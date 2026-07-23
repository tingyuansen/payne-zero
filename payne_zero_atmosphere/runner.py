"""The atmosphere iteration driver.

Flow of `run_atmosphere_model`: resolve the run setup (config.py,
run_setup.py) -> read or produce the initial atmosphere -> generate or load selected
lines (line_selection.py) -> iterate {equation of state -> continuum +
line opacity -> radiative transfer -> temperature/pressure correction ->
convection} until converged (convergence.py) -> quantize the converged columns
in memory and write the structured-atmosphere product (synthesis_bridge.py).
Unported original branches fail loudly via NotImplementedError guards.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from .atmosphere_io import (
    ModelAtmosphere,
    format_atmosphere_deck,
    parse_atmosphere_deck,
)
from .config import AtmosphereConfig
from .convergence import (
    deep_layer_relative_temperature_change,
    max_normalized_column_delta,
    temperature_changes_within_limits,
)
from .convection import (
    ConvectionFiniteDifferenceSamples,
    ConvectionResult,
    compute_convection,
    compute_disabled_convection_diagnostics,
)
from .continuum_opacity import (
    ContinuumAtmosphereState,
    RosselandOpacityTable,
    active_continuum_reference_frequencies,
    assemble_continuum_line_selection_threshold,
    build_continuum_atmosphere_state,
    build_opacity_sampling_grid,
    compute_continuum_opacity_columns,
    create_rosseland_opacity_table,
)
from .doppler import update_doppler_line_strength_factors
from .specific_internal_energy import compute_atomic_specific_internal_energy
from .equation_of_state import populate_all_species, populate_species
from .line_catalog import (
    LineTransitionCatalog,
    SelectedLineCatalog,
    read_line_transition_catalog,
    read_selected_line_catalog,
)
from .line_selection import generate_selected_lines
from .molecular_data import (
    find_default_molecular_equilibrium_catalog,
    read_molecular_equilibrium_catalog,
)
from .molecular_equilibrium import (
    MolecularEquilibriumState,
    initialize_molecular_equilibrium_state,
    restore_molecular_equation_density,
    save_molecular_equation_density,
    set_molecular_specific_internal_energy_mode,
)
from .hydrostatic import integrate_hydrostatic_pressure
from .line_opacity import (
    LineOpacityState,
    accumulate_selected_line_opacity,
    accumulate_transition_line_opacity,
    allocate_line_opacity_state,
)
from .radiative_pressure import (
    RadiativePressureState,
    accumulate_radiative_pressure,
    initialize_radiative_pressure_state,
)
from .radiative_transfer import remap_to_grid
from .radiative_transfer import load_radiative_transfer_tables
from .transfer_kernels import accumulate_transfer_range_parallel
from .transfer_kernels import transfer_chunk_count
from .rosseland_mean import rosseland_mean_step
from .run_setup import RunSetup, resolve_run_setup
from .runtime_state import (
    AtmosphereRuntimeState,
    build_runtime_state,
    update_charge_square_density,
)
from .synthesis_bridge import (
    infer_synthesis_source_catalog_root,
    save_product_structured_atmosphere,
    save_structured_atmosphere_from_runtime_state,
)
from .temperature_correction import (
    TemperatureCorrectionResult,
    TemperatureCorrectionState,
    apply_temperature_correction,
    ingest_temperature_correction_rosseland_table,
    initialize_temperature_correction_state,
)


def _progress(message: str) -> None:
    # Opt-in, off-by-default progress indicator for long multi-iteration solves
    # (PAYNE_ZERO_ATMOSPHERE_PROGRESS=1). Silent by default: does not affect the
    # default byte-identical output.
    if os.getenv("PAYNE_ZERO_ATMOSPHERE_PROGRESS"):
        print(f"[payne-zero-atmosphere] {message}", flush=True)


@dataclass(frozen=True)
class AtmosphereRunResult:
    """Converged or terminal atmosphere state plus solver diagnostics."""

    atmosphere: ModelAtmosphere
    iterations_completed: int
    converged: bool
    diagnostics: dict[str, Any]


@dataclass
class AtmospherePopulationState:
    """Prepared thermodynamic, population, and Doppler state."""

    setup: RunSetup
    runtime_state: AtmosphereRuntimeState
    fractional_doppler_widths: np.ndarray
    partition_normalized_population_over_mass_density_and_fractional_doppler_width: (
        np.ndarray
    )
    temperature_iteration_cache: dict[str, int]
    molecular_state: MolecularEquilibriumState | None = None


@dataclass
class OpacityState:
    """Prepared opacity state before monochromatic transfer and corrections."""

    population_state: AtmospherePopulationState
    continuum_atmosphere: ContinuumAtmosphereState
    opacity_wavelength_grid_nm: np.ndarray
    opacity_frequency_hz: np.ndarray
    frequency_weights: np.ndarray
    active_continuum_indices: np.ndarray
    active_continuum_frequency_hz: np.ndarray
    continuum_absorption: np.ndarray
    continuum_scattering: np.ndarray
    continuum_source: np.ndarray
    continuum_line_selection_threshold: np.ndarray
    continuum_reference_wavelength_nm: np.ndarray
    wavelength_bin_edges: np.ndarray
    line_opacity: LineOpacityState
    rosseland_table: RosselandOpacityTable
    selected_line_catalog: SelectedLineCatalog | None = None
    transition_line_catalog: LineTransitionCatalog | None = None


@dataclass
class TransferAccumulation:
    """Frequency-integrated transfer accumulators before mode-3 finalization."""

    opacity_state: OpacityState
    frequency_start_index: int
    frequency_stop_index: int
    rosseland_accumulator: np.ndarray
    radiative_pressure_state: RadiativePressureState
    temperature_correction_state: TemperatureCorrectionState


@dataclass
class IterationFinalization:
    """Final opacity, radiative-pressure, and correction outputs before remapping."""

    transfer_accumulation: TransferAccumulation
    rosseland_opacity: np.ndarray
    rosseland_optical_depth: np.ndarray
    radiative_pressure_state: RadiativePressureState
    temperature_correction_result: TemperatureCorrectionResult
    convection_result: ConvectionResult | None = None
    convection_finite_difference_samples: ConvectionFiniteDifferenceSamples | None = (
        None
    )


@dataclass
class IterationRemap:
    """Atmosphere and state columns after correction-grid remapping."""

    finalization: IterationFinalization
    atmosphere: ModelAtmosphere
    standard_rosseland_optical_depth: np.ndarray
    integrated_radiation_pressure: np.ndarray
    turbulent_pressure: np.ndarray


def prepare_population_state(
    config: AtmosphereConfig,
    *,
    temperature_iteration_index: int = 1,
    setup: RunSetup | None = None,
    molecular_thermal_energy_erg: np.ndarray | None = None,
) -> AtmospherePopulationState:
    """Run the first parity-checked pre-opacity population phase."""

    setup = resolve_run_setup(config) if setup is None else setup
    state = build_runtime_state(setup.atmosphere)
    update_charge_square_density(
        thermal_energy_erg=setup.atmosphere.thermal_energy_erg,
        state=state,
    )

    temperature_iteration_cache: dict[str, int] = {}
    molecular_state: MolecularEquilibriumState | None = None
    if setup.molecules_enabled:
        molecule_catalog_path = (
            config.inputs.molecules_path or find_default_molecular_equilibrium_catalog()
        )
        if molecule_catalog_path is None:
            raise FileNotFoundError(
                "molecular mode requires a molecular-equilibrium catalog"
            )
        molecular_catalog = read_molecular_equilibrium_catalog(molecule_catalog_path)
        molecular_thermal_energy = (
            setup.atmosphere.thermal_energy_erg
            if molecular_thermal_energy_erg is None
            else np.asarray(molecular_thermal_energy_erg, dtype=np.float64)
        )
        if molecular_thermal_energy.shape != setup.atmosphere.temperature.shape:
            raise ValueError("molecular_thermal_energy_erg must match the atmosphere")
        molecular_state = initialize_molecular_equilibrium_state(
            temperature_k=setup.atmosphere.temperature,
            thermal_energy_erg=molecular_thermal_energy,
            gas_pressure=state.gas_pressure,
            runtime_state=state,
            catalog=molecular_catalog,
        )

    if setup.pressure_iteration_enabled:
        if setup.molecules_enabled:
            populate_species(
                code=0.0,
                population_mode=1,
                output=np.zeros((setup.atmosphere.layers, 1), dtype=np.float64),
                molecules_enabled=True,
                pressure_iteration_enabled=True,
                temperature_k=setup.atmosphere.temperature,
                thermal_energy_erg=setup.atmosphere.thermal_energy_erg,
                state=state,
                temperature_iteration_index=int(temperature_iteration_index),
                temperature_iteration_cache=temperature_iteration_cache,
                molecular_state=molecular_state,
            )
        populate_all_species(
            temperature_k=setup.atmosphere.temperature,
            thermal_energy_erg=setup.atmosphere.thermal_energy_erg,
            state=state,
            molecules_enabled=setup.molecules_enabled,
            pressure_iteration_enabled=True,
            temperature_iteration_index=int(temperature_iteration_index),
            temperature_iteration_cache=temperature_iteration_cache,
            molecular_state=molecular_state,
        )

    fractional_doppler_widths, population_over_density_and_width = (
        update_doppler_line_strength_factors(
            thermal_energy_erg=setup.atmosphere.thermal_energy_erg,
            microturbulence=setup.atmosphere.microturbulence,
            state=state,
        )
    )

    return AtmospherePopulationState(
        setup=setup,
        runtime_state=state,
        fractional_doppler_widths=fractional_doppler_widths,
        partition_normalized_population_over_mass_density_and_fractional_doppler_width=(
            population_over_density_and_width
        ),
        temperature_iteration_cache=temperature_iteration_cache,
        molecular_state=molecular_state,
    )


def prepare_structured_handoff_population_state(
    config: AtmosphereConfig,
    *,
    temperature_iteration_index: int = 1,
    setup: RunSetup | None = None,
    molecular_thermal_energy_erg: np.ndarray | None = None,
) -> AtmospherePopulationState:
    """Build packed synthesis populations at fixed final electron density."""

    setup = resolve_run_setup(config) if setup is None else setup
    state = build_runtime_state(setup.atmosphere)
    update_charge_square_density(
        thermal_energy_erg=setup.atmosphere.thermal_energy_erg,
        state=state,
    )

    temperature_iteration_cache: dict[str, int] = {}
    molecular_state: MolecularEquilibriumState | None = None
    if setup.molecules_enabled:
        molecule_catalog_path = (
            config.inputs.molecules_path or find_default_molecular_equilibrium_catalog()
        )
        if molecule_catalog_path is None:
            raise FileNotFoundError(
                "molecular mode requires a molecular-equilibrium catalog"
            )
        molecular_catalog = read_molecular_equilibrium_catalog(molecule_catalog_path)
        molecular_thermal_energy = (
            setup.atmosphere.thermal_energy_erg
            if molecular_thermal_energy_erg is None
            else np.asarray(molecular_thermal_energy_erg, dtype=np.float64)
        )
        if molecular_thermal_energy.shape != setup.atmosphere.temperature.shape:
            raise ValueError("molecular_thermal_energy_erg must match the atmosphere")
        molecular_state = initialize_molecular_equilibrium_state(
            temperature_k=setup.atmosphere.temperature,
            thermal_energy_erg=molecular_thermal_energy,
            gas_pressure=state.gas_pressure,
            runtime_state=state,
            catalog=molecular_catalog,
        )
        populate_species(
            code=0.0,
            population_mode=1,
            output=np.zeros((setup.atmosphere.layers, 1), dtype=np.float64),
            molecules_enabled=True,
            pressure_iteration_enabled=True,
            temperature_k=setup.atmosphere.temperature,
            thermal_energy_erg=setup.atmosphere.thermal_energy_erg,
            state=state,
            temperature_iteration_index=int(temperature_iteration_index),
            temperature_iteration_cache=temperature_iteration_cache,
            molecular_state=molecular_state,
        )

    populate_all_species(
        temperature_k=setup.atmosphere.temperature,
        thermal_energy_erg=setup.atmosphere.thermal_energy_erg,
        state=state,
        molecules_enabled=setup.molecules_enabled,
        pressure_iteration_enabled=False,
        temperature_iteration_index=int(temperature_iteration_index),
        temperature_iteration_cache=temperature_iteration_cache,
        molecular_state=molecular_state,
    )

    fractional_doppler_widths, population_over_density_and_width = (
        update_doppler_line_strength_factors(
            thermal_energy_erg=setup.atmosphere.thermal_energy_erg,
            microturbulence=setup.atmosphere.microturbulence,
            state=state,
        )
    )

    return AtmospherePopulationState(
        setup=setup,
        runtime_state=state,
        fractional_doppler_widths=fractional_doppler_widths,
        partition_normalized_population_over_mass_density_and_fractional_doppler_width=(
            population_over_density_and_width
        ),
        temperature_iteration_cache=temperature_iteration_cache,
        molecular_state=molecular_state,
    )


def compute_convection_finite_difference_samples(
    *,
    atmosphere: ModelAtmosphere,
    runtime_state: AtmosphereRuntimeState,
    absolute_radiation_pressure: np.ndarray,
    rosseland_optical_depth: np.ndarray,
    temperature_iteration_seed: int,
    temperature_iteration_cache: dict[str, int],
    molecules_enabled: bool = False,
    molecular_state: MolecularEquilibriumState | None = None,
    molecular_thermal_energy_tracks_perturbation: bool = False,
) -> ConvectionFiniteDifferenceSamples:
    """Compute finite-difference energy and density samples for convection."""

    if molecules_enabled and molecular_state is None:
        raise ValueError("molecular_state is required for molecular convection")

    layer_count = atmosphere.layers
    original_temperature = atmosphere.temperature.copy()
    original_pressure = runtime_state.gas_pressure.copy()
    original_electron_density = runtime_state.electron_density.copy()
    original_total_nuclei_number_density = (
        runtime_state.total_nuclei_number_density.copy()
    )
    original_mass_density = runtime_state.mass_density.copy()
    original_ion_stage_population_density = (
        runtime_state.ion_stage_populations_by_packed_slot.copy()
    )
    original_partition_normalized_ion_stage_population_density = (
        runtime_state.partition_normalized_populations_by_packed_slot.copy()
    )
    original_specific_internal_energy = runtime_state.specific_internal_energy.copy()
    original_cache = dict(temperature_iteration_cache)
    original_molecular_populations = None
    original_partition_normalized_molecular_populations = None
    original_molecular_equation_densities = None
    original_previous_molecular_equation_densities = None
    original_molecular_thermal_energy = None
    original_specific_internal_energy_mode = False
    saved_molecular_seed = None
    if molecular_state is not None:
        original_molecular_populations = molecular_state.molecular_populations.copy()
        original_partition_normalized_molecular_populations = (
            molecular_state.partition_normalized_molecular_populations.copy()
        )
        original_molecular_equation_densities = (
            molecular_state.molecular_equation_densities.copy()
        )
        original_previous_molecular_equation_densities = (
            molecular_state.previous_molecular_equation_densities.copy()
        )
        original_molecular_thermal_energy = molecular_state.thermal_energy_erg.copy()
        original_specific_internal_energy_mode = (
            molecular_state.specific_internal_energy_mode_enabled
        )
        if molecules_enabled:
            saved_molecular_seed = save_molecular_equation_density(molecular_state)
            set_molecular_specific_internal_energy_mode(molecular_state, True)
    dilution = 1.0 - np.exp(-np.asarray(rosseland_optical_depth, dtype=np.float64))
    absolute_radiation_pressure_array = np.asarray(
        absolute_radiation_pressure,
        dtype=np.float64,
    )
    dummy_output = np.zeros((layer_count, 1), dtype=np.float64)

    def recompute_pressure_iteration_state(temperature_iteration_index: int) -> None:
        if (
            molecules_enabled
            and molecular_state is not None
            and molecular_thermal_energy_tracks_perturbation
        ):
            molecular_state.thermal_energy_erg[:] = atmosphere.thermal_energy_erg
        populate_species(
            code=0.0,
            population_mode=1,
            output=dummy_output,
            molecules_enabled=bool(molecules_enabled),
            molecular_state=molecular_state,
            pressure_iteration_enabled=True,
            temperature_k=atmosphere.temperature,
            thermal_energy_erg=atmosphere.thermal_energy_erg,
            state=runtime_state,
            temperature_iteration_index=int(temperature_iteration_index),
            temperature_iteration_cache=temperature_iteration_cache,
        )

    try:
        atmosphere.temperature[:] = original_temperature * 1.001
        recompute_pressure_iteration_state(int(temperature_iteration_seed) + 1)
        specific_internal_energy_plus_temperature = (
            runtime_state.specific_internal_energy
            + 3.0
            * absolute_radiation_pressure_array
            / np.maximum(runtime_state.mass_density, 1.0e-300)
            * (1.0 + dilution * (1.001**4 - 1.0))
        ).copy()
        density_plus_temperature = runtime_state.mass_density.copy()

        atmosphere.temperature[:] = original_temperature * 0.999
        recompute_pressure_iteration_state(int(temperature_iteration_seed) + 2)
        specific_internal_energy_minus_temperature = (
            runtime_state.specific_internal_energy
            + 3.0
            * absolute_radiation_pressure_array
            / np.maximum(runtime_state.mass_density, 1.0e-300)
            * (1.0 + dilution * (0.999**4 - 1.0))
        ).copy()
        density_minus_temperature = runtime_state.mass_density.copy()

        atmosphere.temperature[:] = original_temperature
        runtime_state.gas_pressure[:] = original_pressure * 1.001
        recompute_pressure_iteration_state(int(temperature_iteration_seed) + 3)
        specific_internal_energy_plus_pressure = (
            runtime_state.specific_internal_energy
            + 3.0
            * absolute_radiation_pressure_array
            / np.maximum(runtime_state.mass_density, 1.0e-300)
        ).copy()
        density_plus_pressure = runtime_state.mass_density.copy()

        runtime_state.gas_pressure[:] = original_pressure * 0.999
        recompute_pressure_iteration_state(int(temperature_iteration_seed) + 4)
        specific_internal_energy_minus_pressure = (
            runtime_state.specific_internal_energy
            + 3.0
            * absolute_radiation_pressure_array
            / np.maximum(runtime_state.mass_density, 1.0e-300)
        ).copy()
        density_minus_pressure = runtime_state.mass_density.copy()
    finally:
        atmosphere.temperature[:] = original_temperature
        runtime_state.gas_pressure[:] = original_pressure
        runtime_state.electron_density[:] = original_electron_density
        runtime_state.total_nuclei_number_density[:] = (
            original_total_nuclei_number_density
        )
        runtime_state.mass_density[:] = original_mass_density
        runtime_state.ion_stage_populations_by_packed_slot[:] = (
            original_ion_stage_population_density
        )
        runtime_state.partition_normalized_populations_by_packed_slot[:] = (
            original_partition_normalized_ion_stage_population_density
        )
        if np.any(original_specific_internal_energy != 0.0):
            runtime_state.specific_internal_energy[:] = (
                original_specific_internal_energy
            )
        else:
            runtime_state.specific_internal_energy[:] = (
                compute_atomic_specific_internal_energy(
                    temperature_k=atmosphere.temperature,
                    state=runtime_state,
                )
            )
        temperature_iteration_cache.clear()
        temperature_iteration_cache.update(original_cache)
        if molecular_state is not None:
            if original_molecular_populations is not None:
                molecular_state.molecular_populations[:] = (
                    original_molecular_populations
                )
            if original_partition_normalized_molecular_populations is not None:
                molecular_state.partition_normalized_molecular_populations[:] = (
                    original_partition_normalized_molecular_populations
                )
            if original_molecular_equation_densities is not None:
                molecular_state.molecular_equation_densities[:] = (
                    original_molecular_equation_densities
                )
            if original_previous_molecular_equation_densities is not None:
                molecular_state.previous_molecular_equation_densities[:] = (
                    original_previous_molecular_equation_densities
                )
            if original_molecular_thermal_energy is not None:
                molecular_state.thermal_energy_erg[:] = (
                    original_molecular_thermal_energy
                )
            if molecules_enabled:
                restore_molecular_equation_density(
                    molecular_state,
                    saved_molecular_seed,
                )
            molecular_state.specific_internal_energy_mode_enabled = (
                original_specific_internal_energy_mode
            )

    return ConvectionFiniteDifferenceSamples(
        specific_internal_energy_plus_temperature=specific_internal_energy_plus_temperature,
        specific_internal_energy_minus_temperature=specific_internal_energy_minus_temperature,
        specific_internal_energy_plus_pressure=specific_internal_energy_plus_pressure,
        specific_internal_energy_minus_pressure=specific_internal_energy_minus_pressure,
        density_plus_temperature=density_plus_temperature,
        density_minus_temperature=density_minus_temperature,
        density_plus_pressure=density_plus_pressure,
        density_minus_pressure=density_minus_pressure,
    )


def _planck_source_and_stimulated_emission(
    *,
    frequency_hz: float,
    h_over_kt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Planck source and stimulated-emission columns for one frequency."""

    frequency = float(frequency_hz)
    exponential = np.exp(-frequency * np.asarray(h_over_kt, dtype=np.float64))
    stimulated = np.maximum(1.0 - exponential, 1.0e-300)
    frequency_15 = frequency / 1.0e15
    planck_source = 1.47439e-2 * (frequency_15**3) * exponential / stimulated
    return np.asarray(planck_source, dtype=np.float64), np.asarray(
        stimulated, dtype=np.float64
    )


def _empty_first_iteration_rosseland_table(layer_count: int) -> RosselandOpacityTable:
    """Return the empty Rosseland table used during the first iteration."""

    return create_rosseland_opacity_table(int(layer_count))


def _existing_optional_path(path: Path | None) -> bool:
    return path is not None and Path(path).exists()


def _generate_standard_selected_lines(
    config: AtmosphereConfig,
    population_state: AtmospherePopulationState,
    continuum_line_selection_threshold: np.ndarray,
    wavelength_bin_edges: np.ndarray,
):
    """Generate the selected-line catalog from the ported standard catalogs.

    Computed fresh each solve and returned in memory -- selection is fast enough (fused
    parallel kernels over the resident catalogs) that no disk cache is needed.
    """

    inputs = config.inputs
    ported_catalogs = (
        inputs.predicted_atomic_lines_path,
        inputs.observed_atomic_lines_path,
        inputs.high_excitation_lines_path,
        inputs.diatomic_lines_path,
        inputs.titanium_oxide_lines_path,
        inputs.water_lines_path,
        inputs.h3plus_lines_path,
    )
    if not any(_existing_optional_path(path) for path in ported_catalogs):
        raise NotImplementedError(
            "IFOP(15)=1 requires either a preselected selected-line catalog "
            "(selected_line_catalog_path) or at least one of the raw source "
            "catalogs: predicted / observed / high-excitation atomic, diatomic, "
            "TiO, water, or H3+ lines."
        )

    # Selection is fast (fused parallel kernels + resident catalogs), so it is computed
    # fresh each solve and returned IN MEMORY -- no disk cache, no temp selected-line file
    # written and read back. The generation is byte-identical run to run for identical inputs.
    return generate_selected_lines(
        partition_normalized_population_over_mass_density_and_fractional_doppler_width=(
            population_state.partition_normalized_population_over_mass_density_and_fractional_doppler_width
        ),
        continuum_line_selection_threshold=continuum_line_selection_threshold,
        packed_continuum_wavelengths=wavelength_bin_edges,
        hc_over_kt=population_state.setup.atmosphere.hc_over_kt,
        predicted_atomic_lines_path=inputs.predicted_atomic_lines_path,
        observed_atomic_lines_path=inputs.observed_atomic_lines_path,
        high_excitation_lines_path=inputs.high_excitation_lines_path,
        diatomic_lines_path=inputs.diatomic_lines_path,
        titanium_oxide_lines_path=inputs.titanium_oxide_lines_path,
        water_lines_path=inputs.water_lines_path,
        h3plus_lines_path=inputs.h3plus_lines_path,
    )


def prepare_opacity_state(
    config: AtmosphereConfig,
    *,
    population_state: AtmospherePopulationState | None = None,
    temperature_iteration_index: int = 1,
    rosseland_table: RosselandOpacityTable | None = None,
    selected_line_catalog: SelectedLineCatalog | None = None,
    transition_line_catalog: LineTransitionCatalog | None = None,
) -> OpacityState:
    """Prepare continuum and optional line absorption for the next transfer pass.

    It builds the opacity-sampling grid and line-selection thresholds, then
    accumulates selected and detailed transition absorption from canonical
    catalogs. Fresh line selection remains an explicit preceding step.
    """

    prepared = (
        prepare_population_state(
            config,
            temperature_iteration_index=temperature_iteration_index,
        )
        if population_state is None
        else population_state
    )
    setup = prepared.setup
    atmosphere = setup.atmosphere
    runtime_state = prepared.runtime_state
    line_flags = [int(value) for value in setup.opacity_flags]
    if len(line_flags) < 20:
        line_flags.extend([0] * (20 - len(line_flags)))

    continuum_atmosphere = build_continuum_atmosphere_state(atmosphere, runtime_state)
    rosseland = rosseland_table or _empty_first_iteration_rosseland_table(
        continuum_atmosphere.layers,
    )
    opacity_wavelength_grid_nm, frequency_weights = build_opacity_sampling_grid(
        setup.effective_temperature,
    )
    opacity_frequency_hz = 2.99792458e17 / np.maximum(
        opacity_wavelength_grid_nm,
        1.0e-300,
    )

    continuum_absorption, continuum_scattering, continuum_source = (
        compute_continuum_opacity_columns(
            continuum_atmosphere,
            opacity_frequency_hz,
            opacity_flags=line_flags,
            rosseland_table=rosseland,
        )
    )

    active_indices, active_frequency_hz = active_continuum_reference_frequencies(
        setup.effective_temperature,
    )
    active_absorption, active_scattering, _ = compute_continuum_opacity_columns(
        continuum_atmosphere,
        active_frequency_hz,
        opacity_flags=line_flags,
        rosseland_table=rosseland,
    )
    continuum_table, continuum_reference_wavelength_nm, wavelength_bin_edges = (
        assemble_continuum_line_selection_threshold(
            effective_temperature=setup.effective_temperature,
            temperature_k=atmosphere.temperature,
            active_continuum_absorption=active_absorption,
            active_continuum_scattering=active_scattering,
        )
    )

    line_opacity = allocate_line_opacity_state(
        layer_count=atmosphere.layers,
        wavelength_count=opacity_wavelength_grid_nm.size,
    )
    selected_lines = selected_line_catalog
    transition_lines = transition_line_catalog
    if line_flags[14] == 1:
        if selected_lines is None and config.inputs.selected_line_catalog_path is None:
            selected_lines = _generate_standard_selected_lines(
                config, prepared, continuum_table, wavelength_bin_edges
            )
        elif selected_lines is None:
            selected_lines = read_selected_line_catalog(
                config.inputs.selected_line_catalog_path
            )
        line_opacity = accumulate_selected_line_opacity(
            selected_lines=selected_lines,
            opacity_wavelength_grid_nm=opacity_wavelength_grid_nm,
            wavelength_bin_edges=wavelength_bin_edges,
            continuum_line_selection_threshold=continuum_table,
            temperature=atmosphere.temperature,
            hc_over_kt=atmosphere.hc_over_kt,
            electron_density=runtime_state.electron_density,
            ion_stage_populations_by_packed_slot=runtime_state.ion_stage_populations_by_packed_slot,
            partition_normalized_population_over_mass_density_and_fractional_doppler_width=(
                prepared.partition_normalized_population_over_mass_density_and_fractional_doppler_width
            ),
            fractional_doppler_widths=prepared.fractional_doppler_widths,
            wavelength_start_index=1,
            wavelength_stop_index=opacity_wavelength_grid_nm.size,
        )

    if line_flags[16] == 1:
        if (
            transition_lines is None
            and config.inputs.detailed_line_catalog_path is None
        ):
            raise NotImplementedError(
                "IFOP(17)=1 requires a detailed-transition catalog "
                "(detailed_line_catalog_path)."
            )
        if transition_lines is None:
            transition_lines = read_line_transition_catalog(
                config.inputs.detailed_line_catalog_path
            )
        line_opacity = accumulate_transition_line_opacity(
            transition_lines=transition_lines,
            opacity_wavelength_grid_nm=opacity_wavelength_grid_nm,
            wavelength_bin_edges=wavelength_bin_edges,
            continuum_line_selection_threshold=continuum_table,
            temperature=atmosphere.temperature,
            hc_over_kt=atmosphere.hc_over_kt,
            electron_density=runtime_state.electron_density,
            ion_stage_populations_by_packed_slot=runtime_state.ion_stage_populations_by_packed_slot,
            partition_normalized_population_over_mass_density_and_fractional_doppler_width=(
                prepared.partition_normalized_population_over_mass_density_and_fractional_doppler_width
            ),
            fractional_doppler_widths=prepared.fractional_doppler_widths,
            partition_normalized_populations_by_packed_slot=runtime_state.partition_normalized_populations_by_packed_slot,
            mass_density=runtime_state.mass_density,
            base_line_mass_absorption_coefficient=line_opacity.line_mass_absorption_coefficient,
            wavelength_start_index=1,
            wavelength_stop_index=opacity_wavelength_grid_nm.size,
        )

    return OpacityState(
        population_state=prepared,
        continuum_atmosphere=continuum_atmosphere,
        opacity_wavelength_grid_nm=opacity_wavelength_grid_nm,
        opacity_frequency_hz=opacity_frequency_hz,
        frequency_weights=frequency_weights,
        active_continuum_indices=active_indices,
        active_continuum_frequency_hz=active_frequency_hz,
        continuum_absorption=continuum_absorption,
        continuum_scattering=continuum_scattering,
        continuum_source=continuum_source,
        continuum_line_selection_threshold=continuum_table,
        continuum_reference_wavelength_nm=continuum_reference_wavelength_nm,
        wavelength_bin_edges=wavelength_bin_edges,
        line_opacity=line_opacity,
        rosseland_table=rosseland,
        selected_line_catalog=selected_lines,
        transition_line_catalog=transition_lines,
    )


def accumulate_transfer_state(
    opacity_state: OpacityState,
    *,
    frequency_start_index: int = 0,
    frequency_stop_index: int | None = None,
    temperature_correction_state: TemperatureCorrectionState | None = None,
) -> TransferAccumulation:
    """Accumulate transfer, opacity, pressure, and correction terms.

    Runs the compiled per-frequency transfer kernel (the sole production path).
    """

    setup = opacity_state.population_state.setup
    atmosphere = setup.atmosphere
    layer_count = atmosphere.layers
    frequency_count = int(opacity_state.opacity_frequency_hz.size)
    start = max(0, int(frequency_start_index))
    stop = (
        frequency_count if frequency_stop_index is None else int(frequency_stop_index)
    )
    stop = min(max(start, stop), frequency_count)

    h_over_kt = atmosphere.h_over_kt
    column_mass = atmosphere.column_mass
    temperature = atmosphere.temperature
    target_integrated_eddington_flux = (
        5.6697e-5 / 12.5664 * setup.effective_temperature**4
    )

    rosseland_accumulator = np.zeros(layer_count, dtype=np.float64)
    rosseland_accumulator, _ = rosseland_mean_step(
        rosseland_accumulator,
        mode=1,
        frequency_weight=0.0,
        planck_source=np.zeros(layer_count, dtype=np.float64),
        frequency_hz=0.0,
        h_over_kt=h_over_kt,
        temperature_k=temperature,
        stimulated_emission=np.ones(layer_count, dtype=np.float64),
        total_opacity=np.ones(layer_count, dtype=np.float64),
        frequency_count=frequency_count,
        column_mass=column_mass,
    )
    radiative_pressure = initialize_radiative_pressure_state(layer_count)
    accumulate_radiative_pressure(
        radiative_pressure,
        mode=1,
        frequency_weight=0.0,
        total_opacity=np.ones(layer_count, dtype=np.float64),
        monochromatic_eddington_flux=np.zeros(layer_count, dtype=np.float64),
        mean_intensity=np.zeros(layer_count, dtype=np.float64),
        surface_second_moment=0.0,
        target_integrated_eddington_flux=target_integrated_eddington_flux,
        column_mass=column_mass,
    )
    temperature_correction = (
        initialize_temperature_correction_state(layer_count)
        if temperature_correction_state is None
        else temperature_correction_state
    )
    apply_temperature_correction(
        temperature_correction,
        mode=1,
        frequency_weight=0.0,
        column_mass=column_mass,
        total_opacity=np.ones(layer_count, dtype=np.float64),
        monochromatic_eddington_flux=np.zeros(layer_count, dtype=np.float64),
        mean_intensity_minus_source=np.zeros(layer_count, dtype=np.float64),
        monochromatic_optical_depth=np.zeros(layer_count, dtype=np.float64),
        planck_source=np.zeros(layer_count, dtype=np.float64),
        frequency_hz=0.0,
        h_over_kt=h_over_kt,
        temperature_k=temperature,
        stimulated_emission=np.ones(layer_count, dtype=np.float64),
        scattering_fraction=np.zeros(layer_count, dtype=np.float64),
        target_integrated_eddington_flux=target_integrated_eddington_flux,
        effective_temperature=setup.effective_temperature,
        frequency_count=frequency_count,
    )

    if stop > start:
        tables = load_radiative_transfer_tables()
        planck_all = np.zeros((frequency_count, layer_count), dtype=np.float64)
        stimulated_all = np.ones((frequency_count, layer_count), dtype=np.float64)
        # Vectorized BNU / stimulated-emission over the [start, stop) frequency
        # block: each (frequency, layer) pair is independent, so the per-frequency
        # Python loop collapses to a single broadcast (element-for-element the
        # scalar _planck_source_and_stimulated_emission). numpy ufuncs replace
        # the interpreter overhead; the block matmul-free math stays fp64-exact.
        block_frequency = np.ascontiguousarray(
            opacity_state.opacity_frequency_hz[start:stop], dtype=np.float64
        )
        h_over_kt_row = np.asarray(h_over_kt, dtype=np.float64)[None, :]
        block_exponential = np.exp(-block_frequency[:, None] * h_over_kt_row)
        block_stimulated = np.maximum(1.0 - block_exponential, 1.0e-300)
        block_frequency_15 = block_frequency / 1.0e15
        block_planck = (
            1.47439e-2
            * (block_frequency_15[:, None] ** 3)
            * block_exponential
            / block_stimulated
        )
        planck_all[start:stop] = block_planck
        stimulated_all[start:stop] = block_stimulated
        surface_constant = np.array(
            [radiative_pressure.surface_radiation_pressure_constant],
            dtype=np.float64,
        )
        chunk_count = min(
            transfer_chunk_count(),
            max(1, stop - start),
        )
        accumulate_transfer_range_parallel(
            int(chunk_count),
            start,
            stop,
            np.ascontiguousarray(opacity_state.opacity_frequency_hz, dtype=np.float64),
            np.ascontiguousarray(opacity_state.frequency_weights, dtype=np.float64),
            planck_all,
            stimulated_all,
            np.ascontiguousarray(opacity_state.continuum_absorption, dtype=np.float64),
            np.ascontiguousarray(opacity_state.continuum_scattering, dtype=np.float64),
            np.ascontiguousarray(opacity_state.continuum_source, dtype=np.float64),
            np.ascontiguousarray(
                opacity_state.line_opacity.line_mass_absorption_coefficient,
                dtype=np.float32,
            ),
            np.ascontiguousarray(column_mass, dtype=np.float64),
            np.ascontiguousarray(h_over_kt, dtype=np.float64),
            np.ascontiguousarray(temperature, dtype=np.float64),
            np.ascontiguousarray(tables.transfer_optical_depth_grid, dtype=np.float64),
            np.ascontiguousarray(tables.mean_intensity_operator, dtype=np.float32),
            np.ascontiguousarray(tables.eddington_flux_operator, dtype=np.float32),
            np.ascontiguousarray(tables.second_moment_weights, dtype=np.float32),
            float(target_integrated_eddington_flux),
            float(setup.effective_temperature),
            int(frequency_count),
            rosseland_accumulator,
            radiative_pressure.radiation_energy_density,
            radiative_pressure.integrated_eddington_flux,
            radiative_pressure.radiative_acceleration,
            surface_constant,
            temperature_correction.absorption_heating_derivative,
            temperature_correction.mean_intensity_minus_source_integral,
            temperature_correction.integrated_eddington_flux,
            temperature_correction.diagonal_lambda_accumulator,
        )
        radiative_pressure.surface_radiation_pressure_constant = float(
            surface_constant[0]
        )

    return TransferAccumulation(
        opacity_state=opacity_state,
        frequency_start_index=start,
        frequency_stop_index=stop,
        rosseland_accumulator=rosseland_accumulator,
        radiative_pressure_state=radiative_pressure,
        temperature_correction_state=temperature_correction,
    )


def finalize_transfer_state(
    transfer_accumulation: TransferAccumulation,
    *,
    iteration_index: int = 1,
    temperature_iteration_seed: int | None = None,
    convection_enabled: bool | int = False,
    convective_flux: np.ndarray | None = None,
    previous_convective_flux: np.ndarray | None = None,
    logarithmic_temperature_pressure_gradient: np.ndarray | None = None,
    adiabatic_gradient: np.ndarray | None = None,
    pressure_scale_height: np.ndarray | None = None,
    total_pressure: np.ndarray | None = None,
    log_density_temperature_derivative_at_constant_total_pressure: (
        np.ndarray | None
    ) = None,
    heat_capacity: np.ndarray | None = None,
    mixing_length: float = 1.0,
    integrated_radiation_pressure: np.ndarray | None = None,
    turbulent_pressure: np.ndarray | None = None,
    molecular_convection_thermal_tracks_perturbation: bool = False,
) -> IterationFinalization:
    """Finalize one transfer pass on the original Rosseland-depth grid.

    This mirrors the old driver boundary immediately after the monochromatic
    frequency loop: final opacity, radiative pressure, lookup ingest, and
    temperature correction. It does not yet apply the later depth remap.
    """

    opacity_state = transfer_accumulation.opacity_state
    setup = opacity_state.population_state.setup
    atmosphere = setup.atmosphere
    runtime_state = opacity_state.population_state.runtime_state
    layer_count = atmosphere.layers
    frequency_count = int(opacity_state.opacity_frequency_hz.size)
    h_over_kt = atmosphere.h_over_kt
    temperature = atmosphere.temperature
    column_mass = atmosphere.column_mass
    target_integrated_eddington_flux = (
        5.6697e-5 / 12.5664 * setup.effective_temperature**4
    )
    zeros = np.zeros(layer_count, dtype=np.float64)
    ones = np.ones(layer_count, dtype=np.float64)

    rosseland_opacity, rosseland_optical_depth = rosseland_mean_step(
        transfer_accumulation.rosseland_accumulator,
        mode=3,
        frequency_weight=0.0,
        planck_source=zeros,
        frequency_hz=0.0,
        h_over_kt=h_over_kt,
        temperature_k=temperature,
        stimulated_emission=ones,
        total_opacity=ones,
        frequency_count=frequency_count,
        column_mass=column_mass,
    )
    radiative_pressure = transfer_accumulation.radiative_pressure_state
    accumulate_radiative_pressure(
        radiative_pressure,
        mode=3,
        frequency_weight=0.0,
        total_opacity=ones,
        monochromatic_eddington_flux=zeros,
        mean_intensity=zeros,
        surface_second_moment=0.0,
        target_integrated_eddington_flux=target_integrated_eddington_flux,
        column_mass=column_mass,
    )
    temperature_correction = transfer_accumulation.temperature_correction_state
    ingest_temperature_correction_rosseland_table(
        temperature_correction,
        temperature_k=temperature,
        gas_pressure=runtime_state.gas_pressure,
        rosseland_opacity=rosseland_opacity,
    )
    convection_result: ConvectionResult | None = None
    finite_difference_samples: ConvectionFiniteDifferenceSamples | None = None
    if int(convection_enabled) == 1 and convective_flux is None:
        finite_difference_samples = compute_convection_finite_difference_samples(
            atmosphere=atmosphere,
            runtime_state=runtime_state,
            absolute_radiation_pressure=(
                radiative_pressure.absolute_radiation_pressure
            ),
            rosseland_optical_depth=rosseland_optical_depth,
            temperature_iteration_seed=(
                int(iteration_index) * 10
                if temperature_iteration_seed is None
                else int(temperature_iteration_seed)
            ),
            temperature_iteration_cache=(
                opacity_state.population_state.temperature_iteration_cache
            ),
            molecules_enabled=setup.molecules_enabled,
            molecular_state=opacity_state.population_state.molecular_state,
            molecular_thermal_energy_tracks_perturbation=(
                molecular_convection_thermal_tracks_perturbation
            ),
        )
        total_pressure_for_convection = setup.surface_gravity_cgs * column_mass + float(
            setup.surface_radiation_pressure_constant
        )
        turbulent_pressure_for_convection = (
            np.zeros(layer_count, dtype=np.float64)
            if turbulent_pressure is None
            else np.asarray(turbulent_pressure, dtype=np.float64)
        )
        total_pressure_for_convection = (
            total_pressure_for_convection + turbulent_pressure_for_convection
        )
        convection_zero_top_layer_count = (
            int(setup.convection.zero_top_layer_count)
            if int(setup.convection.zero_top_layer_count) > 0
            else 36
        )
        convection_result = compute_convection(
            rosseland_table=temperature_correction.rosseland_opacity_table,
            column_mass=column_mass,
            rosseland_optical_depth=rosseland_optical_depth,
            temperature_k=temperature,
            gas_pressure=runtime_state.gas_pressure,
            mass_density=runtime_state.mass_density,
            rosseland_opacity=rosseland_opacity,
            microturbulence=atmosphere.microturbulence,
            absolute_radiation_pressure=(
                radiative_pressure.absolute_radiation_pressure
            ),
            total_pressure=total_pressure_for_convection,
            surface_gravity_cgs=setup.surface_gravity_cgs,
            target_integrated_eddington_flux=target_integrated_eddington_flux,
            mixing_length=setup.convection.mixing_length,
            overshoot_weight=setup.convection.overshoot_weight,
            convection_enabled=True,
            zero_top_layer_count=convection_zero_top_layer_count,
            specific_internal_energy_plus_temperature=(
                finite_difference_samples.specific_internal_energy_plus_temperature
            ),
            specific_internal_energy_minus_temperature=(
                finite_difference_samples.specific_internal_energy_minus_temperature
            ),
            specific_internal_energy_plus_pressure=(
                finite_difference_samples.specific_internal_energy_plus_pressure
            ),
            specific_internal_energy_minus_pressure=(
                finite_difference_samples.specific_internal_energy_minus_pressure
            ),
            density_plus_temperature=(
                finite_difference_samples.density_plus_temperature
            ),
            density_minus_temperature=(
                finite_difference_samples.density_minus_temperature
            ),
            density_plus_pressure=(finite_difference_samples.density_plus_pressure),
            density_minus_pressure=(finite_difference_samples.density_minus_pressure),
        )
        convective_flux = convection_result.convective_flux
        previous_convective_flux = convection_result.raw_convective_flux
        logarithmic_temperature_pressure_gradient = (
            convection_result.logarithmic_temperature_pressure_gradient
        )
        adiabatic_gradient = convection_result.adiabatic_gradient
        pressure_scale_height = convection_result.pressure_scale_height
        total_pressure = total_pressure_for_convection
        log_density_temperature_derivative_at_constant_total_pressure = convection_result.log_density_temperature_derivative_at_constant_total_pressure
        heat_capacity = convection_result.heat_capacity
        mixing_length = setup.convection.mixing_length

    correction_result = apply_temperature_correction(
        temperature_correction,
        mode=3,
        frequency_weight=0.0,
        column_mass=column_mass,
        total_opacity=ones,
        monochromatic_eddington_flux=zeros,
        mean_intensity_minus_source=zeros,
        monochromatic_optical_depth=zeros,
        planck_source=zeros,
        frequency_hz=0.0,
        h_over_kt=h_over_kt,
        temperature_k=temperature,
        stimulated_emission=ones,
        scattering_fraction=zeros,
        target_integrated_eddington_flux=target_integrated_eddington_flux,
        effective_temperature=setup.effective_temperature,
        frequency_count=frequency_count,
        rosseland_optical_depth=rosseland_optical_depth,
        rosseland_opacity=rosseland_opacity,
        iteration_index=int(iteration_index),
        convection_enabled=convection_enabled,
        convective_flux=convective_flux,
        previous_convective_flux=previous_convective_flux,
        logarithmic_temperature_pressure_gradient=logarithmic_temperature_pressure_gradient,
        adiabatic_gradient=adiabatic_gradient,
        pressure_scale_height=pressure_scale_height,
        total_pressure=total_pressure,
        mass_density=runtime_state.mass_density,
        log_density_temperature_derivative_at_constant_total_pressure=(
            log_density_temperature_derivative_at_constant_total_pressure
        ),
        heat_capacity=heat_capacity,
        mixing_length=mixing_length,
        integrated_radiation_pressure=(
            radiative_pressure.integrated_radiation_pressure
            if integrated_radiation_pressure is None
            else integrated_radiation_pressure
        ),
        turbulent_pressure=(
            np.zeros(layer_count, dtype=np.float64)
            if turbulent_pressure is None
            else turbulent_pressure
        ),
        surface_gravity_cgs=setup.surface_gravity_cgs,
    )
    if correction_result is None:
        raise RuntimeError("the final temperature correction returned no result")

    return IterationFinalization(
        transfer_accumulation=transfer_accumulation,
        rosseland_opacity=rosseland_opacity,
        rosseland_optical_depth=rosseland_optical_depth,
        radiative_pressure_state=radiative_pressure,
        temperature_correction_result=correction_result,
        convection_result=convection_result,
        convection_finite_difference_samples=finite_difference_samples,
    )


def remap_finalized_iteration_state(
    finalization: IterationFinalization,
    *,
    convective_flux: np.ndarray | None = None,
    convective_velocity: np.ndarray | None = None,
    turbulent_pressure: np.ndarray | None = None,
    completed_iterations: int | None = None,
    standard_log_tau_step: float = 0.125,
    standard_log_tau_start: float = -6.875,
) -> IterationRemap:
    """Apply temperature corrections and remap state to the standard grid.

    The correction returns temperature and column mass on its native depth
    grid. Every layer field is then remapped to the standard Rosseland
    optical-depth grid so the output columns share one depth coordinate.
    """

    transfer = finalization.transfer_accumulation
    opacity_state = transfer.opacity_state
    setup = opacity_state.population_state.setup
    source_atmosphere = setup.atmosphere
    runtime_state = opacity_state.population_state.runtime_state
    layer_count = source_atmosphere.layers
    correction = finalization.temperature_correction_result
    rosseland_optical_depth = finalization.rosseland_optical_depth
    standard_rosseland_optical_depth = np.float64(10.0) ** (
        float(standard_log_tau_start)
        + np.arange(layer_count, dtype=np.float64) * float(standard_log_tau_step)
    )

    corrected_column_mass = np.asarray(correction.column_mass, dtype=np.float64)
    corrected_temperature = np.asarray(correction.temperature, dtype=np.float64)
    gas_pressure = np.asarray(runtime_state.gas_pressure, dtype=np.float64)
    electron_density = np.asarray(runtime_state.electron_density, dtype=np.float64)
    rosseland_opacity = np.asarray(finalization.rosseland_opacity, dtype=np.float64)
    integrated_radiation_pressure = np.asarray(
        finalization.radiative_pressure_state.integrated_radiation_pressure,
        dtype=np.float64,
    )
    microturbulence = np.asarray(source_atmosphere.microturbulence, dtype=np.float64)
    turbulent = (
        np.zeros(layer_count, dtype=np.float64)
        if turbulent_pressure is None
        else np.asarray(turbulent_pressure, dtype=np.float64)
    )

    remapped_column_mass, _ = remap_to_grid(
        rosseland_optical_depth,
        corrected_column_mass,
        standard_rosseland_optical_depth,
    )
    remapped_temperature, _ = remap_to_grid(
        rosseland_optical_depth,
        corrected_temperature,
        standard_rosseland_optical_depth,
    )
    remapped_gas_pressure, _ = remap_to_grid(
        rosseland_optical_depth,
        gas_pressure,
        standard_rosseland_optical_depth,
    )
    remapped_electron_density, _ = remap_to_grid(
        rosseland_optical_depth,
        electron_density,
        standard_rosseland_optical_depth,
    )
    remapped_rosseland_opacity, _ = remap_to_grid(
        rosseland_optical_depth,
        rosseland_opacity,
        standard_rosseland_optical_depth,
    )
    remapped_integrated_radiation_pressure, _ = remap_to_grid(
        rosseland_optical_depth,
        integrated_radiation_pressure,
        standard_rosseland_optical_depth,
    )
    remapped_microturbulence, _ = remap_to_grid(
        rosseland_optical_depth,
        microturbulence,
        standard_rosseland_optical_depth,
    )
    remapped_turbulent_pressure, _ = remap_to_grid(
        rosseland_optical_depth,
        turbulent,
        standard_rosseland_optical_depth,
    )
    remapped_radiative_acceleration, _ = remap_to_grid(
        rosseland_optical_depth,
        finalization.radiative_pressure_state.radiative_acceleration,
        standard_rosseland_optical_depth,
    )

    output_convective_flux = (
        np.zeros(layer_count, dtype=np.float64)
        if convective_flux is None
        else np.asarray(convective_flux, dtype=np.float64).copy()
    )
    if output_convective_flux.size > 2:
        output_convective_flux[1:-1] = np.asarray(
            correction.convective_flux, dtype=np.float64
        )[1:-1]
    output_convective_velocity = (
        np.zeros(layer_count, dtype=np.float64)
        if convective_velocity is None
        else np.asarray(convective_velocity, dtype=np.float64).copy()
    )
    remapped_convective_flux, _ = remap_to_grid(
        rosseland_optical_depth,
        output_convective_flux,
        standard_rosseland_optical_depth,
    )
    remapped_convective_velocity, _ = remap_to_grid(
        rosseland_optical_depth,
        output_convective_velocity,
        standard_rosseland_optical_depth,
    )

    metadata = source_atmosphere.metadata.copy()
    metadata["surface_radiation_pressure_line"] = (
        "PRADK "
        f"{float(finalization.radiative_pressure_state.surface_radiation_pressure_constant):.4E}"
    )
    if completed_iterations is not None:
        metadata["begin_line"] = (
            f"BEGIN                    ITERATION{int(completed_iterations):4d} COMPLETED"
        )

    atmosphere = ModelAtmosphere(
        column_mass=remapped_column_mass,
        temperature=remapped_temperature,
        gas_pressure=remapped_gas_pressure,
        electron_density=remapped_electron_density,
        rosseland_opacity=remapped_rosseland_opacity,
        radiative_acceleration=remapped_radiative_acceleration,
        microturbulence=remapped_microturbulence,
        convective_flux=remapped_convective_flux,
        convective_velocity=remapped_convective_velocity,
        metadata=metadata,
        fixed_column_abundance_values=source_atmosphere.fixed_column_abundance_values.copy(),
    )

    return IterationRemap(
        finalization=finalization,
        atmosphere=atmosphere,
        standard_rosseland_optical_depth=standard_rosseland_optical_depth,
        integrated_radiation_pressure=remapped_integrated_radiation_pressure,
        turbulent_pressure=remapped_turbulent_pressure,
    )


def finalize_remapped_iteration(
    remapped_iteration: IterationRemap,
    *,
    iterations_completed: int,
    converged: bool = False,
    diagnostics: dict[str, Any] | None = None,
) -> AtmosphereRunResult:
    """Apply the fixed-column quantization contract to a remapped atmosphere."""

    remapped_iteration.atmosphere.metadata["begin_line"] = (
        f"BEGIN                    ITERATION{int(iterations_completed):4d} COMPLETED"
    )
    quantized_atmosphere = parse_atmosphere_deck(
        format_atmosphere_deck(remapped_iteration.atmosphere),
        source="final fixed-column quantization",
    )
    return AtmosphereRunResult(
        atmosphere=quantized_atmosphere,
        iterations_completed=int(iterations_completed),
        converged=bool(converged),
        diagnostics={} if diagnostics is None else dict(diagnostics),
    )


def _write_debug_state_npz(
    path: Path,
    *,
    remapped_iteration: IterationRemap,
    opacity_state: OpacityState,
    iterations_completed: int,
) -> None:
    """Write a physically named debug snapshot for solver triage."""

    atmosphere = remapped_iteration.atmosphere
    finalization = remapped_iteration.finalization
    transfer = finalization.transfer_accumulation
    runtime = opacity_state.population_state.runtime_state
    correction = finalization.temperature_correction_result
    convection = finalization.convection_result
    temperature_state = transfer.temperature_correction_state

    arrays: dict[str, np.ndarray] = {
        "debug_schema_version": np.asarray([4], dtype=np.int32),
        "column_mass": np.asarray(atmosphere.column_mass, dtype=np.float64),
        "temperature": np.asarray(atmosphere.temperature, dtype=np.float64),
        "thermal_energy_erg": np.asarray(
            atmosphere.thermal_energy_erg, dtype=np.float64
        ),
        "gas_pressure": np.asarray(atmosphere.gas_pressure, dtype=np.float64),
        "electron_density": np.asarray(atmosphere.electron_density, dtype=np.float64),
        "microturbulence": np.asarray(atmosphere.microturbulence, dtype=np.float64),
        "total_nuclei_number_density": np.asarray(
            runtime.total_nuclei_number_density, dtype=np.float64
        ),
        "mass_density": np.asarray(runtime.mass_density, dtype=np.float64),
        "charge_square_density": np.asarray(
            runtime.charge_square_density, dtype=np.float64
        ),
        "specific_internal_energy": np.asarray(
            runtime.specific_internal_energy, dtype=np.float64
        ),
        "mean_nuclear_mass_amu": np.asarray(
            runtime.mean_nuclear_mass_amu, dtype=np.float64
        ),
        "elemental_abundances_by_layer": np.asarray(
            runtime.elemental_abundances_by_layer, dtype=np.float64
        ),
        "ion_stage_populations_by_packed_slot": np.asarray(
            runtime.ion_stage_populations_by_packed_slot, dtype=np.float64
        ),
        "partition_normalized_populations_by_packed_slot": np.asarray(
            runtime.partition_normalized_populations_by_packed_slot, dtype=np.float64
        ),
        "rosseland_opacity": np.asarray(atmosphere.rosseland_opacity, dtype=np.float64),
        "radiative_acceleration": np.asarray(
            atmosphere.radiative_acceleration, dtype=np.float64
        ),
        "iterations_completed": np.asarray([int(iterations_completed)], dtype=np.int32),
        "molecules_enabled": np.asarray(
            [1 if opacity_state.population_state.molecular_state is not None else 0],
            dtype=np.int32,
        ),
        "standard_rosseland_optical_depth": np.asarray(
            remapped_iteration.standard_rosseland_optical_depth, dtype=np.float64
        ),
        "integrated_radiation_pressure": np.asarray(
            remapped_iteration.integrated_radiation_pressure, dtype=np.float64
        ),
        "integrated_eddington_flux": np.asarray(
            temperature_state.integrated_eddington_flux, dtype=np.float64
        ),
        "mean_intensity_minus_source_integral": np.asarray(
            temperature_state.mean_intensity_minus_source_integral,
            dtype=np.float64,
        ),
        "absorption_heating_derivative": np.asarray(
            temperature_state.absorption_heating_derivative,
            dtype=np.float64,
        ),
        "diagonal_lambda_accumulator": np.asarray(
            temperature_state.diagonal_lambda_accumulator,
            dtype=np.float64,
        ),
        "flux_error_percent": np.asarray(
            correction.flux_error_percent, dtype=np.float64
        ),
        "flux_derivative": np.asarray(correction.flux_derivative, dtype=np.float64),
        "flux_temperature_derivative": np.asarray(
            correction.flux_temperature_derivative, dtype=np.float64
        ),
        "lambda_temperature_derivative": np.asarray(
            correction.lambda_temperature_derivative, dtype=np.float64
        ),
        "temperature_correction": np.asarray(
            correction.temperature_correction, dtype=np.float64
        ),
        "convective_flux": np.asarray(atmosphere.convective_flux, dtype=np.float64),
        "convective_velocity": np.asarray(
            atmosphere.convective_velocity, dtype=np.float64
        ),
    }
    if runtime.major_isotope_mass_amu is not None:
        arrays["major_isotope_mass_amu"] = np.asarray(
            runtime.major_isotope_mass_amu, dtype=np.float64
        )
    if runtime.fractional_doppler_widths is not None:
        arrays["fractional_doppler_widths"] = np.asarray(
            runtime.fractional_doppler_widths, dtype=np.float64
        )
    if (
        runtime.partition_normalized_population_over_mass_density_and_fractional_doppler_width
        is not None
    ):
        arrays[
            "partition_normalized_population_over_mass_density_and_fractional_doppler_width"
        ] = np.asarray(
            runtime.partition_normalized_population_over_mass_density_and_fractional_doppler_width,
            dtype=np.float64,
        )
    if convection is not None:
        arrays["logarithmic_temperature_pressure_gradient"] = np.asarray(
            convection.logarithmic_temperature_pressure_gradient,
            dtype=np.float64,
        )
        arrays["heat_capacity"] = np.asarray(convection.heat_capacity, dtype=np.float64)
        arrays["log_density_temperature_derivative_at_constant_total_pressure"] = (
            np.asarray(
                convection.log_density_temperature_derivative_at_constant_total_pressure,
                dtype=np.float64,
            )
        )
        arrays["adiabatic_gradient"] = np.asarray(
            convection.adiabatic_gradient, dtype=np.float64
        )
        arrays["pressure_scale_height"] = np.asarray(
            convection.pressure_scale_height, dtype=np.float64
        )
    finite_difference = finalization.convection_finite_difference_samples
    if finite_difference is not None:
        arrays["specific_internal_energy_plus_temperature"] = np.asarray(
            finite_difference.specific_internal_energy_plus_temperature,
            dtype=np.float64,
        )
        arrays["specific_internal_energy_minus_temperature"] = np.asarray(
            finite_difference.specific_internal_energy_minus_temperature,
            dtype=np.float64,
        )
        arrays["specific_internal_energy_plus_pressure"] = np.asarray(
            finite_difference.specific_internal_energy_plus_pressure,
            dtype=np.float64,
        )
        arrays["specific_internal_energy_minus_pressure"] = np.asarray(
            finite_difference.specific_internal_energy_minus_pressure,
            dtype=np.float64,
        )
        arrays["density_plus_temperature"] = np.asarray(
            finite_difference.density_plus_temperature,
            dtype=np.float64,
        )
        arrays["density_minus_temperature"] = np.asarray(
            finite_difference.density_minus_temperature,
            dtype=np.float64,
        )
        arrays["density_plus_pressure"] = np.asarray(
            finite_difference.density_plus_pressure,
            dtype=np.float64,
        )
        arrays["density_minus_pressure"] = np.asarray(
            finite_difference.density_minus_pressure,
            dtype=np.float64,
        )
    molecular_state = opacity_state.population_state.molecular_state
    if molecular_state is not None:
        catalog = molecular_state.catalog
        arrays["molecule_codes"] = np.asarray(catalog.molecule_codes, dtype=np.float64)
        arrays["molecular_equilibrium_coefficients"] = np.asarray(
            catalog.equilibrium_coefficients, dtype=np.float64
        )
        arrays["molecular_component_start_indices"] = np.asarray(
            catalog.component_start_indices, dtype=np.int32
        )
        arrays["molecular_component_equation_indices"] = np.asarray(
            catalog.component_equation_indices, dtype=np.int32
        )
        arrays["molecular_equation_species_codes"] = np.asarray(
            catalog.equation_species_codes, dtype=np.int32
        )
        arrays["molecular_populations"] = np.asarray(
            molecular_state.molecular_populations,
            dtype=np.float64,
        )
        arrays["partition_normalized_molecular_populations"] = np.asarray(
            molecular_state.partition_normalized_molecular_populations,
            dtype=np.float64,
        )
        arrays["molecular_equation_densities"] = np.asarray(
            molecular_state.molecular_equation_densities,
            dtype=np.float64,
        )

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, **arrays)


def _copy_iteration_atmosphere(
    atmosphere: ModelAtmosphere,
    *,
    gas_pressure: np.ndarray | None = None,
) -> ModelAtmosphere:
    """Return an independent atmosphere object for the next exact iteration."""

    return ModelAtmosphere(
        column_mass=np.asarray(atmosphere.column_mass, dtype=np.float64).copy(),
        temperature=np.asarray(atmosphere.temperature, dtype=np.float64).copy(),
        gas_pressure=(
            np.asarray(atmosphere.gas_pressure, dtype=np.float64).copy()
            if gas_pressure is None
            else np.asarray(gas_pressure, dtype=np.float64).copy()
        ),
        electron_density=np.asarray(
            atmosphere.electron_density, dtype=np.float64
        ).copy(),
        rosseland_opacity=np.asarray(
            atmosphere.rosseland_opacity, dtype=np.float64
        ).copy(),
        radiative_acceleration=np.asarray(
            atmosphere.radiative_acceleration,
            dtype=np.float64,
        ).copy(),
        microturbulence=np.asarray(atmosphere.microturbulence, dtype=np.float64).copy(),
        convective_flux=np.asarray(atmosphere.convective_flux, dtype=np.float64).copy(),
        convective_velocity=np.asarray(
            atmosphere.convective_velocity,
            dtype=np.float64,
        ).copy(),
        metadata=dict(atmosphere.metadata),
        fixed_column_abundance_values=dict(atmosphere.fixed_column_abundance_values),
    )


def run_atmosphere_model(config: AtmosphereConfig) -> AtmosphereRunResult:
    """Run the pykurucz-aligned exact atmosphere solver.

    The current product runner covers fixed continuum-only and standard atomic
    line-opacity iterations, with turbulence, NLTE, hydrogen wing opacity, and
    raw molecular selectors off. Molecules are direct-gated for continuum-only
    and standard/detailed line-opacity runs.
    Within that envelope it executes the same setup -> opacity -> transfer ->
    correction -> remap -> write boundary that the component parity tests pin
    to pykurucz.
    """

    run_start_time = time.perf_counter()
    setup_start_time = time.perf_counter()
    setup = resolve_run_setup(config)
    setup_seconds = time.perf_counter() - setup_start_time
    _require_supported_run_setup(setup)
    remapped: IterationRemap | None = None
    previous_rosseland_table: RosselandOpacityTable | None = None
    previous_surface_radiation_pressure_constant = float(
        setup.surface_radiation_pressure_constant
    )
    molecular_thermal_energy_reference = np.asarray(
        setup.atmosphere.thermal_energy_erg,
        dtype=np.float64,
    ).copy()
    iteration_itemp = 0
    temperature_correction_state = initialize_temperature_correction_state(
        setup.atmosphere.layers,
    )
    selected_line_catalog: SelectedLineCatalog | None = None
    transition_line_catalog: LineTransitionCatalog | None = None
    opacity: OpacityState | None = None
    transfer: TransferAccumulation | None = None
    completed_iterations = 0
    consecutive_converged_iterations = 0
    converged = False
    last_deep_layer_relative_temperature_change = float("inf")
    last_all_layer_relative_temperature_change = float("inf")
    iteration_timings: list[dict[str, float | int]] = []

    for iteration_index in range(1, int(setup.iterations) + 1):
        iteration_start_time = time.perf_counter()
        iteration_timing: dict[str, float | int] = {"iteration": int(iteration_index)}
        _progress(f"iteration {iteration_index}/{int(setup.iterations)}: start")
        completed_iterations = iteration_index
        iteration_itemp += iteration_index
        stage_start_time = time.perf_counter()
        if remapped is None:
            iteration_atmosphere = _copy_iteration_atmosphere(setup.atmosphere)
        else:
            gas_pressure = remapped.atmosphere.gas_pressure
            if setup.pressure_iteration_enabled:
                gas_pressure = integrate_hydrostatic_pressure(
                    remapped.atmosphere,
                    surface_gravity_cgs=setup.surface_gravity_cgs,
                    integrated_radiation_pressure=remapped.integrated_radiation_pressure,
                    turbulent_pressure=remapped.turbulent_pressure,
                )
            iteration_atmosphere = _copy_iteration_atmosphere(
                remapped.atmosphere,
                gas_pressure=gas_pressure,
            )

        iteration_setup = replace(
            setup,
            atmosphere=iteration_atmosphere,
            surface_radiation_pressure_constant=previous_surface_radiation_pressure_constant,
        )
        iteration_timing["prepare_iteration_seconds"] = (
            time.perf_counter() - stage_start_time
        )
        stage_start_time = time.perf_counter()
        population = prepare_population_state(
            config,
            temperature_iteration_index=iteration_itemp,
            setup=iteration_setup,
            molecular_thermal_energy_erg=molecular_thermal_energy_reference,
        )
        iteration_timing["population_seconds"] = time.perf_counter() - stage_start_time
        _progress(f"iteration {iteration_index}/{int(setup.iterations)}: opacity")
        stage_start_time = time.perf_counter()
        opacity = prepare_opacity_state(
            config,
            population_state=population,
            temperature_iteration_index=iteration_itemp,
            rosseland_table=previous_rosseland_table,
            selected_line_catalog=selected_line_catalog,
            transition_line_catalog=transition_line_catalog,
        )
        selected_line_catalog = opacity.selected_line_catalog
        transition_line_catalog = opacity.transition_line_catalog
        iteration_timing["opacity_seconds"] = time.perf_counter() - stage_start_time
        _progress(f"iteration {iteration_index}/{int(setup.iterations)}: transfer")
        stage_start_time = time.perf_counter()
        transfer = accumulate_transfer_state(
            opacity,
            temperature_correction_state=temperature_correction_state,
        )
        iteration_timing["transfer_seconds"] = time.perf_counter() - stage_start_time
        _progress(f"iteration {iteration_index}/{int(setup.iterations)}: finalization")
        stage_start_time = time.perf_counter()
        finalization = finalize_transfer_state(
            transfer,
            iteration_index=iteration_index,
            temperature_iteration_seed=iteration_itemp * 10,
            convection_enabled=setup.convection.enabled,
            molecular_convection_thermal_tracks_perturbation=bool(
                config.molecular_convection_thermal_tracks_perturbation
            ),
        )
        iteration_timing["finalization_seconds"] = (
            time.perf_counter() - stage_start_time
        )
        stage_start_time = time.perf_counter()
        total_pressure = (
            iteration_setup.surface_gravity_cgs * iteration_setup.atmosphere.column_mass
            + float(iteration_setup.surface_radiation_pressure_constant)
        )
        radiative_pressure = finalization.radiative_pressure_state
        temperature_correction = transfer.temperature_correction_state
        convection_flux = None
        convection_velocity = None
        if finalization.convection_result is not None:
            convection_flux = finalization.convection_result.convective_flux
            convection_velocity = finalization.convection_result.convective_velocity
        else:
            convection_diagnostics = compute_disabled_convection_diagnostics(
                column_mass=iteration_setup.atmosphere.column_mass,
                rosseland_optical_depth=finalization.rosseland_optical_depth,
                temperature_k=iteration_setup.atmosphere.temperature,
                gas_pressure=population.runtime_state.gas_pressure,
                mass_density=population.runtime_state.mass_density,
                rosseland_opacity=finalization.rosseland_opacity,
                absolute_radiation_pressure=(
                    radiative_pressure.absolute_radiation_pressure
                ),
                total_pressure=total_pressure,
                surface_gravity_cgs=iteration_setup.surface_gravity_cgs,
                target_integrated_eddington_flux=5.6697e-5
                / 12.5664
                * iteration_setup.effective_temperature**4,
                mixing_length=iteration_setup.convection.mixing_length,
                rosseland_table=temperature_correction.rosseland_opacity_table,
                overshoot_weight=iteration_setup.convection.overshoot_weight,
                zero_top_layer_count=(
                    int(iteration_setup.convection.zero_top_layer_count)
                    if int(iteration_setup.convection.zero_top_layer_count) > 0
                    else 36
                ),
            )
            convection_flux = convection_diagnostics.convective_flux
            convection_velocity = convection_diagnostics.convective_velocity
        remapped = remap_finalized_iteration_state(
            finalization,
            convective_flux=convection_flux,
            convective_velocity=convection_velocity,
            completed_iterations=iteration_index,
        )
        iteration_timing["remap_seconds"] = time.perf_counter() - stage_start_time
        stage_start_time = time.perf_counter()
        previous_rosseland_table = temperature_correction.rosseland_opacity_table
        previous_surface_radiation_pressure_constant = (
            radiative_pressure.surface_radiation_pressure_constant
        )
        last_deep_layer_relative_temperature_change = (
            deep_layer_relative_temperature_change(
                iteration_setup.atmosphere.temperature,
                remapped.atmosphere.temperature,
            )
        )
        last_all_layer_relative_temperature_change = max_normalized_column_delta(
            iteration_setup.atmosphere.temperature,
            remapped.atmosphere.temperature,
            floor=1.0,
            symmetric=True,
        )
        iteration_absolute_flux_error_percent = np.abs(
            remapped.finalization.temperature_correction_result.flux_error_percent
        )
        iteration_timing.update(
            {
                "deep_layer_relative_temperature_change": float(
                    last_deep_layer_relative_temperature_change
                ),
                "all_layer_relative_temperature_change": float(
                    last_all_layer_relative_temperature_change
                ),
                "median_absolute_flux_error_percent": float(
                    np.median(iteration_absolute_flux_error_percent)
                ),
                "p95_absolute_flux_error_percent": float(
                    np.percentile(iteration_absolute_flux_error_percent, 95.0)
                ),
                "maximum_absolute_flux_error_percent": float(
                    np.max(iteration_absolute_flux_error_percent)
                ),
            }
        )
        temperature_change_within_limit = temperature_changes_within_limits(
            deep_layer_change=last_deep_layer_relative_temperature_change,
            all_layer_change=last_all_layer_relative_temperature_change,
            maximum_deep_layer_change=(
                setup.maximum_deep_layer_relative_temperature_change
            ),
            maximum_all_layer_change=(
                setup.maximum_all_layer_relative_temperature_change
            ),
        )
        if (
            setup.enable_convergence_stop
            and iteration_index >= int(setup.minimum_iterations_before_convergence)
            and temperature_change_within_limit
        ):
            consecutive_converged_iterations += 1
        else:
            consecutive_converged_iterations = 0
        if setup.enable_convergence_stop and consecutive_converged_iterations >= int(
            setup.required_consecutive_converged_iterations
        ):
            converged = True
            iteration_timing["convergence_seconds"] = (
                time.perf_counter() - stage_start_time
            )
            iteration_timing["total_seconds"] = (
                time.perf_counter() - iteration_start_time
            )
            iteration_timings.append(iteration_timing)
            _progress(
                f"iteration {iteration_index}/{int(setup.iterations)}: converged "
                f"deep_layer_relative_temperature_change={last_deep_layer_relative_temperature_change:.6e}"
            )
            break
        iteration_timing["convergence_seconds"] = time.perf_counter() - stage_start_time
        iteration_timing["total_seconds"] = time.perf_counter() - iteration_start_time
        iteration_timings.append(iteration_timing)
        _progress(
            f"iteration {iteration_index}/{int(setup.iterations)}: done "
            f"deep_layer_relative_temperature_change={last_deep_layer_relative_temperature_change:.6e}"
        )

    if remapped is None or opacity is None or transfer is None:
        raise RuntimeError("atmosphere solver completed no iterations")

    line_opacity_enabled = bool(setup.opacity_flags[14] or setup.opacity_flags[16])
    molecule_label = "molecules" if setup.molecules_enabled else "no_molecules"
    if line_opacity_enabled:
        supported_branch = (
            f"one_iteration_{molecule_label}_line_opacity"
            if int(setup.iterations) == 1
            else f"fixed_iteration_{molecule_label}_line_opacity"
        )
    else:
        supported_branch = f"fixed_iteration_{molecule_label}_no_lines"
    final_absolute_flux_error_percent = np.abs(
        remapped.finalization.temperature_correction_result.flux_error_percent
    )
    diagnostics = {
        "supported_branch": supported_branch,
        "layer_count": int(setup.atmosphere.layers),
        "frequency_count": int(opacity.opacity_frequency_hz.size),
        "frequency_start_index": int(transfer.frequency_start_index),
        "frequency_stop_index": int(transfer.frequency_stop_index),
        "line_selection_enabled": bool(setup.opacity_flags[14]),
        "detailed_line_enabled": bool(setup.opacity_flags[16]),
        "molecules_enabled": bool(setup.molecules_enabled),
        "convection_enabled": bool(setup.convection.enabled),
        "deep_layer_relative_temperature_change": float(
            last_deep_layer_relative_temperature_change
        ),
        "all_layer_relative_temperature_change": float(
            last_all_layer_relative_temperature_change
        ),
        "median_absolute_flux_error_percent": float(
            np.median(final_absolute_flux_error_percent)
        ),
        "p95_absolute_flux_error_percent": float(
            np.percentile(final_absolute_flux_error_percent, 95.0)
        ),
        "maximum_absolute_flux_error_percent": float(
            np.max(final_absolute_flux_error_percent)
        ),
        "maximum_deep_layer_relative_temperature_change": float(
            setup.maximum_deep_layer_relative_temperature_change
        ),
        "maximum_all_layer_relative_temperature_change": (
            None
            if setup.maximum_all_layer_relative_temperature_change is None
            else float(setup.maximum_all_layer_relative_temperature_change)
        ),
        "enable_convergence_stop": bool(setup.enable_convergence_stop),
        "minimum_iterations_before_convergence": int(
            setup.minimum_iterations_before_convergence
        ),
        "required_consecutive_converged_iterations": int(
            setup.required_consecutive_converged_iterations
        ),
        "consecutive_converged_iterations": int(consecutive_converged_iterations),
        "setup_seconds": float(setup_seconds),
        "total_seconds": float(time.perf_counter() - run_start_time),
        "iteration_timings": iteration_timings,
    }
    result = finalize_remapped_iteration(
        remapped,
        iterations_completed=int(completed_iterations),
        converged=bool(converged),
        diagnostics=diagnostics,
    )
    diagnostics = result.diagnostics

    if config.outputs.debug_state_path is not None:
        debug_state_path = Path(config.outputs.debug_state_path)
        _write_debug_state_npz(
            debug_state_path,
            remapped_iteration=remapped,
            opacity_state=opacity,
            iterations_completed=int(completed_iterations),
        )
        diagnostic_population = opacity.population_state
        diagnostic_population_source = "last_opacity_iteration"
        if converged:
            try:
                final_population_setup = replace(setup, atmosphere=remapped.atmosphere)
                diagnostic_population = prepare_structured_handoff_population_state(
                    config,
                    temperature_iteration_index=int(iteration_itemp) + 1,
                    setup=final_population_setup,
                    molecular_thermal_energy_erg=remapped.atmosphere.thermal_energy_erg,
                )
                diagnostic_population_source = "final_remapped_atmosphere"
            except RuntimeError as exc:
                diagnostics["diagnostic_runtime_structured_atmosphere_warning"] = str(
                    exc
                )
        diagnostic_structured_path = (
            debug_state_path.parent
            / "payne_zero_runtime_state_structured_diagnostic.npz"
        )
        diagnostic_structured_path.unlink(missing_ok=True)
        try:
            save_structured_atmosphere_from_runtime_state(
                diagnostic_structured_path,
                atmosphere=remapped.atmosphere,
                runtime_state=diagnostic_population.runtime_state,
                molecular_state=diagnostic_population.molecular_state,
            )
        except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
            message = "diagnostic structured handoff was not written: " + str(exc)
            existing = diagnostics.get(
                "diagnostic_runtime_structured_atmosphere_warning"
            )
            diagnostics["diagnostic_runtime_structured_atmosphere_warning"] = (
                f"{existing}; {message}" if existing else message
            )
        else:
            diagnostics["diagnostic_runtime_structured_atmosphere_path"] = str(
                diagnostic_structured_path
            )
            diagnostics["diagnostic_runtime_structured_population_source"] = (
                diagnostic_population_source
            )

    if config.outputs.structured_atmosphere_path is not None:
        structured_start_time = time.perf_counter()
        structured_atmosphere_path = Path(config.outputs.structured_atmosphere_path)
        structured_atmosphere_path.unlink(missing_ok=True)
        if not converged:
            diagnostics["structured_atmosphere_warning"] = (
                "product structured handoff was not written because the "
                "atmosphere did not satisfy the convergence criterion"
            )
        else:
            try:
                save_product_structured_atmosphere(
                    result.atmosphere,
                    structured_atmosphere_path,
                    source_catalog_root=infer_synthesis_source_catalog_root(
                        config.inputs.molecules_path
                    ),
                    molecular_lines=bool(setup.molecules_enabled),
                    device="cpu",
                    dtype="float64",
                )
                diagnostics["structured_atmosphere_path"] = str(
                    structured_atmosphere_path
                )
                diagnostics["structured_atmosphere_source"] = (
                    "final_fixed_column_quantized_arrays"
                )
            except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
                diagnostics["structured_atmosphere_warning"] = (
                    "product structured handoff was not written: " + str(exc)
                )
        diagnostics["structured_atmosphere_seconds"] = float(
            time.perf_counter() - structured_start_time
        )
    return result


def _require_supported_run_setup(setup: RunSetup) -> None:
    """Fail before the runner silently skips an unsupported physical branch."""

    if setup.iterations < 1:
        raise NotImplementedError(
            "run_atmosphere_model requires at least one iteration"
        )
    if setup.turbulence.enabled:
        raise NotImplementedError(
            "run_atmosphere_model does not support the turbulent-pressure branch"
        )
    opacity_flags = [int(value) for value in setup.opacity_flags]
    if len(opacity_flags) < 20:
        opacity_flags.extend([0] * (20 - len(opacity_flags)))
    if opacity_flags[13] == 1:
        raise NotImplementedError(
            "run_atmosphere_model does not port HLINOP hydrogen wings"
        )
