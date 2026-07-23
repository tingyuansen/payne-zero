"""The synthesis engine pipeline.

One `SynthesisPipeline` init loads physics tables from the data home and
compiles the requested wavelength window from the converted source
catalogs (atomic_lines.py, source_catalog_molecular_compiler.py; caches
outside the tree). `run` then executes: equation of state -> continuum ->
atomic/hydrogen/helium/molecular line opacity -> radiative transfer ->
total flux, continuum flux, and normalized flux.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from .atmosphere import ATMOSPHERE_SCHEMA_VERSION, load_atmosphere_npz
from .constants import (
    BOLTZMANN_ERG_PER_K,
    REFERENCE_ATOMIC_MASS_GRAM,
    REFERENCE_BOLTZMANN_ERG_PER_K,
    REFERENCE_BOLTZMANN_EV_PER_K,
    LIGHT_SPEED_CM_PER_S,
    LIGHT_SPEED_NM_PER_S,
    PLANCK_ERG_SECOND,
)
from . import paths as runtime_paths
from .device import REFERENCE_DTYPE, resolve_runtime
from . import equation_of_state
from . import continuum as continuum_engine
from . import line_opacity as line_opacity_engine
from . import hydrogen_lines
from . import radiative_transfer
from . import atomic_lines
from . import molecular_lines as molecular_lines_engine

_SYNTHESIS_TABLE_DIR = runtime_paths.SYNTHESIS_TABLE_DIR


# Line-profile reach tests consume an edge block of this width.  Keeping one
# complete block outside each requested boundary prevents a finite synthesis
# window from changing the first or last returned native sample.
WINDOW_CONTEXT_SAMPLES = 16


@dataclass
class SpectrumResult:
    """Spectrum output from one forward pass."""

    wavelength_nm: np.ndarray
    eddington_flux_total_per_frequency: np.ndarray
    eddington_flux_continuum_per_frequency: np.ndarray
    normalized_flux: np.ndarray
    continuum_absorption: np.ndarray | None
    continuum_scattering: np.ndarray | None
    line_mass_absorption_coefficient: np.ndarray | None
    line_source: np.ndarray | None
    spectral_operator_seconds: float = 0.0
    spectral_operator_name: str | None = None


def _apply_spectral_operator_in_wavelength_density(
    eddington_flux_total_per_frequency: torch.Tensor,
    eddington_flux_continuum_per_frequency: torch.Tensor,
    input_wavelength_nm: np.ndarray,
    spectral_operator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """Apply an instrument operator to wavelength-density flux.

    Radiative transfer returns H_nu, while an observed wavelength-space
    line-spread function acts on F_lambda. The common factor of 4 pi cancels
    from the normalized spectrum, so this boundary applies only the
    c / lambda^2 Jacobian before convolution. It converts the convolved outputs
    back to H_nu so SpectrumResult retains its established internal contract;
    the public API restores physical F_lambda.
    """

    dtype = eddington_flux_total_per_frequency.dtype
    device = eddington_flux_total_per_frequency.device
    input_wavelength = torch.as_tensor(
        input_wavelength_nm,
        dtype=dtype,
        device=device,
    )
    input_jacobian = LIGHT_SPEED_NM_PER_S / input_wavelength.square()
    total_per_wavelength = eddington_flux_total_per_frequency * input_jacobian
    continuum_per_wavelength = eddington_flux_continuum_per_frequency * input_jacobian
    (
        convolved_total_per_wavelength,
        convolved_continuum_per_wavelength,
        normalized_flux,
    ) = spectral_operator.convolve_fluxes(
        total_per_wavelength,
        continuum_per_wavelength,
    )

    output_wavelength_nm = np.asarray(
        spectral_operator.output_wavelength_nm,
        np.float64,
    )
    output_wavelength = torch.as_tensor(
        output_wavelength_nm,
        dtype=dtype,
        device=device,
    )
    output_jacobian = LIGHT_SPEED_NM_PER_S / output_wavelength.square()
    return (
        convolved_total_per_wavelength / output_jacobian,
        convolved_continuum_per_wavelength / output_jacobian,
        normalized_flux,
        output_wavelength_nm,
    )


def _atomic_catalog_for_kernels(
    atomic_catalog: "atomic_lines.LineCatalog",
    tables: dict,
) -> dict:
    """Convert the public line catalog into the kernel-facing mapping.

    The opacity kernels consume the same field names as ``LineCatalog``.  The
    remaining table keys come from packaged reference tables and are decoded in
    the specific kernel that uses them.
    """
    kernel_catalog = {
        "wavelength_nm": np.asarray(atomic_catalog.wavelength_nm, np.float64),
        "index_wavelength_nm": np.asarray(
            atomic_catalog.index_wavelength_nm, np.float64
        ),
        "oscillator_strength": np.asarray(
            atomic_catalog.oscillator_strength, np.float64
        ),
        "log_oscillator_strength": np.asarray(
            atomic_catalog.log_oscillator_strength, np.float64
        ),
        "lower_excitation_cm": np.asarray(
            atomic_catalog.lower_excitation_cm, np.float64
        ),
        "radiative_damping": np.asarray(atomic_catalog.radiative_damping, np.float64),
        "stark_damping": np.asarray(atomic_catalog.stark_damping, np.float64),
        "van_der_waals_damping": np.asarray(
            atomic_catalog.van_der_waals_damping, np.float64
        ),
        "raw_radiative_damping_log": np.asarray(
            atomic_catalog.raw_radiative_damping_log,
            np.float64,
        ),
        "raw_stark_damping_log": np.asarray(
            atomic_catalog.raw_stark_damping_log, np.float64
        ),
        "raw_van_der_waals_damping_log": np.asarray(
            atomic_catalog.raw_van_der_waals_damping_log, np.float64
        ),
        "ion_stage": np.asarray(atomic_catalog.ion_stage, np.int64),
        "atomic_number": np.asarray(atomic_catalog.atomic_number, np.int64),
        "species_code": np.asarray(atomic_catalog.species_code, np.float64),
        "line_size": np.asarray(atomic_catalog.line_size, np.int64),
        "line_type": np.asarray(atomic_catalog.line_type, np.int64),
        "lower_principal_quantum_number": np.asarray(
            atomic_catalog.lower_principal_quantum_number,
            np.int64,
        ),
        "upper_principal_quantum_number": np.asarray(
            atomic_catalog.upper_principal_quantum_number,
            np.int64,
        ),
    }
    # Window-independent Harris profile tables are shared by every catalog slice.
    for table_name in (
        "harris_profile_h0_table",
        "harris_profile_h1_table",
        "harris_profile_h2_table",
    ):
        kernel_catalog[table_name] = np.asarray(tables[table_name])
    for table_name in (
        "radiative_damping_sums",
        "impact_electron_density_thresholds_cm3",
        "stark_knm_table",
        "stark_probability_table",
        "stark_wing_correction_c",
        "stark_wing_correction_d",
        "stark_pressure_grid",
        "stark_beta_grid",
    ):
        kernel_catalog[table_name] = np.asarray(tables[table_name])
    kernel_catalog["hydrogen_continuum_edges"] = np.asarray(
        tables["hydrogen_continuum_edges"],
        np.float64,
    )
    return kernel_catalog


# Invariant physics data files; these are not model outputs or cached answers.
_ATOMIC_MASS_TABLE = _SYNTHESIS_TABLE_DIR / "atomic_masses.npz"
_CONTINUUM_EDGE_TABLE = _SYNTHESIS_TABLE_DIR / "continuum_edge_grid.npz"


def load_atomic_masses(path: Path | None = None) -> np.ndarray:
    """Load the 99-element atomic-mass table in amu."""
    if path is None:
        # Override the bundled 99-element atomic-mass table via
        # PAYNE_ZERO_SYNTHESIS_ATOMIC_MASS_TABLE (kept: lets users supply an
        # alternate mass table without code changes).
        path = Path(
            os.environ.get(
                "PAYNE_ZERO_SYNTHESIS_ATOMIC_MASS_TABLE", str(_ATOMIC_MASS_TABLE)
            )
        )
    atomic_mass_table = np.load(path, allow_pickle=False)
    return np.asarray(atomic_mass_table["atomic_mass_amu"], np.float64)[:99]


def compute_doppler_per_ion(temperature, microturbulence, atomic_masses) -> np.ndarray:
    """Doppler widths ``v_D/c`` per depth, ion stage, and species slot."""
    temperature = np.asarray(temperature, np.float64)
    microturbulence = np.asarray(microturbulence, np.float64)
    atomic_masses = np.asarray(atomic_masses, np.float64)
    n_depths = temperature.size
    thermal_energy_erg = REFERENCE_BOLTZMANN_ERG_PER_K * temperature
    doppler = np.zeros((n_depths, 6, 139), dtype=np.float64)
    valid = atomic_masses > 0
    # width[d, z] for valid Z; broadcast over (depth, element).
    width = np.zeros((n_depths, 99), dtype=np.float64)
    arg = (
        2.0
        * thermal_energy_erg[:, None]
        / (atomic_masses[None, :] * REFERENCE_ATOMIC_MASS_GRAM)
        + (microturbulence**2)[:, None]
    )
    width[:, valid] = np.sqrt(arg[:, valid]) / LIGHT_SPEED_CM_PER_S
    # the same width for every ion stage of the element (atomic slots 0..98).
    doppler[:, :, :99] = width[:, None, :]
    doppler[:, :, :99][:, :, ~valid] = 0.0
    return doppler


def _standard_microturbulence_column(
    microturbulence,
    n_depths: int,
    *,
    default_microturbulence_cm_s: float = 2.0e5,
) -> np.ndarray:
    """Return a positive microturbulence column for bridge synthesis."""
    if microturbulence is None:
        return np.full(n_depths, float(default_microturbulence_cm_s), np.float64)
    microturbulence_cm_s = np.asarray(microturbulence, np.float64)
    if microturbulence_cm_s.ndim == 0:
        microturbulence_cm_s = np.full(
            n_depths, float(microturbulence_cm_s), np.float64
        )
    else:
        microturbulence_cm_s = microturbulence_cm_s.reshape(-1)
    if microturbulence_cm_s.size != n_depths:
        raise ValueError(
            f"microturbulence has {microturbulence_cm_s.size} layers, expected {n_depths}"
        )
    valid_microturbulence = np.isfinite(microturbulence_cm_s) & (
        microturbulence_cm_s > 0.0
    )
    if not np.any(valid_microturbulence):
        return np.full(n_depths, float(default_microturbulence_cm_s), np.float64)
    if np.all(valid_microturbulence):
        return microturbulence_cm_s.astype(np.float64, copy=False)
    filled_microturbulence_cm_s = microturbulence_cm_s.astype(np.float64, copy=True)
    filled_microturbulence_cm_s[~valid_microturbulence] = float(
        np.median(filled_microturbulence_cm_s[valid_microturbulence])
    )
    return filled_microturbulence_cm_s


def _build_edge_grid(edge_table_path: Path = _CONTINUUM_EDGE_TABLE):
    """Build the invariant continuum ionization-edge interpolation grid."""
    if not hasattr(_build_edge_grid, "_cache"):
        with np.load(edge_table_path, allow_pickle=False) as edge_table:
            _build_edge_grid._cache = dict(
                signed_continuum_edge_frequency_hz=np.asarray(
                    edge_table["signed_continuum_edge_frequency_hz"], np.float64
                ),
                continuum_edge_wavelength_nm=np.asarray(
                    edge_table["continuum_edge_wavelength_nm"], np.float64
                ),
                continuum_edge_wavenumber_cm=np.asarray(
                    edge_table["continuum_edge_wavenumber_cm"], np.float64
                ),
                continuum_edge_sample_frequency_hz=np.asarray(
                    edge_table["continuum_edge_sample_frequency_hz"],
                    np.float64,
                ),
                continuum_edge_midpoint_wavelength_nm=np.asarray(
                    edge_table["continuum_edge_midpoint_wavelength_nm"], np.float64
                ),
                edge_interval_width_squared_over_two_nm2=np.asarray(
                    edge_table["continuum_edge_interval_width_squared_over_two_nm2"],
                    np.float64,
                ),
            )
    return _build_edge_grid._cache


def build_structured_atmosphere_from_columns(
    *,
    temperature,
    column_mass,
    gas_pressure,
    electron_density,
    elemental_abundances,
    mean_nuclear_mass_amu: float,
    microturbulence=None,
    eos_tables: "equation_of_state.EOSTables",
    electron_density_seed=None,
    tol: float = 1.0e-5,
    atomic_masses=None,
    mass_density=None,
    molecular_species_codes=None,
    molecules_path=None,
) -> dict:
    """Build the native structured-atmosphere mapping from depth columns."""

    def bridge_start() -> float:
        return time.perf_counter()

    def bridge_end(name: str, t0: float) -> None:
        return None

    t_bridge = bridge_start()
    temperature_array = np.asarray(temperature, np.float64)
    column_mass = np.asarray(column_mass, np.float64)
    gas_pressure_array = np.asarray(gas_pressure, np.float64)
    electron_density_array = np.asarray(electron_density, np.float64)
    elemental_abundances = np.asarray(elemental_abundances, np.float64)
    n_depths = temperature_array.size
    microturbulence = _standard_microturbulence_column(microturbulence, n_depths)
    if atomic_masses is None:
        atomic_masses = load_atomic_masses()
    bridge_end("bridge.inputs", t_bridge)

    # (1) the self-consistent EOS state (populations + n_e).
    t_bridge = bridge_start()
    molecular_species_codes = (
        np.asarray(molecular_species_codes, np.int64)
        if molecular_species_codes is not None
        else np.zeros(0, np.int64)
    )
    use_molecules = molecular_species_codes.size > 0
    # The atmosphere solver already converged the electron density for the
    # final structure, so the bridge fills the population struct AT that
    # density (no charge-balance fixed point rerun). solve_population_state
    # remains the public API for callers that need the full resolve.
    population_state = equation_of_state.solve_population_state_at_electron_density(
        temperature_array,
        gas_pressure_array,
        elemental_abundances,
        tables=eos_tables,
        electron_density=(
            electron_density_array
            if electron_density_seed is None
            else electron_density_seed
        ),
        mean_nuclear_mass_amu=mean_nuclear_mass_amu,
        mass_density=mass_density,
        molecules=use_molecules,
        molecules_path=molecules_path,
    )
    bridge_end("bridge.population_state", t_bridge)

    # (2) derived molecular H2 and H II populations.
    t_bridge = bridge_start()
    hydrogen_neutral_population = np.asarray(
        population_state.hydrogen_neutral_population, np.float64
    )
    hydrogen_partition_normalized_ion_stage_populations = np.asarray(
        population_state.hydrogen_partition_normalized_ion_stage_populations, np.float64
    ).copy()
    helium_neutral_population = np.asarray(
        population_state.helium_neutral_population, np.float64
    )
    helium_singly_ionized_population = np.asarray(
        population_state.helium_singly_ionized_population, np.float64
    )
    carbon_partition_normalized_ion_stage_populations = np.asarray(
        population_state.carbon_partition_normalized_ion_stage_populations, np.float64
    ).copy()
    magnesium_neutral_partition_normalized_population = np.asarray(
        population_state.magnesium_neutral_partition_normalized_population, np.float64
    )
    aluminum_neutral_partition_normalized_population = np.asarray(
        population_state.aluminum_neutral_partition_normalized_population, np.float64
    )
    silicon_neutral_partition_normalized_population = np.asarray(
        population_state.silicon_neutral_partition_normalized_population, np.float64
    )
    iron_neutral_partition_normalized_population = np.asarray(
        population_state.iron_neutral_partition_normalized_population, np.float64
    )
    thermal_energy_ev = temperature_array * REFERENCE_BOLTZMANN_EV_PER_K
    natural_log_temperature = np.log(temperature_array)
    molecular_hydrogen_equilibrium_factor = np.exp(
        4.478 / thermal_energy_ev
        - 4.64584e1
        + (
            1.63660e-3
            + (
                -4.93992e-7
                + (
                    1.11822e-10
                    + (
                        -1.49567e-14
                        + (1.06206e-18 - 3.08720e-23 * temperature_array)
                        * temperature_array
                    )
                    * temperature_array
                )
                * temperature_array
            )
            * temperature_array
        )
        * temperature_array
        - 1.5 * natural_log_temperature
    )
    molecular_hydrogen_population = (
        hydrogen_neutral_population**2 * molecular_hydrogen_equilibrium_factor
    )
    molecular_hydrogen_population[temperature_array > 9000.0] = 0.0
    hydrogen_ionized_population = np.asarray(
        population_state.hydrogen_ionized_population,
        np.float64,
    )
    bridge_end("bridge.basic_pop_arrays", t_bridge)

    # Optional cool-star molecular slot fill for molecular bands. The atomic
    # EOS fills stages 0..5 for real elements; molecular bands read synthetic stage 5
    # columns keyed by species code. Rebuild those from molecular equilibrium
    # rather than borrowing the atomic-only abundance distribution.
    # from the reference atmosphere.
    partition_normalized_populations = np.asarray(
        population_state.partition_normalized_populations, np.float64
    ).copy()
    ion_stage_populations = np.asarray(
        population_state.ion_stage_populations, np.float64
    ).copy()
    if use_molecules:
        from . import molecular_equilibrium as _molecular_equilibrium

        t_bridge = bridge_start()
        partition_functions = (
            population_state.eos.partition_functions.detach().cpu().double().numpy()
        )
        molecular_metadata = _molecular_equilibrium.molecular_equilibrium_metadata(
            molecules_path
        )
        molecular_equilibrium_elements = [
            int(atomic_number)
            for atomic_number in molecular_metadata.equation_species_codes[
                : molecular_metadata.equation_count
            ]
            if 1 <= int(atomic_number) <= 99
        ]
        bridge_partition_functions = partition_functions.copy()
        bridge_end("bridge.molecular_inputs", t_bridge)
        t_bridge = bridge_start()
        partition_without_ground_floor = (
            equation_of_state.partition_functions_for_elements(
                temperature_array,
                gas_pressure_array,
                np.asarray(population_state.electron_density, np.float64),
                tables=eos_tables,
                elements=molecular_equilibrium_elements,
                nion=6,
                apply_ground_partition=False,
            )
        )
        for atomic_number, partition_by_ion in partition_without_ground_floor.items():
            n_stages_to_copy = min(
                partition_by_ion.shape[1], bridge_partition_functions.shape[2]
            )
            bridge_partition_functions[:, atomic_number - 1, :n_stages_to_copy] = (
                partition_by_ion[:, :n_stages_to_copy]
            )
        bridge_end("bridge.partition_without_ground_floor", t_bridge)
        # Reuse the population-state molecular solve when it is present; the
        # re-solve branch below only serves population states built without
        # molecules.
        reuse_molecular_populations = (
            getattr(population_state, "molecular_equation_densities", None) is not None
            and getattr(population_state, "molecular_populations", None) is not None
        )
        if reuse_molecular_populations:
            t_bridge = bridge_start()
            molecular_equation_densities = np.asarray(
                population_state.molecular_equation_densities, np.float64
            )
            molecular_populations = np.asarray(
                population_state.molecular_populations, np.float64
            )
            bridge_end("bridge.reuse_molecular_equilibrium", t_bridge)
        else:
            t_bridge = bridge_start()
            ion_formation_constants = (
                equation_of_state.molecular_ion_formation_constants_from_seed(
                    temperature_array,
                    gas_pressure_array,
                    np.asarray(population_state.electron_density, np.float64),
                    tables=eos_tables,
                    meta=molecular_metadata,
                )
            )
            bridge_end("bridge.ion_formation_constants", t_bridge)
            t_bridge = bridge_start()
            (
                _total_nuclei_number_density_unused,
                molecular_populations_t,
                molecular_equation_densities_t,
                _electron_density_unused,
            ) = _molecular_equilibrium.solve_molecular_equilibrium(
                temperature_array,
                gas_pressure_array,
                np.asarray(population_state.electron_density, np.float64),
                elemental_abundances,
                ion_formation_constants,
                molecules_path=molecules_path,
                device=eos_tables.device,
                dtype=REFERENCE_DTYPE
                if eos_tables.device.type != "mps"
                else eos_tables.dtype,
                tol=max(tol, 1.0e-4),
            )
            bridge_end("bridge.solve_molecular_equilibrium", t_bridge)
            molecular_equation_densities = (
                molecular_equation_densities_t.detach().cpu().double().numpy()
            )
            molecular_populations = (
                molecular_populations_t.detach().cpu().double().numpy()
            )
        t_bridge = bridge_start()
        # The molecular population-state solve already returns the
        # reference-compatible atomic ion slots. Never rebuild ordinary atomic
        # line populations from the molecular-equilibrium densities: for trace
        # elements such as Li, that neutral-equation total is not the atomic
        # line-population total and it makes the 610.5/670.8 nm Li lines
        # catastrophically too strong. The synthetic molecular slot below is
        # the only slot molecular bands consume.
        hydrogen_molecule_indices = np.where(
            np.abs(
                molecular_metadata.molecule_codes[: molecular_metadata.molecule_count]
                - 101.0
            )
            < 1.0e-3
        )[0]
        if hydrogen_molecule_indices.size:
            molecular_hydrogen_population = np.asarray(
                molecular_populations[:, int(hydrogen_molecule_indices[0])],
                np.float64,
            )
        bridge_end("bridge.molecular_hydrogen_population", t_bridge)
        t_bridge = bridge_start()
        line_populations_by_species = (
            _molecular_equilibrium.molecular_line_populations_by_species_code(
                temperature=temperature_array,
                equation_densities=molecular_equation_densities,
                neutral_partition=bridge_partition_functions[:, :, 0],
                species_codes=np.unique(molecular_species_codes),
                molecules_path=molecules_path,
            )
        )
        for (
            species_code,
            molecular_line_population,
        ) in line_populations_by_species.items():
            element_index = int(species_code) // 6 - 1
            if 0 <= element_index < partition_normalized_populations.shape[2]:
                partition_normalized_populations[:, 5, element_index] = np.asarray(
                    molecular_line_population, np.float64
                )
        bridge_end("bridge.molecular_line_populations_by_species", t_bridge)

    # (3) Doppler widths, (4) the edge grid, (5) the thermodynamic helpers.
    t_bridge = bridge_start()
    doppler = compute_doppler_per_ion(temperature_array, microturbulence, atomic_masses)
    edges = _build_edge_grid()
    hc_over_kt = (
        PLANCK_ERG_SECOND
        * LIGHT_SPEED_CM_PER_S
        / (BOLTZMANN_ERG_PER_K * temperature_array)
    )
    bridge_end("bridge.doppler_edges", t_bridge)

    t_bridge = bridge_start()
    struct = {
        "atmosphere_schema_version": np.asarray(
            [ATMOSPHERE_SCHEMA_VERSION],
            dtype=np.int32,
        ),
        "temperature": temperature_array,
        "column_mass": column_mass,
        "gas_pressure": gas_pressure_array,
        "electron_density": np.asarray(population_state.electron_density, np.float64),
        "mass_density": np.asarray(population_state.mass_density, np.float64),
        "microturbulence": microturbulence,
        "hc_over_kt": hc_over_kt,
        "hydrogen_neutral_population": hydrogen_neutral_population,
        "hydrogen_ionized_population": hydrogen_ionized_population,
        "hydrogen_partition_normalized_ion_stage_populations": hydrogen_partition_normalized_ion_stage_populations,
        "helium_neutral_population": helium_neutral_population,
        "helium_singly_ionized_population": helium_singly_ionized_population,
        "molecular_hydrogen_population": molecular_hydrogen_population,
        "carbon_partition_normalized_ion_stage_populations": carbon_partition_normalized_ion_stage_populations,
        "magnesium_neutral_partition_normalized_population": magnesium_neutral_partition_normalized_population,
        "aluminum_neutral_partition_normalized_population": aluminum_neutral_partition_normalized_population,
        "silicon_neutral_partition_normalized_population": silicon_neutral_partition_normalized_population,
        "iron_neutral_partition_normalized_population": iron_neutral_partition_normalized_population,
        "partition_normalized_populations": partition_normalized_populations,
        "ion_stage_populations": ion_stage_populations,
        "fractional_doppler_widths": doppler,
        "elemental_abundances": elemental_abundances,
        "signed_continuum_edge_frequency_hz": edges[
            "signed_continuum_edge_frequency_hz"
        ],
        "continuum_edge_wavelength_nm": edges["continuum_edge_wavelength_nm"],
        "continuum_edge_midpoint_wavelength_nm": edges[
            "continuum_edge_midpoint_wavelength_nm"
        ],
        "continuum_edge_interval_width_squared_over_two_nm2": edges[
            "edge_interval_width_squared_over_two_nm2"
        ],
    }
    bridge_end("bridge.struct_pack", t_bridge)
    return struct


# ----------------------------------------------------------------------
#  Device-resident window invariants and their in-process cache.
# ----------------------------------------------------------------------
@dataclass
class WindowInvariants:
    """Everything a synthesis window compiles/uploads that is independent of
    the atmosphere: the geometric grid, the compiled atomic/molecular line
    catalogs, the device-resident invariant tensors, and the physics tables.

    ``synthesis_wavelength_nm`` includes the native boundary context used by
    kernels; ``wavelength_nm`` and ``n_wl`` describe only the exact requested
    public grid, selected by ``output_slice``.

    One bundle serves every ``SynthesisPipeline`` over the same
    (window, resolution, device, dtype, molecular on/off, catalog identity,
    metal chunking). The hydrogen entry is a TEMPLATE: its only
    atmosphere-dependent field (``merge_wavenumber_by_depth``) is replaced per
    pipeline via ``hydrogen_lines.merge_wavenumber_by_depth``.
    """

    key: tuple
    device: torch.device
    dtype: torch.dtype
    molecular_lines: bool
    metal_chunk: int
    grid_obj: "atomic_lines.Grid"
    synthesis_wavelength_nm: np.ndarray
    wavelength_nm: np.ndarray
    output_slice: slice
    n_synthesis_wl: int
    n_wl: int
    n_atomic: int
    atomic_kernel_catalog: dict
    has_metal: bool
    has_helium: bool
    has_hydrogen: bool
    continuum_tables: "continuum_engine.ContinuumTables"
    transfer_tables: "radiative_transfer.TransferTables"
    metal_invariant_chunks: list
    helium_invariants: Optional["line_opacity_engine.AtomicInvariants"]
    hydrogen_invariants_template: Optional["hydrogen_lines.HydrogenInvariants"]
    molecular_invariants: Optional["molecular_lines_engine.MolecularLineInvariants"]
    n_molecular: int
    build_profile: dict


# In-process cache: repeated synthesize() calls over the same window reuse the
# device-resident invariants instead of rebuilding/re-uploading them per call.
# Keyed ONLY by physical inputs (window bounds, resolution, device, dtype,
# molecular on/off, source-catalog/table file identity) plus the metal line
# chunk width (which shapes the compiled chunk structure; chunking is
# accumulation-order-only). Deleting or disabling the cache never changes
# physics: a hit returns tensors value-identical to a fresh build.
_WINDOW_INVARIANT_CACHE: dict[tuple, WindowInvariants] = {}


def window_invariant_cache_enabled() -> bool:
    """Cache switch: PAYNE_ZERO_SYNTHESIS_DISABLE_INVARIANT_CACHE=1 disables it (A/B checks)."""
    return os.environ.get("PAYNE_ZERO_SYNTHESIS_DISABLE_INVARIANT_CACHE", "0") != "1"


def clear_window_invariant_cache() -> None:
    """Drop every cached window-invariant bundle (frees device memory)."""
    _WINDOW_INVARIANT_CACHE.clear()


def _molecular_compiled_cache_contract(
    start_wavelength_nm: float,
    end_wavelength_nm: float,
    resolution: float,
    *,
    cache_enabled: bool | None = None,
) -> tuple[Path | None, dict]:
    """Return the exact persistent product path and its source-bound identity.

    Prewarming and the runtime compiler share this calculation so a completed
    window manifest cannot accidentally accept a cache for another window or
    another source-catalog version.
    """
    if cache_enabled is None:
        cache_enabled = os.environ.get(
            "PAYNE_ZERO_SYNTHESIS_MOLECULAR_COMPILED_CACHE", "1"
        ) not in ("0", "false", "False", "no", "off", "")

    molecular_source_dir = runtime_paths.source_catalog_path("molecules")
    source_paths = (
        molecular_source_dir / "molecular_band_lines.npz",
        molecular_source_dir / "titanium_oxide_lines.npy",
    )
    checksums_path = molecular_source_dir.parent / "CHECKSUMS.sha256"
    source_checksums: dict[str, str] = {}
    if checksums_path.is_file():
        for line in checksums_path.read_text().splitlines():
            fields = line.split(maxsplit=1)
            if len(fields) == 2:
                source_checksums[fields[1].removeprefix("./")] = fields[0]

    source_file_identity: list[list[str | int]] = []
    for path in source_paths:
        relative_name = f"molecules/{path.name}"
        checksum = source_checksums.get(relative_name)
        if checksum is not None:
            source_file_identity.append([relative_name, checksum])
            continue
        # Older external trees without CHECKSUMS retain a safe, non-portable
        # stat identity.  Missing files remain part of the identity so the
        # subsequent source rebuild fails with the original useful context.
        try:
            source_stat = path.stat()
            source_file_identity.append(
                [
                    str(path.resolve()),
                    int(source_stat.st_size),
                    int(
                        getattr(
                            source_stat,
                            "st_mtime_ns",
                            int(source_stat.st_mtime * 1e9),
                        )
                    ),
                ]
            )
        except OSError:
            source_file_identity.append([str(path), 0, 0])

    identity = {
        "schema": 3,
        "start_wavelength_nm": float(start_wavelength_nm),
        "end_wavelength_nm": float(end_wavelength_nm),
        "resolution": float(resolution),
        "use_energy_level_wavelengths": True,
        "compiler": "payne_zero_synthesis.source_catalog_molecular_compiler",
        "files": source_file_identity,
    }
    if not cache_enabled:
        return None, identity
    cache_dir = Path(
        os.environ.get(
            "PAYNE_ZERO_SYNTHESIS_MOLECULAR_COMPILED_CACHE_DIR",
            str(runtime_paths.PACKAGE_CACHE_ROOT / "molecular_compiled"),
        )
    ).expanduser()
    cache_digest = hashlib.blake2b(
        json.dumps(identity, sort_keys=True).encode("utf-8"), digest_size=16
    ).hexdigest()
    return cache_dir / f"compiled_molecules_{cache_digest}.npz", identity


def _file_identity(path: Path) -> tuple:
    """(resolved path, size, mtime_ns) fingerprint of one physical input file."""
    path = Path(path)
    try:
        file_stat = path.stat()
        return (
            str(path.resolve()),
            int(file_stat.st_size),
            int(getattr(file_stat, "st_mtime_ns", int(file_stat.st_mtime * 1e9))),
        )
    except OSError:
        return (str(path), 0, 0)


def _window_invariant_key(
    wl_start_nm: float,
    wl_end_nm: float,
    resolution: float,
    molecular_lines: bool,
    runtime_device: torch.device,
    work_dtype: torch.dtype,
    tables_path: Path,
    transfer_tables_path: Path,
    continuum_tables_path: Path,
    metal_chunk: int,
) -> tuple:
    """Cache key over the physical inputs that determine the invariant bundle."""
    input_files = [
        _file_identity(
            runtime_paths.source_catalog_path("lines", "atomic_source_lines_parsed.npz")
        ),
        _file_identity(tables_path),
        _file_identity(transfer_tables_path),
        _file_identity(continuum_tables_path),
    ]
    if molecular_lines:
        molecular_source_dir = runtime_paths.source_catalog_path("molecules")
        input_files.extend(
            [
                _file_identity(molecular_source_dir / "manifest.json"),
                _file_identity(molecular_source_dir / "molecular_band_lines.npz"),
                _file_identity(molecular_source_dir / "titanium_oxide_lines.npy"),
            ]
        )
    return (
        float(wl_start_nm),
        float(wl_end_nm),
        float(resolution),
        bool(molecular_lines),
        str(runtime_device),
        str(work_dtype),
        int(metal_chunk),
        ("symmetric_native_context", WINDOW_CONTEXT_SAMPLES),
        tuple(input_files),
    )


def _window_grid_contract(
    wl_start_nm: float,
    wl_end_nm: float,
    resolution: float,
) -> tuple[
    "atomic_lines.Grid",
    "atomic_lines.Grid",
    np.ndarray,
    np.ndarray,
    slice,
]:
    """Return requested and internally padded geometric-grid products.

    The requested grid is built exactly once with the established ``Grid``
    implementation. Context samples are then grown outward from its endpoints,
    rather than regenerating the target samples from a different lower bound;
    consequently the interior is bitwise identical to the historical requested
    wavelength array.
    """

    requested_grid = atomic_lines.Grid(wl_start_nm, wl_end_nm, resolution)
    requested_wavelength_nm = requested_grid.build()
    if requested_wavelength_nm.size == 0:
        raise ValueError("requested wavelength window contains no grid samples")

    ratio = requested_grid.ratio
    blue_context = np.empty(WINDOW_CONTEXT_SAMPLES, dtype=np.float64)
    wavelength_nm = float(requested_wavelength_nm[0])
    for offset in range(WINDOW_CONTEXT_SAMPLES - 1, -1, -1):
        wavelength_nm /= ratio
        blue_context[offset] = wavelength_nm
    red_context = np.empty(WINDOW_CONTEXT_SAMPLES, dtype=np.float64)
    wavelength_nm = float(requested_wavelength_nm[-1])
    for offset in range(WINDOW_CONTEXT_SAMPLES):
        wavelength_nm *= ratio
        red_context[offset] = wavelength_nm

    synthesis_wavelength_nm = np.concatenate(
        (blue_context, requested_wavelength_nm, red_context)
    )
    # The context Grid supplies catalog-window metadata. Its build is not used
    # for synthesis because rebuilding from the wider blue bound can move the
    # requested float64 samples by a few ulps.
    synthesis_grid = atomic_lines.Grid(
        float(np.nextafter(synthesis_wavelength_nm[0], -np.inf)),
        float(np.nextafter(synthesis_wavelength_nm[-1], np.inf)),
        resolution,
    )
    output_slice = slice(
        WINDOW_CONTEXT_SAMPLES,
        WINDOW_CONTEXT_SAMPLES + requested_wavelength_nm.size,
    )
    if not np.array_equal(
        synthesis_wavelength_nm[output_slice], requested_wavelength_nm
    ):  # pragma: no cover - construction invariant
        raise RuntimeError("synthesis context changed the requested wavelength grid")
    return (
        requested_grid,
        synthesis_grid,
        requested_wavelength_nm,
        synthesis_wavelength_nm,
        output_slice,
    )


class SynthesisPipeline:
    """Device-resident synthesis pipeline for one geometric wavelength_nm grid."""

    # Metal wing chunks keep the full optical window under the local 24 GB MPS pool.
    # Larger chunks launch less often but raise peak memory.
    METAL_CHUNK = 40_000

    @staticmethod
    def _slice_atomic_catalog(
        atomic_kernel_catalog: dict,
        line_indices: np.ndarray,
    ) -> dict:
        """A per-line-index view of the atomic catalog plus shared tables.

        Per-line catalog arrays are sliced to ``line_indices``; static profile
        tables pass through whole.
        """
        line_indices = np.asarray(line_indices)
        per_line_keys = {
            "wavelength_nm",
            "index_wavelength_nm",
            "oscillator_strength",
            "log_oscillator_strength",
            "lower_excitation_cm",
            "radiative_damping",
            "stark_damping",
            "van_der_waals_damping",
            "raw_radiative_damping_log",
            "raw_stark_damping_log",
            "raw_van_der_waals_damping_log",
            "ion_stage",
            "atomic_number",
            "species_code",
            "line_size",
            "line_type",
            "lower_principal_quantum_number",
            "upper_principal_quantum_number",
        }
        catalog_slice = {}
        for field_name, field_value in atomic_kernel_catalog.items():
            if field_name in per_line_keys:
                catalog_slice[field_name] = np.asarray(field_value)[line_indices]
            else:
                catalog_slice[field_name] = field_value
        line_type_slice = catalog_slice["line_type"]
        catalog_slice["helium_line_type"] = line_type_slice[
            np.isin(line_type_slice, [-3, -4, -6])
        ].astype(np.int64)
        catalog_slice["helium_line_center_cutoff_ratio"] = atomic_kernel_catalog.get(
            "helium_line_center_cutoff_ratio",
            np.float64(line_opacity_engine.LINE_CENTER_CUTOFF_RATIO),
        )
        return catalog_slice

    def __init__(
        self,
        atmosphere: Union[Path, str, dict, "np.lib.npyio.NpzFile"],
        source_path: Optional[Path] = None,
        wl_start_nm: float = 400.0,
        wl_end_nm: float = 900.0,
        resolution: float = 20000.0,
        tables_path: Path = _SYNTHESIS_TABLE_DIR / "line_profile_tables.npz",
        transfer_tables_path: Path = _SYNTHESIS_TABLE_DIR / "transfer_tables.npz",
        continuum_tables_path: Path = _SYNTHESIS_TABLE_DIR / "continuum_tables.npz",
        molecular_lines: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        metal_chunk: Optional[int] = None,
        window_invariants: Optional[WindowInvariants] = None,
    ):
        self.device, self.dtype = resolve_runtime(device, dtype)
        runtime_device = self.device
        self.molecular_lines = molecular_lines

        def start_init_profile() -> float:
            return time.perf_counter()

        def end_init_profile(name: str, t0: float) -> None:
            return None

        # Window-invariant bundle: geometric grid, compiled catalogs,
        # device-resident invariant tensors, and physics tables — everything
        # independent of the atmosphere. Resolved through the in-process
        # cache (kill switch: PAYNE_ZERO_SYNTHESIS_DISABLE_INVARIANT_CACHE=1)
        # or taken from an explicitly shared bundle (the batched path).
        self.metal_chunk = (
            int(metal_chunk) if metal_chunk is not None else int(self.METAL_CHUNK)
        )
        t_init = start_init_profile()
        if window_invariants is not None:
            bundle = window_invariants
            requested_key_head = (
                float(wl_start_nm),
                float(wl_end_nm),
                float(resolution),
                bool(molecular_lines),
                str(runtime_device),
                str(self.dtype),
                int(self.metal_chunk),
            )
            if bundle.key[:7] != requested_key_head:
                raise ValueError(
                    "window_invariants bundle does not match the requested "
                    f"window configuration: bundle {bundle.key[:7]} vs "
                    f"requested {requested_key_head}"
                )
        else:
            bundle = window_invariants_for(
                wl_start_nm=wl_start_nm,
                wl_end_nm=wl_end_nm,
                resolution=resolution,
                molecular_lines=molecular_lines,
                runtime_device=runtime_device,
                work_dtype=self.dtype,
                tables_path=tables_path,
                transfer_tables_path=transfer_tables_path,
                continuum_tables_path=continuum_tables_path,
                metal_chunk=self.metal_chunk,
            )
        self._window_invariants = bundle
        self.grid_obj = bundle.grid_obj
        self.wavelength_nm = bundle.wavelength_nm
        self.synthesis_wavelength_nm = bundle.synthesis_wavelength_nm
        self.output_slice = bundle.output_slice
        self.n_wl = bundle.n_synthesis_wl
        self.n_atomic = bundle.n_atomic
        self._atomic_kernel_catalog = bundle.atomic_kernel_catalog
        self.has_metal = bundle.has_metal
        self.has_helium = bundle.has_helium
        self.has_hydrogen = bundle.has_hydrogen
        self.continuum_tables = bundle.continuum_tables
        self._metal_invariant_chunks = bundle.metal_invariant_chunks
        self._helium_invariants = bundle.helium_invariants
        self.molecular_invariants = bundle.molecular_invariants
        self.n_molecular = bundle.n_molecular
        self.transfer_tables = bundle.transfer_tables
        end_init_profile("init.window_invariants", t_init)

        # Structured atmospheres enter the pipeline with native public names.
        t_init = start_init_profile()
        if isinstance(atmosphere, (str, Path)):
            atm = load_atmosphere_npz(atmosphere)
        else:
            atm = atmosphere
        self._atm = atm
        continuum_atmosphere_fields = (
            "temperature",
            "mass_density",
            "electron_density",
            "hydrogen_partition_normalized_ion_stage_populations",
            "hydrogen_neutral_population",
            "helium_neutral_population",
            "helium_singly_ionized_population",
            "carbon_partition_normalized_ion_stage_populations",
            "magnesium_neutral_partition_normalized_population",
            "aluminum_neutral_partition_normalized_population",
            "silicon_neutral_partition_normalized_population",
            "iron_neutral_partition_normalized_population",
            "partition_normalized_populations",
            "ion_stage_populations",
            "signed_continuum_edge_frequency_hz",
            "continuum_edge_wavelength_nm",
            "continuum_edge_midpoint_wavelength_nm",
            "continuum_edge_interval_width_squared_over_two_nm2",
        )
        self._continuum_atmosphere = {
            field_name: np.asarray(atm[field_name])
            for field_name in continuum_atmosphere_fields
        }
        temperature = np.asarray(atm["temperature"], np.float64)
        self.n_depths = temperature.size
        electron_density = np.asarray(atm["electron_density"], np.float64)
        end_init_profile("init.atmosphere_view", t_init)

        # Hydrogen invariants: the shared window template plus this
        # atmosphere's Inglis-Teller merge limits — the template's only
        # atmosphere-dependent field, rebuilt here on every call so cached
        # bundles never carry per-star state.
        t_init = start_init_profile()
        self.hydrogen_invariants = (
            dataclasses.replace(
                bundle.hydrogen_invariants_template,
                merge_wavenumber_by_depth=hydrogen_lines.merge_wavenumber_by_depth(
                    electron_density
                ),
            )
            if bundle.hydrogen_invariants_template is not None
            else None
        )
        end_init_profile("init.hydrogen_invariants", t_init)

        t_init = start_init_profile()
        column_mass = np.asarray(atm["column_mass"], np.float64)
        self.column_mass = torch.as_tensor(
            column_mass, dtype=self.dtype, device=runtime_device
        )
        end_init_profile("init.column_mass", t_init)

        # Per-depth state stays on host until each kernel uploads the arrays it needs.
        t_init = start_init_profile()

        def as_host_float64_array(array_like):
            return np.asarray(array_like, np.float64)

        collision_density_proxy = (
            as_host_float64_array(atm["hydrogen_neutral_population"])
            + 0.42 * as_host_float64_array(atm["helium_neutral_population"])
            + 0.85 * as_host_float64_array(atm["molecular_hydrogen_population"])
        ) * (temperature / 1e4) ** 0.3
        self._partition_normalized_populations = as_host_float64_array(
            atm["partition_normalized_populations"]
        )
        self._fractional_doppler_widths = as_host_float64_array(
            atm["fractional_doppler_widths"]
        )
        self._mass_density = as_host_float64_array(atm["mass_density"])
        self._electron_density = electron_density
        self._temperature = temperature
        self._hc_over_kt = as_host_float64_array(atm["hc_over_kt"])
        self._collision_density_proxy = collision_density_proxy
        self._microturbulence = as_host_float64_array(atm["microturbulence"])
        self._helium_neutral_population = as_host_float64_array(
            atm["helium_neutral_population"]
        )
        self._molecular_hydrogen_population = as_host_float64_array(
            atm["molecular_hydrogen_population"]
        )
        self._hydrogen_partition_normalized_ion_stage_populations = (
            as_host_float64_array(
                atm["hydrogen_partition_normalized_ion_stage_populations"]
            )
        )
        end_init_profile("init.depth_state", t_init)

        # Optional exact transfer source; otherwise LTE Planck B_nu is rebuilt.
        t_init = start_init_profile()
        self._source_from_ref = source_path is not None
        if self._source_from_ref:
            source_reference_npz = np.load(source_path)
            self._line_source = torch.as_tensor(
                as_host_float64_array(source_reference_npz["line_source"]),
                dtype=self.dtype,
                device=runtime_device,
            )
            self._line_scattering_ref = torch.as_tensor(
                as_host_float64_array(source_reference_npz["line_scattering"]),
                dtype=self.dtype,
                device=runtime_device,
            )
        else:
            self._line_source = None
            self._line_scattering_ref = None
        self._temperature_device = torch.as_tensor(
            temperature,
            dtype=self.dtype,
            device=runtime_device,
        )
        end_init_profile("init.source_state", t_init)

    # ------------------------------------------------------------------
    @staticmethod
    def _compile_molecular(start_wavelength_nm, end_wavelength_nm, resolution) -> dict:
        """Compile the full-window molecular text + TiO line arrays.

        The optical parity grid is shipped as a compiled structure-of-arrays
        asset. Other windows require an explicit external source-catalog root
        and write derived caches outside the package.
        """
        from . import source_catalog_molecular_compiler as mc

        molecular_source_dir = runtime_paths.source_catalog_path("molecules")
        band_catalog_path = molecular_source_dir / "molecular_band_lines.npz"
        titanium_oxide_binary_path = molecular_source_dir / "titanium_oxide_lines.npy"
        cache_enabled = os.environ.get(
            "PAYNE_ZERO_SYNTHESIS_MOLECULAR_COMPILED_CACHE",
            "1",
        ) not in (
            "0",
            "false",
            "False",
            "no",
            "off",
            "",
        )
        rebuild_cache = os.environ.get(
            "PAYNE_ZERO_SYNTHESIS_REBUILD_MOLECULAR_COMPILED_CACHE",
            "0",
        ) in (
            "1",
            "true",
            "True",
            "yes",
            "on",
        )
        cache_path, cache_identity = _molecular_compiled_cache_contract(
            start_wavelength_nm,
            end_wavelength_nm,
            resolution,
            cache_enabled=cache_enabled,
        )
        if cache_path is not None:
            if cache_path.exists() and not rebuild_cache:
                try:
                    with np.load(cache_path, allow_pickle=False) as compiled_cache_npz:
                        if "__meta__" not in compiled_cache_npz.files:
                            raise ValueError("compiled molecular cache lacks metadata")
                        meta = json.loads(
                            bytes(compiled_cache_npz["__meta__"]).decode("utf-8")
                        )
                        if meta.get("cache_identity") != cache_identity:
                            raise ValueError(
                                "compiled molecular cache identity does not match"
                            )
                        compiled = {
                            field_name: np.asarray(compiled_cache_npz[field_name])
                            for field_name in compiled_cache_npz.files
                            if field_name != "__meta__"
                        }
                        compiled["log_grid_ratio"] = float(meta["log_grid_ratio"])
                        compiled["grid_origin_index"] = int(meta["grid_origin_index"])
                    return compiled
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    # A persistent cache is only an acceleration.  Rebuild a
                    # corrupt or identity-mismatched product from its sources.
                    pass
        # Raw multi-GB catalogs are needed only on a cache miss.  A portable
        # compiled cache can therefore be installed or copied to a compute node
        # without also copying the molecular source archive.
        molecular_manifest = json.loads(
            (molecular_source_dir / "manifest.json").read_text()
        )
        band_names = [entry["band"] for entry in molecular_manifest["text_sources"]]
        text_catalog = mc.compile_molecular_text(
            band_catalog_path=band_catalog_path,
            band_names=band_names,
            start_wavelength_nm=start_wavelength_nm,
            end_wavelength_nm=end_wavelength_nm,
            resolution=resolution,
            use_energy_level_wavelengths=True,
        )
        tio_catalog = mc.compile_tio_schwenke(
            bin_path=titanium_oxide_binary_path,
            start_wavelength_nm=start_wavelength_nm,
            end_wavelength_nm=end_wavelength_nm,
            resolution=resolution,
            use_vacuum_wavelengths=True,
        )
        compiled = {
            field_name: np.concatenate(
                [text_catalog[field_name], tio_catalog[field_name]]
            )
            for field_name in text_catalog
        }
        log_grid_ratio = float(np.log(1.0 + 1.0 / resolution))
        grid_origin_index = int(np.floor(np.log(start_wavelength_nm) / log_grid_ratio))
        if np.exp(grid_origin_index * log_grid_ratio) < start_wavelength_nm:
            grid_origin_index += 1
        compiled["log_grid_ratio"] = np.asarray(log_grid_ratio)
        compiled["grid_origin_index"] = np.asarray(grid_origin_index)
        if cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                meta = {
                    "cache_identity": cache_identity,
                    "start_wavelength_nm": float(start_wavelength_nm),
                    "end_wavelength_nm": float(end_wavelength_nm),
                    "resolution": float(resolution),
                    "log_grid_ratio": log_grid_ratio,
                    "grid_origin_index": grid_origin_index,
                    "n_lines": int(np.asarray(compiled["center_index_1based"]).size),
                }
                np.savez(
                    cache_path,
                    **compiled,
                    __meta__=np.frombuffer(
                        json.dumps(meta, sort_keys=True).encode("utf-8"),
                        dtype=np.uint8,
                    ),
                )
            except Exception:
                pass  # cache is an acceleration, not part of the physics contract
        return compiled

    # ------------------------------------------------------------------
    #  The forward pass.
    # ------------------------------------------------------------------
    def run(
        self,
        keep_slabs: bool = False,
        spectral_operator=None,
    ) -> SpectrumResult:
        """One device-resident forward pass over the synthesis context grid.

        ``keep_slabs=False`` by default. Everything stays on ``self.device``
        through the exact requested-grid crop; returned slabs never expose the
        internal context samples.
        """
        runtime_device, work_dtype = self.device, self.dtype

        def start_profile() -> float:
            return time.perf_counter()

        def end_profile(name: str, t0: float) -> None:
            return None

        # Stage 1: continuum opacity on the synthesis grid.
        t_stage = start_profile()
        continuum_absorption, continuum_scattering = continuum_engine.continuum(
            self.synthesis_wavelength_nm,
            self._continuum_atmosphere,
            self.continuum_tables,
        )
        continuum_total = continuum_absorption + continuum_scattering
        end_profile("continuum", t_stage)

        # Stage 2: one shared line-absorption slab for atomic, H, He, and molecular opacity.
        t_stage = start_profile()
        line_mass_absorption_coefficient = torch.zeros(
            self.n_depths,
            self.n_wl,
            dtype=line_opacity_engine.ACCUMULATION_DTYPE,
            device=runtime_device,
        )
        end_profile("line.init", t_stage)

        atomic_state = {
            "partition_normalized_populations": self._partition_normalized_populations,
            "fractional_doppler_widths": self._fractional_doppler_widths,
            "mass_density": self._mass_density,
            "electron_density": self._electron_density,
            "temperature": self._temperature,
            "hc_over_kt": self._hc_over_kt,
            "collision_density_proxy": self._collision_density_proxy,
            "continuum_opacity": continuum_total,
            "helium_core_weight_grid": None,
            "helium_tail_weight_grid": None,
        }
        # Metal lines are chunked by line so the wing-walk temporary tensors stay bounded.
        t_stage = start_profile()
        for metal_invariants in self._metal_invariant_chunks:
            line_mass_absorption_coefficient = (
                line_mass_absorption_coefficient
                + line_opacity_engine.accumulate_atomic(
                    metal_invariants,
                    atomic_state,
                    do_metal=True,
                    do_helium=False,
                    apply_stim=True,
                )
            )
        end_profile("line.atomic_metal", t_stage)
        if self.has_helium and self._helium_invariants is not None:
            t_stage = start_profile()
            line_mass_absorption_coefficient = (
                line_mass_absorption_coefficient
                + line_opacity_engine.accumulate_atomic(
                    self._helium_invariants,
                    atomic_state,
                    do_metal=False,
                    do_helium=True,
                    apply_stim=True,
                )
            )
            end_profile("line.helium", t_stage)

        if self.has_hydrogen and self.hydrogen_invariants is not None:
            hydrogen_state = {
                "temperature": self._temperature,
                "electron_density": self._electron_density,
                "mass_density": self._mass_density,
                "hc_over_kt": self._hc_over_kt,
                "helium_neutral_population": self._helium_neutral_population,
                "molecular_hydrogen_population": self._molecular_hydrogen_population,
                "hydrogen_partition_normalized_ion_stage_populations": (
                    self._hydrogen_partition_normalized_ion_stage_populations
                ),
                "microturbulence": self._microturbulence,
                "hydrogen_neutral_partition_normalized_population": (
                    self._partition_normalized_populations[:, 0, 0]
                ),
                "hydrogen_fractional_doppler_width": self._fractional_doppler_widths[
                    :, 0, 0
                ],
                "continuum_opacity": continuum_total,
            }
            t_stage = start_profile()
            line_mass_absorption_coefficient = (
                line_mass_absorption_coefficient
                + hydrogen_lines.accumulate_hydrogen(
                    self.hydrogen_invariants,
                    hydrogen_state,
                    apply_stim=True,
                )
            )
            end_profile("line.hydrogen", t_stage)

        if self.molecular_lines and self.molecular_invariants is not None:
            molecular_state = {
                "partition_normalized_populations": (
                    self._partition_normalized_populations
                ),
                "mass_density": self._mass_density,
                "electron_density": self._electron_density,
                "temperature": self._temperature,
                "hc_over_kt": self._hc_over_kt,
                "microturbulence": self._microturbulence,
                "collision_density_proxy": self._collision_density_proxy,
                "continuum_opacity": continuum_total,
            }
            t_stage = start_profile()
            molecular_absorption = molecular_lines_engine.accumulate_molecular(
                self.molecular_invariants,
                molecular_state,
                apply_stim=True,
            )
            line_mass_absorption_coefficient = (
                line_mass_absorption_coefficient
                + molecular_absorption.to(line_opacity_engine.ACCUMULATION_DTYPE)
            )
            end_profile("line.molecular", t_stage)

        # Stage 3: assemble the source and solve radiative transfer.
        t_stage = start_profile()
        if self._source_from_ref:
            line_source = self._line_source
            line_scattering = self._line_scattering_ref
        else:
            wavelength_tensor = torch.as_tensor(
                self.synthesis_wavelength_nm,
                dtype=work_dtype,
                device=runtime_device,
            )
            line_source = radiative_transfer.planck_bnu(
                wavelength_tensor, self._temperature_device
            )
            line_scattering = torch.zeros(
                self.n_depths,
                self.n_wl,
                dtype=work_dtype,
                device=runtime_device,
            )
        end_profile("source", t_stage)

        t_stage = start_profile()
        (
            eddington_flux_total_per_frequency,
            eddington_flux_continuum_per_frequency,
            normalized_flux,
        ) = radiative_transfer.solve_spectrum(
            continuum_absorption.to(work_dtype),
            continuum_scattering.to(work_dtype),
            line_mass_absorption_coefficient.to(work_dtype),
            line_scattering,
            line_source,
            self.column_mass,
            self.transfer_tables,
            assert_no_saturated_core=False,
        )
        end_profile("rt", t_stage)

        # The line/profile/transfer solve owns symmetric native-grid context,
        # but every result boundary owns exactly the caller's requested grid.
        # Crop on device so context samples never become hidden public output or
        # an unexpected input shape for a prepared instrument operator.
        eddington_flux_total_per_frequency = eddington_flux_total_per_frequency[
            self.output_slice
        ]
        eddington_flux_continuum_per_frequency = eddington_flux_continuum_per_frequency[
            self.output_slice
        ]
        normalized_flux = normalized_flux[self.output_slice]

        # Stage 4 is identity unless the caller supplies a device-resident
        # spectral operator. Instrument operators consume wavelength-density
        # total and continuum flux together so normalized output is their
        # convolved ratio; no spectrum returns to the host between radiative
        # transfer and convolution.
        output_wavelength_nm = self.wavelength_nm
        spectral_operator_seconds = 0.0
        spectral_operator_name = None
        if spectral_operator is not None:
            (
                eddington_flux_total_per_frequency,
                eddington_flux_continuum_per_frequency,
                normalized_flux,
                output_wavelength_nm,
            ) = _apply_spectral_operator_in_wavelength_density(
                eddington_flux_total_per_frequency,
                eddington_flux_continuum_per_frequency,
                self.wavelength_nm,
                spectral_operator,
            )
            spectral_operator_seconds = float(spectral_operator.last_seconds)
            spectral_operator_name = str(spectral_operator.name)

        def tensor_to_host_float64(tensor):
            return tensor.detach().to("cpu").numpy().astype(np.float64)

        t_stage = start_profile()
        result = SpectrumResult(
            wavelength_nm=output_wavelength_nm,
            eddington_flux_total_per_frequency=tensor_to_host_float64(
                eddington_flux_total_per_frequency
            ),
            eddington_flux_continuum_per_frequency=tensor_to_host_float64(
                eddington_flux_continuum_per_frequency
            ),
            normalized_flux=tensor_to_host_float64(normalized_flux),
            continuum_absorption=(
                tensor_to_host_float64(continuum_absorption[:, self.output_slice])
                if keep_slabs
                else None
            ),
            continuum_scattering=(
                tensor_to_host_float64(continuum_scattering[:, self.output_slice])
                if keep_slabs
                else None
            ),
            line_mass_absorption_coefficient=tensor_to_host_float64(
                line_mass_absorption_coefficient[:, self.output_slice]
            )
            if keep_slabs
            else None,
            line_source=(
                tensor_to_host_float64(line_source[:, self.output_slice])
                if keep_slabs
                else None
            ),
            spectral_operator_seconds=spectral_operator_seconds,
            spectral_operator_name=spectral_operator_name,
        )
        end_profile("host_copy", t_stage)
        return result


# ----------------------------------------------------------------------
#  Window-invariant build + cache accessor.
# ----------------------------------------------------------------------
def _build_window_invariants(
    *,
    wl_start_nm: float,
    wl_end_nm: float,
    resolution: float,
    molecular_lines: bool,
    runtime_device: torch.device,
    work_dtype: torch.dtype,
    tables_path: Path,
    transfer_tables_path: Path,
    continuum_tables_path: Path,
    metal_chunk: int,
    key: tuple,
) -> WindowInvariants:
    """Compile + upload every atmosphere-independent invariant for one window.

    This is the code that used to run inside ``SynthesisPipeline.__init__`` on
    every call; the values are identical whether it runs fresh or is served
    from the in-process cache.
    """
    build_profile: dict[str, float] = {}

    def timed_section(name: str, t0: float) -> None:
        build_profile[name] = build_profile.get(name, 0.0) + (time.perf_counter() - t0)

    t_section = time.perf_counter()
    (
        _requested_grid_obj,
        grid_obj,
        wavelength_nm,
        synthesis_wavelength_nm,
        output_slice,
    ) = _window_grid_contract(wl_start_nm, wl_end_nm, resolution)
    n_wl = wavelength_nm.size
    n_synthesis_wl = synthesis_wavelength_nm.size
    timed_section("init.grid", t_section)

    # Every window compiles from the full converted source catalog
    # (cached outside the tree; the cache is deletable and bitwise-safe).
    t_section = time.perf_counter()
    atomic_catalog = atomic_lines.load_catalog(
        (grid_obj.start_wavelength_nm, grid_obj.end_wavelength_nm),
        grid_obj,
        catalog_path=runtime_paths.source_catalog_path(
            "lines", "atomic_source_lines_parsed.npz"
        ),
        sort="catalog",
    )
    n_atomic = len(atomic_catalog)
    timed_section("init.atomic_catalog", t_section)

    t_section = time.perf_counter()
    line_profile_tables = np.load(tables_path, allow_pickle=False)
    atomic_kernel_catalog = _atomic_catalog_for_kernels(
        atomic_catalog, line_profile_tables
    )
    harris_profile_h0_table = line_profile_tables["harris_profile_h0_table"]
    harris_profile_h1_table = line_profile_tables["harris_profile_h1_table"]
    harris_profile_h2_table = line_profile_tables["harris_profile_h2_table"]
    timed_section("init.atomic_catalog_mapping", t_section)

    t_section = time.perf_counter()
    line_type = atomic_kernel_catalog["line_type"]
    metal_type = (line_type == 0) | (line_type == 1) | (line_type == 3)
    has_metal = bool(np.any(metal_type))
    has_helium = bool(np.any(np.isin(line_type, [-3, -4, -6])))
    has_hydrogen = bool(np.any(np.isin(line_type, [-1, -2])))

    # The packaged optical catalog carries line centers but not helium merge-taper metadata.
    he_mask = np.isin(line_type, [-3, -4, -6])
    atomic_kernel_catalog["helium_line_type"] = line_type[he_mask].astype(np.int64)
    atomic_kernel_catalog["helium_line_center_cutoff_ratio"] = np.float64(
        line_opacity_engine.LINE_CENTER_CUTOFF_RATIO
    )
    timed_section("init.component_flags", t_section)

    t_section = time.perf_counter()
    continuum_tables = continuum_engine.ContinuumTables.from_npz(
        continuum_tables_path,
        device=runtime_device,
        dtype=work_dtype,
    )
    timed_section("init.continuum_tables", t_section)

    t_section = time.perf_counter()
    metal_line_indices = np.where(metal_type)[0]
    metal_invariant_chunks = []
    for chunk_start in range(0, metal_line_indices.size, metal_chunk):
        chunk_indices = metal_line_indices[chunk_start : chunk_start + metal_chunk]
        chunk_catalog = SynthesisPipeline._slice_atomic_catalog(
            atomic_kernel_catalog, chunk_indices
        )
        metal_invariant_chunks.append(
            line_opacity_engine.precompute_invariants(
                chunk_catalog,
                synthesis_wavelength_nm,
                runtime_device=runtime_device,
            )
        )
    timed_section("init.metal_invariants", t_section)

    t_section = time.perf_counter()
    helium_line_indices = np.where(he_mask)[0]
    helium_invariants = (
        line_opacity_engine.precompute_invariants(
            SynthesisPipeline._slice_atomic_catalog(
                atomic_kernel_catalog, helium_line_indices
            ),
            synthesis_wavelength_nm,
            runtime_device=runtime_device,
        )
        if has_helium
        else None
    )
    timed_section("init.he_invariants", t_section)

    # The hydrogen block is a TEMPLATE: `merge_wavenumber_by_depth` (its only
    # atmosphere-dependent field) is built with a placeholder here and replaced
    # per pipeline from the actual electron density.
    t_section = time.perf_counter()
    hydrogen_invariants_template = (
        hydrogen_lines.precompute_invariants(
            catalog=atomic_kernel_catalog,
            wavelength_grid_nm=synthesis_wavelength_nm,
            electron_density=np.ones(1, np.float64),
            compute_device=runtime_device,
        )
        if has_hydrogen
        else None
    )
    timed_section("init.hydrogen_invariants_template", t_section)

    molecular_invariants = None
    n_molecular = 0
    if molecular_lines:
        t_section = time.perf_counter()
        compiled = SynthesisPipeline._compile_molecular(
            grid_obj.start_wavelength_nm,
            grid_obj.end_wavelength_nm,
            resolution,
        )
        timed_section("init.molecular_compile", t_section)
        t_section = time.perf_counter()
        molecular_catalog = molecular_lines_engine.build_catalog_from_arrays(compiled)
        # Compiled molecular centers are relative to a regenerated catalog grid.
        # Re-index them onto the exact synthesis array, whose requested interior
        # is deliberately preserved bitwise.
        molecular_catalog.center_index = line_opacity_engine.nearest_grid_indices(
            synthesis_wavelength_nm,
            molecular_catalog.wavelength_nm,
        ).astype(np.int32)
        n_molecular = len(molecular_catalog)
        timed_section("init.molecular_catalog", t_section)
        t_section = time.perf_counter()
        molecular_invariants = molecular_lines_engine.precompute_invariants(
            catalog=molecular_catalog,
            wavelength_grid_nm=synthesis_wavelength_nm,
            harris_profile_h0_table=harris_profile_h0_table,
            harris_profile_h1_table=harris_profile_h1_table,
            harris_profile_h2_table=harris_profile_h2_table,
            runtime_device=runtime_device,
        )
        timed_section("init.molecular_invariants", t_section)

    t_section = time.perf_counter()
    transfer_tables = radiative_transfer.TransferTables.from_npz(
        transfer_tables_path,
        device=runtime_device,
        dtype=work_dtype,
    )
    timed_section("init.radiative_transfer_tables", t_section)

    return WindowInvariants(
        key=key,
        device=runtime_device,
        dtype=work_dtype,
        molecular_lines=molecular_lines,
        metal_chunk=int(metal_chunk),
        grid_obj=grid_obj,
        synthesis_wavelength_nm=synthesis_wavelength_nm,
        wavelength_nm=wavelength_nm,
        output_slice=output_slice,
        n_synthesis_wl=n_synthesis_wl,
        n_wl=n_wl,
        n_atomic=n_atomic,
        atomic_kernel_catalog=atomic_kernel_catalog,
        has_metal=has_metal,
        has_helium=has_helium,
        has_hydrogen=has_hydrogen,
        continuum_tables=continuum_tables,
        transfer_tables=transfer_tables,
        metal_invariant_chunks=metal_invariant_chunks,
        helium_invariants=helium_invariants,
        hydrogen_invariants_template=hydrogen_invariants_template,
        molecular_invariants=molecular_invariants,
        n_molecular=n_molecular,
        build_profile=build_profile,
    )


def window_invariants_for(
    *,
    wl_start_nm: float,
    wl_end_nm: float,
    resolution: float,
    molecular_lines: bool,
    runtime_device: torch.device,
    work_dtype: torch.dtype,
    tables_path: Path = _SYNTHESIS_TABLE_DIR / "line_profile_tables.npz",
    transfer_tables_path: Path = _SYNTHESIS_TABLE_DIR / "transfer_tables.npz",
    continuum_tables_path: Path = _SYNTHESIS_TABLE_DIR / "continuum_tables.npz",
    metal_chunk: Optional[int] = None,
) -> WindowInvariants:
    """The cache-aware accessor for one window's invariant bundle.

    Cache hits return the resident bundle; misses build it. Disable with
    PAYNE_ZERO_SYNTHESIS_DISABLE_INVARIANT_CACHE=1 (every call then builds
    fresh — identical values, only slower).
    """
    if metal_chunk is None:
        metal_chunk = SynthesisPipeline.METAL_CHUNK
    key = _window_invariant_key(
        wl_start_nm,
        wl_end_nm,
        resolution,
        molecular_lines,
        runtime_device,
        work_dtype,
        tables_path,
        transfer_tables_path,
        continuum_tables_path,
        metal_chunk,
    )
    cache_enabled = window_invariant_cache_enabled()
    if cache_enabled:
        cached_bundle = _WINDOW_INVARIANT_CACHE.get(key)
        if cached_bundle is not None:
            return cached_bundle
    bundle = _build_window_invariants(
        wl_start_nm=wl_start_nm,
        wl_end_nm=wl_end_nm,
        resolution=resolution,
        molecular_lines=molecular_lines,
        runtime_device=runtime_device,
        work_dtype=work_dtype,
        tables_path=tables_path,
        transfer_tables_path=transfer_tables_path,
        continuum_tables_path=continuum_tables_path,
        metal_chunk=metal_chunk,
        key=key,
    )
    if cache_enabled:
        _WINDOW_INVARIANT_CACHE[key] = bundle
    return bundle


# ----------------------------------------------------------------------
#  Batched forward pass: N structured atmospheres over ONE window.
# ----------------------------------------------------------------------
