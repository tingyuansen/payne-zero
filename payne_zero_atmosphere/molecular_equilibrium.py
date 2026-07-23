# ruff: noqa: E402
"""Molecular chemical equilibrium.

Solves the coupled dissociation equations over the equilibrium catalog
(molecular_data.py) for the cool-star population state; feeds both the
equation of state and the neutral-density correction used by the
structured-atmosphere handoff.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ._numba_cache import configure_numba_cache

# The compiled molecular-equilibrium kernels are the sole production
# path; numba is a hard requirement.
configure_numba_cache()
try:
    import numba
except ImportError as exc:  # pragma: no cover - numba is a hard requirement
    raise ImportError(
        "numba is required: the compiled molecular-equilibrium kernels are the "
        "sole production path for payne_zero_atmosphere."
    ) from exc

_NUMBA_AVAILABLE = True

from .constants import (
    BOLTZMANN_ERG_PER_K_REFERENCE,
    LIGHT_SPEED_CM_PER_S_REFERENCE as _LIGHT_SPEED_CM_PER_S_REFERENCE,
    PLANCK_ERG_SECOND_REFERENCE,
)
from .data_files import atmosphere_table_path

from .equation_of_state import (
    saha_partition_depth,
    saha_partition_depth_batch,
)
from .molecular_data import MolecularEquilibriumCatalog
from .constants import ATOMIC_MASS_GRAM_REFERENCE
from .runtime_state import REFERENCE_ATOMIC_MASS_AMU, AtmosphereRuntimeState


_MAX_NEWTON_ITERATIONS = 200
_NEWTON_TOLERANCE = 1.0e-4
_MOLECULAR_TABLE_PATH = atmosphere_table_path("molecular_equilibrium_tables.npz")
# Precomputed in Python (CPython float ** int and float ** float) so the
# compiled kernels reuse the exact doubles for these constant subexpressions.
_PLANCK_ERG_SECOND_SQUARED = PLANCK_ERG_SECOND_REFERENCE**2
_TWO_POW_THREE_HALVES = 2.0**1.5


@dataclass
class MolecularEquilibriumState:
    """Mutable molecular-equilibrium state."""

    temperature_k: np.ndarray
    thermal_energy_erg: np.ndarray
    gas_pressure: np.ndarray
    runtime_state: AtmosphereRuntimeState
    catalog: MolecularEquilibriumCatalog
    molecular_populations: np.ndarray
    partition_normalized_molecular_populations: np.ndarray
    molecular_equation_densities: np.ndarray
    previous_molecular_equation_densities: np.ndarray
    specific_internal_energy_mode_enabled: bool = False


def _load_hydrogen_molecule_partition_table() -> np.ndarray:
    with np.load(_MOLECULAR_TABLE_PATH, allow_pickle=False) as data:
        return np.asarray(data["h2_partition_function"], dtype=np.float64)


_HYDROGEN_MOLECULE_PARTITION_TABLE = _load_hydrogen_molecule_partition_table()


def initialize_molecular_equilibrium_state(
    *,
    temperature_k: np.ndarray,
    thermal_energy_erg: np.ndarray,
    gas_pressure: np.ndarray,
    runtime_state: AtmosphereRuntimeState,
    catalog: MolecularEquilibriumCatalog,
) -> MolecularEquilibriumState:
    """Allocate molecular-equilibrium work arrays for one atmosphere."""

    layer_count = int(np.asarray(temperature_k).size)
    equation_count = max(int(catalog.equation_count), 1)
    return MolecularEquilibriumState(
        temperature_k=np.asarray(temperature_k, dtype=np.float64),
        thermal_energy_erg=np.asarray(thermal_energy_erg, dtype=np.float64),
        gas_pressure=np.asarray(gas_pressure, dtype=np.float64),
        runtime_state=runtime_state,
        catalog=catalog,
        molecular_populations=np.zeros(
            (layer_count, int(catalog.molecule_count)),
            dtype=np.float64,
        ),
        partition_normalized_molecular_populations=np.zeros(
            (layer_count, int(catalog.molecule_count)),
            dtype=np.float64,
        ),
        molecular_equation_densities=np.zeros(
            (layer_count, equation_count), dtype=np.float64
        ),
        previous_molecular_equation_densities=np.zeros(
            (layer_count, equation_count), dtype=np.float64
        ),
        specific_internal_energy_mode_enabled=False,
    )


def _interp_hydrogen_molecule_partition(temperature_k: float) -> float:
    temperature = float(temperature_k)
    if not np.isfinite(temperature) or temperature <= 100.0:
        temperature = 100.0
    elif temperature >= 19900.0:
        temperature = 19900.0
    index = min(199, max(1, int(temperature / 100.0)))
    lower = _HYDROGEN_MOLECULE_PARTITION_TABLE[index - 1]
    upper = _HYDROGEN_MOLECULE_PARTITION_TABLE[index]
    return float(lower + (upper - lower) * (temperature - index * 100.0) / 100.0)


def hydrogen_molecule_equilibrium_constant(temperature_k: float) -> float:
    """Return the H2 equilibrium constant used by molecular equilibrium."""

    temperature = float(temperature_k)
    if not np.isfinite(temperature) or temperature <= 0.0:
        temperature = 1.0
    partition = _interp_hydrogen_molecule_partition(temperature)
    denominator_argument = (
        2.0
        * np.pi
        * 1.008
        * ATOMIC_MASS_GRAM_REFERENCE
        * BOLTZMANN_ERG_PER_K_REFERENCE
        / (PLANCK_ERG_SECOND_REFERENCE**2)
        * temperature
    )
    if not np.isfinite(denominator_argument) or denominator_argument <= 0.0:
        denominator_argument = 1.0e-300
    denominator = denominator_argument**1.5
    exponent = (
        36118.11
        * PLANCK_ERG_SECOND_REFERENCE
        * _LIGHT_SPEED_CM_PER_S_REFERENCE
        / BOLTZMANN_ERG_PER_K_REFERENCE
        / max(temperature, 1.0e-30)
    )
    value = partition * (2.0**1.5) / 4.0 / max(denominator, 1.0e-300) * np.exp(exponent)
    return float(value) if np.isfinite(value) else 0.0


def _saha_single(
    molecular_state: MolecularEquilibriumState,
    *,
    layer_index: int,
    atomic_number: int,
    ion_stage_count: int,
    population_mode: int,
    temperature_override: float | None = None,
    use_charge_square: bool = True,
) -> np.ndarray:
    runtime_state = molecular_state.runtime_state
    temperature = (
        float(molecular_state.temperature_k[layer_index])
        if temperature_override is None
        else float(temperature_override)
    )
    charge_square_density = (
        float(max(runtime_state.charge_square_density[layer_index], 1.0e-30))
        if use_charge_square
        else None
    )
    return saha_partition_depth(
        temperature_k=temperature,
        electron_density_cm3=float(runtime_state.electron_density[layer_index]),
        total_nuclei_number_density_cm3=float(
            runtime_state.total_nuclei_number_density[layer_index]
        ),
        elemental_abundance=float(
            runtime_state.elemental_abundances_by_layer[layer_index, atomic_number - 1]
        ),
        atomic_number=int(atomic_number),
        ion_stage_count=int(ion_stage_count),
        population_mode=int(population_mode),
        charge_square_density_cm3=charge_square_density,
    )


def _saha_depth_arrays(
    molecular_state: MolecularEquilibriumState,
    layer_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Depth arrays for ``saha_partition_depth_batch`` mirroring `_saha_single`.

    Returns ``(temperature_k, electron_density, clamped_charge_square)``;
    every element is the exact double the corresponding per-layer
    `_saha_single` call passes (including the ``max(..., 1e-30)``
    charge-square clamp, applied elementwise).
    """

    runtime_state = molecular_state.runtime_state
    temperatures = np.ascontiguousarray(
        molecular_state.temperature_k[:layer_count], dtype=np.float64
    )
    electron_densities = np.ascontiguousarray(
        runtime_state.electron_density[:layer_count], dtype=np.float64
    )
    charge_square_densities = np.maximum(
        np.asarray(runtime_state.charge_square_density[:layer_count], dtype=np.float64),
        1.0e-30,
    )
    return temperatures, electron_densities, charge_square_densities


def _abundance_vector_for_layer(
    molecular_state: MolecularEquilibriumState,
    layer_index: int,
) -> np.ndarray:
    catalog = molecular_state.catalog
    abundances = np.zeros(catalog.equation_count, dtype=np.float64)
    for equation_index in range(1, catalog.equation_count):
        species_code = int(catalog.equation_species_codes[equation_index])
        if 0 < species_code < 100:
            abundances[equation_index] = max(
                float(
                    molecular_state.runtime_state.elemental_abundances_by_layer[
                        layer_index, species_code - 1
                    ]
                ),
                1.0e-20,
            )
    if int(catalog.equation_species_codes[catalog.equation_count - 1]) == 100:
        abundances[catalog.equation_count - 1] = 0.0
    return abundances


def solve_molecular_equilibrium(
    molecular_state: MolecularEquilibriumState,
    *,
    population_mode: int = 1,
) -> None:
    """Compute molecular equilibrium and update runtime density arrays."""

    catalog = molecular_state.catalog
    runtime_state = molecular_state.runtime_state
    layer_count = int(molecular_state.temperature_k.size)
    equation_count = int(catalog.equation_count)
    if equation_count <= 0:
        return

    equation_density = np.zeros(equation_count, dtype=np.float64)
    first_total_density = float(
        molecular_state.gas_pressure[0]
        / max(
            float(molecular_state.temperature_k[0]) * BOLTZMANN_ERG_PER_K_REFERENCE,
            1.0e-300,
        )
    )
    equation_density[0] = first_total_density / 2.0
    if float(molecular_state.temperature_k[0]) < 4000.0:
        equation_density[0] = first_total_density
    electron_seed = equation_density[0] / 10.0
    first_layer_abundance = _abundance_vector_for_layer(molecular_state, 0)
    for equation_index in range(1, equation_count):
        equation_density[equation_index] = (
            electron_seed * first_layer_abundance[equation_index]
        )
    if int(catalog.equation_species_codes[equation_count - 1]) == 100:
        equation_density[equation_count - 1] = electron_seed
    runtime_state.electron_density[0] = electron_seed

    for layer_index in range(layer_count):
        if layer_index > 0:
            ratio = float(
                molecular_state.gas_pressure[layer_index]
                / max(molecular_state.gas_pressure[layer_index - 1], 1.0e-300)
            )
            runtime_state.electron_density[layer_index] = (
                runtime_state.electron_density[layer_index - 1] * ratio
            )
            equation_density *= ratio
        if molecular_state.specific_internal_energy_mode_enabled and np.any(
            molecular_state.previous_molecular_equation_densities[layer_index] != 0.0
        ):
            equation_density[:] = molecular_state.previous_molecular_equation_densities[
                layer_index
            ]

        equation_density = solve_molecular_equilibrium_layer(
            molecular_state,
            layer_index,
            equation_density,
        )
        molecular_state.molecular_equation_densities[layer_index, :equation_count] = (
            equation_density[:equation_count]
        )
        runtime_state.total_nuclei_number_density[layer_index] = equation_density[0]
        runtime_state.mass_density[layer_index] = (
            runtime_state.total_nuclei_number_density[layer_index]
            * runtime_state.mean_nuclear_mass_amu[layer_index]
            * ATOMIC_MASS_GRAM_REFERENCE
        )
        if int(catalog.equation_species_codes[equation_count - 1]) == 100:
            runtime_state.electron_density[layer_index] = equation_density[
                equation_count - 1
            ]

        constants = compute_equilibrium_constants_for_layer(
            molecular_state, layer_index
        )
        for molecule_index in range(catalog.molecule_count):
            term = float(constants[molecule_index])
            component_start = int(catalog.component_start_indices[molecule_index])
            component_stop = int(catalog.component_start_indices[molecule_index + 1])
            for component_index in range(component_start, component_stop):
                equation_index = int(
                    catalog.component_equation_indices[component_index]
                )
                if equation_index == equation_count:
                    term = term / max(equation_density[equation_count - 1], 1.0e-300)
                else:
                    term = term * equation_density[equation_index]
            molecular_state.molecular_populations[layer_index, molecule_index] = term

    if not molecular_state.specific_internal_energy_mode_enabled:
        molecular_state.previous_molecular_equation_densities[:, :equation_count] = (
            molecular_state.molecular_equation_densities[:, :equation_count]
        )
    else:
        runtime_state.specific_internal_energy[:] = (
            compute_molecular_specific_internal_energy(molecular_state)
        )
        return

    if int(population_mode) in (2, 12):
        return

    _fill_partition_normalized_molecular_densities(molecular_state)
    runtime_state.specific_internal_energy[:] = (
        1.5
        * np.asarray(
            molecular_state.gas_pressure,
            dtype=np.float64,
        )
        / np.maximum(runtime_state.mass_density, 1.0e-300)
    )


def _fill_partition_normalized_molecular_densities(
    molecular_state: MolecularEquilibriumState,
) -> None:
    catalog = molecular_state.catalog
    layer_count = int(molecular_state.temperature_k.size)
    equation_count = int(catalog.equation_count)

    # Depth-batched Saha evaluation (same element/stages/mode across layers): one
    # saha_partition_depth_batch call per species/molecule; every row is
    # bit-identical to the per-layer `_saha_single` call it replaces.
    (
        saha_temperatures,
        saha_electron_densities,
        saha_charge_square_densities,
    ) = _saha_depth_arrays(molecular_state, layer_count)

    for equation_index in range(1, equation_count):
        species_code = int(catalog.equation_species_codes[equation_index])
        if species_code <= 0:
            continue
        if species_code == 100:
            temperature = np.asarray(molecular_state.temperature_k, dtype=np.float64)
            molecular_state.molecular_equation_densities[:, equation_index] = (
                molecular_state.molecular_equation_densities[:, equation_index]
                / (
                    2.0
                    * 2.4148e15
                    * temperature
                    * np.sqrt(np.maximum(temperature, 1.0e-300))
                )
            )
            continue
        atomic_mass = (
            float(REFERENCE_ATOMIC_MASS_AMU[species_code - 1])
            if 1 <= species_code <= REFERENCE_ATOMIC_MASS_AMU.size
            else float(species_code)
        )
        partition_rows = saha_partition_depth_batch(
            saha_temperatures,
            saha_electron_densities,
            species_code,
            1,
            3,
            saha_charge_square_densities,
        )
        for layer_index in range(layer_count):
            partition_values = partition_rows[layer_index]
            partition = float(partition_values[0]) if partition_values.size else 1.0
            temperature = float(molecular_state.temperature_k[layer_index])
            denominator = (
                partition
                * 1.8786e20
                * np.sqrt(max((atomic_mass * temperature) ** 3, 1.0e-300))
            )
            molecular_state.molecular_equation_densities[
                layer_index, equation_index
            ] /= max(
                denominator,
                1.0e-300,
            )

    for molecule_index in range(catalog.molecule_count):
        first_coefficient = float(catalog.equilibrium_coefficients[0, molecule_index])
        component_start = int(catalog.component_start_indices[molecule_index])
        component_stop = int(catalog.component_start_indices[molecule_index + 1])
        component_count = component_stop - component_start
        if first_coefficient != 0.0:
            molecule_mass = 0.0
            for component_index in range(component_start, component_stop):
                equation_index = int(
                    catalog.component_equation_indices[component_index]
                )
                if equation_index >= equation_count:
                    continue
                species_code = int(catalog.equation_species_codes[equation_index])
                if 1 <= species_code <= REFERENCE_ATOMIC_MASS_AMU.size:
                    molecule_mass += float(REFERENCE_ATOMIC_MASS_AMU[species_code - 1])
            for layer_index in range(layer_count):
                temperature = float(molecular_state.temperature_k[layer_index])
                thermal_energy_ev = temperature / 11604.5
                value = np.exp(first_coefficient / max(thermal_energy_ev, 1.0e-300))
                for component_index in range(component_start, component_stop):
                    equation_index = int(
                        catalog.component_equation_indices[component_index]
                    )
                    if equation_index == equation_count:
                        value = value / max(
                            molecular_state.molecular_equation_densities[
                                layer_index, equation_count - 1
                            ],
                            1.0e-300,
                        )
                    else:
                        value = (
                            value
                            * molecular_state.molecular_equation_densities[
                                layer_index,
                                equation_index,
                            ]
                        )
                value *= 1.8786e20 * np.sqrt(
                    max((molecule_mass * temperature) ** 3, 1.0e-300)
                )
                molecular_state.partition_normalized_molecular_populations[
                    layer_index,
                    molecule_index,
                ] = value
            continue

        atomic_number = int(catalog.molecule_codes[molecule_index])
        partition_rows = saha_partition_depth_batch(
            saha_temperatures,
            saha_electron_densities,
            atomic_number,
            max(component_count, 1),
            3,
            saha_charge_square_densities,
        )
        for layer_index in range(layer_count):
            partition_values = partition_rows[layer_index]
            partition = float(partition_values[0]) if partition_values.size else 1.0
            molecular_state.partition_normalized_molecular_populations[
                layer_index,
                molecule_index,
            ] = molecular_state.molecular_populations[
                layer_index, molecule_index
            ] / max(
                partition,
                1.0e-300,
            )


def populate_molecular_species(
    molecular_state: MolecularEquilibriumState,
    *,
    code: float,
    population_mode: int,
    output: np.ndarray,
) -> None:
    """Populate one output slice from the solved molecular state."""

    catalog = molecular_state.catalog
    runtime_state = molecular_state.runtime_state
    code_value = float(code)
    mode = int(population_mode)
    output_array = np.asarray(output, dtype=np.float64)
    layer_count = output_array.shape[0]
    output_array[:, :] = 0.0

    if code_value >= 100.0:
        for molecule_index in range(catalog.molecule_count):
            if abs(float(catalog.molecule_codes[molecule_index]) - code_value) < 1.0e-3:
                source = (
                    molecular_state.partition_normalized_molecular_populations[
                        :, molecule_index
                    ]
                    if mode in (1, 11)
                    else molecular_state.molecular_populations[:, molecule_index]
                )
                output_array[:, 0] = source[:layer_count]
                return

    if code_value < 100.0:
        code_cursor = code_value
        requested_count = 1
        if mode in (11, 12):
            requested_count = int((code_cursor - float(int(code_cursor))) * 100.0 + 1.5)
        fallback_to_saha = False
        atomic_number = int(code_value)

        for index in range(1, requested_count + 1):
            ion = requested_count - index + 1
            found_exact = False
            for molecule_index in range(catalog.molecule_count):
                if (
                    abs(float(catalog.molecule_codes[molecule_index]) - code_cursor)
                    < 1.0e-3
                ):
                    source = (
                        molecular_state.partition_normalized_molecular_populations[
                            :, molecule_index
                        ]
                        if mode in (1, 11)
                        else molecular_state.molecular_populations[:, molecule_index]
                    )
                    output_array[:layer_count, ion - 1] = source[:layer_count]
                    found_exact = True
                    break
            if found_exact:
                code_cursor -= 0.01
                continue

            found_element_family = False
            for molecule_index in range(catalog.molecule_count):
                if int(catalog.molecule_codes[molecule_index]) == atomic_number:
                    found_element_family = True
                    break
            if found_element_family:
                output_array[:layer_count, ion - 1] = 0.0
                code_cursor -= 0.01
                continue

            fallback_to_saha = True
            break

        if not fallback_to_saha:
            return

        ion_stage_count = int((code_value - float(atomic_number)) * 100.0 + 1.5)
        output_count = ion_stage_count if mode in (11, 12) else 1
        # Depth-batched Saha evaluation (fixed element/stages/mode across layers;
        # `_saha_single(..., use_charge_square=False)` per layer is row i of
        # this batch, bit-identical).
        value_rows = saha_partition_depth_batch(
            np.ascontiguousarray(
                molecular_state.temperature_k[:layer_count], dtype=np.float64
            ),
            np.ascontiguousarray(
                runtime_state.electron_density[:layer_count], dtype=np.float64
            ),
            atomic_number,
            ion_stage_count,
            mode,
            None,
        )
        for layer_index in range(layer_count):
            values = value_rows[layer_index]
            copy_count = min(output_count, values.size, output_array.shape[1])
            output_array[layer_index, :copy_count] = (
                values[:copy_count]
                * runtime_state.total_nuclei_number_density[layer_index]
                * runtime_state.elemental_abundances_by_layer[
                    layer_index, atomic_number - 1
                ]
            )
        return

    raise ValueError(f"molecule code {code_value:.2f} not found in molecular table")


def set_molecular_specific_internal_energy_mode(
    molecular_state: MolecularEquilibriumState,
    enabled: bool,
) -> None:
    """Set the convection finite-difference specific-energy warm-start mode."""

    molecular_state.specific_internal_energy_mode_enabled = bool(enabled)


def save_molecular_equation_density(
    molecular_state: MolecularEquilibriumState,
) -> np.ndarray:
    """Return a copy of the molecular-equilibrium warm-start state."""

    return molecular_state.previous_molecular_equation_densities.copy()


def restore_molecular_equation_density(
    molecular_state: MolecularEquilibriumState,
    saved: np.ndarray | None,
) -> None:
    """Restore a saved molecular-equilibrium warm-start state."""

    if saved is not None:
        molecular_state.previous_molecular_equation_densities[:] = saved


# --- compiled (numba) molecular-equilibrium per-layer stack ---
#
# Each function below carries the `_compiled` suffix and uses libm math.* for
# the numpy scalar functions so association order is fixed across platforms. Saha
# values (_saha_single) are precomputed in Python through the compiled Saha
# stack in equation_of_state and passed into the kernels as arrays;
# np.linalg.solve stays the exact numpy LAPACK call.

if _NUMBA_AVAILABLE:
    _njit = numba.njit(cache=True, nogil=True)
    _njit_inline = numba.njit(cache=True, nogil=True, inline="always")

    @_njit_inline
    def _python_float_max_compiled(first, second):
        # Python max(first, second): keep first unless second compares greater
        # (NaN-propagation identical to builtins.max on two floats).
        return second if second > first else first

    @_njit_inline
    def _interp_hydrogen_molecule_partition_compiled(temperature_k, h2_partition_table):
        temperature = temperature_k
        if not math.isfinite(temperature) or temperature <= 100.0:
            temperature = 100.0
        elif temperature >= 19900.0:
            temperature = 19900.0
        index = min(199, max(1, int(temperature / 100.0)))
        lower = h2_partition_table[index - 1]
        upper = h2_partition_table[index]
        return lower + (upper - lower) * (temperature - index * 100.0) / 100.0

    @_njit_inline
    def _hydrogen_molecule_equilibrium_constant_compiled(
        temperature_k, h2_partition_table
    ):
        temperature = temperature_k
        if not math.isfinite(temperature) or temperature <= 0.0:
            temperature = 1.0
        partition = _interp_hydrogen_molecule_partition_compiled(
            temperature, h2_partition_table
        )
        denominator_argument = (
            2.0
            * np.pi
            * 1.008
            * ATOMIC_MASS_GRAM_REFERENCE
            * BOLTZMANN_ERG_PER_K_REFERENCE
            / _PLANCK_ERG_SECOND_SQUARED
            * temperature
        )
        if not math.isfinite(denominator_argument) or denominator_argument <= 0.0:
            denominator_argument = 1.0e-300
        denominator = denominator_argument**1.5
        exponent = (
            36118.11
            * PLANCK_ERG_SECOND_REFERENCE
            * _LIGHT_SPEED_CM_PER_S_REFERENCE
            / BOLTZMANN_ERG_PER_K_REFERENCE
            / _python_float_max_compiled(temperature, 1.0e-30)
        )
        value = (
            partition
            * _TWO_POW_THREE_HALVES
            / 4.0
            / _python_float_max_compiled(denominator, 1.0e-300)
            * math.exp(exponent)
        )
        if math.isfinite(value):
            return value
        return 0.0

    @_njit
    def _equilibrium_constants_kernel(
        temperature,
        thermal_energy_ev,
        natural_log_temperature,
        molecule_count,
        molecule_codes,
        equilibrium_coefficients,
        component_start_indices,
        h2_partition_table,
        saha_constants,
    ):
        constants = np.zeros(molecule_count, dtype=np.float64)
        for molecule_index in range(molecule_count):
            component_start = component_start_indices[molecule_index]
            component_stop = component_start_indices[molecule_index + 1]
            component_count = component_stop - component_start
            first_coefficient = equilibrium_coefficients[0, molecule_index]
            molecule_code = molecule_codes[molecule_index]

            if first_coefficient != 0.0:
                ion_count = int(
                    (molecule_code - float(int(molecule_code))) * 100.0 + 0.5
                )
                if abs(molecule_code - 101.0) < 0.005:
                    if temperature <= 20000.0:
                        constants[molecule_index] = (
                            _hydrogen_molecule_equilibrium_constant_compiled(
                                temperature, h2_partition_table
                            )
                        )
                    continue
                if temperature > 10000.0:
                    continue
                coeff_2 = equilibrium_coefficients[1, molecule_index]
                coeff_3 = equilibrium_coefficients[2, molecule_index]
                coeff_4 = equilibrium_coefficients[3, molecule_index]
                coeff_5 = equilibrium_coefficients[4, molecule_index]
                coeff_6 = equilibrium_coefficients[5, molecule_index]
                polynomial = (
                    coeff_3
                    + (-coeff_4 + (coeff_5 - coeff_6 * temperature) * temperature)
                    * temperature
                )
                exponent = (
                    first_coefficient
                    / _python_float_max_compiled(thermal_energy_ev, 1.0e-30)
                    - coeff_2
                    + polynomial * temperature
                    - 1.5
                    * float(component_count - ion_count - ion_count - 1)
                    * natural_log_temperature
                )
                constants[molecule_index] = math.exp(exponent)
                continue

            if component_count <= 1:
                constants[molecule_index] = 1.0
                continue

            # Saha-driven constants (population mode 12) are precomputed in Python
            # with the exact scalar expression and passed in.
            constants[molecule_index] = saha_constants[molecule_index]

        return constants

    @_njit
    def _newton_matrix_kernel(
        equation_count,
        molecule_count,
        equation_density,
        abundance,
        equilibrium_constants,
        residual_seed,
        equation_species_codes,
        component_start_indices,
        component_equation_indices,
    ):
        jacobian = np.zeros((equation_count, equation_count), dtype=np.float64)
        residual = np.zeros(equation_count, dtype=np.float64)

        residual[0] = residual_seed
        for equation_index in range(1, equation_count):
            residual[0] += equation_density[equation_index]
            jacobian[0, equation_index] = 1.0
            residual[equation_index] = (
                equation_density[equation_index]
                - abundance[equation_index] * equation_density[0]
            )
            jacobian[equation_index, equation_index] = 1.0
            jacobian[equation_index, 0] = -abundance[equation_index]
        if equation_species_codes[equation_count - 1] == 100:
            residual[equation_count - 1] = -equation_density[equation_count - 1]
            jacobian[equation_count - 1, equation_count - 1] = -1.0

        for molecule_index in range(molecule_count):
            component_start = component_start_indices[molecule_index]
            component_stop = component_start_indices[molecule_index + 1]
            component_count = component_stop - component_start
            if component_count <= 1:
                continue
            term = equilibrium_constants[molecule_index]
            if term == 0.0:
                continue

            for component_index in range(component_start, component_stop):
                equation_index = component_equation_indices[component_index]
                if equation_index == equation_count:
                    term = term / _python_float_max_compiled(
                        equation_density[equation_count - 1], 1.0e-300
                    )
                else:
                    term = term * equation_density[equation_index]

            residual[0] += term
            for component_index in range(component_start, component_stop):
                raw_equation_index = component_equation_indices[component_index]
                if raw_equation_index == equation_count:
                    equation_index = equation_count - 1
                    derivative = -term / _python_float_max_compiled(
                        equation_density[equation_index], 1.0e-300
                    )
                else:
                    equation_index = raw_equation_index
                    derivative = term / _python_float_max_compiled(
                        equation_density[equation_index], 1.0e-300
                    )
                residual[equation_index] += term
                jacobian[0, equation_index] += derivative
                for other_component_index in range(component_start, component_stop):
                    raw_other_index = component_equation_indices[other_component_index]
                    other_index = (
                        equation_count - 1
                        if raw_other_index == equation_count
                        else raw_other_index
                    )
                    jacobian[other_index, equation_index] += derivative

            last_equation_index = component_equation_indices[component_stop - 1]
            if (
                last_equation_index < equation_count
                and equation_species_codes[last_equation_index] == 100
            ):
                for component_index in range(component_start, component_stop):
                    equation_index = component_equation_indices[component_index]
                    if equation_index >= equation_count:
                        equation_index = equation_count - 1
                    derivative = term / _python_float_max_compiled(
                        equation_density[equation_index], 1.0e-300
                    )
                    if equation_index == equation_count - 1:
                        residual[equation_index] -= term + term
                    for other_component_index in range(component_start, component_stop):
                        raw_other_index = component_equation_indices[
                            other_component_index
                        ]
                        other_index = (
                            equation_count - 1
                            if raw_other_index == equation_count
                            else raw_other_index
                        )
                        if other_index == equation_count - 1:
                            jacobian[other_index, equation_index] -= (
                                derivative + derivative
                            )

        return jacobian, residual

    @_njit
    def _newton_update_kernel(equation_count, equation_density, previous_delta, delta):
        still_iterating = False
        scale = 100.0
        for equation_index in range(equation_count):
            ratio = abs(delta[equation_index]) / _python_float_max_compiled(
                abs(equation_density[equation_index]),
                1.0e-300,
            )
            if ratio > _NEWTON_TOLERANCE:
                still_iterating = True
            if previous_delta[equation_index] * delta[equation_index] < 0.0:
                delta[equation_index] *= 0.69
            updated = equation_density[equation_index] - delta[equation_index]
            floor = equation_density[equation_index] / 100.0
            if abs(updated) >= floor:
                equation_density[equation_index] = abs(updated)
            else:
                equation_density[equation_index] = (
                    equation_density[equation_index] / scale
                )
                if previous_delta[equation_index] * delta[equation_index] < 0.0:
                    scale = math.sqrt(scale)
            previous_delta[equation_index] = delta[equation_index]
        return still_iterating

    @_njit
    def _molecular_energy_layer_kernel(
        temperature_plus,
        temperature_minus,
        thermal_energy,
        thermal_energy_ev,
        h_over_ck,
        base_energy,
        molecule_count,
        equation_count,
        molecular_density_row,
        molecule_codes,
        equilibrium_coefficients,
        component_start_indices,
        component_equation_indices,
        equation_species_codes,
        h2_partition_table,
        saha3_plus,
        saha3_minus,
        branchb_plus,
        branchb_minus,
        branchb_ionization,
        branchb_ready,
    ):
        energy = base_energy
        for molecule_index in range(molecule_count):
            molecular_density = molecular_density_row[molecule_index]
            if molecular_density <= 0.0:
                continue

            first_coefficient = equilibrium_coefficients[0, molecule_index]
            molecule_code = molecule_codes[molecule_index]
            component_start = component_start_indices[molecule_index]
            component_stop = component_start_indices[molecule_index + 1]

            if first_coefficient != 0.0:
                if abs(molecule_code - 101.0) < 0.005:
                    partition_plus = (
                        _interp_hydrogen_molecule_partition_compiled(
                            temperature_plus, h2_partition_table
                        )
                        + 1.0e-30
                    )
                    partition_minus = (
                        _interp_hydrogen_molecule_partition_compiled(
                            temperature_minus, h2_partition_table
                        )
                        + 1.0e-30
                    )
                    dissociation_over_kt = 36118.11 * h_over_ck
                else:
                    coeff_2 = equilibrium_coefficients[1, molecule_index]
                    coeff_3 = equilibrium_coefficients[2, molecule_index]
                    coeff_4 = equilibrium_coefficients[3, molecule_index]
                    coeff_5 = equilibrium_coefficients[4, molecule_index]
                    coeff_6 = equilibrium_coefficients[5, molecule_index]
                    partition_plus = (
                        math.exp(
                            -coeff_2
                            + (
                                coeff_3
                                + (
                                    -coeff_4
                                    + (coeff_5 - coeff_6 * temperature_plus)
                                    * temperature_plus
                                )
                                * temperature_plus
                            )
                            * temperature_plus
                        )
                        + 1.0e-30
                    )
                    partition_minus = (
                        math.exp(
                            -coeff_2
                            + (
                                coeff_3
                                + (
                                    -coeff_4
                                    + (coeff_5 - coeff_6 * temperature_minus)
                                    * temperature_minus
                                )
                                * temperature_minus
                            )
                            * temperature_minus
                        )
                        + 1.0e-30
                    )
                    for component_index in range(component_start, component_stop):
                        equation_index = component_equation_indices[component_index]
                        if equation_index >= equation_count:
                            continue
                        species_code = equation_species_codes[equation_index]
                        if species_code > 0 and species_code < 100:
                            partition_plus *= saha3_plus[equation_index]
                            partition_minus *= saha3_minus[equation_index]
                    dissociation_over_kt = (
                        first_coefficient
                        / _python_float_max_compiled(thermal_energy_ev, 1.0e-300)
                    )

                partition_derivative = (
                    (partition_plus - partition_minus)
                    / _python_float_max_compiled(
                        partition_plus + partition_minus, 1.0e-30
                    )
                    * 2.0
                    * 500.0
                )
                contribution = (
                    molecular_density
                    * thermal_energy
                    * (-dissociation_over_kt + partition_derivative)
                )
                energy += contribution
                continue

            if branchb_ready[molecule_index] == 0:
                continue
            plus_partition = branchb_plus[molecule_index]
            minus_partition = branchb_minus[molecule_index]
            partition_derivative = (
                (plus_partition - minus_partition)
                / _python_float_max_compiled(plus_partition + minus_partition, 1.0e-30)
                * 2.0
                * 500.0
            )
            contribution = (
                molecular_density
                * thermal_energy
                * (
                    branchb_ionization[molecule_index]
                    / _python_float_max_compiled(thermal_energy_ev, 1.0e-300)
                    + partition_derivative
                )
            )
            energy += contribution

        return energy


@dataclass
class _MolecularEquilibriumKernelCache:
    """Catalog arrays repacked once for the compiled kernels (values unchanged)."""

    molecule_codes: np.ndarray
    equilibrium_coefficients: np.ndarray
    component_start_indices: np.ndarray
    component_equation_indices: np.ndarray
    equation_species_codes: np.ndarray
    # (molecule_index, atomic_number, component_count) needing Saha mode 12
    # in the equilibrium-constants branch (first coefficient zero, >1 component).
    saha_rows: tuple
    # (molecule_index, atomic_number, ion_count) for the energy-density atomic
    # branch (first coefficient zero, 1 <= Z <= 99).
    branchb_rows: tuple
    # (equation_index, species_code) with 0 < species_code < 100 for the
    # energy-density mode-3 partition factors.
    mode3_rows: tuple


def _catalog_kernel_cache(
    catalog: MolecularEquilibriumCatalog,
) -> _MolecularEquilibriumKernelCache:
    cached = getattr(catalog, "_molecular_equilibrium_kernel_cache", None)
    if cached is not None:
        return cached

    molecule_count = int(catalog.molecule_count)
    equation_count = int(catalog.equation_count)
    saha_rows = []
    branchb_rows = []
    for molecule_index in range(molecule_count):
        first_coefficient = float(catalog.equilibrium_coefficients[0, molecule_index])
        if first_coefficient != 0.0:
            continue
        component_start = int(catalog.component_start_indices[molecule_index])
        component_stop = int(catalog.component_start_indices[molecule_index + 1])
        component_count = component_stop - component_start
        atomic_number = int(catalog.molecule_codes[molecule_index])
        if component_count > 1:
            saha_rows.append((molecule_index, atomic_number, component_count))
        if 1 <= atomic_number <= 99:
            branchb_rows.append(
                (molecule_index, atomic_number, max(component_count, 1))
            )
    mode3_rows = []
    for equation_index in range(1, equation_count):
        species_code = int(catalog.equation_species_codes[equation_index])
        if 0 < species_code < 100:
            mode3_rows.append((equation_index, species_code))

    cached = _MolecularEquilibriumKernelCache(
        molecule_codes=np.ascontiguousarray(catalog.molecule_codes, dtype=np.float64),
        equilibrium_coefficients=np.ascontiguousarray(
            catalog.equilibrium_coefficients, dtype=np.float64
        ),
        component_start_indices=np.ascontiguousarray(
            catalog.component_start_indices, dtype=np.int64
        ),
        component_equation_indices=np.ascontiguousarray(
            catalog.component_equation_indices, dtype=np.int64
        ),
        equation_species_codes=np.ascontiguousarray(
            catalog.equation_species_codes, dtype=np.int64
        ),
        saha_rows=tuple(saha_rows),
        branchb_rows=tuple(branchb_rows),
        mode3_rows=tuple(mode3_rows),
    )
    catalog._molecular_equilibrium_kernel_cache = cached
    return cached


def _compute_equilibrium_constants_for_layer_compiled(
    molecular_state: MolecularEquilibriumState,
    layer_index: int,
) -> np.ndarray:
    """Compute molecular equilibrium constants for one layer.

    The mode-12 Saha branch is evaluated here in Python (values come from
    the compiled Saha stack); the H2 and dissociation-polynomial branches run
    in the numba kernel.
    """

    catalog = molecular_state.catalog
    cache = _catalog_kernel_cache(catalog)
    temperature = float(molecular_state.temperature_k[layer_index])
    thermal_energy_ev = temperature / 11604.5
    natural_log_temperature = float(np.log(max(temperature, 1.0e-300)))

    saha_constants = np.zeros(int(catalog.molecule_count), dtype=np.float64)
    for molecule_index, atomic_number, component_count in cache.saha_rows:
        fractions = _saha_single(
            molecular_state,
            layer_index=layer_index,
            atomic_number=atomic_number,
            ion_stage_count=component_count,
            population_mode=12,
        )
        if fractions.size < component_count or fractions[0] <= 0.0:
            continue
        ion_count = component_count - 1
        saha_constants[molecule_index] = float(
            fractions[component_count - 1]
            / fractions[0]
            * max(molecular_state.runtime_state.electron_density[layer_index], 1.0e-300)
            ** ion_count
        )

    return _equilibrium_constants_kernel(
        temperature,
        thermal_energy_ev,
        natural_log_temperature,
        int(catalog.molecule_count),
        cache.molecule_codes,
        cache.equilibrium_coefficients,
        cache.component_start_indices,
        _HYDROGEN_MOLECULE_PARTITION_TABLE,
        saha_constants,
    )


def _solve_molecular_equilibrium_layer_compiled(
    molecular_state: MolecularEquilibriumState,
    layer_index: int,
    equation_density_seed: np.ndarray,
) -> np.ndarray:
    """Run the damped-Newton molecular-equilibrium solve for one layer.

    Jacobian/residual assembly and the damped Newton update run as numba
    kernels; ``np.linalg.solve`` (and the ``lstsq`` fallback) stays the exact
    numpy LAPACK call.
    """

    catalog = molecular_state.catalog
    cache = _catalog_kernel_cache(catalog)
    equation_count = int(catalog.equation_count)
    equation_density = np.asarray(equation_density_seed, dtype=np.float64).copy()
    previous_delta = np.zeros(equation_count, dtype=np.float64)
    abundance = _abundance_vector_for_layer(molecular_state, layer_index)
    equilibrium_constants = _compute_equilibrium_constants_for_layer_compiled(
        molecular_state,
        layer_index,
    )
    thermal_energy = (
        float(molecular_state.temperature_k[layer_index])
        * BOLTZMANN_ERG_PER_K_REFERENCE
    )
    residual_seed = -float(
        molecular_state.gas_pressure[layer_index] / max(thermal_energy, 1.0e-300)
    )

    for _ in range(_MAX_NEWTON_ITERATIONS):
        jacobian, residual = _newton_matrix_kernel(
            equation_count,
            int(catalog.molecule_count),
            equation_density,
            abundance,
            equilibrium_constants,
            residual_seed,
            cache.equation_species_codes,
            cache.component_start_indices,
            cache.component_equation_indices,
        )

        try:
            delta = np.linalg.solve(jacobian, residual)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(jacobian, residual, rcond=None)
        if delta.size != equation_count:
            raise RuntimeError("molecular solver returned wrong vector size")

        still_iterating = _newton_update_kernel(
            equation_count,
            equation_density,
            previous_delta,
            np.ascontiguousarray(delta, dtype=np.float64),
        )
        if not still_iterating:
            return equation_density

    return equation_density


def _compute_molecular_specific_internal_energy_compiled(
    molecular_state: MolecularEquilibriumState,
) -> np.ndarray:
    """Compute molecular specific internal energy.

    The mode-3/mode-5 Saha values used inside the per-layer molecule loop
    are precomputed here in Python through the compiled Saha stack,
    depth-batched: one ``saha_partition_depth_batch`` call per (row, +/-
    perturbation) covers every layer (branch B batches only the layers passing
    the density guard), each row equal to the per-layer `_saha_single` call it
    replaces. The molecule loop runs as a kernel.
    """

    catalog = molecular_state.catalog
    runtime_state = molecular_state.runtime_state
    cache = _catalog_kernel_cache(catalog)
    layer_count = int(molecular_state.temperature_k.size)
    molecule_count = int(catalog.molecule_count)
    equation_count = int(catalog.equation_count)
    specific_internal_energy = np.zeros(layer_count, dtype=np.float64)

    # Exact per-layer scalars, elementwise: max(T, 1.0) * 1.001 / 0.999 and
    # the `_saha_single` electron-density / clamped charge-square inputs.
    temperature_clamped_array = np.maximum(
        np.asarray(molecular_state.temperature_k[:layer_count], dtype=np.float64),
        1.0,
    )
    temperature_plus_array = temperature_clamped_array * 1.001
    temperature_minus_array = temperature_clamped_array * 0.999
    (
        _saha_temperatures_unused,
        electron_density_array,
        charge_square_density_array,
    ) = _saha_depth_arrays(molecular_state, layer_count)

    saha3_plus_by_layer = np.zeros((layer_count, equation_count), dtype=np.float64)
    saha3_minus_by_layer = np.zeros((layer_count, equation_count), dtype=np.float64)
    for equation_index, species_code in cache.mode3_rows:
        plus_rows = saha_partition_depth_batch(
            temperature_plus_array,
            electron_density_array,
            species_code,
            1,
            3,
            charge_square_density_array,
        )
        minus_rows = saha_partition_depth_batch(
            temperature_minus_array,
            electron_density_array,
            species_code,
            1,
            3,
            charge_square_density_array,
        )
        saha3_plus_by_layer[:, equation_index] = plus_rows[:, 0]
        saha3_minus_by_layer[:, equation_index] = minus_rows[:, 0]

    branchb_plus_by_layer = np.zeros((layer_count, molecule_count), dtype=np.float64)
    branchb_minus_by_layer = np.zeros((layer_count, molecule_count), dtype=np.float64)
    branchb_ionization_by_layer = np.zeros(
        (layer_count, molecule_count), dtype=np.float64
    )
    branchb_ready_by_layer = np.zeros((layer_count, molecule_count), dtype=np.uint8)
    for molecule_index, atomic_number, ion_count in cache.branchb_rows:
        # The oracle's per-layer guard is `density <= 0.0: continue`; negate
        # it (rather than testing `> 0.0`) so NaN densities keep computing,
        # exactly as the oracle does.
        density_column = np.asarray(
            molecular_state.molecular_populations[:layer_count, molecule_index],
            dtype=np.float64,
        )
        active_layers = np.flatnonzero(~(density_column <= 0.0))
        if active_layers.size == 0:
            continue
        plus_rows = saha_partition_depth_batch(
            temperature_plus_array[active_layers],
            electron_density_array[active_layers],
            atomic_number,
            ion_count,
            5,
            charge_square_density_array[active_layers],
        )
        minus_rows = saha_partition_depth_batch(
            temperature_minus_array[active_layers],
            electron_density_array[active_layers],
            atomic_number,
            ion_count,
            5,
            charge_square_density_array[active_layers],
        )
        for row_position in range(active_layers.size):
            layer_index = int(active_layers[row_position])
            plus = plus_rows[row_position]
            minus = minus_rows[row_position]
            plus_partition = (
                float(plus[ion_count - 1]) if plus.size >= ion_count else 1.0
            )
            minus_partition = (
                float(minus[ion_count - 1]) if minus.size >= ion_count else 1.0
            )
            plus_partition = max(plus_partition, minus_partition)
            ionization_index = 30 + ion_count
            ionization_energy = (
                float(plus[ionization_index]) if plus.size > ionization_index else 0.0
            )
            branchb_plus_by_layer[layer_index, molecule_index] = plus_partition
            branchb_minus_by_layer[layer_index, molecule_index] = minus_partition
            branchb_ionization_by_layer[layer_index, molecule_index] = ionization_energy
            branchb_ready_by_layer[layer_index, molecule_index] = 1

    for layer_index in range(layer_count):
        temperature = max(float(molecular_state.temperature_k[layer_index]), 1.0)
        thermal_energy = float(molecular_state.thermal_energy_erg[layer_index])
        thermal_energy_ev = temperature / 11604.5
        h_over_ck = (
            PLANCK_ERG_SECOND_REFERENCE
            * _LIGHT_SPEED_CM_PER_S_REFERENCE
            / max(thermal_energy, 1.0e-300)
        )
        total_density = float(molecular_state.gas_pressure[layer_index]) / max(
            thermal_energy,
            1.0e-300,
        )
        base_energy = 1.5 * total_density * thermal_energy
        temperature_plus = temperature * 1.001
        temperature_minus = temperature * 0.999
        density_row = molecular_state.molecular_populations[layer_index]

        energy = _molecular_energy_layer_kernel(
            temperature_plus,
            temperature_minus,
            thermal_energy,
            thermal_energy_ev,
            h_over_ck,
            base_energy,
            molecule_count,
            equation_count,
            density_row,
            cache.molecule_codes,
            cache.equilibrium_coefficients,
            cache.component_start_indices,
            cache.component_equation_indices,
            cache.equation_species_codes,
            _HYDROGEN_MOLECULE_PARTITION_TABLE,
            saha3_plus_by_layer[layer_index],
            saha3_minus_by_layer[layer_index],
            branchb_plus_by_layer[layer_index],
            branchb_minus_by_layer[layer_index],
            branchb_ionization_by_layer[layer_index],
            branchb_ready_by_layer[layer_index],
        )
        density = max(float(runtime_state.mass_density[layer_index]), 1.0e-300)
        specific_internal_energy[layer_index] = energy / density

    return specific_internal_energy


def compute_equilibrium_constants_for_layer(
    molecular_state: MolecularEquilibriumState,
    layer_index: int,
) -> np.ndarray:
    """Compute molecular equilibrium constants for one depth."""

    return _compute_equilibrium_constants_for_layer_compiled(
        molecular_state, layer_index
    )


def solve_molecular_equilibrium_layer(
    molecular_state: MolecularEquilibriumState,
    layer_index: int,
    equation_density_seed: np.ndarray,
) -> np.ndarray:
    """Run the molecular-equilibrium Newton solve for one depth layer."""

    return _solve_molecular_equilibrium_layer_compiled(
        molecular_state, layer_index, equation_density_seed
    )


def compute_molecular_specific_internal_energy(
    molecular_state: MolecularEquilibriumState,
) -> np.ndarray:
    """Compute molecular specific internal energy, matching the reference kernel."""

    return _compute_molecular_specific_internal_energy_compiled(molecular_state)
