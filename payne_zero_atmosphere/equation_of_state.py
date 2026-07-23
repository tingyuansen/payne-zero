# ruff: noqa: E402
"""Equation of state: ionization, partition functions, populations.

Saha/Boltzmann solve over the packed level tables in the data home
(special light-element partition functions, iron-group grids, packed level
metadata), producing the per-layer population state that opacity consumes.
Table loaders live at the bottom; the iron-group partition function and
the packed-metadata decode are the parity-sensitive pieces.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from dataclasses import dataclass
import math

import numpy as np

from ._numba_cache import configure_numba_cache

# The compiled Saha/partition kernels are the sole production path; numba is a
# hard requirement.
configure_numba_cache()
try:
    import numba
except ImportError as exc:  # pragma: no cover - numba is a hard requirement
    raise ImportError(
        "numba is required: the compiled Saha/partition kernels are the sole "
        "production path for payne_zero_atmosphere."
    ) from exc

_NUMBA_AVAILABLE = True

from .constants import (
    BOLTZMANN_ERG_PER_K_REFERENCE,
    BOLTZMANN_EV_PER_K_REFERENCE,
    LIGHT_SPEED_CM_PER_S_EXACT as LIGHT_SPEED_CM_PER_S,
    PLANCK_ERG_SECOND_REFERENCE,
    WAVENUMBER_PER_EV_REFERENCE,
)
from .data_files import atmosphere_table_path, load_table_arrays

from .population_layout import (
    atomic_population_slot_start,
    decode_population_code,
    ion_stage_count_for_atomic_number,
    population_job_schedule,
)
from .constants import ATOMIC_MASS_GRAM_REFERENCE
from .runtime_state import AtmosphereRuntimeState

if TYPE_CHECKING:
    from .molecular_equilibrium import MolecularEquilibriumState


ELECTRON_CHARGE_ESU_REFERENCE = 4.801e-10
SAHA_COEFFICIENT_REFERENCE = 2.0 * 2.4148e15
ION_FRACTION_SCALE = np.array([0.001, 0.01, 0.1, 1.0], dtype=np.float64)
# Precomputed in Python (CPython float ** int) so the compiled kernel reuses
# the exact double produced for ELECTRON_CHARGE**2.
_ELECTRON_CHARGE_ESU_SQUARED = ELECTRON_CHARGE_ESU_REFERENCE**2


def _validate_atomic_number(atomic_number: int) -> int:
    value = int(atomic_number)
    if not 1 <= value <= 99:
        raise ValueError(f"atomic number must be in [1, 99], got {atomic_number}")
    return value


def _start_and_available_ion_count(
    atomic_number: int,
    table_offsets: np.ndarray,
) -> tuple[int, int]:
    if atomic_number <= 28:
        start = int(table_offsets[atomic_number - 1])
        available_count = int(table_offsets[atomic_number] - start)
    else:
        start = 3 * atomic_number + 54
        available_count = 3
    if atomic_number == 6:
        start = 354
        available_count = 6
    if atomic_number == 7:
        start = 360
        available_count = 6
    if 20 <= atomic_number < 29:
        available_count = 10
    return start, available_count


# --- compiled (numba) Saha/partition stack ---
#
# Each function below carries the `_compiled` suffix and uses libm math.* for
# the np scalar funcs so association order is fixed across platforms.

if _NUMBA_AVAILABLE:
    _njit = numba.njit(cache=True, nogil=True)
    _njit_inline = numba.njit(cache=True, nogil=True, inline="always")

    @_njit_inline
    def _python_float_max_compiled(first, second):
        # Python max(first, second): keep first unless second compares greater
        # (NaN-propagation identical to builtins.max on two floats).
        return second if second > first else first

    @_njit_inline
    def _python_float_min_compiled(first, second):
        # Python min(first, second): keep first unless second compares smaller.
        return second if second < first else first

    @_njit_inline
    def _numpy_small_sum_compiled(values):
        # np.sum on a 1-D float64 array dispatches to numpy's pairwise
        # summation; replicate its exact accumulation order for n <= 128
        # (sequential below 8 elements, 8-way accumulators above).
        n = values.shape[0]
        if n < 8:
            res = 0.0
            for i in range(n):
                res += values[i]
            return res
        r0 = values[0]
        r1 = values[1]
        r2 = values[2]
        r3 = values[3]
        r4 = values[4]
        r5 = values[5]
        r6 = values[6]
        r7 = values[7]
        i = 8
        while i < n - (n % 8):
            r0 += values[i + 0]
            r1 += values[i + 1]
            r2 += values[i + 2]
            r3 += values[i + 3]
            r4 += values[i + 4]
            r5 += values[i + 5]
            r6 += values[i + 6]
            r7 += values[i + 7]
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += values[i]
            i += 1
        return res

    @_njit_inline
    def _start_and_available_ion_count_compiled(atomic_number, table_offsets):
        if atomic_number <= 28:
            start = int(table_offsets[atomic_number - 1])
            available_count = int(table_offsets[atomic_number] - start)
        else:
            start = 3 * atomic_number + 54
            available_count = 3
        if atomic_number == 6:
            start = 354
            available_count = 6
        if atomic_number == 7:
            start = 360
            available_count = 6
        if atomic_number >= 20 and atomic_number < 29:
            available_count = 10
        return start, available_count

    @_njit_inline
    def _ionization_potential_index_compiled(atomic_number, ion_stage):
        if atomic_number <= 30:
            return atomic_number * (atomic_number + 1) // 2 + ion_stage - 1
        return atomic_number * 5 + 341 + ion_stage - 1

    @_njit_inline
    def _occupation_term_compiled(density_parameter, ion_charge, thermal_energy_ev):
        x = math.sqrt(
            13.595 * ion_charge * ion_charge / (thermal_energy_ev * density_parameter)
        )
        polynomial = (1.0 / 3.0) + (
            1.0
            - (0.5 + (1.0 / 18.0 + density_parameter / 120.0) * density_parameter)
            * density_parameter
        ) * density_parameter
        return x * x * x * polynomial

    @_njit_inline
    def _occupation_correction_compiled(
        partition_value,
        ion_charge,
        statistical_weight,
        ionization_potential_ev,
        lowering_potential_ev,
        thermal_energy_ev,
        lower_density_parameter,
    ):
        if (
            thermal_energy_ev <= 0.0
            or lower_density_parameter <= 0.0
            or lowering_potential_ev <= 0.0
        ):
            return _python_float_max_compiled(partition_value, 1.0)
        upper_density_parameter = lowering_potential_ev / thermal_energy_ev
        if upper_density_parameter <= 0.0:
            return _python_float_max_compiled(partition_value, 1.0)

        correction = statistical_weight * math.exp(
            -ionization_potential_ev / thermal_energy_ev
        )
        correction *= _occupation_term_compiled(
            upper_density_parameter, ion_charge, thermal_energy_ev
        ) - _occupation_term_compiled(
            lower_density_parameter, ion_charge, thermal_energy_ev
        )
        return _python_float_max_compiled(partition_value + correction, 1.0)

    @_njit
    def _special_partition_compiled(table_index, hc_over_kt, level_grid):
        """Special light-element partition functions.

        ``level_grid`` holds the SpecialPartitionTables level arrays (minus
        the offsets) as zero-padded rows in _SPECIAL_PARTITION_KEYS order —
        a boxing-cheap layout. Each branch re-views the rows it reads under
        descriptive names so the arithmetic below stays readable.
        """

        if table_index == 1:
            hydrogen_neutral_level_energy_cm = level_grid[0]
            hydrogen_neutral_level_statistical_weight = level_grid[1]
            partition_value = 2.0
            for idx in range(1, 6):
                partition_value += hydrogen_neutral_level_statistical_weight[
                    idx
                ] * math.exp(-hydrogen_neutral_level_energy_cm[idx] * hc_over_kt)
            density_parameter = 109677.576 / (6.5 * 6.5) * hc_over_kt
            return partition_value, density_parameter, True

        if table_index == 3:
            helium_neutral_level_energy_cm = level_grid[2]
            helium_neutral_level_statistical_weight = level_grid[3]
            partition_value = 1.0
            for idx in range(1, 29):
                partition_value += helium_neutral_level_statistical_weight[
                    idx
                ] * math.exp(-helium_neutral_level_energy_cm[idx] * hc_over_kt)
            density_parameter = 109677.576 / (5.5 * 5.5) * hc_over_kt
            return partition_value, density_parameter, True

        if table_index == 4:
            helium_singly_ionized_level_energy_cm = level_grid[4]
            helium_singly_ionized_level_statistical_weight = level_grid[5]
            partition_value = 2.0
            for idx in range(1, 6):
                partition_value += helium_singly_ionized_level_statistical_weight[
                    idx
                ] * math.exp(-helium_singly_ionized_level_energy_cm[idx] * hc_over_kt)
            density_parameter = 4.0 * 109722.267 / (6.5 * 6.5) * hc_over_kt
            return partition_value, density_parameter, True

        if table_index == 354:
            carbon_neutral_level_energy_cm = level_grid[6]
            carbon_neutral_level_statistical_weight = level_grid[7]
            partition_value = 1.0 + 3.0 * math.exp(-16.42 * hc_over_kt)
            partition_value += 5.0 * math.exp(-43.42 * hc_over_kt)
            for idx in range(1, 14):
                partition_value += carbon_neutral_level_statistical_weight[
                    idx
                ] * math.exp(-carbon_neutral_level_energy_cm[idx] * hc_over_kt)
            partition_value += (
                108.0 * math.exp(-80000.0 * hc_over_kt)
                + 189.0 * math.exp(-84000.0 * hc_over_kt)
                + 247.0 * math.exp(-87000.0 * hc_over_kt)
                + 231.0 * math.exp(-88000.0 * hc_over_kt)
                + 190.0 * math.exp(-89000.0 * hc_over_kt)
                + 300.0 * math.exp(-90000.0 * hc_over_kt)
            )
            return partition_value, 0.0, True

        if table_index == 355:
            carbon_singly_ionized_level_energy_cm = level_grid[8]
            carbon_singly_ionized_level_statistical_weight = level_grid[9]
            partition_value = 2.0 + 4.0 * math.exp(-63.42 * hc_over_kt)
            for idx in range(1, 6):
                partition_value += carbon_singly_ionized_level_statistical_weight[
                    idx
                ] * math.exp(-carbon_singly_ionized_level_energy_cm[idx] * hc_over_kt)
            partition_value += (
                6.0 * math.exp(-131731.80 * hc_over_kt)
                + 4.0 * math.exp(-142027.1 * hc_over_kt)
                + 10.0 * math.exp(-145550.13 * hc_over_kt)
                + 10.0 * math.exp(-150463.62 * hc_over_kt)
                + 2.0 * math.exp(-157234.07 * hc_over_kt)
                + 6.0 * math.exp(-162500.0 * hc_over_kt)
                + 42.0 * math.exp(-168000.0 * hc_over_kt)
                + 56.0 * math.exp(-178000.0 * hc_over_kt)
                + 102.0 * math.exp(-183000.0 * hc_over_kt)
                + 400.0 * math.exp(-188000.0 * hc_over_kt)
            )
            return partition_value, 0.0, True

        if table_index == 51:
            magnesium_neutral_level_energy_cm = level_grid[10]
            magnesium_neutral_level_statistical_weight = level_grid[11]
            partition_value = 1.0
            for idx in range(1, 11):
                partition_value += magnesium_neutral_level_statistical_weight[
                    idx
                ] * math.exp(-magnesium_neutral_level_energy_cm[idx] * hc_over_kt)
            partition_value += (
                5.0 * math.exp(-53134.0 * hc_over_kt)
                + 15.0 * math.exp(-54192.0 * hc_over_kt)
                + 28.0 * math.exp(-54676.0 * hc_over_kt)
                + 9.0 * math.exp(-57853.0 * hc_over_kt)
            )
            density_parameter = 109734.83 / (4.5 * 4.5) * hc_over_kt
            return partition_value, density_parameter, True

        if table_index == 52:
            magnesium_singly_ionized_level_energy_cm = level_grid[12]
            magnesium_singly_ionized_level_statistical_weight = level_grid[13]
            partition_value = 2.0
            for idx in range(1, 6):
                partition_value += magnesium_singly_ionized_level_statistical_weight[
                    idx
                ] * math.exp(
                    -magnesium_singly_ionized_level_energy_cm[idx] * hc_over_kt
                )
            partition_value += (
                10.0 * math.exp(-93310.80 * hc_over_kt)
                + 14.0 * math.exp(-93799.70 * hc_over_kt)
                + 6.0 * math.exp(-97464.32 * hc_over_kt)
                + 10.0 * math.exp(-103419.82 * hc_over_kt)
                + 14.0 * math.exp(-103689.89 * hc_over_kt)
                + 18.0 * math.exp(-103705.66 * hc_over_kt)
            )
            density_parameter = 4.0 * 109734.83 / (5.5 * 5.5) * hc_over_kt
            return partition_value, density_parameter, True

        if table_index == 57:
            aluminum_neutral_level_energy_cm = level_grid[14]
            aluminum_neutral_level_statistical_weight = level_grid[15]
            partition_value = 2.0 + 4.0 * math.exp(-112.061 * hc_over_kt)
            for idx in range(1, 9):
                partition_value += aluminum_neutral_level_statistical_weight[
                    idx
                ] * math.exp(-aluminum_neutral_level_energy_cm[idx] * hc_over_kt)
            partition_value += 10.0 * math.exp(-42235.0 * hc_over_kt)
            partition_value += 14.0 * math.exp(-43831.0 * hc_over_kt)
            density_parameter = 109735.08 / (5.5 * 5.5) * hc_over_kt
            return partition_value, density_parameter, True

        if table_index == 63:
            silicon_neutral_level_energy_cm = level_grid[16]
            silicon_neutral_level_statistical_weight = level_grid[17]
            partition_value = 1.0 + 3.0 * math.exp(-77.115 * hc_over_kt)
            partition_value += 5.0 * math.exp(-223.157 * hc_over_kt)
            for idx in range(1, 11):
                partition_value += silicon_neutral_level_statistical_weight[
                    idx
                ] * math.exp(-silicon_neutral_level_energy_cm[idx] * hc_over_kt)
            partition_value += (
                76.0 * math.exp(-53000.0 * hc_over_kt)
                + 71.0 * math.exp(-57000.0 * hc_over_kt)
                + 191.0 * math.exp(-60000.0 * hc_over_kt)
                + 240.0 * math.exp(-62000.0 * hc_over_kt)
                + 251.0 * math.exp(-63000.0 * hc_over_kt)
                + 300.0 * math.exp(-65000.0 * hc_over_kt)
            )
            return partition_value, 0.0, True

        if table_index == 64:
            silicon_singly_ionized_level_energy_cm = level_grid[18]
            silicon_singly_ionized_level_statistical_weight = level_grid[19]
            partition_value = 2.0 + 4.0 * math.exp(-287.32 * hc_over_kt)
            for idx in range(1, 6):
                partition_value += silicon_singly_ionized_level_statistical_weight[
                    idx
                ] * math.exp(-silicon_singly_ionized_level_energy_cm[idx] * hc_over_kt)
            partition_value += (
                6.0 * math.exp(-81231.59 * hc_over_kt)
                + 6.0 * math.exp(-83937.08 * hc_over_kt)
                + 10.0 * math.exp(-101024.09 * hc_over_kt)
                + 14.0 * math.exp(-103556.35 * hc_over_kt)
                + 10.0 * math.exp(-108800.0 * hc_over_kt)
                + 42.0 * math.exp(-115000.0 * hc_over_kt)
                + 6.0 * math.exp(-121000.0 * hc_over_kt)
                + 38.0 * math.exp(-125000.0 * hc_over_kt)
                + 34.0 * math.exp(-132000.0 * hc_over_kt)
            )
            density_parameter = 4.0 * 109734.83 / (4.5 * 4.5) * hc_over_kt
            return partition_value, density_parameter, True

        if table_index == 367:
            oxygen_neutral_level_energy_cm = level_grid[22]
            oxygen_neutral_level_statistical_weight = level_grid[23]
            partition_value = 5.0 + 3.0 * math.exp(-158.265 * hc_over_kt)
            partition_value += math.exp(-226.977 * hc_over_kt)
            for idx in range(1, 13):
                partition_value += oxygen_neutral_level_statistical_weight[
                    idx
                ] * math.exp(-oxygen_neutral_level_energy_cm[idx] * hc_over_kt)
            partition_value += (
                15.0 * math.exp(-101140.0 * hc_over_kt)
                + 131.0 * math.exp(-103000.0 * hc_over_kt)
                + 128.0 * math.exp(-105000.0 * hc_over_kt)
                + 600.0 * math.exp(-107000.0 * hc_over_kt)
            )
            return partition_value, 0.0, True

        if table_index == 45:
            sodium_neutral_level_energy_cm = level_grid[20]
            sodium_neutral_level_statistical_weight = level_grid[21]
            partition_value = 2.0
            for idx in range(1, 8):
                partition_value += sodium_neutral_level_statistical_weight[
                    idx
                ] * math.exp(-sodium_neutral_level_energy_cm[idx] * hc_over_kt)
            partition_value += 10.0 * math.exp(-34548.745 * hc_over_kt)
            partition_value += 14.0 * math.exp(-34586.96 * hc_over_kt)
            density_parameter = 109734.83 / (4.5 * 4.5) * hc_over_kt
            return partition_value, density_parameter, True

        if table_index == 14:
            boron_neutral_level_energy_cm = level_grid[24]
            boron_neutral_level_statistical_weight = level_grid[25]
            partition_value = 2.0 + 4.0 * math.exp(-15.25 * hc_over_kt)
            for idx in range(1, 7):
                partition_value += boron_neutral_level_statistical_weight[
                    idx
                ] * math.exp(-boron_neutral_level_energy_cm[idx] * hc_over_kt)
            partition_value += (
                6.0 * math.exp(-57786.80 * hc_over_kt)
                + 10.0 * math.exp(-59989.0 * hc_over_kt)
                + 14.0 * math.exp(-60031.03 * hc_over_kt)
                + 2.0 * math.exp(-63561.0 * hc_over_kt)
            )
            density_parameter = 109734.83 / (4.5 * 4.5) * hc_over_kt
            return partition_value, density_parameter, True

        if table_index == 91:
            potassium_neutral_level_energy_cm = level_grid[26]
            potassium_neutral_level_statistical_weight = level_grid[27]
            partition_value = 2.0
            for idx in range(1, 8):
                partition_value += potassium_neutral_level_statistical_weight[
                    idx
                ] * math.exp(-potassium_neutral_level_energy_cm[idx] * hc_over_kt)
            partition_value += 10.0 * math.exp(-27397.077 * hc_over_kt)
            partition_value += 14.0 * math.exp(-28127.85 * hc_over_kt)
            density_parameter = 109734.83 / (5.5 * 5.5) * hc_over_kt
            return partition_value, density_parameter, True

        return 1.0, 0.0, False

    @_njit_inline
    def _temperature_bin_compiled(log10_temperature):
        if log10_temperature > 4.0:
            upper_bin_1based = int((log10_temperature - 4.0) / 0.05) + 31
            upper_bin_1based = min(upper_bin_1based, 56)
            weight = (log10_temperature - (upper_bin_1based - 31) * 0.05 - 4.0) / 0.05
        elif log10_temperature < 3.7:
            upper_bin_1based = int((log10_temperature - 3.32) / 0.02) + 2
            upper_bin_1based = max(upper_bin_1based, 2)
            weight = (log10_temperature - (upper_bin_1based - 2) * 0.02 - 3.32) / 0.02
        else:
            upper_bin_1based = int((log10_temperature - 3.7) / 0.03) + 21
            weight = (log10_temperature - (upper_bin_1based - 21) * 0.03 - 3.7) / 0.03

        upper_index = upper_bin_1based - 1
        lower_index = upper_index - 1
        return lower_index, upper_index, weight

    @_njit_inline
    def _interpolate_temperature_compiled(
        grid,
        lowering_index,
        lower_temperature_index,
        upper_temperature_index,
        temperature_weight,
        ion_stage_index,
        element_index,
    ):
        return (
            temperature_weight
            * grid[
                lowering_index, upper_temperature_index, ion_stage_index, element_index
            ]
            + (1.0 - temperature_weight)
            * grid[
                lowering_index, lower_temperature_index, ion_stage_index, element_index
            ]
        )

    @_njit
    def _iron_group_partition_function_compiled(
        grid, atomic_number, ion_stage, log10_temperature, lowering_energy_cm
    ):
        # Range validation lives in the dispatcher; kernel inputs are
        # guaranteed valid (Z 20..28, ion stage 1..10) by construction of the
        # work-ion loop.
        lower_temperature_index, upper_temperature_index, temperature_weight = (
            _temperature_bin_compiled(log10_temperature)
        )
        ion_stage_index = ion_stage - 1
        element_index = atomic_number - 20
        lowering_energy = lowering_energy_cm

        if lowering_energy < _DEBYE_LOWERING_GRID_CM[0]:
            return _interpolate_temperature_compiled(
                grid,
                0,
                lower_temperature_index,
                upper_temperature_index,
                temperature_weight,
                ion_stage_index,
                element_index,
            )

        for upper_lowering_index in range(1, len(_DEBYE_LOWERING_GRID_CM)):
            if lowering_energy < _DEBYE_LOWERING_GRID_CM[upper_lowering_index]:
                lowering_weight = (
                    math.log10(lowering_energy)
                    - _DEBYE_LOWERING_LOG10_GRID[upper_lowering_index - 1]
                ) / 0.30103
                upper_value = _interpolate_temperature_compiled(
                    grid,
                    upper_lowering_index,
                    lower_temperature_index,
                    upper_temperature_index,
                    temperature_weight,
                    ion_stage_index,
                    element_index,
                )
                lower_value = _interpolate_temperature_compiled(
                    grid,
                    upper_lowering_index - 1,
                    lower_temperature_index,
                    upper_temperature_index,
                    temperature_weight,
                    ion_stage_index,
                    element_index,
                )
                return (
                    lowering_weight * upper_value
                    + (1.0 - lowering_weight) * lower_value
                )

        return _interpolate_temperature_compiled(
            grid,
            len(_DEBYE_LOWERING_GRID_CM) - 1,
            lower_temperature_index,
            upper_temperature_index,
            temperature_weight,
            ion_stage_index,
            element_index,
        )

    @_njit
    def _format_saha_output_compiled(
        base_mode,
        requested_ion_count,
        work_ion_count,
        partition_values,
        ionization_potential_ev,
        ion_fractions,
    ):
        output_count = min(requested_ion_count, work_ion_count)
        if base_mode == 1:
            output = np.zeros(requested_ion_count, dtype=np.float64)
            for idx in range(output_count):
                output[idx] = ion_fractions[idx] / _python_float_max_compiled(
                    partition_values[idx], 1.0e-300
                )
            return output
        if base_mode == 2:
            output = np.zeros(requested_ion_count, dtype=np.float64)
            for idx in range(output_count):
                output[idx] = ion_fractions[idx]
            return output
        if base_mode == 3:
            output = np.zeros(requested_ion_count, dtype=np.float64)
            for idx in range(output_count):
                output[idx] = partition_values[idx]
            return output
        if base_mode == 4:
            output = np.zeros(requested_ion_count, dtype=np.float64)
            products = np.empty(work_ion_count - 1, dtype=np.float64)
            for idx in range(1, work_ion_count):
                products[idx - 1] = ion_fractions[idx] * float(idx)
            output[0] = _numpy_small_sum_compiled(products)
            return output
        if base_mode == 5:
            output = np.zeros(61, dtype=np.float64)
            for idx in range(output_count):
                output[idx] = partition_values[idx]
            output[31] = 0.0
            accumulated_potential = 0.0
            for idx in range(output_count):
                accumulated_potential += ionization_potential_ev[idx]
                if 32 + idx < output.size:
                    output[32 + idx] = accumulated_potential
            return output
        # Unreachable: the dispatcher validates the mode before kernel entry.
        return np.zeros(1, dtype=np.float64)

    @_njit
    def _saha_partition_depth_kernel(
        temperature_k,
        electron_density_cm3,
        atomic_number,
        ion_stage_count,
        population_mode,
        charge_square_density_value,
        has_charge_square_density,
        packed_level_metadata,
        ionization_potential_table,
        iron_group_grid,
        element_block_offsets,
        special_level_grid,
        ion_fraction_scale,
    ):
        ion_stage_count = max(1, ion_stage_count)
        temperature = _python_float_max_compiled(temperature_k, 1.0)
        electron_density = _python_float_max_compiled(electron_density_cm3, 1.0e-40)
        thermal_energy_erg = BOLTZMANN_ERG_PER_K_REFERENCE * temperature
        thermal_energy_ev = BOLTZMANN_EV_PER_K_REFERENCE * temperature
        hc_over_kt = (
            PLANCK_ERG_SECOND_REFERENCE * LIGHT_SPEED_CM_PER_S
        ) / _python_float_max_compiled(thermal_energy_erg, 1.0e-300)

        mode = population_mode
        base_mode = mode if mode <= 10 else mode - 10
        return_all_ion_stages = mode >= 10

        if not has_charge_square_density:
            charge_square_density = _python_float_max_compiled(
                2.0 * electron_density, 1.0e-30
            )
        else:
            charge_square_density = _python_float_max_compiled(
                charge_square_density_value, 1.0e-30
            )
        debye_length = math.sqrt(
            thermal_energy_erg
            / (12.5664 * _ELECTRON_CHARGE_ESU_SQUARED * charge_square_density)
        )
        lowering_potential = _python_float_min_compiled(
            1.0, 1.44e-7 / _python_float_max_compiled(debye_length, 1.0e-300)
        )

        start_index, available_ion_count = _start_and_available_ion_count_compiled(
            atomic_number,
            element_block_offsets,
        )
        work_ion_count = min(ion_stage_count + 2, available_ion_count)
        table_index = start_index - 1

        partition_values = np.ones(work_ion_count, dtype=np.float64)
        ionization_potential_ev = np.zeros(work_ion_count, dtype=np.float64)
        lowering_potential_by_stage = np.zeros(work_ion_count, dtype=np.float64)
        ion_fractions = np.zeros(work_ion_count, dtype=np.float64)

        for ion_stage in range(1, work_ion_count + 1):
            ion_charge = float(ion_stage)
            table_index += 1
            # Table-index range is validated by the dispatcher before entry.

            stage_lowering_potential = lowering_potential * ion_charge
            packed_value = packed_level_metadata[5, table_index - 1]
            packed_ionization = packed_value // 100
            statistical_weight = float(packed_value - packed_ionization * 100)
            stage_ionization_potential_ev = float(packed_ionization) / 1000.0

            potential_index = (
                _ionization_potential_index_compiled(atomic_number, ion_stage) - 1
            )
            if (
                0 <= potential_index
                and potential_index < ionization_potential_table.size
            ):
                potential_value = ionization_potential_table[potential_index]
                if potential_value > 0.0:
                    stage_ionization_potential_ev = (
                        potential_value / WAVENUMBER_PER_EV_REFERENCE
                    )
                elif (
                    0 <= potential_index - 1
                    and potential_index - 1 < ionization_potential_table.size
                    and ionization_potential_table[potential_index - 1] > 0.0
                ):
                    stage_ionization_potential_ev = (
                        ionization_potential_table[potential_index - 1]
                        / WAVENUMBER_PER_EV_REFERENCE
                    )
            if stage_ionization_potential_ev <= 0.0 and ion_stage > 1:
                stage_ionization_potential_ev = ionization_potential_ev[ion_stage - 2]

            lowering_potential_by_stage[ion_stage - 1] = stage_lowering_potential
            ionization_potential_ev[ion_stage - 1] = stage_ionization_potential_ev

            if atomic_number >= 20 and atomic_number < 29:
                partition_values[ion_stage - 1] = _python_float_max_compiled(
                    _iron_group_partition_function_compiled(
                        iron_group_grid,
                        atomic_number,
                        ion_stage,
                        math.log10(temperature) if temperature > 0.0 else 0.0,
                        stage_lowering_potential * WAVENUMBER_PER_EV_REFERENCE,
                    ),
                    1.0,
                )
                continue

            special_value, density_parameter, used_special = (
                _special_partition_compiled(
                    table_index,
                    hc_over_kt,
                    special_level_grid,
                )
            )
            if used_special:
                partition_value = _python_float_max_compiled(special_value, 1.0)
                if density_parameter > 0.0:
                    partition_value = _occupation_correction_compiled(
                        partition_value,
                        ion_charge,
                        _python_float_max_compiled(statistical_weight, 2.0),
                        stage_ionization_potential_ev,
                        stage_lowering_potential,
                        thermal_energy_ev,
                        density_parameter,
                    )
                partition_values[ion_stage - 1] = _python_float_max_compiled(
                    partition_value, 1.0
                )
                continue

            safe_temperature = (
                temperature
                if (math.isfinite(temperature) and temperature > 0.0)
                else 1.0
            )
            reference_temperature = _python_float_max_compiled(
                stage_ionization_potential_ev * 2000.0 / 11.0, 1.0e-12
            )
            temperature_bin = max(
                1, min(9, int(safe_temperature / reference_temperature - 0.5))
            )
            temperature_delta = (
                safe_temperature / reference_temperature - float(temperature_bin) - 0.5
            )
            minimum_partition = 1.0
            metadata_row = (temperature_bin + 1) // 2
            packed_partition = packed_level_metadata[metadata_row - 1, table_index - 1]
            first_packed = packed_partition // 100000
            second_packed = packed_partition - first_packed * 100000
            second_value = second_packed // 10
            scale_index = max(1, min(4, second_packed - second_value * 10))

            if temperature_bin % 2 == 1:
                left_partition = (
                    float(first_packed) * ion_fraction_scale[scale_index - 1]
                )
                right_partition = (
                    float(second_value) * ion_fraction_scale[scale_index - 1]
                )
                if temperature_delta < 0.0 and scale_index <= 1:
                    rounded_left = int(left_partition)
                    if rounded_left == int(right_partition + 0.5):
                        minimum_partition = float(rounded_left)
            else:
                left_partition = (
                    float(second_value) * ion_fraction_scale[scale_index - 1]
                )
                next_packed = packed_level_metadata[metadata_row, table_index - 1]
                next_first_packed = next_packed // 100000
                next_scale_index = max(1, min(4, int(next_packed % 10)))
                right_partition = (
                    float(next_first_packed) * ion_fraction_scale[next_scale_index - 1]
                )

            partition_value = _python_float_max_compiled(
                minimum_partition,
                left_partition + (right_partition - left_partition) * temperature_delta,
            )

            if temperature < reference_temperature * 2.0:
                partition_values[ion_stage - 1] = _python_float_max_compiled(
                    partition_value, 1.0
                )
                continue

            if (
                statistical_weight != 0.0
                and stage_lowering_potential >= 0.1
                and temperature >= reference_temperature * 4.0
            ):
                effective_thermal_energy_ev = thermal_energy_ev
                if temperature > reference_temperature * 11.0:
                    effective_thermal_energy_ev = (
                        reference_temperature * 11.0
                    ) * BOLTZMANN_EV_PER_K_REFERENCE
                density_parameter = 0.1 / _python_float_max_compiled(
                    effective_thermal_energy_ev, 1.0e-30
                )
                partition_value = _occupation_correction_compiled(
                    partition_value,
                    ion_charge,
                    statistical_weight,
                    stage_ionization_potential_ev,
                    stage_lowering_potential,
                    effective_thermal_energy_ev,
                    density_parameter,
                )
            partition_values[ion_stage - 1] = _python_float_max_compiled(
                partition_value, 1.0
            )

        if base_mode != 3 and base_mode != 5:
            saha_factor = (
                SAHA_COEFFICIENT_REFERENCE * temperature * math.sqrt(temperature)
            )
            saha_factor /= electron_density
            for ion_stage in range(2, work_ion_count + 1):
                idx = ion_stage - 1
                ion_fractions[idx] = (
                    saha_factor
                    * partition_values[idx]
                    / _python_float_max_compiled(partition_values[idx - 1], 1.0e-300)
                    * math.exp(
                        -(
                            ionization_potential_ev[idx - 1]
                            - lowering_potential_by_stage[idx - 1]
                        )
                        / _python_float_max_compiled(thermal_energy_ev, 1.0e-30)
                    )
                )
            ion_fractions[0] = 1.0
            reverse_index = work_ion_count + 1
            for _ in range(2, work_ion_count + 1):
                reverse_index -= 1
                ion_fractions[0] = (
                    1.0 + ion_fractions[reverse_index - 1] * ion_fractions[0]
                )
            ion_fractions[0] = 1.0 / _python_float_max_compiled(
                ion_fractions[0], 1.0e-300
            )
            for ion_stage in range(2, work_ion_count + 1):
                idx = ion_stage - 1
                ion_fractions[idx] = ion_fractions[idx - 1] * ion_fractions[idx]

        if return_all_ion_stages:
            return _format_saha_output_compiled(
                base_mode,
                ion_stage_count,
                work_ion_count,
                partition_values,
                ionization_potential_ev,
                ion_fractions,
            )

        selected_index = min(max(ion_stage_count, 1), work_ion_count) - 1
        output = np.zeros(ion_stage_count, dtype=np.float64)
        if base_mode == 1:
            output[0] = ion_fractions[selected_index] / _python_float_max_compiled(
                partition_values[selected_index],
                1.0e-300,
            )
            return output
        if base_mode == 2:
            output[0] = ion_fractions[selected_index]
            return output
        if base_mode == 3:
            output[0] = partition_values[selected_index]
            return output
        if base_mode == 4:
            products = np.empty(work_ion_count - 1, dtype=np.float64)
            for idx in range(1, work_ion_count):
                products[idx - 1] = ion_fractions[idx] * float(idx)
            output[0] = _numpy_small_sum_compiled(products)
            return output
        if base_mode == 5:
            return _format_saha_output_compiled(
                base_mode,
                ion_stage_count,
                work_ion_count,
                partition_values,
                ionization_potential_ev,
                ion_fractions,
            )
        # Unreachable: the dispatcher validates the mode before kernel entry.
        return output

    @_njit
    def _saha_partition_depth_batch_kernel(
        temperatures,
        electron_densities,
        atomic_number,
        ion_stage_count,
        population_mode,
        charge_square_densities,
        has_charge_square_density,
        output,
        packed_level_metadata,
        ionization_potential_table,
        iron_group_grid,
        element_block_offsets,
        special_level_grid,
        ion_fraction_scale,
    ):
        """Depth loop over the per-depth kernel: one boxing per batch.

        Each row is the unmodified `_saha_partition_depth_kernel` result for
        that depth's (temperature, electron density, charge-square density),
        so every element is bit-identical to the per-call path by
        construction.
        """

        for depth_index in range(temperatures.size):
            output[depth_index, :] = _saha_partition_depth_kernel(
                temperatures[depth_index],
                electron_densities[depth_index],
                atomic_number,
                ion_stage_count,
                population_mode,
                charge_square_densities[depth_index],
                has_charge_square_density,
                packed_level_metadata,
                ionization_potential_table,
                iron_group_grid,
                element_block_offsets,
                special_level_grid,
                ion_fraction_scale,
            )

    @numba.njit(parallel=True, nogil=True, cache=True)
    def _iterate_electron_density_parallel(
        temperature_k,
        thermal_energy_erg,
        gas_pressure,
        electron_density,
        total_nuclei_number_density,
        charge_square_density,
        elemental_abundances_by_layer,
        mean_nuclear_mass_amu,
        mass_density,
        ion_stage_populations_by_packed_slot,
        ion_stage_count_by_z,
        slot_start_by_z,
        atomic_mass_gram,
        max_iterations,
        tolerance,
        converged_flags,
        packed_level_metadata,
        ionization_potential_table,
        iron_group_grid,
        element_block_offsets,
        special_level_grid,
        ion_fraction_scale,
    ):
        """Pure-njit parallel electron-density depth sweep.

        The 80 depth layers are independent (each converges its own electron
        density and writes only its own row of the state arrays), so the sweep
        is a ``numba.prange`` over layers. Each layer runs the exact Newton /
        Saha loop of the scalar ``_iterate_electron_density_layer`` but calls
        the compiled ``_saha_partition_depth_kernel`` directly (tables boxed
        once for the whole kernel instead of once per ~425k scalar calls) --
        eliminating both the Python-wrapper boxing and the GIL, so the sweep
        scales across cores where the old thread pool was flat. Convergence
        failures are reported via ``converged_flags`` (the caller raises in
        Python) so no dynamic-message exception is needed inside prange.
        """
        layer_count = temperature_k.shape[0]
        population_width = ion_stage_populations_by_packed_slot.shape[1]
        for layer_index in numba.prange(layer_count):
            total_particle_density = gas_pressure[layer_index] / max(
                thermal_energy_erg[layer_index], 1.0e-300
            )
            total_nuclei_number_density[layer_index] = (
                total_particle_density - electron_density[layer_index]
            )
            layer_converged = False
            for _ in range(max_iterations):
                for slot in range(population_width):
                    ion_stage_populations_by_packed_slot[layer_index, slot] = 0.0
                updated_electron_density = 0.0
                charge_square = 0.0
                for atomic_number in range(1, 100):
                    ion_stage_count = ion_stage_count_by_z[atomic_number]
                    ion_fractions = _saha_partition_depth_kernel(
                        temperature_k[layer_index],
                        electron_density[layer_index],
                        atomic_number,
                        ion_stage_count,
                        12,
                        max(charge_square_density[layer_index], 1.0e-30),
                        True,
                        packed_level_metadata,
                        ionization_potential_table,
                        iron_group_grid,
                        element_block_offsets,
                        special_level_grid,
                        ion_fraction_scale,
                    )
                    slot_start = slot_start_by_z[atomic_number]
                    abundance_scale = (
                        total_nuclei_number_density[layer_index]
                        * elemental_abundances_by_layer[layer_index, atomic_number - 1]
                    )
                    for ion_stage_index in range(ion_stage_count):
                        population = ion_fractions[ion_stage_index] * abundance_scale
                        output_index = slot_start + ion_stage_index
                        if output_index < population_width:
                            ion_stage_populations_by_packed_slot[
                                layer_index, output_index
                            ] = population
                        charge_square += population * (
                            ion_stage_index * ion_stage_index
                        )
                        updated_electron_density += population * ion_stage_index

                updated_electron_density = max(
                    updated_electron_density,
                    electron_density[layer_index] * 0.5,
                )
                updated_electron_density = 0.5 * (
                    updated_electron_density + electron_density[layer_index]
                )
                relative_error = abs(
                    (electron_density[layer_index] - updated_electron_density)
                    / max(updated_electron_density, 1.0e-300)
                )
                electron_density[layer_index] = updated_electron_density
                total_nuclei_number_density[layer_index] = (
                    total_particle_density - electron_density[layer_index]
                )
                charge_square_density[layer_index] = (
                    charge_square + electron_density[layer_index]
                )
                if relative_error < tolerance:
                    layer_converged = True
                    break

            if layer_converged:
                converged_flags[layer_index] = 1
                mass_density[layer_index] = (
                    total_nuclei_number_density[layer_index]
                    * mean_nuclear_mass_amu[layer_index]
                    * atomic_mass_gram
                )
            else:
                converged_flags[layer_index] = 0


_SAHA_KERNEL_TABLES: "tuple | None" = None


def _saha_kernel_table_arrays():
    """Unpack the lru_cache(1) table loaders once into plain kernel inputs.

    The unpacked tuple is cached for the process lifetime, matching the
    lru_cache(1) loaders it mirrors: if those loaders were ever cleared and
    pointed at different table files, this cache must be reset too (set
    ``_SAHA_KERNEL_TABLES = None``) — table swapping mid-process is not a
    supported workflow anywhere in this package.
    """

    global _SAHA_KERNEL_TABLES
    if _SAHA_KERNEL_TABLES is None:
        special_tables = load_special_partition_tables()
        level_arrays = [
            getattr(special_tables, name) for name in _SPECIAL_PARTITION_KEYS[1:]
        ]
        # Zero-padded row-per-array layout: element values are copied bits, so
        # kernel reads are identical to reading the individual arrays; a single
        # 2-D argument keeps the per-call numba boxing overhead small.
        special_level_grid = np.zeros(
            (len(level_arrays), max(array.size for array in level_arrays)),
            dtype=np.float64,
        )
        for row, array in enumerate(level_arrays):
            special_level_grid[row, : array.size] = array
        _SAHA_KERNEL_TABLES = (
            load_packed_level_metadata(),
            load_ionization_potential_table_cm(),
            load_iron_group_partition_grid(),
            special_tables.element_block_offsets,
            special_level_grid,
            ION_FRACTION_SCALE,
        )
    return _SAHA_KERNEL_TABLES


def _saha_partition_depth_compiled(
    temperature_k: float,
    electron_density_cm3: float,
    atomic_number: int,
    ion_stage_count: int,
    population_mode: int,
    charge_square_density_cm3: float | None,
) -> np.ndarray:
    """Recompute Saha fractions or partition functions for one depth."""

    (
        packed_level_metadata,
        ionization_potential_table,
        iron_group_grid,
        element_block_offsets,
        special_level_grid,
        ion_fraction_scale,
    ) = _saha_kernel_table_arrays()

    atomic_number = _validate_atomic_number(atomic_number)
    clamped_ion_count = max(1, int(ion_stage_count))
    mode = int(population_mode)
    base_mode = mode if mode <= 10 else mode - 10

    # Guard out-of-range table indices and unsupported modes so the kernel
    # only ever runs validated inputs.
    start_index, available_ion_count = _start_and_available_ion_count(
        atomic_number,
        element_block_offsets,
    )
    work_ion_count = min(clamped_ion_count + 2, available_ion_count)
    table_index = start_index
    for _ in range(work_ion_count):
        if table_index < 1 or table_index > packed_level_metadata.shape[1]:
            raise ValueError(f"partition table index out of range: {table_index}")
        table_index += 1
    if mode >= 10 or base_mode == 5:
        if base_mode not in (1, 2, 3, 4, 5):
            raise NotImplementedError(f"population mode {base_mode} is not implemented")
    elif base_mode not in (1, 2, 3, 4):
        raise NotImplementedError(f"population mode {mode} is not implemented")
    if 20 <= atomic_number < 29 and math.isinf(max(float(temperature_k), 1.0)):
        # Iron-group elements hit int(inf) inside _temperature_bin at infinite
        # temperature; raise CPython's OverflowError verbatim.
        raise OverflowError("cannot convert float infinity to integer")

    return _saha_partition_depth_kernel(
        float(temperature_k),
        float(electron_density_cm3),
        atomic_number,
        int(ion_stage_count),
        mode,
        0.0 if charge_square_density_cm3 is None else float(charge_square_density_cm3),
        charge_square_density_cm3 is not None,
        packed_level_metadata,
        ionization_potential_table,
        iron_group_grid,
        element_block_offsets,
        special_level_grid,
        ion_fraction_scale,
    )


def saha_partition_depth(
    *,
    temperature_k: float,
    electron_density_cm3: float,
    total_nuclei_number_density_cm3: float,
    elemental_abundance: float,
    atomic_number: int,
    ion_stage_count: int,
    population_mode: int,
    charge_square_density_cm3: float | None = None,
) -> np.ndarray:
    """Return Saha fractions, partition functions, or diagnostics.

    The result depends only on temperature, electron density, element,
    requested ion stages, mode, and charge-square density.  Neutral atom
    density and abundance are accepted to keep the population call site
    explicit and preserve the validated numerical contract.
    """

    _ = (total_nuclei_number_density_cm3, elemental_abundance)
    return _saha_partition_depth_compiled(
        float(temperature_k),
        float(electron_density_cm3),
        int(atomic_number),
        int(ion_stage_count),
        int(population_mode),
        None if charge_square_density_cm3 is None else float(charge_square_density_cm3),
    )


def _saha_batch_output_width(ion_stage_count: int, population_mode: int) -> int:
    """Per-depth Saha/partition output length for fixed stages and mode.

    Mirrors the kernel/oracle exactly: base mode 5 always emits the 61-slot
    diagnostic block; every other mode emits ``max(1, ion_stage_count)``
    entries (the kernel clamps the requested count before allocating).
    """

    mode = int(population_mode)
    base_mode = mode if mode <= 10 else mode - 10
    if base_mode == 5:
        return 61
    return max(1, int(ion_stage_count))


def saha_partition_depth_batch(
    temperature_k: np.ndarray,
    electron_density_cm3: np.ndarray,
    atomic_number: int,
    ion_stage_count: int,
    population_mode: int,
    charge_square_density_cm3: np.ndarray | None = None,
) -> np.ndarray:
    """Depth-batched ``saha_partition_depth`` for fixed (Z, stages, mode).

    Returns a ``(depth_count, output_width)`` float64 array whose row ``i``
    is bit-identical to ``saha_partition_depth`` called with
    ``temperature_k[i]``, ``electron_density_cm3[i]`` and (when given)
    ``charge_square_density_cm3[i]``: the compiled path loops the exact
    per-depth kernel inside one numba driver call (tables boxed once per
    batch instead of once per depth).
    """

    temperatures = np.ascontiguousarray(temperature_k, dtype=np.float64)
    electron_densities = np.ascontiguousarray(electron_density_cm3, dtype=np.float64)
    if temperatures.ndim != 1:
        raise ValueError("saha_partition_depth_batch expects 1-D depth arrays")
    if electron_densities.shape != temperatures.shape:
        raise ValueError("electron-density array must match the temperature array")
    has_charge_square = charge_square_density_cm3 is not None
    if has_charge_square:
        charge_square_densities = np.ascontiguousarray(
            charge_square_density_cm3, dtype=np.float64
        )
        if charge_square_densities.shape != temperatures.shape:
            raise ValueError(
                "charge-square-density array must match the temperature array"
            )
    else:
        charge_square_densities = np.zeros_like(temperatures)

    depth_count = int(temperatures.size)
    output_width = _saha_batch_output_width(ion_stage_count, population_mode)

    (
        packed_level_metadata,
        ionization_potential_table,
        iron_group_grid,
        element_block_offsets,
        special_level_grid,
        ion_fraction_scale,
    ) = _saha_kernel_table_arrays()

    # Batch-invariant guards, transcribed from _saha_partition_depth_compiled
    # (Z, stages, and mode are fixed across the depth loop, so they validate
    # once); the iron-group infinite-temperature OverflowError is the only
    # per-depth guard and is vectorized below.
    atomic_number = _validate_atomic_number(atomic_number)
    clamped_ion_count = max(1, int(ion_stage_count))
    mode = int(population_mode)
    base_mode = mode if mode <= 10 else mode - 10
    start_index, available_ion_count = _start_and_available_ion_count(
        atomic_number,
        element_block_offsets,
    )
    work_ion_count = min(clamped_ion_count + 2, available_ion_count)
    table_index = start_index
    for _ in range(work_ion_count):
        if table_index < 1 or table_index > packed_level_metadata.shape[1]:
            raise ValueError(f"partition table index out of range: {table_index}")
        table_index += 1
    if mode >= 10 or base_mode == 5:
        if base_mode not in (1, 2, 3, 4, 5):
            raise NotImplementedError(f"population mode {base_mode} is not implemented")
    elif base_mode not in (1, 2, 3, 4):
        raise NotImplementedError(f"population mode {mode} is not implemented")
    if (
        20 <= atomic_number < 29
        and depth_count > 0
        and bool(np.any(np.isinf(np.maximum(temperatures, 1.0))))
    ):
        raise OverflowError("cannot convert float infinity to integer")

    output = np.empty((depth_count, output_width), dtype=np.float64)
    if depth_count > 0:
        _saha_partition_depth_batch_kernel(
            temperatures,
            electron_densities,
            atomic_number,
            int(ion_stage_count),
            mode,
            charge_square_densities,
            has_charge_square,
            output,
            packed_level_metadata,
            ionization_potential_table,
            iron_group_grid,
            element_block_offsets,
            special_level_grid,
            ion_fraction_scale,
        )
    return output


def _iterate_electron_density_layer(
    layer_index: int,
    *,
    temperature_k: np.ndarray,
    thermal_energy_erg: np.ndarray,
    state: AtmosphereRuntimeState,
    max_iterations: int,
    tolerance: float,
) -> None:
    total_particle_density = state.gas_pressure[layer_index] / max(
        thermal_energy_erg[layer_index],
        1.0e-300,
    )
    state.total_nuclei_number_density[layer_index] = (
        total_particle_density - state.electron_density[layer_index]
    )

    converged = False
    for _ in range(max_iterations):
        state.ion_stage_populations_by_packed_slot[layer_index, :] = 0.0
        updated_electron_density = 0.0
        charge_square_density = 0.0

        for atomic_number in range(1, 100):
            ion_stage_count = ion_stage_count_for_atomic_number(atomic_number)
            ion_fractions = saha_partition_depth(
                temperature_k=float(temperature_k[layer_index]),
                electron_density_cm3=float(state.electron_density[layer_index]),
                total_nuclei_number_density_cm3=float(
                    state.total_nuclei_number_density[layer_index]
                ),
                elemental_abundance=float(
                    state.elemental_abundances_by_layer[layer_index, atomic_number - 1]
                ),
                atomic_number=atomic_number,
                ion_stage_count=ion_stage_count,
                population_mode=12,
                charge_square_density_cm3=float(
                    max(state.charge_square_density[layer_index], 1.0e-30)
                ),
            )
            slot_start = atomic_population_slot_start(atomic_number)
            abundance_scale = (
                state.total_nuclei_number_density[layer_index]
                * state.elemental_abundances_by_layer[layer_index, atomic_number - 1]
            )

            for ion_stage_index in range(ion_stage_count):
                population = ion_fractions[ion_stage_index] * abundance_scale
                output_index = slot_start + ion_stage_index
                if output_index < state.ion_stage_populations_by_packed_slot.shape[1]:
                    state.ion_stage_populations_by_packed_slot[
                        layer_index, output_index
                    ] = population
                charge_square_density += population * (ion_stage_index**2)
                updated_electron_density += population * ion_stage_index

        updated_electron_density = max(
            updated_electron_density,
            state.electron_density[layer_index] * 0.5,
        )
        updated_electron_density = 0.5 * (
            updated_electron_density + state.electron_density[layer_index]
        )
        relative_error = abs(
            (state.electron_density[layer_index] - updated_electron_density)
            / max(updated_electron_density, 1.0e-300)
        )

        state.electron_density[layer_index] = updated_electron_density
        state.total_nuclei_number_density[layer_index] = (
            total_particle_density - state.electron_density[layer_index]
        )
        state.charge_square_density[layer_index] = (
            charge_square_density + state.electron_density[layer_index]
        )
        if relative_error < tolerance:
            converged = True
            break

    if not converged:
        raise RuntimeError(
            f"electron-density iteration did not converge at depth index {layer_index}"
        )

    state.mass_density[layer_index] = (
        state.total_nuclei_number_density[layer_index]
        * state.mean_nuclear_mass_amu[layer_index]
        * ATOMIC_MASS_GRAM_REFERENCE
    )


def iterate_electron_density(
    *,
    temperature_k: np.ndarray,
    thermal_energy_erg: np.ndarray,
    state: AtmosphereRuntimeState,
    max_iterations: int = 200,
    tolerance: float = 1.0e-4,
) -> None:
    """Iterate electron density and atomic ion-stage populations."""

    temperature = np.asarray(temperature_k, dtype=np.float64)
    thermal_energy = np.asarray(thermal_energy_erg, dtype=np.float64)
    layer_count = temperature.size
    if thermal_energy.shape != temperature.shape:
        raise ValueError("thermal_energy_erg must match temperature_k shape")
    if state.ion_stage_populations_by_packed_slot.shape[0] != layer_count:
        raise ValueError(
            "state.ion_stage_populations_by_packed_slot must match atmosphere layer count"
        )

    # The 80 depth layers are independent -- each converges its own electron
    # density and writes only its own row of the state arrays -- so the
    # sweep runs as a single njit(parallel=True) prange over layers (GIL-free,
    # Saha tables boxed once for the whole sweep, ~12x over the per-layer Python
    # reference at one thread and ~4x more across cores; bit-identical results).
    # The serial per-layer loop below is the single-layer / no-numba fallback.
    use_parallel = _NUMBA_AVAILABLE and layer_count > 1
    if not use_parallel:
        for layer_index in range(layer_count):
            _iterate_electron_density_layer(
                layer_index,
                temperature_k=temperature,
                thermal_energy_erg=thermal_energy,
                state=state,
                max_iterations=int(max_iterations),
                tolerance=float(tolerance),
            )
        return

    # --- pure-njit(parallel=True) prange path ---
    # Precompute the Z-indexed ion-stage-count / slot-start maps and validate
    # every (Z, mode=12) table access once (these are batch-invariant across
    # layers), then run the compiled depth sweep.
    (
        packed_level_metadata,
        ionization_potential_table,
        iron_group_grid,
        element_block_offsets,
        special_level_grid,
        ion_fraction_scale,
    ) = _saha_kernel_table_arrays()
    ion_stage_count_by_z = np.zeros(100, dtype=np.int64)
    slot_start_by_z = np.zeros(100, dtype=np.int64)
    for atomic_number in range(1, 100):
        ion_stage_count = ion_stage_count_for_atomic_number(atomic_number)
        ion_stage_count_by_z[atomic_number] = ion_stage_count
        slot_start_by_z[atomic_number] = atomic_population_slot_start(atomic_number)
        # Batch-invariant guards, transcribed from _saha_partition_depth_compiled
        # so the prange only ever runs validated (Z, stages, mode) inputs.
        z_validated = _validate_atomic_number(atomic_number)
        start_index, available_ion_count = _start_and_available_ion_count(
            z_validated, element_block_offsets
        )
        work_ion_count = min(max(1, int(ion_stage_count)) + 2, available_ion_count)
        table_index = start_index
        for _ in range(work_ion_count):
            if table_index < 1 or table_index > packed_level_metadata.shape[1]:
                raise ValueError(f"partition table index out of range: {table_index}")
            table_index += 1
        if 20 <= z_validated < 29 and bool(
            np.any(np.isinf(np.maximum(temperature, 1.0)))
        ):
            raise OverflowError("cannot convert float infinity to integer")

    converged_flags = np.zeros(layer_count, dtype=np.int64)
    abundances = np.ascontiguousarray(
        state.elemental_abundances_by_layer, dtype=np.float64
    )
    _iterate_electron_density_parallel(
        np.ascontiguousarray(temperature, dtype=np.float64),
        np.ascontiguousarray(thermal_energy, dtype=np.float64),
        np.ascontiguousarray(state.gas_pressure, dtype=np.float64),
        state.electron_density,
        state.total_nuclei_number_density,
        state.charge_square_density,
        abundances,
        np.ascontiguousarray(state.mean_nuclear_mass_amu, dtype=np.float64),
        state.mass_density,
        state.ion_stage_populations_by_packed_slot,
        ion_stage_count_by_z,
        slot_start_by_z,
        float(ATOMIC_MASS_GRAM_REFERENCE),
        int(max_iterations),
        float(tolerance),
        converged_flags,
        packed_level_metadata,
        ionization_potential_table,
        iron_group_grid,
        element_block_offsets,
        special_level_grid,
        ion_fraction_scale,
    )
    for layer_index in range(layer_count):
        if converged_flags[layer_index] == 0:
            raise RuntimeError(
                "electron-density iteration did not converge at depth index "
                f"{layer_index}"
            )


def _fill_atomic_population_slice(
    *,
    atomic_number: int,
    ion_stage_count: int,
    population_mode: int,
    output: np.ndarray,
    temperature_k: np.ndarray,
    state: AtmosphereRuntimeState,
) -> None:
    layer_count = temperature_k.size
    abundance_by_layer = state.elemental_abundances_by_layer[:, int(atomic_number) - 1]

    # Depth-batched Saha evaluation: row i is bit-identical to
    # the per-layer saha_partition_depth call (total_nuclei_number_density/abundance are
    # discarded, so passing them per layer is unnecessary). The mode<10 vs >=10 /
    # copy_count post-processing below is preserved exactly.
    charge_square_by_layer = np.maximum(
        np.asarray(state.charge_square_density, dtype=np.float64), 1.0e-30
    )
    values_batch = saha_partition_depth_batch(
        np.asarray(temperature_k, dtype=np.float64),
        np.asarray(state.electron_density, dtype=np.float64),
        int(atomic_number),
        int(ion_stage_count),
        int(population_mode),
        charge_square_by_layer,
    )

    if population_mode < 10:
        output[:, 0] = (
            np.ascontiguousarray(values_batch[:, 0], dtype=np.float64)
            * state.total_nuclei_number_density
            * abundance_by_layer
        )
        return

    copy_count = min(output.shape[1], values_batch.shape[1])
    fraction_table = np.zeros((layer_count, copy_count), dtype=np.float64)
    fraction_table[:, :copy_count] = values_batch[:, :copy_count]

    output[:, :] = 0.0
    output[:, :copy_count] = (
        fraction_table
        * (state.total_nuclei_number_density * abundance_by_layer)[:, np.newaxis]
    )


def populate_species(
    *,
    code: float,
    population_mode: int,
    output: np.ndarray,
    molecules_enabled: bool,
    pressure_iteration_enabled: bool,
    temperature_k: np.ndarray,
    thermal_energy_erg: np.ndarray,
    state: AtmosphereRuntimeState,
    temperature_iteration_index: int,
    temperature_iteration_cache: dict[str, int],
    molecular_state: "MolecularEquilibriumState | None" = None,
) -> None:
    """Populate one packed species output slice.

    This uses atomic Saha populations when molecules are disabled and the
    molecular-equilibrium state when molecules are enabled.
    """

    temperature = np.asarray(temperature_k, dtype=np.float64)
    output_array = np.asarray(output, dtype=np.float64)
    if output_array.shape[0] != temperature.size:
        raise ValueError("output must have one row per atmosphere layer")

    previous_iteration = temperature_iteration_cache.get("pops_itemp", -1)

    if molecules_enabled:
        if molecular_state is None:
            raise ValueError("molecular_state is required when molecules are enabled")
        if (
            pressure_iteration_enabled
            and temperature_iteration_index != previous_iteration
        ):
            from .molecular_equilibrium import solve_molecular_equilibrium

            solve_molecular_equilibrium(
                molecular_state,
                population_mode=int(population_mode),
            )
            temperature_iteration_cache["pops_itemp"] = int(temperature_iteration_index)
        if float(code) == 0.0:
            return
        from .molecular_equilibrium import populate_molecular_species

        populate_molecular_species(
            molecular_state,
            code=float(code),
            population_mode=int(population_mode),
            output=output_array,
        )
        return

    if pressure_iteration_enabled and temperature_iteration_index != previous_iteration:
        iterate_electron_density(
            temperature_k=temperature,
            thermal_energy_erg=thermal_energy_erg,
            state=state,
        )
        temperature_iteration_cache["pops_itemp"] = int(temperature_iteration_index)

    if float(code) == 0.0:
        return
    if float(code) >= 100.0:
        raise ValueError(
            "molecule code requested while molecular populations are unavailable"
        )

    atomic_number, ion_stage_count = decode_population_code(float(code))
    _fill_atomic_population_slice(
        atomic_number=atomic_number,
        ion_stage_count=ion_stage_count,
        population_mode=int(population_mode),
        output=output_array,
        temperature_k=temperature,
        state=state,
    )


def populate_all_species(
    *,
    temperature_k: np.ndarray,
    thermal_energy_erg: np.ndarray,
    state: AtmosphereRuntimeState,
    molecules_enabled: bool,
    pressure_iteration_enabled: bool,
    temperature_iteration_index: int,
    temperature_iteration_cache: dict[str, int],
    molecular_state: "MolecularEquilibriumState | None" = None,
) -> None:
    """Run the packed species schedule into runtime population tables."""

    temperature = np.asarray(temperature_k, dtype=np.float64)
    if state.ion_stage_populations_by_packed_slot.shape[0] != temperature.size:
        raise ValueError(
            "ion_stage_populations_by_packed_slot must match atmosphere layer count"
        )
    if (
        state.partition_normalized_populations_by_packed_slot.shape[0]
        != temperature.size
    ):
        raise ValueError(
            "partition_normalized_populations_by_packed_slot must match atmosphere layer count"
        )
    if molecules_enabled and molecular_state is None:
        raise ValueError("molecular_state is required when molecules are enabled")

    for job in population_job_schedule(include_molecules=bool(molecules_enabled)):
        destination = (
            state.ion_stage_populations_by_packed_slot
            if job.target == "ion_stage_populations_by_packed_slot"
            else state.partition_normalized_populations_by_packed_slot
        )
        output = destination[:, job.start_slot : job.start_slot + job.output_slots]
        populate_species(
            code=job.code,
            population_mode=job.mode,
            output=output,
            molecules_enabled=bool(molecules_enabled),
            molecular_state=molecular_state,
            pressure_iteration_enabled=pressure_iteration_enabled,
            temperature_k=temperature,
            thermal_energy_erg=thermal_energy_erg,
            state=state,
            temperature_iteration_index=temperature_iteration_index,
            temperature_iteration_cache=temperature_iteration_cache,
        )


# --- packaged EOS tables (merged from equation_of_state_tables.py) ---


class EquationOfStateTableError(RuntimeError):
    """Raised when a required packaged EOS table is missing or malformed."""


@dataclass(frozen=True)
class SpecialPartitionTables:
    """Level tables used by special partition-function branches."""

    element_block_offsets: np.ndarray
    hydrogen_neutral_level_energy_cm: np.ndarray
    hydrogen_neutral_level_statistical_weight: np.ndarray
    helium_neutral_level_energy_cm: np.ndarray
    helium_neutral_level_statistical_weight: np.ndarray
    helium_singly_ionized_level_energy_cm: np.ndarray
    helium_singly_ionized_level_statistical_weight: np.ndarray
    carbon_neutral_level_energy_cm: np.ndarray
    carbon_neutral_level_statistical_weight: np.ndarray
    carbon_singly_ionized_level_energy_cm: np.ndarray
    carbon_singly_ionized_level_statistical_weight: np.ndarray
    magnesium_neutral_level_energy_cm: np.ndarray
    magnesium_neutral_level_statistical_weight: np.ndarray
    magnesium_singly_ionized_level_energy_cm: np.ndarray
    magnesium_singly_ionized_level_statistical_weight: np.ndarray
    aluminum_neutral_level_energy_cm: np.ndarray
    aluminum_neutral_level_statistical_weight: np.ndarray
    silicon_neutral_level_energy_cm: np.ndarray
    silicon_neutral_level_statistical_weight: np.ndarray
    silicon_singly_ionized_level_energy_cm: np.ndarray
    silicon_singly_ionized_level_statistical_weight: np.ndarray
    sodium_neutral_level_energy_cm: np.ndarray
    sodium_neutral_level_statistical_weight: np.ndarray
    oxygen_neutral_level_energy_cm: np.ndarray
    oxygen_neutral_level_statistical_weight: np.ndarray
    boron_neutral_level_energy_cm: np.ndarray
    boron_neutral_level_statistical_weight: np.ndarray
    potassium_neutral_level_energy_cm: np.ndarray
    potassium_neutral_level_statistical_weight: np.ndarray


_SPECIAL_PARTITION_KEYS = (
    "element_block_offsets",
    "hydrogen_neutral_level_energy_cm",
    "hydrogen_neutral_level_statistical_weight",
    "helium_neutral_level_energy_cm",
    "helium_neutral_level_statistical_weight",
    "helium_singly_ionized_level_energy_cm",
    "helium_singly_ionized_level_statistical_weight",
    "carbon_neutral_level_energy_cm",
    "carbon_neutral_level_statistical_weight",
    "carbon_singly_ionized_level_energy_cm",
    "carbon_singly_ionized_level_statistical_weight",
    "magnesium_neutral_level_energy_cm",
    "magnesium_neutral_level_statistical_weight",
    "magnesium_singly_ionized_level_energy_cm",
    "magnesium_singly_ionized_level_statistical_weight",
    "aluminum_neutral_level_energy_cm",
    "aluminum_neutral_level_statistical_weight",
    "silicon_neutral_level_energy_cm",
    "silicon_neutral_level_statistical_weight",
    "silicon_singly_ionized_level_energy_cm",
    "silicon_singly_ionized_level_statistical_weight",
    "sodium_neutral_level_energy_cm",
    "sodium_neutral_level_statistical_weight",
    "oxygen_neutral_level_energy_cm",
    "oxygen_neutral_level_statistical_weight",
    "boron_neutral_level_energy_cm",
    "boron_neutral_level_statistical_weight",
    "potassium_neutral_level_energy_cm",
    "potassium_neutral_level_statistical_weight",
)


@lru_cache(maxsize=1)
def load_iron_group_partition_grid() -> np.ndarray:
    """Return the PFIRON grid with shape [lowering, temperature, ion, element]."""

    table = np.asarray(
        load_table_arrays(
            atmosphere_table_path("iron_group_partition_tables.npz"),
            ("iron_group_partition_grid",),
            error_type=EquationOfStateTableError,
        )["iron_group_partition_grid"],
        dtype=np.float64,
    )
    if table.shape != (7, 56, 10, 9):
        raise EquationOfStateTableError(
            f"PFIRON grid has shape {table.shape}, expected (7, 56, 10, 9)"
        )
    return table


@lru_cache(maxsize=1)
def load_ionization_potential_table_cm() -> np.ndarray:
    """Return the packed ionization-potential table in cm^-1."""

    table = np.asarray(
        load_table_arrays(
            atmosphere_table_path("ionization_potential_tables.npz"),
            ("ionization_potential_cm",),
            error_type=EquationOfStateTableError,
        )["ionization_potential_cm"],
        dtype=np.float64,
    )
    if table.ndim != 1:
        raise EquationOfStateTableError(
            "Ionization-potential table must be one-dimensional"
        )
    return table


@lru_cache(maxsize=1)
def load_packed_level_metadata() -> np.ndarray:
    """Return the packed level metadata table used by Saha populations."""

    table = np.asarray(
        load_table_arrays(
            atmosphere_table_path("packed_level_metadata.npz"),
            ("packed_level_metadata",),
            error_type=EquationOfStateTableError,
        )["packed_level_metadata"],
        dtype=np.int64,
    )
    if table.shape[0] != 6:
        raise EquationOfStateTableError(
            f"Packed level metadata has shape {table.shape}, expected first axis 6"
        )
    return table


@lru_cache(maxsize=1)
def load_special_partition_tables() -> SpecialPartitionTables:
    """Return named special partition-function tables."""

    arrays = load_table_arrays(
        atmosphere_table_path("special_partition_tables.npz"),
        _SPECIAL_PARTITION_KEYS,
        error_type=EquationOfStateTableError,
    )
    return SpecialPartitionTables(
        **{
            name: np.asarray(arrays[name], dtype=np.float64)
            for name in _SPECIAL_PARTITION_KEYS
        }
    )


# --- iron-group partition function (merged from partition_functions.py) ---

IRON_GROUP_ATOMIC_NUMBER_RANGE = range(20, 29)
IRON_GROUP_ION_STAGE_RANGE = range(1, 11)

_DEBYE_LOWERING_GRID_CM = (500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 32000.0)
_DEBYE_LOWERING_LOG10_GRID = (
    2.69897,
    3.0,
    3.30103,
    3.60206,
    3.90309,
    4.20412,
    4.50515,
)


def _temperature_bin(log10_temperature: float) -> tuple[int, int, float]:
    """Return lower/upper grid indices and interpolation weight for PFIRON."""

    if log10_temperature > 4.0:
        upper_bin_1based = int((log10_temperature - 4.0) / 0.05) + 31
        upper_bin_1based = min(upper_bin_1based, 56)
        weight = (log10_temperature - (upper_bin_1based - 31) * 0.05 - 4.0) / 0.05
    elif log10_temperature < 3.7:
        upper_bin_1based = int((log10_temperature - 3.32) / 0.02) + 2
        upper_bin_1based = max(upper_bin_1based, 2)
        weight = (log10_temperature - (upper_bin_1based - 2) * 0.02 - 3.32) / 0.02
    else:
        upper_bin_1based = int((log10_temperature - 3.7) / 0.03) + 21
        weight = (log10_temperature - (upper_bin_1based - 21) * 0.03 - 3.7) / 0.03

    upper_index = upper_bin_1based - 1
    lower_index = upper_index - 1
    return lower_index, upper_index, weight


def _interpolate_temperature(
    *,
    lowering_index: int,
    lower_temperature_index: int,
    upper_temperature_index: int,
    temperature_weight: float,
    ion_stage_index: int,
    element_index: int,
) -> float:
    grid = load_iron_group_partition_grid()
    return float(
        temperature_weight
        * grid[lowering_index, upper_temperature_index, ion_stage_index, element_index]
        + (1.0 - temperature_weight)
        * grid[lowering_index, lower_temperature_index, ion_stage_index, element_index]
    )


def iron_group_partition_function(
    *,
    atomic_number: int,
    ion_stage: int,
    log10_temperature: float,
    lowering_energy_cm: float,
) -> float:
    """Evaluate the iron-group partition function.

    Parameters use physical names, but the interpolation is intentionally the
    PFIRON table algorithm: atomic number 20..28, ion stage 1..10, log10(T),
    and Debye lowering energy in cm^-1.
    """

    if atomic_number not in IRON_GROUP_ATOMIC_NUMBER_RANGE:
        raise ValueError(
            f"atomic_number={atomic_number} outside iron-group range 20..28"
        )
    if ion_stage not in IRON_GROUP_ION_STAGE_RANGE:
        raise ValueError(f"ion_stage={ion_stage} outside PFIRON range 1..10")

    lower_temperature_index, upper_temperature_index, temperature_weight = (
        _temperature_bin(float(log10_temperature))
    )
    ion_stage_index = int(ion_stage) - 1
    element_index = int(atomic_number) - 20
    lowering_energy = float(lowering_energy_cm)

    if lowering_energy < _DEBYE_LOWERING_GRID_CM[0]:
        return _interpolate_temperature(
            lowering_index=0,
            lower_temperature_index=lower_temperature_index,
            upper_temperature_index=upper_temperature_index,
            temperature_weight=temperature_weight,
            ion_stage_index=ion_stage_index,
            element_index=element_index,
        )

    for upper_lowering_index in range(1, len(_DEBYE_LOWERING_GRID_CM)):
        if lowering_energy < _DEBYE_LOWERING_GRID_CM[upper_lowering_index]:
            lowering_weight = (
                math.log10(lowering_energy)
                - _DEBYE_LOWERING_LOG10_GRID[upper_lowering_index - 1]
            ) / 0.30103
            upper_value = _interpolate_temperature(
                lowering_index=upper_lowering_index,
                lower_temperature_index=lower_temperature_index,
                upper_temperature_index=upper_temperature_index,
                temperature_weight=temperature_weight,
                ion_stage_index=ion_stage_index,
                element_index=element_index,
            )
            lower_value = _interpolate_temperature(
                lowering_index=upper_lowering_index - 1,
                lower_temperature_index=lower_temperature_index,
                upper_temperature_index=upper_temperature_index,
                temperature_weight=temperature_weight,
                ion_stage_index=ion_stage_index,
                element_index=element_index,
            )
            return float(
                lowering_weight * upper_value + (1.0 - lowering_weight) * lower_value
            )

    return _interpolate_temperature(
        lowering_index=len(_DEBYE_LOWERING_GRID_CM) - 1,
        lower_temperature_index=lower_temperature_index,
        upper_temperature_index=upper_temperature_index,
        temperature_weight=temperature_weight,
        ion_stage_index=ion_stage_index,
        element_index=element_index,
    )
