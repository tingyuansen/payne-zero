# ruff: noqa: E402
"""Line-opacity accumulation kernels.

Two stages: the selected-line accumulation over the packed words from
line_selection.py, and the detailed transition stage (hydrogen
Balmer/Paschen wings included) over line_catalog.py records. The compiled
(numba) kernels here are the sole production path; numba is a hard
requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

from ._numba_cache import configure_numba_cache

# The compiled selected-line and transition kernels are the sole production
# path; numba is a hard requirement.
configure_numba_cache()
try:
    import numba
except ImportError as exc:  # pragma: no cover - numba is a hard requirement
    raise ImportError(
        "numba is required: the compiled line-opacity kernels are the sole "
        "production path for payne_zero_atmosphere."
    ) from exc

_NUMBA_AVAILABLE = True

from .constants import LIGHT_SPEED_NM_PER_S as _LIGHT_SPEED_NM_PER_SECOND
from .line_catalog import LineTransitionCatalog, SelectedLineCatalog
from .hydrogen_line_profile import (
    HydrogenLineProfileEvaluator,
    _exponential_integral_table as _hydrogen_exponential_integral_table,
    compute_hydrogen_molecule_population,
    load_hydrogen_line_profile_tables,
)
from .line_profile_math import (
    build_fast_exponential_tables,
    build_selection_log_lookup,
    build_voigt_profile_basis,
    fast_exponential_lookup,
    load_hydrogen_continuum_selector_table,
)


_RATIO_LOG_STEP = np.log(1.0 + 1.0 / 2_000_000.0)
_CLASSICAL_LINE_STRENGTH_SCALE = 0.026538 / 1.77245 / _LIGHT_SPEED_NM_PER_SECOND
_DAMPING_SCALE = 1.0 / 12.5664 / _LIGHT_SPEED_NM_PER_SECOND


def _line_opacity_chunk_count() -> int:
    """One accumulation chunk per numba thread (= ``NUMBA_NUM_THREADS``, which
    defaults to the CPU count).

    Both opacity accumulations -- selected-line and detailed-transition -- run
    the compiled kernel over contiguous line-chunks and sum the per-chunk
    float32 buffers in chunk order. Each line's deposit is identical to the
    serial kernel (the center-index / continuum-column walks are memoryless),
    so the only difference from a serial pass is the deterministic float32 sum
    regrouping (~ulp per cell, ~4e-6 in-band). The whole driver runs inside the
    compiled ``prange`` (no Python thread pool, GIL-free), so chunking is always
    on for real workloads and there is no toggle.
    """
    try:
        return max(1, int(numba.get_num_threads()))
    except Exception:  # pragma: no cover - numba always available here
        return max(1, (os.cpu_count() or 1))


if _NUMBA_AVAILABLE:
    _njit = numba.njit(cache=True, nogil=True)
    _njit_inline = numba.njit(cache=True, nogil=True, inline="always")

    @_njit_inline
    def _fast_exponential_lookup_compiled(x, integer_step, fractional_step):
        if not (x == x) or x < 0.0 or x >= 1001.0:
            return 0.0
        integer_index = int(x)
        fractional_index = int((x - float(integer_index)) * 1000.0 + 1.5)
        if fractional_index < 1:
            fractional_index = 1
        if fractional_index > 1001:
            fractional_index = 1001
        return float(
            integer_step[integer_index] * fractional_step[fractional_index - 1]
        )

    @_njit_inline
    def _voigt_profile_compiled(offset, damping, gaussian, first, second):
        table_index = int(offset * 200.0 + 1.5)
        if table_index < 1:
            table_index = 1
        if table_index > 2001:
            table_index = 2001
        zero_based_index = table_index - 1

        if damping >= 0.2:
            if damping > 1.4 or damping + offset > 3.2:
                damping_squared = damping * damping
                offset_squared = offset * offset
                denominator = (damping_squared + offset_squared) * 1.4142
                profile = damping * 0.79788 / denominator
                if damping > 100.0:
                    return profile
                damping_fraction = damping_squared / denominator
                offset_fraction = offset_squared / denominator
                correction = (
                    (
                        (damping_fraction - 10.0 * offset_fraction)
                        * damping_fraction
                        * 3.0
                    )
                    + 15.0 * offset_fraction * offset_fraction
                    + 3.0 * offset_squared
                    - damping_squared
                )
                return (correction / (denominator * denominator) + 1.0) * profile

            offset_squared = offset * offset
            adjusted_first = (
                first[zero_based_index] + gaussian[zero_based_index] * 1.12838
            )
            adjusted_second = (
                second[zero_based_index]
                + adjusted_first * 1.12838
                - gaussian[zero_based_index]
            )
            third = (
                (1.0 - second[zero_based_index]) * 0.37613
                - adjusted_first * 0.66667 * offset_squared
                + adjusted_second * 1.12838
            )
            fourth = (3.0 * third - adjusted_first) * 0.37613 + gaussian[
                zero_based_index
            ] * 0.66667 * offset_squared * offset_squared
            profile = (
                ((fourth * damping + third) * damping + adjusted_second) * damping
                + adjusted_first
            ) * damping + gaussian[zero_based_index]
            scale = (
                (-0.122727278 * damping + 0.532770573) * damping - 0.96284325
            ) * damping
            return profile * (scale + 0.979895032)

        if offset > 10.0:
            return 0.5642 * damping / (offset * offset)
        return (
            second[zero_based_index] * damping + first[zero_based_index]
        ) * damping + gaussian[zero_based_index]

    @_njit
    def _accumulate_selected_line_wings_compiled(
        line_mass_absorption_coefficient,
        depth_index,
        center_index,
        vacuum_wavelength_nm,
        line_center_absorption,
        damping_parameter,
        doppler_wavelength_width,
        continuum_threshold,
        wavelength_grid,
        gaussian,
        first,
        second,
    ):
        if doppler_wavelength_width <= 0.0:
            return

        wavelength_count = wavelength_grid.shape[0]
        red_limit = min(center_index + 101, wavelength_count)
        if damping_parameter <= 0.2:
            for wavelength_index in range(center_index, red_limit):
                offset = float(wavelength_grid[wavelength_index] - vacuum_wavelength_nm)
                voigt_offset = offset / doppler_wavelength_width
                if voigt_offset > 10.0:
                    contribution = (
                        line_center_absorption * 0.5642 * damping_parameter
                    ) / (voigt_offset * voigt_offset)
                else:
                    contribution = line_center_absorption * _voigt_profile_compiled(
                        voigt_offset,
                        damping_parameter,
                        gaussian,
                        first,
                        second,
                    )
                line_mass_absorption_coefficient[depth_index, wavelength_index] += (
                    np.float32(contribution)
                )
                if contribution < continuum_threshold:
                    break

            for red_step in range(1, 101):
                wavelength_index = center_index - red_step
                if wavelength_index < 0:
                    break
                offset = float(vacuum_wavelength_nm - wavelength_grid[wavelength_index])
                voigt_offset = offset / doppler_wavelength_width
                if voigt_offset > 10.0:
                    contribution = (
                        line_center_absorption * 0.5642 * damping_parameter
                    ) / (voigt_offset * voigt_offset)
                else:
                    contribution = line_center_absorption * _voigt_profile_compiled(
                        voigt_offset,
                        damping_parameter,
                        gaussian,
                        first,
                        second,
                    )
                line_mass_absorption_coefficient[depth_index, wavelength_index] += (
                    np.float32(contribution)
                )
                if contribution < continuum_threshold:
                    break
            return

        for wavelength_index in range(center_index, red_limit):
            voigt_offset = float(
                wavelength_grid[wavelength_index] - vacuum_wavelength_nm
            )
            voigt_offset = voigt_offset / doppler_wavelength_width
            contribution = line_center_absorption * _voigt_profile_compiled(
                voigt_offset, damping_parameter, gaussian, first, second
            )
            line_mass_absorption_coefficient[depth_index, wavelength_index] += (
                np.float32(contribution)
            )
            if contribution < continuum_threshold:
                break

        for red_step in range(1, 101):
            wavelength_index = center_index - red_step
            if wavelength_index < 0:
                break
            voigt_offset = float(
                vacuum_wavelength_nm - wavelength_grid[wavelength_index]
            )
            voigt_offset = voigt_offset / doppler_wavelength_width
            contribution = line_center_absorption * _voigt_profile_compiled(
                voigt_offset, damping_parameter, gaussian, first, second
            )
            line_mass_absorption_coefficient[depth_index, wavelength_index] += (
                np.float32(contribution)
            )
            if contribution < continuum_threshold:
                break

    @_njit
    def _accumulate_selected_line_opacity_compiled(
        line_count,
        packed_wavelength_index,
        vacuum_wavelength_nm_by_line,
        packed_species_slot,
        lower_excitation_index,
        log_strength_index,
        radiative_damping_index,
        stark_damping_index,
        van_der_waals_damping_index,
        wavelength_grid,
        bin_edges,
        continuum_threshold,
        hc_over_kt_cm,
        electrons,
        line_population_widths,
        doppler,
        neutral_collision_density,
        selection_lookup,
        integer_step,
        fractional_step,
        gaussian,
        first,
        second,
        wavelength_start_index,
        stop_index,
    ):
        layer_count = hc_over_kt_cm.shape[0]
        wavelength_count = wavelength_grid.shape[0]
        line_mass_absorption_coefficient = np.zeros(
            (layer_count, wavelength_count), dtype=np.float32
        )
        depth_gate = np.zeros(layer_count + 2, dtype=np.int32)
        continuum_column = 0
        center_index = max(0, int(wavelength_start_index) - 1)
        previous_packed_wavelength = 0
        wavelength_start = float(
            wavelength_grid[max(0, int(wavelength_start_index) - 1)] - 1.0
        )
        wavelength_stop = float(
            wavelength_grid[min(stop_index, wavelength_count) - 1] + 1.0
        )
        contributing_lines = 0

        for line_index in range(line_count):
            packed_wavelength = int(packed_wavelength_index[line_index])
            if packed_wavelength < previous_packed_wavelength:
                continuum_column = 0
                center_index = max(0, int(wavelength_start_index) - 1)

            while continuum_column < bin_edges.shape[0] and packed_wavelength >= int(
                bin_edges[continuum_column]
            ):
                continuum_column += 1
            if continuum_column >= continuum_threshold.shape[1]:
                previous_packed_wavelength = packed_wavelength
                continue

            species_slot = abs(int(packed_species_slot[line_index])) // 10
            if species_slot < 1 or species_slot > line_population_widths.shape[1]:
                previous_packed_wavelength = packed_wavelength
                continue

            vacuum_wavelength_nm = vacuum_wavelength_nm_by_line[line_index]
            if (
                vacuum_wavelength_nm < wavelength_start
                or vacuum_wavelength_nm > wavelength_stop
            ):
                previous_packed_wavelength = packed_wavelength
                continue

            while (
                center_index < wavelength_count
                and vacuum_wavelength_nm >= wavelength_grid[center_index]
            ):
                center_index += 1
            if center_index >= wavelength_count:
                previous_packed_wavelength = packed_wavelength
                continue

            log_strength = int(log_strength_index[line_index])
            lower_excitation = int(lower_excitation_index[line_index])
            radiative_index = int(radiative_damping_index[line_index])
            stark_index = int(stark_damping_index[line_index])
            van_der_waals_index = int(van_der_waals_damping_index[line_index])
            if (
                log_strength < 1
                or lower_excitation < 1
                or radiative_index < 1
                or stark_index < 1
                or van_der_waals_index < 1
            ):
                previous_packed_wavelength = packed_wavelength
                continue
            if (
                log_strength > selection_lookup.shape[0]
                or lower_excitation > selection_lookup.shape[0]
                or radiative_index > selection_lookup.shape[0]
                or stark_index > selection_lookup.shape[0]
                or van_der_waals_index > selection_lookup.shape[0]
            ):
                previous_packed_wavelength = packed_wavelength
                continue

            wavelength_f32 = np.float32(vacuum_wavelength_nm)
            classical_strength = np.float32(
                np.float32(_CLASSICAL_LINE_STRENGTH_SCALE)
                * wavelength_f32
                * selection_lookup[log_strength - 1]
            )
            excitation = float(selection_lookup[lower_excitation - 1])
            radiative_damping = np.float32(
                selection_lookup[radiative_index - 1]
                * wavelength_f32
                * np.float32(_DAMPING_SCALE)
            )
            stark_damping = np.float32(
                selection_lookup[stark_index - 1]
                * wavelength_f32
                * np.float32(_DAMPING_SCALE)
            )
            van_der_waals_damping = np.float32(
                selection_lookup[van_der_waals_index - 1]
                * wavelength_f32
                * np.float32(_DAMPING_SCALE)
            )

            line_touched_depth_gate = False
            for depth_1based in range(8, layer_count + 1, 8):
                gate_index = depth_1based + 1
                depth_gate[gate_index] = 0
                depth_index = depth_1based - 1
                center_absorption = np.float32(
                    classical_strength
                    * line_population_widths[depth_index, species_slot - 1]
                )
                if (
                    center_absorption
                    < continuum_threshold[depth_index, continuum_column]
                ):
                    continue
                center_absorption = np.float32(
                    center_absorption
                    * _fast_exponential_lookup_compiled(
                        excitation * float(hc_over_kt_cm[depth_index]),
                        integer_step,
                        fractional_step,
                    )
                )
                if (
                    center_absorption
                    < continuum_threshold[depth_index, continuum_column]
                ):
                    continue
                depth_gate[gate_index] = 1
                line_touched_depth_gate = True
                doppler_width = np.float64(doppler[depth_index, species_slot - 1])
                if doppler_width <= 0.0:
                    continue
                damping_parameter = np.float64(
                    np.float32(
                        np.float32(
                            radiative_damping
                            + stark_damping * electrons[depth_index]
                            + van_der_waals_damping
                            * neutral_collision_density[depth_index]
                        )
                        / np.float32(doppler_width)
                    )
                )
                _accumulate_selected_line_wings_compiled(
                    line_mass_absorption_coefficient,
                    depth_index,
                    center_index,
                    vacuum_wavelength_nm,
                    np.float64(center_absorption),
                    damping_parameter,
                    doppler_width * vacuum_wavelength_nm,
                    np.float64(continuum_threshold[depth_index, continuum_column]),
                    wavelength_grid,
                    gaussian,
                    first,
                    second,
                )

            for block_end_1based in range(8, layer_count + 1, 8):
                if (
                    depth_gate[block_end_1based - 7] + depth_gate[block_end_1based + 1]
                    == 0
                ):
                    continue
                for depth_1based in range(block_end_1based - 7, block_end_1based):
                    depth_index = depth_1based - 1
                    center_absorption = np.float32(
                        classical_strength
                        * line_population_widths[depth_index, species_slot - 1]
                    )
                    if (
                        center_absorption
                        < continuum_threshold[depth_index, continuum_column]
                    ):
                        continue
                    center_absorption = np.float32(
                        center_absorption
                        * _fast_exponential_lookup_compiled(
                            excitation * float(hc_over_kt_cm[depth_index]),
                            integer_step,
                            fractional_step,
                        )
                    )
                    if (
                        center_absorption
                        < continuum_threshold[depth_index, continuum_column]
                    ):
                        continue
                    doppler_width = np.float64(doppler[depth_index, species_slot - 1])
                    if doppler_width <= 0.0:
                        continue
                    damping_parameter = np.float64(
                        np.float32(
                            np.float32(
                                radiative_damping
                                + stark_damping * electrons[depth_index]
                                + van_der_waals_damping
                                * neutral_collision_density[depth_index]
                            )
                            / np.float32(doppler_width)
                        )
                    )
                    _accumulate_selected_line_wings_compiled(
                        line_mass_absorption_coefficient,
                        depth_index,
                        center_index,
                        vacuum_wavelength_nm,
                        np.float64(center_absorption),
                        damping_parameter,
                        doppler_width * vacuum_wavelength_nm,
                        np.float64(continuum_threshold[depth_index, continuum_column]),
                        wavelength_grid,
                        gaussian,
                        first,
                        second,
                    )

            if line_touched_depth_gate:
                contributing_lines += 1
            previous_packed_wavelength = packed_wavelength

        return line_mass_absorption_coefficient, contributing_lines

    @_njit_inline
    def _voigt_profile_f64_compiled(
        frequency_offset, damping_parameter, gaussian_table, first_table, second_table
    ):
        """Exact transcription of ``evaluate_voigt_profile`` (float64 path)."""
        table_index = int(frequency_offset * 200.0 + 1.5)
        if table_index < 1:
            table_index = 1
        if table_index > 2001:
            table_index = 2001
        zero_based_index = table_index - 1

        gaussian = gaussian_table[zero_based_index]
        first = first_table[zero_based_index]
        second = second_table[zero_based_index]
        damping = damping_parameter
        offset = frequency_offset

        if damping >= 0.2:
            if damping > 1.4 or damping + offset > 3.2:
                damping_squared = damping * damping
                offset_squared = offset * offset
                denominator = (damping_squared + offset_squared) * 1.4142
                profile = damping * 0.79788 / denominator
                if damping > 100.0:
                    return profile
                damping_fraction = damping_squared / denominator
                offset_fraction = offset_squared / denominator
                denominator_squared = denominator * denominator
                correction = (
                    (
                        (damping_fraction - 10.0 * offset_fraction)
                        * damping_fraction
                        * 3.0
                    )
                    + 15.0 * offset_fraction * offset_fraction
                    + 3.0 * offset_squared
                    - damping_squared
                )
                return (correction / denominator_squared + 1.0) * profile

            offset_squared = offset * offset
            adjusted_first = first + gaussian * 1.12838
            adjusted_second = second + adjusted_first * 1.12838 - gaussian
            third = (
                (1.0 - second) * 0.37613
                - adjusted_first * 0.66667 * offset_squared
                + adjusted_second * 1.12838
            )
            fourth = (3.0 * third - adjusted_first) * 0.37613 + gaussian * 0.66667 * (
                offset_squared * offset_squared
            )
            profile = (
                ((fourth * damping + third) * damping + adjusted_second) * damping
                + adjusted_first
            ) * damping + gaussian
            scale = (
                (-0.122727278 * damping + 0.532770573) * damping - 0.96284325
            ) * damping
            return profile * (scale + 0.979895032)

        if offset > 10.0:
            return 0.5642 * damping / (offset * offset)
        return (second * damping + first) * damping + gaussian

    @_njit
    def _transition_wings_compiled(
        line_mass_absorption_coefficient,
        layer_index,
        center_index,
        vacuum_wavelength_nm,
        line_center_absorption,
        damping_parameter,
        doppler_wavelength_width,
        continuum_threshold,
        wavelength_grid,
        wing_steps,
        has_blue_cutoff,
        blue_cutoff_nm,
        gaussian_table,
        first_table,
        second_table,
    ):
        """Accumulate a line's Voigt wings into the absorption grid (float64)."""
        if doppler_wavelength_width <= 0.0:
            return

        wavelength_count = wavelength_grid.shape[0]
        red_limit = min(center_index + wing_steps + 1, wavelength_count)

        for wavelength_index in range(center_index, red_limit):
            offset = wavelength_grid[wavelength_index] - vacuum_wavelength_nm
            voigt_offset = offset / doppler_wavelength_width
            if damping_parameter <= 0.2 and voigt_offset > 10.0:
                contribution = (
                    line_center_absorption
                    * 0.5642
                    * damping_parameter
                    / (voigt_offset * voigt_offset)
                )
            else:
                contribution = line_center_absorption * _voigt_profile_f64_compiled(
                    voigt_offset,
                    damping_parameter,
                    gaussian_table,
                    first_table,
                    second_table,
                )
            line_mass_absorption_coefficient[layer_index, wavelength_index] += (
                np.float32(contribution)
            )
            if contribution < continuum_threshold:
                break

        for red_step in range(1, wing_steps + 1):
            wavelength_index = center_index - red_step
            if wavelength_index < 0:
                break
            if has_blue_cutoff and wavelength_grid[wavelength_index] < blue_cutoff_nm:
                break
            offset = vacuum_wavelength_nm - wavelength_grid[wavelength_index]
            voigt_offset = offset / doppler_wavelength_width
            if damping_parameter <= 0.2 and voigt_offset > 10.0:
                contribution = (
                    line_center_absorption
                    * 0.5642
                    * damping_parameter
                    / (voigt_offset * voigt_offset)
                )
            else:
                contribution = line_center_absorption * _voigt_profile_f64_compiled(
                    voigt_offset,
                    damping_parameter,
                    gaussian_table,
                    first_table,
                    second_table,
                )
            line_mass_absorption_coefficient[layer_index, wavelength_index] += (
                np.float32(contribution)
            )
            if contribution < continuum_threshold:
                break

    @_njit
    def _accumulate_transition_run_compiled(
        start_index,
        stop_line_index,
        line_type,
        packed_wavelength_index,
        vacuum_wavelength_nm_by_line,
        packed_species_slot,
        oscillator_strength,
        lower_excitation_cm,
        radiative_damping_by_line,
        stark_damping_by_line,
        van_der_waals_damping_by_line,
        selector_index_by_line,
        selector_species_slot_by_line,
        wavelength_grid,
        bin_edges,
        continuum_threshold,
        hc_over_kt_cm,
        electrons,
        line_population_widths,
        doppler,
        neutral_collision_density,
        continuum_selector,
        hydrogen_level_dissolution_wavenumber_cm,
        integer_step,
        fractional_step,
        gaussian_table,
        first_table,
        second_table,
        depth_gate,
        line_mass_absorption_coefficient,
        continuum_column,
        center_index,
        wavelength_start_index,
        stop_index,
    ):
        """Run the normal-transition (type 0/2/3) per-line loop over a slice.

        Mirrors the Python loop body exactly; returns updated walk state, the
        number of processed lines, and whether the outer loop must stop.
        """
        layer_count = hc_over_kt_cm.shape[0]
        wavelength_count = wavelength_grid.shape[0]
        last_grid_value = wavelength_grid[min(stop_index, wavelength_count) - 1]
        processed = 0
        stop_flag = False
        stopped_at = stop_line_index

        for line_index in range(start_index, stop_line_index):
            if line_type[line_index] == 2:
                continue

            vacuum_wavelength_nm = vacuum_wavelength_nm_by_line[line_index]
            if vacuum_wavelength_nm > last_grid_value:
                stop_flag = True
                stopped_at = line_index
                break

            packed_wavelength = packed_wavelength_index[line_index]
            while (
                continuum_column < bin_edges.shape[0]
                and packed_wavelength >= bin_edges[continuum_column]
            ):
                continuum_column += 1
            if continuum_column >= continuum_threshold.shape[1]:
                continue

            while (
                center_index < wavelength_count
                and vacuum_wavelength_nm >= wavelength_grid[center_index]
            ):
                center_index += 1
            if center_index >= wavelength_count:
                stop_flag = True
                stopped_at = line_index
                break

            species_slot = packed_species_slot[line_index]
            if species_slot < 1 or species_slot > line_population_widths.shape[1]:
                processed += 1
                continue

            line_oscillator_strength = oscillator_strength[line_index]
            line_lower_excitation_cm = lower_excitation_cm[line_index]
            radiative_damping = radiative_damping_by_line[line_index]
            stark_damping = stark_damping_by_line[line_index]
            van_der_waals_damping = van_der_waals_damping_by_line[line_index]
            selector_index = selector_index_by_line[line_index]
            selector_species_slot = selector_species_slot_by_line[line_index]
            normal_selector_index = selector_index
            if normal_selector_index > 10:
                normal_selector_index = 0

            for depth_1based in range(8, layer_count + 1, 8):
                gate_index = depth_1based + 1
                depth_gate[gate_index] = 0
                depth_index = depth_1based - 1
                center_absorption = (
                    line_oscillator_strength
                    * line_population_widths[depth_index, species_slot - 1]
                )
                if (
                    center_absorption
                    < continuum_threshold[depth_index, continuum_column]
                ):
                    continue
                center_absorption *= _fast_exponential_lookup_compiled(
                    line_lower_excitation_cm * hc_over_kt_cm[depth_index],
                    integer_step,
                    fractional_step,
                )
                if (
                    center_absorption
                    < continuum_threshold[depth_index, continuum_column]
                ):
                    continue
                doppler_width = doppler[depth_index, species_slot - 1]
                if doppler_width <= 0.0:
                    continue
                depth_gate[gate_index] = 1
                damping_parameter = (
                    radiative_damping
                    + stark_damping * electrons[depth_index]
                    + van_der_waals_damping * neutral_collision_density[depth_index]
                ) / doppler_width
                has_blue_cutoff = False
                blue_cutoff_nm = 0.0
                if (
                    normal_selector_index > 0
                    and 1 <= selector_species_slot <= continuum_selector.shape[1]
                ):
                    denominator = (
                        continuum_selector[
                            normal_selector_index - 1, selector_species_slot - 1
                        ]
                        - hydrogen_level_dissolution_wavenumber_cm[depth_index]
                    )
                    if denominator != 0.0:
                        has_blue_cutoff = True
                        blue_cutoff_nm = 1.0e7 / denominator
                if has_blue_cutoff and vacuum_wavelength_nm < blue_cutoff_nm:
                    continue
                _transition_wings_compiled(
                    line_mass_absorption_coefficient,
                    depth_index,
                    center_index,
                    vacuum_wavelength_nm,
                    center_absorption,
                    damping_parameter,
                    doppler_width * vacuum_wavelength_nm,
                    continuum_threshold[depth_index, continuum_column],
                    wavelength_grid,
                    2000,
                    has_blue_cutoff,
                    blue_cutoff_nm,
                    gaussian_table,
                    first_table,
                    second_table,
                )

            for block_end_1based in range(8, layer_count + 1, 8):
                if (
                    depth_gate[block_end_1based - 7] + depth_gate[block_end_1based + 1]
                    == 0
                ):
                    continue
                for depth_1based in range(block_end_1based - 7, block_end_1based):
                    depth_index = depth_1based - 1
                    center_absorption = (
                        line_oscillator_strength
                        * line_population_widths[depth_index, species_slot - 1]
                    )
                    if (
                        center_absorption
                        < continuum_threshold[depth_index, continuum_column]
                    ):
                        continue
                    center_absorption *= _fast_exponential_lookup_compiled(
                        line_lower_excitation_cm * hc_over_kt_cm[depth_index],
                        integer_step,
                        fractional_step,
                    )
                    if (
                        center_absorption
                        < continuum_threshold[depth_index, continuum_column]
                    ):
                        continue
                    doppler_width = doppler[depth_index, species_slot - 1]
                    if doppler_width <= 0.0:
                        continue
                    damping_parameter = (
                        radiative_damping
                        + stark_damping * electrons[depth_index]
                        + van_der_waals_damping * neutral_collision_density[depth_index]
                    ) / doppler_width
                    has_blue_cutoff = False
                    blue_cutoff_nm = 0.0
                    if (
                        normal_selector_index > 0
                        and 1 <= selector_species_slot <= continuum_selector.shape[1]
                    ):
                        denominator = (
                            continuum_selector[
                                normal_selector_index - 1,
                                selector_species_slot - 1,
                            ]
                            - hydrogen_level_dissolution_wavenumber_cm[depth_index]
                        )
                        if denominator != 0.0:
                            has_blue_cutoff = True
                            blue_cutoff_nm = 1.0e7 / denominator
                    if has_blue_cutoff and vacuum_wavelength_nm < blue_cutoff_nm:
                        continue
                    _transition_wings_compiled(
                        line_mass_absorption_coefficient,
                        depth_index,
                        center_index,
                        vacuum_wavelength_nm,
                        center_absorption,
                        damping_parameter,
                        doppler_width * vacuum_wavelength_nm,
                        continuum_threshold[depth_index, continuum_column],
                        wavelength_grid,
                        2000,
                        has_blue_cutoff,
                        blue_cutoff_nm,
                        gaussian_table,
                        first_table,
                        second_table,
                    )

            processed += 1

        return continuum_column, center_index, processed, stop_flag, stopped_at

    @_njit_inline
    def _hydrogen_fast_exponential_compiled(x, integer_table, fractional_table):
        """Exact transcription of hydrogen_line_profile._fast_exponential."""
        if x < 0.0 or x >= 1001.0:
            return 0.0
        integer_index = int(x)
        fractional_index = int((x - float(integer_index)) * 1000.0 + 1.5)
        if fractional_index < 1:
            fractional_index = 1
        if fractional_index > 1001:
            fractional_index = 1001
        return integer_table[integer_index] * fractional_table[fractional_index - 1]

    @_njit_inline
    def _hydrogen_fast_exponential_integral_compiled(x, integral_table):
        """Exact transcription of _fast_exponential_integral (table passed in)."""
        if x > 20.0:
            return 0.0
        if x >= 0.5:
            table_index = int(x * 100.0 + 0.5)
            if table_index < 1:
                table_index = 1
            if table_index > 2000:
                table_index = 2000
            return integral_table[table_index - 1]
        if x <= 0.0:
            return 0.0
        return (1.0 - 0.22464 * x) * x - np.log(x) - 0.57721

    @_njit
    def _hydrogen_stark_probability_compiled(
        beta,
        pressure,
        lower_level,
        upper_level,
        stark_probability_table,
        stark_pressure_grid,
        stark_beta_grid,
        stark_wing_correction_c,
        stark_wing_correction_d,
    ):
        """Exact transcription of hydrogen_line_profile._stark_probability."""
        correction = 1.0
        beta_squared = beta * beta
        sqrt_beta = np.sqrt(max(beta, 1.0e-300))
        if beta <= 500.0:
            table_index = 7
            level_delta = upper_level - lower_level
            if lower_level <= 3 and level_delta <= 2:
                table_index = 2 * (lower_level - 1) + level_delta
            pressure_low_index = min(int(5.0 * pressure) + 1, 4)
            if pressure_low_index < 1:
                pressure_low_index = 1
            pressure_high_index = pressure_low_index + 1
            high_pressure_weight = 5.0 * (
                pressure - stark_pressure_grid[pressure_low_index - 1]
            )
            low_pressure_weight = 1.0 - high_pressure_weight
            if beta <= 25.12:
                beta_high_index = int(
                    np.searchsorted(stark_beta_grid, beta, side="left")
                )
                if beta_high_index < 1:
                    beta_high_index = 1
                if beta_high_index > 14:
                    beta_high_index = 14
                beta_low_index = beta_high_index - 1
                beta_denominator = (
                    stark_beta_grid[beta_high_index] - stark_beta_grid[beta_low_index]
                )
                if beta_denominator == 0.0:
                    high_beta_weight = 0.0
                else:
                    high_beta_weight = (
                        beta - stark_beta_grid[beta_low_index]
                    ) / beta_denominator
                low_beta_weight = 1.0 - high_beta_weight
                high_beta_correction = (
                    stark_probability_table[
                        pressure_high_index - 1, beta_high_index, table_index - 1
                    ]
                    * high_pressure_weight
                    + stark_probability_table[
                        pressure_low_index - 1, beta_high_index, table_index - 1
                    ]
                    * low_pressure_weight
                )
                low_beta_correction = (
                    stark_probability_table[
                        pressure_high_index - 1, beta_low_index, table_index - 1
                    ]
                    * high_pressure_weight
                    + stark_probability_table[
                        pressure_low_index - 1, beta_low_index, table_index - 1
                    ]
                    * low_pressure_weight
                )
                correction = (
                    1.0
                    + high_beta_correction * high_beta_weight
                    + low_beta_correction * low_beta_weight
                )
                low_beta_profile = 0.0
                high_beta_profile = 0.0
                blend = max(min(0.5 * (10.0 - beta), 1.0), 0.0)
                if beta <= 10.0:
                    low_beta_profile = 8.0 / (83.0 + (2.0 + 0.95 * beta_squared) * beta)
                if beta >= 8.0:
                    high_beta_profile = (
                        1.5 / sqrt_beta + 27.0 / beta_squared
                    ) / beta_squared
                return (
                    low_beta_profile * blend + high_beta_profile * (1.0 - blend)
                ) * correction
            c_value = (
                stark_wing_correction_c[pressure_high_index - 1, table_index - 1]
                * high_pressure_weight
                + stark_wing_correction_c[pressure_low_index - 1, table_index - 1]
                * low_pressure_weight
            )
            d_value = (
                stark_wing_correction_d[pressure_high_index - 1, table_index - 1]
                * high_pressure_weight
                + stark_wing_correction_d[pressure_low_index - 1, table_index - 1]
                * low_pressure_weight
            )
            correction = 1.0 + d_value / (c_value + beta * sqrt_beta)
        return (1.5 / sqrt_beta + 27.0 / beta_squared) / beta_squared * correction

    @_njit
    def _hydrogen_profile_compiled(
        layer,
        wavelength_offset_nm,
        # line setup scalars
        lower_level,
        upper_level,
        line_frequency_hz,
        line_wavelength_a,
        beta_scale,
        stark_c1_factor,
        stark_c2_factor,
        setup_radiative_width,
        setup_resonance_width,
        setup_van_der_waals_width,
        setup_stark_width,
        low_density_impact_numerator,
        impact_electron_density_threshold_cm3,
        stark_component_offsets_hz,
        stark_component_weights,
        # per-layer evaluator arrays
        hydrogen_fractional_doppler_width,
        field_strength_arr,
        temperature_density_he,
        temperature_density_h2,
        hydrogen_neutral_population,
        hydrogen_neutral_partition_normalized_population,
        hydrogen_ionized_population,
        electron_density_arr,
        low_density_impact_factor,
        high_density_impact_factor,
        stark_linear_density_coefficient,
        stark_quadratic_density_coefficient,
        stark_gamma_thermal_correction,
        stark_gamma_density_correction,
        pressure_parameter,
        # tables
        h2_quasimolecular_cutoff_table,
        h2plus_quasimolecular_cutoff_table,
        stark_probability_table,
        stark_pressure_grid,
        stark_beta_grid,
        stark_wing_correction_c,
        stark_wing_correction_d,
        exponential_integer,
        exponential_fraction,
        exponential_integral_table,
    ):
        """Exact transcription of HydrogenLineProfileEvaluator.profile_for_setup."""
        LIGHT_A = 2.99792458e18
        LIGHT_CM = 2.99792458e10
        SQRT_PI = 1.77245
        PI = 3.14159

        wavelength_a = line_wavelength_a + wavelength_offset_nm * 10.0
        if wavelength_a <= 0.0:
            return 0.0
        frequency = LIGHT_A / wavelength_a
        frequency_offset = abs(frequency - line_frequency_hz)
        doppler_width = hydrogen_fractional_doppler_width[layer]
        if doppler_width <= 0.0:
            return 0.0
        stark_width = setup_stark_width * field_strength_arr[layer]
        van_der_waals_width = (
            setup_van_der_waals_width * temperature_density_he[layer]
            + 2.0 * setup_van_der_waals_width * temperature_density_h2[layer]
        )
        radiative_width = setup_radiative_width
        resonance_width = setup_resonance_width * hydrogen_neutral_population[layer]
        lorentz_width = resonance_width + van_der_waals_width + radiative_width
        profile_mode = 1
        if not (doppler_width >= stark_width and doppler_width >= lorentz_width):
            profile_mode = 2
            if lorentz_width < stark_width:
                profile_mode = 3
        half_width = line_frequency_hz * max(doppler_width, lorentz_width, stark_width)
        in_core = abs(frequency_offset) <= half_width
        doppler_frequency_width = line_frequency_hz * doppler_width
        if doppler_frequency_width <= 0.0:
            return 0.0

        stark_wavelength_offset = (
            -10.0 * wavelength_offset_nm / line_wavelength_a * line_frequency_hz
        )

        # _doppler_profile
        doppler_value = 0.0
        for stark_component_index in range(stark_component_offsets_hz.shape[0]):
            distance = (
                abs(
                    frequency
                    - line_frequency_hz
                    - stark_component_offsets_hz[stark_component_index]
                )
                / doppler_frequency_width
            )
            if distance <= 7.0:
                doppler_value += (
                    _hydrogen_fast_exponential_compiled(
                        distance * distance, exponential_integer, exponential_fraction
                    )
                    * stark_component_weights[stark_component_index]
                )

        # _lorentz_profile
        lorentz_value = 0.0
        lorentz_resonance = resonance_width
        if lower_level == 1 and upper_level == 2:
            lorentz_resonance = resonance_width * 4.0
            total_width = lorentz_resonance + van_der_waals_width + radiative_width
            lyman_half_width = line_frequency_hz * total_width
            if frequency > (82259.105 - 4000.0) * LIGHT_CM:
                resonance_profile = (
                    lorentz_resonance
                    * line_frequency_hz
                    / PI
                    / (
                        frequency_offset * frequency_offset
                        + lyman_half_width * lyman_half_width
                    )
                    * SQRT_PI
                    * doppler_frequency_width
                )
            else:
                cutoff = 0.0
                if frequency >= 50000.0 * LIGHT_CM:
                    spacing = 200.0 * LIGHT_CM
                    frequency_22000 = (82259.105 - 22000.0) * LIGHT_CM
                    if frequency < frequency_22000:
                        cutoff = (
                            h2_quasimolecular_cutoff_table[1]
                            - h2_quasimolecular_cutoff_table[0]
                        ) / spacing * (
                            frequency - frequency_22000
                        ) + h2_quasimolecular_cutoff_table[0]
                    else:
                        cutoff_index = int((frequency - frequency_22000) / spacing)
                        cutoff_index = max(
                            0,
                            min(
                                cutoff_index,
                                h2_quasimolecular_cutoff_table.shape[0] - 2,
                            ),
                        )
                        cutoff_frequency = cutoff_index * spacing + frequency_22000
                        cutoff = (
                            h2_quasimolecular_cutoff_table[cutoff_index + 1]
                            - h2_quasimolecular_cutoff_table[cutoff_index]
                        ) / spacing * (
                            frequency - cutoff_frequency
                        ) + h2_quasimolecular_cutoff_table[cutoff_index]
                    cutoff = (
                        10.0 ** (cutoff - 14.0)
                        * hydrogen_neutral_partition_normalized_population[layer]
                        * 2.0
                        / LIGHT_CM
                    )
                resonance_profile = cutoff * SQRT_PI * doppler_frequency_width
            radiative_profile = (
                radiative_width
                * line_frequency_hz
                / PI
                / (
                    frequency_offset * frequency_offset
                    + lyman_half_width * lyman_half_width
                )
                * SQRT_PI
                * doppler_frequency_width
            )
            if frequency <= 2.463e15:
                radiative_profile = 0.0
            van_der_waals_profile = (
                van_der_waals_width
                * line_frequency_hz
                / PI
                / (
                    frequency_offset * frequency_offset
                    + lyman_half_width * lyman_half_width
                )
                * SQRT_PI
                * doppler_frequency_width
            )
            if frequency < 1.8e15:
                van_der_waals_profile = 0.0
            lorentz_value = (
                resonance_profile + radiative_profile + van_der_waals_profile
            )
        else:
            lorentz_half_width = line_frequency_hz * (
                resonance_width + van_der_waals_width + radiative_width
            )
            if lorentz_half_width <= 0.0:
                lorentz_value = 0.0
            else:
                lorentz_value = (
                    lorentz_half_width
                    / PI
                    / (
                        frequency_offset * frequency_offset
                        + lorentz_half_width * lorentz_half_width
                    )
                    * SQRT_PI
                    * doppler_frequency_width
                )

        # _stark_profile
        stark_value = 0.0
        layer_field_strength = field_strength_arr[layer]
        if layer_field_strength > 0.0:
            low_density_impact_weight = 1.0 / (
                1.0
                + electron_density_arr[layer] / impact_electron_density_threshold_cm3
            )
            impact_broadening_factor = (
                low_density_impact_numerator
                * low_density_impact_factor[layer]
                * low_density_impact_weight
                + high_density_impact_factor[layer] * (1.0 - low_density_impact_weight)
            )
            linear_impact_parameter = (
                stark_linear_density_coefficient[layer]
                * stark_c1_factor
                * impact_broadening_factor
            )
            quadratic_impact_parameter = (
                stark_quadratic_density_coefficient[layer] * stark_c2_factor
            )
            if linear_impact_parameter <= 0.0:
                linear_impact_parameter = 0.0
            if quadratic_impact_parameter <= 0.0:
                quadratic_impact_parameter = 0.0
            impact_width_scale = 6.77 * np.sqrt(max(linear_impact_parameter, 0.0))
            log_term = 0.0
            if linear_impact_parameter > 0.0 and quadratic_impact_parameter > 0.0:
                log_term = np.log(
                    np.sqrt(quadratic_impact_parameter) / linear_impact_parameter
                )
            zero_offset_impact_width = (
                impact_width_scale
                * max(0.0, 0.2114 + log_term)
                * (
                    1.0
                    - stark_gamma_thermal_correction[layer]
                    - stark_gamma_density_correction[layer]
                )
            )
            beta = abs(stark_wavelength_offset) / layer_field_strength * beta_scale
            linear_impact_argument = linear_impact_parameter * beta
            quadratic_impact_argument = quadratic_impact_parameter * beta * beta
            impact_width = zero_offset_impact_width
            if not (
                quadratic_impact_argument <= 1.0e-4 and linear_impact_argument <= 1.0e-5
            ):
                impact_width = (
                    impact_width_scale
                    * (
                        0.5
                        * _hydrogen_fast_exponential_compiled(
                            min(80.0, linear_impact_argument),
                            exponential_integer,
                            exponential_fraction,
                        )
                        + _hydrogen_fast_exponential_integral_compiled(
                            linear_impact_argument, exponential_integral_table
                        )
                        - 0.5
                        * _hydrogen_fast_exponential_integral_compiled(
                            quadratic_impact_argument, exponential_integral_table
                        )
                    )
                    * (
                        1.0
                        - stark_gamma_thermal_correction[layer]
                        / (1.0 + (90.0 * linear_impact_argument) ** 3.0)
                        - stark_gamma_density_correction[layer]
                        / (1.0 + 2000.0 * linear_impact_argument)
                    )
                )
                if impact_width <= 1.0e-20:
                    impact_width = 0.0

            probability = _hydrogen_stark_probability_compiled(
                beta,
                pressure_parameter[layer],
                lower_level,
                upper_level,
                stark_probability_table,
                stark_pressure_grid,
                stark_beta_grid,
                stark_wing_correction_c,
                stark_wing_correction_d,
            )
            if upper_level <= 2:
                probability *= 0.5
                if frequency >= (82259.105 - 20000.0) * LIGHT_CM:
                    if frequency <= (82259.105 - 4000.0) * LIGHT_CM:
                        frequency_15000 = (82259.105 - 15000.0) * LIGHT_CM
                        spacing = 100.0 * LIGHT_CM
                        if frequency < frequency_15000:
                            cutoff = (
                                h2plus_quasimolecular_cutoff_table[1]
                                - h2plus_quasimolecular_cutoff_table[0]
                            ) / spacing * (
                                frequency - frequency_15000
                            ) + h2plus_quasimolecular_cutoff_table[0]
                        else:
                            cutoff_index = int((frequency - frequency_15000) / spacing)
                            cutoff_index = max(
                                0,
                                min(
                                    cutoff_index,
                                    h2plus_quasimolecular_cutoff_table.shape[0] - 2,
                                ),
                            )
                            cutoff_frequency = cutoff_index * spacing + frequency_15000
                            cutoff = (
                                h2plus_quasimolecular_cutoff_table[cutoff_index + 1]
                                - h2plus_quasimolecular_cutoff_table[cutoff_index]
                            ) / spacing * (
                                frequency - cutoff_frequency
                            ) + h2plus_quasimolecular_cutoff_table[cutoff_index]
                        cutoff = (
                            10.0 ** (cutoff - 14.0)
                            / LIGHT_CM
                            * hydrogen_ionized_population[layer]
                        )
                        stark_value += cutoff * SQRT_PI * doppler_frequency_width
                    else:
                        beta4000 = 4000.0 * LIGHT_CM / layer_field_strength * beta_scale
                        probability4000 = (
                            _hydrogen_stark_probability_compiled(
                                beta4000,
                                pressure_parameter[layer],
                                lower_level,
                                upper_level,
                                stark_probability_table,
                                stark_pressure_grid,
                                stark_beta_grid,
                                stark_wing_correction_c,
                                stark_wing_correction_d,
                            )
                            * 0.5
                            / layer_field_strength
                            * beta_scale
                        )
                        cutoff4000 = (
                            10.0 ** (-11.07 - 14.0)
                            / LIGHT_CM
                            * hydrogen_ionized_population[layer]
                        )
                        if probability4000 != 0.0:
                            stark_value += (
                                cutoff4000
                                / probability4000
                                * probability
                                / layer_field_strength
                                * beta_scale
                                * SQRT_PI
                                * doppler_frequency_width
                            )

            lorentz_component = 0.0
            if impact_width > 0.0:
                lorentz_component = (
                    impact_width / PI / (impact_width * impact_width + beta * beta)
                )
            satellite_blend_square = (0.9 * linear_impact_argument) ** 2.0
            satellite_enhancement = (
                satellite_blend_square
                + 0.03 * np.sqrt(max(linear_impact_argument, 0.0))
            ) / (satellite_blend_square + 1.0)
            stark_value += (
                (probability * (1.0 + satellite_enhancement) + lorentz_component)
                / layer_field_strength
                * beta_scale
                * SQRT_PI
                * doppler_frequency_width
            )

        if in_core:
            if profile_mode == 1:
                value = doppler_value
            elif profile_mode == 2:
                value = lorentz_value
            else:
                value = stark_value
        else:
            value = doppler_value + lorentz_value + stark_value
        return max(value, 0.0)

    @_njit
    def _hydrogen_line_deposit_compiled(
        line_mass_absorption_coefficient,
        center_index,
        vacuum_wavelength_nm,
        scaled_oscillator_strength,
        lower_hydrogen_level,
        continuum_column,
        continuum_threshold,
        wavelength_grid,
        hydrogen_boltzmann_widths,
        continuum_selector_value,
        hydrogen_level_dissolution_wavenumber_cm,
        # line setup
        lower_level,
        upper_level,
        line_frequency_hz,
        line_wavelength_a,
        beta_scale,
        stark_c1_factor,
        stark_c2_factor,
        setup_radiative_width,
        setup_resonance_width,
        setup_van_der_waals_width,
        setup_stark_width,
        low_density_impact_numerator,
        impact_electron_density_threshold_cm3,
        stark_component_offsets_hz,
        stark_component_weights,
        # evaluator per-layer arrays
        hydrogen_fractional_doppler_width,
        field_strength_arr,
        temperature_density_he,
        temperature_density_h2,
        hydrogen_neutral_population,
        hydrogen_neutral_partition_normalized_population,
        hydrogen_ionized_population,
        electron_density_arr,
        low_density_impact_factor,
        high_density_impact_factor,
        stark_linear_density_coefficient,
        stark_quadratic_density_coefficient,
        stark_gamma_thermal_correction,
        stark_gamma_density_correction,
        pressure_parameter,
        # tables
        h2_quasimolecular_cutoff_table,
        h2plus_quasimolecular_cutoff_table,
        stark_probability_table,
        stark_pressure_grid,
        stark_beta_grid,
        stark_wing_correction_c,
        stark_wing_correction_d,
        exponential_integer,
        exponential_fraction,
        exponential_integral_table,
    ):
        """Mirror of the hydrogen depth/wavelength deposit loop in
        accumulate_transition_line_opacity (line_type == -1 branch)."""
        layer_count = hydrogen_fractional_doppler_width.shape[0]
        wavelength_count = wavelength_grid.shape[0]
        # NEP-50 mirror: the scalar path multiplies np.float32 by a weak
        # Python float, which numpy 2.x computes in float32; replicate by
        # casting both factors to float32 before the multiply.
        scaled_oscillator_f32 = np.float32(scaled_oscillator_strength)
        for depth_index in range(layer_count):
            center_absorption = np.float32(
                scaled_oscillator_f32
                * np.float32(
                    hydrogen_boltzmann_widths[depth_index, lower_hydrogen_level - 1]
                )
            )
            if center_absorption < continuum_threshold[depth_index, continuum_column]:
                continue
            denominator = (
                continuum_selector_value
                - hydrogen_level_dissolution_wavenumber_cm[depth_index]
            )
            if denominator == 0.0:
                continue
            blue_continuum_cutoff_nm = 1.0e7 / denominator
            for wavelength_index in range(
                center_index, min(center_index + 2001, wavelength_count)
            ):
                if wavelength_grid[wavelength_index] < blue_continuum_cutoff_nm:
                    continue
                wavelength_offset = np.float32(
                    wavelength_grid[wavelength_index] - vacuum_wavelength_nm
                )
                hydrogen_profile = np.float32(
                    _hydrogen_profile_compiled(
                        depth_index,
                        float(wavelength_offset),
                        lower_level,
                        upper_level,
                        line_frequency_hz,
                        line_wavelength_a,
                        beta_scale,
                        stark_c1_factor,
                        stark_c2_factor,
                        setup_radiative_width,
                        setup_resonance_width,
                        setup_van_der_waals_width,
                        setup_stark_width,
                        low_density_impact_numerator,
                        impact_electron_density_threshold_cm3,
                        stark_component_offsets_hz,
                        stark_component_weights,
                        hydrogen_fractional_doppler_width,
                        field_strength_arr,
                        temperature_density_he,
                        temperature_density_h2,
                        hydrogen_neutral_population,
                        hydrogen_neutral_partition_normalized_population,
                        hydrogen_ionized_population,
                        electron_density_arr,
                        low_density_impact_factor,
                        high_density_impact_factor,
                        stark_linear_density_coefficient,
                        stark_quadratic_density_coefficient,
                        stark_gamma_thermal_correction,
                        stark_gamma_density_correction,
                        pressure_parameter,
                        h2_quasimolecular_cutoff_table,
                        h2plus_quasimolecular_cutoff_table,
                        stark_probability_table,
                        stark_pressure_grid,
                        stark_beta_grid,
                        stark_wing_correction_c,
                        stark_wing_correction_d,
                        exponential_integer,
                        exponential_fraction,
                        exponential_integral_table,
                    )
                )
                contribution = np.float32(center_absorption * hydrogen_profile)
                line_mass_absorption_coefficient[depth_index, wavelength_index] += (
                    contribution
                )
                if contribution < continuum_threshold[depth_index, continuum_column]:
                    break
            for red_step in range(1, 2001):
                wavelength_index = center_index - red_step
                if wavelength_index < 0:
                    break
                if wavelength_grid[wavelength_index] < blue_continuum_cutoff_nm:
                    break
                wavelength_offset = np.float32(
                    wavelength_grid[wavelength_index] - vacuum_wavelength_nm
                )
                hydrogen_profile = np.float32(
                    _hydrogen_profile_compiled(
                        depth_index,
                        float(wavelength_offset),
                        lower_level,
                        upper_level,
                        line_frequency_hz,
                        line_wavelength_a,
                        beta_scale,
                        stark_c1_factor,
                        stark_c2_factor,
                        setup_radiative_width,
                        setup_resonance_width,
                        setup_van_der_waals_width,
                        setup_stark_width,
                        low_density_impact_numerator,
                        impact_electron_density_threshold_cm3,
                        stark_component_offsets_hz,
                        stark_component_weights,
                        hydrogen_fractional_doppler_width,
                        field_strength_arr,
                        temperature_density_he,
                        temperature_density_h2,
                        hydrogen_neutral_population,
                        hydrogen_neutral_partition_normalized_population,
                        hydrogen_ionized_population,
                        electron_density_arr,
                        low_density_impact_factor,
                        high_density_impact_factor,
                        stark_linear_density_coefficient,
                        stark_quadratic_density_coefficient,
                        stark_gamma_thermal_correction,
                        stark_gamma_density_correction,
                        pressure_parameter,
                        h2_quasimolecular_cutoff_table,
                        h2plus_quasimolecular_cutoff_table,
                        stark_probability_table,
                        stark_pressure_grid,
                        stark_beta_grid,
                        stark_wing_correction_c,
                        stark_wing_correction_d,
                        exponential_integer,
                        exponential_fraction,
                        exponential_integral_table,
                    )
                )
                contribution = np.float32(center_absorption * hydrogen_profile)
                line_mass_absorption_coefficient[depth_index, wavelength_index] += (
                    contribution
                )
                if contribution < continuum_threshold[depth_index, continuum_column]:
                    break

    @numba.njit(parallel=True, nogil=True, cache=True)
    def _accumulate_selected_line_opacity_parallel(
        chunk_count,
        line_count,
        packed_wavelength_index,
        vacuum_wavelength_nm_by_line,
        packed_species_slot,
        lower_excitation_index,
        log_strength_index,
        radiative_damping_index,
        stark_damping_index,
        van_der_waals_damping_index,
        wavelength_grid,
        bin_edges,
        continuum_threshold,
        hc_over_kt_cm,
        electrons,
        line_population_widths,
        doppler,
        neutral_collision_density,
        selection_lookup,
        integer_step,
        fractional_step,
        gaussian,
        first,
        second,
        wavelength_start_index,
        stop_index,
    ):
        layer_count = hc_over_kt_cm.shape[0]
        wavelength_count = wavelength_grid.shape[0]
        buffers = np.zeros(
            (chunk_count, layer_count, wavelength_count), dtype=np.float32
        )
        counts = np.zeros(chunk_count, dtype=np.int64)
        bounds = np.empty(chunk_count + 1, dtype=np.int64)
        for c in range(chunk_count + 1):
            bounds[c] = (line_count * c) // chunk_count
        for c in numba.prange(chunk_count):
            start = bounds[c]
            stop = bounds[c + 1]
            if stop <= start:
                continue
            buffer, count = _accumulate_selected_line_opacity_compiled(
                stop - start,
                packed_wavelength_index[start:stop],
                vacuum_wavelength_nm_by_line[start:stop],
                packed_species_slot[start:stop],
                lower_excitation_index[start:stop],
                log_strength_index[start:stop],
                radiative_damping_index[start:stop],
                stark_damping_index[start:stop],
                van_der_waals_damping_index[start:stop],
                wavelength_grid,
                bin_edges,
                continuum_threshold,
                hc_over_kt_cm,
                electrons,
                line_population_widths,
                doppler,
                neutral_collision_density,
                selection_lookup,
                integer_step,
                fractional_step,
                gaussian,
                first,
                second,
                wavelength_start_index,
                stop_index,
            )
            buffers[c] = buffer
            counts[c] = count
        total = np.zeros((layer_count, wavelength_count), dtype=np.float32)
        total_count = 0
        for c in range(chunk_count):
            total += buffers[c]
            total_count += counts[c]
        return total, total_count

    @_njit
    def _accumulate_transition_range_compiled(
        range_start,
        range_stop,
        out_absorption,
        line_type,
        special_indices,
        packed_wavelength_index,
        vacuum_wavelength_nm_by_line,
        packed_species_slot,
        oscillator_strength,
        lower_excitation_cm,
        radiative_damping_by_line,
        stark_damping_by_line,
        van_der_waals_damping_by_line,
        selector_index_by_line,
        selector_species_slot_by_line,
        wavelength_grid,
        bin_edges,
        continuum_threshold,
        hc_over_kt_cm,
        electrons,
        line_population_widths,
        doppler,
        neutral_collision_density,
        continuum_selector,
        hydrogen_level_dissolution_wavenumber_cm,
        integer_step,
        fractional_step,
        gaussian_table,
        first_table,
        second_table,
        wavelength_start_index,
        stop_index,
        partition_normalized_ion_stage_populations,
        mass_density,
        has_partitioned,
        hydrogen_boltzmann_widths,
        continuum_selector_col0,
        h_lower_level_line,
        h_upper_level_line,
        h_line_frequency_hz_line,
        h_line_wavelength_a_line,
        h_beta_scale_line,
        h_stark_c1_factor_line,
        h_stark_c2_factor_line,
        h_radiative_width_line,
        h_resonance_width_line,
        h_van_der_waals_width_line,
        h_stark_width_line,
        hydrogen_low_density_impact_numerator_by_line,
        hydrogen_impact_electron_density_threshold_cm3_by_line,
        h_stark_offsets_line,
        h_stark_weights_line,
        h_stark_component_count_line,
        h_valid_line,
        kernel_h_doppler,
        kernel_h_field,
        kernel_h_td_he,
        kernel_h_td_h2,
        kernel_h_neutral,
        kernel_h_ground,
        kernel_h_ionized,
        kernel_h_electrons,
        hydrogen_low_density_impact_factor_by_layer,
        hydrogen_high_density_impact_factor_by_layer,
        kernel_h_c1,
        kernel_h_c2,
        kernel_h_gcon1,
        kernel_h_gcon2,
        kernel_h_pressure,
        kernel_h2_cutoff,
        kernel_h2plus_cutoff,
        kernel_h_stark_prob,
        kernel_h_pressure_grid,
        kernel_h_beta_grid,
        kernel_h_stark_c,
        kernel_h_stark_d,
        kernel_h_exp_integer,
        kernel_h_exp_fraction,
        kernel_h_exp_integral,
    ):
        """Pure-njit transcription of the Python ``_run_line_range`` driver.

        Processes transition lines ``[range_start, range_stop)`` into
        ``out_absorption`` with a freshly re-seeded, memoryless walk so any
        contiguous line-chunk reproduces the serial deposit for that range.
        Handles normal runs (type 0/2/3), hydrogen (-1), autoionizing (1), and
        merged-continuum records identically to the scalar Python path.
        """
        layer_count = hc_over_kt_cm.shape[0]
        wavelength_count = wavelength_grid.shape[0]
        n_special = special_indices.shape[0]
        line_total = line_type.shape[0]
        depth_gate = np.zeros(layer_count + 2, dtype=np.int32)
        continuum_column = 0
        center_index = wavelength_start_index - 1
        if center_index < 0:
            center_index = 0
        processed = 0
        skip_until = -1
        stop_all_lines = False
        last_grid_value = wavelength_grid[min(stop_index, wavelength_count) - 1]

        line_index = range_start
        while line_index < range_stop:
            if stop_all_lines or line_index < skip_until:
                line_index += 1
                continue

            current_type = line_type[line_index]
            is_normal = current_type == 0 or current_type == 2 or current_type == 3
            if is_normal:
                lo_i = 0
                hi_i = n_special
                while lo_i < hi_i:
                    mid = (lo_i + hi_i) // 2
                    if special_indices[mid] < line_index:
                        lo_i = mid + 1
                    else:
                        hi_i = mid
                insert_at = lo_i
                if insert_at < n_special:
                    run_stop = special_indices[insert_at]
                else:
                    run_stop = line_total
                if run_stop > range_stop:
                    run_stop = range_stop
                (
                    continuum_column,
                    center_index,
                    processed_delta,
                    run_stop_flag,
                    _stopped_at,
                ) = _accumulate_transition_run_compiled(
                    line_index,
                    run_stop,
                    line_type,
                    packed_wavelength_index,
                    vacuum_wavelength_nm_by_line,
                    packed_species_slot,
                    oscillator_strength,
                    lower_excitation_cm,
                    radiative_damping_by_line,
                    stark_damping_by_line,
                    van_der_waals_damping_by_line,
                    selector_index_by_line,
                    selector_species_slot_by_line,
                    wavelength_grid,
                    bin_edges,
                    continuum_threshold,
                    hc_over_kt_cm,
                    electrons,
                    line_population_widths,
                    doppler,
                    neutral_collision_density,
                    continuum_selector,
                    hydrogen_level_dissolution_wavenumber_cm,
                    integer_step,
                    fractional_step,
                    gaussian_table,
                    first_table,
                    second_table,
                    depth_gate,
                    out_absorption,
                    continuum_column,
                    center_index,
                    wavelength_start_index,
                    stop_index,
                )
                processed += processed_delta
                if run_stop_flag:
                    stop_all_lines = True
                    line_index += 1
                    continue
                skip_until = run_stop
                line_index += 1
                continue

            vacuum_wavelength_nm = vacuum_wavelength_nm_by_line[line_index]
            if vacuum_wavelength_nm > last_grid_value:
                break

            packed_wavelength = packed_wavelength_index[line_index]
            while continuum_column < bin_edges.shape[0] and (
                packed_wavelength >= bin_edges[continuum_column]
            ):
                continuum_column += 1
            if continuum_column >= continuum_threshold.shape[1]:
                line_index += 1
                continue

            while center_index < wavelength_count and (
                vacuum_wavelength_nm >= wavelength_grid[center_index]
            ):
                center_index += 1
            if center_index >= wavelength_count:
                break

            species_slot = packed_species_slot[line_index]
            line_oscillator_strength = oscillator_strength[line_index]
            line_lower_excitation_cm = lower_excitation_cm[line_index]
            radiative_damping = radiative_damping_by_line[line_index]
            stark_damping = stark_damping_by_line[line_index]
            van_der_waals_damping = van_der_waals_damping_by_line[line_index]

            if current_type == -1:
                if h_valid_line[line_index] == 0:
                    line_index += 1
                    continue
                selector_index = selector_index_by_line[line_index]
                scaled_oscillator_strength = np.float32(line_oscillator_strength)
                comp_count = h_stark_component_count_line[line_index]
                stark_offsets = h_stark_offsets_line[line_index, 0:comp_count]
                stark_weights = h_stark_weights_line[line_index, 0:comp_count]
                _hydrogen_line_deposit_compiled(
                    out_absorption,
                    center_index,
                    vacuum_wavelength_nm,
                    float(scaled_oscillator_strength),
                    h_lower_level_line[line_index],
                    continuum_column,
                    continuum_threshold,
                    wavelength_grid,
                    hydrogen_boltzmann_widths,
                    continuum_selector_col0[selector_index - 1],
                    hydrogen_level_dissolution_wavenumber_cm,
                    h_lower_level_line[line_index],
                    h_upper_level_line[line_index],
                    h_line_frequency_hz_line[line_index],
                    h_line_wavelength_a_line[line_index],
                    h_beta_scale_line[line_index],
                    h_stark_c1_factor_line[line_index],
                    h_stark_c2_factor_line[line_index],
                    h_radiative_width_line[line_index],
                    h_resonance_width_line[line_index],
                    h_van_der_waals_width_line[line_index],
                    h_stark_width_line[line_index],
                    hydrogen_low_density_impact_numerator_by_line[line_index],
                    hydrogen_impact_electron_density_threshold_cm3_by_line[line_index],
                    stark_offsets,
                    stark_weights,
                    kernel_h_doppler,
                    kernel_h_field,
                    kernel_h_td_he,
                    kernel_h_td_h2,
                    kernel_h_neutral,
                    kernel_h_ground,
                    kernel_h_ionized,
                    kernel_h_electrons,
                    hydrogen_low_density_impact_factor_by_layer,
                    hydrogen_high_density_impact_factor_by_layer,
                    kernel_h_c1,
                    kernel_h_c2,
                    kernel_h_gcon1,
                    kernel_h_gcon2,
                    kernel_h_pressure,
                    kernel_h2_cutoff,
                    kernel_h2plus_cutoff,
                    kernel_h_stark_prob,
                    kernel_h_pressure_grid,
                    kernel_h_beta_grid,
                    kernel_h_stark_c,
                    kernel_h_stark_d,
                    kernel_h_exp_integer,
                    kernel_h_exp_fraction,
                    kernel_h_exp_integral,
                )
                processed += 1
                line_index += 1
                continue

            if current_type == 1:
                if (
                    species_slot < 1
                    or species_slot
                    > partition_normalized_ion_stage_populations.shape[1]
                ):
                    line_index += 1
                    continue
                line_frequency_hz = _LIGHT_SPEED_NM_PER_SECOND / max(
                    vacuum_wavelength_nm, 1.0e-300
                )
                shore_asymmetry = stark_damping
                shore_width = van_der_waals_damping
                if shore_width == 0.0 or radiative_damping == 0.0:
                    line_index += 1
                    continue
                for depth_index in range(layer_count):
                    center_absorption = (
                        shore_width
                        * line_oscillator_strength
                        * partition_normalized_ion_stage_populations[
                            depth_index, species_slot - 1
                        ]
                        / max(mass_density[depth_index], 1.0e-300)
                    )
                    if (
                        center_absorption
                        < continuum_threshold[depth_index, continuum_column]
                    ):
                        continue
                    center_absorption *= _fast_exponential_lookup_compiled(
                        line_lower_excitation_cm * hc_over_kt_cm[depth_index],
                        integer_step,
                        fractional_step,
                    )
                    if (
                        center_absorption
                        < continuum_threshold[depth_index, continuum_column]
                    ):
                        continue
                    blue_stop = center_index + 2001
                    if blue_stop > wavelength_count:
                        blue_stop = wavelength_count
                    for wavelength_index in range(center_index, blue_stop):
                        reduced_frequency = (
                            2.0
                            * (
                                _LIGHT_SPEED_NM_PER_SECOND
                                / max(wavelength_grid[wavelength_index], 1.0e-300)
                                - line_frequency_hz
                            )
                            / radiative_damping
                        )
                        contribution = (
                            center_absorption
                            * (shore_asymmetry * reduced_frequency + shore_width)
                            / (reduced_frequency * reduced_frequency + 1.0)
                            / shore_width
                        )
                        out_absorption[depth_index, wavelength_index] += np.float32(
                            contribution
                        )
                        if (
                            contribution
                            < continuum_threshold[depth_index, continuum_column]
                        ):
                            break
                    for red_step in range(1, 2001):
                        wavelength_index = center_index - red_step
                        if wavelength_index < 0:
                            wavelength_index = 0
                        reduced_frequency = (
                            2.0
                            * (
                                _LIGHT_SPEED_NM_PER_SECOND
                                / max(wavelength_grid[wavelength_index], 1.0e-300)
                                - line_frequency_hz
                            )
                            / radiative_damping
                        )
                        contribution = (
                            center_absorption
                            * (shore_asymmetry * reduced_frequency + shore_width)
                            / (reduced_frequency * reduced_frequency + 1.0)
                            / shore_width
                        )
                        out_absorption[depth_index, wavelength_index] += np.float32(
                            contribution
                        )
                        if (
                            contribution
                            < continuum_threshold[depth_index, continuum_column]
                        ):
                            break
                processed += 1
                line_index += 1
                continue

            if (
                species_slot < 1
                or species_slot > partition_normalized_ion_stage_populations.shape[1]
            ):
                line_index += 1
                continue
            if species_slot == 4:
                effective_charge = 2.0
            else:
                effective_charge = 1.0
            last_level = float(current_type)
            if last_level == 0.0:
                line_index += 1
                continue
            shifted_wavelength = 1.0e7 / (
                1.0e7 / vacuum_wavelength_nm
                - 109737.312
                * effective_charge
                * effective_charge
                / (last_level * last_level)
            )
            for depth_index in range(layer_count):
                merged_wavelength = 1.0e7 / (
                    1.0e7 / vacuum_wavelength_nm
                    - hydrogen_level_dissolution_wavenumber_cm[depth_index]
                    * effective_charge
                    * effective_charge
                )
                maximum_wavelength = max(merged_wavelength, shifted_wavelength)
                continuum_absorption = (
                    line_oscillator_strength
                    * partition_normalized_ion_stage_populations[
                        depth_index, species_slot - 1
                    ]
                    * _fast_exponential_lookup_compiled(
                        line_lower_excitation_cm * hc_over_kt_cm[depth_index],
                        integer_step,
                        fractional_step,
                    )
                    / max(mass_density[depth_index], 1.0e-300)
                )
                merged_stop = center_index + 1001
                if merged_stop > wavelength_count:
                    merged_stop = wavelength_count
                for wavelength_index in range(center_index, merged_stop):
                    if maximum_wavelength < wavelength_grid[wavelength_index]:
                        break
                    out_absorption[depth_index, wavelength_index] += np.float32(
                        continuum_absorption
                    )
            processed += 1
            line_index += 1

        return processed

    @numba.njit(parallel=True, nogil=True, cache=True)
    def _accumulate_transition_line_opacity_parallel(
        chunk_count,
        line_count,
        base_line_mass_absorption_coefficient,
        line_type,
        special_indices,
        packed_wavelength_index,
        vacuum_wavelength_nm_by_line,
        packed_species_slot,
        oscillator_strength,
        lower_excitation_cm,
        radiative_damping_by_line,
        stark_damping_by_line,
        van_der_waals_damping_by_line,
        selector_index_by_line,
        selector_species_slot_by_line,
        wavelength_grid,
        bin_edges,
        continuum_threshold,
        hc_over_kt_cm,
        electrons,
        line_population_widths,
        doppler,
        neutral_collision_density,
        continuum_selector,
        hydrogen_level_dissolution_wavenumber_cm,
        integer_step,
        fractional_step,
        gaussian_table,
        first_table,
        second_table,
        wavelength_start_index,
        stop_index,
        partition_normalized_ion_stage_populations,
        mass_density,
        has_partitioned,
        hydrogen_boltzmann_widths,
        continuum_selector_col0,
        h_lower_level_line,
        h_upper_level_line,
        h_line_frequency_hz_line,
        h_line_wavelength_a_line,
        h_beta_scale_line,
        h_stark_c1_factor_line,
        h_stark_c2_factor_line,
        h_radiative_width_line,
        h_resonance_width_line,
        h_van_der_waals_width_line,
        h_stark_width_line,
        hydrogen_low_density_impact_numerator_by_line,
        hydrogen_impact_electron_density_threshold_cm3_by_line,
        h_stark_offsets_line,
        h_stark_weights_line,
        h_stark_component_count_line,
        h_valid_line,
        kernel_h_doppler,
        kernel_h_field,
        kernel_h_td_he,
        kernel_h_td_h2,
        kernel_h_neutral,
        kernel_h_ground,
        kernel_h_ionized,
        kernel_h_electrons,
        hydrogen_low_density_impact_factor_by_layer,
        hydrogen_high_density_impact_factor_by_layer,
        kernel_h_c1,
        kernel_h_c2,
        kernel_h_gcon1,
        kernel_h_gcon2,
        kernel_h_pressure,
        kernel_h2_cutoff,
        kernel_h2plus_cutoff,
        kernel_h_stark_prob,
        kernel_h_pressure_grid,
        kernel_h_beta_grid,
        kernel_h_stark_c,
        kernel_h_stark_d,
        kernel_h_exp_integer,
        kernel_h_exp_fraction,
        kernel_h_exp_integral,
    ):
        layer_count = hc_over_kt_cm.shape[0]
        wavelength_count = wavelength_grid.shape[0]
        buffers = np.zeros(
            (chunk_count, layer_count, wavelength_count), dtype=np.float32
        )
        counts = np.zeros(chunk_count, dtype=np.int64)
        bounds = np.empty(chunk_count + 1, dtype=np.int64)
        for c in range(chunk_count + 1):
            bounds[c] = (line_count * c) // chunk_count
        for c in numba.prange(chunk_count):
            start = bounds[c]
            stop = bounds[c + 1]
            if stop <= start:
                continue
            counts[c] = _accumulate_transition_range_compiled(
                start,
                stop,
                buffers[c],
                line_type,
                special_indices,
                packed_wavelength_index,
                vacuum_wavelength_nm_by_line,
                packed_species_slot,
                oscillator_strength,
                lower_excitation_cm,
                radiative_damping_by_line,
                stark_damping_by_line,
                van_der_waals_damping_by_line,
                selector_index_by_line,
                selector_species_slot_by_line,
                wavelength_grid,
                bin_edges,
                continuum_threshold,
                hc_over_kt_cm,
                electrons,
                line_population_widths,
                doppler,
                neutral_collision_density,
                continuum_selector,
                hydrogen_level_dissolution_wavenumber_cm,
                integer_step,
                fractional_step,
                gaussian_table,
                first_table,
                second_table,
                wavelength_start_index,
                stop_index,
                partition_normalized_ion_stage_populations,
                mass_density,
                has_partitioned,
                hydrogen_boltzmann_widths,
                continuum_selector_col0,
                h_lower_level_line,
                h_upper_level_line,
                h_line_frequency_hz_line,
                h_line_wavelength_a_line,
                h_beta_scale_line,
                h_stark_c1_factor_line,
                h_stark_c2_factor_line,
                h_radiative_width_line,
                h_resonance_width_line,
                h_van_der_waals_width_line,
                h_stark_width_line,
                hydrogen_low_density_impact_numerator_by_line,
                hydrogen_impact_electron_density_threshold_cm3_by_line,
                h_stark_offsets_line,
                h_stark_weights_line,
                h_stark_component_count_line,
                h_valid_line,
                kernel_h_doppler,
                kernel_h_field,
                kernel_h_td_he,
                kernel_h_td_h2,
                kernel_h_neutral,
                kernel_h_ground,
                kernel_h_ionized,
                kernel_h_electrons,
                hydrogen_low_density_impact_factor_by_layer,
                hydrogen_high_density_impact_factor_by_layer,
                kernel_h_c1,
                kernel_h_c2,
                kernel_h_gcon1,
                kernel_h_gcon2,
                kernel_h_pressure,
                kernel_h2_cutoff,
                kernel_h2plus_cutoff,
                kernel_h_stark_prob,
                kernel_h_pressure_grid,
                kernel_h_beta_grid,
                kernel_h_stark_c,
                kernel_h_stark_d,
                kernel_h_exp_integer,
                kernel_h_exp_fraction,
                kernel_h_exp_integral,
            )
        total = base_line_mass_absorption_coefficient.copy()
        total_count = 0
        for c in range(chunk_count):
            total += buffers[c]
            total_count += counts[c]
        return total, total_count


def _hydrogen_neutral_level_energies_cm() -> np.ndarray:
    levels = np.zeros(100, dtype=np.float64)
    levels[1:10] = [
        82259.105,
        97492.302,
        102823.893,
        105291.651,
        106632.160,
        107440.444,
        107965.051,
        108324.720,
        108581.988,
    ]
    for principal_quantum_number in range(11, 101):
        levels[principal_quantum_number - 1] = 109678.764 - 109677.576 / float(
            principal_quantum_number * principal_quantum_number
        )
    return levels


@dataclass(frozen=True)
class LineOpacityState:
    """Line-opacity slab plus the number of selected lines that contributed."""

    line_mass_absorption_coefficient: np.ndarray
    selected_line_count: int = 0


def allocate_line_opacity_state(
    *, layer_count: int, wavelength_count: int
) -> LineOpacityState:
    """Allocate the line-absorption workspace."""

    return LineOpacityState(
        line_mass_absorption_coefficient=np.zeros(
            (layer_count, wavelength_count), dtype=np.float64
        ),
        selected_line_count=0,
    )


def _sanitize_float32(values: np.ndarray, *, ceiling: float = 1.0e30) -> np.ndarray:
    finite = np.where(np.isfinite(np.asarray(values, dtype=np.float64)), values, 0.0)
    return np.ascontiguousarray(np.clip(finite, -ceiling, ceiling), dtype=np.float32)


def _sanitize_float64(values: np.ndarray) -> np.ndarray:
    finite = np.where(np.isfinite(np.asarray(values, dtype=np.float64)), values, 0.0)
    return np.ascontiguousarray(finite, dtype=np.float64)


def accumulate_selected_line_opacity(
    *,
    selected_lines: SelectedLineCatalog,
    opacity_wavelength_grid_nm: np.ndarray,
    wavelength_bin_edges: np.ndarray,
    continuum_line_selection_threshold: np.ndarray,
    temperature: np.ndarray,
    hc_over_kt: np.ndarray,
    electron_density: np.ndarray,
    ion_stage_populations_by_packed_slot: np.ndarray,
    partition_normalized_population_over_mass_density_and_fractional_doppler_width: np.ndarray,
    fractional_doppler_widths: np.ndarray,
    wavelength_start_index: int = 1,
    wavelength_stop_index: int | None = None,
) -> LineOpacityState:
    """Accumulate line opacity from compact selected-line records.

    Real workloads run the compiled kernel over parallel line-chunks (one chunk
    per numba thread; see ``_line_opacity_chunk_count``); a single line or a
    single thread falls back to the serial compiled kernel.
    """

    wavelength_grid = np.ascontiguousarray(opacity_wavelength_grid_nm, dtype=np.float64)
    bin_edges = np.ascontiguousarray(wavelength_bin_edges, dtype=np.int64)
    continuum_threshold = _sanitize_float32(continuum_line_selection_threshold)
    safe_temperature = np.asarray(temperature, dtype=np.float64)
    safe_temperature = np.where(
        np.isfinite(safe_temperature) & (safe_temperature > 0.0),
        safe_temperature,
        1.0,
    )
    hc_over_kt_cm = _sanitize_float32(hc_over_kt, ceiling=1.0e10)
    electrons = _sanitize_float32(electron_density, ceiling=1.0e30)
    line_population_widths = _sanitize_float32(
        partition_normalized_population_over_mass_density_and_fractional_doppler_width
    )
    doppler = _sanitize_float32(fractional_doppler_widths, ceiling=1.0e10)
    populations = _sanitize_float64(ion_stage_populations_by_packed_slot)

    layer_count = int(safe_temperature.size)
    wavelength_count = int(wavelength_grid.size)
    stop_index = (
        wavelength_count
        if wavelength_stop_index is None
        else int(wavelength_stop_index)
    )
    if layer_count != 80:
        raise ValueError(
            f"Selected-line opacity expects 80 depth layers, got {layer_count}."
        )
    if populations.shape[1] <= 840:
        raise ValueError("population density table must include packed slot 841")

    neutral_collision_density = (
        populations[:, 0] + 0.42 * populations[:, 2] + 0.85 * populations[:, 840]
    ) * (np.maximum(safe_temperature, 1.0) / 10000.0) ** 0.3
    neutral_collision_density = _sanitize_float32(neutral_collision_density)

    exponential_tables = build_fast_exponential_tables()
    profile_basis = build_voigt_profile_basis()
    kernel_arguments = (
        np.ascontiguousarray(selected_lines.packed_wavelength_index, dtype=np.int32),
        np.ascontiguousarray(
            np.exp(
                np.asarray(selected_lines.packed_wavelength_index, dtype=np.float64)
                * _RATIO_LOG_STEP
            ),
            dtype=np.float64,
        ),
        np.ascontiguousarray(selected_lines.packed_species_slot, dtype=np.int16),
        np.ascontiguousarray(selected_lines.lower_excitation_index, dtype=np.int16),
        np.ascontiguousarray(selected_lines.log_strength_index, dtype=np.int16),
        np.ascontiguousarray(selected_lines.radiative_damping_index, dtype=np.int16),
        np.ascontiguousarray(selected_lines.stark_damping_index, dtype=np.int16),
        np.ascontiguousarray(
            selected_lines.van_der_waals_damping_index, dtype=np.int16
        ),
        wavelength_grid,
        bin_edges,
        continuum_threshold,
        hc_over_kt_cm,
        electrons,
        line_population_widths,
        doppler,
        neutral_collision_density,
        np.ascontiguousarray(build_selection_log_lookup(), dtype=np.float32),
        np.ascontiguousarray(exponential_tables.integer_step, dtype=np.float64),
        np.ascontiguousarray(exponential_tables.fractional_step, dtype=np.float64),
        np.ascontiguousarray(profile_basis.gaussian_profile, dtype=np.float64),
        np.ascontiguousarray(profile_basis.first_correction, dtype=np.float64),
        np.ascontiguousarray(profile_basis.second_correction, dtype=np.float64),
        int(wavelength_start_index),
        int(stop_index),
    )
    chunk_count = min(
        _line_opacity_chunk_count(),
        max(1, int(selected_lines.line_count)),
    )
    if chunk_count > 1 and int(selected_lines.line_count) > 1:
        line_mass_absorption_coefficient, contributing_lines = (
            _accumulate_selected_line_opacity_parallel(
                int(chunk_count),
                int(selected_lines.line_count),
                *kernel_arguments,
            )
        )
    else:
        # Trivial input (<= 1 line or single-thread) -> the serial compiled
        # kernel; identical deposits, no chunk-reduction overhead.
        line_mass_absorption_coefficient, contributing_lines = (
            _accumulate_selected_line_opacity_compiled(
                int(selected_lines.line_count),
                *kernel_arguments,
            )
        )
    return LineOpacityState(
        line_mass_absorption_coefficient=line_mass_absorption_coefficient,
        selected_line_count=int(contributing_lines),
    )


def accumulate_transition_line_opacity(
    *,
    transition_lines: LineTransitionCatalog,
    opacity_wavelength_grid_nm: np.ndarray,
    wavelength_bin_edges: np.ndarray,
    continuum_line_selection_threshold: np.ndarray,
    temperature: np.ndarray,
    hc_over_kt: np.ndarray,
    electron_density: np.ndarray,
    ion_stage_populations_by_packed_slot: np.ndarray,
    partition_normalized_population_over_mass_density_and_fractional_doppler_width: np.ndarray,
    fractional_doppler_widths: np.ndarray,
    partition_normalized_populations_by_packed_slot: np.ndarray | None = None,
    mass_density: np.ndarray | None = None,
    base_line_mass_absorption_coefficient: np.ndarray | None = None,
    wavelength_start_index: int = 1,
    wavelength_stop_index: int | None = None,
) -> LineOpacityState:
    """Accumulate detailed-transition opacity from XLINOP records.

    This covers XLINOP `TYPE=0/3` normal records, `TYPE=1` autoionizing
    records, merged-continuum records, and `TYPE=-1` hydrogen-line records.
    """

    wavelength_grid = np.ascontiguousarray(opacity_wavelength_grid_nm, dtype=np.float64)
    bin_edges = np.ascontiguousarray(wavelength_bin_edges, dtype=np.int64)
    continuum_threshold = np.ascontiguousarray(
        continuum_line_selection_threshold, dtype=np.float64
    )
    safe_temperature = np.asarray(temperature, dtype=np.float64)
    hc_over_kt_cm = np.ascontiguousarray(hc_over_kt, dtype=np.float64)
    electrons = np.ascontiguousarray(electron_density, dtype=np.float64)
    populations = np.ascontiguousarray(
        ion_stage_populations_by_packed_slot, dtype=np.float64
    )
    line_population_widths = np.ascontiguousarray(
        partition_normalized_population_over_mass_density_and_fractional_doppler_width,
        dtype=np.float64,
    )
    doppler = np.ascontiguousarray(fractional_doppler_widths, dtype=np.float64)
    partition_normalized_ion_stage_populations = (
        None
        if partition_normalized_populations_by_packed_slot is None
        else np.ascontiguousarray(
            partition_normalized_populations_by_packed_slot, dtype=np.float64
        )
    )
    density = (
        None
        if mass_density is None
        else np.ascontiguousarray(mass_density, dtype=np.float64)
    )

    layer_count = int(safe_temperature.size)
    wavelength_count = int(wavelength_grid.size)
    stop_index = (
        wavelength_count
        if wavelength_stop_index is None
        else int(wavelength_stop_index)
    )
    if layer_count != 80:
        raise ValueError(
            f"Transition-line opacity expects 80 depth layers, got {layer_count}."
        )
    if populations.shape[1] <= 840:
        raise ValueError("population density table must include packed slot 841")

    neutral_collision_density = (
        populations[:, 0] + 0.42 * populations[:, 2] + 0.85 * populations[:, 840]
    ) * (safe_temperature / 10000.0) ** 0.3
    stark_level_cutoff = 1600.0 / np.maximum(electrons, 1.0e-300) ** (2.0 / 15.0)
    hydrogen_level_dissolution_wavenumber_cm = 109737.312 / np.maximum(
        stark_level_cutoff * stark_level_cutoff,
        1.0e-300,
    )

    if base_line_mass_absorption_coefficient is None:
        line_mass_absorption_coefficient = np.zeros(
            (layer_count, wavelength_count), dtype=np.float32
        )
    else:
        line_mass_absorption_coefficient = np.asarray(
            base_line_mass_absorption_coefficient, dtype=np.float32
        ).copy()

    continuum_selector = load_hydrogen_continuum_selector_table()
    exponential_tables = build_fast_exponential_tables()
    hydrogen_neutral_level_energy_cm = _hydrogen_neutral_level_energies_cm()
    hydrogen_boltzmann_widths = None
    hydrogen_profile_evaluator = None
    if transition_lines.line_count and np.any(transition_lines.line_type == -1):
        if partition_normalized_ion_stage_populations is None:
            raise ValueError(
                "Hydrogen transition opacity requires partitioned populations."
            )
        if (
            populations.shape[1] <= 2
            or partition_normalized_ion_stage_populations.shape[1] <= 2
            or doppler.shape[1] < 1
        ):
            raise ValueError(
                "Hydrogen transition opacity requires H I, H II, He I, and H Doppler columns."
            )
        hydrogen_tables = load_hydrogen_line_profile_tables()
        molecular_hydrogen = compute_hydrogen_molecule_population(
            temperature=safe_temperature,
            hydrogen_neutral_partition_normalized_population=partition_normalized_ion_stage_populations[
                :, 0
            ],
            hydrogen_departure_coefficient=np.ones(layer_count, dtype=np.float64),
            tables=hydrogen_tables,
        )
        hydrogen_boltzmann_widths = (
            np.exp(-np.outer(hc_over_kt_cm, hydrogen_neutral_level_energy_cm))
            * line_population_widths[:, 0:1]
        )
        hydrogen_profile_evaluator = HydrogenLineProfileEvaluator(
            temperature=safe_temperature,
            electron_density=electrons,
            hydrogen_neutral_population=populations[:, 0],
            hydrogen_ionized_population=populations[:, 1],
            hydrogen_neutral_partition_normalized_population=partition_normalized_ion_stage_populations[
                :, 0
            ],
            helium_neutral_population=populations[:, 2],
            hydrogen_fractional_doppler_width=doppler[:, 0],
            molecular_hydrogen_population=molecular_hydrogen,
            tables=hydrogen_tables,
        )
    processed_lines = 0

    # Contiguous runs of normal records (line_type 0/2/3) are handled by the
    # compiled kernel _accumulate_transition_run_compiled; special records
    # (hydrogen, autoionizing, merged-continuum) are handled per-line below,
    # hydrogen via _hydrogen_line_deposit_compiled. The per-line walk state
    # (continuum_column/center_index) is a MEMORYLESS monotonic advance driven
    # purely by each line's own packed_wavelength / vacuum_wavelength (lines are
    # wavelength-sorted), so re-seeding it to (0, start-1) at any line index
    # reproduces the exact state the serial walk reaches there. That lets the
    # per-line loop be split into contiguous line-chunks that each re-seed and
    # accumulate into a PRIVATE float32 buffer -- the same chunk pattern the selected-line opacity path
    # uses (see _line_opacity_chunk_count). Buffers are summed in
    # chunk order, so the only difference from the serial path is a deterministic
    # float32 sum regrouping (~ulp per cell), immaterial under the spectrum gate.
    has_transition_lines = transition_lines.line_count > 0
    if has_transition_lines:
        kernel_line_type = np.ascontiguousarray(
            transition_lines.line_type, dtype=np.int64
        )
        kernel_is_normal = np.isin(kernel_line_type, (0, 2, 3))
        kernel_special_indices = np.nonzero(~kernel_is_normal)[0]
        kernel_packed_wavelength = np.ascontiguousarray(
            transition_lines.packed_wavelength_index, dtype=np.int64
        )
        kernel_vacuum_wavelength = np.ascontiguousarray(
            transition_lines.vacuum_wavelength_nm, dtype=np.float64
        )
        kernel_species_slot = np.ascontiguousarray(
            transition_lines.packed_species_slot, dtype=np.int64
        )
        kernel_oscillator_strength = np.ascontiguousarray(
            transition_lines.oscillator_strength, dtype=np.float64
        )
        kernel_lower_excitation = np.ascontiguousarray(
            transition_lines.lower_excitation_cm, dtype=np.float64
        )
        kernel_radiative_damping = np.ascontiguousarray(
            transition_lines.radiative_damping, dtype=np.float64
        )
        kernel_stark_damping = np.ascontiguousarray(
            transition_lines.stark_damping, dtype=np.float64
        )
        kernel_van_der_waals_damping = np.ascontiguousarray(
            transition_lines.van_der_waals_damping, dtype=np.float64
        )
        kernel_selector_index = np.ascontiguousarray(
            transition_lines.hydrogen_continuum_selector_index, dtype=np.int64
        )
        kernel_selector_slot = np.ascontiguousarray(
            transition_lines.continuum_species_slot, dtype=np.int64
        )
        kernel_profile_basis = build_voigt_profile_basis()
        kernel_gaussian = np.ascontiguousarray(
            kernel_profile_basis.gaussian_profile, dtype=np.float64
        )
        kernel_first = np.ascontiguousarray(
            kernel_profile_basis.first_correction, dtype=np.float64
        )
        kernel_second = np.ascontiguousarray(
            kernel_profile_basis.second_correction, dtype=np.float64
        )
        kernel_integer_step = np.ascontiguousarray(
            exponential_tables.integer_step, dtype=np.float64
        )
        kernel_fractional_step = np.ascontiguousarray(
            exponential_tables.fractional_step, dtype=np.float64
        )
        kernel_selector_table = np.ascontiguousarray(
            continuum_selector, dtype=np.float64
        )
        kernel_dissolved_energy = np.ascontiguousarray(
            hydrogen_level_dissolution_wavenumber_cm, dtype=np.float64
        )
        if hydrogen_profile_evaluator is not None:
            _hydrogen_evaluator = hydrogen_profile_evaluator
            _hydrogen_tables = _hydrogen_evaluator.tables
            kernel_h_doppler = np.ascontiguousarray(
                _hydrogen_evaluator.hydrogen_fractional_doppler_width, dtype=np.float64
            )
            kernel_h_field = np.ascontiguousarray(
                _hydrogen_evaluator.field_strength, dtype=np.float64
            )
            kernel_h_td_he = np.ascontiguousarray(
                _hydrogen_evaluator.temperature_density_he, dtype=np.float64
            )
            kernel_h_td_h2 = np.ascontiguousarray(
                _hydrogen_evaluator.temperature_density_h2, dtype=np.float64
            )
            kernel_h_neutral = np.ascontiguousarray(
                _hydrogen_evaluator.hydrogen_neutral_population, dtype=np.float64
            )
            kernel_h_ground = np.ascontiguousarray(
                _hydrogen_evaluator.hydrogen_neutral_partition_normalized_population,
                dtype=np.float64,
            )
            kernel_h_ionized = np.ascontiguousarray(
                _hydrogen_evaluator.hydrogen_ionized_population, dtype=np.float64
            )
            kernel_h_electrons = np.ascontiguousarray(
                _hydrogen_evaluator.electron_density, dtype=np.float64
            )
            hydrogen_low_density_impact_factor_by_layer = np.ascontiguousarray(
                _hydrogen_evaluator.low_density_impact_factor, dtype=np.float64
            )
            hydrogen_high_density_impact_factor_by_layer = np.ascontiguousarray(
                _hydrogen_evaluator.high_density_impact_factor, dtype=np.float64
            )
            kernel_h_c1 = np.ascontiguousarray(
                _hydrogen_evaluator.stark_linear_density_coefficient, dtype=np.float64
            )
            kernel_h_c2 = np.ascontiguousarray(
                _hydrogen_evaluator.stark_quadratic_density_coefficient,
                dtype=np.float64,
            )
            kernel_h_gcon1 = np.ascontiguousarray(
                _hydrogen_evaluator.stark_gamma_thermal_correction, dtype=np.float64
            )
            kernel_h_gcon2 = np.ascontiguousarray(
                _hydrogen_evaluator.stark_gamma_density_correction, dtype=np.float64
            )
            kernel_h_pressure = np.ascontiguousarray(
                _hydrogen_evaluator.pressure_parameter, dtype=np.float64
            )
            kernel_h2_cutoff = np.ascontiguousarray(
                _hydrogen_tables.h2_quasimolecular_cutoff_table, dtype=np.float64
            )
            kernel_h2plus_cutoff = np.ascontiguousarray(
                _hydrogen_tables.h2plus_quasimolecular_cutoff_table, dtype=np.float64
            )
            kernel_h_stark_prob = np.ascontiguousarray(
                _hydrogen_tables.stark_probability_table, dtype=np.float64
            )
            kernel_h_pressure_grid = np.ascontiguousarray(
                _hydrogen_tables.stark_pressure_grid, dtype=np.float64
            )
            kernel_h_beta_grid = np.ascontiguousarray(
                _hydrogen_tables.stark_beta_grid, dtype=np.float64
            )
            kernel_h_stark_c = np.ascontiguousarray(
                _hydrogen_tables.stark_wing_correction_c, dtype=np.float64
            )
            kernel_h_stark_d = np.ascontiguousarray(
                _hydrogen_tables.stark_wing_correction_d, dtype=np.float64
            )
            kernel_h_exp_integer = np.ascontiguousarray(
                _hydrogen_evaluator.exponential_integer, dtype=np.float64
            )
            kernel_h_exp_fraction = np.ascontiguousarray(
                _hydrogen_evaluator.exponential_fraction, dtype=np.float64
            )
            kernel_h_exp_integral = np.ascontiguousarray(
                _hydrogen_exponential_integral_table(), dtype=np.float64
            )

        # --- Precompute per-line hydrogen line-setup arrays for the pure-njit
        # parallel driver. The scalar setup (HydrogenLineProfileEvaluator.
        # line_setup) depends only on (lower_level, upper_level) and is cached,
        # so this is O(distinct H lines) and runs ONCE (outside any prange).
        # Every field the compiled hydrogen deposit needs is flattened onto a
        # per-line array indexed by transition-line index; validity mirrors the
        # scalar branch exactly (selector_index==0 or line_setup None -> skip).
        n_lines_all = int(transition_lines.line_count)
        h_lower_level_line = np.zeros(n_lines_all, dtype=np.int64)
        h_upper_level_line = np.zeros(n_lines_all, dtype=np.int64)
        h_line_frequency_hz_line = np.zeros(n_lines_all, dtype=np.float64)
        h_line_wavelength_a_line = np.zeros(n_lines_all, dtype=np.float64)
        h_beta_scale_line = np.zeros(n_lines_all, dtype=np.float64)
        h_stark_c1_factor_line = np.zeros(n_lines_all, dtype=np.float64)
        h_stark_c2_factor_line = np.zeros(n_lines_all, dtype=np.float64)
        h_radiative_width_line = np.zeros(n_lines_all, dtype=np.float64)
        h_resonance_width_line = np.zeros(n_lines_all, dtype=np.float64)
        h_van_der_waals_width_line = np.zeros(n_lines_all, dtype=np.float64)
        h_stark_width_line = np.zeros(n_lines_all, dtype=np.float64)
        hydrogen_low_density_impact_numerator_by_line = np.zeros(
            n_lines_all, dtype=np.float64
        )
        hydrogen_impact_electron_density_threshold_cm3_by_line = np.zeros(
            n_lines_all, dtype=np.float64
        )
        h_stark_component_count_line = np.zeros(n_lines_all, dtype=np.int64)
        h_valid_line = np.zeros(n_lines_all, dtype=np.int64)
        _h_setup_offsets = {}
        _h_setup_weights = {}
        _max_components = 1
        if hydrogen_profile_evaluator is not None:
            _line_types = np.asarray(transition_lines.line_type)
            _h_line_indices = np.nonzero(_line_types == -1)[0]
            for _li in _h_line_indices:
                _li = int(_li)
                lower_hydrogen_level = int(transition_lines.lower_hydrogen_level[_li])
                upper_hydrogen_level = int(transition_lines.upper_hydrogen_level[_li])
                if lower_hydrogen_level < 1 or lower_hydrogen_level > 100:
                    raise ValueError(
                        f"Hydrogen line has lower level {lower_hydrogen_level}, expected 1..100."
                    )
                if upper_hydrogen_level < 1 or upper_hydrogen_level > 100:
                    raise ValueError(
                        f"Hydrogen line has upper level {upper_hydrogen_level}, expected 1..100."
                    )
                selector_index = int(
                    transition_lines.hydrogen_continuum_selector_index[_li]
                )
                if selector_index == 0:
                    continue
                if selector_index < 1 or selector_index > continuum_selector.shape[0]:
                    raise ValueError(
                        f"Hydrogen line has selector index {selector_index}, "
                        f"expected 1..{continuum_selector.shape[0]}."
                    )
                line_setup = hydrogen_profile_evaluator.line_setup(
                    lower_hydrogen_level, upper_hydrogen_level
                )
                if line_setup is None:
                    continue
                h_valid_line[_li] = 1
                h_lower_level_line[_li] = int(line_setup.lower_level)
                h_upper_level_line[_li] = int(line_setup.upper_level)
                h_line_frequency_hz_line[_li] = float(line_setup.line_frequency_hz)
                h_line_wavelength_a_line[_li] = float(line_setup.line_wavelength_a)
                h_beta_scale_line[_li] = float(line_setup.beta_scale)
                h_stark_c1_factor_line[_li] = float(line_setup.stark_c1_factor)
                h_stark_c2_factor_line[_li] = float(line_setup.stark_c2_factor)
                h_radiative_width_line[_li] = float(line_setup.radiative_width)
                h_resonance_width_line[_li] = float(line_setup.resonance_width)
                h_van_der_waals_width_line[_li] = float(line_setup.van_der_waals_width)
                h_stark_width_line[_li] = float(line_setup.stark_width)
                hydrogen_low_density_impact_numerator_by_line[_li] = float(
                    line_setup.low_density_impact_numerator
                )
                hydrogen_impact_electron_density_threshold_cm3_by_line[_li] = float(
                    line_setup.impact_electron_density_threshold_cm3
                )
                _offsets = np.ascontiguousarray(
                    line_setup.stark_component_offsets_hz, dtype=np.float64
                )
                _weights = np.ascontiguousarray(
                    line_setup.stark_component_weights, dtype=np.float64
                )
                _h_setup_offsets[_li] = _offsets
                _h_setup_weights[_li] = _weights
                h_stark_component_count_line[_li] = int(_offsets.shape[0])
                if int(_offsets.shape[0]) > _max_components:
                    _max_components = int(_offsets.shape[0])
        h_stark_offsets_line = np.zeros(
            (n_lines_all, _max_components), dtype=np.float64
        )
        h_stark_weights_line = np.zeros(
            (n_lines_all, _max_components), dtype=np.float64
        )
        for _li, _offsets in _h_setup_offsets.items():
            _c = _offsets.shape[0]
            h_stark_offsets_line[_li, 0:_c] = _offsets
            h_stark_weights_line[_li, 0:_c] = _h_setup_weights[_li][0:_c]
        # continuum_selector column 0 (used by the hydrogen deposit) as a 1-D
        # contiguous array so the njit driver can index it by selector slot.
        continuum_selector_col0 = np.ascontiguousarray(
            kernel_selector_table[:, 0], dtype=np.float64
        )
        # special-record indices (everything that is not a normal 0/2/3 run)
        kernel_special_indices_i64 = np.ascontiguousarray(
            kernel_special_indices, dtype=np.int64
        )
        # partitioned populations + mass density for the type-1/merged sub-paths
        # (always present for real decks; size-0 placeholders when absent so the
        # njit kernel type-unifies).
        if partition_normalized_ion_stage_populations is None:
            partition_normalized_populations_kernel = np.zeros(
                (layer_count, 0), dtype=np.float64
            )
            has_partition_normalized_flag = 0
        else:
            partition_normalized_populations_kernel = np.ascontiguousarray(
                partition_normalized_ion_stage_populations, dtype=np.float64
            )
            has_partition_normalized_flag = 1
        if density is None:
            density_kernel = np.zeros(layer_count, dtype=np.float64)
        else:
            density_kernel = np.ascontiguousarray(density, dtype=np.float64)
        # size-0 hydrogen deposit tables when there are no hydrogen lines
        if hydrogen_profile_evaluator is None:
            _z1 = np.zeros(0, dtype=np.float64)
            _z2 = np.zeros((0, 0), dtype=np.float64)
            _z3 = np.zeros((0, 0, 0), dtype=np.float64)
            kernel_hydrogen_boltzmann_widths = _z2
            kernel_h_doppler = _z1
            kernel_h_field = _z1
            kernel_h_td_he = _z1
            kernel_h_td_h2 = _z1
            kernel_h_neutral = _z1
            kernel_h_ground = _z1
            kernel_h_ionized = _z1
            kernel_h_electrons = _z1
            hydrogen_low_density_impact_factor_by_layer = _z1
            hydrogen_high_density_impact_factor_by_layer = _z1
            kernel_h_c1 = _z1
            kernel_h_c2 = _z1
            kernel_h_gcon1 = _z1
            kernel_h_gcon2 = _z1
            kernel_h_pressure = _z1
            kernel_h2_cutoff = _z1
            kernel_h2plus_cutoff = _z1
            kernel_h_stark_prob = _z3
            kernel_h_pressure_grid = _z1
            kernel_h_beta_grid = _z1
            kernel_h_stark_c = _z2
            kernel_h_stark_d = _z2
            kernel_h_exp_integer = _z1
            kernel_h_exp_fraction = _z1
            kernel_h_exp_integral = _z1
        else:
            kernel_hydrogen_boltzmann_widths = np.ascontiguousarray(
                hydrogen_boltzmann_widths, dtype=np.float64
            )

    def _run_line_range(range_start, range_stop, out_absorption):
        # Process transition lines [range_start, range_stop) into out_absorption
        # with a freshly re-seeded, memoryless walk. depth_gate is scratch state
        # local to this range so concurrent ranges never share it.
        depth_gate = np.zeros(layer_count + 2, dtype=np.int32)
        continuum_column = 0
        center_index = max(0, int(wavelength_start_index) - 1)
        processed = 0
        skip_until = -1
        stop_all_lines = False
        for line_index in range(range_start, range_stop):
            if stop_all_lines or line_index < skip_until:
                continue
            if kernel_is_normal[line_index]:
                insert_at = int(np.searchsorted(kernel_special_indices, line_index))
                run_stop = (
                    int(kernel_special_indices[insert_at])
                    if insert_at < kernel_special_indices.size
                    else int(transition_lines.line_count)
                )
                # A normal run may not spill past this chunk's line range.
                if run_stop > range_stop:
                    run_stop = range_stop
                (
                    continuum_column,
                    center_index,
                    processed_delta,
                    run_stop_flag,
                    _stopped_at,
                ) = _accumulate_transition_run_compiled(
                    line_index,
                    run_stop,
                    kernel_line_type,
                    kernel_packed_wavelength,
                    kernel_vacuum_wavelength,
                    kernel_species_slot,
                    kernel_oscillator_strength,
                    kernel_lower_excitation,
                    kernel_radiative_damping,
                    kernel_stark_damping,
                    kernel_van_der_waals_damping,
                    kernel_selector_index,
                    kernel_selector_slot,
                    wavelength_grid,
                    bin_edges,
                    continuum_threshold,
                    hc_over_kt_cm,
                    electrons,
                    line_population_widths,
                    doppler,
                    neutral_collision_density,
                    kernel_selector_table,
                    kernel_dissolved_energy,
                    kernel_integer_step,
                    kernel_fractional_step,
                    kernel_gaussian,
                    kernel_first,
                    kernel_second,
                    depth_gate,
                    out_absorption,
                    continuum_column,
                    center_index,
                    int(wavelength_start_index),
                    int(stop_index),
                )
                processed += int(processed_delta)
                if run_stop_flag:
                    stop_all_lines = True
                    continue
                skip_until = run_stop
                continue
            line_type = int(transition_lines.line_type[line_index])
            if line_type == 2:
                continue

            vacuum_wavelength_nm = float(
                transition_lines.vacuum_wavelength_nm[line_index]
            )
            if vacuum_wavelength_nm > float(
                wavelength_grid[min(stop_index, wavelength_count) - 1]
            ):
                break

            packed_wavelength = int(
                transition_lines.packed_wavelength_index[line_index]
            )
            while continuum_column < bin_edges.size and packed_wavelength >= int(
                bin_edges[continuum_column]
            ):
                continuum_column += 1
            if continuum_column >= continuum_threshold.shape[1]:
                continue

            while center_index < wavelength_count and vacuum_wavelength_nm >= float(
                wavelength_grid[center_index]
            ):
                center_index += 1
            if center_index >= wavelength_count:
                break

            packed_species_slot = int(transition_lines.packed_species_slot[line_index])
            normal_transition = line_type in (0, 3)
            if normal_transition and (
                packed_species_slot < 1
                or packed_species_slot > line_population_widths.shape[1]
            ):
                processed += 1
                continue

            oscillator_strength = float(
                transition_lines.oscillator_strength[line_index]
            )
            lower_excitation_cm = float(
                transition_lines.lower_excitation_cm[line_index]
            )
            radiative_damping = float(transition_lines.radiative_damping[line_index])
            stark_damping = float(transition_lines.stark_damping[line_index])
            van_der_waals_damping = float(
                transition_lines.van_der_waals_damping[line_index]
            )
            selector_index = int(
                transition_lines.hydrogen_continuum_selector_index[line_index]
            )

            if line_type == -1:
                if (
                    hydrogen_boltzmann_widths is None
                    or hydrogen_profile_evaluator is None
                ):
                    raise ValueError("Hydrogen transition opacity was not initialized.")
                lower_hydrogen_level = int(
                    transition_lines.lower_hydrogen_level[line_index]
                )
                upper_hydrogen_level = int(
                    transition_lines.upper_hydrogen_level[line_index]
                )
                if lower_hydrogen_level < 1 or lower_hydrogen_level > 100:
                    raise ValueError(
                        f"Hydrogen line has lower level {lower_hydrogen_level}, expected 1..100."
                    )
                if upper_hydrogen_level < 1 or upper_hydrogen_level > 100:
                    raise ValueError(
                        f"Hydrogen line has upper level {upper_hydrogen_level}, expected 1..100."
                    )
                if selector_index == 0:
                    continue
                if selector_index < 1 or selector_index > continuum_selector.shape[0]:
                    raise ValueError(
                        f"Hydrogen line has selector index {selector_index}, "
                        f"expected 1..{continuum_selector.shape[0]}."
                    )
                line_setup = hydrogen_profile_evaluator.line_setup(
                    lower_hydrogen_level,
                    upper_hydrogen_level,
                )
                if line_setup is None:
                    continue
                scaled_oscillator_strength = np.float32(oscillator_strength)
                _hydrogen_line_deposit_compiled(
                    out_absorption,
                    center_index,
                    vacuum_wavelength_nm,
                    float(scaled_oscillator_strength),
                    lower_hydrogen_level,
                    continuum_column,
                    continuum_threshold,
                    wavelength_grid,
                    hydrogen_boltzmann_widths,
                    float(continuum_selector[selector_index - 1, 0]),
                    kernel_dissolved_energy,
                    int(line_setup.lower_level),
                    int(line_setup.upper_level),
                    float(line_setup.line_frequency_hz),
                    float(line_setup.line_wavelength_a),
                    float(line_setup.beta_scale),
                    float(line_setup.stark_c1_factor),
                    float(line_setup.stark_c2_factor),
                    float(line_setup.radiative_width),
                    float(line_setup.resonance_width),
                    float(line_setup.van_der_waals_width),
                    float(line_setup.stark_width),
                    float(line_setup.low_density_impact_numerator),
                    float(line_setup.impact_electron_density_threshold_cm3),
                    np.ascontiguousarray(
                        line_setup.stark_component_offsets_hz, dtype=np.float64
                    ),
                    np.ascontiguousarray(
                        line_setup.stark_component_weights, dtype=np.float64
                    ),
                    kernel_h_doppler,
                    kernel_h_field,
                    kernel_h_td_he,
                    kernel_h_td_h2,
                    kernel_h_neutral,
                    kernel_h_ground,
                    kernel_h_ionized,
                    kernel_h_electrons,
                    hydrogen_low_density_impact_factor_by_layer,
                    hydrogen_high_density_impact_factor_by_layer,
                    kernel_h_c1,
                    kernel_h_c2,
                    kernel_h_gcon1,
                    kernel_h_gcon2,
                    kernel_h_pressure,
                    kernel_h2_cutoff,
                    kernel_h2plus_cutoff,
                    kernel_h_stark_prob,
                    kernel_h_pressure_grid,
                    kernel_h_beta_grid,
                    kernel_h_stark_c,
                    kernel_h_stark_d,
                    kernel_h_exp_integer,
                    kernel_h_exp_fraction,
                    kernel_h_exp_integral,
                )
                processed += 1
                continue

            normal_selector_index = selector_index
            if normal_selector_index > 10:
                normal_selector_index = 0

            if line_type == 1:
                if (
                    partition_normalized_ion_stage_populations is None
                    or density is None
                ):
                    raise ValueError(
                        "Autoionizing transition opacity requires partitioned populations "
                        "and mass density."
                    )
                if (
                    packed_species_slot < 1
                    or packed_species_slot
                    > partition_normalized_ion_stage_populations.shape[1]
                ):
                    continue
                line_frequency_hz = _LIGHT_SPEED_NM_PER_SECOND / max(
                    vacuum_wavelength_nm, 1.0e-300
                )
                shore_asymmetry = stark_damping
                shore_width = van_der_waals_damping
                if shore_width == 0.0 or radiative_damping == 0.0:
                    continue
                for depth_index in range(layer_count):
                    center_absorption = (
                        shore_width
                        * oscillator_strength
                        * float(
                            partition_normalized_ion_stage_populations[
                                depth_index, packed_species_slot - 1
                            ]
                        )
                        / max(float(density[depth_index]), 1.0e-300)
                    )
                    if center_absorption < float(
                        continuum_threshold[depth_index, continuum_column]
                    ):
                        continue
                    center_absorption *= fast_exponential_lookup(
                        lower_excitation_cm * float(hc_over_kt_cm[depth_index]),
                        exponential_tables,
                    )
                    if center_absorption < float(
                        continuum_threshold[depth_index, continuum_column]
                    ):
                        continue
                    for wavelength_index in range(
                        center_index, min(center_index + 2001, wavelength_count)
                    ):
                        reduced_frequency = (
                            2.0
                            * (
                                _LIGHT_SPEED_NM_PER_SECOND
                                / max(
                                    float(wavelength_grid[wavelength_index]), 1.0e-300
                                )
                                - line_frequency_hz
                            )
                            / radiative_damping
                        )
                        contribution = (
                            center_absorption
                            * (shore_asymmetry * reduced_frequency + shore_width)
                            / (reduced_frequency * reduced_frequency + 1.0)
                            / shore_width
                        )
                        out_absorption[depth_index, wavelength_index] += contribution
                        if contribution < float(
                            continuum_threshold[depth_index, continuum_column]
                        ):
                            break
                    for red_step in range(1, 2001):
                        wavelength_index = max(center_index - red_step, 0)
                        reduced_frequency = (
                            2.0
                            * (
                                _LIGHT_SPEED_NM_PER_SECOND
                                / max(
                                    float(wavelength_grid[wavelength_index]), 1.0e-300
                                )
                                - line_frequency_hz
                            )
                            / radiative_damping
                        )
                        contribution = (
                            center_absorption
                            * (shore_asymmetry * reduced_frequency + shore_width)
                            / (reduced_frequency * reduced_frequency + 1.0)
                            / shore_width
                        )
                        out_absorption[depth_index, wavelength_index] += contribution
                        if contribution < float(
                            continuum_threshold[depth_index, continuum_column]
                        ):
                            break
                processed += 1
                continue

            if not normal_transition:
                if (
                    partition_normalized_ion_stage_populations is None
                    or density is None
                ):
                    raise ValueError(
                        "Merged-continuum transition opacity requires partitioned populations "
                        "and mass density."
                    )
                if (
                    packed_species_slot < 1
                    or packed_species_slot
                    > partition_normalized_ion_stage_populations.shape[1]
                ):
                    continue
                effective_charge = 2.0 if packed_species_slot == 4 else 1.0
                last_level = float(line_type)
                if last_level == 0.0:
                    continue
                shifted_wavelength = 1.0e7 / (
                    1.0e7 / vacuum_wavelength_nm
                    - 109737.312
                    * effective_charge
                    * effective_charge
                    / (last_level * last_level)
                )
                for depth_index in range(layer_count):
                    merged_wavelength = 1.0e7 / (
                        1.0e7 / vacuum_wavelength_nm
                        - float(hydrogen_level_dissolution_wavenumber_cm[depth_index])
                        * effective_charge
                        * effective_charge
                    )
                    maximum_wavelength = max(merged_wavelength, shifted_wavelength)
                    continuum_absorption = (
                        oscillator_strength
                        * float(
                            partition_normalized_ion_stage_populations[
                                depth_index, packed_species_slot - 1
                            ]
                        )
                        * fast_exponential_lookup(
                            lower_excitation_cm * float(hc_over_kt_cm[depth_index]),
                            exponential_tables,
                        )
                        / max(float(density[depth_index]), 1.0e-300)
                    )
                    for wavelength_index in range(
                        center_index, min(center_index + 1001, wavelength_count)
                    ):
                        if maximum_wavelength < float(
                            wavelength_grid[wavelength_index]
                        ):
                            break
                        out_absorption[depth_index, wavelength_index] += (
                            continuum_absorption
                        )
                processed += 1
                continue

        return processed

    line_count = int(transition_lines.line_count)
    use_parallel = has_transition_lines and _NUMBA_AVAILABLE
    chunk_count = _line_opacity_chunk_count() if use_parallel else 1
    chunk_count = max(1, min(chunk_count, line_count))
    if not use_parallel or chunk_count <= 1 or line_count <= 1:
        # Serial reference path: the pure-Python driver over the whole range.
        processed_lines = _run_line_range(
            0, line_count, line_mass_absorption_coefficient
        )
    else:
        # Pure-njit(parallel=True)+prange over contiguous line-chunks. Each
        # chunk re-seeds the memoryless continuum-column/center-index walk in a
        # private float32 buffer; buffers are summed in chunk order (the same
        # deterministic float32 regrouping the selected-line opacity path uses).
        # No Python-level thread pool -> no GIL cap on the per-chunk dispatch.
        line_mass_absorption_coefficient, processed_lines = (
            _accumulate_transition_line_opacity_parallel(
                int(chunk_count),
                int(line_count),
                line_mass_absorption_coefficient,
                kernel_line_type,
                kernel_special_indices_i64,
                kernel_packed_wavelength,
                kernel_vacuum_wavelength,
                kernel_species_slot,
                kernel_oscillator_strength,
                kernel_lower_excitation,
                kernel_radiative_damping,
                kernel_stark_damping,
                kernel_van_der_waals_damping,
                kernel_selector_index,
                kernel_selector_slot,
                wavelength_grid,
                bin_edges,
                continuum_threshold,
                hc_over_kt_cm,
                electrons,
                line_population_widths,
                doppler,
                neutral_collision_density,
                kernel_selector_table,
                kernel_dissolved_energy,
                kernel_integer_step,
                kernel_fractional_step,
                kernel_gaussian,
                kernel_first,
                kernel_second,
                int(wavelength_start_index),
                int(stop_index),
                partition_normalized_populations_kernel,
                density_kernel,
                has_partition_normalized_flag,
                kernel_hydrogen_boltzmann_widths,
                continuum_selector_col0,
                h_lower_level_line,
                h_upper_level_line,
                h_line_frequency_hz_line,
                h_line_wavelength_a_line,
                h_beta_scale_line,
                h_stark_c1_factor_line,
                h_stark_c2_factor_line,
                h_radiative_width_line,
                h_resonance_width_line,
                h_van_der_waals_width_line,
                h_stark_width_line,
                hydrogen_low_density_impact_numerator_by_line,
                hydrogen_impact_electron_density_threshold_cm3_by_line,
                h_stark_offsets_line,
                h_stark_weights_line,
                h_stark_component_count_line,
                h_valid_line,
                kernel_h_doppler,
                kernel_h_field,
                kernel_h_td_he,
                kernel_h_td_h2,
                kernel_h_neutral,
                kernel_h_ground,
                kernel_h_ionized,
                kernel_h_electrons,
                hydrogen_low_density_impact_factor_by_layer,
                hydrogen_high_density_impact_factor_by_layer,
                kernel_h_c1,
                kernel_h_c2,
                kernel_h_gcon1,
                kernel_h_gcon2,
                kernel_h_pressure,
                kernel_h2_cutoff,
                kernel_h2plus_cutoff,
                kernel_h_stark_prob,
                kernel_h_pressure_grid,
                kernel_h_beta_grid,
                kernel_h_stark_c,
                kernel_h_stark_d,
                kernel_h_exp_integer,
                kernel_h_exp_fraction,
                kernel_h_exp_integral,
            )
        )
        processed_lines = int(processed_lines)

    return LineOpacityState(
        line_mass_absorption_coefficient=line_mass_absorption_coefficient,
        selected_line_count=processed_lines,
    )
