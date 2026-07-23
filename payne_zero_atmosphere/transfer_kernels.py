"""Compiled per-frequency transfer accumulation loop.

This is the sole production transfer path (the largest single CPU cost in the
solve); numba is a hard requirement. The operator matvecs (``coefj @ grid`` /
``coefh @ grid``) reduce under numba's BLAS, so flux and mean intensity differ
from a plain-numpy reduction at the float32 ulp (~1e-7 relative), ~4 orders
below the product gate's material tolerance.
"""

from __future__ import annotations

import os

import numpy as np

from ._numba_cache import configure_numba_cache

configure_numba_cache()
try:
    import numba
except ImportError as exc:  # pragma: no cover - numba is a hard requirement
    raise ImportError(
        "numba is required: the compiled transfer kernel is the sole "
        "production path for payne_zero_atmosphere."
    ) from exc

_NUMBA_AVAILABLE = True


accumulate_transfer_range_compiled = None
accumulate_transfer_range_parallel = None


def transfer_chunk_count() -> int:
    """Frequency-chunk count for the parallel transfer accumulation: one chunk
    per numba thread (= ``NUMBA_NUM_THREADS``, which defaults to the CPU count).

    Each chunk runs the serial per-frequency kernel on a contiguous frequency
    slice into its own private accumulator buffers; the buffers are summed in
    chunk order. Every frequency's deposit is byte-for-byte the serial kernel's
    (frequencies are independent), so the only difference vs a serial pass is
    the float32 reduction regrouping (~ulp per cell), deterministic for a fixed
    chunk count. Mirrors the line-opacity chunking in line_opacity.py.
    """

    try:
        return max(1, int(numba.get_num_threads()))
    except Exception:  # pragma: no cover - defensive
        return max(1, (os.cpu_count() or 1))


if _NUMBA_AVAILABLE:
    _njit = numba.njit(cache=True, nogil=True)

    @_njit
    def _parabolic_coefficients_compiled(value, coordinate):
        count = value.shape[0]
        constant = np.zeros(count, dtype=np.float64)
        linear = np.zeros(count, dtype=np.float64)
        quadratic = np.zeros(count, dtype=np.float64)
        if count == 0:
            return constant, linear, quadratic
        if count == 1:
            constant[0] = value[0]
            return constant, linear, quadratic

        denominator = coordinate[1] - coordinate[0]
        linear[0] = (value[1] - value[0]) / denominator if denominator != 0.0 else 0.0
        constant[0] = value[0] - coordinate[0] * linear[0]

        last = count - 1
        denominator = coordinate[count - 1] - coordinate[last - 1]
        linear[count - 1] = (
            (value[count - 1] - value[last - 1]) / denominator
            if denominator != 0.0
            else 0.0
        )
        constant[count - 1] = (
            value[count - 1] - coordinate[count - 1] * linear[count - 1]
        )
        if count == 2:
            return constant, linear, quadratic

        for index in range(1, last):
            left = index - 1
            denominator = coordinate[index] - coordinate[left]
            slope = (
                (value[index] - value[left]) / denominator
                if denominator != 0.0
                else 0.0
            )
            high_width = (coordinate[index + 1] - coordinate[index]) * (
                coordinate[index + 1] - coordinate[left]
            )
            first_term = value[index + 1] / high_width if high_width != 0.0 else 0.0
            wide_width = coordinate[index + 1] - coordinate[left]
            next_width = coordinate[index + 1] - coordinate[index]
            local_width = coordinate[index] - coordinate[left]
            second_term = 0.0
            if local_width != 0.0:
                left_term = value[left] / wide_width if wide_width != 0.0 else 0.0
                right_term = value[index] / next_width if next_width != 0.0 else 0.0
                second_term = (left_term - right_term) / local_width
            quadratic[index] = first_term + second_term
            linear[index] = (
                slope - (coordinate[index] + coordinate[left]) * quadratic[index]
            )
            constant[index] = (
                value[left]
                - coordinate[left] * slope
                + coordinate[index] * coordinate[left] * quadratic[index]
            )

        quadratic[1] = 0.0
        denominator = coordinate[2] - coordinate[1]
        linear[1] = (value[2] - value[1]) / denominator if denominator != 0.0 else 0.0
        constant[1] = value[1] - coordinate[1] * linear[1]

        if count > 3:
            quadratic[2] = 0.0
            denominator = coordinate[3] - coordinate[2]
            linear[2] = (
                (value[3] - value[2]) / denominator if denominator != 0.0 else 0.0
            )
            constant[2] = value[2] - coordinate[2] * linear[2]

        for index in range(1, last):
            if quadratic[index] == 0.0:
                continue
            next_index = min(index + 1, count - 1)
            denominator = abs(quadratic[next_index]) + abs(quadratic[index])
            weight = (
                abs(quadratic[next_index]) / denominator if denominator > 0.0 else 0.0
            )
            constant[index] = constant[next_index] + weight * (
                constant[index] - constant[next_index]
            )
            linear[index] = linear[next_index] + weight * (
                linear[index] - linear[next_index]
            )
            quadratic[index] = quadratic[next_index] + weight * (
                quadratic[index] - quadratic[next_index]
            )

        constant[last - 1] = constant[count - 1]
        linear[last - 1] = linear[count - 1]
        quadratic[last - 1] = quadratic[count - 1]
        return constant, linear, quadratic

    @_njit
    def _integrate_on_depth_grid_compiled(coordinate, value, surface_value):
        integral = np.zeros(value.shape[0], dtype=np.float64)
        if value.shape[0] == 0:
            return integral
        constant, linear, quadratic = _parabolic_coefficients_compiled(
            value, coordinate
        )
        integral[0] = surface_value
        if value.shape[0] == 1:
            return integral
        for index in range(value.shape[0] - 1):
            dx = coordinate[index + 1] - coordinate[index]
            segment = constant[index] + 0.5 * linear[index] * (
                coordinate[index + 1] + coordinate[index]
            )
            segment += (quadratic[index] / 3.0) * (
                (coordinate[index + 1] + coordinate[index]) * coordinate[index + 1]
                + coordinate[index] * coordinate[index]
            )
            integral[index + 1] = integral[index] + segment * dx
        return integral

    @_njit
    def _differentiate_on_depth_grid_compiled(coordinate, value):
        derivative = np.zeros(value.shape[0], dtype=np.float64)
        if value.shape[0] < 2:
            return derivative

        low_width = coordinate[1] - coordinate[0]
        high_width = coordinate[value.shape[0] - 1] - coordinate[value.shape[0] - 2]
        derivative[0] = (value[1] - value[0]) / low_width if low_width != 0.0 else 0.0
        derivative[value.shape[0] - 1] = (
            (value[value.shape[0] - 1] - value[value.shape[0] - 2]) / high_width
            if high_width != 0.0
            else 0.0
        )
        if value.shape[0] == 2:
            return derivative

        direction = (
            abs(coordinate[1] - coordinate[0]) / low_width if low_width != 0.0 else 1.0
        )
        for index in range(1, value.shape[0] - 1):
            scale = max(abs(value[index - 1]), abs(value[index]), abs(value[index + 1]))
            scale = (
                scale / abs(coordinate[index]) if coordinate[index] != 0.0 else scale
            )
            if scale == 0.0:
                scale = 1.0
            right_width = coordinate[index + 1] - coordinate[index]
            left_width = coordinate[index] - coordinate[index - 1]
            if right_width == 0.0 or left_width == 0.0:
                derivative[index] = 0.0
                continue
            right_slope = (value[index + 1] - value[index]) / right_width / scale
            left_slope = (value[index] - value[index - 1]) / left_width / scale
            right_denominator = (
                direction * np.sqrt(1.0 + right_slope * right_slope) + 1.0
            )
            left_denominator = direction * np.sqrt(1.0 + left_slope * left_slope) + 1.0
            if abs(right_denominator) < 1.0e-30 or abs(left_denominator) < 1.0e-30:
                derivative[index] = 0.5 * (left_slope + right_slope) * scale
                continue
            right_tangent = right_slope / right_denominator
            left_tangent = left_slope / left_denominator
            denominator = 1.0 - right_tangent * left_tangent
            if abs(denominator) < 1.0e-30:
                derivative[index] = 0.5 * (left_slope + right_slope) * scale
            else:
                derivative[index] = (right_tangent + left_tangent) / denominator * scale
        return derivative

    @_njit
    def _remap_to_grid_compiled(old_grid, old_values, new_grid, remapped):
        if old_grid.shape[0] == 0 or new_grid.shape[0] == 0:
            for i in range(new_grid.shape[0]):
                remapped[i] = 0.0
            return 0

        source_count = old_grid.shape[0]
        grid_1based = np.empty(source_count + 1, dtype=np.float64)
        value_1based = np.empty(source_count + 1, dtype=np.float64)
        grid_1based[0] = 0.0
        value_1based[0] = 0.0
        for i in range(source_count):
            grid_1based[i + 1] = old_grid[i]
            value_1based[i + 1] = old_values[i]

        source_index = 2
        previous_source_index = 0
        forward_quadratic = 0.0
        forward_linear = 0.0
        forward_constant = 0.0
        backward_quadratic = 0.0
        backward_linear = 0.0
        backward_constant = 0.0
        constant = 0.0
        linear = 0.0
        quadratic = 0.0

        for target_index in range(new_grid.shape[0]):
            target_value = new_grid[target_index]
            while True:
                if target_value < grid_1based[source_index]:
                    if source_index == previous_source_index:
                        break
                    if source_index == 2 or source_index == 3:
                        source_index = min(source_count, source_index)
                        quadratic = 0.0
                        width = (
                            grid_1based[source_index] - grid_1based[source_index - 1]
                        )
                        linear = (
                            (
                                value_1based[source_index]
                                - value_1based[source_index - 1]
                            )
                            / width
                            if width != 0.0
                            else 0.0
                        )
                        constant = (
                            value_1based[source_index]
                            - grid_1based[source_index] * linear
                        )
                        previous_source_index = source_index
                        break

                    left_index = source_index - 1
                    if (
                        source_index > previous_source_index + 1
                        or source_index == 3
                        or source_index == 4
                    ):
                        width = grid_1based[left_index] - grid_1based[source_index - 2]
                        right_width = (
                            grid_1based[source_index] - grid_1based[left_index]
                        )
                        wide_width = (
                            grid_1based[source_index] - grid_1based[source_index - 2]
                        )
                        slope = (
                            (value_1based[left_index] - value_1based[source_index - 2])
                            / width
                            if width != 0.0
                            else 0.0
                        )
                        first_term = (
                            value_1based[source_index] / (right_width * wide_width)
                            if right_width != 0.0 and wide_width != 0.0
                            else 0.0
                        )
                        left_term = (
                            value_1based[source_index - 2] / wide_width
                            if wide_width != 0.0
                            else 0.0
                        )
                        right_term = (
                            value_1based[left_index] / right_width
                            if right_width != 0.0
                            else 0.0
                        )
                        second_term = (
                            (left_term - right_term) / width if width != 0.0 else 0.0
                        )
                        backward_quadratic = first_term + second_term
                        backward_linear = (
                            slope
                            - (grid_1based[left_index] + grid_1based[source_index - 2])
                            * backward_quadratic
                        )
                        backward_constant = (
                            value_1based[source_index - 2]
                            - grid_1based[source_index - 2] * slope
                            + grid_1based[left_index]
                            * grid_1based[source_index - 2]
                            * backward_quadratic
                        )
                    else:
                        backward_quadratic = forward_quadratic
                        backward_linear = forward_linear
                        backward_constant = forward_constant

                    if source_index >= source_count:
                        quadratic = backward_quadratic
                        linear = backward_linear
                        constant = backward_constant
                        previous_source_index = source_index
                        break

                    width = grid_1based[source_index] - grid_1based[left_index]
                    right_width = (
                        grid_1based[source_index + 1] - grid_1based[source_index]
                    )
                    wide_width = grid_1based[source_index + 1] - grid_1based[left_index]
                    slope = (
                        (value_1based[source_index] - value_1based[left_index]) / width
                        if width != 0.0
                        else 0.0
                    )
                    first_term = (
                        value_1based[source_index + 1] / (right_width * wide_width)
                        if right_width != 0.0 and wide_width != 0.0
                        else 0.0
                    )
                    left_term = (
                        value_1based[left_index] / wide_width
                        if wide_width != 0.0
                        else 0.0
                    )
                    right_term = (
                        value_1based[source_index] / right_width
                        if right_width != 0.0
                        else 0.0
                    )
                    second_term = (
                        (left_term - right_term) / width if width != 0.0 else 0.0
                    )
                    forward_quadratic = first_term + second_term
                    forward_linear = (
                        slope
                        - (grid_1based[source_index] + grid_1based[left_index])
                        * forward_quadratic
                    )
                    forward_constant = (
                        value_1based[left_index]
                        - grid_1based[left_index] * slope
                        + grid_1based[source_index]
                        * grid_1based[left_index]
                        * forward_quadratic
                    )
                    weight = 0.0
                    if abs(forward_quadratic) != 0.0:
                        weight = abs(forward_quadratic) / (
                            abs(forward_quadratic) + abs(backward_quadratic)
                        )
                    constant = forward_constant + weight * (
                        backward_constant - forward_constant
                    )
                    linear = forward_linear + weight * (
                        backward_linear - forward_linear
                    )
                    quadratic = forward_quadratic + weight * (
                        backward_quadratic - forward_quadratic
                    )
                    previous_source_index = source_index
                    break

                source_index += 1
                if source_index > source_count:
                    source_index = min(source_count, source_index)
                    quadratic = 0.0
                    width = grid_1based[source_index] - grid_1based[source_index - 1]
                    linear = (
                        (value_1based[source_index] - value_1based[source_index - 1])
                        / width
                        if width != 0.0
                        else 0.0
                    )
                    constant = (
                        value_1based[source_index] - grid_1based[source_index] * linear
                    )
                    previous_source_index = source_index
                    break

            remapped[target_index] = (
                constant + (linear + quadratic * target_value) * target_value
            )

        return max(previous_source_index - 1, 0)

    @_njit
    def _exponential_integral_compiled(order, argument):
        a0 = -44178.5471728217
        a1 = 57721.7247139444
        a2 = 9938.31388962037
        a3 = 1842.11088668
        a4 = 101.093806161906
        a5 = 5.03416184097568
        b0 = 76537.3323337614
        b1 = 32597.1881290275
        b2 = 6106.10794245759
        b3 = 635.419418378382
        b4 = 37.2298352833327
        c0 = 4.65627107975096e-7
        c1 = 0.999979577051595
        c2 = 9.04161556946329
        c3 = 24.3784088791317
        c4 = 23.0192559391333
        c5 = 6.90522522784444
        c6 = 0.430967839469389
        d1 = 10.0411643829054
        d2 = 32.4264210695138
        d3 = 41.2807841891424
        d4 = 20.4494785013794
        d5 = 3.31909213593302
        d6 = 0.103400130404874
        e0 = -0.999999999998447
        e1 = -26.6271060431811
        e2 = -241.055827097015
        e3 = -895.927957772937
        e4 = -1298.85688746484
        e5 = -545.374158883133
        e6 = -5.66575206533869
        f1 = 28.6271060422192
        f2 = 292.310039388533
        f3 = 1332.78537748257
        f4 = 2777.61949509163
        f5 = 2404.01713225909
        f6 = 631.6574832808

        x = argument
        if x <= 0.0:
            first_order = 0.0
        else:
            exponential = np.exp(-x)
            if x > 4.0:
                first_order = (
                    exponential
                    + exponential
                    * (
                        e0
                        + (e1 + (e2 + (e3 + (e4 + (e5 + e6 / x) / x) / x) / x) / x) / x
                    )
                    / (x + f1 + (f2 + (f3 + (f4 + (f5 + f6 / x) / x) / x) / x) / x)
                ) / x
            elif x > 1.0:
                first_order = (
                    exponential
                    * (
                        c6
                        + (c5 + (c4 + (c3 + (c2 + (c1 + c0 * x) * x) * x) * x) * x) * x
                    )
                    / (d6 + (d5 + (d4 + (d3 + (d2 + (d1 + x) * x) * x) * x) * x) * x)
                )
            else:
                first_order = (
                    a0 + (a1 + (a2 + (a3 + (a4 + a5 * x) * x) * x) * x) * x
                ) / (b0 + (b1 + (b2 + (b3 + (b4 + x) * x) * x) * x) * x) - np.log(x)
        value = first_order
        for index in range(1, max(order, 1)):
            value = (np.exp(-x) - x * value) / float(index)
        return value

    @_njit
    def _transfer_moments_compiled(
        continuum_absorption,
        continuum_source,
        line_mass_absorption_coefficient,
        line_source,
        continuum_scattering,
        column_mass,
        planck,
        transfer_grid,
        mean_intensity_operator,
        eddington_flux_operator,
        second_moment_weights,
        # outputs (preallocated, layer_count)
        optical_depth_out,
        source_out,
        eddington_flux_out,
        mean_intensity_out,
        mean_intensity_minus_source_out,
        total_opacity_out,
        scattering_fraction_out,
    ):
        """Transfer moments for the scattering-on branch.

        Returns (surface_second_moment, mapped_layer_count).
        """

        grid_count = transfer_grid.shape[0]
        layer_count = column_mass.shape[0]

        thermal_source = np.empty(layer_count, dtype=np.float64)
        for i in range(layer_count):
            total = (
                continuum_absorption[i]
                + line_mass_absorption_coefficient[i]
                + continuum_scattering[i]
                + 0.0
            )
            total_opacity_out[i] = max(total, 1.0e-300)
            scattering_fraction_out[i] = (
                continuum_scattering[i] + 0.0
            ) / total_opacity_out[i]
            thermal_absorption = (
                continuum_absorption[i] + line_mass_absorption_coefficient[i]
            )
            thermal_source[i] = planck[i]
            if thermal_absorption > 0.0:
                thermal_source[i] = (
                    continuum_absorption[i] * continuum_source[i]
                    + line_mass_absorption_coefficient[i] * line_source[i]
                ) / thermal_absorption

        surface_value = total_opacity_out[0] * column_mass[0]
        integ = _integrate_on_depth_grid_compiled(
            column_mass, total_opacity_out, surface_value
        )
        for i in range(layer_count):
            optical_depth_out[i] = integ[i]
            source_out[i] = 0.0
            eddington_flux_out[i] = 0.0
            mean_intensity_out[i] = 0.0
            mean_intensity_minus_source_out[i] = 0.0

        transfer_source_grid = np.zeros(grid_count, dtype=np.float32)
        mapped_layer_count = 0

        if optical_depth_out[0] > transfer_grid[grid_count - 1]:
            mapped_layer_count = 1
        else:
            grid_work = np.empty(grid_count, dtype=np.float64)
            mapped_layer_count = _remap_to_grid_compiled(
                optical_depth_out, thermal_source, transfer_grid, grid_work
            )
            thermal_source_grid = np.empty(grid_count, dtype=np.float32)
            for i in range(grid_count):
                thermal_source_grid[i] = max(
                    np.float32(grid_work[i]), np.float32(1.0e-38)
                )
            mapped_layer_count = _remap_to_grid_compiled(
                optical_depth_out, scattering_fraction_out, transfer_grid, grid_work
            )
            scattering_grid = np.empty(grid_count, dtype=np.float32)
            for i in range(grid_count):
                scattering_grid[i] = max(np.float32(grid_work[i]), np.float32(0.0))
            surface_thermal = max(thermal_source[0], 1.0e-38)
            surface_scattering = max(scattering_fraction_out[0], 0.0)
            for i in range(grid_count):
                if transfer_grid[i] < optical_depth_out[0]:
                    thermal_source_grid[i] = np.float32(surface_thermal)
                    scattering_grid[i] = np.float32(surface_scattering)
            for i in range(grid_count):
                transfer_source_grid[i] = thermal_source_grid[i]
            diagonal = np.empty(grid_count, dtype=np.float32)
            thermal_term_grid = np.empty(grid_count, dtype=np.float32)
            for i in range(grid_count):
                diagonal[i] = (
                    np.float32(1.0) - scattering_grid[i] * mean_intensity_operator[i, i]
                )
                thermal_term_grid[i] = (
                    np.float32(1.0) - scattering_grid[i]
                ) * thermal_source_grid[i]

            for _ in range(grid_count):
                needs_another_iteration = False
                for reverse_index in range(grid_count):
                    grid_index = grid_count - 1 - reverse_index
                    mean_source = np.float32(
                        np.dot(
                            mean_intensity_operator[grid_index, :], transfer_source_grid
                        )
                    )
                    numerator = np.float32(
                        mean_source * scattering_grid[grid_index]
                        + thermal_term_grid[grid_index]
                        - transfer_source_grid[grid_index]
                    )
                    denominator = diagonal[grid_index]
                    if abs(float(denominator)) < 1.0e-37:
                        if float(denominator) >= 0.0:
                            denominator = np.float32(1.0e-37)
                        else:
                            denominator = np.float32(-1.0e-37)
                    source_correction = np.float32(numerator / denominator)
                    source_base = transfer_source_grid[grid_index]
                    if abs(float(source_base)) < 1.0e-37:
                        if float(source_base) >= 0.0:
                            source_base = np.float32(1.0e-37)
                        else:
                            source_base = np.float32(-1.0e-37)
                    relative_error = np.float32(
                        abs(float(source_correction / source_base))
                    )
                    if relative_error > np.float32(1.0e-5):
                        needs_another_iteration = True
                    updated = np.float32(
                        transfer_source_grid[grid_index] + source_correction
                    )
                    transfer_source_grid[grid_index] = np.float32(
                        max(float(updated), 1.0e-37)
                    )
                if not needs_another_iteration:
                    break

            head_count = (
                mapped_layer_count if mapped_layer_count < layer_count else layer_count
            )
            source_grid_f64 = np.empty(grid_count, dtype=np.float64)
            for i in range(grid_count):
                source_grid_f64[i] = float(transfer_source_grid[i])
            head_work = np.empty(head_count, dtype=np.float64)
            _remap_to_grid_compiled(
                transfer_grid,
                source_grid_f64,
                optical_depth_out[:head_count],
                head_work,
            )
            for i in range(head_count):
                source_out[i] = head_work[i]

        if mapped_layer_count == layer_count:
            mean_source_vec = np.dot(mean_intensity_operator, transfer_source_grid)
            flux_vec = np.dot(eddington_flux_operator, transfer_source_grid)
            grid_mms = np.empty(grid_count, dtype=np.float64)
            grid_flux = np.empty(grid_count, dtype=np.float64)
            for i in range(grid_count):
                grid_mms[i] = float(
                    np.float32(-transfer_source_grid[i] + mean_source_vec[i])
                )
                grid_flux[i] = float(flux_vec[i])
            head_work = np.empty(mapped_layer_count, dtype=np.float64)
            _remap_to_grid_compiled(
                transfer_grid,
                grid_mms,
                optical_depth_out[:mapped_layer_count],
                head_work,
            )
            for i in range(mapped_layer_count):
                mean_intensity_minus_source_out[i] = head_work[i]
            _remap_to_grid_compiled(
                transfer_grid,
                grid_flux,
                optical_depth_out[:mapped_layer_count],
                head_work,
            )
            for i in range(mapped_layer_count):
                eddington_flux_out[i] = head_work[i]
            for i in range(mapped_layer_count):
                value = mean_intensity_minus_source_out[i] + source_out[i]
                mean_intensity_out[i] = max(value, 1.0e-38)
            surface_second_moment = float(
                np.dot(second_moment_weights, transfer_source_grid)
            )
            return surface_second_moment, mapped_layer_count

        first_deep_layer = mapped_layer_count + 1
        if mapped_layer_count == 1:
            first_deep_layer = 1
        for i in range(first_deep_layer - 1, layer_count):
            source_out[i] = thermal_source[i]
        derivative_start = max(mapped_layer_count - 1, 1)
        derivative_start_index = derivative_start - 1
        mean_start_index = mapped_layer_count - 1

        for _ in range(grid_count):
            invalid_source = False
            for i in range(derivative_start_index, layer_count):
                if source_out[i] <= 0.0:
                    invalid_source = True
                    break
            if invalid_source:
                for i in range(derivative_start_index, layer_count):
                    thermal_source[i] = planck[i]
                    source_out[i] = planck[i]
            deriv = _differentiate_on_depth_grid_compiled(
                optical_depth_out[derivative_start_index:],
                source_out[derivative_start_index:],
            )
            for i in range(derivative_start_index, layer_count):
                eddington_flux_out[i] = deriv[i - derivative_start_index] / 3.0
            negative_flux = False
            for i in range(derivative_start_index, layer_count):
                if eddington_flux_out[i] <= 0.0:
                    negative_flux = True
                    break
            if negative_flux:
                invalid_source = True
                for i in range(derivative_start_index, layer_count):
                    thermal_source[i] = planck[i]
                    source_out[i] = planck[i]
                deriv = _differentiate_on_depth_grid_compiled(
                    optical_depth_out[derivative_start_index:],
                    source_out[derivative_start_index:],
                )
                for i in range(derivative_start_index, layer_count):
                    eddington_flux_out[i] = deriv[i - derivative_start_index] / 3.0
            deriv2 = _differentiate_on_depth_grid_compiled(
                optical_depth_out[mean_start_index:],
                eddington_flux_out[mean_start_index:],
            )
            for i in range(mean_start_index, layer_count):
                mean_intensity_minus_source_out[i] = deriv2[i - mean_start_index]

            accumulated_error = 0.0
            for layer_index in range(first_deep_layer - 1, layer_count):
                if invalid_source:
                    mean_intensity_minus_source_out[layer_index] = 0.0
                mean_intensity_out[layer_index] = (
                    mean_intensity_minus_source_out[layer_index]
                    + source_out[layer_index]
                )
                updated_source = (
                    1.0 - scattering_fraction_out[layer_index]
                ) * thermal_source[layer_index] + scattering_fraction_out[
                    layer_index
                ] * mean_intensity_out[layer_index]
                accumulated_error += abs(
                    updated_source - source_out[layer_index]
                ) / max(
                    abs(updated_source),
                    1.0e-300,
                )
                source_out[layer_index] = updated_source
            if accumulated_error < 1.0e-5:
                break

        if mapped_layer_count == 1:
            surface_second_moment = mean_intensity_out[0] / 3.0
            return surface_second_moment, mapped_layer_count

        mean_source_vec = np.dot(mean_intensity_operator, transfer_source_grid)
        flux_vec = np.dot(eddington_flux_operator, transfer_source_grid)
        grid_mms = np.empty(grid_count, dtype=np.float64)
        grid_flux = np.empty(grid_count, dtype=np.float64)
        for i in range(grid_count):
            grid_mms[i] = float(
                np.float32(-transfer_source_grid[i] + mean_source_vec[i])
            )
            grid_flux[i] = float(flux_vec[i])
        head_work = np.empty(mapped_layer_count, dtype=np.float64)
        _remap_to_grid_compiled(
            transfer_grid, grid_mms, optical_depth_out[:mapped_layer_count], head_work
        )
        for i in range(mapped_layer_count):
            mean_intensity_minus_source_out[i] = head_work[i]
        _remap_to_grid_compiled(
            transfer_grid, grid_flux, optical_depth_out[:mapped_layer_count], head_work
        )
        for i in range(mapped_layer_count):
            eddington_flux_out[i] = head_work[i]
        for i in range(mapped_layer_count):
            safe_source = max(source_out[i], 1.0e-38)
            value = mean_intensity_minus_source_out[i] + safe_source
            mean_intensity_out[i] = max(value, 1.0e-38)
        surface_second_moment = float(
            np.dot(second_moment_weights, transfer_source_grid)
        )
        return surface_second_moment, mapped_layer_count

    @_njit
    def accumulate_transfer_range_compiled(
        range_start,
        range_stop,
        frequency_hz,
        frequency_weights,
        planck_all,
        stimulated_all,
        continuum_absorption_slab,
        continuum_scattering_slab,
        continuum_source_slab,
        line_mass_absorption_coefficient_slab,
        column_mass,
        h_over_kt,
        temperature,
        transfer_grid,
        mean_intensity_operator,
        eddington_flux_operator,
        second_moment_weights,
        target_integrated_eddington_flux,
        effective_temperature,
        frequency_count,
        rosseland_accumulator,
        radiation_energy_density,
        integrated_eddington_flux,
        radiative_acceleration,
        surface_radiation_pressure_constant,
        temperature_correction_heating_derivative,
        temperature_correction_mean_intensity_minus_source_integral,
        temperature_correction_integrated_eddington_flux,
        temperature_correction_diagonal_lambda,
    ):
        layer_count = column_mass.shape[0]
        continuum_absorption = np.empty(layer_count, dtype=np.float64)
        continuum_scattering = np.empty(layer_count, dtype=np.float64)
        continuum_source = np.empty(layer_count, dtype=np.float64)
        line_mass_absorption_coefficient = np.empty(layer_count, dtype=np.float64)
        optical_depth = np.empty(layer_count, dtype=np.float64)
        source = np.empty(layer_count, dtype=np.float64)
        monochromatic_eddington_flux = np.empty(layer_count, dtype=np.float64)
        mean_intensity = np.empty(layer_count, dtype=np.float64)
        mean_intensity_minus_source = np.empty(layer_count, dtype=np.float64)
        total_opacity = np.empty(layer_count, dtype=np.float64)
        scattering_fraction = np.empty(layer_count, dtype=np.float64)

        for frequency_index in range(range_start, range_stop):
            frequency = frequency_hz[frequency_index]
            frequency_weight = frequency_weights[frequency_index]
            planck = planck_all[frequency_index]
            stimulated = stimulated_all[frequency_index]
            for i in range(layer_count):
                continuum_absorption[i] = continuum_absorption_slab[i, frequency_index]
                continuum_scattering[i] = continuum_scattering_slab[i, frequency_index]
                continuum_source[i] = continuum_source_slab[i, frequency_index]
                line_mass_absorption_coefficient[i] = (
                    float(line_mass_absorption_coefficient_slab[i, frequency_index])
                    * stimulated[i]
                )

            surface_second_moment, _mapped = _transfer_moments_compiled(
                continuum_absorption,
                continuum_source,
                line_mass_absorption_coefficient,
                planck,
                continuum_scattering,
                column_mass,
                planck,
                transfer_grid,
                mean_intensity_operator,
                eddington_flux_operator,
                second_moment_weights,
                optical_depth,
                source,
                monochromatic_eddington_flux,
                mean_intensity,
                mean_intensity_minus_source,
                total_opacity,
                scattering_fraction,
            )

            any_negative_flux = False
            for i in range(layer_count):
                if monochromatic_eddington_flux[i] < 0.0:
                    any_negative_flux = True
                    break
            if any_negative_flux:
                for i in range(layer_count):
                    monochromatic_eddington_flux[i] = max(
                        monochromatic_eddington_flux[i], 1.0e-99
                    )
                    mean_intensity[i] = max(mean_intensity[i], 1.0e-99)
                    source[i] = max(source[i], 1.0e-99)

            # RADIAP mode 2
            for i in range(layer_count):
                radiation_energy_density[i] += mean_intensity[i] * frequency_weight
                integrated_eddington_flux[i] += (
                    monochromatic_eddington_flux[i] * frequency_weight
                )
                radiative_acceleration[i] += (
                    total_opacity[i]
                    * monochromatic_eddington_flux[i]
                    * frequency_weight
                )
            surface_radiation_pressure_constant[0] += (
                surface_second_moment * frequency_weight
            )

            # Frequency-integrated temperature-correction accumulation.
            opacity_derivative = _differentiate_on_depth_grid_compiled(
                column_mass, total_opacity
            )
            for i in range(layer_count):
                temperature_correction_heating_derivative[i] += (
                    opacity_derivative[i]
                    / max(total_opacity[i], 1.0e-300)
                    * monochromatic_eddington_flux[i]
                    * frequency_weight
                )
                temperature_correction_mean_intensity_minus_source_integral[i] += (
                    total_opacity[i] * mean_intensity_minus_source[i] * frequency_weight
                )
                temperature_correction_integrated_eddington_flux[i] += (
                    monochromatic_eddington_flux[i] * frequency_weight
                )

            next_term = 0.0
            for layer_index in range(layer_count):
                previous_term = next_term
                depth_step = 1.0e-10
                if layer_index != layer_count - 1:
                    depth_step = (
                        optical_depth[layer_index + 1] - optical_depth[layer_index]
                    )
                depth_step = max(1.0e-10, depth_step)
                if depth_step <= 0.01:
                    next_term = (
                        (0.922784335098467 - np.log(depth_step)) * depth_step / 4.0
                        + depth_step * depth_step / 12.0
                        - depth_step**3.0 / 96.0
                        + depth_step**4.0 / 720.0
                    )
                else:
                    exponential_integral = 0.0
                    if depth_step < 10.0:
                        exponential_integral = _exponential_integral_compiled(
                            3, depth_step
                        )
                    if (
                        effective_temperature <= 4250.0
                        and depth_step > 0.005
                        and depth_step < 0.02
                    ):
                        exponential_integral = 0.0
                    next_term = (
                        0.5 * (depth_step + exponential_integral - 0.5) / depth_step
                    )
                diagonal_mean_intensity = previous_term + next_term
                planck_derivative = (
                    planck[layer_index]
                    * frequency
                    * h_over_kt[layer_index]
                    / max(temperature[layer_index] * stimulated[layer_index], 1.0e-300)
                )
                if frequency_count == 1:
                    planck_derivative = (
                        target_integrated_eddington_flux
                        * 16.0
                        / max(
                            temperature[layer_index],
                            1.0e-300,
                        )
                    )
                temperature_correction_diagonal_lambda[layer_index] += (
                    total_opacity[layer_index]
                    * (diagonal_mean_intensity - 1.0)
                    / max(
                        1.0
                        - scattering_fraction[layer_index] * diagonal_mean_intensity,
                        1.0e-300,
                    )
                    * (1.0 - scattering_fraction[layer_index])
                    * planck_derivative
                    * frequency_weight
                )

            # Rosseland-mean mode 2
            for i in range(layer_count):
                source_derivative = (
                    planck[i]
                    * frequency
                    * h_over_kt[i]
                    / max(temperature[i] * stimulated[i], 1.0e-300)
                )
                if frequency_count == 1:
                    source_derivative = (
                        4.0 * (5.6697e-5 / 3.14159) * temperature[i] ** 3
                    )
                rosseland_accumulator[i] += (
                    source_derivative
                    / max(total_opacity[i], 1.0e-300)
                    * frequency_weight
                )

    @numba.njit(parallel=True, nogil=True, cache=True)
    def accumulate_transfer_range_parallel(
        chunk_count,
        range_start,
        range_stop,
        frequency_hz,
        frequency_weights,
        planck_all,
        stimulated_all,
        continuum_absorption_slab,
        continuum_scattering_slab,
        continuum_source_slab,
        line_mass_absorption_coefficient_slab,
        column_mass,
        h_over_kt,
        temperature,
        transfer_grid,
        mean_intensity_operator,
        eddington_flux_operator,
        second_moment_weights,
        target_integrated_eddington_flux,
        effective_temperature,
        frequency_count,
        rosseland_accumulator,
        radiation_energy_density,
        integrated_eddington_flux,
        radiative_acceleration,
        surface_radiation_pressure_constant,
        temperature_correction_heating_derivative,
        temperature_correction_mean_intensity_minus_source_integral,
        temperature_correction_integrated_eddington_flux,
        temperature_correction_diagonal_lambda,
    ):
        """Parallel per-frequency transfer accumulation over frequency chunks.

        Splits ``[range_start, range_stop)`` into ``chunk_count`` contiguous
        frequency chunks. Each chunk runs the serial per-frequency kernel on
        its slice into its OWN private zero-initialized accumulator buffers,
        then the per-chunk buffers are summed in chunk order onto the shared
        accumulators. Frequencies are independent, so each frequency's deposit
        is byte-identical to the serial path; only the float32 reduction is
        regrouped (~ulp per cell, deterministic for a fixed chunk count).
        """

        layer_count = column_mass.shape[0]
        total_span = range_stop - range_start
        if total_span <= 0 or chunk_count <= 0:
            return

        # Private per-chunk accumulators (start at zero; the serial kernel +=s).
        rosseland_accumulator_by_chunk = np.zeros(
            (chunk_count, layer_count), dtype=np.float64
        )
        radiation_energy_density_by_chunk = np.zeros(
            (chunk_count, layer_count), dtype=np.float64
        )
        integrated_eddington_flux_by_chunk = np.zeros(
            (chunk_count, layer_count), dtype=np.float64
        )
        radiative_acceleration_by_chunk = np.zeros(
            (chunk_count, layer_count), dtype=np.float64
        )
        surface_radiation_pressure_by_chunk = np.zeros(
            (chunk_count, 1), dtype=np.float64
        )
        heating_derivative_by_chunk = np.zeros(
            (chunk_count, layer_count), dtype=np.float64
        )
        mean_intensity_minus_source_by_chunk = np.zeros(
            (chunk_count, layer_count), dtype=np.float64
        )
        temperature_correction_integrated_eddington_flux_by_chunk = np.zeros(
            (chunk_count, layer_count), dtype=np.float64
        )
        diagonal_lambda_by_chunk = np.zeros(
            (chunk_count, layer_count), dtype=np.float64
        )

        bounds = np.empty(chunk_count + 1, dtype=np.int64)
        for c in range(chunk_count + 1):
            bounds[c] = range_start + (total_span * c) // chunk_count

        for c in numba.prange(chunk_count):
            chunk_start = bounds[c]
            chunk_stop = bounds[c + 1]
            if chunk_stop <= chunk_start:
                continue
            accumulate_transfer_range_compiled(
                chunk_start,
                chunk_stop,
                frequency_hz,
                frequency_weights,
                planck_all,
                stimulated_all,
                continuum_absorption_slab,
                continuum_scattering_slab,
                continuum_source_slab,
                line_mass_absorption_coefficient_slab,
                column_mass,
                h_over_kt,
                temperature,
                transfer_grid,
                mean_intensity_operator,
                eddington_flux_operator,
                second_moment_weights,
                target_integrated_eddington_flux,
                effective_temperature,
                frequency_count,
                rosseland_accumulator_by_chunk[c],
                radiation_energy_density_by_chunk[c],
                integrated_eddington_flux_by_chunk[c],
                radiative_acceleration_by_chunk[c],
                surface_radiation_pressure_by_chunk[c],
                heating_derivative_by_chunk[c],
                mean_intensity_minus_source_by_chunk[c],
                temperature_correction_integrated_eddington_flux_by_chunk[c],
                diagonal_lambda_by_chunk[c],
            )

        # Sum the private buffers in chunk order onto the shared accumulators.
        for c in range(chunk_count):
            for i in range(layer_count):
                rosseland_accumulator[i] += rosseland_accumulator_by_chunk[c, i]
                radiation_energy_density[i] += radiation_energy_density_by_chunk[c, i]
                integrated_eddington_flux[i] += integrated_eddington_flux_by_chunk[c, i]
                radiative_acceleration[i] += radiative_acceleration_by_chunk[c, i]
                temperature_correction_heating_derivative[i] += (
                    heating_derivative_by_chunk[c, i]
                )
                temperature_correction_mean_intensity_minus_source_integral[i] += (
                    mean_intensity_minus_source_by_chunk[c, i]
                )
                temperature_correction_integrated_eddington_flux[i] += (
                    temperature_correction_integrated_eddington_flux_by_chunk[c, i]
                )
                temperature_correction_diagonal_lambda[i] += diagonal_lambda_by_chunk[
                    c, i
                ]
            surface_radiation_pressure_constant[0] += (
                surface_radiation_pressure_by_chunk[c, 0]
            )
