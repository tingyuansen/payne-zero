"""Radiative-transfer depth-grid helpers and table loading.

Provides the depth-grid interpolation helpers (parabolic coefficients,
integrate, differentiate, remap) shared with the Rosseland and pressure
steps, and loads the packed JOSH quadrature tables. The per-frequency
transfer solve itself runs in the compiled kernel in transfer_kernels.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data_files import atmosphere_table_path


_SOURCE_FLOOR = 1.0e-38
_ITERATION_TOLERANCE = 1.0e-5


# --- depth-grid interpolation helpers (merged from radiative_transfer_math.py) ---


def parabolic_coefficients(
    values: np.ndarray,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return parabolic interpolation coefficients on a one-dimensional grid."""

    value = np.asarray(values, dtype=np.float64)
    coordinate = np.asarray(grid, dtype=np.float64)
    count = value.size
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
    denominator = coordinate[-1] - coordinate[last - 1]
    linear[-1] = (
        (value[-1] - value[last - 1]) / denominator if denominator != 0.0 else 0.0
    )
    constant[-1] = value[-1] - coordinate[-1] * linear[-1]
    if count == 2:
        return constant, linear, quadratic

    for index in range(1, last):
        left = index - 1
        denominator = coordinate[index] - coordinate[left]
        slope = (
            (value[index] - value[left]) / denominator if denominator != 0.0 else 0.0
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
        linear[2] = (value[3] - value[2]) / denominator if denominator != 0.0 else 0.0
        constant[2] = value[2] - coordinate[2] * linear[2]

    for index in range(1, last):
        if quadratic[index] == 0.0:
            continue
        next_index = min(index + 1, count - 1)
        denominator = abs(quadratic[next_index]) + abs(quadratic[index])
        weight = abs(quadratic[next_index]) / denominator if denominator > 0.0 else 0.0
        constant[index] = constant[next_index] + weight * (
            constant[index] - constant[next_index]
        )
        linear[index] = linear[next_index] + weight * (
            linear[index] - linear[next_index]
        )
        quadratic[index] = quadratic[next_index] + weight * (
            quadratic[index] - quadratic[next_index]
        )

    constant[last - 1] = constant[-1]
    linear[last - 1] = linear[-1]
    quadratic[last - 1] = quadratic[-1]
    return constant, linear, quadratic


def integrate_on_depth_grid(
    grid: np.ndarray,
    values: np.ndarray,
    *,
    surface_value: float,
) -> np.ndarray:
    """Integrate on a monotonic depth grid using parabolic intervals."""

    coordinate = np.asarray(grid, dtype=np.float64)
    value = np.asarray(values, dtype=np.float64)
    integral = np.zeros(value.size, dtype=np.float64)
    if value.size == 0:
        return integral
    constant, linear, quadratic = parabolic_coefficients(value, coordinate)
    integral[0] = float(surface_value)
    if value.size == 1:
        return integral
    for index in range(value.size - 1):
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


def differentiate_on_depth_grid(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Differentiate on a monotonic depth grid using parabolic intervals."""

    coordinate = np.asarray(grid, dtype=np.float64)
    value = np.asarray(values, dtype=np.float64)
    derivative = np.zeros(value.size, dtype=np.float64)
    if value.size < 2:
        return derivative

    low_width = coordinate[1] - coordinate[0]
    high_width = coordinate[-1] - coordinate[-2]
    derivative[0] = (value[1] - value[0]) / low_width if low_width != 0.0 else 0.0
    derivative[-1] = (value[-1] - value[-2]) / high_width if high_width != 0.0 else 0.0
    if value.size == 2:
        return derivative

    direction = (
        abs(coordinate[1] - coordinate[0]) / low_width if low_width != 0.0 else 1.0
    )
    for index in range(1, value.size - 1):
        scale = max(abs(value[index - 1]), abs(value[index]), abs(value[index + 1]))
        scale = scale / abs(coordinate[index]) if coordinate[index] != 0.0 else scale
        if scale == 0.0:
            scale = 1.0
        right_width = coordinate[index + 1] - coordinate[index]
        left_width = coordinate[index] - coordinate[index - 1]
        if right_width == 0.0 or left_width == 0.0:
            derivative[index] = 0.0
            continue
        right_slope = (value[index + 1] - value[index]) / right_width / scale
        left_slope = (value[index] - value[index - 1]) / left_width / scale
        right_denominator = direction * np.sqrt(1.0 + right_slope * right_slope) + 1.0
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


def remap_to_grid(
    source_grid: np.ndarray,
    source_values: np.ndarray,
    target_grid: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Remap values with the validated piecewise-quadratic convention."""

    old_grid = np.asarray(source_grid, dtype=np.float64)
    old_values = np.asarray(source_values, dtype=np.float64)
    new_grid = np.asarray(target_grid, dtype=np.float64)
    remapped = np.zeros(new_grid.size, dtype=np.float64)
    if old_grid.size == 0 or new_grid.size == 0:
        return remapped, 0

    source_count = old_grid.size
    grid_1based = np.empty(source_count + 1, dtype=np.float64)
    value_1based = np.empty(source_count + 1, dtype=np.float64)
    grid_1based[1:] = old_grid
    value_1based[1:] = old_values

    source_index = 2
    previous_source_index = 0
    forward_quadratic = forward_linear = forward_constant = 0.0
    backward_quadratic = backward_linear = backward_constant = 0.0
    constant = linear = quadratic = 0.0

    for target_index, target_value in enumerate(new_grid):
        while True:
            if target_value < grid_1based[source_index]:
                if source_index == previous_source_index:
                    break
                if source_index == 2 or source_index == 3:
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

                left_index = source_index - 1
                if (
                    source_index > previous_source_index + 1
                    or source_index == 3
                    or source_index == 4
                ):
                    width = grid_1based[left_index] - grid_1based[source_index - 2]
                    right_width = grid_1based[source_index] - grid_1based[left_index]
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
                right_width = grid_1based[source_index + 1] - grid_1based[source_index]
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
                    value_1based[left_index] / wide_width if wide_width != 0.0 else 0.0
                )
                right_term = (
                    value_1based[source_index] / right_width
                    if right_width != 0.0
                    else 0.0
                )
                second_term = (left_term - right_term) / width if width != 0.0 else 0.0
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
                linear = forward_linear + weight * (backward_linear - forward_linear)
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

    return remapped, max(previous_source_index - 1, 0)


# --- transfer-operator tables (merged from radiative_transfer_tables.py) ---

_DEFAULT_TRANSFER_TABLE_PATH = atmosphere_table_path("radiative_transfer_tables.npz")
_TRANSFER_TABLE_CACHE: "RadiativeTransferTables | None" = None


class RadiativeTransferTableError(RuntimeError):
    """Raised when the packaged radiative-transfer tables cannot be loaded."""


@dataclass(frozen=True)
class RadiativeTransferTables:
    """Fixed optical-depth operators used by the radiative-transfer solve."""

    surface_eddington_flux_weights: np.ndarray
    second_moment_weights: np.ndarray
    transfer_optical_depth_grid: np.ndarray
    mean_intensity_operator: np.ndarray
    eddington_flux_operator: np.ndarray


def load_radiative_transfer_tables(
    path: Path | None = None,
    *,
    force_reload: bool = False,
) -> RadiativeTransferTables:
    """Load packaged transfer operators with their validated working precision."""

    global _TRANSFER_TABLE_CACHE
    table_path = path or _DEFAULT_TRANSFER_TABLE_PATH
    if force_reload or _TRANSFER_TABLE_CACHE is None:
        if not table_path.exists():
            raise RadiativeTransferTableError(
                f"Missing radiative-transfer table archive: {table_path}"
            )
        with np.load(table_path, allow_pickle=False) as arrays:
            required = {
                "surface_eddington_flux_weights",
                "second_moment_weights",
                "transfer_optical_depth_grid",
                "mean_intensity_operator",
                "eddington_flux_operator",
            }
            missing = sorted(required.difference(arrays.files))
            if missing:
                raise RadiativeTransferTableError(
                    f"{table_path.name} is missing required keys: {', '.join(missing)}"
                )
            _TRANSFER_TABLE_CACHE = RadiativeTransferTables(
                surface_eddington_flux_weights=np.asarray(
                    arrays["surface_eddington_flux_weights"], dtype=np.float32
                ),
                second_moment_weights=np.asarray(
                    arrays["second_moment_weights"], dtype=np.float32
                ),
                transfer_optical_depth_grid=np.asarray(
                    arrays["transfer_optical_depth_grid"], dtype=np.float64
                ),
                mean_intensity_operator=np.asarray(
                    arrays["mean_intensity_operator"], dtype=np.float32
                ),
                eddington_flux_operator=np.asarray(
                    arrays["eddington_flux_operator"], dtype=np.float32
                ),
            )
    return _TRANSFER_TABLE_CACHE
