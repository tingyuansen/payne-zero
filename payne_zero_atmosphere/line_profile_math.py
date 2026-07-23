"""Profile-table math used by the atmosphere line-opacity kernels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache

import numpy as np

from .data_files import atmosphere_table_path


@dataclass(frozen=True)
class FastExponentialTables:
    """Two-stage exponential lookup table used in line absorption."""

    integer_step: np.ndarray
    fractional_step: np.ndarray


@dataclass(frozen=True)
class VoigtProfileBasis:
    """Precomputed tables for the validated Voigt-profile approximation."""

    gaussian_profile: np.ndarray
    first_correction: np.ndarray
    second_correction: np.ndarray


def build_hydrogen_continuum_selector_table() -> np.ndarray:
    """Hydrogen continuum-edge selector table for line-opacity cutoffs (Kurucz CONTX)."""

    selector = np.zeros((25, 16), dtype=np.float64)
    selector[:10, 0] = np.array(
        [
            109678.764,
            27419.659,
            12186.462,
            6854.871,
            4387.113,
            3046.604,
            2238.320,
            1713.711,
            1354.044,
            1096.776,
        ],
        dtype=np.float64,
    )
    selector[:10, 1] = np.array(
        [
            198310.760,
            38454.691,
            32033.214,
            29223.753,
            27175.760,
            15073.868,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    selector[:10, 2] = np.array(
        [
            438908.850,
            109726.529,
            48766.491,
            27430.925,
            17555.715,
            12191.437,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    selector[:10, 3] = np.array(
        [
            90883.840,
            90867.420,
            90840.420,
            90820.420,
            90804.000,
            90777.000,
            80691.180,
            80627.760,
            69235.820,
            69172.400,
        ],
        dtype=np.float64,
    )
    selector[:10, 5] = np.array(
        [61671.020, 39820.615, 39800.556, 39759.842, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    selector[:10, 7] = np.array(
        [48278.370, 48166.309, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    selector[:10, 9] = np.array(
        [
            66035.000,
            65957.885,
            65811.843,
            65747.550,
            65670.435,
            65524.393,
            59736.150,
            59448.700,
            50640.630,
            50553.180,
        ],
        dtype=np.float64,
    )
    return selector


@lru_cache(maxsize=1)
def load_hydrogen_continuum_selector_table() -> np.ndarray:
    """Return a cached copy of the XLINOP continuum selector table."""

    return build_hydrogen_continuum_selector_table()


@lru_cache(maxsize=1)
def build_selection_log_lookup() -> np.ndarray:
    """Return the packed TABLOG lookup table used by selected-line records."""

    index = np.arange(1, 32769, dtype=np.float64)
    return 10.0 ** ((index - 16384.0) * 0.001)


@lru_cache(maxsize=1)
def build_fast_exponential_tables() -> FastExponentialTables:
    """Return the two lookup arrays for exp(-x)."""

    index = np.arange(1001, dtype=np.float64)
    return FastExponentialTables(
        integer_step=np.exp(-index),
        fractional_step=np.exp(-index * 0.001),
    )


def fast_exponential_lookup(
    x: float,
    tables: FastExponentialTables | None = None,
) -> float:
    """Evaluate the table approximation to exp(-x)."""

    if not np.isfinite(x) or x < 0.0 or x >= 1001.0:
        return 0.0
    lookup = tables or build_fast_exponential_tables()
    integer_index = int(x)
    fractional_index = int((x - float(integer_index)) * 1000.0 + 1.5)
    if fractional_index < 1:
        fractional_index = 1
    if fractional_index > 1001:
        fractional_index = 1001
    return float(
        lookup.integer_step[integer_index]
        * lookup.fractional_step[fractional_index - 1]
    )


def _remap_profile_table(
    source_grid: np.ndarray,
    source_values: np.ndarray,
    target_grid: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_grid, dtype=np.float64)
    values = np.asarray(source_values, dtype=np.float64)
    target = np.asarray(target_grid, dtype=np.float64)
    if source.ndim != 1 or values.shape != source.shape or source.size < 2:
        raise ValueError(
            "source_grid and source_values must be matching one-dimensional "
            "arrays with at least two samples"
        )
    if target.ndim != 1 or np.any(np.diff(source) <= 0.0):
        raise ValueError(
            "target_grid must be one-dimensional and source_grid increasing"
        )
    remapped = np.zeros_like(target, dtype=np.float64)
    source_count = int(source.size)
    current_source = 1
    previous_source = -1
    quadratic = linear = constant = 0.0
    forward_quadratic = forward_linear = forward_constant = 0.0
    backward_quadratic = backward_linear = backward_constant = 0.0

    for target_index, target_value in enumerate(target):
        while current_source < source_count and target_value >= source[current_source]:
            current_source += 1
        if current_source == previous_source:
            remapped[target_index] = (
                constant + (linear + quadratic * target_value) * target_value
            )
            continue

        if current_source > 1:
            left = current_source - 1
            if current_source != 2:
                if current_source <= previous_source + 1 and current_source != 3:
                    backward_quadratic = forward_quadratic
                    backward_linear = forward_linear
                    backward_constant = forward_constant
                    if current_source == source_count:
                        quadratic = backward_quadratic
                        linear = backward_linear
                        constant = backward_constant
                        previous_source = current_source
                        remapped[target_index] = (
                            constant
                            + (linear + quadratic * target_value) * target_value
                        )
                        continue
                else:
                    left2 = current_source - 2
                    slope = (values[left] - values[left2]) / (
                        source[left] - source[left2]
                    )
                    backward_quadratic = values[current_source] / (
                        (source[current_source] - source[left])
                        * (source[current_source] - source[left2])
                    ) + (
                        values[left2] / (source[current_source] - source[left2])
                        - values[left] / (source[current_source] - source[left])
                    ) / (source[left] - source[left2])
                    backward_linear = (
                        slope - (source[left] + source[left2]) * backward_quadratic
                    )
                    backward_constant = (
                        values[left2]
                        - source[left2] * slope
                        + source[left] * source[left2] * backward_quadratic
                    )
                    if current_source >= source_count - 1:
                        quadratic = backward_quadratic
                        linear = backward_linear
                        constant = backward_constant
                        previous_source = current_source
                        remapped[target_index] = (
                            constant
                            + (linear + quadratic * target_value) * target_value
                        )
                        continue

                slope = (values[current_source] - values[left]) / (
                    source[current_source] - source[left]
                )
                forward_quadratic = values[current_source + 1] / (
                    (source[current_source + 1] - source[current_source])
                    * (source[current_source + 1] - source[left])
                ) + (
                    values[left] / (source[current_source + 1] - source[left])
                    - values[current_source]
                    / (source[current_source + 1] - source[current_source])
                ) / (source[current_source] - source[left])
                forward_linear = (
                    slope - (source[current_source] + source[left]) * forward_quadratic
                )
                forward_constant = (
                    values[left]
                    - source[left] * slope
                    + source[current_source] * source[left] * forward_quadratic
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
                previous_source = current_source
                remapped[target_index] = (
                    constant + (linear + quadratic * target_value) * target_value
                )
                continue

        current_source = min(source_count - 1, current_source)
        quadratic = 0.0
        linear = (values[current_source] - values[current_source - 1]) / (
            source[current_source] - source[current_source - 1]
        )
        constant = values[current_source] - source[current_source] * linear
        previous_source = current_source
        remapped[target_index] = (
            constant + (linear + quadratic * target_value) * target_value
        )

    return remapped


@lru_cache(maxsize=1)
def build_voigt_profile_basis() -> VoigtProfileBasis:
    """Return the three Voigt-profile basis tables."""

    line_tables = load_line_opacity_tables()
    profile_grid = np.arange(2001, dtype=np.float64) / 200.0
    first_correction = _remap_profile_table(
        line_tables.voigt_interpolation_table,
        line_tables.hydrogen_profile_table,
        profile_grid,
    )
    squared_offset = profile_grid * profile_grid
    gaussian_profile = np.exp(-squared_offset)
    second_correction = gaussian_profile - 2.0 * squared_offset * gaussian_profile
    return VoigtProfileBasis(
        gaussian_profile=gaussian_profile,
        first_correction=first_correction,
        second_correction=second_correction,
    )


def evaluate_voigt_profile(
    frequency_offset: float,
    damping_parameter: float,
    basis: VoigtProfileBasis | None = None,
) -> float:
    """Evaluate the validated scalar Voigt approximation."""

    profile_basis = basis or build_voigt_profile_basis()
    table_index = int(frequency_offset * 200.0 + 1.5)
    if table_index < 1:
        table_index = 1
    if table_index > 2001:
        table_index = 2001
    zero_based_index = table_index - 1

    gaussian = profile_basis.gaussian_profile[zero_based_index]
    first = profile_basis.first_correction[zero_based_index]
    second = profile_basis.second_correction[zero_based_index]
    damping = float(damping_parameter)
    offset = float(frequency_offset)

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
                ((damping_fraction - 10.0 * offset_fraction) * damping_fraction * 3.0)
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


# --- packaged line-opacity tables (merged from line_opacity_tables.py) ---

_DEFAULT_TABLE_PATH = atmosphere_table_path("line_opacity_tables.npz")
_TABLE_CACHE: "LineOpacityTables | None" = None


class LineOpacityTableError(RuntimeError):
    """Raised when packaged line-opacity tables are missing or malformed."""


@dataclass(frozen=True)
class LineOpacityTables:
    """Small interpolation tables shared by LINOP/XLINOP Voigt helpers."""

    voigt_interpolation_table: np.ndarray
    hydrogen_profile_table: np.ndarray


def load_line_opacity_tables(
    path: Path | None = None,
    *,
    force_reload: bool = False,
) -> LineOpacityTables:
    """Load packaged line-profile helper tables with modern field names."""

    global _TABLE_CACHE
    table_path = path or _DEFAULT_TABLE_PATH
    if force_reload or _TABLE_CACHE is None:
        if not table_path.exists():
            raise LineOpacityTableError(
                f"Missing line-opacity table archive: {table_path}"
            )
        with np.load(table_path, allow_pickle=False) as arrays:
            required = {"voigt_interpolation_table", "hydrogen_profile_table"}
            missing = sorted(required.difference(arrays.files))
            if missing:
                raise LineOpacityTableError(
                    f"{table_path.name} is missing required keys: {', '.join(missing)}"
                )
            _TABLE_CACHE = LineOpacityTables(
                voigt_interpolation_table=np.asarray(
                    arrays["voigt_interpolation_table"], dtype=np.float64
                ),
                hydrogen_profile_table=np.asarray(
                    arrays["hydrogen_profile_table"], dtype=np.float64
                ),
            )
    return _TABLE_CACHE
