"""Continuum opacity on a depth x frequency grid.

The runtime path keeps immutable cross-section tables resident and evaluates the
per-depth opacity terms on the selected torch device. Discrete table lookups stay
in fp64 host arithmetic because their bracket choices must be dtype-independent;
the resulting vectors are then used by the device kernels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .constants import (
    BOLTZMANN_ERG_PER_K,
    NATURAL_LOG_10,
    REFERENCE_BOLTZMANN_EV_PER_K,
    REFERENCE_SAHA_COEFFICIENT,
    REFERENCE_WAVENUMBER_PER_EV,
    LIGHT_SPEED_ANGSTROM_PER_S,
    LIGHT_SPEED_CM_PER_S,
    LIGHT_SPEED_NM_PER_S,
    PLANCK_ERG_SECOND,
)
from .device import DEFAULT_DTYPE, device as _device
from . import paths as _runtime_paths

# Continuum constants and source-table coefficients.

RYDBERG_WAVENUMBER_CM = 109677.576
HYDROGEN_MAXIMUM_EXPLICIT_LEVEL = 6

_HYDROGEN_HIGH_LEVEL_BOUND_FREE_TRANSITIONS = (
    (15, 487.456, 450.0, 109191.313),
    (14, 559.579, 392.0, 109119.188),
    (13, 648.980, 338.0, 109029.789),
    (12, 761.649, 288.0, 108917.117),
    (11, 906.426, 242.0, 108772.336),
    (10, 1096.776, 200.0, 108581.992),
    (9, 1354.044, 162.0, 108324.719),
    (8, 1713.713, 128.0, 107965.051),
    (7, 2238.320, 98.0, 107440.444),
)
_HYDROGEN_LOW_LEVEL_BOUND_FREE_TRANSITIONS = (
    (6, 3046.604, 72.0, 106632.160),
    (5, 4387.113, 50.0, 105291.651),
    (4, 6854.871, 32.0, 102823.893),
    (3, 12186.462, 18.0, 97492.302),
    (2, 27419.659, 8.0, 82259.105),
)
_HELIUM_NEUTRAL_N5_AUTOIONIZATION_TRANSITIONS = (
    (4368.190, 3.0, 193942.57, 28),
    (4388.260, 9.0, 193922.5, 27),
    (4388.260, 27.0, 193922.5, 26),
    (4389.390, 7.0, 193921.37, 25),
    (4389.450, 15.0, 193921.31, 24),
    (4392.369, 5.0, 193918.391, 23),
    (4393.515, 15.0, 193917.245, 22),
    (4509.980, 9.0, 193800.78, 21),
    (4647.133, 1.0, 193663.627, 20),
    (4963.671, 3.0, 193347.089, 19),
)
_HELIUM_NEUTRAL_N4_AUTOIONIZATION_TRANSITIONS = (
    (6817.943, 3.0, 191492.817, 18),
    (6858.680, 7.0, 191452.08, 17),
    (6858.960, 21.0, 191451.80, 16),
    (6864.201, 5.0, 191446.559, 15),
    (6866.172, 15.0, 191444.588, 14),
    (7093.620, 9.0, 191217.14, 13),
    (7370.429, 1.0, 190940.331, 12),
    (8012.550, 3.0, 190298.210, 11),
)
_HELIUM_NEUTRAL_N3_AUTOIONIZATION_TRANSITIONS = (
    (12101.289, (58.81, -2.89), 3.0, 186209.471, 10),
    (12205.695, (85.20, -3.69), 5.0, 186105.065, 9),
    (12209.106, (85.20, -3.69), 15.0, 186101.654, 8),
    (12746.066, (49.30, -2.60), 9.0, 185564.694, 7),
    (13445.824, (23.85, -1.86), 1.0, 184864.936, 6),
    (15073.868, (12.69, -1.54), 3.0, 183236.892, 5),
)
_HELIUM_NEUTRAL_N2_AUTOIONIZATION_TRANSITIONS = (
    (27175.760, (81.35, -3.5), 3.0, 171135.000, 4),
    (29223.753, (61.21, -2.9), 9.0, 169087.007, 3),
    (32033.214, (26.83, -1.91), 1.0, 166277.546, 2),
)
_HELIUM_SINGLY_IONIZED_HIGH_LEVEL_BOUND_FREE_TRANSITIONS = (
    (5418.390, 162.0, 433490.46, 59049.0),
    (6857.660, 128.0, 432051.19, 32768.0),
    (8956.950, 98.0, 429951.90, 16807.0),
)
_HELIUM_SINGLY_IONIZED_LOW_LEVEL_BOUND_FREE_TRANSITIONS = (
    (12191.437, 72.0, 426717.413, 7776.0, (1.0986, -2.704e13, 1.229e27)),
    (17555.715, 50.0, 421353.135, 3125.0, (1.102, -3.909e13, 2.371e27)),
    (27430.925, 32.0, 411477.925, 1024.0, (1.101, -5.765e13, 4.593e27)),
    (48766.491, 18.0, 390142.359, 243.0, (1.101, -9.863e13, 1.035e28)),
    (109726.529, 8.0, 329182.321, 32.0, (1.105, -2.375e14, 4.077e28)),
    (438908.850, 2.0, 0.0, 1.0, (0.9916, 2.719e13, -2.268e30)),
)

# He I cross-section tables for the full Marr-West plus autoionization stack.
# The compact analytic expression is too low in the hot-star far UV and IR.
_HELIUM_GROUND_CROSS_SECTION_50_TO_505_A = np.array(
    [
        7.58,
        7.46,
        7.33,
        7.19,
        7.06,
        6.94,
        6.81,
        6.68,
        6.55,
        6.43,
        6.30,
        6.18,
        6.05,
        5.93,
        5.81,
        5.69,
        5.57,
        5.45,
        5.33,
        5.21,
        5.10,
        4.98,
        4.87,
        4.76,
        4.64,
        4.53,
        4.42,
        4.31,
        4.20,
        4.09,
        4.00,
        3.88,
        3.78,
        3.68,
        3.57,
        3.47,
        3.37,
        3.27,
        3.18,
        3.08,
        2.98,
        2.89,
        2.80,
        2.70,
        2.61,
        2.52,
        2.44,
        2.35,
        2.26,
        2.18,
        2.10,
        2.02,
        1.94,
        1.86,
        1.78,
        1.70,
        1.63,
        1.55,
        1.48,
        1.41,
        1.34,
        1.28,
        1.21,
        1.14,
        1.08,
        1.02,
        0.961,
        0.903,
        0.847,
        0.792,
        0.738,
        0.687,
        0.637,
        0.588,
        0.542,
        0.497,
        0.454,
        0.412,
        0.373,
        0.335,
        0.299,
        0.265,
        0.233,
        0.202,
        0.174,
        0.147,
        0.123,
        0.100,
        0.0795,
        0.0609,
        0.0443,
        0.0315,
    ],
    dtype=np.float64,
)
_HELIUM_GROUND_CROSS_SECTION_20_TO_50_A = np.array(
    [
        0.0315,
        0.0282,
        0.0250,
        0.0220,
        0.0193,
        0.0168,
        0.0145,
        0.0124,
        0.0105,
        0.00885,
        0.00736,
        0.00604,
        0.00489,
        0.00389,
        0.00303,
        0.00231,
    ],
    dtype=np.float64,
)
_HELIUM_GROUND_CROSS_SECTION_10_TO_20_A = np.array(
    [
        0.00231,
        0.00199,
        0.00171,
        0.00145,
        0.00122,
        0.00101,
        0.000832,
        0.000673,
        0.000535,
        0.000417,
        0.000318,
    ],
    dtype=np.float64,
)
_HELIUM_GROUND_CROSS_SECTION_BELOW_10_A = np.array(
    [
        0.000318,
        0.000274,
        0.000235,
        0.000200,
        0.000168,
        0.000139,
        0.000115,
        0.000093,
        0.000074,
        0.000057,
        0.000044,
        0.000032,
        0.000023,
        0.000016,
        0.000010,
        0.000006,
        0.000003,
        0.000001,
        0.0000006,
        0.0000003,
        0.0,
    ],
    dtype=np.float64,
)
_HELIUM_NEUTRAL_2S_SINGLET_LOG10_FREQUENCY_HZ = np.array(
    [
        15.947182,
        15.913654,
        15.877320,
        15.837666,
        15.794025,
        15.745503,
        15.690869,
        15.628361,
        15.555317,
        15.467455,
        15.357189,
        15.289399,
        15.251073,
        15.209035,
        15.162487,
        14.982421,
    ],
    dtype=np.float64,
)
_HELIUM_NEUTRAL_2S_SINGLET_LOG10_CROSS_SECTION_CM2 = np.array(
    [
        -19.635557,
        -19.159345,
        -18.958474,
        -18.809535,
        -18.676481,
        -18.546006,
        -18.410962,
        -18.264821,
        -18.100205,
        -17.909165,
        -17.684370,
        -17.557867,
        -17.490360,
        -17.417876,
        -17.349386,
        -17.084441,
    ],
    dtype=np.float64,
)
_HELIUM_NEUTRAL_2S_TRIPLET_LOG10_FREQUENCY_HZ = np.array(
    [
        15.956523,
        15.923736,
        15.888271,
        15.849649,
        15.807255,
        15.760271,
        15.707580,
        15.647601,
        15.577992,
        15.495055,
        15.392451,
        15.330345,
        15.295609,
        15.257851,
        15.216496,
        15.061770,
    ],
    dtype=np.float64,
)
_HELIUM_NEUTRAL_2S_TRIPLET_LOG10_CROSS_SECTION_CM2 = np.array(
    [
        -18.426022,
        -18.610700,
        -18.593051,
        -18.543304,
        -18.465513,
        -18.378707,
        -18.278574,
        -18.164329,
        -18.033346,
        -17.882435,
        -17.705542,
        -17.605584,
        -17.553459,
        -17.500667,
        -17.451318,
        -17.266686,
    ],
    dtype=np.float64,
)
_HELIUM_NEUTRAL_2P_SINGLET_LOG10_FREQUENCY_HZ = np.array(
    [
        15.939981,
        15.905870,
        15.868850,
        15.828377,
        15.783742,
        15.733988,
        15.677787,
        15.613218,
        15.537343,
        15.445346,
        15.328474,
        15.255641,
        15.214064,
        15.168081,
        15.116647,
        14.911002,
    ],
    dtype=np.float64,
)
_HELIUM_NEUTRAL_2P_SINGLET_LOG10_CROSS_SECTION_CM2 = np.array(
    [
        -18.798876,
        -19.685922,
        -20.011664,
        -20.143030,
        -20.091354,
        -19.908333,
        -19.656788,
        -19.367745,
        -19.043016,
        -18.674484,
        -18.240861,
        -17.989700,
        -17.852015,
        -17.702677,
        -17.525347,
        -16.816344,
    ],
    dtype=np.float64,
)
_HELIUM_NEUTRAL_2P_TRIPLET_LOG10_FREQUENCY_HZ = np.array(
    [
        15.943031,
        15.909169,
        15.872441,
        15.832318,
        15.788107,
        15.738880,
        15.683351,
        15.619667,
        15.545012,
        15.454805,
        15.340813,
        15.270195,
        15.230054,
        15.185821,
        15.136567,
        14.942557,
    ],
    dtype=np.float64,
)
_HELIUM_NEUTRAL_2P_TRIPLET_LOG10_CROSS_SECTION_CM2 = np.array(
    [
        -19.791021,
        -19.697886,
        -19.591421,
        -19.471855,
        -19.337053,
        -19.183958,
        -19.009750,
        -18.807990,
        -18.570571,
        -18.288361,
        -17.943476,
        -17.738737,
        -17.624154,
        -17.497163,
        -17.403183,
        -17.032999,
    ],
    dtype=np.float64,
)
_HELIUM_NEUTRAL_STATISTICAL_WEIGHTS = np.array(
    [1.0, 3.0, 1.0, 9.0, 3.0, 3.0, 1.0, 9.0, 20.0, 3.0], dtype=np.float64
)
_HELIUM_NEUTRAL_EXCITATION_EV = np.array(
    [
        0.0,
        19.819,
        20.615,
        20.964,
        21.217,
        22.718,
        22.920,
        23.006,
        23.073,
        23.086,
    ],
    dtype=np.float64,
)
_HELIUM_NEUTRAL_EDGE_FREQUENCIES_HZ = np.array(
    [
        5.945209e15,
        1.152844e15,
        0.9603331e15,
        0.8761076e15,
        0.8147104e15,
        0.4519048e15,
        0.4030971e15,
        0.3821191e15,
        0.3660215e15,
        0.3627891e15,
    ],
    dtype=np.float64,
)

_DEFAULT_CONTINUUM_TABLES = _runtime_paths.SYNTHESIS_TABLE_DIR / "continuum_tables.npz"


# Invariant tables uploaded once; discrete lookups keep fp64 host arrays.


@dataclass
class ContinuumTables:
    """Continuum cross-section, Gaunt, and Rayleigh tables.

    The discrete interpolators read these in fp64 (their bracket selection must be
    dtype-independent). Device copies are kept only for the Coulomb free-free
    table used in depth-vector bilinear interpolation.
    """

    arrays: dict  # Raw continuum table arrays keyed by packaged NPZ name.
    device: torch.device
    dtype: torch.dtype
    hydrogen_neutral_level_energy_ev: np.ndarray
    hydrogen_neutral_level_statistical_weight: np.ndarray
    hminus_freefree_log_table: np.ndarray
    hminus_freefree_log_wavelength_grid: np.ndarray
    hminus_freefree_temperature_count: int
    coulomb_freefree_gaunt_table_device: torch.Tensor

    @classmethod
    def from_npz(
        cls, path: Path = _DEFAULT_CONTINUUM_TABLES, device=None, dtype=None
    ) -> "ContinuumTables":
        with np.load(path, allow_pickle=False) as data:
            arrays = {
                key: np.asarray(data[key]).astype(np.float64) for key in data.files
            }
        return cls.from_dict(arrays, device=device, dtype=dtype)

    @classmethod
    def from_dict(cls, arrays: dict, device=None, dtype=None) -> "ContinuumTables":
        if device is None:
            device = _device()
        if dtype is None:
            dtype = DEFAULT_DTYPE
        hydrogen_neutral_level_energy_ev = (
            arrays["hydrogen_neutral_level_energy_cm"] / REFERENCE_WAVENUMBER_PER_EV
        )
        hydrogen_neutral_level_statistical_weight = arrays[
            "hydrogen_neutral_level_statistical_weight"
        ]

        hminus_theta_grid = arrays["hminus_freefree_theta_grid"]
        hminus_inverse_wavelength_grid = arrays[
            "hminus_freefree_inverse_wavelength_grid"
        ]
        hminus_freefree_short_wavelength = arrays[
            "hminus_freefree_short_wavelength_table"
        ]
        hminus_freefree_long_wavelength = arrays[
            "hminus_freefree_long_wavelength_table"
        ]
        n_temperature_grid = hminus_theta_grid.size
        freefree_table = np.zeros((n_temperature_grid, 22))
        for temperature_index in range(n_temperature_grid):
            for wavelength_index in range(22):
                if wavelength_index < 11:
                    freefree_table[temperature_index, wavelength_index] = (
                        hminus_freefree_short_wavelength[
                            wavelength_index, temperature_index
                        ]
                    )
                else:
                    freefree_table[temperature_index, wavelength_index] = (
                        hminus_freefree_long_wavelength[
                            wavelength_index - 11, temperature_index
                        ]
                    )
        freefree_log_table = np.zeros((22, n_temperature_grid))
        for wavelength_index in range(22):
            for temperature_index in range(n_temperature_grid):
                freefree_log_table[wavelength_index, temperature_index] = np.log(
                    freefree_table[temperature_index, wavelength_index]
                    / hminus_theta_grid[temperature_index]
                    * 5040.0
                    * BOLTZMANN_ERG_PER_K
                )
        freefree_log_wavelength_grid = np.log(91.134 / hminus_inverse_wavelength_grid)

        return cls(
            arrays=arrays,
            device=device,
            dtype=dtype,
            hydrogen_neutral_level_energy_ev=hydrogen_neutral_level_energy_ev,
            hydrogen_neutral_level_statistical_weight=(
                hydrogen_neutral_level_statistical_weight
            ),
            hminus_freefree_log_table=freefree_log_table,
            hminus_freefree_log_wavelength_grid=freefree_log_wavelength_grid,
            hminus_freefree_temperature_count=n_temperature_grid,
            coulomb_freefree_gaunt_table_device=torch.as_tensor(
                arrays["coulomb_freefree_gaunt_table"], dtype=dtype, device=device
            ),
        )


# Host-side fp64 interpolation helpers.


def _parabolic_interpolate(source_x, source_y, target_x):
    """Parabolic interpolation used by continuum table setup."""
    source_x = np.asarray(source_x, np.float64)
    source_y = np.asarray(source_y, np.float64)
    target_x = np.asarray(target_x, np.float64)
    n_source = source_x.size
    n_target = target_x.size
    target_y = np.zeros(n_target)
    if n_source == 0 or n_target == 0:
        return target_y

    # This helper intentionally keeps the validated one-based table walk.  The
    # readable names below describe the same constant + linear*x + quadratic*x^2
    # coefficients that the scalar continuum tables use.
    indexed_x = np.empty(n_source + 1)
    indexed_y = np.empty(n_source + 1)
    indexed_x[1:] = source_x
    indexed_y[1:] = source_y

    upper_index = 2
    last_coefficient_index = 0
    forward_quadratic = forward_linear = forward_constant = 0.0
    backward_quadratic = backward_linear = backward_constant = 0.0
    constant = linear = quadratic = 0.0

    for target_index in range(1, n_target + 1):
        x_value = target_x[target_index - 1]
        while True:
            if x_value < indexed_x[upper_index]:
                if upper_index == last_coefficient_index:
                    break
                if upper_index == 2 or upper_index == 3:
                    upper_index = min(n_source, upper_index)
                    quadratic = 0.0
                    linear = (indexed_y[upper_index] - indexed_y[upper_index - 1]) / (
                        indexed_x[upper_index] - indexed_x[upper_index - 1]
                    )
                    constant = indexed_y[upper_index] - indexed_x[upper_index] * linear
                    last_coefficient_index = upper_index
                    break

                lower_index = upper_index - 1
                if (
                    upper_index > last_coefficient_index + 1
                    or upper_index == 3
                    or upper_index == 4
                ):
                    lower2_index = upper_index - 2
                    backward_slope = (
                        indexed_y[lower_index] - indexed_y[lower2_index]
                    ) / (indexed_x[lower_index] - indexed_x[lower2_index])
                    backward_quadratic = indexed_y[upper_index] / (
                        (indexed_x[upper_index] - indexed_x[lower_index])
                        * (indexed_x[upper_index] - indexed_x[lower2_index])
                    ) + (
                        indexed_y[lower2_index]
                        / (indexed_x[upper_index] - indexed_x[lower2_index])
                        - indexed_y[lower_index]
                        / (indexed_x[upper_index] - indexed_x[lower_index])
                    ) / (indexed_x[lower_index] - indexed_x[lower2_index])
                    backward_linear = (
                        backward_slope
                        - (indexed_x[lower_index] + indexed_x[lower2_index])
                        * backward_quadratic
                    )
                    backward_constant = (
                        indexed_y[lower2_index]
                        - indexed_x[lower2_index] * backward_slope
                        + indexed_x[lower_index]
                        * indexed_x[lower2_index]
                        * backward_quadratic
                    )
                    if upper_index >= n_source:
                        quadratic = backward_quadratic
                        linear = backward_linear
                        constant = backward_constant
                        last_coefficient_index = upper_index
                        break
                else:
                    backward_quadratic = forward_quadratic
                    backward_linear = forward_linear
                    backward_constant = forward_constant
                    if upper_index == n_source:
                        quadratic = backward_quadratic
                        linear = backward_linear
                        constant = backward_constant
                        last_coefficient_index = upper_index
                        break

                forward_slope = (indexed_y[upper_index] - indexed_y[lower_index]) / (
                    indexed_x[upper_index] - indexed_x[lower_index]
                )
                forward_quadratic = indexed_y[upper_index + 1] / (
                    (indexed_x[upper_index + 1] - indexed_x[upper_index])
                    * (indexed_x[upper_index + 1] - indexed_x[lower_index])
                ) + (
                    indexed_y[lower_index]
                    / (indexed_x[upper_index + 1] - indexed_x[lower_index])
                    - indexed_y[upper_index]
                    / (indexed_x[upper_index + 1] - indexed_x[upper_index])
                ) / (indexed_x[upper_index] - indexed_x[lower_index])
                forward_linear = (
                    forward_slope
                    - (indexed_x[upper_index] + indexed_x[lower_index])
                    * forward_quadratic
                )
                forward_constant = (
                    indexed_y[lower_index]
                    - indexed_x[lower_index] * forward_slope
                    + indexed_x[upper_index]
                    * indexed_x[lower_index]
                    * forward_quadratic
                )
                blend_weight = (
                    abs(forward_quadratic)
                    / (abs(forward_quadratic) + abs(backward_quadratic))
                    if abs(forward_quadratic) != 0.0
                    else 0.0
                )
                constant = forward_constant + blend_weight * (
                    backward_constant - forward_constant
                )
                linear = forward_linear + blend_weight * (
                    backward_linear - forward_linear
                )
                quadratic = forward_quadratic + blend_weight * (
                    backward_quadratic - forward_quadratic
                )
                last_coefficient_index = upper_index
                break

            upper_index += 1
            if upper_index > n_source:
                upper_index = min(n_source, upper_index)
                quadratic = 0.0
                linear = (indexed_y[upper_index] - indexed_y[upper_index - 1]) / (
                    indexed_x[upper_index] - indexed_x[upper_index - 1]
                )
                constant = indexed_y[upper_index] - indexed_x[upper_index] * linear
                last_coefficient_index = upper_index
                break
        target_y[target_index - 1] = constant + (linear + quadratic * x_value) * x_value
    return target_y


def _linear_interpolate(source_x, source_y, target_x):
    """Linear interpolation/extrapolation helper for host-side fp64 setup."""
    source_x = np.asarray(source_x, np.float64)
    source_y = np.asarray(source_y, np.float64)
    target_x = np.asarray(target_x, np.float64)
    n_source = source_x.size
    n_target = target_x.size
    target_y = np.zeros(n_target)
    upper_index = 1
    for target_index in range(n_target):
        while (
            upper_index < n_source - 1
            and target_x[target_index] >= source_x[upper_index]
        ):
            upper_index += 1
        denominator = source_x[upper_index] - source_x[upper_index - 1]
        if abs(denominator) < 1e-40:
            target_y[target_index] = source_y[upper_index - 1]
        else:
            interpolation_fraction = (
                target_x[target_index] - source_x[upper_index - 1]
            ) / denominator
            target_y[target_index] = (
                source_y[upper_index - 1]
                + (source_y[upper_index] - source_y[upper_index - 1])
                * interpolation_fraction
            )
    return target_y


def _seaton_photoionization_cross_section(
    threshold_frequency: float,
    threshold_cross_section: float,
    power: float,
    asymmetry: float,
    frequency: float,
) -> float:
    """Seaton photoionization cross-section formula."""
    if frequency < threshold_frequency:
        return 0.0
    frequency_ratio = threshold_frequency / frequency
    return (
        threshold_cross_section
        * (asymmetry + (1.0 - asymmetry) * frequency_ratio)
        * frequency_ratio**power
    )


def _karzas_latter_cross_section(
    continuum_arrays,
    frequency_hz,
    effective_charge_squared,
    principal_quantum_number,
    angular_quantum_number,
):
    """Karzas-Latter hydrogenic bound-free cross-section."""
    log10_frequency_table = continuum_arrays["karzas_latter_log10_frequency_hz"]
    total_level_log10_cross_section_cm2 = continuum_arrays[
        "karzas_latter_total_log10_cross_section_cm2"
    ]
    angular_level_log10_cross_section_cm2 = continuum_arrays[
        "karzas_latter_angular_log10_cross_section_cm2"
    ]
    high_level_energy_offset_rydberg = continuum_arrays[
        "karzas_latter_high_level_energy_offset_rydberg"
    ]
    if (
        frequency_hz <= 0.0
        or effective_charge_squared <= 0.0
        or principal_quantum_number <= 0
    ):
        return 0.0
    if angular_quantum_number < 0:
        angular_quantum_number = 0
    scaled_log_frequency = math.log10(frequency_hz / effective_charge_squared)
    if principal_quantum_number <= 15:
        frequency_column = log10_frequency_table[:, principal_quantum_number - 1]
        if scaled_log_frequency < frequency_column[-1]:
            return 0.0
        if (
            angular_quantum_number >= principal_quantum_number
            or principal_quantum_number > 6
        ):
            log10_cross_section_cm2_column = total_level_log10_cross_section_cm2[
                :, principal_quantum_number - 1
            ]
        else:
            log10_cross_section_cm2_column = angular_level_log10_cross_section_cm2[
                angular_quantum_number,
                principal_quantum_number - 1,
                :,
            ]
            if np.isnan(log10_cross_section_cm2_column[0]):
                return 0.0
        left_index = 1
        right_index = frequency_column.size - 1
        bracket_index = frequency_column.size
        while left_index <= right_index:
            mid_index = (left_index + right_index) // 2
            if scaled_log_frequency > frequency_column[mid_index]:
                bracket_index = mid_index
                right_index = mid_index - 1
            else:
                left_index = mid_index + 1
        if bracket_index >= frequency_column.size:
            return float(
                np.exp(log10_cross_section_cm2_column[-1] * NATURAL_LOG_10)
                / effective_charge_squared
            )
        denominator = (
            frequency_column[bracket_index - 1] - frequency_column[bracket_index]
        )
        if abs(denominator) < 1e-15:
            return float(
                np.exp(
                    log10_cross_section_cm2_column[bracket_index - 1] * NATURAL_LOG_10
                )
                / effective_charge_squared
            )
        interpolation_fraction = (
            scaled_log_frequency - frequency_column[bracket_index]
        ) / denominator
        log10_cross_section_cm2 = (
            log10_cross_section_cm2_column[bracket_index - 1]
            - log10_cross_section_cm2_column[bracket_index]
        ) * interpolation_fraction + log10_cross_section_cm2_column[bracket_index]
        return float(
            np.exp(log10_cross_section_cm2 * NATURAL_LOG_10) / effective_charge_squared
        )
    inverse_level_squared = 1.0 / (principal_quantum_number * principal_quantum_number)
    rydberg_frequency = 109677.576 * LIGHT_SPEED_CM_PER_S
    lowest_scaled_log_frequency = math.log10(rydberg_frequency * inverse_level_squared)
    if scaled_log_frequency < lowest_scaled_log_frequency:
        return 0.0
    for table_index in range(1, 28):
        current_log_frequency = math.log10(
            (high_level_energy_offset_rydberg[table_index] + inverse_level_squared)
            * rydberg_frequency
        )
        if scaled_log_frequency > current_log_frequency:
            previous_log_frequency = (
                math.log10(
                    (
                        high_level_energy_offset_rydberg[table_index - 1]
                        + inverse_level_squared
                    )
                    * rydberg_frequency
                )
                if table_index - 1 >= 1
                else lowest_scaled_log_frequency
            )
            denominator = previous_log_frequency - current_log_frequency
            if denominator == 0.0:
                return 0.0
            interpolation_fraction = (
                scaled_log_frequency - current_log_frequency
            ) / denominator
            log10_cross_section_cm2 = (
                total_level_log10_cross_section_cm2[table_index - 1, 14]
                - total_level_log10_cross_section_cm2[table_index, 14]
            ) * interpolation_fraction + total_level_log10_cross_section_cm2[
                table_index, 14
            ]
            return float(
                np.exp(log10_cross_section_cm2 * NATURAL_LOG_10)
                / effective_charge_squared
            )
    return float(
        np.exp(total_level_log10_cross_section_cm2[28, 14] * NATURAL_LOG_10)
        / effective_charge_squared
    )


def _helium_ground_photoionization_cross_section(frequency_hz: float) -> float:
    """Ground-state neutral-helium photoionization cross-section."""
    if frequency_hz < 5.945209e15:
        return 0.0
    wavelength_angstrom = LIGHT_SPEED_ANGSTROM_PER_S / frequency_hz
    if wavelength_angstrom > 50.0:
        table_index = int(93.0 - (wavelength_angstrom - 50.0) / 5.0)
        table_index = min(92, max(2, table_index))
        table = _HELIUM_GROUND_CROSS_SECTION_50_TO_505_A
        return (
            (wavelength_angstrom - (92 - table_index) * 5 - 50)
            / 5.0
            * (table[table_index - 2] - table[table_index - 1])
            + table[table_index - 1]
        ) * 1.0e-18
    if wavelength_angstrom > 20.0:
        table_index = int(17.0 - (wavelength_angstrom - 20.0) / 2.0)
        table_index = min(16, max(2, table_index))
        table = _HELIUM_GROUND_CROSS_SECTION_20_TO_50_A
        return (
            (wavelength_angstrom - (16 - table_index) * 2 - 20)
            / 2.0
            * (table[table_index - 2] - table[table_index - 1])
            + table[table_index - 1]
        ) * 1.0e-18
    if wavelength_angstrom > 10.0:
        table_index = int(12.0 - (wavelength_angstrom - 10.0) / 1.0)
        table_index = min(11, max(2, table_index))
        table = _HELIUM_GROUND_CROSS_SECTION_10_TO_20_A
        return (
            (wavelength_angstrom - (11 - table_index) * 1 - 10)
            / 1.0
            * (table[table_index - 2] - table[table_index - 1])
            + table[table_index - 1]
        ) * 1.0e-18
    table_index = int(22.0 - wavelength_angstrom / 0.5)
    table_index = min(21, max(2, table_index))
    table = _HELIUM_GROUND_CROSS_SECTION_BELOW_10_A
    return (
        (wavelength_angstrom - (21 - table_index) * 0.5)
        / 0.5
        * (table[table_index - 2] - table[table_index - 1])
        + table[table_index - 1]
    ) * 1.0e-18


def _helium_singlet_s_cross_section(frequency_hz: float) -> float:
    if frequency_hz < 32033.214 * LIGHT_SPEED_CM_PER_S:
        return 0.0
    if frequency_hz > 2.4 * 109722.267 * LIGHT_SPEED_CM_PER_S:
        wavenumber_cm = frequency_hz / LIGHT_SPEED_CM_PER_S
        rydberg_offset = (wavenumber_cm - 32033.214) / 109722.267
        resonance_coordinate = 2.0 * (rydberg_offset - 2.612316) / 0.00322
        return (
            0.008175
            * (484940.0 / wavenumber_cm) ** 2.71
            * 8.067e-18
            * (resonance_coordinate + 76.21) ** 2
            / (1.0 + resonance_coordinate**2)
        )
    log10_frequency = math.log10(frequency_hz)
    table_index = 15
    for candidate_index in range(1, 16):
        if (
            log10_frequency
            > _HELIUM_NEUTRAL_2S_SINGLET_LOG10_FREQUENCY_HZ[candidate_index]
        ):
            table_index = candidate_index
            break
    log10_cross_section_cm2 = (
        log10_frequency - _HELIUM_NEUTRAL_2S_SINGLET_LOG10_FREQUENCY_HZ[table_index]
    ) / (
        _HELIUM_NEUTRAL_2S_SINGLET_LOG10_FREQUENCY_HZ[table_index - 1]
        - _HELIUM_NEUTRAL_2S_SINGLET_LOG10_FREQUENCY_HZ[table_index]
    ) * (
        _HELIUM_NEUTRAL_2S_SINGLET_LOG10_CROSS_SECTION_CM2[table_index - 1]
        - _HELIUM_NEUTRAL_2S_SINGLET_LOG10_CROSS_SECTION_CM2[table_index]
    ) + _HELIUM_NEUTRAL_2S_SINGLET_LOG10_CROSS_SECTION_CM2[table_index]
    return 10.0**log10_cross_section_cm2


def _helium_triplet_s_cross_section(frequency_hz: float) -> float:
    if frequency_hz < 38454.691 * LIGHT_SPEED_CM_PER_S:
        return 0.0
    if frequency_hz > 2.4 * 109722.267 * LIGHT_SPEED_CM_PER_S:
        wavenumber_cm = frequency_hz / LIGHT_SPEED_CM_PER_S
        rydberg_offset = (wavenumber_cm - 38454.691) / 109722.267
        resonance_coordinate = 2.0 * (rydberg_offset - 2.47898) / 0.000780
        return (
            0.01521
            * (470310.0 / wavenumber_cm) ** 3.12
            * 8.067e-18
            * (resonance_coordinate - 122.4) ** 2
            / (1.0 + resonance_coordinate**2)
        )
    log10_frequency = math.log10(frequency_hz)
    table_index = 15
    for candidate_index in range(1, 16):
        if (
            log10_frequency
            > _HELIUM_NEUTRAL_2S_TRIPLET_LOG10_FREQUENCY_HZ[candidate_index]
        ):
            table_index = candidate_index
            break
    log10_cross_section_cm2 = (
        log10_frequency - _HELIUM_NEUTRAL_2S_TRIPLET_LOG10_FREQUENCY_HZ[table_index]
    ) / (
        _HELIUM_NEUTRAL_2S_TRIPLET_LOG10_FREQUENCY_HZ[table_index - 1]
        - _HELIUM_NEUTRAL_2S_TRIPLET_LOG10_FREQUENCY_HZ[table_index]
    ) * (
        _HELIUM_NEUTRAL_2S_TRIPLET_LOG10_CROSS_SECTION_CM2[table_index - 1]
        - _HELIUM_NEUTRAL_2S_TRIPLET_LOG10_CROSS_SECTION_CM2[table_index]
    ) + _HELIUM_NEUTRAL_2S_TRIPLET_LOG10_CROSS_SECTION_CM2[table_index]
    return 10.0**log10_cross_section_cm2


def _helium_singlet_p_cross_section(frequency_hz: float) -> float:
    if frequency_hz < 27175.76 * LIGHT_SPEED_CM_PER_S:
        return 0.0
    if frequency_hz > 2.4 * 109722.267 * LIGHT_SPEED_CM_PER_S:
        wavenumber_cm = frequency_hz / LIGHT_SPEED_CM_PER_S
        rydberg_offset = (wavenumber_cm - 27175.76) / 109722.267
        singlet_s_coordinate = 2.0 * (rydberg_offset - 2.446534) / 0.01037
        singlet_d_coordinate = 2.0 * (rydberg_offset - 2.59427) / 0.00538
        return (
            0.0009487
            * (466750.0 / wavenumber_cm) ** 3.69
            * 8.067e-18
            * (
                (singlet_s_coordinate - 29.30) ** 2 / (1.0 + singlet_s_coordinate**2)
                + (singlet_d_coordinate + 172.4) ** 2 / (1.0 + singlet_d_coordinate**2)
            )
        )
    log10_frequency = math.log10(frequency_hz)
    table_index = 15
    for candidate_index in range(1, 16):
        if (
            log10_frequency
            > _HELIUM_NEUTRAL_2P_SINGLET_LOG10_FREQUENCY_HZ[candidate_index]
        ):
            table_index = candidate_index
            break
    log10_cross_section_cm2 = (
        log10_frequency - _HELIUM_NEUTRAL_2P_SINGLET_LOG10_FREQUENCY_HZ[table_index]
    ) / (
        _HELIUM_NEUTRAL_2P_SINGLET_LOG10_FREQUENCY_HZ[table_index - 1]
        - _HELIUM_NEUTRAL_2P_SINGLET_LOG10_FREQUENCY_HZ[table_index]
    ) * (
        _HELIUM_NEUTRAL_2P_SINGLET_LOG10_CROSS_SECTION_CM2[table_index - 1]
        - _HELIUM_NEUTRAL_2P_SINGLET_LOG10_CROSS_SECTION_CM2[table_index]
    ) + _HELIUM_NEUTRAL_2P_SINGLET_LOG10_CROSS_SECTION_CM2[table_index]
    return 10.0**log10_cross_section_cm2


def _helium_triplet_p_cross_section(frequency_hz: float) -> float:
    if frequency_hz < 29223.753 * LIGHT_SPEED_CM_PER_S:
        return 0.0
    log10_frequency = math.log10(frequency_hz)
    table_index = 15
    for candidate_index in range(1, 16):
        if (
            log10_frequency
            > _HELIUM_NEUTRAL_2P_TRIPLET_LOG10_FREQUENCY_HZ[candidate_index]
        ):
            table_index = candidate_index
            break
    log10_cross_section_cm2 = (
        log10_frequency - _HELIUM_NEUTRAL_2P_TRIPLET_LOG10_FREQUENCY_HZ[table_index]
    ) / (
        _HELIUM_NEUTRAL_2P_TRIPLET_LOG10_FREQUENCY_HZ[table_index - 1]
        - _HELIUM_NEUTRAL_2P_TRIPLET_LOG10_FREQUENCY_HZ[table_index]
    ) * (
        _HELIUM_NEUTRAL_2P_TRIPLET_LOG10_CROSS_SECTION_CM2[table_index - 1]
        - _HELIUM_NEUTRAL_2P_TRIPLET_LOG10_CROSS_SECTION_CM2[table_index]
    ) + _HELIUM_NEUTRAL_2P_TRIPLET_LOG10_CROSS_SECTION_CM2[table_index]
    return 10.0**log10_cross_section_cm2


def _neutral_helium_frequency_grids(
    continuum_tables, frequencies_hz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Frequency-only neutral-helium transition grids."""
    frequency_grid_hz = np.asarray(frequencies_hz, np.float64)
    transition_grid = np.zeros((10, frequency_grid_hz.size), dtype=np.float64)
    edge_masks = [
        frequency_grid_hz >= threshold_frequency
        for threshold_frequency in _HELIUM_NEUTRAL_EDGE_FREQUENCIES_HZ
    ]
    if np.any(edge_masks[0]):
        transition_grid[0, edge_masks[0]] = np.array(
            [
                _helium_ground_photoionization_cross_section(float(frequency_hz))
                for frequency_hz in frequency_grid_hz[edge_masks[0]]
            ]
        )
    if np.any(edge_masks[1]):
        transition_grid[1, edge_masks[1]] = np.array(
            [
                _helium_triplet_s_cross_section(float(frequency_hz))
                for frequency_hz in frequency_grid_hz[edge_masks[1]]
            ]
        )
    if np.any(edge_masks[2]):
        transition_grid[2, edge_masks[2]] = np.array(
            [
                _helium_singlet_s_cross_section(float(frequency_hz))
                for frequency_hz in frequency_grid_hz[edge_masks[2]]
            ]
        )
    if np.any(edge_masks[3]):
        transition_grid[3, edge_masks[3]] = np.array(
            [
                _helium_triplet_p_cross_section(float(frequency_hz))
                for frequency_hz in frequency_grid_hz[edge_masks[3]]
            ]
        )
    if np.any(edge_masks[4]):
        transition_grid[4, edge_masks[4]] = np.array(
            [
                _helium_singlet_p_cross_section(float(frequency_hz))
                for frequency_hz in frequency_grid_hz[edge_masks[4]]
            ]
        )
    karzas_latter_transitions = {
        5: (1.236439, 3, 0),
        6: (1.102898, 3, 0),
        7: (1.045499, 3, 1),
        8: (1.001427, 3, 2),
        9: (0.9926, 3, 1),
    }
    for transition_index, (
        effective_charge_squared,
        principal_quantum_number,
        angular_quantum_number,
    ) in karzas_latter_transitions.items():
        if np.any(edge_masks[transition_index]):
            transition_grid[transition_index, edge_masks[transition_index]] = np.array(
                [
                    _karzas_latter_cross_section(
                        continuum_tables.arrays,
                        float(frequency_hz),
                        effective_charge_squared,
                        principal_quantum_number,
                        angular_quantum_number,
                    )
                    for frequency_hz in frequency_grid_hz[edge_masks[transition_index]]
                ]
            )

    helium_rydberg_frequency_hz = 109722.273 * LIGHT_SPEED_CM_PER_S
    for ionization_limit_cm, levels in (
        (527490.06, ((171135.000, 4), (169087.0, 3), (166277.546, 2), (159856.069, 1))),
        (
            588451.59,
            (
                (186209.471, 9),
                (186101.0, 8),
                (185564.0, 7),
                (184864.0, 6),
                (183236.0, 5),
            ),
        ),
    ):
        for level_energy_cm, transition_index in levels:
            threshold_frequency_hz = (
                ionization_limit_cm - level_energy_cm
            ) * LIGHT_SPEED_CM_PER_S
            active_frequency_mask = frequency_grid_hz >= threshold_frequency_hz
            if np.any(active_frequency_mask):
                effective_charge_squared = (
                    threshold_frequency_hz / helium_rydberg_frequency_hz
                )
                transition_grid[transition_index, active_frequency_mask] += np.array(
                    [
                        _karzas_latter_cross_section(
                            continuum_tables.arrays,
                            float(frequency_hz),
                            effective_charge_squared,
                            1,
                            0,
                        )
                        for frequency_hz in frequency_grid_hz[active_frequency_mask]
                    ]
                )

    transition_autoionization_grid = np.zeros(
        (28, frequency_grid_hz.size), dtype=np.float64
    )
    for principal_quantum_number in range(4, 28):
        effective_charge_squared = 4.0 - 3.0 / (
            principal_quantum_number * principal_quantum_number
        )
        transition_autoionization_grid[principal_quantum_number, :] = np.array(
            [
                _karzas_latter_cross_section(
                    continuum_tables.arrays,
                    float(frequency_hz),
                    effective_charge_squared,
                    1,
                    0,
                )
                for frequency_hz in frequency_grid_hz
            ]
        )
    return transition_grid, transition_autoionization_grid


_MAGNESIUM_SINGLY_IONIZED_LEVEL_ENERGY_CM = np.array(
    [
        112197.0,
        108900.0,
        103705.66,
        103689.89,
        103419.82,
        97464.32,
        92790.51,
        93799.70,
        93310.80,
        80639.85,
        69804.95,
        71490.54,
        35730.36,
        0.0,
    ],
    dtype=np.float64,
)
_MAGNESIUM_SINGLY_IONIZED_LEVEL_STATISTICAL_WEIGHT = np.array(
    [98.0, 72.0, 18.0, 14.0, 10.0, 6.0, 2.0, 14.0, 10.0, 6.0, 2.0, 10.0, 6.0, 2.0],
    dtype=np.float64,
)
_MAGNESIUM_SINGLY_IONIZED_QUANTUM_NUMBERS = (
    (7, 7),
    (6, 6),
    (5, 4),
    (5, 3),
    (5, 2),
    (5, 1),
    (5, 0),
    (4, 3),
    (4, 2),
    (4, 1),
    (4, 0),
    (3, 2),
    (3, 1),
)
_MAGNESIUM_SINGLY_IONIZED_EFFECTIVE_CHARGE_NUMERATOR = np.array(
    [49.0, 36.0, 25.0, 25.0, 25.0, 25.0, 25.0, 16.0, 16.0, 16.0, 16.0, 9.0, 9.0],
    dtype=np.float64,
)
_MAGNESIUM_SINGLY_IONIZED_IONIZATION_LIMIT_CM = 121267.61
_MAGNESIUM_RYDBERG_WAVENUMBER_CM = 109732.298
_MAGNESIUM_SINGLY_IONIZED_CHARGE = 2.0
_MAGNESIUM_SINGLY_IONIZED_FREE_FREE_THRESHOLD_CM = (
    _MAGNESIUM_SINGLY_IONIZED_IONIZATION_LIMIT_CM
    - _MAGNESIUM_RYDBERG_WAVENUMBER_CM * _MAGNESIUM_SINGLY_IONIZED_CHARGE** 2 / (8.0**2)
)


def _light_element_frequency_grids(
    continuum_tables, frequencies_hz: np.ndarray
) -> dict[str, np.ndarray]:
    """Frequency-only light-element photoionization grids: N I, O I, Mg II, Ca II."""
    frequency_grid_hz = np.asarray(frequencies_hz, np.float64)
    wavenumber_cm = frequency_grid_hz / LIGHT_SPEED_CM_PER_S
    nitrogen_edge_cross_sections = np.zeros(
        (3, frequency_grid_hz.size), dtype=np.float64
    )
    nitrogen_seaton_edges = (
        (3.517915e15, 1.142e-17, 2.0, 4.29),
        (2.941534e15, 4.41e-18, 1.5, 3.85),
        (2.653317e15, 4.2e-18, 1.5, 4.34),
    )
    for edge_index, (
        edge_frequency_hz,
        threshold_cross_section,
        power_law_index,
        seaton_shape,
    ) in enumerate(nitrogen_seaton_edges):
        nitrogen_edge_cross_sections[edge_index] = np.array(
            [
                _seaton_photoionization_cross_section(
                    edge_frequency_hz,
                    threshold_cross_section,
                    power_law_index,
                    seaton_shape,
                    float(frequency_hz),
                )
                for frequency_hz in frequency_grid_hz
            ]
        )
    oxygen_911_cross_section = np.array(
        [
            _seaton_photoionization_cross_section(
                3.28805e15, 2.94e-18, 1.0, 2.66, float(frequency_hz)
            )
            for frequency_hz in frequency_grid_hz
        ]
    )
    calcium_edge_cross_sections = np.zeros(
        (3, frequency_grid_hz.size), dtype=np.float64
    )
    calcium_edge_cross_sections[0] = np.where(
        frequency_grid_hz >= 2.870454e15,
        5.4e-20 * (2.870454e15 / frequency_grid_hz) ** 3,
        0.0,
    )
    calcium_edge_cross_sections[1] = np.where(
        frequency_grid_hz >= 2.460127e15,
        1.64e-17 * np.sqrt(2.460127e15 / frequency_grid_hz),
        0.0,
    )
    calcium_edge_cross_sections[2] = np.array(
        [
            _seaton_photoionization_cross_section(
                2.110779e15, 4.13e-18, 3.0, 0.69, float(frequency_hz)
            )
            for frequency_hz in frequency_grid_hz
        ]
    )

    magnesium_ionized_cross_sections = np.zeros(
        (14, frequency_grid_hz.size), dtype=np.float64
    )
    for transition_index, (
        principal_quantum_number,
        angular_quantum_number,
    ) in enumerate(_MAGNESIUM_SINGLY_IONIZED_QUANTUM_NUMBERS):
        threshold_wavenumber = (
            _MAGNESIUM_SINGLY_IONIZED_IONIZATION_LIMIT_CM
            - _MAGNESIUM_SINGLY_IONIZED_LEVEL_ENERGY_CM[transition_index]
        )
        active_frequency_mask = wavenumber_cm >= threshold_wavenumber
        if np.any(active_frequency_mask):
            effective_charge_squared = (
                _MAGNESIUM_SINGLY_IONIZED_EFFECTIVE_CHARGE_NUMERATOR[transition_index]
                / _MAGNESIUM_RYDBERG_WAVENUMBER_CM
                * threshold_wavenumber
            )
            magnesium_ionized_cross_sections[
                transition_index, active_frequency_mask
            ] = np.array(
                [
                    _karzas_latter_cross_section(
                        continuum_tables.arrays,
                        float(frequency_hz),
                        effective_charge_squared,
                        principal_quantum_number,
                        angular_quantum_number,
                    )
                    for frequency_hz in frequency_grid_hz[active_frequency_mask]
                ]
            )
    ground_threshold_wavenumber = (
        _MAGNESIUM_SINGLY_IONIZED_IONIZATION_LIMIT_CM
        - _MAGNESIUM_SINGLY_IONIZED_LEVEL_ENERGY_CM[13]
    )
    ground_edge_mask = wavenumber_cm >= ground_threshold_wavenumber
    if np.any(ground_edge_mask):
        threshold_ratio = ground_threshold_wavenumber / np.maximum(
            wavenumber_cm[ground_edge_mask],
            1.0e-300,
        )
        magnesium_ionized_cross_sections[13, ground_edge_mask] = 0.14e-18 * (
            6.700 * threshold_ratio**4 - 5.700 * threshold_ratio**5
        )
    return {
        "nitrogen_edge_cross_sections": nitrogen_edge_cross_sections,
        "oxygen_911_cross_section": oxygen_911_cross_section,
        "calcium_edge_cross_sections": calcium_edge_cross_sections,
        "magnesium_ionized_cross_section_rows": magnesium_ionized_cross_sections,
    }


def _rayleigh_polarizability_factor(continuum_arrays, frequency_hz):
    """Gavrila Rayleigh polarizability factor G(nu), evaluated on host fp64."""
    gavrila_main = continuum_arrays["hydrogen_rayleigh_gavrila_main_table"]
    gavrila_ab = continuum_arrays["hydrogen_rayleigh_gavrila_ab_table"]
    gavrila_bc = continuum_arrays["hydrogen_rayleigh_gavrila_bc_table"]
    gavrila_cd = continuum_arrays["hydrogen_rayleigh_gavrila_cd_table"]
    gavrila_lyman = continuum_arrays["hydrogen_rayleigh_gavrila_lyman_continuum_table"]
    gavrila_lyman_frequency_grid = continuum_arrays[
        "hydrogen_rayleigh_gavrila_lyman_frequency_ratio_grid"
    ]
    lyman_frequency_hz = 3.288051e15
    main_interval_hz = 3.288051e13

    if frequency_hz < lyman_frequency_hz * 0.01:
        return gavrila_main[0] * (frequency_hz / main_interval_hz) ** 2
    if frequency_hz <= lyman_frequency_hz * 0.74:
        one_based_index = int(frequency_hz / main_interval_hz)
        one_based_index = max(1, min(one_based_index + 1, 74))
        if one_based_index >= len(gavrila_main):
            one_based_index = len(gavrila_main) - 1
        if one_based_index > 1:
            return gavrila_main[one_based_index - 2] + (
                gavrila_main[one_based_index - 1] - gavrila_main[one_based_index - 2]
            ) / main_interval_hz * (
                frequency_hz - (one_based_index - 1) * main_interval_hz
            )
        return gavrila_main[0]
    if frequency_hz < lyman_frequency_hz * 0.755:
        return 15.57
    if frequency_hz <= lyman_frequency_hz * 0.885:
        interval_hz = 1.644026e13
        one_based_index = max(
            1,
            min(int((frequency_hz - lyman_frequency_hz * 0.755) / interval_hz) + 2, 27),
        )
        if one_based_index >= len(gavrila_ab):
            one_based_index = len(gavrila_ab) - 1
        if one_based_index > 1:
            # Preserve the validated source-table left edge; it differs from the denominator.
            left_frequency_hz = (
                lyman_frequency_hz * 0.755 + (one_based_index - 2) * 1.664026e13
            )
            return gavrila_ab[one_based_index - 2] + (
                gavrila_ab[one_based_index - 1] - gavrila_ab[one_based_index - 2]
            ) / interval_hz * (frequency_hz - left_frequency_hz)
        return gavrila_ab[0]
    if frequency_hz < lyman_frequency_hz * 0.890:
        return 8.0
    if frequency_hz <= lyman_frequency_hz * 0.936:
        interval_hz = 0.657610e13
        one_based_index = max(
            1,
            min(int((frequency_hz - lyman_frequency_hz * 0.890) / interval_hz) + 2, 24),
        )
        if one_based_index >= len(gavrila_bc):
            one_based_index = len(gavrila_bc) - 1
        if one_based_index > 1:
            left_frequency_hz = (
                lyman_frequency_hz * 0.890 + (one_based_index - 2) * interval_hz
            )
            return gavrila_bc[one_based_index - 2] + (
                gavrila_bc[one_based_index - 1] - gavrila_bc[one_based_index - 2]
            ) / interval_hz * (frequency_hz - left_frequency_hz)
        return gavrila_bc[0]
    if frequency_hz < lyman_frequency_hz * 0.938:
        return 9.0
    if frequency_hz <= lyman_frequency_hz * 0.959:
        interval_hz = 0.3288051e13
        one_based_index = max(
            1,
            min(int((frequency_hz - lyman_frequency_hz * 0.938) / interval_hz) + 2, 22),
        )
        if one_based_index >= len(gavrila_cd):
            one_based_index = len(gavrila_cd) - 1
        if one_based_index > 1:
            left_frequency_hz = (
                lyman_frequency_hz * 0.938 + (one_based_index - 2) * interval_hz
            )
            return gavrila_cd[one_based_index - 2] + (
                gavrila_cd[one_based_index - 1] - gavrila_cd[one_based_index - 2]
            ) / interval_hz * (frequency_hz - left_frequency_hz)
        return gavrila_cd[0]
    if frequency_hz <= lyman_frequency_hz:
        return gavrila_lyman[0]
    return _parabolic_interpolate(
        gavrila_lyman_frequency_grid,
        gavrila_lyman,
        np.array([frequency_hz / lyman_frequency_hz]),
    )[0]


def _silicon_singly_ionized_peach_frequency_row(
    continuum_arrays, frequency_hz, natural_log_frequency
):
    """Frequency-only Si II Peach opacity row evaluated on host fp64.

    Precomputing this row once per sampled frequency leaves the per-depth
    temperature blend unchanged while avoiding repeated bracket searches.
    """
    peach_rows = continuum_arrays["silicon_singly_ionized_peach_cross_section_table"]
    frequency_threshold_hz = continuum_arrays[
        "silicon_singly_ionized_peach_threshold_frequencies_hz"
    ]
    natural_log_frequency_grid = continuum_arrays[
        "silicon_singly_ionized_peach_natural_log_frequency_grid"
    ]
    frequency_bin = 0
    for threshold_index in range(7):
        if frequency_hz > frequency_threshold_hz[threshold_index]:
            frequency_bin = threshold_index + 1
            break
    else:
        frequency_bin = 8
    frequency_weight = (
        (natural_log_frequency - natural_log_frequency_grid[frequency_bin - 1])
        / (
            natural_log_frequency_grid[frequency_bin]
            - natural_log_frequency_grid[frequency_bin - 1]
        )
        if 0 < frequency_bin < 9
        else 0.0
    )
    if frequency_bin > 2:
        frequency_bin = 2 * frequency_bin - 2
    frequency_bin = min(frequency_bin, 13)
    complementary_weight = 1.0 - frequency_weight
    if frequency_bin < 14:
        frequency_row = (
            peach_rows[frequency_bin] * frequency_weight
            + peach_rows[frequency_bin - 1] * complementary_weight
            if frequency_bin > 0
            else peach_rows[0]
        )
    else:
        frequency_row = peach_rows[13]
    return frequency_row


def _silicon_singly_ionized_peach_opacity(
    continuum_arrays,
    frequency_hz,
    natural_log_frequency,
    temperature_host,
    natural_log_temperature_host,
    frequency_row=None,
):
    """Si II Peach-table opacity, depth-batched on host fp64.

    ``frequency_row`` may provide the precomputed frequency-only Peach row.
    """
    natural_log_temperature_grid = continuum_arrays[
        "silicon_singly_ionized_peach_natural_log_temperature_grid"
    ]
    n_layers = temperature_host.size
    temperature_index = np.clip((temperature_host / 2000.0).astype(int) - 4, 1, 5)
    temperature_weight = (
        natural_log_temperature_host
        - natural_log_temperature_grid[temperature_index - 1]
    ) / (
        natural_log_temperature_grid[temperature_index]
        - natural_log_temperature_grid[temperature_index - 1]
    )
    if frequency_row is None:
        frequency_row = _silicon_singly_ionized_peach_frequency_row(
            continuum_arrays,
            frequency_hz,
            natural_log_frequency,
        )
    result = np.zeros(n_layers)
    for depth_index in range(n_layers):
        row_index = temperature_index[depth_index] - 1
        log10_cross_section_cm2 = (
            frequency_row[row_index] * (1.0 - temperature_weight[depth_index])
            + frequency_row[row_index + 1] * temperature_weight[depth_index]
            if row_index < 5
            else frequency_row[5]
        )
        result[depth_index] = np.exp(log10_cross_section_cm2) * 6.0
    return result


# Coulomb free-free Gaunt factor.


def _coulomb_freefree_gaunt(
    continuum_tables,
    charge,
    natural_log_frequency,
    temperature,
    natural_log_temperature,
    energy_first_layout=False,
):
    """Coulomb free-free Gaunt factor for one frequency and all depths."""
    if charge < 1 or charge > 6:
        return torch.ones_like(temperature)
    coulomb_table = continuum_tables.coulomb_freefree_gaunt_table_device
    charge_log_offset = float(
        continuum_tables.arrays["coulomb_freefree_charge_log_offset"][charge - 1]
    )
    gamma_log = 10.39638 - natural_log_temperature / 1.15129 + charge_log_offset
    photon_energy_log = (
        natural_log_frequency - natural_log_temperature
    ) / 1.15129 - 20.63764
    gamma_index = torch.clamp((gamma_log + 7.0).to(torch.int64), 1, 10)
    energy_index = torch.clamp((photon_energy_log + 9.0).to(torch.int64), 1, 11)
    gamma_weight = gamma_log - (gamma_index.to(temperature.dtype) - 7.0)
    energy_weight = photon_energy_log - (energy_index.to(temperature.dtype) - 9.0)
    gamma_axis_index = gamma_index - 1
    energy_axis_index = energy_index - 1
    n_columns = coulomb_table.shape[1]

    def gather_table(row_index, column_index):
        return coulomb_table.reshape(-1)[row_index * n_columns + column_index]

    if energy_first_layout:
        # Alternate table layout: table[energy, gamma].
        corner00 = gather_table(energy_axis_index, gamma_axis_index)
        corner01 = gather_table(energy_axis_index + 1, gamma_axis_index)
        corner10 = gather_table(energy_axis_index, gamma_axis_index + 1)
        corner11 = gather_table(energy_axis_index + 1, gamma_axis_index + 1)
        return (1.0 - gamma_weight) * (
            (1.0 - energy_weight) * corner00 + energy_weight * corner01
        ) + gamma_weight * ((1.0 - energy_weight) * corner10 + energy_weight * corner11)

    # Table layout: [gamma, energy], with the validated edge guards.
    corner00 = gather_table(gamma_axis_index, energy_axis_index)
    energy_axis_next = torch.minimum(
        energy_axis_index + 1, torch.tensor(10, device=temperature.device)
    )
    gamma_axis_next = torch.minimum(
        gamma_axis_index + 1, torch.tensor(11, device=temperature.device)
    )
    corner01 = torch.where(
        energy_index < 11, gather_table(gamma_axis_index, energy_axis_next), corner00
    )
    corner10 = torch.where(
        gamma_index < 10, gather_table(gamma_axis_next, energy_axis_index), corner00
    )
    corner11 = torch.where(
        (gamma_index < 10) & (energy_index < 11),
        gather_table(gamma_axis_next, energy_axis_next),
        corner00,
    )
    return (1.0 - gamma_weight) * (
        (1.0 - energy_weight) * corner00 + energy_weight * corner01
    ) + gamma_weight * ((1.0 - energy_weight) * corner10 + energy_weight * corner11)


def _coulomb_freefree_gaunt_grid(
    continuum_tables,
    charge,
    log_frequency_vector,
    temperature,
    natural_log_temperature,
    energy_first_layout=False,
):
    """Batched Coulomb free-free Gaunt factor, returned as ``(frequency, depth)``."""
    n_frequencies = log_frequency_vector.shape[0]
    n_layers = temperature.shape[0]
    if charge < 1 or charge > 6:
        return torch.ones(
            n_frequencies, n_layers, dtype=temperature.dtype, device=temperature.device
        )
    coulomb_table = continuum_tables.coulomb_freefree_gaunt_table_device
    charge_log_offset = float(
        continuum_tables.arrays["coulomb_freefree_charge_log_offset"][charge - 1]
    )
    gamma_log = 10.39638 - natural_log_temperature / 1.15129 + charge_log_offset
    photon_energy_log = (
        log_frequency_vector[:, None] - natural_log_temperature[None, :]
    ) / 1.15129 - 20.63764
    gamma_index = torch.clamp((gamma_log + 7.0).to(torch.int64), 1, 10)
    energy_index = torch.clamp((photon_energy_log + 9.0).to(torch.int64), 1, 11)
    gamma_weight = gamma_log - (gamma_index.to(temperature.dtype) - 7.0)
    energy_weight = photon_energy_log - (energy_index.to(temperature.dtype) - 9.0)
    gamma_axis_index = gamma_index - 1
    energy_axis_index = energy_index - 1
    n_columns = coulomb_table.shape[1]

    gamma_axis_batch = gamma_axis_index[None, :].expand(n_frequencies, -1)
    gamma_index_batch = gamma_index[None, :].expand(n_frequencies, -1)
    gamma_weight_batch = gamma_weight[None, :].expand(n_frequencies, -1)

    def gather_table(row_index, column_index):
        return coulomb_table.reshape(-1)[row_index * n_columns + column_index]

    if energy_first_layout:
        corner00 = gather_table(energy_axis_index, gamma_axis_batch)
        corner01 = gather_table(energy_axis_index + 1, gamma_axis_batch)
        corner10 = gather_table(energy_axis_index, gamma_axis_batch + 1)
        corner11 = gather_table(energy_axis_index + 1, gamma_axis_batch + 1)
        return (1.0 - gamma_weight_batch) * (
            (1.0 - energy_weight) * corner00 + energy_weight * corner01
        ) + gamma_weight_batch * (
            (1.0 - energy_weight) * corner10 + energy_weight * corner11
        )

    corner00 = gather_table(gamma_axis_batch, energy_axis_index)
    energy_axis_next = torch.minimum(
        energy_axis_index + 1, torch.tensor(10, device=temperature.device)
    )
    gamma_axis_next = torch.minimum(
        gamma_axis_batch + 1, torch.tensor(11, device=temperature.device)
    )
    corner01 = torch.where(
        energy_index < 11, gather_table(gamma_axis_batch, energy_axis_next), corner00
    )
    corner10 = torch.where(
        gamma_index_batch < 10,
        gather_table(gamma_axis_next, energy_axis_index),
        corner00,
    )
    corner11 = torch.where(
        (gamma_index_batch < 10) & (energy_index < 11),
        gather_table(gamma_axis_next, energy_axis_next),
        corner00,
    )
    return (1.0 - gamma_weight_batch) * (
        (1.0 - energy_weight) * corner00 + energy_weight * corner01
    ) + gamma_weight_batch * (
        (1.0 - energy_weight) * corner10 + energy_weight * corner11
    )


# Population gathering.


def build_pops(atmosphere: dict, device=None, dtype=None) -> dict:
    """Gather the per-depth populations each continuum term reads.

    `atmosphere` is a dict-like of the native atmosphere arrays. Returns device
    tensors plus host fp64 temperature arrays for the discrete Si II / Peach
    lookups.
    """
    if device is None:
        device = _device()
    if dtype is None:
        dtype = DEFAULT_DTYPE

    def to_device_tensor(array_like):
        return torch.as_tensor(np.asarray(array_like), dtype=dtype, device=device)

    partition_normalized_populations = np.asarray(
        atmosphere["partition_normalized_populations"]
    )
    ion_stage_populations = np.asarray(atmosphere["ion_stage_populations"])
    n_depths = np.asarray(atmosphere["temperature"]).size
    if "hydrogen_ionized_population" in atmosphere:
        hydrogen_ionized_population = np.asarray(
            atmosphere["hydrogen_ionized_population"],
            dtype=np.float64,
        )
    else:
        hydrogen_ionized_population = np.asarray(
            atmosphere["hydrogen_partition_normalized_ion_stage_populations"],
            dtype=np.float64,
        )[:, 1]

    pops = dict(
        temperature=to_device_tensor(atmosphere["temperature"]),
        mass_density=to_device_tensor(atmosphere["mass_density"]),
        electron_density=to_device_tensor(atmosphere["electron_density"]),
        hydrogen_partition_normalized_ion_stage_populations=to_device_tensor(
            atmosphere["hydrogen_partition_normalized_ion_stage_populations"]
        ),
        hydrogen_neutral_population=to_device_tensor(
            atmosphere["hydrogen_neutral_population"]
        ),
        hydrogen_ionized_population=to_device_tensor(hydrogen_ionized_population),
        helium_neutral_partition_normalized_population=to_device_tensor(
            partition_normalized_populations[:, 0, 1]
        ),
        helium_singly_ionized_partition_normalized_population=to_device_tensor(
            partition_normalized_populations[:, 1, 1]
        ),
        helium_doubly_ionized_partition_normalized_population=to_device_tensor(
            partition_normalized_populations[:, 2, 1]
        ),
        helium_neutral_population=to_device_tensor(
            atmosphere["helium_neutral_population"]
        ),
        helium_singly_ionized_population=to_device_tensor(
            atmosphere["helium_singly_ionized_population"]
        ),
        carbon_partition_normalized_ion_stage_populations=to_device_tensor(
            atmosphere["carbon_partition_normalized_ion_stage_populations"]
        ),
        magnesium_neutral_partition_normalized_population=to_device_tensor(
            atmosphere["magnesium_neutral_partition_normalized_population"]
        ),
        aluminum_neutral_partition_normalized_population=to_device_tensor(
            atmosphere["aluminum_neutral_partition_normalized_population"]
        ),
        silicon_neutral_partition_normalized_population=to_device_tensor(
            atmosphere["silicon_neutral_partition_normalized_population"]
        ),
        iron_neutral_partition_normalized_population=to_device_tensor(
            atmosphere["iron_neutral_partition_normalized_population"]
        ),
        nitrogen_neutral_partition_normalized_population=to_device_tensor(
            partition_normalized_populations[:, 0, 6]
        ),
        oxygen_neutral_partition_normalized_population=to_device_tensor(
            partition_normalized_populations[:, 0, 7]
        ),
        magnesium_singly_ionized_partition_normalized_population=to_device_tensor(
            partition_normalized_populations[:, 1, 11]
        ),
        silicon_singly_ionized_partition_normalized_population=to_device_tensor(
            partition_normalized_populations[:, 1, 13]
        ),
        calcium_singly_ionized_partition_normalized_population=to_device_tensor(
            partition_normalized_populations[:, 1, 19]
        ),
    )
    hot_metal_population_slots = np.zeros((n_depths, 21))
    hot_metal_population_slots[:, 0:4] = partition_normalized_populations[:, 0:4, 5]
    hot_metal_population_slots[:, 4:9] = partition_normalized_populations[:, 0:5, 6]
    hot_metal_population_slots[:, 9:15] = partition_normalized_populations[:, 0:6, 7]
    hot_metal_population_slots[:, 15:21] = partition_normalized_populations[:, 0:6, 9]
    charge_square_population_sum = np.zeros((n_depths, 5))
    for element_index in (5, 6, 7, 9, 11, 13, 15, 25):
        for charge in range(1, 6):
            charge_square_population_sum[:, charge - 1] += (
                charge * charge
            ) * ion_stage_populations[:, charge, element_index]
    pops["hot_metal_populations"] = to_device_tensor(hot_metal_population_slots)
    pops["charge_square_population_sum"] = to_device_tensor(
        charge_square_population_sum
    )

    # Si II Peach interpolation uses the host fp64 temperature brackets.
    temperature_host = np.asarray(atmosphere["temperature"], dtype=np.float64)
    pops["_temperature_host"] = temperature_host
    pops["_natural_log_temperature_host"] = np.log(np.maximum(temperature_host, 1e-10))
    return pops


def pops_from_population_state(
    population_state, temperature, mass_density, device=None, dtype=None
) -> dict:
    """Adapt an EOS population solve into the continuum population dictionary."""
    atmosphere = dict(
        temperature=np.asarray(temperature, np.float64),
        mass_density=np.asarray(mass_density, np.float64),
        electron_density=np.asarray(population_state.electron_density, np.float64),
        partition_normalized_populations=np.asarray(
            population_state.partition_normalized_populations, np.float64
        ),
        ion_stage_populations=np.asarray(
            population_state.ion_stage_populations, np.float64
        ),
        hydrogen_partition_normalized_ion_stage_populations=np.asarray(
            population_state.hydrogen_partition_normalized_ion_stage_populations,
            np.float64,
        ),
        hydrogen_neutral_population=np.asarray(
            population_state.hydrogen_neutral_population, np.float64
        ),
        hydrogen_ionized_population=np.asarray(
            population_state.hydrogen_ionized_population, np.float64
        ),
        helium_neutral_population=np.asarray(
            population_state.helium_neutral_population, np.float64
        ),
        helium_singly_ionized_population=np.asarray(
            population_state.helium_singly_ionized_population, np.float64
        ),
        carbon_partition_normalized_ion_stage_populations=np.asarray(
            population_state.carbon_partition_normalized_ion_stage_populations,
            np.float64,
        ),
        magnesium_neutral_partition_normalized_population=np.asarray(
            population_state.magnesium_neutral_partition_normalized_population,
            np.float64,
        ),
        aluminum_neutral_partition_normalized_population=np.asarray(
            population_state.aluminum_neutral_partition_normalized_population,
            np.float64,
        ),
        silicon_neutral_partition_normalized_population=np.asarray(
            population_state.silicon_neutral_partition_normalized_population, np.float64
        ),
        iron_neutral_partition_normalized_population=np.asarray(
            population_state.iron_neutral_partition_normalized_population, np.float64
        ),
    )
    return build_pops(atmosphere, device=device, dtype=dtype)


# Per-source opacity terms, depth-batched.


def _planck_nu(frequency_hz, temperature):
    planck_prefactor = 2.0 * PLANCK_ERG_SECOND / LIGHT_SPEED_CM_PER_S**2
    photon_energy_over_kt = (
        PLANCK_ERG_SECOND * frequency_hz / (BOLTZMANN_ERG_PER_K * temperature)
    )
    planck_source = torch.where(
        photon_energy_over_kt < 1e-6,
        2.0
        * BOLTZMANN_ERG_PER_K
        * temperature
        * frequency_hz**2
        / LIGHT_SPEED_CM_PER_S**2,
        planck_prefactor
        * frequency_hz**3
        / torch.expm1(
            torch.where(
                photon_energy_over_kt < 1e-6,
                torch.ones_like(photon_energy_over_kt),
                photon_energy_over_kt,
            )
        ),
    )
    return torch.where(
        torch.isfinite(planck_source), planck_source, torch.zeros_like(planck_source)
    )


def _hydrogen_partition(continuum_tables, temperature):
    thermal_energy_ev = REFERENCE_BOLTZMANN_EV_PER_K * temperature
    partition_function = torch.zeros_like(temperature)
    for level_index in range(HYDROGEN_MAXIMUM_EXPLICIT_LEVEL):
        partition_function = partition_function + float(
            continuum_tables.hydrogen_neutral_level_statistical_weight[level_index]
        ) * torch.exp(
            -float(continuum_tables.hydrogen_neutral_level_energy_ev[level_index])
            / thermal_energy_ev
        )
    return partition_function


def _hminus_ff_table(continuum_tables, frequency_hz):
    """Frequency-only H-minus free-free table sampled over the temperature grid.

    The wavelength interpolation depends only on frequency, so this vector is
    precomputed once per synthesis grid and reused by the depth-dependent stage.
    """
    wavelength_nm = LIGHT_SPEED_NM_PER_S / frequency_hz
    log_wavelength = math.log(wavelength_nm)
    return np.array(
        [
            math.exp(
                _linear_interpolate(
                    continuum_tables.hminus_freefree_log_wavelength_grid,
                    continuum_tables.hminus_freefree_log_table[:, temperature_index],
                    np.array([log_wavelength]),
                )[0]
            )
            for temperature_index in range(
                continuum_tables.hminus_freefree_temperature_count
            )
        ]
    )


def _hminus_bf_scalar(continuum_tables, frequency_hz):
    """Frequency-only H-minus bound-free cross-section."""
    if frequency_hz > 1.82365e14:
        wavelength_nm = LIGHT_SPEED_NM_PER_S / frequency_hz
        return float(
            _parabolic_interpolate(
                continuum_tables.arrays["hminus_boundfree_wavelength_nm"],
                continuum_tables.arrays["hminus_boundfree_cross_section_cm2"],
                np.array([wavelength_nm]),
            )[0]
        )
    return 0.0


def _hminus_opacity(
    continuum_tables,
    frequency_hz,
    pops,
    photon_boltzmann_factor,
    stimulated_emission,
    hminus_freefree_by_theta=None,
    hminus_boundfree_cross_section_cm2=None,
):
    """H- bound-free plus free-free opacity in cm^2/g per layer.

    Optional precomputed frequency-only tables skip wavelength interpolation; the
    per-depth theta interpolation and opacity arithmetic remain unchanged.
    """
    temperature = pops["temperature"]
    mass_density_safe = torch.clamp(pops["mass_density"], min=1e-30)
    electron_density = pops["electron_density"]
    hydrogen_neutral_partition_normalized_population = pops[
        "hydrogen_partition_normalized_ion_stage_populations"
    ][:, 0]
    dtype = temperature.dtype
    thermal_energy_ev = temperature * REFERENCE_BOLTZMANN_EV_PER_K
    theta = 5040.0 / temperature

    hminus_population_factor = (
        torch.exp(0.754209 / thermal_energy_ev)
        / (2.0 * REFERENCE_SAHA_COEFFICIENT * temperature * torch.sqrt(temperature))
        * hydrogen_neutral_partition_normalized_population
        * electron_density
    )

    # The theta grid is decreasing with depth, so each layer keeps the single-point
    # interpolation used by the validated scalar path.
    if hminus_freefree_by_theta is None:
        hminus_freefree_by_theta = _hminus_ff_table(continuum_tables, frequency_hz)
    theta_np = theta.detach().cpu().numpy().astype(np.float64)
    theta_grid = continuum_tables.arrays["hminus_freefree_theta_grid"]
    freefree_theta_values = np.array(
        [
            _linear_interpolate(
                theta_grid, hminus_freefree_by_theta, np.array([theta_np[layer]])
            )[0]
            for layer in range(theta_np.size)
        ]
    )
    freefree_theta_tensor = torch.as_tensor(
        freefree_theta_values,
        dtype=dtype,
        device=temperature.device,
    )
    hminus_freefree = (
        freefree_theta_tensor
        * hydrogen_neutral_partition_normalized_population
        * 2.0
        * electron_density
        / mass_density_safe
        * 1e-26
    )

    if hminus_boundfree_cross_section_cm2 is None:
        hminus_boundfree_cross_section_cm2 = _hminus_bf_scalar(
            continuum_tables, frequency_hz
        )
    hminus_boundfree = (
        hminus_boundfree_cross_section_cm2
        * 1e-18
        * (1.0 - photon_boltzmann_factor)
        * hminus_population_factor
        / mass_density_safe
    )
    return hminus_boundfree + hminus_freefree


def _hminus_opacity_grid(
    continuum_tables, pops, stimulated_emission_grid, frequency_invariants
):
    """Batched H- opacity on ``(depth, frequency)`` for the sampled grid.

    The scalar routine is intentionally preserved for the bit-exact fallback.  This path
    keeps the same per-cell arithmetic order where practical, but evaluates the
    temperature-dependent theta interpolation for every frequency in one NumPy gather.
    """
    temperature = pops["temperature"]
    mass_density_safe = torch.clamp(pops["mass_density"], min=1e-30)
    electron_density = pops["electron_density"]
    hydrogen_neutral_partition_normalized_population = pops[
        "hydrogen_partition_normalized_ion_stage_populations"
    ][:, 0]
    dtype = temperature.dtype
    device = temperature.device
    thermal_energy_ev = temperature * REFERENCE_BOLTZMANN_EV_PER_K
    theta = 5040.0 / temperature

    hminus_population_factor = (
        torch.exp(0.754209 / thermal_energy_ev)
        / (2.0 * REFERENCE_SAHA_COEFFICIENT * temperature * torch.sqrt(temperature))
        * hydrogen_neutral_partition_normalized_population
        * electron_density
    )

    # Vectorized equivalent of the single-depth theta interpolation.  The
    # `side="right"` search reproduces the monotonic cursor interval choice.
    theta_np = theta.detach().cpu().numpy().astype(np.float64)
    theta_grid = continuum_tables.arrays["hminus_freefree_theta_grid"]
    theta_hi = np.searchsorted(theta_grid, theta_np, side="right")
    theta_hi = np.clip(theta_hi, 1, theta_grid.size - 1)
    theta_lo = theta_hi - 1
    theta_span = theta_grid[theta_hi] - theta_grid[theta_lo]
    theta_weight = np.zeros_like(theta_np)
    valid_span = np.abs(theta_span) >= 1e-40
    theta_weight[valid_span] = (
        theta_np[valid_span] - theta_grid[theta_lo[valid_span]]
    ) / theta_span[valid_span]
    freefree_lo = frequency_invariants.hminus_freefree_rows[:, theta_lo]
    freefree_hi = frequency_invariants.hminus_freefree_rows[:, theta_hi]
    freefree_by_frequency_depth = np.where(
        valid_span[None, :],
        freefree_lo + (freefree_hi - freefree_lo) * theta_weight[None, :],
        freefree_lo,
    )
    freefree_theta_tensor = torch.as_tensor(
        np.ascontiguousarray(freefree_by_frequency_depth.T),
        dtype=dtype,
        device=device,
    )

    hminus_freefree = (
        freefree_theta_tensor
        * hydrogen_neutral_partition_normalized_population[:, None]
    )
    hminus_freefree = hminus_freefree * 2.0
    hminus_freefree = hminus_freefree * electron_density[:, None]
    hminus_freefree = hminus_freefree / mass_density_safe[:, None]
    hminus_freefree = hminus_freefree * 1e-26

    hminus_boundfree = frequency_invariants.tensor(
        "hminus_boundfree_cross_section_cm2", dtype, device
    )[None, :]
    hminus_boundfree = hminus_boundfree * 1e-18
    hminus_boundfree = hminus_boundfree * stimulated_emission_grid
    hminus_boundfree = hminus_boundfree * hminus_population_factor[:, None]
    hminus_boundfree = hminus_boundfree / mass_density_safe[:, None]
    return hminus_boundfree + hminus_freefree


def _hydrogen_opacity(
    continuum_tables,
    frequency_hz,
    pops,
    photon_boltzmann_factor,
    stimulated_emission,
    hc_over_kt,
    coulomb_table_energy_first=False,
    coulomb_freefree_charge1=None,
):
    """Hydrogen bound-free plus free-free opacity for one frequency."""
    temperature = pops["temperature"]
    mass_density_safe = torch.clamp(pops["mass_density"], min=1e-30)
    electron_density = pops["electron_density"]
    natural_log_temperature = torch.log(torch.clamp(temperature, min=1e-10))
    hydrogen_neutral_partition_normalized_population = pops[
        "hydrogen_partition_normalized_ion_stage_populations"
    ][:, 0]
    hydrogen_ionized_population = pops.get(
        "hydrogen_ionized_population",
        pops["hydrogen_partition_normalized_ion_stage_populations"][:, 1],
    )
    wavenumber_cm = frequency_hz / LIGHT_SPEED_CM_PER_S
    frequency_cubed_factor = 2.815e29 / (frequency_hz * frequency_hz * frequency_hz)

    hydrogen_opacity = (
        frequency_cubed_factor
        * 2.0
        / 2.0
        / (RYDBERG_WAVENUMBER_CM * hc_over_kt)
        * (
            torch.exp(-max(109250.336, 109678.764 - wavenumber_cm) * hc_over_kt)
            - torch.exp(-109678.764 * hc_over_kt)
        )
        * stimulated_emission
    )

    for (
        principal_quantum_number,
        threshold_wavenumber,
        statistical_weight,
        excitation_cm,
    ) in _HYDROGEN_HIGH_LEVEL_BOUND_FREE_TRANSITIONS:
        if wavenumber_cm >= threshold_wavenumber:
            cross_section = _karzas_latter_cross_section(
                continuum_tables.arrays,
                frequency_hz,
                1.0,
                principal_quantum_number,
                principal_quantum_number,
            )
            hydrogen_opacity = hydrogen_opacity + (
                cross_section
                * statistical_weight
                * torch.exp(-excitation_cm * hc_over_kt)
                * stimulated_emission
            )
    for (
        principal_quantum_number,
        threshold_wavenumber,
        statistical_weight,
        excitation_cm,
    ) in _HYDROGEN_LOW_LEVEL_BOUND_FREE_TRANSITIONS:
        if wavenumber_cm >= threshold_wavenumber:
            cross_section = _karzas_latter_cross_section(
                continuum_tables.arrays,
                frequency_hz,
                1.0,
                principal_quantum_number,
                principal_quantum_number,
            )
            hydrogen_opacity = hydrogen_opacity + (
                cross_section
                * statistical_weight
                * torch.exp(-excitation_cm * hc_over_kt)
                * (1.0 - photon_boltzmann_factor)
            )
    if wavenumber_cm >= 109678.764:
        cross_section = _karzas_latter_cross_section(
            continuum_tables.arrays, frequency_hz, 1.0, 1, 1
        )
        hydrogen_opacity = hydrogen_opacity + cross_section * 2.0 * (
            1.0 - photon_boltzmann_factor
        )

    hydrogen_opacity = (
        hydrogen_opacity
        * hydrogen_neutral_partition_normalized_population
        / mass_density_safe
    )
    coulomb_freefree = (
        _coulomb_freefree_gaunt(
            continuum_tables,
            1,
            math.log(frequency_hz),
            temperature,
            natural_log_temperature,
            energy_first_layout=coulomb_table_energy_first,
        )
        if coulomb_freefree_charge1 is None
        else coulomb_freefree_charge1
    )
    hydrogen_opacity = (
        hydrogen_opacity
        + 3.6919e8
        / torch.sqrt(temperature)
        * coulomb_freefree
        / frequency_hz
        * electron_density
        / frequency_hz
        * hydrogen_ionized_population
        / frequency_hz
        * stimulated_emission
        / mass_density_safe
    )
    return hydrogen_opacity


def _hydrogen_opacity_grid(
    pops,
    stimulated_emission_grid,
    hc_over_kt,
    coulomb_freefree_charge1_grid,
    frequency_invariants,
    dtype,
    device,
):
    """Batched H I bound-free + free-free opacity on ``(depth, frequency)``."""
    temperature = pops["temperature"]
    mass_density_safe = torch.clamp(pops["mass_density"], min=1e-30)
    electron_density = pops["electron_density"]
    hydrogen_neutral_partition_normalized_population = pops[
        "hydrogen_partition_normalized_ion_stage_populations"
    ][:, 0]
    hydrogen_ionized_population = pops.get(
        "hydrogen_ionized_population",
        pops["hydrogen_partition_normalized_ion_stage_populations"][:, 1],
    )
    frequency_hz = frequency_invariants.tensor("frequencies_hz", dtype, device)
    frequency_cubed_factor = 2.815e29 / (frequency_hz * frequency_hz * frequency_hz)
    tail_edge = frequency_invariants.tensor("hydrogen_tail_edge", dtype, device)

    hydrogen_opacity = (
        frequency_cubed_factor[None, :]
        * 2.0
        / 2.0
        / (RYDBERG_WAVENUMBER_CM * hc_over_kt[:, None])
    )
    hydrogen_opacity = hydrogen_opacity * (
        torch.exp(-tail_edge[None, :] * hc_over_kt[:, None])
        - torch.exp(-109678.764 * hc_over_kt[:, None])
    )
    hydrogen_opacity = hydrogen_opacity * stimulated_emission_grid

    high_level_cross_sections = frequency_invariants.tensor(
        "hydrogen_high_level_photoionization_cross_sections", dtype, device
    )
    for level_index, (
        _level,
        _threshold,
        statistical_weight,
        excitation_cm,
    ) in enumerate(_HYDROGEN_HIGH_LEVEL_BOUND_FREE_TRANSITIONS):
        hydrogen_opacity = (
            hydrogen_opacity
            + high_level_cross_sections[level_index][None, :]
            * statistical_weight
            * torch.exp(-excitation_cm * hc_over_kt[:, None])
            * stimulated_emission_grid
        )

    low_level_cross_sections = frequency_invariants.tensor(
        "hydrogen_low_level_photoionization_cross_sections", dtype, device
    )
    for level_index, (
        _level,
        _threshold,
        statistical_weight,
        excitation_cm,
    ) in enumerate(_HYDROGEN_LOW_LEVEL_BOUND_FREE_TRANSITIONS):
        hydrogen_opacity = (
            hydrogen_opacity
            + low_level_cross_sections[level_index][None, :]
            * statistical_weight
            * torch.exp(-excitation_cm * hc_over_kt[:, None])
            * stimulated_emission_grid
        )

    ground_cross_section = frequency_invariants.tensor(
        "hydrogen_ground_level_photoionization_cross_section", dtype, device
    )
    hydrogen_opacity = (
        hydrogen_opacity
        + ground_cross_section[None, :] * 2.0 * stimulated_emission_grid
    )
    hydrogen_opacity = (
        hydrogen_opacity
        * hydrogen_neutral_partition_normalized_population[:, None]
        / mass_density_safe[:, None]
    )

    hydrogen_opacity = (
        hydrogen_opacity
        + 3.6919e8
        / torch.sqrt(temperature[:, None])
        * coulomb_freefree_charge1_grid
        / frequency_hz[None, :]
        * electron_density[:, None]
        / frequency_hz[None, :]
        * hydrogen_ionized_population[:, None]
        / frequency_hz[None, :]
        * stimulated_emission_grid
        / mass_density_safe[:, None]
    )
    return hydrogen_opacity


def _scattering_opacity(
    continuum_tables,
    frequency_hz,
    pops,
    rayleigh_factor=None,
    hydrogen_partition=None,
):
    """Hydrogen Rayleigh plus Thomson scattering for one frequency."""
    temperature = pops["temperature"]
    mass_density_safe = torch.clamp(pops["mass_density"], min=1e-30)
    electron_density = pops["electron_density"]
    hydrogen_partition = (
        _hydrogen_partition(continuum_tables, temperature)
        if hydrogen_partition is None
        else hydrogen_partition
    )
    hydrogen_neutral_over_partition = (
        pops["hydrogen_neutral_population"] / hydrogen_partition
    )
    if rayleigh_factor is None:
        rayleigh_factor = _rayleigh_polarizability_factor(
            continuum_tables.arrays, frequency_hz
        )
    hydrogen_rayleigh_scattering = (
        6.65e-25
        * rayleigh_factor**2
        * hydrogen_neutral_over_partition
        * 2.0
        / mass_density_safe
    )
    electron_scattering = 0.6653e-24 * electron_density / mass_density_safe
    return hydrogen_rayleigh_scattering, electron_scattering


def _scattering_opacity_grid(
    pops, frequency_invariants, hydrogen_partition, dtype, device
):
    """Rayleigh + Thomson scattering on the full ``(depth, frequency)`` sampled grid."""
    mass_density_safe = torch.clamp(pops["mass_density"], min=1e-30)
    electron_density = pops["electron_density"]
    rayleigh_factor = frequency_invariants.tensor("rayleigh_factor", dtype, device)
    hydrogen_neutral_over_partition = (
        pops["hydrogen_neutral_population"] / hydrogen_partition
    )
    hydrogen_rayleigh_scattering = (
        6.65e-25
        * (rayleigh_factor * rayleigh_factor)[None, :]
        * hydrogen_neutral_over_partition[:, None]
        * 2.0
        / mass_density_safe[:, None]
    )
    electron_scattering = 0.6653e-24 * electron_density / mass_density_safe
    return hydrogen_rayleigh_scattering + electron_scattering[:, None]


# C I bound-free level data for the sampled-continuum metal package.
# 25 lower levels: (excitation energy cm^-1, statistical weight).  Levels 0-13 photoionise
# to the C II ground limit, 14-18 are the Luo-Pradhan resonance levels, 19-24 the n=2 group.
_CARBON_RYDBERG_WAVENUMBER_CM = 109732.298
_CARBON_NEUTRAL_LEVEL_ENERGY_CM = (
    79314.86,
    78731.27,
    78529.62,
    78309.76,
    78226.35,
    77679.82,
    73975.91,
    72610.72,
    71374.90,
    70743.95,
    69722.00,
    68856.33,
    61981.82,
    60373.00,
    21648.01,
    10192.63,
    43.42,
    16.42,
    0.00,
    119878.0,
    105798.7,
    97878.0,
    75254.93,
    64088.85,
    33735.20,
)
_CARBON_NEUTRAL_LEVEL_STATISTICAL_WEIGHT = (
    9.0,
    3.0,
    7.0,
    15.0,
    21.0,
    5.0,
    1.0,
    5.0,
    9.0,
    3.0,
    15.0,
    3.0,
    3.0,
    9.0,
    1.0,
    5.0,
    5.0,
    3.0,
    1.0,
    3.0,
    3.0,
    5.0,
    12.0,
    15.0,
    5.0,
)
_CARBON_NEUTRAL_GROUND_IONIZATION_LIMIT_CM = (
    90862.70  # C II 2P limit (ground), the 110.1 nm C I edge
)
_CARBON_NEUTRAL_LOWER_FINE_STRUCTURE_LIMIT_CM = 90820.42  # Luo-Pradhan 2P1/2 limit
_CARBON_NEUTRAL_UPPER_FINE_STRUCTURE_LIMIT_CM = (
    _CARBON_NEUTRAL_LOWER_FINE_STRUCTURE_LIMIT_CM + 63.42
)
_CARBON_NEUTRAL_EXCITED_IONIZATION_LIMIT_CM = (
    _CARBON_NEUTRAL_LOWER_FINE_STRUCTURE_LIMIT_CM + 43003.3
)
_CARBON_NEUTRAL_FREE_FREE_LIMIT_CM = _CARBON_NEUTRAL_LOWER_FINE_STRUCTURE_LIMIT_CM


def _carbon_neutral_boundfree_opacity(continuum_tables, frequency_hz, hc_over_kt):
    """Carbon I far-UV bound-free cross-section per depth."""
    wavenumber_cm = frequency_hz / LIGHT_SPEED_CM_PER_S
    if frequency_hz > 3.28805e15:
        return None
    hc_over_kt_cm_host = hc_over_kt.detach().cpu().numpy().astype(np.float64)
    level_energies_cm = np.asarray(_CARBON_NEUTRAL_LEVEL_ENERGY_CM)
    statistical_weights = np.asarray(_CARBON_NEUTRAL_LEVEL_STATISTICAL_WEIGHT)
    boltzmann_weight_by_level = statistical_weights[:, None] * np.exp(
        -level_energies_cm[:, None] * hc_over_kt_cm_host[None, :]
    )

    cross_section_by_level = np.zeros(25)
    # Group 1: levels 0-13 photoionise to the C II ground limit (n=3, l=2/1/0).
    for level_index in range(14):
        threshold_wavenumber = (
            _CARBON_NEUTRAL_GROUND_IONIZATION_LIMIT_CM - level_energies_cm[level_index]
        )
        if wavenumber_cm < threshold_wavenumber:
            continue
        effective_charge_squared = (
            9.0 / _CARBON_RYDBERG_WAVENUMBER_CM * threshold_wavenumber
        )
        orbital_quantum_number = (
            2 if level_index < 6 else (1 if level_index < 12 else 0)
        )
        cross_section_by_level[level_index] = _karzas_latter_cross_section(
            continuum_tables.arrays,
            frequency_hz,
            effective_charge_squared,
            3,
            orbital_quantum_number,
        )
    # Group 2: Luo-Pradhan levels 14-18 (1S, 1D, 3P), the two C II 2P fine-structure limits.
    for ionization_limit_wavenumber, fine_structure_weight in (
        (_CARBON_NEUTRAL_LOWER_FINE_STRUCTURE_LIMIT_CM, 1.0 / 3.0),
        (_CARBON_NEUTRAL_UPPER_FINE_STRUCTURE_LIMIT_CM, 2.0 / 3.0),
    ):
        if wavenumber_cm >= ionization_limit_wavenumber - level_energies_cm[14]:
            singlet_s_tail = 10.0 ** (
                -16.80
                - (wavenumber_cm - ionization_limit_wavenumber + level_energies_cm[14])
                / 3.0
                / _CARBON_RYDBERG_WAVENUMBER_CM
            )
            singlet_s_resonance_coordinate = (wavenumber_cm - 97700.0) * 2.0 / 2743.0
            singlet_s_resonance = (
                68e-18 * singlet_s_resonance_coordinate + 118e-18
            ) / (singlet_s_resonance_coordinate * singlet_s_resonance_coordinate + 1.0)
            cross_section_by_level[14] += (
                singlet_s_tail + singlet_s_resonance
            ) * fine_structure_weight
        if wavenumber_cm >= ionization_limit_wavenumber - level_energies_cm[15]:
            singlet_d_tail = 10.0 ** (
                -16.80
                - (wavenumber_cm - ionization_limit_wavenumber + level_energies_cm[15])
                / 3.0
                / _CARBON_RYDBERG_WAVENUMBER_CM
            )
            singlet_d_coordinate_1 = (wavenumber_cm - 93917.0) * 2.0 / 9230.0
            singlet_d_resonance_1 = (22e-18 * singlet_d_coordinate_1 + 26e-18) / (
                singlet_d_coordinate_1 * singlet_d_coordinate_1 + 1.0
            )
            singlet_d_coordinate_2 = (wavenumber_cm - 111130.0) * 2.0 / 2743.0
            singlet_d_resonance_2 = (-10.5e-18 * singlet_d_coordinate_2 + 46e-18) / (
                singlet_d_coordinate_2 * singlet_d_coordinate_2 + 1.0
            )
            cross_section_by_level[15] += (
                singlet_d_tail + singlet_d_resonance_1 + singlet_d_resonance_2
            ) * fine_structure_weight
        for level_index in range(16, 19):
            if (
                wavenumber_cm
                >= ionization_limit_wavenumber - level_energies_cm[level_index]
            ):
                cross_section_by_level[level_index] += (
                    10.0
                    ** (
                        -16.80
                        - (
                            wavenumber_cm
                            - ionization_limit_wavenumber
                            + level_energies_cm[level_index]
                        )
                        / 3.0
                        / _CARBON_RYDBERG_WAVENUMBER_CM
                    )
                    * fine_structure_weight
                )
    # Group 3: levels 19-24 photoionise to the higher limit (n=2, l=1, degeneracy 3).
    for level_index in range(19, 25):
        threshold_wavenumber = (
            _CARBON_NEUTRAL_EXCITED_IONIZATION_LIMIT_CM - level_energies_cm[level_index]
        )
        if wavenumber_cm < threshold_wavenumber:
            continue
        cross_section_by_level[level_index] = (
            _karzas_latter_cross_section(
                continuum_tables.arrays,
                frequency_hz,
                4.0 / _CARBON_RYDBERG_WAVENUMBER_CM * threshold_wavenumber,
                2,
                1,
            )
            * 3.0
        )

    # Kramers-Gaunt n>=4 free-free toward the ground limit (gfactor 6).
    frequency_cubed_factor = 2.815e29 / (frequency_hz * frequency_hz * frequency_hz)
    freefree_threshold_wavenumber = max(
        _CARBON_NEUTRAL_FREE_FREE_LIMIT_CM - _CARBON_RYDBERG_WAVENUMBER_CM / 16.0,
        _CARBON_NEUTRAL_FREE_FREE_LIMIT_CM - wavenumber_cm,
    )
    kramers_freefree = (
        frequency_cubed_factor
        * 6.0
        / (_CARBON_RYDBERG_WAVENUMBER_CM * hc_over_kt_cm_host)
        * (
            np.exp(-freefree_threshold_wavenumber * hc_over_kt_cm_host)
            - np.exp(-_CARBON_NEUTRAL_FREE_FREE_LIMIT_CM * hc_over_kt_cm_host)
        )
    )
    opacity_by_depth = (
        kramers_freefree + cross_section_by_level @ boltzmann_weight_by_level
    )
    return torch.as_tensor(
        opacity_by_depth,
        dtype=hc_over_kt.dtype,
        device=hc_over_kt.device,
    )


# Near-UV metal bound-free forest. The scalar synthesis path keeps the compact
# optical-edge floors; the sampled-continuum path uses the full multi-edge UV set.

_METAL_RYDBERG_WAVENUMBER_CM = 109732.298  # Rydberg used by metal Karzas-Latter calls


# Mg I sampled-continuum term, 15 lower levels.
_MAGNESIUM_NEUTRAL_LEVEL_ENERGY_CM = np.array(
    [
        54676.710,
        54676.438,
        54192.284,
        53134.642,
        49346.729,
        47957.034,
        47847.797,
        46403.065,
        43503.333,
        41197.043,
        35051.264,
        21919.178,
        21870.464,
        21850.405,
        0.0,
    ]
)
_MAGNESIUM_NEUTRAL_LEVEL_STATISTICAL_WEIGHT = np.array(
    [21.0, 7.0, 15.0, 5.0, 3.0, 15.0, 9.0, 5.0, 1.0, 3.0, 3.0, 5.0, 3.0, 1.0, 1.0]
)
_MAGNESIUM_NEUTRAL_IONIZATION_LIMIT_CM = 61671.02


def _magnesium_neutral_boundfree_opacity(continuum_tables, frequency_hz, hc_over_kt):
    """Magnesium I far-UV bound-free opacity per depth."""
    if frequency_hz > 3.28805e15:
        return None
    wavenumber_cm = frequency_hz / LIGHT_SPEED_CM_PER_S
    hc_over_kt_cm_host = hc_over_kt.detach().cpu().numpy().astype(np.float64)
    level_energies_cm = _MAGNESIUM_NEUTRAL_LEVEL_ENERGY_CM
    ionization_limit_wavenumber = _MAGNESIUM_NEUTRAL_IONIZATION_LIMIT_CM
    boltzmann_weight_by_level = _MAGNESIUM_NEUTRAL_LEVEL_STATISTICAL_WEIGHT[
        :, None
    ] * np.exp(-level_energies_cm[:, None] * hc_over_kt_cm_host[None, :])
    cross_section_by_level = np.zeros(15)

    # Levels 0-4: Karzas-Latter cross-sections with l=3/3/2/2/1.
    for level_index, orbital_quantum_number in ((0, 3), (1, 3), (2, 2), (3, 2), (4, 1)):
        threshold_wavenumber = (
            ionization_limit_wavenumber - level_energies_cm[level_index]
        )
        if wavenumber_cm >= threshold_wavenumber:
            cross_section_by_level[level_index] = _karzas_latter_cross_section(
                continuum_tables.arrays,
                frequency_hz,
                16.0 / _METAL_RYDBERG_WAVENUMBER_CM * threshold_wavenumber,
                4,
                orbital_quantum_number,
            )
    # Levels 5-9: analytic single power law.
    for level_index, (edge_wavenumber, coefficient, exponent) in (
        (5, (13713.986, 25e-18, 2.7)),
        (6, (13823.223, 33.8e-18, 2.8)),
        (7, (15267.955, 45e-18, 2.7)),
        (8, (18167.687, 0.43e-18, 2.6)),
        (9, (20473.617, 2.1e-18, 2.6)),
    ):
        if (
            wavenumber_cm
            >= ionization_limit_wavenumber - level_energies_cm[level_index]
        ):
            cross_section_by_level[level_index] = (
                coefficient * (edge_wavenumber / wavenumber_cm) ** exponent
            )
    # Level 10: two-term.
    if wavenumber_cm >= ionization_limit_wavenumber - level_energies_cm[10]:
        cross_section_by_level[10] = (
            16e-18 * (26619.756 / wavenumber_cm) ** 2.1
            - 7.8e-18 * (26619.756 / wavenumber_cm) ** 9.5
        )
    # Levels 11-13: max of two power laws.
    for level_index in range(11, 14):
        if (
            wavenumber_cm
            >= ionization_limit_wavenumber - level_energies_cm[level_index]
        ):
            cross_section_by_level[level_index] = max(
                20e-18 * (39759.842 / wavenumber_cm) ** 2.7,
                40e-18 * (39759.842 / wavenumber_cm) ** 14,
            )
    # Level 14: ground-state analytic.
    if wavenumber_cm >= ionization_limit_wavenumber - level_energies_cm[14]:
        cross_section_by_level[14] = (
            1.1e-18
            * ((ionization_limit_wavenumber - level_energies_cm[14]) / wavenumber_cm)
            ** 10
        )
    # Kramers-Gaunt n>=5 free-free (gfactor 2).
    frequency_cubed_factor = 2.815e29 / (frequency_hz * frequency_hz * frequency_hz)
    freefree_threshold_wavenumber = max(
        ionization_limit_wavenumber - _METAL_RYDBERG_WAVENUMBER_CM / 25.0,
        ionization_limit_wavenumber - wavenumber_cm,
    )
    kramers_freefree = (
        frequency_cubed_factor
        * 2.0
        / (_METAL_RYDBERG_WAVENUMBER_CM * hc_over_kt_cm_host)
        * (
            np.exp(-freefree_threshold_wavenumber * hc_over_kt_cm_host)
            - np.exp(-ionization_limit_wavenumber * hc_over_kt_cm_host)
        )
    )
    opacity_by_depth = (
        kramers_freefree + cross_section_by_level @ boltzmann_weight_by_level
    )
    return torch.as_tensor(
        opacity_by_depth,
        dtype=hc_over_kt.dtype,
        device=hc_over_kt.device,
    )


# Si I sampled-continuum term, 33 lower levels.
_SILICON_NEUTRAL_LEVEL_ENERGY_CM = np.array(
    [
        59962.284,
        59100.0,
        59077.112,
        58893.40,
        58801.529,
        58777.0,
        57488.974,
        56503.346,
        54225.621,
        53387.34,
        53362.24,
        51612.012,
        50533.424,
        50189.389,
        49965.894,
        49399.670,
        49128.131,
        48161.459,
        47351.554,
        47284.061,
        40991.884,
        39859.920,
        15394.370,
        6298.850,
        223.157,
        77.115,
        0.000,
        94000.0,
        79664.0,
        72000.0,
        56698.738,
        45303.310,
        33326.053,
    ]
)
_SILICON_NEUTRAL_LEVEL_STATISTICAL_WEIGHT = np.array(
    [
        9.0,
        56.0,
        15.0,
        7.0,
        3.0,
        28.0,
        21.0,
        5.0,
        15.0,
        3.0,
        7.0,
        1.0,
        9.0,
        5.0,
        21.0,
        3.0,
        9.0,
        15.0,
        5.0,
        3.0,
        3.0,
        9.0,
        1.0,
        5.0,
        5.0,
        3.0,
        1.0,
        3.0,
        3.0,
        5.0,
        12.0,
        15.0,
        5.0,
    ]
)
# (n, l) and effective-charge factor for the group-1 Karzas-Latter levels 0-21.
_SILICON_NEUTRAL_QUANTUM_NUMBERS = (
    (4, 2),
    (4, 3),
    (4, 2),
    (4, 2),
    (4, 2),
    (4, 3),
    (4, 2),
    (4, 2),
    (3, 2),
    (3, 2),
    (3, 2),
    (4, 1),
    (3, 2),
    (4, 1),
    (3, 2),
    (4, 1),
    (4, 1),
    (4, 1),
    (3, 2),
    (4, 1),
    (4, 0),
    (4, 0),
)
_SILICON_NEUTRAL_EFFECTIVE_CHARGE_NUMERATOR = (
    16.0,
    16.0,
    16.0,
    16.0,
    16.0,
    16.0,
    16.0,
    16.0,
    9.0,
    9.0,
    9.0,
    16.0,
    9.0,
    16.0,
    9.0,
    16.0,
    16.0,
    16.0,
    9.0,
    16.0,
    16.0,
    16.0,
)
_SILICON_NEUTRAL_GROUND_IONIZATION_LIMIT_CM = 65939.18
_SILICON_NEUTRAL_FINE_STRUCTURE_LIMIT_CM = 65747.55
_SILICON_NEUTRAL_EXCITED_IONIZATION_LIMIT_CM = 65747.5 + 42824.35


def _silicon_neutral_boundfree_opacity(continuum_tables, frequency_hz, hc_over_kt):
    """Silicon I far-UV bound-free opacity per depth."""
    if frequency_hz > 3.28805e15:
        return None
    wavenumber_cm = frequency_hz / LIGHT_SPEED_CM_PER_S
    hc_over_kt_cm_host = hc_over_kt.detach().cpu().numpy().astype(np.float64)
    level_energies_cm = _SILICON_NEUTRAL_LEVEL_ENERGY_CM
    boltzmann_weight_by_level = _SILICON_NEUTRAL_LEVEL_STATISTICAL_WEIGHT[
        :, None
    ] * np.exp(-level_energies_cm[:, None] * hc_over_kt_cm_host[None, :])
    cross_section_by_level = np.zeros(33)

    # Group 1: levels 0-21 use Karzas-Latter cross-sections with per-level (n, l).
    for level_index in range(22):
        threshold_wavenumber = (
            _SILICON_NEUTRAL_GROUND_IONIZATION_LIMIT_CM - level_energies_cm[level_index]
        )
        if wavenumber_cm >= threshold_wavenumber:
            principal_quantum_number, orbital_quantum_number = (
                _SILICON_NEUTRAL_QUANTUM_NUMBERS[level_index]
            )
            cross_section_by_level[level_index] = _karzas_latter_cross_section(
                continuum_tables.arrays,
                frequency_hz,
                _SILICON_NEUTRAL_EFFECTIVE_CHARGE_NUMERATOR[level_index]
                / _METAL_RYDBERG_WAVENUMBER_CM
                * threshold_wavenumber,
                principal_quantum_number,
                orbital_quantum_number,
            )
    # Group 2 + 2b: Nahar-Pradhan resonance levels 22-26 at the two Si II 2P limits.
    for ionization_limit_wavenumber, fine_structure_weight in (
        (_SILICON_NEUTRAL_FINE_STRUCTURE_LIMIT_CM, 1.0 / 3.0),
        (_SILICON_NEUTRAL_FINE_STRUCTURE_LIMIT_CM + 287.45, 2.0 / 3.0),
    ):
        if wavenumber_cm >= ionization_limit_wavenumber - level_energies_cm[22]:
            singlet_s_coordinate = (wavenumber_cm - 70000.0) * 2.0 / 6500.0
            singlet_s_resonance = (97e-18 * singlet_s_coordinate + 94e-18) / (
                singlet_s_coordinate * singlet_s_coordinate + 1.0
            )
            cross_section_by_level[22] += (
                37e-18 * (50353.180 / wavenumber_cm) ** 2.40 + singlet_s_resonance
            ) * fine_structure_weight
        if wavenumber_cm >= ionization_limit_wavenumber - level_energies_cm[23]:
            singlet_d_coordinate = (wavenumber_cm - 78600.0) * 2.0 / 13000.0
            singlet_d_resonance = (-10e-18 * singlet_d_coordinate + 77e-18) / (
                singlet_d_coordinate * singlet_d_coordinate + 1.0
            )
            cross_section_by_level[23] += (
                24.5e-18 * (59448.700 / wavenumber_cm) ** 1.85 + singlet_d_resonance
            ) * fine_structure_weight
        for level_index in (24, 25, 26):
            if (
                wavenumber_cm
                >= ionization_limit_wavenumber - level_energies_cm[level_index]
            ):
                threshold_ratio = 65524.393 / wavenumber_cm
                effective_weight = (
                    (2.0 / 3.0) if level_index == 25 else fine_structure_weight
                )
                cross_section_by_level[level_index] += (
                    72e-18 * threshold_ratio**1.90
                    if wavenumber_cm <= 74000.0
                    else 93e-18 * threshold_ratio**4.00
                ) * effective_weight
    # Group 3: levels 27-32 use Karzas-Latter cross-sections at the higher limit.
    for level_index in range(27, 33):
        threshold_wavenumber = (
            _SILICON_NEUTRAL_EXCITED_IONIZATION_LIMIT_CM
            - level_energies_cm[level_index]
        )
        if wavenumber_cm >= threshold_wavenumber:
            cross_section_by_level[level_index] = (
                _karzas_latter_cross_section(
                    continuum_tables.arrays,
                    frequency_hz,
                    9.0 / _METAL_RYDBERG_WAVENUMBER_CM * threshold_wavenumber,
                    3,
                    1,
                )
                * 3.0
            )
    # Kramers-Gaunt n>=5 free-free (gfactor 6).
    frequency_cubed_factor = 2.815e29 / (frequency_hz * frequency_hz * frequency_hz)
    freefree_threshold_wavenumber = max(
        _SILICON_NEUTRAL_FINE_STRUCTURE_LIMIT_CM - _METAL_RYDBERG_WAVENUMBER_CM / 25.0,
        _SILICON_NEUTRAL_FINE_STRUCTURE_LIMIT_CM - wavenumber_cm,
    )
    kramers_freefree = (
        frequency_cubed_factor
        * 6.0
        / (_METAL_RYDBERG_WAVENUMBER_CM * hc_over_kt_cm_host)
        * (
            np.exp(-freefree_threshold_wavenumber * hc_over_kt_cm_host)
            - np.exp(-_SILICON_NEUTRAL_FINE_STRUCTURE_LIMIT_CM * hc_over_kt_cm_host)
        )
    )
    opacity_by_depth = (
        kramers_freefree + cross_section_by_level @ boltzmann_weight_by_level
    )
    return torch.as_tensor(
        opacity_by_depth,
        dtype=hc_over_kt.dtype,
        device=hc_over_kt.device,
    )


# Al I sampled-continuum term: a twin 2P fine-structure edge.
_ALUMINUM_NEUTRAL_IONIZATION_LIMIT_CM = 48278.37


def _aluminum_neutral_boundfree_opacity(
    _continuum_tables,
    frequency_hz,
    hc_over_kt,
):
    """Aluminum I far-UV bound-free opacity per depth.

    Frequency-only cross-section; broadcast to all depths so the metal helper
    contract remains element-uniform.
    """
    if frequency_hz > 3.28805e15:
        return None
    wavenumber_cm = frequency_hz / LIGHT_SPEED_CM_PER_S
    cross_section = 0.0

    upper_fine_structure_edge = _ALUMINUM_NEUTRAL_IONIZATION_LIMIT_CM - 112.061
    if wavenumber_cm >= upper_fine_structure_edge:
        cross_section += (
            6.5e-17 * (upper_fine_structure_edge / wavenumber_cm) ** 5 * 4.0
        )
    if wavenumber_cm >= _ALUMINUM_NEUTRAL_IONIZATION_LIMIT_CM:
        cross_section += (
            6.5e-17 * (_ALUMINUM_NEUTRAL_IONIZATION_LIMIT_CM / wavenumber_cm) ** 5 * 2.0
        )

    if cross_section == 0.0:
        return None
    return cross_section * torch.ones_like(hc_over_kt)


# Fe I sampled-continuum term: 48 resonance transitions.
_IRON_NEUTRAL_TRANSITION_STATISTICAL_WEIGHT = np.array(
    [
        25.0,
        35.0,
        21.0,
        15.0,
        9.0,
        35.0,
        33.0,
        21.0,
        27.0,
        49.0,
        9.0,
        21.0,
        27.0,
        9.0,
        9.0,
        25.0,
        33.0,
        15.0,
        35.0,
        3.0,
        5.0,
        11.0,
        15.0,
        13.0,
        15.0,
        9.0,
        21.0,
        15.0,
        21.0,
        25.0,
        35.0,
        9.0,
        5.0,
        45.0,
        27.0,
        21.0,
        15.0,
        21.0,
        15.0,
        25.0,
        21.0,
        35.0,
        5.0,
        15.0,
        45.0,
        35.0,
        55.0,
        25.0,
    ]
)
_IRON_NEUTRAL_TRANSITION_ENERGY_CM = np.array(
    [
        500.0,
        7500.0,
        12500.0,
        17500.0,
        19000.0,
        19500.0,
        19500.0,
        21000.0,
        22000.0,
        23000.0,
        23000.0,
        24000.0,
        24000.0,
        24500.0,
        24500.0,
        26000.0,
        26500.0,
        26500.0,
        27000.0,
        27500.0,
        28500.0,
        29000.0,
        29500.0,
        29500.0,
        29500.0,
        30000.0,
        31500.0,
        31500.0,
        33500.0,
        33500.0,
        34000.0,
        34500.0,
        34500.0,
        35000.0,
        35500.0,
        37000.0,
        37000.0,
        37000.0,
        38500.0,
        40000.0,
        40000.0,
        41000.0,
        41000.0,
        43000.0,
        43000.0,
        43000.0,
        43000.0,
        44000.0,
    ]
)
_IRON_NEUTRAL_TRANSITION_WAVENUMBER_CM = np.array(
    [
        63500.0,
        58500.0,
        53500.0,
        59500.0,
        45000.0,
        44500.0,
        44500.0,
        43000.0,
        58000.0,
        41000.0,
        54000.0,
        40000.0,
        40000.0,
        57500.0,
        55500.0,
        38000.0,
        57500.0,
        57500.0,
        37000.0,
        54500.0,
        53500.0,
        55000.0,
        34500.0,
        34500.0,
        34500.0,
        34000.0,
        32500.0,
        32500.0,
        32500.0,
        32500.0,
        32000.0,
        29500.0,
        29500.0,
        31000.0,
        30500.0,
        29000.0,
        27000.0,
        54000.0,
        27500.0,
        24000.0,
        47000.0,
        23000.0,
        44000.0,
        42000.0,
        42000.0,
        21000.0,
        42000.0,
        42000.0,
    ]
)


def _iron_neutral_boundfree_opacity(_continuum_tables, frequency_hz, hc_over_kt):
    """Iron I bound-free opacity per depth for a 48-transition forest."""
    wavenumber_cm = frequency_hz / LIGHT_SPEED_CM_PER_S
    if wavenumber_cm < 21000.0:
        return None
    hc_over_kt_cm_host = hc_over_kt.detach().cpu().numpy().astype(np.float64)
    active_transition_mask = _IRON_NEUTRAL_TRANSITION_WAVENUMBER_CM <= wavenumber_cm
    if not np.any(active_transition_mask):
        return None

    # Lorentzian-quartic resonance profile per active transition (scalar in freq).
    active_transition_wavenumber = _IRON_NEUTRAL_TRANSITION_WAVENUMBER_CM[
        active_transition_mask
    ]
    resonance_cross_section = 3e-18 / (
        1.0
        + (
            (active_transition_wavenumber + 3000.0 - wavenumber_cm)
            / active_transition_wavenumber
            / 0.1
        )
        ** 4
    )
    boltzmann_weight_by_transition = _IRON_NEUTRAL_TRANSITION_STATISTICAL_WEIGHT[
        active_transition_mask
    ][:, None] * np.exp(
        -_IRON_NEUTRAL_TRANSITION_ENERGY_CM[active_transition_mask][:, None]
        * hc_over_kt_cm_host[None, :]
    )
    opacity_by_depth = resonance_cross_section @ boltzmann_weight_by_transition
    return torch.as_tensor(
        opacity_by_depth,
        dtype=hc_over_kt.dtype,
        device=hc_over_kt.device,
    )


# Metal bound-free helper table: each function returns a depth-vector
# cross-section, and the caller applies population, stimulated emission, and
# density scaling uniformly.
_METAL_BOUND_FREE_HELPERS = (
    (
        "carbon_neutral_partition_normalized_population",
        _carbon_neutral_boundfree_opacity,
    ),
    (
        "magnesium_neutral_partition_normalized_population",
        _magnesium_neutral_boundfree_opacity,
    ),
    (
        "silicon_neutral_partition_normalized_population",
        _silicon_neutral_boundfree_opacity,
    ),
    (
        "aluminum_neutral_partition_normalized_population",
        _aluminum_neutral_boundfree_opacity,
    ),
    ("iron_neutral_partition_normalized_population", _iron_neutral_boundfree_opacity),
)


def _precompute_metal_boundfree_frequency_rows(continuum_tables, frequency_grid_hz):
    """Frequency-only metal bound-free rows for the sampled-grid batch lane."""
    n_frequencies = frequency_grid_hz.size
    wavenumber_cm = frequency_grid_hz / LIGHT_SPEED_CM_PER_S

    carbon_cross_section_rows = np.zeros((25, n_frequencies), dtype=np.float64)
    carbon_freefree_prefactor = np.zeros(n_frequencies, dtype=np.float64)
    carbon_freefree_threshold = np.zeros(n_frequencies, dtype=np.float64)

    magnesium_cross_section_rows = np.zeros(
        (_MAGNESIUM_NEUTRAL_LEVEL_ENERGY_CM.size, n_frequencies), dtype=np.float64
    )
    magnesium_freefree_prefactor = np.zeros(n_frequencies, dtype=np.float64)
    magnesium_freefree_threshold = np.zeros(n_frequencies, dtype=np.float64)

    silicon_cross_section_rows = np.zeros(
        (_SILICON_NEUTRAL_LEVEL_ENERGY_CM.size, n_frequencies), dtype=np.float64
    )
    silicon_freefree_prefactor = np.zeros(n_frequencies, dtype=np.float64)
    silicon_freefree_threshold = np.zeros(n_frequencies, dtype=np.float64)

    aluminum_cross_section = np.zeros(n_frequencies, dtype=np.float64)
    iron_cross_section_rows = np.zeros(
        (_IRON_NEUTRAL_TRANSITION_WAVENUMBER_CM.size, n_frequencies), dtype=np.float64
    )

    for frequency_index, frequency_hz in enumerate(frequency_grid_hz):
        wavenumber = wavenumber_cm[frequency_index]
        if frequency_hz <= 3.28805e15:
            for level_index in range(14):
                threshold_wavenumber = (
                    _CARBON_NEUTRAL_GROUND_IONIZATION_LIMIT_CM
                    - _CARBON_NEUTRAL_LEVEL_ENERGY_CM[level_index]
                )
                if wavenumber >= threshold_wavenumber:
                    effective_charge_squared = (
                        9.0 / _CARBON_RYDBERG_WAVENUMBER_CM * threshold_wavenumber
                    )
                    orbital_quantum_number = (
                        2 if level_index < 6 else (1 if level_index < 12 else 0)
                    )
                    carbon_cross_section_rows[level_index, frequency_index] = (
                        _karzas_latter_cross_section(
                            continuum_tables.arrays,
                            float(frequency_hz),
                            effective_charge_squared,
                            3,
                            orbital_quantum_number,
                        )
                    )
            for ionization_limit_wavenumber, fine_structure_weight in (
                (_CARBON_NEUTRAL_LOWER_FINE_STRUCTURE_LIMIT_CM, 1.0 / 3.0),
                (_CARBON_NEUTRAL_UPPER_FINE_STRUCTURE_LIMIT_CM, 2.0 / 3.0),
            ):
                if (
                    wavenumber
                    >= ionization_limit_wavenumber - _CARBON_NEUTRAL_LEVEL_ENERGY_CM[14]
                ):
                    singlet_s_tail = 10.0 ** (
                        -16.80
                        - (
                            wavenumber
                            - ionization_limit_wavenumber
                            + _CARBON_NEUTRAL_LEVEL_ENERGY_CM[14]
                        )
                        / 3.0
                        / _CARBON_RYDBERG_WAVENUMBER_CM
                    )
                    singlet_s_coordinate = (wavenumber - 97700.0) * 2.0 / 2743.0
                    singlet_s_resonance = (68e-18 * singlet_s_coordinate + 118e-18) / (
                        singlet_s_coordinate * singlet_s_coordinate + 1.0
                    )
                    carbon_cross_section_rows[14, frequency_index] += (
                        singlet_s_tail + singlet_s_resonance
                    ) * fine_structure_weight
                if (
                    wavenumber
                    >= ionization_limit_wavenumber - _CARBON_NEUTRAL_LEVEL_ENERGY_CM[15]
                ):
                    singlet_d_tail = 10.0 ** (
                        -16.80
                        - (
                            wavenumber
                            - ionization_limit_wavenumber
                            + _CARBON_NEUTRAL_LEVEL_ENERGY_CM[15]
                        )
                        / 3.0
                        / _CARBON_RYDBERG_WAVENUMBER_CM
                    )
                    singlet_d_coordinate_1 = (wavenumber - 93917.0) * 2.0 / 9230.0
                    singlet_d_resonance_1 = (
                        22e-18 * singlet_d_coordinate_1 + 26e-18
                    ) / (singlet_d_coordinate_1 * singlet_d_coordinate_1 + 1.0)
                    singlet_d_coordinate_2 = (wavenumber - 111130.0) * 2.0 / 2743.0
                    singlet_d_resonance_2 = (
                        -10.5e-18 * singlet_d_coordinate_2 + 46e-18
                    ) / (singlet_d_coordinate_2 * singlet_d_coordinate_2 + 1.0)
                    carbon_cross_section_rows[15, frequency_index] += (
                        singlet_d_tail + singlet_d_resonance_1 + singlet_d_resonance_2
                    ) * fine_structure_weight
                for level_index in range(16, 19):
                    if (
                        wavenumber
                        >= ionization_limit_wavenumber
                        - _CARBON_NEUTRAL_LEVEL_ENERGY_CM[level_index]
                    ):
                        carbon_cross_section_rows[level_index, frequency_index] += (
                            10.0
                            ** (
                                -16.80
                                - (
                                    wavenumber
                                    - ionization_limit_wavenumber
                                    + _CARBON_NEUTRAL_LEVEL_ENERGY_CM[level_index]
                                )
                                / 3.0
                                / _CARBON_RYDBERG_WAVENUMBER_CM
                            )
                            * fine_structure_weight
                        )
            for level_index in range(19, 25):
                threshold_wavenumber = (
                    _CARBON_NEUTRAL_EXCITED_IONIZATION_LIMIT_CM
                    - _CARBON_NEUTRAL_LEVEL_ENERGY_CM[level_index]
                )
                if wavenumber >= threshold_wavenumber:
                    carbon_cross_section_rows[level_index, frequency_index] = (
                        _karzas_latter_cross_section(
                            continuum_tables.arrays,
                            float(frequency_hz),
                            4.0 / _CARBON_RYDBERG_WAVENUMBER_CM * threshold_wavenumber,
                            2,
                            1,
                        )
                        * 3.0
                    )
            carbon_freefree_prefactor[frequency_index] = (
                2.815e29
                / (frequency_hz * frequency_hz * frequency_hz)
                * 6.0
                / _CARBON_RYDBERG_WAVENUMBER_CM
            )
            carbon_freefree_threshold[frequency_index] = max(
                _CARBON_NEUTRAL_FREE_FREE_LIMIT_CM
                - _CARBON_RYDBERG_WAVENUMBER_CM / 16.0,
                _CARBON_NEUTRAL_FREE_FREE_LIMIT_CM - wavenumber,
            )

            for level_index, orbital_quantum_number in (
                (0, 3),
                (1, 3),
                (2, 2),
                (3, 2),
                (4, 1),
            ):
                threshold_wavenumber = (
                    _MAGNESIUM_NEUTRAL_IONIZATION_LIMIT_CM
                    - _MAGNESIUM_NEUTRAL_LEVEL_ENERGY_CM[level_index]
                )
                if wavenumber >= threshold_wavenumber:
                    magnesium_cross_section_rows[level_index, frequency_index] = (
                        _karzas_latter_cross_section(
                            continuum_tables.arrays,
                            float(frequency_hz),
                            16.0 / _METAL_RYDBERG_WAVENUMBER_CM * threshold_wavenumber,
                            4,
                            orbital_quantum_number,
                        )
                    )
            for level_index, (edge_wavenumber, coefficient, exponent) in (
                (5, (13713.986, 25e-18, 2.7)),
                (6, (13823.223, 33.8e-18, 2.8)),
                (7, (15267.955, 45e-18, 2.7)),
                (8, (18167.687, 0.43e-18, 2.6)),
                (9, (20473.617, 2.1e-18, 2.6)),
            ):
                if (
                    wavenumber
                    >= _MAGNESIUM_NEUTRAL_IONIZATION_LIMIT_CM
                    - _MAGNESIUM_NEUTRAL_LEVEL_ENERGY_CM[level_index]
                ):
                    magnesium_cross_section_rows[level_index, frequency_index] = (
                        coefficient * (edge_wavenumber / wavenumber) ** exponent
                    )
            if (
                wavenumber
                >= _MAGNESIUM_NEUTRAL_IONIZATION_LIMIT_CM
                - _MAGNESIUM_NEUTRAL_LEVEL_ENERGY_CM[10]
            ):
                magnesium_cross_section_rows[10, frequency_index] = (
                    16e-18 * (26619.756 / wavenumber) ** 2.1
                    - 7.8e-18 * (26619.756 / wavenumber) ** 9.5
                )
            for level_index in range(11, 14):
                if (
                    wavenumber
                    >= _MAGNESIUM_NEUTRAL_IONIZATION_LIMIT_CM
                    - _MAGNESIUM_NEUTRAL_LEVEL_ENERGY_CM[level_index]
                ):
                    magnesium_cross_section_rows[level_index, frequency_index] = max(
                        20e-18 * (39759.842 / wavenumber) ** 2.7,
                        40e-18 * (39759.842 / wavenumber) ** 14,
                    )
            if (
                wavenumber
                >= _MAGNESIUM_NEUTRAL_IONIZATION_LIMIT_CM
                - _MAGNESIUM_NEUTRAL_LEVEL_ENERGY_CM[14]
            ):
                magnesium_cross_section_rows[14, frequency_index] = (
                    1.1e-18
                    * (
                        (
                            _MAGNESIUM_NEUTRAL_IONIZATION_LIMIT_CM
                            - _MAGNESIUM_NEUTRAL_LEVEL_ENERGY_CM[14]
                        )
                        / wavenumber
                    )
                    ** 10
                )
            magnesium_freefree_prefactor[frequency_index] = (
                2.815e29
                / (frequency_hz * frequency_hz * frequency_hz)
                * 2.0
                / _METAL_RYDBERG_WAVENUMBER_CM
            )
            magnesium_freefree_threshold[frequency_index] = max(
                _MAGNESIUM_NEUTRAL_IONIZATION_LIMIT_CM
                - _METAL_RYDBERG_WAVENUMBER_CM / 25.0,
                _MAGNESIUM_NEUTRAL_IONIZATION_LIMIT_CM - wavenumber,
            )

            for level_index in range(22):
                threshold_wavenumber = (
                    _SILICON_NEUTRAL_GROUND_IONIZATION_LIMIT_CM
                    - _SILICON_NEUTRAL_LEVEL_ENERGY_CM[level_index]
                )
                if wavenumber >= threshold_wavenumber:
                    principal_quantum_number, orbital_quantum_number = (
                        _SILICON_NEUTRAL_QUANTUM_NUMBERS[level_index]
                    )
                    silicon_cross_section_rows[level_index, frequency_index] = (
                        _karzas_latter_cross_section(
                            continuum_tables.arrays,
                            float(frequency_hz),
                            _SILICON_NEUTRAL_EFFECTIVE_CHARGE_NUMERATOR[level_index]
                            / _METAL_RYDBERG_WAVENUMBER_CM
                            * threshold_wavenumber,
                            principal_quantum_number,
                            orbital_quantum_number,
                        )
                    )
            for ionization_limit_wavenumber, fine_structure_weight in (
                (_SILICON_NEUTRAL_FINE_STRUCTURE_LIMIT_CM, 1.0 / 3.0),
                (_SILICON_NEUTRAL_FINE_STRUCTURE_LIMIT_CM + 287.45, 2.0 / 3.0),
            ):
                if (
                    wavenumber
                    >= ionization_limit_wavenumber
                    - _SILICON_NEUTRAL_LEVEL_ENERGY_CM[22]
                ):
                    singlet_s_coordinate = (wavenumber - 70000.0) * 2.0 / 6500.0
                    singlet_s_resonance = (97e-18 * singlet_s_coordinate + 94e-18) / (
                        singlet_s_coordinate * singlet_s_coordinate + 1.0
                    )
                    silicon_cross_section_rows[22, frequency_index] += (
                        37e-18 * (50353.180 / wavenumber) ** 2.40 + singlet_s_resonance
                    ) * fine_structure_weight
                if (
                    wavenumber
                    >= ionization_limit_wavenumber
                    - _SILICON_NEUTRAL_LEVEL_ENERGY_CM[23]
                ):
                    singlet_d_coordinate = (wavenumber - 78600.0) * 2.0 / 13000.0
                    singlet_d_resonance = (-10e-18 * singlet_d_coordinate + 77e-18) / (
                        singlet_d_coordinate * singlet_d_coordinate + 1.0
                    )
                    silicon_cross_section_rows[23, frequency_index] += (
                        24.5e-18 * (59448.700 / wavenumber) ** 1.85
                        + singlet_d_resonance
                    ) * fine_structure_weight
                for level_index in (24, 25, 26):
                    if (
                        wavenumber
                        >= ionization_limit_wavenumber
                        - _SILICON_NEUTRAL_LEVEL_ENERGY_CM[level_index]
                    ):
                        threshold_ratio = 65524.393 / wavenumber
                        effective_weight = (
                            (2.0 / 3.0) if level_index == 25 else fine_structure_weight
                        )
                        silicon_cross_section_rows[level_index, frequency_index] += (
                            72e-18 * threshold_ratio**1.90
                            if wavenumber <= 74000.0
                            else 93e-18 * threshold_ratio**4.00
                        ) * effective_weight
            for level_index in range(27, 33):
                threshold_wavenumber = (
                    _SILICON_NEUTRAL_EXCITED_IONIZATION_LIMIT_CM
                    - _SILICON_NEUTRAL_LEVEL_ENERGY_CM[level_index]
                )
                if wavenumber >= threshold_wavenumber:
                    silicon_cross_section_rows[level_index, frequency_index] = (
                        _karzas_latter_cross_section(
                            continuum_tables.arrays,
                            float(frequency_hz),
                            9.0 / _METAL_RYDBERG_WAVENUMBER_CM * threshold_wavenumber,
                            3,
                            1,
                        )
                        * 3.0
                    )
            silicon_freefree_prefactor[frequency_index] = (
                2.815e29
                / (frequency_hz * frequency_hz * frequency_hz)
                * 6.0
                / _METAL_RYDBERG_WAVENUMBER_CM
            )
            silicon_freefree_threshold[frequency_index] = max(
                _SILICON_NEUTRAL_FINE_STRUCTURE_LIMIT_CM
                - _METAL_RYDBERG_WAVENUMBER_CM / 25.0,
                _SILICON_NEUTRAL_FINE_STRUCTURE_LIMIT_CM - wavenumber,
            )

            aluminum_upper_fine_structure_edge = (
                _ALUMINUM_NEUTRAL_IONIZATION_LIMIT_CM - 112.061
            )
            if wavenumber >= aluminum_upper_fine_structure_edge:
                aluminum_cross_section[frequency_index] += (
                    6.5e-17
                    * (aluminum_upper_fine_structure_edge / wavenumber) ** 5
                    * 4.0
                )
            if wavenumber >= _ALUMINUM_NEUTRAL_IONIZATION_LIMIT_CM:
                aluminum_cross_section[frequency_index] += (
                    6.5e-17
                    * (_ALUMINUM_NEUTRAL_IONIZATION_LIMIT_CM / wavenumber) ** 5
                    * 2.0
                )

        if wavenumber >= 21000.0:
            active_transition_mask = (
                _IRON_NEUTRAL_TRANSITION_WAVENUMBER_CM <= wavenumber
            )
            active_transition_wavenumber = _IRON_NEUTRAL_TRANSITION_WAVENUMBER_CM[
                active_transition_mask
            ]
            iron_cross_section_rows[active_transition_mask, frequency_index] = 3e-18 / (
                1.0
                + (
                    (active_transition_wavenumber + 3000.0 - wavenumber)
                    / active_transition_wavenumber
                    / 0.1
                )
                ** 4
            )

    return dict(
        carbon_cross_section_rows=carbon_cross_section_rows,
        carbon_freefree_prefactor=carbon_freefree_prefactor,
        carbon_freefree_threshold=carbon_freefree_threshold,
        magnesium_cross_section_rows=magnesium_cross_section_rows,
        magnesium_freefree_prefactor=magnesium_freefree_prefactor,
        magnesium_freefree_threshold=magnesium_freefree_threshold,
        silicon_cross_section_rows=silicon_cross_section_rows,
        silicon_freefree_prefactor=silicon_freefree_prefactor,
        silicon_freefree_threshold=silicon_freefree_threshold,
        aluminum_cross_section=aluminum_cross_section,
        iron_cross_section_rows=iron_cross_section_rows,
    )


def _minor_terms(
    continuum_tables,
    frequency_hz,
    pops,
    photon_boltzmann_factor,
    stimulated_emission,
    hc_over_kt,
    coulomb_table_energy_first=False,
    hydrogen_partition=None,
):
    """Minor continuous absorption and scattering terms for one frequency."""
    temperature = pops["temperature"]
    mass_density_safe = torch.clamp(pops["mass_density"], min=1e-30)
    electron_density = pops["electron_density"]
    thermal_energy_ev = temperature * REFERENCE_BOLTZMANN_EV_PER_K
    natural_log_temperature = torch.log(torch.clamp(temperature, min=1e-10))
    hydrogen_neutral_partition_normalized_population = pops[
        "hydrogen_partition_normalized_ion_stage_populations"
    ][:, 0]
    hydrogen_ionized_partition_normalized_population = pops[
        "hydrogen_partition_normalized_ion_stage_populations"
    ][:, 1]
    wavenumber_cm = frequency_hz / LIGHT_SPEED_CM_PER_S
    minor_absorption = torch.zeros_like(temperature)
    minor_scattering = torch.zeros_like(temperature)

    if frequency_hz <= 3.28805e15:
        natural_log_frequency = math.log(frequency_hz)
        frequency_1e15 = frequency_hz / 1.0e15
        h2plus_frequency_polynomial = (
            -3.0233e3
            + (
                3.7797e2
                + (
                    -1.82496e1
                    + (3.9207e-1 - 3.1672e-3 * natural_log_frequency)
                    * natural_log_frequency
                )
                * natural_log_frequency
            )
            * natural_log_frequency
        )
        h2plus_energy_polynomial = (
            -7.342e-3
            + (
                -2.409e0
                + (
                    1.028e0
                    + (
                        -4.230e-1
                        + (1.224e-1 - 1.351e-2 * frequency_1e15) * frequency_1e15
                    )
                    * frequency_1e15
                )
                * frequency_1e15
            )
            * frequency_1e15
        )
        minor_absorption = minor_absorption + (
            torch.exp(
                -h2plus_energy_polynomial / thermal_energy_ev
                + h2plus_frequency_polynomial
                + torch.log(
                    torch.clamp(
                        hydrogen_neutral_partition_normalized_population, min=1e-40
                    )
                )
            )
            * 2.0
            * hydrogen_ionized_partition_normalized_population
            / mass_density_safe
            * stimulated_emission
        )

    helium_minus_temperature_coefficient = (
        3.397e-01 + (-5.216e14 + 7.039e30 / frequency_hz) / frequency_hz
    )
    helium_minus_offset = (
        -4.116e03 + (1.067e19 + 8.135e34 / frequency_hz) / frequency_hz
    )
    helium_minus_inverse_temperature = (
        5.081e08 + (-8.724e22 - 5.659e37 / frequency_hz) / frequency_hz
    )
    minor_absorption = minor_absorption + (
        (
            helium_minus_temperature_coefficient * temperature
            + helium_minus_offset
            + helium_minus_inverse_temperature / temperature
        )
        / 1.0e15
        * electron_density
        / 1.0e15
        * pops["helium_neutral_population"]
        / 1.0e15
        / mass_density_safe
    )

    if coulomb_table_energy_first:
        # Full near-UV metal forest: each helper returns a depth cross-section.
        for population_key, cross_section_fn in _METAL_BOUND_FREE_HELPERS:
            metal_cross_section = cross_section_fn(
                continuum_tables,
                frequency_hz,
                hc_over_kt,
            )
            if metal_cross_section is None:
                continue
            metal_population = (
                pops["carbon_partition_normalized_ion_stage_populations"][:, 0]
                if population_key == "carbon_neutral_partition_normalized_population"
                else pops[population_key]
            )
            minor_absorption = (
                minor_absorption
                + metal_cross_section
                * stimulated_emission
                * metal_population
                / mass_density_safe
            )
    else:
        # Compact optical-edge floors for the scalar synthesis layout.
        carbon_visible_boundfree = 1e-30 * torch.ones_like(temperature)
        if wavenumber_cm >= 22006.370:
            carbon_visible_boundfree = carbon_visible_boundfree + (
                2.1e-18
                * (22006.370 / wavenumber_cm) ** 1.5
                * 3.0
                * torch.exp(-68856.33 * hc_over_kt)
                * stimulated_emission
            )
        minor_absorption = (
            minor_absorption
            + carbon_visible_boundfree
            * pops["carbon_partition_normalized_ion_stage_populations"][:, 0]
            / mass_density_safe
        )

        magnesium_visible_edges = [
            (13713.986, 25e-18, 2.7, 15.0, 47957.034),
            (13823.223, 33.8e-18, 2.8, 9.0, 47847.797),
            (15267.955, 45e-18, 2.7, 5.0, 46403.065),
            (18167.687, 0.43e-18, 2.6, 1.0, 43503.333),
            (20473.617, 2.1e-18, 2.6, 3.0, 41197.043),
        ]
        magnesium_visible_boundfree = 1e-30 * torch.ones_like(temperature)
        for (
            threshold_wavenumber,
            cross_section0,
            power,
            statistical_weight,
            excitation_cm,
        ) in magnesium_visible_edges:
            if wavenumber_cm >= threshold_wavenumber:
                magnesium_visible_boundfree = magnesium_visible_boundfree + (
                    cross_section0
                    * (threshold_wavenumber / wavenumber_cm) ** power
                    * statistical_weight
                    * torch.exp(-excitation_cm * hc_over_kt)
                    * stimulated_emission
                )
        minor_absorption = (
            minor_absorption
            + magnesium_visible_boundfree
            * pops["magnesium_neutral_partition_normalized_population"]
            / mass_density_safe
        )

        aluminum_visible_edges = [
            (8002.467, 50e-18, 3, 6.0, 40275.903),
            (9346.231, 50e-18, 3, 10.0, 38932.139),
            (10588.957, 56.7e-18, 1.9, 2.0, 37689.413),
            (15318.007, 14.5e-18, 1, 6.0, 32960.363),
            (15842.129, 47e-18, 1.83, 10.0, 32436.241),
        ]
        aluminum_visible_boundfree = 1e-30 * torch.ones_like(temperature)
        aluminum_ionization_boltzmann = torch.exp(-48278.37 * hc_over_kt)
        for (
            threshold_wavenumber,
            cross_section0,
            power,
            statistical_weight,
            excitation_cm,
        ) in aluminum_visible_edges:
            if wavenumber_cm >= threshold_wavenumber:
                aluminum_visible_boundfree = aluminum_visible_boundfree + (
                    cross_section0
                    * (threshold_wavenumber / wavenumber_cm) ** power
                    * statistical_weight
                    * torch.exp(-excitation_cm * hc_over_kt)
                    * (1.0 - aluminum_ionization_boltzmann * photon_boltzmann_factor)
                )
        minor_absorption = (
            minor_absorption
            + aluminum_visible_boundfree
            * pops["aluminum_neutral_partition_normalized_population"]
            / mass_density_safe
        )

        silicon_visible_boundfree = 1e-30 * torch.ones_like(temperature)
        if wavenumber_cm >= 17777.641:
            silicon_visible_boundfree = silicon_visible_boundfree + (
                18e-18
                * (17777.641 / wavenumber_cm) ** 3
                * 15.0
                * torch.exp(-48161.459 * hc_over_kt)
                * (1.0 - photon_boltzmann_factor)
            )
        minor_absorption = (
            minor_absorption
            + silicon_visible_boundfree
            * pops["silicon_neutral_partition_normalized_population"]
            / mass_density_safe
        )

    # Helium Rayleigh scattering.
    helium_wavelength_angstrom = LIGHT_SPEED_ANGSTROM_PER_S / min(frequency_hz, 5.15e15)
    helium_wavelength_squared = helium_wavelength_angstrom**2
    helium_rayleigh_cross_section = (
        5.484e-14
        / (helium_wavelength_squared * helium_wavelength_squared)
        * (
            1.0
            + (2.44e5 + 5.94e10 / max(helium_wavelength_squared - 2.90e5, 1e-10))
            / helium_wavelength_squared
        )
        ** 2
    )
    minor_scattering = (
        minor_scattering
        + helium_rayleigh_cross_section
        * pops["helium_neutral_population"]
        / mass_density_safe
    )

    # Molecular-hydrogen Rayleigh scattering.
    h2_temperature_polynomial = (
        1.63660e-3
        + (
            -4.93992e-7
            + (
                1.11822e-10
                + (
                    -1.49567e-14
                    + (1.06206e-18 - 3.08720e-23 * temperature) * temperature
                )
                * temperature
            )
            * temperature
        )
        * temperature
    ) * temperature
    hydrogen_partition = (
        _hydrogen_partition(continuum_tables, temperature)
        if hydrogen_partition is None
        else hydrogen_partition
    )
    molecular_hydrogen_density = (
        (pops["hydrogen_neutral_population"] / hydrogen_partition * 2.0) ** 2
        * torch.exp(
            torch.clamp(
                4.478 / thermal_energy_ev
                - 4.64584e1
                + h2_temperature_polynomial
                - 1.5 * natural_log_temperature,
                -100,
                100,
            )
        )
        / mass_density_safe
    )
    h2_wavelength_angstrom = LIGHT_SPEED_ANGSTROM_PER_S / min(frequency_hz, 2.922e15)
    h2_wavelength_squared = h2_wavelength_angstrom**2
    h2_rayleigh_cross_section = (
        8.14e-13
        + 1.28e-6 / h2_wavelength_squared
        + 1.61 / (h2_wavelength_squared * h2_wavelength_squared)
    ) / (h2_wavelength_squared * h2_wavelength_squared)
    minor_scattering = (
        minor_scattering + h2_rayleigh_cross_section * molecular_hydrogen_density
    )
    return minor_absorption, minor_scattering


def _kramers_freefree_grid(
    freefree_prefactor,
    freefree_threshold,
    ionization_limit,
    hc_over_kt,
    dtype,
    device,
):
    prefactor = (
        freefree_prefactor
        if torch.is_tensor(freefree_prefactor)
        else torch.as_tensor(freefree_prefactor, dtype=dtype, device=device)
    )
    threshold = (
        freefree_threshold
        if torch.is_tensor(freefree_threshold)
        else torch.as_tensor(freefree_threshold, dtype=dtype, device=device)
    )
    return (
        prefactor[None, :]
        / hc_over_kt[:, None]
        * (
            torch.exp(-threshold[None, :] * hc_over_kt[:, None])
            - torch.exp(-ionization_limit * hc_over_kt[:, None])
        )
    )


def _metal_boundfree_opacity_grid(
    pops,
    stimulated_emission_grid,
    hc_over_kt,
    frequency_invariants,
    dtype,
    device,
):
    """Batched near-UV metal bound-free forest."""
    mass_density_safe = torch.clamp(pops["mass_density"], min=1e-30)
    metal_absorption = torch.zeros_like(stimulated_emission_grid)

    def constant_matrix(values):
        return (
            values
            if torch.is_tensor(values)
            else torch.as_tensor(values, dtype=dtype, device=device)
        )

    carbon_boltzmann = constant_matrix(_CARBON_NEUTRAL_LEVEL_STATISTICAL_WEIGHT)[
        None, :
    ] * torch.exp(
        -constant_matrix(_CARBON_NEUTRAL_LEVEL_ENERGY_CM)[None, :] * hc_over_kt[:, None]
    )
    carbon_cross_section = carbon_boltzmann @ frequency_invariants.tensor(
        "carbon_boundfree_cross_section_rows", dtype, device
    )
    carbon_cross_section = carbon_cross_section + _kramers_freefree_grid(
        frequency_invariants.tensor("carbon_freefree_prefactor", dtype, device),
        frequency_invariants.tensor("carbon_freefree_threshold", dtype, device),
        _CARBON_NEUTRAL_FREE_FREE_LIMIT_CM,
        hc_over_kt,
        dtype,
        device,
    )
    metal_absorption = (
        metal_absorption
        + carbon_cross_section
        * stimulated_emission_grid
        * pops["carbon_partition_normalized_ion_stage_populations"][:, 0][:, None]
        / mass_density_safe[:, None]
    )

    magnesium_boltzmann = constant_matrix(_MAGNESIUM_NEUTRAL_LEVEL_STATISTICAL_WEIGHT)[
        None, :
    ] * torch.exp(
        -constant_matrix(_MAGNESIUM_NEUTRAL_LEVEL_ENERGY_CM)[None, :]
        * hc_over_kt[:, None]
    )
    magnesium_cross_section = magnesium_boltzmann @ frequency_invariants.tensor(
        "magnesium_boundfree_cross_section_rows", dtype, device
    )
    magnesium_cross_section = magnesium_cross_section + _kramers_freefree_grid(
        frequency_invariants.tensor("magnesium_freefree_prefactor", dtype, device),
        frequency_invariants.tensor("magnesium_freefree_threshold", dtype, device),
        _MAGNESIUM_NEUTRAL_IONIZATION_LIMIT_CM,
        hc_over_kt,
        dtype,
        device,
    )
    metal_absorption = (
        metal_absorption
        + magnesium_cross_section
        * stimulated_emission_grid
        * pops["magnesium_neutral_partition_normalized_population"][:, None]
        / mass_density_safe[:, None]
    )

    silicon_boltzmann = constant_matrix(_SILICON_NEUTRAL_LEVEL_STATISTICAL_WEIGHT)[
        None, :
    ] * torch.exp(
        -constant_matrix(_SILICON_NEUTRAL_LEVEL_ENERGY_CM)[None, :]
        * hc_over_kt[:, None]
    )
    silicon_cross_section = silicon_boltzmann @ frequency_invariants.tensor(
        "silicon_boundfree_cross_section_rows", dtype, device
    )
    silicon_cross_section = silicon_cross_section + _kramers_freefree_grid(
        frequency_invariants.tensor("silicon_freefree_prefactor", dtype, device),
        frequency_invariants.tensor("silicon_freefree_threshold", dtype, device),
        _SILICON_NEUTRAL_FINE_STRUCTURE_LIMIT_CM,
        hc_over_kt,
        dtype,
        device,
    )
    metal_absorption = (
        metal_absorption
        + silicon_cross_section
        * stimulated_emission_grid
        * pops["silicon_neutral_partition_normalized_population"][:, None]
        / mass_density_safe[:, None]
    )

    aluminum_cross_section = frequency_invariants.tensor(
        "aluminum_boundfree_cross_section", dtype, device
    )
    metal_absorption = (
        metal_absorption
        + aluminum_cross_section[None, :]
        * stimulated_emission_grid
        * pops["aluminum_neutral_partition_normalized_population"][:, None]
        / mass_density_safe[:, None]
    )

    iron_boltzmann = constant_matrix(_IRON_NEUTRAL_TRANSITION_STATISTICAL_WEIGHT)[
        None, :
    ] * torch.exp(
        -constant_matrix(_IRON_NEUTRAL_TRANSITION_ENERGY_CM)[None, :]
        * hc_over_kt[:, None]
    )
    iron_cross_section = iron_boltzmann @ frequency_invariants.tensor(
        "iron_boundfree_cross_section_rows", dtype, device
    )
    metal_absorption = (
        metal_absorption
        + iron_cross_section
        * stimulated_emission_grid
        * pops["iron_neutral_partition_normalized_population"][:, None]
        / mass_density_safe[:, None]
    )
    return metal_absorption


def _minor_terms_grid(
    pops,
    stimulated_emission_grid,
    hc_over_kt,
    hydrogen_partition,
    frequency_invariants,
    dtype,
    device,
):
    """Batched minor absorption/scattering grids."""
    temperature = pops["temperature"]
    mass_density_safe = torch.clamp(pops["mass_density"], min=1e-30)
    electron_density = pops["electron_density"]
    thermal_energy_ev = temperature * REFERENCE_BOLTZMANN_EV_PER_K
    natural_log_temperature = torch.log(torch.clamp(temperature, min=1e-10))
    hydrogen_neutral_partition_normalized_population = pops[
        "hydrogen_partition_normalized_ion_stage_populations"
    ][:, 0]
    hydrogen_ionized_partition_normalized_population = pops[
        "hydrogen_partition_normalized_ion_stage_populations"
    ][:, 1]
    frequency_hz = frequency_invariants.tensor("frequencies_hz", dtype, device)
    natural_log_frequency = frequency_invariants.tensor(
        "natural_log_frequency", dtype, device
    )
    minor_absorption = torch.zeros_like(stimulated_emission_grid)

    h2plus_active = (frequency_hz <= 3.28805e15).to(dtype)
    frequency_1e15 = frequency_hz / 1.0e15
    h2plus_frequency_polynomial = (
        -3.0233e3
        + (
            3.7797e2
            + (
                -1.82496e1
                + (3.9207e-1 - 3.1672e-3 * natural_log_frequency)
                * natural_log_frequency
            )
            * natural_log_frequency
        )
        * natural_log_frequency
    )
    h2plus_energy_polynomial = (
        -7.342e-3
        + (
            -2.409e0
            + (
                1.028e0
                + (-4.230e-1 + (1.224e-1 - 1.351e-2 * frequency_1e15) * frequency_1e15)
                * frequency_1e15
            )
            * frequency_1e15
        )
        * frequency_1e15
    )
    minor_absorption = minor_absorption + (
        torch.exp(
            -h2plus_energy_polynomial[None, :] / thermal_energy_ev[:, None]
            + h2plus_frequency_polynomial[None, :]
            + torch.log(
                torch.clamp(hydrogen_neutral_partition_normalized_population, min=1e-40)
            )[:, None]
        )
        * 2.0
        * hydrogen_ionized_partition_normalized_population[:, None]
        / mass_density_safe[:, None]
        * stimulated_emission_grid
        * h2plus_active[None, :]
    )

    helium_minus_temperature_coefficient = (
        3.397e-01 + (-5.216e14 + 7.039e30 / frequency_hz) / frequency_hz
    )
    helium_minus_offset = (
        -4.116e03 + (1.067e19 + 8.135e34 / frequency_hz) / frequency_hz
    )
    helium_minus_inverse_temperature = (
        5.081e08 + (-8.724e22 - 5.659e37 / frequency_hz) / frequency_hz
    )
    minor_absorption = minor_absorption + (
        (
            helium_minus_temperature_coefficient[None, :] * temperature[:, None]
            + helium_minus_offset[None, :]
            + helium_minus_inverse_temperature[None, :] / temperature[:, None]
        )
        / 1.0e15
        * electron_density[:, None]
        / 1.0e15
        * pops["helium_neutral_population"][:, None]
        / 1.0e15
        / mass_density_safe[:, None]
    )

    minor_absorption = minor_absorption + _metal_boundfree_opacity_grid(
        pops,
        stimulated_emission_grid,
        hc_over_kt,
        frequency_invariants,
        dtype,
        device,
    )

    helium_wavelength_angstrom = LIGHT_SPEED_ANGSTROM_PER_S / torch.minimum(
        frequency_hz,
        torch.full_like(frequency_hz, 5.15e15),
    )
    helium_wavelength_squared = helium_wavelength_angstrom * helium_wavelength_angstrom
    helium_rayleigh_cross_section = (
        5.484e-14
        / (helium_wavelength_squared * helium_wavelength_squared)
        * (
            1.0
            + (
                2.44e5
                + 5.94e10 / torch.clamp(helium_wavelength_squared - 2.90e5, min=1e-10)
            )
            / helium_wavelength_squared
        )
        ** 2
    )
    minor_scattering = (
        helium_rayleigh_cross_section[None, :]
        * pops["helium_neutral_population"][:, None]
        / mass_density_safe[:, None]
    )

    h2_temperature_polynomial = (
        1.63660e-3
        + (
            -4.93992e-7
            + (
                1.11822e-10
                + (
                    -1.49567e-14
                    + (1.06206e-18 - 3.08720e-23 * temperature) * temperature
                )
                * temperature
            )
            * temperature
        )
        * temperature
    ) * temperature
    molecular_hydrogen_density = (
        (pops["hydrogen_neutral_population"] / hydrogen_partition * 2.0) ** 2
        * torch.exp(
            torch.clamp(
                4.478 / thermal_energy_ev
                - 4.64584e1
                + h2_temperature_polynomial
                - 1.5 * natural_log_temperature,
                -100,
                100,
            )
        )
        / mass_density_safe
    )
    h2_wavelength_angstrom = LIGHT_SPEED_ANGSTROM_PER_S / torch.minimum(
        frequency_hz,
        torch.full_like(frequency_hz, 2.922e15),
    )
    h2_wavelength_squared = h2_wavelength_angstrom * h2_wavelength_angstrom
    h2_rayleigh_cross_section = (
        8.14e-13
        + 1.28e-6 / h2_wavelength_squared
        + 1.61 / (h2_wavelength_squared * h2_wavelength_squared)
    ) / (h2_wavelength_squared * h2_wavelength_squared)
    minor_scattering = minor_scattering + (
        h2_rayleigh_cross_section[None, :] * molecular_hydrogen_density[:, None]
    )
    return minor_absorption, minor_scattering


def _helium_opacity(
    continuum_tables,
    frequency_hz,
    pops,
    photon_boltzmann_factor,
    stimulated_emission,
    hc_over_kt,
    coulomb_table_energy_first=False,
    helium_neutral_coulomb_freefree=None,
    helium_ionized_coulomb_freefree=None,
):
    """Helium I and II bound-free plus free-free opacity for one frequency."""
    temperature = pops["temperature"]
    mass_density_safe = torch.clamp(pops["mass_density"], min=1e-30)
    electron_density = pops["electron_density"]
    natural_log_temperature = torch.log(torch.clamp(temperature, min=1e-10))
    wavenumber_cm = frequency_hz / LIGHT_SPEED_CM_PER_S
    natural_log_frequency = math.log(frequency_hz)
    hydrogenic_frequency_factor = 2.815e29 / (
        frequency_hz * frequency_hz * frequency_hz
    )

    rydberg_he = 109722.267
    helium_neutral_boundfree = (
        hydrogenic_frequency_factor
        * 4.0
        / 2.0
        / (rydberg_he * hc_over_kt)
        * (
            torch.exp(-max(195262.919, 198310.76 - wavenumber_cm) * hc_over_kt)
            - torch.exp(-198310.76 * hc_over_kt)
        )
        * stimulated_emission
    )
    for threshold_wavenumber, statistical_weight, excitation_cm, branch_index in (
        _HELIUM_NEUTRAL_N5_AUTOIONIZATION_TRANSITIONS
        + _HELIUM_NEUTRAL_N4_AUTOIONIZATION_TRANSITIONS
    ):
        if wavenumber_cm >= threshold_wavenumber:
            cross_section = hydrogenic_frequency_factor / (
                3125.0 if branch_index >= 19 else 1024.0
            )
            helium_neutral_boundfree = (
                helium_neutral_boundfree
                + cross_section
                * statistical_weight
                * torch.exp(-excitation_cm * hc_over_kt)
                * stimulated_emission
            )
    for (
        threshold_wavenumber,
        coefficient_pair,
        statistical_weight,
        excitation_cm,
        _branch_index,
    ) in (
        _HELIUM_NEUTRAL_N3_AUTOIONIZATION_TRANSITIONS
        + _HELIUM_NEUTRAL_N2_AUTOIONIZATION_TRANSITIONS
    ):
        if wavenumber_cm >= threshold_wavenumber:
            cross_section = math.exp(
                coefficient_pair[0] + coefficient_pair[1] * natural_log_frequency
            )
            helium_neutral_boundfree = (
                helium_neutral_boundfree
                + cross_section
                * statistical_weight
                * torch.exp(-excitation_cm * hc_over_kt)
                * stimulated_emission
            )
    if wavenumber_cm >= 38454.691:
        helium_neutral_boundfree = (
            helium_neutral_boundfree
            + math.exp(
                -390.026
                + (21.035 - 0.318 * natural_log_frequency) * natural_log_frequency
            )
            * 3.0
            * torch.exp(-159856.069 * hc_over_kt)
            * stimulated_emission
        )
    if wavenumber_cm >= 198310.760:
        helium_neutral_boundfree = (
            helium_neutral_boundfree
            + math.exp(33.32 - 2.0 * natural_log_frequency) * stimulated_emission
        )
    helium_neutral_boundfree = (
        helium_neutral_boundfree
        * pops["helium_neutral_partition_normalized_population"]
        / mass_density_safe
    )

    helium_neutral_coulomb_freefree = (
        _coulomb_freefree_gaunt(
            continuum_tables,
            1,
            natural_log_frequency,
            temperature,
            natural_log_temperature,
            energy_first_layout=coulomb_table_energy_first,
        )
        if helium_neutral_coulomb_freefree is None
        else helium_neutral_coulomb_freefree
    )
    # The free-free prefactor differs between the standalone and sampled
    # continuum layouts; the layout switch keeps the validated parity behavior.
    helium_neutral_freefree_constant = (
        3.6919e8 if coulomb_table_energy_first else 3.619e8
    )
    helium_neutral_opacity = (
        helium_neutral_boundfree
        + helium_neutral_freefree_constant
        / torch.sqrt(temperature)
        * helium_neutral_coulomb_freefree
        / frequency_hz
        * electron_density
        / frequency_hz
        * pops["helium_singly_ionized_population"]
        / frequency_hz
        * stimulated_emission
        / mass_density_safe
    )

    rydberg_he2 = 438889.068
    helium_ionized_over_density = (
        pops["helium_singly_ionized_partition_normalized_population"]
        / mass_density_safe
    )
    helium_ionized_opacity = (
        hydrogenic_frequency_factor
        * 16.0
        * 2.0
        / 2.0
        / (rydberg_he2 * hc_over_kt)
        * (
            torch.exp(-max(434519.959, 438908.85 - wavenumber_cm) * hc_over_kt)
            - torch.exp(-438908.85 * hc_over_kt)
        )
        * stimulated_emission
        * helium_ionized_over_density
    )
    for (
        threshold_wavenumber,
        statistical_weight,
        excitation_cm,
        divisor,
    ) in _HELIUM_SINGLY_IONIZED_HIGH_LEVEL_BOUND_FREE_TRANSITIONS:
        if wavenumber_cm >= threshold_wavenumber:
            helium_ionized_opacity = (
                helium_ionized_opacity
                + hydrogenic_frequency_factor
                * 16.0
                / divisor
                * statistical_weight
                * torch.exp(-excitation_cm * hc_over_kt)
                * stimulated_emission
                * helium_ionized_over_density
            )
    for (
        threshold_wavenumber,
        statistical_weight,
        excitation_cm,
        divisor,
        polynomial,
    ) in _HELIUM_SINGLY_IONIZED_LOW_LEVEL_BOUND_FREE_TRANSITIONS:
        if wavenumber_cm >= threshold_wavenumber:
            cross_section = (
                hydrogenic_frequency_factor
                * 16.0
                / divisor
                * (
                    polynomial[0]
                    + (polynomial[1] + polynomial[2] / frequency_hz) / frequency_hz
                )
            )
            stimulated_factor = (
                stimulated_emission
                if excitation_cm == 0.0
                else torch.exp(-excitation_cm * hc_over_kt) * stimulated_emission
            )
            helium_ionized_opacity = (
                helium_ionized_opacity
                + cross_section
                * statistical_weight
                * stimulated_factor
                * helium_ionized_over_density
            )
    helium_ionized_coulomb_freefree = (
        _coulomb_freefree_gaunt(
            continuum_tables,
            2,
            natural_log_frequency,
            temperature,
            natural_log_temperature,
            energy_first_layout=coulomb_table_energy_first,
        )
        if helium_ionized_coulomb_freefree is None
        else helium_ionized_coulomb_freefree
    )
    helium_ionized_opacity = (
        helium_ionized_opacity
        + 3.6919e8
        * 4.0
        / torch.sqrt(temperature)
        * helium_ionized_coulomb_freefree
        / frequency_hz
        * electron_density
        / frequency_hz
        * pops["helium_doubly_ionized_partition_normalized_population"]
        / frequency_hz
        * stimulated_emission
        / mass_density_safe
    )
    return helium_neutral_opacity, helium_ionized_opacity


def _helium_opacity_grid(
    pops,
    photon_boltzmann_grid,
    stimulated_emission_grid,
    hc_over_kt,
    helium_neutral_coulomb_freefree_grid,
    helium_ionized_coulomb_freefree_grid,
    frequency_invariants,
    dtype,
    device,
):
    """Batched He I + He II bound-free/free-free opacity on ``(depth, frequency)``."""
    temperature = pops["temperature"]
    mass_density_safe = torch.clamp(pops["mass_density"], min=1e-30)
    electron_density = pops["electron_density"]
    frequency_hz = frequency_invariants.tensor("frequencies_hz", dtype, device)
    wavenumber_cm = frequency_hz / LIGHT_SPEED_CM_PER_S
    hydrogenic_frequency_factor = 2.815e29 / (
        frequency_hz * frequency_hz * frequency_hz
    )

    helium_neutral_population = pops["helium_neutral_partition_normalized_population"]
    helium_neutral_thermal_energy_ev = temperature * REFERENCE_BOLTZMANN_EV_PER_K
    helium_neutral_boltzmann = torch.exp(
        -torch.as_tensor(_HELIUM_NEUTRAL_EXCITATION_EV, dtype=dtype, device=device)[
            None, :
        ]
        / torch.clamp(helium_neutral_thermal_energy_ev[:, None], min=1e-300)
    )
    helium_neutral_boltzmann = (
        helium_neutral_boltzmann
        * torch.as_tensor(
            _HELIUM_NEUTRAL_STATISTICAL_WEIGHTS, dtype=dtype, device=device
        )[None, :]
    )
    helium_neutral_boltzmann = (
        helium_neutral_boltzmann
        * helium_neutral_population[:, None]
        / mass_density_safe[:, None]
    )

    high_level_quantum_numbers = torch.arange(4, 28, dtype=dtype, device=device)
    helium_neutral_high_level_boltzmann = torch.exp(
        -24.587
        * (1.0 - 1.0 / (high_level_quantum_numbers * high_level_quantum_numbers))
        / torch.clamp(helium_neutral_thermal_energy_ev[:, None], min=1e-300)
    )
    helium_neutral_high_level_boltzmann = (
        helium_neutral_high_level_boltzmann
        * 4.0
        * high_level_quantum_numbers[None, :]
        * high_level_quantum_numbers[None, :]
        * helium_neutral_population[:, None]
        / mass_density_safe[:, None]
    )

    helium_neutral_transition_cross_sections = frequency_invariants.tensor(
        "neutral_helium_low_level_photoionization_cross_sections",
        dtype,
        device,
    )
    helium_neutral_boundfree = (
        helium_neutral_boltzmann @ helium_neutral_transition_cross_sections
    )
    high_level_mask = (frequency_hz >= 1.25408e16).to(dtype)
    high_level_cross_sections = frequency_invariants.tensor(
        "neutral_helium_high_level_photoionization_cross_sections",
        dtype,
        device,
    )[4:28]
    helium_neutral_boundfree = (
        helium_neutral_boundfree
        + (helium_neutral_high_level_boltzmann @ high_level_cross_sections)
        * high_level_mask[None, :]
    )

    helium_neutral_excited_scale = (
        helium_neutral_population
        * (4.0 / 2.0 / 13.595)
        * helium_neutral_thermal_energy_ev
        / mass_density_safe
    )
    helium_neutral_excited_boltzmann = (
        torch.exp(-23.730 / torch.clamp(helium_neutral_thermal_energy_ev, min=1e-300))
        * helium_neutral_excited_scale
    )
    helium_neutral_limit_boltzmann = (
        torch.exp(-24.587 / torch.clamp(helium_neutral_thermal_energy_ev, min=1e-300))
        * helium_neutral_excited_scale
    )
    low_frequency_mask = frequency_hz < 2.055e14
    helium_neutral_excited_population = torch.where(
        low_frequency_mask[None, :],
        helium_neutral_limit_boltzmann[:, None]
        / torch.clamp(photon_boltzmann_grid, min=1e-300),
        helium_neutral_excited_boltzmann[:, None],
    )
    helium_neutral_opacity = (
        helium_neutral_excited_population - helium_neutral_limit_boltzmann[:, None]
    ) * hydrogenic_frequency_factor[None, :]
    helium_neutral_opacity = helium_neutral_opacity + helium_neutral_boundfree
    helium_neutral_freefree_scale = (
        electron_density
        * pops["helium_singly_ionized_population"]
        / mass_density_safe
        / torch.sqrt(torch.clamp(temperature, min=1e-300))
    )
    helium_neutral_opacity = (
        helium_neutral_opacity
        + helium_neutral_coulomb_freefree_grid
        * (3.6919e8 / (frequency_hz * frequency_hz * frequency_hz))[None, :]
        * helium_neutral_freefree_scale[:, None]
    )
    helium_neutral_opacity = helium_neutral_opacity * stimulated_emission_grid

    rydberg_he2 = 438889.068
    helium_ionized_over_density = (
        pops["helium_singly_ionized_partition_normalized_population"]
        / mass_density_safe
    )
    helium_ionized_edge = torch.maximum(
        torch.full_like(wavenumber_cm, 434519.959),
        438908.85 - wavenumber_cm,
    )
    helium_ionized_opacity = (
        hydrogenic_frequency_factor[None, :]
        * 16.0
        * 2.0
        / 2.0
        / (rydberg_he2 * hc_over_kt[:, None])
    )
    helium_ionized_opacity = helium_ionized_opacity * (
        torch.exp(-helium_ionized_edge[None, :] * hc_over_kt[:, None])
        - torch.exp(-438908.85 * hc_over_kt[:, None])
    )
    helium_ionized_opacity = (
        helium_ionized_opacity
        * stimulated_emission_grid
        * helium_ionized_over_density[:, None]
    )
    for (
        threshold_wavenumber,
        statistical_weight,
        excitation_cm,
        divisor,
    ) in _HELIUM_SINGLY_IONIZED_HIGH_LEVEL_BOUND_FREE_TRANSITIONS:
        active = (wavenumber_cm >= threshold_wavenumber).to(dtype)
        helium_ionized_opacity = (
            helium_ionized_opacity
            + (hydrogenic_frequency_factor * 16.0 / divisor * active)[None, :]
            * statistical_weight
            * torch.exp(-excitation_cm * hc_over_kt[:, None])
            * stimulated_emission_grid
            * helium_ionized_over_density[:, None]
        )
    for (
        threshold_wavenumber,
        statistical_weight,
        excitation_cm,
        divisor,
        polynomial,
    ) in _HELIUM_SINGLY_IONIZED_LOW_LEVEL_BOUND_FREE_TRANSITIONS:
        active = (wavenumber_cm >= threshold_wavenumber).to(dtype)
        cross_section = (
            hydrogenic_frequency_factor
            * 16.0
            / divisor
            * (
                polynomial[0]
                + (polynomial[1] + polynomial[2] / frequency_hz) / frequency_hz
            )
            * active
        )
        stimulated_factor = (
            stimulated_emission_grid
            if excitation_cm == 0.0
            else torch.exp(-excitation_cm * hc_over_kt[:, None])
            * stimulated_emission_grid
        )
        helium_ionized_opacity = (
            helium_ionized_opacity
            + cross_section[None, :]
            * statistical_weight
            * stimulated_factor
            * helium_ionized_over_density[:, None]
        )
    helium_ionized_opacity = (
        helium_ionized_opacity
        + 3.6919e8
        * 4.0
        / torch.sqrt(temperature[:, None])
        * helium_ionized_coulomb_freefree_grid
        / frequency_hz[None, :]
        * electron_density[:, None]
        / frequency_hz[None, :]
        * pops["helium_doubly_ionized_partition_normalized_population"][:, None]
        / frequency_hz[None, :]
        * stimulated_emission_grid
        / mass_density_safe[:, None]
    )
    return helium_neutral_opacity, helium_ionized_opacity


def _hot_metal_and_silicon_singly_ionized_opacity(
    continuum_tables,
    frequency_hz,
    pops,
    stimulated_emission,
    thermal_energy_ev,
    natural_log_temperature,
    coulomb_table_energy_first=False,
    coulomb_freefree_by_charge=None,
    silicon_singly_ionized_peach_frequency_row=None,
    silicon_singly_ionized_peach_base_cross_section=None,
):
    """Hot-metal and Si II Peach opacity for one frequency."""
    temperature = pops["temperature"]
    mass_density_safe = torch.clamp(pops["mass_density"], min=1e-30)
    electron_density = pops["electron_density"]
    natural_log_frequency = math.log(frequency_hz)
    sqrt_temperature = torch.sqrt(torch.clamp(temperature, min=1e-30))
    charge_weighted_freefree = torch.zeros_like(temperature)
    for charge in range(1, 6):
        coulomb_freefree = (
            _coulomb_freefree_gaunt(
                continuum_tables,
                charge,
                natural_log_frequency,
                temperature,
                natural_log_temperature,
                energy_first_layout=coulomb_table_energy_first,
            )
            if coulomb_freefree_by_charge is None
            else coulomb_freefree_by_charge[charge - 1]
        )
        charge_weighted_freefree = (
            charge_weighted_freefree
            + coulomb_freefree * pops["charge_square_population_sum"][:, charge - 1]
        )
    hot_metal_opacity = (
        charge_weighted_freefree
        * (3.6919e8 / frequency_hz**3)
        * (electron_density / sqrt_temperature)
    )
    thermal_energy_ev_safe = torch.clamp(thermal_energy_ev, min=1e-30)
    hot_metal_transitions = continuum_tables.arrays[
        "hot_metal_boundfree_transition_table"
    ]
    population_column = np.clip(
        hot_metal_transitions[:, 6].astype(np.int64) - 1,
        0,
        20,
    )
    for transition_index in range(hot_metal_transitions.shape[0]):
        (
            threshold_frequency,
            cross_section0,
            alpha0,
            power0,
            multiplier0,
            excitation_ev,
            _population_slot,
        ) = hot_metal_transitions[transition_index]
        if frequency_hz < threshold_frequency:
            continue
        frequency_ratio = threshold_frequency / frequency_hz
        cross_section = (
            cross_section0
            * (alpha0 + frequency_ratio - alpha0 * frequency_ratio)
            * math.sqrt(frequency_ratio ** int(power0))
        )
        transition_opacity = (
            cross_section
            * pops["hot_metal_populations"][:, population_column[transition_index]]
            * multiplier0
        )
        excitation_factor = torch.exp(-excitation_ev / thermal_energy_ev_safe)
        hot_metal_opacity = torch.where(
            transition_opacity > hot_metal_opacity / 100.0,
            hot_metal_opacity + transition_opacity * excitation_factor,
            hot_metal_opacity,
        )
    hot_metal_opacity = hot_metal_opacity * stimulated_emission / mass_density_safe

    if silicon_singly_ionized_peach_base_cross_section is None:
        silicon_singly_ionized_peach_base_cross_section_host = (
            _silicon_singly_ionized_peach_opacity(
                continuum_tables.arrays,
                frequency_hz,
                natural_log_frequency,
                pops["_temperature_host"],
                pops["_natural_log_temperature_host"],
                frequency_row=silicon_singly_ionized_peach_frequency_row,
            )
        )
        silicon_singly_ionized_peach_base_cross_section = torch.as_tensor(
            silicon_singly_ionized_peach_base_cross_section_host,
            dtype=temperature.dtype,
            device=temperature.device,
        )
    silicon_singly_ionized_opacity = (
        silicon_singly_ionized_peach_base_cross_section
        * pops["silicon_singly_ionized_partition_normalized_population"]
        * stimulated_emission
        / mass_density_safe
    )
    return hot_metal_opacity, silicon_singly_ionized_opacity


def _silicon_singly_ionized_peach_base_grid(pops, frequency_invariants, dtype, device):
    """Depth-blended Si II Peach cross-section on ``(depth, frequency)``.

    This is the T-dependent part of ``_silicon_singly_ionized_peach_opacity`` without the final population/stim/rho
    scaling.  It removes the remaining per-frequency host loop over 80 depths.
    """
    temperature_host = np.asarray(pops["_temperature_host"], dtype=np.float64)
    natural_log_temperature_host = np.asarray(
        pops["_natural_log_temperature_host"], dtype=np.float64
    )
    silicon_singly_ionized_peach_frequency_rows = (
        frequency_invariants.silicon_singly_ionized_peach_frequency_rows
    )
    temperature_index = np.clip((temperature_host / 2000.0).astype(int) - 4, 1, 5)
    temperature_weight = (
        natural_log_temperature_host
        - frequency_invariants.silicon_singly_ionized_peach_natural_log_temperature_grid[
            temperature_index - 1
        ]
    ) / (
        frequency_invariants.silicon_singly_ionized_peach_natural_log_temperature_grid[
            temperature_index
        ]
        - frequency_invariants.silicon_singly_ionized_peach_natural_log_temperature_grid[
            temperature_index - 1
        ]
    )
    lower_index = temperature_index - 1
    upper_index = temperature_index
    cross_section_by_frequency_depth = (
        silicon_singly_ionized_peach_frequency_rows[:, lower_index]
        * (1.0 - temperature_weight[None, :])
        + silicon_singly_ionized_peach_frequency_rows[:, upper_index]
        * temperature_weight[None, :]
    )
    return torch.as_tensor(
        np.ascontiguousarray((np.exp(cross_section_by_frequency_depth) * 6.0).T),
        dtype=dtype,
        device=device,
    )


def _light_element_opacity_grid(
    pops,
    stimulated_emission_grid,
    hc_over_kt,
    thermal_energy_ev,
    frequency_invariants,
    dtype,
    device,
):
    """High-impact light-element photoionization terms: N I, O I, Mg II, Ca II."""
    mass_density_safe = torch.clamp(pops["mass_density"], min=1e-30)
    frequency_hz = frequency_invariants.tensor("frequencies_hz", dtype, device)
    wavenumber_cm = frequency_hz / LIGHT_SPEED_CM_PER_S
    nitrogen_edge_cross_sections = frequency_invariants.tensor(
        "nitrogen_edge_cross_sections", dtype, device
    )
    oxygen_edge_cross_section = frequency_invariants.tensor(
        "oxygen_911_cross_section", dtype, device
    )
    calcium_edge_cross_sections = frequency_invariants.tensor(
        "calcium_edge_cross_sections", dtype, device
    )

    thermal_energy_ev_safe = torch.clamp(thermal_energy_ev, min=1e-300)
    nitrogen_1130_population_factor = 6.0 * torch.exp(-3.575 / thermal_energy_ev_safe)
    nitrogen_1020_population_factor = 10.0 * torch.exp(-2.384 / thermal_energy_ev_safe)
    nitrogen_cross_section = (
        4.0 * nitrogen_edge_cross_sections[0][None, :]
        + nitrogen_1020_population_factor[:, None]
        * nitrogen_edge_cross_sections[1][None, :]
        + nitrogen_1130_population_factor[:, None]
        * nitrogen_edge_cross_sections[2][None, :]
    )
    light_element_opacity = (
        nitrogen_cross_section
        * pops["nitrogen_neutral_partition_normalized_population"][:, None]
        * stimulated_emission_grid
        / mass_density_safe[:, None]
    )

    light_element_opacity = (
        light_element_opacity
        + (9.0 * oxygen_edge_cross_section[None, :])
        * pops["oxygen_neutral_partition_normalized_population"][:, None]
        * stimulated_emission_grid
        / mass_density_safe[:, None]
    )

    calcium_1218_population_factor = 10.0 * torch.exp(-1.697 / thermal_energy_ev_safe)
    calcium_1420_population_factor = 6.0 * torch.exp(-3.142 / thermal_energy_ev_safe)
    calcium_cross_section = (
        2.0 * calcium_edge_cross_sections[0][None, :]
        + calcium_1218_population_factor[:, None]
        * calcium_edge_cross_sections[1][None, :]
        + calcium_1420_population_factor[:, None]
        * calcium_edge_cross_sections[2][None, :]
    )
    light_element_opacity = (
        light_element_opacity
        + calcium_cross_section
        * pops["calcium_singly_ionized_partition_normalized_population"][:, None]
        * stimulated_emission_grid
        / mass_density_safe[:, None]
    )

    magnesium_cross_section_table = frequency_invariants.tensor(
        "magnesium_ionized_cross_section_rows", dtype, device
    )
    magnesium_boltzmann_weights = torch.as_tensor(
        _MAGNESIUM_SINGLY_IONIZED_LEVEL_STATISTICAL_WEIGHT, dtype=dtype, device=device
    )[None, :] * torch.exp(
        -torch.as_tensor(
            _MAGNESIUM_SINGLY_IONIZED_LEVEL_ENERGY_CM, dtype=dtype, device=device
        )[None, :]
        * hc_over_kt[:, None]
    )
    magnesium_boundfree = magnesium_boltzmann_weights @ magnesium_cross_section_table
    magnesium_limit_factor = torch.exp(
        -_MAGNESIUM_SINGLY_IONIZED_IONIZATION_LIMIT_CM * hc_over_kt
    )
    magnesium_frequency_factor = (
        2.815e29
        / torch.clamp(frequency_hz * frequency_hz * frequency_hz, min=1e-300)
        * _MAGNESIUM_SINGLY_IONIZED_CHARGE**4
    )
    magnesium_prefactor = magnesium_frequency_factor[None, :] / (
        _MAGNESIUM_RYDBERG_WAVENUMBER_CM
        * _MAGNESIUM_SINGLY_IONIZED_CHARGE**2
        * hc_over_kt[:, None]
    )
    magnesium_tail_edge = torch.maximum(
        torch.full_like(
            wavenumber_cm, _MAGNESIUM_SINGLY_IONIZED_FREE_FREE_THRESHOLD_CM
        ),
        _MAGNESIUM_SINGLY_IONIZED_IONIZATION_LIMIT_CM - wavenumber_cm,
    )
    magnesium_cross_section = magnesium_prefactor * (
        torch.exp(-magnesium_tail_edge[None, :] * hc_over_kt[:, None])
        - magnesium_limit_factor[:, None]
    )
    magnesium_cross_section = magnesium_cross_section + magnesium_boundfree
    light_element_opacity = (
        light_element_opacity
        + magnesium_cross_section
        * pops["magnesium_singly_ionized_partition_normalized_population"][:, None]
        * stimulated_emission_grid
        / mass_density_safe[:, None]
    )
    return light_element_opacity


def _hot_metal_opacity_grid(
    continuum_tables,
    pops,
    stimulated_emission_grid,
    thermal_energy_ev,
    coulomb_freefree_by_charge,
    frequency_invariants,
    dtype,
    device,
):
    """Batched hot-metal free-free + transition opacity on ``(depth, frequency)``."""
    temperature = pops["temperature"]
    mass_density_safe = torch.clamp(pops["mass_density"], min=1e-30)
    electron_density = pops["electron_density"]
    frequency_hz = frequency_invariants.tensor("frequencies_hz", dtype, device)
    sqrt_temperature = torch.sqrt(torch.clamp(temperature, min=1e-30))
    charge_weighted_freefree = torch.zeros(
        temperature.shape[0],
        frequency_hz.shape[0],
        dtype=dtype,
        device=device,
    )
    for charge in range(1, 6):
        charge_weighted_freefree = (
            charge_weighted_freefree
            + coulomb_freefree_by_charge[charge].transpose(0, 1)
            * pops["charge_square_population_sum"][:, charge - 1][:, None]
        )
    hot_metal_opacity = (
        charge_weighted_freefree
        * (3.6919e8 / (frequency_hz * frequency_hz * frequency_hz))[None, :]
        * (electron_density / sqrt_temperature)[:, None]
    )

    thermal_energy_ev_safe = torch.clamp(thermal_energy_ev, min=1e-30)
    hot_metal_transitions = continuum_tables.arrays[
        "hot_metal_boundfree_transition_table"
    ]
    population_column = np.clip(
        hot_metal_transitions[:, 6].astype(np.int64) - 1,
        0,
        20,
    )
    for transition_index in range(hot_metal_transitions.shape[0]):
        (
            threshold_frequency,
            cross_section0,
            alpha0,
            power0,
            multiplier0,
            excitation_ev,
            _population_slot,
        ) = hot_metal_transitions[transition_index]
        active = (frequency_hz >= float(threshold_frequency)).to(dtype)
        frequency_ratio = (
            torch.as_tensor(float(threshold_frequency), dtype=dtype, device=device)
            / frequency_hz
        )
        cross_section = (
            cross_section0
            * (alpha0 + frequency_ratio - alpha0 * frequency_ratio)
            * torch.sqrt(frequency_ratio ** int(power0))
            * active
        )
        transition_opacity = (
            cross_section[None, :]
            * pops["hot_metal_populations"][:, population_column[transition_index]][
                :, None
            ]
            * multiplier0
        )
        excitation_factor = torch.exp(-excitation_ev / thermal_energy_ev_safe)[:, None]
        hot_metal_opacity = torch.where(
            transition_opacity > hot_metal_opacity / 100.0,
            hot_metal_opacity + transition_opacity * excitation_factor,
            hot_metal_opacity,
        )
    return hot_metal_opacity * stimulated_emission_grid / mass_density_safe[:, None]


# Frequency-grid continuum driver.


@dataclass
class FrequencyInvariants:
    """Frequency-only continuum precompute for one sampled-frequency grid."""

    frequencies_hz: np.ndarray
    coulomb_table_energy_first: bool
    natural_log_frequency: np.ndarray
    hminus_freefree_rows: np.ndarray
    hminus_boundfree_cross_section_cm2: np.ndarray
    silicon_singly_ionized_peach_frequency_rows: np.ndarray
    silicon_singly_ionized_peach_natural_log_temperature_grid: np.ndarray
    rayleigh_factor: np.ndarray
    hydrogen_high_level_photoionization_cross_sections: np.ndarray
    hydrogen_low_level_photoionization_cross_sections: np.ndarray
    hydrogen_ground_level_photoionization_cross_section: np.ndarray
    hydrogen_tail_edge: np.ndarray
    neutral_helium_low_level_photoionization_cross_sections: np.ndarray
    neutral_helium_high_level_photoionization_cross_sections: np.ndarray
    nitrogen_edge_cross_sections: np.ndarray
    oxygen_911_cross_section: np.ndarray
    calcium_edge_cross_sections: np.ndarray
    magnesium_ionized_cross_section_rows: np.ndarray
    carbon_boundfree_cross_section_rows: np.ndarray
    carbon_freefree_prefactor: np.ndarray
    carbon_freefree_threshold: np.ndarray
    magnesium_boundfree_cross_section_rows: np.ndarray
    magnesium_freefree_prefactor: np.ndarray
    magnesium_freefree_threshold: np.ndarray
    silicon_boundfree_cross_section_rows: np.ndarray
    silicon_freefree_prefactor: np.ndarray
    silicon_freefree_threshold: np.ndarray
    aluminum_boundfree_cross_section: np.ndarray
    iron_boundfree_cross_section_rows: np.ndarray
    _tensor_cache: dict[tuple[str, str, str], torch.Tensor] = field(
        default_factory=dict, init=False, repr=False
    )

    def tensor(
        self, name: str, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Return a cached device tensor view of a frequency-invariant array."""
        compute_device = torch.device(device)
        cache_key = (name, str(dtype), str(compute_device))
        cached_tensor = self._tensor_cache.get(cache_key)
        if cached_tensor is None:
            host_array = np.ascontiguousarray(
                np.asarray(getattr(self, name), dtype=np.float64)
            )
            cached_tensor = torch.as_tensor(
                host_array, dtype=dtype, device=compute_device
            )
            self._tensor_cache[cache_key] = cached_tensor
        return cached_tensor


def build_frequency_invariants(
    continuum_tables, frequencies_hz, coulomb_table_energy_first=False
):
    """Precompute frequency-only continuum lookups for a fixed sampled grid.

    The host fp64 routines match the scalar per-frequency path; the cache only
    changes when the sampled frequency grid changes.
    """
    frequency_grid_hz = np.asarray(
        [float(frequency_hz) for frequency_hz in frequencies_hz],
        dtype=np.float64,
    )
    natural_log_frequency = np.array(
        [math.log(float(frequency_hz)) for frequency_hz in frequency_grid_hz],
        dtype=np.float64,
    )
    hminus_freefree_rows = np.array(
        [
            _hminus_ff_table(continuum_tables, float(frequency_hz))
            for frequency_hz in frequency_grid_hz
        ]
    )
    hminus_boundfree_cross_section_cm2 = np.array(
        [
            _hminus_bf_scalar(continuum_tables, float(frequency_hz))
            for frequency_hz in frequency_grid_hz
        ],
        dtype=np.float64,
    )
    silicon_singly_ionized_peach_frequency_rows = np.asarray(
        [
            _silicon_singly_ionized_peach_frequency_row(
                continuum_tables.arrays, float(frequency_hz), float(log_frequency_value)
            )
            for frequency_hz, log_frequency_value in zip(
                frequency_grid_hz, natural_log_frequency
            )
        ],
        dtype=np.float64,
    )
    rayleigh_factor = np.array(
        [
            _rayleigh_polarizability_factor(
                continuum_tables.arrays, float(frequency_hz)
            )
            for frequency_hz in frequency_grid_hz
        ],
        dtype=np.float64,
    )
    wavenumber_cm = frequency_grid_hz / LIGHT_SPEED_CM_PER_S
    hydrogen_high_level_photoionization_cross_sections = np.zeros(
        (len(_HYDROGEN_HIGH_LEVEL_BOUND_FREE_TRANSITIONS), frequency_grid_hz.size),
        dtype=np.float64,
    )
    hydrogen_low_level_photoionization_cross_sections = np.zeros(
        (len(_HYDROGEN_LOW_LEVEL_BOUND_FREE_TRANSITIONS), frequency_grid_hz.size),
        dtype=np.float64,
    )
    hydrogen_ground_level_photoionization_cross_section = np.zeros(
        frequency_grid_hz.size, dtype=np.float64
    )
    for frequency_index, frequency_hz in enumerate(frequency_grid_hz):
        wavenumber = wavenumber_cm[frequency_index]
        for level_index, (
            principal_quantum_number,
            threshold_wavenumber,
            _statistical_weight,
            _excitation_cm,
        ) in enumerate(_HYDROGEN_HIGH_LEVEL_BOUND_FREE_TRANSITIONS):
            if wavenumber >= threshold_wavenumber:
                hydrogen_high_level_photoionization_cross_sections[
                    level_index, frequency_index
                ] = _karzas_latter_cross_section(
                    continuum_tables.arrays,
                    float(frequency_hz),
                    1.0,
                    principal_quantum_number,
                    principal_quantum_number,
                )
        for level_index, (
            principal_quantum_number,
            threshold_wavenumber,
            _statistical_weight,
            _excitation_cm,
        ) in enumerate(_HYDROGEN_LOW_LEVEL_BOUND_FREE_TRANSITIONS):
            if wavenumber >= threshold_wavenumber:
                hydrogen_low_level_photoionization_cross_sections[
                    level_index, frequency_index
                ] = _karzas_latter_cross_section(
                    continuum_tables.arrays,
                    float(frequency_hz),
                    1.0,
                    principal_quantum_number,
                    principal_quantum_number,
                )
        if wavenumber >= 109678.764:
            hydrogen_ground_level_photoionization_cross_section[frequency_index] = (
                _karzas_latter_cross_section(
                    continuum_tables.arrays,
                    float(frequency_hz),
                    1.0,
                    1,
                    1,
                )
            )
    hydrogen_tail_edge = np.maximum(109250.336, 109678.764 - wavenumber_cm).astype(
        np.float64
    )
    (
        neutral_helium_low_level_photoionization_cross_sections,
        neutral_helium_high_level_photoionization_cross_sections,
    ) = _neutral_helium_frequency_grids(continuum_tables, frequency_grid_hz)
    light_element_rows = _light_element_frequency_grids(
        continuum_tables, frequency_grid_hz
    )
    metal_boundfree_rows = _precompute_metal_boundfree_frequency_rows(
        continuum_tables,
        frequency_grid_hz,
    )
    return FrequencyInvariants(
        frequencies_hz=frequency_grid_hz,
        coulomb_table_energy_first=bool(coulomb_table_energy_first),
        natural_log_frequency=natural_log_frequency,
        hminus_freefree_rows=hminus_freefree_rows,
        hminus_boundfree_cross_section_cm2=hminus_boundfree_cross_section_cm2,
        silicon_singly_ionized_peach_frequency_rows=silicon_singly_ionized_peach_frequency_rows,
        silicon_singly_ionized_peach_natural_log_temperature_grid=np.asarray(
            continuum_tables.arrays[
                "silicon_singly_ionized_peach_natural_log_temperature_grid"
            ],
            dtype=np.float64,
        ),
        rayleigh_factor=rayleigh_factor,
        hydrogen_high_level_photoionization_cross_sections=hydrogen_high_level_photoionization_cross_sections,
        hydrogen_low_level_photoionization_cross_sections=hydrogen_low_level_photoionization_cross_sections,
        hydrogen_ground_level_photoionization_cross_section=hydrogen_ground_level_photoionization_cross_section,
        hydrogen_tail_edge=hydrogen_tail_edge,
        neutral_helium_low_level_photoionization_cross_sections=neutral_helium_low_level_photoionization_cross_sections,
        neutral_helium_high_level_photoionization_cross_sections=neutral_helium_high_level_photoionization_cross_sections,
        nitrogen_edge_cross_sections=light_element_rows["nitrogen_edge_cross_sections"],
        oxygen_911_cross_section=light_element_rows["oxygen_911_cross_section"],
        calcium_edge_cross_sections=light_element_rows["calcium_edge_cross_sections"],
        magnesium_ionized_cross_section_rows=light_element_rows[
            "magnesium_ionized_cross_section_rows"
        ],
        carbon_boundfree_cross_section_rows=metal_boundfree_rows[
            "carbon_cross_section_rows"
        ],
        carbon_freefree_prefactor=metal_boundfree_rows["carbon_freefree_prefactor"],
        carbon_freefree_threshold=metal_boundfree_rows["carbon_freefree_threshold"],
        magnesium_boundfree_cross_section_rows=metal_boundfree_rows[
            "magnesium_cross_section_rows"
        ],
        magnesium_freefree_prefactor=metal_boundfree_rows[
            "magnesium_freefree_prefactor"
        ],
        magnesium_freefree_threshold=metal_boundfree_rows[
            "magnesium_freefree_threshold"
        ],
        silicon_boundfree_cross_section_rows=metal_boundfree_rows[
            "silicon_cross_section_rows"
        ],
        silicon_freefree_prefactor=metal_boundfree_rows["silicon_freefree_prefactor"],
        silicon_freefree_threshold=metal_boundfree_rows["silicon_freefree_threshold"],
        aluminum_boundfree_cross_section=metal_boundfree_rows["aluminum_cross_section"],
        iron_boundfree_cross_section_rows=metal_boundfree_rows[
            "iron_cross_section_rows"
        ],
    )


def _compute_at_freqs(
    continuum_tables,
    frequencies_hz,
    pops,
    coulomb_table_energy_first=False,
    frequency_invariants=None,
):
    """Return continuum absorption and scattering on a sampled frequency grid."""
    temperature = pops["temperature"]
    n_frequencies = len(frequencies_hz)
    n_depths = temperature.shape[0]
    dtype = temperature.dtype
    device = temperature.device

    hc_over_kt = (
        PLANCK_ERG_SECOND / (BOLTZMANN_ERG_PER_K * temperature) * LIGHT_SPEED_CM_PER_S
    )
    thermal_energy_ev = temperature * REFERENCE_BOLTZMANN_EV_PER_K
    natural_log_temperature = torch.log(torch.clamp(temperature, min=1e-10))
    hydrogen_partition = _hydrogen_partition(continuum_tables, temperature)

    if frequency_invariants is not None:
        assert bool(frequency_invariants.coulomb_table_energy_first) == bool(
            coulomb_table_energy_first
        ), "FrequencyInvariants layout mismatch"
        assert frequency_invariants.frequencies_hz.shape[0] == n_frequencies, (
            "FrequencyInvariants grid size mismatch"
        )
        log_frequency_host = frequency_invariants.natural_log_frequency
    else:
        # Match the scalar `_coulomb_freefree_gaunt` path exactly: per-element `math.log` on
        # host fp64 avoids rare one-ULP bracket changes from a batched torch log.
        log_frequency_host = np.array(
            [math.log(float(frequency)) for frequency in frequencies_hz],
            dtype=np.float64,
        )

    if frequency_invariants is not None:
        log_frequency_tensor = frequency_invariants.tensor(
            "natural_log_frequency", dtype, device
        )
    else:
        log_frequency_tensor = torch.as_tensor(
            log_frequency_host, dtype=dtype, device=device
        )
    coulomb_freefree_by_charge = {
        charge: _coulomb_freefree_gaunt_grid(
            continuum_tables,
            charge,
            log_frequency_tensor,
            temperature,
            natural_log_temperature,
            energy_first_layout=coulomb_table_energy_first,
        )
        for charge in range(1, 6)
    }

    # Batched (depth, frequency) grid terms are the production path whenever the
    # window-invariant bundle is available; the per-frequency Python loop below
    # remains the fallback when no invariants are supplied (e.g. deck import).
    batch_terms = frequency_invariants is not None
    photon_boltzmann_grid = stimulated_emission_grid = None
    hminus_grid = hydrogen_grid = minor_grids = helium_grids = None
    hot_metal_grid = silicon_singly_ionized_grid = (
        light_element_and_silicon_singly_ionized_grid
    ) = None
    if batch_terms:
        if frequency_invariants is not None:
            frequency_tensor = frequency_invariants.tensor(
                "frequencies_hz", dtype, device
            )
        else:
            frequency_tensor = torch.as_tensor(
                np.asarray(frequencies_hz, np.float64), dtype=dtype, device=device
            )
        photon_boltzmann_grid = torch.exp(
            -(PLANCK_ERG_SECOND * frequency_tensor[None, :])
            / (BOLTZMANN_ERG_PER_K * temperature[:, None])
        )
        stimulated_emission_grid = 1.0 - photon_boltzmann_grid
        hminus_grid = _hminus_opacity_grid(
            continuum_tables,
            pops,
            stimulated_emission_grid,
            frequency_invariants,
        )
        hydrogen_grid = _hydrogen_opacity_grid(
            pops,
            stimulated_emission_grid,
            hc_over_kt,
            coulomb_freefree_by_charge[1].transpose(0, 1),
            frequency_invariants,
            dtype,
            device,
        )
        if coulomb_table_energy_first:
            minor_grids = _minor_terms_grid(
                pops,
                stimulated_emission_grid,
                hc_over_kt,
                hydrogen_partition,
                frequency_invariants,
                dtype,
                device,
            )
        helium_grids = _helium_opacity_grid(
            pops,
            photon_boltzmann_grid,
            stimulated_emission_grid,
            hc_over_kt,
            coulomb_freefree_by_charge[1].transpose(0, 1),
            coulomb_freefree_by_charge[2].transpose(0, 1),
            frequency_invariants,
            dtype,
            device,
        )
        hot_metal_grid = _hot_metal_opacity_grid(
            continuum_tables,
            pops,
            stimulated_emission_grid,
            thermal_energy_ev,
            coulomb_freefree_by_charge,
            frequency_invariants,
            dtype,
            device,
        )
        silicon_singly_ionized_peach_base_grid = (
            _silicon_singly_ionized_peach_base_grid(
                pops, frequency_invariants, dtype, device
            )
        )
        silicon_singly_ionized_grid = (
            silicon_singly_ionized_peach_base_grid
            * pops["silicon_singly_ionized_partition_normalized_population"][:, None]
            * stimulated_emission_grid
            / torch.clamp(pops["mass_density"], min=1e-30)[:, None]
        )
        light_element_and_silicon_singly_ionized_grid = (
            _light_element_opacity_grid(
                pops,
                stimulated_emission_grid,
                hc_over_kt,
                thermal_energy_ev,
                frequency_invariants,
                dtype,
                device,
            )
            + silicon_singly_ionized_grid
        )

    # Grid finalize sums the materialized continuum grids in scalar term order,
    # the production path whenever every batched term is present; falls through
    # to the per-frequency loop otherwise.
    grid_finalize = (
        batch_terms
        and minor_grids is not None
        and helium_grids is not None
        and hot_metal_grid is not None
        and light_element_and_silicon_singly_ionized_grid is not None
    )
    if grid_finalize:
        # All expensive continuum terms are already materialized as (depth, frequency)
        # grids.  Sum them in the same term order as the scalar column loop, but avoid
        # the Python loop over the 90k-150k sampled frequencies.
        continuum_absorption = hminus_grid
        continuum_absorption = continuum_absorption + hydrogen_grid
        continuum_absorption = continuum_absorption + minor_grids[0]
        continuum_absorption = continuum_absorption + helium_grids[0]
        continuum_absorption = continuum_absorption + helium_grids[1]
        continuum_absorption = continuum_absorption + hot_metal_grid
        continuum_absorption = (
            continuum_absorption + light_element_and_silicon_singly_ionized_grid
        )
        continuum_scattering = _scattering_opacity_grid(
            pops,
            frequency_invariants,
            hydrogen_partition,
            dtype,
            device,
        )
        continuum_scattering = continuum_scattering + minor_grids[1]
        return continuum_absorption, continuum_scattering

    continuum_absorption = torch.zeros(
        n_depths, n_frequencies, dtype=dtype, device=device
    )
    continuum_scattering = torch.zeros(
        n_depths, n_frequencies, dtype=dtype, device=device
    )
    for frequency_index, frequency_hz in enumerate(frequencies_hz):
        frequency_hz = float(frequency_hz)
        coulomb_freefree_charge1 = coulomb_freefree_by_charge[1][frequency_index]
        coulomb_freefree_charge2 = coulomb_freefree_by_charge[2][frequency_index]
        coulomb_freefree_charges = [
            coulomb_freefree_by_charge[charge][frequency_index]
            for charge in range(1, 6)
        ]
        hminus_freefree_by_theta = (
            frequency_invariants.hminus_freefree_rows[frequency_index]
            if frequency_invariants is not None
            else None
        )
        hminus_boundfree_cross_section_cm2 = (
            float(
                frequency_invariants.hminus_boundfree_cross_section_cm2[frequency_index]
            )
            if frequency_invariants is not None
            else None
        )
        silicon_singly_ionized_peach_frequency_row = (
            frequency_invariants.silicon_singly_ionized_peach_frequency_rows[
                frequency_index
            ]
            if frequency_invariants is not None
            else None
        )
        rayleigh_factor = (
            float(frequency_invariants.rayleigh_factor[frequency_index])
            if frequency_invariants is not None
            else None
        )
        if photon_boltzmann_grid is not None:
            photon_boltzmann_factor = photon_boltzmann_grid[:, frequency_index]
            stimulated_emission = stimulated_emission_grid[:, frequency_index]
        else:
            photon_boltzmann_factor = torch.exp(
                -PLANCK_ERG_SECOND * frequency_hz / (BOLTZMANN_ERG_PER_K * temperature)
            )
            stimulated_emission = 1.0 - photon_boltzmann_factor
        if hminus_grid is not None:
            absorption_column = hminus_grid[:, frequency_index]
        else:
            absorption_column = _hminus_opacity(
                continuum_tables,
                frequency_hz,
                pops,
                photon_boltzmann_factor,
                stimulated_emission,
                hminus_freefree_by_theta=hminus_freefree_by_theta,
                hminus_boundfree_cross_section_cm2=hminus_boundfree_cross_section_cm2,
            )
        if hydrogen_grid is not None:
            absorption_column = absorption_column + hydrogen_grid[:, frequency_index]
        else:
            absorption_column = absorption_column + _hydrogen_opacity(
                continuum_tables,
                frequency_hz,
                pops,
                photon_boltzmann_factor,
                stimulated_emission,
                hc_over_kt,
                coulomb_table_energy_first=coulomb_table_energy_first,
                coulomb_freefree_charge1=coulomb_freefree_charge1,
            )
        if minor_grids is not None:
            minor_absorption = minor_grids[0][:, frequency_index]
            minor_scattering = minor_grids[1][:, frequency_index]
        else:
            minor_absorption, minor_scattering = _minor_terms(
                continuum_tables,
                frequency_hz,
                pops,
                photon_boltzmann_factor,
                stimulated_emission,
                hc_over_kt,
                coulomb_table_energy_first=coulomb_table_energy_first,
                hydrogen_partition=hydrogen_partition,
            )
        absorption_column = absorption_column + minor_absorption
        if helium_grids is not None:
            helium_neutral_opacity = helium_grids[0][:, frequency_index]
            helium_ionized_opacity = helium_grids[1][:, frequency_index]
        else:
            helium_neutral_opacity, helium_ionized_opacity = _helium_opacity(
                continuum_tables,
                frequency_hz,
                pops,
                photon_boltzmann_factor,
                stimulated_emission,
                hc_over_kt,
                coulomb_table_energy_first=coulomb_table_energy_first,
                helium_neutral_coulomb_freefree=coulomb_freefree_charge1,
                helium_ionized_coulomb_freefree=coulomb_freefree_charge2,
            )
        if (
            hot_metal_grid is not None
            and light_element_and_silicon_singly_ionized_grid is not None
        ):
            hot_metal_opacity = hot_metal_grid[:, frequency_index]
            light_element_and_silicon_singly_ionized_opacity = (
                light_element_and_silicon_singly_ionized_grid[:, frequency_index]
            )
        else:
            silicon_singly_ionized_peach_base_cross_section = (
                silicon_singly_ionized_grid[:, frequency_index]
                if silicon_singly_ionized_grid is not None
                else None
            )
            hot_metal_opacity, light_element_and_silicon_singly_ionized_opacity = (
                _hot_metal_and_silicon_singly_ionized_opacity(
                    continuum_tables,
                    frequency_hz,
                    pops,
                    stimulated_emission,
                    thermal_energy_ev,
                    natural_log_temperature,
                    coulomb_table_energy_first=coulomb_table_energy_first,
                    coulomb_freefree_by_charge=coulomb_freefree_charges,
                    silicon_singly_ionized_peach_frequency_row=silicon_singly_ionized_peach_frequency_row,
                    silicon_singly_ionized_peach_base_cross_section=silicon_singly_ionized_peach_base_cross_section,
                )
            )
        absorption_column = (
            absorption_column
            + helium_neutral_opacity
            + helium_ionized_opacity
            + hot_metal_opacity
            + light_element_and_silicon_singly_ionized_opacity
        )
        hydrogen_rayleigh_scattering, electron_scattering = _scattering_opacity(
            continuum_tables,
            frequency_hz,
            pops,
            rayleigh_factor=rayleigh_factor,
            hydrogen_partition=hydrogen_partition,
        )
        continuum_absorption[:, frequency_index] = absorption_column
        continuum_scattering[:, frequency_index] = (
            hydrogen_rayleigh_scattering + electron_scattering + minor_scattering
        )
    return continuum_absorption, continuum_scattering


def build_edge_sample_frequencies(
    signed_edge_frequency_hz,
    continuum_edge_wavelength_nm,
):
    """Return the three continuum sample frequencies used in each edge interval."""
    signed_edge_frequency_hz = np.asarray(signed_edge_frequency_hz, np.float64)
    continuum_edge_wavelength_nm = np.asarray(continuum_edge_wavelength_nm, np.float64)
    n_edges = signed_edge_frequency_hz.size
    sample_frequencies_hz = np.empty(3 * (n_edges - 1))
    for edge_index in range(n_edges - 1):
        midpoint_wavelength_nm = (
            continuum_edge_wavelength_nm[edge_index]
            + continuum_edge_wavelength_nm[edge_index + 1]
        ) / 2.0
        sample_frequencies_hz[3 * edge_index] = (
            abs(signed_edge_frequency_hz[edge_index]) / 1.0000001
        )
        sample_frequencies_hz[3 * edge_index + 1] = (
            LIGHT_SPEED_NM_PER_S / midpoint_wavelength_nm
        )
        sample_frequencies_hz[3 * edge_index + 2] = (
            abs(signed_edge_frequency_hz[edge_index + 1]) * 1.0000001
        )
    return sample_frequencies_hz


def compute_sampled_continuum(
    continuum_tables: ContinuumTables,
    frequencies_hz,
    pops: dict,
    frequency_invariants: "Optional[FrequencyInvariants]" = None,
):
    """Continuum absorption, scattering, and LTE source on the sampled frequency grid."""
    frequency_grid_hz = np.asarray(frequencies_hz, np.float64)
    continuum_absorption, continuum_scattering = _compute_at_freqs(
        continuum_tables,
        frequency_grid_hz,
        pops,
        coulomb_table_energy_first=True,
        frequency_invariants=frequency_invariants,
    )
    temperature = pops["temperature"]
    frequency_tensor = torch.as_tensor(
        frequency_grid_hz,
        dtype=temperature.dtype,
        device=temperature.device,
    )
    continuum_source = _planck_nu(frequency_tensor[None, :], temperature[:, None])
    return continuum_absorption, continuum_scattering, continuum_source


def continuum(
    wavelength_grid_nm,
    atmosphere: dict,
    continuum_tables: ContinuumTables,
    pops: Optional[dict] = None,
):
    """Return continuum absorption and scattering on the synthesis wavelength grid."""
    device = continuum_tables.device
    dtype = continuum_tables.dtype
    if pops is None:
        pops = build_pops(atmosphere, device=device, dtype=dtype)
    n_depths = pops["temperature"].shape[0]

    signed_edge_frequency_hz = np.asarray(
        atmosphere["signed_continuum_edge_frequency_hz"], np.float64
    )
    edge_wavelength_nm = np.asarray(
        atmosphere["continuum_edge_wavelength_nm"], np.float64
    )
    edge_midpoint_nm = np.asarray(
        atmosphere["continuum_edge_midpoint_wavelength_nm"], np.float64
    )
    edge_interval_width_squared_over_two_nm2 = np.asarray(
        atmosphere["continuum_edge_interval_width_squared_over_two_nm2"],
        np.float64,
    )
    synthesis_wavelength_nm = np.asarray(wavelength_grid_nm, np.float64)

    sample_frequencies_hz = build_edge_sample_frequencies(
        signed_edge_frequency_hz,
        edge_wavelength_nm,
    )
    edge_indices = np.clip(
        np.searchsorted(
            edge_wavelength_nm, np.abs(synthesis_wavelength_nm), side="right"
        )
        - 1,
        0,
        edge_wavelength_nm.size - 2,
    )
    used_edge_indices = np.unique(edge_indices)

    sample_indices = np.concatenate(
        [
            [3 * edge_index, 3 * edge_index + 1, 3 * edge_index + 2]
            for edge_index in used_edge_indices
        ]
    )
    absorption_samples, scattering_samples = _compute_at_freqs(
        continuum_tables,
        sample_frequencies_hz[sample_indices],
        pops,
    )

    log_absorption_by_edge = torch.zeros(
        n_depths, edge_wavelength_nm.size - 1, 3, dtype=dtype, device=device
    )
    log_scattering_by_edge = torch.zeros(
        n_depths, edge_wavelength_nm.size - 1, 3, dtype=dtype, device=device
    )
    for sample_block_index, edge_index in enumerate(used_edge_indices):
        sample_slice = slice(3 * sample_block_index, 3 * sample_block_index + 3)
        log_absorption_by_edge[:, edge_index, :] = torch.log10(
            torch.clamp(absorption_samples[:, sample_slice], min=1e-30)
        )
        log_scattering_by_edge[:, edge_index, :] = torch.log10(
            torch.clamp(scattering_samples[:, sample_slice], min=1e-30)
        )

    absorption = torch.zeros(
        n_depths, synthesis_wavelength_nm.size, dtype=dtype, device=device
    )
    scattering = torch.zeros(
        n_depths, synthesis_wavelength_nm.size, dtype=dtype, device=device
    )
    for edge_index in used_edge_indices:
        in_edge = edge_indices == edge_index
        if not np.any(in_edge):
            continue
        wavelength_in_edge = synthesis_wavelength_nm[in_edge]
        left_wavelength_nm = edge_wavelength_nm[edge_index]
        right_wavelength_nm = edge_wavelength_nm[edge_index + 1]
        midpoint_nm = edge_midpoint_nm[edge_index]
        interval_width_squared_over_two = (
            edge_interval_width_squared_over_two_nm2[edge_index]
            if edge_interval_width_squared_over_two_nm2[edge_index] != 0.0
            else 1e-20
        )
        left_basis = (
            (wavelength_in_edge - midpoint_nm)
            * (wavelength_in_edge - right_wavelength_nm)
            / interval_width_squared_over_two
        )
        center_basis = (
            (left_wavelength_nm - wavelength_in_edge)
            * (wavelength_in_edge - right_wavelength_nm)
            * 2.0
            / interval_width_squared_over_two
        )
        right_basis = (
            (wavelength_in_edge - left_wavelength_nm)
            * (wavelength_in_edge - midpoint_nm)
            / interval_width_squared_over_two
        )
        left_basis_t = torch.as_tensor(left_basis, dtype=dtype, device=device)
        center_basis_t = torch.as_tensor(center_basis, dtype=dtype, device=device)
        right_basis_t = torch.as_tensor(right_basis, dtype=dtype, device=device)
        wavelength_indices_t = torch.as_tensor(
            np.nonzero(in_edge)[0], dtype=torch.int64, device=device
        )
        log_absorption = (
            log_absorption_by_edge[:, edge_index, 0][:, None] * left_basis_t[None, :]
            + log_absorption_by_edge[:, edge_index, 1][:, None]
            * center_basis_t[None, :]
            + log_absorption_by_edge[:, edge_index, 2][:, None] * right_basis_t[None, :]
        )
        log_scattering = (
            log_scattering_by_edge[:, edge_index, 0][:, None] * left_basis_t[None, :]
            + log_scattering_by_edge[:, edge_index, 1][:, None]
            * center_basis_t[None, :]
            + log_scattering_by_edge[:, edge_index, 2][:, None] * right_basis_t[None, :]
        )
        absorption[:, wavelength_indices_t] = 10.0**log_absorption
        scattering[:, wavelength_indices_t] = 10.0**log_scattering
    return absorption, scattering


__all__ = [
    "ContinuumTables",
    "build_pops",
    "pops_from_population_state",
    "continuum",
    "compute_sampled_continuum",
    "build_edge_sample_frequencies",
    "FrequencyInvariants",
    "build_frequency_invariants",
]
