# ruff: noqa: E402
"""Continuum-opacity boundary state and physical table loaders."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ._numba_cache import configure_numba_cache

from .data_files import atmosphere_table_path, load_table_arrays

from .atmosphere_io import ModelAtmosphere
from .microturbulence import _piecewise_quadratic_remap
from .runtime_state import AtmosphereRuntimeState

# The compiled continuum kernels (Karzas-Latter cross sections, the LUKEOP
# lukewarm-metal hot loop, and the ROSSTAB nearest-quadrant lookup) are the
# sole production path; numba is a hard requirement.
configure_numba_cache()
try:
    import numba
except ImportError as exc:  # pragma: no cover - numba is a hard requirement
    raise ImportError(
        "numba is required: the compiled continuum kernels are the sole "
        "production path for payne_zero_atmosphere."
    ) from exc

_NUMBA_AVAILABLE = True


# LINTER query arrays below this size use the serial njit kernel; prange
# thread-dispatch overhead makes small arrays slower under many threads.
_LINTER_PARALLEL_MIN_SIZE = 8192


from .constants import (
    ATOMIC_MASS_GRAM_REFERENCE,
    BOLTZMANN_ERG_PER_K_EXACT,
    BOLTZMANN_ERG_PER_K_REFERENCE,
    BOLTZMANN_EV_PER_K_REFERENCE,
    REFERENCE_NATURAL_LOG_10,
    LIGHT_SPEED_ANGSTROM_PER_S,
    LIGHT_SPEED_CM_PER_S_EXACT,
    LIGHT_SPEED_CM_PER_S_REFERENCE,
    LIGHT_SPEED_NM_PER_S,
    PLANCK_ERG_SECOND_EXACT,
    PLANCK_ERG_SECOND_REFERENCE,
    WAVENUMBER_PER_EV_REFERENCE,
)

_CONTINUUM_REFERENCE_WAVELENGTHS_343_NM = np.array(
    [
        9.09,
        9.35,
        9.61,
        9.77,
        9.96,
        10.20,
        10.38,
        10.56,
        10.77,
        11.04,
        11.40,
        11.78,
        12.13,
        12.48,
        12.71,
        12.84,
        13.05,
        13.24,
        13.39,
        13.66,
        13.98,
        14.33,
        14.72,
        15.10,
        15.52,
        15.88,
        16.20,
        16.60,
        17.03,
        17.34,
        17.68,
        18.02,
        18.17,
        18.61,
        19.10,
        19.39,
        19.84,
        20.18,
        20.50,
        21.05,
        21.62,
        21.98,
        22.30,
        22.68,
        23.00,
        23.40,
        24.00,
        24.65,
        25.24,
        25.68,
        26.00,
        26.40,
        26.85,
        27.35,
        27.85,
        28.40,
        29.0,
        29.6,
        30.1,
        30.8,
        31.8,
        32.8,
        33.8,
        34.8,
        35.7,
        36.6,
        37.5,
        38.5,
        39.5,
        40.5,
        41.4,
        42.2,
        43.0,
        44.1,
        45.1,
        46.0,
        47.0,
        48.0,
        49.0,
        50.0,
        50.6,
        51.4,
        53.0,
        55.0,
        56.7,
        58.5,
        60.5,
        62.5,
        64.5,
        66.3,
        68.0,
        70.0,
        71.6,
        73.0,
        75.0,
        77.0,
        79.0,
        81.0,
        83.0,
        85.0,
        87.0,
        89.0,
        90.6,
        92.6,
        96.0,
        100.0,
        104.0,
        108.0,
        111.5,
        114.5,
        118.0,
        122.0,
        126.0,
        130.0,
        134.0,
        138.0,
        142.0,
        146.0,
        150.0,
        154.0,
        160.0,
        165.0,
        169.0,
        173.0,
        177.5,
        182.0,
        186.0,
        190.5,
        195.0,
        200.0,
        204.5,
        208.5,
        212.5,
        217.5,
        222.5,
        227.5,
        232.5,
        237.5,
        242.5,
        248.0,
        253.0,
        257.5,
        262.5,
        267.5,
        272.5,
        277.5,
        282.5,
        287.5,
        295.0,
        305.0,
        315.0,
        325.0,
        335.0,
        345.0,
        355.0,
        362.0,
        367.0,
        375.0,
        385.0,
        395.0,
        405.0,
        415.0,
        425.0,
        435.0,
        455.0,
        465.0,
        475.0,
        485.0,
        495.0,
        505.0,
        515.0,
        525.0,
        535.0,
        545.0,
        555.0,
        565.0,
        575.0,
        585.0,
        595.0,
        605.0,
        615.0,
        625.0,
        635.0,
        645.0,
        655.0,
        665.0,
        675.0,
        685.0,
        695.0,
        705.0,
        715.0,
        725.0,
        735.0,
        745.0,
        755.0,
        765.0,
        775.0,
        785.0,
        795.0,
        805.0,
        815.0,
        825.0,
        835.0,
        845.0,
        855.0,
        865.0,
        875.0,
        885.0,
        895.0,
        905.0,
        915.0,
        925.0,
        935.0,
        945.0,
        955.0,
        965.0,
        975.0,
        985.0,
        995.0,
        1012.5,
        1037.5,
        1062.5,
        1087.5,
        1112.5,
        1137.5,
        1162.5,
        1187.5,
        1212.5,
        1237.5,
        1262.5,
        1287.5,
        1312.5,
        1337.5,
        1362.5,
        1387.5,
        1412.5,
        1442.0,
        1467.0,
        1487.5,
        1512.5,
        1537.5,
        1562.5,
        1587.5,
        1620.0,
        1660.0,
        1700.0,
        1740.0,
        1780.0,
        1820.0,
        1860.0,
        1900.0,
        1940.0,
        1980.0,
        2025.0,
        2075.0,
        2125.0,
        2175.0,
        2225.0,
        2265.0,
        2290.0,
        2325.0,
        2375.0,
        2425.0,
        2475.0,
        2525.0,
        2575.0,
        2625.0,
        2675.0,
        2725.0,
        2775.0,
        2825.0,
        2875.0,
        2925.0,
        2975.0,
        3025.0,
        3075.0,
        3125.0,
        3175.0,
        3240.0,
        3340.0,
        3450.0,
        3550.0,
        3650.0,
        3750.0,
        3850.0,
        3950.0,
        4050.0,
        4150.0,
        4250.0,
        4350.0,
        4450.0,
        4550.0,
        4650.0,
        4750.0,
        4850.0,
        4950.0,
        5050.0,
        5150.0,
        5250.0,
        5350.0,
        5450.0,
        5550.0,
        5650.0,
        5750.0,
        5850.0,
        5950.0,
        6050.0,
        6150.0,
        6250.0,
        6350.0,
        6500.0,
        6700.0,
        6900.0,
        7100.0,
        7300.0,
        7500.0,
        7700.0,
        7900.0,
        8100.0,
        8300.0,
        8500.0,
        8700.0,
        8900.0,
        9100.0,
        9300.0,
        9500.0,
        9700.0,
        9900.0,
        10000.0,
        20000.0,
        40000.0,
        60000.0,
        80000.0,
        100000.0,
        120000.0,
        140000.0,
        160000.0,
        200000.0,
        240000.0,
        280000.0,
        320000.0,
        360000.0,
        400000.0,
    ],
    dtype=np.float64,
)

_HELIUM_NEUTRAL_STATISTICAL_WEIGHTS = np.array(
    [1.0, 3.0, 1.0, 9.0, 3.0, 3.0, 1.0, 9.0, 20.0, 3.0],
    dtype=np.float64,
)
_HELIUM_NEUTRAL_EXCITATION_EV = np.array(
    [0.0, 19.819, 20.615, 20.964, 21.217, 22.718, 22.920, 23.006, 23.073, 23.086],
    dtype=np.float64,
)
_HELIUM_NEUTRAL_THRESHOLD_FREQUENCY_HZ = np.array(
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
_HELIUM_GROUND_CROSS_SECTION_50_505 = np.array(
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
_HELIUM_GROUND_CROSS_SECTION_20_50 = np.array(
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
_HELIUM_GROUND_CROSS_SECTION_10_20 = np.array(
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
_HELIUM_GROUND_CROSS_SECTION_0_10 = np.array(
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
_HELIUM_1S2S_SINGLET_LOG_FREQUENCY = np.array(
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
_HELIUM_1S2S_SINGLET_LOG_CROSS_SECTION = np.array(
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
_HELIUM_1S2S_TRIPLET_LOG_FREQUENCY = np.array(
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
_HELIUM_1S2S_TRIPLET_LOG_CROSS_SECTION = np.array(
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
_HELIUM_1S2P_SINGLET_LOG_FREQUENCY = np.array(
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
_HELIUM_1S2P_SINGLET_LOG_CROSS_SECTION = np.array(
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
_HELIUM_1S2P_TRIPLET_LOG_FREQUENCY = np.array(
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
_HELIUM_1S2P_TRIPLET_LOG_CROSS_SECTION = np.array(
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


class ContinuumOpacityTableError(RuntimeError):
    """Raised when a packaged continuum table is missing or malformed."""


@dataclass(frozen=True)
class ContinuumOpacityTables:
    """Physical tables used by continuum-opacity branches."""

    coulomb_freefree_charge_log_offset: np.ndarray
    hminus_boundfree_wavelength_nm: np.ndarray
    hminus_boundfree_cross_section_cm2: np.ndarray
    hminus_freefree_inverse_wavelength_grid: np.ndarray
    hminus_freefree_theta_grid: np.ndarray
    hminus_freefree_short_wavelength_table: np.ndarray
    hminus_freefree_long_wavelength_table: np.ndarray
    hydrogen_rayleigh_gavrila_main_table: np.ndarray
    hydrogen_rayleigh_gavrila_ab_table: np.ndarray
    hydrogen_rayleigh_gavrila_bc_table: np.ndarray
    hydrogen_rayleigh_gavrila_cd_table: np.ndarray
    hydrogen_rayleigh_gavrila_lyman_continuum_table: np.ndarray
    hydrogen_rayleigh_gavrila_lyman_frequency_ratio_grid: np.ndarray
    coulomb_freefree_gaunt_table: np.ndarray
    hot_metal_boundfree_transition_table: np.ndarray
    silicon_singly_ionized_peach_cross_section_table: np.ndarray
    silicon_singly_ionized_peach_threshold_frequencies_hz: np.ndarray
    silicon_singly_ionized_peach_natural_log_frequency_grid: np.ndarray
    silicon_singly_ionized_peach_natural_log_temperature_grid: np.ndarray
    ch_partition_table: np.ndarray
    oh_partition_table: np.ndarray
    ch_cross_section_table: np.ndarray
    oh_cross_section_table: np.ndarray
    hydrogen_molecule_h2_collision_table: np.ndarray
    hydrogen_molecule_he_collision_table: np.ndarray
    hydrogen_neutral_level_energy_cm: np.ndarray
    hydrogen_neutral_level_statistical_weight: np.ndarray


@dataclass(frozen=True)
class KarzasLatterTables:
    """Karzas-Latter hydrogenic bound-free cross-section tables."""

    karzas_latter_log10_frequency_hz: np.ndarray
    karzas_latter_total_log10_cross_section_cm2: np.ndarray
    karzas_latter_angular_log10_cross_section_cm2: np.ndarray
    karzas_latter_high_level_energy_offset_rydberg: np.ndarray


@dataclass(frozen=True)
class ContinuumLevelTables:
    """Atomic level tables shared by continuum-opacity branches."""

    hydrogen_neutral_level_energy_cm: np.ndarray
    hydrogen_neutral_level_statistical_weight: np.ndarray
    helium_neutral_level_energy_cm: np.ndarray
    helium_neutral_level_statistical_weight: np.ndarray
    helium_singly_ionized_level_energy_cm: np.ndarray
    helium_singly_ionized_level_statistical_weight: np.ndarray
    carbon_neutral_level_energy_cm: np.ndarray
    carbon_neutral_level_statistical_weight: np.ndarray
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
    potassium_neutral_level_energy_cm: np.ndarray
    potassium_neutral_level_statistical_weight: np.ndarray
    calcium_neutral_level_energy_cm: np.ndarray
    calcium_neutral_level_statistical_weight: np.ndarray
    calcium_singly_ionized_level_energy_cm: np.ndarray
    calcium_singly_ionized_level_statistical_weight: np.ndarray
    element_block_offsets: np.ndarray
    partition_interpolation_scale: np.ndarray


@dataclass(frozen=True)
class MolecularEquilibriumTables:
    """Small molecular-equilibrium tables used by continuum branches."""

    h2_partition_function: np.ndarray


@dataclass
class RosselandOpacityTable:
    """Normalized log(T), log(P), log(kappa_R) interpolation table."""

    normalized_log_temperature: np.ndarray
    normalized_log_pressure: np.ndarray
    log10_rosseland_opacity: np.ndarray
    entry_count: int
    log_temperature_origin: float
    log_pressure_origin: float
    log_temperature_span: float
    log_pressure_span: float


@dataclass(frozen=True)
class ContinuumAtmosphereState:
    """Atmosphere and population arrays consumed by continuum opacity."""

    temperature: np.ndarray
    mass_density: np.ndarray
    electron_density: np.ndarray
    gas_pressure: np.ndarray
    hydrogen_partition_normalized_ion_stage_populations: np.ndarray
    hydrogen_neutral_population: np.ndarray
    hydrogen_ionized_population: np.ndarray
    helium_neutral_population: np.ndarray
    helium_singly_ionized_population: np.ndarray
    helium_neutral_partition_normalized_population: np.ndarray
    helium_singly_ionized_partition_normalized_population: np.ndarray
    elemental_abundances_by_layer: np.ndarray
    hydrogen_departure_coefficients: np.ndarray
    microturbulence: np.ndarray
    ion_stage_populations_by_packed_slot: np.ndarray
    partition_normalized_populations_by_packed_slot: np.ndarray
    ch_population: np.ndarray
    oh_population: np.ndarray

    @property
    def layers(self) -> int:
        return int(self.temperature.size)


_CONTINUUM_TABLE_KEYS = (
    "coulomb_freefree_charge_log_offset",
    "hminus_boundfree_wavelength_nm",
    "hminus_boundfree_cross_section_cm2",
    "hminus_freefree_inverse_wavelength_grid",
    "hminus_freefree_theta_grid",
    "hminus_freefree_short_wavelength_table",
    "hminus_freefree_long_wavelength_table",
    "hydrogen_rayleigh_gavrila_main_table",
    "hydrogen_rayleigh_gavrila_ab_table",
    "hydrogen_rayleigh_gavrila_bc_table",
    "hydrogen_rayleigh_gavrila_cd_table",
    "hydrogen_rayleigh_gavrila_lyman_continuum_table",
    "hydrogen_rayleigh_gavrila_lyman_frequency_ratio_grid",
    "coulomb_freefree_gaunt_table",
    "hot_metal_boundfree_transition_table",
    "silicon_singly_ionized_peach_cross_section_table",
    "silicon_singly_ionized_peach_threshold_frequencies_hz",
    "silicon_singly_ionized_peach_natural_log_frequency_grid",
    "silicon_singly_ionized_peach_natural_log_temperature_grid",
    "ch_partition_table",
    "oh_partition_table",
    "ch_cross_section_table",
    "oh_cross_section_table",
    "hydrogen_molecule_h2_collision_table",
    "hydrogen_molecule_he_collision_table",
    "hydrogen_neutral_level_energy_cm",
    "hydrogen_neutral_level_statistical_weight",
)

_CONTINUUM_TABLE_SHAPES = {
    "coulomb_freefree_charge_log_offset": (6,),
    "hminus_boundfree_wavelength_nm": (85,),
    "hminus_boundfree_cross_section_cm2": (85,),
    "hminus_freefree_inverse_wavelength_grid": (22,),
    "hminus_freefree_theta_grid": (11,),
    "hminus_freefree_short_wavelength_table": (11, 11),
    "hminus_freefree_long_wavelength_table": (11, 11),
    "hydrogen_rayleigh_gavrila_main_table": (74,),
    "hydrogen_rayleigh_gavrila_ab_table": (27,),
    "hydrogen_rayleigh_gavrila_bc_table": (24,),
    "hydrogen_rayleigh_gavrila_cd_table": (22,),
    "hydrogen_rayleigh_gavrila_lyman_continuum_table": (64,),
    "hydrogen_rayleigh_gavrila_lyman_frequency_ratio_grid": (64,),
    "coulomb_freefree_gaunt_table": (12, 11),
    "hot_metal_boundfree_transition_table": (60, 7),
    "silicon_singly_ionized_peach_cross_section_table": (14, 6),
    "silicon_singly_ionized_peach_threshold_frequencies_hz": (7,),
    "silicon_singly_ionized_peach_natural_log_frequency_grid": (9,),
    "silicon_singly_ionized_peach_natural_log_temperature_grid": (6,),
    "ch_partition_table": (41,),
    "oh_partition_table": (41,),
    "ch_cross_section_table": (106, 15),
    "oh_cross_section_table": (130, 15),
    "hydrogen_molecule_h2_collision_table": (81, 7),
    "hydrogen_molecule_he_collision_table": (81, 7),
    "hydrogen_neutral_level_energy_cm": (6,),
    "hydrogen_neutral_level_statistical_weight": (6,),
}

_KARZAS_TABLE_KEYS = (
    "karzas_latter_log10_frequency_hz",
    "karzas_latter_total_log10_cross_section_cm2",
    "karzas_latter_angular_log10_cross_section_cm2",
    "karzas_latter_high_level_energy_offset_rydberg",
)

_KARZAS_TABLE_SHAPES = {
    "karzas_latter_log10_frequency_hz": (29, 15),
    "karzas_latter_total_log10_cross_section_cm2": (29, 15),
    "karzas_latter_angular_log10_cross_section_cm2": (6, 6, 29),
    "karzas_latter_high_level_energy_offset_rydberg": (29,),
}

_CONTINUUM_LEVEL_TABLE_KEYS = (
    "hydrogen_neutral_level_energy_cm",
    "hydrogen_neutral_level_statistical_weight",
    "helium_neutral_level_energy_cm",
    "helium_neutral_level_statistical_weight",
    "helium_singly_ionized_level_energy_cm",
    "helium_singly_ionized_level_statistical_weight",
    "carbon_neutral_level_energy_cm",
    "carbon_neutral_level_statistical_weight",
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
    "potassium_neutral_level_energy_cm",
    "potassium_neutral_level_statistical_weight",
    "calcium_neutral_level_energy_cm",
    "calcium_neutral_level_statistical_weight",
    "calcium_singly_ionized_level_energy_cm",
    "calcium_singly_ionized_level_statistical_weight",
    "element_block_offsets",
    "partition_interpolation_scale",
)

_MOLECULAR_TABLE_KEYS = ("h2_partition_function",)


def _validate_shapes(
    *,
    filename: str,
    arrays: dict[str, np.ndarray],
    expected_shapes: dict[str, tuple[int, ...]],
) -> None:
    for key, expected_shape in expected_shapes.items():
        actual_shape = arrays[key].shape
        if actual_shape != expected_shape:
            raise ContinuumOpacityTableError(
                f"{filename}:{key} has shape {actual_shape}, expected {expected_shape}"
            )


@lru_cache(maxsize=1)
def load_continuum_opacity_tables() -> ContinuumOpacityTables:
    """Return packaged physical tables used by continuum opacity."""

    filename = "continuum_opacity_tables.npz"
    arrays = load_table_arrays(
        atmosphere_table_path(filename),
        _CONTINUUM_TABLE_KEYS,
        error_type=ContinuumOpacityTableError,
    )
    _validate_shapes(
        filename=filename,
        arrays=arrays,
        expected_shapes=_CONTINUUM_TABLE_SHAPES,
    )
    return ContinuumOpacityTables(
        **{
            name: np.asarray(arrays[name], dtype=np.float64)
            for name in _CONTINUUM_TABLE_KEYS
        }
    )


@lru_cache(maxsize=1)
def load_karzas_latter_tables() -> KarzasLatterTables:
    """Return the packaged Karzas-Latter cross-section interpolation tables."""

    filename = "karzas_latter_tables.npz"
    arrays = load_table_arrays(
        atmosphere_table_path(filename),
        _KARZAS_TABLE_KEYS,
        error_type=ContinuumOpacityTableError,
    )
    _validate_shapes(
        filename=filename,
        arrays=arrays,
        expected_shapes=_KARZAS_TABLE_SHAPES,
    )
    return KarzasLatterTables(
        **{
            name: np.asarray(arrays[name], dtype=np.float64)
            for name in _KARZAS_TABLE_KEYS
        }
    )


@lru_cache(maxsize=1)
def load_continuum_level_tables() -> ContinuumLevelTables:
    """Return the atomic level-table archive used by continuum branches."""

    arrays = load_table_arrays(
        atmosphere_table_path("continuum_level_tables.npz"),
        _CONTINUUM_LEVEL_TABLE_KEYS,
        error_type=ContinuumOpacityTableError,
    )
    return ContinuumLevelTables(
        **{name: np.asarray(arrays[name]) for name in _CONTINUUM_LEVEL_TABLE_KEYS}
    )


@lru_cache(maxsize=1)
def load_molecular_equilibrium_tables() -> MolecularEquilibriumTables:
    """Return the H2 partition table used by molecular continuum terms."""

    arrays = load_table_arrays(
        atmosphere_table_path("molecular_equilibrium_tables.npz"),
        _MOLECULAR_TABLE_KEYS,
        error_type=ContinuumOpacityTableError,
    )
    if arrays["h2_partition_function"].shape != (200,):
        raise ContinuumOpacityTableError(
            "molecular_equilibrium_tables.npz:h2_partition_function has shape "
            f"{arrays['h2_partition_function'].shape}, expected (200,)"
        )
    return MolecularEquilibriumTables(
        **{
            name: np.asarray(arrays[name], dtype=np.float64)
            for name in _MOLECULAR_TABLE_KEYS
        }
    )


def build_continuum_atmosphere_state(
    atmosphere: ModelAtmosphere,
    state: AtmosphereRuntimeState,
) -> ContinuumAtmosphereState:
    """Build the continuum-opacity state from the prepared runtime."""

    layer_count = atmosphere.layers
    ion_stage_populations_by_packed_slot = np.asarray(
        state.ion_stage_populations_by_packed_slot, dtype=np.float64
    )
    partition_normalized_populations_by_packed_slot = np.asarray(
        state.partition_normalized_populations_by_packed_slot,
        dtype=np.float64,
    )

    if ion_stage_populations_by_packed_slot.shape[0] != layer_count:
        raise ValueError(
            "ion_stage_populations_by_packed_slot must match atmosphere layer count"
        )
    if partition_normalized_populations_by_packed_slot.shape[0] != layer_count:
        raise ValueError(
            "partition_normalized_populations_by_packed_slot must match atmosphere layer count"
        )
    if ion_stage_populations_by_packed_slot.shape[1] < 2:
        raise ValueError(
            "ion_stage_populations_by_packed_slot must include hydrogen neutral and ionized slots"
        )
    if partition_normalized_populations_by_packed_slot.shape[1] <= 847:
        raise ValueError(
            "partition_normalized_populations_by_packed_slot must include CH and OH continuum slots"
        )

    hydrogen_departure_coefficients = state.hydrogen_departure_coefficients
    if hydrogen_departure_coefficients is None:
        hydrogen_departure_coefficients = np.ones((layer_count, 6), dtype=np.float64)
    hydrogen_departure_coefficients = np.asarray(
        hydrogen_departure_coefficients,
        dtype=np.float64,
    )
    if hydrogen_departure_coefficients.shape[0] != layer_count:
        raise ValueError(
            "hydrogen_departure_coefficients must match atmosphere layer count"
        )

    hydrogen_partition_normalized_ion_stage_populations = np.column_stack(
        [
            partition_normalized_populations_by_packed_slot[:, 0],
            partition_normalized_populations_by_packed_slot[:, 1],
        ]
    )

    return ContinuumAtmosphereState(
        temperature=np.asarray(atmosphere.temperature, dtype=np.float64),
        mass_density=np.asarray(state.mass_density, dtype=np.float64),
        electron_density=np.asarray(state.electron_density, dtype=np.float64),
        gas_pressure=np.asarray(state.gas_pressure, dtype=np.float64),
        hydrogen_partition_normalized_ion_stage_populations=np.asarray(
            hydrogen_partition_normalized_ion_stage_populations, dtype=np.float64
        ),
        hydrogen_neutral_population=np.asarray(
            ion_stage_populations_by_packed_slot[:, 0], dtype=np.float64
        ),
        hydrogen_ionized_population=np.asarray(
            ion_stage_populations_by_packed_slot[:, 1], dtype=np.float64
        ),
        helium_neutral_population=np.asarray(
            ion_stage_populations_by_packed_slot[:, 2],
            dtype=np.float64,
        ),
        helium_singly_ionized_population=np.asarray(
            ion_stage_populations_by_packed_slot[:, 3],
            dtype=np.float64,
        ),
        helium_neutral_partition_normalized_population=np.asarray(
            partition_normalized_populations_by_packed_slot[:, 2],
            dtype=np.float64,
        ),
        helium_singly_ionized_partition_normalized_population=np.asarray(
            partition_normalized_populations_by_packed_slot[:, 3],
            dtype=np.float64,
        ),
        elemental_abundances_by_layer=np.asarray(
            state.elemental_abundances_by_layer,
            dtype=np.float64,
        ),
        hydrogen_departure_coefficients=hydrogen_departure_coefficients,
        microturbulence=np.asarray(atmosphere.microturbulence, dtype=np.float64),
        ion_stage_populations_by_packed_slot=ion_stage_populations_by_packed_slot,
        partition_normalized_populations_by_packed_slot=partition_normalized_populations_by_packed_slot,
        ch_population=np.asarray(
            partition_normalized_populations_by_packed_slot[:, 845], dtype=np.float64
        ),
        oh_population=np.asarray(
            partition_normalized_populations_by_packed_slot[:, 847], dtype=np.float64
        ),
    )


@lru_cache(maxsize=1)
def build_continuum_reference_wavelength_grid() -> tuple[np.ndarray, np.ndarray]:
    """Return continuum reference wavelengths and packed selection thresholds."""

    wavelength_nm = np.empty(344, dtype=np.float64)
    wavelength_nm[:343] = _CONTINUUM_REFERENCE_WAVELENGTHS_343_NM
    wavelength_nm[343] = wavelength_nm[342]

    packed_wavelength_index = np.empty(344, dtype=np.int64)
    logarithmic_grid_step = np.log(1.0 + 1.0 / 2_000_000.0)
    packed_wavelength_index[:343] = (
        np.log(wavelength_nm[:343]) / logarithmic_grid_step + 0.5
    ).astype(np.int64)
    packed_wavelength_index[343] = 2**30
    return wavelength_nm, packed_wavelength_index


def active_continuum_reference_frequencies(
    effective_temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return active KAPCONT reference columns and their frequencies."""

    opacity_wavelength_grid_nm, _ = build_opacity_sampling_grid(
        float(effective_temperature)
    )
    continuum_wavelength_nm, _ = build_continuum_reference_wavelength_grid()
    active_indices = np.nonzero(
        continuum_wavelength_nm[:343] > opacity_wavelength_grid_nm[0]
    )[0]
    frequencies_hz = LIGHT_SPEED_NM_PER_S / np.maximum(
        continuum_wavelength_nm[active_indices],
        1.0e-300,
    )
    return active_indices, np.asarray(frequencies_hz, dtype=np.float64)


def assemble_continuum_line_selection_threshold(
    *,
    effective_temperature: float,
    temperature_k: np.ndarray,
    active_continuum_absorption: np.ndarray,
    active_continuum_scattering: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the continuum threshold used to select and accumulate spectral lines.

    The table is ``1e-3 * (absorption + scattering)`` divided by the
    stimulated-emission factor; it is not a raw continuum-opacity table.
    """

    temperature = np.asarray(temperature_k, dtype=np.float64)
    if temperature.ndim != 1:
        raise ValueError("temperature_k must be one-dimensional")

    active_absorption = np.asarray(active_continuum_absorption, dtype=np.float64)
    active_scattering = np.asarray(active_continuum_scattering, dtype=np.float64)
    active_indices, active_frequency_hz = active_continuum_reference_frequencies(
        effective_temperature,
    )

    expected_shape = (temperature.size, active_indices.size)
    if active_absorption.shape != expected_shape:
        raise ValueError(
            f"active_continuum_absorption has shape {active_absorption.shape}, "
            f"expected {expected_shape}"
        )
    if active_scattering.shape != expected_shape:
        raise ValueError(
            f"active_continuum_scattering has shape {active_scattering.shape}, "
            f"expected {expected_shape}"
        )

    reference_wavelength_nm, packed_wavelength_index = (
        build_continuum_reference_wavelength_grid()
    )
    tabulated_continuum = np.zeros((temperature.size, 344), dtype=np.float32)
    h_over_kt = PLANCK_ERG_SECOND_REFERENCE / np.maximum(
        BOLTZMANN_ERG_PER_K_REFERENCE * temperature,
        1.0e-300,
    )

    if active_indices.size:
        stimulated_emission = np.maximum(
            1.0 - np.exp(-np.outer(h_over_kt, active_frequency_hz)),
            1.0e-300,
        )
        tabulated_continuum[:, active_indices] = np.asarray(
            (active_absorption + active_scattering) * 1.0e-3 / stimulated_emission,
            dtype=np.float32,
        )

    inactive_indices = np.nonzero(
        reference_wavelength_nm[:343]
        <= build_opacity_sampling_grid(float(effective_temperature))[0][0]
    )[0]
    if inactive_indices.size:
        inactive_frequency_hz = LIGHT_SPEED_NM_PER_S / np.maximum(
            reference_wavelength_nm[inactive_indices],
            1.0e-300,
        )
        inactive_stimulated_emission = np.maximum(
            1.0 - np.exp(-np.outer(h_over_kt, inactive_frequency_hz)),
            1.0e-300,
        )
        tabulated_continuum[:, inactive_indices] = np.asarray(
            1.0e10 * 1.0e-3 / inactive_stimulated_emission,
            dtype=np.float32,
        )

    tabulated_continuum[:, 343] = tabulated_continuum[:, 342]
    return tabulated_continuum, reference_wavelength_nm, packed_wavelength_index


def _h2_equilibrium_constant(
    temperature_k: np.ndarray,
    *,
    tables: MolecularEquilibriumTables | None = None,
) -> np.ndarray:
    """Return the H2 equilibrium constant for each layer."""

    table = load_molecular_equilibrium_tables() if tables is None else tables
    temperature = np.asarray(temperature_k, dtype=np.float64)
    safe_temperature = np.where(
        np.isfinite(temperature) & (temperature > 100.0),
        temperature,
        100.0,
    )
    safe_temperature = np.minimum(safe_temperature, 19900.0)
    table_index = np.floor(safe_temperature / 100.0).astype(np.int64)
    table_index = np.minimum(199, np.maximum(1, table_index))
    partition = (
        table.h2_partition_function[table_index - 1]
        + (
            table.h2_partition_function[table_index]
            - table.h2_partition_function[table_index - 1]
        )
        * (safe_temperature - table_index * 100.0)
        / 100.0
    )
    denominator = (
        2.0
        * 3.14159
        * 1.008
        * ATOMIC_MASS_GRAM_REFERENCE
        * BOLTZMANN_ERG_PER_K_REFERENCE
        / (PLANCK_ERG_SECOND_REFERENCE**2)
        * safe_temperature
    ) ** 1.5
    equilibrium = (
        partition
        * (2.0**1.5)
        / 4.0
        / np.maximum(denominator, 1.0e-300)
        * np.exp(
            36118.11
            * PLANCK_ERG_SECOND_REFERENCE
            * LIGHT_SPEED_CM_PER_S_REFERENCE
            / BOLTZMANN_ERG_PER_K_REFERENCE
            / safe_temperature
        )
    )
    return np.where(np.isfinite(equilibrium), equilibrium, 0.0)


def compute_molecular_hydrogen_population(
    *,
    temperature_k: np.ndarray,
    hydrogen_neutral_partition_normalized_population: np.ndarray,
    hydrogen_departure_coefficient: np.ndarray | None = None,
    tables: MolecularEquilibriumTables | None = None,
) -> np.ndarray:
    """Return the H2 population used by Rayleigh and collision opacity."""

    temperature = np.asarray(temperature_k, dtype=np.float64)
    ground_population = np.asarray(
        hydrogen_neutral_partition_normalized_population, dtype=np.float64
    )
    if hydrogen_departure_coefficient is None:
        departure = np.ones_like(temperature, dtype=np.float64)
    else:
        departure = np.asarray(hydrogen_departure_coefficient, dtype=np.float64)
    return (ground_population * 2.0 * departure) ** 2 * _h2_equilibrium_constant(
        temperature,
        tables=tables,
    )


def _hydrogen_neutral_partition_normalized_population_from_neutral(
    *,
    temperature_k: np.ndarray,
    hydrogen_neutral_population: np.ndarray,
) -> np.ndarray:
    """Return the recomputed hydrogen ground-state population."""

    continuum_tables = load_continuum_opacity_tables()
    temperature = np.asarray(temperature_k, dtype=np.float64)
    thermal_energy_ev = BOLTZMANN_EV_PER_K_REFERENCE * temperature
    partition = np.zeros_like(temperature, dtype=np.float64)
    hydrogen_neutral_level_energy_ev = (
        continuum_tables.hydrogen_neutral_level_energy_cm / WAVENUMBER_PER_EV_REFERENCE
    )
    for statistical_weight, energy_ev in zip(
        continuum_tables.hydrogen_neutral_level_statistical_weight,
        hydrogen_neutral_level_energy_ev,
    ):
        with np.errstate(over="ignore", invalid="ignore"):
            boltzmann_factor = np.exp(-energy_ev / thermal_energy_ev)
        partition += statistical_weight * np.where(
            np.isfinite(boltzmann_factor),
            boltzmann_factor,
            0.0,
        )
    return np.asarray(hydrogen_neutral_population, dtype=np.float64) / np.maximum(
        partition,
        1.0e-300,
    )


def _linear_interpolate_with_extrapolation(
    x_table: np.ndarray,
    y_table: np.ndarray,
    x_new: np.ndarray,
) -> np.ndarray:
    """Return linear interpolation with endpoint extrapolation."""

    x_old = np.asarray(x_table, dtype=np.float64)
    y_old = np.asarray(y_table, dtype=np.float64)
    x_values = np.asarray(x_new, dtype=np.float64)
    if x_old.ndim != 1 or y_old.ndim != 1 or x_values.ndim != 1:
        raise ValueError("LINTER inputs must be one-dimensional")
    if x_old.size != y_old.size:
        raise ValueError("LINTER x/y tables must have the same length")
    if x_old.size < 2:
        raise ValueError("LINTER requires at least two table points")

    if _NUMBA_AVAILABLE and x_values.size > 0:
        xo = np.ascontiguousarray(x_old)
        yo = np.ascontiguousarray(y_old)
        xn = np.ascontiguousarray(x_values)
        # prange only pays off on large query arrays; below the threshold the
        # serial njit kernel avoids thread-dispatch overhead (measured to slow
        # small arrays otherwise) while still removing the pure-Python overhead.
        if x_values.size >= _LINTER_PARALLEL_MIN_SIZE:
            return _linear_interpolate_kernel_parallel(xo, yo, xn)
        return _linear_interpolate_kernel_serial(xo, yo, xn)

    result = np.zeros(x_values.size, dtype=np.float64)
    table_index = 1
    for output_index, value in enumerate(x_values):
        while table_index < x_old.size - 1 and value >= x_old[table_index]:
            table_index += 1
        denominator = x_old[table_index] - x_old[table_index - 1]
        if abs(denominator) < 1.0e-40:
            result[output_index] = y_old[table_index - 1]
        else:
            weight = (value - x_old[table_index - 1]) / denominator
            result[output_index] = (
                y_old[table_index - 1]
                + (y_old[table_index] - y_old[table_index - 1]) * weight
            )
    return result


def _planck_frequency_exact(
    *,
    temperature_k: np.ndarray,
    frequency_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Planck B_nu, exp(-h nu/kT), and stimulated-emission factor."""

    temperature = np.ascontiguousarray(temperature_k, dtype=np.float64)
    frequency = np.ascontiguousarray(frequency_hz, dtype=np.float64)
    if _NUMBA_AVAILABLE:
        # Compiled prange kernel over frequencies (scales with NUMBA_NUM_THREADS);
        # element-for-element the numpy math below, modulo ~ulp libm divergence.
        return _planck_frequency_exact_kernel(temperature, frequency)
    hnu_over_kt = (
        PLANCK_ERG_SECOND_EXACT
        * frequency[None, :]
        / np.maximum(BOLTZMANN_ERG_PER_K_EXACT * temperature[:, None], 1.0e-300)
    )
    exponential = np.exp(-hnu_over_kt)
    stimulated_emission = 1.0 - exponential

    planck = np.zeros((temperature.size, frequency.size), dtype=np.float64)
    rayleigh_jeans = hnu_over_kt < 1.0e-6
    if np.any(rayleigh_jeans):
        planck[rayleigh_jeans] = (
            2.0
            * BOLTZMANN_ERG_PER_K_EXACT
            * temperature[:, None]
            * (frequency[None, :] ** 2)
            / LIGHT_SPEED_CM_PER_S_EXACT**2
        )[rayleigh_jeans]
    if np.any(~rayleigh_jeans):
        full_planck = (
            2.0
            * PLANCK_ERG_SECOND_EXACT
            / LIGHT_SPEED_CM_PER_S_EXACT**2
            * (frequency[None, :] ** 3)
            / np.expm1(hnu_over_kt)
        )
        planck[~rayleigh_jeans] = full_planck[~rayleigh_jeans]
    planck[~np.isfinite(planck)] = 0.0
    return planck, exponential, stimulated_emission


# Level lists used by the compiled LUKEOP kernel; these mirror the literal
# (principal, orbital) tuples in compute_lukewarm_metal_opacity_columns.
_LUKEWARM_MAGNESIUM_PRINCIPAL = np.array(
    [7, 6, 5, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3], dtype=np.int64
)
_LUKEWARM_MAGNESIUM_ANGULAR = np.array(
    [7, 6, 4, 3, 2, 1, 0, 3, 2, 1, 0, 2, 1], dtype=np.int64
)
_LUKEWARM_CARBON_PRINCIPAL = np.array(
    [5, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3], dtype=np.int64
)
_LUKEWARM_CARBON_ANGULAR = np.array(
    [4, 3, 2, 1, 0, 3, 2, 1, 0, 2, 1, 0], dtype=np.int64
)


_helium_low_level_grid_kernel = None

if _NUMBA_AVAILABLE:
    _njit = numba.njit(cache=True, nogil=True)
    _njit_inline = numba.njit(cache=True, nogil=True, inline="always")

    @numba.njit(parallel=True, nogil=True, cache=True)
    def _planck_frequency_exact_kernel(temperature, frequency):
        # Parallel over frequencies: each frequency writes its own column across
        # all layers (disjoint scatter, no reduction). Reproduces the numpy
        # _planck_frequency_exact math element-for-element; the ~ulp libm
        # divergence (math.exp/expm1 vs np.exp/np.expm1) is immaterial under the
        # spectrum gate. NUMBA_NUM_THREADS governs the thread count.
        n_temp = temperature.shape[0]
        n_freq = frequency.shape[0]
        planck = np.zeros((n_temp, n_freq), dtype=np.float64)
        exponential = np.empty((n_temp, n_freq), dtype=np.float64)
        stimulated = np.empty((n_temp, n_freq), dtype=np.float64)
        planck_prefactor = (
            2.0
            * PLANCK_ERG_SECOND_EXACT
            / (LIGHT_SPEED_CM_PER_S_EXACT * LIGHT_SPEED_CM_PER_S_EXACT)
        )
        rayleigh_prefactor = (
            2.0
            * BOLTZMANN_ERG_PER_K_EXACT
            / (LIGHT_SPEED_CM_PER_S_EXACT * LIGHT_SPEED_CM_PER_S_EXACT)
        )
        for frequency_index in numba.prange(n_freq):
            nu = frequency[frequency_index]
            nu2 = nu * nu
            nu3 = nu2 * nu
            for layer_index in range(n_temp):
                temp = temperature[layer_index]
                denominator = BOLTZMANN_ERG_PER_K_EXACT * temp
                if denominator < 1.0e-300:
                    denominator = 1.0e-300
                hnu_over_kt = PLANCK_ERG_SECOND_EXACT * nu / denominator
                exp_value = math.exp(-hnu_over_kt)
                exponential[layer_index, frequency_index] = exp_value
                stimulated[layer_index, frequency_index] = 1.0 - exp_value
                if hnu_over_kt < 1.0e-6:
                    value = rayleigh_prefactor * temp * nu2
                else:
                    value = planck_prefactor * nu3 / math.expm1(hnu_over_kt)
                if not np.isfinite(value):
                    value = 0.0
                planck[layer_index, frequency_index] = value
        return planck, exponential, stimulated

    @numba.njit(parallel=True, nogil=True, cache=True)
    def _coulomb_freefree_gaunt_kernel(
        frequency_log,
        temperature_log,
        z4log,
        gaunt_table,
    ):
        # Parallel over frequencies: each frequency writes its own column of the
        # COULFF grid across layers (disjoint scatter). Reproduces the numpy
        # _coulomb_freefree_gaunt_grid bilinear interpolation element-for-element.
        n_temp = temperature_log.shape[0]
        n_freq = frequency_log.shape[0]
        out = np.empty((n_temp, n_freq), dtype=np.float64)
        for frequency_index in numba.prange(n_freq):
            flog = frequency_log[frequency_index]
            for layer_index in range(n_temp):
                tlog = temperature_log[layer_index]
                gamlog = 10.39638 - tlog / 1.15129 + z4log
                hvktlg = (flog - tlog) / 1.15129 - 20.63764
                igam = int(gamlog + 7.0)
                if igam < 1:
                    igam = 1
                elif igam > 10:
                    igam = 10
                ihvkt = int(hvktlg + 9.0)
                if ihvkt < 1:
                    ihvkt = 1
                elif ihvkt > 11:
                    ihvkt = 11
                p_weight = gamlog - (igam - 7.0)
                q_weight = hvktlg - (ihvkt - 9.0)
                ig = igam - 1
                ih = ihvkt - 1
                a00 = gaunt_table[ih, ig]
                a01 = gaunt_table[ih + 1, ig]
                a10 = gaunt_table[ih, ig + 1]
                a11 = gaunt_table[ih + 1, ig + 1]
                out[layer_index, frequency_index] = (1.0 - p_weight) * (
                    (1.0 - q_weight) * a00 + q_weight * a01
                ) + p_weight * ((1.0 - q_weight) * a10 + q_weight * a11)
        return out

    @_njit_inline
    def _linter_point(value, x_old, y_old, last):
        table_index = 1
        while table_index < last and value >= x_old[table_index]:
            table_index += 1
        denominator = x_old[table_index] - x_old[table_index - 1]
        if abs(denominator) < 1.0e-40:
            return y_old[table_index - 1]
        weight = (value - x_old[table_index - 1]) / denominator
        return (
            y_old[table_index - 1]
            + (y_old[table_index] - y_old[table_index - 1]) * weight
        )

    @_njit
    def _linear_interpolate_kernel_serial(x_old, y_old, x_values):
        # Serial njit LINTER: removes the pure-Python per-point overhead. Used
        # for small query arrays where prange thread overhead would dominate.
        n = x_values.shape[0]
        last = x_old.shape[0] - 1
        result = np.empty(n, dtype=np.float64)
        for output_index in range(n):
            result[output_index] = _linter_point(
                x_values[output_index], x_old, y_old, last
            )
        return result

    @numba.njit(parallel=True, nogil=True, cache=True)
    def _linear_interpolate_kernel_parallel(x_old, y_old, x_values):
        # ATLAS LINTER linear interpolation/extrapolation, parallel over the
        # (sorted) query points. Each output element independently locates its
        # bracket [table_index-1, table_index] the serial walk would settle on
        # -- for monotonic x_values this yields the identical index, so results
        # match element-for-element. Disjoint output slots (pure scatter).
        n = x_values.shape[0]
        last = x_old.shape[0] - 1
        result = np.empty(n, dtype=np.float64)
        for output_index in numba.prange(n):
            result[output_index] = _linter_point(
                x_values[output_index], x_old, y_old, last
            )
        return result

    @numba.njit(parallel=True, nogil=True, cache=True)
    def _iron_neutral_branch_kernel(
        wavenumber,
        hc_over_kt,
        transition_weight,
        transition_energy_cm,
        transition_threshold_cm,
    ):
        # FE1OP branch accumulation, parallel over frequencies. Each frequency
        # sums the 48 Fe I branch contributions across layers into its own column
        # (disjoint scatter). Reproduces the per-transition numpy accumulation:
        # cross_section(freq) * weight * exp(-energy * hc_over_kt(layer)).
        n_layer = hc_over_kt.shape[0]
        n_freq = wavenumber.shape[0]
        n_trans = transition_weight.shape[0]
        branch = np.zeros((n_layer, n_freq), dtype=np.float64)
        # boltzmann[trans, layer] = weight * exp(-energy * hc_over_kt[layer])
        boltzmann = np.empty((n_trans, n_layer), dtype=np.float64)
        for t in range(n_trans):
            w = transition_weight[t]
            e = transition_energy_cm[t]
            for layer_index in range(n_layer):
                boltzmann[t, layer_index] = w * math.exp(-e * hc_over_kt[layer_index])
        for frequency_index in numba.prange(n_freq):
            wn = wavenumber[frequency_index]
            for t in range(n_trans):
                threshold_cm = transition_threshold_cm[t]
                if wn < threshold_cm:
                    continue
                ratio = (threshold_cm + 3000.0 - wn) / threshold_cm / 0.1
                cross_section = 3.0e-18 / (1.0 + ratio * ratio * ratio * ratio)
                for layer_index in range(n_layer):
                    branch[layer_index, frequency_index] += (
                        boltzmann[t, layer_index] * cross_section
                    )
        return branch

    @_njit_inline
    def _helium_ground_cross_section_compiled(frequency_hz):
        # ATLAS He I ground-state photoionization cross section. Byte-for-byte
        # the scalar Python _helium_ground_cross_section; the module-level
        # ATLAS ground-state tables are frozen as numba constants.
        if frequency_hz < 5.945209e15:
            return 0.0
        wavelength_angstrom = LIGHT_SPEED_ANGSTROM_PER_S / frequency_hz
        if wavelength_angstrom > 50.0:
            table_index = int(93.0 - (wavelength_angstrom - 50.0) / 5.0)
            if table_index > 92:
                table_index = 92
            if table_index < 2:
                table_index = 2
            return (
                (wavelength_angstrom - (92 - table_index) * 5.0 - 50.0)
                / 5.0
                * (
                    _HELIUM_GROUND_CROSS_SECTION_50_505[table_index - 2]
                    - _HELIUM_GROUND_CROSS_SECTION_50_505[table_index - 1]
                )
                + _HELIUM_GROUND_CROSS_SECTION_50_505[table_index - 1]
            ) * 1.0e-18
        if wavelength_angstrom > 20.0:
            table_index = int(17.0 - (wavelength_angstrom - 20.0) / 2.0)
            if table_index > 16:
                table_index = 16
            if table_index < 2:
                table_index = 2
            return (
                (wavelength_angstrom - (16 - table_index) * 2.0 - 20.0)
                / 2.0
                * (
                    _HELIUM_GROUND_CROSS_SECTION_20_50[table_index - 2]
                    - _HELIUM_GROUND_CROSS_SECTION_20_50[table_index - 1]
                )
                + _HELIUM_GROUND_CROSS_SECTION_20_50[table_index - 1]
            ) * 1.0e-18
        if wavelength_angstrom > 10.0:
            table_index = int(12.0 - (wavelength_angstrom - 10.0) / 1.0)
            if table_index > 11:
                table_index = 11
            if table_index < 2:
                table_index = 2
            return (
                (wavelength_angstrom - (11 - table_index) - 10.0)
                * (
                    _HELIUM_GROUND_CROSS_SECTION_10_20[table_index - 2]
                    - _HELIUM_GROUND_CROSS_SECTION_10_20[table_index - 1]
                )
                + _HELIUM_GROUND_CROSS_SECTION_10_20[table_index - 1]
            ) * 1.0e-18
        table_index = int(22.0 - wavelength_angstrom / 0.5)
        if table_index > 21:
            table_index = 21
        if table_index < 2:
            table_index = 2
        return (
            (wavelength_angstrom - (21 - table_index) * 0.5)
            / 0.5
            * (
                _HELIUM_GROUND_CROSS_SECTION_0_10[table_index - 2]
                - _HELIUM_GROUND_CROSS_SECTION_0_10[table_index - 1]
            )
            + _HELIUM_GROUND_CROSS_SECTION_0_10[table_index - 1]
        ) * 1.0e-18

    @_njit_inline
    def _helium_tabulated_cross_section_compiled(
        frequency_hz,
        threshold_wavenumber_cm,
        log_frequency_table,
        log10_cross_section_cm2_table,
        high_frequency_kind,  # 0=none, 1=1s2s_singlet, 2=1s2s_triplet, 3=1s2p_singlet
    ):
        # Byte-for-byte the scalar Python _helium_tabulated_cross_section; the
        # string kind is passed as an int code so the branch is numba-typed.
        if frequency_hz < threshold_wavenumber_cm * LIGHT_SPEED_CM_PER_S_EXACT:
            return 0.0
        if (
            high_frequency_kind != 0
            and frequency_hz > 2.4 * 109722.267 * LIGHT_SPEED_CM_PER_S_EXACT
        ):
            wavenumber = frequency_hz / LIGHT_SPEED_CM_PER_S_EXACT
            if high_frequency_kind == 1:
                kinetic_energy = (wavenumber - 32033.214) / 109722.267
                epsilon = 2.0 * (kinetic_energy - 2.612316) / 0.00322
                return (
                    0.008175
                    * (484940.0 / wavenumber) ** 2.71
                    * 8.067e-18
                    * (epsilon + 76.21) ** 2
                    / (1.0 + epsilon**2)
                )
            if high_frequency_kind == 2:
                kinetic_energy = (wavenumber - 38454.691) / 109722.267
                epsilon = 2.0 * (kinetic_energy - 2.47898) / 0.000780
                return (
                    0.01521
                    * (470310.0 / wavenumber) ** 3.12
                    * 8.067e-18
                    * (epsilon - 122.4) ** 2
                    / (1.0 + epsilon**2)
                )
            if high_frequency_kind == 3:
                kinetic_energy = (wavenumber - 27175.76) / 109722.267
                epsilon_s = 2.0 * (kinetic_energy - 2.446534) / 0.01037
                epsilon_d = 2.0 * (kinetic_energy - 2.59427) / 0.00538
                return (
                    0.0009487
                    * (466750.0 / wavenumber) ** 3.69
                    * 8.067e-18
                    * (
                        (epsilon_s - 29.30) ** 2 / (1.0 + epsilon_s**2)
                        + (epsilon_d + 172.4) ** 2 / (1.0 + epsilon_d**2)
                    )
                )

        log10_frequency = np.log10(frequency_hz)
        table_index = 15
        for candidate_index in range(1, 16):
            if log10_frequency > log_frequency_table[candidate_index]:
                table_index = candidate_index
                break
        log10_cross_section_cm2 = (
            log10_frequency - log_frequency_table[table_index]
        ) / (
            log_frequency_table[table_index - 1] - log_frequency_table[table_index]
        ) * (
            log10_cross_section_cm2_table[table_index - 1]
            - log10_cross_section_cm2_table[table_index]
        ) + log10_cross_section_cm2_table[table_index]
        return 10.0**log10_cross_section_cm2

    @numba.njit(parallel=True, nogil=True, cache=True)
    def _helium_low_level_grid_kernel(
        frequency,
        threshold_frequency_hz,
        s2s_triplet_log_frequency,
        s2s_triplet_log10_cross_section_cm2,
        s2s_singlet_log_frequency,
        s2s_singlet_log10_cross_section_cm2,
        s2p_triplet_log_frequency,
        s2p_triplet_log10_cross_section_cm2,
        s2p_singlet_log_frequency,
        s2p_singlet_log10_cross_section_cm2,
    ):
        # Fill low_level_cross_sections[0..4, :] for the 5 He I low levels.
        # Each frequency writes its own column (disjoint scatter); parallel over
        # independent frequencies. The ~1e-32 log10/pow libm divergence is
        # immaterial under the spectrum gate. NUMBA_NUM_THREADS governs threads.
        count = frequency.shape[0]
        low = np.zeros((10, count), dtype=np.float64)
        for frequency_index in numba.prange(count):
            current_frequency = frequency[frequency_index]
            if current_frequency >= threshold_frequency_hz[0]:
                low[0, frequency_index] = _helium_ground_cross_section_compiled(
                    current_frequency
                )
            if current_frequency >= threshold_frequency_hz[1]:
                low[1, frequency_index] = _helium_tabulated_cross_section_compiled(
                    current_frequency,
                    38454.691,
                    s2s_triplet_log_frequency,
                    s2s_triplet_log10_cross_section_cm2,
                    2,
                )
            if current_frequency >= threshold_frequency_hz[2]:
                low[2, frequency_index] = _helium_tabulated_cross_section_compiled(
                    current_frequency,
                    32033.214,
                    s2s_singlet_log_frequency,
                    s2s_singlet_log10_cross_section_cm2,
                    1,
                )
            if current_frequency >= threshold_frequency_hz[3]:
                low[3, frequency_index] = _helium_tabulated_cross_section_compiled(
                    current_frequency,
                    29223.753,
                    s2p_triplet_log_frequency,
                    s2p_triplet_log10_cross_section_cm2,
                    0,
                )
            if current_frequency >= threshold_frequency_hz[4]:
                low[4, frequency_index] = _helium_tabulated_cross_section_compiled(
                    current_frequency,
                    27175.76,
                    s2p_singlet_log_frequency,
                    s2p_singlet_log10_cross_section_cm2,
                    3,
                )
        return low

    @_njit_inline
    def _karzas_latter_point_compiled(
        current_frequency,
        charge_squared,
        shell,
        angular_momentum,
        karzas_latter_log10_frequency_hz,
        karzas_latter_total_log10_cross_section_cm2,
        karzas_latter_angular_log10_cross_section_cm2,
        karzas_latter_high_level_energy_offset_rydberg,
    ):
        """One iteration of the Karzas-Latter cross-section loop.

        ``shell``/``angular_momentum`` must already be normalized ints and
        ``charge_squared > 0`` / ``shell > 0`` guaranteed by the caller.
        libm calls (math.exp/math.log10) match the np.* scalar calls;
        float-literal exponents match CPython's float**int (libm pow) where
        numba would otherwise lower an integer power to repeated
        multiplication.
        """

        if current_frequency <= 0.0:
            return 0.0
        ln10 = math.log(10.0)
        rydberg_frequency = 109677.576 * LIGHT_SPEED_CM_PER_S_EXACT
        log10_frequency = math.log10(current_frequency / charge_squared)

        if shell <= 15:
            frequency_column = karzas_latter_log10_frequency_hz[:, shell - 1]
            if log10_frequency < frequency_column[-1]:
                return 0.0
            if angular_momentum >= shell or shell > 6:
                value_column = karzas_latter_total_log10_cross_section_cm2[:, shell - 1]
            else:
                value_column = karzas_latter_angular_log10_cross_section_cm2[
                    angular_momentum,
                    shell - 1,
                    :,
                ]
                if math.isnan(value_column[0]):
                    return 0.0

            bracket = frequency_column.size
            left = 1
            right = frequency_column.size - 1
            while left <= right:
                midpoint = (left + right) // 2
                if log10_frequency > frequency_column[midpoint]:
                    bracket = midpoint
                    right = midpoint - 1
                else:
                    left = midpoint + 1

            if bracket >= frequency_column.size:
                return math.exp(value_column[-1] * ln10) / charge_squared
            denominator = frequency_column[bracket - 1] - frequency_column[bracket]
            if abs(denominator) < 1.0e-15:
                return math.exp(value_column[bracket - 1] * ln10) / charge_squared
            weight = (log10_frequency - frequency_column[bracket]) / denominator
            log10_cross_section_cm2 = (
                value_column[bracket - 1] - value_column[bracket]
            ) * weight + value_column[bracket]
            return math.exp(log10_cross_section_cm2 * ln10) / charge_squared

        high_shell_frequency = np.empty(29, dtype=np.float64)
        inverse_shell_squared = 1.0 / (shell * shell)
        high_shell_frequency[-1] = math.log10(rydberg_frequency * inverse_shell_squared)
        if log10_frequency < high_shell_frequency[-1]:
            return 0.0
        result = 0.0
        completed_without_bracket = True
        for table_index in range(1, 28):
            high_shell_frequency[table_index] = math.log10(
                (
                    karzas_latter_high_level_energy_offset_rydberg[table_index]
                    + inverse_shell_squared
                )
                * rydberg_frequency
            )
            if log10_frequency > high_shell_frequency[table_index]:
                completed_without_bracket = False
                denominator = (
                    high_shell_frequency[table_index - 1]
                    - high_shell_frequency[table_index]
                )
                if denominator == 0.0:
                    break
                weight = (
                    log10_frequency - high_shell_frequency[table_index]
                ) / denominator
                value_column = karzas_latter_total_log10_cross_section_cm2[:, 14]
                log10_cross_section_cm2 = (
                    value_column[table_index - 1] - value_column[table_index]
                ) * weight + value_column[table_index]
                result = math.exp(log10_cross_section_cm2 * ln10) / charge_squared
                break
        if completed_without_bracket:
            result = (
                math.exp(karzas_latter_total_log10_cross_section_cm2[28, 14] * ln10)
                / charge_squared
            )
        return result

    @numba.njit(parallel=True, nogil=True, cache=True)
    def _karzas_latter_cross_section_grid_kernel(
        frequency,
        charge_squared,
        shell,
        angular_momentum,
        karzas_latter_log10_frequency_hz,
        karzas_latter_total_log10_cross_section_cm2,
        karzas_latter_angular_log10_cross_section_cm2,
        karzas_latter_high_level_energy_offset_rydberg,
    ):
        # Each frequency writes its own result[index] (disjoint scatter), so the
        # per-frequency loop parallelizes over independent frequencies with no
        # cross-frequency accumulation. The point kernel is a pure function of
        # its frequency; the ~ulp libm divergence is immaterial under the
        # spectrum gate. Chunking is governed by NUMBA_NUM_THREADS.
        result = np.zeros(frequency.size, dtype=np.float64)
        if charge_squared <= 0.0 or shell <= 0:
            return result
        for index in numba.prange(frequency.shape[0]):
            result[index] = _karzas_latter_point_compiled(
                frequency[index],
                charge_squared,
                shell,
                angular_momentum,
                karzas_latter_log10_frequency_hz,
                karzas_latter_total_log10_cross_section_cm2,
                karzas_latter_angular_log10_cross_section_cm2,
                karzas_latter_high_level_energy_offset_rydberg,
            )
        return result

    @_njit_inline
    def _seaton_bound_free_cross_section_compiled(
        threshold_frequency_hz,
        threshold_cross_section,
        power,
        asymptotic_constant,
        frequency_hz,
    ):
        if frequency_hz < threshold_frequency_hz:
            return 0.0
        ratio = threshold_frequency_hz / frequency_hz
        exponent = int(2.0 * power + 0.01)
        return (
            threshold_cross_section
            * (asymptotic_constant + (1.0 - asymptotic_constant) * ratio)
            * math.sqrt(ratio ** float(exponent))
        )

    @numba.njit(parallel=True, nogil=True, cache=True)
    def _lukewarm_metal_absorption_kernel(
        absorption,
        frequency,
        wavenumber,
        stimulated_emission,
        exp_hnu_over_kt,
        hc_over_kt,
        thermal_energy_ev,
        mass_density,
        nitrogen_population,
        oxygen_population,
        carbon_ionized_population,
        magnesium_ionized_population,
        silicon_singly_ionized_population,
        calcium_ionized_population,
        magnesium_energy_cm,
        magnesium_boltzmann,
        magnesium_limit_boltzmann,
        magnesium_principal,
        magnesium_angular,
        magnesium_effective_charge_number,
        magnesium_limit,
        magnesium_rydberg,
        magnesium_charge_fourth,
        magnesium_denominator_scale,
        magnesium_kramers_threshold,
        carbon_energy_cm,
        carbon_boltzmann,
        carbon_boltzmann_1,
        carbon_boltzmann_2,
        carbon_principal,
        carbon_angular,
        carbon_rydberg,
        carbon_frequency_factor,
        carbon_denominator_scale,
        carbon_limit_1,
        carbon_limit_2,
        carbon_limit_3,
        carbon_kramers_1,
        carbon_kramers_2,
        silicon_frequency_factor,
        silicon_temperature_index,
        silicon_temperature_fraction,
        silicon_boltzmann_helper,
        silicon_singly_ionized_table,
        karzas_latter_log10_frequency_hz,
        karzas_latter_total_log10_cross_section_cm2,
        karzas_latter_angular_log10_cross_section_cm2,
        karzas_latter_high_level_energy_offset_rydberg,
    ):
        """The per-frequency LUKEOP (lukewarm-metal) loop.

        The frequency-independent layer vectors (nitrogen/calcium Boltzmann
        factors) are hoisted out of the frequency loop; each element is the
        same deterministic libm expression recomputed per frequency, so the
        hoist is value-identical. The per-frequency cross-section dot products
        use np.dot so numba resolves BLAS dgemv.
        """

        layer_count = hc_over_kt.shape[0]
        nitrogen_c1130 = np.empty(layer_count, dtype=np.float64)
        nitrogen_c1020 = np.empty(layer_count, dtype=np.float64)
        calcium_c1218 = np.empty(layer_count, dtype=np.float64)
        calcium_c1420 = np.empty(layer_count, dtype=np.float64)
        for layer_index in range(layer_count):
            nitrogen_c1130[layer_index] = 6.0 * math.exp(
                -3.575 / thermal_energy_ev[layer_index]
            )
            nitrogen_c1020[layer_index] = 10.0 * math.exp(
                -2.384 / thermal_energy_ev[layer_index]
            )
            calcium_c1218[layer_index] = 10.0 * math.exp(
                -1.697 / thermal_energy_ev[layer_index]
            )
            calcium_c1420[layer_index] = 6.0 * math.exp(
                -3.142 / thermal_energy_ev[layer_index]
            )

        # Each frequency writes its own absorption[:, frequency_index] column
        # (disjoint scatter); the frequency-independent layer vectors above are
        # hoisted out and shared read-only. Parallel over independent
        # frequencies; NUMBA_NUM_THREADS governs the thread count.
        for frequency_index in numba.prange(frequency.shape[0]):
            current_frequency = frequency[frequency_index]
            current_wavenumber = wavenumber[frequency_index]

            nitrogen_x853 = _seaton_bound_free_cross_section_compiled(
                3.517915e15,
                1.142e-17,
                2.0,
                4.29,
                current_frequency,
            )
            nitrogen_x1020 = _seaton_bound_free_cross_section_compiled(
                2.941534e15,
                4.41e-18,
                1.5,
                3.85,
                current_frequency,
            )
            nitrogen_x1130 = _seaton_bound_free_cross_section_compiled(
                2.653317e15,
                4.2e-18,
                1.5,
                4.34,
                current_frequency,
            )
            oxygen_x911 = _seaton_bound_free_cross_section_compiled(
                3.28805e15,
                2.94e-18,
                1.0,
                2.66,
                current_frequency,
            )

            magnesium_cross_section = np.zeros(14, dtype=np.float64)
            for level_index in range(13):
                threshold = magnesium_limit - magnesium_energy_cm[level_index]
                if current_wavenumber < threshold:
                    break
                magnesium_cross_section[level_index] = _karzas_latter_point_compiled(
                    current_frequency,
                    magnesium_effective_charge_number[level_index]
                    / magnesium_rydberg
                    * threshold,
                    magnesium_principal[level_index],
                    magnesium_angular[level_index],
                    karzas_latter_log10_frequency_hz,
                    karzas_latter_total_log10_cross_section_cm2,
                    karzas_latter_angular_log10_cross_section_cm2,
                    karzas_latter_high_level_energy_offset_rydberg,
                )
            if current_wavenumber >= magnesium_limit - magnesium_energy_cm[13]:
                ratio = (magnesium_limit - magnesium_energy_cm[13]) / max(
                    current_wavenumber,
                    1.0e-300,
                )
                magnesium_cross_section[13] = 0.14e-18 * (
                    6.700 * ratio**4.0 - 5.700 * ratio**5.0
                )
            magnesium_dot = np.dot(magnesium_cross_section, magnesium_boltzmann)
            magnesium_frequency_cubed = (
                2.815e29
                / max(current_frequency**3.0, 1.0e-300)
                * magnesium_charge_fourth
            )
            magnesium_exponent_wavenumber = max(
                magnesium_kramers_threshold,
                magnesium_limit - current_wavenumber,
            )

            carbon_cross_section = np.zeros(34, dtype=np.float64)
            for level_index in range(12):
                threshold = carbon_limit_1 - carbon_energy_cm[level_index]
                if current_wavenumber < threshold:
                    break
                principal_quantum_number = carbon_principal[level_index]
                if principal_quantum_number == 5:
                    charge_factor = 25.0
                elif principal_quantum_number == 4:
                    charge_factor = 16.0
                else:
                    charge_factor = 9.0
                carbon_cross_section[level_index] = _karzas_latter_point_compiled(
                    current_frequency,
                    charge_factor / carbon_rydberg * threshold,
                    principal_quantum_number,
                    carbon_angular[level_index],
                    karzas_latter_log10_frequency_hz,
                    karzas_latter_total_log10_cross_section_cm2,
                    karzas_latter_angular_log10_cross_section_cm2,
                    karzas_latter_high_level_energy_offset_rydberg,
                )
            for level_index in range(13, 19):
                threshold = carbon_limit_2 - carbon_energy_cm[level_index]
                if current_wavenumber < threshold:
                    break
                carbon_cross_section[level_index] = _karzas_latter_point_compiled(
                    current_frequency,
                    9.0 / carbon_rydberg * threshold,
                    3,
                    2,
                    karzas_latter_log10_frequency_hz,
                    karzas_latter_total_log10_cross_section_cm2,
                    karzas_latter_angular_log10_cross_section_cm2,
                    karzas_latter_high_level_energy_offset_rydberg,
                )
            for level_index in range(19, 25):
                threshold = carbon_limit_2 - carbon_energy_cm[level_index]
                if current_wavenumber < threshold:
                    break
                carbon_cross_section[level_index] = _karzas_latter_point_compiled(
                    current_frequency,
                    9.0 / carbon_rydberg * threshold,
                    3,
                    1,
                    karzas_latter_log10_frequency_hz,
                    karzas_latter_total_log10_cross_section_cm2,
                    karzas_latter_angular_log10_cross_section_cm2,
                    karzas_latter_high_level_energy_offset_rydberg,
                )
            for level_index in range(25, 27):
                threshold = carbon_limit_2 - carbon_energy_cm[level_index]
                if current_wavenumber < threshold:
                    break
                carbon_cross_section[level_index] = _karzas_latter_point_compiled(
                    current_frequency,
                    9.0 / carbon_rydberg * threshold,
                    3,
                    0,
                    karzas_latter_log10_frequency_hz,
                    karzas_latter_total_log10_cross_section_cm2,
                    karzas_latter_angular_log10_cross_section_cm2,
                    karzas_latter_high_level_energy_offset_rydberg,
                )
            for level_index in range(31, 34):
                threshold = carbon_limit_3 - carbon_energy_cm[level_index]
                if current_wavenumber < threshold:
                    break
                carbon_cross_section[level_index] = 3.0 * _karzas_latter_point_compiled(
                    current_frequency,
                    4.0 / carbon_rydberg * threshold,
                    2,
                    1,
                    karzas_latter_log10_frequency_hz,
                    karzas_latter_total_log10_cross_section_cm2,
                    karzas_latter_angular_log10_cross_section_cm2,
                    karzas_latter_high_level_energy_offset_rydberg,
                )
            carbon_dot = np.dot(carbon_cross_section, carbon_boltzmann)
            carbon_frequency_cubed = carbon_frequency_factor / max(
                current_frequency**3.0,
                1.0e-300,
            )
            carbon_exponent_wavenumber_1 = max(
                carbon_kramers_1,
                carbon_limit_1 - current_wavenumber,
            )
            carbon_exponent_wavenumber_2 = max(
                carbon_kramers_2,
                carbon_limit_2 - current_wavenumber,
            )

            silicon_uses_table = current_wavenumber >= 12192.48
            table_w0 = 0
            table_w1 = 0
            wavenumber_fraction = 0.0
            silicon_frequency_cubed = 0.0
            if silicon_uses_table:
                wavenumber_bin = int(current_wavenumber * 0.001)
                wavenumber_bin = max(min(wavenumber_bin, 199), 1)
                wavenumber_fraction = (
                    current_wavenumber - wavenumber_bin * 1000.0
                ) / 1000.0
                table_w0 = wavenumber_bin - 1
                table_w1 = wavenumber_bin
            else:
                silicon_frequency_cubed = silicon_frequency_factor / max(
                    current_frequency**3.0,
                    1.0e-300,
                )

            calcium_x1044 = 0.0
            calcium_x1218 = 0.0
            calcium_x1420 = 0.0
            if current_frequency >= 2.870454e15:
                calcium_x1044 = 5.4e-20 * (2.870454e15 / current_frequency) ** 3.0
            if current_frequency >= 2.460127e15:
                calcium_x1218 = 1.64e-17 * math.sqrt(2.460127e15 / current_frequency)
            if current_frequency >= 2.110779e15:
                calcium_x1420 = _seaton_bound_free_cross_section_compiled(
                    2.110779e15,
                    4.13e-18,
                    3.0,
                    0.69,
                    current_frequency,
                )

            for layer_index in range(layer_count):
                hc_layer = hc_over_kt[layer_index]
                stimulated_layer = stimulated_emission[layer_index, frequency_index]
                density_layer = mass_density[layer_index]

                nitrogen_opacity = (
                    nitrogen_x853 * 4.0
                    + nitrogen_x1020 * nitrogen_c1020[layer_index]
                    + nitrogen_x1130 * nitrogen_c1130[layer_index]
                )
                oxygen_opacity = oxygen_x911 * 9.0

                magnesium_profile = (
                    magnesium_frequency_cubed
                    / (magnesium_denominator_scale * hc_layer)
                    * (
                        math.exp(-magnesium_exponent_wavenumber * hc_layer)
                        - magnesium_limit_boltzmann[layer_index]
                    )
                    + magnesium_dot[layer_index]
                )

                carbon_profile = (
                    carbon_frequency_cubed
                    / (carbon_denominator_scale * hc_layer)
                    * (
                        math.exp(-carbon_exponent_wavenumber_1 * hc_layer)
                        - carbon_boltzmann_1[layer_index]
                    )
                )
                carbon_profile += (
                    carbon_frequency_cubed
                    * 9.0
                    / (carbon_denominator_scale * hc_layer)
                    * (
                        math.exp(-carbon_exponent_wavenumber_2 * hc_layer)
                        - carbon_boltzmann_2[layer_index]
                    )
                )
                carbon_profile += carbon_dot[layer_index]

                if silicon_uses_table:
                    table_t0 = silicon_temperature_index[layer_index] - 1
                    table_t1 = silicon_temperature_index[layer_index]
                    temperature_fraction = silicon_temperature_fraction[layer_index]
                    h00 = silicon_singly_ionized_table[table_w0, table_t0]
                    h01 = silicon_singly_ionized_table[table_w0, table_t1]
                    h10 = silicon_singly_ionized_table[table_w1, table_t0]
                    h11 = silicon_singly_ionized_table[table_w1, table_t1]
                    h0 = h00 * (1.0 - temperature_fraction) + h01 * temperature_fraction
                    h1 = h10 * (1.0 - temperature_fraction) + h11 * temperature_fraction
                    silicon_singly_ionized_opacity = (
                        math.exp(
                            h0 * (1.0 - wavenumber_fraction) + h1 * wavenumber_fraction
                        )
                        * silicon_singly_ionized_population[layer_index]
                        * stimulated_layer
                        / density_layer
                    )
                else:
                    silicon_profile = (
                        silicon_frequency_cubed
                        * (
                            1.0
                            / max(
                                exp_hnu_over_kt[layer_index, frequency_index],
                                1.0e-300,
                            )
                            - 1.0
                        )
                        * silicon_boltzmann_helper[layer_index]
                    )
                    silicon_singly_ionized_opacity = (
                        silicon_profile
                        * silicon_singly_ionized_population[layer_index]
                        * stimulated_layer
                        / density_layer
                    )

                calcium_opacity = (
                    calcium_x1044 * 2.0
                    + calcium_x1218 * calcium_c1218[layer_index]
                    + calcium_x1420 * calcium_c1420[layer_index]
                )

                absorption[layer_index, frequency_index] = (
                    nitrogen_opacity
                    * nitrogen_population[layer_index]
                    * stimulated_layer
                    / density_layer
                    + oxygen_opacity
                    * oxygen_population[layer_index]
                    * stimulated_layer
                    / density_layer
                    + calcium_opacity
                    * calcium_ionized_population[layer_index]
                    * stimulated_layer
                    / density_layer
                    + carbon_profile
                    * carbon_ionized_population[layer_index]
                    * stimulated_layer
                    / density_layer
                    + magnesium_profile
                    * magnesium_ionized_population[layer_index]
                    * stimulated_layer
                    / density_layer
                    + silicon_singly_ionized_opacity
                )

    @_njit
    def _evaluate_rosseland_opacity_kernel(
        normalized_log_temperature,
        normalized_log_pressure,
        log10_rosseland_opacity,
        entry_count,
        log_temperature_origin,
        log_pressure_origin,
        log_temperature_span,
        log_pressure_span,
        temperature_k,
        gas_pressure,
    ):
        if entry_count <= 0:
            return 1.0

        normalized_temperature = (
            math.log10(max(temperature_k, 1.0e-300)) - log_temperature_origin
        ) / log_temperature_span
        normalized_pressure = (
            math.log10(max(gas_pressure, 1.0e-300)) - log_pressure_origin
        ) / log_pressure_span

        radius_pp = 1.0e30
        radius_pm = 1.0e30
        radius_mp = 1.0e30
        radius_mm = 1.0e30
        index_pp = -1
        index_pm = -1
        index_mp = -1
        index_mm = -1
        value_pp = 0.0
        value_pm = 0.0
        value_mp = 0.0
        value_mm = 0.0
        for table_index in range(entry_count):
            pressure_delta = normalized_log_pressure[table_index] - normalized_pressure
            temperature_delta = (
                normalized_log_temperature[table_index] - normalized_temperature
            )
            radius2 = (
                temperature_delta * temperature_delta + pressure_delta * pressure_delta
            )
            if temperature_delta >= 0.0 and pressure_delta >= 0.0:
                if radius2 < radius_pp:
                    radius_pp = radius2
                    index_pp = table_index
                    value_pp = log10_rosseland_opacity[table_index]
            elif temperature_delta >= 0.0 and pressure_delta < 0.0:
                if radius2 < radius_pm:
                    radius_pm = radius2
                    index_pm = table_index
                    value_pm = log10_rosseland_opacity[table_index]
            elif temperature_delta < 0.0 and pressure_delta >= 0.0:
                if radius2 < radius_mp:
                    radius_mp = radius2
                    index_mp = table_index
                    value_mp = log10_rosseland_opacity[table_index]
            else:
                if radius2 < radius_mm:
                    radius_mm = radius2
                    index_mm = table_index
                    value_mm = log10_rosseland_opacity[table_index]

        if index_pp >= 0 and index_pm >= 0 and index_mp >= 0 and index_mm >= 0:
            temperature_pp = normalized_log_temperature[index_pp]
            pressure_pp = normalized_log_pressure[index_pp]
            temperature_pm = normalized_log_temperature[index_pm]
            pressure_pm = normalized_log_pressure[index_pm]
            temperature_mp = normalized_log_temperature[index_mp]
            pressure_mp = normalized_log_pressure[index_mp]
            temperature_mm = normalized_log_temperature[index_mm]
            pressure_mm = normalized_log_pressure[index_mm]

            upper_denominator = max(temperature_pp - temperature_mp, 1.0e-300)
            lower_denominator = max(temperature_pm - temperature_mm, 1.0e-300)
            upper_opacity = (
                (normalized_temperature - temperature_mp) * value_pp
                + (temperature_pp - normalized_temperature) * value_mp
            ) / upper_denominator
            lower_opacity = (
                (normalized_temperature - temperature_mm) * value_pm
                + (temperature_pm - normalized_temperature) * value_mm
            ) / lower_denominator
            upper_pressure = (
                (normalized_temperature - temperature_mp) * pressure_pp
                + (temperature_pp - normalized_temperature) * pressure_mp
            ) / upper_denominator
            lower_pressure = (
                (normalized_temperature - temperature_mm) * pressure_pm
                + (temperature_pm - normalized_temperature) * pressure_mm
            ) / lower_denominator
            log_opacity = (
                (normalized_pressure - lower_pressure) * upper_opacity
                + (upper_pressure - normalized_pressure) * lower_opacity
            ) / max(upper_pressure - lower_pressure, 1.0e-300)
            return 10.0**log_opacity

        weight_pp = 1.0 / (math.sqrt(radius_pp) + 1.0e-5)
        weight_pm = 1.0 / (math.sqrt(radius_pm) + 1.0e-5)
        weight_mp = 1.0 / (math.sqrt(radius_mp) + 1.0e-5)
        weight_mm = 1.0 / (math.sqrt(radius_mm) + 1.0e-5)
        weight_sum = weight_pp + weight_pm + weight_mp + weight_mm
        log_opacity = (
            log10_rosseland_opacity[max(index_pp, 0)] * weight_pp
            + log10_rosseland_opacity[max(index_pm, 0)] * weight_pm
            + log10_rosseland_opacity[max(index_mp, 0)] * weight_mp
            + log10_rosseland_opacity[max(index_mm, 0)] * weight_mm
        ) / max(weight_sum, 1.0e-300)
        return 10.0**log_opacity


def _karzas_latter_cross_section_grid(
    frequency_hz: np.ndarray,
    *,
    effective_charge_squared: float,
    principal_quantum_number: int,
    orbital_angular_momentum: int,
    tables: KarzasLatterTables | None = None,
) -> np.ndarray:
    """Return Karzas-Latter hydrogenic bound-free cross sections."""

    table = load_karzas_latter_tables() if tables is None else tables
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    return _karzas_latter_cross_section_grid_kernel(
        frequency,
        float(effective_charge_squared),
        int(principal_quantum_number),
        max(0, int(orbital_angular_momentum)),
        table.karzas_latter_log10_frequency_hz,
        table.karzas_latter_total_log10_cross_section_cm2,
        table.karzas_latter_angular_log10_cross_section_cm2,
        table.karzas_latter_high_level_energy_offset_rydberg,
    )


def _helium_ground_cross_section(frequency_hz: float) -> float:
    """Return the He I ground-state photoionization cross section."""

    if frequency_hz < 5.945209e15:
        return 0.0
    wavelength_angstrom = LIGHT_SPEED_ANGSTROM_PER_S / frequency_hz
    if wavelength_angstrom > 50.0:
        table_index = int(93.0 - (wavelength_angstrom - 50.0) / 5.0)
        table_index = min(92, max(2, table_index))
        return (
            (wavelength_angstrom - (92 - table_index) * 5.0 - 50.0)
            / 5.0
            * (
                _HELIUM_GROUND_CROSS_SECTION_50_505[table_index - 2]
                - _HELIUM_GROUND_CROSS_SECTION_50_505[table_index - 1]
            )
            + _HELIUM_GROUND_CROSS_SECTION_50_505[table_index - 1]
        ) * 1.0e-18
    if wavelength_angstrom > 20.0:
        table_index = int(17.0 - (wavelength_angstrom - 20.0) / 2.0)
        table_index = min(16, max(2, table_index))
        return (
            (wavelength_angstrom - (16 - table_index) * 2.0 - 20.0)
            / 2.0
            * (
                _HELIUM_GROUND_CROSS_SECTION_20_50[table_index - 2]
                - _HELIUM_GROUND_CROSS_SECTION_20_50[table_index - 1]
            )
            + _HELIUM_GROUND_CROSS_SECTION_20_50[table_index - 1]
        ) * 1.0e-18
    if wavelength_angstrom > 10.0:
        table_index = int(12.0 - (wavelength_angstrom - 10.0) / 1.0)
        table_index = min(11, max(2, table_index))
        return (
            (wavelength_angstrom - (11 - table_index) - 10.0)
            * (
                _HELIUM_GROUND_CROSS_SECTION_10_20[table_index - 2]
                - _HELIUM_GROUND_CROSS_SECTION_10_20[table_index - 1]
            )
            + _HELIUM_GROUND_CROSS_SECTION_10_20[table_index - 1]
        ) * 1.0e-18
    table_index = int(22.0 - wavelength_angstrom / 0.5)
    table_index = min(21, max(2, table_index))
    return (
        (wavelength_angstrom - (21 - table_index) * 0.5)
        / 0.5
        * (
            _HELIUM_GROUND_CROSS_SECTION_0_10[table_index - 2]
            - _HELIUM_GROUND_CROSS_SECTION_0_10[table_index - 1]
        )
        + _HELIUM_GROUND_CROSS_SECTION_0_10[table_index - 1]
    ) * 1.0e-18


def _helium_tabulated_cross_section(
    frequency_hz: float,
    *,
    threshold_wavenumber_cm: float,
    log_frequency_table: np.ndarray,
    log10_cross_section_cm2_table: np.ndarray,
    high_frequency_kind: str | None,
) -> float:
    """Return one He I excited-state photoionization cross section."""

    if frequency_hz < threshold_wavenumber_cm * LIGHT_SPEED_CM_PER_S_EXACT:
        return 0.0
    if (
        high_frequency_kind is not None
        and frequency_hz > 2.4 * 109722.267 * LIGHT_SPEED_CM_PER_S_EXACT
    ):
        wavenumber = frequency_hz / LIGHT_SPEED_CM_PER_S_EXACT
        if high_frequency_kind == "1s2s_singlet":
            kinetic_energy = (wavenumber - 32033.214) / 109722.267
            epsilon = 2.0 * (kinetic_energy - 2.612316) / 0.00322
            return (
                0.008175
                * (484940.0 / wavenumber) ** 2.71
                * 8.067e-18
                * (epsilon + 76.21) ** 2
                / (1.0 + epsilon**2)
            )
        if high_frequency_kind == "1s2s_triplet":
            kinetic_energy = (wavenumber - 38454.691) / 109722.267
            epsilon = 2.0 * (kinetic_energy - 2.47898) / 0.000780
            return (
                0.01521
                * (470310.0 / wavenumber) ** 3.12
                * 8.067e-18
                * (epsilon - 122.4) ** 2
                / (1.0 + epsilon**2)
            )
        if high_frequency_kind == "1s2p_singlet":
            kinetic_energy = (wavenumber - 27175.76) / 109722.267
            epsilon_s = 2.0 * (kinetic_energy - 2.446534) / 0.01037
            epsilon_d = 2.0 * (kinetic_energy - 2.59427) / 0.00538
            return (
                0.0009487
                * (466750.0 / wavenumber) ** 3.69
                * 8.067e-18
                * (
                    (epsilon_s - 29.30) ** 2 / (1.0 + epsilon_s**2)
                    + (epsilon_d + 172.4) ** 2 / (1.0 + epsilon_d**2)
                )
            )

    log10_frequency = np.log10(frequency_hz)
    table_index = 15
    for candidate_index in range(1, 16):
        if log10_frequency > log_frequency_table[candidate_index]:
            table_index = candidate_index
            break
    log10_cross_section_cm2 = (log10_frequency - log_frequency_table[table_index]) / (
        log_frequency_table[table_index - 1] - log_frequency_table[table_index]
    ) * (
        log10_cross_section_cm2_table[table_index - 1]
        - log10_cross_section_cm2_table[table_index]
    ) + log10_cross_section_cm2_table[table_index]
    return 10.0**log10_cross_section_cm2


def _helium_neutral_transition_grid(
    frequency_hz: np.ndarray,
    *,
    karzas_tables: KarzasLatterTables,
) -> tuple[np.ndarray, np.ndarray]:
    """Return He I bound-free grids for the ten low levels and high-n levels."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if _helium_low_level_grid_kernel is not None:
        # Compiled, prange-parallel over frequencies for the 5 He I low levels;
        # the deep levels (5..9) and autoionization additions below stay on the
        # already-parallel _karzas_latter_cross_section_grid path.
        low_level_cross_sections = _helium_low_level_grid_kernel(
            frequency,
            _HELIUM_NEUTRAL_THRESHOLD_FREQUENCY_HZ,
            _HELIUM_1S2S_TRIPLET_LOG_FREQUENCY,
            _HELIUM_1S2S_TRIPLET_LOG_CROSS_SECTION,
            _HELIUM_1S2S_SINGLET_LOG_FREQUENCY,
            _HELIUM_1S2S_SINGLET_LOG_CROSS_SECTION,
            _HELIUM_1S2P_TRIPLET_LOG_FREQUENCY,
            _HELIUM_1S2P_TRIPLET_LOG_CROSS_SECTION,
            _HELIUM_1S2P_SINGLET_LOG_FREQUENCY,
            _HELIUM_1S2P_SINGLET_LOG_CROSS_SECTION,
        )
    else:
        low_level_cross_sections = np.zeros((10, frequency.size), dtype=np.float64)
        for frequency_index, current_frequency in enumerate(frequency):
            if current_frequency >= _HELIUM_NEUTRAL_THRESHOLD_FREQUENCY_HZ[0]:
                low_level_cross_sections[0, frequency_index] = (
                    _helium_ground_cross_section(
                        current_frequency,
                    )
                )
            if current_frequency >= _HELIUM_NEUTRAL_THRESHOLD_FREQUENCY_HZ[1]:
                low_level_cross_sections[1, frequency_index] = (
                    _helium_tabulated_cross_section(
                        current_frequency,
                        threshold_wavenumber_cm=38454.691,
                        log_frequency_table=_HELIUM_1S2S_TRIPLET_LOG_FREQUENCY,
                        log10_cross_section_cm2_table=_HELIUM_1S2S_TRIPLET_LOG_CROSS_SECTION,
                        high_frequency_kind="1s2s_triplet",
                    )
                )
            if current_frequency >= _HELIUM_NEUTRAL_THRESHOLD_FREQUENCY_HZ[2]:
                low_level_cross_sections[2, frequency_index] = (
                    _helium_tabulated_cross_section(
                        current_frequency,
                        threshold_wavenumber_cm=32033.214,
                        log_frequency_table=_HELIUM_1S2S_SINGLET_LOG_FREQUENCY,
                        log10_cross_section_cm2_table=_HELIUM_1S2S_SINGLET_LOG_CROSS_SECTION,
                        high_frequency_kind="1s2s_singlet",
                    )
                )
            if current_frequency >= _HELIUM_NEUTRAL_THRESHOLD_FREQUENCY_HZ[3]:
                low_level_cross_sections[3, frequency_index] = (
                    _helium_tabulated_cross_section(
                        current_frequency,
                        threshold_wavenumber_cm=29223.753,
                        log_frequency_table=_HELIUM_1S2P_TRIPLET_LOG_FREQUENCY,
                        log10_cross_section_cm2_table=_HELIUM_1S2P_TRIPLET_LOG_CROSS_SECTION,
                        high_frequency_kind=None,
                    )
                )
            if current_frequency >= _HELIUM_NEUTRAL_THRESHOLD_FREQUENCY_HZ[4]:
                low_level_cross_sections[4, frequency_index] = (
                    _helium_tabulated_cross_section(
                        current_frequency,
                        threshold_wavenumber_cm=27175.76,
                        log_frequency_table=_HELIUM_1S2P_SINGLET_LOG_FREQUENCY,
                        log10_cross_section_cm2_table=_HELIUM_1S2P_SINGLET_LOG_CROSS_SECTION,
                        high_frequency_kind="1s2p_singlet",
                    )
                )

    fixed_transitions = {
        5: (1.236439, 3, 0),
        6: (1.102898, 3, 0),
        7: (1.045499, 3, 1),
        8: (1.001427, 3, 2),
        9: (0.9926, 3, 1),
    }
    for level_index, (
        charge_squared,
        shell,
        angular_momentum,
    ) in fixed_transitions.items():
        active = frequency >= _HELIUM_NEUTRAL_THRESHOLD_FREQUENCY_HZ[level_index]
        if np.any(active):
            low_level_cross_sections[level_index, active] = (
                _karzas_latter_cross_section_grid(
                    frequency[active],
                    effective_charge_squared=charge_squared,
                    principal_quantum_number=shell,
                    orbital_angular_momentum=angular_momentum,
                    tables=karzas_tables,
                )
            )

    rydberg_frequency = 109722.273 * LIGHT_SPEED_CM_PER_S_EXACT
    autoionization_n2 = [
        (171135.000, 4),
        (169087.0, 3),
        (166277.546, 2),
        (159856.069, 1),
    ]
    for level_cm, target_level in autoionization_n2:
        threshold_frequency = (527490.06 - level_cm) * LIGHT_SPEED_CM_PER_S_EXACT
        active = frequency >= threshold_frequency
        if np.any(active):
            low_level_cross_sections[target_level, active] += (
                _karzas_latter_cross_section_grid(
                    frequency[active],
                    effective_charge_squared=threshold_frequency / rydberg_frequency,
                    principal_quantum_number=1,
                    orbital_angular_momentum=0,
                    tables=karzas_tables,
                )
            )

    autoionization_n3 = [
        (186209.471, 9),
        (186101.0, 8),
        (185564.0, 7),
        (184864.0, 6),
        (183236.0, 5),
    ]
    for level_cm, target_level in autoionization_n3:
        threshold_frequency = (588451.59 - level_cm) * LIGHT_SPEED_CM_PER_S_EXACT
        active = frequency >= threshold_frequency
        if np.any(active):
            low_level_cross_sections[target_level, active] += (
                _karzas_latter_cross_section_grid(
                    frequency[active],
                    effective_charge_squared=threshold_frequency / rydberg_frequency,
                    principal_quantum_number=1,
                    orbital_angular_momentum=0,
                    tables=karzas_tables,
                )
            )

    high_level_cross_sections = np.zeros((28, frequency.size), dtype=np.float64)
    for shell in range(4, 28):
        high_level_cross_sections[shell, :] = _karzas_latter_cross_section_grid(
            frequency,
            effective_charge_squared=4.0 - 3.0 / (shell * shell),
            principal_quantum_number=1,
            orbital_angular_momentum=0,
            tables=karzas_tables,
        )
    return low_level_cross_sections, high_level_cross_sections


def _coulomb_freefree_gaunt_grid(
    ion_charge: int,
    natural_log_frequency: np.ndarray,
    natural_log_temperature: np.ndarray,
    *,
    tables: ContinuumOpacityTables | None = None,
) -> np.ndarray:
    """Return the Coulomb free-free grid over layers and frequencies."""

    table = load_continuum_opacity_tables() if tables is None else tables
    charge = int(ion_charge)
    frequency_log = np.asarray(natural_log_frequency, dtype=np.float64)
    temperature_log = np.asarray(natural_log_temperature, dtype=np.float64)
    if charge < 1 or charge > 6:
        return np.ones((temperature_log.size, frequency_log.size), dtype=np.float64)

    z4log = table.coulomb_freefree_charge_log_offset[charge - 1]
    gaunt_table = np.ascontiguousarray(
        table.coulomb_freefree_gaunt_table, dtype=np.float64
    )
    if _NUMBA_AVAILABLE:
        # Compiled prange kernel over frequencies (scales with NUMBA_NUM_THREADS);
        # element-for-element the numpy bilinear interpolation below.
        return _coulomb_freefree_gaunt_kernel(
            np.ascontiguousarray(frequency_log, dtype=np.float64),
            np.ascontiguousarray(temperature_log, dtype=np.float64),
            float(z4log),
            gaunt_table,
        )
    gamlog = 10.39638 - temperature_log[:, None] / 1.15129 + z4log
    hvktlg = (frequency_log[None, :] - temperature_log[:, None]) / 1.15129 - 20.63764
    igam = np.clip((gamlog + 7.0).astype(np.int64), 1, 10)
    ihvkt = np.clip((hvktlg + 9.0).astype(np.int64), 1, 11)
    p_weight = gamlog - (igam - 7.0)
    q_weight = hvktlg - (ihvkt - 9.0)

    ig = igam - 1
    ih = ihvkt - 1
    a00 = gaunt_table[ih, ig]
    a01 = gaunt_table[ih + 1, ig]
    a10 = gaunt_table[ih, ig + 1]
    a11 = gaunt_table[ih + 1, ig + 1]
    return (1.0 - p_weight) * ((1.0 - q_weight) * a00 + q_weight * a01) + p_weight * (
        (1.0 - q_weight) * a10 + q_weight * a11
    )


def compute_hydrogen_opacity_columns(
    atmosphere: ContinuumAtmosphereState,
    frequency_hz: np.ndarray,
    *,
    continuum_tables: ContinuumOpacityTables | None = None,
    karzas_tables: KarzasLatterTables | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return hydrogen absorption and source columns."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if frequency.ndim != 1:
        raise ValueError("frequency_hz must be one-dimensional")

    continuum_table = (
        load_continuum_opacity_tables()
        if continuum_tables is None
        else continuum_tables
    )
    karzas_table = (
        load_karzas_latter_tables() if karzas_tables is None else karzas_tables
    )
    temperature = np.asarray(atmosphere.temperature, dtype=np.float64)
    mass_density = np.maximum(
        np.asarray(atmosphere.mass_density, dtype=np.float64), 1.0e-300
    )
    electron_density = np.asarray(atmosphere.electron_density, dtype=np.float64)
    hydrogen_neutral_partition_normalized_population = np.asarray(
        atmosphere.hydrogen_partition_normalized_ion_stage_populations[:, 0],
        dtype=np.float64,
    )
    hydrogen_ionized_population = np.asarray(
        atmosphere.hydrogen_ionized_population,
        dtype=np.float64,
    )
    hydrogen_departure = np.asarray(
        atmosphere.hydrogen_departure_coefficients,
        dtype=np.float64,
    )
    if hydrogen_departure.ndim == 1:
        hydrogen_departure = hydrogen_departure[:, None]
    if hydrogen_departure.shape[1] < 6:
        padding = np.ones(
            (temperature.size, 6 - hydrogen_departure.shape[1]), dtype=np.float64
        )
        hydrogen_departure = np.hstack([hydrogen_departure, padding])

    planck_nu, exp_hnu_over_kt, stimulated_emission = _planck_frequency_exact(
        temperature_k=temperature,
        frequency_hz=frequency,
    )
    thermal_energy_ev = np.maximum(temperature * BOLTZMANN_EV_PER_K_REFERENCE, 1.0e-300)
    boltzmann_population = np.zeros((temperature.size, 8), dtype=np.float64)
    for shell in range(1, 9):
        boltzmann_population[:, shell - 1] = (
            np.exp(-(13.595 - 13.595 / float(shell * shell)) / thermal_energy_ev)
            * 2.0
            * float(shell * shell)
            * hydrogen_neutral_partition_normalized_population
            / mass_density
        )
        if shell <= 6:
            boltzmann_population[:, shell - 1] *= hydrogen_departure[:, shell - 1]

    freefree_density_factor = (
        electron_density
        * hydrogen_ionized_population
        / mass_density
        / np.sqrt(np.maximum(temperature, 1.0e-300))
    )
    xr = (
        hydrogen_neutral_partition_normalized_population
        * (thermal_energy_ev / 13.595)
        / mass_density
    )
    boltzmann_extension = np.exp(-13.427 / thermal_energy_ev) * xr
    series_limit_extension = np.exp(-13.595 / thermal_energy_ev) * xr
    coulomb_freefree = _coulomb_freefree_gaunt_grid(
        1,
        np.log(np.maximum(frequency, 1.0e-300)),
        np.log(np.maximum(temperature, 1.0e-300)),
        tables=continuum_table,
    )
    karzas_cross_sections = np.vstack(
        [
            _karzas_latter_cross_section_grid(
                frequency,
                effective_charge_squared=1.0,
                principal_quantum_number=shell,
                orbital_angular_momentum=shell,
                tables=karzas_table,
            )
            for shell in range(1, 9)
        ]
    )

    frequency_cubed = np.maximum(frequency * frequency * frequency, 1.0e-300)
    freefree_coefficient = 3.6919e8 / frequency_cubed
    extension_coefficient = 2.815e29 / frequency_cubed
    extension_population = np.broadcast_to(
        boltzmann_extension[:, None],
        (temperature.size, frequency.size),
    ).copy()
    low_frequency = frequency < 4.05933e13
    if np.any(low_frequency):
        extension_population[:, low_frequency] = series_limit_extension[
            :, None
        ] / np.maximum(exp_hnu_over_kt[:, low_frequency], 1.0e-300)

    absorption = (
        karzas_cross_sections[6, :][None, :] * boltzmann_population[:, 6][:, None]
        + karzas_cross_sections[7, :][None, :] * boltzmann_population[:, 7][:, None]
        + (extension_population - series_limit_extension[:, None])
        * extension_coefficient[None, :]
        + coulomb_freefree
        * freefree_density_factor[:, None]
        * freefree_coefficient[None, :]
    ) * stimulated_emission
    source_numerator = absorption * planck_nu

    for shell_index in range(6):
        departure = np.maximum(hydrogen_departure[:, shell_index], 1.0e-300)
        boundfree_term = (
            karzas_cross_sections[shell_index, :][None, :]
            * boltzmann_population[:, shell_index][:, None]
        )
        absorption += boundfree_term * (1.0 - exp_hnu_over_kt / departure[:, None])
        source_numerator += (
            boundfree_term * planck_nu * stimulated_emission / departure[:, None]
        )

    source = np.where(
        absorption > 0.0,
        source_numerator / np.maximum(absorption, 1.0e-300),
        planck_nu,
    )
    return absorption, source


def compute_helium_neutral_opacity_columns(
    atmosphere: ContinuumAtmosphereState,
    frequency_hz: np.ndarray,
    *,
    continuum_tables: ContinuumOpacityTables | None = None,
    karzas_tables: KarzasLatterTables | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return He I absorption and LTE source columns."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if frequency.ndim != 1:
        raise ValueError("frequency_hz must be one-dimensional")

    continuum_table = (
        load_continuum_opacity_tables()
        if continuum_tables is None
        else continuum_tables
    )
    karzas_table = (
        load_karzas_latter_tables() if karzas_tables is None else karzas_tables
    )
    temperature = np.asarray(atmosphere.temperature, dtype=np.float64)
    mass_density = np.maximum(
        np.asarray(atmosphere.mass_density, dtype=np.float64), 1.0e-300
    )
    electron_density = np.asarray(atmosphere.electron_density, dtype=np.float64)
    helium_neutral_partition_normalized = np.asarray(
        atmosphere.helium_neutral_partition_normalized_population,
        dtype=np.float64,
    )
    helium_singly_ionized = np.asarray(
        atmosphere.helium_singly_ionized_population,
        dtype=np.float64,
    )

    planck_nu, exp_hnu_over_kt, stimulated_emission = _planck_frequency_exact(
        temperature_k=temperature,
        frequency_hz=frequency,
    )
    thermal_energy_ev = np.maximum(temperature * BOLTZMANN_EV_PER_K_REFERENCE, 1.0e-300)
    low_level_population = np.zeros((temperature.size, 10), dtype=np.float64)
    for level_index in range(10):
        low_level_population[:, level_index] = (
            np.exp(-_HELIUM_NEUTRAL_EXCITATION_EV[level_index] / thermal_energy_ev)
            * _HELIUM_NEUTRAL_STATISTICAL_WEIGHTS[level_index]
            * helium_neutral_partition_normalized
            / mass_density
        )

    high_level_population = np.zeros((temperature.size, 28), dtype=np.float64)
    for shell in range(4, 28):
        high_level_population[:, shell] = (
            np.exp(-24.587 * (1.0 - 1.0 / (shell * shell)) / thermal_energy_ev)
            * 4.0
            * shell
            * shell
            * helium_neutral_partition_normalized
            / mass_density
        )

    freefree_density_factor = (
        electron_density
        * helium_singly_ionized
        / mass_density
        / np.sqrt(np.maximum(temperature, 1.0e-300))
    )
    xr = (
        helium_neutral_partition_normalized
        * (4.0 / 2.0 / 13.595)
        * thermal_energy_ev
        / mass_density
    )
    boltzmann_extension = np.exp(-23.730 / thermal_energy_ev) * xr
    series_limit_extension = np.exp(-24.587 / thermal_energy_ev) * xr
    coulomb_freefree = _coulomb_freefree_gaunt_grid(
        1,
        np.log(np.maximum(frequency, 1.0e-300)),
        np.log(np.maximum(temperature, 1.0e-300)),
        tables=continuum_table,
    )
    low_level_cross_sections, high_level_cross_sections = (
        _helium_neutral_transition_grid(
            frequency,
            karzas_tables=karzas_table,
        )
    )

    bound_contribution = low_level_population @ low_level_cross_sections
    high_frequency = frequency >= 1.25408e16
    if np.any(high_frequency):
        bound_contribution[:, high_frequency] += (
            high_level_population @ high_level_cross_sections
        )[:, high_frequency]

    frequency_cubed = frequency * frequency * frequency
    freefree_coefficient = 3.6919e8 / frequency_cubed
    extension_coefficient = 2.815e29 / frequency_cubed
    extension_population = np.empty(
        (temperature.size, frequency.size), dtype=np.float64
    )
    low_frequency = frequency < 2.055e14
    if np.any(low_frequency):
        extension_population[:, low_frequency] = (
            series_limit_extension[:, None] / exp_hnu_over_kt[:, low_frequency]
        )
    if np.any(~low_frequency):
        extension_population[:, ~low_frequency] = boltzmann_extension[:, None]

    absorption = (
        (extension_population - series_limit_extension[:, None])
        * extension_coefficient[None, :]
        + bound_contribution
        + coulomb_freefree
        * freefree_coefficient[None, :]
        * freefree_density_factor[:, None]
    ) * stimulated_emission
    return absorption, planck_nu


def compute_hminus_opacity_columns(
    atmosphere: ContinuumAtmosphereState,
    frequency_hz: np.ndarray,
    *,
    hminus_departure_coefficient: np.ndarray | None = None,
    continuum_tables: ContinuumOpacityTables | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return H-minus absorption and source columns."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if frequency.ndim != 1:
        raise ValueError("frequency_hz must be one-dimensional")

    tables = (
        load_continuum_opacity_tables()
        if continuum_tables is None
        else continuum_tables
    )
    temperature = np.asarray(atmosphere.temperature, dtype=np.float64)
    mass_density = np.maximum(
        np.asarray(atmosphere.mass_density, dtype=np.float64), 1.0e-300
    )
    electron_density = np.asarray(atmosphere.electron_density, dtype=np.float64)
    hydrogen_neutral_partition_normalized_population = np.asarray(
        atmosphere.hydrogen_partition_normalized_ion_stage_populations[:, 0],
        dtype=np.float64,
    )
    hydrogen_departure = np.asarray(
        atmosphere.hydrogen_departure_coefficients[:, 0],
        dtype=np.float64,
    )
    if hminus_departure_coefficient is None:
        hminus_departure = np.ones(temperature.size, dtype=np.float64)
    else:
        hminus_departure = np.asarray(hminus_departure_coefficient, dtype=np.float64)
    if hminus_departure.shape != temperature.shape:
        raise ValueError(
            "hminus_departure_coefficient must match the atmosphere layers"
        )

    planck_nu, exp_hnu_over_kt, stimulated_emission = _planck_frequency_exact(
        temperature_k=temperature,
        frequency_hz=frequency,
    )
    thermal_energy_ev = temperature * BOLTZMANN_EV_PER_K_REFERENCE
    hminus_population = (
        np.exp(0.754209 / np.maximum(thermal_energy_ev, 1.0e-300))
        / (2.0 * 2.4148e15 * temperature * np.sqrt(np.maximum(temperature, 1.0e-300)))
        * hminus_departure
        * hydrogen_departure
        * hydrogen_neutral_partition_normalized_population
        * electron_density
    )

    theta = 5040.0 / temperature
    wfflog = np.log(91.134 / tables.hminus_freefree_inverse_wavelength_grid)
    freefree_table = np.zeros(
        (tables.hminus_freefree_theta_grid.size, 22), dtype=np.float64
    )
    for theta_index in range(tables.hminus_freefree_theta_grid.size):
        for wavelength_index in range(22):
            if wavelength_index < 11:
                freefree_table[theta_index, wavelength_index] = (
                    tables.hminus_freefree_short_wavelength_table[
                        wavelength_index, theta_index
                    ]
                )
            else:
                freefree_table[theta_index, wavelength_index] = (
                    tables.hminus_freefree_long_wavelength_table[
                        wavelength_index - 11, theta_index
                    ]
                )

    freefree_log = np.zeros(
        (22, tables.hminus_freefree_theta_grid.size), dtype=np.float64
    )
    for wavelength_index in range(22):
        for theta_index, theta_grid_value in enumerate(
            tables.hminus_freefree_theta_grid
        ):
            freefree_log[wavelength_index, theta_index] = np.log(
                freefree_table[theta_index, wavelength_index]
                / theta_grid_value
                * 5040.0
                * BOLTZMANN_ERG_PER_K_EXACT
            )

    wavelength_nm = LIGHT_SPEED_NM_PER_S / np.maximum(frequency, 1.0e-30)
    wavelength_log = np.log(wavelength_nm)
    freefree_by_theta = np.empty(
        (tables.hminus_freefree_theta_grid.size, frequency.size),
        dtype=np.float64,
    )
    for theta_index in range(tables.hminus_freefree_theta_grid.size):
        freefree_by_theta[theta_index, :] = np.exp(
            _linear_interpolate_with_extrapolation(
                wfflog,
                freefree_log[:, theta_index],
                wavelength_log,
            )
        )

    freefree_theta = np.empty((temperature.size, frequency.size), dtype=np.float64)
    theta_grid = tables.hminus_freefree_theta_grid
    for layer_index, layer_theta in enumerate(theta):
        table_index = int(np.searchsorted(theta_grid, layer_theta, side="right"))
        table_index = max(1, min(table_index, theta_grid.size - 1))
        denominator = theta_grid[table_index] - theta_grid[table_index - 1]
        if abs(denominator) < 1.0e-40:
            freefree_theta[layer_index, :] = freefree_by_theta[table_index - 1, :]
        else:
            weight = (layer_theta - theta_grid[table_index - 1]) / denominator
            freefree_theta[layer_index, :] = (
                freefree_by_theta[table_index - 1, :]
                + (
                    freefree_by_theta[table_index, :]
                    - freefree_by_theta[table_index - 1, :]
                )
                * weight
            )

    boundfree_cross_section = np.zeros(frequency.size, dtype=np.float64)
    active_boundfree = frequency > 1.82365e14
    if np.any(active_boundfree):
        boundfree_cross_section[active_boundfree] = _piecewise_quadratic_remap(
            tables.hminus_boundfree_wavelength_nm,
            tables.hminus_boundfree_cross_section_cm2,
            wavelength_nm[active_boundfree],
        )

    freefree_absorption = (
        freefree_theta
        * (
            hydrogen_neutral_partition_normalized_population
            * 2.0
            * hydrogen_departure
            * electron_density
            / mass_density
            * 1.0e-26
        )[:, None]
    )
    boundfree_absorption = (
        boundfree_cross_section[None, :]
        * 1.0e-18
        * (1.0 - exp_hnu_over_kt / np.maximum(hminus_departure, 1.0e-40)[:, None])
        * hminus_population[:, None]
        / mass_density[:, None]
    )
    absorption = boundfree_absorption + freefree_absorption

    source_denominator = hminus_departure[:, None] - exp_hnu_over_kt
    boundfree_source = (
        boundfree_absorption
        * planck_nu
        * stimulated_emission
        / np.maximum(source_denominator, 1.0e-40)
    )
    source = np.where(
        absorption > 0.0,
        (boundfree_source + freefree_absorption * planck_nu) / absorption,
        planck_nu,
    )
    return absorption, source


def compute_molecular_hydrogen_ion_opacity_columns(
    atmosphere: ContinuumAtmosphereState,
    frequency_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return H2-plus absorption and LTE source columns."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if frequency.ndim != 1:
        raise ValueError("frequency_hz must be one-dimensional")

    temperature = np.asarray(atmosphere.temperature, dtype=np.float64)
    mass_density = np.maximum(
        np.asarray(atmosphere.mass_density, dtype=np.float64), 1.0e-300
    )
    hydrogen_levels = np.asarray(
        atmosphere.hydrogen_partition_normalized_ion_stage_populations, dtype=np.float64
    )
    if hydrogen_levels.ndim != 2 or hydrogen_levels.shape[1] < 2:
        raise ValueError(
            "hydrogen_partition_normalized_ion_stage_populations must include the first two levels"
        )
    hydrogen_departure = np.asarray(
        atmosphere.hydrogen_departure_coefficients[:, 0],
        dtype=np.float64,
    )

    planck_nu, _, stimulated_emission = _planck_frequency_exact(
        temperature_k=temperature,
        frequency_hz=frequency,
    )
    absorption = np.zeros((temperature.size, frequency.size), dtype=np.float64)
    active = frequency <= 3.28805e15
    if np.any(active):
        active_frequency = frequency[active]
        natural_log_frequency = np.log(active_frequency)
        frequency_1e15 = active_frequency / 1.0e15
        fr_polynomial = (
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
        excitation_energy = (
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
        thermal_energy_ev = np.maximum(
            temperature * BOLTZMANN_EV_PER_K_REFERENCE, 1.0e-300
        )
        ground_population = np.maximum(hydrogen_levels[:, 0], 1.0e-40)
        first_excited_population = hydrogen_levels[:, 1]
        absorption[:, active] = (
            np.exp(
                -excitation_energy[None, :] / thermal_energy_ev[:, None]
                + fr_polynomial[None, :]
                + np.log(ground_population)[:, None]
            )
            * 2.0
            * hydrogen_departure[:, None]
            * first_excited_population[:, None]
            / mass_density[:, None]
            * stimulated_emission[:, active]
        )
    return absorption, planck_nu


def compute_heminus_opacity_columns(
    atmosphere: ContinuumAtmosphereState,
    frequency_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return He-minus absorption and LTE source columns."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if frequency.ndim != 1:
        raise ValueError("frequency_hz must be one-dimensional")

    temperature = np.asarray(atmosphere.temperature, dtype=np.float64)
    electron_density = np.asarray(atmosphere.electron_density, dtype=np.float64)
    helium_neutral = np.asarray(atmosphere.helium_neutral_population, dtype=np.float64)
    mass_density = np.maximum(
        np.asarray(atmosphere.mass_density, dtype=np.float64), 1.0e-300
    )
    planck_nu, _, _ = _planck_frequency_exact(
        temperature_k=temperature,
        frequency_hz=frequency,
    )

    a_coeff = 3.397e-1 + (-5.216e14 + 7.039e30 / frequency) / frequency
    b_coeff = -4.116e3 + (1.067e19 + 8.135e34 / frequency) / frequency
    c_coeff = 5.081e8 + (-8.724e22 - 5.659e37 / frequency) / frequency
    absorption = (
        (
            a_coeff[None, :] * temperature[:, None]
            + b_coeff[None, :]
            + c_coeff[None, :] / temperature[:, None]
        )
        / 1.0e15
        * electron_density[:, None]
        / 1.0e15
        * helium_neutral[:, None]
        / 1.0e15
        / mass_density[:, None]
    )
    return absorption, planck_nu


def compute_helium_ionized_opacity_columns(
    atmosphere: ContinuumAtmosphereState,
    frequency_hz: np.ndarray,
    *,
    continuum_tables: ContinuumOpacityTables | None = None,
    karzas_tables: KarzasLatterTables | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return He II absorption and LTE source columns."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if frequency.ndim != 1:
        raise ValueError("frequency_hz must be one-dimensional")

    continuum_table = (
        load_continuum_opacity_tables()
        if continuum_tables is None
        else continuum_tables
    )
    karzas_table = (
        load_karzas_latter_tables() if karzas_tables is None else karzas_tables
    )
    temperature = np.asarray(atmosphere.temperature, dtype=np.float64)
    mass_density = np.maximum(
        np.asarray(atmosphere.mass_density, dtype=np.float64), 1.0e-300
    )
    electron_density = np.asarray(atmosphere.electron_density, dtype=np.float64)
    helium_singly_ionized_partition_normalized = np.asarray(
        atmosphere.helium_singly_ionized_partition_normalized_population,
        dtype=np.float64,
    )
    helium_doubly_ionized = np.asarray(
        atmosphere.ion_stage_populations_by_packed_slot[:, 4], dtype=np.float64
    )

    planck_nu, exp_hnu_over_kt, stimulated_emission = _planck_frequency_exact(
        temperature_k=temperature,
        frequency_hz=frequency,
    )
    thermal_energy_ev = np.maximum(temperature * BOLTZMANN_EV_PER_K_REFERENCE, 1.0e-300)
    boltzmann_population = np.zeros((temperature.size, 9), dtype=np.float64)
    for shell in range(1, 10):
        boltzmann_population[:, shell - 1] = (
            np.exp(-(54.403 - 54.403 / float(shell * shell)) / thermal_energy_ev)
            * 2.0
            * float(shell * shell)
            * helium_singly_ionized_partition_normalized
            / mass_density
        )

    freefree_density_factor = (
        electron_density
        * helium_doubly_ionized
        / mass_density
        / np.sqrt(np.maximum(temperature, 1.0e-300))
    )
    xr = (
        helium_singly_ionized_partition_normalized
        * (1.0 / 13.595)
        * thermal_energy_ev
        / mass_density
    )
    boltzmann_extension = np.exp(-53.859 / thermal_energy_ev) * xr
    series_limit_extension = np.exp(-54.403 / thermal_energy_ev) * xr
    coulomb_freefree = _coulomb_freefree_gaunt_grid(
        2,
        np.log(np.maximum(frequency, 1.0e-300)),
        np.log(np.maximum(temperature, 1.0e-300)),
        tables=continuum_table,
    )
    karzas_cross_sections = np.vstack(
        [
            _karzas_latter_cross_section_grid(
                frequency,
                effective_charge_squared=4.0,
                principal_quantum_number=shell,
                orbital_angular_momentum=shell,
                tables=karzas_table,
            )
            for shell in range(1, 10)
        ]
    )

    frequency_cubed = frequency * frequency * frequency
    freefree_coefficient = 3.6919e8 / frequency_cubed * 4.0
    extension_coefficient = 2.815e29 * 4.0 / frequency_cubed
    extension_population = np.empty(
        (temperature.size, frequency.size), dtype=np.float64
    )
    low_frequency = frequency < 1.31522e14
    if np.any(low_frequency):
        extension_population[:, low_frequency] = (
            series_limit_extension[:, None] / exp_hnu_over_kt[:, low_frequency]
        )
    if np.any(~low_frequency):
        extension_population[:, ~low_frequency] = boltzmann_extension[:, None]

    absorption = (
        (extension_population - series_limit_extension[:, None])
        * extension_coefficient[None, :]
        + boltzmann_population @ karzas_cross_sections
        + coulomb_freefree
        * freefree_coefficient[None, :]
        * freefree_density_factor[:, None]
    ) * stimulated_emission
    return absorption, planck_nu


def compute_carbon_neutral_opacity_columns(
    atmosphere: ContinuumAtmosphereState,
    frequency_hz: np.ndarray,
    *,
    karzas_tables: KarzasLatterTables | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return C I continuum absorption and its LTE source."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if frequency.ndim != 1:
        raise ValueError("frequency_hz must be one-dimensional")

    temperature = np.asarray(atmosphere.temperature, dtype=np.float64)
    mass_density = np.maximum(
        np.asarray(atmosphere.mass_density, dtype=np.float64), 1.0e-300
    )
    carbon_population = np.asarray(
        atmosphere.partition_normalized_populations_by_packed_slot[:, 20],
        dtype=np.float64,
    )
    planck_nu, _, stimulated_emission = _planck_frequency_exact(
        temperature_k=temperature,
        frequency_hz=frequency,
    )

    lyman_frequency_mask = frequency <= 3.28805e15
    absorption = np.zeros((temperature.size, frequency.size), dtype=np.float64)
    if not np.any(lyman_frequency_mask):
        return absorption, planck_nu

    rydberg_carbon = 109732.298
    level_energy_cm = np.array(
        [
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
        ],
        dtype=np.float64,
    )
    statistical_weight = np.array(
        [
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
        ],
        dtype=np.float64,
    )

    hc_over_kt = (
        PLANCK_ERG_SECOND_EXACT
        * LIGHT_SPEED_CM_PER_S_EXACT
        / np.maximum(BOLTZMANN_ERG_PER_K_EXACT * temperature, 1.0e-300)
    )
    boltzmann_weight = statistical_weight[:, None] * np.exp(
        -level_energy_cm[:, None] * hc_over_kt[None, :]
    )

    wavenumber = frequency / LIGHT_SPEED_CM_PER_S_EXACT
    cross_section_by_level = np.zeros((25, frequency.size), dtype=np.float64)
    ionization_limit_1 = 90862.70
    ionization_limit_2 = 90820.42
    ionization_limit_2b = ionization_limit_2 + 63.42
    ionization_limit_3 = ionization_limit_2 + 43003.3

    for level_index in range(14):
        threshold_cm = ionization_limit_1 - level_energy_cm[level_index]
        active = lyman_frequency_mask & (wavenumber >= threshold_cm)
        if not np.any(active):
            continue
        effective_charge_squared = 9.0 / rydberg_carbon * threshold_cm
        if level_index < 6:
            orbital_angular_momentum = 2
        elif level_index < 12:
            orbital_angular_momentum = 1
        else:
            orbital_angular_momentum = 0
        cross_section_by_level[level_index, active] = _karzas_latter_cross_section_grid(
            frequency[active],
            effective_charge_squared=effective_charge_squared,
            principal_quantum_number=3,
            orbital_angular_momentum=orbital_angular_momentum,
            tables=karzas_tables,
        )

    for ionization_limit, limit_weight in (
        (ionization_limit_2, 1.0 / 3.0),
        (ionization_limit_2b, 2.0 / 3.0),
    ):
        active_1s = lyman_frequency_mask & (
            wavenumber >= ionization_limit - level_energy_cm[14]
        )
        if np.any(active_1s):
            active_wavenumber = wavenumber[active_1s]
            background = 10.0 ** (
                -16.80
                - (active_wavenumber - ionization_limit + level_energy_cm[14])
                / 3.0
                / rydberg_carbon
            )
            resonance = (active_wavenumber - 97700.0) * 2.0 / 2743.0
            resonant_cross_section = (68.0e-18 * resonance + 118.0e-18) / (
                resonance**2 + 1.0
            )
            cross_section_by_level[14, active_1s] += (
                background + resonant_cross_section
            ) * limit_weight

        active_1d = lyman_frequency_mask & (
            wavenumber >= ionization_limit - level_energy_cm[15]
        )
        if np.any(active_1d):
            active_wavenumber = wavenumber[active_1d]
            background = 10.0 ** (
                -16.80
                - (active_wavenumber - ionization_limit + level_energy_cm[15])
                / 3.0
                / rydberg_carbon
            )
            resonance_1 = (active_wavenumber - 93917.0) * 2.0 / 9230.0
            resonant_cross_section_1 = (22.0e-18 * resonance_1 + 26.0e-18) / (
                resonance_1**2 + 1.0
            )
            resonance_2 = (active_wavenumber - 111130.0) * 2.0 / 2743.0
            resonant_cross_section_2 = (-10.5e-18 * resonance_2 + 46.0e-18) / (
                resonance_2**2 + 1.0
            )
            cross_section_by_level[15, active_1d] += (
                background + resonant_cross_section_1 + resonant_cross_section_2
            ) * limit_weight

        for level_index in range(16, 19):
            active = lyman_frequency_mask & (
                wavenumber >= ionization_limit - level_energy_cm[level_index]
            )
            if not np.any(active):
                continue
            cross_section_by_level[level_index, active] += (
                10.0
                ** (
                    -16.80
                    - (
                        wavenumber[active]
                        - ionization_limit
                        + level_energy_cm[level_index]
                    )
                    / 3.0
                    / rydberg_carbon
                )
                * limit_weight
            )

    for level_index in range(19, 25):
        threshold_cm = ionization_limit_3 - level_energy_cm[level_index]
        active = lyman_frequency_mask & (wavenumber >= threshold_cm)
        if not np.any(active):
            continue
        cross_section_by_level[level_index, active] = (
            _karzas_latter_cross_section_grid(
                frequency[active],
                effective_charge_squared=4.0 / rydberg_carbon * threshold_cm,
                principal_quantum_number=2,
                orbital_angular_momentum=1,
                tables=karzas_tables,
            )
            * 3.0
        )

    kramers_limit = ionization_limit_2
    frequency_cubed_factor = 2.815e29 / (frequency**3)
    kramers_lower = np.maximum(
        kramers_limit - rydberg_carbon / 16.0,
        kramers_limit - wavenumber,
    )
    freefree_profile = (
        frequency_cubed_factor[None, :]
        * 6.0
        / (rydberg_carbon * hc_over_kt[:, None])
        * (
            np.exp(-kramers_lower[None, :] * hc_over_kt[:, None])
            - np.exp(-kramers_limit * hc_over_kt)[:, None]
        )
    )
    branch_profile = freefree_profile + boltzmann_weight.T @ cross_section_by_level
    absorption[:, lyman_frequency_mask] = (
        branch_profile[:, lyman_frequency_mask]
        * stimulated_emission[:, lyman_frequency_mask]
        * carbon_population[:, None]
        / mass_density[:, None]
    )
    return absorption, planck_nu


def compute_magnesium_neutral_opacity_columns(
    atmosphere: ContinuumAtmosphereState,
    frequency_hz: np.ndarray,
    *,
    karzas_tables: KarzasLatterTables | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Mg I continuum absorption and its LTE source."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if frequency.ndim != 1:
        raise ValueError("frequency_hz must be one-dimensional")

    temperature = np.asarray(atmosphere.temperature, dtype=np.float64)
    mass_density = np.maximum(
        np.asarray(atmosphere.mass_density, dtype=np.float64), 1.0e-300
    )
    magnesium_population = np.asarray(
        atmosphere.partition_normalized_populations_by_packed_slot[:, 77],
        dtype=np.float64,
    )
    planck_nu, _, stimulated_emission = _planck_frequency_exact(
        temperature_k=temperature,
        frequency_hz=frequency,
    )

    lyman_frequency_mask = frequency <= 3.28805e15
    absorption = np.zeros((temperature.size, frequency.size), dtype=np.float64)
    if not np.any(lyman_frequency_mask):
        return absorption, planck_nu

    rydberg_magnesium = 109732.298
    ionization_limit = 61671.02
    level_energy_cm = np.array(
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
        ],
        dtype=np.float64,
    )
    statistical_weight = np.array(
        [21.0, 7.0, 15.0, 5.0, 3.0, 15.0, 9.0, 5.0, 1.0, 3.0, 3.0, 5.0, 3.0, 1.0, 1.0],
        dtype=np.float64,
    )

    hc_over_kt = (
        PLANCK_ERG_SECOND_EXACT
        * LIGHT_SPEED_CM_PER_S_EXACT
        / np.maximum(BOLTZMANN_ERG_PER_K_EXACT * temperature, 1.0e-300)
    )
    boltzmann_weight = statistical_weight[:, None] * np.exp(
        -level_energy_cm[:, None] * hc_over_kt[None, :]
    )

    wavenumber = frequency / LIGHT_SPEED_CM_PER_S_EXACT
    cross_section_by_level = np.zeros((15, frequency.size), dtype=np.float64)

    for level_index in range(2):
        threshold_cm = ionization_limit - level_energy_cm[level_index]
        active = lyman_frequency_mask & (wavenumber >= threshold_cm)
        if np.any(active):
            cross_section_by_level[level_index, active] = (
                _karzas_latter_cross_section_grid(
                    frequency[active],
                    effective_charge_squared=16.0 / rydberg_magnesium * threshold_cm,
                    principal_quantum_number=4,
                    orbital_angular_momentum=3,
                    tables=karzas_tables,
                )
            )

    for level_index in range(2, 4):
        threshold_cm = ionization_limit - level_energy_cm[level_index]
        active = lyman_frequency_mask & (wavenumber >= threshold_cm)
        if np.any(active):
            cross_section_by_level[level_index, active] = (
                _karzas_latter_cross_section_grid(
                    frequency[active],
                    effective_charge_squared=16.0 / rydberg_magnesium * threshold_cm,
                    principal_quantum_number=4,
                    orbital_angular_momentum=2,
                    tables=karzas_tables,
                )
            )

    active = lyman_frequency_mask & (
        wavenumber >= ionization_limit - level_energy_cm[4]
    )
    if np.any(active):
        cross_section_by_level[4, active] = _karzas_latter_cross_section_grid(
            frequency[active],
            effective_charge_squared=16.0
            / rydberg_magnesium
            * (ionization_limit - level_energy_cm[4]),
            principal_quantum_number=4,
            orbital_angular_momentum=1,
            tables=karzas_tables,
        )

    active = lyman_frequency_mask & (
        wavenumber >= ionization_limit - level_energy_cm[5]
    )
    if np.any(active):
        cross_section_by_level[5, active] = (
            25.0e-18 * (13713.986 / wavenumber[active]) ** 2.7
        )
    active = lyman_frequency_mask & (
        wavenumber >= ionization_limit - level_energy_cm[6]
    )
    if np.any(active):
        cross_section_by_level[6, active] = (
            33.8e-18 * (13823.223 / wavenumber[active]) ** 2.8
        )
    active = lyman_frequency_mask & (
        wavenumber >= ionization_limit - level_energy_cm[7]
    )
    if np.any(active):
        cross_section_by_level[7, active] = (
            45.0e-18 * (15267.955 / wavenumber[active]) ** 2.7
        )
    active = lyman_frequency_mask & (
        wavenumber >= ionization_limit - level_energy_cm[8]
    )
    if np.any(active):
        cross_section_by_level[8, active] = (
            0.43e-18 * (18167.687 / wavenumber[active]) ** 2.6
        )
    active = lyman_frequency_mask & (
        wavenumber >= ionization_limit - level_energy_cm[9]
    )
    if np.any(active):
        cross_section_by_level[9, active] = (
            2.1e-18 * (20473.617 / wavenumber[active]) ** 2.6
        )
    active = lyman_frequency_mask & (
        wavenumber >= ionization_limit - level_energy_cm[10]
    )
    if np.any(active):
        cross_section_by_level[10, active] = (
            16.0e-18 * (26619.756 / wavenumber[active]) ** 2.1
            - 7.8e-18 * (26619.756 / wavenumber[active]) ** 9.5
        )

    for level_index in range(11, 14):
        active = lyman_frequency_mask & (
            wavenumber >= ionization_limit - level_energy_cm[level_index]
        )
        if not np.any(active):
            continue
        shallow_power = 20.0e-18 * (39759.842 / wavenumber[active]) ** 2.7
        steep_power = 40.0e-18 * (39759.842 / wavenumber[active]) ** 14
        cross_section_by_level[level_index, active] = np.maximum(
            shallow_power, steep_power
        )

    active = lyman_frequency_mask & (
        wavenumber >= ionization_limit - level_energy_cm[14]
    )
    if np.any(active):
        cross_section_by_level[14, active] = (
            1.1e-18
            * ((ionization_limit - level_energy_cm[14]) / wavenumber[active]) ** 10
        )

    frequency_cubed_factor = 2.815e29 / (frequency**3)
    kramers_lower = np.maximum(
        ionization_limit - rydberg_magnesium / 25.0,
        ionization_limit - wavenumber,
    )
    freefree_profile = (
        frequency_cubed_factor[None, :]
        * 2.0
        / (rydberg_magnesium * hc_over_kt[:, None])
        * (
            np.exp(-kramers_lower[None, :] * hc_over_kt[:, None])
            - np.exp(-ionization_limit * hc_over_kt)[:, None]
        )
    )
    branch_profile = freefree_profile + boltzmann_weight.T @ cross_section_by_level
    absorption[:, lyman_frequency_mask] = (
        branch_profile[:, lyman_frequency_mask]
        * stimulated_emission[:, lyman_frequency_mask]
        * magnesium_population[:, None]
        / mass_density[:, None]
    )
    return absorption, planck_nu


def compute_silicon_neutral_opacity_columns(
    atmosphere: ContinuumAtmosphereState,
    frequency_hz: np.ndarray,
    *,
    karzas_tables: KarzasLatterTables | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Si I continuum absorption and its LTE source."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if frequency.ndim != 1:
        raise ValueError("frequency_hz must be one-dimensional")

    temperature = np.asarray(atmosphere.temperature, dtype=np.float64)
    mass_density = np.maximum(
        np.asarray(atmosphere.mass_density, dtype=np.float64), 1.0e-300
    )
    silicon_population = np.asarray(
        atmosphere.partition_normalized_populations_by_packed_slot[:, 104],
        dtype=np.float64,
    )
    planck_nu, _, stimulated_emission = _planck_frequency_exact(
        temperature_k=temperature,
        frequency_hz=frequency,
    )

    lyman_frequency_mask = frequency <= 3.28805e15
    absorption = np.zeros((temperature.size, frequency.size), dtype=np.float64)
    if not np.any(lyman_frequency_mask):
        return absorption, planck_nu

    rydberg_silicon = 109732.298
    level_energy_cm = np.array(
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
        ],
        dtype=np.float64,
    )
    statistical_weight = np.array(
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
        ],
        dtype=np.float64,
    )
    karzas_levels = [
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
    ]
    effective_charge_factors = np.array(
        [
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
        ],
        dtype=np.float64,
    )

    hc_over_kt = (
        PLANCK_ERG_SECOND_EXACT
        * LIGHT_SPEED_CM_PER_S_EXACT
        / np.maximum(BOLTZMANN_ERG_PER_K_EXACT * temperature, 1.0e-300)
    )
    boltzmann_weight = statistical_weight[:, None] * np.exp(
        -level_energy_cm[:, None] * hc_over_kt[None, :]
    )

    wavenumber = frequency / LIGHT_SPEED_CM_PER_S_EXACT
    cross_section_by_level = np.zeros((33, frequency.size), dtype=np.float64)

    ionization_limit_1 = 65939.18
    for level_index, (principal_quantum_number, orbital_angular_momentum) in enumerate(
        karzas_levels
    ):
        threshold_cm = ionization_limit_1 - level_energy_cm[level_index]
        active = lyman_frequency_mask & (wavenumber >= threshold_cm)
        if not np.any(active):
            continue
        cross_section_by_level[level_index, active] = _karzas_latter_cross_section_grid(
            frequency[active],
            effective_charge_squared=(
                effective_charge_factors[level_index] / rydberg_silicon * threshold_cm
            ),
            principal_quantum_number=principal_quantum_number,
            orbital_angular_momentum=orbital_angular_momentum,
            tables=karzas_tables,
        )

    for ionization_limit, limit_weight in (
        (65747.55, 1.0 / 3.0),
        (65747.55 + 287.45, 2.0 / 3.0),
    ):
        active_1s = lyman_frequency_mask & (
            wavenumber >= ionization_limit - level_energy_cm[22]
        )
        if np.any(active_1s):
            active_wavenumber = wavenumber[active_1s]
            resonance = (active_wavenumber - 70000.0) * 2.0 / 6500.0
            resonant_cross_section = (97.0e-18 * resonance + 94.0e-18) / (
                resonance**2 + 1.0
            )
            cross_section_by_level[22, active_1s] += (
                37.0e-18 * (50353.180 / active_wavenumber) ** 2.40
                + resonant_cross_section
            ) * limit_weight

        active_1d = lyman_frequency_mask & (
            wavenumber >= ionization_limit - level_energy_cm[23]
        )
        if np.any(active_1d):
            active_wavenumber = wavenumber[active_1d]
            resonance = (active_wavenumber - 78600.0) * 2.0 / 13000.0
            resonant_cross_section = (-10.0e-18 * resonance + 77.0e-18) / (
                resonance**2 + 1.0
            )
            cross_section_by_level[23, active_1d] += (
                24.5e-18 * (59448.700 / active_wavenumber) ** 1.85
                + resonant_cross_section
            ) * limit_weight

        for level_index in (24, 25, 26):
            active = lyman_frequency_mask & (
                wavenumber >= ionization_limit - level_energy_cm[level_index]
            )
            if not np.any(active):
                continue
            ratio = 65524.393 / wavenumber[active]
            effective_weight = (2.0 / 3.0) if level_index == 25 else limit_weight
            cross_section_by_level[level_index, active] += (
                np.where(
                    wavenumber[active] <= 74000.0,
                    72.0e-18 * ratio**1.90,
                    93.0e-18 * ratio**4.00,
                )
                * effective_weight
            )

    ionization_limit_3 = 65747.5 + 42824.35
    for level_index in range(27, 33):
        threshold_cm = ionization_limit_3 - level_energy_cm[level_index]
        active = lyman_frequency_mask & (wavenumber >= threshold_cm)
        if not np.any(active):
            continue
        cross_section_by_level[level_index, active] = (
            _karzas_latter_cross_section_grid(
                frequency[active],
                effective_charge_squared=9.0 / rydberg_silicon * threshold_cm,
                principal_quantum_number=3,
                orbital_angular_momentum=1,
                tables=karzas_tables,
            )
            * 3.0
        )

    freefree_limit = 65747.55
    frequency_cubed_factor = 2.815e29 / (frequency**3)
    kramers_lower = np.maximum(
        freefree_limit - rydberg_silicon / 25.0,
        freefree_limit - wavenumber,
    )
    freefree_profile = (
        frequency_cubed_factor[None, :]
        * 6.0
        / (rydberg_silicon * hc_over_kt[:, None])
        * (
            np.exp(-kramers_lower[None, :] * hc_over_kt[:, None])
            - np.exp(-freefree_limit * hc_over_kt)[:, None]
        )
    )
    branch_profile = freefree_profile + boltzmann_weight.T @ cross_section_by_level
    absorption[:, lyman_frequency_mask] = (
        branch_profile[:, lyman_frequency_mask]
        * stimulated_emission[:, lyman_frequency_mask]
        * silicon_population[:, None]
        / mass_density[:, None]
    )
    return absorption, planck_nu


@lru_cache(maxsize=1)
def _build_silicon_singly_ionized_lukewarm_table() -> np.ndarray:
    """Return the Si II photoionization cross-section table."""

    rydberg_silicon = 109732.298
    ion_charge = 2.0
    ionization_limit = np.array(
        [
            131838.4,
            131838.4,
            131838.4,
            131838.4,
            131838.4,
            131838.4,
            131838.4,
            131838.4,
            131838.4,
            131838.4,
            131838.4,
            184563.09,
            184563.09,
            184563.09,
            184563.09,
            184563.09,
            184563.09,
            184563.09,
            184563.09,
            184563.09,
            184563.09,
            184563.09,
            184563.09,
            184563.09,
            184563.09,
            184563.09,
            184563.09,
            184563.09,
            184563.09,
            184563.09,
            184563.09,
            184563.09,
            184563.09,
            184563.09,
            184563.09,
            184563.09,
            184563.09,
            184563.09,
            184563.09,
            184563.09,
            184563.09,
            254052.92,
            254052.92,
            254052.92,
            131838.4,
            184563.09,
        ],
        dtype=np.float64,
    )
    level_energy_cm = np.array(
        [
            114177.4,
            113760.48,
            112394.92,
            103877.34,
            97972.35,
            103556.36,
            101024.09,
            81231.57,
            79348.67,
            65500.73,
            191.55,
            157396.6,
            157188.8,
            156838.9,
            156836.9,
            155663.4,
            155593.7,
            155555.0,
            153523.1,
            153147.2,
            152977.0,
            152480.7,
            151245.1,
            149905.6,
            140696.0,
            134905.34,
            134136.03,
            132648.5,
            132012.27,
            131815.5,
            126250.9,
            124595.5,
            124373.8,
            121541.76,
            117058.95,
            114415.54,
            108804.1,
            83937.09,
            76665.61,
            55319.11,
            43002.27,
            143990.0,
            135300.5,
            123033.6,
            119645.92,
            167005.92,
        ],
        dtype=np.float64,
    )
    threshold_cm = np.array(
        [
            17661.0,
            18077.92,
            19443.48,
            27961.06,
            33866.05,
            28282.04,
            30814.31,
            50606.83,
            52489.73,
            66337.67,
            131646.85,
            27166.49,
            27374.29,
            27724.19,
            27726.19,
            28899.69,
            28969.39,
            29008.09,
            31039.99,
            31415.89,
            31586.09,
            32082.39,
            33317.99,
            34657.49,
            43867.09,
            49657.75,
            50427.06,
            51914.59,
            52550.82,
            52747.59,
            58312.19,
            59967.59,
            60189.29,
            63021.33,
            67504.14,
            70147.55,
            75758.99,
            100526.0,
            107897.48,
            129243.98,
            141560.82,
            110052.92,
            118752.42,
            131019.32,
            12192.48,
            17557.17,
        ],
        dtype=np.float64,
    )
    statistical_weight = np.array(
        [
            18.0,
            14.0,
            10.0,
            6.0,
            2.0,
            14.0,
            10.0,
            6.0,
            10.0,
            1.0,
            6.0,
            20.0,
            10.0,
            18.0,
            36.0,
            28.0,
            10.0,
            10.0,
            6.0,
            12.0,
            2.0,
            20.0,
            28.0,
            10.0,
            10.0,
            4.0,
            12.0,
            6.0,
            20.0,
            10.0,
            6.0,
            12.0,
            20.0,
            6.0,
            12.0,
            28.0,
            10.0,
            6.0,
            2.0,
            10.0,
            12.0,
            6.0,
            10.0,
            4.0,
            1.0,
            9.0,
        ],
        dtype=np.float64,
    )
    principal_quantum_number = np.array(
        [
            5,
            5,
            5,
            5,
            5,
            4,
            4,
            4,
            3,
            4,
            3,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            3,
            3,
            3,
            3,
            4,
            4,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            6,
            5,
        ],
        dtype=np.int64,
    )
    orbital_angular_momentum = np.array(
        [
            4,
            3,
            2,
            1,
            0,
            3,
            2,
            1,
            2,
            0,
            1,
            3,
            3,
            3,
            3,
            3,
            3,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            0,
            0,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
        ],
        dtype=np.int64,
    )
    effective_charge_squared = (
        principal_quantum_number[:44].astype(np.float64) ** 2
        / rydberg_silicon
        * threshold_cm[:44]
    )
    temperature_factor_table = np.empty(51, dtype=np.float64)
    ground_boltzmann = np.empty(51, dtype=np.float64)
    excited_boltzmann = np.empty(51, dtype=np.float64)
    level_boltzmann = np.empty((44, 51), dtype=np.float64)
    for temperature_index in range(51):
        table_temperature = 10.0 ** (3.48 + (temperature_index + 1) * 0.02)
        temperature_factor_table[temperature_index] = (
            PLANCK_ERG_SECOND_REFERENCE
            * LIGHT_SPEED_CM_PER_S_EXACT
            / BOLTZMANN_ERG_PER_K_REFERENCE
            / table_temperature
        )
        ground_boltzmann[temperature_index] = np.exp(
            -ionization_limit[0] * temperature_factor_table[temperature_index]
        )
        excited_boltzmann[temperature_index] = np.exp(
            -ionization_limit[11] * temperature_factor_table[temperature_index]
        )
        level_boltzmann[:, temperature_index] = statistical_weight[:44] * np.exp(
            -level_energy_cm[:44] * temperature_factor_table[temperature_index]
        )

    table = np.empty((200, 51), dtype=np.float64)
    for wavenumber_index in range(1, 201):
        table_wavenumber = wavenumber_index * 1000.0
        table_frequency = table_wavenumber * LIGHT_SPEED_CM_PER_S_EXACT
        frequency_cubed_factor = 2.815e29 / (table_frequency**3) * ion_charge**4
        cross_section = np.zeros(44, dtype=np.float64)
        for level_index in range(0, 11):
            if table_wavenumber < threshold_cm[level_index]:
                break
            cross_section[level_index] = _karzas_latter_cross_section_grid(
                np.array([table_frequency], dtype=np.float64),
                effective_charge_squared=effective_charge_squared[level_index],
                principal_quantum_number=int(principal_quantum_number[level_index]),
                orbital_angular_momentum=int(orbital_angular_momentum[level_index]),
            )[0]
        for level_index in range(11, 37):
            if table_wavenumber < threshold_cm[level_index]:
                break
            cross_section[level_index] = _karzas_latter_cross_section_grid(
                np.array([table_frequency], dtype=np.float64),
                effective_charge_squared=effective_charge_squared[level_index],
                principal_quantum_number=int(principal_quantum_number[level_index]),
                orbital_angular_momentum=int(orbital_angular_momentum[level_index]),
            )[0]
        for level_index in range(37, 41):
            if table_wavenumber < threshold_cm[level_index]:
                break
            cross_section[level_index] = (
                2.0
                * _karzas_latter_cross_section_grid(
                    np.array([table_frequency], dtype=np.float64),
                    effective_charge_squared=effective_charge_squared[level_index],
                    principal_quantum_number=int(principal_quantum_number[level_index]),
                    orbital_angular_momentum=int(orbital_angular_momentum[level_index]),
                )[0]
            )
        for level_index in range(41, 44):
            if table_wavenumber < threshold_cm[level_index]:
                break
            cross_section[level_index] = (
                3.0
                * _karzas_latter_cross_section_grid(
                    np.array([table_frequency], dtype=np.float64),
                    effective_charge_squared=effective_charge_squared[level_index],
                    principal_quantum_number=int(principal_quantum_number[level_index]),
                    orbital_angular_momentum=int(orbital_angular_momentum[level_index]),
                )[0]
            )
        for temperature_index in range(51):
            h_value = (
                frequency_cubed_factor
                * statistical_weight[44]
                / (
                    rydberg_silicon
                    * ion_charge**2
                    * temperature_factor_table[temperature_index]
                )
                * (
                    np.exp(
                        -max(
                            level_energy_cm[44], ionization_limit[44] - table_wavenumber
                        )
                        * temperature_factor_table[temperature_index]
                    )
                    - ground_boltzmann[temperature_index]
                )
            )
            h_value += (
                frequency_cubed_factor
                * statistical_weight[45]
                / (
                    rydberg_silicon
                    * ion_charge**2
                    * temperature_factor_table[temperature_index]
                )
                * (
                    np.exp(
                        -max(
                            level_energy_cm[45], ionization_limit[45] - table_wavenumber
                        )
                        * temperature_factor_table[temperature_index]
                    )
                    - excited_boltzmann[temperature_index]
                )
            )
            h_value += np.dot(cross_section, level_boltzmann[:, temperature_index])
            table[wavenumber_index - 1, temperature_index] = np.log(
                max(h_value, 1.0e-300)
            )
    return table


def _population_stage_or_zeros(
    partition_normalized_populations_by_packed_slot: np.ndarray,
    *,
    start_slot_1based: int,
    ion_stage_1based: int,
    layer_count: int,
) -> np.ndarray:
    slot_index = start_slot_1based + ion_stage_1based - 2
    if 0 <= slot_index < partition_normalized_populations_by_packed_slot.shape[1]:
        return np.asarray(
            partition_normalized_populations_by_packed_slot[:, slot_index],
            dtype=np.float64,
        )
    return np.zeros(layer_count, dtype=np.float64)


def compute_lukewarm_metal_opacity_columns(
    atmosphere: ContinuumAtmosphereState,
    frequency_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return intermediate-temperature metal absorption and its LTE source."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if frequency.ndim != 1:
        raise ValueError("frequency_hz must be one-dimensional")

    temperature = np.asarray(atmosphere.temperature, dtype=np.float64)
    mass_density = np.maximum(
        np.asarray(atmosphere.mass_density, dtype=np.float64), 1.0e-300
    )
    partition_normalized_population = np.asarray(
        atmosphere.partition_normalized_populations_by_packed_slot,
        dtype=np.float64,
    )
    layer_count = temperature.size
    planck_nu, exp_hnu_over_kt, stimulated_emission = _planck_frequency_exact(
        temperature_k=temperature,
        frequency_hz=frequency,
    )
    wavenumber = frequency / LIGHT_SPEED_CM_PER_S_EXACT
    hc_over_kt = (
        PLANCK_ERG_SECOND_EXACT
        * LIGHT_SPEED_CM_PER_S_EXACT
        / np.maximum(BOLTZMANN_ERG_PER_K_EXACT * temperature, 1.0e-300)
    )
    thermal_energy_ev = BOLTZMANN_EV_PER_K_REFERENCE * temperature

    nitrogen_population = _population_stage_or_zeros(
        partition_normalized_population,
        start_slot_1based=28,
        ion_stage_1based=1,
        layer_count=layer_count,
    )
    oxygen_population = _population_stage_or_zeros(
        partition_normalized_population,
        start_slot_1based=36,
        ion_stage_1based=1,
        layer_count=layer_count,
    )
    carbon_ionized_population = _population_stage_or_zeros(
        partition_normalized_population,
        start_slot_1based=21,
        ion_stage_1based=2,
        layer_count=layer_count,
    )
    magnesium_ionized_population = _population_stage_or_zeros(
        partition_normalized_population,
        start_slot_1based=78,
        ion_stage_1based=2,
        layer_count=layer_count,
    )
    silicon_singly_ionized_population = _population_stage_or_zeros(
        partition_normalized_population,
        start_slot_1based=105,
        ion_stage_1based=2,
        layer_count=layer_count,
    )
    calcium_ionized_population = _population_stage_or_zeros(
        partition_normalized_population,
        start_slot_1based=210,
        ion_stage_1based=2,
        layer_count=layer_count,
    )

    magnesium_energy_cm = np.array(
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
    magnesium_weight = np.array(
        [98.0, 72.0, 18.0, 14.0, 10.0, 6.0, 2.0, 14.0, 10.0, 6.0, 2.0, 10.0, 6.0, 2.0],
        dtype=np.float64,
    )
    magnesium_effective_charge_number = np.array(
        [49.0, 36.0, 25.0, 25.0, 25.0, 25.0, 25.0, 16.0, 16.0, 16.0, 16.0, 9.0, 9.0],
        dtype=np.float64,
    )
    magnesium_limit = 121267.61
    magnesium_rydberg = 109732.298
    magnesium_charge = 2.0
    magnesium_boltzmann = magnesium_weight[:, None] * np.exp(
        -magnesium_energy_cm[:, None] * hc_over_kt[None, :]
    )
    magnesium_limit_boltzmann = np.exp(-magnesium_limit * hc_over_kt)
    magnesium_kramers_threshold = (
        magnesium_limit - magnesium_rydberg * magnesium_charge** 2 / (8.0**2)
    )

    carbon_energy_cm = np.array(
        [
            179073.05,
            178955.94,
            178495.47,
            175292.30,
            173347.84,
            168978.34,
            168124.17,
            162522.34,
            157234.07,
            145550.1,
            131731.8,
            116537.65,
            42.28,
            202188.07,
            199965.31,
            198856.92,
            198431.96,
            196572.80,
            195786.71,
            190000.0,
            188601.54,
            186452.13,
            184690.98,
            182036.89,
            181741.65,
            177787.22,
            167009.29,
            110651.76,
            96493.74,
            74931.11,
            43035.8,
            230407.2,
            150464.6,
            142027.1,
        ],
        dtype=np.float64,
    )
    carbon_weight = np.array(
        [
            18.0,
            14.0,
            10.0,
            6.0,
            2.0,
            14.0,
            10.0,
            6.0,
            1.0,
            10.0,
            6.0,
            1.0,
            3.0,
            6.0,
            10.0,
            12.0,
            10.0,
            20.0,
            28.0,
            2.0,
            10.0,
            12.0,
            4.0,
            6.0,
            20.0,
            6.0,
            12.0,
            6.0,
            2.0,
            10.0,
            12.0,
            6.0,
            10.0,
            4.0,
        ],
        dtype=np.float64,
    )
    carbon_rydberg = 109732.298
    carbon_charge = 2.0
    carbon_frequency_factor = 2.815e29 * carbon_charge**4
    carbon_limit_1 = 196664.7
    carbon_limit_2 = carbon_limit_1 + 52367.06
    carbon_limit_3 = carbon_limit_1 + 137425.70
    carbon_boltzmann = carbon_weight[:, None] * np.exp(
        -carbon_energy_cm[:, None] * hc_over_kt[None, :]
    )
    carbon_boltzmann_1 = np.exp(-carbon_limit_1 * hc_over_kt)
    carbon_boltzmann_2 = np.exp(-carbon_limit_2 * hc_over_kt)
    carbon_kramers_1 = carbon_limit_1 - carbon_rydberg * carbon_charge**2 / (6.0**2)
    carbon_kramers_2 = carbon_limit_2 - carbon_rydberg * carbon_charge**2 / (4.0**2)

    temperature_log10 = (
        np.log(np.maximum(temperature, 1.0e-300)) / REFERENCE_NATURAL_LOG_10
    )
    silicon_temperature_index = np.clip(
        ((temperature_log10 - 3.48) / 0.02).astype(np.int64),
        1,
        50,
    )
    silicon_temperature_fraction = (
        temperature_log10 - 3.48 - silicon_temperature_index * 0.02
    ) / 0.02
    silicon_boltzmann_helper = (
        np.exp(-131838.4 * hc_over_kt) + 9.0 * np.exp(-184563.09 * hc_over_kt)
    ) / (109732.298 * 4.0 * hc_over_kt)
    silicon_singly_ionized_table = _build_silicon_singly_ionized_lukewarm_table()

    absorption = np.zeros((layer_count, frequency.size), dtype=np.float64)
    karzas_tables = load_karzas_latter_tables()
    _lukewarm_metal_absorption_kernel(
        absorption,
        frequency,
        wavenumber,
        stimulated_emission,
        exp_hnu_over_kt,
        hc_over_kt,
        thermal_energy_ev,
        mass_density,
        nitrogen_population,
        oxygen_population,
        carbon_ionized_population,
        magnesium_ionized_population,
        silicon_singly_ionized_population,
        calcium_ionized_population,
        magnesium_energy_cm,
        magnesium_boltzmann,
        magnesium_limit_boltzmann,
        _LUKEWARM_MAGNESIUM_PRINCIPAL,
        _LUKEWARM_MAGNESIUM_ANGULAR,
        magnesium_effective_charge_number,
        magnesium_limit,
        magnesium_rydberg,
        magnesium_charge**4,
        magnesium_rydberg * magnesium_charge**2,
        magnesium_kramers_threshold,
        carbon_energy_cm,
        carbon_boltzmann,
        carbon_boltzmann_1,
        carbon_boltzmann_2,
        _LUKEWARM_CARBON_PRINCIPAL,
        _LUKEWARM_CARBON_ANGULAR,
        carbon_rydberg,
        carbon_frequency_factor,
        carbon_rydberg * carbon_charge**2,
        carbon_limit_1,
        carbon_limit_2,
        carbon_limit_3,
        carbon_kramers_1,
        carbon_kramers_2,
        2.815e29 * (2.0**4),
        silicon_temperature_index,
        silicon_temperature_fraction,
        silicon_boltzmann_helper,
        silicon_singly_ionized_table,
        karzas_tables.karzas_latter_log10_frequency_hz,
        karzas_tables.karzas_latter_total_log10_cross_section_cm2,
        karzas_tables.karzas_latter_angular_log10_cross_section_cm2,
        karzas_tables.karzas_latter_high_level_energy_offset_rydberg,
    )
    return absorption, planck_nu


def _ch_molecular_cross_section_grid(
    frequency_hz: np.ndarray,
    temperature_k: np.ndarray,
    *,
    tables: ContinuumOpacityTables,
) -> np.ndarray:
    """Return CH cross-section times partition function."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    temperature = np.asarray(temperature_k, dtype=np.float64)
    opacity = np.zeros((temperature.size, frequency.size), dtype=np.float64)

    photon_energy_ev = (
        frequency / LIGHT_SPEED_CM_PER_S_EXACT / WAVENUMBER_PER_EV_REFERENCE
    )
    energy_index = np.asarray(photon_energy_ev * 10.0, dtype=np.int64)
    active_frequency = (energy_index >= 20) & (energy_index < 105)
    cool_layer = temperature < 9000.0
    if not np.any(active_frequency) or not np.any(cool_layer):
        return opacity

    idx = energy_index[active_frequency]
    energy_lower = idx.astype(np.float64) * 0.1
    energy_fraction = (photon_energy_ev[active_frequency] - energy_lower) / 0.1
    cross_section_log = (
        tables.ch_cross_section_table[idx, :]
        + (
            tables.ch_cross_section_table[idx + 1, :]
            - tables.ch_cross_section_table[idx, :]
        )
        * energy_fraction[:, None]
    )

    partition_index = np.clip(
        np.asarray((temperature - 1000.0) / 200.0, dtype=np.int64),
        0,
        39,
    )
    partition_lower_temperature = partition_index.astype(np.float64) * 200.0 + 1000.0
    partition_function = (
        tables.ch_partition_table[partition_index]
        + (
            tables.ch_partition_table[partition_index + 1]
            - tables.ch_partition_table[partition_index]
        )
        * (temperature - partition_lower_temperature)
        / 200.0
    )

    temperature_index = np.clip(
        np.asarray((temperature - 2000.0) / 500.0, dtype=np.int64),
        0,
        13,
    )
    temperature_lower = temperature_index.astype(np.float64) * 500.0 + 2000.0
    temperature_fraction = (temperature - temperature_lower) / 500.0
    interpolated_log10_cross_section_cm2 = (
        cross_section_log[:, temperature_index]
        + (
            cross_section_log[:, temperature_index + 1]
            - cross_section_log[:, temperature_index]
        )
        * temperature_fraction[None, :]
    )

    values = (
        np.exp(interpolated_log10_cross_section_cm2 * REFERENCE_NATURAL_LOG_10).T
        * partition_function[:, None]
    )
    opacity[:, active_frequency] = values
    opacity[~cool_layer, :] = 0.0
    return opacity


def _oh_molecular_cross_section_grid(
    frequency_hz: np.ndarray,
    temperature_k: np.ndarray,
    *,
    tables: ContinuumOpacityTables,
) -> np.ndarray:
    """Return OH cross-section times partition function."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    temperature = np.asarray(temperature_k, dtype=np.float64)
    opacity = np.zeros((temperature.size, frequency.size), dtype=np.float64)

    photon_energy_ev = (
        frequency / LIGHT_SPEED_CM_PER_S_EXACT / WAVENUMBER_PER_EV_REFERENCE
    )
    energy_index = np.asarray(photon_energy_ev * 10.0, dtype=np.int64) - 20
    active_frequency = (energy_index > 0) & (energy_index < 130)
    cool_layer = temperature < 9000.0
    if not np.any(active_frequency) or not np.any(cool_layer):
        return opacity

    idx = energy_index[active_frequency] - 1
    energy_lower = energy_index[active_frequency].astype(np.float64) * 0.1 + 2.0
    energy_fraction = (photon_energy_ev[active_frequency] - energy_lower) / 0.1
    cross_section_log = (
        tables.oh_cross_section_table[idx, :]
        + (
            tables.oh_cross_section_table[idx + 1, :]
            - tables.oh_cross_section_table[idx, :]
        )
        * energy_fraction[:, None]
    )

    partition_index = np.clip(
        np.asarray((temperature - 1000.0) / 200.0, dtype=np.int64),
        0,
        39,
    )
    partition_lower_temperature = partition_index.astype(np.float64) * 200.0 + 1000.0
    partition_function = (
        tables.oh_partition_table[partition_index]
        + (
            tables.oh_partition_table[partition_index + 1]
            - tables.oh_partition_table[partition_index]
        )
        * (temperature - partition_lower_temperature)
        / 200.0
    )

    temperature_index = np.clip(
        np.asarray((temperature - 2000.0) / 500.0, dtype=np.int64),
        0,
        13,
    )
    temperature_lower = temperature_index.astype(np.float64) * 500.0 + 2000.0
    temperature_fraction = (temperature - temperature_lower) / 500.0
    interpolated_log10_cross_section_cm2 = (
        cross_section_log[:, temperature_index]
        + (
            cross_section_log[:, temperature_index + 1]
            - cross_section_log[:, temperature_index]
        )
        * temperature_fraction[None, :]
    )

    values = (
        np.exp(interpolated_log10_cross_section_cm2 * REFERENCE_NATURAL_LOG_10).T
        * partition_function[:, None]
    )
    opacity[:, active_frequency] = values
    opacity[~cool_layer, :] = 0.0
    return opacity


def _h2_collision_absorption_grid(
    frequency_hz: np.ndarray,
    *,
    temperature_k: np.ndarray,
    hydrogen_neutral_partition_normalized_population: np.ndarray,
    hydrogen_departure_coefficient: np.ndarray,
    helium_neutral_population: np.ndarray,
    mass_density: np.ndarray,
    stimulated_emission: np.ndarray,
    continuum_tables: ContinuumOpacityTables,
    molecular_tables: MolecularEquilibriumTables | None = None,
) -> np.ndarray:
    """Return H2 collision-induced absorption."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    temperature = np.asarray(temperature_k, dtype=np.float64)
    absorption = np.zeros((temperature.size, frequency.size), dtype=np.float64)

    wavenumber = frequency / LIGHT_SPEED_CM_PER_S_EXACT
    active_frequency = wavenumber <= 20000.0
    if not np.any(active_frequency):
        return absorption

    molecular_hydrogen = compute_molecular_hydrogen_population(
        temperature_k=temperature,
        hydrogen_neutral_partition_normalized_population=hydrogen_neutral_partition_normalized_population,
        hydrogen_departure_coefficient=hydrogen_departure_coefficient,
        tables=molecular_tables,
    )
    molecular_hydrogen = np.where(temperature > 20000.0, 0.0, molecular_hydrogen)

    active_wavenumber = wavenumber[active_frequency]
    wavenumber_index = np.asarray(active_wavenumber / 250.0, dtype=np.int64)
    wavenumber_index = np.minimum(wavenumber_index, 79)
    wavenumber_fraction = (
        active_wavenumber - 250.0 * wavenumber_index.astype(np.float64)
    ) / 250.0
    idx0 = np.minimum(wavenumber_index, 80)
    idx1 = np.minimum(wavenumber_index + 1, 80)

    h2h2_by_temperature = (
        continuum_tables.hydrogen_molecule_h2_collision_table[idx0, :]
        * (1.0 - wavenumber_fraction[:, None])
        + continuum_tables.hydrogen_molecule_h2_collision_table[idx1, :]
        * wavenumber_fraction[:, None]
    )
    h2he_by_temperature = (
        continuum_tables.hydrogen_molecule_he_collision_table[idx0, :]
        * (1.0 - wavenumber_fraction[:, None])
        + continuum_tables.hydrogen_molecule_he_collision_table[idx1, :]
        * wavenumber_fraction[:, None]
    )

    temperature_index = np.asarray(temperature / 1000.0, dtype=np.int64)
    temperature_index = np.clip(temperature_index, 1, 6)
    temperature_fraction = np.clip(
        (temperature - 1000.0 * temperature_index.astype(np.float64)) / 1000.0,
        0.0,
        1.0,
    )
    active_indices = np.nonzero(active_frequency)[0]
    for layer_index in range(temperature.size):
        table_index = temperature_index[layer_index]
        h2h2_log = h2h2_by_temperature[:, table_index - 1] * temperature_fraction[
            layer_index
        ] + h2h2_by_temperature[:, table_index] * (
            1.0 - temperature_fraction[layer_index]
        )
        h2he_log = h2he_by_temperature[:, table_index - 1] * temperature_fraction[
            layer_index
        ] + h2he_by_temperature[:, table_index] * (
            1.0 - temperature_fraction[layer_index]
        )
        absorption[layer_index, active_indices] = (
            (
                10.0**h2he_log * helium_neutral_population[layer_index]
                + 10.0**h2h2_log * molecular_hydrogen[layer_index]
            )
            * molecular_hydrogen[layer_index]
            / mass_density[layer_index]
            * stimulated_emission[layer_index, active_indices]
        )
    return absorption


def compute_molecular_continuum_opacity_columns(
    atmosphere: ContinuumAtmosphereState,
    frequency_hz: np.ndarray,
    *,
    continuum_tables: ContinuumOpacityTables | None = None,
    molecular_tables: MolecularEquilibriumTables | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return molecular continuum absorption and its LTE source."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if frequency.ndim != 1:
        raise ValueError("frequency_hz must be one-dimensional")

    tables = (
        load_continuum_opacity_tables()
        if continuum_tables is None
        else continuum_tables
    )
    temperature = np.asarray(atmosphere.temperature, dtype=np.float64)
    mass_density = np.maximum(
        np.asarray(atmosphere.mass_density, dtype=np.float64), 1.0e-300
    )
    planck_nu, _, stimulated_emission = _planck_frequency_exact(
        temperature_k=temperature,
        frequency_hz=frequency,
    )

    absorption = np.zeros((temperature.size, frequency.size), dtype=np.float64)
    if np.min(temperature) < 9000.0:
        absorption += (
            _ch_molecular_cross_section_grid(frequency, temperature, tables=tables)
            * (np.asarray(atmosphere.ch_population, dtype=np.float64) / mass_density)[
                :, None
            ]
            * stimulated_emission
        )
        absorption += (
            _oh_molecular_cross_section_grid(frequency, temperature, tables=tables)
            * (np.asarray(atmosphere.oh_population, dtype=np.float64) / mass_density)[
                :, None
            ]
            * stimulated_emission
        )
        hydrogen_neutral_partition_normalized_population = (
            _hydrogen_neutral_partition_normalized_population_from_neutral(
                temperature_k=temperature,
                hydrogen_neutral_population=atmosphere.hydrogen_neutral_population,
            )
        )
        hydrogen_departure = np.asarray(
            atmosphere.hydrogen_departure_coefficients[:, 0],
            dtype=np.float64,
        )
        absorption += _h2_collision_absorption_grid(
            frequency,
            temperature_k=temperature,
            hydrogen_neutral_partition_normalized_population=hydrogen_neutral_partition_normalized_population,
            hydrogen_departure_coefficient=hydrogen_departure,
            helium_neutral_population=np.asarray(
                atmosphere.helium_neutral_population,
                dtype=np.float64,
            ),
            mass_density=mass_density,
            stimulated_emission=stimulated_emission,
            continuum_tables=tables,
            molecular_tables=molecular_tables,
        )

    return absorption, planck_nu


def _copy_partition_normalized_population_block(
    destination: np.ndarray,
    *,
    destination_start: int,
    source: np.ndarray,
    source_start_1based: int,
    ion_count: int,
) -> None:
    """Copy a packed ion block using the validated hot-metal slot convention."""

    source_start = source_start_1based - 1
    source_stop = min(source_start + ion_count, source.shape[1])
    count = max(0, source_stop - source_start)
    if count:
        destination[:, destination_start : destination_start + count] = source[
            :,
            source_start:source_stop,
        ]


def compute_hot_metal_opacity_columns(
    atmosphere: ContinuumAtmosphereState,
    frequency_hz: np.ndarray,
    *,
    continuum_tables: ContinuumOpacityTables | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return hot-star metal continuum absorption and its LTE source."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if frequency.ndim != 1:
        raise ValueError("frequency_hz must be one-dimensional")

    tables = (
        load_continuum_opacity_tables()
        if continuum_tables is None
        else continuum_tables
    )
    temperature = np.asarray(atmosphere.temperature, dtype=np.float64)
    electron_density = np.asarray(atmosphere.electron_density, dtype=np.float64)
    mass_density = np.maximum(
        np.asarray(atmosphere.mass_density, dtype=np.float64), 1.0e-300
    )
    partition_normalized_population = np.asarray(
        atmosphere.partition_normalized_populations_by_packed_slot,
        dtype=np.float64,
    )
    ion_stage_populations_by_packed_slot = np.asarray(
        atmosphere.ion_stage_populations_by_packed_slot, dtype=np.float64
    )

    planck_nu, _, stimulated_emission = _planck_frequency_exact(
        temperature_k=temperature,
        frequency_hz=frequency,
    )
    temperature_log = np.log(np.maximum(temperature, 1.0e-10))
    thermal_energy_ev = BOLTZMANN_EV_PER_K_REFERENCE * temperature
    hot_partition_normalized_population = np.zeros(
        (temperature.size, 21), dtype=np.float64
    )
    charge_square_population_sum = np.zeros((temperature.size, 5), dtype=np.float64)

    _copy_partition_normalized_population_block(
        hot_partition_normalized_population,
        destination_start=0,
        source=partition_normalized_population,
        source_start_1based=21,
        ion_count=4,
    )
    _copy_partition_normalized_population_block(
        hot_partition_normalized_population,
        destination_start=4,
        source=partition_normalized_population,
        source_start_1based=28,
        ion_count=5,
    )
    _copy_partition_normalized_population_block(
        hot_partition_normalized_population,
        destination_start=9,
        source=partition_normalized_population,
        source_start_1based=36,
        ion_count=6,
    )
    _copy_partition_normalized_population_block(
        hot_partition_normalized_population,
        destination_start=15,
        source=partition_normalized_population,
        source_start_1based=55,
        ion_count=6,
    )

    for source_start_1based in (21, 28, 36, 55, 78, 105, 136, 351):
        for ion_charge in range(1, 6):
            source_index = source_start_1based + ion_charge - 1
            if 0 <= source_index < ion_stage_populations_by_packed_slot.shape[1]:
                charge_square_population_sum[:, ion_charge - 1] += (
                    ion_charge * ion_charge
                ) * ion_stage_populations_by_packed_slot[:, source_index]

    absorption = np.zeros((temperature.size, frequency.size), dtype=np.float64)
    for frequency_start in range(0, frequency.size, 4096):
        frequency_stop = min(frequency_start + 4096, frequency.size)
        frequency_chunk = frequency[frequency_start:frequency_stop]
        freefree_sum = np.zeros(
            (temperature.size, frequency_chunk.size), dtype=np.float64
        )
        for ion_charge in range(1, 6):
            freefree_sum += (
                _coulomb_freefree_gaunt_grid(
                    ion_charge,
                    np.log(frequency_chunk),
                    temperature_log,
                    tables=tables,
                )
                * charge_square_population_sum[:, ion_charge - 1][:, None]
            )

        opacity_chunk = (
            freefree_sum
            * (3.6919e8 / (frequency_chunk[None, :] ** 3))
            * (
                electron_density[:, None]
                / np.sqrt(np.maximum(temperature, 1.0e-30))[:, None]
            )
        )

        for transition in tables.hot_metal_boundfree_transition_table:
            threshold_frequency = transition[0]
            active = frequency_chunk >= threshold_frequency
            if not np.any(active):
                continue
            cross_section = transition[1]
            alpha = transition[2]
            power = transition[3]
            multiplier = transition[4]
            excitation_energy_ev = transition[5]
            population_index = int(np.clip(int(transition[6]) - 1, 0, 20))
            ratio = threshold_frequency / frequency_chunk[active]
            transition_cross_section = (
                cross_section
                * (alpha + ratio - alpha * ratio)
                * np.sqrt(ratio ** int(power))
            )
            weighted_cross_section = (
                transition_cross_section[None, :]
                * hot_partition_normalized_population[:, population_index][:, None]
                * multiplier
            )
            threshold = opacity_chunk[:, active] / 100.0
            opacity_chunk[:, active] += np.where(
                weighted_cross_section > threshold,
                weighted_cross_section
                * np.exp(
                    -excitation_energy_ev / np.maximum(thermal_energy_ev, 1.0e-30)
                )[:, None],
                0.0,
            )

        absorption[:, frequency_start:frequency_stop] = (
            opacity_chunk
            * stimulated_emission[:, frequency_start:frequency_stop]
            / mass_density[:, None]
        )
    return absorption, planck_nu


def compute_aluminum_neutral_opacity_columns(
    atmosphere: ContinuumAtmosphereState,
    frequency_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Al I continuum absorption and its LTE source."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if frequency.ndim != 1:
        raise ValueError("frequency_hz must be one-dimensional")

    temperature = np.asarray(atmosphere.temperature, dtype=np.float64)
    mass_density = np.maximum(
        np.asarray(atmosphere.mass_density, dtype=np.float64), 1.0e-300
    )
    aluminum_population = np.asarray(
        atmosphere.partition_normalized_populations_by_packed_slot[:, 90],
        dtype=np.float64,
    )
    planck_nu, _, stimulated_emission = _planck_frequency_exact(
        temperature_k=temperature,
        frequency_hz=frequency,
    )
    wavenumber = frequency / LIGHT_SPEED_CM_PER_S_EXACT
    branch_cross_section = np.zeros(frequency.size, dtype=np.float64)
    ionization_limit = 48278.37

    active = frequency <= 3.28805e15
    upper_edge = active & (wavenumber >= ionization_limit - 112.061)
    branch_cross_section[upper_edge] = (
        6.5e-17 * ((ionization_limit - 112.061) / wavenumber[upper_edge]) ** 5 * 4.0
    )
    lower_edge = active & (wavenumber >= ionization_limit)
    branch_cross_section[lower_edge] += (
        6.5e-17 * (ionization_limit / wavenumber[lower_edge]) ** 5 * 2.0
    )

    absorption = (
        aluminum_population[:, None]
        * stimulated_emission
        / mass_density[:, None]
        * branch_cross_section[None, :]
    )
    return absorption, planck_nu


def compute_iron_neutral_opacity_columns(
    atmosphere: ContinuumAtmosphereState,
    frequency_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Fe I continuum absorption and its LTE source."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if frequency.ndim != 1:
        raise ValueError("frequency_hz must be one-dimensional")

    temperature = np.asarray(atmosphere.temperature, dtype=np.float64)
    mass_density = np.maximum(
        np.asarray(atmosphere.mass_density, dtype=np.float64), 1.0e-300
    )
    iron_population = np.asarray(
        atmosphere.partition_normalized_populations_by_packed_slot[:, 350],
        dtype=np.float64,
    )
    planck_nu, _, stimulated_emission = _planck_frequency_exact(
        temperature_k=temperature,
        frequency_hz=frequency,
    )

    wavenumber = frequency / LIGHT_SPEED_CM_PER_S_EXACT
    active_frequency = wavenumber >= 21000.0
    absorption = np.zeros((temperature.size, frequency.size), dtype=np.float64)
    if not np.any(active_frequency):
        return absorption, planck_nu

    transition_weight = np.array(
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
        ],
        dtype=np.float64,
    )
    transition_energy_cm = np.array(
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
        ],
        dtype=np.float64,
    )
    transition_threshold_cm = np.array(
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
        ],
        dtype=np.float64,
    )

    hc_over_kt = (
        PLANCK_ERG_SECOND_EXACT
        * LIGHT_SPEED_CM_PER_S_EXACT
        / np.maximum(BOLTZMANN_ERG_PER_K_EXACT * temperature, 1.0e-300)
    )
    if _NUMBA_AVAILABLE:
        # Compiled prange kernel over frequencies: each frequency accumulates its
        # own column over the 48 Fe I branches (disjoint scatter). Reproduces the
        # per-transition numpy accumulation element-for-element.
        branch_profile = _iron_neutral_branch_kernel(
            np.ascontiguousarray(wavenumber),
            np.ascontiguousarray(hc_over_kt),
            transition_weight,
            transition_energy_cm,
            transition_threshold_cm,
        )
    else:
        branch_profile = np.zeros((temperature.size, frequency.size), dtype=np.float64)
        for weight, energy_cm, threshold_cm in zip(
            transition_weight,
            transition_energy_cm,
            transition_threshold_cm,
        ):
            use_frequency = wavenumber >= threshold_cm
            if not np.any(use_frequency):
                continue
            cross_section = np.zeros(frequency.size, dtype=np.float64)
            cross_section[use_frequency] = 3.0e-18 / (
                1.0
                + (
                    (threshold_cm + 3000.0 - wavenumber[use_frequency])
                    / threshold_cm
                    / 0.1
                )
                ** 4
            )
            branch_profile += (
                weight
                * np.exp(-energy_cm * hc_over_kt)[:, None]
                * cross_section[None, :]
            )

    absorption[:, active_frequency] = (
        branch_profile[:, active_frequency]
        * stimulated_emission[:, active_frequency]
        * iron_population[:, None]
        / mass_density[:, None]
    )
    return absorption, planck_nu


def create_rosseland_opacity_table(
    layer_count: int,
    *,
    entries_per_layer: int = 60,
) -> RosselandOpacityTable:
    """Return an empty ROSSTAB-style table for Rosseland opacity lookups."""

    entry_capacity = max(1, int(layer_count) * int(entries_per_layer))
    return RosselandOpacityTable(
        normalized_log_temperature=np.zeros(entry_capacity, dtype=np.float64),
        normalized_log_pressure=np.zeros(entry_capacity, dtype=np.float64),
        log10_rosseland_opacity=np.zeros(entry_capacity, dtype=np.float64),
        entry_count=0,
        log_temperature_origin=0.0,
        log_pressure_origin=0.0,
        log_temperature_span=1.0,
        log_pressure_span=1.0,
    )


def ingest_rosseland_opacity_table(
    table: RosselandOpacityTable,
    *,
    temperature_k: np.ndarray,
    gas_pressure: np.ndarray,
    rosseland_opacity: np.ndarray,
) -> None:
    """Append one atmosphere column to the ROSSTAB-style lookup table."""

    temperature = np.asarray(temperature_k, dtype=np.float64)
    pressure = np.asarray(gas_pressure, dtype=np.float64)
    opacity = np.asarray(rosseland_opacity, dtype=np.float64)
    if temperature.shape != pressure.shape or temperature.shape != opacity.shape:
        raise ValueError(
            "temperature_k, gas_pressure, and rosseland_opacity must match"
        )

    if table.entry_count == 0:
        table.log_temperature_origin = np.log10(max(float(temperature[0]), 1.0e-300))
        table.log_pressure_origin = np.log10(max(float(pressure[0]), 1.0e-300))
        table.log_temperature_span = (
            np.log10(max(float(temperature[-1]), 1.0e-300))
            - table.log_temperature_origin
        )
        table.log_pressure_span = (
            np.log10(max(float(pressure[-1]), 1.0e-300)) - table.log_pressure_origin
        )
        if abs(table.log_temperature_span) < 1.0e-300:
            table.log_temperature_span = 1.0
        if abs(table.log_pressure_span) < 1.0e-300:
            table.log_pressure_span = 1.0

    for layer_index in range(temperature.size):
        if table.entry_count >= table.normalized_log_temperature.size:
            break
        table.normalized_log_temperature[table.entry_count] = (
            np.log10(max(float(temperature[layer_index]), 1.0e-300))
            - table.log_temperature_origin
        ) / table.log_temperature_span
        table.normalized_log_pressure[table.entry_count] = (
            np.log10(max(float(pressure[layer_index]), 1.0e-300))
            - table.log_pressure_origin
        ) / table.log_pressure_span
        table.log10_rosseland_opacity[table.entry_count] = np.log10(
            max(float(opacity[layer_index]), 1.0e-300)
        )
        table.entry_count += 1


def evaluate_rosseland_opacity(
    table: RosselandOpacityTable,
    *,
    temperature_k: float,
    gas_pressure: float,
) -> float:
    """Evaluate nearest-quadrant Rosseland-opacity interpolation."""

    return float(
        _evaluate_rosseland_opacity_kernel(
            table.normalized_log_temperature,
            table.normalized_log_pressure,
            table.log10_rosseland_opacity,
            int(table.entry_count),
            float(table.log_temperature_origin),
            float(table.log_pressure_origin),
            float(table.log_temperature_span),
            float(table.log_pressure_span),
            float(temperature_k),
            float(gas_pressure),
        )
    )


def compute_rosseland_continuum_opacity_columns(
    atmosphere: ContinuumAtmosphereState,
    frequency_hz: np.ndarray,
    *,
    rosseland_table: RosselandOpacityTable,
) -> tuple[np.ndarray, np.ndarray]:
    """Return interpolated continuum absorption and thermal source columns."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if frequency.ndim != 1:
        raise ValueError("frequency_hz must be one-dimensional")

    temperature = np.asarray(atmosphere.temperature, dtype=np.float64)
    pressure = np.asarray(atmosphere.gas_pressure, dtype=np.float64)
    opacity = np.array(
        [
            evaluate_rosseland_opacity(
                rosseland_table,
                temperature_k=float(layer_temperature),
                gas_pressure=float(layer_pressure),
            )
            for layer_temperature, layer_pressure in zip(temperature, pressure)
        ],
        dtype=np.float64,
    )
    source = 5.667e-5 / 12.5664 * temperature**4 * 4.0
    return (
        np.broadcast_to(opacity[:, None], (temperature.size, frequency.size)).copy(),
        np.broadcast_to(source[:, None], (temperature.size, frequency.size)).copy(),
    )


def compute_continuum_opacity_columns(
    atmosphere: ContinuumAtmosphereState,
    frequency_hz: np.ndarray,
    *,
    opacity_flags: list[int] | tuple[int, ...] | None = None,
    rosseland_table: RosselandOpacityTable | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return continuum absorption, scattering, and thermal-source columns."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if frequency.ndim != 1:
        raise ValueError("frequency_hz must be one-dimensional")

    flags = (
        [1] * 20 if opacity_flags is None else [int(value) for value in opacity_flags]
    )
    if len(flags) < 20:
        flags.extend([0] * (20 - len(flags)))

    planck_nu, _, _ = _planck_frequency_exact(
        temperature_k=atmosphere.temperature,
        frequency_hz=frequency,
    )
    layer_count = atmosphere.layers
    absorption = np.zeros((layer_count, frequency.size), dtype=np.float64)
    source_numerator = np.zeros_like(absorption)

    if flags[0] == 1:
        hydrogen_absorption, hydrogen_source = compute_hydrogen_opacity_columns(
            atmosphere,
            frequency,
        )
        absorption += hydrogen_absorption
        source_numerator += hydrogen_absorption * hydrogen_source

    if flags[2] == 1:
        hminus_absorption, hminus_source = compute_hminus_opacity_columns(
            atmosphere,
            frequency,
        )
        absorption += hminus_absorption
        source_numerator += hminus_absorption * hminus_source

    thermal_absorption = np.zeros_like(absorption)
    if flags[1] == 1:
        h2plus_absorption, _ = compute_molecular_hydrogen_ion_opacity_columns(
            atmosphere,
            frequency,
        )
        thermal_absorption += h2plus_absorption
    if flags[4] == 1:
        helium_neutral_absorption, _ = compute_helium_neutral_opacity_columns(
            atmosphere,
            frequency,
        )
        thermal_absorption += helium_neutral_absorption
    if flags[5] == 1:
        helium_ionized_absorption, _ = compute_helium_ionized_opacity_columns(
            atmosphere,
            frequency,
        )
        thermal_absorption += helium_ionized_absorption
    if flags[6] == 1:
        heminus_absorption, _ = compute_heminus_opacity_columns(
            atmosphere,
            frequency,
        )
        thermal_absorption += heminus_absorption
    if flags[8] == 1:
        molecular_absorption, _ = compute_molecular_continuum_opacity_columns(
            atmosphere,
            frequency,
        )
        carbon_absorption, _ = compute_carbon_neutral_opacity_columns(
            atmosphere,
            frequency,
        )
        magnesium_absorption, _ = compute_magnesium_neutral_opacity_columns(
            atmosphere,
            frequency,
        )
        aluminum_absorption, _ = compute_aluminum_neutral_opacity_columns(
            atmosphere,
            frequency,
        )
        silicon_absorption, _ = compute_silicon_neutral_opacity_columns(
            atmosphere,
            frequency,
        )
        iron_absorption, _ = compute_iron_neutral_opacity_columns(
            atmosphere,
            frequency,
        )
        thermal_absorption += (
            molecular_absorption
            + carbon_absorption
            + magnesium_absorption
            + aluminum_absorption
            + silicon_absorption
            + iron_absorption
        )
    if flags[9] == 1:
        lukewarm_absorption, _ = compute_lukewarm_metal_opacity_columns(
            atmosphere,
            frequency,
        )
        thermal_absorption += lukewarm_absorption
    if flags[10] == 1:
        hot_absorption, _ = compute_hot_metal_opacity_columns(
            atmosphere,
            frequency,
        )
        thermal_absorption += hot_absorption

    absorption += thermal_absorption
    source_numerator += thermal_absorption * planck_nu

    if flags[18] == 1 and rosseland_table is not None:
        rosseland_absorption, rosseland_source = (
            compute_rosseland_continuum_opacity_columns(
                atmosphere,
                frequency,
                rosseland_table=rosseland_table,
            )
        )
        absorption += rosseland_absorption
        source_numerator += rosseland_absorption * rosseland_source

    source = planck_nu.copy()
    active = absorption > 0.0
    source[active] = source_numerator[active] / absorption[active]

    scattering = compute_continuum_scattering_columns(
        atmosphere,
        frequency,
        opacity_flags=flags,
    )
    return absorption, scattering, source


def compute_light_element_continuum_columns(
    atmosphere: ContinuumAtmosphereState,
    frequency_hz: np.ndarray,
    *,
    opacity_flags: list[int] | tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the assembled continuum for the H and He branches."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if frequency.ndim != 1:
        raise ValueError("frequency_hz must be one-dimensional")

    flags = (
        [1] * 20 if opacity_flags is None else [int(value) for value in opacity_flags]
    )
    if len(flags) < 20:
        flags.extend([0] * (20 - len(flags)))

    planck_nu, _, _ = _planck_frequency_exact(
        temperature_k=atmosphere.temperature,
        frequency_hz=frequency,
    )
    layer_count = atmosphere.layers
    absorption = np.zeros((layer_count, frequency.size), dtype=np.float64)
    source_numerator = np.zeros_like(absorption)

    if flags[0] == 1:
        hydrogen_absorption, hydrogen_source = compute_hydrogen_opacity_columns(
            atmosphere,
            frequency,
        )
        absorption += hydrogen_absorption
        source_numerator += hydrogen_absorption * hydrogen_source

    if flags[2] == 1:
        hminus_absorption, hminus_source = compute_hminus_opacity_columns(
            atmosphere,
            frequency,
        )
        absorption += hminus_absorption
        source_numerator += hminus_absorption * hminus_source

    thermal_absorption = np.zeros_like(absorption)
    if flags[1] == 1:
        h2plus_absorption, _ = compute_molecular_hydrogen_ion_opacity_columns(
            atmosphere,
            frequency,
        )
        thermal_absorption += h2plus_absorption
    if flags[4] == 1:
        helium_neutral_absorption, _ = compute_helium_neutral_opacity_columns(
            atmosphere,
            frequency,
        )
        thermal_absorption += helium_neutral_absorption
    if flags[5] == 1:
        helium_ionized_absorption, _ = compute_helium_ionized_opacity_columns(
            atmosphere,
            frequency,
        )
        thermal_absorption += helium_ionized_absorption
    if flags[6] == 1:
        heminus_absorption, _ = compute_heminus_opacity_columns(
            atmosphere,
            frequency,
        )
        thermal_absorption += heminus_absorption

    absorption += thermal_absorption
    source_numerator += thermal_absorption * planck_nu
    source = planck_nu.copy()
    active = absorption > 0.0
    source[active] = source_numerator[active] / absorption[active]

    scattering = compute_continuum_scattering_columns(
        atmosphere,
        frequency,
        opacity_flags=flags,
    )
    return absorption, scattering, source


def compute_continuum_scattering_columns(
    atmosphere: ContinuumAtmosphereState,
    frequency_hz: np.ndarray,
    *,
    opacity_flags: list[int] | tuple[int, ...] | None = None,
    molecular_tables: MolecularEquilibriumTables | None = None,
) -> np.ndarray:
    """Return continuum-scattering columns for the requested frequency grid."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if frequency.ndim != 1:
        raise ValueError("frequency_hz must be one-dimensional")

    flags = (
        [1] * 20 if opacity_flags is None else [int(value) for value in opacity_flags]
    )
    if len(flags) < 13:
        flags.extend([0] * (13 - len(flags)))

    temperature = np.asarray(atmosphere.temperature, dtype=np.float64)
    mass_density = np.maximum(
        np.asarray(atmosphere.mass_density, dtype=np.float64), 1.0e-300
    )
    electron_density = np.asarray(atmosphere.electron_density, dtype=np.float64)
    layer_count = temperature.size
    scattering = np.zeros((layer_count, frequency.size), dtype=np.float64)

    if flags[11] == 1:
        scattering += (0.6653e-24 * electron_density / mass_density)[:, None]

    hydrogen_neutral_partition_normalized_population: np.ndarray | None = None
    if flags[3] == 1:
        hydrogen_neutral_partition_normalized_population = (
            _hydrogen_neutral_partition_normalized_population_from_neutral(
                temperature_k=temperature,
                hydrogen_neutral_population=atmosphere.hydrogen_neutral_population,
            )
        )
        hydrogen_departure = np.asarray(
            atmosphere.hydrogen_departure_coefficients[:, 0],
            dtype=np.float64,
        )
        population_over_density = (
            hydrogen_neutral_partition_normalized_population
            * 2.0
            * hydrogen_departure
            / mass_density
        )
        wavelength_angstrom = LIGHT_SPEED_ANGSTROM_PER_S / np.minimum(
            frequency, 2.463e15
        )
        wavelength_squared = wavelength_angstrom * wavelength_angstrom
        cross_section = (
            5.799e-13
            + 1.422e-6 / wavelength_squared
            + 2.784 / (wavelength_squared * wavelength_squared)
        ) / (wavelength_squared * wavelength_squared)
        scattering += population_over_density[:, None] * cross_section[None, :]

    if flags[7] == 1:
        helium_neutral = np.asarray(
            atmosphere.helium_neutral_population, dtype=np.float64
        )
        wave = LIGHT_SPEED_ANGSTROM_PER_S / np.minimum(frequency, 5.15e15)
        wave_squared = wave * wave
        cross_section = (
            5.484e-14
            / (wave_squared * wave_squared)
            * (
                1.0
                + (2.44e5 + 5.94e10 / np.maximum(wave_squared - 2.90e5, 1.0e-10))
                / wave_squared
            )
            ** 2
        )
        scattering += (helium_neutral / mass_density)[:, None] * cross_section[None, :]

    if flags[12] == 1 and hydrogen_neutral_partition_normalized_population is not None:
        hydrogen_departure = np.asarray(
            atmosphere.hydrogen_departure_coefficients[:, 0],
            dtype=np.float64,
        )
        molecular_hydrogen = compute_molecular_hydrogen_population(
            temperature_k=temperature,
            hydrogen_neutral_partition_normalized_population=hydrogen_neutral_partition_normalized_population,
            hydrogen_departure_coefficient=hydrogen_departure,
            tables=molecular_tables,
        )
        molecular_hydrogen = np.where(temperature > 20000.0, 0.0, molecular_hydrogen)
        wave = LIGHT_SPEED_ANGSTROM_PER_S / np.minimum(frequency, 2.922e15)
        wave_squared = wave * wave
        cross_section = (
            8.14e-13 + 1.28e-6 / wave_squared + 1.61 / (wave_squared * wave_squared)
        ) / (wave_squared * wave_squared)
        scattering += (molecular_hydrogen / mass_density)[:, None] * cross_section[
            None, :
        ]

    return scattering


@lru_cache(maxsize=16)
def build_opacity_sampling_grid(
    effective_temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return opacity-sampling wavelengths and frequency quadrature weights."""

    grid_size = 30000
    carbon_edge_start = 11601
    lyman_edge_start = 9599
    helium_neutral_edge_start = 7027
    helium_ionized_edge_start = 3577

    start_index = 1
    temperature = float(effective_temperature)
    if temperature < 30000.0:
        start_index = helium_ionized_edge_start
    if temperature < 13000.0:
        start_index = helium_neutral_edge_start
    if temperature < 7250.0:
        start_index = lyman_edge_start
    if temperature < 4500.0:
        start_index = carbon_edge_start

    one_based_index = np.arange(1, grid_size + 1, dtype=np.float64)
    wavelength_nm = 10.0 ** (1.0 + 0.0001 * (one_based_index + start_index - 1.0))
    frequency_weights = np.zeros(grid_size, dtype=np.float64)

    frequency_weights[0] = (
        LIGHT_SPEED_NM_PER_S / wavelength_nm[0]
        - LIGHT_SPEED_NM_PER_S / wavelength_nm[1]
    ) * 1.5
    frequency_weights[1:-1] = (
        LIGHT_SPEED_NM_PER_S / wavelength_nm[:-2]
        - LIGHT_SPEED_NM_PER_S / wavelength_nm[2:]
    ) * 0.5
    frequency_weights[-1] = (
        LIGHT_SPEED_NM_PER_S / wavelength_nm[-2]
        + LIGHT_SPEED_NM_PER_S / wavelength_nm[-1]
    ) * 0.25

    return wavelength_nm, frequency_weights
