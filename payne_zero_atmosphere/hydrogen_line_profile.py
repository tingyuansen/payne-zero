"""Hydrogen Stark-profile evaluator (the original HPROF).

Builds per-line setups (Stark widths, components, Griem parameters) from
the hydrogen tables in the data home and evaluates the quasi-static +
impact profile per layer and frequency. The compiled mirror of this
evaluator lives in line_opacity.py and is byte-exact against this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .constants import (
    LIGHT_SPEED_ANGSTROM_PER_S as _LIGHT_SPEED_A_PER_SECOND,
    LIGHT_SPEED_CM_PER_S_EXACT as _LIGHT_SPEED_CM_PER_SECOND,
)
from .data_files import atmosphere_table_path


_DEFAULT_TABLE_PATH = atmosphere_table_path("hydrogen_line_profile_tables.npz")

_RYDBERG_HYDROGEN_HZ = 3.2880515e15
_SQRT_PI_APPROX = 1.77245
_PI_APPROX = 3.14159


class HydrogenLineProfileTableError(RuntimeError):
    """Raised when packaged hydrogen-line profile tables are missing or malformed."""


@dataclass(frozen=True)
class HydrogenLineProfileTables:
    stark_probability_table: np.ndarray
    stark_wing_correction_c: np.ndarray
    stark_wing_correction_d: np.ndarray
    stark_pressure_grid: np.ndarray
    stark_beta_grid: np.ndarray
    alpha_line_component_offsets: np.ndarray
    alpha_line_component_weights: np.ndarray
    alpha_line_component_start: np.ndarray
    alpha_line_component_count: np.ndarray
    stark_component_offsets: np.ndarray
    stark_component_weights: np.ndarray
    stark_component_count: np.ndarray
    h2plus_quasimolecular_cutoff_table: np.ndarray
    h2_quasimolecular_cutoff_table: np.ndarray
    lyman_radiative_damping_sums: np.ndarray
    radiative_damping_sums: np.ndarray
    impact_electron_density_thresholds_cm3: np.ndarray
    stark_knm_table: np.ndarray
    h2_partition_function: np.ndarray


@dataclass(frozen=True)
class HydrogenLineSetup:
    lower_level: int
    upper_level: int
    line_frequency_hz: float
    line_wavelength_a: float
    beta_scale: float
    stark_c1_factor: float
    stark_c2_factor: float
    radiative_width: float
    resonance_width: float
    van_der_waals_width: float
    stark_width: float
    low_density_impact_numerator: float
    impact_electron_density_threshold_cm3: float
    stark_component_offsets_hz: np.ndarray
    stark_component_weights: np.ndarray


def _load_hydrogen_line_profile_tables(path: Path) -> HydrogenLineProfileTables:
    if not path.exists():
        raise HydrogenLineProfileTableError(
            f"Missing hydrogen-line profile table: {path}"
        )
    with np.load(path, allow_pickle=False) as arrays:
        required = {
            "stark_probability_table",
            "stark_wing_correction_c",
            "stark_wing_correction_d",
            "stark_pressure_grid",
            "stark_beta_grid",
            "alpha_line_component_offsets",
            "alpha_line_component_weights",
            "alpha_line_component_start",
            "alpha_line_component_count",
            "stark_component_offsets",
            "stark_component_weights",
            "stark_component_count",
            "h2plus_quasimolecular_cutoff_table",
            "h2_quasimolecular_cutoff_table",
            "lyman_radiative_damping_sums",
            "radiative_damping_sums",
            "impact_electron_density_thresholds_cm3",
            "stark_knm_table",
        }
        missing = sorted(required.difference(arrays.files))
        if missing:
            raise HydrogenLineProfileTableError(
                f"{path.name} is missing required keys: {', '.join(missing)}"
            )
        integer_fields = {
            "alpha_line_component_start",
            "alpha_line_component_count",
            "stark_component_count",
        }
        loaded = {
            name: np.asarray(
                arrays[name],
                dtype=np.int64 if name in integer_fields else np.float64,
            )
            for name in sorted(required)
        }
    # The molecular H2 partition function lives with the molecular-equilibrium
    # tables; the hydrogen occupation-probability path shares that single copy.
    equilibrium_path = path.parent / "molecular_equilibrium_tables.npz"
    if not equilibrium_path.exists():
        raise HydrogenLineProfileTableError(
            f"Missing molecular-equilibrium table: {equilibrium_path}"
        )
    with np.load(equilibrium_path, allow_pickle=False) as equilibrium_arrays:
        loaded["h2_partition_function"] = np.asarray(
            equilibrium_arrays["h2_partition_function"], dtype=np.float64
        )
    return HydrogenLineProfileTables(**loaded)


@lru_cache(maxsize=1)
def load_hydrogen_line_profile_tables(
    path: Path | None = None,
) -> HydrogenLineProfileTables:
    """Load packaged HPROF4 hydrogen-line profile tables with modern names."""

    return _load_hydrogen_line_profile_tables(path or _DEFAULT_TABLE_PATH)


@lru_cache(maxsize=256)
def _hydrogen_oscillator_strength(lower_level: int, upper_level: int) -> float:
    if upper_level <= lower_level:
        return 0.0
    lower = float(lower_level)
    upper = float(upper_level)
    lower_upper = upper - lower
    asymptotic_weight = 0.2027 / (lower**0.71)
    correction = 0.124 / lower
    base = lower * 1.9603
    blend_weight = 0.45 - 2.4 / (lower**3) * (lower - 1.0)
    lower_upper_power = lower_upper**1.2
    interpolation_weight = (lower_upper_power - 1.0) / (
        lower_upper_power + blend_weight
    )
    strength = base * (upper / (lower_upper * (upper + lower))) ** 3
    return strength * (
        1.0
        - interpolation_weight * asymptotic_weight
        - (0.222 + correction / upper) * (1.0 - interpolation_weight)
    )


def _build_exponential_tables() -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(1001, dtype=np.float64)
    return np.exp(-index), np.exp(-index * 0.001)


def _fast_exponential(
    x: float, integer_table: np.ndarray, fractional_table: np.ndarray
) -> float:
    if x < 0.0 or x >= 1001.0:
        return 0.0
    integer_index = int(x)
    fractional_index = int((x - float(integer_index)) * 1000.0 + 1.5)
    if fractional_index < 1:
        fractional_index = 1
    if fractional_index > 1001:
        fractional_index = 1001
    return float(integer_table[integer_index] * fractional_table[fractional_index - 1])


def _exponential_integral(order: int, x: float) -> float:
    a0, a1, a2, a3, a4, a5 = (
        -44178.5471728217,
        57721.7247139444,
        9938.31388962037,
        1842.11088668,
        101.093806161906,
        5.03416184097568,
    )
    b0, b1, b2, b3, b4 = (
        76537.3323337614,
        32597.1881290275,
        6106.10794245759,
        635.419418378382,
        37.2298352833327,
    )
    c0, c1, c2, c3, c4, c5, c6 = (
        4.65627107975096e-7,
        0.999979577051595,
        9.04161556946329,
        24.3784088791317,
        23.0192559391333,
        6.90522522784444,
        0.430967839469389,
    )
    d1, d2, d3, d4, d5, d6 = (
        10.0411643829054,
        32.4264210695138,
        41.2807841891424,
        20.4494785013794,
        3.31909213593302,
        0.103400130404874,
    )
    e0, e1, e2, e3, e4, e5, e6 = (
        -0.999999999998447,
        -26.6271060431811,
        -241.055827097015,
        -895.927957772937,
        -1298.85688746484,
        -545.374158883133,
        -5.66575206533869,
    )
    f1, f2, f3, f4, f5, f6 = (
        28.6271060422192,
        292.310039388533,
        1332.78537748257,
        2777.61949509163,
        2404.01713225909,
        631.6574832808,
    )

    exponential = float(np.exp(-x))
    if x > 4.0:
        first_order = (
            exponential
            + exponential
            * (e0 + (e1 + (e2 + (e3 + (e4 + (e5 + e6 / x) / x) / x) / x) / x) / x)
            / (x + f1 + (f2 + (f3 + (f4 + (f5 + f6 / x) / x) / x) / x) / x)
        ) / x
    elif x > 1.0:
        first_order = (
            exponential
            * (c6 + (c5 + (c4 + (c3 + (c2 + (c1 + c0 * x) * x) * x) * x) * x) * x)
            / (d6 + (d5 + (d4 + (d3 + (d2 + (d1 + x) * x) * x) * x) * x) * x)
        )
    elif x > 0.0:
        first_order = (a0 + (a1 + (a2 + (a3 + (a4 + a5 * x) * x) * x) * x) * x) / (
            b0 + (b1 + (b2 + (b3 + (b4 + x) * x) * x) * x) * x
        ) - np.log(x)
    else:
        first_order = 0.0
    if order == 1:
        return first_order

    value = first_order
    for index in range(1, order):
        value = (exponential - x * value) / float(index)
    return value


@lru_cache(maxsize=1)
def _exponential_integral_table() -> np.ndarray:
    table = np.zeros(2000, dtype=np.float64)
    for index in range(1, 2001):
        table[index - 1] = _exponential_integral(1, float(index) * 0.01)
    return table


def _fast_exponential_integral(x: float) -> float:
    if x > 20.0:
        return 0.0
    if x >= 0.5:
        table_index = int(x * 100.0 + 0.5)
        if table_index < 1:
            table_index = 1
        if table_index > 2000:
            table_index = 2000
        return float(_exponential_integral_table()[table_index - 1])
    if x <= 0.0:
        return 0.0
    return (1.0 - 0.22464 * x) * x - np.log(x) - 0.57721


def _stark_probability(
    beta: float,
    pressure_parameter: float,
    lower_level: int,
    upper_level: int,
    tables: HydrogenLineProfileTables,
) -> float:
    correction = 1.0
    beta_squared = beta * beta
    sqrt_beta = np.sqrt(max(beta, 1.0e-300))
    if beta <= 500.0:
        table_index = 7
        level_delta = upper_level - lower_level
        if lower_level <= 3 and level_delta <= 2:
            table_index = 2 * (lower_level - 1) + level_delta
        pressure_low_index = min(int(5.0 * pressure_parameter) + 1, 4)
        if pressure_low_index < 1:
            pressure_low_index = 1
        pressure_high_index = pressure_low_index + 1
        high_pressure_weight = 5.0 * (
            pressure_parameter
            - float(tables.stark_pressure_grid[pressure_low_index - 1])
        )
        low_pressure_weight = 1.0 - high_pressure_weight
        if beta <= 25.12:
            beta_high_index = int(
                np.searchsorted(tables.stark_beta_grid, beta, side="left")
            )
            if beta_high_index < 1:
                beta_high_index = 1
            if beta_high_index > 14:
                beta_high_index = 14
            beta_low_index = beta_high_index - 1
            beta_denominator = float(
                tables.stark_beta_grid[beta_high_index]
                - tables.stark_beta_grid[beta_low_index]
            )
            high_beta_weight = (
                0.0
                if beta_denominator == 0.0
                else (beta - float(tables.stark_beta_grid[beta_low_index]))
                / beta_denominator
            )
            low_beta_weight = 1.0 - high_beta_weight
            high_beta_correction = (
                float(
                    tables.stark_probability_table[
                        pressure_high_index - 1,
                        beta_high_index,
                        table_index - 1,
                    ]
                )
                * high_pressure_weight
                + float(
                    tables.stark_probability_table[
                        pressure_low_index - 1,
                        beta_high_index,
                        table_index - 1,
                    ]
                )
                * low_pressure_weight
            )
            low_beta_correction = (
                float(
                    tables.stark_probability_table[
                        pressure_high_index - 1,
                        beta_low_index,
                        table_index - 1,
                    ]
                )
                * high_pressure_weight
                + float(
                    tables.stark_probability_table[
                        pressure_low_index - 1,
                        beta_low_index,
                        table_index - 1,
                    ]
                )
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
            float(
                tables.stark_wing_correction_c[pressure_high_index - 1, table_index - 1]
            )
            * high_pressure_weight
            + float(
                tables.stark_wing_correction_c[pressure_low_index - 1, table_index - 1]
            )
            * low_pressure_weight
        )
        d_value = (
            float(
                tables.stark_wing_correction_d[pressure_high_index - 1, table_index - 1]
            )
            * high_pressure_weight
            + float(
                tables.stark_wing_correction_d[pressure_low_index - 1, table_index - 1]
            )
            * low_pressure_weight
        )
        correction = 1.0 + d_value / (c_value + beta * sqrt_beta)
    return (1.5 / sqrt_beta + 27.0 / beta_squared) / beta_squared * correction


def molecular_hydrogen_equilibrium_constant(
    temperature: np.ndarray | float,
    *,
    tables: HydrogenLineProfileTables | None = None,
) -> np.ndarray:
    table = load_hydrogen_line_profile_tables() if tables is None else tables
    temp = np.asarray(temperature, dtype=np.float64)
    temp = np.where(np.isfinite(temp) & (temp > 100.0), temp, 100.0)
    temp = np.minimum(temp, 19900.0)
    table_index = np.floor(temp / 100.0).astype(np.int64)
    table_index = np.minimum(199, np.maximum(1, table_index))
    partition = table.h2_partition_function[table_index - 1] + (
        table.h2_partition_function[table_index]
        - table.h2_partition_function[table_index - 1]
    ) / 100.0 * (temp - table_index * 100.0)
    equilibrium = partition * (2.0**1.5) / 4.0
    equilibrium /= (
        2.0 * 3.14159 * 1.008 * 1.660e-24 * 1.38054e-16 / 6.6256e-27**2 * temp
    ) ** 1.5
    equilibrium *= np.exp(36118.11 * 6.6256e-27 * 2.997925e10 / 1.38054e-16 / temp)
    return equilibrium


def compute_hydrogen_molecule_population(
    *,
    temperature: np.ndarray,
    hydrogen_neutral_partition_normalized_population: np.ndarray,
    hydrogen_departure_coefficient: np.ndarray | None = None,
    tables: HydrogenLineProfileTables | None = None,
) -> np.ndarray:
    temp = np.asarray(temperature, dtype=np.float64)
    ground = np.asarray(
        hydrogen_neutral_partition_normalized_population, dtype=np.float64
    )
    if hydrogen_departure_coefficient is None:
        departure = np.ones_like(temp, dtype=np.float64)
    else:
        departure = np.asarray(hydrogen_departure_coefficient, dtype=np.float64)
    return (ground * 2.0 * departure) ** 2 * molecular_hydrogen_equilibrium_constant(
        temp,
        tables=tables,
    )


@dataclass
class HydrogenLineProfileEvaluator:
    temperature: np.ndarray
    electron_density: np.ndarray
    hydrogen_neutral_population: np.ndarray
    hydrogen_ionized_population: np.ndarray
    hydrogen_neutral_partition_normalized_population: np.ndarray
    helium_neutral_population: np.ndarray
    hydrogen_fractional_doppler_width: np.ndarray
    molecular_hydrogen_population: np.ndarray
    tables: HydrogenLineProfileTables
    pressure_parameter: np.ndarray = field(init=False)
    field_strength: np.ndarray = field(init=False)
    high_density_impact_factor: np.ndarray = field(init=False)
    low_density_impact_factor: np.ndarray = field(init=False)
    temperature_density_he: np.ndarray = field(init=False)
    temperature_density_h2: np.ndarray = field(init=False)
    stark_linear_density_coefficient: np.ndarray = field(init=False)
    stark_quadratic_density_coefficient: np.ndarray = field(init=False)
    stark_gamma_thermal_correction: np.ndarray = field(init=False)
    stark_gamma_density_correction: np.ndarray = field(init=False)
    exponential_integer: np.ndarray = field(init=False)
    exponential_fraction: np.ndarray = field(init=False)
    _line_cache: dict[tuple[int, int], HydrogenLineSetup] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        temp = np.asarray(self.temperature, dtype=np.float64)
        electrons = np.asarray(self.electron_density, dtype=np.float64)
        self.temperature = temp
        self.electron_density = electrons
        self.hydrogen_neutral_population = np.asarray(
            self.hydrogen_neutral_population,
            dtype=np.float64,
        )
        self.hydrogen_ionized_population = np.asarray(
            self.hydrogen_ionized_population,
            dtype=np.float64,
        )
        self.hydrogen_neutral_partition_normalized_population = np.asarray(
            self.hydrogen_neutral_partition_normalized_population,
            dtype=np.float64,
        )
        self.helium_neutral_population = np.asarray(
            self.helium_neutral_population,
            dtype=np.float64,
        )
        self.hydrogen_fractional_doppler_width = np.asarray(
            self.hydrogen_fractional_doppler_width, dtype=np.float64
        )
        self.molecular_hydrogen_population = np.asarray(
            self.molecular_hydrogen_population,
            dtype=np.float64,
        )

        electron_sixth_root = np.maximum(electrons, 1.0e-300) ** 0.1666667
        temperature_10000 = temp / 10000.0
        temperature_factor = np.maximum(temperature_10000, 1.0e-300) ** 0.3
        self.pressure_parameter = (
            electron_sixth_root * 0.08989 / np.sqrt(np.maximum(temp, 1.0e-300))
        )
        self.field_strength = electron_sixth_root**4 * 1.25e-9
        self.high_density_impact_factor = 2.0 / (
            1.0
            + 0.012
            / np.maximum(temp, 1.0e-300)
            * np.sqrt(np.maximum(electrons / np.maximum(temp, 1.0e-300), 0.0))
        )
        self.low_density_impact_factor = temperature_factor / np.maximum(
            electron_sixth_root, 1.0e-300
        )
        self.temperature_density_he = temperature_factor * np.maximum(
            self.helium_neutral_population,
            0.0,
        )
        self.temperature_density_h2 = temperature_factor * np.maximum(
            self.molecular_hydrogen_population,
            0.0,
        )
        self.stark_linear_density_coefficient = (
            self.field_strength * 78940.0 / np.maximum(temp, 1.0e-300)
        )
        self.stark_quadratic_density_coefficient = (
            self.field_strength
            * self.field_strength
            / 5.96e-23
            / np.maximum(
                electrons,
                1.0e-300,
            )
        )
        self.stark_gamma_thermal_correction = 0.2 + 0.09 * np.sqrt(
            np.maximum(temperature_10000, 0.0)
        ) / (1.0 + electrons / 1.0e13)
        self.stark_gamma_density_correction = 0.2 / (1.0 + electrons / 1.0e15)
        self.exponential_integer, self.exponential_fraction = (
            _build_exponential_tables()
        )
        self._line_cache = {}

    def line_setup(
        self, lower_level: int, upper_level: int
    ) -> HydrogenLineSetup | None:
        key = (int(lower_level), int(upper_level))
        if key in self._line_cache:
            return self._line_cache[key]
        lower = int(lower_level)
        upper = int(upper_level)
        level_delta = upper - lower
        if level_delta <= 0:
            return None

        lower_float = float(lower)
        upper_float = float(upper)
        lower_squared = lower_float * lower_float
        upper_squared = upper_float * upper_float
        combined_squared = upper_squared * lower_squared
        inverse_level_gap = (upper_squared - lower_squared) / combined_squared

        if level_delta <= 3 and lower <= 4:
            knm = float(self.tables.stark_knm_table[lower - 1, level_delta - 1])
        else:
            knm = (
                5.5e-5
                / inverse_level_gap
                * combined_squared
                / (1.0 + 0.13 / float(level_delta))
            )
        low_density_impact_numerator = 320.0
        if upper == 2:
            low_density_impact_numerator = 550.0
        if upper == 3:
            low_density_impact_numerator = 380.0
        impact_electron_density_threshold_cm3 = 1.0e13
        if level_delta <= 3:
            impact_electron_density_threshold_cm3 = 1.0e14
        if level_delta <= 2 and lower <= 2:
            impact_electron_density_threshold_cm3 = float(
                self.tables.impact_electron_density_thresholds_cm3[
                    lower - 1, level_delta - 1
                ]
            )

        line_frequency = _RYDBERG_HYDROGEN_HZ * inverse_level_gap
        beta_scale = _LIGHT_SPEED_A_PER_SECOND / (line_frequency * line_frequency) / knm
        line_wavelength = _LIGHT_SPEED_A_PER_SECOND / line_frequency
        stark_c1_factor = (
            knm / line_wavelength * inverse_level_gap * (upper_squared - lower_squared)
        )
        stark_c2_factor = (knm / line_wavelength) ** 2
        radiative_width = float(
            self.tables.radiative_damping_sums[lower - 1]
            + self.tables.radiative_damping_sums[upper - 1]
        )
        if lower == 1:
            radiative_width = float(self.tables.lyman_radiative_damping_sums[upper - 1])
        radiative_width = radiative_width / 12.5664 / line_frequency
        resonance_width = (
            _hydrogen_oscillator_strength(1, upper)
            / upper_float
            / (1.0 - 1.0 / upper_squared)
        )
        if lower != 1:
            resonance_width += (
                _hydrogen_oscillator_strength(1, lower)
                / lower_float
                / (1.0 - 1.0 / lower_squared)
            )
        resonance_width = resonance_width * 3.92e-24 / inverse_level_gap
        van_der_waals_width = (
            4.45e-26
            / inverse_level_gap
            * (upper_squared * (7.0 * upper_squared + 5.0)) ** 0.4
        )
        stark_width = 1.6678e-18 * line_frequency * knm

        if lower > 4 or upper > 10:
            stark_component_offsets = np.asarray([0.0], dtype=np.float64)
            stark_component_weights = np.asarray([1.0], dtype=np.float64)
        elif level_delta != 1:
            stark_component_count = int(self.tables.stark_component_count[lower - 1])
            stark_component_offsets = np.asarray(
                self.tables.stark_component_offsets[:stark_component_count, lower - 1]
                * 1.0e7,
                dtype=np.float64,
            )
            stark_component_weights = np.asarray(
                self.tables.stark_component_weights[:stark_component_count, lower - 1]
                / lower_squared,
                dtype=np.float64,
            )
        else:
            stark_component_count = int(
                self.tables.alpha_line_component_count[lower - 1]
            )
            start = int(self.tables.alpha_line_component_start[lower - 1]) - 1
            stop = start + stark_component_count
            stark_component_offsets = np.asarray(
                self.tables.alpha_line_component_offsets[start:stop] * 1.0e7,
                dtype=np.float64,
            )
            stark_component_weights = np.asarray(
                self.tables.alpha_line_component_weights[start:stop]
                / lower_squared
                / 3.0,
                dtype=np.float64,
            )

        setup = HydrogenLineSetup(
            lower_level=lower,
            upper_level=upper,
            line_frequency_hz=line_frequency,
            line_wavelength_a=line_wavelength,
            beta_scale=beta_scale,
            stark_c1_factor=stark_c1_factor,
            stark_c2_factor=stark_c2_factor,
            radiative_width=radiative_width,
            resonance_width=resonance_width,
            van_der_waals_width=van_der_waals_width,
            stark_width=stark_width,
            low_density_impact_numerator=low_density_impact_numerator,
            impact_electron_density_threshold_cm3=impact_electron_density_threshold_cm3,
            stark_component_offsets_hz=stark_component_offsets,
            stark_component_weights=stark_component_weights,
        )
        self._line_cache[key] = setup
        return setup

    def profile(
        self,
        lower_level: int,
        upper_level: int,
        layer_index: int,
        wavelength_offset_nm: float,
    ) -> float:
        setup = self.line_setup(lower_level, upper_level)
        if setup is None:
            return 0.0
        return self.profile_for_setup(setup, layer_index, wavelength_offset_nm)

    def profile_for_setup(
        self,
        setup: HydrogenLineSetup,
        layer_index: int,
        wavelength_offset_nm: float,
    ) -> float:
        layer = int(layer_index)
        wavelength_a = setup.line_wavelength_a + float(wavelength_offset_nm) * 10.0
        if wavelength_a <= 0.0:
            return 0.0
        frequency = _LIGHT_SPEED_A_PER_SECOND / wavelength_a
        frequency_offset = abs(frequency - setup.line_frequency_hz)
        doppler_width = float(self.hydrogen_fractional_doppler_width[layer])
        if doppler_width <= 0.0:
            return 0.0
        stark_width = setup.stark_width * float(self.field_strength[layer])
        van_der_waals_width = setup.van_der_waals_width * float(
            self.temperature_density_he[layer]
        ) + 2.0 * setup.van_der_waals_width * float(self.temperature_density_h2[layer])
        radiative_width = setup.radiative_width
        resonance_width = setup.resonance_width * float(
            self.hydrogen_neutral_population[layer]
        )
        lorentz_width = resonance_width + van_der_waals_width + radiative_width
        profile_mode = 1
        if not (doppler_width >= stark_width and doppler_width >= lorentz_width):
            profile_mode = 2
            if lorentz_width < stark_width:
                profile_mode = 3
        half_width = setup.line_frequency_hz * max(
            doppler_width, lorentz_width, stark_width
        )
        in_core = abs(frequency_offset) <= half_width
        doppler_frequency_width = setup.line_frequency_hz * doppler_width
        if doppler_frequency_width <= 0.0:
            return 0.0

        stark_wavelength_offset = (
            -10.0
            * float(wavelength_offset_nm)
            / setup.line_wavelength_a
            * setup.line_frequency_hz
        )
        if in_core:
            if profile_mode == 1:
                value = self._doppler_profile(setup, frequency, doppler_frequency_width)
            elif profile_mode == 2:
                value = self._lorentz_profile(
                    setup,
                    layer,
                    frequency,
                    frequency_offset,
                    doppler_frequency_width,
                    resonance_width,
                    van_der_waals_width,
                    radiative_width,
                )
            else:
                value = self._stark_profile(
                    setup,
                    layer,
                    frequency,
                    stark_wavelength_offset,
                    doppler_frequency_width,
                )
        else:
            value = (
                self._doppler_profile(setup, frequency, doppler_frequency_width)
                + self._lorentz_profile(
                    setup,
                    layer,
                    frequency,
                    frequency_offset,
                    doppler_frequency_width,
                    resonance_width,
                    van_der_waals_width,
                    radiative_width,
                )
                + self._stark_profile(
                    setup,
                    layer,
                    frequency,
                    stark_wavelength_offset,
                    doppler_frequency_width,
                )
            )
        return max(value, 0.0)

    def _doppler_profile(
        self,
        setup: HydrogenLineSetup,
        frequency: float,
        doppler_frequency_width: float,
    ) -> float:
        value = 0.0
        for offset, weight in zip(
            setup.stark_component_offsets_hz,
            setup.stark_component_weights,
            strict=False,
        ):
            distance = (
                abs(frequency - setup.line_frequency_hz - float(offset))
                / doppler_frequency_width
            )
            if distance <= 7.0:
                value += _fast_exponential(
                    distance * distance,
                    self.exponential_integer,
                    self.exponential_fraction,
                ) * float(weight)
        return value

    def _lorentz_profile(
        self,
        setup: HydrogenLineSetup,
        layer: int,
        frequency: float,
        frequency_offset: float,
        doppler_frequency_width: float,
        resonance_width: float,
        van_der_waals_width: float,
        radiative_width: float,
    ) -> float:
        if setup.lower_level == 1 and setup.upper_level == 2:
            resonance_width *= 4.0
            total_width = resonance_width + van_der_waals_width + radiative_width
            half_width = setup.line_frequency_hz * total_width
            if frequency > (82259.105 - 4000.0) * _LIGHT_SPEED_CM_PER_SECOND:
                resonance_profile = (
                    resonance_width
                    * setup.line_frequency_hz
                    / _PI_APPROX
                    / (frequency_offset * frequency_offset + half_width * half_width)
                    * _SQRT_PI_APPROX
                    * doppler_frequency_width
                )
            else:
                cutoff = 0.0
                if frequency >= 50000.0 * _LIGHT_SPEED_CM_PER_SECOND:
                    spacing = 200.0 * _LIGHT_SPEED_CM_PER_SECOND
                    frequency_22000 = (82259.105 - 22000.0) * _LIGHT_SPEED_CM_PER_SECOND
                    if frequency < frequency_22000:
                        cutoff = (
                            self.tables.h2_quasimolecular_cutoff_table[1]
                            - self.tables.h2_quasimolecular_cutoff_table[0]
                        ) / spacing * (
                            frequency - frequency_22000
                        ) + self.tables.h2_quasimolecular_cutoff_table[0]
                    else:
                        cutoff_index = int((frequency - frequency_22000) / spacing)
                        cutoff_index = max(
                            0,
                            min(
                                cutoff_index,
                                self.tables.h2_quasimolecular_cutoff_table.size - 2,
                            ),
                        )
                        cutoff_frequency = cutoff_index * spacing + frequency_22000
                        cutoff = (
                            self.tables.h2_quasimolecular_cutoff_table[cutoff_index + 1]
                            - self.tables.h2_quasimolecular_cutoff_table[cutoff_index]
                        ) / spacing * (
                            frequency - cutoff_frequency
                        ) + self.tables.h2_quasimolecular_cutoff_table[cutoff_index]
                    cutoff = (
                        10.0 ** (cutoff - 14.0)
                        * float(
                            self.hydrogen_neutral_partition_normalized_population[layer]
                        )
                        * 2.0
                        / _LIGHT_SPEED_CM_PER_SECOND
                    )
                resonance_profile = cutoff * _SQRT_PI_APPROX * doppler_frequency_width
            radiative_profile = (
                radiative_width
                * setup.line_frequency_hz
                / _PI_APPROX
                / (frequency_offset * frequency_offset + half_width * half_width)
                * _SQRT_PI_APPROX
                * doppler_frequency_width
            )
            if frequency <= 2.463e15:
                radiative_profile = 0.0
            van_der_waals_profile = (
                van_der_waals_width
                * setup.line_frequency_hz
                / _PI_APPROX
                / (frequency_offset * frequency_offset + half_width * half_width)
                * _SQRT_PI_APPROX
                * doppler_frequency_width
            )
            if frequency < 1.8e15:
                van_der_waals_profile = 0.0
            return resonance_profile + radiative_profile + van_der_waals_profile

        half_width = setup.line_frequency_hz * (
            resonance_width + van_der_waals_width + radiative_width
        )
        if half_width <= 0.0:
            return 0.0
        return (
            half_width
            / _PI_APPROX
            / (frequency_offset * frequency_offset + half_width * half_width)
            * _SQRT_PI_APPROX
            * doppler_frequency_width
        )

    def _stark_profile(
        self,
        setup: HydrogenLineSetup,
        layer: int,
        frequency: float,
        stark_wavelength_offset: float,
        doppler_frequency_width: float,
    ) -> float:
        field_strength = float(self.field_strength[layer])
        if field_strength <= 0.0:
            return 0.0
        low_density_impact_weight = 1.0 / (
            1.0
            + float(self.electron_density[layer])
            / setup.impact_electron_density_threshold_cm3
        )
        impact_broadening_factor = setup.low_density_impact_numerator * float(
            self.low_density_impact_factor[layer]
        ) * low_density_impact_weight + float(
            self.high_density_impact_factor[layer]
        ) * (1.0 - low_density_impact_weight)
        linear_impact_parameter = (
            float(self.stark_linear_density_coefficient[layer])
            * setup.stark_c1_factor
            * impact_broadening_factor
        )
        quadratic_impact_parameter = (
            float(self.stark_quadratic_density_coefficient[layer])
            * setup.stark_c2_factor
        )
        linear_impact_parameter = max(linear_impact_parameter, 0.0)
        quadratic_impact_parameter = max(quadratic_impact_parameter, 0.0)
        impact_width_scale = 6.77 * np.sqrt(linear_impact_parameter)
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
                - float(self.stark_gamma_thermal_correction[layer])
                - float(self.stark_gamma_density_correction[layer])
            )
        )
        beta = abs(stark_wavelength_offset) / field_strength * setup.beta_scale
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
                    * _fast_exponential(
                        min(80.0, linear_impact_argument),
                        self.exponential_integer,
                        self.exponential_fraction,
                    )
                    + _fast_exponential_integral(linear_impact_argument)
                    - 0.5 * _fast_exponential_integral(quadratic_impact_argument)
                )
                * (
                    1.0
                    - float(self.stark_gamma_thermal_correction[layer])
                    / (1.0 + (90.0 * linear_impact_argument) ** 3)
                    - float(self.stark_gamma_density_correction[layer])
                    / (1.0 + 2000.0 * linear_impact_argument)
                )
            )
            if impact_width <= 1.0e-20:
                impact_width = 0.0

        probability = _stark_probability(
            beta,
            float(self.pressure_parameter[layer]),
            setup.lower_level,
            setup.upper_level,
            self.tables,
        )
        profile = 0.0
        if setup.upper_level <= 2:
            probability *= 0.5
            if frequency >= (82259.105 - 20000.0) * _LIGHT_SPEED_CM_PER_SECOND:
                if frequency <= (82259.105 - 4000.0) * _LIGHT_SPEED_CM_PER_SECOND:
                    frequency_15000 = (82259.105 - 15000.0) * _LIGHT_SPEED_CM_PER_SECOND
                    spacing = 100.0 * _LIGHT_SPEED_CM_PER_SECOND
                    if frequency < frequency_15000:
                        cutoff = (
                            self.tables.h2plus_quasimolecular_cutoff_table[1]
                            - self.tables.h2plus_quasimolecular_cutoff_table[0]
                        ) / spacing * (
                            frequency - frequency_15000
                        ) + self.tables.h2plus_quasimolecular_cutoff_table[0]
                    else:
                        cutoff_index = int((frequency - frequency_15000) / spacing)
                        cutoff_index = max(
                            0,
                            min(
                                cutoff_index,
                                self.tables.h2plus_quasimolecular_cutoff_table.size - 2,
                            ),
                        )
                        cutoff_frequency = cutoff_index * spacing + frequency_15000
                        cutoff = (
                            self.tables.h2plus_quasimolecular_cutoff_table[
                                cutoff_index + 1
                            ]
                            - self.tables.h2plus_quasimolecular_cutoff_table[
                                cutoff_index
                            ]
                        ) / spacing * (
                            frequency - cutoff_frequency
                        ) + self.tables.h2plus_quasimolecular_cutoff_table[cutoff_index]
                    cutoff = (
                        10.0 ** (cutoff - 14.0)
                        / _LIGHT_SPEED_CM_PER_SECOND
                        * float(self.hydrogen_ionized_population[layer])
                    )
                    profile += cutoff * _SQRT_PI_APPROX * doppler_frequency_width
                else:
                    beta4000 = (
                        4000.0
                        * _LIGHT_SPEED_CM_PER_SECOND
                        / field_strength
                        * setup.beta_scale
                    )
                    probability4000 = (
                        _stark_probability(
                            beta4000,
                            float(self.pressure_parameter[layer]),
                            setup.lower_level,
                            setup.upper_level,
                            self.tables,
                        )
                        * 0.5
                        / field_strength
                        * setup.beta_scale
                    )
                    cutoff4000 = (
                        10.0 ** (-11.07 - 14.0)
                        / _LIGHT_SPEED_CM_PER_SECOND
                        * float(self.hydrogen_ionized_population[layer])
                    )
                    if probability4000 != 0.0:
                        profile += (
                            cutoff4000
                            / probability4000
                            * probability
                            / field_strength
                            * setup.beta_scale
                            * _SQRT_PI_APPROX
                            * doppler_frequency_width
                        )

        lorentz_component = 0.0
        if impact_width > 0.0:
            lorentz_component = (
                impact_width / _PI_APPROX / (impact_width * impact_width + beta * beta)
            )
        satellite_blend_square = (0.9 * linear_impact_argument) ** 2
        satellite_enhancement = (
            satellite_blend_square + 0.03 * np.sqrt(max(linear_impact_argument, 0.0))
        ) / (satellite_blend_square + 1.0)
        profile += (
            (probability * (1.0 + satellite_enhancement) + lorentz_component)
            / field_strength
            * setup.beta_scale
            * _SQRT_PI_APPROX
            * doppler_frequency_width
        )
        return profile
