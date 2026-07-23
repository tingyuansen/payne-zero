"""Torch hydrogen line opacity (Stark quasi-static + impact profiles).

Consumes the canonical hydrogen Stark tables (stark_* keys in
`line_profile_tables.npz`) and deposits Balmer/Paschen/Brackett series
opacity on the device wavelength grid, mirroring the validated branch
structure with `torch.where` regime blends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .constants import (
    BOLTZMANN_ERG_PER_K,
    CLASSICAL_LINE_STRENGTH_COEFFICIENT,
    HYDROGEN_PROFILE_ATOMIC_MASS_GRAM,
    LIGHT_SPEED_ANGSTROM_PER_S,
    LIGHT_SPEED_CM_PER_S,
    LIGHT_SPEED_NM_PER_S,
    PLANCK_ERG_SECOND,
)
from .device import ACCUMULATION_DTYPE, device, to_dev
from .line_opacity import (
    LINE_CENTER_CUTOFF_RATIO,
    fast_ex,
    _fastex_tables,
    highp_dtype,
)

# Validated hydrogen-profile constants.
HYDROGEN_RYDBERG_FREQUENCY_HZ = 3.2880515e15  # Hz  (Rydberg frequency)
LIGHT_SPEED_KM_PER_S = 299792.458  # km / s
HYDROGEN_ATOMIC_MASS_AMU = 1.008  # amu

# H-level energies (cm^-1), n=1..8; above n=8 use the Rydberg formula
_HYDROGEN_LEVEL_ENERGY_CM = np.array(
    [
        0.0,
        82259.105,
        97492.302,
        102823.893,
        105291.651,
        106632.160,
        107440.444,
        107965.051,
    ]
)
_HYDROGEN_RYDBERG_WAVENUMBER_CM, _HYDROGEN_IONIZATION_LIMIT_WAVENUMBER_CM = (
    109677.576,
    109678.764,
)


def _hydrogen_level_energy_cm(nn: int) -> float:
    """Energy of hydrogen level nn in cm^-1 (table for low nn, Rydberg above)."""
    if nn <= 0:
        return 0.0
    return (
        float(_HYDROGEN_LEVEL_ENERGY_CM[nn - 1])
        if nn - 1 < _HYDROGEN_LEVEL_ENERGY_CM.size
        else _HYDROGEN_IONIZATION_LIMIT_WAVENUMBER_CM
        - _HYDROGEN_RYDBERG_WAVENUMBER_CM / float(nn * nn)
    )


# Fine-structure offsets/weights for hydrogen sub-lines. Computing these from
# compact tables keeps the full optical range self-contained, including Halpha
# and other Delta-n == 1 transitions that are not present in narrow-window caches.
# Delta-n == 1 (Balmer/Paschen/... alpha lines) component offsets in
# 1e7 Hz units and weights. The embedded table keeps alpha-line
# fine structure available even when a narrow input catalog omits those records.
_ALPHA_LINE_COMPONENT_OFFSET_1E7_HZ = np.array(
    [
        -730.0,
        370.0,
        188.0,
        515.0,
        327.0,
        619.0,
        -772.0,
        -473.0,
        -369.0,
        120.0,
        1256.0,
        162.0,
        285.0,
        -161.0,
        -38.3,
        6.82,
        -174.0,
        -147.0,
        -101.0,
        -77.5,
        55.0,
        126.0,
        275.0,
        139.0,
        -60.0,
        3.7,
        27.0,
        -69.0,
        -42.0,
        -18.0,
        -5.5,
        -9.1,
        -33.0,
        -24.0,
    ],
    dtype=np.float64,
)
_ALPHA_LINE_COMPONENT_WEIGHT = np.array(
    [
        1.0,
        2.0,
        1.0,
        2.0,
        1.0,
        2.0,
        1.0,
        2.0,
        3.0,
        1.0,
        2.0,
        1.0,
        2.0,
        1.0,
        4.0,
        6.0,
        1.0,
        2.0,
        3.0,
        4.0,
        1.0,
        2.0,
        1.0,
        2.0,
        1.0,
        4.0,
        6.0,
        1.0,
        7.0,
        6.0,
        4.0,
        4.0,
        4.0,
        5.0,
    ],
    dtype=np.float64,
)
_ALPHA_LINE_COMPONENT_START_INDEX = np.array(
    [1, 3, 10, 21], dtype=np.int64
)  # slice start per n (1..4)
_ALPHA_LINE_COMPONENT_COUNT = np.array(
    [2, 7, 11, 14], dtype=np.int64
)  # number of components per n
# Delta-n != 1 (Hbeta/Hgamma/...) component offsets and weights.
_GENERAL_LINE_COMPONENT_OFFSET_1E7_HZ = np.array(
    [
        [0.0, 468.0, 260.0, 140.0],
        [0.0, 576.0, 290.0, 150.0],
        [0.0, -522.0, -33.0, 18.0],
        [0.0, 0.0, -140.0, -27.0],
        [0.0, 0.0, 0.0, -51.0],
    ],
    dtype=np.float64,
)
_GENERAL_LINE_COMPONENT_WEIGHT = np.array(
    [
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0],
        [0.0, 2.0, 4.0, 4.0],
        [0.0, 0.0, 3.0, 6.0],
        [0.0, 0.0, 0.0, 4.0],
    ],
    dtype=np.float64,
)
_GENERAL_LINE_COMPONENT_COUNT = np.array(
    [1, 3, 4, 5], dtype=np.int64
)  # number of components per n


def _fine_structure(lower_level: int, upper_level: int):
    """Fine-structure component offsets and weights for one transition.

    Returns `(offsets_hz, weights)`.  Transitions outside the compact Stark
    fine-structure tables use one centered unit-weight component.
    """
    level_separation = upper_level - lower_level
    lower_level_squared = float(lower_level) * float(lower_level)
    if lower_level > 4 or upper_level > 10:
        return np.array([0.0]), np.array([1.0])
    if level_separation != 1:
        component_count = int(_GENERAL_LINE_COMPONENT_COUNT[lower_level - 1])
        offsets = (
            _GENERAL_LINE_COMPONENT_OFFSET_1E7_HZ[:component_count, lower_level - 1]
            * 1.0e7
        )
        weights = (
            _GENERAL_LINE_COMPONENT_WEIGHT[:component_count, lower_level - 1]
            / lower_level_squared
        )
        return offsets.copy(), weights.copy()
    component_count = int(_ALPHA_LINE_COMPONENT_COUNT[lower_level - 1])
    component_start = int(_ALPHA_LINE_COMPONENT_START_INDEX[lower_level - 1])
    offsets = (
        _ALPHA_LINE_COMPONENT_OFFSET_1E7_HZ[
            component_start : component_start + component_count
        ]
        * 1.0e7
    )
    weights = (
        _ALPHA_LINE_COMPONENT_WEIGHT[
            component_start : component_start + component_count
        ]
        / lower_level_squared
        / 3.0
    )
    return offsets.copy(), weights.copy()


def _hf_nm(lower_level: int, upper_level: int) -> float:
    """Hydrogen absorption oscillator strength f_{n->m} (Menzel-Pekeris form).

    Feeds the resonance (self-broadening) width. Note resonance broadening of the
    n->m line is set by the transitions to the GROUND state, so the profile calls
    f_{1->m} and f_{1->n}, not f_{n->m}.
    """
    if upper_level <= lower_level:
        return 0.0
    lower_level_float = float(lower_level)
    upper_level_float = float(upper_level)
    infinity_gaunt_factor = 0.2027 / lower_level_float**0.71
    correction_gaunt_factor = 0.124 / lower_level_float
    strength_scale = lower_level_float * 1.9603
    blend_offset = 0.45 - 2.4 / lower_level_float**3 * (lower_level_float - 1.0)
    level_gap = upper_level_float - lower_level_float
    base_strength = (
        strength_scale
        * (upper_level_float / (level_gap * (upper_level_float + lower_level_float)))
        ** 3
    )
    gap_power = level_gap**1.2
    blend_weight = (gap_power - 1.0) / (gap_power + blend_offset)
    return base_strength * (
        1.0
        - blend_weight * infinity_gaunt_factor
        - (0.222 + correction_gaunt_factor / upper_level_float) * (1.0 - blend_weight)
    )


def _stark_quasi_static_profile(
    beta: torch.Tensor,
    pressure_parameter: torch.Tensor,
    lower_level: int,
    upper_level: int,
    stark_probability_table: torch.Tensor,
    stark_wing_correction_c_table: torch.Tensor,
    stark_wing_correction_d_table: torch.Tensor,
    stark_pressure_grid: torch.Tensor,
    stark_beta_grid: torch.Tensor,
) -> torch.Tensor:
    """Quasi-static linear-Stark profile S(beta): Holtsmark average + corrections.

    `beta` is `[depth, wavelength]`; `pressure_parameter` is `[depth, 1]`.
    The three beta regimes are blended with `torch.where` to preserve the
    validated branch structure.
    """
    beta_squared = beta * beta
    sqrt_beta = torch.sqrt(torch.clamp(beta, min=0.0))
    sqrt_beta_safe = torch.where(sqrt_beta > 0, sqrt_beta, torch.ones_like(sqrt_beta))
    beta_squared_safe = torch.where(
        beta_squared > 0,
        beta_squared,
        torch.ones_like(beta_squared),
    )

    # transition-table column index (scalar per (n,m))
    level_separation = upper_level - lower_level
    table_index_1based = (
        2 * (lower_level - 1) + level_separation
        if (lower_level <= 3 and level_separation <= 2)
        else 7
    )
    table_index_1based = min(max(table_index_1based, 1), 7)
    table_index = table_index_1based - 1

    # --- pressure-parameter bracket (per depth, broadcast over wavelength_count) ---
    pressure_lower_index = torch.clamp(
        (5.0 * pressure_parameter).to(torch.int64) + 1,
        1,
        4,
    )
    pressure_upper_index = pressure_lower_index + 1
    pressure_upper_weight = torch.clamp(
        5.0 * (pressure_parameter - stark_pressure_grid[pressure_lower_index - 1]),
        0.0,
        1.0,
    )
    pressure_lower_weight = 1.0 - pressure_upper_weight

    # --- beta <= 25.12 : bilinear-interp correction + near/asymptotic blend ---
    # Bracket beta, then interpolate in pressure at both beta nodes.
    beta_count = stark_beta_grid.numel()
    beta_upper_index = torch.searchsorted(stark_beta_grid, beta.contiguous())
    beta_upper_index = torch.clamp(beta_upper_index, 1, beta_count - 1)
    beta_lower_index = beta_upper_index - 1
    beta_interval = (
        stark_beta_grid[beta_upper_index] - stark_beta_grid[beta_lower_index]
    )
    beta_upper_weight = torch.where(
        beta_interval > 0,
        (beta - stark_beta_grid[beta_lower_index])
        / torch.where(beta_interval > 0, beta_interval, torch.ones_like(beta_interval)),
        torch.zeros_like(beta),
    )
    beta_lower_weight = 1.0 - beta_upper_weight

    depth_count = beta.shape[0]
    pressure_upper_table_index = (pressure_upper_index - 1).expand(
        depth_count, beta.shape[1]
    )
    pressure_lower_table_index = (pressure_lower_index - 1).expand(
        depth_count, beta.shape[1]
    )
    tab = stark_probability_table[table_index]  # [5, beta_count]
    # flat gather over the (pp, beta) plane
    nbcols = tab.shape[1]
    tab_flat = tab.reshape(-1)
    correction_upper_beta = (
        tab_flat[pressure_upper_table_index * nbcols + beta_upper_index]
        * pressure_upper_weight
        + tab_flat[pressure_lower_table_index * nbcols + beta_upper_index]
        * pressure_lower_weight
    )
    correction_lower_beta = (
        tab_flat[pressure_upper_table_index * nbcols + beta_lower_index]
        * pressure_upper_weight
        + tab_flat[pressure_lower_table_index * nbcols + beta_lower_index]
        * pressure_lower_weight
    )
    near_correction = (
        1.0
        + correction_upper_beta * beta_upper_weight
        + correction_lower_beta * beta_lower_weight
    )

    near_core_weight = torch.clamp(0.5 * (10.0 - beta), 0.0, 1.0)
    near_core_term = torch.where(
        beta <= 10.0,
        8.0 / (83.0 + (2.0 + 0.95 * beta_squared) * beta),
        torch.zeros_like(beta),
    )
    wing_asymptotic_term = torch.where(
        beta >= 8.0,
        (1.5 / sqrt_beta_safe + 27.0 / beta_squared_safe) / beta_squared_safe,
        torch.zeros_like(beta),
    )
    near_profile = (
        near_core_term * near_core_weight
        + wing_asymptotic_term * (1.0 - near_core_weight)
    ) * near_correction

    # --- 25.12 < beta <= 500 : asymptotic form * (c,d) correction ---
    # stark_wing_correction_c_table/stark_wing_correction_d_table shape (5,7): pressure row x transition column.
    stark_wing_correction_c = (
        stark_wing_correction_c_table[pressure_lower_index - 1, table_index]
        * pressure_upper_weight
        + stark_wing_correction_c_table[pressure_upper_index - 1, table_index]
        * pressure_lower_weight
    )
    stark_wing_correction_d = (
        stark_wing_correction_d_table[pressure_lower_index - 1, table_index]
        * pressure_upper_weight
        + stark_wing_correction_d_table[pressure_upper_index - 1, table_index]
        * pressure_lower_weight
    )
    wing_denominator = stark_wing_correction_c + beta * sqrt_beta
    wing_denominator = torch.where(
        wing_denominator == 0.0,
        torch.full_like(wing_denominator, 1e-30),
        wing_denominator,
    )
    wing_correction = 1.0 + stark_wing_correction_d / wing_denominator

    asymptotic_profile = (
        1.5 / sqrt_beta_safe + 27.0 / beta_squared_safe
    ) / beta_squared_safe

    # --- assemble the three regimes branchlessly ---
    regime_correction = torch.where(
        beta <= 25.12,
        torch.ones_like(beta),
        torch.where(beta <= 500.0, wing_correction, torch.ones_like(beta)),
    )
    profile = torch.where(
        beta <= 25.12,
        near_profile,
        asymptotic_profile * regime_correction,
    )
    profile = torch.where(beta > 0.0, profile, torch.zeros_like(profile))
    return profile


def _fast_ex_gauss(argument: torch.Tensor) -> torch.Tensor:
    """Guarded exp(-x) for the Gaussian core (zero past x = 80), branchless."""
    return torch.where(
        argument > 80.0, torch.zeros_like(argument), torch.exp(-argument)
    )


# Hydrogen Stark profile, vectorized over the wavelength grid for one transition.
@dataclass
class HydrogenProfileTables:
    """GPU-resident tables used by the hydrogen Stark profile."""

    radiative_damping_sums: torch.Tensor
    impact_electron_density_thresholds_cm3: torch.Tensor
    stark_knm_table: torch.Tensor
    stark_probability_table: torch.Tensor
    stark_wing_correction_c: torch.Tensor
    stark_wing_correction_d: torch.Tensor
    stark_pressure_grid: torch.Tensor
    stark_beta_grid: torch.Tensor


def _hydrogen_profile_grid(
    lower_level: int,
    upper_level: int,
    delta_lambda_nm: torch.Tensor,
    profile_state: dict,
    tables: HydrogenProfileTables,
    component_frequency_offsets_hz: torch.Tensor,
    component_weights: torch.Tensor,
    component_count: int,
) -> torch.Tensor:
    """Hydrogen Stark profile phi(Delta-lambda) for one transition.

    `delta_lambda_nm` is [D, wavelength_count] (wavelength offset of every grid pixel from the
    line center, per depth -- identical across depths but kept 2-D for broadcast).
    `profile_state` carries the per-depth state as [D,1] tensors. Returns
    [D, wavelength_count] profile values, normalized so line_strength * phi is opacity.
    """
    level_separation = upper_level - lower_level
    if level_separation <= 0:
        return torch.zeros_like(delta_lambda_nm)

    lower_level_float = float(lower_level)
    upper_level_float = float(upper_level)
    lower_level_squared = lower_level_float * lower_level_float
    upper_level_squared = upper_level_float * upper_level_float
    level_square_difference = upper_level_squared - lower_level_squared
    level_square_product = upper_level_squared * lower_level_squared
    transition_inverse_wavelength = level_square_difference / level_square_product
    if lower_level <= 4 and level_separation <= 3:
        stark_knm_constant = float(
            tables.stark_knm_table[lower_level - 1, level_separation - 1]
        )
    else:
        stark_knm_constant = (
            5.5e-5
            / transition_inverse_wavelength
            * level_square_product
            / (1.0 + 0.13 / float(level_separation))
        )
    line_frequency_hz = HYDROGEN_RYDBERG_FREQUENCY_HZ * transition_inverse_wavelength
    line_wavelength_angstrom = LIGHT_SPEED_ANGSTROM_PER_S / line_frequency_hz
    beta_frequency_scale = LIGHT_SPEED_ANGSTROM_PER_S / (
        line_frequency_hz * line_frequency_hz * stark_knm_constant
    )
    impact_linear_scale = (
        stark_knm_constant
        / line_wavelength_angstrom
        * transition_inverse_wavelength
        * level_square_difference
    )
    impact_quadratic_scale = (stark_knm_constant / line_wavelength_angstrom) ** 2

    radiative_table_size = tables.radiative_damping_sums.shape[0]
    if lower_level <= radiative_table_size and upper_level <= radiative_table_size:
        radiative_damping = float(
            tables.radiative_damping_sums[lower_level - 1]
            + tables.radiative_damping_sums[upper_level - 1]
        )
    elif lower_level <= radiative_table_size:
        radiative_damping = float(tables.radiative_damping_sums[lower_level - 1])
    else:
        radiative_damping = 0.0
    radiative_damping = radiative_damping / 12.5664 / line_frequency_hz
    resonance_coefficient = (
        _hf_nm(1, upper_level) / upper_level_float / (1.0 - 1.0 / upper_level_squared)
    )
    if lower_level != 1:
        resonance_coefficient += (
            _hf_nm(1, lower_level)
            / lower_level_float
            / (1.0 - 1.0 / lower_level_squared)
        )
    resonance_coefficient *= 3.579e-24 / transition_inverse_wavelength
    van_der_waals_coefficient = (
        4.45e-26
        / transition_inverse_wavelength
        * (upper_level_squared * (7.0 * upper_level_squared + 5.0)) ** 0.4
    )

    helium_perturber_density = profile_state["helium_perturber_density"]
    molecular_hydrogen_perturber_density = profile_state[
        "molecular_hydrogen_perturber_density"
    ]
    holtsmark_field = profile_state["holtsmark_field"]
    hydrogen_doppler_fraction = profile_state["hydrogen_doppler_fraction"]
    impact_linear_coefficient = profile_state["impact_linear_coefficient"]
    impact_quadratic_coefficient = profile_state["impact_quadratic_coefficient"]
    low_density_impact_factor = profile_state["low_density_impact_factor"]
    high_density_impact_factor = profile_state["high_density_impact_factor"]
    impact_correction_1 = profile_state["impact_correction_1"]
    impact_correction_2 = profile_state["impact_correction_2"]
    pressure_parameter = profile_state["pressure_parameter"]
    electron_density = profile_state["electron_density"]
    hydrogen_neutral_population = profile_state["hydrogen_neutral_population"]

    van_der_waals_half_width = (
        van_der_waals_coefficient * helium_perturber_density
        + 2.0 * van_der_waals_coefficient * molecular_hydrogen_perturber_density
    )
    radiative_half_width = radiative_damping
    stark_scale = 1.6678e-18 * line_frequency_hz * stark_knm_constant
    resonance_half_width = resonance_coefficient * hydrogen_neutral_population * 2.0
    stark_half_width = stark_scale * holtsmark_field
    lorentz_half_width = (
        resonance_half_width + van_der_waals_half_width + radiative_half_width
    )

    wavelength_angstrom = line_wavelength_angstrom + delta_lambda_nm * 10.0
    wavelength_angstrom_safe = torch.where(
        wavelength_angstrom > 0,
        wavelength_angstrom,
        torch.ones_like(wavelength_angstrom),
    )
    signed_frequency_offset = (
        -LIGHT_SPEED_ANGSTROM_PER_S
        * (delta_lambda_nm * 10.0)
        / (wavelength_angstrom_safe * line_wavelength_angstrom)
    )
    frequency_detuning = signed_frequency_offset.abs()
    doppler_fraction_safe = torch.clamp(hydrogen_doppler_fraction, min=1e-40)
    doppler_width_hz = line_frequency_hz * doppler_fraction_safe
    core_half_width_hz = line_frequency_hz * torch.maximum(
        torch.maximum(doppler_fraction_safe, lorentz_half_width),
        stark_half_width,
    )
    inside_core = frequency_detuning <= core_half_width_hz

    # Select the core approximation from the broadening mechanism that dominates
    # each depth; Stark is the fall-through case.
    doppler_dominant = (doppler_fraction_safe >= stark_half_width) & (
        doppler_fraction_safe >= lorentz_half_width
    )
    lorentz_at_least_stark = lorentz_half_width >= stark_half_width
    use_doppler_core = doppler_dominant  # [D,1]
    use_lorentz_core = (~doppler_dominant) & lorentz_at_least_stark

    # Sum the fine-structure components into the Doppler core.
    core = torch.zeros_like(frequency_detuning)
    doppler_width_safe = torch.clamp(doppler_width_hz, min=1e-30)
    for component_index in range(component_count):
        component_detuning = (
            signed_frequency_offset
            - float(component_frequency_offsets_hz[component_index])
        ).abs() / doppler_width_safe
        contrib = torch.where(
            component_detuning <= 7.0,
            _fast_ex_gauss(component_detuning * component_detuning)
            * float(component_weights[component_index]),
            torch.zeros_like(component_detuning),
        )
        core = core + contrib

    lorentz_width_hz = line_frequency_hz * lorentz_half_width
    frequency_detuning_squared = frequency_detuning * frequency_detuning
    lorentz = torch.where(
        lorentz_width_hz > 0.0,
        lorentz_width_hz
        / math.pi
        / (frequency_detuning_squared + lorentz_width_hz * lorentz_width_hz)
        * 1.77245
        * doppler_width_hz,
        torch.zeros_like(frequency_detuning),
    )

    # Electron-impact width gamma: impact-broadening Lorentzian in beta.
    if upper_level > 3:
        impact_density_numerator = 320.0
    elif upper_level == 2:
        impact_density_numerator = 550.0
    else:
        impact_density_numerator = 380.0
    impact_density_threshold = 1.0e14 if level_separation <= 3 else 1.0e13
    if (
        level_separation <= 2
        and 1 <= lower_level <= 2
        and lower_level <= tables.impact_electron_density_thresholds_cm3.shape[0]
        and level_separation <= tables.impact_electron_density_thresholds_cm3.shape[1]
    ):
        impact_density_threshold = float(
            tables.impact_electron_density_thresholds_cm3[
                lower_level - 1, level_separation - 1
            ]
        )
    safe_impact_density_threshold = max(impact_density_threshold, 1e-30)
    low_density_weight = 1.0 / (
        1.0 + torch.clamp(electron_density, min=0.0) / safe_impact_density_threshold
    )
    impact_scale = (
        impact_density_numerator * low_density_impact_factor * low_density_weight
        + high_density_impact_factor * (1.0 - low_density_weight)
    )

    linear_impact_term = impact_linear_coefficient * impact_linear_scale * impact_scale
    quadratic_impact_term = impact_quadratic_coefficient * impact_quadratic_scale
    holtsmark_field_safe = torch.clamp(holtsmark_field, min=1e-30)
    beta = frequency_detuning / holtsmark_field_safe * beta_frequency_scale
    linear_impact_argument = linear_impact_term * beta
    quadratic_impact_argument = quadratic_impact_term * beta * beta

    impact_gamma_scale = 6.77 * torch.sqrt(torch.clamp(linear_impact_term, min=1e-30))
    ratio = torch.where(
        (linear_impact_term > 0) & (quadratic_impact_term > 0),
        torch.sqrt(torch.clamp(quadratic_impact_term, min=0.0))
        / torch.clamp(linear_impact_term, min=1e-30),
        torch.zeros_like(linear_impact_term),
    )
    log_term = torch.where(
        ratio > 0, torch.log(torch.clamp(ratio, min=1e-30)), torch.zeros_like(ratio)
    )
    # Low-impact analytic gamma, broadcast from [D,1] to [D,wavelength_count].
    gamma_simple = (
        impact_gamma_scale
        * torch.clamp(0.2114 + log_term, min=0.0)
        * (1.0 - impact_correction_1 - impact_correction_2)
    )
    gamma_simple = gamma_simple.expand_as(beta)
    # Exponential-integral form for the high-impact branch.
    linear_exponential_integral = _first_exponential_integral_approximation(
        linear_impact_argument
    )
    quadratic_exponential_integral = _first_exponential_integral_approximation(
        quadratic_impact_argument
    )
    gamma_exponential_integral = (
        impact_gamma_scale
        * (
            0.5 * _fast_ex_gauss(torch.clamp(linear_impact_argument, max=80.0))
            + linear_exponential_integral
            - 0.5 * quadratic_exponential_integral
        )
        * (
            1.0
            - impact_correction_1 / (1.0 + (90.0 * linear_impact_argument) ** 3)
            - impact_correction_2 / (1.0 + 2000.0 * linear_impact_argument)
        )
    )
    use_exponential_integral = (quadratic_impact_argument > 1e-4) & (
        linear_impact_argument > 1e-5
    )
    gamma = torch.where(
        use_exponential_integral, gamma_exponential_integral, gamma_simple
    )
    impact_profile = torch.where(
        gamma > 0.0,
        gamma / math.pi / (gamma * gamma + beta * beta),
        torch.zeros_like(beta),
    )

    # Quasi-static Stark term plus the impact profile, normalized by F0.
    quasi_static_profile = _stark_quasi_static_profile(
        beta,
        pressure_parameter,
        lower_level,
        upper_level,
        tables.stark_probability_table,
        tables.stark_wing_correction_c,
        tables.stark_wing_correction_d,
        tables.stark_pressure_grid,
        tables.stark_beta_grid,
    )
    near_core_blend_square = (0.9 * linear_impact_argument) ** 2
    quasi_static_screening = (
        near_core_blend_square
        + 0.03 * torch.sqrt(torch.clamp(linear_impact_argument, min=0.0))
    ) / (near_core_blend_square + 1.0)
    stark_core = (
        (quasi_static_profile * (1.0 + quasi_static_screening) + impact_profile)
        / holtsmark_field_safe
        * beta_frequency_scale
        * 1.77245
        * doppler_width_hz
    )

    # In the core choose the dominant broadening mechanism; in the wing sum them.
    doppler_profile = torch.clamp(core, min=0.0)
    lorentz_profile = torch.clamp(lorentz, min=0.0)
    stark_profile = torch.clamp(stark_core, min=0.0)
    core_profile = torch.where(
        use_doppler_core,
        doppler_profile,
        torch.where(use_lorentz_core, lorentz_profile, stark_profile),
    )
    wing_profile = torch.clamp(core + lorentz + stark_core, min=0.0)
    profile = torch.where(inside_core, core_profile, wing_profile)
    profile = torch.where(wavelength_angstrom > 0, profile, torch.zeros_like(profile))
    return profile


def _first_exponential_integral_approximation(argument: torch.Tensor) -> torch.Tensor:
    """Vectorized piecewise approximation to the first exponential integral.

    Computes all four branches and `torch.where`-selects by the branch points
    (0.01, 1, 30). Inputs <= 0 -> 0. Matches the scalar reference bit-for-bit
    in fp64 (each branch is the identical arithmetic). (spec §7.5)
    """
    argument_safe = torch.clamp(argument, min=1e-300)
    log_argument = torch.log(argument_safe)
    small = -log_argument - 0.577215 + argument
    mid = (
        -log_argument
        - 0.57721566
        + argument
        * (
            0.99999193
            + argument
            * (
                -0.24991055
                + argument
                * (0.05519968 + argument * (-0.00976004 + argument * 0.00107857))
            )
        )
    )
    numerator = argument * (argument + 2.334733) + 0.25062
    denominator = (argument * (argument + 3.330657) + 1.681534) * argument
    safe_denominator = torch.where(
        denominator != 0,
        denominator,
        torch.ones_like(denominator),
    )
    large = (
        numerator
        / safe_denominator
        * torch.exp(
            -torch.clamp(argument, max=700.0),
        )
    )

    result = torch.where(
        argument <= 0.01,
        small,
        torch.where(
            argument <= 1.0,
            mid,
            torch.where(argument > 30.0, torch.zeros_like(argument), large),
        ),
    )
    result = torch.where(argument <= 0.0, torch.zeros_like(argument), result)
    return result


# Per-depth hydrogen state.
def _hydrogen_state(
    temperature,
    electron_density,
    helium_neutral_population,
    molecular_hydrogen_population,
    hydrogen_partition_normalized_ion_stage_populations,
    microturbulence,
    dtype,
    compute_device,
) -> dict:
    """Return small [depth, 1] Stark-profile coefficient arrays.

    This state is computed in host fp64 because pressure-parameter table
    bracketing is index-sensitive; the finished columns are then uploaded to the
    synthesis device.
    """
    temperature = torch.clamp(temperature.to(torch.float64), min=1.0)
    electron_density = torch.clamp(electron_density.to(torch.float64), min=1e-40)
    electron_density_sixth_root = electron_density ** (1.0 / 6.0)

    holtsmark_field = electron_density_sixth_root**4 * 1.25e-9
    pressure_parameter = electron_density_sixth_root * 0.08989 / torch.sqrt(temperature)
    high_density_impact_factor = 2.0 / (
        1.0 + 0.012 / temperature * torch.sqrt(electron_density / temperature)
    )
    temperature_perturber_scale = (temperature / 1.0e4) ** 0.3
    low_density_impact_factor = (
        temperature_perturber_scale / electron_density_sixth_root
    )
    impact_linear_coefficient = holtsmark_field * 78940.0 / temperature
    impact_quadratic_coefficient = holtsmark_field**2 / 5.96e-23 / electron_density
    impact_correction_1 = 0.2 + 0.09 * torch.sqrt(
        torch.clamp(temperature / 1e4, min=1e-12)
    ) / (1.0 + electron_density / 1.0e13)
    impact_correction_2 = 0.2 / (1.0 + electron_density / 1.0e15)

    thermal_velocity_fraction = (
        torch.sqrt(
            2.0
            * BOLTZMANN_ERG_PER_K
            * temperature
            / (HYDROGEN_ATOMIC_MASS_AMU * HYDROGEN_PROFILE_ATOMIC_MASS_GRAM)
        )
        / LIGHT_SPEED_CM_PER_S
    )
    turbulent_velocity_fraction = (
        microturbulence.to(torch.float64) / 1e5 / LIGHT_SPEED_KM_PER_S
    )
    hydrogen_doppler_fraction = torch.sqrt(
        thermal_velocity_fraction * thermal_velocity_fraction
        + turbulent_velocity_fraction * turbulent_velocity_fraction
    )

    def as_column_on_device(values):
        return values.to(dtype).to(compute_device)[:, None]

    return dict(
        helium_perturber_density=as_column_on_device(
            temperature_perturber_scale * helium_neutral_population.to(torch.float64)
        ),
        molecular_hydrogen_perturber_density=as_column_on_device(
            temperature_perturber_scale
            * molecular_hydrogen_population.to(torch.float64)
        ),
        holtsmark_field=as_column_on_device(holtsmark_field),
        hydrogen_doppler_fraction=as_column_on_device(hydrogen_doppler_fraction),
        impact_linear_coefficient=as_column_on_device(impact_linear_coefficient),
        impact_quadratic_coefficient=as_column_on_device(impact_quadratic_coefficient),
        low_density_impact_factor=as_column_on_device(low_density_impact_factor),
        high_density_impact_factor=as_column_on_device(high_density_impact_factor),
        impact_correction_1=as_column_on_device(impact_correction_1),
        impact_correction_2=as_column_on_device(impact_correction_2),
        pressure_parameter=as_column_on_device(pressure_parameter),
        electron_density=as_column_on_device(electron_density),
        hydrogen_neutral_population=as_column_on_device(
            hydrogen_partition_normalized_ion_stage_populations[:, 0].to(torch.float64)
        ),
    )


def _center_index(wavelength_grid: np.ndarray, value: float) -> int:
    """Return the nearest log-grid index relative to the first wavelength cell."""
    log_grid_step = np.log(wavelength_grid[1] / wavelength_grid[0])
    origin_index = int(np.log(wavelength_grid[0]) / log_grid_step + 0.5)
    return int(np.log(value) / log_grid_step + 0.5) - origin_index


def _merged_continuum_limits(
    series_limit_wavelength_nm: float,
    last_resolved_upper_level: int,
    merge_wavenumber_by_depth: np.ndarray,
):
    """Return per-depth plateau and taper limits for a series-limit record."""
    rydberg_wavenumber = _HYDROGEN_RYDBERG_WAVENUMBER_CM
    denominator_shift = 1.0e7 / series_limit_wavelength_nm - rydberg_wavenumber / float(
        last_resolved_upper_level * last_resolved_upper_level
    )
    shifted_series_limit_wavelength_nm = (
        1.0e7 / denominator_shift
        if abs(denominator_shift) > 1e-12
        else (series_limit_wavelength_nm + series_limit_wavelength_nm)
    )
    denominator_merge = 1.0e7 / series_limit_wavelength_nm - merge_wavenumber_by_depth
    safe_denominator_merge = np.where(
        np.abs(denominator_merge) > 1e-12,
        denominator_merge,
        np.ones_like(denominator_merge),
    )
    merge_wavelength_nm = np.where(
        np.abs(denominator_merge) > 1e-12,
        1.0e7 / safe_denominator_merge,
        shifted_series_limit_wavelength_nm + shifted_series_limit_wavelength_nm,
    )
    merge_wavelength_nm = np.where(
        merge_wavelength_nm < 0.0,
        shifted_series_limit_wavelength_nm + shifted_series_limit_wavelength_nm,
        merge_wavelength_nm,
    )
    merge_wavelength_nm = np.maximum(
        merge_wavelength_nm,
        shifted_series_limit_wavelength_nm,
    )
    merge_wavelength_nm = np.minimum(
        shifted_series_limit_wavelength_nm + shifted_series_limit_wavelength_nm,
        merge_wavelength_nm,
    )
    denominator_tail = 1.0e7 / merge_wavelength_nm - 500.0
    safe_denominator_tail = np.where(
        np.abs(denominator_tail) > 1e-12,
        denominator_tail,
        np.ones_like(denominator_tail),
    )
    tail_wavelength_nm = np.where(
        np.abs(denominator_tail) > 1e-12,
        1.0e7 / safe_denominator_tail,
        merge_wavelength_nm + merge_wavelength_nm,
    )
    tail_wavelength_nm = np.where(
        tail_wavelength_nm < 0.0,
        merge_wavelength_nm + merge_wavelength_nm,
        tail_wavelength_nm,
    )
    tail_wavelength_nm = np.minimum(
        merge_wavelength_nm + merge_wavelength_nm,
        tail_wavelength_nm,
    )
    return merge_wavelength_nm, tail_wavelength_nm


def _line_continuum_merge_limits(
    merge_wavenumber_by_depth: np.ndarray,
    series_limit_wavenumber_cm: float,
    shifted_series_limit_wavelength_nm: float,
):
    """Return the per-depth merge wavelength and taper end for a resolved line."""
    denominator = series_limit_wavenumber_cm - merge_wavenumber_by_depth
    merge_wavelength_nm = np.where(
        denominator > 0.0,
        1.0e7 / np.where(denominator > 0, denominator, 1.0),
        shifted_series_limit_wavelength_nm + shifted_series_limit_wavelength_nm,
    )
    continuum_merge_wavelength_nm = np.maximum(
        shifted_series_limit_wavelength_nm,
        merge_wavelength_nm,
    )
    tail_inner_wavenumber = np.where(
        continuum_merge_wavelength_nm > 0.0,
        1.0e7
        / np.where(
            continuum_merge_wavelength_nm > 0,
            continuum_merge_wavelength_nm,
            1.0,
        )
        - 500.0,
        -1.0,
    )
    taper_end_wavelength_nm = np.where(
        tail_inner_wavenumber > 0.0,
        1.0e7 / np.where(tail_inner_wavenumber > 0, tail_inner_wavenumber, 1.0),
        continuum_merge_wavelength_nm + continuum_merge_wavelength_nm,
    )
    continuum_merge_wavelength_nm = np.minimum(
        shifted_series_limit_wavelength_nm + shifted_series_limit_wavelength_nm,
        continuum_merge_wavelength_nm,
    )
    taper_end_wavelength_nm = np.where(
        taper_end_wavelength_nm < 0.0,
        continuum_merge_wavelength_nm + continuum_merge_wavelength_nm,
        taper_end_wavelength_nm,
    )
    taper_end_wavelength_nm = np.minimum(
        continuum_merge_wavelength_nm + continuum_merge_wavelength_nm,
        taper_end_wavelength_nm,
    )
    return continuum_merge_wavelength_nm, taper_end_wavelength_nm


@dataclass
class HydrogenLine:
    """Per-line static data for one Balmer line (spec §7.4 INVARIANT)."""

    n_lower: int
    n_upper: int
    line_wavelength_nm: float
    center_index: int
    classical_strength: float
    lower_excitation_cm: float
    simple: bool
    series_limit_wavenumber_cm: float
    shifted_series_limit_wavelength_nm: float
    component_frequency_offsets: torch.Tensor
    component_weights: torch.Tensor
    component_count: int
    red_neighbor_wavelength_nm: float
    far_red_neighbor_wavelength_nm: float
    blue_neighbor_wavelength_nm: float
    far_blue_neighbor_wavelength_nm: float
    red_dominance_cutoff_nm: float
    blue_dominance_cutoff_nm: float


@dataclass
class HydrogenMergedCont:
    """Hydrogen series-limit pseudo-continuum record."""

    n_lower: int
    series_limit_wavelength_nm: float
    merged_continuum_strength: float
    lower_excitation_cm: float
    last_resolved_upper_level: int


@dataclass
class HydrogenInvariants:
    """GPU-resident invariant block for the hydrogen Stark engine."""

    wavelength_grid: torch.Tensor  # fp64 wavelengths on device
    wavelength_grid_host: np.ndarray  # fp64 wavelengths on host, for index math
    wavelength_count: int
    lines: list  # list[HydrogenLine]
    merged: list  # list[HydrogenMergedCont] (series-limit pseudo-continua)
    tables: HydrogenProfileTables
    merge_wavenumber_by_depth: np.ndarray  # Inglis-Teller merge wavenumber per depth
    component_map: dict  # (n_lower, n_upper) -> component tensors
    exponential_integer_table: torch.Tensor
    exponential_fraction_table: torch.Tensor


def _component_tensors_for_transition(
    invariants: "HydrogenInvariants",
    lower_level: int,
    upper_level: int,
):
    """Fine-structure component tensors for one transition.

    A neighbor transition without tabulated fine structure contributes no
    Doppler core.
    """
    components = invariants.component_map.get((lower_level, upper_level))
    if components is None:
        zeros = invariants.wavelength_grid.new_zeros(1)
        return zeros, zeros, 0
    return components


def merge_wavenumber_by_depth(electron_density) -> np.ndarray:
    """Inglis-Teller merge wavenumber per depth from the electron density.

    This is the only atmosphere-dependent field of ``HydrogenInvariants``;
    the window-invariant cache rebuilds it per atmosphere through this helper
    so the cached template never carries per-star state.
    """
    electron_density_floor = np.maximum(
        np.asarray(electron_density, np.float64),
        1e-40,
    )
    inglis_teller_level = 1600.0 / np.power(electron_density_floor, 2.0 / 15.0)
    merge_level = np.maximum(inglis_teller_level - 1.5, 1.0)
    return _HYDROGEN_RYDBERG_WAVENUMBER_CM / np.maximum(
        merge_level * merge_level, 1e-12
    )


def precompute_invariants(
    catalog,
    wavelength_grid_nm,
    electron_density,
    compute_device=None,
) -> HydrogenInvariants:
    """Build the GPU-resident hydrogen invariant block from the catalog (once).

    `catalog` is the atomic line-catalog mapping; `wavelength_grid_nm` is the
    synthesis grid; `electron_density` sets the per-depth Inglis-Teller merge
    level used by the series-limit pseudo-continuum records.

    Routes type -1/-2, neutral (ion 1) Balmer lines (n_lower >= 2) here.
    """
    compute_device = compute_device if compute_device is not None else device()
    profile_dtype = highp_dtype(compute_device)
    wavelength_grid = np.asarray(wavelength_grid_nm, np.float64)
    wavelength_count = wavelength_grid.size

    line_type = catalog["line_type"].astype(np.int64)
    ion_stage = catalog["ion_stage"].astype(np.int64)
    lower_level_all = catalog["lower_principal_quantum_number"].astype(np.int64)
    upper_level_all = catalog["upper_principal_quantum_number"].astype(np.int64)

    # Hydrogen records split into resolved Stark lines and series-limit merged
    # continuum records; the latter use n_upper == 99 in the packaged catalog.
    hydrogen_record_mask = np.isin(line_type, [-1, -2]) & (ion_stage == 1)
    merged_continuum_mask = hydrogen_record_mask & (upper_level_all == 99)
    line_indices = np.where(hydrogen_record_mask & ~merged_continuum_mask)[0]
    merged_indices = np.where(merged_continuum_mask)[0]

    wavelength_nm_all = np.asarray(catalog["wavelength_nm"], np.float64)
    index_wavelength_nm_all = np.asarray(catalog["index_wavelength_nm"], np.float64)
    oscillator_strength_all = np.asarray(catalog["oscillator_strength"], np.float64)
    lower_excitation_cm_all = np.asarray(catalog["lower_excitation_cm"], np.float64)
    hydrogen_continuum_edges = np.asarray(
        catalog["hydrogen_continuum_edges"],
        np.float64,
    )

    # Fine-structure components are computed from embedded hydrogen tables for
    # every (n_lower, n_upper) pair.  Packaged narrow-window catalogs only carried
    # precomputed entries for a few beta/gamma/delta lines, so computing here is
    # the general full-optical path and keeps Halpha/Palpha cores populated.
    def _component_entry(lower_level: int, upper_level: int):
        component_offsets_hz, component_weights = _fine_structure(
            lower_level,
            upper_level,
        )
        return component_offsets_hz, component_weights, int(component_offsets_hz.size)

    # Inglis-Teller merge level -> merge wavenumber per depth
    hydrogen_merge_wavenumber = merge_wavenumber_by_depth(electron_density)

    lines = []
    for catalog_index in line_indices:
        line_wavelength_nm = float(wavelength_nm_all[catalog_index])
        center_index = _center_index(
            wavelength_grid,
            float(index_wavelength_nm_all[catalog_index]),
        )
        classical_strength = (
            CLASSICAL_LINE_STRENGTH_COEFFICIENT
            * float(oscillator_strength_all[catalog_index])
            / (LIGHT_SPEED_NM_PER_S / line_wavelength_nm)
        )
        lower_level = max(int(lower_level_all[catalog_index]), 1)
        upper_level = max(int(upper_level_all[catalog_index]), lower_level + 1)
        if lower_level < 2:
            # Lyman lines are out of scope for the optical gate; skip with a guard.
            raise NotImplementedError(
                f"Lyman line n_lower={lower_level} not supported (Balmer engine only)"
            )
        lower_energy_cm = _hydrogen_level_energy_cm(lower_level)
        red_neighbor_wavelength_nm = (
            1.0e7 / (_hydrogen_level_energy_cm(upper_level - 1) - lower_energy_cm)
            if upper_level - 1 > lower_level
            else line_wavelength_nm
        )
        far_red_neighbor_wavelength_nm = (
            1.0e7 / (_hydrogen_level_energy_cm(upper_level - 2) - lower_energy_cm)
            if upper_level - 2 > lower_level
            else line_wavelength_nm
        )
        blue_neighbor_wavelength_nm = 1.0e7 / (
            _hydrogen_level_energy_cm(upper_level + 1) - lower_energy_cm
        )
        far_blue_neighbor_wavelength_nm = 1.0e7 / (
            _hydrogen_level_energy_cm(upper_level + 2) - lower_energy_cm
        )
        red_dominance_cutoff_nm = 1.0e7 / (
            hydrogen_continuum_edges[0]
            - _HYDROGEN_RYDBERG_WAVENUMBER_CM / (upper_level - 0.8) ** 2
            - lower_energy_cm
        )
        blue_dominance_cutoff_nm = 1.0e7 / (
            hydrogen_continuum_edges[0]
            - _HYDROGEN_RYDBERG_WAVENUMBER_CM / (upper_level + 0.8) ** 2
            - lower_energy_cm
        )
        continuum_edge_cm = float(
            hydrogen_continuum_edges[
                max(1, min(lower_level, hydrogen_continuum_edges.size)) - 1
            ]
        )
        shifted_limit_wavelength_nm = 1.0e7 / (
            continuum_edge_cm - _HYDROGEN_RYDBERG_WAVENUMBER_CM / 81.0**2
        )
        component_offsets_hz, component_weights, component_count = _component_entry(
            lower_level,
            upper_level,
        )
        lines.append(
            HydrogenLine(
                n_lower=lower_level,
                n_upper=upper_level,
                line_wavelength_nm=line_wavelength_nm,
                center_index=center_index,
                classical_strength=classical_strength,
                lower_excitation_cm=float(lower_excitation_cm_all[catalog_index]),
                simple=(upper_level <= lower_level + 2),
                series_limit_wavenumber_cm=continuum_edge_cm,
                shifted_series_limit_wavelength_nm=shifted_limit_wavelength_nm,
                component_frequency_offsets=torch.as_tensor(
                    component_offsets_hz,
                    dtype=profile_dtype,
                ).to(compute_device),
                component_weights=torch.as_tensor(
                    component_weights,
                    dtype=profile_dtype,
                ).to(compute_device),
                component_count=component_count,
                red_neighbor_wavelength_nm=red_neighbor_wavelength_nm,
                far_red_neighbor_wavelength_nm=far_red_neighbor_wavelength_nm,
                blue_neighbor_wavelength_nm=blue_neighbor_wavelength_nm,
                far_blue_neighbor_wavelength_nm=far_blue_neighbor_wavelength_nm,
                red_dominance_cutoff_nm=red_dominance_cutoff_nm,
                blue_dominance_cutoff_nm=blue_dominance_cutoff_nm,
            )
        )

    merged = []
    for catalog_index in merged_indices:
        lower_level = max(int(lower_level_all[catalog_index]), 1)
        merged.append(
            HydrogenMergedCont(
                n_lower=lower_level,
                series_limit_wavelength_nm=float(wavelength_nm_all[catalog_index]),
                merged_continuum_strength=(
                    float(oscillator_strength_all[catalog_index])
                    * 2.0
                    * float(lower_level)
                    * float(lower_level)
                ),
                lower_excitation_cm=float(lower_excitation_cm_all[catalog_index]),
                last_resolved_upper_level=81,
            )
        )

    def as_device_table(values):
        return torch.as_tensor(
            np.asarray(values, np.float64),
            dtype=profile_dtype,
        ).to(compute_device)

    tables = HydrogenProfileTables(
        radiative_damping_sums=as_device_table(catalog["radiative_damping_sums"]),
        impact_electron_density_thresholds_cm3=as_device_table(
            catalog["impact_electron_density_thresholds_cm3"]
        ),
        stark_knm_table=as_device_table(catalog["stark_knm_table"]),
        stark_probability_table=as_device_table(catalog["stark_probability_table"]),
        stark_wing_correction_c=as_device_table(catalog["stark_wing_correction_c"]),
        stark_wing_correction_d=as_device_table(catalog["stark_wing_correction_d"]),
        stark_pressure_grid=as_device_table(catalog["stark_pressure_grid"]),
        stark_beta_grid=as_device_table(catalog["stark_beta_grid"]),
    )
    exponential_integer_table, exponential_fraction_table = _fastex_tables(
        compute_device,
        profile_dtype,
    )

    # Neighbor wing tests reuse the same component tensors as the main line.
    component_transition_keys = set()
    for line in lines:
        lower_level, upper_level = line.n_lower, line.n_upper
        component_transition_keys.add((lower_level, upper_level))
        component_transition_keys.add(
            (lower_level, max(upper_level - 2, lower_level + 1))
        )
        component_transition_keys.add((lower_level, upper_level + 2))
    component_map_tensors = {}
    for lower_level, upper_level in component_transition_keys:
        component_offsets_hz, component_weights, component_count = _component_entry(
            lower_level,
            upper_level,
        )
        component_map_tensors[(lower_level, upper_level)] = (
            torch.as_tensor(component_offsets_hz, dtype=profile_dtype).to(
                compute_device
            ),
            torch.as_tensor(component_weights, dtype=profile_dtype).to(compute_device),
            component_count,
        )

    return HydrogenInvariants(
        wavelength_grid=torch.as_tensor(wavelength_grid, dtype=profile_dtype).to(
            compute_device
        ),
        wavelength_grid_host=wavelength_grid,
        wavelength_count=wavelength_count,
        lines=lines,
        merged=merged,
        tables=tables,
        merge_wavenumber_by_depth=hydrogen_merge_wavenumber,
        component_map=component_map_tensors,
        exponential_integer_table=exponential_integer_table,
        exponential_fraction_table=exponential_fraction_table,
    )


# Outward line-wing walk.  Each depth deposits a contiguous run until the first
# cutoff, merge-boundary, or stronger-neighbor stop.
def _run_mask(terminates_walk, start_offset):
    """Return the deposit mask for a depth-batched outward walk."""
    depth_count, step_count = terminates_walk.shape
    compute_device = terminates_walk.device
    step_positions = torch.arange(step_count, device=compute_device)[None, :]
    entered_region = step_positions >= start_offset[:, None]
    effective_stop = terminates_walk & entered_region
    run_is_open = torch.cumprod((~effective_stop).to(torch.int8), dim=1).to(torch.bool)
    return entered_region & run_is_open & (~terminates_walk)


def _deposit_side(
    line_mass_absorption_coefficient,
    columns,
    deposit_values,
    cutoff_values,
    is_simple_line,
    continuum_merge_mask,
    neighbor_values,
    neighbor_stop_mask,
    skip_below_merge,
    stop_below_merge,
    start_offset,
):
    """Deposit one wing side, with columns already ordered outward from center."""
    step_count = columns.numel()
    if step_count == 0:
        return
    merge_mask_window = (
        continuum_merge_mask[:, columns]
        if not is_simple_line
        else torch.zeros_like(deposit_values, dtype=torch.bool)
    )

    below_cutoff = (deposit_values < cutoff_values) | (deposit_values <= 0.0)
    neighbor_dominates = neighbor_stop_mask & (neighbor_values >= deposit_values)
    terminates_walk = below_cutoff | neighbor_dominates
    if stop_below_merge:
        terminates_walk = terminates_walk | merge_mask_window
    if skip_below_merge:
        terminates_walk = terminates_walk & (~merge_mask_window)

    deposit_mask = _run_mask(terminates_walk, start_offset)
    if skip_below_merge:
        deposit_mask = deposit_mask & (~merge_mask_window)
    deposit_mask = deposit_mask & (~neighbor_dominates)
    line_mass_absorption_coefficient[:, columns] = line_mass_absorption_coefficient[
        :, columns
    ] + torch.where(
        deposit_mask,
        deposit_values,
        torch.zeros_like(deposit_values),
    )


def _deposit_walk(
    line_mass_absorption_coefficient,
    deposit_values,
    cutoff_values,
    center_index,
    window_width,
    window_wavelength_grid,
    line,
    continuum_merge_mask,
    red_neighbor_values,
    blue_neighbor_values,
    active_depth,
):
    """Deposit one line window by the center-out red/blue cutoff walk.

    Red-side pixels below the continuum-merge wavelength are transparent skips;
    blue-side pixels below that boundary stop the walk.
    """
    depth_count = line_mass_absorption_coefficient.shape[0]
    compute_device = line_mass_absorption_coefficient.device
    is_simple_line = line.simple

    if 0 <= center_index < window_width:
        center_below_cutoff = (
            deposit_values[:, center_index] < cutoff_values[:, center_index]
        ) | (deposit_values[:, center_index] <= 0.0)
        center_deposit_mask = (~center_below_cutoff) & active_depth
        if not is_simple_line:
            center_deposit_mask = center_deposit_mask & (
                ~continuum_merge_mask[:, center_index]
            )
        line_mass_absorption_coefficient[:, center_index] = (
            line_mass_absorption_coefficient[:, center_index]
            + torch.where(
                center_deposit_mask,
                deposit_values[:, center_index],
                torch.zeros_like(deposit_values[:, center_index]),
            )
        )

    # --- RED wing: walk order = increasing grid index ---
    red_start = max(center_index + 1, 0)
    if red_start < window_width:
        red_columns = torch.arange(red_start, window_width, device=compute_device)
        red_wavelengths = window_wavelength_grid[red_columns][None, :].expand(
            depth_count, -1
        )
        if not is_simple_line:
            red_neighbor_limit_stop = red_wavelengths > line.red_neighbor_wavelength_nm
            red_neighbor_stop_mask = red_wavelengths > line.red_dominance_cutoff_nm
        else:
            red_neighbor_limit_stop = torch.zeros_like(
                red_wavelengths, dtype=torch.bool
            )
            red_neighbor_stop_mask = torch.zeros_like(red_wavelengths, dtype=torch.bool)
        red_values = deposit_values[:, red_columns]
        red_cutoff_values = cutoff_values[:, red_columns]
        red_neighbor_values = (
            red_neighbor_values[:, red_columns]
            if red_neighbor_values is not None
            else torch.zeros_like(red_values)
        )
        red_cutoff_values = torch.where(
            red_neighbor_limit_stop,
            torch.full_like(red_cutoff_values, float("inf")),
            red_cutoff_values,
        )
        red_cutoff_values = torch.where(
            active_depth[:, None],
            red_cutoff_values,
            torch.full_like(red_cutoff_values, float("inf")),
        )
        red_start_offset = torch.zeros(
            depth_count, device=compute_device, dtype=torch.int64
        )
        _deposit_side(
            line_mass_absorption_coefficient,
            red_columns,
            red_values,
            red_cutoff_values,
            is_simple_line,
            continuum_merge_mask,
            red_neighbor_values,
            red_neighbor_stop_mask,
            skip_below_merge=(not is_simple_line),
            stop_below_merge=False,
            start_offset=red_start_offset,
        )

    # --- BLUE wing: walk order = decreasing grid index ---
    blue_start = min(center_index - 1, window_width - 1)
    if blue_start >= 0:
        blue_columns = torch.arange(blue_start, -1, -1, device=compute_device)
        blue_wavelengths = window_wavelength_grid[blue_columns][None, :].expand(
            depth_count, -1
        )
        blue_values = deposit_values[:, blue_columns]
        blue_cutoff_values = cutoff_values[:, blue_columns]
        if not is_simple_line:
            blue_neighbor_limit_stop = (
                blue_wavelengths < line.blue_neighbor_wavelength_nm
            )
            blue_neighbor_stop_mask = blue_wavelengths < line.blue_dominance_cutoff_nm
            blue_neighbor_values = (
                blue_neighbor_values[:, blue_columns]
                if blue_neighbor_values is not None
                else torch.zeros_like(blue_values)
            )
            blue_cutoff_values = torch.where(
                blue_neighbor_limit_stop,
                torch.full_like(blue_cutoff_values, float("inf")),
                blue_cutoff_values,
            )
            stop_below_merge = True
        else:
            blue_neighbor_stop_mask = torch.zeros_like(
                blue_wavelengths, dtype=torch.bool
            )
            blue_neighbor_values = torch.zeros_like(blue_values)
            stop_below_merge = False
        blue_cutoff_values = torch.where(
            active_depth[:, None],
            blue_cutoff_values,
            torch.full_like(blue_cutoff_values, float("inf")),
        )
        blue_start_offset = torch.zeros(
            depth_count, device=compute_device, dtype=torch.int64
        )
        _deposit_side(
            line_mass_absorption_coefficient,
            blue_columns,
            blue_values,
            blue_cutoff_values,
            is_simple_line,
            continuum_merge_mask,
            blue_neighbor_values,
            blue_neighbor_stop_mask,
            skip_below_merge=False,
            stop_below_merge=stop_below_merge,
            start_offset=blue_start_offset,
        )


# Hydrogen reach windows.
#
# Stark wings are wide but deposit only into a contiguous run around each line
# center. The run ends once the opacity drops below the shared line cutoff at
# every depth, so evaluating a bounded window avoids the dense full-grid profile
# cost while preserving the deposited pixels.

_LINE_REACH_INITIAL_HALF_WINDOW = 64  # smallest half-window probed
_LINE_REACH_EDGE_BLOCK_SIZE = 16  # sub-cutoff edge block required before growth stops
_LINE_REACH_GROWTH_FACTOR = (
    2.0  # forward-only doubling factor; smaller factors are unsafe
)


def _reach_window(
    invariants,
    line,
    line_amplitude,
    passcut,
    profile_state,
    stimulated_emission_factor,
    continuum_opacity,
    wavelength_grid,
    compute_device,
):
    """Return a conservative half-window, in grid pixels, for one line.

    The edge block is evaluated on a forward doubling ladder. Growth stops only
    when an on-grid edge block is below cutoff at every depth. This deliberately
    overshoots the true deposit region instead of trying a bisection that could
    miss non-monotone depth unions.
    """
    center_index = line.center_index
    wavelength_count = invariants.wavelength_count
    max_grid_offset = max(
        center_index,
        (wavelength_count - 1) - center_index,
        0,
    )
    if max_grid_offset == 0:
        return 0

    # Evaluate all ladder edge blocks in one call; this avoids one kernel launch
    # per growth step on accelerator backends.
    rungs = []
    current_rung_radius = min(_LINE_REACH_INITIAL_HALF_WINDOW, max_grid_offset)
    while True:
        rungs.append(current_rung_radius)
        if current_rung_radius >= max_grid_offset:
            break
        current_rung_radius = min(
            max(
                current_rung_radius + _LINE_REACH_EDGE_BLOCK_SIZE,
                int(current_rung_radius * _LINE_REACH_GROWTH_FACTOR),
            ),
            max_grid_offset,
        )

    # Keep the rung id for each edge column so one batched profile evaluation can
    # reduce back to a stop decision per rung.
    edge_column_blocks, rung_index_blocks = [], []
    for rung_index, rung_radius in enumerate(rungs):
        block_start = max(rung_radius - _LINE_REACH_EDGE_BLOCK_SIZE + 1, 1)
        offsets = torch.arange(block_start, rung_radius + 1, device=compute_device)
        edge_columns = torch.cat([center_index + offsets, center_index - offsets])
        edge_columns = edge_columns[
            (edge_columns >= 0) & (edge_columns < wavelength_count)
        ]
        edge_column_blocks.append(edge_columns)
        rung_index_blocks.append(
            torch.full(
                (edge_columns.numel(),),
                rung_index,
                dtype=torch.int64,
                device=compute_device,
            )
        )
    edge_columns = (
        torch.cat(edge_column_blocks)
        if edge_column_blocks
        else torch.empty(0, dtype=torch.int64, device=compute_device)
    )
    rung_index_by_column = (
        torch.cat(rung_index_blocks)
        if rung_index_blocks
        else torch.empty(0, dtype=torch.int64, device=compute_device)
    )

    rung_above = torch.zeros(len(rungs), dtype=torch.bool, device=compute_device)
    if edge_columns.numel():
        wavelength_offset_nm = (
            wavelength_grid[edge_columns][None, :] - line.line_wavelength_nm
        ).expand(stimulated_emission_factor.shape[0], -1)
        profile = _hydrogen_profile_grid(
            line.n_lower,
            line.n_upper,
            wavelength_offset_nm,
            profile_state,
            invariants.tables,
            line.component_frequency_offsets,
            line.component_weights,
            line.component_count,
        )
        edge_opacity = (
            line_amplitude[:, None]
            * profile
            * stimulated_emission_factor[:, edge_columns]
        )
        edge_column_above_cutoff = (
            (
                edge_opacity
                >= continuum_opacity[:, edge_columns] * LINE_CENTER_CUTOFF_RATIO
            )
            & passcut[:, None]
        ).any(dim=0)
        rung_above.index_put_(
            (rung_index_by_column,),
            edge_column_above_cutoff,
            accumulate=True,
        )

    # Stop at the first on-grid edge block that is below cutoff. Entirely
    # off-grid edge blocks keep growing because the gap to the grid is unknown.
    rung_has_on_grid_columns = torch.tensor(
        [columns.numel() > 0 for columns in edge_column_blocks],
        device=compute_device,
    )
    stop_rung = rung_has_on_grid_columns & (~rung_above)
    if bool(stop_rung.any()):
        reach = int(rungs[int(torch.argmax(stop_rung.to(torch.int8)))])
    else:
        reach = int(rungs[-1])  # never cleared -> the grid edge bounds it
    return reach


def accumulate_hydrogen(
    invariants: HydrogenInvariants, state: dict, apply_stim: bool = True
) -> torch.Tensor:
    """Accumulate hydrogen Stark-line opacity on the synthesis grid."""
    compute_device = invariants.wavelength_grid.device
    profile_dtype = invariants.wavelength_grid.dtype
    depth_count = state["temperature"].shape[0]
    wavelength_count = invariants.wavelength_count

    mass_density = to_dev(state["mass_density"], profile_dtype, compute_device)
    hc_over_kt = to_dev(
        state["hc_over_kt"],
        profile_dtype,
        compute_device,
    )
    hydrogen_neutral_partition_normalized_population = to_dev(
        state["hydrogen_neutral_partition_normalized_population"],
        profile_dtype,
        compute_device,
    )
    hydrogen_fractional_doppler_width = to_dev(
        state["hydrogen_fractional_doppler_width"],
        profile_dtype,
        compute_device,
    )
    continuum_opacity = to_dev(
        state["continuum_opacity"], profile_dtype, compute_device
    )

    # Accumulator dtype = the device's highest float (fp64 on CPU -> machine
    # precision; fp32 on MPS -> the format floor). fp16/bf16 are banned from the
    # accumulation (the wide H wings vanish under the deep-layer floor).
    accumulator_dtype = profile_dtype
    hydrogen_opacity = torch.zeros(
        (depth_count, wavelength_count),
        dtype=accumulator_dtype,
        device=compute_device,
    )

    if not invariants.lines and not invariants.merged:
        return (
            hydrogen_opacity.to(ACCUMULATION_DTYPE)
            if compute_device.type == "mps"
            else hydrogen_opacity
        )

    # stimulated-emission factor on the grid, per depth (spec §1.7)
    wavelength_grid = invariants.wavelength_grid
    temperature = to_dev(state["temperature"], profile_dtype, compute_device)
    photon_temperature_factor = PLANCK_ERG_SECOND / (BOLTZMANN_ERG_PER_K * temperature)
    stimulated_emission_factor = 1.0 - torch.exp(
        -(LIGHT_SPEED_NM_PER_S / wavelength_grid)[None, :]
        * photon_temperature_factor[:, None]
    )

    # per-depth hydrogen profile state ([D,1] tensors). Computed in fp64 on host
    # (index-sensitive; ~80 values) then uploaded -- see _hydrogen_state.
    import numpy as _np

    def _h(key):
        return torch.as_tensor(_np.asarray(state[key], _np.float64))

    profile_state = _hydrogen_state(
        _h("temperature"),
        _h("electron_density"),
        _h("helium_neutral_population"),
        _h("molecular_hydrogen_population"),
        _h("hydrogen_partition_normalized_ion_stage_populations"),
        _h("microturbulence"),
        profile_dtype,
        compute_device,
    )

    # Build all line/depth amplitudes at once, then evaluate profiles only for
    # lines with at least one depth that clears the center cutoff.
    active_depth = (
        (hydrogen_neutral_partition_normalized_population > 0)
        & (hydrogen_fractional_doppler_width > 0)
        & (mass_density > 0)
    )
    doppler_width_safe = torch.where(
        hydrogen_fractional_doppler_width > 0,
        hydrogen_fractional_doppler_width,
        torch.ones_like(hydrogen_fractional_doppler_width),
    )
    mass_density_safe = torch.where(
        mass_density > 0,
        mass_density,
        torch.ones_like(mass_density),
    )
    population_doppler_ratio = torch.where(
        active_depth,
        hydrogen_neutral_partition_normalized_population
        / (mass_density_safe * doppler_width_safe),
        torch.zeros_like(hydrogen_neutral_partition_normalized_population),
    )

    classical_strength = torch.as_tensor(
        [line_record.classical_strength for line_record in invariants.lines],
        dtype=profile_dtype,
        device=compute_device,
    )
    lower_excitation_cm = torch.as_tensor(
        [line_record.lower_excitation_cm for line_record in invariants.lines],
        dtype=profile_dtype,
        device=compute_device,
    )
    center_columns = torch.as_tensor(
        [
            max(0, min(line_record.center_index, wavelength_count - 1))
            for line_record in invariants.lines
        ],
        dtype=torch.int64,
        device=compute_device,
    )

    pre_excitation_strength = (
        classical_strength[:, None] * population_doppler_ratio[None, :]
    )
    excitation_weight = fast_ex(
        lower_excitation_cm[:, None] * hc_over_kt[None, :],
        invariants.exponential_integer_table,
        invariants.exponential_fraction_table,
    )
    line_amplitude_all = pre_excitation_strength * excitation_weight
    center_cutoff = (
        continuum_opacity[:, center_columns].transpose(0, 1) * LINE_CENTER_CUTOFF_RATIO
    )
    passes_center_cutoff_all = (
        active_depth[None, :]
        & (pre_excitation_strength >= center_cutoff)
        & (line_amplitude_all >= center_cutoff)
    )
    line_amplitude_all = torch.where(
        passes_center_cutoff_all,
        line_amplitude_all,
        torch.zeros_like(line_amplitude_all),
    )
    live_line_mask = passes_center_cutoff_all.any(dim=1)

    # cutoff continuum on the full grid (sliced per-window below)
    continuum_cutoff = continuum_opacity * LINE_CENTER_CUTOFF_RATIO

    # Only lines that clear the center cutoff evaluate their Stark profile
    # windows; inactive catalog records never touch the wavelength grid.
    # One host transfer for the whole mask instead of one sync per line.
    live_line_mask_host = live_line_mask.cpu().tolist()
    for line_index, line in enumerate(invariants.lines):
        if not live_line_mask_host[line_index]:
            continue
        line_amplitude = line_amplitude_all[line_index]
        passes_center_cutoff = passes_center_cutoff_all[line_index]
        center_index = line.center_index

        # reach half-window (grown until the edge is below cutoff at every depth)
        reach = _reach_window(
            invariants,
            line,
            line_amplitude,
            passes_center_cutoff,
            profile_state,
            stimulated_emission_factor,
            continuum_opacity,
            wavelength_grid,
            compute_device,
        )
        window_start = max(center_index - reach, 0)
        window_stop = min(center_index + reach, wavelength_count - 1)
        if window_stop < window_start:
            continue  # window entirely off-grid
        window_columns = slice(window_start, window_stop + 1)
        window_width = window_stop - window_start + 1
        window_center_index = center_index - window_start

        window_wavelength_grid = wavelength_grid[window_columns]
        window_grid_by_depth = window_wavelength_grid[None, :].expand(depth_count, -1)
        stimulated_emission_window = stimulated_emission_factor[:, window_columns]
        window_cutoff = continuum_cutoff[:, window_columns]

        # Hydrogen Stark profile on the active line window, per depth.
        wavelength_offset_nm = window_grid_by_depth - line.line_wavelength_nm
        profile = _hydrogen_profile_grid(
            line.n_lower,
            line.n_upper,
            wavelength_offset_nm,
            profile_state,
            invariants.tables,
            line.component_frequency_offsets,
            line.component_weights,
            line.component_count,
        )
        opacity_window = line_amplitude[:, None] * profile * stimulated_emission_window

        # Continuum-merge taper; inert for low Balmer lines.
        continuum_merge_wavelength_host, taper_end_wavelength_host = (
            _line_continuum_merge_limits(
                invariants.merge_wavenumber_by_depth,
                line.series_limit_wavenumber_cm,
                line.shifted_series_limit_wavelength_nm,
            )
        )
        continuum_merge_wavelength = torch.as_tensor(
            continuum_merge_wavelength_host,
            dtype=profile_dtype,
        ).to(compute_device)[:, None]
        taper_end_wavelength = torch.as_tensor(
            taper_end_wavelength_host,
            dtype=profile_dtype,
        ).to(compute_device)[:, None]

        red_neighbor_opacity = None
        blue_neighbor_opacity = None
        if not line.simple:
            taper_active = taper_end_wavelength > continuum_merge_wavelength
            taper_denominator = torch.clamp(
                taper_end_wavelength - continuum_merge_wavelength,
                min=1e-30,
            )
            taper_ramp = (
                window_grid_by_depth - continuum_merge_wavelength
            ) / taper_denominator
            in_taper = taper_active & (window_grid_by_depth < taper_end_wavelength)
            opacity_window = torch.where(
                in_taper,
                opacity_window * taper_ramp,
                opacity_window,
            )
            continuum_merge_mask = window_grid_by_depth < continuum_merge_wavelength

            # Neighbor profiles are needed only when the walk crosses the red/blue
            # dominance cuts.  If the active window stays inside those cuts, the
            # neighbor opacity slabs would never be read.
            red_neighbor_upper_level = max(line.n_upper - 2, line.n_lower + 1)
            blue_neighbor_upper_level = line.n_upper + 2
            need_red = bool(
                (window_wavelength_grid > line.red_dominance_cutoff_nm).any()
            )
            need_blue = bool(
                (window_wavelength_grid < line.blue_dominance_cutoff_nm).any()
            )
            if need_red:
                red_component_offsets_hz, red_component_weights, red_component_count = (
                    _component_tensors_for_transition(
                        invariants,
                        line.n_lower,
                        red_neighbor_upper_level,
                    )
                )
                red_neighbor_profile = _hydrogen_profile_grid(
                    line.n_lower,
                    red_neighbor_upper_level,
                    window_grid_by_depth - line.far_red_neighbor_wavelength_nm,
                    profile_state,
                    invariants.tables,
                    red_component_offsets_hz,
                    red_component_weights,
                    red_component_count,
                )
                red_neighbor_opacity = (
                    line_amplitude[:, None]
                    * red_neighbor_profile
                    * stimulated_emission_window
                )
                red_neighbor_opacity = torch.where(
                    in_taper,
                    red_neighbor_opacity * taper_ramp,
                    red_neighbor_opacity,
                )
                red_neighbor_opacity = red_neighbor_opacity.to(accumulator_dtype)
            if need_blue:
                (
                    blue_component_offsets_hz,
                    blue_component_weights,
                    blue_component_count,
                ) = _component_tensors_for_transition(
                    invariants,
                    line.n_lower,
                    blue_neighbor_upper_level,
                )
                blue_neighbor_profile = _hydrogen_profile_grid(
                    line.n_lower,
                    blue_neighbor_upper_level,
                    window_grid_by_depth - line.far_blue_neighbor_wavelength_nm,
                    profile_state,
                    invariants.tables,
                    blue_component_offsets_hz,
                    blue_component_weights,
                    blue_component_count,
                )
                blue_neighbor_opacity = (
                    line_amplitude[:, None]
                    * blue_neighbor_profile
                    * stimulated_emission_window
                )
                blue_neighbor_opacity = torch.where(
                    in_taper,
                    blue_neighbor_opacity * taper_ramp,
                    blue_neighbor_opacity,
                )
                blue_neighbor_opacity = blue_neighbor_opacity.to(accumulator_dtype)
        else:
            continuum_merge_mask = torch.zeros_like(
                window_grid_by_depth, dtype=torch.bool
            )

        # --- outward walk with the 1e-3*continuum cutoff (contiguous-run mask) ---
        # _deposit_walk accumulates into the resident opacity window view.
        opacity_window = torch.where(
            passes_center_cutoff[:, None],
            opacity_window,
            torch.zeros_like(opacity_window),
        )
        _deposit_walk(
            hydrogen_opacity[:, window_columns],
            opacity_window.to(accumulator_dtype),
            window_cutoff.to(accumulator_dtype),
            window_center_index,
            window_width,
            window_wavelength_grid,
            line,
            continuum_merge_mask,
            red_neighbor_opacity,
            blue_neighbor_opacity,
            passes_center_cutoff,
        )

    if invariants.merged:
        population_per_mass = torch.where(
            mass_density > 0,
            hydrogen_neutral_partition_normalized_population / mass_density_safe,
            torch.zeros_like(hydrogen_neutral_partition_normalized_population),
        )
        continuum_cutoff_accumulator = continuum_cutoff.to(accumulator_dtype)
        merge_wavenumber_by_depth = invariants.merge_wavenumber_by_depth
        wavelength_grid_host = invariants.wavelength_grid_host
        grid_column_indices = torch.arange(
            wavelength_count,
            device=compute_device,
        )[None, :]
        for merged_record in invariants.merged:
            merged_boltzmann_weight = fast_ex(
                torch.as_tensor(
                    [merged_record.lower_excitation_cm],
                    dtype=profile_dtype,
                    device=compute_device,
                )[:, None]
                * hc_over_kt[None, :],
                invariants.exponential_integer_table,
                invariants.exponential_fraction_table,
            )[0]
            merged_opacity = (
                merged_record.merged_continuum_strength
                * population_per_mass
                * merged_boltzmann_weight
            )
            merge_wavelength_nm, tail_wavelength_nm = _merged_continuum_limits(
                merged_record.series_limit_wavelength_nm,
                merged_record.last_resolved_upper_level,
                merge_wavenumber_by_depth,
            )
            plateau_start_index = max(
                _center_index(
                    wavelength_grid_host,
                    merged_record.series_limit_wavelength_nm,
                ),
                0,
            )
            merge_index = np.searchsorted(
                wavelength_grid_host,
                merge_wavelength_nm,
                side="left",
            )
            tail_index = np.minimum(
                np.searchsorted(wavelength_grid_host, tail_wavelength_nm, side="right"),
                wavelength_count,
            )
            ramp_denominator = np.maximum(
                tail_index - np.maximum(merge_index, plateau_start_index),
                1,
            ).astype(np.float64)
            merge_index_tensor = torch.as_tensor(
                merge_index,
                dtype=torch.int64,
                device=compute_device,
            )[:, None]
            tail_index_tensor = torch.as_tensor(
                tail_index,
                dtype=torch.int64,
                device=compute_device,
            )[:, None]
            ramp_denominator_tensor = torch.as_tensor(
                ramp_denominator,
                dtype=profile_dtype,
                device=compute_device,
            )[:, None]
            ramp = (
                torch.clamp(
                    (tail_index_tensor - grid_column_indices).to(profile_dtype),
                    min=0.0,
                )
                / ramp_denominator_tensor
            )
            in_linear_ramp = grid_column_indices >= merge_index_tensor
            merged_value = torch.where(
                in_linear_ramp,
                merged_opacity[:, None] * ramp,
                merged_opacity[:, None].expand(-1, wavelength_count),
            )
            merged_value = merged_value * stimulated_emission_factor

            wavelength_after_limit = (
                wavelength_grid[None, :] >= merged_record.series_limit_wavelength_nm
            )
            in_deposit_range = (grid_column_indices >= plateau_start_index) & (
                grid_column_indices < tail_index_tensor
            )
            skipped_blueward = ~wavelength_after_limit
            merged_value = torch.where(
                in_deposit_range & wavelength_after_limit,
                merged_value,
                torch.zeros_like(merged_value),
            )
            terminates_merged_walk = (
                (merged_value < continuum_cutoff_accumulator.to(profile_dtype))
                & in_deposit_range
                & (~skipped_blueward)
            )
            walk_is_open = torch.cumprod(
                (~terminates_merged_walk).to(torch.int8),
                dim=1,
            ).to(torch.bool)
            open_before_column = torch.ones_like(walk_is_open)
            open_before_column[:, 1:] = walk_is_open[:, :-1]
            deposit = (
                in_deposit_range
                & (~skipped_blueward)
                & open_before_column
                & (~terminates_merged_walk)
            )
            hydrogen_opacity = hydrogen_opacity + torch.where(
                deposit,
                merged_value.to(accumulator_dtype),
                torch.zeros_like(merged_value).to(accumulator_dtype),
            )

    if not apply_stim:
        emission_factor_safe = torch.where(
            stimulated_emission_factor > 0,
            stimulated_emission_factor,
            torch.ones_like(stimulated_emission_factor),
        )
        hydrogen_opacity = hydrogen_opacity / emission_factor_safe
    # Return in the device accumulator policy: fp32 on MPS, fp64 on CPU.
    return (
        hydrogen_opacity.to(ACCUMULATION_DTYPE)
        if compute_device.type == "mps"
        else hydrogen_opacity
    )
