"""GPU-resident opacity for atomic, autoionizing, and helium lines.

The kernel combines a static line catalog with one structured atmosphere. It
builds per-depth line amplitudes, applies the catalog-specific damping model,
and scatters surviving center/wing contributions onto the wavelength grid.
Hydrogen and molecular bands live in separate modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .constants import (
    BOLTZMANN_ERG_PER_K,
    CLASSICAL_LINE_STRENGTH_COEFFICIENT,
    LIGHT_SPEED_CM_PER_S,
    LIGHT_SPEED_NM_PER_S,
    PLANCK_ERG_SECOND,
)
from .device import ACCUMULATION_DTYPE, DEFAULT_DTYPE, device, to_dev


def highp_dtype(runtime_device) -> torch.dtype:
    """Highest float dtype the device supports: fp64 on CPU/CUDA, fp32 on MPS.

    Line-amplitude cutoffs are narrow enough to benefit from fp64. Apple MPS
    has no fp64 support, so it uses the documented fp32 parity floor.
    """
    return (
        torch.float32 if torch.device(runtime_device).type == "mps" else torch.float64
    )


NANOMETER_TO_CENTIMETER = 1.0e-7
LINE_CENTER_CUTOFF_RATIO = 1e-3
WING_CUTOFF_FLOOR_RATIO = 1e-8
MAX_WING_PROFILE_STEPS = 1_000_000
HELIUM3_ISOTOPE_SCALE = 1.155

# Lines with short wings use a batched [depth, line, offset] scatter; the few
# long-wing lines fall back to the per-line walk.
NARROW_WING_MAX_REACH = 128

# Pad around helium's analytic untapered reach. The merge taper can only shorten
# an untapered wing; actively tapered lines use the full grid.
HE_REACH_PAD = 4

# Reach buckets for the narrow-line scatter. Most lines stop within a few pixels,
# so each bucket sweeps only up to its own widest member.
NARROW_WING_REACH_TIERS = (1, 8, 32, NARROW_WING_MAX_REACH)


def interpolate_harris_profile(
    doppler_offset: torch.Tensor,
    damping_ratio: torch.Tensor,
    harris_profile_h0_table: torch.Tensor,
    harris_profile_h1_table: torch.Tensor,
    harris_profile_h2_table: torch.Tensor,
) -> torch.Tensor:
    """Vectorized Harris line profile for a Doppler offset and damping ratio."""
    abs_offset = doppler_offset.abs()
    table_index = torch.clamp(
        (abs_offset * 200.0 + 0.5).to(torch.int64),
        0,
        harris_profile_h0_table.numel() - 1,
    )
    h0_profile_value = harris_profile_h0_table[table_index]
    h1_profile_value = harris_profile_h1_table[table_index]
    h2_profile_value = harris_profile_h2_table[table_index]

    damping_ratio = (
        damping_ratio.to(doppler_offset.dtype)
        if damping_ratio.dtype != doppler_offset.dtype
        else damping_ratio
    )
    damping_squared = damping_ratio * damping_ratio
    offset_squared = doppler_offset * doppler_offset

    offset_squared_safe = torch.where(
        offset_squared > 0,
        offset_squared,
        torch.ones_like(offset_squared),
    )
    low_damping_tail = 0.5642 * damping_ratio / offset_squared_safe
    low_damping_core = (
        h2_profile_value * damping_ratio + h1_profile_value
    ) * damping_ratio + h0_profile_value
    low_damping_profile = torch.where(
        abs_offset > 10.0, low_damping_tail, low_damping_core
    )

    asymptotic_denominator = (damping_squared + offset_squared) * 1.4142
    asymptotic_denominator_safe = torch.where(
        asymptotic_denominator > 0,
        asymptotic_denominator,
        torch.ones_like(asymptotic_denominator),
    )
    asymptotic_base = damping_ratio * 0.79788 / asymptotic_denominator_safe
    damping_fraction = damping_squared / asymptotic_denominator_safe
    offset_fraction = offset_squared / asymptotic_denominator_safe
    denominator_squared = asymptotic_denominator_safe * asymptotic_denominator_safe
    asymptotic_correction = (
        (
            (damping_fraction - 10.0 * offset_fraction) * damping_fraction * 3.0
            + 15.0 * offset_fraction * offset_fraction
        )
        + 3.0 * offset_squared
        - damping_squared
    ) / denominator_squared + 1.0
    high_damping_profile = torch.where(
        damping_ratio <= 100.0,
        asymptotic_correction * asymptotic_base,
        asymptotic_base,
    )

    blend_1 = h1_profile_value + h0_profile_value * 1.12838
    blend_2 = h2_profile_value + blend_1 * 1.12838 - h0_profile_value
    blend_3 = (
        (1.0 - h2_profile_value) * 0.37613
        - blend_1 * 0.66667 * offset_squared
        + blend_2 * 1.12838
    )
    blend_4 = (
        3.0 * blend_3 - blend_1
    ) * 0.37613 + h0_profile_value * 0.66667 * offset_squared * offset_squared
    blend_polynomial = (
        ((blend_4 * damping_ratio + blend_3) * damping_ratio + blend_2) * damping_ratio
        + blend_1
    ) * damping_ratio + h0_profile_value
    blend_scale = (
        (-0.122727278 * damping_ratio + 0.532770573) * damping_ratio - 0.96284325
    ) * damping_ratio + 0.979895032
    mid_profile = blend_polynomial * blend_scale

    use_asymptotic = (damping_ratio > 1.4) | ((damping_ratio + abs_offset) > 3.2)
    return torch.where(
        damping_ratio < 0.2,
        low_damping_profile,
        torch.where(use_asymptotic, high_damping_profile, mid_profile),
    )


def harris_profile_at_line_center(
    damping_ratio: torch.Tensor,
    harris_profile_h0_table: torch.Tensor,
    harris_profile_h1_table: torch.Tensor,
    harris_profile_h2_table: torch.Tensor,
) -> torch.Tensor:
    """Vectorized Harris profile at zero Doppler offset, floored at 1e-30."""
    h0_origin_profile = float(harris_profile_h0_table[0])
    h1_origin_profile = float(harris_profile_h1_table[0])
    h2_origin_profile = float(harris_profile_h2_table[0])
    blend_0 = h0_origin_profile
    blend_1 = h1_origin_profile + blend_0 * 1.12838
    blend_2 = h2_origin_profile + blend_1 * 1.12838 - blend_0
    blend_3 = (1.0 - h2_origin_profile) * 0.37613 + blend_2 * 1.12838
    blend_4 = (3.0 * blend_3 - blend_1) * 0.37613

    damping_ratio = damping_ratio.to(DEFAULT_DTYPE)
    low_damping_profile = (
        h2_origin_profile * damping_ratio + h1_origin_profile
    ) * damping_ratio + h0_origin_profile
    mid_profile = (
        (
            ((blend_4 * damping_ratio + blend_3) * damping_ratio + blend_2)
            * damping_ratio
            + blend_1
        )
        * damping_ratio
        + blend_0
    ) * (
        ((-0.122727278 * damping_ratio + 0.532770573) * damping_ratio - 0.96284325)
        * damping_ratio
        + 0.979895032
    )
    # The asymptotic expression is selected only above 1.4.  Evaluating it at
    # a zero/very small damping ratio creates an unused 0/0-like branch whose
    # backward pass can be NaN even though ``torch.where`` selects the low-
    # damping polynomial.  Clamp only the inactive asymptotic branch; values in
    # its actual domain are unchanged.
    asymptotic_ratio = torch.clamp(damping_ratio, min=1.4)
    damping_squared = asymptotic_ratio * asymptotic_ratio
    asymptotic_denominator = torch.clamp(damping_squared * 1.4142, min=1e-40)
    asymptotic_base = asymptotic_ratio * 0.79788 / asymptotic_denominator
    damping_fraction = damping_squared / asymptotic_denominator
    high_damping_profile = torch.where(
        damping_ratio <= 100.0,
        (
            (damping_fraction * damping_fraction * 3.0 - damping_squared)
            / torch.clamp(asymptotic_denominator * asymptotic_denominator, min=1e-40)
            + 1.0
        )
        * asymptotic_base,
        asymptotic_base,
    )
    profile_at_zero = torch.where(
        damping_ratio < 0.2,
        low_damping_profile,
        torch.where(damping_ratio > 1.4, high_damping_profile, mid_profile),
    )
    return torch.clamp(profile_at_zero, min=1e-30)


def _fastex_tables(runtime_device, dtype=None):
    if dtype is None:
        dtype = highp_dtype(runtime_device)
    table_index = torch.arange(1001, dtype=torch.float64)
    exponential_integer_table = torch.exp(-table_index).to(dtype).to(runtime_device)
    exponential_fraction_table = (
        torch.exp(-table_index * 0.001).to(dtype).to(runtime_device)
    )
    return exponential_integer_table, exponential_fraction_table


def fast_ex(
    exponent_argument: torch.Tensor,
    exponential_integer_table: torch.Tensor,
    exponential_fraction_table: torch.Tensor,
) -> torch.Tensor:
    """Table-quantized exp(-x) used by the line opacity Boltzmann factors."""
    exponent_argument = exponent_argument.to(exponential_integer_table.dtype)
    table_size = exponential_integer_table.numel()
    integer_index = torch.floor(exponent_argument).to(torch.int64)
    in_table = (exponent_argument > 0.0) & (integer_index < table_size)
    integer_index_clamped = torch.clamp(integer_index, 0, table_size - 1)
    fractional_part = exponent_argument - integer_index_clamped.to(
        exponent_argument.dtype
    )
    fractional_index = torch.clamp(
        torch.floor(fractional_part * 1000.0 + 0.5).to(torch.int64),
        0,
        exponential_fraction_table.numel() - 1,
    )
    table_value = (
        exponential_integer_table[integer_index_clamped]
        * exponential_fraction_table[fractional_index]
    )
    exact_value = torch.exp(-exponent_argument)
    boltzmann_factor = torch.where(in_table, table_value, exact_value)
    boltzmann_factor = torch.where(
        exponent_argument == 0.0,
        torch.ones_like(boltzmann_factor),
        boltzmann_factor,
    )
    return boltzmann_factor


def nearest_grid_indices(wavelength_grid_nm, line_wavelength_nm):
    """Nearest log-grid center index for each line wavelength."""
    wavelength_grid = np.asarray(wavelength_grid_nm, np.float64)
    line_wavelength = np.asarray(line_wavelength_nm, np.float64)
    log_spacing = np.log(wavelength_grid[1] / wavelength_grid[0])
    start_index = int(np.log(wavelength_grid[0]) / log_spacing + 0.5)
    nearest_log_index = (np.log(line_wavelength) / log_spacing + 0.5).astype(np.int64)
    nearest_indices = nearest_log_index - start_index
    nearest_indices[line_wavelength < wavelength_grid[0]] = -1
    nearest_indices[line_wavelength > wavelength_grid[-1]] = wavelength_grid.size
    return nearest_indices


def nearest_grid_indices_raw(
    wavelength_grid_nm,
    line_wavelength_nm,
    origin_wavelength_nm,
):
    """Nearest raw wing-anchor index before clipping to the synthesis window."""
    wavelength_grid = np.asarray(wavelength_grid_nm, np.float64)
    line_wavelength = np.asarray(line_wavelength_nm, np.float64)
    log_spacing = np.log(wavelength_grid[1] / wavelength_grid[0])
    origin_index_floor = int(np.floor(np.log(origin_wavelength_nm) / log_spacing))
    reconstructed_start = np.exp(origin_index_floor * log_spacing)
    # `origin_wavelength_nm` is itself a grid point. On Linux/NumPy the exp/log round-trip can
    # reconstruct it one ULP low, while macOS reconstructs it exactly; a strict compare
    # then shifts every full-window wing anchor by one pixel. Only bump when the
    # reconstructed origin is meaningfully below the actual grid start, not just an ULP
    # artefact.
    origin_floor = origin_wavelength_nm - 64.0 * np.spacing(origin_wavelength_nm)
    if reconstructed_start < origin_floor:
        origin_index_floor += 1
        reconstructed_start = np.exp(origin_index_floor * log_spacing)
    return np.rint(np.log(line_wavelength / reconstructed_start) / log_spacing).astype(
        np.int64
    )


@dataclass
class AtomicInvariants:
    """GPU-resident static data and numerical tables."""

    wavelength_grid: torch.Tensor
    n_wavelengths: int
    grid_resolution: float

    metal_catalog_index: torch.Tensor
    metal_classical_strength: torch.Tensor
    metal_lower_excitation_cm: torch.Tensor
    metal_radiative_damping: torch.Tensor
    metal_stark_damping: torch.Tensor
    metal_van_der_waals_damping: torch.Tensor
    metal_wavelength_nm: torch.Tensor
    metal_population_ion_stage_index: torch.Tensor
    metal_population_element_index: torch.Tensor
    metal_center_index: torch.Tensor
    metal_wing_index: torch.Tensor
    metal_center_clamped: torch.Tensor
    metal_wing_clamped: torch.Tensor

    auto_catalog_index: torch.Tensor
    auto_oscillator_strength: torch.Tensor
    auto_lower_excitation_cm: torch.Tensor
    auto_radiative_damping: torch.Tensor
    auto_stark_damping: torch.Tensor
    auto_van_der_waals_damping: torch.Tensor
    auto_wavelength_nm: torch.Tensor
    auto_population_ion_stage_index: torch.Tensor
    auto_population_element_index: torch.Tensor
    auto_center_index: torch.Tensor
    auto_center_clamped: torch.Tensor

    helium_classical_strength: torch.Tensor
    helium_lower_excitation_cm: torch.Tensor
    helium_radiative_damping: torch.Tensor
    helium_stark_damping: torch.Tensor
    helium_van_der_waals_damping: torch.Tensor
    helium_wavelength_nm: torch.Tensor
    helium_population_ion_stage_index: torch.Tensor
    helium_population_element_index: torch.Tensor
    helium_center_index: torch.Tensor
    helium_line_type: torch.Tensor
    helium_cutoff: float

    harris_profile_h0_table: torch.Tensor
    harris_profile_h1_table: torch.Tensor
    harris_profile_h2_table: torch.Tensor
    exponential_integer_table: torch.Tensor
    exponential_fraction_table: torch.Tensor


def precompute_invariants(
    catalog: dict,
    wavelength_grid_nm: np.ndarray,
    runtime_device=None,
) -> AtomicInvariants:
    """Build the GPU-resident invariant block from the catalog (once per run).

    `catalog` is the atomic line-catalog mapping; `wavelength_grid_nm` is the synthesis grid.
    Type 0/3 records use the metal Voigt path, type 1 records use the
    autoionizing path, and type -3/-4/-6 records use the helium path. Hydrogen
    records are handled in `hydrogen_lines.py`.

    """
    compute_device = runtime_device if runtime_device is not None else device()
    work_dtype = highp_dtype(compute_device)

    wavelength_grid = np.asarray(wavelength_grid_nm, np.float64)
    n_wavelengths = wavelength_grid.size
    ratio = wavelength_grid[1] / wavelength_grid[0]
    grid_resolution = 1.0 / (ratio - 1.0) if ratio > 1.0 else 300000.0

    line_type = catalog["line_type"].astype(np.int64)
    atomic_number = catalog["atomic_number"].astype(np.int64)
    ion_stage = catalog["ion_stage"].astype(np.int64)
    population_element_index = atomic_number - 1
    population_ion_stage_index = ion_stage - 1

    oscillator_strength = np.asarray(catalog["oscillator_strength"], np.float64)
    wavelength_nm = np.asarray(catalog["wavelength_nm"], np.float64)
    index_wavelength_nm = np.asarray(catalog["index_wavelength_nm"], np.float64)
    lower_excitation_cm = np.asarray(catalog["lower_excitation_cm"], np.float64)
    radiative_damping = np.asarray(catalog["radiative_damping"], np.float64)
    stark_damping = np.asarray(catalog["stark_damping"], np.float64)
    van_der_waals_damping = np.asarray(catalog["van_der_waals_damping"], np.float64)
    raw_radiative_damping_log = np.asarray(
        catalog.get("raw_radiative_damping_log", np.zeros_like(radiative_damping)),
        np.float64,
    )
    raw_stark_damping_log = np.asarray(
        catalog.get("raw_stark_damping_log", np.zeros_like(stark_damping)),
        np.float64,
    )
    raw_van_der_waals_damping_log = np.asarray(
        catalog.get(
            "raw_van_der_waals_damping_log", np.zeros_like(van_der_waals_damping)
        ),
        np.float64,
    )

    frequency_hz = LIGHT_SPEED_NM_PER_S / wavelength_nm
    classical_line_strength = (
        CLASSICAL_LINE_STRENGTH_COEFFICIENT * oscillator_strength / frequency_hz
    )
    damping_denominator = 12.5664 * frequency_hz

    # Autoionizing Shore/Fano profiles use raw damping constants; ordinary
    # catalog damping is normalized by 4*pi*nu, so reconstruct that branch here.
    with np.errstate(over="ignore"):
        auto_radiative_damping = np.where(
            raw_radiative_damping_log != 0.0,
            np.power(10.0, raw_radiative_damping_log),
            radiative_damping * damping_denominator,
        )
        auto_van_der_waals_damping = np.where(
            raw_van_der_waals_damping_log != 0.0,
            np.power(10.0, raw_van_der_waals_damping_log),
            van_der_waals_damping * damping_denominator,
        )
        auto_stark_damping = np.where(
            raw_stark_damping_log > 0.0,
            -np.power(10.0, -raw_stark_damping_log),
            np.where(
                raw_stark_damping_log < 0.0,
                np.power(10.0, raw_stark_damping_log),
                stark_damping * damping_denominator,
            ),
        )

    center_indices = nearest_grid_indices(wavelength_grid, index_wavelength_nm)
    wing_anchor_indices = nearest_grid_indices_raw(
        wavelength_grid,
        index_wavelength_nm,
        float(wavelength_grid[0]),
    )

    n_supported_ion_stages = 6
    n_supported_elements = 139
    valid_population_slot = (
        (population_element_index >= 0)
        & (population_element_index < n_supported_elements)
        & (population_ion_stage_index >= 0)
        & (population_ion_stage_index < n_supported_ion_stages)
    )

    # Catalog type codes route the atomic lines into the validated kernels.
    metal_line_mask = ((line_type == 0) | (line_type == 3)) & valid_population_slot
    autoionizing_line_mask = (line_type == 1) & valid_population_slot
    helium_line_mask = np.isin(line_type, [-3, -4, -6])
    metal_line_indices = np.where(metal_line_mask)[0]
    autoionizing_line_indices = np.where(autoionizing_line_mask)[0]
    helium_line_indices = np.where(helium_line_mask)[0]

    def to_device_tensor(array, dtype):
        return torch.as_tensor(array, dtype=dtype).to(compute_device)

    metal_center_clamped = np.clip(
        center_indices[metal_line_indices],
        0,
        n_wavelengths - 1,
    )
    metal_wing_clamped = np.clip(
        wing_anchor_indices[metal_line_indices],
        0,
        n_wavelengths - 1,
    )
    auto_center_clamped = np.clip(
        center_indices[autoionizing_line_indices],
        0,
        n_wavelengths - 1,
    )
    helium_center_clamped = np.clip(
        center_indices[helium_line_indices],
        0,
        n_wavelengths - 1,
    )

    exponential_integer_table, exponential_fraction_table = _fastex_tables(
        compute_device,
        work_dtype,
    )
    return AtomicInvariants(
        wavelength_grid=to_device_tensor(wavelength_grid, work_dtype),
        n_wavelengths=n_wavelengths,
        grid_resolution=grid_resolution,
        metal_catalog_index=to_device_tensor(metal_line_indices, torch.int64),
        metal_classical_strength=to_device_tensor(
            classical_line_strength[metal_line_indices],
            DEFAULT_DTYPE,
        ),
        metal_lower_excitation_cm=to_device_tensor(
            lower_excitation_cm[metal_line_indices],
            work_dtype,
        ),
        metal_radiative_damping=to_device_tensor(
            radiative_damping[metal_line_indices],
            DEFAULT_DTYPE,
        ),
        metal_stark_damping=to_device_tensor(
            stark_damping[metal_line_indices], DEFAULT_DTYPE
        ),
        metal_van_der_waals_damping=to_device_tensor(
            van_der_waals_damping[metal_line_indices], DEFAULT_DTYPE
        ),
        metal_wavelength_nm=to_device_tensor(
            wavelength_nm[metal_line_indices], work_dtype
        ),
        metal_population_ion_stage_index=to_device_tensor(
            population_ion_stage_index[metal_line_indices],
            torch.int64,
        ),
        metal_population_element_index=to_device_tensor(
            population_element_index[metal_line_indices],
            torch.int64,
        ),
        metal_center_index=to_device_tensor(
            center_indices[metal_line_indices], torch.int64
        ),
        metal_wing_index=to_device_tensor(
            wing_anchor_indices[metal_line_indices],
            torch.int64,
        ),
        metal_center_clamped=to_device_tensor(metal_center_clamped, torch.int64),
        metal_wing_clamped=to_device_tensor(metal_wing_clamped, torch.int64),
        auto_catalog_index=to_device_tensor(autoionizing_line_indices, torch.int64),
        auto_oscillator_strength=to_device_tensor(
            oscillator_strength[autoionizing_line_indices],
            DEFAULT_DTYPE,
        ),
        auto_lower_excitation_cm=to_device_tensor(
            lower_excitation_cm[autoionizing_line_indices],
            work_dtype,
        ),
        auto_radiative_damping=to_device_tensor(
            auto_radiative_damping[autoionizing_line_indices],
            DEFAULT_DTYPE,
        ),
        auto_stark_damping=to_device_tensor(
            auto_stark_damping[autoionizing_line_indices],
            DEFAULT_DTYPE,
        ),
        auto_van_der_waals_damping=to_device_tensor(
            auto_van_der_waals_damping[autoionizing_line_indices],
            DEFAULT_DTYPE,
        ),
        auto_wavelength_nm=to_device_tensor(
            wavelength_nm[autoionizing_line_indices],
            work_dtype,
        ),
        auto_population_ion_stage_index=to_device_tensor(
            population_ion_stage_index[autoionizing_line_indices],
            torch.int64,
        ),
        auto_population_element_index=to_device_tensor(
            population_element_index[autoionizing_line_indices],
            torch.int64,
        ),
        auto_center_index=to_device_tensor(
            center_indices[autoionizing_line_indices],
            torch.int64,
        ),
        auto_center_clamped=to_device_tensor(auto_center_clamped, torch.int64),
        helium_classical_strength=to_device_tensor(
            classical_line_strength[helium_line_indices],
            DEFAULT_DTYPE,
        ),
        helium_lower_excitation_cm=to_device_tensor(
            lower_excitation_cm[helium_line_indices],
            work_dtype,
        ),
        helium_radiative_damping=to_device_tensor(
            radiative_damping[helium_line_indices],
            DEFAULT_DTYPE,
        ),
        helium_stark_damping=to_device_tensor(
            stark_damping[helium_line_indices], DEFAULT_DTYPE
        ),
        helium_van_der_waals_damping=to_device_tensor(
            van_der_waals_damping[helium_line_indices], DEFAULT_DTYPE
        ),
        helium_wavelength_nm=to_device_tensor(
            wavelength_nm[helium_line_indices],
            work_dtype,
        ),
        helium_population_ion_stage_index=to_device_tensor(
            population_ion_stage_index[helium_line_indices],
            torch.int64,
        ),
        helium_population_element_index=to_device_tensor(
            population_element_index[helium_line_indices],
            torch.int64,
        ),
        helium_center_index=to_device_tensor(helium_center_clamped, torch.int64),
        helium_line_type=to_device_tensor(
            catalog["helium_line_type"].astype(np.int64),
            torch.int64,
        ),
        helium_cutoff=float(catalog["helium_line_center_cutoff_ratio"]),
        harris_profile_h0_table=to_device_tensor(
            catalog["harris_profile_h0_table"], work_dtype
        ),
        harris_profile_h1_table=to_device_tensor(
            catalog["harris_profile_h1_table"], work_dtype
        ),
        harris_profile_h2_table=to_device_tensor(
            catalog["harris_profile_h2_table"], work_dtype
        ),
        exponential_integer_table=exponential_integer_table,
        exponential_fraction_table=exponential_fraction_table,
    )


# Depth-batched outward wing walk.
def _wing_walk_metal(
    line_mass_absorption_coefficient_grid,
    invariants,
    wing_anchor_columns,
    wing_profile_amplitude,
    wing_damping_ratio,
    doppler_width_nm,
    line_wavelength_nm,
    wing_cutoff_opacity,
):
    """Deposit one metal line's red and blue wings, vectorized over depth."""
    n_wavelengths = invariants.n_wavelengths
    grid_resolution = invariants.grid_resolution

    doppler_fraction = torch.where(
        doppler_width_nm > 0,
        doppler_width_nm / line_wavelength_nm,
        torch.full_like(doppler_width_nm, 1e-10),
    )
    active_depths = doppler_width_nm > 0
    if not bool(active_depths.any()):
        return

    ten_doppler_steps = (10.0 * doppler_fraction * grid_resolution).to(torch.int64)
    doppler_offset_per_pixel = torch.where(
        doppler_fraction > 0,
        1.0 / (doppler_fraction * grid_resolution),
        torch.ones_like(doppler_fraction),
    )

    max_near_steps = int(ten_doppler_steps.max().item()) if active_depths.any() else 0
    near_cutoff_step = ten_doppler_steps.clone()
    crossed_cutoff = torch.zeros_like(active_depths)
    opacity_at_ten_doppler_steps = torch.zeros_like(doppler_fraction)

    low_damping_branch = wing_damping_ratio < 0.2
    if max_near_steps >= 1:
        near_steps = torch.arange(
            1, max_near_steps + 1, device=line_mass_absorption_coefficient_grid.device
        )
        doppler_offset = near_steps[None, :] * doppler_offset_per_pixel[:, None]
        profile = harris_wing_walk_profile(
            doppler_offset,
            wing_damping_ratio[:, None],
            invariants,
            low_damping_branch[:, None],
        )
        opacity_value = wing_profile_amplitude[:, None] * profile
        valid_step = near_steps[None, :] <= ten_doppler_steps[:, None]
        below = (opacity_value < wing_cutoff_opacity[:, None]) & valid_step
        any_below = below.any(dim=1)
        first_below = torch.argmax(below.to(torch.int8), dim=1) + 1  # 1-based step
        near_cutoff_step = torch.where(any_below, first_below, ten_doppler_steps)
        crossed_cutoff = any_below
        ten_doppler_column = torch.clamp(ten_doppler_steps - 1, min=0)
        opacity_at_ten_doppler_steps = torch.where(
            ten_doppler_steps >= 1,
            opacity_value.gather(1, ten_doppler_column[:, None]).squeeze(1),
            torch.zeros_like(doppler_fraction),
        )

    never_crossed = (~crossed_cutoff) & (ten_doppler_steps >= 1)
    near_cutoff_step = torch.where(
        never_crossed,
        torch.full_like(near_cutoff_step, -1),
        near_cutoff_step,
    )

    use_far_wing = near_cutoff_step == -1
    far_wing_coefficient = torch.where(
        (ten_doppler_steps > 0) & (opacity_at_ten_doppler_steps > 0),
        opacity_at_ten_doppler_steps
        * ten_doppler_steps.to(doppler_fraction.dtype) ** 2,
        torch.zeros_like(doppler_fraction),
    )
    far_wing_reach_steps = torch.where(
        (wing_cutoff_opacity > 0) & (far_wing_coefficient > 0),
        (torch.sqrt(far_wing_coefficient / wing_cutoff_opacity) + 1.0).to(torch.int64),
        torch.zeros_like(ten_doppler_steps),
    )
    far_wing_reach_steps = torch.clamp(far_wing_reach_steps, max=MAX_WING_PROFILE_STEPS)
    reach_steps = torch.where(use_far_wing, far_wing_reach_steps, near_cutoff_step)
    reach_steps = torch.where(active_depths, reach_steps, torch.zeros_like(reach_steps))

    global_reach = int(reach_steps.max().item()) if active_depths.any() else 0
    if global_reach <= 0:
        return

    damping_ratio_column = wing_damping_ratio[:, None]
    low_damping_column = low_damping_branch[:, None]
    CHUNK = 4096
    offset_start = 1
    while offset_start <= global_reach:
        offset_stop = min(offset_start + CHUNK - 1, global_reach)
        offsets = torch.arange(
            offset_start,
            offset_stop + 1,
            device=line_mass_absorption_coefficient_grid.device,
        )  # [C]
        within_reach = offsets[None, :] <= reach_steps[:, None]  # [D,C]
        doppler_offset = offsets[None, :] * doppler_offset_per_pixel[:, None]
        profile = harris_wing_walk_profile(
            doppler_offset,
            damping_ratio_column,
            invariants,
            low_damping_column,
        )
        near_opacity = wing_profile_amplitude[:, None] * profile
        far_wing_mask = use_far_wing[:, None] & (
            offsets[None, :] > ten_doppler_steps[:, None]
        )
        offsets_float = offsets[None, :].to(doppler_fraction.dtype)
        far_opacity = far_wing_coefficient[:, None] / (offsets_float * offsets_float)
        opacity_value = torch.where(far_wing_mask, far_opacity, near_opacity)
        opacity_value = torch.where(
            within_reach, opacity_value, torch.zeros_like(opacity_value)
        )

        red_columns = wing_anchor_columns[:, None] + offsets[None, :]
        blue_columns = wing_anchor_columns[:, None] - offsets[None, :]
        red_ok = within_reach & (red_columns >= 0) & (red_columns < n_wavelengths)
        blue_ok = within_reach & (blue_columns >= 0) & (blue_columns < n_wavelengths)

        _scatter_add_rows(
            line_mass_absorption_coefficient_grid, red_columns, opacity_value, red_ok
        )
        _scatter_add_rows(
            line_mass_absorption_coefficient_grid, blue_columns, opacity_value, blue_ok
        )
        offset_start = offset_stop + 1


def harris_wing_walk_profile(
    doppler_offset, damping_ratio, invariants, use_low_damping_branch
):
    """Profile used by the metal wing walk's near-wing branch."""
    abs_offset = doppler_offset.abs()
    table_index = torch.clamp(
        (abs_offset * 200.0 + 0.5).to(torch.int64),
        0,
        invariants.harris_profile_h0_table.numel() - 1,
    )
    h0_profile_value = invariants.harris_profile_h0_table[table_index]
    h1_profile_value = invariants.harris_profile_h1_table[table_index]
    offset_squared = doppler_offset * doppler_offset
    offset_squared_safe = torch.where(
        offset_squared > 0,
        offset_squared,
        torch.ones_like(doppler_offset),
    )
    cheap_tail = 0.5642 * damping_ratio / offset_squared_safe
    cheap_core = h0_profile_value + damping_ratio * h1_profile_value
    cheap_profile = torch.where(abs_offset > 10.0, cheap_tail, cheap_core)

    full_profile = interpolate_harris_profile(
        doppler_offset,
        damping_ratio.expand_as(doppler_offset)
        if damping_ratio.shape != doppler_offset.shape
        else damping_ratio,
        invariants.harris_profile_h0_table,
        invariants.harris_profile_h1_table,
        invariants.harris_profile_h2_table,
    )
    low_damping_mask = (
        use_low_damping_branch.expand_as(doppler_offset)
        if use_low_damping_branch.shape != doppler_offset.shape
        else use_low_damping_branch
    )
    return torch.where(low_damping_mask, cheap_profile, full_profile)


def _scatter_add_rows(
    line_mass_absorption_coefficient_grid, columns, values, deposit_mask
):
    """Add one value per depth/column entry into the line-opacity slab."""
    n_wavelengths = line_mass_absorption_coefficient_grid.shape[1]
    clamped_columns = torch.clamp(columns, 0, n_wavelengths - 1)
    depth_offsets = (
        torch.arange(
            line_mass_absorption_coefficient_grid.shape[0],
            device=line_mass_absorption_coefficient_grid.device,
        )[:, None]
        * n_wavelengths
    )
    flat_indices = (depth_offsets + clamped_columns)[deposit_mask]
    flat_values = values[deposit_mask].to(line_mass_absorption_coefficient_grid.dtype)
    line_mass_absorption_coefficient_grid.view(-1).index_put_(
        (flat_indices,), flat_values, accumulate=True
    )


def _scatter_add_3d(
    line_mass_absorption_coefficient_grid, columns, values, deposit_mask
):
    """Add a [depth, line, offset] block into the line-opacity slab."""
    n_wavelengths = line_mass_absorption_coefficient_grid.shape[1]
    n_depths = line_mass_absorption_coefficient_grid.shape[0]
    clamped_columns = torch.clamp(columns, 0, n_wavelengths - 1)
    depth_offsets = (
        torch.arange(
            n_depths, device=line_mass_absorption_coefficient_grid.device
        ).view(n_depths, 1, 1)
        * n_wavelengths
    )
    flat_indices = (depth_offsets + clamped_columns)[deposit_mask]
    flat_values = values[deposit_mask].to(line_mass_absorption_coefficient_grid.dtype)
    line_mass_absorption_coefficient_grid.view(-1).index_put_(
        (flat_indices,), flat_values, accumulate=True
    )


def _wing_reach_batched(
    invariants,
    wing_anchor_columns,
    wing_profile_amplitude,
    wing_damping_ratio,
    doppler_width_nm,
    line_wavelength_nm,
    wing_cutoff_opacity,
    wing_depth_pairs,
):
    """Return per-depth/per-line wing reach geometry without depositing opacity."""
    device = wing_anchor_columns.device
    grid_resolution = invariants.grid_resolution

    doppler_fraction = torch.where(
        doppler_width_nm > 0,
        doppler_width_nm / line_wavelength_nm[None, :],
        torch.full_like(doppler_width_nm, 1e-10),
    )
    active_pairs = (doppler_width_nm > 0) & wing_depth_pairs
    ten_doppler_steps = (10.0 * doppler_fraction * grid_resolution).to(torch.int64)
    doppler_offset_per_pixel = torch.where(
        doppler_fraction > 0,
        1.0 / (doppler_fraction * grid_resolution),
        torch.ones_like(doppler_fraction),
    )

    low_damping_branch = wing_damping_ratio < 0.2
    max_near_steps = (
        int(ten_doppler_steps[active_pairs].max().item())
        if bool(active_pairs.any())
        else 0
    )
    near_cutoff_step = ten_doppler_steps.clone()
    crossed_cutoff = torch.zeros_like(active_pairs)
    opacity_at_ten_doppler_steps = torch.zeros_like(doppler_fraction)
    if max_near_steps >= 1:
        near_steps = torch.arange(1, max_near_steps + 1, device=device)
        doppler_offset = (
            near_steps.view(1, 1, -1) * doppler_offset_per_pixel[:, :, None]
        )
        profile = harris_wing_walk_profile(
            doppler_offset,
            wing_damping_ratio[:, :, None],
            invariants,
            low_damping_branch[:, :, None],
        )
        opacity_value = wing_profile_amplitude[:, :, None] * profile
        within_near_wing = near_steps.view(1, 1, -1) <= ten_doppler_steps[:, :, None]
        below_cutoff = (
            opacity_value < wing_cutoff_opacity[:, :, None]
        ) & within_near_wing
        any_below_cutoff = below_cutoff.any(dim=2)
        first_below_cutoff = torch.argmax(below_cutoff.to(torch.int8), dim=2) + 1
        near_cutoff_step = torch.where(
            any_below_cutoff, first_below_cutoff, ten_doppler_steps
        )
        crossed_cutoff = any_below_cutoff
        ten_doppler_column = torch.clamp(ten_doppler_steps - 1, min=0)
        opacity_at_ten_doppler_steps = torch.where(
            ten_doppler_steps >= 1,
            opacity_value.gather(2, ten_doppler_column[:, :, None]).squeeze(2),
            torch.zeros_like(doppler_fraction),
        )

    never_crossed = (~crossed_cutoff) & (ten_doppler_steps >= 1)
    near_cutoff_step = torch.where(
        never_crossed,
        torch.full_like(near_cutoff_step, -1),
        near_cutoff_step,
    )

    use_far_wing = near_cutoff_step == -1
    far_wing_coefficient = torch.where(
        (ten_doppler_steps > 0) & (opacity_at_ten_doppler_steps > 0),
        opacity_at_ten_doppler_steps
        * ten_doppler_steps.to(doppler_fraction.dtype) ** 2,
        torch.zeros_like(doppler_fraction),
    )
    far_wing_reach_steps = torch.where(
        (wing_cutoff_opacity > 0) & (far_wing_coefficient > 0),
        (torch.sqrt(far_wing_coefficient / wing_cutoff_opacity) + 1.0).to(torch.int64),
        torch.zeros_like(ten_doppler_steps),
    )
    far_wing_reach_steps = torch.clamp(far_wing_reach_steps, max=MAX_WING_PROFILE_STEPS)
    reach_steps = torch.where(use_far_wing, far_wing_reach_steps, near_cutoff_step)
    reach_steps = torch.where(active_pairs, reach_steps, torch.zeros_like(reach_steps))
    return (
        reach_steps,
        use_far_wing,
        ten_doppler_steps,
        doppler_offset_per_pixel,
        far_wing_coefficient,
    )


def _wing_walk_narrow_batched(
    line_mass_absorption_coefficient_grid,
    invariants,
    wing_anchor_columns,
    wing_profile_amplitude,
    wing_damping_ratio,
    reach_steps,
    use_far_wing,
    ten_doppler_steps,
    doppler_offset_per_pixel,
    far_wing_coefficient,
    n_wavelengths,
):
    """Deposit red and blue wings for narrow lines, bucketed by reach length."""
    n_lines = wing_anchor_columns.numel()
    if n_lines == 0:
        return
    line_reach_steps = reach_steps.max(dim=0).values
    lower_reach_bound = 0
    for upper_reach_bound in NARROW_WING_REACH_TIERS:
        in_reach_tier = (line_reach_steps > lower_reach_bound) & (
            line_reach_steps <= upper_reach_bound
        )
        lower_reach_bound = upper_reach_bound
        if not bool(in_reach_tier.any()):
            continue
        tier_line_indices = torch.nonzero(in_reach_tier, as_tuple=False).squeeze(1)
        _wing_walk_narrow_core(
            line_mass_absorption_coefficient_grid,
            invariants,
            wing_anchor_columns[tier_line_indices],
            wing_profile_amplitude[:, tier_line_indices],
            wing_damping_ratio[:, tier_line_indices],
            reach_steps[:, tier_line_indices],
            use_far_wing[:, tier_line_indices],
            ten_doppler_steps[:, tier_line_indices],
            doppler_offset_per_pixel[:, tier_line_indices],
            far_wing_coefficient[:, tier_line_indices],
            n_wavelengths,
        )


def _wing_walk_narrow_core(
    line_mass_absorption_coefficient_grid,
    invariants,
    wing_anchor_columns,
    wing_profile_amplitude,
    wing_damping_ratio,
    reach_steps,
    use_far_wing,
    ten_doppler_steps,
    doppler_offset_per_pixel,
    far_wing_coefficient,
    n_wavelengths,
):
    """Fixed-width batched sweep for one reach bucket of narrow lines."""
    n_lines = wing_anchor_columns.numel()
    if n_lines == 0:
        return
    window_width = int(reach_steps.max().item())
    if window_width <= 0:
        return

    offsets = torch.arange(
        1, window_width + 1, device=line_mass_absorption_coefficient_grid.device
    )
    within_reach = offsets.view(1, 1, -1) <= reach_steps[:, :, None]

    doppler_offset = offsets.view(1, 1, -1) * doppler_offset_per_pixel[:, :, None]
    low_damping_branch = wing_damping_ratio < 0.2
    profile = harris_wing_walk_profile(
        doppler_offset,
        wing_damping_ratio[:, :, None],
        invariants,
        low_damping_branch[:, :, None],
    )
    near_opacity = wing_profile_amplitude[:, :, None] * profile
    far_wing_mask = use_far_wing[:, :, None] & (
        offsets.view(1, 1, -1) > ten_doppler_steps[:, :, None]
    )
    offsets_float = offsets.view(1, 1, -1).to(doppler_offset.dtype)
    far_opacity = far_wing_coefficient[:, :, None] / (offsets_float * offsets_float)
    opacity_value = torch.where(far_wing_mask, far_opacity, near_opacity)
    opacity_value = torch.where(
        within_reach, opacity_value, torch.zeros_like(opacity_value)
    )

    anchor_columns = wing_anchor_columns.view(1, -1, 1)
    red_columns = anchor_columns + offsets.view(1, 1, -1)
    blue_columns = anchor_columns - offsets.view(1, 1, -1)
    red_columns = red_columns.expand(
        line_mass_absorption_coefficient_grid.shape[0], -1, -1
    )
    blue_columns = blue_columns.expand(
        line_mass_absorption_coefficient_grid.shape[0], -1, -1
    )
    red_ok = within_reach & (red_columns >= 0) & (red_columns < n_wavelengths)
    blue_ok = within_reach & (blue_columns >= 0) & (blue_columns < n_wavelengths)

    _scatter_add_3d(
        line_mass_absorption_coefficient_grid, red_columns, opacity_value, red_ok
    )
    _scatter_add_3d(
        line_mass_absorption_coefficient_grid, blue_columns, opacity_value, blue_ok
    )


def _wing_walk_helium(
    line_mass_absorption_coefficient_grid,
    continuum_opacity,
    invariants,
    center_column,
    line_wavelength_nm,
    effective_amplitude,
    doppler_width_nm,
    damping_ratio,
    line_cutoff,
    merge_start_wavelength_nm,
    merge_tail_wavelength_nm,
    window=None,
):
    """Depth-batched helium wing walk on the wavelength grid."""
    n_wavelengths = invariants.n_wavelengths
    wavelength_grid = invariants.wavelength_grid
    active_depths = (effective_amplitude > 0) & (doppler_width_nm > 0)
    if not bool(active_depths.any()):
        return

    window_start, window_stop = (
        (0, n_wavelengths) if window is None else (int(window[0]), int(window[1]))
    )

    # Helium cutoff tests sit close to the rounding floor; use the grid dtype
    # so CPU/CUDA keep fp64 while MPS keeps its available fp32 path.
    work_dtype = invariants.wavelength_grid.dtype
    doppler_width_nm = doppler_width_nm.to(work_dtype)
    effective_amplitude = effective_amplitude.to(work_dtype)
    damping_ratio = damping_ratio.to(work_dtype)
    merge_start_wavelength_nm = merge_start_wavelength_nm.to(work_dtype)
    merge_tail_wavelength_nm = merge_tail_wavelength_nm.to(work_dtype)
    continuum_opacity = continuum_opacity.to(work_dtype)
    harris_profile_h0_table = invariants.harris_profile_h0_table.to(work_dtype)
    harris_profile_h1_table = invariants.harris_profile_h1_table.to(work_dtype)
    harris_profile_h2_table = invariants.harris_profile_h2_table.to(work_dtype)

    doppler_width_safe = torch.where(
        doppler_width_nm > 0,
        doppler_width_nm,
        torch.ones_like(doppler_width_nm),
    )
    damping_ratio_safe = torch.clamp(damping_ratio, min=1e-12)
    has_merge_start = merge_start_wavelength_nm > 0.0
    has_merge_tail = merge_tail_wavelength_nm > 0.0
    taper_base_wavelength_nm = torch.where(
        merge_start_wavelength_nm > 0,
        merge_start_wavelength_nm,
        torch.full_like(merge_start_wavelength_nm, line_wavelength_nm),
    )

    center_column_int = int(center_column)
    window_wavelength_nm = wavelength_grid[None, window_start:window_stop].to(
        work_dtype
    )
    doppler_offset = (
        window_wavelength_nm - line_wavelength_nm
    ).abs() / doppler_width_safe[:, None]
    profile = interpolate_harris_profile(
        doppler_offset,
        damping_ratio_safe[:, None],
        harris_profile_h0_table,
        harris_profile_h1_table,
        harris_profile_h2_table,
    )
    opacity_value = effective_amplitude[:, None] * profile

    taper_denominator = torch.clamp(
        merge_tail_wavelength_nm - taper_base_wavelength_nm,
        min=1e-12,
    )
    taper_fraction = (
        window_wavelength_nm - taper_base_wavelength_nm[:, None]
    ) / taper_denominator[:, None]
    taper_active = has_merge_tail[:, None] & (
        window_wavelength_nm < merge_tail_wavelength_nm[:, None]
    )
    opacity_value = torch.where(
        taper_active, opacity_value * taper_fraction, opacity_value
    )
    below_merge_start = has_merge_start[:, None] & (
        window_wavelength_nm <= merge_start_wavelength_nm[:, None]
    )
    cutoff_opacity = continuum_opacity[:, window_start:window_stop] * line_cutoff

    _walk_directional(
        line_mass_absorption_coefficient_grid,
        opacity_value,
        cutoff_opacity,
        below_merge_start,
        active_depths,
        center_column_int,
        +1,
        window_start,
        window_stop,
        skip_below_merge_start=True,
    )
    _walk_directional(
        line_mass_absorption_coefficient_grid,
        opacity_value,
        cutoff_opacity,
        below_merge_start,
        active_depths,
        center_column_int,
        -1,
        window_start,
        window_stop,
        skip_below_merge_start=False,
    )


def _walk_directional(
    line_mass_absorption_coefficient_grid,
    opacity_value,
    cutoff_opacity,
    below_merge_start,
    active_depths,
    center_column,
    direction,
    window_start,
    window_stop,
    skip_below_merge_start,
):
    """Deposit one helium wing until each depth reaches its cutoff pixel."""
    n_depths = line_mass_absorption_coefficient_grid.shape[0]
    device = line_mass_absorption_coefficient_grid.device
    if direction > 0:
        columns = torch.arange(center_column, window_stop, device=device)
    else:
        columns = torch.arange(center_column - 1, window_start - 1, -1, device=device)
    if columns.numel() == 0:
        return
    local_columns = columns - window_start
    walk_opacity = opacity_value[:, local_columns]
    walk_cutoff = cutoff_opacity[:, local_columns]
    walk_below_merge_start = below_merge_start[:, local_columns]

    below_cutoff = walk_opacity < walk_cutoff
    if skip_below_merge_start:
        stop = below_cutoff & (~walk_below_merge_start)
    else:
        stop = below_cutoff | walk_below_merge_start

    any_stop = stop.any(dim=1)
    first_stop = torch.where(
        any_stop,
        torch.argmax(stop.to(torch.int8), dim=1),
        torch.full((n_depths,), columns.numel(), device=device, dtype=torch.int64),
    )
    positions = torch.arange(columns.numel(), device=device)[None, :]
    deposit_mask = (positions < first_stop[:, None]) & active_depths[:, None]
    if skip_below_merge_start:
        deposit_mask = deposit_mask & (~walk_below_merge_start)

    absolute_columns = columns[None, :].expand(n_depths, -1)
    _scatter_add_rows(
        line_mass_absorption_coefficient_grid,
        absolute_columns,
        walk_opacity,
        deposit_mask,
    )


# Per-iteration entry point.
def accumulate_atomic(
    invariants: AtomicInvariants,
    state: dict,
    do_metal=True,
    do_helium=True,
    apply_stim=True,
    wing_mode="batched",
    output_line_mass_absorption_coefficient: torch.Tensor | None = None,
    host_accumulator: np.ndarray | None = None,
) -> torch.Tensor:
    """Accumulate atomic line opacity on the synthesis wavelength grid."""
    runtime_device = invariants.wavelength_grid.device
    work_dtype = invariants.wavelength_grid.dtype
    n_depths = state["temperature"].shape[0]
    n_wavelengths = invariants.n_wavelengths

    partition_normalized_populations = to_dev(
        state["partition_normalized_populations"], work_dtype, runtime_device
    )
    fractional_doppler_widths = to_dev(
        state["fractional_doppler_widths"], work_dtype, runtime_device
    )
    mass_density = to_dev(state["mass_density"], work_dtype, runtime_device)
    electron_density = to_dev(state["electron_density"], DEFAULT_DTYPE, runtime_device)
    temperature = to_dev(state["temperature"], work_dtype, runtime_device)
    hc_over_kt = to_dev(
        state["hc_over_kt"],
        work_dtype,
        runtime_device,
    )
    collision_density_proxy = to_dev(
        state["collision_density_proxy"],
        DEFAULT_DTYPE,
        runtime_device,
    )
    continuum_opacity = to_dev(
        state["continuum_opacity"], DEFAULT_DTYPE, runtime_device
    )

    if host_accumulator is not None and (do_helium or apply_stim):
        raise ValueError(
            "host_accumulator is only supported for unstimmed metal-only deposits"
        )
    line_mass_absorption_coefficient = (
        output_line_mass_absorption_coefficient
        if output_line_mass_absorption_coefficient is not None
        else torch.zeros(
            (n_depths, n_wavelengths),
            dtype=ACCUMULATION_DTYPE,
            device=runtime_device,
        )
    )

    if do_metal:
        _accumulate_metal(
            line_mass_absorption_coefficient,
            invariants,
            partition_normalized_populations,
            fractional_doppler_widths,
            mass_density,
            electron_density,
            hc_over_kt,
            collision_density_proxy,
            continuum_opacity,
            wing_mode=wing_mode,
            host_accumulator=host_accumulator,
        )
        _accumulate_autoionizing(
            line_mass_absorption_coefficient,
            invariants,
            partition_normalized_populations,
            mass_density,
            hc_over_kt,
            continuum_opacity,
        )
    if do_helium:
        _accumulate_helium(
            line_mass_absorption_coefficient,
            invariants,
            partition_normalized_populations,
            fractional_doppler_widths,
            mass_density,
            electron_density,
            temperature,
            hc_over_kt,
            collision_density_proxy,
            continuum_opacity,
            state,
        )

    if apply_stim:
        frequency_grid_hz = (LIGHT_SPEED_NM_PER_S / invariants.wavelength_grid).to(
            DEFAULT_DTYPE
        )
        photon_temperature_factor = (
            PLANCK_ERG_SECOND / (BOLTZMANN_ERG_PER_K * temperature)
        ).to(DEFAULT_DTYPE)
        stimulated_emission_factor = 1.0 - torch.exp(
            -frequency_grid_hz[None, :] * photon_temperature_factor[:, None]
        )
        line_mass_absorption_coefficient = (
            line_mass_absorption_coefficient * stimulated_emission_factor
        )
    return line_mass_absorption_coefficient


def _accumulate_autoionizing(
    line_mass_absorption_coefficient_grid,
    invariants,
    partition_normalized_populations,
    mass_density,
    hc_over_kt,
    continuum_opacity,
):
    """Accumulate type-1 autoionizing line opacity."""
    n_auto_lines = int(invariants.auto_oscillator_strength.numel())
    if n_auto_lines == 0:
        return

    profile_dtype = invariants.wavelength_grid.dtype
    n_depths = line_mass_absorption_coefficient_grid.shape[0]
    n_wavelengths = invariants.n_wavelengths
    freq_grid = LIGHT_SPEED_CM_PER_S / (
        invariants.wavelength_grid.to(profile_dtype) * NANOMETER_TO_CENTIMETER
    )
    flat_populations = partition_normalized_populations.reshape(n_depths, -1)
    population_columns = (
        invariants.auto_population_ion_stage_index
        * partition_normalized_populations.shape[2]
        + invariants.auto_population_element_index
    )
    auto_populations = flat_populations[:, population_columns]
    mass_density_column = mass_density[:, None]
    mass_density_safe = torch.where(
        mass_density_column > 0,
        mass_density_column,
        torch.ones_like(mass_density_column),
    )
    population_per_mass = torch.where(
        mass_density_column > 0,
        auto_populations / mass_density_safe,
        torch.zeros_like(auto_populations),
    )
    excitation_weight = fast_ex(
        invariants.auto_lower_excitation_cm[None, :] * hc_over_kt[:, None],
        invariants.exponential_integer_table,
        invariants.exponential_fraction_table,
    )
    line_amplitude = (
        invariants.auto_van_der_waals_damping.to(profile_dtype)[None, :]
        * invariants.auto_oscillator_strength.to(profile_dtype)[None, :]
        * population_per_mass
        * excitation_weight
    )
    center_on_grid = (invariants.auto_center_index >= 0) & (
        invariants.auto_center_index < n_wavelengths
    )
    center_cutoff = (
        continuum_opacity[:, invariants.auto_center_clamped].to(profile_dtype)
        * LINE_CENTER_CUTOFF_RATIO
    )
    center_deposits = (
        (population_per_mass > 0)
        & (line_amplitude > 0)
        & (line_amplitude >= center_cutoff)
        & center_on_grid[None, :]
    )

    # Hoist the loop-control scalars to the host in one transfer each so the
    # per-line loop pays no device sync per line. Values are unchanged; only
    # where the host reads them moves.
    center_on_grid_host = center_on_grid.cpu().tolist()
    line_has_active_depth_host = center_deposits.any(dim=0).cpu().tolist()
    auto_center_index_host = invariants.auto_center_index.cpu().tolist()

    for auto_index in range(n_auto_lines):
        if not center_on_grid_host[auto_index]:
            continue
        if not line_has_active_depth_host[auto_index]:
            continue
        active_depths = center_deposits[:, auto_index]
        center_column = int(auto_center_index_host[auto_index])
        amplitude_column = line_amplitude[:, auto_index]
        line_mass_absorption_coefficient_grid[:, center_column] += torch.where(
            active_depths,
            amplitude_column,
            torch.zeros_like(amplitude_column),
        ).to(line_mass_absorption_coefficient_grid.dtype)

        line_wavelength_nm = invariants.auto_wavelength_nm[auto_index].to(profile_dtype)
        line_frequency_hz = LIGHT_SPEED_CM_PER_S / (
            line_wavelength_nm * NANOMETER_TO_CENTIMETER
        )
        radiative_width = max(
            abs(float(invariants.auto_radiative_damping[auto_index].item())), 1.0e-30
        )
        shore_asymmetry = float(invariants.auto_stark_damping[auto_index].item())
        shore_baseline = float(invariants.auto_van_der_waals_damping[auto_index].item())
        if abs(shore_baseline) < 1.0e-30:
            shore_baseline = 1.0e-30

        red_active = active_depths.clone()
        blue_active = active_depths.clone()
        max_offset = max(center_column, n_wavelengths - center_column - 1)

        # Blocked shore walk. The per-offset survival mask is a running AND
        # (once a depth falls below the cutoff it never re-activates), so a
        # block-wise cumulative product reproduces the sequential walk
        # value-for-value while syncing the host once per block instead of
        # twice per offset.
        shore_block = 1024
        for block_start in range(1, max_offset + 1, shore_block):
            if not (bool(red_active.any()) or bool(blue_active.any())):
                break
            block_stop = min(block_start + shore_block - 1, max_offset)
            offsets = torch.arange(
                block_start, block_stop + 1, device=amplitude_column.device
            )
            for direction, active in (("red", red_active), ("blue", blue_active)):
                if direction == "red":
                    columns = center_column + offsets
                    valid = columns < n_wavelengths
                else:
                    columns = center_column - offsets
                    valid = columns >= 0
                if not bool(valid.any()) or not bool(active.any()):
                    if direction == "red":
                        red_active = torch.zeros_like(red_active)
                    else:
                        blue_active = torch.zeros_like(blue_active)
                    continue
                columns_clamped = torch.clamp(columns, 0, n_wavelengths - 1)
                shore_offset = (
                    2.0
                    * (freq_grid[columns_clamped] - line_frequency_hz)
                    / radiative_width
                )
                profile_ratio = (
                    (shore_asymmetry * shore_offset + shore_baseline)
                    / (shore_offset * shore_offset + 1.0)
                    / shore_baseline
                )
                opacity_value = amplitude_column[:, None] * profile_ratio[None, :]
                cutoff = (
                    continuum_opacity[:, columns_clamped].to(profile_dtype)
                    * LINE_CENTER_CUTOFF_RATIO
                )
                keep_step = (
                    valid[None, :] & (opacity_value > 0) & (opacity_value >= cutoff)
                )
                survival = torch.cumprod(keep_step.to(torch.int8), dim=1).to(torch.bool)
                survival &= active[:, None]
                deposit = torch.where(
                    survival, opacity_value, torch.zeros_like(opacity_value)
                ).to(line_mass_absorption_coefficient_grid.dtype)
                line_mass_absorption_coefficient_grid.index_add_(
                    1,
                    columns_clamped,
                    torch.where(valid[None, :], deposit, torch.zeros_like(deposit)),
                )
                carry = survival[:, -1]
                if direction == "red":
                    red_active = carry
                else:
                    blue_active = carry


def _accumulate_metal(
    line_mass_absorption_coefficient_grid,
    invariants,
    partition_normalized_populations,
    fractional_doppler_widths,
    mass_density,
    electron_density,
    hc_over_kt,
    collision_density_proxy,
    continuum_opacity,
    wing_mode="batched",
    host_accumulator=None,
):
    """Accumulate metal-line centers and Voigt wings."""
    device = line_mass_absorption_coefficient_grid.device
    work_dtype = invariants.wavelength_grid.dtype
    n_wavelengths = invariants.n_wavelengths
    n_depths = line_mass_absorption_coefficient_grid.shape[0]

    metal_classical_strength = invariants.metal_classical_strength
    metal_lower_excitation_cm = invariants.metal_lower_excitation_cm
    metal_radiative_damping = invariants.metal_radiative_damping
    metal_stark_damping = invariants.metal_stark_damping
    metal_van_der_waals_damping = invariants.metal_van_der_waals_damping
    metal_wavelength_nm = invariants.metal_wavelength_nm
    metal_center_index = invariants.metal_center_index
    metal_wing_index = invariants.metal_wing_index
    metal_center_clamped = invariants.metal_center_clamped
    metal_wing_clamped = invariants.metal_wing_clamped

    population_ion_stage_index = invariants.metal_population_ion_stage_index
    population_element_index = invariants.metal_population_element_index

    flat_populations = partition_normalized_populations.reshape(n_depths, -1)
    population_columns = (
        population_ion_stage_index * partition_normalized_populations.shape[2]
        + population_element_index
    )
    metal_populations = flat_populations[:, population_columns]
    metal_doppler_fractions = fractional_doppler_widths.reshape(n_depths, -1)[
        :, population_columns
    ]

    mass_density_column = mass_density[:, None]
    valid_metal_levels = (
        (metal_populations > 0)
        & (metal_doppler_fractions > 0)
        & (mass_density_column > 0)
    )
    doppler_fraction_safe = torch.where(
        metal_doppler_fractions > 0,
        metal_doppler_fractions,
        torch.ones_like(metal_doppler_fractions),
    )
    mass_density_safe = torch.where(
        mass_density_column > 0,
        mass_density_column,
        torch.ones_like(mass_density_column),
    )
    population_doppler_ratio = torch.where(
        valid_metal_levels,
        metal_populations / (mass_density_safe * doppler_fraction_safe),
        torch.zeros_like(metal_populations),
    )

    classical_strength = metal_classical_strength.to(work_dtype)[None, :]
    pre_excitation_strength = classical_strength * population_doppler_ratio

    excitation_weight = fast_ex(
        metal_lower_excitation_cm[None, :] * hc_over_kt[:, None],
        invariants.exponential_integer_table,
        invariants.exponential_fraction_table,
    )
    line_amplitude = pre_excitation_strength * excitation_weight

    center_cutoff = (
        continuum_opacity[:, metal_center_clamped].to(work_dtype)
        * LINE_CENTER_CUTOFF_RATIO
    )
    passes_center_cutoff = (
        valid_metal_levels
        & (pre_excitation_strength >= center_cutoff)
        & (line_amplitude >= center_cutoff)
        & (line_amplitude > 0)
    )

    total_damping = (
        metal_radiative_damping[None, :].to(work_dtype)
        + metal_stark_damping[None, :].to(work_dtype)
        * electron_density.to(work_dtype)[:, None]
        + metal_van_der_waals_damping[None, :].to(work_dtype)
        * collision_density_proxy.to(work_dtype)[:, None]
    )
    doppler_fraction = metal_doppler_fractions
    damping_ratio = torch.where(
        (metal_doppler_fractions > 0) & (doppler_fraction > 0),
        total_damping / doppler_fraction,
        torch.zeros_like(total_damping),
    )

    center_damping_ratio = damping_ratio
    low_damping_center_opacity = line_amplitude * (1.0 - 1.128 * center_damping_ratio)
    full_center_profile = harris_profile_at_line_center(
        center_damping_ratio.to(DEFAULT_DTYPE),
        invariants.harris_profile_h0_table,
        invariants.harris_profile_h1_table,
        invariants.harris_profile_h2_table,
    ).to(work_dtype)
    full_profile_center_opacity = line_amplitude * full_center_profile
    center_opacity = torch.where(
        center_damping_ratio < 0.2,
        low_damping_center_opacity,
        full_profile_center_opacity,
    )
    center_deposits = (
        passes_center_cutoff & (center_damping_ratio >= 0.0) & (line_amplitude > 0)
    )
    center_opacity = torch.where(
        center_deposits, center_opacity, torch.zeros_like(center_opacity)
    )

    center_index = metal_center_index
    center_on_grid = (center_index >= 0) & (center_index < n_wavelengths)
    center_columns = torch.clamp(center_index, 0, n_wavelengths - 1)[None, :].expand(
        n_depths,
        -1,
    )
    center_mask = center_deposits & center_on_grid[None, :]
    _scatter_add_rows(
        line_mass_absorption_coefficient_grid,
        center_columns,
        center_opacity,
        center_mask,
    )

    wing_damping_ratio = torch.clamp(center_damping_ratio, min=1e-12)
    wing_center_profile = harris_profile_at_line_center(
        wing_damping_ratio.to(DEFAULT_DTYPE),
        invariants.harris_profile_h0_table,
        invariants.harris_profile_h1_table,
        invariants.harris_profile_h2_table,
    ).to(work_dtype)
    wing_profile_amplitude = torch.where(
        center_opacity > 0,
        center_opacity / wing_center_profile,
        torch.zeros_like(center_opacity),
    )

    wing_cutoff_reference = torch.maximum(
        continuum_opacity[:, metal_wing_clamped].to(work_dtype)
        * LINE_CENTER_CUTOFF_RATIO,
        continuum_opacity[:, metal_wing_clamped].to(work_dtype)
        * WING_CUTOFF_FLOOR_RATIO,
    )

    line_wavelength_nm = metal_wavelength_nm
    doppler_width_nm = metal_doppler_fractions * line_wavelength_nm[None, :]

    max_profile_steps = MAX_WING_PROFILE_STEPS
    wing_anchor_index = metal_wing_index
    wing_anchor_reachable = (wing_anchor_index >= -max_profile_steps) & (
        wing_anchor_index <= n_wavelengths - 1 + max_profile_steps
    )
    wing_depth_pairs = center_deposits & (center_opacity > 0)
    line_has_wing = wing_depth_pairs.any(dim=0) & wing_anchor_reachable
    if not bool(line_has_wing.any()):
        return

    live_line_indices = torch.nonzero(line_has_wing, as_tuple=False).squeeze(1)
    live_wing_pairs = wing_depth_pairs[:, live_line_indices]
    live_wing_profile_amplitude = torch.where(
        live_wing_pairs,
        wing_profile_amplitude[:, live_line_indices],
        torch.zeros_like(wing_profile_amplitude[:, live_line_indices]),
    )
    live_doppler_width_nm = torch.where(
        live_wing_pairs,
        doppler_width_nm[:, live_line_indices],
        torch.zeros_like(doppler_width_nm[:, live_line_indices]),
    )
    live_wing_damping_ratio = wing_damping_ratio[:, live_line_indices]
    live_wing_cutoff = wing_cutoff_reference[:, live_line_indices]
    live_wavelength_nm = line_wavelength_nm[live_line_indices]
    live_wing_anchor_columns = wing_anchor_index[live_line_indices]

    if wing_mode == "loop":
        for live_position in range(int(live_line_indices.numel())):
            wing_anchor_column = int(live_wing_anchor_columns[live_position].item())
            wing_anchor_by_depth = torch.full(
                (n_depths,),
                wing_anchor_column,
                device=device,
                dtype=torch.int64,
            )
            _wing_walk_metal(
                line_mass_absorption_coefficient_grid,
                invariants,
                wing_anchor_by_depth,
                live_wing_profile_amplitude[:, live_position],
                live_wing_damping_ratio[:, live_position],
                live_doppler_width_nm[:, live_position],
                float(live_wavelength_nm[live_position].item()),
                live_wing_cutoff[:, live_position],
            )
        return

    (
        reach_steps,
        use_far_wing,
        ten_doppler_steps,
        doppler_offset_per_pixel,
        far_wing_coefficient,
    ) = _wing_reach_batched(
        invariants,
        live_wing_anchor_columns,
        live_wing_profile_amplitude,
        live_wing_damping_ratio,
        live_doppler_width_nm,
        live_wavelength_nm,
        live_wing_cutoff,
        live_wing_pairs,
    )

    line_reach_steps = reach_steps.max(dim=0).values
    is_narrow_wing = line_reach_steps <= NARROW_WING_MAX_REACH
    narrow_line_indices = torch.nonzero(is_narrow_wing, as_tuple=False).squeeze(1)
    wide_line_indices = torch.nonzero(~is_narrow_wing, as_tuple=False).squeeze(1)

    if narrow_line_indices.numel() > 0:
        _wing_walk_narrow_batched(
            line_mass_absorption_coefficient_grid,
            invariants,
            live_wing_anchor_columns[narrow_line_indices],
            live_wing_profile_amplitude[:, narrow_line_indices],
            live_wing_damping_ratio[:, narrow_line_indices],
            reach_steps[:, narrow_line_indices],
            use_far_wing[:, narrow_line_indices],
            ten_doppler_steps[:, narrow_line_indices],
            doppler_offset_per_pixel[:, narrow_line_indices],
            far_wing_coefficient[:, narrow_line_indices],
            n_wavelengths,
        )

    for live_position in wide_line_indices.tolist():
        wing_anchor_column = int(live_wing_anchor_columns[live_position].item())
        wing_anchor_by_depth = torch.full(
            (n_depths,),
            wing_anchor_column,
            device=device,
            dtype=torch.int64,
        )
        _wing_walk_metal(
            line_mass_absorption_coefficient_grid,
            invariants,
            wing_anchor_by_depth,
            live_wing_profile_amplitude[:, live_position],
            live_wing_damping_ratio[:, live_position],
            live_doppler_width_nm[:, live_position],
            float(live_wavelength_nm[live_position].item()),
            live_wing_cutoff[:, live_position],
        )


def _accumulate_helium(
    line_mass_absorption_coefficient_grid,
    invariants,
    partition_normalized_populations,
    fractional_doppler_widths,
    mass_density,
    electron_density,
    temperature,
    hc_over_kt,
    collision_density_proxy,
    continuum_opacity,
    state,
):
    """Accumulate helium line opacity with per-depth continuum-merge tapers."""
    compute_device = line_mass_absorption_coefficient_grid.device
    work_dtype = invariants.wavelength_grid.dtype
    n_depths = line_mass_absorption_coefficient_grid.shape[0]
    n_helium_lines = invariants.helium_classical_strength.numel()
    if n_helium_lines == 0:
        return

    merge_start_by_depth = state.get("helium_core_weight_grid")
    merge_tail_by_depth = state.get("helium_tail_weight_grid")
    if merge_start_by_depth is None:
        merge_start_by_depth = torch.zeros(
            (n_depths, n_helium_lines),
            dtype=DEFAULT_DTYPE,
            device=compute_device,
        )
    else:
        merge_start_by_depth = to_dev(
            merge_start_by_depth, DEFAULT_DTYPE, compute_device
        )
    if merge_tail_by_depth is None:
        merge_tail_by_depth = torch.zeros(
            (n_depths, n_helium_lines),
            dtype=DEFAULT_DTYPE,
            device=compute_device,
        )
    else:
        merge_tail_by_depth = to_dev(merge_tail_by_depth, DEFAULT_DTYPE, compute_device)

    flat_populations = partition_normalized_populations.reshape(n_depths, -1)
    flat_fractional_doppler_widths = fractional_doppler_widths.reshape(n_depths, -1)
    population_columns = (
        invariants.helium_population_ion_stage_index
        * partition_normalized_populations.shape[2]
        + invariants.helium_population_element_index
    )
    helium_populations = flat_populations[:, population_columns]
    helium_doppler_fractions = flat_fractional_doppler_widths[:, population_columns]
    mass_density_column = mass_density[:, None]
    doppler_fraction_safe = torch.clamp(helium_doppler_fractions, min=1e-40)
    valid_he_levels = (
        (helium_populations > 0)
        & (helium_doppler_fractions > 0)
        & (mass_density_column > 0)
    )
    mass_density_safe = torch.where(
        mass_density_column > 0,
        mass_density_column,
        torch.ones_like(mass_density_column),
    )
    population_doppler_ratio = torch.where(
        valid_he_levels,
        helium_populations / (mass_density_safe * doppler_fraction_safe),
        torch.zeros_like(helium_populations),
    )
    pre_excitation_strength = (
        invariants.helium_classical_strength.to(work_dtype)[None, :]
        * population_doppler_ratio
    )
    excitation_weight = fast_ex(
        invariants.helium_lower_excitation_cm[None, :] * hc_over_kt[:, None],
        invariants.exponential_integer_table,
        invariants.exponential_fraction_table,
    )
    center_cutoff = (
        continuum_opacity[:, invariants.helium_center_index].to(work_dtype)
        * invariants.helium_cutoff
    )
    valid_he_levels = valid_he_levels & (pre_excitation_strength >= center_cutoff)
    line_amplitude = pre_excitation_strength * excitation_weight
    valid_he_levels = valid_he_levels & (line_amplitude >= center_cutoff)

    total_damping = (
        invariants.helium_radiative_damping[None, :].to(work_dtype)
        + invariants.helium_stark_damping[None, :].to(work_dtype)
        * electron_density.to(work_dtype)[:, None]
        + invariants.helium_van_der_waals_damping[None, :].to(work_dtype)
        * collision_density_proxy.to(work_dtype)[:, None]
    )
    damping_ratio = total_damping / doppler_fraction_safe

    line_amplitude = torch.where(
        valid_he_levels,
        line_amplitude,
        torch.zeros_like(line_amplitude),
    )
    doppler_width_nm = torch.where(
        valid_he_levels,
        helium_doppler_fractions * invariants.helium_wavelength_nm[None, :],
        torch.zeros_like(helium_doppler_fractions),
    )
    damping_ratio = torch.where(
        valid_he_levels,
        damping_ratio,
        torch.zeros_like(damping_ratio),
    )

    is_helium3 = (invariants.helium_line_type == -4)[None, :]
    effective_amplitude = torch.where(
        is_helium3,
        line_amplitude / HELIUM3_ISOTOPE_SCALE,
        line_amplitude,
    )
    effective_doppler_width_nm = torch.where(
        is_helium3,
        doppler_width_nm * HELIUM3_ISOTOPE_SCALE,
        doppler_width_nm,
    )
    effective_damping_ratio = torch.where(
        is_helium3,
        damping_ratio / HELIUM3_ISOTOPE_SCALE,
        damping_ratio,
    )

    live_lines = (effective_amplitude > 0).any(dim=0)
    if not bool(live_lines.any()):
        return

    continuum_opacity_hp = continuum_opacity.to(work_dtype)
    has_merge_taper = (merge_start_by_depth > 0).any(dim=0) | (
        merge_tail_by_depth > 0
    ).any(dim=0)
    live_without_taper = live_lines & (~has_merge_taper)

    untapered_reach = _helium_reach_batched(
        invariants,
        continuum_opacity_hp,
        effective_amplitude,
        effective_doppler_width_nm,
        effective_damping_ratio,
        live_without_taper,
    )

    for helium_index in torch.nonzero(live_lines, as_tuple=False).squeeze(1).tolist():
        center_column = int(invariants.helium_center_index[helium_index].item())
        line_wavelength_nm = float(invariants.helium_wavelength_nm[helium_index].item())
        if bool(has_merge_taper[helium_index].item()):
            window = None
        else:
            window_radius = int(untapered_reach[helium_index].item()) + HE_REACH_PAD
            window = (
                max(0, center_column - window_radius),
                min(invariants.n_wavelengths, center_column + window_radius + 1),
            )
        _wing_walk_helium(
            line_mass_absorption_coefficient_grid,
            continuum_opacity_hp,
            invariants,
            center_column,
            line_wavelength_nm,
            effective_amplitude[:, helium_index],
            effective_doppler_width_nm[:, helium_index],
            effective_damping_ratio[:, helium_index],
            invariants.helium_cutoff,
            merge_start_by_depth[:, helium_index].to(DEFAULT_DTYPE),
            merge_tail_by_depth[:, helium_index].to(DEFAULT_DTYPE),
            window=window,
        )


def _helium_reach_batch_size(
    device: torch.device | str,
    selected_lines: int,
    n_wavelengths: int,
) -> int:
    """Bound accelerator reach tensors without changing per-line results.

    The reach search allocates tensors proportional to wavelength samples times
    the number of helium lines in a batch.  Sixteen lines is efficient for the
    usual survey windows, but a native optical grid is an order of magnitude
    longer.  Cap that product so the same public path remains memory safe.
    """

    device_type = torch.device(device).type
    # CPU hosts have more addressable memory than one accelerator allocation,
    # so retain the historical all-line batch on ordinary grids.  Still cap a
    # native optical grid to avoid multi-tens-of-GB reach temporaries.
    product_budget = 2_500_000 if device_type == "cpu" else 1_500_000
    wavelength_bounded = max(1, product_budget // max(1, int(n_wavelengths)))
    backend_limit = int(selected_lines) if device_type == "cpu" else 16
    return min(backend_limit, int(selected_lines), wavelength_bounded)


def _helium_reach_batched(
    invariants,
    continuum_opacity_hp,
    effective_amplitude,
    effective_doppler_width_nm,
    effective_damping_ratio,
    selected_lines,
):
    """Furthest untapered above-cutoff pixel offset for each helium line."""
    compute_device = effective_amplitude.device
    n_depths, n_helium_lines = effective_amplitude.shape
    untapered_reach = torch.zeros(
        n_helium_lines,
        dtype=torch.int64,
        device=compute_device,
    )
    selected_indices = torch.nonzero(selected_lines, as_tuple=False).squeeze(1)
    if selected_indices.numel() == 0:
        return untapered_reach

    wavelength_grid = invariants.wavelength_grid
    cutoff_opacity = continuum_opacity_hp * invariants.helium_cutoff
    # Chunking only regroups independent per-line reach calculations. Keep
    # accelerator temporaries bounded; CPU can walk the selected set at once.
    batch_size = _helium_reach_batch_size(
        compute_device,
        int(selected_indices.numel()),
        int(wavelength_grid.numel()),
    )

    for start in range(0, int(selected_indices.numel()), batch_size):
        line_indices = selected_indices[start : start + batch_size]
        center_columns = invariants.helium_center_index[line_indices]
        line_wavelength_nm = invariants.helium_wavelength_nm[line_indices]
        line_amplitude = effective_amplitude[:, line_indices]
        damping_ratio_safe = torch.clamp(
            effective_damping_ratio[:, line_indices],
            min=1e-12,
        )
        doppler_width_safe = torch.where(
            effective_doppler_width_nm[:, line_indices] > 0,
            effective_doppler_width_nm[:, line_indices],
            torch.ones_like(effective_doppler_width_nm[:, line_indices]),
        )

        probe_radius = 8
        while True:
            offsets = torch.arange(1, probe_radius + 1, device=compute_device)
            red_columns = torch.clamp(
                center_columns[:, None] + offsets[None, :],
                0,
                invariants.n_wavelengths - 1,
            )
            blue_columns = torch.clamp(
                center_columns[:, None] - offsets[None, :],
                0,
                invariants.n_wavelengths - 1,
            )

            def above_cutoff(columns):
                probe_wavelength_nm = wavelength_grid[columns]
                doppler_offset = (
                    probe_wavelength_nm[None, :, :] - line_wavelength_nm[None, :, None]
                ).abs() / doppler_width_safe[:, :, None]
                profile = interpolate_harris_profile(
                    doppler_offset,
                    damping_ratio_safe[:, :, None],
                    invariants.harris_profile_h0_table,
                    invariants.harris_profile_h1_table,
                    invariants.harris_profile_h2_table,
                )
                return (line_amplitude[:, :, None] * profile) >= cutoff_opacity[
                    :, columns
                ]

            above_cutoff_mask = above_cutoff(red_columns) | above_cutoff(blue_columns)
            any_offset = above_cutoff_mask.any(dim=0)
            furthest_offset = (
                torch.where(
                    any_offset,
                    offsets[None, :],
                    torch.zeros_like(offsets)[None, :],
                )
                .max(dim=1)
                .values
            )
            outer_edge_hit = any_offset[:, -1].any()
            if not bool(outer_edge_hit) or probe_radius >= invariants.n_wavelengths:
                untapered_reach[line_indices] = furthest_offset
                break
            probe_radius = min(probe_radius * 4, invariants.n_wavelengths)

    return untapered_reach
