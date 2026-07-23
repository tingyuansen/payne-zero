"""Batched radiative transfer for continuum-normalized spectra.

Each wavelength is one batch row. The solver integrates optical depth over the
atmosphere, maps the thermal source and scattering fraction onto a fixed
51-point optical-depth grid, iterates the scattering source with backward
Gauss-Seidel, and returns the emergent Eddington flux. The fixed transfer
operators are loaded once and reused for every spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .constants import BOLTZMANN_ERG_PER_K, LIGHT_SPEED_NM_PER_S, PLANCK_ERG_SECOND
from .device import DEVICE, DEFAULT_DTYPE

FLOAT32_POSITIVE_FLOOR = 1.0e-38
ITER_TOL = 1.0e-5
MAX_ITER = 51

DEFAULT_SWEEPS = 8

PLANCK_PREFACTOR = 1.47439e-2


@dataclass
class TransferTables:
    """Wavelength-independent transfer operators resident on one device."""

    transfer_optical_depth_grid: torch.Tensor
    mean_intensity_operator: torch.Tensor
    mean_intensity_diagonal: torch.Tensor
    surface_eddington_flux_weights: torch.Tensor

    @property
    def n_grid(self) -> int:
        return self.transfer_optical_depth_grid.shape[0]

    @classmethod
    def from_npz(
        cls,
        transfer_tables_path: str | Path,
        device: torch.device | None = None,
        dtype: torch.dtype = DEFAULT_DTYPE,
    ) -> "TransferTables":
        """Load transfer tables onto ``device`` in ``dtype``."""
        compute_device = device if device is not None else DEVICE
        transfer_data = np.load(str(transfer_tables_path))
        transfer_optical_depth_grid = torch.as_tensor(
            transfer_data["transfer_optical_depth_grid"],
            dtype=dtype,
            device=compute_device,
        )
        mean_intensity_operator = torch.as_tensor(
            transfer_data["mean_intensity_operator"],
            dtype=dtype,
            device=compute_device,
        )
        surface_eddington_flux_weights = torch.as_tensor(
            transfer_data["surface_eddington_flux_weights"],
            dtype=dtype,
            device=compute_device,
        )
        mean_intensity_diagonal = torch.diagonal(mean_intensity_operator).contiguous()
        return cls(
            transfer_optical_depth_grid=transfer_optical_depth_grid,
            mean_intensity_operator=mean_intensity_operator,
            mean_intensity_diagonal=mean_intensity_diagonal,
            surface_eddington_flux_weights=surface_eddington_flux_weights,
        )

    def to(
        self, device: torch.device, dtype: torch.dtype | None = None
    ) -> "TransferTables":
        """Move/convert all operators to `device` (and optionally cast)."""

        def move_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            target_dtype = dtype if dtype is not None else tensor.dtype
            return tensor.to(device=device, dtype=target_dtype)

        return TransferTables(
            transfer_optical_depth_grid=move_tensor(self.transfer_optical_depth_grid),
            mean_intensity_operator=move_tensor(self.mean_intensity_operator),
            mean_intensity_diagonal=move_tensor(self.mean_intensity_diagonal),
            surface_eddington_flux_weights=move_tensor(
                self.surface_eddington_flux_weights
            ),
        )


def planck_bnu(wavelength_nm: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    """Planck B_nu(T) on a ``[depth, wavelength]`` grid."""
    frequency_hz = LIGHT_SPEED_NM_PER_S / wavelength_nm
    photon_energy_over_thermal_energy = (PLANCK_ERG_SECOND * frequency_hz)[None, :] / (
        BOLTZMANN_ERG_PER_K * temperature[:, None]
    )
    boltzmann_factor = torch.exp(-photon_energy_over_thermal_energy)
    return (
        PLANCK_PREFACTOR
        * (frequency_hz[None, :] / 1e15) ** 3
        * boltzmann_factor
        / (1.0 - boltzmann_factor)
    )


def source_and_alpha(
    continuum_absorption: torch.Tensor,
    continuum_source: torch.Tensor,
    line_mass_absorption_coefficient: torch.Tensor,
    line_source: torch.Tensor,
    continuum_scattering: torch.Tensor,
    line_scattering: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total extinction, scattering fraction, and thermal source."""
    total_extinction = torch.clamp(
        continuum_absorption
        + line_mass_absorption_coefficient
        + continuum_scattering
        + line_scattering,
        min=FLOAT32_POSITIVE_FLOOR,
    )
    scattering_fraction = torch.clamp(
        (continuum_scattering + line_scattering) / total_extinction,
        0.0,
        1.0,
    )
    absorbing_opacity = continuum_absorption + line_mass_absorption_coefficient
    thermal_source = torch.where(
        absorbing_opacity > 0,
        (
            continuum_absorption * continuum_source
            + line_mass_absorption_coefficient * line_source
        )
        / absorbing_opacity,
        continuum_source,
    )
    return total_extinction, scattering_fraction, thermal_source


def _parabolic_interval_coefficients(
    values: torch.Tensor,
    depth_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit ``constant + linear*x + quadratic*x^2`` over each depth interval."""
    _batch_size, n_depth = values.shape
    constant = torch.zeros_like(values)
    linear = torch.zeros_like(values)
    quadratic = torch.zeros_like(values)
    if n_depth == 1:
        constant[:, 0] = values[:, 0]
        return constant, linear, quadratic

    linear[:, 0] = (values[:, 1] - values[:, 0]) / (depth_grid[1] - depth_grid[0])
    constant[:, 0] = values[:, 0] - depth_grid[0] * linear[:, 0]
    last_depth_index = n_depth - 1
    linear[:, -1] = (values[:, -1] - values[:, last_depth_index - 1]) / (
        depth_grid[-1] - depth_grid[last_depth_index - 1]
    )
    constant[:, -1] = values[:, -1] - depth_grid[-1] * linear[:, -1]
    if n_depth == 2:
        return constant, linear, quadratic

    center_depth = depth_grid[1:last_depth_index]
    previous_depth = depth_grid[0 : last_depth_index - 1]
    next_depth = depth_grid[2:n_depth]
    center_value = values[:, 1:last_depth_index]
    previous_value = values[:, 0 : last_depth_index - 1]
    next_value = values[:, 2:n_depth]

    left_slope = (center_value - previous_value) / (center_depth - previous_depth)
    interior_quadratic = next_value / (
        (next_depth - center_depth) * (next_depth - previous_depth)
    ) + (
        previous_value / (next_depth - previous_depth)
        - center_value / (next_depth - center_depth)
    ) / (center_depth - previous_depth)
    interior_linear = left_slope - (center_depth + previous_depth) * interior_quadratic
    interior_constant = (
        previous_value
        - previous_depth * left_slope
        + center_depth * previous_depth * interior_quadratic
    )
    constant[:, 1:last_depth_index] = interior_constant
    linear[:, 1:last_depth_index] = interior_linear
    quadratic[:, 1:last_depth_index] = interior_quadratic

    # Force points 2 and 3 (0-based 1,2) linear — no stable interior parabola near the top.
    quadratic[:, 1] = 0.0
    linear[:, 1] = (values[:, 2] - values[:, 1]) / (depth_grid[2] - depth_grid[1])
    constant[:, 1] = values[:, 1] - depth_grid[1] * linear[:, 1]
    if n_depth > 3:
        quadratic[:, 2] = 0.0
        linear[:, 2] = (values[:, 3] - values[:, 2]) / (depth_grid[3] - depth_grid[2])
        constant[:, 2] = values[:, 2] - depth_grid[2] * linear[:, 2]

    interval_index = torch.arange(1, last_depth_index, device=values.device)
    neighbor_index = torch.clamp(interval_index + 1, max=n_depth - 1)
    current_quadratic = quadratic[:, interval_index]
    neighbor_quadratic = quadratic[:, neighbor_index]
    current_constant = constant[:, interval_index]
    current_linear = linear[:, interval_index]
    neighbor_constant = constant[:, neighbor_index]
    neighbor_linear = linear[:, neighbor_index]

    blend_denominator = neighbor_quadratic.abs() + current_quadratic.abs()
    safe_blend_denominator = torch.where(
        blend_denominator > 0,
        blend_denominator,
        torch.ones_like(blend_denominator),
    )
    blend_weight = torch.where(
        blend_denominator > 0,
        neighbor_quadratic.abs() / safe_blend_denominator,
        torch.zeros_like(blend_denominator),
    ).detach()
    blended_constant = neighbor_constant + blend_weight * (
        current_constant - neighbor_constant
    )
    blended_linear = neighbor_linear + blend_weight * (current_linear - neighbor_linear)
    blended_quadratic = neighbor_quadratic + blend_weight * (
        current_quadratic - neighbor_quadratic
    )
    has_curvature = current_quadratic != 0.0
    constant[:, interval_index] = torch.where(
        has_curvature, blended_constant, current_constant
    )
    linear[:, interval_index] = torch.where(
        has_curvature, blended_linear, current_linear
    )
    quadratic[:, interval_index] = torch.where(
        has_curvature, blended_quadratic, current_quadratic
    )

    constant[:, last_depth_index - 1] = constant[:, -1]
    linear[:, last_depth_index - 1] = linear[:, -1]
    quadratic[:, last_depth_index - 1] = quadratic[:, -1]
    return constant, linear, quadratic


def integrate_optical_depth(
    column_mass: torch.Tensor,
    extinction: torch.Tensor,
    surface_tau: torch.Tensor,
) -> torch.Tensor:
    """Cumulative optical depth from parabolic interval integrals.

    ``column_mass`` is the shared depth grid [depth], ``extinction`` is
    [wavelength, depth], and ``surface_tau`` is the top half-cell seed for each
    wavelength.  The only sequential dependence is a prefix sum over depth.
    """
    constant, linear, quadratic = _parabolic_interval_coefficients(
        extinction, column_mass
    )
    n_depth = extinction.shape[1]
    optical_depth = torch.empty_like(extinction)
    optical_depth[:, 0] = surface_tau
    if n_depth == 1:
        return optical_depth

    left_column_mass = column_mass[0 : n_depth - 1]
    right_column_mass = column_mass[1:n_depth]
    interval_width = right_column_mass - left_column_mass
    interval_average = (
        constant[:, 0 : n_depth - 1]
        + 0.5 * linear[:, 0 : n_depth - 1] * (right_column_mass + left_column_mass)
        + (quadratic[:, 0 : n_depth - 1] / 3.0)
        * (
            (right_column_mass + left_column_mass) * right_column_mass
            + left_column_mass * left_column_mass
        )
    )
    interval_tau = interval_average * interval_width
    optical_depth[:, 1:] = surface_tau[:, None] + torch.cumsum(interval_tau, dim=1)
    return optical_depth


def _interpolate_to_transfer_grid(
    optical_depth: torch.Tensor,
    depth_values: torch.Tensor,
    transfer_depth_grid: torch.Tensor,
) -> torch.Tensor:
    """Parabolic interpolation from physical depths onto the fixed transfer grid.

    Each target depth chooses the bracketing physical layers and evaluates the
    same parabolic stencil for all wavelength rows at once.
    """
    n_wavelength, n_physical_depth = optical_depth.shape
    n_transfer_depth = transfer_depth_grid.shape[0]

    transfer_depths = (
        transfer_depth_grid[None, :].expand(n_wavelength, n_transfer_depth).contiguous()
    )
    bracket_ordinal = torch.searchsorted(optical_depth, transfer_depths, right=True) + 1
    bracket_ordinal = torch.clamp(bracket_ordinal, min=2, max=n_physical_depth)

    upper_index = bracket_ordinal - 1
    lower_index = upper_index - 1
    previous_index = upper_index - 2
    forward_index = upper_index + 1

    def gather_rows(
        row_values: torch.Tensor, row_indices: torch.Tensor
    ) -> torch.Tensor:
        bounded_indices = torch.clamp(row_indices, 0, n_physical_depth - 1)
        return torch.gather(row_values, 1, bounded_indices)

    upper_depth = gather_rows(optical_depth, upper_index)
    upper_value = gather_rows(depth_values, upper_index)
    lower_depth = gather_rows(optical_depth, lower_index)
    lower_value = gather_rows(depth_values, lower_index)
    previous_depth = gather_rows(optical_depth, previous_index)
    previous_value = gather_rows(depth_values, previous_index)
    forward_depth = gather_rows(optical_depth, forward_index)
    forward_value = gather_rows(depth_values, forward_index)

    linear_slope = (upper_value - lower_value) / (upper_depth - lower_depth)
    linear_constant = upper_value - upper_depth * linear_slope
    linear_curvature = torch.zeros_like(linear_constant)

    def finite_denominator(value: torch.Tensor) -> torch.Tensor:
        """Replace only boundary-stencil zero denominators in inactive branches."""

        return torch.where(value != 0.0, value, torch.ones_like(value))

    lower_previous_span = finite_denominator(lower_depth - previous_depth)
    upper_lower_span = finite_denominator(upper_depth - lower_depth)
    upper_previous_span = finite_denominator(upper_depth - previous_depth)
    forward_upper_span = finite_denominator(forward_depth - upper_depth)
    forward_lower_span = finite_denominator(forward_depth - lower_depth)

    backward_slope_seed = (lower_value - previous_value) / lower_previous_span
    backward_curvature = upper_value / (
        upper_lower_span * upper_previous_span
    ) + (
        previous_value / upper_previous_span - lower_value / upper_lower_span
    ) / lower_previous_span
    backward_slope = (
        backward_slope_seed - (lower_depth + previous_depth) * backward_curvature
    )
    backward_constant = (
        previous_value
        - previous_depth * backward_slope_seed
        + lower_depth * previous_depth * backward_curvature
    )

    forward_slope_seed = (upper_value - lower_value) / upper_lower_span
    forward_curvature = forward_value / (
        forward_upper_span * forward_lower_span
    ) + (
        lower_value / forward_lower_span - upper_value / forward_upper_span
    ) / upper_lower_span
    forward_slope = forward_slope_seed - (upper_depth + lower_depth) * forward_curvature
    forward_constant = (
        lower_value
        - lower_depth * forward_slope_seed
        + upper_depth * lower_depth * forward_curvature
    )
    forward_curvature_abs = forward_curvature.abs()
    curvature_sum = forward_curvature_abs + backward_curvature.abs()
    # Keep the inactive zero-curvature branch finite for autograd.  A raw 0/0
    # is hidden by ``where`` in the forward pass but still poisons its backward
    # graph with NaNs.
    safe_curvature_sum = torch.where(
        curvature_sum > 0.0,
        curvature_sum,
        torch.ones_like(curvature_sum),
    )
    curvature_weight = torch.where(
        forward_curvature_abs != 0.0,
        forward_curvature_abs / safe_curvature_sum,
        torch.zeros_like(forward_curvature),
    ).detach()
    # This limiter selects a local interpolation stencil.  Its derivative is
    # undefined when both curvatures approach zero (a locally linear source),
    # while the selected interpolation remains smooth in the depth values.
    # Treat the selection as fixed during backpropagation, as is already done
    # for the discrete searchsorted bracket above.
    blended_constant = forward_constant + curvature_weight * (
        backward_constant - forward_constant
    )
    blended_slope = forward_slope + curvature_weight * (backward_slope - forward_slope)
    blended_curvature = forward_curvature + curvature_weight * (
        backward_curvature - forward_curvature
    )

    use_linear = (bracket_ordinal == 2) | (bracket_ordinal == 3)
    use_backward = (~use_linear) & (bracket_ordinal >= n_physical_depth)

    constant = torch.where(
        use_linear,
        linear_constant,
        torch.where(use_backward, backward_constant, blended_constant),
    )
    slope = torch.where(
        use_linear,
        linear_slope,
        torch.where(use_backward, backward_slope, blended_slope),
    )
    curvature = torch.where(
        use_linear,
        linear_curvature,
        torch.where(use_backward, backward_curvature, blended_curvature),
    )

    transfer_depth = transfer_depth_grid[None, :]
    return constant + (slope + curvature * transfer_depth) * transfer_depth


def solve_scattering_source(
    thermal_source_grid: torch.Tensor,
    scattering_fraction_grid: torch.Tensor,
    lambda_operator: torch.Tensor,
    lambda_operator_diagonal: torch.Tensor,
    sweeps: int = DEFAULT_SWEEPS,
) -> torch.Tensor:
    """Solve the scattering source on the fixed transfer grid.

    The iteration is intentionally fp32: that is the arithmetic used by the
    validated reference.  Each grid point depends on already-updated deeper
    points, so the depth loop remains sequential while the wavelength batch stays
    vectorized.
    """
    n_transfer_depth = thermal_source_grid.shape[1]
    lambda_operator = lambda_operator.to(torch.float32)
    operator_diagonal = lambda_operator_diagonal.to(torch.float32)
    scattering_fraction = scattering_fraction_grid.to(torch.float32)
    thermal_source = thermal_source_grid.to(torch.float32)
    floor = torch.tensor(
        FLOAT32_POSITIVE_FLOOR, dtype=torch.float32, device=thermal_source_grid.device
    )

    thermal_emission = thermal_source * (1.0 - scattering_fraction)
    gauss_seidel_denominator = 1.0 - scattering_fraction * operator_diagonal

    scattering_fraction_by_depth = scattering_fraction.transpose(0, 1).contiguous()
    denominator_by_depth = gauss_seidel_denominator.transpose(0, 1).contiguous()
    thermal_emission_by_depth = thermal_emission.transpose(0, 1).contiguous()
    source_by_depth = thermal_source.transpose(0, 1).contiguous()

    for sweep_index in range(sweeps):
        for depth_index in range(n_transfer_depth - 1, -1, -1):
            mean_intensity = torch.matmul(lambda_operator[depth_index], source_by_depth)
            correction = (
                mean_intensity * scattering_fraction_by_depth[depth_index]
                + thermal_emission_by_depth[depth_index]
                - source_by_depth[depth_index]
            ) / denominator_by_depth[depth_index]
            source_by_depth[depth_index] = torch.maximum(
                source_by_depth[depth_index] + correction,
                floor,
            )
    return source_by_depth.transpose(0, 1).contiguous()


def _differentiate_rows(depth_grid: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Row-wise derivatives on nonuniform depth grids."""
    n_rows, n_depth = values.shape
    derivative = torch.zeros_like(values)
    if n_depth < 2:
        return derivative

    derivative[:, 0] = (values[:, 1] - values[:, 0]) / (
        depth_grid[:, 1] - depth_grid[:, 0]
    )
    derivative[:, -1] = (values[:, -1] - values[:, -2]) / (
        depth_grid[:, -1] - depth_grid[:, -2]
    )
    if n_depth == 2:
        return derivative

    center_depth = depth_grid[:, 1:-1]
    previous_values = values[:, :-2]
    center_values = values[:, 1:-1]
    next_values = values[:, 2:]
    previous_depths = depth_grid[:, :-2]
    center_depths = depth_grid[:, 1:-1]
    next_depths = depth_grid[:, 2:]

    value_scale = torch.maximum(
        torch.maximum(previous_values.abs(), center_values.abs()),
        next_values.abs(),
    )
    value_scale = torch.where(
        center_depth != 0.0, value_scale / center_depth.abs(), value_scale
    )
    value_scale = torch.where(
        value_scale == 0.0, torch.ones_like(value_scale), value_scale
    )

    right_slope = (
        (next_values - center_values) / (next_depths - center_depths) / value_scale
    )
    left_slope = (
        (center_values - previous_values)
        / (center_depths - previous_depths)
        / value_scale
    )
    orientation = torch.where(
        depth_grid[:, 1:2] != depth_grid[:, 0:1],
        (depth_grid[:, 1:2] - depth_grid[:, 0:1]).abs()
        / (depth_grid[:, 1:2] - depth_grid[:, 0:1]),
        torch.ones((n_rows, 1), dtype=depth_grid.dtype, device=depth_grid.device),
    )
    right_angle = right_slope / (
        orientation * torch.sqrt(1.0 + right_slope * right_slope) + 1.0
    )
    left_angle = left_slope / (
        orientation * torch.sqrt(1.0 + left_slope * left_slope) + 1.0
    )
    derivative[:, 1:-1] = (
        (right_angle + left_angle) / (1.0 - right_angle * left_angle) * value_scale
    )
    return derivative


def _saturated_core_flux(
    optical_depth: torch.Tensor,
    thermal_source: torch.Tensor,
    scattering_fraction: torch.Tensor,
) -> torch.Tensor:
    """Fallback flux for saturated cores whose surface is below the fixed grid."""
    dtype = optical_depth.dtype
    floor = torch.tensor(
        FLOAT32_POSITIVE_FLOOR, dtype=dtype, device=optical_depth.device
    )
    source = thermal_source.clone()

    if optical_depth.shape[1] > 1:
        depth_step = torch.diff(optical_depth, dim=1).abs()
        if depth_step.shape[1] > 2:
            min_deep_step = depth_step[:, 2:].min(dim=1).values
        else:
            min_deep_step = depth_step.min(dim=1).values
    else:
        min_deep_step = torch.zeros(
            optical_depth.shape[0], dtype=dtype, device=optical_depth.device
        )
    unstable_depth_spacing = min_deep_step < (1.0e-4 * optical_depth[:, 0].abs())

    monochromatic_eddington_flux = (
        _differentiate_rows(optical_depth, thermal_source) / 3.0
    )
    surface_eddington_flux = monochromatic_eddington_flux[:, 0]
    active = ~unstable_depth_spacing

    for iteration_index in range(MAX_ITER):
        if not bool(active.any()):
            break
        monochromatic_eddington_flux = _differentiate_rows(optical_depth, source) / 3.0
        curvature_correction = _differentiate_rows(
            optical_depth, monochromatic_eddington_flux
        )
        surface_eddington_flux = torch.where(
            active, monochromatic_eddington_flux[:, 0], surface_eddington_flux
        )

        max_correction = (
            (scattering_fraction * curvature_correction).abs().max(dim=1).values
        )
        max_thermal_source = torch.clamp(
            thermal_source.abs().max(dim=1).values, min=FLOAT32_POSITIVE_FLOOR
        )
        diverged = max_correction > max_thermal_source

        mean_intensity = curvature_correction + source
        updated_source = (
            1.0 - scattering_fraction
        ) * thermal_source + scattering_fraction * mean_intensity
        relative_change = (updated_source - source).abs() / torch.clamp(
            updated_source.abs(), min=floor
        )
        converged = relative_change.sum(dim=1) < ITER_TOL

        update = active & ~diverged
        source = torch.where(update[:, None], updated_source, source)
        active = active & ~diverged & ~converged
    return surface_eddington_flux


def _solve_flux_rows(
    continuum_absorption: torch.Tensor,
    continuum_source: torch.Tensor,
    line_mass_absorption_coefficient: torch.Tensor,
    line_source: torch.Tensor,
    continuum_scattering: torch.Tensor,
    line_scattering: torch.Tensor,
    column_mass: torch.Tensor,
    tables: TransferTables,
    sweeps: int = DEFAULT_SWEEPS,
    assert_no_saturated_core: bool = True,
) -> torch.Tensor:
    """Emergent Eddington flux for a batch of wavelength rows."""
    transfer_depth_grid = tables.transfer_optical_depth_grid
    extinction, scattering_fraction, thermal_source = source_and_alpha(
        continuum_absorption,
        continuum_source,
        line_mass_absorption_coefficient,
        line_source,
        continuum_scattering,
        line_scattering,
    )

    surface_tau = extinction[:, 0] * column_mass[0]
    optical_depth = integrate_optical_depth(column_mass, extinction, surface_tau)

    if assert_no_saturated_core:
        if bool((optical_depth[:, 0] > transfer_depth_grid[-1]).any()):
            raise NotImplementedError(
                "Saturated-core transfer requested. Pass assert_no_saturated_core=False to continue."
            )

    transfer_thermal_source = torch.clamp(
        _interpolate_to_transfer_grid(
            optical_depth, thermal_source, transfer_depth_grid
        ),
        min=FLOAT32_POSITIVE_FLOOR,
    )
    transfer_scattering_fraction = torch.clamp(
        _interpolate_to_transfer_grid(
            optical_depth, scattering_fraction, transfer_depth_grid
        ),
        0.0,
        1.0,
    )

    above_atmosphere = transfer_depth_grid[None, :] < optical_depth[:, 0:1]
    surface_source = torch.clamp(thermal_source[:, 0:1], min=FLOAT32_POSITIVE_FLOOR)
    surface_scattering = torch.clamp(scattering_fraction[:, 0:1], 0.0, 1.0)
    transfer_thermal_source = torch.where(
        above_atmosphere,
        surface_source.expand_as(transfer_thermal_source),
        transfer_thermal_source,
    )
    transfer_scattering_fraction = torch.where(
        above_atmosphere,
        surface_scattering.expand_as(transfer_scattering_fraction),
        transfer_scattering_fraction,
    )

    source = solve_scattering_source(
        transfer_thermal_source,
        transfer_scattering_fraction,
        tables.mean_intensity_operator,
        tables.mean_intensity_diagonal,
        sweeps=sweeps,
    )

    high_precision = (
        torch.float64 if transfer_depth_grid.device.type != "mps" else torch.float32
    )
    surface_eddington_flux_per_frequency = torch.matmul(
        source.to(high_precision),
        tables.surface_eddington_flux_weights.to(high_precision),
    )
    saturated = optical_depth[:, 0] > transfer_depth_grid[-1]
    if not assert_no_saturated_core and bool(saturated.any()):
        saturated_flux_per_frequency = _saturated_core_flux(
            optical_depth[saturated].to(high_precision),
            thermal_source[saturated].to(high_precision),
            scattering_fraction[saturated].to(high_precision),
        )
        surface_eddington_flux_per_frequency = (
            surface_eddington_flux_per_frequency.clone()
        )
        surface_eddington_flux_per_frequency[saturated] = (
            saturated_flux_per_frequency.to(surface_eddington_flux_per_frequency.dtype)
        )
    return surface_eddington_flux_per_frequency


def solve_spectrum(
    continuum_absorption: torch.Tensor,
    continuum_scattering: torch.Tensor,
    line_mass_absorption_coefficient: torch.Tensor,
    line_scattering: torch.Tensor,
    planck_source: torch.Tensor,
    column_mass: torch.Tensor,
    tables: TransferTables,
    sweeps: int = DEFAULT_SWEEPS,
    assert_no_saturated_core: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total and continuum Eddington flux ``H_nu`` plus their ratio.

    Opacity and source arrays arrive as [depth, wavelength].  The transfer
    solver uses wavelength as the batch axis, then stacks total and continuum
    rows into one call so both fluxes share the same operator launch pattern.
    """
    continuum_absorption_rows = continuum_absorption.transpose(0, 1).contiguous()
    continuum_scattering_rows = continuum_scattering.transpose(0, 1).contiguous()
    line_mass_absorption_coefficient_rows = line_mass_absorption_coefficient.transpose(
        0, 1
    ).contiguous()
    line_scattering_rows = line_scattering.transpose(0, 1).contiguous()
    source_rows = planck_source.transpose(0, 1).contiguous()
    wavelength_count = source_rows.shape[0]

    zero_line_mass_absorption_coefficient = torch.zeros_like(
        line_mass_absorption_coefficient_rows
    )

    stacked_continuum_absorption = torch.cat(
        (continuum_absorption_rows, continuum_absorption_rows),
        dim=0,
    )
    stacked_source = torch.cat((source_rows, source_rows), dim=0)
    stacked_line_mass_absorption_coefficient = torch.cat(
        (line_mass_absorption_coefficient_rows, zero_line_mass_absorption_coefficient),
        dim=0,
    )
    stacked_continuum_scattering = torch.cat(
        (continuum_scattering_rows, continuum_scattering_rows),
        dim=0,
    )
    stacked_line_scattering = torch.cat(
        (line_scattering_rows, zero_line_mass_absorption_coefficient), dim=0
    )
    if assert_no_saturated_core is None:
        # Default to the strict check; the shipped pipeline passes an explicit
        # value (False) at its call sites, so this default only applies to
        # direct callers that leave the flag unset.
        assert_no_saturated_core = True

    surface_eddington_flux_per_frequency = _solve_flux_rows(
        stacked_continuum_absorption,
        stacked_source,
        stacked_line_mass_absorption_coefficient,
        stacked_source,
        stacked_continuum_scattering,
        stacked_line_scattering,
        column_mass,
        tables,
        sweeps=sweeps,
        assert_no_saturated_core=assert_no_saturated_core,
    )

    eddington_flux_total_per_frequency = surface_eddington_flux_per_frequency[
        :wavelength_count
    ]
    eddington_flux_continuum_per_frequency = surface_eddington_flux_per_frequency[
        wavelength_count:
    ]

    normalized_flux = (
        eddington_flux_total_per_frequency / eddington_flux_continuum_per_frequency
    )
    return (
        eddington_flux_total_per_frequency,
        eddington_flux_continuum_per_frequency,
        normalized_flux,
    )
