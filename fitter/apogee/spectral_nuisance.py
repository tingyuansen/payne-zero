"""GPU-native kinematic and continuum nuisance model for APOGEE fitting.

Macroscopic broadening and residual radial velocity act on the uniform
log-wavelength synthesis grid before the wavelength-dependent instrument LSF.
The high-resolution total/continuum pair is cached on device so nuisance
derivatives never repeat spectral synthesis. Continuum coefficients are linear
and are profiled exactly at every nonlinear trial.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import torch
from torch.nn import functional as torch_functional

from fitter.apogee.lsf import APOGEEDR14LSF, DEFAULT_ASSET


LIGHT_SPEED_KM_S = 299_792.458


def _chip_slices(apstar_pixel: np.ndarray) -> tuple[slice, ...]:
    """Return contiguous detector segments from retained apStar pixels."""

    pixels = np.asarray(apstar_pixel, np.int64)
    if pixels.ndim != 1 or pixels.size == 0:
        raise ValueError("apStar pixels must be a non-empty one-dimensional array")
    if not np.all(np.diff(pixels) > 0):
        raise ValueError("apStar pixels must be strictly increasing")
    boundaries = np.flatnonzero(np.diff(pixels) > 1) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [pixels.size]))
    return tuple(slice(int(start), int(stop)) for start, stop in zip(starts, stops))


def _legendre_chip_basis(
    apstar_pixel: np.ndarray,
    *,
    order: int,
) -> np.ndarray:
    """Block-diagonal Legendre basis, one polynomial per detector segment."""

    if order < 0:
        raise ValueError("continuum order must be nonnegative")
    pixels = np.asarray(apstar_pixel, np.int64)
    chips = _chip_slices(pixels)
    basis = np.zeros((pixels.size, len(chips) * (order + 1)), np.float64)
    for chip_index, chip in enumerate(chips):
        count = chip.stop - chip.start
        coordinate = np.linspace(-1.0, 1.0, count, dtype=np.float64)
        columns = np.polynomial.legendre.legvander(coordinate, order)
        start = chip_index * (order + 1)
        basis[chip, start : start + order + 1] = columns
    return basis


class APOGEESpectralNuisance:
    """Broadening + residual RV + DR14 LSF with a cached native spectrum."""

    name = "apogee_dr14_r8_combo_mean_with_kinematics"

    def __init__(
        self,
        input_wavelength_nm: np.ndarray,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
        asset_path=DEFAULT_ASSET,
        maximum_vmacro_km_s: float = 30.0,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        wavelength = np.asarray(input_wavelength_nm, np.float64)
        if wavelength.ndim != 1 or wavelength.size < 3:
            raise ValueError("input wavelengths must be a one-dimensional grid")
        log_step = np.diff(np.log(wavelength))
        if not np.allclose(log_step, log_step[0], rtol=2.0e-9, atol=2.0e-13):
            raise ValueError(
                "kinematic projection requires a uniform log-wavelength grid"
            )
        if maximum_vmacro_km_s <= 0.0:
            raise ValueError("maximum_vmacro_km_s must be positive")
        self.input_wavelength_nm = wavelength
        self.log_wavelength_step = float(np.mean(log_step))
        self.velocity_step_km_s = LIGHT_SPEED_KM_S * self.log_wavelength_step
        self.maximum_vmacro_km_s = float(maximum_vmacro_km_s)
        self._pixel_index = torch.arange(
            wavelength.size, dtype=torch.int64, device=self.device
        )
        self.lsf = APOGEEDR14LSF(
            wavelength,
            device=self.device,
            dtype=self.dtype,
            asset_path=asset_path,
        )
        self.output_wavelength_nm = self.lsf.output_wavelength_nm
        self.apstar_pixel = self.lsf.apstar_pixel
        self.residual_rv_km_s = 0.0
        self.vmacro_km_s = 0.0
        self.last_seconds = 0.0
        self.last_kinematic_seconds = 0.0
        self.last_native_pair: torch.Tensor | None = None
        self.benchmark_seconds_per_flux_pair = float("nan")

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def set_parameters(self, *, residual_rv_km_s: float, vmacro_km_s: float) -> None:
        """Set the kinematics used by the next synthesis/operator call."""

        if not math.isfinite(residual_rv_km_s):
            raise ValueError("residual radial velocity must be finite")
        if (
            not math.isfinite(vmacro_km_s)
            or not 0.0 <= vmacro_km_s <= self.maximum_vmacro_km_s
        ):
            raise ValueError(
                f"vmacro must be between 0 and {self.maximum_vmacro_km_s:g} km/s"
            )
        self.residual_rv_km_s = float(residual_rv_km_s)
        self.vmacro_km_s = float(vmacro_km_s)

    def _broaden_pair(self, pair: torch.Tensor, vmacro_km_s: float) -> torch.Tensor:
        if vmacro_km_s <= 1.0e-8:
            return pair
        radius = max(
            1,
            int(math.ceil(5.0 * vmacro_km_s / self.velocity_step_km_s)),
        )
        offset = torch.arange(
            -radius,
            radius + 1,
            dtype=pair.dtype,
            device=pair.device,
        )
        velocity = offset * self.velocity_step_km_s
        kernel = torch.exp(-0.5 * (velocity / vmacro_km_s).square())
        kernel = kernel / torch.sum(kernel)
        padded = torch_functional.pad(
            pair.unsqueeze(1), (radius, radius), mode="reflect"
        )
        return torch_functional.conv1d(padded, kernel.reshape(1, 1, -1))[:, 0]

    def _shift_pair(self, pair: torch.Tensor, residual_rv_km_s: float) -> torch.Tensor:
        if abs(residual_rv_km_s) <= 1.0e-12:
            return pair
        if residual_rv_km_s <= -0.99 * LIGHT_SPEED_KM_S:
            raise ValueError("radial velocity is outside the Doppler domain")
        shift_pixel = math.log1p(residual_rv_km_s / LIGHT_SPEED_KM_S) / (
            self.log_wavelength_step
        )
        left_offset = math.floor(-shift_pixel)
        fraction = float(-shift_pixel - left_offset)
        left = torch.clamp(
            self._pixel_index + left_offset,
            min=0,
            max=pair.shape[-1] - 1,
        )
        right = torch.clamp(left + 1, max=pair.shape[-1] - 1)
        return pair[:, left] + fraction * (pair[:, right] - pair[:, left])

    def project_pair(
        self,
        total_flux: torch.Tensor,
        continuum_flux: torch.Tensor,
        *,
        residual_rv_km_s: float,
        vmacro_km_s: float,
        cache_native: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project one native total/continuum pair to the apStar grid."""

        if total_flux.shape != continuum_flux.shape or total_flux.ndim != 1:
            raise ValueError("total and continuum must be matching 1-D tensors")
        self.set_parameters(
            residual_rv_km_s=residual_rv_km_s,
            vmacro_km_s=vmacro_km_s,
        )
        native_pair = torch.stack((total_flux, continuum_flux), dim=0)
        if cache_native:
            self.last_native_pair = native_pair.detach()
        self._synchronize()
        start = time.perf_counter()
        broadened = self._broaden_pair(native_pair, self.vmacro_km_s)
        shifted = self._shift_pair(broadened, self.residual_rv_km_s)
        projected_total, projected_continuum, normalized = self.lsf.convolve_fluxes(
            shifted[0], shifted[1]
        )
        self.last_seconds = time.perf_counter() - start
        self.last_kinematic_seconds = max(
            0.0, self.last_seconds - self.lsf.last_seconds
        )
        return projected_total, projected_continuum, normalized

    def convolve_fluxes(
        self,
        total_flux: torch.Tensor,
        continuum_flux: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Synthesis spectral-operator interface using the configured kinematics."""

        return self.project_pair(
            total_flux,
            continuum_flux,
            residual_rv_km_s=self.residual_rv_km_s,
            vmacro_km_s=self.vmacro_km_s,
            cache_native=True,
        )

    def project_cached(
        self,
        *,
        residual_rv_km_s: float,
        vmacro_km_s: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-project the last synthesized native pair without re-synthesis."""

        if self.last_native_pair is None:
            raise RuntimeError("no native spectrum has been cached")
        return self.project_pair(
            self.last_native_pair[0],
            self.last_native_pair[1],
            residual_rv_km_s=residual_rv_km_s,
            vmacro_km_s=vmacro_km_s,
            cache_native=False,
        )

    def prepare(self, *, repeats: int = 20) -> float:
        """Warm the complete kinematic + LSF path and report setup wall time."""

        if repeats < 1:
            raise ValueError("repeats must be positive")
        configured_rv = self.residual_rv_km_s
        configured_vmacro = self.vmacro_km_s
        start = time.perf_counter()
        try:
            self.lsf.prepare(repeats=max(1, repeats))
            continuum = torch.ones(
                self.input_wavelength_nm.size, dtype=self.dtype, device=self.device
            )
            total = continuum - 0.05 * torch.exp(
                -0.5
                * (
                    (self._pixel_index.to(self.dtype) - continuum.numel() / 2.0) / 4.0
                ).square()
            )
            for _ in range(2):
                self.project_pair(
                    total,
                    continuum,
                    residual_rv_km_s=1.5,
                    vmacro_km_s=6.0,
                )
            self._synchronize()
            benchmark_start = time.perf_counter()
            for _ in range(repeats):
                self.project_pair(
                    total,
                    continuum,
                    residual_rv_km_s=1.5,
                    vmacro_km_s=6.0,
                )
            self._synchronize()
            self.benchmark_seconds_per_flux_pair = (
                time.perf_counter() - benchmark_start
            ) / repeats
        finally:
            # Benchmarking must never change the physical state selected by
            # the caller, which may configure nuisance values before warming
            # this operator.
            self.set_parameters(
                residual_rv_km_s=configured_rv,
                vmacro_km_s=configured_vmacro,
            )
        return time.perf_counter() - start

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "input_pixels": int(self.input_wavelength_nm.size),
            "output_pixels": int(self.output_wavelength_nm.size),
            "velocity_step_km_s": self.velocity_step_km_s,
            "configured_residual_rv_km_s": self.residual_rv_km_s,
            "configured_vmacro_km_s": self.vmacro_km_s,
            "maximum_vmacro_km_s": self.maximum_vmacro_km_s,
            "broadening_profile": "Gaussian sigma on the log-wavelength grid",
            "benchmark_seconds_per_flux_pair": self.benchmark_seconds_per_flux_pair,
            "lsf": self.lsf.metadata(),
        }


@dataclass(frozen=True)
class ProfiledContinuumResult:
    flux: torch.Tensor
    coefficients: torch.Tensor
    chi_square: torch.Tensor
    objective: torch.Tensor


class APOGEEContinuumProfiler:
    """Exact weighted variable projection for three-chip multiplicative continua."""

    def __init__(
        self,
        *,
        apstar_pixel: np.ndarray,
        observed_flux: np.ndarray | torch.Tensor,
        inverse_variance: np.ndarray | torch.Tensor,
        good_pixel_mask: np.ndarray | torch.Tensor | None = None,
        order: int = 2,
        coefficient_prior_sigma: float | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if coefficient_prior_sigma is not None and coefficient_prior_sigma <= 0.0:
            raise ValueError("coefficient_prior_sigma must be positive")
        self.device = torch.device(device)
        self.dtype = dtype
        pixels = np.asarray(apstar_pixel, np.int64)
        basis = _legendre_chip_basis(pixels, order=order)
        self.apstar_pixel = pixels
        self.order = int(order)
        self.chip_count = len(_chip_slices(pixels))
        self.basis = torch.as_tensor(basis, dtype=dtype, device=self.device)
        self.observed = torch.as_tensor(observed_flux, dtype=dtype, device=self.device)
        self.inverse_variance = torch.as_tensor(
            inverse_variance, dtype=dtype, device=self.device
        )
        if self.observed.shape != (pixels.size,) or self.inverse_variance.shape != (
            pixels.size,
        ):
            raise ValueError(
                "observed flux and inverse variance must match apStar pixels"
            )
        if good_pixel_mask is None:
            mask = torch.ones(pixels.size, dtype=torch.bool, device=self.device)
        else:
            mask = torch.as_tensor(
                good_pixel_mask, dtype=torch.bool, device=self.device
            )
        finite = torch.isfinite(self.observed) & torch.isfinite(self.inverse_variance)
        self.good_pixel_mask = mask & finite & (self.inverse_variance > 0.0)
        if int(torch.count_nonzero(self.good_pixel_mask)) <= self.basis.shape[1]:
            raise ValueError("too few good pixels to profile the continuum")
        self.weight = torch.where(
            self.good_pixel_mask,
            self.inverse_variance,
            torch.zeros_like(self.inverse_variance),
        )
        self.regularization = torch.zeros(
            (self.basis.shape[1], self.basis.shape[1]),
            dtype=dtype,
            device=self.device,
        )
        if coefficient_prior_sigma is not None:
            self.regularization.diagonal().fill_(coefficient_prior_sigma**-2)
        self.coefficient_prior_sigma = coefficient_prior_sigma
        self.good_pixel_count = int(torch.count_nonzero(self.good_pixel_mask))

    def profile(self, model_flux: torch.Tensor) -> ProfiledContinuumResult:
        """Return the profiled flux and likelihood for one nonlinear model."""

        model = torch.as_tensor(model_flux, dtype=self.dtype, device=self.device)
        if model.shape != self.observed.shape:
            raise ValueError("model flux must match the observed apStar grid")
        design = model[:, None] * self.basis
        weighted_design = design * self.weight[:, None]
        normal = design.T @ weighted_design + self.regularization
        rhs = design.T @ (self.weight * (self.observed - model))
        coefficients = torch.linalg.solve(normal, rhs)
        fitted = model * (1.0 + self.basis @ coefficients)
        residual = fitted - self.observed
        chi_square = torch.sum(self.weight * residual.square())
        objective = 0.5 * chi_square / self.good_pixel_count
        return ProfiledContinuumResult(
            flux=fitted,
            coefficients=coefficients,
            chi_square=chi_square,
            objective=objective,
        )


@dataclass(frozen=True)
class CoarseVelocityResult:
    radial_velocity_km_s: float
    integer_shift_pixels: int
    subpixel_shift: float
    peak_correlation: float
    velocity_step_km_s: float


def estimate_coarse_velocity(
    observed_flux: np.ndarray | torch.Tensor,
    template_flux: np.ndarray | torch.Tensor,
    *,
    wavelength_nm: np.ndarray,
    apstar_pixel: np.ndarray,
    inverse_variance: np.ndarray | torch.Tensor | None = None,
    good_pixel_mask: np.ndarray | torch.Tensor | None = None,
    maximum_velocity_km_s: float = 500.0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> CoarseVelocityResult:
    """Vectorized integer-pixel CCF with a parabolic subpixel peak."""

    if maximum_velocity_km_s <= 0.0:
        raise ValueError("maximum_velocity_km_s must be positive")
    runtime_device = torch.device(device)
    wavelength = np.asarray(wavelength_nm, np.float64)
    pixels = np.asarray(apstar_pixel, np.int64)
    if wavelength.shape != pixels.shape:
        raise ValueError("wavelengths and apStar pixels must have matching shapes")
    velocity_step = LIGHT_SPEED_KM_S * float(np.mean(np.diff(np.log(wavelength))))
    maximum_shift = int(math.ceil(maximum_velocity_km_s / velocity_step))
    observed = torch.as_tensor(observed_flux, dtype=dtype, device=runtime_device)
    template = torch.as_tensor(template_flux, dtype=dtype, device=runtime_device)
    if observed.shape != wavelength.shape or template.shape != wavelength.shape:
        raise ValueError("observed and template flux must match the wavelength grid")
    if inverse_variance is None:
        weight = torch.ones_like(observed)
    else:
        weight = torch.as_tensor(inverse_variance, dtype=dtype, device=runtime_device)
    if good_pixel_mask is None:
        mask = torch.ones_like(observed, dtype=torch.bool)
    else:
        mask = torch.as_tensor(good_pixel_mask, dtype=torch.bool, device=runtime_device)
    mask = mask & torch.isfinite(observed) & torch.isfinite(template) & (weight > 0.0)
    weight = torch.where(mask, weight, torch.zeros_like(weight))

    observed_feature = 1.0 - observed
    template_feature = 1.0 - template
    chip_id = np.empty(pixels.size, np.int64)
    for index, chip in enumerate(_chip_slices(pixels)):
        chip_id[chip] = index
        chip_tensor = torch.arange(chip.start, chip.stop, device=runtime_device)
        chip_weight = weight[chip_tensor]
        normalization = torch.clamp(torch.sum(chip_weight), min=torch.finfo(dtype).tiny)
        observed_feature[chip_tensor] -= (
            torch.sum(chip_weight * observed_feature[chip_tensor]) / normalization
        )
        template_feature[chip_tensor] -= (
            torch.sum(chip_weight * template_feature[chip_tensor]) / normalization
        )

    shifts = torch.arange(
        -maximum_shift,
        maximum_shift + 1,
        dtype=torch.int64,
        device=runtime_device,
    )
    base = torch.arange(pixels.size, dtype=torch.int64, device=runtime_device)
    source = base.unsqueeze(0) - shifts.unsqueeze(1)
    valid_bounds = (source >= 0) & (source < pixels.size)
    clipped = torch.clamp(source, 0, pixels.size - 1)
    chip_tensor = torch.as_tensor(chip_id, dtype=torch.int64, device=runtime_device)
    valid = valid_bounds & (chip_tensor[clipped] == chip_tensor[base].unsqueeze(0))
    shifted_template = template_feature[clipped]
    shifted_weight = weight.unsqueeze(0) * valid
    numerator = torch.sum(
        shifted_weight * observed_feature.unsqueeze(0) * shifted_template,
        dim=1,
    )
    observed_norm = torch.sum(
        shifted_weight * observed_feature.unsqueeze(0).square(), dim=1
    )
    template_norm = torch.sum(shifted_weight * shifted_template.square(), dim=1)
    correlation = numerator / torch.sqrt(
        torch.clamp(observed_norm * template_norm, min=torch.finfo(dtype).tiny)
    )
    peak = int(torch.argmax(correlation))
    integer_shift = int(shifts[peak])
    subpixel = 0.0
    if 0 < peak < correlation.numel() - 1:
        left, center, right = (
            float(correlation[peak - 1]),
            float(correlation[peak]),
            float(correlation[peak + 1]),
        )
        denominator = left - 2.0 * center + right
        if denominator < 0.0:
            subpixel = float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))
    return CoarseVelocityResult(
        radial_velocity_km_s=(integer_shift + subpixel) * velocity_step,
        integer_shift_pixels=integer_shift,
        subpixel_shift=subpixel,
        peak_correlation=float(correlation[peak]),
        velocity_step_km_s=velocity_step,
    )
