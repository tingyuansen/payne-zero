"""Survey-agnostic projection from a native spectrum to observed pixels."""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch.nn import functional as torch_functional


LIGHT_SPEED_KM_S = 299_792.458


class ObservedSpectrumOperator:
    """Apply kinematics, a simple LSF, and resampling on one Torch device.

    The input grid must be uniform in logarithmic wavelength, as produced by
    Payne Zero synthesis. The output grid may be any strictly increasing set
    of wavelengths within the input interval. An instrument can be described
    by either a constant resolving power or ``lsf_kernel``, a one-dimensional,
    odd-length convolution kernel sampled in native log-wavelength pixels. The
    middle entry is the zero-offset sample. Entries must be finite and
    nonnegative; the operator normalizes their sum. This single kernel is
    shift-invariant and is applied before resampling. For wavelength-dependent
    or detector-specific line-spread functions, callers may instead supply
    their own object implementing the same ``output_wavelength_nm`` and
    ``convolve_fluxes`` interface.

    The operator is differentiable with respect to the input flux. Radial
    velocity and Gaussian broadening are scalar configuration values intended
    for finite-difference or other bounded optimization.
    """

    name = "generic_observed_spectrum"

    def __init__(
        self,
        input_wavelength_nm: np.ndarray,
        output_wavelength_nm: np.ndarray,
        *,
        resolving_power: float | None = None,
        lsf_kernel: np.ndarray | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        maximum_broadening_sigma_km_s: float = 100.0,
    ) -> None:
        input_wavelength = np.asarray(input_wavelength_nm, np.float64)
        output_wavelength = np.asarray(output_wavelength_nm, np.float64)
        if input_wavelength.ndim != 1 or input_wavelength.size < 3:
            raise ValueError("input_wavelength_nm must be a one-dimensional grid")
        if output_wavelength.ndim != 1 or output_wavelength.size < 1:
            raise ValueError("output_wavelength_nm must be one-dimensional")
        if not np.all(np.isfinite(input_wavelength)) or np.any(
            input_wavelength <= 0.0
        ):
            raise ValueError("input wavelengths must be finite and positive")
        if not np.all(np.isfinite(output_wavelength)) or np.any(
            output_wavelength <= 0.0
        ):
            raise ValueError("output wavelengths must be finite and positive")
        input_log_step = np.diff(np.log(input_wavelength))
        if not np.all(input_log_step > 0.0) or not np.allclose(
            input_log_step,
            input_log_step[0],
            rtol=2.0e-9,
            atol=2.0e-13,
        ):
            raise ValueError(
                "input_wavelength_nm must be strictly increasing and uniform "
                "in logarithmic wavelength"
            )
        if output_wavelength.size > 1 and not np.all(
            np.diff(output_wavelength) > 0.0
        ):
            raise ValueError("output_wavelength_nm must be strictly increasing")
        if (
            output_wavelength[0] < input_wavelength[0]
            or output_wavelength[-1] > input_wavelength[-1]
        ):
            raise ValueError("output wavelengths must lie within the input grid")
        if resolving_power is not None and lsf_kernel is not None:
            raise ValueError("choose resolving_power or lsf_kernel, not both")
        if resolving_power is not None and (
            not math.isfinite(resolving_power) or resolving_power <= 0.0
        ):
            raise ValueError("resolving_power must be finite and positive")
        if (
            not math.isfinite(maximum_broadening_sigma_km_s)
            or maximum_broadening_sigma_km_s < 0.0
        ):
            raise ValueError(
                "maximum_broadening_sigma_km_s must be finite and nonnegative"
            )
        if not dtype.is_floating_point:
            raise TypeError("dtype must be a floating-point Torch dtype")

        self.device = torch.device(device)
        self.dtype = dtype
        self.input_wavelength_nm = input_wavelength
        self.output_wavelength_nm = output_wavelength
        self.log_wavelength_step = float(np.mean(input_log_step))
        self.velocity_step_km_s = (
            LIGHT_SPEED_KM_S * self.log_wavelength_step
        )
        self.resolving_power = (
            None if resolving_power is None else float(resolving_power)
        )
        self.maximum_broadening_sigma_km_s = float(
            maximum_broadening_sigma_km_s
        )
        self.radial_velocity_km_s = 0.0
        self.broadening_sigma_km_s = 0.0
        self.last_seconds = 0.0

        self._input_wavelength = torch.as_tensor(
            input_wavelength, device=self.device, dtype=self.dtype
        )
        self.device = self._input_wavelength.device
        self._pixel_index = torch.arange(
            input_wavelength.size, device=self.device, dtype=torch.int64
        )

        output_tensor = torch.as_tensor(
            output_wavelength, device=self.device, dtype=self.dtype
        )
        right = torch.searchsorted(self._input_wavelength, output_tensor)
        right = torch.clamp(right, min=1, max=input_wavelength.size - 1)
        left = right - 1
        denominator = self._input_wavelength[right] - self._input_wavelength[left]
        self._sample_left = left
        self._sample_right = right
        self._sample_fraction = (
            output_tensor - self._input_wavelength[left]
        ) / denominator

        self._lsf_kernel: torch.Tensor | None = None
        if lsf_kernel is not None:
            kernel = np.asarray(lsf_kernel, np.float64)
            if (
                kernel.ndim != 1
                or kernel.size < 1
                or kernel.size % 2 != 1
                or not np.all(np.isfinite(kernel))
                or np.any(kernel < 0.0)
                or float(np.sum(kernel)) <= 0.0
            ):
                raise ValueError(
                    "lsf_kernel must be a one-dimensional, finite, nonnegative, "
                    "odd-length vector with positive sum"
                )
            self._lsf_kernel = torch.as_tensor(
                kernel / np.sum(kernel), device=self.device, dtype=self.dtype
            )

    def set_parameters(
        self,
        *,
        radial_velocity_km_s: float = 0.0,
        broadening_sigma_km_s: float = 0.0,
    ) -> None:
        """Set residual velocity and Gaussian broadening for later calls."""

        if (
            not math.isfinite(radial_velocity_km_s)
            or radial_velocity_km_s <= -0.99 * LIGHT_SPEED_KM_S
        ):
            raise ValueError("radial_velocity_km_s is outside the Doppler domain")
        if (
            not math.isfinite(broadening_sigma_km_s)
            or not 0.0
            <= broadening_sigma_km_s
            <= self.maximum_broadening_sigma_km_s
        ):
            raise ValueError(
                "broadening_sigma_km_s must lie between zero and "
                "maximum_broadening_sigma_km_s"
            )
        self.radial_velocity_km_s = float(radial_velocity_km_s)
        self.broadening_sigma_km_s = float(broadening_sigma_km_s)

    def _shift(self, pair: torch.Tensor) -> torch.Tensor:
        if abs(self.radial_velocity_km_s) <= 1.0e-12:
            return pair
        shift_pixel = math.log1p(
            self.radial_velocity_km_s / LIGHT_SPEED_KM_S
        ) / self.log_wavelength_step
        left_offset = math.floor(-shift_pixel)
        fraction = float(-shift_pixel - left_offset)
        left = torch.clamp(
            self._pixel_index + left_offset,
            min=0,
            max=pair.shape[-1] - 1,
        )
        right = torch.clamp(left + 1, max=pair.shape[-1] - 1)
        return pair[:, left] + fraction * (pair[:, right] - pair[:, left])

    def _gaussian_kernel(self) -> torch.Tensor | None:
        instrument_sigma = (
            0.0
            if self.resolving_power is None
            else LIGHT_SPEED_KM_S
            / (2.0 * math.sqrt(2.0 * math.log(2.0)) * self.resolving_power)
        )
        sigma_km_s = math.hypot(
            instrument_sigma, self.broadening_sigma_km_s
        )
        if sigma_km_s <= 1.0e-12:
            return None
        sigma_pixel = sigma_km_s / self.velocity_step_km_s
        radius = max(1, int(math.ceil(4.0 * sigma_pixel)))
        if radius >= self.input_wavelength_nm.size:
            raise ValueError("broadening kernel is wider than the input spectrum")
        offset = torch.arange(
            -radius, radius + 1, device=self.device, dtype=self.dtype
        )
        kernel = torch.exp(-0.5 * (offset / sigma_pixel).square())
        return kernel / torch.sum(kernel)

    @staticmethod
    def _apply_kernel(pair: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        radius = kernel.numel() // 2
        padded = torch_functional.pad(
            pair[:, None, :], (radius, radius), mode="reflect"
        )
        return torch_functional.conv1d(
            padded, kernel.reshape(1, 1, -1)
        )[:, 0, :]

    def _sample(self, pair: torch.Tensor) -> torch.Tensor:
        fraction = self._sample_fraction[None, :]
        return (
            pair[:, self._sample_left]
            + fraction
            * (pair[:, self._sample_right] - pair[:, self._sample_left])
        )

    def convolve_fluxes(
        self,
        total_flux: torch.Tensor,
        continuum_flux: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project native-grid total and continuum flux to the output pixels.

        Both inputs are one-dimensional tensors matching
        ``input_wavelength_nm`` and must use this operator's device and dtype.
        The three returned tensors follow ``output_wavelength_nm``.
        """

        if (
            not isinstance(total_flux, torch.Tensor)
            or not isinstance(continuum_flux, torch.Tensor)
            or total_flux.ndim != 1
            or continuum_flux.shape != total_flux.shape
            or total_flux.shape[0] != self.input_wavelength_nm.size
        ):
            raise ValueError(
                "total_flux and continuum_flux must match the input wavelength grid"
            )
        if (
            total_flux.device != self.device
            or continuum_flux.device != self.device
            or total_flux.dtype != self.dtype
            or continuum_flux.dtype != self.dtype
        ):
            raise ValueError("flux tensors must use the operator device and dtype")

        started = time.perf_counter()
        pair = self._shift(torch.stack((total_flux, continuum_flux), dim=0))
        gaussian_kernel = self._gaussian_kernel()
        if gaussian_kernel is not None:
            pair = self._apply_kernel(pair, gaussian_kernel)
        if self._lsf_kernel is not None:
            pair = self._apply_kernel(pair, self._lsf_kernel)
        projected = self._sample(pair)
        normalized = projected[0] / torch.clamp(projected[1], min=1.0e-30)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()
        self.last_seconds = time.perf_counter() - started
        return projected[0], projected[1], normalized

    def metadata(self) -> dict[str, object]:
        """Describe the fixed wavelength geometry and current nuisance values."""

        return {
            "name": self.name,
            "input_pixels": int(self.input_wavelength_nm.size),
            "output_pixels": int(self.output_wavelength_nm.size),
            "velocity_step_km_s": self.velocity_step_km_s,
            "resolving_power": self.resolving_power,
            "custom_lsf_kernel_pixels": (
                None if self._lsf_kernel is None else int(self._lsf_kernel.numel())
            ),
            "radial_velocity_km_s": self.radial_velocity_km_s,
            "broadening_sigma_km_s": self.broadening_sigma_km_s,
            "last_seconds": self.last_seconds,
            "device": str(self.device),
            "dtype": str(self.dtype),
        }
