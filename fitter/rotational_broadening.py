"""Device-resident rotational broadening for logarithmic wavelength grids."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as torch_functional


LIGHT_SPEED_KM_S = 299_792.458


class RotationalBroadening:
    r"""Apply the standard linear-limb-darkened rotational profile.

    The wavelength grid must be uniform in :math:`\log\lambda`, so every pixel
    has the same velocity width. Geometry is built once on ``device`` and the
    last discrete kernel is cached. Repeated calls therefore remain entirely
    in Torch on CPU, CUDA, or Apple Metal.

    ``vsini_km_s`` is a scalar configuration value rather than a trainable
    tensor. The operation is differentiable with respect to ``flux``; fit
    projected rotation with finite differences or another scalar optimizer.
    """

    def __init__(
        self,
        wavelength_nm: np.ndarray,
        *,
        maximum_vsini_km_s: float,
        limb_darkening: float = 0.6,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        wavelength = np.asarray(wavelength_nm, dtype=np.float64)
        if wavelength.ndim != 1 or wavelength.size < 3:
            raise ValueError("wavelength_nm must be a one-dimensional grid")
        if not np.all(np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
            raise ValueError("wavelength_nm must be finite and positive")
        log_step = np.diff(np.log(wavelength))
        if not np.all(log_step > 0.0):
            raise ValueError("wavelength_nm must be strictly increasing")
        if not np.allclose(log_step, log_step[0], rtol=2.0e-9, atol=2.0e-13):
            raise ValueError(
                "rotational broadening requires a uniform log-wavelength grid"
            )
        if not math.isfinite(maximum_vsini_km_s) or maximum_vsini_km_s <= 0.0:
            raise ValueError("maximum_vsini_km_s must be finite and positive")
        if not math.isfinite(limb_darkening) or not 0.0 <= limb_darkening <= 1.0:
            raise ValueError("limb_darkening must be between zero and one")
        if not dtype.is_floating_point:
            raise TypeError("dtype must be a floating-point Torch dtype")

        self.wavelength_nm = wavelength
        self.device = torch.device(device)
        self.dtype = dtype
        self.log_wavelength_step = float(np.mean(log_step))
        self.velocity_step_km_s = LIGHT_SPEED_KM_S * self.log_wavelength_step
        self.maximum_vsini_km_s = float(maximum_vsini_km_s)
        self.limb_darkening = float(limb_darkening)
        self.maximum_radius = max(
            1,
            int(math.ceil(self.maximum_vsini_km_s / self.velocity_step_km_s)),
        )
        if self.maximum_radius >= wavelength.size:
            raise ValueError(
                "maximum_vsini_km_s is too large for the supplied wavelength grid"
            )
        self._full_offsets = torch.arange(
            -self.maximum_radius,
            self.maximum_radius + 1,
            dtype=self.dtype,
            device=self.device,
        )
        # Torch may canonicalize an accelerator request (for example ``mps``)
        # to an indexed device (``mps:0``). Use the resident tensor's device
        # for subsequent strict no-copy input checks.
        self.device = self._full_offsets.device
        self._cached_vsini_km_s: float | None = None
        self._cached_kernel: torch.Tensor | None = None

    def kernel(self, vsini_km_s: float) -> torch.Tensor:
        r"""Return the normalized discrete Gray rotation kernel on device.

        For projected velocity :math:`v_e = v\sin i` and
        :math:`x=v/v_e`, the unnormalized profile is

        .. math::

           2(1-\epsilon)\sqrt{1-x^2}
           + \frac{\pi\epsilon}{2}(1-x^2), \quad |x|\leq 1.

        Discrete normalization preserves a flat spectrum exactly.
        """

        value = float(vsini_km_s)
        if not math.isfinite(value) or not 0.0 < value <= self.maximum_vsini_km_s:
            raise ValueError(
                "vsini_km_s must be finite, positive, and no larger than "
                "maximum_vsini_km_s"
            )
        if value == self._cached_vsini_km_s and self._cached_kernel is not None:
            return self._cached_kernel

        radius = max(1, int(math.ceil(value / self.velocity_step_km_s)))
        center = self.maximum_radius
        offsets = self._full_offsets[center - radius : center + radius + 1]
        velocity = offsets * self.velocity_step_km_s
        x_squared = (velocity / value).square()
        inside = x_squared <= 1.0
        one_minus_x_squared = torch.clamp(1.0 - x_squared, min=0.0)
        epsilon = self.limb_darkening
        profile = (
            2.0 * (1.0 - epsilon) * torch.sqrt(one_minus_x_squared)
            + 0.5 * math.pi * epsilon * one_minus_x_squared
        )
        profile = torch.where(inside, profile, torch.zeros_like(profile))
        kernel = profile / torch.sum(profile)
        self._cached_vsini_km_s = value
        self._cached_kernel = kernel
        return kernel

    def __call__(self, flux: torch.Tensor, *, vsini_km_s: float) -> torch.Tensor:
        """Broaden one spectrum or a batch whose final axis is wavelength."""

        if not isinstance(flux, torch.Tensor):
            raise TypeError("flux must be a Torch tensor")
        if flux.device != self.device:
            raise ValueError(f"flux must be on {self.device}")
        if flux.dtype != self.dtype:
            raise ValueError(f"flux must have dtype {self.dtype}")
        if flux.ndim < 1 or flux.shape[-1] != self.wavelength_nm.size:
            raise ValueError("the final flux axis must match wavelength_nm")
        if not math.isfinite(vsini_km_s) or vsini_km_s < 0.0:
            raise ValueError("vsini_km_s must be finite and nonnegative")
        if vsini_km_s == 0.0:
            return flux

        kernel = self.kernel(vsini_km_s)
        radius = kernel.numel() // 2
        batch_shape = flux.shape[:-1]
        flattened = flux.reshape(-1, 1, flux.shape[-1])
        padded = torch_functional.pad(flattened, (radius, radius), mode="reflect")
        broadened = torch_functional.conv1d(padded, kernel.reshape(1, 1, -1))
        return broadened.reshape(*batch_shape, flux.shape[-1])

    def metadata(self) -> dict[str, object]:
        """Describe the fixed geometry and physical profile."""

        return {
            "profile": "Gray linear-limb-darkened rotation",
            "limb_darkening": self.limb_darkening,
            "velocity_step_km_s": self.velocity_step_km_s,
            "maximum_vsini_km_s": self.maximum_vsini_km_s,
            "device": str(self.device),
            "dtype": str(self.dtype),
        }
