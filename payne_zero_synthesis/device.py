"""Device selection and dtype policy for the synthesis kernels.

The public API prefers CUDA, then Apple Metal, then CPU.  Its omitted-dtype
policy uses fp32 on Metal, whose backend does not support fp64, and fp64 on
CUDA or CPU.  Callers can still request fp32 explicitly on any device.
"""

from __future__ import annotations

import torch


def _default_device() -> torch.device:
    """Prefer discrete CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _default_device()

# Opacity accumulation stays fp32 on every backend. DEFAULT_DTYPE is the
# Metal-compatible work precision; REFERENCE_DTYPE is the CUDA/CPU default.
ACCUMULATION_DTYPE = torch.float32
DEFAULT_DTYPE = torch.float32
REFERENCE_DTYPE = torch.float64


def device() -> torch.device:
    """Return the default compute device for resident synthesis data."""
    return DEVICE


def resolve_runtime(
    requested_device: torch.device | str | None = None,
    requested_dtype: torch.dtype | None = None,
) -> tuple[torch.device, torch.dtype]:
    """Resolve the public device and dtype defaults as one compatible pair."""

    runtime_device = (
        torch.device(requested_device)
        if requested_device is not None
        else device()
    )
    runtime_dtype = (
        requested_dtype
        if requested_dtype is not None
        else (DEFAULT_DTYPE if runtime_device.type == "mps" else REFERENCE_DTYPE)
    )
    if runtime_device.type == "mps" and runtime_dtype == torch.float64:
        raise ValueError(
            "Apple Metal (MPS) does not support float64; omit dtype or use float32"
        )
    return runtime_device, runtime_dtype


def to_dev(
    value, dtype=DEFAULT_DTYPE, device: torch.device | None = None
) -> torch.Tensor:
    """Convert an array-like object to a tensor on the selected compute device."""
    return torch.as_tensor(value, dtype=dtype).to(
        device if device is not None else DEVICE
    )
