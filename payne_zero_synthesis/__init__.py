"""Payne Zero spectrum-synthesis public API."""

__version__ = "1.3"

from .api import (
    Spectrum,
    build_structured_atmosphere,
    clear_window_invariant_cache,
    save_structured_atmosphere,
    synthesize,
    window_invariant_cache_enabled,
)
from .atmosphere import (
    REQUIRED_ATMOSPHERE_ARRAYS,
    load_atmosphere_npz,
    validate_atmosphere_npz,
)

__all__ = [
    "REQUIRED_ATMOSPHERE_ARRAYS",
    "Spectrum",
    "__version__",
    "build_structured_atmosphere",
    "clear_window_invariant_cache",
    "load_atmosphere_npz",
    "save_structured_atmosphere",
    "synthesize",
    "validate_atmosphere_npz",
    "window_invariant_cache_enabled",
]
