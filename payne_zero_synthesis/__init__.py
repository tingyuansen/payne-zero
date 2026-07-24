"""Payne Zero spectrum-synthesis public API."""

__version__ = "1.3"

from .api import (
    ForwardTimings,
    InitializedAtmosphere,
    LabelSpectrum,
    Spectrum,
    build_structured_atmosphere,
    clear_window_invariant_cache,
    initialize_atmosphere_from_labels,
    save_structured_atmosphere,
    synthesize,
    synthesize_from_labels,
    window_invariant_cache_enabled,
)
from .atmosphere import (
    REQUIRED_ATMOSPHERE_ARRAYS,
    load_atmosphere_npz,
    load_atmosphere_product_metadata,
    validate_atmosphere_npz,
)

__all__ = [
    "REQUIRED_ATMOSPHERE_ARRAYS",
    "ForwardTimings",
    "InitializedAtmosphere",
    "LabelSpectrum",
    "Spectrum",
    "__version__",
    "build_structured_atmosphere",
    "clear_window_invariant_cache",
    "initialize_atmosphere_from_labels",
    "load_atmosphere_npz",
    "load_atmosphere_product_metadata",
    "save_structured_atmosphere",
    "synthesize",
    "synthesize_from_labels",
    "validate_atmosphere_npz",
    "window_invariant_cache_enabled",
]
