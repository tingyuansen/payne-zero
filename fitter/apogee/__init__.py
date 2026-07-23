"""APOGEE normalized-spectrum fitting with the DR14 combined LSF model.

The public API exposes the forward model, nuisance projection, continuum
profiling, atomic calibration, and optimizer. It does not download survey data
or assume an APOGEE data-release catalog.
"""

from .api import fit_apogee_spectrum
from .forward_model import (
    APOGEE_SYNTHESIS_RESOLUTION,
    APOGEE_SYNTHESIS_R_GRID,
    FastForwardModel,
)
from .lsf import APOGEEDR14LSF
from .optimizer import NormalizedSpectrumInput
from .spectral_nuisance import (
    APOGEEContinuumProfiler,
    APOGEESpectralNuisance,
    estimate_coarse_velocity,
)


def calibrated_window_invariants(*args, **kwargs):
    """Load and apply an atomic calibration to resident synthesis invariants."""

    from .atomic_calibration import calibrated_window_invariants as implementation

    return implementation(*args, **kwargs)


def validate_atomic_calibration(*args, **kwargs):
    """Validate a portable atomic-calibration NPZ without fitting a spectrum."""

    from .atomic_calibration import validate_atomic_calibration as implementation

    return implementation(*args, **kwargs)


__all__ = [
    "APOGEEContinuumProfiler",
    "APOGEEDR14LSF",
    "APOGEESpectralNuisance",
    "APOGEE_SYNTHESIS_R_GRID",
    "APOGEE_SYNTHESIS_RESOLUTION",
    "FastForwardModel",
    "NormalizedSpectrumInput",
    "calibrated_window_invariants",
    "estimate_coarse_velocity",
    "fit_apogee_spectrum",
    "validate_atomic_calibration",
]
