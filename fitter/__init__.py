"""Instrument-agnostic normalized-spectrum fitting."""

from .normalized import (
    FitConfiguration,
    FitResult,
    NormalizedSpectrum,
    PhysicalAtmosphereCheck,
    PhysicalAtmosphereConfiguration,
    PhysicalAtmosphereCorrection,
    PhysicalAtmosphereResult,
    ResolvedPhysicalAtmosphereControls,
    fit_normalized_spectrum,
    refine_with_physical_atmosphere,
)
from .instrument import ObservedSpectrumOperator
from .rotational_broadening import RotationalBroadening

__all__ = [
    "FitConfiguration",
    "FitResult",
    "NormalizedSpectrum",
    "ObservedSpectrumOperator",
    "PhysicalAtmosphereCheck",
    "PhysicalAtmosphereConfiguration",
    "PhysicalAtmosphereCorrection",
    "PhysicalAtmosphereResult",
    "ResolvedPhysicalAtmosphereControls",
    "RotationalBroadening",
    "fit_normalized_spectrum",
    "refine_with_physical_atmosphere",
]
