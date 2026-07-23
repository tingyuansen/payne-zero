"""Hydrostatic pressure updates for the atmosphere runner."""

from __future__ import annotations

import warnings

import numpy as np

from .atmosphere_io import ModelAtmosphere


def integrate_hydrostatic_pressure(
    atmosphere: ModelAtmosphere,
    *,
    surface_gravity_cgs: float,
    integrated_radiation_pressure: np.ndarray,
    turbulent_pressure: np.ndarray,
    pressure_constant: float = 0.0,
) -> np.ndarray:
    """Return gas pressure from the hydrostatic-balance update."""

    column_mass = np.asarray(atmosphere.column_mass, dtype=np.float64)
    radiation = np.asarray(integrated_radiation_pressure, dtype=np.float64)
    turbulent = np.asarray(turbulent_pressure, dtype=np.float64)
    pressure = (
        float(surface_gravity_cgs) * column_mass
        - radiation
        - turbulent
        - float(pressure_constant)
    )
    if np.any(pressure <= 0.0):
        floor = np.maximum(1.0e-6 * float(surface_gravity_cgs) * column_mass, 1.0e-30)
        bad = pressure <= 0.0
        worst = int(np.argmin(pressure))
        warnings.warn(
            "Hydrostatic P non-positive at "
            f"{int(bad.sum())} layer(s) (worst layer {worst}: "
            f"P={pressure[worst]:.3e}); flooring positive.",
            RuntimeWarning,
        )
        pressure = np.where(bad, floor, pressure)
    return pressure


def update_total_pressure(
    column_mass: np.ndarray,
    *,
    surface_gravity_cgs: float,
    pressure_zero_point: float,
) -> np.ndarray:
    """Return total pressure from gravity, column mass, and surface pressure."""

    return float(surface_gravity_cgs) * np.asarray(
        column_mass, dtype=np.float64
    ) + float(pressure_zero_point)
