"""Radiative acceleration and pressure accumulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .radiative_transfer import integrate_on_depth_grid


@dataclass
class RadiativePressureState:
    """Frequency-integrated RADIAP accumulators."""

    integrated_eddington_flux: np.ndarray
    radiation_energy_density: np.ndarray
    radiative_acceleration: np.ndarray
    integrated_radiation_pressure: np.ndarray
    absolute_radiation_pressure: np.ndarray
    surface_radiation_pressure_constant: float


def initialize_radiative_pressure_state(layer_count: int) -> RadiativePressureState:
    """Return a zeroed RADIAP accumulator state."""

    zeros = np.zeros(int(layer_count), dtype=np.float64)
    return RadiativePressureState(
        integrated_eddington_flux=zeros.copy(),
        radiation_energy_density=zeros.copy(),
        radiative_acceleration=zeros.copy(),
        integrated_radiation_pressure=zeros.copy(),
        absolute_radiation_pressure=zeros.copy(),
        surface_radiation_pressure_constant=0.0,
    )


def accumulate_radiative_pressure(
    state: RadiativePressureState,
    *,
    mode: int,
    frequency_weight: float,
    total_opacity: np.ndarray,
    monochromatic_eddington_flux: np.ndarray,
    mean_intensity: np.ndarray,
    surface_second_moment: float,
    target_integrated_eddington_flux: float,
    column_mass: np.ndarray,
) -> None:
    """Apply one radiative-pressure accumulation step in place."""

    if int(mode) == 1:
        state.integrated_eddington_flux[:] = 0.0
        state.radiation_energy_density[:] = 0.0
        state.radiative_acceleration[:] = 0.0
        state.surface_radiation_pressure_constant = 0.0
        return

    if int(mode) == 2:
        weight = float(frequency_weight)
        hnu = np.asarray(monochromatic_eddington_flux, dtype=np.float64)
        jnu = np.asarray(mean_intensity, dtype=np.float64)
        opacity = np.asarray(total_opacity, dtype=np.float64)
        state.radiation_energy_density += jnu * weight
        state.integrated_eddington_flux += hnu * weight
        state.radiative_acceleration += opacity * hnu * weight
        state.surface_radiation_pressure_constant += (
            float(surface_second_moment) * weight
        )
        return

    if int(mode) != 3:
        raise ValueError(f"Unsupported radiative-pressure mode: {mode}")

    conversion = 12.5664 / 2.99792458e10
    state.radiation_energy_density *= conversion
    state.radiative_acceleration *= conversion
    flux_ratio = state.integrated_eddington_flux / max(
        float(target_integrated_eddington_flux), 1.0e-300
    )
    too_bright = flux_ratio > 1.0
    state.radiative_acceleration[too_bright] *= float(
        target_integrated_eddington_flux
    ) / np.maximum(
        state.integrated_eddington_flux[too_bright],
        1.0e-300,
    )
    maximum_flux_ratio = float(np.max(flux_ratio)) if flux_ratio.size else 0.0
    state.surface_radiation_pressure_constant *= conversion
    if maximum_flux_ratio > 1.0:
        state.surface_radiation_pressure_constant /= maximum_flux_ratio

    column_mass = np.asarray(column_mass, dtype=np.float64)
    state.integrated_radiation_pressure[:] = integrate_on_depth_grid(
        column_mass,
        state.radiative_acceleration,
        surface_value=float(state.radiative_acceleration[0] * column_mass[0]),
    )
    state.absolute_radiation_pressure[:] = (
        state.integrated_radiation_pressure + state.surface_radiation_pressure_constant
    )
