"""Rosseland-mean opacity accumulation for the atmosphere solver."""

from __future__ import annotations

import numpy as np

from .radiative_transfer import (
    integrate_on_depth_grid,
)


_STEFAN_BOLTZMANN_OVER_PI = 5.6697e-5 / 3.14159


def rosseland_mean_step(
    rosseland_accumulator: np.ndarray,
    *,
    mode: int,
    frequency_weight: float,
    planck_source: np.ndarray,
    frequency_hz: float,
    h_over_kt: np.ndarray,
    temperature_k: np.ndarray,
    stimulated_emission: np.ndarray,
    total_opacity: np.ndarray,
    frequency_count: int,
    column_mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one Rosseland-mean mode and return opacity plus optical depth."""

    accumulator = np.asarray(rosseland_accumulator, dtype=np.float64)
    temperature = np.asarray(temperature_k, dtype=np.float64)
    optical_depth = np.zeros_like(accumulator)

    if int(mode) == 1:
        accumulator[:] = 0.0
        return accumulator, optical_depth

    if int(mode) == 2:
        planck = np.asarray(planck_source, dtype=np.float64)
        h_over_kt = np.asarray(h_over_kt, dtype=np.float64)
        stimulated = np.asarray(stimulated_emission, dtype=np.float64)
        opacity = np.asarray(total_opacity, dtype=np.float64)
        source_derivative = (
            planck
            * float(frequency_hz)
            * h_over_kt
            / np.maximum(
                temperature * stimulated,
                1.0e-300,
            )
        )
        if int(frequency_count) == 1:
            source_derivative = 4.0 * _STEFAN_BOLTZMANN_OVER_PI * temperature**3
        accumulator += (
            source_derivative / np.maximum(opacity, 1.0e-300) * float(frequency_weight)
        )
        return accumulator, optical_depth

    if int(mode) != 3:
        raise ValueError(f"Unsupported Rosseland mode: {mode}")

    accumulator[:] = (
        4.0
        * _STEFAN_BOLTZMANN_OVER_PI
        * temperature**3
        / np.maximum(
            accumulator,
            1.0e-300,
        )
    )
    optical_depth = integrate_on_depth_grid(
        np.asarray(column_mass, dtype=np.float64),
        accumulator,
        surface_value=float(
            accumulator[0] * np.asarray(column_mass, dtype=np.float64)[0]
        ),
    )
    return accumulator, optical_depth
