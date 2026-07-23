"""Doppler-width state used by line absorption."""

from __future__ import annotations

import numpy as np

from .constants import LIGHT_SPEED_CM_PER_S_EXACT as LIGHT_SPEED_CM_PER_S
from .constants import ATOMIC_MASS_GRAM_REFERENCE
from .runtime_state import AtmosphereRuntimeState


def update_doppler_line_strength_factors(
    *,
    thermal_energy_erg: np.ndarray,
    microturbulence: np.ndarray,
    state: AtmosphereRuntimeState,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute fractional widths and partition-normalized line-strength factors."""

    thermal_energy = np.asarray(thermal_energy_erg, dtype=np.float64)
    microturbulence_cm_s = np.asarray(microturbulence, dtype=np.float64)
    mass_density = np.asarray(state.mass_density, dtype=np.float64)
    partition_normalized_ion_stage_populations = np.asarray(
        state.partition_normalized_populations_by_packed_slot, dtype=np.float64
    )
    major_isotope_mass_amu = np.asarray(state.major_isotope_mass_amu, dtype=np.float64)

    layers = thermal_energy.size
    ion_slots = partition_normalized_ion_stage_populations.shape[1]
    if major_isotope_mass_amu.size < ion_slots:
        raise ValueError(
            f"major isotope mass table has {major_isotope_mass_amu.size} slots, "
            f"but runtime state requires {ion_slots}"
        )

    fractional_doppler_widths = np.zeros((layers, ion_slots), dtype=np.float64)
    population_over_density_and_width = np.zeros((layers, ion_slots), dtype=np.float64)
    if ion_slots <= 1:
        state.fractional_doppler_widths = fractional_doppler_widths
        state.partition_normalized_population_over_mass_density_and_fractional_doppler_width = population_over_density_and_width
        return fractional_doppler_widths, population_over_density_and_width

    isotope_mass = major_isotope_mass_amu[: ion_slots - 1]
    thermal_velocity_squared = np.divide(
        2.0 * thermal_energy[:, None],
        isotope_mass[None, :] * ATOMIC_MASS_GRAM_REFERENCE,
        out=np.full((layers, ion_slots - 1), np.inf, dtype=np.float64),
        where=isotope_mass[None, :] > 0.0,
    )
    doppler = (
        np.sqrt(thermal_velocity_squared + microturbulence_cm_s[:, None] ** 2)
        / LIGHT_SPEED_CM_PER_S
    )
    fractional_doppler_widths[:, : ion_slots - 1] = doppler

    density_safe = np.maximum(mass_density[:, None], 1.0e-300)
    population_over_density_and_width[:, : ion_slots - 1] = np.divide(
        partition_normalized_ion_stage_populations[:, : ion_slots - 1],
        doppler * density_safe,
        out=np.zeros((layers, ion_slots - 1), dtype=np.float64),
        where=doppler > 0.0,
    )

    state.fractional_doppler_widths = fractional_doppler_widths
    state.partition_normalized_population_over_mass_density_and_fractional_doppler_width = population_over_density_and_width
    return fractional_doppler_widths, population_over_density_and_width
