"""Atomic specific internal energy used by convection finite differences."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .constants import (
    BOLTZMANN_ERG_PER_K_REFERENCE,
    LIGHT_SPEED_CM_PER_S_EXACT as LIGHT_SPEED_CM_PER_S,
    PLANCK_ERG_SECOND_REFERENCE,
)
from .equation_of_state import (
    saha_partition_depth_batch,
)
from .equation_of_state import load_ionization_potential_table_cm
from .population_layout import (
    atomic_population_slot_start,
    ion_stage_count_for_atomic_number,
)
from .runtime_state import AtmosphereRuntimeState


_SPECIFIC_INTERNAL_ENERGY_SLOT_COUNT = 840


@lru_cache(maxsize=1)
def _ionization_potential_sums() -> np.ndarray:
    """Return atomic specific internal energy for packed ion-stage slots."""

    ionization_potential = np.asarray(
        load_ionization_potential_table_cm(),
        dtype=np.float64,
    )
    if ionization_potential.size < 999:
        raise RuntimeError("ionization-potential table is too small for POTIONSUM")

    potential_sum = np.zeros(999, dtype=np.float64)
    packed_index = 0
    for atomic_number in range(1, 31):
        packed_index += 1
        potential_sum[packed_index - 1] = 0.0
        for _ion_stage in range(2, atomic_number + 2):
            packed_index += 1
            potential_sum[packed_index - 1] = (
                ionization_potential[packed_index - 2] + potential_sum[packed_index - 2]
            )

    for _atomic_number in range(31, 100):
        packed_index += 1
        potential_sum[packed_index - 1] = 0.0
        for _ion_stage in range(4):
            packed_index += 1
            potential_sum[packed_index - 1] = (
                ionization_potential[packed_index - 2] + potential_sum[packed_index - 2]
            )
    return potential_sum


def compute_atomic_specific_internal_energy(
    *,
    temperature_k: np.ndarray,
    state: AtmosphereRuntimeState,
) -> np.ndarray:
    """Compute atomic specific internal energy in erg g^-1 by layer."""

    temperature = np.asarray(temperature_k, dtype=np.float64)
    layer_count = int(temperature.size)
    specific_internal_energy = np.zeros(layer_count, dtype=np.float64)
    potential_sum = _ionization_potential_sums()
    if layer_count == 0:
        return specific_internal_energy

    # Depth-batched Saha evaluation: one call per
    # (element, +/- perturbation) instead of two per (layer, element). Each
    # batch row is bit-identical to the per-layer call it replaces (same
    # kernel, same scalar inputs: max(T, 1) * 1.001 / 0.999 computed
    # elementwise, the layer's electron density and unclamped charge-square
    # density), so the per-layer accumulation below is unchanged from the
    # original loop.
    safe_temperature_array = np.maximum(temperature, 1.0)
    temperature_plus_array = safe_temperature_array * 1.001
    temperature_minus_array = safe_temperature_array * 0.999
    electron_density_array = np.ascontiguousarray(
        state.electron_density[:layer_count], dtype=np.float64
    )
    charge_square_density_array = np.ascontiguousarray(
        state.charge_square_density[:layer_count], dtype=np.float64
    )

    partition_plus_by_layer = np.ones(
        (layer_count, _SPECIFIC_INTERNAL_ENERGY_SLOT_COUNT), dtype=np.float64
    )
    partition_minus_by_layer = np.ones(
        (layer_count, _SPECIFIC_INTERNAL_ENERGY_SLOT_COUNT), dtype=np.float64
    )
    for atomic_number in range(1, 100):
        ion_stage_count = (
            ion_stage_count_for_atomic_number(atomic_number)
            if atomic_number <= 30
            else 3
        )
        slot_start = atomic_population_slot_start(atomic_number)
        copy_count = min(
            ion_stage_count,
            _SPECIFIC_INTERNAL_ENERGY_SLOT_COUNT - slot_start,
        )
        if copy_count <= 0:
            continue

        plus_values = saha_partition_depth_batch(
            temperature_plus_array,
            electron_density_array,
            atomic_number,
            ion_stage_count,
            13,
            charge_square_density_array,
        )
        minus_values = saha_partition_depth_batch(
            temperature_minus_array,
            electron_density_array,
            atomic_number,
            ion_stage_count,
            13,
            charge_square_density_array,
        )
        partition_plus_by_layer[:, slot_start : slot_start + copy_count] = plus_values[
            :, :copy_count
        ]
        partition_minus_by_layer[:, slot_start : slot_start + copy_count] = (
            minus_values[:, :copy_count]
        )

    # Elementwise on the same doubles the original per-layer expression saw,
    # so every derivative entry is bit-identical.
    partition_derivative_by_layer = (
        (partition_plus_by_layer - partition_minus_by_layer)
        / np.maximum(partition_plus_by_layer + partition_minus_by_layer, 1.0e-30)
        * 1000.0
    )

    for layer_index, layer_temperature in enumerate(temperature):
        safe_temperature = max(float(layer_temperature), 1.0)
        thermal_energy = BOLTZMANN_ERG_PER_K_REFERENCE * safe_temperature
        hc_over_kt = (
            PLANCK_ERG_SECOND_REFERENCE
            * LIGHT_SPEED_CM_PER_S
            / max(thermal_energy, 1.0e-300)
        )
        total_particle_density = (
            state.electron_density[layer_index]
            + state.total_nuclei_number_density[layer_index]
        )

        energy = 1.5 * total_particle_density * thermal_energy
        partition_derivative = partition_derivative_by_layer[layer_index]
        for packed_slot in range(_SPECIFIC_INTERNAL_ENERGY_SLOT_COUNT):
            energy += (
                state.ion_stage_populations_by_packed_slot[layer_index, packed_slot]
                * thermal_energy
                * (
                    potential_sum[packed_slot] * hc_over_kt
                    + partition_derivative[packed_slot]
                )
            )
        specific_internal_energy[layer_index] = energy / np.maximum(
            state.mass_density[layer_index],
            1.0e-300,
        )
    return specific_internal_energy
