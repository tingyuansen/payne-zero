"""Torch equation of state and population solver for synthesis.

Partition functions and Saha solve from `partition_saha_inputs.npz`, then
per-layer ion populations and Doppler widths in the packed slot layout the
opacity kernels index (`_population_slot_maps` is shared with the
atmosphere-side bridge so both packages agree on the layout).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .constants import (
    REFERENCE_ATOMIC_MASS_GRAM,
    REFERENCE_BOLTZMANN_ERG_PER_K,
    REFERENCE_BOLTZMANN_EV_PER_K,
    REFERENCE_PLANCK_ERG_SECOND,
    REFERENCE_NATURAL_LOG_10,
    REFERENCE_SAHA_COEFFICIENT,
    REFERENCE_WAVENUMBER_PER_EV,
    LIGHT_SPEED_CM_PER_S,
)
from .device import DEFAULT_DTYPE, REFERENCE_DTYPE, device as _device
from .ground_partition_table import (
    FIRST_RANGE_LABELS,
    SECOND_RANGE_LABELS,
    ground_partition_values,
)
from . import paths as _runtime_paths

# Rounded EOS literals carried at the validated reference precision.


FLOAT64_POSITIVE_FLOOR = 1e-300
_FIRST_RANGE_LABELS = tuple(int(label) for label in FIRST_RANGE_LABELS)
_SECOND_RANGE_LABELS = tuple(int(label) for label in SECOND_RANGE_LABELS)

_DEFAULT_INPUTS = _runtime_paths.SYNTHESIS_TABLE_DIR / "partition_saha_inputs.npz"

_SPECIAL_LEVEL_TABLE_FIELDS = {
    "hydrogen_neutral": (
        "hydrogen_neutral_level_energy_cm",
        "hydrogen_neutral_level_statistical_weight",
    ),
    "helium_neutral": (
        "helium_neutral_level_energy_cm",
        "helium_neutral_level_statistical_weight",
    ),
    "helium_singly_ionized": (
        "helium_singly_ionized_level_energy_cm",
        "helium_singly_ionized_level_statistical_weight",
    ),
    "carbon_neutral": (
        "carbon_neutral_level_energy_cm",
        "carbon_neutral_level_statistical_weight",
    ),
    "carbon_singly_ionized": (
        "carbon_singly_ionized_level_energy_cm",
        "carbon_singly_ionized_level_statistical_weight",
    ),
    "oxygen_neutral": (
        "oxygen_neutral_level_energy_cm",
        "oxygen_neutral_level_statistical_weight",
    ),
    "magnesium_neutral": (
        "magnesium_neutral_level_energy_cm",
        "magnesium_neutral_level_statistical_weight",
    ),
    "magnesium_singly_ionized": (
        "magnesium_singly_ionized_level_energy_cm",
        "magnesium_singly_ionized_level_statistical_weight",
    ),
    "aluminum_neutral": (
        "aluminum_neutral_level_energy_cm",
        "aluminum_neutral_level_statistical_weight",
    ),
    "silicon_neutral": (
        "silicon_neutral_level_energy_cm",
        "silicon_neutral_level_statistical_weight",
    ),
    "silicon_singly_ionized": (
        "silicon_singly_ionized_level_energy_cm",
        "silicon_singly_ionized_level_statistical_weight",
    ),
    "calcium_neutral": (
        "calcium_neutral_level_energy_cm",
        "calcium_neutral_level_statistical_weight",
    ),
    "calcium_singly_ionized": (
        "calcium_singly_ionized_level_energy_cm",
        "calcium_singly_ionized_level_statistical_weight",
    ),
    "sodium_neutral": (
        "sodium_neutral_level_energy_cm",
        "sodium_neutral_level_statistical_weight",
    ),
    "boron_neutral": (
        "boron_neutral_level_energy_cm",
        "boron_neutral_level_statistical_weight",
    ),
    "potassium_neutral": (
        "potassium_neutral_level_energy_cm",
        "potassium_neutral_level_statistical_weight",
    ),
}

# Per-element high-lying terms differ enough that the dispatch stays explicit.

# Invariant tables uploaded once.


@dataclass
class EOSTables:
    """Atomic-data tables resident on a device.

    Built from the packaged EOS table bundle. Numeric arrays become torch
    tensors on `device`; host fp64 copies remain for discrete table lookups where
    dtype-dependent bracket choices would break parity.
    """

    device: torch.device
    dtype: torch.dtype
    packed_partition_table: torch.Tensor
    packed_partition_table_int: np.ndarray
    ionization_potential_cm: torch.Tensor
    iron_group_partition_grid: torch.Tensor
    iron_group_lower_potential_grid: torch.Tensor
    iron_group_lower_potential_log_grid: torch.Tensor
    element_block_offsets: np.ndarray
    partition_interpolation_scale: torch.Tensor
    ground_partition_table: torch.Tensor  # (605, 80)
    special_level_tables: dict[str, torch.Tensor]
    ionization_potential_cm_host: np.ndarray
    # Host copies for discrete packed-table lookups; fp32 bracket arithmetic can
    # choose a different table cell near boundaries.
    partition_interpolation_scale_host: np.ndarray
    iron_group_partition_grid_host: np.ndarray
    iron_group_lower_potential_grid_host: np.ndarray
    iron_group_lower_potential_log_grid_host: np.ndarray

    @classmethod
    def from_npz(
        cls, path: Path = _DEFAULT_INPUTS, device=None, dtype=None
    ) -> "EOSTables":
        with np.load(path, allow_pickle=False) as data:
            table_data = {key: data[key] for key in data.files}
        return cls.from_dict(table_data, device=device, dtype=dtype)

    @classmethod
    def from_dict(cls, d: dict, device=None, dtype=None) -> "EOSTables":
        if device is None:
            device = _device()
        if dtype is None:
            dtype = DEFAULT_DTYPE

        def array(name):
            return np.asarray(d[name])

        def tensor_from_array(name):
            return torch.as_tensor(array(name), dtype=dtype, device=device)

        special_level_tables = {
            field_name: tensor_from_array(field_name)
            for field_pair in _SPECIAL_LEVEL_TABLE_FIELDS.values()
            for field_name in field_pair
        }
        packed_partition_table = array("packed_partition_table")
        return cls(
            device=device,
            dtype=dtype,
            packed_partition_table=torch.as_tensor(
                packed_partition_table, dtype=dtype, device=device
            ),
            packed_partition_table_int=packed_partition_table.astype(np.int64),
            ionization_potential_cm=tensor_from_array("ionization_potential_cm"),
            iron_group_partition_grid=tensor_from_array("iron_group_partition_grid"),
            iron_group_lower_potential_grid=tensor_from_array(
                "iron_group_lower_potential_grid"
            ),
            iron_group_lower_potential_log_grid=tensor_from_array(
                "iron_group_lower_potential_log_grid"
            ),
            element_block_offsets=array("element_block_offsets").astype(np.int64),
            partition_interpolation_scale=tensor_from_array(
                "partition_interpolation_scale"
            ),
            ground_partition_table=tensor_from_array("ground_partition_table"),
            special_level_tables=special_level_tables,
            ionization_potential_cm_host=array("ionization_potential_cm").astype(
                np.float64
            ),
            partition_interpolation_scale_host=array(
                "partition_interpolation_scale"
            ).astype(np.float64),
            iron_group_partition_grid_host=array("iron_group_partition_grid").astype(
                np.float64
            ),
            iron_group_lower_potential_grid_host=array(
                "iron_group_lower_potential_grid"
            ).astype(np.float64),
            iron_group_lower_potential_log_grid_host=array(
                "iron_group_lower_potential_log_grid"
            ).astype(np.float64),
        )


# Static per-element dispatch resolved on the host.


def _ionization_potential_index_1based(atomic_number: int, ion_stage: int) -> int:
    """1-based index into the flat ionization-potential table."""
    if atomic_number <= 30:
        return atomic_number * (atomic_number + 1) // 2 + ion_stage - 1
    return atomic_number * 5 + 341 + ion_stage - 1


def _element_partition_block(atomic_number: int, element_block_offsets: np.ndarray):
    """Return the first table column and available ion stages for one element."""
    if atomic_number <= 28:
        first_species_column_1based = int(element_block_offsets[atomic_number - 1])
        available_ion_count = (
            int(
                element_block_offsets[atomic_number]
                - element_block_offsets[atomic_number - 1]
            )
            if atomic_number < len(element_block_offsets)
            else 3
        )
    else:
        first_species_column_1based = 3 * atomic_number + 54
        available_ion_count = 3
    if atomic_number == 6:
        first_species_column_1based, available_ion_count = 354, 6
    elif atomic_number == 7:
        first_species_column_1based, available_ion_count = 360, 7
    elif atomic_number == 8:
        first_species_column_1based, available_ion_count = 367, 8
    if 20 <= atomic_number < 29:
        available_ion_count = 10
    return first_species_column_1based, available_ion_count


def _ground_partition_values_np(
    packed_ion_slot: int, temperature_host: np.ndarray
) -> np.ndarray:
    """Evaluate the ground-state correction on the host temperature grid."""
    temperature_array = np.asarray(temperature_host, np.float64)
    if packed_ion_slot <= 0:
        return np.ones_like(temperature_array)
    if packed_ion_slot <= len(_FIRST_RANGE_LABELS):
        label = _FIRST_RANGE_LABELS[packed_ion_slot - 1]
    elif packed_ion_slot < 169:
        return np.ones_like(temperature_array)
    elif packed_ion_slot - 169 < len(_SECOND_RANGE_LABELS):
        label = _SECOND_RANGE_LABELS[packed_ion_slot - 169]
    else:
        label = 666
    return ground_partition_values(int(label), temperature_array)


# Iron-group partition grid, depth-batched bilinear interpolation.


def _iron_group_partition(
    tables: EOSTables,
    atomic_number: int,
    ion_stage: int,
    log10_temperature_host: np.ndarray,
    potential_lowering_cm: torch.Tensor,
    dtype,
    device,
) -> torch.Tensor:
    """Iron-group partition interpolation for one element and ion stage.

    Temperature and lowering brackets are discrete host-fp64 decisions so the
    table cell matches the reference implementation exactly.
    """
    partition_grid = tables.iron_group_partition_grid_host
    lowering_grid = tables.iron_group_lower_potential_grid_host
    log_lowering_grid = tables.iron_group_lower_potential_log_grid_host
    element_index = atomic_number - 20
    ion_index = ion_stage - 1
    lowering_host = potential_lowering_cm.detach().cpu().numpy().astype(np.float64)
    n_depths = log10_temperature_host.shape[0]
    log10_temperature = log10_temperature_host

    hot_temperature_index = np.clip(
        ((log10_temperature - 4.0) / 0.05 + 31.0).astype(np.int64),
        1,
        56,
    )
    hot_temperature_weight = (
        log10_temperature
        - (hot_temperature_index.astype(np.float64) - 31.0) * 0.05
        - 4.0
    ) / 0.05
    cool_temperature_index = np.maximum(
        ((log10_temperature - 3.32) / 0.02 + 2.0).astype(np.int64),
        2,
    )
    cool_temperature_weight = (
        log10_temperature
        - (cool_temperature_index.astype(np.float64) - 2.0) * 0.02
        - 3.32
    ) / 0.02
    mid_temperature_index = ((log10_temperature - 3.7) / 0.03 + 21.0).astype(np.int64)
    mid_temperature_weight = (
        log10_temperature
        - (mid_temperature_index.astype(np.float64) - 21.0) * 0.03
        - 3.7
    ) / 0.03

    hot_temperature = log10_temperature > 4.0
    cool_temperature = log10_temperature < 3.7
    temperature_index = np.where(
        hot_temperature,
        hot_temperature_index,
        np.where(cool_temperature, cool_temperature_index, mid_temperature_index),
    )
    temperature_weight = np.where(
        hot_temperature,
        hot_temperature_weight,
        np.where(cool_temperature, cool_temperature_weight, mid_temperature_weight),
    )
    temperature_index = np.clip(temperature_index, 1, 56)
    temperature_upper_index = temperature_index - 1
    temperature_lower_index = np.maximum(temperature_upper_index - 1, 0)

    partition_slice = partition_grid[:, :, ion_index, element_index]

    weak_lowering_value = (
        temperature_weight * partition_slice[0, temperature_upper_index]
        + (1.0 - temperature_weight) * partition_slice[0, temperature_lower_index]
    )

    n_lowering_grid = lowering_grid.shape[0]
    lowering_index = np.full(n_depths, n_lowering_grid - 1, dtype=np.int64)
    lowering_found = np.zeros(n_depths, dtype=bool)
    for grid_index in range(1, n_lowering_grid):
        bracket_hit = (~lowering_found) & (lowering_host < lowering_grid[grid_index])
        lowering_index = np.where(bracket_hit, grid_index, lowering_index)
        lowering_found = lowering_found | bracket_hit

    safe_lowering = np.maximum(lowering_host, 1e-30)
    lowering_weight = (
        np.log10(safe_lowering) - log_lowering_grid[lowering_index - 1]
    ) / 0.30103
    current_upper_temp = partition_slice[lowering_index, temperature_upper_index]
    current_lower_temp = partition_slice[lowering_index, temperature_lower_index]
    previous_upper_temp = partition_slice[lowering_index - 1, temperature_upper_index]
    previous_lower_temp = partition_slice[lowering_index - 1, temperature_lower_index]
    finite_lowering_value = lowering_weight * (
        temperature_weight * current_upper_temp
        + (1.0 - temperature_weight) * current_lower_temp
    ) + (1.0 - lowering_weight) * (
        temperature_weight * previous_upper_temp
        + (1.0 - temperature_weight) * previous_lower_temp
    )

    partition_value = np.where(
        lowering_host < lowering_grid[0],
        weak_lowering_value,
        finite_lowering_value,
    )
    return torch.as_tensor(partition_value, dtype=dtype, device=device)


# Special light-element partition sums, depth-batched.


def _boltzmann_level_sum(tables, species_key, n_levels, hc_over_kt):
    """Depth-batched sum of excited-level Boltzmann weights."""
    energy_field, weight_field = _SPECIAL_LEVEL_TABLE_FIELDS[species_key]
    level_energy_cm = tables.special_level_tables[energy_field]
    statistical_weight = tables.special_level_tables[weight_field]
    partition_sum = torch.zeros_like(hc_over_kt)
    for level_index in range(1, n_levels):
        partition_sum = partition_sum + statistical_weight[level_index] * torch.exp(
            -level_energy_cm[level_index] * hc_over_kt
        )
    return partition_sum


def _boltzmann_level_sum_with_base(
    tables,
    base_partition,
    species_key,
    n_levels,
    hc_over_kt,
):
    """Depth-batched Boltzmann sum with an existing base partition term."""
    energy_field, weight_field = _SPECIAL_LEVEL_TABLE_FIELDS[species_key]
    level_energy_cm = tables.special_level_tables[energy_field]
    statistical_weight = tables.special_level_tables[weight_field]
    partition_sum = base_partition.clone()
    for level_index in range(1, n_levels):
        partition_sum = partition_sum + statistical_weight[level_index] * torch.exp(
            -level_energy_cm[level_index] * hc_over_kt
        )
    return partition_sum


def _special_partition(tables, species_column, temperature, hc_over_kt):
    """Depth-batched special light-element partition functions.

    Returns ``(partition_function, statistical_weight_override,
    occupation_correction)`` or ``None`` for ions handled by the ordinary packed
    partition table.
    """
    zero_correction = torch.zeros_like(hc_over_kt)

    if species_column == 2:
        return torch.ones_like(hc_over_kt), None, zero_correction
    if species_column == 1:  # H I
        base_partition = torch.full_like(hc_over_kt, 2.0)
        partition_function = torch.where(
            temperature >= 9000.0,
            _boltzmann_level_sum(tables, "hydrogen_neutral", 6, hc_over_kt) + 2.0,
            base_partition,
        )
        return partition_function, None, 109677.576 / 6.5 / 6.5 * hc_over_kt
    if species_column == 3:  # He I
        base_partition = torch.ones_like(hc_over_kt)
        partition_function = torch.where(
            temperature >= 15000.0,
            _boltzmann_level_sum(tables, "helium_neutral", 29, hc_over_kt) + 1.0,
            base_partition,
        )
        return partition_function, None, 109677.576 / 5.5 / 5.5 * hc_over_kt
    if species_column == 4:  # He II
        base_partition = torch.full_like(hc_over_kt, 2.0)
        partition_function = torch.where(
            temperature >= 30000.0,
            _boltzmann_level_sum(
                tables,
                "helium_singly_ionized",
                6,
                hc_over_kt,
            )
            + 2.0,
            base_partition,
        )
        return partition_function, None, 4.0 * 109722.267 / 6.5 / 6.5 * hc_over_kt
    if species_column == 354:  # C I
        base_partition = (
            1.0
            + 3.0 * torch.exp(-16.42 * hc_over_kt)
            + 5.0 * torch.exp(-43.42 * hc_over_kt)
        )
        partition_function = _boltzmann_level_sum_with_base(
            tables, base_partition, "carbon_neutral", 14, hc_over_kt
        )
        partition_function = partition_function + (
            108.0 * torch.exp(-80000.0 * hc_over_kt)
            + 189.0 * torch.exp(-84000.0 * hc_over_kt)
            + 247.0 * torch.exp(-87000.0 * hc_over_kt)
            + 231.0 * torch.exp(-88000.0 * hc_over_kt)
            + 190.0 * torch.exp(-89000.0 * hc_over_kt)
            + 300.0 * torch.exp(-90000.0 * hc_over_kt)
        )
        return partition_function, None, zero_correction
    if species_column == 355:  # C II
        base_partition = 2.0 + 4.0 * torch.exp(-63.42 * hc_over_kt)
        partition_function = _boltzmann_level_sum_with_base(
            tables, base_partition, "carbon_singly_ionized", 6, hc_over_kt
        )
        partition_function = partition_function + (
            6.0 * torch.exp(-131731.80 * hc_over_kt)
            + 4.0 * torch.exp(-142027.1 * hc_over_kt)
            + 10.0 * torch.exp(-145550.13 * hc_over_kt)
            + 10.0 * torch.exp(-150463.62 * hc_over_kt)
            + 2.0 * torch.exp(-157234.07 * hc_over_kt)
            + 6.0 * torch.exp(-162500.0 * hc_over_kt)
            + 42.0 * torch.exp(-168000.0 * hc_over_kt)
            + 56.0 * torch.exp(-178000.0 * hc_over_kt)
            + 102.0 * torch.exp(-183000.0 * hc_over_kt)
            + 400.0 * torch.exp(-188000.0 * hc_over_kt)
        )
        return partition_function, None, zero_correction
    if species_column == 51:  # Mg I
        partition_function = (
            _boltzmann_level_sum(tables, "magnesium_neutral", 11, hc_over_kt) + 1.0
        )
        partition_function = partition_function + (
            5.0 * torch.exp(-53134.0 * hc_over_kt)
            + 15.0 * torch.exp(-54192.0 * hc_over_kt)
            + 28.0 * torch.exp(-54676.0 * hc_over_kt)
            + 9.0 * torch.exp(-57853.0 * hc_over_kt)
        )
        return partition_function, 4.0, 109734.83 / 4.5 / 4.5 * hc_over_kt
    if species_column == 52:  # Mg II
        partition_function = (
            _boltzmann_level_sum(tables, "magnesium_singly_ionized", 6, hc_over_kt)
            + 2.0
        )
        partition_function = partition_function + (
            10.0 * torch.exp(-93310.80 * hc_over_kt)
            + 14.0 * torch.exp(-93799.70 * hc_over_kt)
            + 6.0 * torch.exp(-97464.32 * hc_over_kt)
            + 10.0 * torch.exp(-103419.82 * hc_over_kt)
            + 14.0 * torch.exp(-103689.89 * hc_over_kt)
            + 18.0 * torch.exp(-103705.66 * hc_over_kt)
        )
        return partition_function, 2.0, 4.0 * 109734.83 / 5.5 / 5.5 * hc_over_kt
    if species_column == 57:  # Al I
        base_partition = 2.0 + 4.0 * torch.exp(-112.061 * hc_over_kt)
        partition_function = _boltzmann_level_sum_with_base(
            tables, base_partition, "aluminum_neutral", 9, hc_over_kt
        )
        partition_function = (
            partition_function
            + 10.0 * torch.exp(-42235.0 * hc_over_kt)
            + 14.0 * torch.exp(-43831.0 * hc_over_kt)
        )
        return partition_function, 2.0, 109735.08 / 5.5 / 5.5 * hc_over_kt
    if species_column == 63:  # Si I
        base_partition = (
            1.0
            + 3.0 * torch.exp(-77.115 * hc_over_kt)
            + 5.0 * torch.exp(-223.157 * hc_over_kt)
        )
        partition_function = _boltzmann_level_sum_with_base(
            tables, base_partition, "silicon_neutral", 11, hc_over_kt
        )
        partition_function = partition_function + (
            76.0 * torch.exp(-53000.0 * hc_over_kt)
            + 71.0 * torch.exp(-57000.0 * hc_over_kt)
            + 191.0 * torch.exp(-60000.0 * hc_over_kt)
            + 240.0 * torch.exp(-62000.0 * hc_over_kt)
            + 251.0 * torch.exp(-63000.0 * hc_over_kt)
            + 300.0 * torch.exp(-65000.0 * hc_over_kt)
        )
        return partition_function, None, zero_correction
    if species_column == 64:  # Si II
        base_partition = 2.0 + 4.0 * torch.exp(-287.32 * hc_over_kt)
        partition_function = _boltzmann_level_sum_with_base(
            tables, base_partition, "silicon_singly_ionized", 6, hc_over_kt
        )
        partition_function = partition_function + (
            6.0 * torch.exp(-81231.59 * hc_over_kt)
            + 6.0 * torch.exp(-83937.08 * hc_over_kt)
            + 10.0 * torch.exp(-101024.09 * hc_over_kt)
            + 14.0 * torch.exp(-103556.35 * hc_over_kt)
            + 10.0 * torch.exp(-108800.0 * hc_over_kt)
            + 42.0 * torch.exp(-115000.0 * hc_over_kt)
            + 6.0 * torch.exp(-121000.0 * hc_over_kt)
            + 38.0 * torch.exp(-125000.0 * hc_over_kt)
            + 34.0 * torch.exp(-132000.0 * hc_over_kt)
        )
        return partition_function, 2.0, 4.0 * 109734.83 / 4.5 / 4.5 * hc_over_kt
    if species_column == 96:  # Ca I
        partition_function = (
            _boltzmann_level_sum(tables, "calcium_neutral", 8, hc_over_kt) + 1.0
        )
        partition_function = partition_function + (
            28.0 * torch.exp(-37000.0 * hc_over_kt)
            + 67.0 * torch.exp(-40000.0 * hc_over_kt)
            + 21.0 * torch.exp(-43000.0 * hc_over_kt)
            + 34.0 * torch.exp(-48000.0 * hc_over_kt)
        )
        return partition_function, 4.0, 109734.82 / 4.5 / 4.5 * hc_over_kt
    if species_column == 97:  # Ca II
        partition_function = (
            _boltzmann_level_sum(tables, "calcium_singly_ionized", 5, hc_over_kt) + 2.0
        )
        partition_function = partition_function + 12.0 * torch.exp(
            -68000.0 * hc_over_kt
        )
        return partition_function, 2.0, 109734.83 / 4.5 / 4.5 * hc_over_kt
    if species_column == 367:  # O I
        base_partition = (
            5.0
            + 3.0 * torch.exp(-158.265 * hc_over_kt)
            + torch.exp(-226.977 * hc_over_kt)
        )
        partition_function = _boltzmann_level_sum_with_base(
            tables, base_partition, "oxygen_neutral", 13, hc_over_kt
        )
        partition_function = partition_function + (
            15.0 * torch.exp(-101140.0 * hc_over_kt)
            + 131.0 * torch.exp(-103000.0 * hc_over_kt)
            + 128.0 * torch.exp(-105000.0 * hc_over_kt)
            + 600.0 * torch.exp(-107000.0 * hc_over_kt)
        )
        return partition_function, None, zero_correction
    if species_column == 45:  # Na I
        partition_function = (
            _boltzmann_level_sum(tables, "sodium_neutral", 8, hc_over_kt) + 2.0
        )
        partition_function = (
            partition_function
            + 10.0 * torch.exp(-34548.745 * hc_over_kt)
            + 14.0 * torch.exp(-34586.96 * hc_over_kt)
        )
        return partition_function, 2.0, 109734.83 / 4.5 / 4.5 * hc_over_kt
    if species_column == 14:  # B I
        base_partition = 2.0 + 4.0 * torch.exp(-15.25 * hc_over_kt)
        partition_function = _boltzmann_level_sum_with_base(
            tables, base_partition, "boron_neutral", 7, hc_over_kt
        )
        partition_function = partition_function + (
            6.0 * torch.exp(-57786.80 * hc_over_kt)
            + 10.0 * torch.exp(-59989.0 * hc_over_kt)
            + 14.0 * torch.exp(-60031.03 * hc_over_kt)
            + 2.0 * torch.exp(-63561.0 * hc_over_kt)
        )
        return partition_function, 2.0, 109734.83 / 4.5 / 4.5 * hc_over_kt
    if species_column == 91:  # K I
        partition_function = (
            _boltzmann_level_sum(tables, "potassium_neutral", 8, hc_over_kt) + 2.0
        )
        partition_function = (
            partition_function
            + 10.0 * torch.exp(-27397.077 * hc_over_kt)
            + 14.0 * torch.exp(-28127.85 * hc_over_kt)
        )
        return partition_function, 2.0, 109734.83 / 5.5 / 5.5 * hc_over_kt
    return None


# Occupation correction and Debye lowering, depth-batched.


def _debye_lowering(electron_density, thermal_energy_erg, gas_pressure):
    """Per-unit-charge Debye lowering in eV, depth-batched."""
    effective_charge_density = 2.0 * electron_density
    pressure_excess_charge = 2.0 * electron_density - gas_pressure / thermal_energy_erg
    effective_charge_density = torch.where(
        pressure_excess_charge > 0.0,
        effective_charge_density + 2.0 * pressure_excess_charge,
        effective_charge_density,
    )
    effective_charge_density = torch.where(
        effective_charge_density == 0.0,
        torch.ones_like(effective_charge_density),
        effective_charge_density,
    )
    debye_radius = torch.sqrt(
        thermal_energy_erg / 2.8965e-18 / effective_charge_density
    )
    return torch.clamp(1.44e-7 / debye_radius, max=1.0)


def _debye_lowering_np(electron_density, thermal_energy_erg, gas_pressure):
    """Host fp64 Debye lowering for discrete occupation gates."""
    electron_density = np.asarray(electron_density, np.float64)
    thermal_energy_erg = np.asarray(thermal_energy_erg, np.float64)
    gas_pressure = np.asarray(gas_pressure, np.float64)
    effective_charge_density = 2.0 * electron_density
    pressure_excess_charge = 2.0 * electron_density - gas_pressure / thermal_energy_erg
    effective_charge_density = np.where(
        pressure_excess_charge > 0.0,
        effective_charge_density + 2.0 * pressure_excess_charge,
        effective_charge_density,
    )
    effective_charge_density = np.where(
        effective_charge_density == 0.0,
        1.0,
        effective_charge_density,
    )
    debye_radius = np.sqrt(thermal_energy_erg / 2.8965e-18 / effective_charge_density)
    return np.minimum(1.0, 1.44e-7 / debye_radius)


def _occupation_term(cutoff_parameter, ion_charge, thermal_energy_ev):
    """Effective high-lying state count down to an occupation cutoff."""
    scaled_cutoff = (
        torch.sqrt(
            13.595 * ion_charge * ion_charge / thermal_energy_ev / cutoff_parameter
        )
        ** 3
    )
    polynomial = (
        1.0 / 3.0
        + (
            1.0
            - (0.5 + (1.0 / 18.0 + cutoff_parameter / 120.0) * cutoff_parameter)
            * cutoff_parameter
        )
        * cutoff_parameter
    )
    return scaled_cutoff * polynomial


# Packed ordinary-ion partition lookup, depth-batched.


def _packed_partition_interpolation(
    tables,
    packed_column_index,
    ionization_potential_ev,
    temperature_host,
    dtype,
    device,
):
    """Ordinary-ion partition interpolation from the packed table."""
    packed_partition_table = tables.packed_partition_table_int
    n_table_rows = packed_partition_table.shape[0]
    interpolation_scale = tables.partition_interpolation_scale_host

    reference_temperature = ionization_potential_ev * 2000.0 / 11.0
    temperature_bracket = np.clip(
        (temperature_host / reference_temperature - 0.5).astype(np.int64),
        1,
        9,
    )
    temperature_fraction = (
        temperature_host / reference_temperature
        - temperature_bracket.astype(np.float64)
        - 0.5
    )
    row_index = np.clip((temperature_bracket + 1) // 2 - 1, 0, n_table_rows - 1)

    packed_value = packed_partition_table[row_index, packed_column_index]
    upper_digits = packed_value // 100000
    lower_digits_with_scale = packed_value - upper_digits * 100000
    lower_digits = lower_digits_with_scale // 10
    scale_digit = lower_digits_with_scale - lower_digits * 10
    scale_index = np.clip(scale_digit - 1, 0, interpolation_scale.shape[0] - 1)

    odd_left_value = upper_digits.astype(np.float64) * interpolation_scale[scale_index]
    odd_right_value = lower_digits.astype(np.float64) * interpolation_scale[scale_index]

    next_row_index = np.clip(row_index + 1, 0, n_table_rows - 1)
    next_packed_value = packed_partition_table[next_row_index, packed_column_index]
    next_scale_index = np.clip(
        (next_packed_value % 10) - 1,
        0,
        interpolation_scale.shape[0] - 1,
    )
    even_left_value = lower_digits.astype(np.float64) * interpolation_scale[scale_index]
    even_right_value = (next_packed_value // 100000).astype(
        np.float64
    ) * interpolation_scale[next_scale_index]

    odd_bracket = (temperature_bracket % 2) == 1
    left_value = np.where(odd_bracket, odd_left_value, even_left_value)
    right_value = np.where(odd_bracket, odd_right_value, even_right_value)

    floor_condition = (
        odd_bracket
        & (temperature_fraction < 0.0)
        & (scale_index <= 0)
        & (left_value == np.floor(right_value + 0.5))
    )
    floor_value = np.where(floor_condition, left_value, 1.0)

    partition_value = np.maximum(
        floor_value,
        left_value + (right_value - left_value) * temperature_fraction,
    )
    return torch.as_tensor(partition_value, dtype=dtype, device=device)


# Per-element partition builder and Saha ladder, depth-batched.


def _build_partition_state_for_element(
    tables,
    atomic_number,
    requested_ion_count,
    state,
    *,
    apply_ground_partition: bool = True,
):
    """Build partition functions and ionization state for one element.

    Returns ``(partition_by_stage, ionization_potential_ev,
    potential_lowering_ev, ion_stage_count)`` with tensors shaped
    ``(ion_stage, depth)``.
    """
    temperature = state["temperature"]
    thermal_energy_ev = state["thermal_energy_ev"]
    hc_over_kt = state["hc_over_kt"]
    potential_lowering = state["potential_lowering"]
    temperature_host = state["temperature_host"]
    natural_log_temperature_host = state["natural_log_temperature_host"]
    potential_lowering_host = state["potential_lowering_host"]
    dtype = temperature.dtype
    device = temperature.device
    n_depths = temperature.shape[0]

    first_species_column_1based, available_ion_count = _element_partition_block(
        atomic_number,
        tables.element_block_offsets,
    )
    first_species_column_0based = first_species_column_1based - 1
    ion_stage_count = min(requested_ion_count + 2, available_ion_count)

    partition_by_stage = torch.ones(
        ion_stage_count, n_depths, dtype=dtype, device=device
    )
    ionization_potential_ev = torch.zeros(
        ion_stage_count, n_depths, dtype=dtype, device=device
    )
    potential_lowering_ev = torch.zeros(
        ion_stage_count, n_depths, dtype=dtype, device=device
    )

    for ion_stage in range(1, ion_stage_count + 1):
        ion_charge = float(ion_stage)
        stage_index = ion_stage - 1
        potential_lowering_ev[stage_index] = potential_lowering * ion_charge

        potential_index = (
            _ionization_potential_index_1based(atomic_number, ion_stage) - 1
        )
        ionization_potential_value_ev = 0.0
        if 0 <= potential_index < tables.ionization_potential_cm_host.size:
            ionization_potential_value_ev = (
                tables.ionization_potential_cm_host[potential_index]
                / REFERENCE_WAVENUMBER_PER_EV
            )
            if ionization_potential_value_ev == 0.0 and potential_index > 0:
                ionization_potential_value_ev = (
                    tables.ionization_potential_cm_host[potential_index - 1]
                    / REFERENCE_WAVENUMBER_PER_EV
                )
        ionization_potential_ev[stage_index] = ionization_potential_value_ev

        if 20 <= atomic_number < 29:
            partition_by_stage[stage_index] = _iron_group_partition(
                tables,
                atomic_number,
                ion_stage,
                natural_log_temperature_host / REFERENCE_NATURAL_LOG_10,
                potential_lowering_ev[stage_index] * REFERENCE_WAVENUMBER_PER_EV,
                dtype,
                device,
            )
            continue

        species_column = first_species_column_0based + ion_stage
        packed_column_index = species_column - 1
        statistical_weight_override = 0.0
        occupation_correction = torch.zeros(n_depths, dtype=dtype, device=device)
        if packed_column_index < tables.packed_partition_table_int.shape[1]:
            packed_metadata = int(
                tables.packed_partition_table_int[5, packed_column_index]
            )
            statistical_weight_override = float(
                packed_metadata - (packed_metadata // 100) * 100
            )

        handled = _special_partition(
            tables,
            species_column,
            temperature,
            hc_over_kt,
        )
        if handled is not None:
            special_partition, special_weight_override, occupation_correction = handled
            partition_by_stage[stage_index] = special_partition
            if special_weight_override is not None:
                statistical_weight_override = float(special_weight_override)
        elif (
            packed_column_index < tables.packed_partition_table_int.shape[1]
            and ionization_potential_value_ev > 0.0
        ):
            partition_by_stage[stage_index] = _packed_partition_interpolation(
                tables,
                packed_column_index,
                ionization_potential_value_ev,
                temperature_host,
                dtype,
                device,
            )
        else:
            partition_by_stage[stage_index] = torch.ones(
                n_depths,
                dtype=dtype,
                device=device,
            )

        # Keep regime gates on host fp64 temperature/lowering, matching the reference.
        if ionization_potential_value_ev > 0.0:
            reference_temperature = ionization_potential_value_ev * 2000.0 / 11.0
            low_temperature_host = temperature_host < (reference_temperature * 2.0)
            low_temperature_mask = torch.as_tensor(low_temperature_host, device=device)
            packed_ion_slot = (atomic_number - 1) * 6 + ion_stage
            if apply_ground_partition:
                ground_partition_row = torch.as_tensor(
                    _ground_partition_values_np(packed_ion_slot, temperature_host),
                    dtype=dtype,
                    device=device,
                )
                corrected_partition = torch.maximum(
                    partition_by_stage[stage_index],
                    ground_partition_row,
                )
                apply_low_temperature = low_temperature_mask & (
                    ground_partition_row > 0.0
                )
                partition_by_stage[stage_index] = torch.where(
                    apply_low_temperature,
                    corrected_partition,
                    partition_by_stage[stage_index],
                )
            skip_high_temperature_correction = low_temperature_mask

            occupation_correction_positive = occupation_correction > 0.0
            special_occupation_bypass = bool(occupation_correction_positive.any())
            if statistical_weight_override > 0.0 or special_occupation_bypass:
                high_temperature_mask = torch.as_tensor(
                    temperature_host >= (reference_temperature * 4.0),
                    device=device,
                )
                enough_lowering = torch.as_tensor(
                    (potential_lowering_host * ion_charge) >= 0.1,
                    device=device,
                )
                lowering_gate = occupation_correction_positive | enough_lowering
                temperature_gate = (
                    occupation_correction_positive | high_temperature_mask
                )
                occupation_gate = (
                    lowering_gate
                    & temperature_gate
                    & (~skip_high_temperature_correction)
                )
                capped_temperature_mask = torch.as_tensor(
                    temperature_host > (reference_temperature * 11.0),
                    device=device,
                )
                capped_thermal_energy_ev = torch.where(
                    capped_temperature_mask,
                    torch.full_like(
                        temperature,
                        (reference_temperature * 11.0) * REFERENCE_BOLTZMANN_EV_PER_K,
                    ),
                    thermal_energy_ev,
                )
                lower_occupation_cutoff = torch.where(
                    occupation_correction <= 0.0,
                    0.1 / capped_thermal_energy_ev,
                    occupation_correction,
                )
                upper_occupation_cutoff = (
                    potential_lowering_ev[stage_index] / capped_thermal_energy_ev
                )
                if statistical_weight_override > 0.0:
                    occupation_addition = (
                        statistical_weight_override
                        * torch.exp(
                            -ionization_potential_value_ev / capped_thermal_energy_ev
                        )
                        * (
                            _occupation_term(
                                upper_occupation_cutoff,
                                ion_charge,
                                capped_thermal_energy_ev,
                            )
                            - _occupation_term(
                                lower_occupation_cutoff,
                                ion_charge,
                                capped_thermal_energy_ev,
                            )
                        )
                    )
                    partition_by_stage[stage_index] = torch.where(
                        occupation_gate,
                        partition_by_stage[stage_index] + occupation_addition,
                        partition_by_stage[stage_index],
                    )

    return (
        partition_by_stage,
        ionization_potential_ev,
        potential_lowering_ev,
        ion_stage_count,
    )


def _saha_ladder(
    partition_by_stage,
    ionization_potential_ev,
    potential_lowering_ev,
    temperature,
    thermal_energy_ev,
    electron_density,
):
    """Return ion-stage fractions divided by partition functions."""
    ion_stage_count, n_depths = partition_by_stage.shape
    dtype = partition_by_stage.dtype
    device = partition_by_stage.device
    # Saha ratios are accumulated in log space to avoid fp32 overflow.
    log_cf = (
        math.log(2.0 * REFERENCE_SAHA_COEFFICIENT)
        + 1.5 * torch.log(temperature)
        - torch.log(electron_density)
    )

    log_stage_ratio = torch.zeros(ion_stage_count, n_depths, dtype=dtype, device=device)
    dtype_tiny = torch.finfo(dtype).tiny
    for ion_stage in range(2, ion_stage_count + 1):
        stage_index = ion_stage - 1
        log_ratio = (
            log_cf
            + torch.log(torch.clamp(partition_by_stage[stage_index], min=dtype_tiny))
            - torch.log(
                torch.clamp(partition_by_stage[stage_index - 1], min=dtype_tiny)
            )
            - (
                ionization_potential_ev[stage_index - 1]
                - potential_lowering_ev[stage_index - 1]
            )
            / thermal_energy_ev
        )
        log_stage_ratio[stage_index] = torch.where(
            partition_by_stage[stage_index - 1] > 0.0,
            log_ratio,
            torch.full_like(log_ratio, -float("inf")),
        )

    log_stage_population = torch.cumsum(log_stage_ratio, dim=0)
    log_stage_max = torch.amax(log_stage_population, dim=0, keepdim=True)
    stage_weight = torch.exp(log_stage_population - log_stage_max)
    stage_norm = torch.sum(stage_weight, dim=0, keepdim=True)
    stage_fraction = stage_weight / stage_norm

    stage_fractions_over_partition = torch.where(
        partition_by_stage > 0.0,
        stage_fraction / partition_by_stage,
        torch.zeros_like(stage_fraction),
    )
    return stage_fractions_over_partition


# Public API.


@dataclass
class EOSResult:
    partition_functions: torch.Tensor  # (depth, 99, 6)
    partition_normalized_populations: torch.Tensor  # (depth, 99, 6)
    ion_stage_fractions_over_partition: torch.Tensor  # (depth, 99, 6)


def populations(
    state: dict, tables: EOSTables, n_elements: int = 99, max_ion: int = 6
) -> EOSResult:
    """Return partition functions and Saha populations for all elements."""
    device = tables.device
    dtype = tables.dtype
    temperature = state["temperature"].to(device=device, dtype=dtype)
    thermal_energy_ev = state["thermal_energy_ev"].to(device=device, dtype=dtype)
    thermal_energy_erg = state["thermal_energy_erg"].to(device=device, dtype=dtype)
    hc_over_kt = state["hc_over_kt"].to(device=device, dtype=dtype)
    natural_log_temperature = state["natural_log_temperature"].to(
        device=device, dtype=dtype
    )
    gas_pressure = state["gas_pressure"].to(device=device, dtype=dtype)
    electron_density = state["electron_density"].to(device=device, dtype=dtype)
    total_nuclei_number_density = state["total_nuclei_number_density"].to(
        device=device, dtype=dtype
    )
    elemental_abundances = state["elemental_abundances"]
    if torch.is_tensor(elemental_abundances):
        elemental_abundances = elemental_abundances.to(device=device, dtype=dtype)
    else:
        elemental_abundances = torch.as_tensor(
            np.asarray(elemental_abundances), dtype=dtype, device=device
        )
    ion_stage_count = np.asarray(state["ion_stage_count"]).astype(np.int64)

    n_depths = temperature.shape[0]
    potential_lowering = _debye_lowering(
        electron_density, thermal_energy_erg, gas_pressure
    )

    # Host fp64 copies keep table-cell and regime decisions stable across
    # device dtypes. Prefer loader-provided copies, otherwise copy from tensors.
    if "temperature_host" in state:
        temperature_host = np.asarray(state["temperature_host"], dtype=np.float64)
    else:
        temperature_host = (
            state["temperature"].detach().to("cpu", torch.float64).numpy()
        )
    if "natural_log_temperature_host" in state:
        natural_log_temperature_host = np.asarray(
            state["natural_log_temperature_host"], dtype=np.float64
        )
    else:
        natural_log_temperature_host = (
            state["natural_log_temperature"].detach().to("cpu", torch.float64).numpy()
        )
    # potlow in fp64 from fp64 host inputs (matches debye_lowering exactly)
    if {"electron_density_host", "thermal_energy_erg_host", "gas_pressure_host"} <= set(
        state
    ):
        potential_lowering_host = _debye_lowering_np(
            state["electron_density_host"],
            state["thermal_energy_erg_host"],
            state["gas_pressure_host"],
        )
    else:
        potential_lowering_host = (
            potential_lowering.detach().to("cpu", torch.float64).numpy()
        )

    partition_state = dict(
        temperature=temperature,
        thermal_energy_ev=thermal_energy_ev,
        hc_over_kt=hc_over_kt,
        natural_log_temperature=natural_log_temperature,
        potential_lowering=potential_lowering,
        temperature_host=temperature_host,
        natural_log_temperature_host=natural_log_temperature_host,
        potential_lowering_host=potential_lowering_host,
    )

    partition_functions = torch.zeros(
        n_depths, n_elements, max_ion, dtype=dtype, device=device
    )
    partition_normalized_populations = torch.zeros(
        n_depths, n_elements, max_ion, dtype=dtype, device=device
    )
    ion_stage_fractions_over_partition = torch.zeros(
        n_depths, n_elements, max_ion, dtype=dtype, device=device
    )

    for atomic_number in range(1, n_elements + 1):
        requested_ion_count = int(ion_stage_count[atomic_number - 1])
        (
            partition_by_stage,
            ionization_potential_ev,
            potential_lowering_ev,
            element_ion_stage_count,
        ) = _build_partition_state_for_element(
            tables,
            atomic_number,
            requested_ion_count,
            partition_state,
        )
        fractions_over_partition = _saha_ladder(
            partition_by_stage,
            ionization_potential_ev,
            potential_lowering_ev,
            temperature,
            thermal_energy_ev,
            electron_density,
        )
        stored_ion_count = min(max_ion, element_ion_stage_count)
        partition_functions[:, atomic_number - 1, :stored_ion_count] = (
            partition_by_stage[:stored_ion_count].transpose(0, 1)
        )
        ion_stage_fractions_over_partition[:, atomic_number - 1, :stored_ion_count] = (
            fractions_over_partition[:stored_ion_count].transpose(0, 1)
        )
        partition_normalized_populations[:, atomic_number - 1, :stored_ion_count] = (
            fractions_over_partition[:stored_ion_count].transpose(0, 1)
            * (total_nuclei_number_density * elemental_abundances[atomic_number - 1])[
                None, :
            ].transpose(0, 1)
        )

    return EOSResult(
        partition_functions=partition_functions,
        partition_normalized_populations=partition_normalized_populations,
        ion_stage_fractions_over_partition=ion_stage_fractions_over_partition,
    )


def partition_functions_for_elements(
    temperature,
    gas_pressure,
    electron_density,
    *,
    tables: EOSTables,
    elements,
    nion: int = 1,
    apply_ground_partition: bool = True,
) -> dict[int, np.ndarray]:
    """Return partition functions for selected elements.

    The molecular-population bridge sometimes needs the same ion partitions
    without the low-temperature ground correction. Use
    `apply_ground_partition=False` only for that parity path.
    """
    state = derived_state(temperature, gas_pressure, electron_density, tables=tables)
    device = tables.device
    dtype = tables.dtype
    temperature = state["temperature"].to(device=device, dtype=dtype)
    thermal_energy_ev = state["thermal_energy_ev"].to(device=device, dtype=dtype)
    thermal_energy_erg = state["thermal_energy_erg"].to(device=device, dtype=dtype)
    hc_over_kt = state["hc_over_kt"].to(device=device, dtype=dtype)
    natural_log_temperature = state["natural_log_temperature"].to(
        device=device, dtype=dtype
    )
    gas_pressure = state["gas_pressure"].to(device=device, dtype=dtype)
    electron_density = state["electron_density"].to(device=device, dtype=dtype)
    potential_lowering = _debye_lowering(
        electron_density, thermal_energy_erg, gas_pressure
    )
    if "temperature_host" in state:
        temperature_host = np.asarray(state["temperature_host"], dtype=np.float64)
    else:
        temperature_host = temperature.detach().to("cpu", torch.float64).numpy()
    if "natural_log_temperature_host" in state:
        natural_log_temperature_host = np.asarray(
            state["natural_log_temperature_host"], dtype=np.float64
        )
    else:
        natural_log_temperature_host = (
            natural_log_temperature.detach().to("cpu", torch.float64).numpy()
        )
    if {"electron_density_host", "thermal_energy_erg_host", "gas_pressure_host"} <= set(
        state
    ):
        potential_lowering_host = _debye_lowering_np(
            state["electron_density_host"],
            state["thermal_energy_erg_host"],
            state["gas_pressure_host"],
        )
    else:
        potential_lowering_host = (
            potential_lowering.detach().to("cpu", torch.float64).numpy()
        )
    partition_state = dict(
        temperature=temperature,
        thermal_energy_ev=thermal_energy_ev,
        hc_over_kt=hc_over_kt,
        natural_log_temperature=natural_log_temperature,
        potential_lowering=potential_lowering,
        temperature_host=temperature_host,
        natural_log_temperature_host=natural_log_temperature_host,
        potential_lowering_host=potential_lowering_host,
    )
    partition_by_element: dict[int, np.ndarray] = {}
    for atomic_number in sorted(
        {int(element) for element in elements if 1 <= int(element) <= 99}
    ):
        (
            partition_by_stage,
            _ionization_energy_by_stage,
            _ionization_potential_by_stage,
            element_ion_stage_count,
        ) = _build_partition_state_for_element(
            tables,
            atomic_number,
            max(int(nion), 1),
            partition_state,
            apply_ground_partition=apply_ground_partition,
        )
        stored_ion_count = min(int(nion), int(element_ion_stage_count))
        partition_by_element[atomic_number] = (
            partition_by_stage[:stored_ion_count]
            .transpose(0, 1)
            .detach()
            .cpu()
            .double()
            .numpy()
        )
    return partition_by_element


# Self-consistent electron density from charge balance.


def _ion_stage_count_for_atomic_number(atomic_number: int) -> int:
    """Number of ion stages included in the charge-balance solve for one element."""
    if atomic_number == 1:
        return 2
    if atomic_number == 2:
        return 3
    if atomic_number in (3, 4, 5):
        return 4
    if 6 <= atomic_number <= 16:
        return 6
    if 17 <= atomic_number <= 28:
        return 5
    return 3


# The shared per-element ion-stage count is resolved once.
ION_STAGE_COUNT_BY_ATOMIC_NUMBER = np.array(
    [
        _ion_stage_count_for_atomic_number(atomic_number)
        for atomic_number in range(1, 100)
    ],
    dtype=np.int64,
)
_ION_CHARGE = np.arange(6, dtype=np.float64)  # charge of ion index 0..5


def derived_state(
    temperature, gas_pressure, electron_density, *, tables: EOSTables
) -> dict:
    """Build the depth-state dict consumed by `populations()`.

    Reproduces the derived thermal energy, inverse-temperature factor,
    temperature log, and neutral-atom density at the validated constants.
    Carries the fp64 host copies the discrete EOS bracket/gate lookups need, so
    the table-cell selection is dtype-independent. Inputs are depth vectors; the
    returned tensors live on ``tables``.
    """
    temperature = np.asarray(temperature, np.float64)
    gas_pressure = np.asarray(gas_pressure, np.float64)
    electron_density = np.asarray(electron_density, np.float64)
    thermal_energy_erg = temperature * REFERENCE_BOLTZMANN_ERG_PER_K
    thermal_energy_ev = REFERENCE_BOLTZMANN_EV_PER_K * temperature
    hc_over_kt = (REFERENCE_PLANCK_ERG_SECOND * LIGHT_SPEED_CM_PER_S) / np.maximum(
        thermal_energy_erg, FLOAT64_POSITIVE_FLOOR
    )
    natural_log_temperature = np.log(np.maximum(temperature, FLOAT64_POSITIVE_FLOOR))
    total_nuclei_number_density = np.maximum(
        gas_pressure / np.maximum(thermal_energy_erg, FLOAT64_POSITIVE_FLOOR)
        - electron_density,
        FLOAT64_POSITIVE_FLOOR,
    )

    def to_device_tensor(array):
        return torch.as_tensor(array, dtype=tables.dtype, device=tables.device)

    return dict(
        temperature=to_device_tensor(temperature),
        thermal_energy_ev=to_device_tensor(thermal_energy_ev),
        thermal_energy_erg=to_device_tensor(thermal_energy_erg),
        hc_over_kt=to_device_tensor(hc_over_kt),
        natural_log_temperature=to_device_tensor(natural_log_temperature),
        gas_pressure=to_device_tensor(gas_pressure),
        electron_density=to_device_tensor(electron_density),
        total_nuclei_number_density=to_device_tensor(total_nuclei_number_density),
        ion_stage_count=ION_STAGE_COUNT_BY_ATOMIC_NUMBER,
        temperature_host=temperature,
        natural_log_temperature_host=natural_log_temperature,
        electron_density_host=electron_density,
        thermal_energy_erg_host=thermal_energy_erg,
        gas_pressure_host=gas_pressure,
    )


def molecular_seed_electron_density(
    temperature,
    gas_pressure,
    electron_density,
) -> np.ndarray:
    """Depth-wise electron-density seed for molecular ion formation constants."""
    temperature = np.asarray(temperature, np.float64)
    gas_pressure = np.asarray(gas_pressure, np.float64)
    electron_density = np.asarray(electron_density, np.float64)
    thermal_energy_erg = temperature * REFERENCE_BOLTZMANN_ERG_PER_K
    seed = np.empty_like(electron_density, dtype=np.float64)
    if seed.size == 0:
        return seed
    seed[0] = (
        gas_pressure[0]
        / np.maximum(thermal_energy_erg[0], FLOAT64_POSITIVE_FLOOR)
        / 20.0
    )
    if seed.size > 1:
        pressure_ratio = gas_pressure[1:] / np.maximum(
            gas_pressure[:-1], FLOAT64_POSITIVE_FLOOR
        )
        seed[1:] = electron_density[:-1] * pressure_ratio
    return np.maximum(seed, FLOAT64_POSITIVE_FLOOR)


def molecular_ion_formation_constants_from_seed(
    temperature,
    gas_pressure,
    electron_density,
    *,
    tables: EOSTables,
    meta,
) -> np.ndarray:
    """Build molecular atomic-ion formation constants from the seed Saha state."""
    from . import molecular_equilibrium as _molecular_equilibrium

    seed_electron_density = molecular_seed_electron_density(
        temperature, gas_pressure, electron_density
    )
    state = derived_state(
        temperature, gas_pressure, seed_electron_density, tables=tables
    )
    state["elemental_abundances"] = torch.ones(
        99, dtype=tables.dtype, device=tables.device
    )
    eos_seed = populations(state, tables)
    ion_stage_fraction = (
        (eos_seed.ion_stage_fractions_over_partition * eos_seed.partition_functions)
        .detach()
        .cpu()
        .to(torch.float64)
        .numpy()
    )
    return _molecular_equilibrium.ion_formation_constants_from_saha(
        meta, ion_stage_fraction, seed_electron_density
    )


@dataclass
class ElectronDensityResult:
    """Converged charge balance + the full per-ion EOS state at the fixed point."""

    electron_density: np.ndarray
    total_nuclei_number_density: np.ndarray
    mass_density: np.ndarray
    eos: EOSResult


def _mass_density_from_composition(
    total_nuclei_number_density: np.ndarray,
    elemental_abundances: np.ndarray,
    mean_nuclear_mass_amu,
) -> np.ndarray:
    """Return mass density in g cm^-3 from a nuclei-density scale."""

    nuclei_density = np.asarray(total_nuclei_number_density, np.float64)
    if mean_nuclear_mass_amu is None:
        abundance_matrix = np.asarray(elemental_abundances, np.float64)
        if abundance_matrix.ndim == 1:
            abundance_matrix = np.broadcast_to(
                abundance_matrix,
                (nuclei_density.size, abundance_matrix.size),
            )
        elif abundance_matrix.ndim == 2 and abundance_matrix.shape[0] == 1:
            abundance_matrix = np.broadcast_to(
                abundance_matrix,
                (nuclei_density.size, abundance_matrix.shape[1]),
            )
        if (
            abundance_matrix.ndim != 2
            or abundance_matrix.shape[0] != nuclei_density.size
            or abundance_matrix.shape[1] < 99
        ):
            raise ValueError(
                "elemental_abundances must have shape (99,) or "
                "(n_depth, n_element >= 99)"
            )
        atomic_mass_path = Path(
            os.environ.get(
                "PAYNE_ZERO_SYNTHESIS_ATOMIC_MASS_TABLE",
                str(_runtime_paths.SYNTHESIS_TABLE_DIR / "atomic_masses.npz"),
            )
        )
        with np.load(atomic_mass_path, allow_pickle=False) as atomic_mass_table:
            atomic_masses_amu = np.asarray(
                atomic_mass_table["atomic_mass_amu"], np.float64
            )[:99]
        abundance_sum = np.sum(abundance_matrix[:, :99], axis=1)
        if np.any(~np.isfinite(abundance_sum)) or np.any(abundance_sum <= 0.0):
            raise ValueError(
                "elemental_abundances must have a finite positive sum to derive "
                "mass density"
            )
        mean_nuclear_mass_amu_array = (
            np.sum(abundance_matrix[:, :99] * atomic_masses_amu[None, :], axis=1)
            / abundance_sum
        )
    else:
        mean_nuclear_mass_amu_array = np.asarray(mean_nuclear_mass_amu, np.float64)
    if np.any(~np.isfinite(mean_nuclear_mass_amu_array)) or np.any(
        mean_nuclear_mass_amu_array <= 0.0
    ):
        raise ValueError("mean_nuclear_mass_amu must be finite and strictly positive")
    return nuclei_density * mean_nuclear_mass_amu_array * REFERENCE_ATOMIC_MASS_GRAM


def solve_electron_density(
    temperature,
    gas_pressure,
    elemental_abundances,
    *,
    tables: EOSTables,
    mean_nuclear_mass_amu=None,
    electron_density_seed=None,
    max_iter: int = 200,
    tol: float = 1e-4,
    molecules: bool = False,
    molecules_path=None,
) -> ElectronDensityResult:
    """Self-consistent electron density at fixed temperature and gas pressure."""
    temperature = np.asarray(temperature, np.float64)
    gas_pressure = np.asarray(gas_pressure, np.float64)
    n_depths = temperature.size
    thermal_energy_erg = temperature * REFERENCE_BOLTZMANN_ERG_PER_K
    total_particle_density = gas_pressure / np.maximum(
        thermal_energy_erg, FLOAT64_POSITIVE_FLOOR
    )

    abundance_matrix = np.asarray(elemental_abundances, np.float64)
    if abundance_matrix.ndim == 1:
        abundance_matrix = np.broadcast_to(abundance_matrix, (n_depths, 99))
    if electron_density_seed is None:
        electron_density_current = total_particle_density * 0.5
    else:
        electron_density_current = np.asarray(electron_density_seed, np.float64).copy()
        default_seed = total_particle_density * 0.5
        bad_seed = ~np.isfinite(electron_density_current) | (
            electron_density_current <= 0.0
        )
        if np.any(bad_seed):
            electron_density_current = np.where(
                bad_seed, default_seed, electron_density_current
            )

    ion_charge = _ION_CHARGE[None, None, :]
    unit_abundances = torch.ones(99, dtype=tables.dtype, device=tables.device)
    static_state = dict(
        temperature=torch.as_tensor(
            temperature, dtype=tables.dtype, device=tables.device
        ),
        thermal_energy_ev=torch.as_tensor(
            REFERENCE_BOLTZMANN_EV_PER_K * temperature,
            dtype=tables.dtype,
            device=tables.device,
        ),
        thermal_energy_erg=torch.as_tensor(
            thermal_energy_erg, dtype=tables.dtype, device=tables.device
        ),
        hc_over_kt=torch.as_tensor(
            (REFERENCE_PLANCK_ERG_SECOND * LIGHT_SPEED_CM_PER_S)
            / np.maximum(thermal_energy_erg, FLOAT64_POSITIVE_FLOOR),
            dtype=tables.dtype,
            device=tables.device,
        ),
        natural_log_temperature=torch.as_tensor(
            np.log(np.maximum(temperature, FLOAT64_POSITIVE_FLOOR)),
            dtype=tables.dtype,
            device=tables.device,
        ),
        gas_pressure=torch.as_tensor(
            gas_pressure, dtype=tables.dtype, device=tables.device
        ),
        elemental_abundances=unit_abundances,
        ion_stage_count=ION_STAGE_COUNT_BY_ATOMIC_NUMBER,
        temperature_host=temperature,
        natural_log_temperature_host=np.log(
            np.maximum(temperature, FLOAT64_POSITIVE_FLOOR)
        ),
        thermal_energy_erg_host=thermal_energy_erg,
        gas_pressure_host=gas_pressure,
    )

    def _state_for(
        electron_density: np.ndarray,
        total_nuclei_number_density: np.ndarray,
    ) -> dict:
        state = dict(static_state)
        state["electron_density"] = torch.as_tensor(
            electron_density, dtype=tables.dtype, device=tables.device
        )
        state["total_nuclei_number_density"] = torch.as_tensor(
            total_nuclei_number_density, dtype=tables.dtype, device=tables.device
        )
        state["electron_density_host"] = np.asarray(electron_density, np.float64)
        return state

    eos_result = None
    for iteration_index in range(max_iter):
        total_nuclei_number_density = np.maximum(
            total_particle_density - electron_density_current, FLOAT64_POSITIVE_FLOOR
        )
        state = _state_for(electron_density_current, total_nuclei_number_density)
        eos_result = populations(state, tables)
        ion_stage_fraction = (
            (
                eos_result.ion_stage_fractions_over_partition
                * eos_result.partition_functions
            )
            .detach()
            .cpu()
            .to(torch.float64)
            .numpy()
        )
        weighted = ion_stage_fraction * (
            total_nuclei_number_density[:, None, None] * abundance_matrix[:, :, None]
        )
        new_electron_density = (weighted * ion_charge).sum(axis=(1, 2))
        new_electron_density = np.maximum(
            new_electron_density, electron_density_current * 0.5
        )
        new_electron_density = 0.5 * (new_electron_density + electron_density_current)
        relative_error = np.abs(
            (electron_density_current - new_electron_density)
            / np.maximum(new_electron_density, FLOAT64_POSITIVE_FLOOR)
        )
        electron_density_current = new_electron_density
        if np.all(relative_error < tol):
            break
    else:
        raise RuntimeError(
            f"electron density did not converge in {max_iter} iterations"
        )

    total_nuclei_number_density_override = None
    if molecules:
        from . import molecular_equilibrium as _molecular_equilibrium

        abundance_vector = abundance_matrix[0]
        meta = _molecular_equilibrium.molecular_equilibrium_metadata(molecules_path)
        ion_formation_constants = molecular_ion_formation_constants_from_seed(
            temperature,
            gas_pressure,
            electron_density_current,
            tables=tables,
            meta=meta,
        )
        (
            total_nuclei_number_density_t,
            _molecular_populations,
            _molecular_equation_densities,
            electron_t,
        ) = _molecular_equilibrium.solve_molecular_equilibrium(
            temperature,
            gas_pressure,
            electron_density_current,
            abundance_vector,
            ion_formation_constants,
            molecules_path=molecules_path,
            device=tables.device,
            dtype=REFERENCE_DTYPE if tables.device.type != "mps" else tables.dtype,
            tol=tol,
        )
        electron_density_current = electron_t.detach().cpu().to(torch.float64).numpy()
        total_nuclei_number_density_molecular = (
            total_nuclei_number_density_t.detach().cpu().to(torch.float64).numpy()
        )
        total_nuclei_number_density_override = np.asarray(
            total_nuclei_number_density_molecular, np.float64
        )
        state = _state_for(
            electron_density_current, total_nuclei_number_density_override
        )
        eos_result = populations(state, tables)

    if total_nuclei_number_density_override is None:
        total_nuclei_number_density = np.maximum(
            total_particle_density - electron_density_current, FLOAT64_POSITIVE_FLOOR
        )
    else:
        total_nuclei_number_density = np.maximum(
            total_nuclei_number_density_override, FLOAT64_POSITIVE_FLOOR
        )
    mass_density = _mass_density_from_composition(
        total_nuclei_number_density,
        abundance_matrix,
        mean_nuclear_mass_amu,
    )
    return ElectronDensityResult(
        electron_density=electron_density_current,
        total_nuclei_number_density=total_nuclei_number_density,
        mass_density=mass_density,
        eos=eos_result,
    )


# Full population state consumed by continuum and line opacity.


@dataclass
class PopulationState:
    """Population bundle consumed by continuum and line-opacity kernels."""

    electron_density: np.ndarray
    total_nuclei_number_density: np.ndarray
    mass_density: np.ndarray
    partition_normalized_populations: np.ndarray
    ion_stage_populations: np.ndarray
    hydrogen_neutral_population: np.ndarray
    hydrogen_ionized_population: np.ndarray
    hydrogen_partition_normalized_ion_stage_populations: np.ndarray
    helium_neutral_population: np.ndarray
    helium_singly_ionized_population: np.ndarray
    carbon_partition_normalized_ion_stage_populations: np.ndarray
    magnesium_neutral_partition_normalized_population: np.ndarray
    aluminum_neutral_partition_normalized_population: np.ndarray
    silicon_neutral_partition_normalized_population: np.ndarray
    iron_neutral_partition_normalized_population: np.ndarray
    eos: EOSResult
    molecular_populations: Optional[np.ndarray] = None
    molecular_equation_densities: Optional[np.ndarray] = None


# The two schedules below deliberately differ in their fractional parts
# (ion-stage counts) for Cl, Ca, and the iron group: MODE12 lists the ion
# stages the EOS totals track, MODE11 the per-ion slots line opacity reads.
# This is not a copy-paste error; see _population_slot_maps.
_POPULATION_MODE12_CALLS: tuple[tuple[float, int], ...] = (
    (1.01, 1),
    (2.02, 3),
    (3.03, 6),
    (4.03, 10),
    (5.03, 15),
    (6.05, 21),
    (7.05, 28),
    (8.05, 36),
    (9.05, 45),
    (10.05, 55),
    (11.05, 66),
    (12.05, 78),
    (13.05, 91),
    (14.05, 105),
    (15.05, 120),
    (16.05, 136),
    (17.04, 153),
    (18.04, 171),
    (19.04, 190),
    (20.04, 210),
    (21.04, 231),
    (22.04, 253),
    (23.04, 276),
    (24.04, 300),
    (25.04, 325),
    (26.04, 351),
    (27.04, 378),
    (28.04, 406),
    (29.02, 435),
    (30.02, 465),
)
_POPULATION_MODE11_CALLS: tuple[tuple[float, int], ...] = (
    (1.01, 1),
    (2.02, 3),
    (3.03, 6),
    (4.03, 10),
    (5.03, 15),
    (6.05, 21),
    (7.05, 28),
    (8.05, 36),
    (9.05, 45),
    (10.05, 55),
    (11.05, 66),
    (12.05, 78),
    (13.05, 91),
    (14.05, 105),
    (15.05, 120),
    (16.05, 136),
    (17.05, 153),
    (18.04, 171),
    (19.05, 190),
    (20.09, 210),
    (21.09, 231),
    (22.09, 253),
    (23.09, 276),
    (24.09, 300),
    (25.09, 325),
    (26.09, 351),
    (27.09, 378),
    (28.09, 406),
    (29.02, 435),
    (30.02, 465),
)


def _population_output_count(population_code: float) -> int:
    fractional_part = float(population_code) - float(int(population_code))
    return max(1, int(fractional_part * 100.0 + 1.5))


def _population_slot_maps(max_slot: int = 1006) -> tuple[np.ndarray, np.ndarray]:
    atomic_number_by_slot = np.zeros(max_slot + 1, np.int64)
    ion_stage_by_slot = np.zeros(max_slot + 1, np.int64)
    # Line opacity follows the per-ion slot schedule, not the ion-stage total
    # schedule; the two differ for Cl, Ca, and iron-group higher stages.
    for population_code, start_slot_1based in _POPULATION_MODE11_CALLS:
        atomic_number = int(population_code)
        for ion_offset in range(_population_output_count(population_code)):
            slot = start_slot_1based + ion_offset
            if slot <= max_slot:
                atomic_number_by_slot[slot] = atomic_number
                ion_stage_by_slot[slot] = ion_offset + 1
    for atomic_number in range(31, 100):
        start_slot_1based = 496 + (atomic_number - 31) * 5
        for ion_offset in range(_population_output_count(float(atomic_number) + 0.02)):
            slot = start_slot_1based + ion_offset
            if slot <= max_slot:
                atomic_number_by_slot[slot] = atomic_number
                ion_stage_by_slot[slot] = ion_offset + 1
    return atomic_number_by_slot, ion_stage_by_slot


def _molecular_population_block(
    *,
    output_code: float,
    mode: int,
    output_count: int,
    molecule_codes: np.ndarray,
    molecular_line_populations: np.ndarray,
    molecular_populations: np.ndarray,
    partition_normalized_populations_by_atomic_number_and_stage: np.ndarray,
    ion_stage_populations_by_atomic_number_and_stage: np.ndarray,
) -> np.ndarray:
    n_depths = molecular_line_populations.shape[0]
    output_populations = np.zeros((n_depths, int(output_count)), dtype=np.float64)
    molecule_code_index = {
        int(round(float(molecule_code) * 100.0)): molecule_index
        for molecule_index, molecule_code in enumerate(molecule_codes)
    }
    molecule_families = {
        int(molecule_code)
        for molecule_code in molecule_codes
        if float(molecule_code) < 100.0
    }
    source_populations = (
        molecular_line_populations if mode in (1, 11) else molecular_populations
    )

    if output_code >= 100.0:
        molecule_index = molecule_code_index.get(int(round(float(output_code) * 100.0)))
        if molecule_index is not None:
            output_populations[:, 0] = source_populations[:, molecule_index]
        return output_populations

    atomic_number = int(output_code)
    n_output_stages = int(output_count) if mode in (11, 12) else 1
    current_code = float(output_code)
    fallback_to_eos = False
    for output_stage in range(1, n_output_stages + 1):
        ion_stage = n_output_stages - output_stage + 1
        molecule_index = molecule_code_index.get(int(round(current_code * 100.0)))
        if molecule_index is not None:
            output_populations[:, ion_stage - 1] = source_populations[:, molecule_index]
            current_code -= 0.01
            continue
        if atomic_number in molecule_families:
            current_code -= 0.01
            continue
        fallback_to_eos = True
        break

    if not fallback_to_eos:
        return output_populations

    if not (
        1
        <= atomic_number
        <= partition_normalized_populations_by_atomic_number_and_stage.shape[1]
    ):
        return output_populations
    fallback_populations = (
        partition_normalized_populations_by_atomic_number_and_stage
        if mode in (1, 11)
        else ion_stage_populations_by_atomic_number_and_stage
    )
    copy_count = min(output_populations.shape[1], fallback_populations.shape[2])
    output_populations[:, :copy_count] = fallback_populations[
        :, atomic_number - 1, :copy_count
    ]
    return output_populations


def _molecule_backed_population_state(
    *,
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    elemental_abundances: np.ndarray,
    electron_result: ElectronDensityResult,
    partition_normalized_populations_by_atomic_number_and_stage: np.ndarray,
    ion_stage_populations_by_atomic_number_and_stage: np.ndarray,
    tables: EOSTables,
    molecules_path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from . import molecular_equilibrium as _molecular_equilibrium

    abundance_vector = np.asarray(
        elemental_abundances[0]
        if elemental_abundances.ndim == 2
        else elemental_abundances,
        np.float64,
    )
    meta = _molecular_equilibrium.molecular_equilibrium_metadata(molecules_path)
    ion_formation_constants = molecular_ion_formation_constants_from_seed(
        temperature,
        gas_pressure,
        electron_result.electron_density,
        tables=tables,
        meta=meta,
    )
    (
        total_nuclei_number_density_t,
        molecular_populations_t,
        equation_densities_t,
        _electron,
    ) = _molecular_equilibrium.solve_molecular_equilibrium(
        temperature,
        gas_pressure,
        electron_result.electron_density,
        abundance_vector,
        ion_formation_constants,
        molecules_path=molecules_path,
        device=tables.device,
        dtype=tables.dtype,
        tol=1e-4,
    )
    molecular_total_nuclei_number_density = (
        total_nuclei_number_density_t.detach().cpu().double().numpy()
    )
    molecular_populations = molecular_populations_t.detach().cpu().double().numpy()
    molecular_equation_densities = equation_densities_t.detach().cpu().double().numpy()
    partition_functions = (
        electron_result.eos.partition_functions.detach().cpu().to(torch.float64).numpy()
    )
    (
        molecule_codes,
        molecular_line_populations,
    ) = _molecular_equilibrium.all_molecular_line_populations(
        temperature=temperature,
        equation_densities=molecular_equation_densities,
        molecular_populations=molecular_populations,
        neutral_partition=partition_functions[:, :, 0],
        partition_functions=partition_functions,
        molecules_path=molecules_path,
    )

    # Molecule-backed species use molecular populations. Other species keep the
    # EOS ion fractions but share the molecule-coupled nuclei-density scale.
    # This matters for elements with no molecular residual slot.
    nuclei_density_scale = molecular_total_nuclei_number_density / np.maximum(
        electron_result.total_nuclei_number_density, 1.0e-300
    )
    ion_population_fallback = (
        partition_normalized_populations_by_atomic_number_and_stage
        * nuclei_density_scale[:, None, None]
    )
    stage_population_fallback = (
        ion_stage_populations_by_atomic_number_and_stage
        * nuclei_density_scale[:, None, None]
    )

    n_depths = temperature.size
    stage_population_slots = np.zeros((n_depths, 1006), dtype=np.float64)
    per_ion_population_slots = np.zeros((n_depths, 1006), dtype=np.float64)

    for population_code, start_slot_1based in _POPULATION_MODE12_CALLS:
        output_count = _population_output_count(population_code)
        slot_start = start_slot_1based - 1
        slot_stop = slot_start + output_count
        stage_population_slots[:, slot_start:slot_stop] = _molecular_population_block(
            output_code=population_code,
            mode=12,
            output_count=output_count,
            molecule_codes=molecule_codes,
            molecular_line_populations=molecular_line_populations,
            molecular_populations=molecular_populations,
            partition_normalized_populations_by_atomic_number_and_stage=(
                ion_population_fallback
            ),
            ion_stage_populations_by_atomic_number_and_stage=stage_population_fallback,
        )
    for population_code, start_slot_1based in _POPULATION_MODE11_CALLS:
        output_count = _population_output_count(population_code)
        slot_start = start_slot_1based - 1
        slot_stop = slot_start + output_count
        per_ion_population_slots[:, slot_start:slot_stop] = _molecular_population_block(
            output_code=population_code,
            mode=11,
            output_count=output_count,
            molecule_codes=molecule_codes,
            molecular_line_populations=molecular_line_populations,
            molecular_populations=molecular_populations,
            partition_normalized_populations_by_atomic_number_and_stage=(
                ion_population_fallback
            ),
            ion_stage_populations_by_atomic_number_and_stage=stage_population_fallback,
        )
    for atomic_number in range(31, 100):
        output_code = float(atomic_number) + 0.02
        start_slot_1based = 496 + (atomic_number - 31) * 5
        output_count = _population_output_count(output_code)
        slot_start = start_slot_1based - 1
        slot_stop = slot_start + output_count
        per_ion_values = _molecular_population_block(
            output_code=output_code,
            mode=11,
            output_count=output_count,
            molecule_codes=molecule_codes,
            molecular_line_populations=molecular_line_populations,
            molecular_populations=molecular_populations,
            partition_normalized_populations_by_atomic_number_and_stage=(
                ion_population_fallback
            ),
            ion_stage_populations_by_atomic_number_and_stage=stage_population_fallback,
        )
        stage_values = _molecular_population_block(
            output_code=output_code,
            mode=12,
            output_count=output_count,
            molecule_codes=molecule_codes,
            molecular_line_populations=molecular_line_populations,
            molecular_populations=molecular_populations,
            partition_normalized_populations_by_atomic_number_and_stage=(
                ion_population_fallback
            ),
            ion_stage_populations_by_atomic_number_and_stage=stage_population_fallback,
        )
        per_ion_population_slots[:, slot_start:slot_stop] = per_ion_values
        stage_population_slots[:, slot_start:slot_stop] = stage_values

    partition_normalized_populations = np.zeros((n_depths, 6, 139), dtype=np.float64)
    atomic_number_by_slot, ion_stage_by_slot = _population_slot_maps()
    last_population_slot = min(
        per_ion_population_slots.shape[1],
        atomic_number_by_slot.size - 1,
    )
    for population_slot_index in range(1, last_population_slot + 1):
        atomic_number = int(atomic_number_by_slot[population_slot_index])
        ion_stage = int(ion_stage_by_slot[population_slot_index])
        if 1 <= atomic_number <= 99 and 1 <= ion_stage <= 6:
            partition_normalized_populations[:, ion_stage - 1, atomic_number - 1] = (
                per_ion_population_slots[:, population_slot_index - 1]
            )

    return (
        partition_normalized_populations,
        stage_population_slots,
        per_ion_population_slots,
        molecular_total_nuclei_number_density,
        molecular_populations,
        molecular_equation_densities,
    )


def _ion_stage_populations_from_packed_slots(
    stage_population_slots: np.ndarray,
) -> np.ndarray:
    """Map packed actual ion-stage totals to the public depth/stage/species cube."""

    n_depths = stage_population_slots.shape[0]
    ion_stage_populations = np.zeros((n_depths, 6, 139), dtype=np.float64)
    for population_code, start_slot_1based in _POPULATION_MODE12_CALLS:
        atomic_number = int(population_code)
        stage_count = min(_population_output_count(population_code), 6)
        slot_start = start_slot_1based - 1
        ion_stage_populations[:, :stage_count, atomic_number - 1] = (
            stage_population_slots[:, slot_start : slot_start + stage_count]
        )
    for atomic_number in range(31, 100):
        population_code = float(atomic_number) + 0.02
        stage_count = min(_population_output_count(population_code), 6)
        slot_start = 496 + (atomic_number - 31) * 5 - 1
        ion_stage_populations[:, :stage_count, atomic_number - 1] = (
            stage_population_slots[:, slot_start : slot_start + stage_count]
        )
    return ion_stage_populations


def solve_population_state(
    temperature,
    gas_pressure,
    elemental_abundances,
    *,
    tables: EOSTables,
    mean_nuclear_mass_amu=None,
    electron_density_seed=None,
    max_iter: int = 200,
    tol: float = 1e-4,
    molecules: bool = False,
    molecules_path=None,
) -> PopulationState:
    """Solve electron density and assemble the population state."""
    electron_result = solve_electron_density(
        temperature,
        gas_pressure,
        elemental_abundances,
        tables=tables,
        mean_nuclear_mass_amu=mean_nuclear_mass_amu,
        electron_density_seed=electron_density_seed,
        max_iter=max_iter,
        tol=tol,
        molecules=molecules,
        molecules_path=molecules_path,
    )
    return _assemble_population_state(
        temperature=temperature,
        gas_pressure=gas_pressure,
        elemental_abundances=elemental_abundances,
        electron_result=electron_result,
        tables=tables,
        molecules=molecules,
        molecules_path=molecules_path,
        use_molecular_mass_density=molecules,
    )


def solve_population_state_at_electron_density(
    temperature,
    gas_pressure,
    elemental_abundances,
    *,
    tables: EOSTables,
    electron_density,
    mean_nuclear_mass_amu=None,
    mass_density=None,
    molecules: bool = False,
    molecules_path=None,
) -> PopulationState:
    """Full-slot EOS fill at an already-solved electron density.

    This is the atmosphere-to-synthesis bridge path: the atmosphere solver has
    already saved electron density for the final atmosphere, so the bridge can
    build the same population struct at that fixed density without rerunning the
    charge-balance fixed-point loop.
    """
    temperature = np.asarray(temperature, np.float64)
    gas_pressure = np.asarray(gas_pressure, np.float64)
    electron_density = np.asarray(electron_density, np.float64)
    state = derived_state(temperature, gas_pressure, electron_density, tables=tables)
    state["elemental_abundances"] = torch.ones(
        99, dtype=tables.dtype, device=tables.device
    )
    eos_result = populations(state, tables)
    total_nuclei_number_density = np.maximum(
        gas_pressure
        / np.maximum(
            temperature * REFERENCE_BOLTZMANN_ERG_PER_K, FLOAT64_POSITIVE_FLOOR
        )
        - electron_density,
        FLOAT64_POSITIVE_FLOOR,
    )
    abundance_matrix = np.asarray(elemental_abundances, np.float64)
    if abundance_matrix.ndim == 1:
        abundance_matrix = np.broadcast_to(
            abundance_matrix,
            (temperature.size, abundance_matrix.size),
        )
    if mass_density is not None:
        mass_density_arr = np.asarray(mass_density, np.float64)
    else:
        mass_density_arr = _mass_density_from_composition(
            total_nuclei_number_density,
            abundance_matrix,
            mean_nuclear_mass_amu,
        )
    electron_result = ElectronDensityResult(
        electron_density=electron_density,
        total_nuclei_number_density=total_nuclei_number_density,
        mass_density=mass_density_arr,
        eos=eos_result,
    )
    return _assemble_population_state(
        temperature=temperature,
        gas_pressure=gas_pressure,
        elemental_abundances=elemental_abundances,
        electron_result=electron_result,
        tables=tables,
        molecules=molecules,
        molecules_path=molecules_path,
        use_molecular_mass_density=(molecules and mass_density is None),
    )


def _assemble_population_state(
    *,
    temperature,
    gas_pressure,
    elemental_abundances,
    electron_result: ElectronDensityResult,
    tables: EOSTables,
    molecules: bool = False,
    molecules_path=None,
    use_molecular_mass_density: bool = False,
) -> PopulationState:
    """Assemble synthesis populations from a solved/fixed electron state."""
    abundance_matrix = np.asarray(elemental_abundances, np.float64)
    n_depths = np.asarray(temperature).size
    if abundance_matrix.ndim == 1:
        abundance_matrix = np.broadcast_to(abundance_matrix, (n_depths, 99))

    # The fixed point used unit abundances; apply the requested abundance vector
    # to the ion-stage fractions when assembling synthesis populations.
    ion_stage_fraction = (
        electron_result.eos.ion_stage_fractions_over_partition.detach()
        .cpu()
        .to(torch.float64)
        .numpy()
    )
    partition_functions = (
        electron_result.eos.partition_functions.detach().cpu().to(torch.float64).numpy()
    )
    abundance_scale = (
        electron_result.total_nuclei_number_density[:, None] * abundance_matrix
    )[:, :, None]
    partition_normalized_populations_by_atomic_number_and_stage = (
        ion_stage_fraction * abundance_scale
    )
    ion_stage_populations_by_atomic_number_and_stage = (
        partition_normalized_populations_by_atomic_number_and_stage
        * partition_functions
    )

    partition_normalized_populations = np.zeros((n_depths, 6, 139), dtype=np.float64)
    partition_normalized_populations[:, :, :99] = (
        partition_normalized_populations_by_atomic_number_and_stage.transpose(0, 2, 1)
    )
    ion_stage_populations = np.zeros((n_depths, 6, 139), dtype=np.float64)
    ion_stage_populations[:, :, :99] = (
        ion_stage_populations_by_atomic_number_and_stage.transpose(0, 2, 1)
    )

    if molecules:
        (
            partition_normalized_populations,
            stage_population_slots,
            per_ion_population_slots,
            molecular_total_nuclei_number_density,
            molecular_populations,
            molecular_equation_densities,
        ) = _molecule_backed_population_state(
            temperature=np.asarray(temperature, np.float64),
            gas_pressure=np.asarray(gas_pressure, np.float64),
            elemental_abundances=abundance_matrix,
            electron_result=electron_result,
            partition_normalized_populations_by_atomic_number_and_stage=(
                partition_normalized_populations_by_atomic_number_and_stage
            ),
            ion_stage_populations_by_atomic_number_and_stage=(
                ion_stage_populations_by_atomic_number_and_stage
            ),
            tables=tables,
            molecules_path=molecules_path,
        )
        mass_density = electron_result.mass_density
        if use_molecular_mass_density:
            # Molecular equilibrium changes the nuclei density in cool atmospheres; use
            # that same density scale for rho so continuum opacity normalizes
            # like the reference converter.
            nuclei_density_scale = molecular_total_nuclei_number_density / np.maximum(
                electron_result.total_nuclei_number_density, 1.0e-300
            )
            mass_density = electron_result.mass_density * nuclei_density_scale
        ion_stage_populations = _ion_stage_populations_from_packed_slots(
            stage_population_slots
        )
        return PopulationState(
            electron_density=electron_result.electron_density,
            total_nuclei_number_density=molecular_total_nuclei_number_density,
            mass_density=mass_density,
            partition_normalized_populations=partition_normalized_populations,
            ion_stage_populations=ion_stage_populations,
            hydrogen_neutral_population=stage_population_slots[:, 0],
            hydrogen_ionized_population=stage_population_slots[:, 1],
            hydrogen_partition_normalized_ion_stage_populations=per_ion_population_slots[
                :, 0:2
            ].copy(),
            helium_neutral_population=stage_population_slots[:, 2],
            # Helium free-free opacity uses the ion-stage total, while level
            # populations use the partition-divided per-ion slots.
            helium_singly_ionized_population=stage_population_slots[:, 3],
            carbon_partition_normalized_ion_stage_populations=per_ion_population_slots[
                :, 20:22
            ].copy(),
            magnesium_neutral_partition_normalized_population=per_ion_population_slots[
                :, 77
            ],
            aluminum_neutral_partition_normalized_population=per_ion_population_slots[
                :, 90
            ],
            silicon_neutral_partition_normalized_population=per_ion_population_slots[
                :, 104
            ],
            iron_neutral_partition_normalized_population=per_ion_population_slots[
                :, 350
            ],
            eos=electron_result.eos,
            molecular_populations=molecular_populations,
            molecular_equation_densities=molecular_equation_densities,
        )

    return PopulationState(
        electron_density=electron_result.electron_density,
        total_nuclei_number_density=electron_result.total_nuclei_number_density,
        mass_density=electron_result.mass_density,
        partition_normalized_populations=partition_normalized_populations,
        ion_stage_populations=ion_stage_populations,
        hydrogen_neutral_population=ion_stage_populations_by_atomic_number_and_stage[
            :, 0, 0
        ],
        hydrogen_ionized_population=ion_stage_populations_by_atomic_number_and_stage[
            :, 0, 1
        ],
        hydrogen_partition_normalized_ion_stage_populations=(
            partition_normalized_populations_by_atomic_number_and_stage[
                :, 0, 0:2
            ].copy()
        ),
        helium_neutral_population=ion_stage_populations_by_atomic_number_and_stage[
            :, 1, 0
        ],
        helium_singly_ionized_population=(
            ion_stage_populations_by_atomic_number_and_stage[:, 1, 1]
        ),
        carbon_partition_normalized_ion_stage_populations=(
            partition_normalized_populations_by_atomic_number_and_stage[
                :, 5, 0:2
            ].copy()
        ),
        magnesium_neutral_partition_normalized_population=(
            partition_normalized_populations_by_atomic_number_and_stage[:, 11, 0]
        ),
        aluminum_neutral_partition_normalized_population=(
            partition_normalized_populations_by_atomic_number_and_stage[:, 12, 0]
        ),
        silicon_neutral_partition_normalized_population=(
            partition_normalized_populations_by_atomic_number_and_stage[:, 13, 0]
        ),
        iron_neutral_partition_normalized_population=(
            partition_normalized_populations_by_atomic_number_and_stage[:, 25, 0]
        ),
        eos=electron_result.eos,
    )


def load_state_from_inputs(
    path: Path = _DEFAULT_INPUTS, device=None, dtype=None
) -> dict:
    """Load the packaged EOS depth-state fixture onto a device."""
    if device is None:
        device = _device()
    if dtype is None:
        dtype = DEFAULT_DTYPE
    with np.load(path, allow_pickle=False) as data:
        tensor_fields = (
            "temperature",
            "thermal_energy_ev",
            "thermal_energy_erg",
            "hc_over_kt",
            "natural_log_temperature",
            "gas_pressure",
            "electron_density",
            "total_nuclei_number_density",
            "elemental_abundances",
        )
        state = {
            field_name: torch.as_tensor(data[field_name], dtype=dtype, device=device)
            for field_name in tensor_fields
        }
        state["ion_stage_count"] = np.asarray(data["ion_stage_count"], dtype=np.int64)
        # Host fp64 copies keep discrete table brackets independent of device dtype.
        state["temperature_host"] = np.asarray(data["temperature"], dtype=np.float64)
        state["natural_log_temperature_host"] = np.asarray(
            data["natural_log_temperature"], dtype=np.float64
        )
        state["electron_density_host"] = np.asarray(
            data["electron_density"], dtype=np.float64
        )
        state["thermal_energy_erg_host"] = np.asarray(
            data["thermal_energy_erg"], dtype=np.float64
        )
        state["gas_pressure_host"] = np.asarray(data["gas_pressure"], dtype=np.float64)
    return state


__all__ = [
    "EOSTables",
    "EOSResult",
    "populations",
    "load_state_from_inputs",
    "derived_state",
    "solve_electron_density",
    "ElectronDensityResult",
    "solve_population_state",
    "solve_population_state_at_electron_density",
    "PopulationState",
    "partition_functions_for_elements",
    "ION_STAGE_COUNT_BY_ATOMIC_NUMBER",
]
