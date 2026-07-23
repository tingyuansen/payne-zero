"""Initial atmosphere state before EOS, opacity, and transfer steps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import ATOMIC_MASS_GRAM_REFERENCE
from .data_files import atmosphere_table_path

from .atmosphere_io import ModelAtmosphere


ION_STAGE_SLOTS = 1006
_ISOTOPE_TABLE_PATH = atmosphere_table_path("isotope_tables.npz")


# Rounded reference atomic masses in amu. Keep this table separate from exact
# CODATA masses; changing it changes the validated equation-of-state result.
REFERENCE_ATOMIC_MASS_AMU = np.array(
    [
        1.008,
        4.003,
        6.939,
        9.013,
        10.81,
        12.01,
        14.01,
        16.00,
        19.00,
        20.18,
        22.99,
        24.31,
        26.98,
        28.09,
        30.98,
        32.07,
        35.45,
        39.95,
        39.10,
        40.08,
        44.96,
        47.90,
        50.94,
        52.00,
        54.94,
        55.85,
        58.94,
        58.71,
        63.55,
        65.37,
        69.72,
        72.60,
        74.92,
        78.96,
        79.91,
        83.80,
        85.48,
        87.63,
        88.91,
        91.22,
        92.91,
        95.95,
        99.00,
        101.1,
        102.9,
        106.4,
        107.9,
        112.4,
        114.8,
        118.7,
        121.8,
        127.6,
        126.9,
        131.3,
        132.9,
        137.4,
        138.9,
        140.1,
        140.9,
        144.3,
        147.0,
        150.4,
        152.0,
        157.3,
        158.9,
        162.5,
        164.9,
        167.3,
        168.9,
        173.0,
        175.0,
        178.5,
        181.0,
        183.9,
        186.3,
        190.2,
        192.2,
        195.1,
        197.0,
        200.6,
        204.4,
        207.2,
        209.0,
        210.0,
        211.0,
        222.0,
        223.0,
        226.1,
        227.1,
        232.0,
        231.0,
        238.0,
        237.0,
        244.0,
        243.0,
        247.0,
        247.0,
        251.0,
        254.0,
    ],
    dtype=np.float64,
)


@dataclass
class AtmosphereRuntimeState:
    """Atmosphere iteration arrays with readable field names.

    This is the pykurucz-faithful pre-iteration seed.  Later physics modules
    should consume these fields directly instead of rebuilding their own COMMON
    state.
    """

    gas_pressure: np.ndarray
    electron_density: np.ndarray
    total_nuclei_number_density: np.ndarray
    mass_density: np.ndarray
    charge_square_density: np.ndarray
    elemental_abundances_by_layer: np.ndarray
    mean_nuclear_mass_amu: np.ndarray
    ion_stage_populations_by_packed_slot: np.ndarray
    partition_normalized_populations_by_packed_slot: np.ndarray
    specific_internal_energy: np.ndarray
    major_isotope_mass_amu: np.ndarray
    fractional_doppler_widths: np.ndarray | None = None
    partition_normalized_population_over_mass_density_and_fractional_doppler_width: (
        np.ndarray | None
    ) = None
    hydrogen_departure_coefficients: np.ndarray | None = None
    metal_departure_coefficients: np.ndarray | None = None
    geometric_depth_below_surface_km: np.ndarray | None = None


def load_major_isotope_masses_amu(path: Path | None = None) -> np.ndarray:
    """Load the reference major-isotope mass vector."""

    table_path = _ISOTOPE_TABLE_PATH if path is None else Path(path)
    with np.load(table_path, allow_pickle=False) as data:
        return np.asarray(data["major_isotope_mass_amu"], dtype=np.float64)


def build_elemental_abundances_by_layer(atmosphere: ModelAtmosphere) -> np.ndarray:
    """Build linear elemental number fractions from deck abundance values."""

    elemental_abundances_by_layer = np.full(
        (atmosphere.layers, 99), 1.0e-30, dtype=np.float64
    )
    elemental_abundances_by_layer[:, 0] = 0.92
    elemental_abundances_by_layer[:, 1] = 0.08
    for atomic_number, value in atmosphere.fixed_column_abundance_values.items():
        if not (1 <= atomic_number <= 99):
            continue
        linear_abundance = float(value) if atomic_number <= 2 else 10.0 ** float(value)
        elemental_abundances_by_layer[:, atomic_number - 1] = max(
            linear_abundance, 1.0e-30
        )
    return elemental_abundances_by_layer


def compute_mean_nuclear_mass_amu(
    elemental_abundances_by_layer: np.ndarray,
) -> np.ndarray:
    """Return mean mass per nucleus in atomic mass units."""

    elemental_abundances = np.asarray(elemental_abundances_by_layer, dtype=np.float64)
    return np.sum(elemental_abundances * REFERENCE_ATOMIC_MASS_AMU[None, :], axis=1)


def build_runtime_state(atmosphere: ModelAtmosphere) -> AtmosphereRuntimeState:
    """Construct the pykurucz-faithful runtime seed from an atmosphere deck."""

    elemental_abundances = build_elemental_abundances_by_layer(atmosphere)
    mean_nuclear_mass_amu = compute_mean_nuclear_mass_amu(elemental_abundances)
    thermal_energy_erg = atmosphere.thermal_energy_erg
    total_particle_density = atmosphere.gas_pressure / np.maximum(
        thermal_energy_erg, 1.0e-300
    )
    total_nuclei_number_density = total_particle_density - atmosphere.electron_density
    mass_density = (
        total_nuclei_number_density * mean_nuclear_mass_amu * ATOMIC_MASS_GRAM_REFERENCE
    )
    layers = atmosphere.layers

    return AtmosphereRuntimeState(
        gas_pressure=np.asarray(atmosphere.gas_pressure, dtype=np.float64).copy(),
        electron_density=np.asarray(
            atmosphere.electron_density, dtype=np.float64
        ).copy(),
        total_nuclei_number_density=np.asarray(
            total_nuclei_number_density, dtype=np.float64
        ),
        mass_density=np.asarray(mass_density, dtype=np.float64),
        charge_square_density=np.maximum(2.0 * atmosphere.electron_density, 1.0e-30),
        elemental_abundances_by_layer=elemental_abundances,
        mean_nuclear_mass_amu=mean_nuclear_mass_amu,
        ion_stage_populations_by_packed_slot=np.zeros(
            (layers, ION_STAGE_SLOTS), dtype=np.float64
        ),
        partition_normalized_populations_by_packed_slot=np.zeros(
            (layers, ION_STAGE_SLOTS), dtype=np.float64
        ),
        specific_internal_energy=np.zeros(layers, dtype=np.float64),
        major_isotope_mass_amu=load_major_isotope_masses_amu(),
        hydrogen_departure_coefficients=np.ones((layers, 6), dtype=np.float64),
        metal_departure_coefficients=np.ones(layers, dtype=np.float64),
    )


def update_charge_square_density(
    *,
    thermal_energy_erg: np.ndarray,
    state: AtmosphereRuntimeState,
) -> np.ndarray:
    """Seed the charge-square density before population iteration."""

    thermal_energy = np.asarray(thermal_energy_erg, dtype=np.float64)
    electron_density = np.asarray(state.electron_density, dtype=np.float64)
    gas_pressure = np.asarray(state.gas_pressure, dtype=np.float64)

    charge_square_density = 2.0 * electron_density
    excess = 2.0 * electron_density - gas_pressure / np.maximum(
        thermal_energy, 1.0e-300
    )
    charge_square_density = charge_square_density.copy()
    charge_square_density[excess > 0.0] += 2.0 * excess[excess > 0.0]
    state.charge_square_density[:] = charge_square_density
    return state.charge_square_density
