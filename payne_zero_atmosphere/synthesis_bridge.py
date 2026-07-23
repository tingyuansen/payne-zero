"""Structured handoff from the atmosphere solver to spectrum synthesis.

The atmosphere iteration carries stage populations, partition-divided
populations, Doppler widths, and molecular densities. Spectrum synthesis
consumes the same physics through a public structured NPZ schema. This module
is the narrow exporter between those representations.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .atmosphere_io import ModelAtmosphere, linear_elemental_abundances
from .constants import (
    BOLTZMANN_ERG_PER_K_EXACT as BOLTZMANN_ERG_PER_K,
    LIGHT_SPEED_CM_PER_S_EXACT as LIGHT_SPEED_CM_PER_S,
    PLANCK_ERG_SECOND_EXACT as PLANCK_ERG_SECOND,
)
from .data_files import data_root

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

_MOLECULAR_SPECIES_CODE_TO_MOLECULE_CODES: dict[int, tuple[float, ...]] = {
    240: (101.0,),
    246: (106.0,),
    252: (107.0,),
    258: (108.0,),
    264: (606.0,),
    270: (607.0,),
    276: (608.0,),
    282: (707.0,),
    288: (708.0,),
    294: (808.0,),
    300: (112.0,),
    306: (113.0,),
    312: (114.0,),
    318: (812.0,),
    324: (813.0,),
    330: (814.0,),
    336: (116.0,),
    342: (120.0,),
    348: (816.0,),
    354: (820.0,),
    360: (821.0,),
    366: (822.0,),
    372: (823.0,),
    378: (103.0,),
    384: (104.0,),
    390: (105.0,),
    396: (109.0,),
    402: (115.0,),
    408: (117.0,),
    414: (121.0,),
    420: (122.0,),
    426: (123.0,),
    432: (124.0,),
    438: (125.0,),
    444: (126.0,),
    492: (111.0,),
    498: (119.0,),
    510: (817.0,),
    516: (824.0,),
    522: (825.0,),
    528: (826.0,),
    534: (10108.0,),
    540: (60808.0,),
    546: (10106.0,),
    552: (60606.0,),
    558: (127.0,),
    564: (128.0,),
    570: (129.0,),
    576: (827.0,),
    582: (828.0,),
    588: (829.0,),
    780: (839.0,),
    786: (840.0,),
    792: (857.0,),
}

# The sibling payne_zero_synthesis package sits next to this package at the
# workspace root; the shared data home holds its physics tables.
_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
_SYNTHESIS_SOURCE_ROOT = _WORKSPACE_ROOT
_DEFAULT_EDGE_GRID = data_root() / "synthesis_tables" / "continuum_edge_grid.npz"
_SYNTHESIS_SOURCE_CATALOG_ENV = "PAYNE_ZERO_SYNTHESIS_SOURCE_CATALOG_ROOT"
_SYNTHESIS_ATMOSPHERE_SCHEMA_VERSION = 4
_MINIMUM_REQUIRED_PACKED_POPULATION_SLOTS = 351


def _ensure_synthesis_import_path() -> None:
    """Make the sibling Payne Zero synthesis package importable from source trees."""

    if _SYNTHESIS_SOURCE_ROOT.exists() and str(_SYNTHESIS_SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(_SYNTHESIS_SOURCE_ROOT))


def _synthesis_population_slot_maps(n_slots):
    """Slot -> (element, ion stage) layout, shared with the synthesis package."""

    _ensure_synthesis_import_path()
    from payne_zero_synthesis.equation_of_state import (  # noqa: PLC0415
        _population_slot_maps,
    )

    return _population_slot_maps(n_slots)


@contextmanager
def _temporary_source_catalog_root(path: Path | None):
    if path is None:
        yield
        return
    previous = os.environ.get(_SYNTHESIS_SOURCE_CATALOG_ENV)
    os.environ[_SYNTHESIS_SOURCE_CATALOG_ENV] = str(Path(path).expanduser().resolve())
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_SYNTHESIS_SOURCE_CATALOG_ENV, None)
        else:
            os.environ[_SYNTHESIS_SOURCE_CATALOG_ENV] = previous


def infer_synthesis_source_catalog_root(molecules_path: Path | None) -> Path | None:
    """Infer a synthesis source-catalog root from a molecules file path."""

    if os.environ.get(_SYNTHESIS_SOURCE_CATALOG_ENV):
        return None
    if molecules_path is None:
        return None
    path = Path(molecules_path).expanduser().resolve()
    if path.parent.name != "lines":
        return None
    root = path.parent.parent
    if (root / "lines" / "molecular_equilibrium_synthesis.npz").exists():
        return root
    return None


def _edge_grid_path(path: Path | None = None) -> Path:
    # Production uses the bundled edge grid; callers may still pass an explicit
    # edge_grid_path through the public API.
    if path is not None:
        return Path(path).expanduser().resolve()
    return _DEFAULT_EDGE_GRID


def _load_edge_grid(path: Path | None = None) -> dict[str, np.ndarray]:
    edge_path = _edge_grid_path(path)
    if not edge_path.exists():
        raise FileNotFoundError(
            "continuum edge grid not found; pass edge_grid_path "
            f"(looked for {edge_path})"
        )
    with np.load(edge_path, allow_pickle=False) as edge_table:
        return {
            "signed_continuum_edge_frequency_hz": np.asarray(
                edge_table["signed_continuum_edge_frequency_hz"], np.float64
            ),
            "continuum_edge_wavelength_nm": np.asarray(
                edge_table["continuum_edge_wavelength_nm"], np.float64
            ),
            "continuum_edge_midpoint_wavelength_nm": np.asarray(
                edge_table["continuum_edge_midpoint_wavelength_nm"], np.float64
            ),
            "continuum_edge_interval_width_squared_over_two_nm2": np.asarray(
                edge_table["continuum_edge_interval_width_squared_over_two_nm2"],
                np.float64,
            ),
        }


def _collapse_abundance_vector(elemental_abundances: np.ndarray) -> np.ndarray:
    abundances = np.asarray(elemental_abundances, np.float64)
    if abundances.ndim == 1:
        if abundances.size < 99:
            raise ValueError("elemental_abundances must contain at least 99 values")
        return abundances[:99].copy()
    if abundances.ndim != 2 or abundances.shape[1] < 99:
        raise ValueError("elemental_abundances must have shape (99,) or (n_depth, 99)")
    if not np.allclose(abundances[:, :99], abundances[0:1, :99], rtol=0.0, atol=0.0):
        raise ValueError(
            "layer-dependent abundances are not representable in the structured "
            "synthesis schema"
        )
    return abundances[0, :99].copy()


def _packed_atomic_cube(
    *,
    ion_stage_populations_by_packed_slot: np.ndarray,
    partition_normalized_populations_by_packed_slot: np.ndarray,
    fractional_doppler_widths_by_packed_slot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ion_stage_populations_by_packed_slot = np.asarray(
        ion_stage_populations_by_packed_slot,
        np.float64,
    )
    partition_normalized_populations_by_packed_slot = np.asarray(
        partition_normalized_populations_by_packed_slot,
        np.float64,
    )
    fractional_doppler_widths_by_packed_slot = np.asarray(
        fractional_doppler_widths_by_packed_slot, np.float64
    )
    if (
        ion_stage_populations_by_packed_slot.ndim != 2
        or partition_normalized_populations_by_packed_slot.ndim != 2
        or fractional_doppler_widths_by_packed_slot.ndim != 2
    ):
        raise ValueError(
            "packed populations and doppler widths must be two-dimensional"
        )
    if not (
        ion_stage_populations_by_packed_slot.shape
        == partition_normalized_populations_by_packed_slot.shape
        == fractional_doppler_widths_by_packed_slot.shape
    ):
        raise ValueError(
            "packed populations and doppler widths must have the same shape"
        )

    n_depths, n_slots = ion_stage_populations_by_packed_slot.shape
    ion_stage_populations = np.zeros((n_depths, 6, 139), dtype=np.float64)
    partition_normalized_populations = np.zeros((n_depths, 6, 139), dtype=np.float64)
    fractional_doppler_widths = np.zeros((n_depths, 6, 139), dtype=np.float64)
    atomic_number_by_slot, ion_stage_by_slot = _synthesis_population_slot_maps(n_slots)
    for slot_1based in range(1, min(n_slots, atomic_number_by_slot.size - 1) + 1):
        atomic_number = int(atomic_number_by_slot[slot_1based])
        ion_stage = int(ion_stage_by_slot[slot_1based])
        if 1 <= atomic_number <= 99 and 1 <= ion_stage <= 6:
            ion_stage_populations[:, ion_stage - 1, atomic_number - 1] = (
                ion_stage_populations_by_packed_slot[:, slot_1based - 1]
            )
            partition_normalized_populations[:, ion_stage - 1, atomic_number - 1] = (
                partition_normalized_populations_by_packed_slot[:, slot_1based - 1]
            )
            fractional_doppler_widths[:, ion_stage - 1, atomic_number - 1] = (
                fractional_doppler_widths_by_packed_slot[:, slot_1based - 1]
            )
    return (
        ion_stage_populations,
        partition_normalized_populations,
        fractional_doppler_widths,
    )


def _molecule_code_key(molecule_code: float) -> int:
    return int(round(float(molecule_code) * 100.0))


def _molecular_population_by_code(
    *,
    molecule_codes: np.ndarray | None,
    populations: np.ndarray | None,
    molecule_code: float,
    n_depths: int,
) -> np.ndarray:
    if molecule_codes is None or populations is None:
        return np.zeros(n_depths, dtype=np.float64)
    codes = np.asarray(molecule_codes, np.float64).ravel()
    values = np.asarray(populations, np.float64)
    if values.ndim != 2 or values.shape[0] != n_depths:
        raise ValueError(
            "molecular population arrays must have shape (n_depth, n_molecule)"
        )
    if codes.size != values.shape[1]:
        raise ValueError(
            "molecule_codes and molecular populations must have the same "
            "n_molecule dimension"
        )
    key = _molecule_code_key(molecule_code)
    matches = np.array([_molecule_code_key(code) == key for code in codes])
    if not np.any(matches):
        return np.zeros(n_depths, dtype=np.float64)
    return np.sum(values[:, matches], axis=1)


def _fill_molecular_species_slots(
    *,
    partition_normalized_populations: np.ndarray,
    molecule_codes: np.ndarray | None,
    partition_normalized_molecular_populations: np.ndarray | None,
) -> None:
    n_depths = partition_normalized_populations.shape[0]
    for (
        species_code,
        molecule_code_group,
    ) in _MOLECULAR_SPECIES_CODE_TO_MOLECULE_CODES.items():
        population_column = int(species_code) // 6 - 1
        if not (0 <= population_column < partition_normalized_populations.shape[2]):
            continue
        population = np.zeros(n_depths, dtype=np.float64)
        for molecule_code in molecule_code_group:
            population += _molecular_population_by_code(
                molecule_codes=molecule_codes,
                populations=partition_normalized_molecular_populations,
                molecule_code=molecule_code,
                n_depths=n_depths,
            )
        if np.any(population > 0.0):
            partition_normalized_populations[:, 5, population_column] = population


def _molecular_hydrogen_population(
    *,
    temperature: np.ndarray,
    ion_stage_populations_by_packed_slot: np.ndarray,
    molecule_codes: np.ndarray | None,
    molecular_populations: np.ndarray | None,
) -> np.ndarray:
    n_depths = temperature.size
    from_molecules = _molecular_population_by_code(
        molecule_codes=molecule_codes,
        populations=molecular_populations,
        molecule_code=101.0,
        n_depths=n_depths,
    )
    if np.any(from_molecules > 0.0):
        return from_molecules

    neutral_hydrogen = np.asarray(
        ion_stage_populations_by_packed_slot[:, 0], np.float64
    )
    thermal_energy_ev = np.asarray(temperature, np.float64) * 8.617333262e-5
    natural_log_temperature = np.log(np.maximum(temperature, 1.0e-300))
    with np.errstate(over="ignore"):
        equilibrium_factor = np.exp(
            4.478 / np.maximum(thermal_energy_ev, 1.0e-300)
            - 46.4584
            + (
                1.63660e-3
                + (
                    -4.93992e-7
                    + (
                        1.11822e-10
                        + (
                            -1.49567e-14
                            + (1.06206e-18 - 3.08720e-23 * temperature) * temperature
                        )
                        * temperature
                    )
                    * temperature
                )
                * temperature
            )
            * temperature
            - 1.5 * natural_log_temperature
        )
    molecular_hydrogen = neutral_hydrogen**2 * equilibrium_factor
    molecular_hydrogen[temperature > 9000.0] = 0.0
    return molecular_hydrogen


def structured_atmosphere_from_packed_state(
    *,
    temperature: np.ndarray,
    column_mass: np.ndarray,
    gas_pressure: np.ndarray,
    electron_density: np.ndarray,
    mass_density: np.ndarray,
    microturbulence: np.ndarray,
    ion_stage_populations_by_packed_slot: np.ndarray,
    partition_normalized_populations_by_packed_slot: np.ndarray,
    fractional_doppler_widths_by_packed_slot: np.ndarray,
    elemental_abundances: np.ndarray,
    molecule_codes: np.ndarray | None = None,
    molecular_populations: np.ndarray | None = None,
    partition_normalized_molecular_populations: np.ndarray | None = None,
    edge_grid_path: Path | None = None,
) -> dict[str, np.ndarray]:
    """Return a structured synthesis atmosphere from packed physical arrays."""

    temperature = np.asarray(temperature, np.float64)
    ion_stage_populations_by_packed_slot = np.asarray(
        ion_stage_populations_by_packed_slot, np.float64
    )
    partition_normalized_populations_by_packed_slot = np.asarray(
        partition_normalized_populations_by_packed_slot,
        np.float64,
    )
    fractional_doppler_widths_by_packed_slot = np.asarray(
        fractional_doppler_widths_by_packed_slot, np.float64
    )
    if temperature.ndim != 1 or temperature.size < 2:
        raise ValueError(
            "temperature must have shape (n_depth,) with at least two layers"
        )
    n_depths = temperature.size
    depth_columns = {
        "column_mass": column_mass,
        "gas_pressure": gas_pressure,
        "electron_density": electron_density,
        "mass_density": mass_density,
        "microturbulence": microturbulence,
    }
    bad_depth_columns = [
        name
        for name, values in depth_columns.items()
        if np.asarray(values).shape != (n_depths,)
    ]
    if bad_depth_columns:
        raise ValueError(
            "packed-state depth columns must have shape (n_depth,): "
            + ", ".join(bad_depth_columns)
        )
    if (
        ion_stage_populations_by_packed_slot.shape
        != partition_normalized_populations_by_packed_slot.shape
    ):
        raise ValueError(
            "ion-stage and partition-normalized population slots must have the same shape"
        )
    if (
        ion_stage_populations_by_packed_slot.shape
        != fractional_doppler_widths_by_packed_slot.shape
    ):
        raise ValueError(
            "ion-stage population and Doppler slots must have the same shape"
        )
    if ion_stage_populations_by_packed_slot.ndim != 2:
        raise ValueError("packed population arrays must have shape (n_depth, n_slot)")
    if ion_stage_populations_by_packed_slot.shape[0] != n_depths:
        raise ValueError(
            "packed population arrays must share the temperature depth axis"
        )
    if (
        ion_stage_populations_by_packed_slot.shape[1]
        < _MINIMUM_REQUIRED_PACKED_POPULATION_SLOTS
    ):
        raise ValueError(
            "packed population arrays must contain at least 351 solver slots"
        )

    codes = None if molecule_codes is None else np.asarray(molecule_codes, np.float64)
    if codes is not None and codes.ndim != 1:
        raise ValueError("molecule_codes must have shape (n_molecule,)")
    for name, values in (
        ("molecular_populations", molecular_populations),
        (
            "partition_normalized_molecular_populations",
            partition_normalized_molecular_populations,
        ),
    ):
        if values is None:
            continue
        if codes is None:
            raise ValueError(f"{name} requires molecule_codes")
        shape = np.asarray(values).shape
        if shape != (n_depths, codes.size):
            raise ValueError(
                f"{name} must have shape (n_depth, n_molecule) matching molecule_codes"
            )

    (
        ion_stage_populations,
        partition_normalized_populations,
        fractional_doppler_widths,
    ) = _packed_atomic_cube(
        ion_stage_populations_by_packed_slot=ion_stage_populations_by_packed_slot,
        partition_normalized_populations_by_packed_slot=(
            partition_normalized_populations_by_packed_slot
        ),
        fractional_doppler_widths_by_packed_slot=(
            fractional_doppler_widths_by_packed_slot
        ),
    )
    _fill_molecular_species_slots(
        partition_normalized_populations=partition_normalized_populations,
        molecule_codes=molecule_codes,
        partition_normalized_molecular_populations=partition_normalized_molecular_populations,
    )

    edge_grid = _load_edge_grid(edge_grid_path)
    hc_over_kt = (
        PLANCK_ERG_SECOND
        * LIGHT_SPEED_CM_PER_S
        / (BOLTZMANN_ERG_PER_K * np.maximum(temperature, 1.0e-300))
    )

    structured = {
        "atmosphere_schema_version": np.asarray(
            [_SYNTHESIS_ATMOSPHERE_SCHEMA_VERSION],
            dtype=np.int32,
        ),
        "temperature": temperature,
        "gas_pressure": np.asarray(gas_pressure, np.float64),
        "electron_density": np.asarray(electron_density, np.float64),
        "mass_density": np.asarray(mass_density, np.float64),
        "column_mass": np.asarray(column_mass, np.float64),
        "ion_stage_populations": ion_stage_populations,
        "partition_normalized_populations": partition_normalized_populations,
        "fractional_doppler_widths": fractional_doppler_widths,
        "hydrogen_neutral_population": ion_stage_populations_by_packed_slot[
            :, 0
        ].copy(),
        "helium_neutral_population": ion_stage_populations_by_packed_slot[:, 2].copy(),
        "helium_singly_ionized_population": ion_stage_populations_by_packed_slot[
            :, 3
        ].copy(),
        "molecular_hydrogen_population": _molecular_hydrogen_population(
            temperature=temperature,
            ion_stage_populations_by_packed_slot=ion_stage_populations_by_packed_slot,
            molecule_codes=molecule_codes,
            molecular_populations=molecular_populations,
        ),
        "hydrogen_partition_normalized_ion_stage_populations": (
            partition_normalized_populations_by_packed_slot[:, 0:2].copy()
        ),
        "carbon_partition_normalized_ion_stage_populations": (
            partition_normalized_populations_by_packed_slot[:, 20:22].copy()
        ),
        "magnesium_neutral_partition_normalized_population": (
            partition_normalized_populations_by_packed_slot[:, 77].copy()
        ),
        "aluminum_neutral_partition_normalized_population": (
            partition_normalized_populations_by_packed_slot[:, 90].copy()
        ),
        "silicon_neutral_partition_normalized_population": (
            partition_normalized_populations_by_packed_slot[:, 104].copy()
        ),
        "iron_neutral_partition_normalized_population": (
            partition_normalized_populations_by_packed_slot[:, 350].copy()
        ),
        "hydrogen_ionized_population": ion_stage_populations_by_packed_slot[
            :, 1
        ].copy(),
        "hc_over_kt": hc_over_kt,
        "microturbulence": np.asarray(microturbulence, np.float64),
        "elemental_abundances": _collapse_abundance_vector(elemental_abundances),
    }
    structured.update(edge_grid)
    return structured


def structured_atmosphere_from_runtime_state(
    *,
    atmosphere: Any,
    runtime_state: Any,
    molecular_state: Any | None = None,
    edge_grid_path: Path | None = None,
) -> dict[str, np.ndarray]:
    """Build a structured atmosphere from live atmosphere-runtime objects."""

    molecule_codes = None
    molecular_populations = None
    partition_normalized_molecular_populations = None
    if molecular_state is not None:
        molecule_codes = np.asarray(molecular_state.catalog.molecule_codes, np.float64)
        molecular_populations = np.asarray(
            molecular_state.molecular_populations, np.float64
        )
        partition_normalized_molecular_populations = np.asarray(
            molecular_state.partition_normalized_molecular_populations,
            np.float64,
        )

    return structured_atmosphere_from_packed_state(
        temperature=atmosphere.temperature,
        column_mass=atmosphere.column_mass,
        gas_pressure=runtime_state.gas_pressure,
        electron_density=runtime_state.electron_density,
        mass_density=runtime_state.mass_density,
        microturbulence=atmosphere.microturbulence,
        ion_stage_populations_by_packed_slot=runtime_state.ion_stage_populations_by_packed_slot,
        partition_normalized_populations_by_packed_slot=(
            runtime_state.partition_normalized_populations_by_packed_slot
        ),
        fractional_doppler_widths_by_packed_slot=runtime_state.fractional_doppler_widths,
        elemental_abundances=runtime_state.elemental_abundances_by_layer,
        molecule_codes=molecule_codes,
        molecular_populations=molecular_populations,
        partition_normalized_molecular_populations=partition_normalized_molecular_populations,
        edge_grid_path=edge_grid_path,
    )


def structured_atmosphere_from_debug_npz(
    debug_npz: Path,
    *,
    edge_grid_path: Path | None = None,
) -> dict[str, np.ndarray]:
    """Build a structured atmosphere from a physically named debug snapshot."""

    source_path = Path(debug_npz).expanduser().resolve()
    with np.load(source_path, allow_pickle=False) as data:
        if "debug_schema_version" not in data:
            raise ValueError(f"{source_path} uses an unsupported legacy debug schema")
        stored_schema_version = np.asarray(data["debug_schema_version"])
        if stored_schema_version.size != 1 or not np.issubdtype(
            stored_schema_version.dtype, np.integer
        ):
            raise ValueError(
                f"{source_path} debug_schema_version must contain exactly one integer"
            )
        schema_version = int(stored_schema_version.reshape(-1)[0])
        if schema_version != 4:
            raise ValueError(
                f"{source_path} uses unsupported debug schema version "
                f"{schema_version}; expected 4"
            )
        microturbulence = np.asarray(data["microturbulence"], np.float64)

        molecule_codes = (
            np.asarray(data["molecule_codes"], np.float64)
            if "molecule_codes" in data
            else None
        )
        molecular_populations = (
            np.asarray(data["molecular_populations"], np.float64)
            if "molecular_populations" in data
            else None
        )
        partition_normalized_molecular_populations = (
            np.asarray(
                data["partition_normalized_molecular_populations"],
                np.float64,
            )
            if "partition_normalized_molecular_populations" in data
            else None
        )

        return structured_atmosphere_from_packed_state(
            temperature=np.asarray(data["temperature"], np.float64),
            column_mass=np.asarray(data["column_mass"], np.float64),
            gas_pressure=np.asarray(data["gas_pressure"], np.float64),
            electron_density=np.asarray(data["electron_density"], np.float64),
            mass_density=np.asarray(data["mass_density"], np.float64),
            microturbulence=microturbulence,
            ion_stage_populations_by_packed_slot=np.asarray(
                data["ion_stage_populations_by_packed_slot"], np.float64
            ),
            partition_normalized_populations_by_packed_slot=np.asarray(
                data["partition_normalized_populations_by_packed_slot"],
                np.float64,
            ),
            fractional_doppler_widths_by_packed_slot=np.asarray(
                data["fractional_doppler_widths"], np.float64
            ),
            elemental_abundances=np.asarray(
                data["elemental_abundances_by_layer"], np.float64
            ),
            molecule_codes=molecule_codes,
            molecular_populations=molecular_populations,
            partition_normalized_molecular_populations=partition_normalized_molecular_populations,
            edge_grid_path=edge_grid_path,
        )


def save_structured_atmosphere(
    atmosphere: dict[str, np.ndarray],
    output_npz: Path,
) -> Path:
    """Write a structured atmosphere NPZ."""

    output_path = Path(output_npz).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path, **{key: np.asarray(value) for key, value in atmosphere.items()}
    )
    return output_path


def save_product_structured_atmosphere(
    atmosphere: ModelAtmosphere,
    output_npz: Path,
    *,
    source_catalog_root: Path | None = None,
    molecular_lines: bool = True,
    device: str = "cpu",
    dtype: str = "float64",
) -> Path:
    """Write the synthesis product from final fixed-column-quantized arrays.

    This is the release handoff path. It intentionally uses the same
    column-based synthesis builder as the product spectrum gate instead of
    exporting live atmosphere-runtime populations.
    """

    _ensure_synthesis_import_path()
    from payne_zero_synthesis import (  # noqa: PLC0415
        build_structured_atmosphere,
        save_structured_atmosphere as save_synthesis_structured_atmosphere,
    )

    with _temporary_source_catalog_root(source_catalog_root):
        structured = build_structured_atmosphere(
            temperature=atmosphere.temperature,
            column_mass=atmosphere.column_mass,
            gas_pressure=atmosphere.gas_pressure,
            electron_density=atmosphere.electron_density,
            elemental_abundances=linear_elemental_abundances(atmosphere),
            microturbulence=atmosphere.microturbulence,
            molecular_lines=bool(molecular_lines),
            device=device,
            dtype=dtype,
        )
    save_synthesis_structured_atmosphere(structured, output_npz)
    return Path(output_npz).expanduser().resolve()


def save_structured_atmosphere_from_runtime_state(
    output_npz: Path,
    *,
    atmosphere: Any,
    runtime_state: Any,
    molecular_state: Any | None = None,
    edge_grid_path: Path | None = None,
) -> Path:
    """Write a structured synthesis atmosphere from live runtime state."""

    structured = structured_atmosphere_from_runtime_state(
        atmosphere=atmosphere,
        runtime_state=runtime_state,
        molecular_state=molecular_state,
        edge_grid_path=edge_grid_path,
    )
    return save_structured_atmosphere(structured, output_npz)


def save_structured_atmosphere_from_debug_npz(
    debug_npz: Path,
    output_npz: Path,
    *,
    edge_grid_path: Path | None = None,
) -> Path:
    """Write a structured synthesis atmosphere from a debug snapshot."""

    structured = structured_atmosphere_from_debug_npz(
        debug_npz,
        edge_grid_path=edge_grid_path,
    )
    return save_structured_atmosphere(structured, output_npz)
