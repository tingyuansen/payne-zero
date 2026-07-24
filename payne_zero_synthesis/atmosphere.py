"""Structured-atmosphere input schema.

`load_atmosphere_npz` validates and loads the native NPZ the atmosphere
solver writes (schema: atmosphere_schema.json); REQUIRED_ATMOSPHERE_ARRAYS
is the contract both packages share.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ATMOSPHERE_SCHEMA_VERSION = 4
ATMOSPHERE_PRODUCT_METADATA_SCHEMA_VERSION = 1
LEGACY_ATMOSPHERE_SCHEMA_VERSIONS = (1, 2, 3)
POPULATION_ION_STAGE_COUNT = 6
POPULATION_SPECIES_COUNT = 139

ATMOSPHERE_PRODUCT_METADATA_FIELDS = (
    "atmosphere_product_metadata_schema",
    "atmosphere_product_role",
    "atmosphere_converged",
    "atmosphere_closure_required",
    "initializer_family",
    "atmosphere_metadata_json",
)

REQUIRED_ATMOSPHERE_ARRAYS = (
    "temperature",
    "gas_pressure",
    "electron_density",
    "mass_density",
    "column_mass",
    "partition_normalized_populations",
    "ion_stage_populations",
    "fractional_doppler_widths",
    "hydrogen_neutral_population",
    "helium_neutral_population",
    "helium_singly_ionized_population",
    "molecular_hydrogen_population",
    "hydrogen_partition_normalized_ion_stage_populations",
    "carbon_partition_normalized_ion_stage_populations",
    "magnesium_neutral_partition_normalized_population",
    "aluminum_neutral_partition_normalized_population",
    "silicon_neutral_partition_normalized_population",
    "iron_neutral_partition_normalized_population",
    "hc_over_kt",
    "microturbulence",
    "elemental_abundances",
    "signed_continuum_edge_frequency_hz",
    "continuum_edge_wavelength_nm",
    "continuum_edge_midpoint_wavelength_nm",
    "continuum_edge_interval_width_squared_over_two_nm2",
)

LEGACY_ATMOSPHERE_ARRAY_ALIASES = {
    "fractional_doppler_widths": "doppler_widths",
    "partition_normalized_populations": "ion_populations",
    "helium_singly_ionized_population": "helium_ionized_population",
    "hydrogen_partition_normalized_ion_stage_populations": (
        "hydrogen_level_populations"
    ),
    "carbon_partition_normalized_ion_stage_populations": ("carbon_level_populations"),
    "magnesium_neutral_partition_normalized_population": (
        "magnesium_neutral_population"
    ),
    "aluminum_neutral_partition_normalized_population": ("aluminum_neutral_population"),
    "silicon_neutral_partition_normalized_population": ("silicon_neutral_population"),
    "iron_neutral_partition_normalized_population": "iron_neutral_population",
    "signed_continuum_edge_frequency_hz": "continuum_edge_frequency",
    "continuum_edge_wavelength_nm": "continuum_edge_wavelength",
    "continuum_edge_midpoint_wavelength_nm": "continuum_edge_midpoint",
    "continuum_edge_interval_width_squared_over_two_nm2": (
        "continuum_edge_delta_squared"
    ),
}


def _stored_array_name(
    data: np.lib.npyio.NpzFile,
    public_name: str,
    *,
    allow_legacy_aliases: bool,
) -> str | None:
    if public_name in data:
        return public_name
    if not allow_legacy_aliases:
        return None
    legacy_name = LEGACY_ATMOSPHERE_ARRAY_ALIASES.get(public_name)
    if legacy_name in data:
        return legacy_name
    return None


def _read_public_array(
    data: np.lib.npyio.NpzFile,
    public_name: str,
    *,
    allow_legacy_aliases: bool,
) -> np.ndarray:
    stored_name = _stored_array_name(
        data,
        public_name,
        allow_legacy_aliases=allow_legacy_aliases,
    )
    if stored_name is not None:
        return np.asarray(data[stored_name])
    raise KeyError(public_name)


def _read_schema_version(data: np.lib.npyio.NpzFile, path: Path) -> int | None:
    if "atmosphere_schema_version" not in data:
        return None
    stored = np.asarray(data["atmosphere_schema_version"])
    if stored.size != 1 or not np.issubdtype(stored.dtype, np.integer):
        raise ValueError(
            f"{path} atmosphere_schema_version must contain exactly one integer"
        )
    return int(stored.reshape(-1)[0])


def _read_scalar_product_field(
    data: np.lib.npyio.NpzFile,
    path: Path,
    name: str,
    *,
    kind: str,
) -> object:
    stored = np.asarray(data[name])
    if stored.size != 1 or stored.dtype.kind not in kind:
        raise ValueError(
            f"{path} {name} must contain exactly one value of dtype kind {kind}"
        )
    return stored.reshape(-1)[0].item()


def load_atmosphere_product_metadata(
    path: str | Path,
) -> dict[str, object] | None:
    """Load optional self-identifying metadata from an atmosphere archive.

    Historical and physically converged schema-v4 products remain valid
    without this extension and return ``None``. An initialized-atmosphere
    archive written by :class:`InitializedAtmosphere` carries the complete
    extension so it cannot be mistaken for a converged physical solution.
    """

    atmosphere_path = Path(path).expanduser()
    if not atmosphere_path.exists():
        raise FileNotFoundError(f"atmosphere file does not exist: {atmosphere_path}")

    with np.load(atmosphere_path, allow_pickle=False) as data:
        present = [
            name for name in ATMOSPHERE_PRODUCT_METADATA_FIELDS if name in data
        ]
        if not present:
            return None
        missing = [
            name for name in ATMOSPHERE_PRODUCT_METADATA_FIELDS if name not in data
        ]
        if missing:
            raise ValueError(
                f"{atmosphere_path} has an incomplete atmosphere-product "
                "metadata extension: " + ", ".join(missing)
            )

        schema = int(
            _read_scalar_product_field(
                data,
                atmosphere_path,
                "atmosphere_product_metadata_schema",
                kind="iu",
            )
        )
        if schema != ATMOSPHERE_PRODUCT_METADATA_SCHEMA_VERSION:
            raise ValueError(
                f"{atmosphere_path} uses unsupported atmosphere-product "
                f"metadata schema {schema}; supported version is "
                f"{ATMOSPHERE_PRODUCT_METADATA_SCHEMA_VERSION}"
            )
        role = str(
            _read_scalar_product_field(
                data,
                atmosphere_path,
                "atmosphere_product_role",
                kind="SU",
            )
        )
        converged = bool(
            _read_scalar_product_field(
                data,
                atmosphere_path,
                "atmosphere_converged",
                kind="b",
            )
        )
        closure_required = bool(
            _read_scalar_product_field(
                data,
                atmosphere_path,
                "atmosphere_closure_required",
                kind="b",
            )
        )
        initializer_family = str(
            _read_scalar_product_field(
                data,
                atmosphere_path,
                "initializer_family",
                kind="SU",
            )
        )
        metadata_text = str(
            _read_scalar_product_field(
                data,
                atmosphere_path,
                "atmosphere_metadata_json",
                kind="SU",
            )
        )

    try:
        detailed = json.loads(metadata_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{atmosphere_path} atmosphere_metadata_json is not valid JSON"
        ) from error
    if not isinstance(detailed, dict):
        raise ValueError(
            f"{atmosphere_path} atmosphere_metadata_json must contain one object"
        )
    if (
        detailed.get("atmosphere_product_role") != role
        or detailed.get("initializer_family") != initializer_family
        or detailed.get("atmosphere_converged") is not converged
        or detailed.get("atmosphere_closure_required") is not closure_required
    ):
        raise ValueError(
            f"{atmosphere_path} typed atmosphere-product fields disagree with "
            "atmosphere_metadata_json"
        )
    return {
        "schema": schema,
        "atmosphere_product_role": role,
        "atmosphere_converged": converged,
        "atmosphere_closure_required": closure_required,
        "initializer_family": initializer_family,
        "labels": detailed.get("labels"),
        "provenance": detailed.get("provenance"),
        "timings": detailed.get("timings"),
    }


def load_atmosphere_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load a structured atmosphere and return canonical public field names."""

    atmosphere_path = Path(path).expanduser()
    if not atmosphere_path.exists():
        raise FileNotFoundError(f"atmosphere file does not exist: {atmosphere_path}")

    with np.load(atmosphere_path, allow_pickle=False) as data:
        schema_version = _read_schema_version(data, atmosphere_path)
        if schema_version is not None:
            supported_versions = (
                *LEGACY_ATMOSPHERE_SCHEMA_VERSIONS,
                ATMOSPHERE_SCHEMA_VERSION,
            )
            if schema_version not in supported_versions:
                raise ValueError(
                    f"{atmosphere_path} uses unsupported atmosphere schema version "
                    f"{schema_version}; supported versions are "
                    + ", ".join(str(version) for version in supported_versions)
                )
        required_stored_arrays = REQUIRED_ATMOSPHERE_ARRAYS
        allow_legacy_aliases = schema_version in (
            None,
            *LEGACY_ATMOSPHERE_SCHEMA_VERSIONS,
        )
        if not allow_legacy_aliases:
            legacy_names = sorted(
                legacy_name
                for legacy_name in LEGACY_ATMOSPHERE_ARRAY_ALIASES.values()
                if legacy_name in data
            )
            if legacy_names:
                raise ValueError(
                    f"{atmosphere_path} schema {schema_version} contains legacy "
                    "array aliases: " + ", ".join(legacy_names)
                )
        if schema_version in (None, 1, 2):
            required_stored_arrays = tuple(
                name
                for name in REQUIRED_ATMOSPHERE_ARRAYS
                if name != "ion_stage_populations"
            )
        missing = [
            public_name
            for public_name in required_stored_arrays
            if _stored_array_name(
                data,
                public_name,
                allow_legacy_aliases=allow_legacy_aliases,
            )
            is None
        ]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"{atmosphere_path} is missing required arrays: {joined}")

        atmosphere = {
            public_name: _read_public_array(
                data,
                public_name,
                allow_legacy_aliases=allow_legacy_aliases,
            )
            for public_name in required_stored_arrays
        }
        if "ion_stage_populations" in data:
            atmosphere["ion_stage_populations"] = np.asarray(
                data["ion_stage_populations"]
            )
        if "hydrogen_ionized_population" in data:
            atmosphere["hydrogen_ionized_population"] = np.asarray(
                data["hydrogen_ionized_population"]
            )
    if "ion_stage_populations" not in atmosphere:
        _validate_loaded_atmosphere(
            atmosphere,
            atmosphere_path,
            require_ion_stage_populations=False,
        )
        atmosphere["ion_stage_populations"] = _reconstruct_ion_stage_populations(
            atmosphere
        )
    _validate_loaded_atmosphere(atmosphere, atmosphere_path)
    return atmosphere


def _reconstruct_ion_stage_populations(
    atmosphere: dict[str, np.ndarray],
) -> np.ndarray:
    """Upgrade pre-v3 products with actual Saha ion-stage number densities."""

    import torch

    from . import equation_of_state

    tables = equation_of_state.EOSTables.from_npz(
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    state = equation_of_state.derived_state(
        atmosphere["temperature"],
        atmosphere["gas_pressure"],
        atmosphere["electron_density"],
        tables=tables,
    )
    state["elemental_abundances"] = torch.as_tensor(
        np.asarray(atmosphere["elemental_abundances"], np.float64),
        dtype=tables.dtype,
        device=tables.device,
    )
    eos = equation_of_state.populations(state, tables)
    partition_functions = (
        eos.partition_functions.detach()
        .cpu()
        .to(torch.float64)
        .numpy()
        .transpose(0, 2, 1)
    )
    stored_partition_normalized = np.asarray(
        atmosphere["partition_normalized_populations"], np.float64
    )
    stored_shape = stored_partition_normalized.shape
    ion_stage_populations = np.zeros(stored_shape, dtype=np.float64)
    stage_count = min(stored_shape[1], partition_functions.shape[1])
    species_count = min(stored_shape[2], partition_functions.shape[2])
    ion_stage_populations[:, :stage_count, :species_count] = (
        stored_partition_normalized[:, :stage_count, :species_count]
        * partition_functions[:, :stage_count, :species_count]
    )
    return ion_stage_populations


def _validate_loaded_atmosphere(
    atmosphere: dict[str, np.ndarray],
    path: Path,
    *,
    require_ion_stage_populations: bool = True,
) -> None:
    for name, values in atmosphere.items():
        array = np.asarray(values)
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"{path} array {name} must have a numeric dtype")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{path} array {name} contains non-finite values")

    temperature = np.asarray(atmosphere["temperature"])
    if temperature.ndim != 1 or temperature.size < 2:
        raise ValueError(
            "temperature must have shape (n_depth,) with at least two layers"
        )
    depth_count = temperature.shape[0]
    depth_columns = (
        "gas_pressure",
        "electron_density",
        "mass_density",
        "column_mass",
        "hydrogen_neutral_population",
        "helium_neutral_population",
        "helium_singly_ionized_population",
        "molecular_hydrogen_population",
        "magnesium_neutral_partition_normalized_population",
        "aluminum_neutral_partition_normalized_population",
        "silicon_neutral_partition_normalized_population",
        "iron_neutral_partition_normalized_population",
        "hc_over_kt",
        "microturbulence",
    )
    bad_depth_arrays = [
        name
        for name in depth_columns
        if np.asarray(atmosphere[name]).shape != (depth_count,)
    ]
    if "hydrogen_ionized_population" in atmosphere and np.asarray(
        atmosphere["hydrogen_ionized_population"]
    ).shape != (depth_count,):
        bad_depth_arrays.append("hydrogen_ionized_population")
    if bad_depth_arrays:
        joined = ", ".join(bad_depth_arrays)
        raise ValueError(f"{path} has inconsistent depth axis for arrays: {joined}")

    positive_columns = (
        "temperature",
        "gas_pressure",
        "electron_density",
        "mass_density",
        "column_mass",
        "hc_over_kt",
    )
    for name in positive_columns:
        if np.any(np.asarray(atmosphere[name]) <= 0.0):
            raise ValueError(f"{path} array {name} must be strictly positive")
    if np.any(np.asarray(atmosphere["microturbulence"]) < 0.0):
        raise ValueError(f"{path} array microturbulence must be non-negative")
    column_mass = np.asarray(atmosphere["column_mass"])
    if np.any(np.diff(column_mass) <= 0.0):
        raise ValueError(f"{path} column_mass must be strictly increasing")

    population_shape = np.asarray(atmosphere["partition_normalized_populations"]).shape
    ion_stage_population_shape = (
        np.asarray(atmosphere["ion_stage_populations"]).shape
        if "ion_stage_populations" in atmosphere
        else None
    )
    doppler_shape = np.asarray(atmosphere["fractional_doppler_widths"]).shape
    if (
        population_shape != doppler_shape
        or (
            require_ion_stage_populations
            and population_shape != ion_stage_population_shape
        )
        or population_shape
        != (depth_count, POPULATION_ION_STAGE_COUNT, POPULATION_SPECIES_COUNT)
    ):
        raise ValueError(
            "partition_normalized_populations, ion_stage_populations, and "
            "fractional_doppler_widths must all have shape (n_depth, 6, 139)"
        )

    for name in (
        "hydrogen_partition_normalized_ion_stage_populations",
        "carbon_partition_normalized_ion_stage_populations",
    ):
        shape = np.asarray(atmosphere[name]).shape
        if shape != (depth_count, 2):
            raise ValueError(f"{name} must have shape (n_depth, 2)")

    nonnegative_columns = (
        "partition_normalized_populations",
        "fractional_doppler_widths",
        "hydrogen_neutral_population",
        "helium_neutral_population",
        "helium_singly_ionized_population",
        "molecular_hydrogen_population",
        "hydrogen_partition_normalized_ion_stage_populations",
        "carbon_partition_normalized_ion_stage_populations",
        "magnesium_neutral_partition_normalized_population",
        "aluminum_neutral_partition_normalized_population",
        "silicon_neutral_partition_normalized_population",
        "iron_neutral_partition_normalized_population",
    )
    if "ion_stage_populations" in atmosphere:
        nonnegative_columns += ("ion_stage_populations",)
    if "hydrogen_ionized_population" in atmosphere:
        nonnegative_columns += ("hydrogen_ionized_population",)
    for name in nonnegative_columns:
        if np.any(np.asarray(atmosphere[name]) < 0.0):
            raise ValueError(f"{path} array {name} must be non-negative")

    abundances = np.asarray(atmosphere["elemental_abundances"])
    if abundances.ndim != 1 or abundances.shape[0] < 99:
        raise ValueError("elemental_abundances must have shape (n_element >= 99,)")
    if np.any(abundances <= 0.0):
        raise ValueError("elemental_abundances must contain positive number fractions")

    edge_frequency = np.asarray(atmosphere["signed_continuum_edge_frequency_hz"])
    edge_wavelength = np.asarray(atmosphere["continuum_edge_wavelength_nm"])
    edge_midpoint = np.asarray(atmosphere["continuum_edge_midpoint_wavelength_nm"])
    edge_interval_width_squared_over_two = np.asarray(
        atmosphere["continuum_edge_interval_width_squared_over_two_nm2"]
    )
    if (
        edge_frequency.ndim != 1
        or edge_wavelength.ndim != 1
        or edge_frequency.shape != edge_wavelength.shape
        or edge_frequency.size < 2
    ):
        raise ValueError(
            "continuum edge frequency and wavelength must share shape (n_edge >= 2,)"
        )
    expected_intervals = edge_frequency.size - 1
    if edge_midpoint.shape != (
        expected_intervals,
    ) or edge_interval_width_squared_over_two.shape != (expected_intervals,):
        raise ValueError(
            "continuum edge midpoint and delta-squared arrays must have shape (n_edge - 1,)"
        )
    if np.any(edge_frequency == 0.0):
        raise ValueError(
            "signed_continuum_edge_frequency_hz must contain non-zero values"
        )
    for name, values in (
        ("continuum_edge_wavelength_nm", edge_wavelength),
        ("continuum_edge_midpoint_wavelength_nm", edge_midpoint),
        (
            "continuum_edge_interval_width_squared_over_two_nm2",
            edge_interval_width_squared_over_two,
        ),
    ):
        if np.any(values <= 0.0):
            raise ValueError(f"{name} must contain strictly positive values")


def validate_atmosphere_npz(path: str | Path) -> tuple[str, ...]:
    """Validate a structured atmosphere NPZ and return public field names."""

    load_atmosphere_npz(path)
    return REQUIRED_ATMOSPHERE_ARRAYS
