"""Portable calibrated atomic parameters and explicit catalog substitution."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path
import struct
from typing import Literal, Mapping

import numpy as np


ATOMIC_CALIBRATION_SCHEMA_VERSION = 4
LEGACY_PORTABLE_ATOMIC_CALIBRATION_SCHEMA_VERSION = 3
LEGACY_ATOMIC_CALIBRATION_SCHEMA_VERSION = 2
ATOMIC_CALIBRATION_SIGNATURE_FIELDS = (
    "wavelength_nm",
    "log_oscillator_strength",
    "lower_excitation_cm",
    "raw_radiative_damping_log",
    "raw_stark_damping_log",
    "raw_van_der_waals_damping_log",
    "ion_stage",
    "atomic_number",
    "species_code",
    "line_size",
    "line_type",
    "lower_principal_quantum_number",
    "upper_principal_quantum_number",
)
_INTEGER_SIGNATURE_FIELDS = {
    "ion_stage",
    "atomic_number",
    "line_size",
    "line_type",
    "lower_principal_quantum_number",
    "upper_principal_quantum_number",
}
_DEX_CORRECTIONS = {
    "loggf": "delta_loggf_dex",
    "vdw": "delta_log_vdw_dex",
    "radiative": "delta_log_radiative_dex",
    "stark": "delta_log_stark_dex",
}
_HASHED_COMPONENT_SIGNATURE_FIELD = "component_row_signature_sha256"
_HASHED_COMPONENT_ORDINAL_FIELD = "component_occurrence_ordinal"
_HASHED_SIGNATURE_DOMAIN = b"payne-zero-atomic-row-signature-v1\0"
_LEGACY_CORRECTIONS = {
    "loggf": "values",
    "vdw": "damping_values",
    "radiative": "radiative_values",
    "stark": "stark_values",
}

# These optional schema-3 fields materialize the final, per-component values.
# Older schema-3 overlays remain valid correction-only products.
ATOMIC_CALIBRATION_ABSOLUTE_FIELDS = {
    "loggf": "component_calibrated_log_oscillator_strength",
    "radiative": "component_calibrated_radiative_damping",
    "stark": "component_calibrated_stark_damping",
    "vdw": "component_calibrated_van_der_waals_damping",
}

_BUNDLED_CALIBRATIONS = {
    "sun_fts_hband": "sun_fts_hband.npz",
    "arcturus_fts_hband_joint_epochs": "arcturus_fts_hband_joint_epochs.npz",
    "sun_arcturus_fts_hband_shared": "sun_arcturus_fts_hband_shared.npz",
}


def _scalar_string(value: np.ndarray, *, field: str) -> str:
    if value.shape != () or value.dtype.kind not in {"S", "U"}:
        raise ValueError(f"atomic calibration {field} must be a scalar string")
    return str(value.item())


def _scalar_sha256(value: np.ndarray, *, field: str) -> str:
    digest = _scalar_string(value, field=field)
    if len(digest) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in digest
    ):
        raise ValueError(
            f"atomic calibration {field} must be a 64-character hexadecimal SHA-256"
        )
    return digest.lower()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_row_signature(
    arrays: Mapping[str, np.ndarray], index: int
) -> str:
    """Hash one parsed catalog row without publishing its source values.

    Version 1 serializes integer fields as signed little-endian int64 values
    and all other identity fields as little-endian IEEE-754 float64 values.
    Field names and type tags are included in the digest, making the identity
    independent of NumPy dtype width and host byte order.
    """

    digest = hashlib.sha256(_HASHED_SIGNATURE_DOMAIN)
    for name in ATOMIC_CALIBRATION_SIGNATURE_FIELDS:
        value = np.asarray(arrays[name])[index].item()
        digest.update(name.encode("ascii") + b"\0")
        if name in _INTEGER_SIGNATURE_FIELDS:
            digest.update(b"i" + int(value).to_bytes(8, "little", signed=True))
        else:
            floating = float(value)
            if not np.isfinite(floating):
                raise ValueError(
                    f"atomic catalog signature field {name} must be finite"
                )
            digest.update(b"f" + struct.pack("<d", floating))
    return digest.hexdigest()


def canonical_atomic_row_identities(
    arrays: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return opaque row hashes and zero-based duplicate ordinals.

    ``arrays`` must contain every name in
    ``ATOMIC_CALIBRATION_SIGNATURE_FIELDS`` as same-length one-dimensional
    arrays. Returned signature and ordinal vectors have that row length.

    The ordinal is counted in the supplied catalog order among rows with the
    same canonical hash. Together these two arrays identify duplicate source
    rows without exposing their catalog values.
    """

    missing = set(ATOMIC_CALIBRATION_SIGNATURE_FIELDS).difference(arrays)
    if missing:
        raise ValueError(
            "atomic catalog lacks signature fields: " + ", ".join(sorted(missing))
        )
    row_count = int(np.asarray(arrays["wavelength_nm"]).size)
    for name in ATOMIC_CALIBRATION_SIGNATURE_FIELDS:
        if np.asarray(arrays[name]).shape != (row_count,):
            raise ValueError(f"atomic catalog {name} must be a row vector")
    signatures = np.asarray(
        [_canonical_row_signature(arrays, index) for index in range(row_count)],
        dtype="U64",
    )
    seen: dict[str, int] = {}
    ordinals = np.empty(row_count, dtype=np.int64)
    for index, signature in enumerate(signatures.tolist()):
        ordinals[index] = seen.get(signature, 0)
        seen[signature] = int(ordinals[index]) + 1
    return signatures, ordinals


def _validate_sha256_vector(value: np.ndarray, *, field: str) -> np.ndarray:
    if value.ndim != 1 or value.dtype.kind not in {"S", "U"}:
        raise ValueError(f"atomic calibration {field} must be a string vector")
    raw = [str(digest) for digest in value.tolist()]
    if any(
        len(digest) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in digest)
        for digest in raw
    ):
        raise ValueError(
            f"atomic calibration {field} entries must be hexadecimal SHA-256 values"
        )
    return np.asarray([digest.lower() for digest in raw], dtype="U64")


def _row_signature(arrays: Mapping[str, np.ndarray], index: int) -> tuple[object, ...]:
    """Return the immutable source identity of one atomic component."""

    return tuple(
        np.asarray(arrays[name])[index].item()
        for name in ATOMIC_CALIBRATION_SIGNATURE_FIELDS
    )


def load_atomic_calibration(
    calibration_path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Load and validate a portable standard-star atomic-parameter product.

    Returns the stored arrays plus an inventory dictionary. The four normalized
    ``_delta_*_dex`` vectors added to the array dictionary are always group
    length, regardless of the readable on-disk schema version.

    Schema 4 is the correction-only public format. It stores opaque source-row
    identities and requires an exact source-catalog digest when applied.
    Schemas 2, 3, and the earlier unversioned layout remain readable for
    backward compatibility.
    """

    path = Path(calibration_path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}

    raw_schema = arrays.get("schema")
    if raw_schema is None:
        schema_version: int | None = None
        correction_fields = _LEGACY_CORRECTIONS
    else:
        if raw_schema.shape != () or not np.issubdtype(raw_schema.dtype, np.integer):
            raise ValueError("atomic calibration schema must be an integer scalar")
        schema_version = int(raw_schema.item())
        if schema_version in {
            ATOMIC_CALIBRATION_SCHEMA_VERSION,
            LEGACY_PORTABLE_ATOMIC_CALIBRATION_SCHEMA_VERSION,
        }:
            correction_fields = _DEX_CORRECTIONS
        elif schema_version == LEGACY_ATOMIC_CALIBRATION_SCHEMA_VERSION:
            correction_fields = _LEGACY_CORRECTIONS
        else:
            raise ValueError(f"unsupported atomic calibration schema {schema_version}")

    required: set[str] = {
        "key",
        "component_group_index",
        *correction_fields.values(),
    }
    if schema_version == ATOMIC_CALIBRATION_SCHEMA_VERSION:
        forbidden_public_fields = {
            "symbol",
            "center_nm",
            *{
                f"component_{name}"
                for name in ATOMIC_CALIBRATION_SIGNATURE_FIELDS
            },
            *ATOMIC_CALIBRATION_ABSOLUTE_FIELDS.values(),
        }.intersection(arrays)
        if forbidden_public_fields:
            raise ValueError(
                "schema-4 atomic calibration exposes forbidden source-derived "
                "fields: " + ", ".join(sorted(forbidden_public_fields))
            )
        required.update(
            {
                "source_catalog_sha256",
                _HASHED_COMPONENT_SIGNATURE_FIELD,
                _HASHED_COMPONENT_ORDINAL_FIELD,
                "scope_wavelength_nm",
                "scope_atomic_number",
                "scope_line_type",
                "scope_ion_stage",
            }
        )
    else:
        required.update(
            f"component_{name}" for name in ATOMIC_CALIBRATION_SIGNATURE_FIELDS
        )
    missing = required.difference(arrays)
    if missing:
        raise ValueError(
            "atomic calibration lacks required portable fields: "
            + ", ".join(sorted(missing))
        )

    keys = arrays["key"]
    if (
        keys.ndim != 1
        or keys.size == 0
        or keys.dtype.kind not in {"S", "U"}
        or np.unique(keys).size != keys.size
    ):
        raise ValueError(
            "atomic calibration key must be a non-empty vector of unique strings"
        )
    group_count = int(keys.size)
    if schema_version == ATOMIC_CALIBRATION_SCHEMA_VERSION:
        arrays["key"] = _validate_sha256_vector(keys, field="key")
        if np.unique(arrays["key"]).size != group_count:
            raise ValueError(
                "atomic calibration key values must remain unique after "
                "SHA-256 case normalization"
            )
    corrections: dict[str, np.ndarray] = {}
    for physical_name, stored_name in correction_fields.items():
        value = arrays[stored_name]
        if value.shape != (group_count,) or not np.issubdtype(value.dtype, np.number):
            raise ValueError(
                f"atomic calibration {stored_name} must be a numeric group vector"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(f"atomic calibration {stored_name} must be finite")
        corrections[physical_name] = np.asarray(value, np.float64)

    group_index = arrays["component_group_index"]
    if group_index.ndim != 1 or not np.issubdtype(group_index.dtype, np.integer):
        raise ValueError(
            "atomic calibration component_group_index must be an integer vector"
        )
    if group_index.size == 0 or np.any(
        (group_index < 0) | (group_index >= group_count)
    ):
        raise ValueError(
            "atomic calibration component_group_index contains no valid components"
        )
    if schema_version == ATOMIC_CALIBRATION_SCHEMA_VERSION:
        signatures = _validate_sha256_vector(
            arrays[_HASHED_COMPONENT_SIGNATURE_FIELD],
            field=_HASHED_COMPONENT_SIGNATURE_FIELD,
        )
        if signatures.shape != group_index.shape:
            raise ValueError(
                f"atomic calibration {_HASHED_COMPONENT_SIGNATURE_FIELD} must "
                "match component_group_index"
            )
        arrays[_HASHED_COMPONENT_SIGNATURE_FIELD] = signatures
        ordinals = arrays[_HASHED_COMPONENT_ORDINAL_FIELD]
        if (
            ordinals.shape != group_index.shape
            or not np.issubdtype(ordinals.dtype, np.integer)
            or np.any(ordinals < 0)
        ):
            raise ValueError(
                f"atomic calibration {_HASHED_COMPONENT_ORDINAL_FIELD} must be "
                "a nonnegative integer component vector"
            )
        identities = set(
            zip(signatures.tolist(), np.asarray(ordinals, np.int64).tolist())
        )
        if len(identities) != len(ordinals):
            raise ValueError("atomic calibration contains duplicate opaque row identities")
        scope_wavelength = np.asarray(arrays["scope_wavelength_nm"])
        if (
            scope_wavelength.shape != (2,)
            or not np.issubdtype(scope_wavelength.dtype, np.number)
            or not np.all(np.isfinite(scope_wavelength))
            or float(scope_wavelength[0]) > float(scope_wavelength[1])
        ):
            raise ValueError(
                "atomic calibration scope_wavelength_nm must be a finite ordered pair"
            )
        for field in ("scope_atomic_number", "scope_line_type", "scope_ion_stage"):
            value = arrays[field]
            if (
                value.ndim != 1
                or value.size == 0
                or not np.issubdtype(value.dtype, np.integer)
                or np.unique(value).size != value.size
            ):
                raise ValueError(
                    f"atomic calibration {field} must be a non-empty unique integer vector"
                )
    else:
        for name in ATOMIC_CALIBRATION_SIGNATURE_FIELDS:
            component = arrays[f"component_{name}"]
            if component.shape != group_index.shape:
                raise ValueError(
                    f"atomic calibration component_{name} must match "
                    "component_group_index"
                )
            if not np.issubdtype(component.dtype, np.number) or not np.all(
                np.isfinite(component)
            ):
                raise ValueError(
                    f"atomic calibration component_{name} must be numeric and finite"
                )
            if name in _INTEGER_SIGNATURE_FIELDS and not np.issubdtype(
                component.dtype, np.integer
            ):
                raise ValueError(
                    f"atomic calibration component_{name} must use an integer dtype"
                )

    absolute_fields_present = {
        name for name in ATOMIC_CALIBRATION_ABSOLUTE_FIELDS.values() if name in arrays
    }
    if (
        schema_version == ATOMIC_CALIBRATION_SCHEMA_VERSION
        and absolute_fields_present
    ):
        raise ValueError(
            "schema-4 atomic calibrations must be correction-only and cannot "
            "contain absolute source-derived parameters"
        )
    if absolute_fields_present and absolute_fields_present != set(
        ATOMIC_CALIBRATION_ABSOLUTE_FIELDS.values()
    ):
        missing_absolute = set(ATOMIC_CALIBRATION_ABSOLUTE_FIELDS.values()).difference(
            absolute_fields_present
        )
        raise ValueError(
            "atomic calibration absolute component values are incomplete: "
            + ", ".join(sorted(missing_absolute))
        )
    for physical_name, stored_name in ATOMIC_CALIBRATION_ABSOLUTE_FIELDS.items():
        if stored_name not in arrays:
            continue
        value = arrays[stored_name]
        if value.shape != group_index.shape or not np.issubdtype(
            value.dtype, np.number
        ):
            raise ValueError(
                f"atomic calibration {stored_name} must be a numeric component vector"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(f"atomic calibration {stored_name} must be finite")
        if physical_name != "loggf" and np.any(value < 0.0):
            raise ValueError(f"atomic calibration {stored_name} must be nonnegative")

    if absolute_fields_present:
        expected_loggf = (
            np.asarray(arrays["component_log_oscillator_strength"], np.float64)
            + corrections["loggf"][np.asarray(group_index, np.int64)]
        )
        calibrated_loggf = np.asarray(
            arrays[ATOMIC_CALIBRATION_ABSOLUTE_FIELDS["loggf"]], np.float64
        )
        if not np.array_equal(calibrated_loggf, expected_loggf):
            raise ValueError(
                "atomic calibration component_calibrated_log_oscillator_strength "
                "disagrees with the retained source values and delta_loggf_dex"
            )

    if schema_version in {
        ATOMIC_CALIBRATION_SCHEMA_VERSION,
        LEGACY_PORTABLE_ATOMIC_CALIBRATION_SCHEMA_VERSION,
    }:
        calibration_name = _scalar_string(
            arrays.get("calibration_name", np.asarray([])), field="calibration_name"
        )
    else:
        calibration_name = _scalar_string(
            arrays.get("calibration_star", np.asarray([])), field="calibration_star"
        )

    metadata: dict[str, object] = {
        "path": str(path),
        "schema_version": schema_version,
        "legacy_schema_marker_missing": schema_version is None,
        "calibration_name": calibration_name,
        "group_count": group_count,
        "component_count": int(group_index.size),
        "correction_fields": dict(correction_fields),
        "identity_representation": (
            "sha256_plus_occurrence_ordinal"
            if schema_version == ATOMIC_CALIBRATION_SCHEMA_VERSION
            else "cleartext_source_fields"
        ),
        "absolute_parameter_fields": (
            dict(ATOMIC_CALIBRATION_ABSOLUTE_FIELDS) if absolute_fields_present else {}
        ),
    }
    for field in (
        "source_catalog_sha256",
        "source_parameter_sha256",
        "evidence_sha256",
        "grouping_sha256",
    ):
        if field in arrays:
            metadata[field] = _scalar_sha256(arrays[field], field=field)
    arrays["_delta_loggf_dex"] = corrections["loggf"]
    arrays["_delta_log_vdw_dex"] = corrections["vdw"]
    arrays["_delta_log_radiative_dex"] = corrections["radiative"]
    arrays["_delta_log_stark_dex"] = corrections["stark"]
    return arrays, metadata


def validate_atomic_calibration(calibration_path: str | Path) -> dict[str, object]:
    """Validate the portable overlay file contract and return its inventory."""

    _, metadata = load_atomic_calibration(calibration_path)
    return metadata


def bundled_atomic_calibration(name: str) -> Path:
    """Return one provenance-bound optional standard-star overlay."""

    try:
        filename = _BUNDLED_CALIBRATIONS[name]
    except KeyError as error:
        choices = ", ".join(sorted(_BUNDLED_CALIBRATIONS))
        raise ValueError(
            f"unknown bundled calibration {name!r}; choose {choices}"
        ) from error
    return Path(files("linelist_calibration").joinpath("data", filename))


def bundled_atomic_calibrations() -> dict[str, Path]:
    """Return the installed standard-star overlays keyed by public name."""

    return {
        name: bundled_atomic_calibration(name)
        for name in sorted(_BUNDLED_CALIBRATIONS)
    }


def apply_atomic_calibration(
    catalog: Mapping[str, np.ndarray],
    calibration_path: str | Path,
    *,
    catalog_scope: Literal["complete", "selected_window"] = "complete",
    source_catalog_path: str | Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Apply one overlay to a private catalog copy.

    Every catalog value is a one-dimensional row array. The mapping must include
    ``ATOMIC_CALIBRATION_SIGNATURE_FIELDS`` and the oscillator-strength and
    damping arrays carried by ``payne_zero_synthesis.atomic_lines.LineCatalog``.
    The returned mapping contains corrected copies with identical shapes.

    ``complete`` requires a one-to-one match for every overlay component and is
    the contract used by catalog export. ``selected_window`` permits overlay
    components absent from an active synthesis catalog, but still requires an
    exact match for every calibratable row present in that catalog. Schema-4
    products also require ``source_catalog_path``; its bytes must match the
    digest embedded in the overlay before any correction is applied.
    """

    if catalog_scope not in {"complete", "selected_window"}:
        raise ValueError("catalog_scope must be 'complete' or 'selected_window'")

    parameters, contract = load_atomic_calibration(calibration_path)
    schema_version = contract["schema_version"]
    if schema_version == ATOMIC_CALIBRATION_SCHEMA_VERSION:
        if source_catalog_path is None:
            raise ValueError(
                "schema-4 atomic calibration application requires "
                "source_catalog_path for exact source hash validation"
            )
        expected_source_hash = str(contract["source_catalog_sha256"])
        actual_source_hash = _sha256_file(source_catalog_path)
        if actual_source_hash != expected_source_hash:
            raise RuntimeError(
                "atomic calibration source catalog SHA-256 mismatch: "
                f"expected {expected_source_hash}, found {actual_source_hash}"
            )
    else:
        actual_source_hash = None
    corrected = {
        name: np.asarray(value).copy() if isinstance(value, np.ndarray) else value
        for name, value in catalog.items()
    }
    row_count = int(np.asarray(corrected["wavelength_nm"]).size)
    group_index = np.asarray(parameters["component_group_index"], np.int64)
    matched_rows: list[int] = []
    matched_groups: list[int] = []
    matched_components: list[int] = []
    unmatched_components: list[int] = []
    if schema_version == ATOMIC_CALIBRATION_SCHEMA_VERSION:
        catalog_signatures, catalog_ordinals = canonical_atomic_row_identities(corrected)
        catalog_by_identity = {
            (str(signature), int(ordinal)): index
            for index, (signature, ordinal) in enumerate(
                zip(catalog_signatures.tolist(), catalog_ordinals.tolist())
            )
        }
        component_signatures = np.asarray(
            parameters[_HASHED_COMPONENT_SIGNATURE_FIELD], dtype="U64"
        )
        component_ordinals = np.asarray(
            parameters[_HASHED_COMPONENT_ORDINAL_FIELD], np.int64
        )
        if catalog_scope == "selected_window":
            overlay_counts: dict[str, int] = {}
            for signature, ordinal in zip(
                component_signatures.tolist(), component_ordinals.tolist()
            ):
                overlay_counts[str(signature)] = max(
                    overlay_counts.get(str(signature), 0), int(ordinal) + 1
                )
            catalog_counts: dict[str, int] = {}
            for signature in catalog_signatures.tolist():
                catalog_counts[str(signature)] = (
                    catalog_counts.get(str(signature), 0) + 1
                )
            partial_duplicates = [
                signature
                for signature, component_count in overlay_counts.items()
                if 0 < catalog_counts.get(signature, 0) < component_count
            ]
            if partial_duplicates:
                raise RuntimeError(
                    "selected catalog contains only part of "
                    f"{len(partial_duplicates)} duplicate component signature groups"
                )
        for component_index, identity in enumerate(
            zip(component_signatures.tolist(), component_ordinals.tolist())
        ):
            row = catalog_by_identity.get((str(identity[0]), int(identity[1])))
            if row is None:
                unmatched_components.append(component_index)
            else:
                matched_rows.append(row)
                matched_groups.append(int(group_index[component_index]))
                matched_components.append(component_index)
    else:
        component_arrays = {
            name: parameters[f"component_{name}"]
            for name in ATOMIC_CALIBRATION_SIGNATURE_FIELDS
        }
        catalog_by_signature: dict[tuple[object, ...], list[int]] = {}
        for index in range(row_count):
            catalog_by_signature.setdefault(
                _row_signature(corrected, index), []
            ).append(index)
        for indices in catalog_by_signature.values():
            indices.reverse()
        if catalog_scope == "selected_window":
            components_by_signature: dict[tuple[object, ...], int] = {}
            for component_index in range(group_index.size):
                signature = _row_signature(component_arrays, component_index)
                components_by_signature[signature] = (
                    components_by_signature.get(signature, 0) + 1
                )
            partial_duplicates = [
                signature
                for signature, component_count in components_by_signature.items()
                if 0
                < len(catalog_by_signature.get(signature, ()))
                < component_count
            ]
            if partial_duplicates:
                raise RuntimeError(
                    "selected catalog contains only part of "
                    f"{len(partial_duplicates)} duplicate component signature groups"
                )
        for component_index in range(group_index.size):
            candidates = catalog_by_signature.get(
                _row_signature(component_arrays, component_index)
            )
            if candidates:
                matched_rows.append(candidates.pop())
                matched_groups.append(int(group_index[component_index]))
                matched_components.append(component_index)
            else:
                unmatched_components.append(component_index)

    if unmatched_components and catalog_scope == "complete":
        raise RuntimeError(
            "atomic calibration contains "
            f"{len(unmatched_components)} components absent from the source catalog"
        )

    rows = np.asarray(matched_rows, np.int64)
    parameter_indices = np.asarray(matched_groups, np.int64)
    component_indices = np.asarray(matched_components, np.int64)
    if schema_version == ATOMIC_CALIBRATION_SCHEMA_VERSION:
        lower_wavelength, upper_wavelength = np.asarray(
            parameters["scope_wavelength_nm"], np.float64
        ).tolist()
        scope_line_type = np.asarray(parameters["scope_line_type"], np.int64)
        scope_ion_stage = np.asarray(parameters["scope_ion_stage"], np.int64)
        scope_atomic_number = np.asarray(
            parameters["scope_atomic_number"], np.int64
        )
    else:
        component_wavelength = np.asarray(
            parameters["component_wavelength_nm"], np.float64
        )
        lower_wavelength = float(np.min(component_wavelength))
        upper_wavelength = float(np.max(component_wavelength))
        scope_line_type = np.asarray((0, 3), np.int64)
        scope_ion_stage = np.asarray((1, 2), np.int64)
        scope_atomic_number = np.unique(
            np.asarray(parameters["component_atomic_number"], np.int64)
        )
    line_type = np.asarray(corrected["line_type"], np.int64)
    ion_stage = np.asarray(corrected["ion_stage"], np.int64)
    wavelength = np.asarray(corrected["wavelength_nm"], np.float64)
    calibratable = (
        np.isin(line_type, scope_line_type)
        & np.isin(ion_stage, scope_ion_stage)
        & (wavelength >= lower_wavelength)
        & (wavelength <= upper_wavelength)
        & np.isin(
            np.asarray(corrected["atomic_number"], np.int64),
            scope_atomic_number,
        )
    )
    unmatched = np.setdiff1d(np.flatnonzero(calibratable), rows, assume_unique=False)
    if unmatched.size:
        raise RuntimeError(
            "atomic calibration component identities do not cover "
            f"{unmatched.size} calibratable catalog rows"
        )

    has_absolute_parameters = bool(contract["absolute_parameter_fields"])
    if has_absolute_parameters:
        calibrated_loggf = np.asarray(
            parameters[ATOMIC_CALIBRATION_ABSOLUTE_FIELDS["loggf"]], np.float64
        )[component_indices]
        calibrated_vdw = np.asarray(
            parameters[ATOMIC_CALIBRATION_ABSOLUTE_FIELDS["vdw"]], np.float64
        )[component_indices]
        calibrated_radiative = np.asarray(
            parameters[ATOMIC_CALIBRATION_ABSOLUTE_FIELDS["radiative"]], np.float64
        )[component_indices]
        calibrated_stark = np.asarray(
            parameters[ATOMIC_CALIBRATION_ABSOLUTE_FIELDS["stark"]], np.float64
        )[component_indices]
        corrected["log_oscillator_strength"][rows] = calibrated_loggf
        corrected["oscillator_strength"][rows] = np.power(10.0, calibrated_loggf)
        corrected["van_der_waals_damping"][rows] = calibrated_vdw
        corrected["radiative_damping"][rows] = calibrated_radiative
        corrected["stark_damping"][rows] = calibrated_stark
        applied_representation = "absolute_component_values"
    else:
        delta_loggf = parameters["_delta_loggf_dex"][parameter_indices]
        delta_vdw = parameters["_delta_log_vdw_dex"][parameter_indices]
        delta_radiative = parameters["_delta_log_radiative_dex"][parameter_indices]
        delta_stark = parameters["_delta_log_stark_dex"][parameter_indices]
        corrected["oscillator_strength"][rows] *= np.power(10.0, delta_loggf)
        corrected["log_oscillator_strength"][rows] += delta_loggf
        corrected["van_der_waals_damping"][rows] *= np.power(10.0, delta_vdw)
        corrected["radiative_damping"][rows] *= np.power(10.0, delta_radiative)
        corrected["stark_damping"][rows] *= np.power(10.0, delta_stark)
        applied_representation = "group_deltas"
    metadata = {
        **contract,
        "catalog_scope": catalog_scope,
        "applied_parameter_representation": applied_representation,
        "matched_groups": int(np.unique(parameter_indices).size),
        "matched_catalog_rows": int(rows.size),
        "unmatched_calibration_components": int(len(unmatched_components)),
        "unmatched_calibratable_rows": int(unmatched.size),
        "wavelength_range_nm": [lower_wavelength, upper_wavelength],
        "source_catalog_hash_verified": (
            actual_source_hash == contract.get("source_catalog_sha256")
            if actual_source_hash is not None
            else False
        ),
    }
    return corrected, metadata


def write_substituted_catalog(
    catalog: Mapping[str, np.ndarray],
    calibration_path: str | Path,
    output_path: str | Path,
    *,
    source_catalog_path: str | Path | None = None,
) -> dict[str, object]:
    """Write corrected same-shape catalog arrays and a JSON sidecar.

    ``catalog`` follows :func:`apply_atomic_calibration`. ``output_path`` gains
    a ``.npz`` suffix when needed. Schema-4 products require the exact
    ``source_catalog_path`` whose SHA-256 is recorded in the overlay.
    """

    destination = Path(output_path).expanduser().resolve()
    if destination.suffix != ".npz":
        destination = Path(f"{destination}.npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    corrected, metadata = apply_atomic_calibration(
        catalog,
        calibration_path,
        catalog_scope="complete",
        source_catalog_path=source_catalog_path,
    )
    np.savez_compressed(destination, **corrected)
    sidecar = destination.with_suffix(destination.suffix + ".json")
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return {**metadata, "output_path": str(destination), "metadata_path": str(sidecar)}


__all__ = [
    "ATOMIC_CALIBRATION_ABSOLUTE_FIELDS",
    "ATOMIC_CALIBRATION_SCHEMA_VERSION",
    "ATOMIC_CALIBRATION_SIGNATURE_FIELDS",
    "apply_atomic_calibration",
    "bundled_atomic_calibration",
    "bundled_atomic_calibrations",
    "canonical_atomic_row_identities",
    "load_atomic_calibration",
    "validate_atomic_calibration",
    "write_substituted_catalog",
]
