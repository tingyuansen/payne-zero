"""Resolution of the bundled full-range source line-catalog tree.

The workspace bundles all physical source catalogs (all stellar types, all
wavelength ranges) under ``source_data_files/source_catalogs/`` at the
workspace root. These are physics inputs with canonical Kurucz file names preserved for
provenance; they are not reference answers or cached model outputs.

Resolution order:

1. ``PAYNE_ZERO_SOURCE_CATALOG_ROOT`` environment variable, when set;
2. the bundled workspace tree next to this package;
3. a loud error naming both options.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re

import numpy as np

from .data_files import data_root

SOURCE_CATALOG_ROOT_ENV = "PAYNE_ZERO_SOURCE_CATALOG_ROOT"

BUNDLED_SOURCE_CATALOG_ROOT = data_root() / "source_catalogs"
SOURCE_CATALOG_CHECKSUMS = BUNDLED_SOURCE_CATALOG_ROOT / "CHECKSUMS.sha256"


class SourceCatalogError(RuntimeError):
    """Raised when the source-catalog tree or a required member is missing."""


def source_catalog_root() -> Path:
    """Return the source-catalog root, preferring the environment override."""

    configured = os.environ.get(SOURCE_CATALOG_ROOT_ENV)
    if configured:
        root = Path(configured).expanduser()
        if not root.is_dir():
            raise SourceCatalogError(
                f"{SOURCE_CATALOG_ROOT_ENV}={root} is not a directory"
            )
        return root
    if BUNDLED_SOURCE_CATALOG_ROOT.is_dir():
        return BUNDLED_SOURCE_CATALOG_ROOT
    raise SourceCatalogError(
        "source-catalog tree not found: expected the bundled tree at "
        f"{BUNDLED_SOURCE_CATALOG_ROOT} or {SOURCE_CATALOG_ROOT_ENV} pointing "
        "at an equivalent external tree"
    )


def _require(path: Path) -> Path:
    if not path.exists():
        raise SourceCatalogError(f"missing source catalog member: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source_catalog_checksums(
    checksum_path: Path | None = None,
) -> dict[str, str]:
    """Load the committed checksum identity for every runtime source catalog."""

    path = checksum_path or SOURCE_CATALOG_CHECKSUMS
    if not path.is_file():
        raise SourceCatalogError(f"missing source-catalog checksum manifest: {path}")
    checksums: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]):
            raise SourceCatalogError(
                f"malformed checksum manifest line {line_number}: {path}"
            )
        relative = fields[1].removeprefix("./")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise SourceCatalogError(
                f"unsafe checksum manifest path on line {line_number}: {relative}"
            )
        normalized = relative_path.as_posix()
        if normalized in checksums:
            raise SourceCatalogError(
                f"duplicate checksum manifest path on line {line_number}: {relative}"
            )
        checksums[normalized] = fields[0].lower()
    if not checksums:
        raise SourceCatalogError(f"empty source-catalog checksum manifest: {path}")
    return checksums


def verify_source_catalog_checksums(
    *,
    root: Path | None = None,
    checksum_path: Path | None = None,
) -> dict[str, object]:
    """Hash every staged source catalog once and require its committed identity."""

    catalog_root = (root or source_catalog_root()).resolve()
    manifest_path = (checksum_path or SOURCE_CATALOG_CHECKSUMS).resolve()
    expected = load_source_catalog_checksums(manifest_path)
    files = []
    for relative, expected_sha256 in expected.items():
        path = _require(catalog_root / relative)
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise SourceCatalogError(
                f"source catalog checksum mismatch: {path}; expected "
                f"{expected_sha256}, found {actual_sha256}"
            )
        files.append(
            {
                "path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": actual_sha256,
            }
        )
    return {
        "status": "verified",
        "root": str(catalog_root),
        "checksum_manifest": str(manifest_path),
        "checksum_manifest_sha256": _sha256(manifest_path),
        "file_count": len(files),
        "total_bytes": sum(int(record["bytes"]) for record in files),
        "files": files,
    }


def source_line_paths(root: Path | None = None) -> dict[str, Path]:
    """Return the full source line-catalog set for general runs.

    Keys match the ``AtmosphereInput`` field names for the source
    catalogs (predicted / observed / high-excitation atomic, diatomic, TiO,
    water, and H3+ lines).
    """

    base = root or source_catalog_root()
    lines_dir = base / "lines"
    return {
        "predicted_atomic_lines_path": _require(
            lines_dir / "predicted_atomic_lines_part1.npy"
        ),
        "observed_atomic_lines_path": _require(lines_dir / "observed_atomic_lines.npy"),
        "high_excitation_lines_path": _require(lines_dir / "high_excitation_lines.npy"),
        "diatomic_lines_path": _require(lines_dir / "diatomic_lines.npy"),
        "titanium_oxide_lines_path": _require(
            base / "molecules/titanium_oxide_lines.npy"
        ),
        "water_lines_path": _require(base / "molecules/water_lines.npy"),
        "detailed_line_catalog_path": _require(
            lines_dir / "detailed_transition_lines.npz"
        ),
    }


def molecular_equilibrium_catalog_path(root: Path | None = None) -> Path:
    """Return the atmosphere-stage molecular-equilibrium catalog."""

    base = root or source_catalog_root()
    return _require(base / "lines" / "molecular_equilibrium_atmosphere.npz")


def atmosphere_source_catalog_paths(root: Path | None = None) -> dict[str, Path]:
    """Return every physical file required by one exact atmosphere solve.

    The predicted atomic catalog is split only to satisfy hosting limits.  The
    runtime API points at ``part1``, whose reader joins all parts, so an install
    preflight must inventory every shard explicitly rather than only the API
    entry point.
    """

    base = root or source_catalog_root()
    runtime_paths = source_line_paths(base)
    predicted_part1 = runtime_paths.pop("predicted_atomic_lines_path")
    stem = predicted_part1.stem[: -len("_part1")]
    part_paths = sorted(
        predicted_part1.parent.glob(f"{stem}_part*.npy"),
        key=lambda path: int(re.search(r"_part(\d+)$", path.stem).group(1)),
    )
    part_numbers = [
        int(re.search(r"_part(\d+)$", path.stem).group(1)) for path in part_paths
    ]
    expected_numbers = list(range(1, len(part_paths) + 1))
    if not part_paths or part_numbers != expected_numbers:
        raise SourceCatalogError(
            "predicted atomic catalog shards must be contiguous from part1; "
            f"found {part_numbers or 'none'} in {predicted_part1.parent}"
        )

    paths = {
        f"predicted_atomic_lines_part{number}": _require(path)
        for number, path in zip(part_numbers, part_paths, strict=True)
    }
    paths.update(runtime_paths)
    paths["molecular_equilibrium_catalog_path"] = molecular_equilibrium_catalog_path(
        base
    )
    return paths


def validate_npy_catalog_file(path: Path) -> None:
    """Validate one NPY header and require its declared payload to be complete."""

    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            header_reader = np.lib.format.read_array_header_1_0
        elif version in {(2, 0), (3, 0)}:
            header_reader = np.lib.format.read_array_header_2_0
        else:
            raise SourceCatalogError(f"unsupported NPY version {version}: {path}")
        shape, _fortran_order, dtype = header_reader(handle)
        payload_offset = handle.tell()
    declared_bytes = int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
    expected_bytes = payload_offset + declared_bytes
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise SourceCatalogError(
            f"truncated or malformed NPY source catalog: {path} declares "
            f"{expected_bytes} bytes but contains {actual_bytes}; repair the source "
            "catalog before prewarming"
        )
