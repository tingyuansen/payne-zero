"""Project a portable atomic calibration into device-resident invariants."""

from __future__ import annotations

import dataclasses
import hashlib
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from linelist_calibration.catalog import (
    ATOMIC_CALIBRATION_SCHEMA_VERSION,
    ATOMIC_CALIBRATION_SIGNATURE_FIELDS,
    apply_atomic_calibration,
    load_atomic_calibration,
    validate_atomic_calibration,
)
from payne_zero_synthesis import line_opacity, paths as synthesis_paths
from payne_zero_synthesis.pipeline import SynthesisPipeline, WindowInvariants


_CALIBRATED_CACHE: dict[
    tuple[object, ...],
    tuple[WindowInvariants, WindowInvariants, dict[str, object]],
] = {}
_FILE_DIGEST_CACHE: dict[str, tuple[tuple[int, ...], str]] = {}
# Compatibility alias for callers that build a portable component signature.
_SIGNATURE_FIELDS = ATOMIC_CALIBRATION_SIGNATURE_FIELDS


def _stat_identity(stat: os.stat_result) -> tuple[int, ...]:
    """Return cheap fields that change whenever a local file is replaced."""

    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


@dataclasses.dataclass(frozen=True)
class _FileSnapshot:
    identity: tuple[str, str]
    content: bytes


def _snapshot_content(stream) -> tuple[bytes, str]:
    digest = hashlib.sha256()
    blocks: list[bytes] = []
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        blocks.append(block)
        digest.update(block)
    return b"".join(blocks), digest.hexdigest()


def _cached_file_identity(path: Path) -> tuple[str, str] | None:
    resolved = path.resolve()
    cached = _FILE_DIGEST_CACHE.get(str(resolved))
    if cached is None or cached[0] != _stat_identity(resolved.stat()):
        return None
    return str(resolved), cached[1]


def _file_snapshot(path: Path) -> _FileSnapshot:
    """Capture one stable byte snapshot and its matching content digest."""

    resolved = path.resolve()
    resolved_string = str(resolved)
    for _ in range(3):
        before = _stat_identity(resolved.stat())
        with resolved.open("rb") as stream:
            opened = _stat_identity(os.fstat(stream.fileno()))
            if opened != before:
                continue
            content, digest = _snapshot_content(stream)
            finished = _stat_identity(os.fstat(stream.fileno()))
        if opened == finished == _stat_identity(resolved.stat()):
            _FILE_DIGEST_CACHE[resolved_string] = (finished, digest)
            return _FileSnapshot((resolved_string, digest), content)
    raise RuntimeError(f"atomic calibration changed while reading: {resolved}")


def _file_identity(path: Path) -> tuple[str, str]:
    """Return path plus content identity without rehashing unchanged files."""

    return _cached_file_identity(path) or _file_snapshot(path).identity


def _load_atomic_calibration(
    calibration_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Compatibility wrapper for the generic public overlay loader."""

    return load_atomic_calibration(calibration_path)


def _calibrated_catalog(
    bundle: WindowInvariants,
    calibration_path: Path,
    snapshot: _FileSnapshot | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Apply a portable overlay to a private copy of the base catalog."""

    path = calibration_path.resolve()
    captured = snapshot or _file_snapshot(path)
    source_catalog_path = synthesis_paths.source_catalog_path(
        "lines", "atomic_source_lines_parsed.npz"
    )
    with tempfile.TemporaryDirectory(prefix="payne-zero-atomic-calibration-") as root:
        snapshot_path = Path(root) / "atomic_calibration.npz"
        snapshot_path.write_bytes(captured.content)
        catalog, metadata = apply_atomic_calibration(
            bundle.atomic_kernel_catalog,
            snapshot_path,
            catalog_scope="selected_window",
            source_catalog_path=source_catalog_path,
        )
    metadata = {**metadata, "path": str(path)}
    return catalog, {
        **metadata,
        # Preserve the public fitter metadata used by earlier result readers.
        "calibration_path": metadata["path"],
        "calibration_star": metadata["calibration_name"],
        "calibration_group_count": metadata["group_count"],
        "matched_groups_in_window": metadata["matched_groups"],
        "implementation": (
            "component identities projected once into resident atomic invariants"
        ),
    }


def calibrated_window_invariants(
    base: WindowInvariants, calibration_path: str | Path
) -> tuple[WindowInvariants, dict[str, object]]:
    """Return a process-cached invariant bundle with calibrated metal tensors."""

    path = Path(calibration_path).expanduser().resolve()
    file_identity = _cached_file_identity(path)
    if file_identity is not None:
        key = (id(base), *file_identity)
        cached = _CALIBRATED_CACHE.get(key)
        if cached is not None and cached[0] is base:
            _, bundle, metadata = cached
            return bundle, {**metadata, "resident_cache_reused": True}

    snapshot = _file_snapshot(path)
    file_identity = snapshot.identity
    key = (id(base), *file_identity)
    cached = _CALIBRATED_CACHE.get(key)
    if cached is not None and cached[0] is base:
        _, bundle, metadata = cached
        return bundle, {**metadata, "resident_cache_reused": True}

    start = time.perf_counter()
    catalog, metadata = _calibrated_catalog(base, path, snapshot)
    line_type = np.asarray(catalog["line_type"], np.int64)
    metal_indices = np.flatnonzero(np.isin(line_type, (0, 1, 3)))
    chunks = []
    for chunk_start in range(0, metal_indices.size, base.metal_chunk):
        indices = metal_indices[chunk_start : chunk_start + base.metal_chunk]
        chunks.append(
            line_opacity.precompute_invariants(
                SynthesisPipeline._slice_atomic_catalog(catalog, indices),
                base.synthesis_wavelength_nm,
                runtime_device=base.device,
            )
        )
    bundle = dataclasses.replace(
        base,
        key=(*base.key, "atomic_calibration", *file_identity),
        atomic_kernel_catalog=catalog,
        metal_invariant_chunks=chunks,
    )
    metadata = {
        **metadata,
        "calibration_artifact_sha256": file_identity[1],
        "compile_seconds": time.perf_counter() - start,
        "resident_cache_reused": False,
    }
    # Retaining and checking the exact base object prevents an ``id(base)``
    # collision after a synthesis-window cache clear and Python id reuse.
    _CALIBRATED_CACHE[key] = base, bundle, metadata
    return bundle, metadata


__all__ = [
    "ATOMIC_CALIBRATION_SCHEMA_VERSION",
    "ATOMIC_CALIBRATION_SIGNATURE_FIELDS",
    "calibrated_window_invariants",
    "validate_atomic_calibration",
]
