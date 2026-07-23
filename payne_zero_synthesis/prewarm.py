"""Populate persistent synthesis caches for a fixed wavelength window."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

from payne_zero_atmosphere._numba_cache import configure_numba_cache

configure_numba_cache()

import torch  # noqa: E402

from . import paths as runtime_paths  # noqa: E402
from . import atomic_lines  # noqa: E402
from . import pipeline as synthesis_pipeline  # noqa: E402
from .pipeline import (  # noqa: E402
    clear_window_invariant_cache,
    window_invariants_for,
)


PREWARM_MANIFEST_SCHEMA_VERSION = 3


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    package_dir = Path(__file__).resolve().parent
    for path in sorted(package_dir.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _inventory(cache_root: Path) -> dict:
    categories = {}
    for name in ("atomic_lines", "molecular_source", "molecular_compiled"):
        directory = cache_root / name
        files = [path for path in directory.rglob("*.npz") if path.is_file()]
        categories[name] = {
            "file_count": len(files),
            "bytes": sum(path.stat().st_size for path in files),
        }
    return categories


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _window_artifact_paths(
    *,
    wavelength_start_nm: float,
    wavelength_end_nm: float,
    resolution: float,
) -> dict[str, Path]:
    """Resolve the two persistent products used by exactly this window."""
    atomic_source = runtime_paths.source_catalog_path(
        "lines", "atomic_source_lines_parsed.npz"
    )
    (
        _requested_grid,
        synthesis_grid,
        _requested_wavelength_nm,
        _synthesis_wavelength_nm,
        _output_slice,
    ) = synthesis_pipeline._window_grid_contract(  # noqa: SLF001
        wavelength_start_nm,
        wavelength_end_nm,
        resolution,
    )
    atomic_key = atomic_lines._cache_key(  # noqa: SLF001
        atomic_source, synthesis_grid, "catalog", True
    )
    atomic_path = (
        runtime_paths.PACKAGE_CACHE_ROOT
        / "atomic_lines"
        / f"atomic_lines_{atomic_key}.npz"
    )
    molecular_path, _ = synthesis_pipeline._molecular_compiled_cache_contract(  # noqa: SLF001
        synthesis_grid.start_wavelength_nm,
        synthesis_grid.end_wavelength_nm,
        resolution,
    )
    if molecular_path is None:
        raise RuntimeError(
            "synthesis prewarm requires the persistent compiled molecular cache"
        )
    return {
        "atomic_catalog": atomic_path.resolve(),
        "compiled_molecular_catalog": molecular_path.resolve(),
    }


def _artifact_record(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": int(stat.st_size),
        "sha256": _sha256(path),
    }


def _required_artifacts_match(records: object, expected_paths: dict[str, Path]) -> bool:
    if not isinstance(records, dict) or set(records) != set(expected_paths):
        return False
    for name, expected_path in expected_paths.items():
        record = records.get(name)
        if not isinstance(record, dict):
            return False
        path = expected_path.resolve()
        if record.get("path") != str(path) or not path.is_file():
            return False
        try:
            stat = path.stat()
            if int(record.get("bytes", -1)) != int(stat.st_size):
                return False
            if record.get("sha256") != _sha256(path):
                return False
        except OSError:
            return False
    return True


def prewarm(
    *,
    wavelength_start_nm: float,
    wavelength_end_nm: float,
    resolution: float,
    force: bool = False,
) -> dict:
    cache_root = runtime_paths.PACKAGE_CACHE_ROOT.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    expected_artifacts = _window_artifact_paths(
        wavelength_start_nm=wavelength_start_nm,
        wavelength_end_nm=wavelength_end_nm,
        resolution=resolution,
    )
    identity = {
        "schema_version": PREWARM_MANIFEST_SCHEMA_VERSION,
        "cache_root": str(cache_root),
        "wavelength_start_nm": float(wavelength_start_nm),
        "wavelength_end_nm": float(wavelength_end_nm),
        "resolution": float(resolution),
        "molecular_lines": True,
        "system": platform.system(),
        "machine": platform.machine(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch": torch.__version__,
        "source_fingerprint": _source_fingerprint(),
        "window_artifact_paths": {
            name: str(path) for name, path in expected_artifacts.items()
        },
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[
        :20
    ]
    manifest_dir = cache_root / "prewarm_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"synthesis_{digest}.json"
    inventory_before = _inventory(cache_root)
    if not force and manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            existing = {}
        if not isinstance(existing, dict):
            existing = {}
        if (
            existing.get("status") == "complete"
            and existing.get("identity") == identity
            and _required_artifacts_match(
                existing.get("required_window_artifacts"), expected_artifacts
            )
        ):
            result = {
                **existing,
                "reused": True,
                "prewarm_seconds_this_call": 0.0,
                "cache_inventory": inventory_before,
            }
            print(json.dumps(result, indent=2))
            return result

    started = time.perf_counter()
    # A previous call in the same Python process may hold a complete bundle in
    # device memory even after a disk artifact was removed.  Clear that layer
    # so this prewarm proves the persistent products can really be loaded.
    clear_window_invariant_cache()
    bundle = window_invariants_for(
        wl_start_nm=wavelength_start_nm,
        wl_end_nm=wavelength_end_nm,
        resolution=resolution,
        molecular_lines=True,
        runtime_device=torch.device("cpu"),
        work_dtype=torch.float64,
    )
    seconds = time.perf_counter() - started
    inventory_after = _inventory(cache_root)
    missing = [name for name, path in expected_artifacts.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"synthesis prewarm did not populate: {', '.join(missing)}")
    required_artifacts = {
        name: _artifact_record(path) for name, path in expected_artifacts.items()
    }
    result = {
        "schema_version": PREWARM_MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "identity": identity,
        "cache_root": str(cache_root),
        "prewarm_seconds": seconds,
        "prewarm_seconds_this_call": seconds,
        "reused": False,
        "cache_inventory_before": inventory_before,
        "cache_inventory": inventory_after,
        "required_window_artifacts": required_artifacts,
        "window": {
            "wavelength_count": int(bundle.n_wl),
            "atomic_line_count": int(bundle.n_atomic),
            "molecular_line_count": int(bundle.n_molecular),
            "build_profile": bundle.build_profile,
        },
    }
    manifest_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wavelength-start-nm", type=float, default=1500.0)
    parser.add_argument("--wavelength-end-nm", type=float, default=1700.0)
    parser.add_argument(
        "--r-grid",
        "--resolution",
        dest="resolution",
        type=float,
        default=20_000.0,
        help="logarithmic wavelength-grid density (not instrumental resolution)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    prewarm(
        wavelength_start_nm=args.wavelength_start_nm,
        wavelength_end_nm=args.wavelength_end_nm,
        resolution=args.resolution,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
