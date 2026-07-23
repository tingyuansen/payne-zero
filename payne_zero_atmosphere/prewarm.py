"""Populate and validate the persistent compiled-kernel cache once per runtime."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import llvmlite
from llvmlite import binding as llvm
import numba
from numba import config as numba_config

from ._numba_cache import configure_numba_cache
from .config import (
    DEFAULT_OPACITY_FLAGS,
    AtmosphereConfig,
    AtmosphereInput,
    AtmosphereOutput,
)
from .runner import run_atmosphere_model
from .source_catalogs import (
    atmosphere_source_catalog_paths,
    molecular_equilibrium_catalog_path,
    source_line_paths,
    validate_npy_catalog_file,
)
from .warm_start import emulator_warm_start_model


PREWARM_MANIFEST_SCHEMA_VERSION = 3
REPRESENTATIVE_BRANCHES = (
    {
        "name": "hot",
        "effective_temperature": 9000.0,
        "log_surface_gravity": 4.0,
        "metallicity": 0.0,
        "alpha_enhancement": 0.0,
        "microturbulence_km_s": 2.0,
        "enable_molecules": True,
    },
    {
        "name": "sun",
        "effective_temperature": 5777.0,
        "log_surface_gravity": 4.44,
        "metallicity": 0.0,
        "alpha_enhancement": 0.0,
        "microturbulence_km_s": 2.0,
        "enable_molecules": True,
    },
    {
        "name": "giant",
        "effective_temperature": 4500.0,
        "log_surface_gravity": 2.0,
        "metallicity": -0.5,
        "alpha_enhancement": 0.2,
        "microturbulence_km_s": 2.0,
        "enable_molecules": True,
    },
    {
        "name": "sun_atomic_only",
        "effective_temperature": 5777.0,
        "log_surface_gravity": 4.44,
        "metallicity": 0.0,
        "alpha_enhancement": 0.0,
        "microturbulence_km_s": 2.0,
        "enable_molecules": False,
        # Exercise the physical scalar transition-opacity fallback. Depending
        # on the Numba dispatcher version, its Python call either reuses the
        # float32 hydrogen kernel or caches a separate float64 specialization.
        # The fresh-process write-free check below is the portable completeness
        # invariant for either runtime behavior.
        "numba_threads": 1,
    },
)

# These cached specializations were the residual cold-JIT paths exposed by a
# fresh hot-dwarf solve after a solar-only prewarm. The fresh-process invariant
# below remains the general guard for additional label-dependent branches.
REQUIRED_REPRESENTATIVE_SPECIALIZATIONS = {
    "line_opacity._accumulate_selected_line_opacity_parallel-": 1,
    "line_opacity._hydrogen_line_deposit_compiled-": 1,
    "line_opacity._accumulate_transition_range_compiled-": 1,
}


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    package_dir = Path(__file__).resolve().parent
    for path in sorted(package_dir.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _runtime_signature(cache_dir: Path) -> dict:
    host_features = llvm.get_host_cpu_features()
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numba": numba.__version__,
        "llvmlite": llvmlite.__version__,
        "llvm": ".".join(str(part) for part in llvm.llvm_version_info),
        "llvm_host_cpu_name": str(llvm.get_host_cpu_name()),
        "llvm_host_cpu_features": host_features.flatten(),
        "numba_cpu_name": numba_config.CPU_NAME or "",
        "numba_cpu_features": numba_config.CPU_FEATURES or "",
        "cache_dir": str(cache_dir.resolve()),
        "source_fingerprint": _source_fingerprint(),
    }


def _required_source_inventory() -> dict[str, dict[str, object]]:
    """Require and record every full-catalog atmosphere physics input."""

    inventory = {}
    for name, path in atmosphere_source_catalog_paths().items():
        if path.suffix == ".npy":
            validate_npy_catalog_file(path)
        stat = path.stat()
        inventory[name] = {
            "path": str(path.resolve()),
            "bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "format_validation": (
                "npy-payload-complete" if path.suffix == ".npy" else "exists"
            ),
        }
    return inventory


def _cache_inventory(cache_dir: Path) -> dict:
    artifacts = sorted(
        (
            path
            for path in cache_dir.rglob("*")
            if path.is_file() and path.suffix in {".nbc", ".nbi"}
        ),
        key=lambda path: str(path.relative_to(cache_dir)),
    )
    records = [
        {
            "path": str(path.relative_to(cache_dir)),
            "bytes": int(path.stat().st_size),
            "mtime_ns": int(path.stat().st_mtime_ns),
        }
        for path in artifacts
    ]
    serialized = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return {
        "artifact_count": len(artifacts),
        "artifact_bytes": sum(path.stat().st_size for path in artifacts),
        "artifact_state_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "artifacts": records,
    }


def _missing_representative_specializations(cache_dir: Path) -> list[str]:
    artifact_names = [path.name for path in cache_dir.rglob("*.nbc")]
    missing = []
    for prefix, required_count in REQUIRED_REPRESENTATIVE_SPECIALIZATIONS.items():
        actual_count = sum(name.startswith(prefix) for name in artifact_names)
        if actual_count < required_count:
            missing.append(f"{prefix} ({actual_count}/{required_count})")
    return missing


def _missing_required_kernel_artifacts(cache_dir: Path) -> list[str]:
    """Return cached outer kernels not exercised by the representative solve."""

    package_dir = Path(__file__).resolve().parent
    required = []
    for path in sorted(package_dir.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorators = " ".join(ast.unparse(item) for item in node.decorator_list)
            cached_njit = "_njit" in decorators or "numba.njit" in decorators
            if cached_njit and "_njit_inline" not in decorators:
                required.append(f"{path.stem}.{node.name}-")
    artifact_names = [path.name for path in cache_dir.rglob("*.nbi")]
    return [
        prefix
        for prefix in required
        if not any(prefix in artifact_name for artifact_name in artifact_names)
    ]


def _run_representative_branches() -> list[dict[str, object]]:
    """Exercise one physical iteration for every ordinary cached branch."""

    branch_results = []
    line_paths = source_line_paths()
    molecules_path = molecular_equilibrium_catalog_path()
    for branch in REPRESENTATIVE_BRANCHES:
        started = time.perf_counter()
        atmosphere, _ = emulator_warm_start_model(
            effective_temperature=branch["effective_temperature"],
            log_surface_gravity=branch["log_surface_gravity"],
            metallicity=branch["metallicity"],
            alpha_enhancement=branch["alpha_enhancement"],
            microturbulence_km_s=branch["microturbulence_km_s"],
            device="cpu",
        )
        if not branch["enable_molecules"]:
            opacity_flags = list(DEFAULT_OPACITY_FLAGS)
            opacity_flags[14] = 0
            opacity_flags[16] = 0
            atmosphere.metadata["opacity_flags"] = "OPACITY IFOP " + " ".join(
                str(value) for value in opacity_flags
            )
        original_numba_threads = int(numba.get_num_threads())
        requested_numba_threads = int(
            branch.get("numba_threads", original_numba_threads)
        )
        try:
            if requested_numba_threads != original_numba_threads:
                numba.set_num_threads(requested_numba_threads)
            result = run_atmosphere_model(
                AtmosphereConfig(
                    inputs=AtmosphereInput(
                        initial_atmosphere=atmosphere,
                        molecules_path=molecules_path,
                        **line_paths,
                    ),
                    outputs=AtmosphereOutput(),
                    iterations=1,
                    enable_molecules=bool(branch["enable_molecules"]),
                    enable_convection=True,
                    enable_convergence_stop=False,
                )
            )
        finally:
            if int(numba.get_num_threads()) != original_numba_threads:
                numba.set_num_threads(original_numba_threads)
        iterations_completed = int(result.iterations_completed)
        if iterations_completed != 1:
            raise RuntimeError(
                f"prewarm branch {branch['name']} completed "
                f"{iterations_completed} iterations instead of one"
            )
        branch_results.append(
            {
                **branch,
                "iterations_completed": iterations_completed,
                "seconds": time.perf_counter() - started,
            }
        )
    return branch_results


def _verify_cache_in_current_process(report_path: Path) -> tuple[bool, dict]:
    """Repeat representative branches and require a write-free cache state."""

    cache_dir = configure_numba_cache().resolve()
    before = _cache_inventory(cache_dir)
    branch_results = _run_representative_branches()
    after = _cache_inventory(cache_dir)
    missing_kernels = _missing_required_kernel_artifacts(cache_dir)
    missing_specializations = _missing_representative_specializations(cache_dir)
    stable = (
        before["artifact_state_sha256"] == after["artifact_state_sha256"]
        and not missing_kernels
        and not missing_specializations
    )
    report = {
        "status": "stable" if stable else "cache-writes-detected",
        "fresh_process": True,
        "representative_branches": branch_results,
        "cache_inventory_before": before,
        "cache_inventory_after": after,
        "missing_required_kernels": missing_kernels,
        "missing_representative_specializations": missing_specializations,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return stable, report


def _verify_in_fresh_process(*, cache_dir: Path, out_dir: Path) -> dict:
    """Launch the write-free smoke check with a fresh Python/Numba runtime."""

    report_path = out_dir / ".fresh_process_cache_verification.json"
    report_path.unlink(missing_ok=True)
    environment = os.environ.copy()
    environment["NUMBA_CACHE_DIR"] = str(cache_dir)
    environment["PAYNE_ZERO_NUMBA_CACHE_DIR"] = str(cache_dir)
    environment["PAYNE_ZERO_ATMOSPHERE_PROGRESS"] = "0"
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "payne_zero_atmosphere.prewarm",
            "--verify-cache-only",
            "--verification-report",
            str(report_path),
        ),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if not report_path.is_file():
        raise RuntimeError(
            "fresh-process prewarm verification produced no report; stderr="
            + completed.stderr[-2000:]
        )
    report = json.loads(report_path.read_text())
    if completed.returncode != 0 or report.get("status") != "stable":
        raise RuntimeError(
            "fresh-process representative atmosphere calls changed the Numba "
            "cache or missed required specializations; inspect "
            f"{report_path} and rerun prewarm"
        )
    report_path.unlink(missing_ok=True)
    return report


def prewarm(*, out_dir: Path, force: bool = False) -> dict:
    cache_dir = configure_numba_cache().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "prewarm_manifest.json"
    runtime_signature = _runtime_signature(cache_dir)
    required_source_catalogs = _required_source_inventory()
    runtime_signature["required_source_catalogs"] = required_source_catalogs
    inventory_before = _cache_inventory(cache_dir)
    missing_before = _missing_required_kernel_artifacts(cache_dir)
    missing_specializations_before = _missing_representative_specializations(
        cache_dir
    )

    if not force and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if (
            existing.get("schema_version") == PREWARM_MANIFEST_SCHEMA_VERSION
            and existing.get("status") == "complete"
            and existing.get("runtime_signature") == runtime_signature
            and existing.get("cache_inventory") == inventory_before
            and existing.get("fresh_process_verification", {}).get("status")
            == "stable"
            and not missing_before
            and not missing_specializations_before
        ):
            result = {
                **existing,
                "reused": True,
                "prewarm_seconds_this_call": 0.0,
                "cache_inventory": inventory_before,
            }
            print(json.dumps(result, indent=2))
            return result

    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    representative_branches = _run_representative_branches()
    inventory_after = _cache_inventory(cache_dir)
    if inventory_after["artifact_count"] == 0:
        raise RuntimeError(f"prewarm produced no Numba artifacts in {cache_dir}")
    missing_required_kernels = _missing_required_kernel_artifacts(cache_dir)
    missing_representative_specializations = (
        _missing_representative_specializations(cache_dir)
    )
    if missing_required_kernels or missing_representative_specializations:
        raise RuntimeError(
            "prewarm did not exercise required cached kernels: "
            + ", ".join(
                missing_required_kernels + missing_representative_specializations
            )
        )
    fresh_process_verification = _verify_in_fresh_process(
        cache_dir=cache_dir,
        out_dir=out_dir,
    )
    verified_inventory = _cache_inventory(cache_dir)
    if verified_inventory != inventory_after:
        raise RuntimeError(
            "fresh-process verification changed the compiled cache despite a "
            "stable report"
        )
    seconds = time.perf_counter() - started
    result = {
        "schema_version": PREWARM_MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_signature": runtime_signature,
        "representative_branches": representative_branches,
        "representative_branch_names": [
            branch["name"] for branch in representative_branches
        ],
        "representative_iterations_per_branch": 1,
        "products_written": [],
        "prewarm_seconds": seconds,
        "prewarm_seconds_this_call": seconds,
        "reused": False,
        "cache_inventory_before": inventory_before,
        "cache_inventory": verified_inventory,
        "required_kernel_artifacts_complete": True,
        "required_representative_specializations": (
            REQUIRED_REPRESENTATIVE_SPECIALIZATIONS
        ),
        "fresh_process_verification": fresh_process_verification,
        "required_source_catalogs": required_source_catalogs,
        "numba_threads": int(numba.get_num_threads()),
        "logical_cpu_count": os.cpu_count(),
    }
    manifest_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-cache-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--verification-report", type=Path, help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    if args.verify_cache_only:
        if args.verification_report is None:
            parser.error("--verify-cache-only requires --verification-report")
        stable, _ = _verify_cache_in_current_process(
            args.verification_report.expanduser().resolve()
        )
        return 0 if stable else 4
    if args.out_dir is None:
        parser.error("--out-dir is required")
    prewarm(out_dir=args.out_dir.expanduser().resolve(), force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
