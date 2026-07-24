"""Verify or install the manifest-bound Payne Zero runtime data.

Runtime arrays and atmosphere-initializer checkpoints are distributed through
Git LFS. This module verifies their exact identities before use or relocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any


DATA_ROOT_ENV = "PAYNE_ZERO_DATA_ROOT"
_SCHEMA = 1
_SHA256 = re.compile(r"[0-9a-f]{64}")
_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = _WORKSPACE_ROOT / "source_data_files/runtime_data_manifest.json"
DEFAULT_DESTINATION = _WORKSPACE_ROOT / "source_data_files"
DEFAULT_GENERATED_ASSET_MANIFEST = (
    _WORKSPACE_ROOT / "source_data_files/generated_asset_manifest.json"
)

_DEFAULT_INITIALIZER_PATHS = {
    "atmosphere_emulator/five_label/checkpoint.pt",
    "atmosphere_emulator/cno8/checkpoint.pt",
}
_DIRECT_XH_INITIALIZER_PATH = "atmosphere_emulator/direct_abundance/checkpoint.pt"
_DIRECT_XH_METADATA_PATH = "atmosphere_emulator/direct_abundance/manifest.json"
_DIRECT_XH_METADATA_SHA256 = (
    "fb59da5e6bd3f8fcba06e0c4c284137e90aab5c4e93165daa74d8ce2ae268710"
)


class RuntimeDataInstallError(RuntimeError):
    """Raised when runtime data cannot be verified or installed safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_runtime_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load and validate the public identity manifest."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeDataInstallError(f"cannot read runtime manifest: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        raise RuntimeDataInstallError(
            f"runtime manifest must be a schema-{_SCHEMA} JSON object: {path}"
        )
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeDataInstallError(f"runtime manifest has no files: {path}")

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(files):
        if not isinstance(record, dict):
            raise RuntimeDataInstallError(
                f"manifest file record {index} is not an object"
            )
        relative = record.get("path")
        expected_bytes = record.get("bytes")
        expected_sha256 = record.get("sha256")
        if not isinstance(relative, str):
            raise RuntimeDataInstallError(f"manifest file record {index} has no path")
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative
        ):
            raise RuntimeDataInstallError(f"unsafe runtime manifest path: {relative!r}")
        if relative in seen:
            raise RuntimeDataInstallError(
                f"duplicate runtime manifest path: {relative}"
            )
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise RuntimeDataInstallError(f"invalid byte count for {relative}")
        if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(
            expected_sha256
        ):
            raise RuntimeDataInstallError(f"invalid SHA-256 for {relative}")
        seen.add(relative)
        normalized.append(
            {"path": relative, "bytes": expected_bytes, "sha256": expected_sha256}
        )
    if payload.get("file_count") != len(normalized):
        raise RuntimeDataInstallError(
            "runtime manifest file_count does not match files"
        )
    if payload.get("total_bytes") != sum(record["bytes"] for record in normalized):
        raise RuntimeDataInstallError(
            "runtime manifest total_bytes does not match files"
        )
    public_metadata = payload.get("public_metadata", [])
    if not isinstance(public_metadata, list):
        raise RuntimeDataInstallError("runtime manifest public_metadata is not a list")
    normalized_metadata: list[dict[str, Any]] = []
    for index, record in enumerate(public_metadata):
        if not isinstance(record, dict):
            raise RuntimeDataInstallError(
                f"public metadata record {index} is not an object"
            )
        relative = record.get("path")
        expected_bytes = record.get("bytes")
        expected_sha256 = record.get("sha256")
        if not isinstance(relative, str):
            raise RuntimeDataInstallError(f"public metadata record {index} has no path")
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative
        ):
            raise RuntimeDataInstallError(f"unsafe runtime manifest path: {relative!r}")
        if relative in seen:
            raise RuntimeDataInstallError(
                f"duplicate runtime manifest path: {relative}"
            )
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise RuntimeDataInstallError(f"invalid byte count for {relative}")
        if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(
            expected_sha256
        ):
            raise RuntimeDataInstallError(f"invalid SHA-256 for {relative}")
        seen.add(relative)
        normalized_metadata.append(
            {"path": relative, "bytes": expected_bytes, "sha256": expected_sha256}
        )
    return {
        **payload,
        "files": normalized,
        "public_metadata": normalized_metadata,
    }


def _checked_file(root: Path, record: dict[str, Any]) -> dict[str, Any]:
    path = root / record["path"]
    if not path.is_file():
        raise RuntimeDataInstallError(f"missing runtime data file: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != record["bytes"]:
        raise RuntimeDataInstallError(
            f"runtime data size mismatch: {path}; expected {record['bytes']}, "
            f"found {actual_bytes}"
        )
    actual_sha256 = _sha256(path)
    if actual_sha256 != record["sha256"]:
        raise RuntimeDataInstallError(
            f"runtime data checksum mismatch: {path}; expected {record['sha256']}, "
            f"found {actual_sha256}"
        )
    return {
        "path": record["path"],
        "bytes": actual_bytes,
        "sha256": actual_sha256,
    }


def verify_runtime_data(
    root: Path,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    include_public_metadata: bool = True,
) -> dict[str, Any]:
    """Verify every manifest-bound runtime file under *root*."""

    manifest = load_runtime_manifest(manifest_path)
    resolved_root = root.expanduser().resolve()
    files = [_checked_file(resolved_root, record) for record in manifest["files"]]
    metadata = (
        [_checked_file(resolved_root, record) for record in manifest["public_metadata"]]
        if include_public_metadata
        else []
    )
    return {
        "status": "verified",
        "root": str(resolved_root),
        "manifest": str(manifest_path.expanduser().resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "file_count": len(files),
        "total_bytes": sum(record["bytes"] for record in files),
        "files": files,
        "public_metadata": metadata,
    }


def _normalize_source_root(root: Path) -> Path:
    resolved = root.expanduser().resolve()
    nested = resolved / "source_data_files"
    if not (resolved / "atmosphere_tables").is_dir() and nested.is_dir():
        return nested
    return resolved


def _safe_relative_path(relative: object, *, context: str) -> Path:
    if not isinstance(relative, str):
        raise RuntimeDataInstallError(f"{context} has no path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
        raise RuntimeDataInstallError(f"unsafe {context} path: {relative!r}")
    return path


def _load_initializer_asset_records(
    generated_manifest_path: Path,
    *,
    include_direct_xh: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Return the exact generated runtime assets selected for installation."""

    try:
        payload = json.loads(generated_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeDataInstallError(
            f"cannot read generated-asset manifest: {generated_manifest_path}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema") != 1:
        raise RuntimeDataInstallError(
            "generated-asset manifest must be a schema-1 JSON object: "
            f"{generated_manifest_path}"
        )
    assets = payload.get("assets")
    if not isinstance(assets, list):
        raise RuntimeDataInstallError("generated-asset manifest has no assets")
    if payload.get("asset_count") != len(assets):
        raise RuntimeDataInstallError(
            "generated-asset manifest asset_count does not match assets"
        )
    if any(
        not isinstance(record, dict)
        or not isinstance(record.get("bytes"), int)
        or isinstance(record.get("bytes"), bool)
        or record["bytes"] <= 0
        for record in assets
    ):
        raise RuntimeDataInstallError(
            "generated-asset manifest contains an invalid asset record"
        )
    if payload.get("total_bytes") != sum(record["bytes"] for record in assets):
        raise RuntimeDataInstallError(
            "generated-asset manifest total_bytes does not match assets"
        )

    wanted = set(_DEFAULT_INITIALIZER_PATHS)
    if include_direct_xh:
        wanted.add(_DIRECT_XH_INITIALIZER_PATH)
    records: dict[str, dict[str, Any]] = {}
    for index, raw_record in enumerate(assets):
        if not isinstance(raw_record, dict):
            raise RuntimeDataInstallError(
                f"generated-asset record {index} is not an object"
            )
        path = _safe_relative_path(
            raw_record.get("path"), context=f"generated-asset record {index}"
        ).as_posix()
        if path not in wanted:
            continue
        expected_delivery = (
            "optional_experimental_runtime"
            if path == _DIRECT_XH_INITIALIZER_PATH
            else "default_runtime"
        )
        expected_bytes = raw_record.get("bytes")
        expected_sha256 = raw_record.get("sha256")
        if raw_record.get("delivery") != expected_delivery:
            raise RuntimeDataInstallError(
                f"generated initializer {path} has an unexpected delivery role"
            )
        if not isinstance(expected_bytes, int) or expected_bytes <= 0:
            raise RuntimeDataInstallError(
                f"generated initializer {path} has an invalid byte count"
            )
        if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(
            expected_sha256
        ):
            raise RuntimeDataInstallError(
                f"generated initializer {path} has an invalid SHA-256"
            )
        if path in records:
            raise RuntimeDataInstallError(
                f"duplicate generated initializer path: {path}"
            )
        records[path] = {
            "path": path,
            "bytes": expected_bytes,
            "sha256": expected_sha256,
        }
    missing = wanted.difference(records)
    if missing:
        raise RuntimeDataInstallError(
            "generated-asset manifest lacks required initializer assets: "
            + ", ".join(sorted(missing))
        )

    direct_metadata: dict[str, Any] | None = None
    if include_direct_xh:
        metadata_path = generated_manifest_path.parent / _DIRECT_XH_METADATA_PATH
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeDataInstallError(
                f"cannot read direct-[X/H] initializer manifest: {metadata_path}"
            ) from exc
        direct_record = (
            metadata.get("checkpoint", {}) if isinstance(metadata, dict) else {}
        )
        selected = records[_DIRECT_XH_INITIALIZER_PATH]
        if (
            _sha256(metadata_path) != _DIRECT_XH_METADATA_SHA256
            or metadata.get("format") != "payne_zero_direct_xh_experimental_asset_v1"
            or metadata.get("automatic_dispatch") is not False
            or metadata.get("decoded_profile_is_final_atmosphere") is not False
            or metadata.get("exact_solver_after_decode_is_mandatory") is not True
            or direct_record.get("path") != "checkpoint.pt"
            or direct_record.get("sha256") != selected["sha256"]
            or direct_record.get("bytes") != selected["bytes"]
        ):
            raise RuntimeDataInstallError(
                "direct-[X/H] initializer manifest changed its required safety contract"
            )
        direct_metadata = {
            "path": _DIRECT_XH_METADATA_PATH,
            "bytes": metadata_path.stat().st_size,
            "sha256": _sha256(metadata_path),
        }
    return [records[path] for path in sorted(records)], direct_metadata


def install_initializer_assets(
    source_root: Path,
    destination_root: Path,
    *,
    generated_manifest_path: Path = DEFAULT_GENERATED_ASSET_MANIFEST,
    include_direct_xh: bool = False,
    replace: bool = False,
) -> dict[str, Any]:
    """Install hash-bound Payne Zero initializer assets into a data root.

    Only the two default checkpoint files and, when explicitly requested, the
    direct-abundance checkpoint and its safety manifest are selected.  Truth
    corpora and Kurucz-derived runtime payloads are never copied by this path.
    """

    source = _normalize_source_root(source_root)
    destination = destination_root.expanduser().resolve()
    manifest_path = generated_manifest_path.expanduser().resolve()
    records, direct_metadata = _load_initializer_asset_records(
        manifest_path, include_direct_xh=include_direct_xh
    )
    selected = records + ([direct_metadata] if direct_metadata is not None else [])

    copy_records: list[dict[str, Any]] = []
    for record in selected:
        target = destination / record["path"]
        if target.is_file():
            try:
                _checked_file(destination, record)
            except RuntimeDataInstallError:
                if not replace:
                    raise RuntimeDataInstallError(
                        "refusing to replace mismatched initializer asset without "
                        f"--replace: {target}"
                    ) from None
                copy_records.append(record)
        elif target.exists():
            raise RuntimeDataInstallError(
                f"initializer destination is not a file: {target}"
            )
        else:
            copy_records.append(record)

    for record in copy_records:
        _checked_file(source, record)

    missing_bytes = sum(record["bytes"] for record in copy_records)
    destination.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination).free < missing_bytes:
        raise RuntimeDataInstallError(
            f"insufficient free space under {destination}: need at least "
            f"{missing_bytes} bytes for initializer assets"
        )

    copied = 0
    for record in copy_records:
        target = destination / record["path"]
        source_path = source / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.payne-zero-partial-{os.getpid()}")
        try:
            shutil.copyfile(source_path, temporary)
            temporary_record = {**record, "path": temporary.name}
            _checked_file(temporary.parent, temporary_record)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        copied += 1

    installed = [_checked_file(destination, record) for record in selected]
    return {
        "status": "installed" if copied else "already_installed",
        "source_root": str(source),
        "root": str(destination),
        "generated_asset_manifest": str(manifest_path),
        "generated_asset_manifest_sha256": _sha256(manifest_path),
        "include_direct_xh": bool(include_direct_xh),
        "file_count": len(installed),
        "total_bytes": sum(record["bytes"] for record in installed),
        "copied_files": copied,
        "files": installed,
    }


def install_runtime_data(
    source_root: Path,
    destination_root: Path,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    replace: bool = False,
) -> dict[str, Any]:
    """Verify *source_root*, then atomically copy it into *destination_root*."""

    manifest = load_runtime_manifest(manifest_path)
    source = _normalize_source_root(source_root)
    destination = destination_root.expanduser().resolve()
    verify_runtime_data(
        source,
        manifest_path=manifest_path,
        include_public_metadata=False,
    )

    copy_records: list[dict[str, Any]] = []
    for record in manifest["files"]:
        target = destination / record["path"]
        if target.is_file():
            try:
                _checked_file(destination, record)
            except RuntimeDataInstallError:
                if not replace:
                    raise RuntimeDataInstallError(
                        f"refusing to replace mismatched runtime file without "
                        f"--replace: {target}"
                    ) from None
                copy_records.append(record)
        elif target.exists():
            raise RuntimeDataInstallError(
                f"runtime destination is not a file: {target}"
            )
        else:
            copy_records.append(record)

    missing_bytes = sum(record["bytes"] for record in copy_records)
    destination.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination).free < missing_bytes:
        raise RuntimeDataInstallError(
            f"insufficient free space under {destination}: need at least "
            f"{missing_bytes} bytes"
        )

    copied = 0
    for record in copy_records:
        source_path = source / record["path"]
        target = destination / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.payne-zero-partial-{os.getpid()}")
        try:
            shutil.copyfile(source_path, temporary)
            temporary_record = {**record, "path": temporary.name}
            _checked_file(temporary.parent, temporary_record)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        copied += 1

    metadata_copied = 0
    metadata_root = manifest_path.expanduser().resolve().parent
    for record in manifest["public_metadata"]:
        source_path = metadata_root / record["path"]
        _checked_file(metadata_root, record)
        target = destination / record["path"]
        if target.is_file():
            try:
                _checked_file(destination, record)
            except RuntimeDataInstallError:
                if not replace:
                    raise RuntimeDataInstallError(
                        f"refusing to replace mismatched runtime metadata without "
                        f"--replace: {target}"
                    ) from None
            else:
                continue
        elif target.exists():
            raise RuntimeDataInstallError(
                f"runtime destination is not a file: {target}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.payne-zero-partial-{os.getpid()}")
        try:
            shutil.copyfile(source_path, temporary)
            temporary_record = {**record, "path": temporary.name}
            _checked_file(temporary.parent, temporary_record)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        metadata_copied += 1

    report = verify_runtime_data(destination, manifest_path=manifest_path)
    return {
        **report,
        "status": "installed" if copied or metadata_copied else "already_installed",
        "source_root": str(source),
        "copied_files": copied,
        "copied_public_metadata_files": metadata_copied,
    }


def _default_destination() -> Path:
    configured = os.environ.get(DATA_ROOT_ENV)
    return Path(configured).expanduser() if configured else DEFAULT_DESTINATION


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser("verify", help="verify an installed tree")
    verify_parser.add_argument("--root", type=Path, default=_default_destination())

    install_parser = subparsers.add_parser(
        "install", help="verify and copy an authorized local data tree"
    )
    install_parser.add_argument("--source-root", type=Path, required=True)
    install_parser.add_argument(
        "--destination-root", type=Path, default=_default_destination()
    )
    install_parser.add_argument("--replace", action="store_true")

    initializer_parser = subparsers.add_parser(
        "install-initializers",
        help="verify and stage Payne Zero initializer checkpoints",
    )
    initializer_parser.add_argument("--source-root", type=Path, required=True)
    initializer_parser.add_argument(
        "--destination-root", type=Path, default=_default_destination()
    )
    initializer_parser.add_argument(
        "--generated-manifest",
        type=Path,
        default=DEFAULT_GENERATED_ASSET_MANIFEST,
    )
    initializer_parser.add_argument("--include-direct-xh", action="store_true")
    initializer_parser.add_argument("--replace", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "verify":
            report = verify_runtime_data(args.root, manifest_path=args.manifest)
        elif args.command == "install":
            report = install_runtime_data(
                args.source_root,
                args.destination_root,
                manifest_path=args.manifest,
                replace=args.replace,
            )
        else:
            report = install_initializer_assets(
                args.source_root,
                args.destination_root,
                generated_manifest_path=args.generated_manifest,
                include_direct_xh=args.include_direct_xh,
                replace=args.replace,
            )
    except RuntimeDataInstallError as exc:
        print(f"Payne Zero runtime-data error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
