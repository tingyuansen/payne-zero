#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
export PAYNE_ZERO_NUMBA_CACHE_DIR="${PAYNE_ZERO_NUMBA_CACHE_DIR:-$REPO_DIR/.cache/payne-zero/numba-atmosphere}"
export PAYNE_ZERO_SYNTHESIS_CACHE_DIR="${PAYNE_ZERO_SYNTHESIS_CACHE_DIR:-$REPO_DIR/.cache/payne-zero/synthesis}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}"
export PAYNE_ZERO_ATMOSPHERE_PROGRESS="${PAYNE_ZERO_ATMOSPHERE_PROGRESS:-1}"
PREWARM_R_GRID="${PAYNE_ZERO_PREWARM_R_GRID:-${PAYNE_ZERO_PREWARM_RESOLUTION:-}}"
DATA_ROOT_INPUT="${PAYNE_ZERO_DATA_ROOT:-$REPO_DIR/source_data_files}"
INCLUDE_DIRECT_XH="${PAYNE_ZERO_INCLUDE_DIRECT_XH:-${PAYNE_ZERO_INCLUDE_EXPERIMENTAL_DIRECT_XH:-0}}"

NUMBA_CACHE_INPUT="${NUMBA_CACHE_DIR:-$PAYNE_ZERO_NUMBA_CACHE_DIR}"
RESOLVED_NUMBA_CACHE_DIR="$("$PYTHON" -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$NUMBA_CACHE_INPUT")"
RESOLVED_SYNTHESIS_CACHE_DIR="$("$PYTHON" -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$PAYNE_ZERO_SYNTHESIS_CACHE_DIR")"
RESOLVED_DATA_ROOT="$("$PYTHON" -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$DATA_ROOT_INPUT")"
export NUMBA_CACHE_DIR="$RESOLVED_NUMBA_CACHE_DIR"
export PAYNE_ZERO_NUMBA_CACHE_DIR="$RESOLVED_NUMBA_CACHE_DIR"
export PAYNE_ZERO_SYNTHESIS_CACHE_DIR="$RESOLVED_SYNTHESIS_CACHE_DIR"
export PAYNE_ZERO_DATA_ROOT="$RESOLVED_DATA_ROOT"

cd "$REPO_DIR"
mkdir -p "$RESOLVED_NUMBA_CACHE_DIR" "$REPO_DIR/.cache/payne-zero/prewarm-atmosphere"

runtime_manifest="$REPO_DIR/source_data_files/runtime_data_manifest.json"
runtime_installer="$REPO_DIR/payne_zero_atmosphere/install_runtime_data.py"
if [[ -n "${PAYNE_ZERO_RUNTIME_DATA_SOURCE:-}" ]]; then
    echo "[payne-zero installer] installing authorized local runtime data"
    "$PYTHON" "$runtime_installer" --manifest "$runtime_manifest" install \
        --source-root "$PAYNE_ZERO_RUNTIME_DATA_SOURCE" \
        --destination-root "$RESOLVED_DATA_ROOT"
else
    echo "[payne-zero installer] verifying authorized local runtime data"
    if ! "$PYTHON" "$runtime_installer" --manifest "$runtime_manifest" verify \
        --root "$RESOLVED_DATA_ROOT"; then
        echo "Set PAYNE_ZERO_RUNTIME_DATA_SOURCE to a complete local runtime-data tree." >&2
        echo "See source_data_files/README.md." >&2
        exit 2
    fi
fi

if [[ "${PAYNE_ZERO_SKIP_LFS_PULL:-0}" != "1" ]]; then
    if git -C "$REPO_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        if ! git lfs version >/dev/null 2>&1; then
            echo "Git LFS is required to install Payne Zero initializer assets." >&2
            echo "Install Git LFS, then rerun ./install.sh." >&2
            exit 2
        fi
        direct_xh_checkpoint="source_data_files/atmosphere_emulator/direct_xh_experimental/checkpoint.pt"
        runtime_lfs_include="source_data_files/atmosphere_emulator/five_label/checkpoint.pt"
        runtime_lfs_include+=",source_data_files/atmosphere_emulator/cno8/checkpoint.pt"
        lfs_exclude="$direct_xh_checkpoint"
        if [[ "$INCLUDE_DIRECT_XH" == "1" ]]; then
            runtime_lfs_include+=",$direct_xh_checkpoint"
            lfs_exclude=""
        fi
        lfs_include="$runtime_lfs_include"
        echo "[payne-zero installer] downloading Payne Zero initializer assets: $lfs_include"
        git lfs install --local
        git lfs pull --include="$lfs_include" --exclude="$lfs_exclude"
    else
        echo "[payne-zero installer] no Git metadata; validating pre-staged runtime data"
    fi
else
    echo "[payne-zero installer] Git LFS pull skipped; validating pre-staged runtime data"
fi
initializer_args=(
    install-initializers
    --generated-manifest "$REPO_DIR/source_data_files/generated_asset_manifest.json"
    --source-root "$REPO_DIR/source_data_files"
    --destination-root "$RESOLVED_DATA_ROOT"
)
if [[ "$INCLUDE_DIRECT_XH" == "1" ]]; then
    initializer_args+=(--include-direct-xh)
fi
echo "[payne-zero installer] staging hash-verified initializer assets"
"$PYTHON" "$runtime_installer" "${initializer_args[@]}"
if [[ "${PAYNE_ZERO_SKIP_PIP_INSTALL:-0}" != "1" ]]; then
    echo "[payne-zero installer] installing the editable package"
    "$PYTHON" -m pip install -e . --no-build-isolation
fi
if [[ "$INCLUDE_DIRECT_XH" == "1" ]]; then
    echo "[payne-zero installer] validating direct-[X/H] initializer"
    "$PYTHON" -c 'from payne_zero_atmosphere.direct_abundance import load_direct_abundance_initializer; load_direct_abundance_initializer(enable_experimental=True, device="cpu")'
fi
if [[ -z "$PREWARM_R_GRID" \
    && -z "${PAYNE_ZERO_PREWARM_WAVELENGTH_START_NM:-}" \
    && -z "${PAYNE_ZERO_PREWARM_WAVELENGTH_END_NM:-}" ]]; then
    echo "[payne-zero installer] building synthesis cache: 400-900 nm, R_grid=20000"
    "$PYTHON" -m payne_zero_synthesis.prewarm \
        --wavelength-start-nm 400 --wavelength-end-nm 900 \
        --r-grid 20000 "$@"
fi
if [[ -n "$PREWARM_R_GRID" \
    || -n "${PAYNE_ZERO_PREWARM_WAVELENGTH_START_NM:-}" \
    || -n "${PAYNE_ZERO_PREWARM_WAVELENGTH_END_NM:-}" ]]; then
    echo "[payne-zero installer] building requested synthesis cache"
    "$PYTHON" -m payne_zero_synthesis.prewarm \
        --wavelength-start-nm "${PAYNE_ZERO_PREWARM_WAVELENGTH_START_NM:-400}" \
        --wavelength-end-nm "${PAYNE_ZERO_PREWARM_WAVELENGTH_END_NM:-900}" \
        --r-grid "${PREWARM_R_GRID:-20000}" \
        "$@"
fi
echo "[payne-zero installer] validating full catalogs and prewarming hot/Sun/giant atmosphere branches"
"$PYTHON" -m payne_zero_atmosphere.prewarm \
    --out-dir "$REPO_DIR/.cache/payne-zero/prewarm-atmosphere" \
    "$@"

echo "Payne Zero installed; persistent Numba cache: $RESOLVED_NUMBA_CACHE_DIR"
echo "Payne Zero synthesis cache: $RESOLVED_SYNTHESIS_CACHE_DIR"
