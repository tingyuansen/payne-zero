"""Stable Numba-cache location shared by all atmosphere kernel modules."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def default_numba_cache_dir() -> Path:
    """Return a persistent, discoverable cache path for this installation."""

    package_parent = Path(__file__).resolve().parent.parent
    if (package_parent / "pyproject.toml").is_file():
        return package_parent / ".cache" / "payne-zero" / "numba-atmosphere"
    cache_home = Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    ).expanduser()
    return cache_home / "payne-zero" / "numba-atmosphere"


def configure_numba_cache() -> Path:
    """Configure the persistent cache before or after importing Numba."""

    existing = os.environ.get("NUMBA_CACHE_DIR")
    if existing:
        cache_dir = Path(existing).expanduser()
    else:
        requested = os.environ.get("PAYNE_ZERO_NUMBA_CACHE_DIR")
        cache_dir = (
            Path(requested).expanduser() if requested else default_numba_cache_dir()
        )
        os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)
    # Numba reads environment configuration during import.  Updating its live
    # setting as well makes ``import numba; import payne_zero_atmosphere`` use
    # the same persistent cache as the normal package-first import order.
    if "numba" in sys.modules:
        from numba import config as numba_config

        numba_config.CACHE_DIR = str(cache_dir)
    return cache_dir
