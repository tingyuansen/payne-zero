"""Runtime paths for Payne Zero synthesis data, caches, and source rebuilds.

All data for both Payne Zero packages lives in one workspace directory,
``source_data_files/``, next to the two code packages:

- ``synthesis_tables/``       physics tables consumed by this package
- ``source_catalogs/``        full-range raw source catalogs (rebuild inputs)

``PAYNE_ZERO_DATA_ROOT`` overrides the data home as a whole;
``PAYNE_ZERO_SYNTHESIS_SOURCE_CATALOG_ROOT`` (or the shared
``PAYNE_ZERO_SOURCE_CATALOG_ROOT``) overrides only the raw source catalogs.
"""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent

DATA_ROOT_ENV = "PAYNE_ZERO_DATA_ROOT"


def _env_path(*names: str, default: Path) -> Path:
    for name in names:
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser()
    return default.expanduser()


def data_root() -> Path:
    """The workspace data home holding all bundled data for both packages."""
    return _env_path(DATA_ROOT_ENV, default=WORKSPACE_ROOT / "source_data_files")


SYNTHESIS_TABLE_DIR = data_root() / "synthesis_tables"

_DEFAULT_CACHE_ROOT = (
    WORKSPACE_ROOT / ".cache" / "payne-zero" / "synthesis"
    if (WORKSPACE_ROOT / "pyproject.toml").is_file()
    else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    / "payne-zero"
    / "synthesis"
)
PACKAGE_CACHE_ROOT = _env_path(
    "PAYNE_ZERO_SYNTHESIS_CACHE_DIR",
    default=_DEFAULT_CACHE_ROOT,
)

SOURCE_CATALOG_ENV = "PAYNE_ZERO_SYNTHESIS_SOURCE_CATALOG_ROOT"
SHARED_SOURCE_CATALOG_ENV = "PAYNE_ZERO_SOURCE_CATALOG_ROOT"


def source_catalog_root() -> Path:
    """Return the raw source-catalog root, or fail with context.

    Resolution order: package-specific env var, shared workspace env var,
    then the bundled workspace data home.
    """
    for name in (SOURCE_CATALOG_ENV, SHARED_SOURCE_CATALOG_ENV):
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser()
    bundled = data_root() / "source_catalogs"
    if bundled.is_dir():
        return bundled
    raise FileNotFoundError(
        f"Source catalogs not found: no bundled tree at {bundled} and neither "
        f"{SOURCE_CATALOG_ENV} nor {SHARED_SOURCE_CATALOG_ENV} is set. Source "
        "catalogs are needed only when rebuilding catalogs or atmospheres "
        "from source inputs."
    )


def source_catalog_path(*parts: str) -> Path:
    """Path inside the raw source-catalog tree."""
    return source_catalog_root().joinpath(*parts)
