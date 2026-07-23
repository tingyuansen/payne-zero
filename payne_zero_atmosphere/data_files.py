"""Resolution of the workspace data home for the atmosphere solver.

All data for both Payne Zero packages lives in one workspace directory,
``source_data_files/``, next to the two code packages. This package reads:

- ``atmosphere_tables/``   packed physics tables (NPZ)
- ``atmosphere_emulator/`` warm-start emulator weights and normalization
- ``source_catalogs/``     full-range raw source catalogs (see
  ``source_catalogs.py`` for their dedicated resolver and env override)

``PAYNE_ZERO_DATA_ROOT`` overrides the data home as a whole.
"""

from __future__ import annotations

import os
from pathlib import Path


DATA_ROOT_ENV = "PAYNE_ZERO_DATA_ROOT"

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def data_root() -> Path:
    """The workspace data home holding all bundled data for both packages."""
    value = os.environ.get(DATA_ROOT_ENV)
    if value:
        return Path(value).expanduser()
    return _WORKSPACE_ROOT / "source_data_files"


def atmosphere_table_dir() -> Path:
    """Directory of the packed atmosphere physics tables."""
    return data_root() / "atmosphere_tables"


def atmosphere_table_path(file_name: str) -> Path:
    """Path of one packed atmosphere physics table."""
    return atmosphere_table_dir() / file_name


def atmosphere_emulator_dir() -> Path:
    """Directory of the warm-start emulator assets."""
    return data_root() / "atmosphere_emulator"


def load_table_arrays(
    path: Path,
    required_keys: tuple[str, ...],
    error_type: type[Exception] = RuntimeError,
) -> dict:
    """Load a packed table NPZ, insisting on the required keys."""
    import numpy as np

    if not path.exists():
        raise error_type(f"Missing packaged table: {path}")
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in required_keys if key not in data.files]
        if missing:
            raise error_type(
                f"{path.name} is missing required keys: {', '.join(missing)}"
            )
        return {key: np.asarray(data[key]) for key in required_keys}
