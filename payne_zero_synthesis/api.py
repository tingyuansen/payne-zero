"""Public structured-atmosphere spectrum-synthesis API.

`synthesize` consumes the native NPZ product. `build_structured_atmosphere`
and `save_structured_atmosphere` construct that product from solver columns.
Everything else in the package is engine machinery behind these calls.

Repeated calls over the same (window, resolution, device, dtype) reuse the
device-resident window invariants through an in-process cache keyed only by
those physical inputs plus catalog/table file identity; disable it with
PAYNE_ZERO_SYNTHESIS_DISABLE_INVARIANT_CACHE=1 (identical outputs, slower) or
drop it with `clear_window_invariant_cache()`.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import torch

from . import synthesis as _synthesis_engine
from .atmosphere import (
    ATMOSPHERE_SCHEMA_VERSION,
    load_atmosphere_npz,
    validate_atmosphere_npz,
)
from .pipeline import (  # noqa: F401 - public re-exports
    clear_window_invariant_cache,
    window_invariant_cache_enabled,
)


SPEED_OF_LIGHT_NM_S = 2.99792458e17
FOUR_PI = 4.0 * np.pi


@dataclass(frozen=True)
class Spectrum:
    """Spectrum and timing on a wavelength grid.

    ``flux_total`` and ``flux_continuum`` are spectral flux densities per
    nanometer. ``normalized_flux`` is their dimensionless ratio.
    """

    wavelength_nm: np.ndarray
    flux_total: np.ndarray
    flux_continuum: np.ndarray
    normalized_flux: np.ndarray
    seconds: float

    def save_npz(self, path: str | Path) -> None:
        output_path = Path(path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            wavelength_nm=self.wavelength_nm,
            flux_total=self.flux_total,
            flux_continuum=self.flux_continuum,
            normalized_flux=self.normalized_flux,
            seconds=np.asarray([self.seconds], np.float64),
        )


def _torch_dtype(name: str | torch.dtype | None) -> torch.dtype | None:
    if name is None:
        return None
    if isinstance(name, torch.dtype):
        return name
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype {name!r}")


def _surface_flux_per_wavelength_nm(
    wavelength_nm: np.ndarray,
    eddington_flux_per_frequency: np.ndarray,
) -> np.ndarray:
    """Convert Eddington ``H_nu`` to physical surface ``F_lambda`` per nm."""

    wavelength_nm_array = np.asarray(wavelength_nm, np.float64)
    if np.any(wavelength_nm_array <= 0.0):
        raise ValueError("wavelength_nm must be strictly positive")
    return (
        FOUR_PI
        * np.asarray(eddington_flux_per_frequency, np.float64)
        * SPEED_OF_LIGHT_NM_S
        / np.square(wavelength_nm_array)
    )


def _wrap(result, seconds: float) -> Spectrum:
    wavelength_nm = np.asarray(result.wavelength_nm, np.float64)
    return Spectrum(
        wavelength_nm=wavelength_nm,
        flux_total=_surface_flux_per_wavelength_nm(
            wavelength_nm, result.eddington_flux_total_per_frequency
        ),
        flux_continuum=_surface_flux_per_wavelength_nm(
            wavelength_nm, result.eddington_flux_continuum_per_frequency
        ),
        normalized_flux=np.asarray(result.normalized_flux, np.float64),
        seconds=float(seconds),
    )


def build_structured_atmosphere(
    *,
    temperature,
    column_mass,
    gas_pressure,
    electron_density,
    elemental_abundances,
    mean_nuclear_mass_amu: float | None = None,
    microturbulence=None,
    mass_density=None,
    molecular_lines: bool = True,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    eos_tolerance: float = 1.0e-5,
) -> dict[str, np.ndarray]:
    """Build the native atmosphere mapping from solver depth columns.

    Use this when an atmosphere solver already has converged thermodynamic
    columns and needs to hand them to Payne Zero synthesis without writing an
    intermediate text deck. The returned mapping uses the public field names
    documented in ``atmosphere_schema.json``.
    """

    return _synthesis_engine.build_structured_atmosphere_from_columns(
        temperature=temperature,
        column_mass=column_mass,
        gas_pressure=gas_pressure,
        electron_density=electron_density,
        elemental_abundances=elemental_abundances,
        mean_nuclear_mass_amu=mean_nuclear_mass_amu,
        microturbulence=microturbulence,
        mass_density=mass_density,
        device=device,
        dtype=_torch_dtype(dtype),
        molecular_lines=molecular_lines,
        eos_tolerance=eos_tolerance,
    )


def save_structured_atmosphere(
    atmosphere: dict[str, np.ndarray],
    path: str | Path,
) -> tuple[str, ...]:
    """Write and validate a native structured atmosphere NPZ."""

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(value) for name, value in atmosphere.items()}
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    output_path.unlink(missing_ok=True)
    temporary_path.unlink(missing_ok=True)
    try:
        with temporary_path.open("wb") as handle:
            np.savez(handle, **arrays)
        canonical_arrays = load_atmosphere_npz(temporary_path)
        canonical_arrays["atmosphere_schema_version"] = np.asarray(
            [ATMOSPHERE_SCHEMA_VERSION], dtype=np.int32
        )
        with temporary_path.open("wb") as handle:
            np.savez(handle, **canonical_arrays)
        names = validate_atmosphere_npz(temporary_path)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return names


def synthesize(
    atmosphere_npz: str | Path,
    *,
    wavelength_start_nm: float = 400.0,
    wavelength_end_nm: float = 900.0,
    resolution: float = 20000.0,
    molecular_lines: bool = True,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    spectral_operator=None,
) -> Spectrum:
    """Synthesize a spectrum from a native structured atmosphere NPZ.

    ``spectral_operator`` is an optional prepared device-resident operator that
    transforms wavelength-density total and continuum flux before their one
    final host transfer.
    """

    atmosphere = load_atmosphere_npz(atmosphere_npz)
    result, seconds = _synthesis_engine.synthesize_structured_atmosphere(
        atmosphere,
        wavelength_start_nm=wavelength_start_nm,
        wavelength_end_nm=wavelength_end_nm,
        resolution=resolution,
        molecular_lines=molecular_lines,
        device=device,
        dtype=_torch_dtype(dtype),
        spectral_operator=spectral_operator,
    )
    return _wrap(result, seconds)
