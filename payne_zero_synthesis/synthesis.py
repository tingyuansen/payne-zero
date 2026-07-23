"""Engine boundary for Payne Zero spectrum synthesis.

This module connects public package calls to the synthesis pipeline:

    structured atmosphere + line data -> continuum-normalized spectrum

Any atmosphere solver can feed the native structured schema.
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

from . import equation_of_state
from . import molecular_equilibrium as molecular_equilibrium
from . import pipeline as synthesis_pipeline
from .device import resolve_runtime


def compute_mean_nuclear_mass_amu(
    elemental_abundances: np.ndarray,
    atomic_masses: np.ndarray | None = None,
) -> float:
    """Return mean mass per nucleus in atomic mass units."""

    abundances = np.asarray(elemental_abundances, np.float64)
    masses = (
        synthesis_pipeline.load_atomic_masses()
        if atomic_masses is None
        else np.asarray(atomic_masses, np.float64)
    )
    return float(np.sum(abundances * masses[:99]) / np.sum(abundances))


def build_structured_atmosphere_from_columns(
    *,
    temperature,
    column_mass,
    gas_pressure,
    electron_density,
    elemental_abundances,
    mean_nuclear_mass_amu: float | None = None,
    microturbulence=None,
    mass_density=None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    molecular_lines: bool = True,
    eos_tolerance: float = 1.0e-5,
) -> dict:
    """Build the native structured atmosphere mapping from solver columns."""

    runtime_device, runtime_dtype = resolve_runtime(device, dtype)
    eos_tables = equation_of_state.EOSTables.from_npz(
        device=runtime_device,
        dtype=runtime_dtype,
    )
    molecular_species_codes = (
        molecular_equilibrium.supported_molecular_species_codes()
        if molecular_lines
        else None
    )
    molecules_path = (
        molecular_equilibrium._default_molecule_table() if molecular_lines else None
    )
    abundances = np.asarray(elemental_abundances, np.float64)
    if mean_nuclear_mass_amu is None:
        mean_nuclear_mass_amu = compute_mean_nuclear_mass_amu(abundances)
    return synthesis_pipeline.build_structured_atmosphere_from_columns(
        temperature=temperature,
        column_mass=column_mass,
        gas_pressure=gas_pressure,
        electron_density=electron_density,
        elemental_abundances=abundances,
        mean_nuclear_mass_amu=float(mean_nuclear_mass_amu),
        microturbulence=microturbulence,
        eos_tables=eos_tables,
        electron_density_seed=electron_density,
        tol=eos_tolerance,
        atomic_masses=synthesis_pipeline.load_atomic_masses(),
        mass_density=mass_density,
        molecular_species_codes=molecular_species_codes,
        molecules_path=molecules_path,
    )


def synthesize_structured_atmosphere(
    atmosphere: str | Path | dict,
    *,
    wavelength_start_nm: float = 400.0,
    wavelength_end_nm: float = 900.0,
    resolution: float = 20000.0,
    molecular_lines: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    spectral_operator=None,
    window_invariants=None,
):
    """Run from a structured atmosphere NPZ and return ``(result, seconds)``."""

    runtime_device, runtime_dtype = resolve_runtime(device, dtype)

    start_time = time.perf_counter()
    pipeline_runner = synthesis_pipeline.SynthesisPipeline(
        atmosphere,
        source_path=None,
        wl_start_nm=wavelength_start_nm,
        wl_end_nm=wavelength_end_nm,
        resolution=resolution,
        molecular_lines=molecular_lines,
        device=runtime_device,
        dtype=runtime_dtype,
        window_invariants=window_invariants,
    )
    result = pipeline_runner.run(
        keep_slabs=False,
        spectral_operator=spectral_operator,
    )
    return result, time.perf_counter() - start_time
