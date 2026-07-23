"""Structured atmosphere-solver inputs, outputs, and controls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .atmosphere_io import ModelAtmosphere


@dataclass(frozen=True)
class AtmosphereInput:
    """Input sources for one atmosphere solve."""

    initial_atmosphere: ModelAtmosphere
    molecules_path: Path | None = None
    selected_line_catalog_path: Path | None = None
    detailed_line_catalog_path: Path | None = None
    predicted_atomic_lines_path: Path | None = None
    observed_atomic_lines_path: Path | None = None
    high_excitation_lines_path: Path | None = None
    diatomic_lines_path: Path | None = None
    titanium_oxide_lines_path: Path | None = None
    water_lines_path: Path | None = None
    h3plus_lines_path: Path | None = None


@dataclass(frozen=True)
class AtmosphereOutput:
    """Output targets produced by one atmosphere solve."""

    structured_atmosphere_path: Path | None = None
    diagnostics_path: Path | None = None
    debug_state_path: Path | None = None


@dataclass(frozen=True)
class AtmosphereConfig:
    """Top-level atmosphere-solver configuration."""

    inputs: AtmosphereInput
    outputs: AtmosphereOutput
    iterations: int = 1
    enable_molecules: bool = False
    enable_convection: bool = True
    enable_convergence_stop: bool = False
    minimum_iterations_before_convergence: int = 3
    required_consecutive_converged_iterations: int = 1
    maximum_deep_layer_relative_temperature_change: float = 5.0e-4
    maximum_all_layer_relative_temperature_change: float | None = None
    molecular_convection_thermal_tracks_perturbation: bool = True


DEFAULT_OPACITY_FLAGS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0]
