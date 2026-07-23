"""Resolve atmosphere inputs before the physics iteration loop."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from .atmosphere_io import ModelAtmosphere
from .config import AtmosphereConfig, DEFAULT_OPACITY_FLAGS
from .microturbulence import standard_microturbulence


@dataclass(frozen=True)
class ConvectionSettings:
    enabled: bool
    mixing_length: float
    overshoot_weight: float
    zero_top_layer_count: int


@dataclass(frozen=True)
class TurbulenceSettings:
    enabled: bool
    density_coefficient: float
    density_power: float
    sound_speed_fraction: float
    constant_velocity_km_s: float


@dataclass
class RunSetup:
    """Resolved run state before EOS, opacity, transfer, and correction steps."""

    atmosphere: ModelAtmosphere
    iterations: int
    enable_convergence_stop: bool
    minimum_iterations_before_convergence: int
    required_consecutive_converged_iterations: int
    maximum_deep_layer_relative_temperature_change: float
    maximum_all_layer_relative_temperature_change: float | None
    surface_gravity_cgs: float
    opacity_flags: list[int]
    molecules_enabled: bool
    pressure_iteration_enabled: bool
    convection: ConvectionSettings
    turbulence: TurbulenceSettings
    surface_radiation_pressure_constant: float
    effective_temperature: float
    log_surface_gravity: float
    standard_rosseland_optical_depth: np.ndarray


def surface_gravity_from_atmosphere(atmosphere: ModelAtmosphere) -> float:
    """Return cgs gravity from the atmosphere's log10(g) metadata."""

    log_surface_gravity = float(atmosphere.metadata.get("log_surface_gravity", "4.44"))
    return 10.0**log_surface_gravity


def surface_radiation_pressure_constant_from_atmosphere(
    atmosphere: ModelAtmosphere,
) -> float:
    """Extract the surface radiation-pressure seed from external metadata."""

    raw = atmosphere.metadata.get("surface_radiation_pressure_line", "")
    match = re.search(r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?", raw)
    if match is None:
        return 0.0
    return float(match.group(0).replace("D", "E").replace("d", "e"))


def opacity_flags_from_atmosphere(atmosphere: ModelAtmosphere) -> list[int]:
    """Parse the 20 external-format opacity flags from atmosphere metadata."""

    values = [
        int(value)
        for value in re.findall(r"-?\d+", atmosphere.metadata.get("opacity_flags", ""))
    ]
    if len(values) >= 20:
        return values[-20:]
    return list(DEFAULT_OPACITY_FLAGS)


def standard_rosseland_optical_depth_grid(layers: int) -> np.ndarray:
    """Return the standard optical-depth grid used for microturbulence."""

    return 10.0 ** (-6.875 + np.arange(int(layers), dtype=np.float64) * 0.125)


def initialize_microturbulence(
    atmosphere: ModelAtmosphere,
    *,
    effective_temperature: float,
    log_surface_gravity: float,
    standard_rosseland_optical_depth: np.ndarray,
) -> None:
    """Fill a missing microturbulence profile from the standard prescription."""

    if not np.any(atmosphere.microturbulence > 0.0):
        atmosphere.microturbulence[:] = standard_microturbulence(
            effective_temperature=effective_temperature,
            log_surface_gravity=log_surface_gravity,
            standard_rosseland_optical_depth=standard_rosseland_optical_depth,
        )


def validate_atmosphere_seed(atmosphere: ModelAtmosphere) -> None:
    """Validate physical layer fields before starting an exact solve."""

    layer_count = atmosphere.layers
    fields = {
        "column_mass": atmosphere.column_mass,
        "temperature": atmosphere.temperature,
        "gas_pressure": atmosphere.gas_pressure,
        "electron_density": atmosphere.electron_density,
        "rosseland_opacity": atmosphere.rosseland_opacity,
        "radiative_acceleration": atmosphere.radiative_acceleration,
        "microturbulence": atmosphere.microturbulence,
        "convective_flux": atmosphere.convective_flux,
        "convective_velocity": atmosphere.convective_velocity,
    }
    for name, values in fields.items():
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (layer_count,) or not np.all(np.isfinite(array)):
            raise ValueError(
                f"atmosphere seed {name} must be a finite ({layer_count},) array"
            )
    for name in (
        "column_mass",
        "temperature",
        "gas_pressure",
        "electron_density",
        "rosseland_opacity",
    ):
        if np.any(np.asarray(fields[name], dtype=np.float64) <= 0.0):
            raise ValueError(f"atmosphere seed {name} must be strictly positive")
    if np.any(np.diff(np.asarray(atmosphere.column_mass, dtype=np.float64)) <= 0.0):
        raise ValueError("atmosphere seed column_mass must be strictly increasing")
    if np.any(np.asarray(atmosphere.microturbulence, dtype=np.float64) < 0.0):
        raise ValueError("atmosphere seed microturbulence must be non-negative")


def resolve_run_setup(config: AtmosphereConfig) -> RunSetup:
    """Validate and normalize the structured state needed before iteration."""

    atmosphere = config.inputs.initial_atmosphere
    validate_atmosphere_seed(atmosphere)

    iterations = max(1, int(config.iterations))

    enable_convergence_stop = bool(config.enable_convergence_stop)
    maximum_deep_layer_relative_temperature_change = float(
        config.maximum_deep_layer_relative_temperature_change
    )
    maximum_all_layer_relative_temperature_change = (
        None
        if config.maximum_all_layer_relative_temperature_change is None
        else float(config.maximum_all_layer_relative_temperature_change)
    )
    if maximum_deep_layer_relative_temperature_change <= 0.0:
        raise ValueError("maximum deep-layer temperature change must be positive")
    if (
        maximum_all_layer_relative_temperature_change is not None
        and maximum_all_layer_relative_temperature_change <= 0.0
    ):
        raise ValueError("maximum all-layer temperature change must be positive")

    surface_gravity_cgs = surface_gravity_from_atmosphere(atmosphere)
    opacity_flags = opacity_flags_from_atmosphere(atmosphere)
    molecules_enabled = bool(config.enable_molecules)

    pressure_iteration_enabled = True
    if atmosphere.metadata.get("pressure_iteration_enabled") is not None:
        pressure_iteration_enabled = bool(
            int(atmosphere.metadata["pressure_iteration_enabled"])
        )

    convection = ConvectionSettings(
        enabled=bool(config.enable_convection),
        mixing_length=1.25,
        overshoot_weight=0.0,
        zero_top_layer_count=0,
    )
    turbulence = TurbulenceSettings(
        enabled=False,
        density_coefficient=0.0,
        density_power=0.0,
        sound_speed_fraction=0.0,
        constant_velocity_km_s=0.0,
    )

    effective_temperature = float(
        atmosphere.metadata.get("effective_temperature", 5778.0)
    )
    log_surface_gravity = float(atmosphere.metadata.get("log_surface_gravity", 4.44))
    standard_rosseland_optical_depth = standard_rosseland_optical_depth_grid(
        atmosphere.layers
    )
    initialize_microturbulence(
        atmosphere,
        effective_temperature=effective_temperature,
        log_surface_gravity=log_surface_gravity,
        standard_rosseland_optical_depth=standard_rosseland_optical_depth,
    )

    return RunSetup(
        atmosphere=atmosphere,
        iterations=iterations,
        enable_convergence_stop=enable_convergence_stop,
        minimum_iterations_before_convergence=max(
            1, int(config.minimum_iterations_before_convergence)
        ),
        required_consecutive_converged_iterations=max(
            1, int(config.required_consecutive_converged_iterations)
        ),
        maximum_deep_layer_relative_temperature_change=maximum_deep_layer_relative_temperature_change,
        maximum_all_layer_relative_temperature_change=(
            maximum_all_layer_relative_temperature_change
        ),
        surface_gravity_cgs=surface_gravity_cgs,
        opacity_flags=opacity_flags,
        molecules_enabled=molecules_enabled,
        pressure_iteration_enabled=pressure_iteration_enabled,
        convection=convection,
        turbulence=turbulence,
        surface_radiation_pressure_constant=surface_radiation_pressure_constant_from_atmosphere(
            atmosphere
        ),
        effective_temperature=effective_temperature,
        log_surface_gravity=log_surface_gravity,
        standard_rosseland_optical_depth=standard_rosseland_optical_depth,
    )
