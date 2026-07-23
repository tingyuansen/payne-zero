"""Convection and thermodynamic-gradient helpers for the atmosphere runner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .temperature_correction import _signed_floor

from .continuum_opacity import RosselandOpacityTable, evaluate_rosseland_opacity
from .radiative_transfer import (
    differentiate_on_depth_grid,
    integrate_on_depth_grid,
    remap_to_grid,
)


@dataclass(frozen=True)
class ConvectionResult:
    """Outputs from one convection evaluation."""

    geometric_depth_below_surface_km: np.ndarray
    logarithmic_temperature_pressure_gradient: np.ndarray
    heat_capacity: np.ndarray
    log_density_temperature_derivative_at_constant_total_pressure: np.ndarray
    sound_speed: np.ndarray
    adiabatic_gradient: np.ndarray
    pressure_scale_height: np.ndarray
    convective_flux: np.ndarray
    convective_velocity: np.ndarray
    raw_convective_flux: np.ndarray
    overshoot_convective_flux: np.ndarray


@dataclass(frozen=True)
class ConvectionFiniteDifferenceSamples:
    """Equation-of-state perturbation samples consumed by convection."""

    specific_internal_energy_plus_temperature: np.ndarray
    specific_internal_energy_minus_temperature: np.ndarray
    specific_internal_energy_plus_pressure: np.ndarray
    specific_internal_energy_minus_pressure: np.ndarray
    density_plus_temperature: np.ndarray
    density_minus_temperature: np.ndarray
    density_plus_pressure: np.ndarray
    density_minus_pressure: np.ndarray


@dataclass(frozen=True)
class DisabledConvectionDiagnostics:
    """Endpoint diagnostics retained when convection is disabled."""

    convective_flux: np.ndarray
    convective_velocity: np.ndarray


def integrate_geometric_depth_below_surface_km(
    *,
    column_mass: np.ndarray,
    mass_density: np.ndarray,
) -> np.ndarray:
    """Integrate inward geometric depth from column mass and density."""

    density = np.asarray(mass_density, dtype=np.float64)
    reciprocal_density = 1.0e-5 / np.maximum(density, 1.0e-300)
    return integrate_on_depth_grid(
        np.asarray(column_mass, dtype=np.float64),
        reciprocal_density,
        surface_value=0.0,
    )


def compute_convection(
    *,
    rosseland_table: RosselandOpacityTable,
    column_mass: np.ndarray,
    rosseland_optical_depth: np.ndarray,
    temperature_k: np.ndarray,
    gas_pressure: np.ndarray,
    mass_density: np.ndarray,
    rosseland_opacity: np.ndarray,
    microturbulence: np.ndarray,
    absolute_radiation_pressure: np.ndarray,
    total_pressure: np.ndarray,
    surface_gravity_cgs: float,
    target_integrated_eddington_flux: float,
    mixing_length: float = 1.0,
    overshoot_weight: float = 1.0,
    convection_enabled: bool | int = True,
    zero_top_layer_count: int = 36,
    specific_internal_energy_plus_temperature: np.ndarray | None = None,
    specific_internal_energy_minus_temperature: np.ndarray | None = None,
    specific_internal_energy_plus_pressure: np.ndarray | None = None,
    specific_internal_energy_minus_pressure: np.ndarray | None = None,
    density_plus_temperature: np.ndarray | None = None,
    density_minus_temperature: np.ndarray | None = None,
    density_plus_pressure: np.ndarray | None = None,
    density_minus_pressure: np.ndarray | None = None,
) -> ConvectionResult:
    """Evaluate convection for one atmospheric iteration.

    The finite-difference arrays are the EOS perturbation samples used by the
    validated reference branch. When omitted, the calculation uses the
    ideal-gas derivative path also used by the disabled-convection diagnostic.
    """

    column_mass = np.asarray(column_mass, dtype=np.float64)
    optical_depth = np.asarray(rosseland_optical_depth, dtype=np.float64)
    temperature = np.asarray(temperature_k, dtype=np.float64)
    pressure = np.asarray(gas_pressure, dtype=np.float64)
    density = np.asarray(mass_density, dtype=np.float64)
    opacity = np.asarray(rosseland_opacity, dtype=np.float64)
    _ = np.asarray(microturbulence, dtype=np.float64)
    absolute_radiation_pressure = np.asarray(
        absolute_radiation_pressure,
        dtype=np.float64,
    )
    pressure_total = np.asarray(total_pressure, dtype=np.float64)

    layer_count = int(temperature.size)
    temperature_derivative = differentiate_on_depth_grid(column_mass, temperature)
    dilution = 1.0 - np.exp(-optical_depth)

    logarithmic_gradient = np.zeros(layer_count, dtype=np.float64)
    heat_capacity = np.zeros(layer_count, dtype=np.float64)
    log_density_temperature_derivative_at_constant_total_pressure = np.zeros(
        layer_count, dtype=np.float64
    )
    sound_speed = np.zeros(layer_count, dtype=np.float64)
    adiabatic_gradient = np.zeros(layer_count, dtype=np.float64)
    pressure_scale_height = np.zeros(layer_count, dtype=np.float64)
    convective_flux = np.zeros(layer_count, dtype=np.float64)
    convective_velocity = np.zeros(layer_count, dtype=np.float64)
    raw_convective_flux = np.zeros(layer_count, dtype=np.float64)
    overshoot_convective_flux = np.zeros(layer_count, dtype=np.float64)
    temperature_step = np.zeros(layer_count, dtype=np.float64)
    local_rosseland = np.zeros(layer_count, dtype=np.float64)

    fd_arrays = (
        specific_internal_energy_plus_temperature,
        specific_internal_energy_minus_temperature,
        specific_internal_energy_plus_pressure,
        specific_internal_energy_minus_pressure,
        density_plus_temperature,
        density_minus_temperature,
        density_plus_pressure,
        density_minus_pressure,
    )
    use_finite_difference = all(array is not None for array in fd_arrays)
    if use_finite_difference:
        edens_t_plus = np.asarray(
            specific_internal_energy_plus_temperature, dtype=np.float64
        )
        edens_t_minus = np.asarray(
            specific_internal_energy_minus_temperature, dtype=np.float64
        )
        edens_p_plus = np.asarray(
            specific_internal_energy_plus_pressure, dtype=np.float64
        )
        edens_p_minus = np.asarray(
            specific_internal_energy_minus_pressure, dtype=np.float64
        )
        rho_t_plus = np.asarray(density_plus_temperature, dtype=np.float64)
        rho_t_minus = np.asarray(density_minus_temperature, dtype=np.float64)
        rho_p_plus = np.asarray(density_plus_pressure, dtype=np.float64)
        rho_p_minus = np.asarray(density_minus_pressure, dtype=np.float64)

    for layer_index in range(layer_count):
        superadiabatic_gradient = 0.0
        if use_finite_difference:
            energy_temperature_derivative = (
                (edens_t_plus[layer_index] - edens_t_minus[layer_index])
                / np.maximum(temperature[layer_index], 1.0e-300)
                * 500.0
            )
            density_temperature_derivative = (
                (rho_t_plus[layer_index] - rho_t_minus[layer_index])
                / np.maximum(temperature[layer_index], 1.0e-300)
                * 500.0
            )
            energy_pressure_derivative = (
                (edens_p_plus[layer_index] - edens_p_minus[layer_index])
                / np.maximum(pressure[layer_index], 1.0e-300)
                * 500.0
            )
            density_pressure_derivative = (
                (rho_p_plus[layer_index] - rho_p_minus[layer_index])
                / np.maximum(pressure[layer_index], 1.0e-300)
                * 500.0
            )
        else:
            gas_constant = pressure[layer_index] / np.maximum(
                density[layer_index] * temperature[layer_index],
                1.0e-300,
            )
            energy_temperature_derivative = 1.5 * gas_constant
            density_temperature_derivative = -density[layer_index] / np.maximum(
                temperature[layer_index],
                1.0e-300,
            )
            energy_pressure_derivative = 0.0
            density_pressure_derivative = density[layer_index] / np.maximum(
                pressure[layer_index],
                1.0e-300,
            )

        pressure_derivative_pressure = 1.0
        pressure_temperature_derivative = (
            4.0
            * absolute_radiation_pressure[layer_index]
            / np.maximum(temperature[layer_index], 1.0e-300)
            * dilution[layer_index]
        )
        density_pressure_safe = _signed_floor(float(density_pressure_derivative))
        constant_volume_heat_capacity = (
            energy_temperature_derivative
            - energy_pressure_derivative
            * density_temperature_derivative
            / density_pressure_safe
        )
        heat_capacity[layer_index] = (
            energy_temperature_derivative
            - energy_pressure_derivative
            * pressure_temperature_derivative
            / np.maximum(pressure_derivative_pressure, 1.0e-300)
            - pressure_total[layer_index]
            / np.maximum(density[layer_index] ** 2, 1.0e-300)
            * (
                density_temperature_derivative
                - density_pressure_derivative
                * pressure_temperature_derivative
                / np.maximum(pressure_derivative_pressure, 1.0e-300)
            )
        )
        if constant_volume_heat_capacity > 0.0:
            sound_speed[layer_index] = np.sqrt(
                max(
                    heat_capacity[layer_index]
                    / constant_volume_heat_capacity
                    * pressure_derivative_pressure
                    / density_pressure_safe,
                    0.0,
                )
            )
        log_density_temperature_derivative_at_constant_total_pressure[layer_index] = (
            temperature[layer_index] / np.maximum(density[layer_index], 1.0e-300)
        ) * (
            density_temperature_derivative
            - density_pressure_derivative
            * pressure_temperature_derivative
            / np.maximum(pressure_derivative_pressure, 1.0e-300)
        )
        if abs(heat_capacity[layer_index]) > 1.0e-300:
            adiabatic_gradient[layer_index] = (
                -pressure_total[layer_index]
                / np.maximum(
                    density[layer_index] * temperature[layer_index],
                    1.0e-300,
                )
                * log_density_temperature_derivative_at_constant_total_pressure[
                    layer_index
                ]
                / heat_capacity[layer_index]
            )
        logarithmic_gradient[layer_index] = (
            pressure_total[layer_index]
            / np.maximum(
                temperature[layer_index] * float(surface_gravity_cgs),
                1.0e-300,
            )
            * temperature_derivative[layer_index]
        )
        pressure_scale_height[layer_index] = pressure_total[layer_index] / np.maximum(
            density[layer_index] * float(surface_gravity_cgs),
            1.0e-300,
        )

        if float(mixing_length) == 0.0 or layer_index < 3:
            continue
        superadiabatic_gradient = (
            logarithmic_gradient[layer_index] - adiabatic_gradient[layer_index]
        )
        if superadiabatic_gradient < 0.0:
            continue

        velocity_coefficient = (
            0.5
            * float(mixing_length)
            * np.sqrt(
                max(
                    -0.5
                    * pressure_total[layer_index]
                    / np.maximum(density[layer_index], 1.0e-300)
                    * log_density_temperature_derivative_at_constant_total_pressure[
                        layer_index
                    ],
                    0.0,
                )
            )
        )
        if velocity_coefficient == 0.0:
            continue

        flux_coefficient = (
            0.5
            * density[layer_index]
            * heat_capacity[layer_index]
            * temperature[layer_index]
            * float(mixing_length)
            / 12.5664
        )
        local_rosseland[layer_index] = evaluate_rosseland_opacity(
            rosseland_table,
            temperature_k=float(temperature[layer_index]),
            gas_pressure=float(pressure[layer_index]),
        )
        previous_temperature_step = 0.0
        iteration_count = 30 if int(convection_enabled) != 0 else 1
        for _iteration in range(iteration_count):
            rosseland_safe = _signed_floor(float(local_rosseland[layer_index]))
            opacity_plus = (
                evaluate_rosseland_opacity(
                    rosseland_table,
                    temperature_k=float(
                        temperature[layer_index] + temperature_step[layer_index]
                    ),
                    gas_pressure=float(pressure[layer_index]),
                )
                / rosseland_safe
            )
            opacity_minus = (
                evaluate_rosseland_opacity(
                    rosseland_table,
                    temperature_k=float(
                        temperature[layer_index] - temperature_step[layer_index]
                    ),
                    gas_pressure=float(pressure[layer_index]),
                )
                / rosseland_safe
            )
            if opacity_plus == 0.0 or opacity_minus == 0.0:
                convective_opacity = 0.0
            else:
                convective_opacity = (
                    2.0
                    / (1.0 / opacity_plus + 1.0 / opacity_minus)
                    * opacity[layer_index]
                )
            denominator_1 = (
                convective_opacity
                * pressure_scale_height[layer_index]
                * density[layer_index]
            )
            denominator_2 = flux_coefficient * 12.5664
            if (
                denominator_1 == 0.0
                or denominator_2 == 0.0
                or velocity_coefficient == 0.0
            ):
                d_factor = 0.0
            else:
                d_factor = (
                    8.0
                    * 5.6697e-5
                    * temperature[layer_index] ** 4
                    / denominator_1
                    / denominator_2
                    / velocity_coefficient
                )
            optical_thickness = (
                convective_opacity
                * density[layer_index]
                * float(mixing_length)
                * pressure_scale_height[layer_index]
            )
            d_factor = d_factor * optical_thickness**2 / (2.0 + optical_thickness**2)
            d_factor = d_factor**2 / 2.0
            ratio_squared = (
                superadiabatic_gradient
                / _signed_floor(float(d_factor + superadiabatic_gradient))
            ) ** 2
            if ratio_squared < 0.5:
                delta = 0.5
                term = 0.5
                numerator = -1.0
                denominator = 2.0
                while term > 1.0e-6:
                    numerator += 2.0
                    denominator += 2.0
                    term = numerator / denominator * ratio_squared * term
                    delta += term
            else:
                delta = (1.0 - np.sqrt(max(1.0 - ratio_squared, 0.0))) / np.maximum(
                    ratio_squared,
                    1.0e-300,
                )
            delta = (
                delta
                * superadiabatic_gradient**2
                / _signed_floor(float(d_factor + superadiabatic_gradient))
            )
            convective_velocity[layer_index] = velocity_coefficient * np.sqrt(
                max(delta, 0.0)
            )
            convective_flux[layer_index] = max(
                flux_coefficient * convective_velocity[layer_index] * delta,
                0.0,
            )
            temperature_step[layer_index] = (
                temperature[layer_index] * float(mixing_length) * delta
            )
            temperature_step[layer_index] = min(
                temperature_step[layer_index],
                temperature[layer_index] * 0.15,
            )
            temperature_step[layer_index] = (
                temperature_step[layer_index] * 0.7 + previous_temperature_step * 0.3
            )
            if (
                previous_temperature_step - 0.5
                < temperature_step[layer_index]
                < previous_temperature_step + 0.5
            ):
                break
            previous_temperature_step = temperature_step[layer_index]

    raw_convective_flux[:] = convective_flux
    geometric_depth_below_surface_km = integrate_geometric_depth_below_surface_km(
        column_mass=column_mass, mass_density=density
    )

    if float(overshoot_weight) > 0.0:
        weight = min(
            max(
                float(
                    np.max(
                        convective_flux
                        / np.maximum(float(target_integrated_eddington_flux), 1.0e-300)
                    )
                ),
                0.0,
            ),
            1.0,
        ) * float(overshoot_weight)
        height_step = np.minimum.reduce(
            [
                pressure_scale_height * 0.5e-5 * weight,
                np.maximum(
                    geometric_depth_below_surface_km[-1]
                    - geometric_depth_below_surface_km,
                    0.0,
                ),
                np.maximum(
                    geometric_depth_below_surface_km
                    - geometric_depth_below_surface_km[0],
                    0.0,
                ),
            ]
        )
        convective_integral = integrate_on_depth_grid(
            geometric_depth_below_surface_km,
            convective_flux,
            surface_value=0.0,
        )
        midpoint_index = max(layer_count // 2 - 1, 0)
        for layer_index in range(midpoint_index, layer_count - 1):
            if height_step[layer_index] == 0.0:
                continue
            left, _ = remap_to_grid(
                geometric_depth_below_surface_km,
                convective_integral,
                np.asarray(
                    [
                        geometric_depth_below_surface_km[layer_index]
                        - height_step[layer_index]
                    ],
                    dtype=np.float64,
                ),
            )
            right, _ = remap_to_grid(
                geometric_depth_below_surface_km,
                convective_integral,
                np.asarray(
                    [
                        geometric_depth_below_surface_km[layer_index]
                        + height_step[layer_index]
                    ],
                    dtype=np.float64,
                ),
            )
            overshoot_convective_flux[layer_index] += (
                (right[0] - left[0]) / height_step[layer_index] / 2.0
            )
        convective_flux = np.maximum(raw_convective_flux, overshoot_convective_flux)

    layers_to_zero = int(max(min(int(zero_top_layer_count), layer_count), 0))
    if layers_to_zero > 0:
        convective_flux[:layers_to_zero] = 0.0

    return ConvectionResult(
        geometric_depth_below_surface_km=geometric_depth_below_surface_km,
        logarithmic_temperature_pressure_gradient=logarithmic_gradient,
        heat_capacity=heat_capacity,
        log_density_temperature_derivative_at_constant_total_pressure=(
            log_density_temperature_derivative_at_constant_total_pressure
        ),
        sound_speed=sound_speed,
        adiabatic_gradient=adiabatic_gradient,
        pressure_scale_height=pressure_scale_height,
        convective_flux=convective_flux,
        convective_velocity=convective_velocity,
        raw_convective_flux=raw_convective_flux,
        overshoot_convective_flux=overshoot_convective_flux,
    )


def compute_disabled_convection_diagnostics(
    *,
    column_mass: np.ndarray,
    rosseland_optical_depth: np.ndarray,
    temperature_k: np.ndarray,
    gas_pressure: np.ndarray,
    mass_density: np.ndarray,
    rosseland_opacity: np.ndarray,
    absolute_radiation_pressure: np.ndarray,
    total_pressure: np.ndarray,
    surface_gravity_cgs: float,
    target_integrated_eddington_flux: float,
    mixing_length: float,
    rosseland_table: RosselandOpacityTable,
    overshoot_weight: float = 1.0,
    zero_top_layer_count: int = 36,
) -> DisabledConvectionDiagnostics:
    """Preserve validated endpoint diagnostics when convection is disabled.

    One local evaluation produces convective flux and velocity before the
    correction step overwrites only the interior flux values. The endpoint
    values therefore remain part of the structured atmosphere contract.
    """

    result = compute_convection(
        rosseland_table=rosseland_table,
        column_mass=column_mass,
        rosseland_optical_depth=rosseland_optical_depth,
        temperature_k=temperature_k,
        gas_pressure=gas_pressure,
        mass_density=mass_density,
        rosseland_opacity=rosseland_opacity,
        microturbulence=np.zeros_like(np.asarray(temperature_k, dtype=np.float64)),
        absolute_radiation_pressure=absolute_radiation_pressure,
        total_pressure=total_pressure,
        surface_gravity_cgs=surface_gravity_cgs,
        target_integrated_eddington_flux=target_integrated_eddington_flux,
        mixing_length=mixing_length,
        overshoot_weight=overshoot_weight,
        convection_enabled=False,
        zero_top_layer_count=zero_top_layer_count,
    )
    return DisabledConvectionDiagnostics(
        convective_flux=result.convective_flux,
        convective_velocity=result.convective_velocity,
    )
