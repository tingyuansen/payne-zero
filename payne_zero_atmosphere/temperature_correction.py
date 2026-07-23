"""Temperature and column-mass correction.

Combines the transfer moments into heating/flux-error terms and applies
the lambda-corrected temperature update plus the column-mass remap that
closes each iteration of the runner loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .continuum_opacity import (
    RosselandOpacityTable,
    create_rosseland_opacity_table,
    evaluate_rosseland_opacity,
    ingest_rosseland_opacity_table,
)
from .radiative_transfer import (
    differentiate_on_depth_grid,
    integrate_on_depth_grid,
    remap_to_grid,
)


@dataclass
class TemperatureCorrectionState:
    """Frequency-integrated temperature-correction accumulators."""

    mean_intensity_minus_source_integral: np.ndarray
    absorption_heating_derivative: np.ndarray
    diagonal_lambda_accumulator: np.ndarray
    integrated_eddington_flux: np.ndarray
    previous_temperature_correction: np.ndarray
    rosseland_opacity_table: RosselandOpacityTable


@dataclass
class TemperatureCorrectionResult:
    """Final correction outputs on the original Rosseland optical-depth grid."""

    temperature: np.ndarray
    flux_error_percent: np.ndarray
    flux_derivative: np.ndarray
    flux_temperature_derivative: np.ndarray
    lambda_temperature_derivative: np.ndarray
    surface_temperature_derivative: np.ndarray
    temperature_correction: np.ndarray
    flux_ratio: np.ndarray
    convective_flux: np.ndarray
    column_mass: np.ndarray
    column_mass_correction: np.ndarray


def initialize_temperature_correction_state(
    layer_count: int,
) -> TemperatureCorrectionState:
    """Return a zeroed temperature-correction accumulator state."""

    zeros = np.zeros(int(layer_count), dtype=np.float64)
    return TemperatureCorrectionState(
        mean_intensity_minus_source_integral=zeros.copy(),
        absorption_heating_derivative=zeros.copy(),
        diagonal_lambda_accumulator=zeros.copy(),
        integrated_eddington_flux=zeros.copy(),
        previous_temperature_correction=zeros.copy(),
        rosseland_opacity_table=create_rosseland_opacity_table(int(layer_count)),
    )


def ingest_temperature_correction_rosseland_table(
    state: TemperatureCorrectionState,
    *,
    temperature_k: np.ndarray,
    gas_pressure: np.ndarray,
    rosseland_opacity: np.ndarray,
) -> None:
    """Append one atmosphere column to the correction opacity lookup."""

    ingest_rosseland_opacity_table(
        state.rosseland_opacity_table,
        temperature_k=temperature_k,
        gas_pressure=gas_pressure,
        rosseland_opacity=rosseland_opacity,
    )


def exponential_integral_approximation(order: int, argument: float) -> float:
    """Return the validated exponential-integral approximation."""

    a0, a1, a2, a3, a4, a5 = (
        -44178.5471728217,
        57721.7247139444,
        9938.31388962037,
        1842.11088668,
        101.093806161906,
        5.03416184097568,
    )
    b0, b1, b2, b3, b4 = (
        76537.3323337614,
        32597.1881290275,
        6106.10794245759,
        635.419418378382,
        37.2298352833327,
    )
    c0, c1, c2, c3, c4, c5, c6 = (
        4.65627107975096e-7,
        0.999979577051595,
        9.04161556946329,
        24.3784088791317,
        23.0192559391333,
        6.90522522784444,
        0.430967839469389,
    )
    d1, d2, d3, d4, d5, d6 = (
        10.0411643829054,
        32.4264210695138,
        41.2807841891424,
        20.4494785013794,
        3.31909213593302,
        0.103400130404874,
    )
    e0, e1, e2, e3, e4, e5, e6 = (
        -0.999999999998447,
        -26.6271060431811,
        -241.055827097015,
        -895.927957772937,
        -1298.85688746484,
        -545.374158883133,
        -5.66575206533869,
    )
    f1, f2, f3, f4, f5, f6 = (
        28.6271060422192,
        292.310039388533,
        1332.78537748257,
        2777.61949509163,
        2404.01713225909,
        631.6574832808,
    )
    x = float(argument)
    if x <= 0.0:
        first_order = 0.0
    else:
        exponential = np.exp(-x)
        if x > 4.0:
            first_order = (
                exponential
                + exponential
                * (e0 + (e1 + (e2 + (e3 + (e4 + (e5 + e6 / x) / x) / x) / x) / x) / x)
                / (x + f1 + (f2 + (f3 + (f4 + (f5 + f6 / x) / x) / x) / x) / x)
            ) / x
        elif x > 1.0:
            first_order = (
                exponential
                * (c6 + (c5 + (c4 + (c3 + (c2 + (c1 + c0 * x) * x) * x) * x) * x) * x)
                / (d6 + (d5 + (d4 + (d3 + (d2 + (d1 + x) * x) * x) * x) * x) * x)
            )
        else:
            first_order = (a0 + (a1 + (a2 + (a3 + (a4 + a5 * x) * x) * x) * x) * x) / (
                b0 + (b1 + (b2 + (b3 + (b4 + x) * x) * x) * x) * x
            ) - np.log(x)
    value = first_order
    for index in range(1, max(int(order), 1)):
        value = (np.exp(-x) - x * value) / float(index)
    return float(value)


def _signed_floor(value: float, floor: float = 1.0e-300) -> float:
    if abs(value) >= floor:
        return value
    return floor if value >= 0.0 else -floor


def _pressure_on_standard_depth_grid(
    *,
    state: TemperatureCorrectionState,
    temperature_k: np.ndarray,
    standard_rosseland_optical_depth: np.ndarray,
    integrated_radiation_pressure: np.ndarray,
    turbulent_pressure: np.ndarray,
    surface_gravity_cgs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve pressure on the corrected optical-depth grid."""

    temperature = np.asarray(temperature_k, dtype=np.float64)
    optical_depth = np.asarray(standard_rosseland_optical_depth, dtype=np.float64)
    integrated_radiation_pressure = np.asarray(
        integrated_radiation_pressure, dtype=np.float64
    )
    turbulent_pressure = np.asarray(turbulent_pressure, dtype=np.float64)
    layer_count = temperature.size
    standard_opacity = np.zeros(layer_count, dtype=np.float64)
    total_pressure = np.zeros(layer_count, dtype=np.float64)
    gas_pressure = np.zeros(layer_count, dtype=np.float64)

    if layer_count > 1:
        optical_depth_step = np.log(
            max(
                float(optical_depth[1] / np.maximum(optical_depth[0], 1.0e-300)),
                1.0e-300,
            )
        )
    else:
        optical_depth_step = 0.0
    pressure_log_4 = pressure_log_3 = pressure_log_2 = pressure_log_1 = 0.0
    dpressure_log_3 = dpressure_log_2 = dpressure_log_1 = 0.0

    standard_opacity[0] = 0.1
    if integrated_radiation_pressure[0] > 0.0:
        standard_opacity[0] = min(
            0.1,
            float(surface_gravity_cgs)
            * optical_depth[0]
            / np.maximum(integrated_radiation_pressure[0], 1.0e-300)
            / 2.0,
        )

    for layer_index in range(layer_count):
        if layer_index == 0:
            pressure_log = np.log(
                max(
                    float(surface_gravity_cgs)
                    / np.maximum(standard_opacity[0], 1.0e-300)
                    * optical_depth[0],
                    1.0e-300,
                )
            )
        elif layer_index <= 3:
            pressure_log = pressure_log_1 + dpressure_log_1
        else:
            pressure_log = (
                3.0 * pressure_log_4
                + 8.0 * dpressure_log_1
                - 4.0 * dpressure_log_2
                + 8.0 * dpressure_log_3
            ) / 3.0

        error = 1.0
        dpressure_log = 0.0
        iteration = 1
        while True:
            pressure_log = min(pressure_log, 709.78)
            total_pressure[layer_index] = np.exp(pressure_log)
            gas_pressure[layer_index] = (
                total_pressure[layer_index]
                + (
                    integrated_radiation_pressure[0]
                    - integrated_radiation_pressure[layer_index]
                )
                - turbulent_pressure[layer_index]
            )
            if gas_pressure[layer_index] <= 0.0:
                gas_pressure[layer_index] = 1.0e-30
                standard_opacity[layer_index] = 0.1
                break
            standard_opacity[layer_index] = evaluate_rosseland_opacity(
                state.rosseland_opacity_table,
                temperature_k=float(temperature[layer_index]),
                gas_pressure=float(gas_pressure[layer_index]),
            )
            dpressure_log = (
                float(surface_gravity_cgs)
                / np.maximum(standard_opacity[layer_index], 1.0e-300)
                * optical_depth[layer_index]
                / np.maximum(total_pressure[layer_index], 1.0e-300)
                * optical_depth_step
            )
            iteration += 1
            if iteration > 1000 or error <= 5.0e-5:
                break

            if layer_index == 0:
                pressure_new = np.log(
                    max(
                        float(surface_gravity_cgs)
                        / np.maximum(standard_opacity[layer_index], 1.0e-300)
                        * optical_depth[layer_index],
                        1.0e-300,
                    )
                )
            elif layer_index <= 3:
                pressure_new = (
                    pressure_log
                    + 2.0 * pressure_log_1
                    + dpressure_log
                    + dpressure_log_1
                ) / 3.0
            else:
                pressure_new = (
                    126.0 * pressure_log_1
                    - 14.0 * pressure_log_3
                    + 9.0 * pressure_log_4
                    + 42.0 * dpressure_log
                    + 108.0 * dpressure_log_1
                    - 54.0 * dpressure_log_2
                    + 24.0 * dpressure_log_3
                ) / 121.0
            error = abs(pressure_new - pressure_log)
            pressure_log = 0.5 * (pressure_new + pressure_log)

        pressure_log_4 = pressure_log_3
        pressure_log_3 = pressure_log_2
        pressure_log_2 = pressure_log_1
        pressure_log_1 = pressure_log
        dpressure_log_3 = dpressure_log_2
        dpressure_log_2 = dpressure_log_1
        dpressure_log_1 = dpressure_log

    return standard_opacity, total_pressure, gas_pressure


def apply_temperature_correction(
    state: TemperatureCorrectionState,
    *,
    mode: int,
    frequency_weight: float,
    column_mass: np.ndarray,
    total_opacity: np.ndarray,
    monochromatic_eddington_flux: np.ndarray,
    mean_intensity_minus_source: np.ndarray,
    monochromatic_optical_depth: np.ndarray,
    planck_source: np.ndarray,
    frequency_hz: float,
    h_over_kt: np.ndarray,
    temperature_k: np.ndarray,
    stimulated_emission: np.ndarray,
    scattering_fraction: np.ndarray,
    target_integrated_eddington_flux: float,
    effective_temperature: float,
    frequency_count: int,
    rosseland_optical_depth: np.ndarray | None = None,
    rosseland_opacity: np.ndarray | None = None,
    iteration_index: int = 1,
    convection_enabled: bool | int = False,
    convective_flux: np.ndarray | None = None,
    previous_convective_flux: np.ndarray | None = None,
    logarithmic_temperature_pressure_gradient: np.ndarray | None = None,
    adiabatic_gradient: np.ndarray | None = None,
    pressure_scale_height: np.ndarray | None = None,
    total_pressure: np.ndarray | None = None,
    mass_density: np.ndarray | None = None,
    log_density_temperature_derivative_at_constant_total_pressure: (
        np.ndarray | None
    ) = None,
    heat_capacity: np.ndarray | None = None,
    mixing_length: float = 1.0,
    smooth_start_layer: int = 0,
    smooth_stop_layer: int = 0,
    smooth_left_weight: float = 0.3,
    smooth_center_weight: float = 0.4,
    smooth_right_weight: float = 0.3,
    integrated_radiation_pressure: np.ndarray | None = None,
    turbulent_pressure: np.ndarray | None = None,
    surface_gravity_cgs: float = 1.0e4,
    standard_log_tau_step: float = 0.125,
    standard_log_tau_start: float = -6.875,
) -> TemperatureCorrectionResult | None:
    """Apply one temperature-correction mode step in place."""

    if int(mode) == 1:
        state.mean_intensity_minus_source_integral[:] = 0.0
        state.absorption_heating_derivative[:] = 0.0
        state.diagonal_lambda_accumulator[:] = 0.0
        state.integrated_eddington_flux[:] = 0.0
        return None

    column_mass = np.asarray(column_mass, dtype=np.float64)
    opacity = np.asarray(total_opacity, dtype=np.float64)
    monochromatic_eddington_flux = np.asarray(
        monochromatic_eddington_flux, dtype=np.float64
    )
    mean_minus_source = np.asarray(mean_intensity_minus_source, dtype=np.float64)
    optical_depth = np.asarray(monochromatic_optical_depth, dtype=np.float64)
    planck = np.asarray(planck_source, dtype=np.float64)
    h_over_kt = np.asarray(h_over_kt, dtype=np.float64)
    temperature = np.asarray(temperature_k, dtype=np.float64)
    stimulated = np.asarray(stimulated_emission, dtype=np.float64)
    scattering = np.asarray(scattering_fraction, dtype=np.float64)

    if int(mode) == 2:
        weight = float(frequency_weight)
        opacity_derivative = differentiate_on_depth_grid(column_mass, opacity)
        state.absorption_heating_derivative += (
            opacity_derivative
            / np.maximum(opacity, 1.0e-300)
            * monochromatic_eddington_flux
            * weight
        )
        state.mean_intensity_minus_source_integral += (
            opacity * mean_minus_source * weight
        )
        state.integrated_eddington_flux += monochromatic_eddington_flux * weight

        next_term = 0.0
        for layer_index in range(temperature.size):
            previous_term = next_term
            depth_step = 1.0e-10
            if layer_index != temperature.size - 1:
                depth_step = optical_depth[layer_index + 1] - optical_depth[layer_index]
            depth_step = max(1.0e-10, float(depth_step))
            if depth_step <= 0.01:
                next_term = (
                    (0.922784335098467 - np.log(depth_step)) * depth_step / 4.0
                    + depth_step * depth_step / 12.0
                    - depth_step**3 / 96.0
                    + depth_step**4 / 720.0
                )
            else:
                exponential_integral = 0.0
                if depth_step < 10.0:
                    exponential_integral = exponential_integral_approximation(
                        3, depth_step
                    )
                if (
                    float(effective_temperature) <= 4250.0
                    and depth_step > 0.005
                    and depth_step < 0.02
                ):
                    exponential_integral = 0.0
                next_term = 0.5 * (depth_step + exponential_integral - 0.5) / depth_step
            diagonal_mean_intensity = previous_term + next_term
            planck_derivative = (
                planck[layer_index]
                * float(frequency_hz)
                * h_over_kt[layer_index]
                / np.maximum(
                    temperature[layer_index] * stimulated[layer_index], 1.0e-300
                )
            )
            if int(frequency_count) == 1:
                planck_derivative = (
                    float(target_integrated_eddington_flux)
                    * 16.0
                    / np.maximum(
                        temperature[layer_index],
                        1.0e-300,
                    )
                )
            state.diagonal_lambda_accumulator[layer_index] += (
                opacity[layer_index]
                * (diagonal_mean_intensity - 1.0)
                / np.maximum(
                    1.0 - scattering[layer_index] * diagonal_mean_intensity, 1.0e-300
                )
                * (1.0 - scattering[layer_index])
                * planck_derivative
                * weight
            )
        return None

    if int(mode) != 3:
        raise ValueError(f"Unsupported temperature-correction mode: {mode}")
    if rosseland_optical_depth is None or rosseland_opacity is None:
        raise ValueError(
            "the final temperature correction requires rosseland_optical_depth "
            "and rosseland_opacity"
        )

    rosseland_optical_depth = np.asarray(rosseland_optical_depth, dtype=np.float64)
    rosseland = np.asarray(rosseland_opacity, dtype=np.float64)
    layer_count = int(temperature.size)

    convective = (
        np.zeros(layer_count, dtype=np.float64)
        if convective_flux is None
        else np.asarray(convective_flux, dtype=np.float64).copy()
    )
    previous_convective = (
        np.zeros(layer_count, dtype=np.float64)
        if previous_convective_flux is None
        else np.asarray(previous_convective_flux, dtype=np.float64)
    )
    log_temperature_pressure_gradient = (
        np.zeros(layer_count, dtype=np.float64)
        if logarithmic_temperature_pressure_gradient is None
        else np.asarray(logarithmic_temperature_pressure_gradient, dtype=np.float64)
    )
    adiabatic = (
        np.zeros(layer_count, dtype=np.float64)
        if adiabatic_gradient is None
        else np.asarray(adiabatic_gradient, dtype=np.float64)
    )
    scale_height = (
        np.ones(layer_count, dtype=np.float64)
        if pressure_scale_height is None
        else np.asarray(pressure_scale_height, dtype=np.float64)
    )
    pressure_total = (
        np.ones(layer_count, dtype=np.float64)
        if total_pressure is None
        else np.asarray(total_pressure, dtype=np.float64)
    )
    density = (
        np.ones(layer_count, dtype=np.float64)
        if mass_density is None
        else np.asarray(mass_density, dtype=np.float64)
    )
    opacity_gradient = (
        np.zeros(layer_count, dtype=np.float64)
        if log_density_temperature_derivative_at_constant_total_pressure is None
        else np.asarray(
            log_density_temperature_derivative_at_constant_total_pressure,
            dtype=np.float64,
        )
    )
    heat_capacity = (
        np.zeros(layer_count, dtype=np.float64)
        if heat_capacity is None
        else np.asarray(heat_capacity, dtype=np.float64)
    )

    dtemperature_dcolumn = differentiate_on_depth_grid(column_mass, temperature)
    dgradient_dcolumn = differentiate_on_depth_grid(
        column_mass, log_temperature_pressure_gradient
    )
    dopacity_dcolumn = differentiate_on_depth_grid(column_mass, rosseland)

    smoothed_convective_flux = np.zeros(layer_count, dtype=np.float64)
    if int(convection_enabled) == 1:
        smoothed_convective_flux[:] = convective
    if layer_count >= 1:
        smoothed_convective_flux[0] = 0.0
    if layer_count >= 2:
        smoothed_convective_flux[1] = 0.0
    if layer_count >= 3:
        temporary_flux = smoothed_convective_flux.copy()
        for layer_index in range(1, layer_count - 1):
            temporary_flux[layer_index] = (
                0.25 * smoothed_convective_flux[layer_index - 1]
                + 0.5 * smoothed_convective_flux[layer_index]
                + 0.25 * smoothed_convective_flux[layer_index + 1]
            )
        temporary_flux[-1] = (
            0.25 * smoothed_convective_flux[-3]
            + 0.25 * smoothed_convective_flux[-2]
            + 0.5 * smoothed_convective_flux[-1]
        )
        smoothed_convective_flux[1:-1] = temporary_flux[1:-1]
        smoothed_convective_flux[-1] = temporary_flux[-1]

    radiative_heating_derivative = (
        state.absorption_heating_derivative
        - state.integrated_eddington_flux
        * dopacity_dcolumn
        / np.maximum(rosseland, 1.0e-300)
    )
    column_correction_coefficient = np.zeros(layer_count, dtype=np.float64)
    convective_derivative = np.zeros(layer_count, dtype=np.float64)
    for layer_index in range(layer_count):
        superadiabatic_gradient = 1.0
        convection_denominator = 0.0
        if (
            smoothed_convective_flux[layer_index] > 0.0
            and previous_convective[layer_index] > 0.0
        ):
            superadiabatic_gradient = (
                log_temperature_pressure_gradient[layer_index] - adiabatic[layer_index]
            )
            velocity_coefficient = (
                0.5
                * float(mixing_length)
                * np.sqrt(
                    max(
                        -0.5
                        * pressure_total[layer_index]
                        / max(density[layer_index], 1.0e-300)
                        * opacity_gradient[layer_index],
                        0.0,
                    )
                )
            )
            flux_coefficient = (
                0.5
                * density[layer_index]
                * heat_capacity[layer_index]
                * temperature[layer_index]
                * float(mixing_length)
                / 12.5664
            )
            if float(mixing_length) > 0.0 and velocity_coefficient > 0.0:
                convection_denominator = (
                    8.0
                    * 5.6697e-5
                    * temperature[layer_index] ** 4
                    / np.maximum(
                        rosseland[layer_index]
                        * scale_height[layer_index]
                        * density[layer_index],
                        1.0e-300,
                    )
                    / np.maximum(flux_coefficient * 12.5664, 1.0e-300)
                    / velocity_coefficient
                )
            optical_thickness = (
                rosseland[layer_index]
                * density[layer_index]
                * float(mixing_length)
                * scale_height[layer_index]
            )
            convection_denominator = (
                convection_denominator
                * optical_thickness
                * optical_thickness
                / (2.0 + optical_thickness * optical_thickness)
            )
            convection_denominator = (
                convection_denominator * convection_denominator / 2.0
            )
            den = _signed_floor(float(convection_denominator + superadiabatic_gradient))
            gradient_safe = _signed_floor(float(superadiabatic_gradient))
            convective_derivative[layer_index] = (
                1.0 + convection_denominator / den
            ) / gradient_safe

        active_convective_flux = 0.0
        if state.integrated_eddington_flux[layer_index] > 0.0:
            if (
                smoothed_convective_flux[layer_index]
                / state.integrated_eddington_flux[layer_index]
                > 1.0e-3
                and previous_convective[layer_index]
                / state.integrated_eddington_flux[layer_index]
                > 1.0e-3
            ):
                active_convective_flux = smoothed_convective_flux[layer_index]
        den = _signed_floor(float(convection_denominator + superadiabatic_gradient))
        gradient_safe = _signed_floor(float(superadiabatic_gradient))
        numerator = radiative_heating_derivative[
            layer_index
        ] + active_convective_flux * (
            dtemperature_dcolumn[layer_index]
            / np.maximum(temperature[layer_index], 1.0e-300)
            * (1.0 - 9.0 * convection_denominator / den)
            + 1.5
            * dgradient_dcolumn[layer_index]
            / gradient_safe
            * (1.0 + convection_denominator / den)
        )
        denominator = (
            state.integrated_eddington_flux[layer_index]
            + smoothed_convective_flux[layer_index]
            * 1.5
            * log_temperature_pressure_gradient[layer_index]
            * convective_derivative[layer_index]
        )
        column_correction_coefficient[layer_index] = numerator / _signed_floor(
            float(denominator)
        )
    if layer_count >= 1:
        column_correction_coefficient[0] = 0.0
    if layer_count >= 2:
        column_correction_coefficient[1] = 0.0

    integrating_factor = np.exp(
        integrate_on_depth_grid(
            column_mass,
            column_correction_coefficient,
            surface_value=0.0,
        )
    )
    flux_denominator = (
        state.integrated_eddington_flux
        + smoothed_convective_flux
        * 1.5
        * log_temperature_pressure_gradient
        * convective_derivative
    )
    flux_denominator_safe = np.where(
        np.abs(flux_denominator) >= 1.0e-300,
        flux_denominator,
        np.where(flux_denominator >= 0.0, 1.0e-300, -1.0e-300),
    )
    integrated_flux_error = (
        integrating_factor
        * (
            state.integrated_eddington_flux
            + smoothed_convective_flux
            - float(target_integrated_eddington_flux)
        )
        / flux_denominator_safe
    )
    optical_depth_correction = integrate_on_depth_grid(
        rosseland_optical_depth,
        integrated_flux_error,
        surface_value=0.0,
    ) / np.maximum(integrating_factor, 1.0e-300)
    optical_depth_correction = np.maximum(
        -rosseland_optical_depth / 3.0,
        np.minimum(rosseland_optical_depth / 3.0, optical_depth_correction),
    )
    flux_temperature_derivative = (
        -optical_depth_correction
        * dtemperature_dcolumn
        / np.maximum(rosseland, 1.0e-300)
    )

    flux_error = (
        (
            state.integrated_eddington_flux
            + smoothed_convective_flux
            - float(target_integrated_eddington_flux)
        )
        / np.maximum(float(target_integrated_eddington_flux), 1.0e-300)
        * 100.0
    )
    flux_derivative = differentiate_on_depth_grid(rosseland_optical_depth, flux_error)
    lambda_temperature_derivative = np.zeros(layer_count, dtype=np.float64)
    maximum_temperature_step = float(effective_temperature) / 25.0
    for layer_index in range(layer_count):
        convective_ratio = smoothed_convective_flux[layer_index] / np.maximum(
            state.integrated_eddington_flux[layer_index],
            1.0e-300,
        )
        if convective_ratio < 1.0e-5:
            flux_derivative[layer_index] = (
                state.mean_intensity_minus_source_integral[layer_index]
                / np.maximum(rosseland[layer_index], 1.0e-300)
                / np.maximum(float(target_integrated_eddington_flux), 1.0e-300)
                * 100.0
            )
        diagonal = (
            state.diagonal_lambda_accumulator[layer_index]
            if abs(state.diagonal_lambda_accumulator[layer_index]) > 1.0e-300
            else np.sign(state.diagonal_lambda_accumulator[layer_index]) * 1.0e-300
        )
        lambda_temperature_derivative[layer_index] = (
            -flux_derivative[layer_index]
            * float(target_integrated_eddington_flux)
            / 100.0
            / diagonal
            * rosseland[layer_index]
        )
        if not (
            convective_ratio < 1.0e-5 and rosseland_optical_depth[layer_index] < 1.0
        ):
            lambda_temperature_derivative[layer_index] = 0.0
            for offset in range(1, 6):
                neighbor_index = layer_index - offset
                if neighbor_index >= 0:
                    lambda_temperature_derivative[neighbor_index] *= 0.5
        lambda_temperature_derivative[layer_index] = float(
            np.clip(
                lambda_temperature_derivative[layer_index],
                -maximum_temperature_step,
                maximum_temperature_step,
            )
        )

    surface_step = (
        (float(target_integrated_eddington_flux) - state.integrated_eddington_flux[0])
        / np.maximum(float(target_integrated_eddington_flux), 1.0e-300)
        * 0.25
        * temperature[0]
    )
    surface_step = float(
        np.clip(surface_step, -maximum_temperature_step, maximum_temperature_step)
    )
    integrated_temperature_step = integrate_on_depth_grid(
        rosseland_optical_depth,
        flux_temperature_derivative + lambda_temperature_derivative,
        surface_value=0.0,
    )
    step_at_tau_01 = remap_to_grid(
        rosseland_optical_depth, integrated_temperature_step, np.array([0.1])
    )[0][0]
    step_at_tau_2 = remap_to_grid(
        rosseland_optical_depth, integrated_temperature_step, np.array([2.0])
    )[0][0]
    average_step = (step_at_tau_2 - step_at_tau_01) / 2.0
    if surface_step * average_step <= 0.0:
        average_step = 0.0
    if abs(average_step) > abs(surface_step):
        average_step = surface_step
    surface_step = surface_step - average_step

    surface_temperature_derivative = np.full(
        layer_count, surface_step, dtype=np.float64
    )
    flux_ratio = smoothed_convective_flux / np.maximum(
        smoothed_convective_flux + state.integrated_eddington_flux,
        1.0e-300,
    )
    flux_temperature_derivative = np.nan_to_num(
        flux_temperature_derivative,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    lambda_temperature_derivative = np.nan_to_num(
        lambda_temperature_derivative,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    surface_temperature_derivative = np.nan_to_num(
        surface_temperature_derivative,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    temperature_correction = (
        flux_temperature_derivative
        + lambda_temperature_derivative
        + surface_temperature_derivative
    )

    for layer_index in range(layer_count):
        skip_damping = False
        if int(convection_enabled) == 1 and flux_ratio[layer_index] > 0.0:
            skip_damping = True
        if int(convection_enabled) == 1 and (layer_index + 1) >= (layer_count / 3.0):
            skip_damping = True
        if int(iteration_index) == 1:
            skip_damping = True
        if not skip_damping:
            previous = state.previous_temperature_correction[layer_index]
            current = temperature_correction[layer_index]
            if previous * current > 0.0 and abs(previous) > abs(current):
                temperature_correction[layer_index] *= 1.25
            if previous * current < 0.0:
                temperature_correction[layer_index] *= 0.5
        state.previous_temperature_correction[layer_index] = temperature_correction[
            layer_index
        ]

    new_temperature = temperature + temperature_correction
    bad_temperature = ~np.isfinite(new_temperature)
    if bad_temperature.any():
        new_temperature = np.where(bad_temperature, temperature, new_temperature)
    new_temperature = np.maximum(new_temperature, 1.0)
    if smooth_start_layer > 0:
        start_index = max(int(smooth_start_layer) - 1, 1)
        stop_index = min(int(smooth_stop_layer) - 1, layer_count - 2)
        if stop_index >= start_index:
            smoothed_temperature = new_temperature.copy()
            for layer_index in range(start_index, stop_index + 1):
                smoothed_temperature[layer_index] = (
                    float(smooth_left_weight) * new_temperature[layer_index - 1]
                    + float(smooth_center_weight) * new_temperature[layer_index]
                    + float(smooth_right_weight) * new_temperature[layer_index + 1]
                )
            new_temperature[start_index : stop_index + 1] = smoothed_temperature[
                start_index : stop_index + 1
            ]

    for reverse_offset in range(1, layer_count):
        layer_index = layer_count - 1 - reverse_offset
        new_temperature[layer_index] = np.fmin(
            new_temperature[layer_index],
            new_temperature[layer_index + 1] - 1.0,
        )
        if not np.isfinite(new_temperature[layer_index]):
            new_temperature[layer_index] = max(temperature[layer_index], 1.0)

    integrated_radiation_pressure = (
        np.zeros(layer_count, dtype=np.float64)
        if integrated_radiation_pressure is None
        else np.asarray(integrated_radiation_pressure, dtype=np.float64)
    )
    turbulent_pressure = (
        np.zeros(layer_count, dtype=np.float64)
        if turbulent_pressure is None
        else np.asarray(turbulent_pressure, dtype=np.float64)
    )
    standard_rosseland_optical_depth = np.float64(10.0) ** (
        float(standard_log_tau_start)
        + np.arange(layer_count, dtype=np.float64) * float(standard_log_tau_step)
    )
    temperature_plus_correction = temperature + temperature_correction

    old_temperature_on_standard_grid, _ = remap_to_grid(
        rosseland_optical_depth,
        temperature,
        standard_rosseland_optical_depth,
    )
    integrated_radiation_pressure_on_standard_grid, _ = remap_to_grid(
        rosseland_optical_depth,
        integrated_radiation_pressure,
        standard_rosseland_optical_depth,
    )
    turbulent_pressure_on_standard_grid, _ = remap_to_grid(
        rosseland_optical_depth,
        turbulent_pressure,
        standard_rosseland_optical_depth,
    )
    _, old_total_pressure, _ = _pressure_on_standard_depth_grid(
        state=state,
        temperature_k=old_temperature_on_standard_grid,
        standard_rosseland_optical_depth=standard_rosseland_optical_depth,
        integrated_radiation_pressure=integrated_radiation_pressure_on_standard_grid,
        turbulent_pressure=turbulent_pressure_on_standard_grid,
        surface_gravity_cgs=float(surface_gravity_cgs),
    )
    new_temperature_on_standard_grid, _ = remap_to_grid(
        rosseland_optical_depth,
        temperature_plus_correction,
        standard_rosseland_optical_depth,
    )
    _, new_total_pressure, _ = _pressure_on_standard_depth_grid(
        state=state,
        temperature_k=new_temperature_on_standard_grid,
        standard_rosseland_optical_depth=standard_rosseland_optical_depth,
        integrated_radiation_pressure=integrated_radiation_pressure_on_standard_grid,
        turbulent_pressure=turbulent_pressure_on_standard_grid,
        surface_gravity_cgs=float(surface_gravity_cgs),
    )
    pressure_fractional_change = (
        new_total_pressure / np.maximum(old_total_pressure, 1.0e-300) - 1.0
    )
    column_fractional_change, _ = remap_to_grid(
        standard_rosseland_optical_depth,
        pressure_fractional_change,
        rosseland_optical_depth,
    )
    column_mass_correction = column_fractional_change * column_mass
    corrected_column_mass = column_mass + column_mass_correction

    return TemperatureCorrectionResult(
        temperature=new_temperature,
        flux_error_percent=flux_error,
        flux_derivative=flux_derivative,
        flux_temperature_derivative=flux_temperature_derivative,
        lambda_temperature_derivative=lambda_temperature_derivative,
        surface_temperature_derivative=surface_temperature_derivative,
        temperature_correction=temperature_correction,
        flux_ratio=flux_ratio,
        convective_flux=smoothed_convective_flux,
        column_mass=corrected_column_mass,
        column_mass_correction=column_mass_correction,
    )
