"""Instrument-agnostic fitting of a normalized spectrum on its observed grid."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np


Array = np.ndarray
ModelFunction = Callable[[Array], Array]
JacobianFunction = Callable[[Array], Array]
ConvergedModelFunction = Callable[[Array], Array]


@dataclass(frozen=True)
class NormalizedSpectrum:
    """Observed normalized flux and weights on one wavelength grid."""

    wavelength: Array
    flux: Array
    inverse_variance: Array
    mask: Array

    def validated(self) -> "NormalizedSpectrum":
        wavelength = np.asarray(self.wavelength, np.float64)
        flux = np.asarray(self.flux, np.float64)
        inverse_variance = np.asarray(self.inverse_variance, np.float64)
        mask = np.asarray(self.mask, bool)
        if not (
            wavelength.ndim == flux.ndim == inverse_variance.ndim == mask.ndim == 1
        ):
            raise ValueError("spectrum arrays must be one-dimensional")
        if not (wavelength.size == flux.size == inverse_variance.size == mask.size):
            raise ValueError("spectrum arrays must have the same length")
        if not np.all(np.isfinite(wavelength)):
            raise ValueError("wavelength must be finite")
        if wavelength.size < 2 or not np.all(np.diff(wavelength) > 0.0):
            raise ValueError("wavelength must be strictly increasing")
        good = mask & np.isfinite(flux) & np.isfinite(inverse_variance)
        good &= inverse_variance > 0.0
        if np.count_nonzero(good) <= 1:
            raise ValueError("at least two positive-weight pixels are required")
        return NormalizedSpectrum(wavelength, flux, inverse_variance, good)


@dataclass(frozen=True)
class FitConfiguration:
    """Parameter bounds, finite-difference scales, and stopping controls."""

    names: tuple[str, ...]
    initial: Array
    lower: Array
    upper: Array
    derivative_steps: Array
    trust_half_width: Array
    maximum_iterations: int = 8
    minimum_objective_improvement: float = 1.0e-6
    minimum_scaled_step: float = 1.0e-4

    def validated(self) -> "FitConfiguration":
        count = len(self.names)
        arrays = tuple(
            np.asarray(value, np.float64)
            for value in (
                self.initial,
                self.lower,
                self.upper,
                self.derivative_steps,
                self.trust_half_width,
            )
        )
        if count == 0 or len(set(self.names)) != count or not all(self.names):
            raise ValueError("parameter names must be unique and non-empty")
        if any(value.shape != (count,) for value in arrays):
            raise ValueError("every parameter array must match names")
        initial, lower, upper, derivative_steps, trust_half_width = arrays
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("parameter arrays must be finite")
        if np.any(lower >= upper) or np.any((initial < lower) | (initial > upper)):
            raise ValueError("parameter bounds or initial values are invalid")
        if np.any(derivative_steps <= 0.0) or np.any(trust_half_width <= 0.0):
            raise ValueError("derivative and trust scales must be positive")
        if self.maximum_iterations < 1:
            raise ValueError("maximum_iterations must be positive")
        if self.minimum_objective_improvement < 0.0 or self.minimum_scaled_step < 0.0:
            raise ValueError("stopping thresholds must be nonnegative")
        return FitConfiguration(
            self.names,
            initial,
            lower,
            upper,
            derivative_steps,
            trust_half_width,
            self.maximum_iterations,
            self.minimum_objective_improvement,
            self.minimum_scaled_step,
        )


@dataclass(frozen=True)
class FitResult:
    """Best fit plus the complete accepted optimization path and model timing."""

    names: tuple[str, ...]
    parameters: Array
    continuum_coefficients: Array
    model_flux: Array
    mean_weighted_squared_residual: float
    converged: bool
    stop_reason: str
    parameter_path: Array
    objective_path: Array
    model_seconds_path: Array
    model_flux_path: Array

    def save(self, output_dir: str | Path) -> None:
        """Write a compact machine-readable summary and full optimization trace."""

        directory = Path(output_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "fit_trace.npz",
            parameter_names=np.asarray(self.names),
            parameter_path=self.parameter_path,
            objective_path=self.objective_path,
            model_seconds_path=self.model_seconds_path,
            model_flux_path=self.model_flux_path,
            continuum_coefficients=self.continuum_coefficients,
        )
        summary = {
            "parameter_names": self.names,
            "parameters": {
                name: float(value)
                for name, value in zip(self.names, self.parameters, strict=True)
            },
            "mean_weighted_squared_residual": self.mean_weighted_squared_residual,
            "converged": self.converged,
            "stop_reason": self.stop_reason,
            "accepted_step_count": int(self.parameter_path.shape[0] - 1),
            "model_seconds_excluding_plotting": float(np.sum(self.model_seconds_path)),
        }
        (directory / "fit_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )


@dataclass(frozen=True)
class PhysicalAtmosphereConfiguration:
    """Acceptance gates and bounded converged-atmosphere correction controls.

    The two maximum values are explicit gates on the fast model at a
    physical-atmosphere-checked point. ``maximum_physical_evaluations`` is an
    operational safety cap, not a claim that a fit should require a fixed
    number of atmosphere calculations.
    """

    maximum_discrepancy_rms: float
    maximum_objective_degradation: float
    minimum_predicted_objective_improvement: float
    maximum_physical_evaluations: int = 4
    correction_derivative_steps: Array | None = None
    correction_trust_half_width: Array | None = None
    correction_trust_fraction: float = 0.5
    trust_shrink_factor: float = 0.5
    minimum_trust_fraction: float = 1.0e-3
    minimum_scaled_step: float = 1.0e-4
    minimum_physical_objective_improvement: float = 0.0
    line_search_fractions: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)

    def validated(self) -> "PhysicalAtmosphereConfiguration":
        scalar_values = (
            self.maximum_discrepancy_rms,
            self.maximum_objective_degradation,
            self.minimum_predicted_objective_improvement,
            self.correction_trust_fraction,
            self.trust_shrink_factor,
            self.minimum_trust_fraction,
            self.minimum_scaled_step,
            self.minimum_physical_objective_improvement,
        )
        if not all(np.isfinite(value) for value in scalar_values):
            raise ValueError("physical-atmosphere controls must be finite")
        if (
            self.maximum_discrepancy_rms < 0.0
            or self.maximum_objective_degradation < 0.0
            or self.minimum_predicted_objective_improvement < 0.0
            or self.minimum_scaled_step < 0.0
            or self.minimum_physical_objective_improvement < 0.0
        ):
            raise ValueError("physical-atmosphere thresholds must be nonnegative")
        if self.maximum_physical_evaluations < 1:
            raise ValueError("maximum_physical_evaluations must be positive")
        if not 0.0 < self.correction_trust_fraction < 1.0:
            raise ValueError("correction_trust_fraction must lie between zero and one")
        if not 0.0 < self.trust_shrink_factor < 1.0:
            raise ValueError("trust_shrink_factor must lie between zero and one")
        if not 0.0 < self.minimum_trust_fraction <= self.correction_trust_fraction:
            raise ValueError(
                "minimum_trust_fraction must be positive and no larger than "
                "correction_trust_fraction"
            )
        fractions = tuple(float(value) for value in self.line_search_fractions)
        if (
            not fractions
            or not all(np.isfinite(value) and 0.0 < value <= 1.0 for value in fractions)
            or any(left <= right for left, right in zip(fractions, fractions[1:]))
        ):
            raise ValueError(
                "line_search_fractions must be finite, positive, and strictly decreasing"
            )

        def optional_positive_array(name: str, value: Array | None) -> Array | None:
            if value is None:
                return None
            selected = np.asarray(value, np.float64)
            if (
                selected.ndim != 1
                or not np.all(np.isfinite(selected))
                or np.any(selected <= 0.0)
            ):
                raise ValueError(f"{name} must be a finite positive vector")
            return selected.copy()

        return PhysicalAtmosphereConfiguration(
            maximum_discrepancy_rms=float(self.maximum_discrepancy_rms),
            maximum_objective_degradation=float(
                self.maximum_objective_degradation
            ),
            minimum_predicted_objective_improvement=float(
                self.minimum_predicted_objective_improvement
            ),
            maximum_physical_evaluations=int(self.maximum_physical_evaluations),
            correction_derivative_steps=optional_positive_array(
                "correction_derivative_steps", self.correction_derivative_steps
            ),
            correction_trust_half_width=optional_positive_array(
                "correction_trust_half_width", self.correction_trust_half_width
            ),
            correction_trust_fraction=float(self.correction_trust_fraction),
            trust_shrink_factor=float(self.trust_shrink_factor),
            minimum_trust_fraction=float(self.minimum_trust_fraction),
            minimum_scaled_step=float(self.minimum_scaled_step),
            minimum_physical_objective_improvement=float(
                self.minimum_physical_objective_improvement
            ),
            line_search_fractions=fractions,
        )


@dataclass(frozen=True)
class ResolvedPhysicalAtmosphereControls:
    """Immutable numerical controls used by one atmosphere refinement."""

    maximum_discrepancy_rms: float
    maximum_objective_degradation: float
    minimum_predicted_objective_improvement: float
    maximum_physical_evaluations: int
    parameter_lower_bounds: Array
    parameter_upper_bounds: Array
    correction_derivative_steps: Array
    correction_trust_half_width: Array
    minimum_trust_half_width: Array
    trust_shrink_factor: float
    minimum_scaled_step: float
    minimum_physical_objective_improvement: float
    line_search_fractions: tuple[float, ...]


@dataclass(frozen=True)
class PhysicalAtmosphereCheck:
    """One converged-model evaluation and its matched fast-model comparison."""

    parameters: Array
    fast_continuum_coefficients: Array
    physical_continuum_coefficients: Array
    fast_model_flux: Array
    physical_model_flux: Array
    fast_objective: float
    physical_objective: float
    discrepancy_rms: float
    objective_degradation: float
    discrepancy_gate_passed: bool
    objective_gate_passed: bool
    accepted: bool
    fast_model_reused: bool
    fast_model_seconds: float
    physical_model_seconds: float


@dataclass(frozen=True)
class PhysicalAtmosphereCorrection:
    """One fresh local discrepancy-corrected fast-model proposal."""

    base_physical_index: int
    trust_half_width: Array
    proposed_parameters: Array
    proposed_scaled_step_norm: float
    selected_line_search_fraction: float
    actual_scaled_step_norm: float
    actual_parameter_step_norm: float
    predicted_objective: float
    predicted_improvement: float
    physical_evaluated: bool
    physical_objective: float
    actual_improvement: float
    actual_to_predicted_ratio: float
    accepted: bool
    fast_evaluation_count: int
    fast_model_seconds: float
    correction_seconds: float
    outcome: str


@dataclass(frozen=True)
class PhysicalAtmosphereResult:
    """Best accepted physical model and the complete refinement trace.

    Stationarity of the local discrepancy-corrected fit and agreement of the
    fast and converged-atmosphere spectra are reported independently.
    ``successful`` is true when either declared stopping route succeeds.
    """

    names: tuple[str, ...]
    parameters: Array
    continuum_coefficients: Array
    model_flux: Array
    mean_weighted_squared_residual: float
    physical_fit_stationary: bool
    fast_physical_gates_passed: bool
    successful: bool
    stop_reason: str
    fast_candidate_parameters: Array
    physical_checks: tuple[PhysicalAtmosphereCheck, ...]
    corrections: tuple[PhysicalAtmosphereCorrection, ...]
    controls: ResolvedPhysicalAtmosphereControls
    total_seconds_excluding_plotting: float

    def save(self, output_dir: str | Path) -> None:
        """Write the physical checks, correction proposals, and model timings."""

        directory = Path(output_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        physical_count = len(self.physical_checks)
        correction_count = len(self.corrections)
        parameter_count = len(self.names)
        pixel_count = self.model_flux.size
        continuum_count = self.continuum_coefficients.size

        def physical_array(
            name: str, shape: tuple[int, ...], dtype=np.float64
        ) -> Array:
            if physical_count == 0:
                return np.empty((0, *shape), dtype=dtype)
            return np.asarray(
                [getattr(check, name) for check in self.physical_checks], dtype=dtype
            )

        def correction_array(
            name: str, shape: tuple[int, ...] = (), dtype=np.float64
        ) -> Array:
            if correction_count == 0:
                return np.empty((0, *shape), dtype=dtype)
            return np.asarray(
                [getattr(correction, name) for correction in self.corrections],
                dtype=dtype,
            )

        np.savez_compressed(
            directory / "physical_atmosphere_trace.npz",
            parameter_names=np.asarray(self.names),
            fast_candidate_parameters=self.fast_candidate_parameters,
            physical_parameter_path=physical_array("parameters", (parameter_count,)),
            physical_fast_continuum_coefficients_path=physical_array(
                "fast_continuum_coefficients", (continuum_count,)
            ),
            physical_continuum_coefficients_path=physical_array(
                "physical_continuum_coefficients", (continuum_count,)
            ),
            physical_fast_model_flux_path=physical_array(
                "fast_model_flux", (pixel_count,)
            ),
            physical_model_flux_path=physical_array(
                "physical_model_flux", (pixel_count,)
            ),
            physical_fast_objective_path=physical_array("fast_objective", ()),
            physical_objective_path=physical_array("physical_objective", ()),
            physical_discrepancy_rms_path=physical_array("discrepancy_rms", ()),
            physical_objective_degradation_path=physical_array(
                "objective_degradation", ()
            ),
            physical_discrepancy_gate_passed_path=physical_array(
                "discrepancy_gate_passed", (), bool
            ),
            physical_objective_gate_passed_path=physical_array(
                "objective_gate_passed", (), bool
            ),
            physical_accepted_path=physical_array("accepted", (), bool),
            physical_fast_model_reused_path=physical_array(
                "fast_model_reused", (), bool
            ),
            physical_fast_model_seconds_path=physical_array(
                "fast_model_seconds", ()
            ),
            physical_model_seconds_path=physical_array(
                "physical_model_seconds", ()
            ),
            correction_base_physical_index=correction_array(
                "base_physical_index", (), np.int64
            ),
            correction_trust_half_width_path=correction_array(
                "trust_half_width", (parameter_count,)
            ),
            correction_parameter_path=correction_array(
                "proposed_parameters", (parameter_count,)
            ),
            correction_proposed_scaled_step_norm_path=correction_array(
                "proposed_scaled_step_norm"
            ),
            correction_selected_line_search_fraction_path=correction_array(
                "selected_line_search_fraction"
            ),
            correction_actual_scaled_step_norm_path=correction_array(
                "actual_scaled_step_norm"
            ),
            correction_actual_parameter_step_norm_path=correction_array(
                "actual_parameter_step_norm"
            ),
            correction_predicted_objective_path=correction_array(
                "predicted_objective"
            ),
            correction_predicted_improvement_path=correction_array(
                "predicted_improvement"
            ),
            correction_physical_evaluated_path=correction_array(
                "physical_evaluated", (), bool
            ),
            correction_physical_objective_path=correction_array(
                "physical_objective"
            ),
            correction_actual_improvement_path=correction_array(
                "actual_improvement"
            ),
            correction_actual_to_predicted_ratio_path=correction_array(
                "actual_to_predicted_ratio"
            ),
            correction_accepted_path=correction_array("accepted", (), bool),
            correction_fast_evaluation_count_path=correction_array(
                "fast_evaluation_count", (), np.int64
            ),
            correction_fast_model_seconds_path=correction_array(
                "fast_model_seconds"
            ),
            correction_seconds_path=correction_array("correction_seconds"),
            correction_outcome_path=np.asarray(
                [correction.outcome for correction in self.corrections], dtype=str
            ),
            control_maximum_discrepancy_rms=np.asarray(
                self.controls.maximum_discrepancy_rms
            ),
            control_maximum_objective_degradation=np.asarray(
                self.controls.maximum_objective_degradation
            ),
            control_minimum_predicted_objective_improvement=np.asarray(
                self.controls.minimum_predicted_objective_improvement
            ),
            control_maximum_physical_evaluations=np.asarray(
                self.controls.maximum_physical_evaluations, dtype=np.int64
            ),
            control_parameter_lower_bounds=self.controls.parameter_lower_bounds,
            control_parameter_upper_bounds=self.controls.parameter_upper_bounds,
            control_correction_derivative_steps=(
                self.controls.correction_derivative_steps
            ),
            control_correction_trust_half_width=(
                self.controls.correction_trust_half_width
            ),
            control_minimum_trust_half_width=(
                self.controls.minimum_trust_half_width
            ),
            control_trust_shrink_factor=np.asarray(
                self.controls.trust_shrink_factor
            ),
            control_minimum_scaled_step=np.asarray(
                self.controls.minimum_scaled_step
            ),
            control_minimum_physical_objective_improvement=np.asarray(
                self.controls.minimum_physical_objective_improvement
            ),
            control_line_search_fractions=np.asarray(
                self.controls.line_search_fractions
            ),
        )
        summary = {
            "parameter_names": self.names,
            "parameters": {
                name: float(value)
                for name, value in zip(self.names, self.parameters, strict=True)
            },
            "mean_weighted_squared_residual": self.mean_weighted_squared_residual,
            "physical_fit_stationary": self.physical_fit_stationary,
            "fast_physical_gates_passed": self.fast_physical_gates_passed,
            "successful": self.successful,
            "stop_reason": self.stop_reason,
            "physical_evaluation_count": physical_count,
            "correction_count": correction_count,
            "accepted_correction_count": int(
                sum(correction.accepted for correction in self.corrections)
            ),
            "physical_model_seconds": float(
                sum(
                    check.physical_model_seconds for check in self.physical_checks
                )
            ),
            "fast_model_seconds": float(
                sum(check.fast_model_seconds for check in self.physical_checks)
                + sum(
                    correction.fast_model_seconds for correction in self.corrections
                )
            ),
            "correction_seconds": float(
                sum(correction.correction_seconds for correction in self.corrections)
            ),
            "refinement_controls": {
                "maximum_discrepancy_rms": (
                    self.controls.maximum_discrepancy_rms
                ),
                "maximum_objective_degradation": (
                    self.controls.maximum_objective_degradation
                ),
                "minimum_predicted_objective_improvement": (
                    self.controls.minimum_predicted_objective_improvement
                ),
                "maximum_physical_evaluations": (
                    self.controls.maximum_physical_evaluations
                ),
                "parameter_lower_bounds": (
                    self.controls.parameter_lower_bounds.tolist()
                ),
                "parameter_upper_bounds": (
                    self.controls.parameter_upper_bounds.tolist()
                ),
                "correction_derivative_steps": (
                    self.controls.correction_derivative_steps.tolist()
                ),
                "correction_trust_half_width": (
                    self.controls.correction_trust_half_width.tolist()
                ),
                "minimum_trust_half_width": (
                    self.controls.minimum_trust_half_width.tolist()
                ),
                "trust_shrink_factor": self.controls.trust_shrink_factor,
                "minimum_scaled_step": self.controls.minimum_scaled_step,
                "minimum_physical_objective_improvement": (
                    self.controls.minimum_physical_objective_improvement
                ),
                "line_search_fractions": self.controls.line_search_fractions,
            },
            "total_seconds_excluding_plotting": float(
                self.total_seconds_excluding_plotting
            ),
        }
        (directory / "physical_atmosphere_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )


def _profile_continuum(
    physical_flux: Array,
    spectrum: NormalizedSpectrum,
    basis: Array,
) -> tuple[Array, Array, float]:
    """Solve the linear multiplicative continuum at fixed nonlinear labels."""

    good = spectrum.mask
    if basis.shape != (spectrum.flux.size, basis.shape[1]):
        raise ValueError("continuum basis has an invalid pixel dimension")
    if basis.shape[1] == 0:
        coefficients = np.empty(0, np.float64)
        fitted = physical_flux
    else:
        design = physical_flux[:, None] * basis
        weighted_design = design[good] * np.sqrt(spectrum.inverse_variance[good, None])
        weighted_target = (spectrum.flux[good] - physical_flux[good]) * np.sqrt(
            spectrum.inverse_variance[good]
        )
        coefficients = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)[0]
        fitted = physical_flux + design @ coefficients
    residual = fitted[good] - spectrum.flux[good]
    chi_square = float(np.dot(residual * spectrum.inverse_variance[good], residual))
    return fitted, coefficients, chi_square / np.count_nonzero(good)


def fit_normalized_spectrum(
    spectrum: NormalizedSpectrum,
    configuration: FitConfiguration,
    model: ModelFunction,
    *,
    jacobian: JacobianFunction | None = None,
    continuum_basis: Array | None = None,
) -> FitResult:
    """Fit any normalized spectrum whose callback returns observed-grid flux.

    The callback owns the physical forward path, including synthesis,
    instrument response, Doppler projection, and resampling. With a continuum
    basis, ``jacobian`` must describe the continuum-profiled output. If it is
    omitted, bounded one-sided finite differences include profiling
    automatically. Plotting is never timed here.
    """

    observed = spectrum.validated()
    fit = configuration.validated()
    basis = (
        np.empty((observed.flux.size, 0), np.float64)
        if continuum_basis is None
        else np.asarray(continuum_basis, np.float64)
    )
    if basis.ndim != 2 or basis.shape[0] != observed.flux.size:
        raise ValueError("continuum_basis must have shape [pixel, coefficient]")
    if not np.all(np.isfinite(basis)):
        raise ValueError("continuum_basis must be finite")

    model_times: list[float] = []

    def evaluate(parameters: Array) -> tuple[Array, Array, float]:
        started = time.perf_counter()
        physical = np.asarray(model(parameters.copy()), np.float64)
        model_times.append(time.perf_counter() - started)
        if physical.shape != observed.flux.shape or not np.all(np.isfinite(physical)):
            raise ValueError("model callback returned an invalid flux array")
        return _profile_continuum(physical, observed, basis)

    parameters = fit.initial.copy()
    selected_flux, coefficients, objective = evaluate(parameters)
    parameter_path = [parameters.copy()]
    objective_path = [objective]
    flux_path = [selected_flux.copy()]
    converged = False
    stop_reason = "maximum_iterations"

    for _ in range(fit.maximum_iterations):
        if jacobian is None:
            matrix = np.empty((observed.flux.size, len(fit.names)), np.float64)
            for index, step in enumerate(fit.derivative_steps):
                direction = (
                    1.0 if parameters[index] + step <= fit.upper[index] else -1.0
                )
                perturbed = parameters.copy()
                perturbed[index] += direction * step
                perturbed_flux, _, _ = evaluate(perturbed)
                matrix[:, index] = (perturbed_flux - selected_flux) / (direction * step)
        else:
            started = time.perf_counter()
            matrix = np.asarray(jacobian(parameters.copy()), np.float64)
            model_times.append(time.perf_counter() - started)
            if matrix.shape != (
                observed.flux.size,
                len(fit.names),
            ) or not np.all(np.isfinite(matrix)):
                raise ValueError("jacobian callback returned an invalid array")

        good = observed.mask
        weighted_matrix = matrix[good] * np.sqrt(observed.inverse_variance[good, None])
        weighted_residual = (selected_flux[good] - observed.flux[good]) * np.sqrt(
            observed.inverse_variance[good]
        )
        scaled_matrix = weighted_matrix * fit.trust_half_width[None, :]
        scaled_delta = np.linalg.lstsq(scaled_matrix, -weighted_residual, rcond=None)[0]
        scaled_delta = np.clip(scaled_delta, -1.0, 1.0)
        if np.linalg.norm(scaled_delta) < fit.minimum_scaled_step:
            converged = True
            stop_reason = "scaled_step"
            break
        delta = scaled_delta * fit.trust_half_width

        accepted = False
        previous_objective = objective
        for fraction in (1.0, 0.5, 0.25, 0.125):
            candidate = np.clip(parameters + fraction * delta, fit.lower, fit.upper)
            candidate_flux, candidate_coefficients, candidate_objective = evaluate(
                candidate
            )
            if candidate_objective < objective:
                parameters = candidate
                selected_flux = candidate_flux
                coefficients = candidate_coefficients
                objective = candidate_objective
                parameter_path.append(parameters.copy())
                objective_path.append(objective)
                flux_path.append(selected_flux.copy())
                accepted = True
                break
        if not accepted:
            stop_reason = "line_search"
            break
        if previous_objective - objective < fit.minimum_objective_improvement:
            converged = True
            stop_reason = "objective_improvement"
            break

    return FitResult(
        names=fit.names,
        parameters=parameters,
        continuum_coefficients=coefficients,
        model_flux=selected_flux,
        mean_weighted_squared_residual=objective,
        converged=converged,
        stop_reason=stop_reason,
        parameter_path=np.asarray(parameter_path),
        objective_path=np.asarray(objective_path),
        model_seconds_path=np.asarray(model_times),
        model_flux_path=np.asarray(flux_path),
    )


def refine_with_physical_atmosphere(
    spectrum: NormalizedSpectrum,
    configuration: FitConfiguration,
    fast_result: FitResult,
    fast_model: ModelFunction,
    converged_model: ConvergedModelFunction,
    physical_configuration: PhysicalAtmosphereConfiguration,
    *,
    continuum_basis: Array | None = None,
) -> PhysicalAtmosphereResult:
    """Check and, when useful, refine a fast fit with converged atmospheres.

    ``fast_model`` and ``converged_model`` return physical normalized flux on
    the observed grid. The latter must solve the atmosphere to convergence and
    synthesize from that atmosphere. Both paths are profiled through the same
    linear continuum basis before comparison.

    When either declared gate fails, the method holds the local
    converged-minus-fast flux discrepancy fixed and builds a fresh bounded,
    one-sided fast-model Jacobian at the accepted physical point. A tighter
    trust-region proposal is sent to the converged model only when its predicted
    objective gain exceeds the declared threshold. The proposal is accepted
    only when the converged model improves the current objective. Rejected
    proposals shrink the trust region while reusing that unchanged local
    Jacobian. An accepted move always rebuilds it. Plotting time is never
    included.
    """

    started_total = time.perf_counter()
    observed = spectrum.validated()
    fit = configuration.validated()
    controls = physical_configuration.validated()
    correction_derivative_steps = (
        fit.derivative_steps
        if controls.correction_derivative_steps is None
        else controls.correction_derivative_steps
    )
    if correction_derivative_steps.shape != (len(fit.names),):
        raise ValueError("correction_derivative_steps must match parameter names")
    if (
        controls.correction_trust_half_width is not None
        and controls.correction_trust_half_width.shape != (len(fit.names),)
    ):
        raise ValueError("correction_trust_half_width must match parameter names")
    if controls.correction_trust_half_width is None:
        initial_correction_trust = (
            fit.trust_half_width * controls.correction_trust_fraction
        )
        minimum_correction_trust = (
            fit.trust_half_width * controls.minimum_trust_fraction
        )
    else:
        initial_correction_trust = controls.correction_trust_half_width.copy()
        minimum_correction_trust = initial_correction_trust * (
            controls.minimum_trust_fraction / controls.correction_trust_fraction
        )

    def immutable_vector(value: Array) -> Array:
        selected = np.asarray(value, np.float64)
        return np.frombuffer(selected.tobytes(), dtype=np.float64).reshape(
            selected.shape
        )

    resolved_controls = ResolvedPhysicalAtmosphereControls(
        maximum_discrepancy_rms=controls.maximum_discrepancy_rms,
        maximum_objective_degradation=controls.maximum_objective_degradation,
        minimum_predicted_objective_improvement=(
            controls.minimum_predicted_objective_improvement
        ),
        maximum_physical_evaluations=controls.maximum_physical_evaluations,
        parameter_lower_bounds=immutable_vector(fit.lower),
        parameter_upper_bounds=immutable_vector(fit.upper),
        correction_derivative_steps=immutable_vector(
            correction_derivative_steps
        ),
        correction_trust_half_width=immutable_vector(initial_correction_trust),
        minimum_trust_half_width=immutable_vector(minimum_correction_trust),
        trust_shrink_factor=controls.trust_shrink_factor,
        minimum_scaled_step=controls.minimum_scaled_step,
        minimum_physical_objective_improvement=(
            controls.minimum_physical_objective_improvement
        ),
        line_search_fractions=controls.line_search_fractions,
    )
    basis = (
        np.empty((observed.flux.size, 0), np.float64)
        if continuum_basis is None
        else np.asarray(continuum_basis, np.float64)
    )
    if basis.ndim != 2 or basis.shape[0] != observed.flux.size:
        raise ValueError("continuum_basis must have shape [pixel, coefficient]")
    if not np.all(np.isfinite(basis)):
        raise ValueError("continuum_basis must be finite")
    if fast_result.names != fit.names:
        raise ValueError("fast_result parameter names must match configuration")

    fast_candidate = np.asarray(fast_result.parameters, np.float64)
    if (
        fast_candidate.shape != (len(fit.names),)
        or not np.all(np.isfinite(fast_candidate))
        or np.any((fast_candidate < fit.lower) | (fast_candidate > fit.upper))
    ):
        raise ValueError("fast_result parameters are invalid for configuration")

    def evaluate_model(
        callback: ModelFunction,
        parameters: Array,
        callback_name: str,
    ) -> tuple[Array, float]:
        started = time.perf_counter()
        physical_flux = np.asarray(callback(parameters.copy()), np.float64)
        seconds = time.perf_counter() - started
        if physical_flux.shape != observed.flux.shape or not np.all(
            np.isfinite(physical_flux)
        ):
            raise ValueError(f"{callback_name} returned an invalid flux array")
        return physical_flux, seconds

    def gate_values(
        fast_flux: Array,
        fast_objective: float,
        physical_flux: Array,
        physical_objective: float,
    ) -> tuple[float, float, bool, bool]:
        difference = physical_flux[observed.mask] - fast_flux[observed.mask]
        discrepancy_rms = float(np.sqrt(np.mean(difference**2)))
        objective_degradation = physical_objective - fast_objective
        return (
            discrepancy_rms,
            objective_degradation,
            discrepancy_rms <= controls.maximum_discrepancy_rms,
            objective_degradation <= controls.maximum_objective_degradation,
        )

    checks: list[PhysicalAtmosphereCheck] = []
    corrections: list[PhysicalAtmosphereCorrection] = []

    current_parameters = fast_candidate.copy()
    current_fast_physical, fast_seconds = evaluate_model(
        fast_model, current_parameters, "fast_model"
    )
    (
        current_fast_flux,
        current_fast_coefficients,
        current_fast_objective,
    ) = _profile_continuum(current_fast_physical, observed, basis)
    current_physical_physical, physical_seconds = evaluate_model(
        converged_model, current_parameters, "converged_model"
    )
    (
        current_physical_flux,
        current_physical_coefficients,
        current_physical_objective,
    ) = _profile_continuum(current_physical_physical, observed, basis)
    (
        discrepancy_rms,
        objective_degradation,
        discrepancy_passed,
        objective_passed,
    ) = gate_values(
        current_fast_flux,
        current_fast_objective,
        current_physical_flux,
        current_physical_objective,
    )
    checks.append(
        PhysicalAtmosphereCheck(
            parameters=current_parameters.copy(),
            fast_continuum_coefficients=current_fast_coefficients.copy(),
            physical_continuum_coefficients=current_physical_coefficients.copy(),
            fast_model_flux=current_fast_flux.copy(),
            physical_model_flux=current_physical_flux.copy(),
            fast_objective=current_fast_objective,
            physical_objective=current_physical_objective,
            discrepancy_rms=discrepancy_rms,
            objective_degradation=objective_degradation,
            discrepancy_gate_passed=discrepancy_passed,
            objective_gate_passed=objective_passed,
            accepted=True,
            fast_model_reused=False,
            fast_model_seconds=fast_seconds,
            physical_model_seconds=physical_seconds,
        )
    )
    current_check_index = 0

    def finish(
        physical_fit_stationary: bool, stop_reason: str
    ) -> PhysicalAtmosphereResult:
        accepted_check = checks[current_check_index]
        gates_passed = bool(
            accepted_check.discrepancy_gate_passed
            and accepted_check.objective_gate_passed
        )
        return PhysicalAtmosphereResult(
            names=fit.names,
            parameters=current_parameters.copy(),
            continuum_coefficients=current_physical_coefficients.copy(),
            model_flux=current_physical_flux.copy(),
            mean_weighted_squared_residual=current_physical_objective,
            physical_fit_stationary=physical_fit_stationary,
            fast_physical_gates_passed=gates_passed,
            successful=bool(physical_fit_stationary or gates_passed),
            stop_reason=stop_reason,
            fast_candidate_parameters=fast_candidate.copy(),
            physical_checks=tuple(checks),
            corrections=tuple(corrections),
            controls=resolved_controls,
            total_seconds_excluding_plotting=time.perf_counter() - started_total,
        )

    if discrepancy_passed and objective_passed:
        return finish(False, "physical_gates")

    trust_half_width = initial_correction_trust.copy()
    minimum_trust = minimum_correction_trust
    cached_matrix: Array | None = None
    cached_fixed_discrepancy: Array | None = None

    while len(checks) < controls.maximum_physical_evaluations:
        if np.all(trust_half_width <= minimum_trust):
            return finish(False, "minimum_trust_region")

        correction_started = time.perf_counter()
        correction_fast_seconds = 0.0
        correction_fast_evaluations = 0
        if cached_matrix is None:
            fixed_discrepancy = (
                current_physical_physical - current_fast_physical
            )
            matrix = np.empty(
                (observed.flux.size, len(fit.names)), np.float64
            )
            for index, requested_step in enumerate(correction_derivative_steps):
                upper_room = fit.upper[index] - current_parameters[index]
                lower_room = current_parameters[index] - fit.lower[index]
                if upper_room >= requested_step:
                    signed_step = requested_step
                elif lower_room >= requested_step:
                    signed_step = -requested_step
                elif upper_room >= lower_room and upper_room > 0.0:
                    signed_step = upper_room
                else:
                    signed_step = -lower_room
                if signed_step == 0.0:
                    raise ValueError(
                        "a finite-difference parameter has no bounded room"
                    )

                perturbed = current_parameters.copy()
                perturbed[index] += signed_step
                perturbed_fast_physical, seconds = evaluate_model(
                    fast_model, perturbed, "fast_model"
                )
                correction_fast_seconds += seconds
                correction_fast_evaluations += 1
                perturbed_flux, _, _ = _profile_continuum(
                    perturbed_fast_physical + fixed_discrepancy,
                    observed,
                    basis,
                )
                matrix[:, index] = (
                    perturbed_flux - current_physical_flux
                ) / signed_step
            cached_matrix = matrix
            cached_fixed_discrepancy = fixed_discrepancy
        else:
            matrix = cached_matrix
            if cached_fixed_discrepancy is None:
                raise RuntimeError("cached correction discrepancy is unavailable")
            fixed_discrepancy = cached_fixed_discrepancy

        good = observed.mask
        square_root_weight = np.sqrt(observed.inverse_variance[good])
        weighted_matrix = matrix[good] * square_root_weight[:, None]
        weighted_residual = (
            current_physical_flux[good] - observed.flux[good]
        ) * square_root_weight
        scaled_matrix = weighted_matrix * trust_half_width[None, :]
        scaled_delta = np.linalg.lstsq(
            scaled_matrix, -weighted_residual, rcond=None
        )[0]
        scaled_delta = np.clip(scaled_delta, -1.0, 1.0)
        scaled_step_norm = float(np.linalg.norm(scaled_delta))

        if scaled_step_norm < controls.minimum_scaled_step:
            corrections.append(
                PhysicalAtmosphereCorrection(
                    base_physical_index=current_check_index,
                    trust_half_width=trust_half_width.copy(),
                    proposed_parameters=current_parameters.copy(),
                    proposed_scaled_step_norm=scaled_step_norm,
                    selected_line_search_fraction=np.nan,
                    actual_scaled_step_norm=0.0,
                    actual_parameter_step_norm=0.0,
                    predicted_objective=current_physical_objective,
                    predicted_improvement=0.0,
                    physical_evaluated=False,
                    physical_objective=np.nan,
                    actual_improvement=np.nan,
                    actual_to_predicted_ratio=np.nan,
                    accepted=False,
                    fast_evaluation_count=correction_fast_evaluations,
                    fast_model_seconds=correction_fast_seconds,
                    correction_seconds=max(
                        0.0,
                        time.perf_counter()
                        - correction_started
                        - correction_fast_seconds,
                    ),
                    outcome="scaled_step",
                )
            )
            return finish(True, "scaled_step")

        full_delta = scaled_delta * trust_half_width
        best_parameters = current_parameters.copy()
        best_fast_physical = current_fast_physical
        best_fast_flux = current_fast_flux
        best_fast_coefficients = current_fast_coefficients
        best_fast_objective = current_fast_objective
        best_predicted_objective = current_physical_objective
        found_new_candidate = False
        selected_fraction = np.nan

        for fraction in controls.line_search_fractions:
            candidate = np.clip(
                current_parameters + fraction * full_delta,
                fit.lower,
                fit.upper,
            )
            if any(
                np.array_equal(candidate, check.parameters) for check in checks
            ):
                continue
            found_new_candidate = True
            candidate_fast_physical, seconds = evaluate_model(
                fast_model, candidate, "fast_model"
            )
            correction_fast_seconds += seconds
            correction_fast_evaluations += 1
            candidate_fast_flux, candidate_fast_coefficients, candidate_fast_objective = (
                _profile_continuum(candidate_fast_physical, observed, basis)
            )
            _, _, candidate_predicted_objective = _profile_continuum(
                candidate_fast_physical + fixed_discrepancy,
                observed,
                basis,
            )
            if candidate_predicted_objective < best_predicted_objective:
                best_parameters = candidate
                best_fast_physical = candidate_fast_physical
                best_fast_flux = candidate_fast_flux
                best_fast_coefficients = candidate_fast_coefficients
                best_fast_objective = candidate_fast_objective
                best_predicted_objective = candidate_predicted_objective
                selected_fraction = float(fraction)
                break

        if not found_new_candidate:
            corrections.append(
                PhysicalAtmosphereCorrection(
                    base_physical_index=current_check_index,
                    trust_half_width=trust_half_width.copy(),
                    proposed_parameters=current_parameters.copy(),
                    proposed_scaled_step_norm=scaled_step_norm,
                    selected_line_search_fraction=np.nan,
                    actual_scaled_step_norm=0.0,
                    actual_parameter_step_norm=0.0,
                    predicted_objective=current_physical_objective,
                    predicted_improvement=0.0,
                    physical_evaluated=False,
                    physical_objective=np.nan,
                    actual_improvement=np.nan,
                    actual_to_predicted_ratio=np.nan,
                    accepted=False,
                    fast_evaluation_count=correction_fast_evaluations,
                    fast_model_seconds=correction_fast_seconds,
                    correction_seconds=max(
                        0.0,
                        time.perf_counter()
                        - correction_started
                        - correction_fast_seconds,
                    ),
                    outcome="no_new_physical_candidate",
                )
            )
            return finish(False, "no_new_physical_candidate")

        predicted_improvement = (
            current_physical_objective - best_predicted_objective
        )
        actual_delta = best_parameters - current_parameters
        actual_scaled_step_norm = float(
            np.linalg.norm(actual_delta / trust_half_width)
        )
        actual_parameter_step_norm = float(np.linalg.norm(actual_delta))
        correction_seconds = max(
            0.0,
            time.perf_counter() - correction_started - correction_fast_seconds,
        )
        if (
            predicted_improvement
            <= controls.minimum_predicted_objective_improvement
        ):
            corrections.append(
                PhysicalAtmosphereCorrection(
                    base_physical_index=current_check_index,
                    trust_half_width=trust_half_width.copy(),
                    proposed_parameters=best_parameters.copy(),
                    proposed_scaled_step_norm=scaled_step_norm,
                    selected_line_search_fraction=selected_fraction,
                    actual_scaled_step_norm=actual_scaled_step_norm,
                    actual_parameter_step_norm=actual_parameter_step_norm,
                    predicted_objective=best_predicted_objective,
                    predicted_improvement=predicted_improvement,
                    physical_evaluated=False,
                    physical_objective=np.nan,
                    actual_improvement=np.nan,
                    actual_to_predicted_ratio=np.nan,
                    accepted=False,
                    fast_evaluation_count=correction_fast_evaluations,
                    fast_model_seconds=correction_fast_seconds,
                    correction_seconds=correction_seconds,
                    outcome="minimum_predicted_improvement",
                )
            )
            return finish(True, "minimum_predicted_improvement")

        candidate_physical_physical, physical_seconds = evaluate_model(
            converged_model, best_parameters, "converged_model"
        )
        (
            candidate_physical_flux,
            candidate_physical_coefficients,
            candidate_physical_objective,
        ) = _profile_continuum(candidate_physical_physical, observed, basis)
        actual_improvement = (
            current_physical_objective - candidate_physical_objective
        )
        accepted = (
            actual_improvement
            > controls.minimum_physical_objective_improvement
        )
        ratio = actual_improvement / predicted_improvement
        (
            discrepancy_rms,
            objective_degradation,
            discrepancy_passed,
            objective_passed,
        ) = gate_values(
            best_fast_flux,
            best_fast_objective,
            candidate_physical_flux,
            candidate_physical_objective,
        )
        checks.append(
            PhysicalAtmosphereCheck(
                parameters=best_parameters.copy(),
                fast_continuum_coefficients=best_fast_coefficients.copy(),
                physical_continuum_coefficients=(
                    candidate_physical_coefficients.copy()
                ),
                fast_model_flux=best_fast_flux.copy(),
                physical_model_flux=candidate_physical_flux.copy(),
                fast_objective=best_fast_objective,
                physical_objective=candidate_physical_objective,
                discrepancy_rms=discrepancy_rms,
                objective_degradation=objective_degradation,
                discrepancy_gate_passed=discrepancy_passed,
                objective_gate_passed=objective_passed,
                accepted=accepted,
                fast_model_reused=True,
                fast_model_seconds=0.0,
                physical_model_seconds=physical_seconds,
            )
        )
        corrections.append(
            PhysicalAtmosphereCorrection(
                base_physical_index=current_check_index,
                trust_half_width=trust_half_width.copy(),
                proposed_parameters=best_parameters.copy(),
                proposed_scaled_step_norm=scaled_step_norm,
                selected_line_search_fraction=selected_fraction,
                actual_scaled_step_norm=actual_scaled_step_norm,
                actual_parameter_step_norm=actual_parameter_step_norm,
                predicted_objective=best_predicted_objective,
                predicted_improvement=predicted_improvement,
                physical_evaluated=True,
                physical_objective=candidate_physical_objective,
                actual_improvement=actual_improvement,
                actual_to_predicted_ratio=ratio,
                accepted=accepted,
                fast_evaluation_count=correction_fast_evaluations,
                fast_model_seconds=correction_fast_seconds,
                correction_seconds=correction_seconds,
                outcome="accepted" if accepted else "rejected",
            )
        )

        if accepted:
            cached_matrix = None
            cached_fixed_discrepancy = None
            current_check_index = len(checks) - 1
            current_parameters = best_parameters.copy()
            current_fast_physical = best_fast_physical
            current_fast_flux = best_fast_flux
            current_fast_coefficients = best_fast_coefficients
            current_fast_objective = best_fast_objective
            current_physical_physical = candidate_physical_physical
            current_physical_flux = candidate_physical_flux
            current_physical_coefficients = candidate_physical_coefficients
            current_physical_objective = candidate_physical_objective
            if discrepancy_passed and objective_passed:
                return finish(False, "physical_gates")
        else:
            trust_half_width *= controls.trust_shrink_factor

    return finish(False, "maximum_physical_evaluations")
