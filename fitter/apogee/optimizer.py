#!/usr/bin/env python3
"""Normalized APOGEE fitter with stellar, velocity, broadening, and continuum terms."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any

import numpy as np
import torch


from fitter.apogee.forward_model import (
    LABEL_LOWER,
    LABEL_NAMES,
    LABEL_UPPER,
    FastForwardModel,
    _resolve_synthesis_r_grid,
)
from fitter.apogee.spectral_nuisance import (
    APOGEEContinuumProfiler,
    APOGEESpectralNuisance,
    CoarseVelocityResult,
    estimate_coarse_velocity,
)


NUISANCE_NAMES = ("residual_rv_km_s", "vmacro_km_s")
PARAMETER_NAMES = LABEL_NAMES + NUISANCE_NAMES
PARAMETER_LOWER = np.concatenate((LABEL_LOWER, [-15.0, 0.0]))
PARAMETER_UPPER = np.concatenate((LABEL_UPPER, [15.0, 20.0]))
DERIVATIVE_STEPS = np.asarray([100.0, 0.10, 0.10, 0.05, 0.20, 0.25, 0.50])
LOCAL_DERIVATIVE_STEPS = np.asarray([50.0, 0.05, 0.05, 0.025, 0.10, 0.10, 0.25])
INITIAL_TRUST_HALF_WIDTH = np.asarray([350.0, 0.45, 0.40, 0.25, 0.70, 5.0, 4.0])
CORRECTION_TRUST_HALF_WIDTH = np.asarray([200.0, 0.25, 0.25, 0.15, 0.80, 3.0, 3.0])
LABEL_TOLERANCE = np.asarray([75.0, 0.08, 0.05, 0.05, 0.20])
CNO_LABEL_NAMES = (
    "carbon_enhancement",
    "nitrogen_enhancement",
    "oxygen_enhancement",
)
CNO_LABEL_LOWER = np.full(3, -0.499)
CNO_LABEL_UPPER = np.full(3, 0.499)
CNO_DERIVATIVE_STEPS = np.full(3, 0.05)
CNO_LOCAL_DERIVATIVE_STEPS = np.full(3, 0.025)
CNO_INITIAL_TRUST_HALF_WIDTH = np.full(3, 0.25)
CNO_CORRECTION_TRUST_HALF_WIDTH = np.full(3, 0.15)
CNO_LABEL_TOLERANCE = np.full(3, 0.08)
MIN_PREDICTED_REDUCED_CHI_SQUARE_IMPROVEMENT = 2.0e-4
MAX_CHORD_CORRECTIONS = 8
INITIAL_LABEL_MODES = ("reference", "controlled_offset")
INITIAL_RV_MODES = ("rest_frame", "coarse_ccf")
CONTROLLED_INITIAL_LABEL_OFFSET = np.asarray(
    [200.0, 0.25, 0.20, -0.10, 0.30], np.float64
)
CONTROLLED_INITIAL_CNO_OFFSET = np.asarray([0.10, -0.10, 0.05], np.float64)


def _initial_stellar_labels(
    reference_labels: np.ndarray,
    *,
    fit_cno8: bool,
    mode: str,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return clipped initial labels and the requested controlled offset."""

    if mode not in INITIAL_LABEL_MODES:
        raise ValueError(f"initial_label_mode must be one of {INITIAL_LABEL_MODES}")
    reference = np.asarray(reference_labels, np.float64)
    offset = np.zeros_like(reference)
    if mode == "controlled_offset":
        offset = CONTROLLED_INITIAL_LABEL_OFFSET.copy()
        if fit_cno8:
            offset = np.concatenate((offset, CONTROLLED_INITIAL_CNO_OFFSET))
    return np.clip(reference + offset, lower, upper), offset


def _initial_residual_velocity(
    mode: str,
    *,
    observed_flux: np.ndarray,
    template_flux: np.ndarray,
    wavelength_nm: np.ndarray,
    apstar_pixel: np.ndarray,
    inverse_variance: np.ndarray,
    good_pixel_mask: np.ndarray,
    maximum_velocity_km_s: float,
    device: str,
    dtype: torch.dtype,
) -> tuple[float, CoarseVelocityResult | None, float]:
    """Return the initial residual RV, optionally from a coarse CCF."""

    if mode not in INITIAL_RV_MODES:
        raise ValueError(f"initial_rv_mode must be one of {INITIAL_RV_MODES}")
    if mode == "rest_frame":
        return 0.0, None, 0.0
    start = time.perf_counter()
    ccf = estimate_coarse_velocity(
        observed_flux,
        template_flux,
        wavelength_nm=wavelength_nm,
        apstar_pixel=apstar_pixel,
        inverse_variance=inverse_variance,
        good_pixel_mask=good_pixel_mask,
        maximum_velocity_km_s=maximum_velocity_km_s,
        device=device,
        dtype=dtype,
    )
    seconds = time.perf_counter() - start
    initial_rv = float(
        np.clip(
            ccf.radial_velocity_km_s,
            -maximum_velocity_km_s,
            maximum_velocity_km_s,
        )
    )
    return initial_rv, ccf, seconds


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    temporary.replace(path)


def _portable_path(path: Path) -> str:
    """Return an install-independent absolute result path."""

    return str(path.expanduser().resolve())


def _parameter_dict(
    values: np.ndarray, names: tuple[str, ...] = PARAMETER_NAMES
) -> dict[str, float]:
    return {name: float(value) for name, value in zip(names, values, strict=True)}


def _parameter_configuration(fit_cno8: bool) -> dict[str, object]:
    """Return the ordered bounds and optimizer scales for one label family."""

    if not fit_cno8:
        return {
            "stellar_names": LABEL_NAMES,
            "parameter_names": PARAMETER_NAMES,
            "lower": PARAMETER_LOWER,
            "upper": PARAMETER_UPPER,
            "derivative_steps": DERIVATIVE_STEPS,
            "local_derivative_steps": LOCAL_DERIVATIVE_STEPS,
            "initial_trust": INITIAL_TRUST_HALF_WIDTH,
            "correction_trust": CORRECTION_TRUST_HALF_WIDTH,
            "label_tolerance": LABEL_TOLERANCE,
        }
    stellar_names = LABEL_NAMES + CNO_LABEL_NAMES
    return {
        "stellar_names": stellar_names,
        "parameter_names": stellar_names + NUISANCE_NAMES,
        "lower": np.concatenate((LABEL_LOWER, CNO_LABEL_LOWER, [-15.0, 0.0])),
        "upper": np.concatenate((LABEL_UPPER, CNO_LABEL_UPPER, [15.0, 20.0])),
        "derivative_steps": np.concatenate(
            (DERIVATIVE_STEPS[:5], CNO_DERIVATIVE_STEPS, DERIVATIVE_STEPS[5:])
        ),
        "local_derivative_steps": np.concatenate(
            (
                LOCAL_DERIVATIVE_STEPS[:5],
                CNO_LOCAL_DERIVATIVE_STEPS,
                LOCAL_DERIVATIVE_STEPS[5:],
            )
        ),
        "initial_trust": np.concatenate(
            (
                INITIAL_TRUST_HALF_WIDTH[:5],
                CNO_INITIAL_TRUST_HALF_WIDTH,
                INITIAL_TRUST_HALF_WIDTH[5:],
            )
        ),
        "correction_trust": np.concatenate(
            (
                CORRECTION_TRUST_HALF_WIDTH[:5],
                CNO_CORRECTION_TRUST_HALF_WIDTH,
                CORRECTION_TRUST_HALF_WIDTH[5:],
            )
        ),
        "label_tolerance": np.concatenate((LABEL_TOLERANCE, CNO_LABEL_TOLERANCE)),
    }


def _broyden_secant_update(
    jacobian: np.ndarray,
    *,
    start_parameters: np.ndarray,
    end_parameters: np.ndarray,
    start_flux: np.ndarray,
    end_flux: np.ndarray,
    parameter_scale: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Apply one scaled minimum-change Jacobian update along an accepted step."""

    matrix = np.asarray(jacobian, np.float64)
    scale = np.asarray(parameter_scale, np.float64)
    parameter_step = (
        np.asarray(end_parameters, np.float64)
        - np.asarray(start_parameters, np.float64)
    ) / scale
    denominator = float(parameter_step @ parameter_step)
    flux_step = np.asarray(end_flux, np.float64) - np.asarray(start_flux, np.float64)
    if denominator <= 1.0e-24:
        return matrix.copy(), {
            "applied": False,
            "scaled_parameter_step_norm": float(np.sqrt(denominator)),
            "relative_secant_mismatch_before": 0.0,
        }
    scaled_jacobian = matrix * scale[None, :]
    mismatch = flux_step - scaled_jacobian @ parameter_step
    flux_norm = max(float(np.linalg.norm(flux_step)), np.finfo(np.float64).tiny)
    relative_mismatch = float(np.linalg.norm(mismatch) / flux_norm)
    scaled_jacobian += np.outer(mismatch, parameter_step) / denominator
    return scaled_jacobian / scale[None, :], {
        "applied": True,
        "scaled_parameter_step_norm": float(np.sqrt(denominator)),
        "relative_secant_mismatch_before": relative_mismatch,
    }


@dataclass
class ModelEvaluation:
    event_index: int
    parameters: np.ndarray
    flux: np.ndarray
    continuum_coefficients: np.ndarray
    objective: float
    chi_square: float
    synthesis_forward: bool
    model_seconds: float
    timings: dict[str, float]


@dataclass(frozen=True)
class NormalizedSpectrumInput:
    """One real normalized spectrum and its catalog comparison point."""

    object_id: str
    wavelength_nm: np.ndarray
    observed_flux: np.ndarray
    inverse_variance: np.ndarray
    good_pixel_mask: np.ndarray
    catalog_labels: np.ndarray
    catalog_vmacro_km_s: float
    metadata: dict[str, Any]
    reference_residual_rv_km_s: float = 0.0
    data_mode: str = "apogee_normalized_spectrum"
    reference_is_truth: bool = False
    carbon_enhancement: float | None = None
    nitrogen_enhancement: float | None = None
    oxygen_enhancement: float | None = None
    atomic_calibration_path: Path | None = None


class OptimizationTrace:
    """Append-safe extended-parameter events and spectra."""

    def __init__(
        self,
        path: Path,
        wavelength_nm: np.ndarray,
        *,
        parameter_names: tuple[str, ...] = PARAMETER_NAMES,
        stellar_names: tuple[str, ...] = LABEL_NAMES,
        store_spectra: bool = True,
    ) -> None:
        self.path = path
        self.spectra_dir = path / "spectra"
        self.path.mkdir(parents=True, exist_ok=True)
        self.store_spectra = store_spectra
        if store_spectra:
            self.spectra_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.path / "events.jsonl"
        if self.events_path.exists():
            self.events_path.unlink()
        self.wavelength_nm = np.asarray(wavelength_nm, np.float64)
        self.parameter_names = parameter_names
        self.stellar_names = stellar_names
        self.events: list[dict[str, Any]] = []
        self.spectra: list[np.ndarray] = []

    def append(
        self,
        *,
        parameters: np.ndarray,
        flux: np.ndarray,
        continuum_coefficients: np.ndarray,
        role: str,
        objective: float,
        chi_square: float,
        synthesis_forward: bool,
        model_seconds: float,
        timings: dict[str, float],
    ) -> int:
        index = len(self.events)
        spectrum_path = None
        if self.store_spectra:
            spectrum_path = self.spectra_dir / f"evaluation_{index:03d}.npy"
            np.save(spectrum_path, np.asarray(flux, np.float64), allow_pickle=False)
        event = {
            "evaluation": index,
            "role": role,
            "parameters": _parameter_dict(parameters, self.parameter_names),
            "labels": {
                name: float(value)
                for name, value in zip(
                    self.stellar_names,
                    parameters[: len(self.stellar_names)],
                    strict=True,
                )
            },
            "nuisance": {
                name: float(value)
                for name, value in zip(
                    NUISANCE_NAMES,
                    parameters[len(self.stellar_names) :],
                    strict=True,
                )
            },
            "continuum_coefficients": np.asarray(continuum_coefficients, np.float64),
            "objective": float(objective),
            "chi_square": float(chi_square),
            "synthesis_forward": bool(synthesis_forward),
            "model_seconds": float(model_seconds),
            "accepted_path": False,
            "spectrum_path": (
                None if spectrum_path is None else _portable_path(spectrum_path)
            ),
            **{name: float(value) for name, value in timings.items()},
        }
        self.events.append(event)
        if self.store_spectra:
            self.spectra.append(np.asarray(flux, np.float64))
        with self.events_path.open("a") as handle:
            handle.write(json.dumps(event, default=_json_default) + "\n")
            handle.flush()
        return index

    def mark_accepted(self, indices: list[int]) -> None:
        for index in indices:
            self.events[index]["accepted_path"] = True

    def consolidate(self) -> Path:
        self.events_path.write_text(
            "".join(
                json.dumps(event, default=_json_default) + "\n" for event in self.events
            )
        )
        output = self.path / "optimization_trace.npz"
        arrays = {
            "wavelength_nm": self.wavelength_nm,
            "parameter_names": np.asarray(self.parameter_names),
            "parameters": np.asarray(
                [
                    [event["parameters"][name] for name in self.parameter_names]
                    for event in self.events
                ],
                np.float64,
            ),
            "continuum_coefficients": np.asarray(
                [event["continuum_coefficients"] for event in self.events], np.float64
            ),
            "objective": np.asarray([event["objective"] for event in self.events]),
            "model_seconds": np.asarray(
                [event["model_seconds"] for event in self.events]
            ),
            "synthesis_forward": np.asarray(
                [event["synthesis_forward"] for event in self.events], np.uint8
            ),
            "accepted_path": np.asarray(
                [event["accepted_path"] for event in self.events], np.uint8
            ),
            "role": np.asarray([event["role"] for event in self.events]),
        }
        if self.store_spectra:
            arrays["normalized_model_flux"] = np.asarray(self.spectra, np.float64)
        np.savez_compressed(output, **arrays)
        return output


def run_one_star(
    result_dir: Path,
    *,
    device: str,
    dtype: str,
    synthesis_r_grid: float | None = None,
    synthesis_resolution: float | None = None,
    fresh_jacobian_rounds: int,
    force: bool,
    spectrum_input: NormalizedSpectrumInput,
    continuum_order: int = 2,
    fit_cno8: bool = False,
    compact_trace: bool = False,
    initial_label_mode: str = "reference",
    initial_rv_mode: str = "rest_frame",
) -> dict[str, Any]:
    """Fit one observed APOGEE spectrum with profiled continuum parameters.

    By default the supplied catalog labels initialize the stellar search and
    the residual velocity starts at zero, as appropriate for an already
    rest-framed APOGEE spectrum. ``controlled_offset`` reproduces the fixed
    displaced-label start used by the controlled recovery experiment.
    ``coarse_ccf`` estimates and uses a residual-velocity start from the input
    spectrum and the initial synthetic template.
    """

    synthesis_r_grid = _resolve_synthesis_r_grid(synthesis_r_grid, synthesis_resolution)
    summary_path = result_dir / "summary.json"
    configuration = _parameter_configuration(fit_cno8)
    stellar_names = configuration["stellar_names"]
    parameter_names = configuration["parameter_names"]
    parameter_lower = np.asarray(configuration["lower"], np.float64)
    parameter_upper = np.asarray(configuration["upper"], np.float64)
    derivative_steps = np.asarray(configuration["derivative_steps"], np.float64)
    local_derivative_steps = np.asarray(
        configuration["local_derivative_steps"], np.float64
    )
    initial_trust_half_width = np.asarray(configuration["initial_trust"], np.float64)
    correction_trust_half_width = np.asarray(
        configuration["correction_trust"], np.float64
    )
    stellar_count = len(stellar_names)
    nuisance_start = stellar_count
    if continuum_order < 0:
        raise ValueError("continuum_order must be nonnegative")
    if initial_label_mode not in INITIAL_LABEL_MODES:
        raise ValueError(f"initial_label_mode must be one of {INITIAL_LABEL_MODES}")
    if initial_rv_mode not in INITIAL_RV_MODES:
        raise ValueError(f"initial_rv_mode must be one of {INITIAL_RV_MODES}")
    if summary_path.is_file() and not force:
        summary = json.loads(summary_path.read_text())
        stored_r_grid = summary.get(
            "synthesis_r_grid", summary.get("synthesis_resolution")
        )
        if (
            stored_r_grid is None
            or float(stored_r_grid) != synthesis_r_grid
            or summary.get("data_mode") != spectrum_input.data_mode
            or int(summary.get("continuum_order", 2)) != continuum_order
            or bool(summary.get("fit_cno8", False)) != fit_cno8
            or tuple(summary.get("stellar_label_names", ())) != tuple(stellar_names)
            or summary.get("initial_label_mode") != initial_label_mode
            or summary.get("initial_rv_mode") != initial_rv_mode
        ):
            raise RuntimeError(
                "existing fit uses a different model or initialization "
                "configuration; select another result directory or pass --force"
            )
        return summary
    if result_dir.exists() and force:
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    reference_labels = np.asarray(spectrum_input.catalog_labels, np.float64)
    if fit_cno8:
        cno = (
            spectrum_input.carbon_enhancement,
            spectrum_input.nitrogen_enhancement,
            spectrum_input.oxygen_enhancement,
        )
        if any(value is None for value in cno):
            raise ValueError("fit_cno8 requires initial C, N, and O enhancements")
        reference_labels = np.concatenate(
            (reference_labels, np.asarray(cno, np.float64))
        )
    reference_rv = float(spectrum_input.reference_residual_rv_km_s)
    reference_vmacro = float(spectrum_input.catalog_vmacro_km_s)
    wavelength_nm = np.asarray(spectrum_input.wavelength_nm, np.float64)
    observed_flux = np.asarray(spectrum_input.observed_flux, np.float64)
    inverse_variance = np.asarray(spectrum_input.inverse_variance, np.float64)
    good_pixel_mask = np.asarray(spectrum_input.good_pixel_mask, bool)

    forward = FastForwardModel(
        device=device,
        dtype=dtype,
        synthesis_r_grid=synthesis_r_grid,
        apogee_dr14_lsf=True,
        fit_spectral_nuisance=True,
        atomic_calibration_path=spectrum_input.atomic_calibration_path,
    )
    setup_seconds = forward.prepare_window()
    operator = forward.spectral_operator
    if not isinstance(operator, APOGEESpectralNuisance):
        raise RuntimeError("spectral nuisance operator was not constructed")
    likelihood_dtype = torch.float32 if device == "mps" else torch.float64
    profiler = APOGEEContinuumProfiler(
        apstar_pixel=operator.apstar_pixel,
        observed_flux=observed_flux,
        inverse_variance=inverse_variance,
        good_pixel_mask=good_pixel_mask,
        order=continuum_order,
        device=device,
        dtype=likelihood_dtype,
    )
    fit_mask = (
        good_pixel_mask
        & np.isfinite(observed_flux)
        & np.isfinite(inverse_variance)
        & (inverse_variance > 0.0)
    )
    fit_weight = np.sqrt(np.where(fit_mask, inverse_variance, 0.0))
    trace = OptimizationTrace(
        result_dir / "trace",
        wavelength_nm,
        parameter_names=parameter_names,
        stellar_names=stellar_names,
        store_spectra=not compact_trace,
    )
    optimizer_start = time.perf_counter()
    synthesis_forward_count = 0
    fixed_abundance_kwargs = (
        {}
        if fit_cno8
        else {
            "carbon_enhancement": spectrum_input.carbon_enhancement,
            "nitrogen_enhancement": spectrum_input.nitrogen_enhancement,
            "oxygen_enhancement": spectrum_input.oxygen_enhancement,
        }
    )

    def abundance_kwargs(stellar_parameters: np.ndarray) -> dict[str, object]:
        if fit_cno8:
            return {
                name: float(value)
                for name, value in zip(
                    CNO_LABEL_NAMES, stellar_parameters[5:8], strict=True
                )
            }
        return dict(fixed_abundance_kwargs)

    def record_profile(
        parameters: np.ndarray,
        model_tensor: torch.Tensor,
        *,
        role: str,
        synthesis_forward: bool,
        start: float,
        timings: dict[str, float],
    ) -> ModelEvaluation:
        profiled = profiler.profile(model_tensor)
        flux = profiled.flux.detach().cpu().numpy().astype(np.float64)
        coefficients = profiled.coefficients.detach().cpu().numpy().astype(np.float64)
        objective = float(profiled.objective.detach().cpu())
        chi_square = float(profiled.chi_square.detach().cpu())
        model_seconds = time.perf_counter() - start
        event_index = trace.append(
            parameters=parameters,
            flux=flux,
            continuum_coefficients=coefficients,
            role=role,
            objective=objective,
            chi_square=chi_square,
            synthesis_forward=synthesis_forward,
            model_seconds=model_seconds,
            timings=timings,
        )
        return ModelEvaluation(
            event_index=event_index,
            parameters=parameters.copy(),
            flux=flux,
            continuum_coefficients=coefficients,
            objective=objective,
            chi_square=chi_square,
            synthesis_forward=synthesis_forward,
            model_seconds=model_seconds,
            timings=timings,
        )

    def expensive_evaluation(parameters: np.ndarray, role: str) -> ModelEvaluation:
        nonlocal synthesis_forward_count
        start = time.perf_counter()
        _, model_flux, timings = forward.evaluate(
            parameters[:5],
            residual_rv_km_s=float(parameters[nuisance_start]),
            vmacro_km_s=float(parameters[nuisance_start + 1]),
            **abundance_kwargs(parameters[:stellar_count]),
        )
        synthesis_forward_count += 1
        model_tensor = torch.as_tensor(
            model_flux, dtype=profiler.dtype, device=profiler.device
        )
        return record_profile(
            parameters,
            model_tensor,
            role=role,
            synthesis_forward=True,
            start=start,
            timings=timings,
        )

    def cached_evaluation(parameters: np.ndarray, role: str) -> ModelEvaluation:
        start = time.perf_counter()
        model_tensor = forward.project_cached_nuisance(
            residual_rv_km_s=float(parameters[nuisance_start]),
            vmacro_km_s=float(parameters[nuisance_start + 1]),
        )
        return record_profile(
            parameters,
            model_tensor,
            role=role,
            synthesis_forward=False,
            start=start,
            timings={
                "kinematic_projection_seconds": operator.last_kinematic_seconds,
                "instrument_lsf_seconds": operator.lsf.last_seconds,
            },
        )

    initial_labels, initial_label_offset = _initial_stellar_labels(
        reference_labels,
        fit_cno8=fit_cno8,
        mode=initial_label_mode,
        lower=parameter_lower[:stellar_count],
        upper=parameter_upper[:stellar_count],
    )
    initial_vmacro = float(np.clip(reference_vmacro, 0.0, 20.0))
    template_start = time.perf_counter()
    _, initial_template, initial_timings = forward.evaluate(
        initial_labels[:5],
        residual_rv_km_s=0.0,
        vmacro_km_s=initial_vmacro,
        **abundance_kwargs(initial_labels),
    )
    synthesis_forward_count += 1
    initial_rv, ccf, ccf_seconds = _initial_residual_velocity(
        initial_rv_mode,
        observed_flux=observed_flux,
        template_flux=initial_template,
        wavelength_nm=wavelength_nm,
        apstar_pixel=operator.apstar_pixel,
        inverse_variance=inverse_variance,
        good_pixel_mask=good_pixel_mask,
        maximum_velocity_km_s=float(
            min(
                abs(parameter_lower[nuisance_start]),
                abs(parameter_upper[nuisance_start]),
            )
        ),
        device=device,
        dtype=profiler.dtype,
    )
    if initial_rv_mode == "rest_frame":
        initial_rv = float(
            np.clip(
                reference_rv,
                parameter_lower[nuisance_start],
                parameter_upper[nuisance_start],
            )
        )
    initial = np.concatenate(
        (
            initial_labels,
            [initial_rv, initial_vmacro],
        )
    )
    initial_tensor = forward.project_cached_nuisance(
        residual_rv_km_s=float(initial[nuisance_start]),
        vmacro_km_s=float(initial[nuisance_start + 1]),
    )
    if ccf is not None:
        initial_timings = {
            **initial_timings,
            "coarse_ccf_seconds": ccf_seconds,
            "coarse_ccf_peak": ccf.peak_correlation,
        }
    base = record_profile(
        initial,
        initial_tensor,
        role=f"initial_{initial_label_mode}_{initial_rv_mode}",
        synthesis_forward=True,
        start=template_start,
        timings=initial_timings,
    )
    if operator.last_native_pair is None:
        raise RuntimeError("initial synthesis did not populate the native flux cache")
    base_native_pair = operator.last_native_pair.detach().clone()

    jacobian = np.empty((wavelength_nm.size, len(parameter_names)), np.float64)
    # The native initial spectrum is still cached. Compute the two nuisance
    # columns first so these probes never repeat synthesis.
    for parameter_index in range(nuisance_start, nuisance_start + 2):
        step = derivative_steps[parameter_index]
        direction = (
            -1.0
            if initial[parameter_index] + step > parameter_upper[parameter_index]
            else 1.0
        )
        perturbed = initial.copy()
        perturbed[parameter_index] += direction * step
        event = cached_evaluation(
            perturbed, f"jacobian_{parameter_names[parameter_index]}"
        )
        jacobian[:, parameter_index] = (event.flux - base.flux) / (direction * step)
    for parameter_index in range(stellar_count):
        step = derivative_steps[parameter_index]
        direction = (
            -1.0
            if initial[parameter_index] + step > parameter_upper[parameter_index]
            else 1.0
        )
        perturbed = initial.copy()
        perturbed[parameter_index] += direction * step
        event = expensive_evaluation(
            perturbed, f"jacobian_{parameter_names[parameter_index]}"
        )
        jacobian[:, parameter_index] = (event.flux - base.flux) / (direction * step)

    weighted_jacobian = jacobian * fit_weight[:, None]
    weighted_target = (observed_flux - base.flux) * fit_weight
    raw_delta, _, rank, singular_values = np.linalg.lstsq(
        weighted_jacobian, weighted_target, rcond=1.0e-8
    )
    trusted_delta = np.clip(
        raw_delta, -initial_trust_half_width, initial_trust_half_width
    )
    candidate_parameters = np.clip(
        initial + trusted_delta, parameter_lower, parameter_upper
    )
    candidate = expensive_evaluation(candidate_parameters, "gauss_newton_candidate")
    if operator.last_native_pair is None:
        raise RuntimeError("candidate synthesis did not populate the native flux cache")
    candidate_native_pair = operator.last_native_pair.detach().clone()
    selected = candidate
    selected_native_pair = candidate_native_pair
    if candidate.objective >= base.objective:
        half_parameters = np.clip(
            initial + 0.5 * trusted_delta, parameter_lower, parameter_upper
        )
        half = expensive_evaluation(half_parameters, "half_step_fallback")
        if operator.last_native_pair is None:
            raise RuntimeError(
                "half-step synthesis did not populate the native flux cache"
            )
        half_native_pair = operator.last_native_pair.detach().clone()
        if half.objective < candidate.objective:
            selected = half
            selected_native_pair = half_native_pair
        else:
            selected = candidate
            selected_native_pair = candidate_native_pair
        if selected.objective >= base.objective:
            selected = base
            selected_native_pair = base_native_pair
    accepted_path = [base.event_index]
    secant_update_history: list[dict[str, Any]] = []
    if selected.event_index != base.event_index:
        accepted_path.append(selected.event_index)
        jacobian, secant = _broyden_secant_update(
            jacobian,
            start_parameters=base.parameters,
            end_parameters=selected.parameters,
            start_flux=base.flux,
            end_flux=selected.flux,
            parameter_scale=initial_trust_half_width,
        )
        secant.update(
            {
                "from_evaluation": base.event_index,
                "to_evaluation": selected.event_index,
            }
        )
        secant_update_history.append(secant)
        weighted_jacobian = jacobian * fit_weight[:, None]

    correction_history: list[dict[str, Any]] = []
    correction_stop_reason = "maximum_corrections_reached"
    for correction_index in range(1, MAX_CHORD_CORRECTIONS + 1):
        weighted_target = (observed_flux - selected.flux) * fit_weight
        correction_delta, _, _, _ = np.linalg.lstsq(
            weighted_jacobian, weighted_target, rcond=1.0e-8
        )
        trusted = np.clip(
            correction_delta,
            -correction_trust_half_width,
            correction_trust_half_width,
        )
        correction_parameters = np.clip(
            selected.parameters + trusted, parameter_lower, parameter_upper
        )
        effective_delta = correction_parameters - selected.parameters
        predicted_flux = selected.flux + jacobian @ effective_delta
        predicted_chi_square = float(
            np.sum(
                inverse_variance[fit_mask]
                * np.square(predicted_flux[fit_mask] - observed_flux[fit_mask])
            )
            / profiler.good_pixel_count
        )
        current_chi_square = 2.0 * selected.objective
        predicted_improvement = current_chi_square - predicted_chi_square
        if predicted_improvement < MIN_PREDICTED_REDUCED_CHI_SQUARE_IMPROVEMENT:
            correction_history.append(
                {
                    "index": correction_index,
                    "evaluated": False,
                    "predicted_reduced_chi_square_improvement": predicted_improvement,
                }
            )
            correction_stop_reason = "predicted_improvement_below_threshold"
            break
        if np.allclose(effective_delta[:stellar_count], 0.0, rtol=0.0, atol=1.0e-12):
            correction = cached_evaluation(
                correction_parameters,
                f"cached_chord_correction_{correction_index}",
            )
        else:
            correction = expensive_evaluation(
                correction_parameters, f"chord_correction_{correction_index}"
            )
        correction_native_pair = (
            operator.last_native_pair.detach().clone()
            if correction.synthesis_forward and operator.last_native_pair is not None
            else selected_native_pair
        )
        accepted = correction.objective < selected.objective
        actual_improvement = 2.0 * (selected.objective - correction.objective)
        history: dict[str, Any] = {
            "index": correction_index,
            "evaluated": True,
            "accepted": accepted,
            "predicted_reduced_chi_square_improvement": predicted_improvement,
            "full_step_accepted": accepted,
            "full_step_actual_reduced_chi_square_improvement": actual_improvement,
            "full_step_actual_to_predicted_improvement_ratio": (
                actual_improvement / predicted_improvement
            ),
            "full_step_evaluation": correction.event_index,
            "full_step_objective": correction.objective,
            "selected_evaluation": correction.event_index,
            "selected_objective": correction.objective,
            "half_step_evaluated": False,
        }
        if not accepted:
            operator.last_native_pair = selected_native_pair
            half_parameters = np.clip(
                selected.parameters + 0.5 * effective_delta,
                parameter_lower,
                parameter_upper,
            )
            if np.allclose(
                half_parameters[:stellar_count],
                selected.parameters[:stellar_count],
                rtol=0.0,
                atol=1.0e-12,
            ):
                half = cached_evaluation(
                    half_parameters,
                    f"cached_chord_correction_{correction_index}_half_step",
                )
            else:
                half = expensive_evaluation(
                    half_parameters,
                    f"chord_correction_{correction_index}_half_step",
                )
            half_native_pair = (
                operator.last_native_pair.detach().clone()
                if half.synthesis_forward and operator.last_native_pair is not None
                else selected_native_pair
            )
            half_actual_improvement = 2.0 * (selected.objective - half.objective)
            accepted = half.objective < selected.objective
            history.update(
                {
                    "accepted": accepted,
                    "half_step_evaluated": True,
                    "half_step_evaluation": half.event_index,
                    "half_step_actual_reduced_chi_square_improvement": (
                        half_actual_improvement
                    ),
                    "half_step_objective": half.objective,
                }
            )
            if accepted:
                correction = half
                correction_native_pair = half_native_pair
                history["selected_evaluation"] = half.event_index
                history["selected_objective"] = half.objective
            else:
                correction_history.append(history)
                operator.last_native_pair = selected_native_pair
                correction_stop_reason = "candidate_and_half_step_not_improved"
                break
        correction_history.append(history)
        previous = selected
        selected = correction
        selected_native_pair = correction_native_pair
        accepted_path.append(selected.event_index)
        jacobian, secant = _broyden_secant_update(
            jacobian,
            start_parameters=previous.parameters,
            end_parameters=selected.parameters,
            start_flux=previous.flux,
            end_flux=selected.flux,
            parameter_scale=correction_trust_half_width,
        )
        secant.update(
            {
                "from_evaluation": previous.event_index,
                "to_evaluation": selected.event_index,
            }
        )
        secant_update_history.append(secant)
        weighted_jacobian = jacobian * fit_weight[:, None]

    fresh_jacobian_history: list[dict[str, Any]] = []
    for fresh_round in range(1, fresh_jacobian_rounds + 1):
        operator.last_native_pair = selected_native_pair
        fresh_base = selected
        fresh_jacobian = np.empty_like(jacobian)
        for parameter_index in range(nuisance_start, nuisance_start + 2):
            step = local_derivative_steps[parameter_index]
            direction = (
                -1.0
                if fresh_base.parameters[parameter_index] + step
                > parameter_upper[parameter_index]
                else 1.0
            )
            perturbed = fresh_base.parameters.copy()
            perturbed[parameter_index] += direction * step
            probe = cached_evaluation(
                perturbed,
                f"fresh_{fresh_round}_jacobian_{parameter_names[parameter_index]}",
            )
            fresh_jacobian[:, parameter_index] = (probe.flux - fresh_base.flux) / (
                direction * step
            )
        for parameter_index in range(stellar_count):
            step = local_derivative_steps[parameter_index]
            direction = (
                -1.0
                if fresh_base.parameters[parameter_index] + step
                > parameter_upper[parameter_index]
                else 1.0
            )
            perturbed = fresh_base.parameters.copy()
            perturbed[parameter_index] += direction * step
            probe = expensive_evaluation(
                perturbed,
                f"fresh_{fresh_round}_jacobian_{parameter_names[parameter_index]}",
            )
            fresh_jacobian[:, parameter_index] = (probe.flux - fresh_base.flux) / (
                direction * step
            )

        weighted_fresh = fresh_jacobian * fit_weight[:, None]
        weighted_target = (observed_flux - fresh_base.flux) * fit_weight
        fresh_delta, _, fresh_rank, fresh_singular_values = np.linalg.lstsq(
            weighted_fresh, weighted_target, rcond=1.0e-8
        )
        trusted = np.clip(
            fresh_delta,
            -correction_trust_half_width,
            correction_trust_half_width,
        )
        fresh_parameters = np.clip(
            fresh_base.parameters + trusted,
            parameter_lower,
            parameter_upper,
        )
        effective_delta = fresh_parameters - fresh_base.parameters
        predicted_flux = fresh_base.flux + fresh_jacobian @ effective_delta
        predicted_reduced_chi_square = float(
            np.sum(
                inverse_variance[fit_mask]
                * np.square(predicted_flux[fit_mask] - observed_flux[fit_mask])
            )
            / profiler.good_pixel_count
        )
        predicted_improvement = (
            2.0 * fresh_base.objective - predicted_reduced_chi_square
        )
        history = {
            "round": fresh_round,
            "rank": int(fresh_rank),
            "singular_values": fresh_singular_values,
            "raw_delta": _parameter_dict(fresh_delta, parameter_names),
            "trusted_delta": _parameter_dict(trusted, parameter_names),
            "predicted_reduced_chi_square": predicted_reduced_chi_square,
            "predicted_reduced_chi_square_improvement": predicted_improvement,
            "evaluated": False,
            "accepted": False,
        }
        if predicted_improvement < MIN_PREDICTED_REDUCED_CHI_SQUARE_IMPROVEMENT:
            operator.last_native_pair = selected_native_pair
            fresh_jacobian_history.append(history)
            break
        fresh_candidate = expensive_evaluation(
            fresh_parameters, f"fresh_{fresh_round}_candidate"
        )
        if operator.last_native_pair is None:
            raise RuntimeError("fresh candidate did not populate the native flux cache")
        fresh_native_pair = operator.last_native_pair.detach().clone()
        accepted = fresh_candidate.objective < fresh_base.objective
        history.update(
            {
                "evaluated": True,
                "accepted": accepted,
                "candidate_evaluation": fresh_candidate.event_index,
                "candidate_objective": fresh_candidate.objective,
            }
        )
        if not accepted:
            operator.last_native_pair = selected_native_pair
            fresh_jacobian_history.append(history)
            break
        previous = fresh_base
        selected = fresh_candidate
        selected_native_pair = fresh_native_pair
        accepted_path.append(selected.event_index)
        fresh_jacobian, _ = _broyden_secant_update(
            fresh_jacobian,
            start_parameters=previous.parameters,
            end_parameters=selected.parameters,
            start_flux=previous.flux,
            end_flux=selected.flux,
            parameter_scale=correction_trust_half_width,
        )
        fresh_chord_history: list[dict[str, Any]] = []
        for correction_index in range(1, MAX_CHORD_CORRECTIONS + 1):
            weighted_fresh = fresh_jacobian * fit_weight[:, None]
            weighted_target = (observed_flux - selected.flux) * fit_weight
            correction_delta, _, _, _ = np.linalg.lstsq(
                weighted_fresh, weighted_target, rcond=1.0e-8
            )
            trusted_correction = np.clip(
                correction_delta,
                -correction_trust_half_width,
                correction_trust_half_width,
            )
            correction_parameters = np.clip(
                selected.parameters + trusted_correction,
                parameter_lower,
                parameter_upper,
            )
            effective_correction = correction_parameters - selected.parameters
            predicted_flux = selected.flux + fresh_jacobian @ effective_correction
            predicted_reduced_chi_square = float(
                np.sum(
                    inverse_variance[fit_mask]
                    * np.square(predicted_flux[fit_mask] - observed_flux[fit_mask])
                )
                / profiler.good_pixel_count
            )
            predicted_correction_improvement = (
                2.0 * selected.objective - predicted_reduced_chi_square
            )
            chord: dict[str, Any] = {
                "index": correction_index,
                "predicted_reduced_chi_square_improvement": (
                    predicted_correction_improvement
                ),
                "evaluated": False,
                "accepted": False,
            }
            if (
                predicted_correction_improvement
                < MIN_PREDICTED_REDUCED_CHI_SQUARE_IMPROVEMENT
            ):
                fresh_chord_history.append(chord)
                break
            if np.allclose(
                correction_parameters[:stellar_count],
                selected.parameters[:stellar_count],
                rtol=0.0,
                atol=1.0e-12,
            ):
                correction = cached_evaluation(
                    correction_parameters,
                    f"fresh_{fresh_round}_chord_{correction_index}",
                )
                correction_native_pair = selected_native_pair
            else:
                correction = expensive_evaluation(
                    correction_parameters,
                    f"fresh_{fresh_round}_chord_{correction_index}",
                )
                if operator.last_native_pair is None:
                    raise RuntimeError("fresh chord correction lost its native pair")
                correction_native_pair = operator.last_native_pair.detach().clone()
            accepted = correction.objective < selected.objective
            chord.update(
                {
                    "evaluated": True,
                    "accepted": accepted,
                    "candidate_evaluation": correction.event_index,
                    "candidate_objective": correction.objective,
                    "half_step_evaluated": False,
                }
            )
            if not accepted:
                operator.last_native_pair = selected_native_pair
                half_parameters = np.clip(
                    selected.parameters + 0.5 * effective_correction,
                    parameter_lower,
                    parameter_upper,
                )
                if np.allclose(
                    half_parameters[:stellar_count],
                    selected.parameters[:stellar_count],
                    rtol=0.0,
                    atol=1.0e-12,
                ):
                    half = cached_evaluation(
                        half_parameters,
                        f"fresh_{fresh_round}_chord_{correction_index}_half",
                    )
                    half_native_pair = selected_native_pair
                else:
                    half = expensive_evaluation(
                        half_parameters,
                        f"fresh_{fresh_round}_chord_{correction_index}_half",
                    )
                    if operator.last_native_pair is None:
                        raise RuntimeError("fresh chord half-step lost its native pair")
                    half_native_pair = operator.last_native_pair.detach().clone()
                accepted = half.objective < selected.objective
                chord.update(
                    {
                        "accepted": accepted,
                        "half_step_evaluated": True,
                        "half_step_evaluation": half.event_index,
                        "half_step_objective": half.objective,
                    }
                )
                if not accepted:
                    fresh_chord_history.append(chord)
                    operator.last_native_pair = selected_native_pair
                    break
                correction = half
                correction_native_pair = half_native_pair
            fresh_chord_history.append(chord)
            previous = selected
            selected = correction
            selected_native_pair = correction_native_pair
            accepted_path.append(selected.event_index)
            fresh_jacobian, _ = _broyden_secant_update(
                fresh_jacobian,
                start_parameters=previous.parameters,
                end_parameters=selected.parameters,
                start_flux=previous.flux,
                end_flux=selected.flux,
                parameter_scale=correction_trust_half_width,
            )
        history["chord_corrections"] = fresh_chord_history
        fresh_jacobian_history.append(history)

    # The initial nuisance columns precede the accepted stellar-label updates.
    # Rebuild these two cheap columns on the accepted native spectrum so their
    # final refinement needs no additional synthesis.
    operator.last_native_pair = selected_native_pair
    nuisance_refinement_history: list[dict[str, Any]] = []
    for refinement_index in range(1, 4):
        nuisance_jacobian = np.empty((wavelength_nm.size, 2), np.float64)
        for local_index, parameter_index in enumerate(
            range(nuisance_start, nuisance_start + 2)
        ):
            step = local_derivative_steps[parameter_index]
            direction = (
                -1.0
                if selected.parameters[parameter_index] + step
                > parameter_upper[parameter_index]
                else 1.0
            )
            perturbed = selected.parameters.copy()
            perturbed[parameter_index] += direction * step
            probe = cached_evaluation(
                perturbed,
                f"final_nuisance_jacobian_{parameter_names[parameter_index]}_{refinement_index}",
            )
            nuisance_jacobian[:, local_index] = (probe.flux - selected.flux) / (
                direction * step
            )
        weighted_nuisance = nuisance_jacobian * fit_weight[:, None]
        weighted_target = (observed_flux - selected.flux) * fit_weight
        nuisance_delta, _, nuisance_rank, _ = np.linalg.lstsq(
            weighted_nuisance, weighted_target, rcond=1.0e-8
        )
        nuisance_delta = np.clip(nuisance_delta, [-2.0, -2.0], [2.0, 2.0])
        refined_parameters = selected.parameters.copy()
        refined_parameters[nuisance_start:] = np.clip(
            refined_parameters[nuisance_start:] + nuisance_delta,
            parameter_lower[nuisance_start:],
            parameter_upper[nuisance_start:],
        )
        refined = cached_evaluation(
            refined_parameters, f"final_nuisance_candidate_{refinement_index}"
        )
        accepted = refined.objective < selected.objective
        nuisance_refinement_history.append(
            {
                "index": refinement_index,
                "rank": int(nuisance_rank),
                "delta": {
                    name: float(value)
                    for name, value in zip(NUISANCE_NAMES, nuisance_delta, strict=True)
                },
                "objective": refined.objective,
                "accepted": accepted,
            }
        )
        if not accepted:
            break
        improvement = selected.objective - refined.objective
        selected = refined
        accepted_path.append(selected.event_index)
        if improvement < 1.0e-4:
            break

    trace.mark_accepted(accepted_path)
    trace_path = trace.consolidate()
    optimizer_seconds = time.perf_counter() - optimizer_start

    # Compare with the supplied external reference after optimizer timing. The
    # reference is diagnostic only and never constrains the likelihood solve.
    diagnostic_start = time.perf_counter()
    diagnostic_labels = np.clip(
        reference_labels,
        parameter_lower[:stellar_count],
        parameter_upper[:stellar_count],
    )
    _, catalog_flux, _ = forward.evaluate(
        diagnostic_labels[:5],
        residual_rv_km_s=reference_rv,
        vmacro_km_s=reference_vmacro,
        **abundance_kwargs(diagnostic_labels),
    )
    catalog_profile = profiler.profile(
        torch.as_tensor(catalog_flux, dtype=profiler.dtype, device=profiler.device)
    )
    reference_profiled_objective = float(catalog_profile.objective.detach().cpu())
    catalog_reference_diagnostic = {
        "profiled_objective": reference_profiled_objective,
        "selected_total_chi_square_advantage": float(
            2.0
            * profiler.good_pixel_count
            * (reference_profiled_objective - selected.objective)
        ),
        "model_seconds_excluded_from_optimizer": (
            time.perf_counter() - diagnostic_start
        ),
        "support_clipped": bool(np.any(diagnostic_labels != reference_labels)),
        "evaluated_stellar_parameters": _parameter_dict(
            diagnostic_labels, stellar_names
        ),
    }

    reference_parameters = np.concatenate(
        (reference_labels, [reference_rv, reference_vmacro])
    )
    difference = selected.parameters - reference_parameters
    absolute_difference = np.abs(difference)
    observed_residual = selected.flux - observed_flux
    observed_rms = float(np.sqrt(np.mean(np.square(observed_residual[fit_mask]))))
    reduced_chi_square = 2.0 * selected.objective
    common_summary = {
        "synthesis_r_grid": synthesis_r_grid,
        # Compatibility field for readers of pre-v1.3 summaries.
        "synthesis_resolution": synthesis_r_grid,
        "continuum_order": continuum_order,
        "fit_cno8": fit_cno8,
        "initial_label_mode": initial_label_mode,
        "initial_rv_mode": initial_rv_mode,
        "requested_initial_label_offset": _parameter_dict(
            initial_label_offset, stellar_names
        ),
        "applied_initial_label_offset": _parameter_dict(
            initial_labels - reference_labels, stellar_names
        ),
        "stellar_label_names": stellar_names,
        "parameter_names": parameter_names,
        "initial_parameters": _parameter_dict(initial, parameter_names),
        "selected_parameters": _parameter_dict(selected.parameters, parameter_names),
        "selected_continuum_coefficients": selected.continuum_coefficients,
        "coarse_ccf": {
            "performed": ccf is not None,
            "used_for_initialization": ccf is not None,
            "radial_velocity_km_s": (None if ccf is None else ccf.radial_velocity_km_s),
            "peak_correlation": None if ccf is None else ccf.peak_correlation,
            "seconds": ccf_seconds,
            "initial_radial_velocity_km_s": initial_rv,
        },
        "initial_objective": base.objective,
        "selected_objective": selected.objective,
        "reduced_chi_square": reduced_chi_square,
        "good_pixel_count": profiler.good_pixel_count,
        "jacobian_rank": int(rank),
        "jacobian_singular_values": singular_values,
        "initial_raw_delta": raw_delta,
        "initial_trusted_delta": trusted_delta,
        "initial_derivative_steps": _parameter_dict(derivative_steps, parameter_names),
        "local_derivative_steps": _parameter_dict(
            local_derivative_steps, parameter_names
        ),
        "minimum_predicted_reduced_chi_square_improvement": (
            MIN_PREDICTED_REDUCED_CHI_SQUARE_IMPROVEMENT
        ),
        "maximum_chord_corrections": MAX_CHORD_CORRECTIONS,
        "correction_history": correction_history,
        "correction_stop_reason": correction_stop_reason,
        "secant_update_history": secant_update_history,
        "fresh_jacobian_history": fresh_jacobian_history,
        "nuisance_refinement_history": nuisance_refinement_history,
        "accepted_evaluations": accepted_path,
        "selected_evaluation": selected.event_index,
        "total_evaluations": len(trace.events),
        "synthesis_forward_count": synthesis_forward_count,
        "cached_nuisance_evaluation_count": int(
            sum(not event["synthesis_forward"] for event in trace.events)
        ),
        "optimizer_model_seconds_excluding_setup_and_plot": optimizer_seconds,
        "mean_synthesis_forward_seconds": float(
            np.mean(
                [
                    event["model_seconds"]
                    for event in trace.events
                    if event["synthesis_forward"]
                ]
            )
        ),
        "mean_cached_nuisance_seconds": float(
            np.mean(
                [
                    event["model_seconds"]
                    for event in trace.events
                    if not event["synthesis_forward"]
                ]
            )
        ),
        "setup_seconds_excluded": setup_seconds,
        "atomic_calibration": forward.atomic_calibration_metadata,
        "spectral_operator": operator.metadata(),
        "trace_path": _portable_path(trace_path),
        "trace_contains_spectra": not compact_trace,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "device": device,
            "dtype": dtype,
            "likelihood_dtype": str(likelihood_dtype),
            "logical_cpu_count": os.cpu_count(),
        },
    }
    standardized_residual = observed_residual[fit_mask] * np.sqrt(
        inverse_variance[fit_mask]
    )
    boundary_hit = np.isclose(
        selected.parameters,
        parameter_lower,
        rtol=0.0,
        atol=np.concatenate(([0.1], np.full(len(parameter_names) - 1, 1.0e-4))),
    ) | np.isclose(
        selected.parameters,
        parameter_upper,
        rtol=0.0,
        atol=np.concatenate(([0.1], np.full(len(parameter_names) - 1, 1.0e-4))),
    )
    fit_kind = (
        "normalized_reference_control"
        if spectrum_input.reference_is_truth
        else "apogee_normalized_spectrum_fit"
    )
    summary = {
        "fit_kind": fit_kind,
        # Compatibility field retained with a production-neutral value.
        "experiment": fit_kind,
        "data_mode": spectrum_input.data_mode,
        "object_id": spectrum_input.object_id,
        "reference_is_truth": spectrum_input.reference_is_truth,
        "catalog_parameters": _parameter_dict(reference_parameters, parameter_names),
        "selected_minus_catalog": _parameter_dict(difference, parameter_names),
        "absolute_catalog_difference": _parameter_dict(
            absolute_difference, parameter_names
        ),
        "input_metadata": spectrum_input.metadata,
        "catalog_reference_diagnostic": catalog_reference_diagnostic,
        "observed_flux_rms": observed_rms,
        "observed_flux_p95_absolute": float(
            np.percentile(np.abs(observed_residual[fit_mask]), 95.0)
        ),
        "standardized_residual_median": float(np.median(standardized_residual)),
        "standardized_residual_p95_absolute": float(
            np.percentile(np.abs(standardized_residual), 95.0)
        ),
        **common_summary,
        "validation": {
            "objective_improved": bool(selected.objective < base.objective),
            "finite_solution": bool(
                np.all(np.isfinite(selected.parameters))
                and np.isfinite(selected.objective)
            ),
            "parameter_bound_hit": {
                name: bool(hit)
                for name, hit in zip(parameter_names, boundary_hit, strict=True)
            },
        },
    }
    _write_json(summary_path, summary)
    return summary
