"""Generic bounded optimization for differentiable line-list parameters."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch


TensorModel = Callable[[torch.Tensor], torch.Tensor]


def _synchronize(device: torch.device) -> None:
    """Finish queued accelerator work before recording a wall-clock boundary."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@dataclass(frozen=True)
class CalibrationData:
    """Observed normalized flux and nonnegative weights of matching shape.

    A weight may be inverse variance, a quality weight, or zero for a masked
    sample. Non-finite flux and zero-weight samples are excluded. The model
    callback must return flux with this exact shape.
    """

    flux: np.ndarray
    weight: np.ndarray

    def tensors(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flux = np.asarray(self.flux, np.float64)
        weight = np.asarray(self.weight, np.float64)
        if flux.shape != weight.shape or flux.size == 0:
            raise ValueError("flux and weight must have the same non-empty shape")
        if np.any(weight < 0.0) or not np.all(np.isfinite(weight)):
            raise ValueError("weights must be finite and nonnegative")
        good = np.isfinite(flux) & (weight > 0.0)
        if not np.any(good):
            raise ValueError("at least one finite positive-weight sample is required")
        clean_flux = np.where(good, flux, 0.0)
        clean_weight = np.where(good, weight, 0.0)
        return (
            torch.as_tensor(clean_flux, device=device, dtype=dtype),
            torch.as_tensor(clean_weight, device=device, dtype=dtype),
        )


@dataclass(frozen=True)
class CalibrationConfiguration:
    """Matching one-dimensional parameter vectors and optimizer controls.

    ``initial``, ``lower``, and ``upper`` have shape ``(P,)`` and use the
    physical order expected by the callback. ``names``, when supplied, has
    length ``P``. The physical atomic model interprets these values as dex.
    ``device`` is a concrete PyTorch device such as ``cpu``, ``mps``, or
    ``cuda``; ``dtype`` is ``float32`` or ``float64``.
    """

    initial: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    names: tuple[str, ...] | None = None
    maximum_iterations: int = 100
    tolerance_gradient: float = 1.0e-7
    tolerance_change: float = 1.0e-9
    device: str = "cpu"
    dtype: str = "float64"

    def validated(self) -> "CalibrationConfiguration":
        initial = np.asarray(self.initial, np.float64)
        lower = np.asarray(self.lower, np.float64)
        upper = np.asarray(self.upper, np.float64)
        if (
            initial.ndim != 1
            or lower.shape != initial.shape
            or upper.shape != initial.shape
        ):
            raise ValueError("initial and bound arrays must be matching vectors")
        if initial.size == 0:
            raise ValueError("at least one parameter is required")
        if not all(np.all(np.isfinite(value)) for value in (initial, lower, upper)):
            raise ValueError("parameter arrays must be finite")
        if np.any(lower >= upper) or np.any((initial < lower) | (initial > upper)):
            raise ValueError("parameter bounds or initial values are invalid")
        if self.names is not None:
            if len(self.names) != initial.size:
                raise ValueError("names must match the parameter vector")
            if len(set(self.names)) != len(self.names) or not all(self.names):
                raise ValueError("parameter names must be unique and non-empty")
        if self.maximum_iterations < 1:
            raise ValueError("maximum_iterations must be positive")
        if (
            not np.isfinite(self.tolerance_gradient)
            or not np.isfinite(self.tolerance_change)
            or self.tolerance_gradient < 0.0
            or self.tolerance_change < 0.0
        ):
            raise ValueError("optimizer tolerances must be finite and nonnegative")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        return CalibrationConfiguration(
            initial=initial,
            lower=lower,
            upper=upper,
            names=self.names,
            maximum_iterations=self.maximum_iterations,
            tolerance_gradient=self.tolerance_gradient,
            tolerance_change=self.tolerance_change,
            device=self.device,
            dtype=self.dtype,
        )


@dataclass(frozen=True)
class CalibrationResult:
    """Optimized physical values and every LBFGS objective evaluation."""

    values: np.ndarray
    initial_loss: float
    final_loss: float
    loss_history: np.ndarray
    evaluation_seconds: np.ndarray
    names: tuple[str, ...] | None

    def save(self, output_dir: str | Path) -> None:
        directory = Path(output_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "calibrated_parameters.npz",
            values=self.values,
            parameter_names=(
                np.asarray(self.names) if self.names is not None else np.asarray([])
            ),
            loss_history=self.loss_history,
            evaluation_seconds=self.evaluation_seconds,
        )
        summary = {
            "parameter_count": int(self.values.size),
            "parameters": (
                {
                    name: float(value)
                    for name, value in zip(self.names, self.values, strict=True)
                }
                if self.names is not None
                else None
            ),
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "objective_reduction_fraction": (
                1.0 - self.final_loss / self.initial_loss
                if self.initial_loss > 0.0
                else 0.0
            ),
            "objective_evaluation_count": int(self.loss_history.size),
            "model_seconds_excluding_io_and_plotting": float(
                np.sum(self.evaluation_seconds)
            ),
        }
        (directory / "calibration_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )


def calibrate_line_parameters(
    data: CalibrationData,
    configuration: CalibrationConfiguration,
    model: TensorModel,
) -> CalibrationResult:
    """Optimize any differentiable physical line-parameter vector.

    ``model`` receives the bounded tensor with shape ``(P,)`` on the configured
    device and dtype. It must return differentiable normalized flux with the
    exact shape of ``data.flux``. Instrument response, normalization, and
    multi-spectrum stacking remain explicit parts of that callback.
    """

    fit = configuration.validated()
    device = torch.device(fit.device)
    dtype = torch.float32 if fit.dtype == "float32" else torch.float64
    observed, weight = data.tensors(device=device, dtype=dtype)
    lower = torch.as_tensor(fit.lower, device=device, dtype=dtype)
    upper = torch.as_tensor(fit.upper, device=device, dtype=dtype)
    fraction = np.clip(
        (fit.initial - fit.lower) / (fit.upper - fit.lower), 1.0e-7, 1.0 - 1.0e-7
    )
    raw = torch.nn.Parameter(
        torch.as_tensor(np.log(fraction / (1.0 - fraction)), device=device, dtype=dtype)
    )
    optimizer = torch.optim.LBFGS(
        [raw],
        lr=1.0,
        max_iter=fit.maximum_iterations,
        tolerance_grad=fit.tolerance_gradient,
        tolerance_change=fit.tolerance_change,
        line_search_fn="strong_wolfe",
    )
    losses: list[float] = []
    seconds: list[float] = []

    def physical_values() -> torch.Tensor:
        return lower + (upper - lower) * torch.sigmoid(raw)

    def closure() -> torch.Tensor:
        _synchronize(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(physical_values())
        if prediction.shape != observed.shape:
            raise ValueError("model returned flux with the wrong shape")
        loss = torch.sum(weight * (prediction - observed).square()) / torch.sum(weight)
        if not torch.isfinite(loss):
            raise RuntimeError("line-calibration objective became non-finite")
        loss.backward()
        if raw.grad is None or not torch.all(torch.isfinite(raw.grad)):
            raise RuntimeError("line-calibration gradient became non-finite")
        _synchronize(device)
        losses.append(float(loss.detach().cpu()))
        seconds.append(time.perf_counter() - started)
        return loss

    optimizer.step(closure)
    _synchronize(device)
    final_started = time.perf_counter()
    with torch.no_grad():
        final_prediction = model(physical_values())
        if final_prediction.shape != observed.shape:
            raise ValueError("model returned flux with the wrong shape")
        final_loss_tensor = torch.sum(
            weight * (final_prediction - observed).square()
        ) / torch.sum(weight)
    if not torch.isfinite(final_loss_tensor):
        raise RuntimeError("line-calibration objective became non-finite")
    _synchronize(device)
    final_loss = float(final_loss_tensor.detach().cpu())
    # This independent no-gradient callback certifies the returned bounded
    # parameters.  Retain it even when its loss exactly repeats the final LBFGS
    # closure so evaluation counts and non-I/O model time remain complete.
    losses.append(final_loss)
    seconds.append(time.perf_counter() - final_started)
    values = physical_values().detach().cpu().numpy().astype(np.float64)
    return CalibrationResult(
        values=values,
        initial_loss=losses[0],
        final_loss=final_loss,
        loss_history=np.asarray(losses, np.float64),
        evaluation_seconds=np.asarray(seconds, np.float64),
        names=fit.names,
    )
