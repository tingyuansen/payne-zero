"""Atmosphere convergence diagnostics used by the production runner."""

from __future__ import annotations

import numpy as np


def max_normalized_column_delta(
    before: np.ndarray,
    after: np.ndarray,
    *,
    floor: float = 1.0e-300,
    symmetric: bool = False,
) -> float:
    """Return the largest layer-wise normalized column change."""

    before_array = np.asarray(before, dtype=np.float64)
    after_array = np.asarray(after, dtype=np.float64)
    if before_array.shape != after_array.shape or before_array.size == 0:
        return float("nan")

    if symmetric:
        denominator = np.maximum.reduce(
            [
                np.abs(before_array),
                np.abs(after_array),
                np.full(before_array.shape, floor, dtype=np.float64),
            ]
        )
    else:
        denominator = np.maximum(np.abs(before_array), floor)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.abs(after_array - before_array) / denominator
    finite = ratios[np.isfinite(ratios)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def deep_layer_relative_temperature_change(
    before: np.ndarray, after: np.ndarray
) -> float:
    """Return the maximum relative temperature change in the deep layers.

    The production threshold is evaluated over layers 39 through ``layers - 6``
    on the standard 80-layer grid. Smaller grids use every layer.
    """

    before_array = np.asarray(before, dtype=np.float64)
    after_array = np.asarray(after, dtype=np.float64)
    if (
        before_array.shape != after_array.shape
        or before_array.ndim != 1
        or before_array.size == 0
    ):
        return float("nan")

    layers = before_array.size
    start = 39
    stop = layers - 5
    if stop - start < 1:
        start, stop = 0, layers

    old_temperature = before_array[start:stop]
    new_temperature = after_array[start:stop]
    if not np.all(np.isfinite(old_temperature)) or not np.all(
        np.isfinite(new_temperature)
    ):
        return float("inf")
    with np.errstate(divide="ignore", invalid="ignore"):
        fractional_temperature_change = np.abs(
            new_temperature - old_temperature
        ) / np.abs(new_temperature)
    if not np.all(np.isfinite(fractional_temperature_change)):
        return float("inf")
    return float(np.max(fractional_temperature_change))


def temperature_changes_within_limits(
    *,
    deep_layer_change: float,
    all_layer_change: float,
    maximum_deep_layer_change: float,
    maximum_all_layer_change: float | None,
) -> bool:
    """Evaluate the declared structural fixed-point stopping limits.

    The optional all-layer test catches slowly relaxing upper layers seen by
    strong-line cores while retaining the historical deep-only behavior when
    no all-layer limit is requested.
    """

    deep_ok = np.isfinite(deep_layer_change) and deep_layer_change < float(
        maximum_deep_layer_change
    )
    if maximum_all_layer_change is None:
        return bool(deep_ok)
    all_ok = np.isfinite(all_layer_change) and all_layer_change < float(
        maximum_all_layer_change
    )
    return bool(deep_ok and all_ok)
