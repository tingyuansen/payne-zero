"""Microturbulence helpers used before the atmosphere iteration loop."""

from __future__ import annotations

import numpy as np


_STANDARD_VELOCITY = np.array(
    [
        0.50e5,
        0.50e5,
        0.50e5,
        0.51e5,
        0.52e5,
        0.55e5,
        0.63e5,
        0.80e5,
        0.90e5,
        1.00e5,
        1.10e5,
        1.20e5,
        1.30e5,
        1.40e5,
        1.46e5,
        1.52e5,
        1.56e5,
        1.60e5,
        1.64e5,
        1.68e5,
        1.71e5,
        1.74e5,
        1.76e5,
        1.78e5,
        1.80e5,
        1.81e5,
        1.82e5,
        1.83e5,
        1.83e5,
        1.83e5,
    ],
    dtype=np.float64,
)

_STANDARD_LOG_TAU = np.array(
    [
        -20.0,
        -3.0,
        -2.67313,
        -2.49296,
        -2.31296,
        -1.95636,
        -1.60768,
        -1.26699,
        -1.10007,
        -0.93587,
        -0.77416,
        -0.61500,
        -0.45564,
        -0.29176,
        -0.18673,
        -0.07193,
        0.01186,
        0.10342,
        0.20400,
        0.31605,
        0.44498,
        0.58875,
        0.74365,
        0.90604,
        1.07181,
        1.23841,
        1.39979,
        1.55300,
        2.00000,
        10.00000,
    ],
    dtype=np.float64,
)

_MAXIMUM_VELOCITY_GRID = np.array(
    [
        [
            3.3,
            4.1,
            5.2,
            6.3,
            7.3,
            8.0,
            8.0,
            8.0,
            8.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            3.0,
            3.7,
            4.6,
            5.5,
            6.4,
            7.7,
            8.0,
            8.0,
            8.0,
            8.0,
            8.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            2.7,
            3.3,
            4.0,
            4.7,
            5.5,
            6.4,
            7.1,
            7.9,
            8.0,
            8.0,
            8.0,
            8.0,
            8.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            2.4,
            2.9,
            3.4,
            3.9,
            4.6,
            5.1,
            5.7,
            6.3,
            6.9,
            7.5,
            8.0,
            8.0,
            8.0,
            4.6,
            0.2,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            2.1,
            2.5,
            2.9,
            3.3,
            3.7,
            4.2,
            4.7,
            5.2,
            5.6,
            6.1,
            6.6,
            7.1,
            7.6,
            8.0,
            4.3,
            0.3,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            1.8,
            2.1,
            2.4,
            2.7,
            3.1,
            3.5,
            3.9,
            4.3,
            4.7,
            5.1,
            5.5,
            5.9,
            6.2,
            6.6,
            7.0,
            4.2,
            0.3,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            1.3,
            1.6,
            1.9,
            2.2,
            2.6,
            2.9,
            3.2,
            3.6,
            4.0,
            4.4,
            4.7,
            5.0,
            5.4,
            5.7,
            6.1,
            6.4,
            3.9,
            0.3,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.9,
            1.2,
            1.5,
            1.8,
            2.1,
            2.4,
            2.7,
            3.0,
            3.4,
            3.7,
            4.0,
            4.3,
            4.6,
            4.9,
            5.3,
            5.6,
            5.9,
            3.7,
            0.4,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.6,
            0.9,
            1.2,
            1.5,
            1.8,
            2.0,
            2.3,
            2.5,
            2.8,
            3.1,
            3.4,
            3.6,
            3.9,
            4.2,
            4.4,
            4.7,
            5.0,
            5.2,
            3.5,
            0.5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.3,
            0.6,
            0.9,
            1.2,
            1.4,
            1.6,
            1.9,
            2.1,
            2.3,
            2.6,
            2.8,
            3.0,
            3.3,
            3.5,
            3.7,
            4.0,
            4.2,
            4.4,
            4.7,
            3.4,
            0.7,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.2,
            0.3,
            0.6,
            0.9,
            1.1,
            1.3,
            1.5,
            1.7,
            1.9,
            2.1,
            2.3,
            2.5,
            2.7,
            2.9,
            3.1,
            3.3,
            3.5,
            3.7,
            3.9,
            4.1,
            3.6,
            1.1,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.1,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
            1.2,
            1.4,
            1.5,
            1.7,
            1.9,
            2.0,
            2.2,
            2.4,
            2.6,
            2.8,
            3.0,
            3.2,
            3.4,
            3.6,
            3.8,
            4.0,
            3.5,
            1.6,
            0.0,
        ],
        [
            0.1,
            0.1,
            0.2,
            0.4,
            0.6,
            0.7,
            0.9,
            1.1,
            1.2,
            1.3,
            1.5,
            1.7,
            1.9,
            2.0,
            2.2,
            2.4,
            2.6,
            2.8,
            3.0,
            3.1,
            3.3,
            3.5,
            3.6,
            3.6,
            2.3,
        ],
    ],
    dtype=np.float64,
)


def _piecewise_quadratic_remap(
    xold: np.ndarray, fold: np.ndarray, xnew: np.ndarray
) -> np.ndarray:
    """Piecewise-quadratic remap on a monotonic coordinate."""

    nold = xold.size
    fnew = np.zeros(xnew.size, dtype=np.float64)
    if nold == 0 or xnew.size == 0:
        return fnew

    xold1 = np.empty(nold + 1, dtype=np.float64)
    fold1 = np.empty(nold + 1, dtype=np.float64)
    xold1[1:] = xold
    fold1[1:] = fold

    old_index = 2
    last_old_index = 0
    cfor = bfor = afor = 0.0
    cbac = bbac = abac = 0.0

    for k, xk in enumerate(xnew):
        while True:
            if xk < xold1[old_index]:
                if old_index == last_old_index:
                    break
                if old_index in (2, 3):
                    old_index = min(nold, old_index)
                    c = 0.0
                    dx = xold1[old_index] - xold1[old_index - 1]
                    b = (
                        (fold1[old_index] - fold1[old_index - 1]) / dx
                        if dx != 0.0
                        else 0.0
                    )
                    a = fold1[old_index] - xold1[old_index] * b
                    last_old_index = old_index
                    break
                previous_old_index = old_index - 1
                if old_index > last_old_index + 1 or old_index in (3, 4):
                    dx_d = xold1[previous_old_index] - xold1[old_index - 2]
                    dx_l1 = xold1[old_index] - xold1[previous_old_index]
                    dx_l2 = xold1[old_index] - xold1[old_index - 2]
                    d = (
                        (fold1[previous_old_index] - fold1[old_index - 2]) / dx_d
                        if dx_d != 0.0
                        else 0.0
                    )
                    term1 = (
                        fold1[old_index] / (dx_l1 * dx_l2)
                        if dx_l1 != 0.0 and dx_l2 != 0.0
                        else 0.0
                    )
                    part_a = fold1[old_index - 2] / dx_l2 if dx_l2 != 0.0 else 0.0
                    part_b = fold1[previous_old_index] / dx_l1 if dx_l1 != 0.0 else 0.0
                    part = (part_a - part_b) / dx_d if dx_d != 0.0 else 0.0
                    cbac = term1 + part
                    bbac = d - (xold1[previous_old_index] + xold1[old_index - 2]) * cbac
                    abac = (
                        fold1[old_index - 2]
                        - xold1[old_index - 2] * d
                        + xold1[previous_old_index] * xold1[old_index - 2] * cbac
                    )
                else:
                    cbac = cfor
                    bbac = bfor
                    abac = afor
                if old_index >= nold:
                    c = cbac
                    b = bbac
                    a = abac
                    last_old_index = old_index
                    break
                dx_d = xold1[old_index] - xold1[previous_old_index]
                dx_p1 = xold1[old_index + 1] - xold1[old_index]
                dx_p2 = xold1[old_index + 1] - xold1[previous_old_index]
                d = (
                    (fold1[old_index] - fold1[previous_old_index]) / dx_d
                    if dx_d != 0.0
                    else 0.0
                )
                term1 = (
                    fold1[old_index + 1] / (dx_p1 * dx_p2)
                    if dx_p1 != 0.0 and dx_p2 != 0.0
                    else 0.0
                )
                part_a = fold1[previous_old_index] / dx_p2 if dx_p2 != 0.0 else 0.0
                part_b = fold1[old_index] / dx_p1 if dx_p1 != 0.0 else 0.0
                part = (part_a - part_b) / dx_d if dx_d != 0.0 else 0.0
                cfor = term1 + part
                bfor = d - (xold1[old_index] + xold1[previous_old_index]) * cfor
                afor = (
                    fold1[previous_old_index]
                    - xold1[previous_old_index] * d
                    + xold1[old_index] * xold1[previous_old_index] * cfor
                )
                weight = (
                    abs(cfor) / (abs(cfor) + abs(cbac)) if abs(cfor) != 0.0 else 0.0
                )
                a = afor + weight * (abac - afor)
                b = bfor + weight * (bbac - bfor)
                c = cfor + weight * (cbac - cfor)
                last_old_index = old_index
                break

            old_index += 1
            if old_index > nold:
                old_index = min(nold, old_index)
                c = 0.0
                dx = xold1[old_index] - xold1[old_index - 1]
                b = (fold1[old_index] - fold1[old_index - 1]) / dx if dx != 0.0 else 0.0
                a = fold1[old_index] - xold1[old_index] * b
                last_old_index = old_index
                break
        fnew[k] = a + (b + c * xk) * xk
    return fnew


def standard_microturbulence(
    *,
    effective_temperature: float,
    log_surface_gravity: float,
    standard_rosseland_optical_depth: np.ndarray,
    requested_maximum_velocity: float = -99.0e5,
) -> np.ndarray:
    """Return the standard microturbulent-velocity profile in cm s^-1."""

    if requested_maximum_velocity == -99.0e5:
        gravity_index = int((log_surface_gravity + 1.0) / 0.5) + 1
        gravity_index = max(1, min(gravity_index, 12))
        temperature_index = int((effective_temperature - 3000.0) / 250.0) + 1
        temperature_index = max(1, min(temperature_index, 24))

        gravity_fraction = (
            log_surface_gravity - ((gravity_index - 1) * 0.5 - 1.0)
        ) / 0.5
        temperature_fraction = (
            effective_temperature - ((temperature_index - 1) * 250.0 + 3000.0)
        ) / 250.0

        g0 = gravity_index - 1
        g1 = min(gravity_index, 12)
        t0 = temperature_index - 1
        t1 = min(temperature_index, 24)
        maximum_velocity = (
            _MAXIMUM_VELOCITY_GRID[g0, t0]
            * (1.0 - gravity_fraction)
            * (1.0 - temperature_fraction)
            + _MAXIMUM_VELOCITY_GRID[g1, t0]
            * gravity_fraction
            * (1.0 - temperature_fraction)
            + _MAXIMUM_VELOCITY_GRID[g0, t1]
            * (1.0 - gravity_fraction)
            * temperature_fraction
            + _MAXIMUM_VELOCITY_GRID[g1, t1] * gravity_fraction * temperature_fraction
        )
        maximum_velocity *= 1.0e5
    else:
        maximum_velocity = abs(requested_maximum_velocity)

    log_tau = np.log10(np.maximum(standard_rosseland_optical_depth, 1.0e-300))
    base_profile = _piecewise_quadratic_remap(
        _STANDARD_LOG_TAU, _STANDARD_VELOCITY, log_tau
    )
    return base_profile * maximum_velocity / 1.83e5
