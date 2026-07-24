#!/usr/bin/env python3
"""Fit two line-strength corrections to synthetic standard-star spectra.

This intentionally small example exercises only the public calibration API.
The analytic line profiles are stand-ins for a differentiable synthesis
callback; no FTS atlas, atmosphere, or retained fit result is required.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from linelist_calibration import (
    CalibrationConfiguration,
    CalibrationData,
    calibrate_line_parameters,
)


@dataclass(frozen=True)
class StandardStar:
    """Labels and fixed analytic response depths for one example spectrum."""

    name: str
    effective_temperature: float
    log_surface_gravity: float
    line_depths: tuple[float, float]


STANDARD_STARS = (
    StandardStar("Sun", 5777.0, 4.44, (0.10, 0.07)),
    StandardStar("Arcturus", 4286.0, 1.66, (0.18, 0.11)),
)
TRUE_CORRECTIONS_DEX = np.asarray([0.08, -0.12])


def build_example() -> tuple[
    CalibrationData,
    CalibrationConfiguration,
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Return a deterministic two-star calibration problem and its callback."""

    wavelength = torch.linspace(1568.5, 1590.0, 800, dtype=torch.float64)
    centers = (1569.073, 1589.275)
    widths = (0.055, 0.075)
    templates = []
    for star in STANDARD_STARS:
        templates.append(
            torch.stack(
                tuple(
                    depth * torch.exp(-0.5 * ((wavelength - center) / width).square())
                    for depth, center, width in zip(
                        star.line_depths, centers, widths, strict=True
                    )
                ),
                dim=1,
            )
        )

    truth = torch.as_tensor(TRUE_CORRECTIONS_DEX, dtype=torch.float64)
    observed = torch.cat(
        tuple(1.0 - template @ torch.pow(10.0, truth) for template in templates)
    )
    data = CalibrationData(observed.numpy(), np.ones(observed.numel()))
    configuration = CalibrationConfiguration(
        initial=np.zeros(2),
        lower=np.full(2, -0.5),
        upper=np.full(2, 0.5),
        names=("Fe_I_1569_loggf_dex", "Si_I_1589_loggf_dex"),
        maximum_iterations=40,
        tolerance_gradient=1.0e-12,
        tolerance_change=1.0e-14,
    )

    def model(corrections_dex: torch.Tensor) -> torch.Tensor:
        strengths = torch.pow(10.0, corrections_dex)
        return torch.cat(
            tuple(
                1.0
                - template.to(
                    device=corrections_dex.device,
                    dtype=corrections_dex.dtype,
                )
                @ strengths
                for template in templates
            )
        )

    return data, configuration, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="optional directory for the NPZ result and JSON summary",
    )
    args = parser.parse_args()

    data, configuration, model = build_example()
    result = calibrate_line_parameters(data, configuration, model)
    if args.output_dir is not None:
        result.save(args.output_dir)

    labels = ", ".join(
        f"{star.name} (Teff={star.effective_temperature:.0f} K, "
        f"logg={star.log_surface_gravity:.2f})"
        for star in STANDARD_STARS
    )
    print(labels)
    for name, value in zip(result.names or (), result.values, strict=True):
        print(f"{name}: {value:+.5f}")
    print(f"objective: {result.initial_loss:.3e} -> {result.final_loss:.3e}")


if __name__ == "__main__":
    main()
