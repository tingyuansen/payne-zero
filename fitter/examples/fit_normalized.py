"""Fit two absorption-line strengths while profiling a linear continuum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fitter import FitConfiguration, NormalizedSpectrum, fit_normalized_spectrum


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        help="optional directory for fit_summary.json and fit_trace.npz",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    wavelength = np.linspace(500.0, 510.0, 600)
    line_profiles = np.column_stack(
        (
            np.exp(-0.5 * ((wavelength - 503.0) / 0.08) ** 2),
            np.exp(-0.5 * ((wavelength - 507.0) / 0.12) ** 2),
        )
    )

    def model(parameters: np.ndarray) -> np.ndarray:
        return 1.0 - line_profiles @ parameters

    truth = np.asarray([0.25, 0.12])
    coordinate = (wavelength - wavelength.mean()) / np.ptp(wavelength)
    continuum_basis = np.column_stack((np.ones(wavelength.size), coordinate))
    continuum = np.asarray([0.012, -0.006])
    observed_flux = model(truth) * (1.0 + continuum_basis @ continuum)
    spectrum = NormalizedSpectrum(
        wavelength=wavelength,
        flux=observed_flux,
        inverse_variance=np.full(wavelength.size, 1.0e6),
        mask=np.ones(wavelength.size, dtype=bool),
    )
    configuration = FitConfiguration(
        names=("line_a", "line_b"),
        initial=np.asarray([0.10, 0.25]),
        lower=np.zeros(2),
        upper=np.full(2, 0.5),
        derivative_steps=np.full(2, 0.01),
        trust_half_width=np.full(2, 0.2),
        maximum_iterations=5,
    )
    result = fit_normalized_spectrum(
        spectrum,
        configuration,
        model,
        continuum_basis=continuum_basis,
    )
    if args.out is not None:
        result.save(args.out)
    print(
        json.dumps(
            {
                "truth": dict(zip(configuration.names, truth, strict=True)),
                "fitted": dict(
                    zip(configuration.names, result.parameters, strict=True)
                ),
                "continuum_coefficients": result.continuum_coefficients.tolist(),
                "mean_weighted_squared_residual": (
                    result.mean_weighted_squared_residual
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
