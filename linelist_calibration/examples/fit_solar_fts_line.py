#!/usr/bin/env python3
"""Calibrate one physical Fe I line against a real solar FTS excerpt."""

from __future__ import annotations

import argparse
from importlib.resources import files
import json
from pathlib import Path

import numpy as np
import torch

from linelist_calibration import (
    AtomicTransition,
    CalibrationConfiguration,
    CalibrationData,
    SynthesisLineCalibrationModel,
    calibrate_line_parameters,
)


TRANSITION_WAVELENGTH_NM = 1568.1802428734834
REGISTERED_RADIAL_VELOCITY_KM_S = 0.5236894203555993
GAUSSIAN_BROADENING_SIGMA_KM_S = 1.55


def _example_data_path(filename: str) -> Path:
    return Path(str(files("linelist_calibration.examples").joinpath("data", filename)))


def load_solar_fts_excerpt() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return wavelength, normalized flux, quality weight, and provenance."""

    path = _example_data_path("solar_fts_fe_i_1568_excerpt.npz")
    with np.load(path, allow_pickle=False) as data:
        wavelength_nm = np.asarray(data["wavelength_nm"], np.float64)
        normalized_flux = np.asarray(data["normalized_flux"], np.float64)
        quality_weight = np.asarray(data["quality_weight"], np.float64)
        metadata = json.loads(bytes(data["metadata_json"]).decode("utf-8"))
    return wavelength_nm, normalized_flux, quality_weight, metadata


def build_example(
    *,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    maximum_iterations: int = 30,
) -> tuple[
    CalibrationData,
    CalibrationConfiguration,
    SynthesisLineCalibrationModel,
    dict,
]:
    """Build the real one-line FTS calibration used by the public tutorial."""

    wavelength_nm, observed_flux, weight, metadata = load_solar_fts_excerpt()
    model = SynthesisLineCalibrationModel(
        _example_data_path("sun_fts_structured_atmosphere.npz"),
        wavelength_start_nm=TRANSITION_WAVELENGTH_NM - 0.22,
        wavelength_end_nm=TRANSITION_WAVELENGTH_NM + 0.22,
        resolution=300_000,
        transitions=(
            AtomicTransition(
                atomic_number=26,
                ion_stage=1,
                wavelength_nm=TRANSITION_WAVELENGTH_NM,
                name="Fe I 1568.180 nm",
            ),
        ),
        observed_wavelength_nm=wavelength_nm,
        radial_velocity_km_s=REGISTERED_RADIAL_VELOCITY_KM_S,
        gaussian_broadening_sigma_km_s=GAUSSIAN_BROADENING_SIGMA_KM_S,
        molecular_lines=True,
        device=device,
        dtype=dtype,
    )
    configuration = CalibrationConfiguration(
        initial=np.asarray([0.0]),
        lower=np.asarray([-1.0]),
        upper=np.asarray([1.5]),
        names=("Fe I 1568.180 nm delta log(gf)",),
        maximum_iterations=int(maximum_iterations),
        device=str(model.device),
        dtype="float32" if model.dtype == torch.float32 else "float64",
    )
    return (
        CalibrationData(flux=observed_flux, weight=weight),
        configuration,
        model,
        metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float64"),
        default="auto",
    )
    parser.add_argument("--maximum-iterations", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/solar-fts-line-calibration"),
    )
    args = parser.parse_args()

    data, configuration, physical_model, metadata = build_example(
        device=args.device,
        dtype=args.dtype,
        maximum_iterations=args.maximum_iterations,
    )
    callback = physical_model.callback(("loggf",))
    baseline = physical_model.baseline_flux(("loggf",))
    result = calibrate_line_parameters(data, configuration, callback)
    fitted_tensor = torch.as_tensor(
        result.values,
        dtype=physical_model.dtype,
        device=physical_model.device,
    )
    with torch.no_grad():
        calibrated = (
            callback(fitted_tensor).detach().cpu().numpy().astype(np.float64)
        )

    output_dir = args.output_dir.expanduser().resolve()
    result.save(output_dir)
    overlay = physical_model.write_atomic_calibration_overlay(
        result.values,
        output_dir / "solar_fts_fe_i_1568_calibration.npz",
        parameter_families=("loggf",),
        calibration_name="solar_fts_fe_i_1568_example",
    )
    np.savez_compressed(
        output_dir / "solar_fts_line_comparison.npz",
        wavelength_nm=physical_model.output_wavelength_nm,
        observed_flux=np.asarray(data.flux, np.float64),
        quality_weight=np.asarray(data.weight, np.float64),
        baseline_flux=baseline,
        calibrated_flux=calibrated,
        fitted_delta_loggf_dex=result.values,
        atlas_metadata_json=np.frombuffer(
            json.dumps(metadata, sort_keys=True).encode("utf-8"),
            dtype=np.uint8,
        ),
    )
    fractional_reduction = (
        1.0 - result.final_loss / result.initial_loss
        if result.initial_loss > 0.0
        else 0.0
    )
    print(f"device: {physical_model.device}")
    print(
        "transition: "
        f"{physical_model.transitions[0].name} "
        f"({physical_model.transitions[0].wavelength_nm:.9f} nm)"
    )
    print(f"fitted delta log(gf): {result.values[0]:+.6f} dex")
    print(f"objective: {result.initial_loss:.6e} -> {result.final_loss:.6e}")
    print(f"weighted residual-power reduction: {100.0 * fractional_reduction:.1f}%")
    print(f"schema-4 overlay: {overlay['overlay_path']}")
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
