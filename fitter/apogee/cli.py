"""Fit one prepared normalized APOGEE spectrum from an NPZ file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .api import fit_apogee_spectrum


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "spectrum",
        type=Path,
        help=(
            "NPZ with (7514,) wavelength_nm, normalized_flux, inverse_variance, "
            "and optional good_pixel_mask arrays on the packaged DR14 grid"
        ),
    )
    parser.add_argument("result_dir", type=Path, help="output directory")
    parser.add_argument("--object-id", required=True, help="source identifier")
    parser.add_argument(
        "--reference-labels",
        type=float,
        nargs=5,
        required=True,
        metavar=("TEFF", "LOGG", "M_H", "ALPHA_M", "VMICRO"),
        help="initial labels in K, dex, dex, dex, and km/s",
    )
    parser.add_argument(
        "--reference-vmacro",
        type=float,
        required=True,
        help="initial macroscopic broadening in km/s",
    )
    parser.add_argument("--c-over-m", "--carbon-enhancement", dest="c_over_m", type=float)
    parser.add_argument("--n-over-m", "--nitrogen-enhancement", dest="n_over_m", type=float)
    parser.add_argument("--o-over-m", "--oxygen-enhancement", dest="o_over_m", type=float)
    parser.add_argument(
        "--fit-cno8",
        action="store_true",
        help="fit [C/M], [N/M], and [O/M] after the five reference labels",
    )
    parser.add_argument("--atomic-calibration", type=Path)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float64"), default="auto"
    )
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument(
        "--synthesis-r-grid",
        "--synthesis-resolution",
        dest="synthesis_r_grid",
        type=float,
        default=300_000.0,
        help="intrinsic logarithmic synthesis-grid density before the APOGEE LSF",
    )
    parser.add_argument("--fresh-jacobian-rounds", type=int, default=1)
    parser.add_argument("--continuum-order", type=int, default=2)
    parser.add_argument(
        "--initial-label-mode",
        choices=("reference", "controlled-offset"),
        default="reference",
        help=(
            "stellar-label start: supplied reference labels (default) or the "
            "fixed offset used by the controlled recovery experiment"
        ),
    )
    parser.add_argument(
        "--initial-rv-mode",
        choices=("rest-frame", "coarse-ccf"),
        default="rest-frame",
        help=(
            "residual-RV start: zero for an already rest-framed spectrum "
            "(default), or a coarse cross-correlation estimate"
        ),
    )
    parser.add_argument("--store-spectra", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result_dir = args.result_dir.expanduser().resolve()
    with np.load(args.spectrum.expanduser().resolve(), allow_pickle=False) as source:
        required = {"wavelength_nm", "normalized_flux", "inverse_variance"}
        missing = required.difference(source.files)
        if missing:
            raise ValueError(f"spectrum NPZ lacks arrays: {sorted(missing)}")
        mask = source["good_pixel_mask"] if "good_pixel_mask" in source.files else None
        summary = fit_apogee_spectrum(
            result_dir,
            object_id=args.object_id,
            wavelength_nm=source["wavelength_nm"],
            normalized_flux=source["normalized_flux"],
            inverse_variance=source["inverse_variance"],
            good_pixel_mask=mask,
            reference_labels=np.asarray(args.reference_labels, np.float64),
            reference_vmacro_km_s=args.reference_vmacro,
            c_over_m=args.c_over_m,
            n_over_m=args.n_over_m,
            o_over_m=args.o_over_m,
            atomic_calibration_path=args.atomic_calibration,
            device=args.device,
            dtype=args.dtype,
            torch_threads=args.torch_threads,
            synthesis_r_grid=args.synthesis_r_grid,
            fresh_jacobian_rounds=args.fresh_jacobian_rounds,
            continuum_order=args.continuum_order,
            fit_cno8=args.fit_cno8,
            initial_label_mode=args.initial_label_mode.replace("-", "_"),
            initial_rv_mode=args.initial_rv_mode.replace("-", "_"),
            compact_trace=not args.store_spectra,
            force=args.force,
        )
    report = {
        "object_id": summary["object_id"],
        "summary_path": str(result_dir / "summary.json"),
        "selected_parameters": summary.get("selected_parameters"),
        "reduced_chi_square": summary.get("reduced_chi_square"),
        "optimizer_model_seconds": summary.get(
            "optimizer_model_seconds_excluding_setup_and_plot"
        ),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
