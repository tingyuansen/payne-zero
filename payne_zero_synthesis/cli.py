"""Command-line synthesis: `python -m payne_zero_synthesis.cli atmosphere.npz
--out spectrum.npz --device cuda --wl-start-nm 400 --wl-end-nm 900
--r-grid 20000` (see README for the full flag list).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .api import synthesize
from .atmosphere import validate_atmosphere_npz


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m payne_zero_synthesis.cli", description=__doc__
    )
    parser.add_argument("atmosphere", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--wl-start-nm", type=float, default=400.0)
    parser.add_argument("--wl-end-nm", type=float, default=900.0)
    parser.add_argument(
        "--r-grid",
        "--resolution",
        dest="resolution",
        type=float,
        default=20000.0,
        help="logarithmic wavelength-grid density (not instrumental resolution)",
    )
    parser.add_argument("--device", default=None, help="cuda, mps, or cpu")
    parser.add_argument(
        "--dtype",
        choices=("float64", "float32"),
        default=None,
        help="default: float32 on Metal, float64 on CUDA or CPU",
    )
    parser.add_argument("--no-molecular-lines", action="store_true")
    args = parser.parse_args(argv)

    if args.validate_only:
        array_names = validate_atmosphere_npz(args.atmosphere)
        print(
            f"{args.atmosphere} is a valid Payne Zero synthesis atmosphere "
            f"({len(array_names)} arrays)"
        )
        return 0
    if args.out is None:
        parser.error("--out is required unless --validate-only is used")

    spectrum = synthesize(
        args.atmosphere,
        wavelength_start_nm=args.wl_start_nm,
        wavelength_end_nm=args.wl_end_nm,
        resolution=args.resolution,
        molecular_lines=not args.no_molecular_lines,
        device=args.device,
        dtype=args.dtype,
    )
    spectrum.save_npz(args.out)
    print(f"wrote {args.out} in {spectrum.seconds:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
