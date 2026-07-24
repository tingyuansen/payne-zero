"""Synthesize from stellar labels or from a structured atmosphere archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from payne_zero_atmosphere.direct_abundance import DIRECT_XH_ATOMIC_NUMBERS
from payne_zero_atmosphere.warm_start import (
    ELEMENT_SYMBOLS,
    parse_abundance_offset,
)

from .api import synthesize, synthesize_from_labels
from .atmosphere import validate_atmosphere_npz


_DIRECT_XH_NAMES = {
    f"{ELEMENT_SYMBOLS[atomic_number].lower()}_over_h": atomic_number
    for atomic_number in DIRECT_XH_ATOMIC_NUMBERS
}


def _parse_abundances(
    entries: list[str],
    abundance_file: Path | None,
) -> dict[int, float]:
    parsed_entries = list(entries)
    if abundance_file is not None:
        payload = json.loads(abundance_file.expanduser().read_text())
        if not isinstance(payload, dict):
            raise ValueError("--abundance-file must contain one JSON object")
        parsed_entries.extend(f"{key}:{value}" for key, value in payload.items())
    result: dict[int, float] = {}
    for entry in parsed_entries:
        atomic_number, value = parse_abundance_offset(entry)
        if atomic_number in result:
            raise ValueError(
                f"duplicate abundance for atomic number {atomic_number}"
            )
        result[atomic_number] = value
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m payne_zero_synthesis.cli", description=__doc__
    )
    parser.add_argument(
        "atmosphere",
        type=Path,
        nargs="?",
        help=(
            "optional structured-atmosphere NPZ; omit it to synthesize directly "
            "from stellar labels"
        ),
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    labels = parser.add_argument_group("stellar-label synthesis")
    labels.add_argument("--effective-temperature", type=float)
    labels.add_argument("--log-surface-gravity", type=float)
    labels.add_argument(
        "--metallicity",
        type=float,
        help="[M/H] for the five- or eight-label initializer",
    )
    labels.add_argument("--alpha-enhancement", type=float, default=0.0)
    labels.add_argument("--microturbulence-km-s", type=float, default=2.0)
    labels.add_argument(
        "--c-over-m",
        type=float,
        help="[C/M]; selects the eight-label initializer",
    )
    labels.add_argument(
        "--n-over-m",
        type=float,
        help="[N/M]; selects the eight-label initializer",
    )
    labels.add_argument(
        "--o-over-m",
        type=float,
        help="[O/M]; selects the eight-label initializer",
    )
    labels.add_argument(
        "--initializer",
        choices=("auto", "five-label", "eight-label", "direct-abundance"),
        default="auto",
    )
    labels.add_argument(
        "--abundance",
        action="append",
        default=[],
        metavar="X:+x",
        help=(
            "individual [X/H], for example Mg:-0.2; repeatable and equivalent "
            "to the element-specific flags"
        ),
    )
    labels.add_argument(
        "--abundance-file",
        type=Path,
        help="JSON object mapping element symbols or atomic numbers to [X/H]",
    )
    labels.add_argument(
        "--save-initialized-atmosphere",
        type=Path,
        help=(
            "optionally retain the predicted, population-bridged atmosphere "
            "with an explicit unconverged-product marker"
        ),
    )
    direct = parser.add_argument_group("individual direct-abundance coordinates")
    for coordinate_name, atomic_number in _DIRECT_XH_NAMES.items():
        symbol = ELEMENT_SYMBOLS[atomic_number]
        direct.add_argument(
            f"--{symbol.lower()}-over-h",
            dest=coordinate_name,
            type=float,
            help=f"[{symbol}/H]",
        )
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
    parser.add_argument(
        "--device", default="auto", help="auto, cuda, mps, or cpu"
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float64", "float32"),
        default="auto",
        help="default: float32 on Metal, float64 on CUDA or CPU",
    )
    parser.add_argument("--no-molecular-lines", action="store_true")
    args = parser.parse_args(argv)

    if args.validate_only:
        if args.atmosphere is None:
            parser.error("--validate-only requires an atmosphere NPZ")
        array_names = validate_atmosphere_npz(args.atmosphere)
        print(
            f"{args.atmosphere} is a valid Payne Zero synthesis atmosphere "
            f"({len(array_names)} arrays)"
        )
        return 0
    if args.out is None:
        parser.error("--out is required unless --validate-only is used")

    labels_requested = (
        args.effective_temperature is not None
        or args.log_surface_gravity is not None
        or args.metallicity is not None
        or args.alpha_enhancement != 0.0
        or args.microturbulence_km_s != 2.0
        or args.c_over_m is not None
        or args.n_over_m is not None
        or args.o_over_m is not None
        or args.initializer != "auto"
        or args.abundance
        or args.abundance_file is not None
        or any(getattr(args, name) is not None for name in _DIRECT_XH_NAMES)
    )
    if args.atmosphere is not None and labels_requested:
        parser.error("choose an atmosphere NPZ or stellar labels, not both")

    if args.atmosphere is not None:
        spectrum = synthesize(
            args.atmosphere,
            wavelength_start_nm=args.wl_start_nm,
            wavelength_end_nm=args.wl_end_nm,
            resolution=args.resolution,
            molecular_lines=not args.no_molecular_lines,
            device=args.device,
            dtype=args.dtype,
        )
    else:
        if args.effective_temperature is None or args.log_surface_gravity is None:
            parser.error(
                "label synthesis requires --effective-temperature and "
                "--log-surface-gravity"
            )
        try:
            x_over_h = _parse_abundances(args.abundance, args.abundance_file)
        except ValueError as error:
            parser.error(str(error))
        for coordinate_name, atomic_number in _DIRECT_XH_NAMES.items():
            value = getattr(args, coordinate_name)
            if value is None:
                continue
            if atomic_number in x_over_h:
                parser.error(
                    f"duplicate [X/H] value for {ELEMENT_SYMBOLS[atomic_number]}"
                )
            x_over_h[atomic_number] = float(value)
        family = {
            "auto": "auto",
            "five-label": "five_label",
            "eight-label": "cno8",
            "direct-abundance": "direct_abundance",
        }[args.initializer]
        direct_requested = family == "direct_abundance" or bool(x_over_h)
        if direct_requested:
            if args.metallicity is not None:
                parser.error(
                    "direct-abundance synthesis uses --fe-over-h, not --metallicity"
                )
            if 26 not in x_over_h:
                parser.error(
                    "direct-abundance synthesis requires --fe-over-h"
                )
            fe_over_h = float(x_over_h[26])
            metallicity = 0.0
        else:
            fe_over_h = None
            metallicity = 0.0 if args.metallicity is None else args.metallicity
        spectrum = synthesize_from_labels(
            effective_temperature=args.effective_temperature,
            log_surface_gravity=args.log_surface_gravity,
            metallicity=metallicity,
            fe_over_h=fe_over_h,
            alpha_enhancement=args.alpha_enhancement,
            microturbulence_km_s=args.microturbulence_km_s,
            c_over_m=args.c_over_m,
            n_over_m=args.n_over_m,
            o_over_m=args.o_over_m,
            x_over_h=x_over_h if direct_requested else None,
            initializer_family=family,
            wavelength_start_nm=args.wl_start_nm,
            wavelength_end_nm=args.wl_end_nm,
            r_grid=args.resolution,
            molecular_lines=not args.no_molecular_lines,
            device=args.device,
            dtype=args.dtype,
        )
        if args.save_initialized_atmosphere is not None:
            spectrum.initialized_atmosphere.save_npz(
                args.save_initialized_atmosphere
            )
            print(
                "wrote initialized atmosphere "
                f"{args.save_initialized_atmosphere} "
                "(marked atmosphere_converged=false)"
            )
    spectrum.save_npz(args.out)
    family_text = (
        ""
        if not hasattr(spectrum, "initializer_family")
        else f" with {spectrum.initializer_family}"
    )
    print(f"wrote {args.out} in {spectrum.seconds:.3f}s{family_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
