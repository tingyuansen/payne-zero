"""Command-line model-atmosphere solver.

``python -m payne_zero_atmosphere --effective-temperature 5777
--log-surface-gravity 4.44 --out DIR`` solves a converged model atmosphere from
a learned warm start and writes the single modern product
``DIR/payne_zero_structured_atmosphere.npz``. No ``.atm`` text files are
written.

This is the first-class solve entry point, and it reproduces the certified
solve path exactly. The load-bearing details:

* Start: a bundled complete-atmosphere initializer; no reference atmosphere or
  comparison oracle is needed at runtime.
* Physics: molecules, convection, and the full source-line catalogs are on.
* Convergence: the production controls are baked in as defaults (a ``15 x 2``
  iteration policy, at least three iterations, one consecutive converged
  iteration, and a maximum deep-layer relative temperature change of
  ``0.0005``). The trial budget is configurable;
  the stopping thresholds define the accepted atmosphere.
* Molecular convection finite differences track each thermal perturbation.
* The converged columns pass through the fixed-column quantization contract in
  memory before the structured product is built; no text atmosphere is written.

See ``payne_zero_atmosphere/README.md`` for threading and data-location options.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Mapping

from .config import AtmosphereConfig, AtmosphereInput, AtmosphereOutput
from .runner import run_atmosphere_model
from .source_catalogs import molecular_equilibrium_catalog_path, source_line_paths
from .warm_start import (
    ELEMENT_SYMBOLS,
    deterministic_initializer_labels,
    emulator_warm_start_model,
    parse_abundance_offset,
)
from .direct_abundance import DIRECT_XH_ATOMIC_NUMBERS

_DIRECT_XH_NAMES = {
    f"{ELEMENT_SYMBOLS[atomic_number].lower()}_over_h": atomic_number
    for atomic_number in DIRECT_XH_ATOMIC_NUMBERS
}

# The in-memory warm start carries the certified continuum and source-line
# switches, so the CLI needs no secondary physics override.


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m payne_zero_atmosphere",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--effective-temperature",
        type=float,
        required=True,
        help="effective temperature [K]",
    )
    parser.add_argument(
        "--log-surface-gravity",
        type=float,
        required=True,
        help="log surface gravity [cgs]",
    )
    parser.add_argument(
        "--metallicity",
        type=float,
        default=0.0,
        help="[M/H] scaled metallicity (default 0)",
    )
    parser.add_argument(
        "--alpha-enhancement",
        type=float,
        default=0.0,
        help="[alpha/M] alpha-element enhancement (default 0)",
    )
    parser.add_argument(
        "--microturbulence-km-s",
        type=float,
        default=2.0,
        help="microturbulent velocity [km/s] (default 2)",
    )
    parser.add_argument(
        "--c-over-m",
        "--carbon-enhancement",
        type=float,
        help=(
            "[C/M]; supplying any CNO coordinate selects the eight-label "
            "CNO-aware initializer"
        ),
    )
    parser.add_argument(
        "--n-over-m",
        "--nitrogen-enhancement",
        type=float,
        help=(
            "[N/M]; supplying any CNO coordinate selects the eight-label "
            "CNO-aware initializer"
        ),
    )
    parser.add_argument(
        "--o-over-m",
        "--oxygen-enhancement",
        type=float,
        help=(
            "[O/M], replacing the alpha-scaled oxygen abundance in the "
            "eight-label CNO-aware initializer"
        ),
    )
    parser.add_argument(
        "--initializer",
        choices=("auto", "direct-abundance"),
        default="auto",
        help=(
            "initializer family: auto selects 5D or 8D; direct-abundance "
            "uses [Fe/H] plus any individual [X/H] coordinates"
        ),
    )
    parser.add_argument(
        "--abundance",
        action="append",
        default=[],
        metavar="N:+x",
        help=(
            "[X/H] by symbol or atomic number, e.g. Fe:+0.3 or 26:+0.3; "
            "repeatable; unspecified elements inherit [Fe/H] with "
            "--initializer direct-abundance"
        ),
    )
    parser.add_argument(
        "--abundance-file",
        type=Path,
        help=(
            "JSON object mapping element symbols or atomic numbers to [X/H]; "
            "unspecified elements inherit Fe; "
            "use with --initializer direct-abundance"
        ),
    )
    direct_group = parser.add_argument_group(
        "individual direct-abundance coordinates",
        "Supplying any --x-over-h coordinate selects the direct-abundance "
        "initializer. Unspecified elements inherit --fe-over-h.",
    )
    for coordinate_name, atomic_number in _DIRECT_XH_NAMES.items():
        symbol = ELEMENT_SYMBOLS[atomic_number]
        direct_group.add_argument(
            f"--{symbol.lower()}-over-h",
            dest=coordinate_name,
            type=float,
            help=f"[{symbol}/H]",
        )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output directory for the products (created if missing)",
    )
    parser.add_argument(
        "--iterations-per-trial",
        type=int,
        default=15,
        help="maximum exact iterations per independent initializer (default 15)",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=2,
        help="maximum deterministic initializer trials (default 2)",
    )
    parser.add_argument(
        "--initializer-seed",
        type=int,
        default=20260713,
        help="deterministic nearby-initializer seed (default 20260713)",
    )
    parser.add_argument(
        "--initializer-jitter-scale",
        type=float,
        default=0.01,
        help="nearby-label offset as a fraction of checkpoint width (default 0.01)",
    )
    return parser


def _parse_abundances(
    entries: list[str], abundance_file: Path | None
) -> dict[int, float]:
    """Parse CLI abundance entries without silently replacing duplicates."""

    parsed_entries = list(entries)
    if abundance_file is not None:
        payload = json.loads(abundance_file.expanduser().read_text())
        if not isinstance(payload, dict):
            raise ValueError("--abundance-file must contain one JSON object")
        parsed_entries.extend(f"{key}:{value}" for key, value in payload.items())

    abundances: dict[int, float] = {}
    for entry in parsed_entries:
        atomic_number, value = parse_abundance_offset(entry)
        if atomic_number in abundances:
            raise ValueError(
                f"duplicate abundance for atomic number {atomic_number}"
            )
        abundances[atomic_number] = value
    return abundances


def _complete_direct_abundances(
    partial: Mapping[int, float],
) -> dict[int, float]:
    """Fill unspecified direct coordinates with the explicitly supplied [Fe/H]."""

    if 26 not in partial:
        raise ValueError("direct-abundance mode requires Fe_over_h / --fe-over-h")
    iron = float(partial[26])
    complete = {atomic_number: iron for atomic_number in DIRECT_XH_ATOMIC_NUMBERS}
    complete.update({int(key): float(value) for key, value in partial.items()})
    return complete


def solve_structured_atmosphere(
    *,
    effective_temperature: float,
    log_surface_gravity: float,
    out_dir: Path | str,
    metallicity: float = 0.0,
    alpha_enhancement: float = 0.0,
    microturbulence_km_s: float = 2.0,
    c_over_m: float | None = None,
    n_over_m: float | None = None,
    o_over_m: float | None = None,
    carbon_over_m: float | None = None,
    nitrogen_over_m: float | None = None,
    oxygen_over_m: float | None = None,
    carbon_enhancement: float | None = None,
    nitrogen_enhancement: float | None = None,
    oxygen_enhancement: float | None = None,
    absolute_abundance_offsets: Mapping[int, float] | None = None,
    initializer: str = "auto",
    abundance_by_atomic_number: Mapping[int, float] | None = None,
    iterations_per_trial: int = 15,
    max_trials: int = 2,
    initializer_seed: int = 20260713,
    initializer_jitter_scale: float = 0.01,
    **element_over_h: float,
) -> Path:
    """Solve a converged model atmosphere and write the modern ``.npz`` product.

    The programmatic twin of ``python -m payne_zero_atmosphere``: it solves from
    the emulator warm start on the certified solve path and writes the single
    product ``<out_dir>/payne_zero_structured_atmosphere.npz`` (the packed solver
    state the synthesizer consumes directly), returning its path. No ``.atm``
    file is written: fixed-column deck quantization happens in memory. A
    temporary directory stages each trial's structured NPZ until a converged
    product can be promoted atomically, then it is discarded.

    ``initializer="auto"`` selects the five- or eight-label initializer.
    ``absolute_abundance_offsets`` then supplies optional exact-solver ``[X/H]``
    overrides. ``initializer="direct-abundance"`` instead accepts ``fe_over_h``
    and any individual ``x_over_h`` coordinates. Unspecified elements inherit
    ``fe_over_h``. The resulting complete 81-coordinate mixture is used for both
    the starting structure and the converged physical solve. Generated callers
    may provide the same coordinates through ``abundance_by_atomic_number``.
    """
    coordinate_options = (
        ("[C/M]", c_over_m, carbon_over_m, carbon_enhancement),
        ("[N/M]", n_over_m, nitrogen_over_m, nitrogen_enhancement),
        ("[O/M]", o_over_m, oxygen_over_m, oxygen_enhancement),
    )
    resolved_cno: list[float | None] = []
    for name, *options in coordinate_options:
        supplied = [value for value in options if value is not None]
        if len(supplied) > 1:
            raise ValueError(f"supply {name} through only one Python keyword")
        resolved_cno.append(supplied[0] if supplied else None)
    carbon_enhancement, nitrogen_enhancement, oxygen_enhancement = resolved_cno

    named_abundances: dict[int, float] = {}
    for coordinate_name, value in element_over_h.items():
        atomic_number = _DIRECT_XH_NAMES.get(coordinate_name.lower())
        if atomic_number is None:
            raise TypeError(
                f"unexpected keyword {coordinate_name!r}; individual abundances "
                "use names such as fe_over_h or mg_over_h"
            )
        named_abundances[atomic_number] = float(value)

    if initializer not in {"auto", "direct-abundance"}:
        raise ValueError("initializer must be 'auto' or 'direct-abundance'")
    direct_abundance = (
        initializer == "direct-abundance"
        or abundance_by_atomic_number is not None
        or bool(named_abundances)
    )
    if direct_abundance:
        if absolute_abundance_offsets is not None:
            raise ValueError(
                "direct-abundance mode uses abundance_by_atomic_number, not "
                "absolute_abundance_offsets"
            )
        supplied_abundances = dict(abundance_by_atomic_number or {})
        duplicates = sorted(set(supplied_abundances) & set(named_abundances))
        if duplicates:
            symbols = ", ".join(ELEMENT_SYMBOLS[z] for z in duplicates)
            raise ValueError(f"duplicate direct-abundance coordinates: {symbols}")
        supplied_abundances.update(named_abundances)
        if not supplied_abundances:
            raise ValueError(
                "direct-abundance mode requires Fe_over_h and any desired "
                "individual X_over_h coordinates"
            )
        if (
            metallicity != 0.0
            or alpha_enhancement != 0.0
            or any(value is not None for value in resolved_cno)
        ):
            raise ValueError(
                "direct-abundance mode replaces [M/H], [alpha/M], and CNO "
                "coordinates with the complete [X/H] mapping"
            )
        from .direct_abundance import complete_direct_abundance_vector

        abundance_by_atomic_number = _complete_direct_abundances(
            supplied_abundances
        )
        complete_direct_abundance_vector(abundance_by_atomic_number)
    elif abundance_by_atomic_number is not None:
        raise ValueError(
            "abundance_by_atomic_number requires initializer='direct-abundance'"
        )

    iterations_per_trial = int(iterations_per_trial)
    max_trials = int(max_trials)
    if iterations_per_trial < 1:
        raise ValueError("iterations_per_trial must be positive")
    if max_trials < 1:
        raise ValueError("max_trials must be positive")

    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    structured_atmosphere_path = out_dir / "payne_zero_structured_atmosphere.npz"
    structured_atmosphere_path.unlink(missing_ok=True)
    offsets = dict(absolute_abundance_offsets or {})
    if direct_abundance:
        initializers: tuple[dict[str, float] | None, ...] = (None,)
    else:
        initializers = deterministic_initializer_labels(
            effective_temperature=effective_temperature,
            log_surface_gravity=log_surface_gravity,
            metallicity=metallicity,
            alpha_enhancement=alpha_enhancement,
            microturbulence_km_s=microturbulence_km_s,
            carbon_enhancement=carbon_enhancement,
            nitrogen_enhancement=nitrogen_enhancement,
            oxygen_enhancement=oxygen_enhancement,
            absolute_abundance_offsets=offsets,
            max_trials=max_trials,
            seed=int(initializer_seed),
            jitter_scale=float(initializer_jitter_scale),
            device="cpu",
        )

    # The converged deck is an INTERNAL step: the structured .npz product is
    # rebuilt from the deck's quantized columns (the load-bearing quantization).
    # Each trial is staged below the output directory so a converged product can
    # be atomically promoted. Failed trials and internal deck text are discarded.
    completed_iterations = 0
    last_deep_layer_relative_temperature_change = float("nan")
    with tempfile.TemporaryDirectory(
        prefix=".payne_zero_atmosphere_", dir=out_dir
    ) as internal_dir:
        internal_root = Path(internal_dir)
        for trial_index, initializer_label in enumerate(initializers):
            trial_dir = internal_root / f"trial_{trial_index:02d}"
            trial_dir.mkdir()
            if direct_abundance:
                from .direct_abundance import _direct_abundance_warm_start_model

                warm_start_atmosphere, _deck_text = (
                    _direct_abundance_warm_start_model(
                        effective_temperature=effective_temperature,
                        log_surface_gravity=log_surface_gravity,
                        microturbulence_km_s=microturbulence_km_s,
                        abundance_by_atomic_number=abundance_by_atomic_number or {},
                        enable_experimental=True,
                        device="cpu",
                    )
                )
            else:
                warm_start_atmosphere, _deck_text = emulator_warm_start_model(
                    effective_temperature=effective_temperature,
                    log_surface_gravity=log_surface_gravity,
                    metallicity=metallicity,
                    alpha_enhancement=alpha_enhancement,
                    microturbulence_km_s=microturbulence_km_s,
                    carbon_enhancement=carbon_enhancement,
                    nitrogen_enhancement=nitrogen_enhancement,
                    oxygen_enhancement=oxygen_enhancement,
                    absolute_abundance_offsets=offsets,
                    device="cpu",
                    initializer_label=initializer_label,
                )
            trial_structured_path = trial_dir / structured_atmosphere_path.name
            config = AtmosphereConfig(
                inputs=AtmosphereInput(
                    initial_atmosphere=warm_start_atmosphere,
                    molecules_path=molecular_equilibrium_catalog_path(),
                    **source_line_paths(),
                ),
                outputs=AtmosphereOutput(
                    structured_atmosphere_path=trial_structured_path,
                ),
                iterations=iterations_per_trial,
                enable_molecules=True,
                enable_convection=True,
                enable_convergence_stop=True,
                minimum_iterations_before_convergence=3,
                required_consecutive_converged_iterations=1,
                maximum_deep_layer_relative_temperature_change=0.0005,
            )
            result = run_atmosphere_model(config)
            completed_iterations += int(result.iterations_completed)
            last_deep_layer_relative_temperature_change = float(
                result.diagnostics.get(
                    "deep_layer_relative_temperature_change", float("nan")
                )
            )
            if not result.converged:
                continue
            if not trial_structured_path.is_file():
                warning = result.diagnostics.get(
                    "structured_atmosphere_warning", "no product was written"
                )
                raise RuntimeError(
                    "converged atmosphere product could not be written: " + str(warning)
                )
            trial_structured_path.replace(structured_atmosphere_path)
            return structured_atmosphere_path

    raise RuntimeError(
        "atmosphere did not converge after "
        f"{completed_iterations} exact iterations across {len(initializers)} trial(s); "
        "final deep_layer_relative_temperature_change="
        f"{last_deep_layer_relative_temperature_change:.6g}"
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    absolute_abundance_offsets = _parse_abundances(
        args.abundance, args.abundance_file
    )
    named_abundances = {
        coordinate_name: getattr(args, coordinate_name)
        for coordinate_name in _DIRECT_XH_NAMES
        if getattr(args, coordinate_name) is not None
    }
    direct_requested = (
        args.initializer == "direct-abundance"
        or args.abundance_file is not None
        or bool(named_abundances)
    )
    structured_atmosphere_path = solve_structured_atmosphere(
        effective_temperature=args.effective_temperature,
        log_surface_gravity=args.log_surface_gravity,
        out_dir=args.out,
        metallicity=args.metallicity,
        alpha_enhancement=args.alpha_enhancement,
        microturbulence_km_s=args.microturbulence_km_s,
        c_over_m=args.c_over_m,
        n_over_m=args.n_over_m,
        o_over_m=args.o_over_m,
        initializer="direct-abundance" if direct_requested else args.initializer,
        abundance_by_atomic_number=(
            absolute_abundance_offsets
            if direct_requested
            else None
        ),
        absolute_abundance_offsets=(
            absolute_abundance_offsets if not direct_requested else None
        ),
        iterations_per_trial=args.iterations_per_trial,
        max_trials=args.max_trials,
        initializer_seed=args.initializer_seed,
        initializer_jitter_scale=args.initializer_jitter_scale,
        **named_abundances,
    )
    print(f"wrote {structured_atmosphere_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
