"""External atmosphere-deck compatibility I/O.

This boundary reads and writes the historical fixed-width pykurucz deck while
keeping the internal physical arrays explicit. Production synthesis consumes
the structured NPZ product instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import numpy as np


@dataclass
class ModelAtmosphere:
    """Structured atmosphere columns plus external-format metadata.

    Every physical column is a one-dimensional NumPy array with the same
    outer-to-inner depth ordering, and column mass increases with index.
    Column mass is in g cm^-2, gas pressure in dyn cm^-2, electron density in
    cm^-3, Rosseland opacity in cm^2 g^-1, radiative acceleration in cm s^-2,
    and microturbulence in cm s^-1. ``fixed_column_abundance_values`` preserves
    the external format's mixed convention and is keyed by atomic number 1
    through 99: values for H and He are linear number fractions, while metal
    values are base-10 logarithms of their number fractions.
    """

    column_mass: np.ndarray
    temperature: np.ndarray
    gas_pressure: np.ndarray
    electron_density: np.ndarray
    rosseland_opacity: np.ndarray
    radiative_acceleration: np.ndarray
    microturbulence: np.ndarray
    convective_flux: np.ndarray
    convective_velocity: np.ndarray
    metadata: dict[str, str] = field(default_factory=dict)
    fixed_column_abundance_values: dict[int, float] = field(default_factory=dict)

    @property
    def layers(self) -> int:
        return int(self.column_mass.size)

    @property
    def thermal_energy_erg(self) -> np.ndarray:
        return self.temperature * 1.38054e-16

    @property
    def h_over_kt(self) -> np.ndarray:
        return 6.6256e-27 / np.maximum(self.thermal_energy_erg, 1.0e-300)

    @property
    def hc_over_kt(self) -> np.ndarray:
        return (6.6256e-27 * 2.99792458e10) / np.maximum(
            self.thermal_energy_erg, 1.0e-300
        )

    @property
    def thermal_energy_ev(self) -> np.ndarray:
        return self.temperature / 11604.5

    @property
    def natural_log_temperature(self) -> np.ndarray:
        return np.log(np.maximum(self.temperature, 1.0e-300))


def _parse_abundance_change(
    line: str,
    fixed_column_abundance_values: dict[int, float],
) -> None:
    """Collect ``ABUNDANCE CHANGE Z value`` pairs from one header line."""

    keyword = "ABUNDANCE CHANGE"
    keyword_start = line.find(keyword)
    if keyword_start < 0:
        return
    pair_fields = line[keyword_start + len(keyword) :].split()
    for pair_index in range(0, len(pair_fields) - 1, 2):
        try:
            atomic_number = int(pair_fields[pair_index])
            abundance = float(pair_fields[pair_index + 1])
        except ValueError:
            break
        fixed_column_abundance_values[atomic_number] = abundance


def _split_run_together_deck_numbers(line: str) -> list[str]:
    # The deck rows are written in fixed-width scientific notation, so two
    # adjacent fields can run together with no separating blank when a value
    # fills its column, e.g. "4.131E+00-1.058E+07".  Re-insert the missing
    # separator before an exponent-free sign that starts the next field,
    # then split on whitespace as usual.
    return re.sub(r"(?<=[0-9])([+-])(?=\d)", r" \1", line).split()


def _metadata_from_header_line(line: str, metadata: dict[str, str]) -> None:
    """Extract TEFF/GRAVITY (or LOG G) metadata from the deck's first line."""

    header_fields = line.strip().split()
    if not header_fields or header_fields[0] != "TEFF":
        return
    metadata["stellar_parameters_line"] = line.rstrip()
    for field_index, token in enumerate(header_fields):
        if token == "TEFF" and field_index + 1 < len(header_fields):
            # Old decks write "TEFF 5777." with a trailing period.
            metadata["effective_temperature"] = (
                f"{float(header_fields[field_index + 1].rstrip('.')):.6f}"
            )
        if token == "GRAVITY" and field_index + 1 < len(header_fields):
            metadata["log_surface_gravity"] = (
                f"{float(header_fields[field_index + 1]):.6f}"
            )
        if (
            token == "G"
            and field_index >= 1
            and header_fields[field_index - 1] == "LOG"
            and field_index + 1 < len(header_fields)
        ):
            try:
                metadata["log_surface_gravity"] = (
                    f"{float(header_fields[field_index + 1]):.6f}"
                )
            except ValueError:
                pass


def parse_atmosphere_deck(
    deck_text: str, *, source: str = "<in-memory deck>"
) -> ModelAtmosphere:
    """Parse READ DECK6 atmosphere text into structured arrays.

    This is the same parser `read_atmosphere_deck` applies to on-disk files;
    the warm-start path calls it directly on freshly formatted deck text so
    the solver input carries exactly the deck-format quantization without a
    disk round-trip.  ``source`` only labels error messages.
    """

    lines = deck_text.splitlines()
    metadata: dict[str, str] = {}
    fixed_column_abundance_values: dict[int, float] = {}
    layer_row_start_index = -1

    # Header phase: everything above "READ DECK6" is free-form metadata
    # (TEFF/GRAVITY line, TITLE, OPACITY IFOP switches, PRESSURE toggle,
    # ABUNDANCE CHANGE pairs).  The whole block is kept verbatim in
    # metadata["predeck_block"] so a rewrite can reproduce it exactly.
    for line_index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        _metadata_from_header_line(raw_line, metadata)
        if line.startswith("TITLE"):
            metadata["title"] = raw_line[5:].strip() if len(raw_line) > 5 else ""
        elif line.startswith("OPACITY IFOP"):
            metadata["opacity_flags"] = raw_line.strip()
        elif line.startswith("PRESSURE"):
            pressure_fields = line.split()
            if len(pressure_fields) >= 2:
                metadata["pressure_iteration_enabled"] = (
                    "1" if pressure_fields[1].upper() == "ON" else "0"
                )
        elif "ABUNDANCE CHANGE" in line:
            _parse_abundance_change(line, fixed_column_abundance_values)
        elif line.startswith("READ DECK6"):
            layer_row_start_index = line_index + 1
            metadata["deck6_header"] = raw_line.strip()
            metadata["predeck_block"] = "\n".join(lines[: line_index + 1])
            break

    if layer_row_start_index < 0:
        raise ValueError(f"READ DECK6 section not found in {source}")

    # Layer-row phase: one row per depth layer with at least the seven
    # columns named in the DECK6 header (RHOX = column mass, T, P, XNE =
    # electron density, ABROSS = Rosseland opacity, ACCRAD = radiative
    # acceleration, VTURB) plus two optional convection columns (FLXCNV,
    # VCONV).  The first line that is not a numeric row ends the table.
    layer_rows: list[list[float]] = []
    layer_row_stop_index = layer_row_start_index
    for line_index in range(layer_row_start_index, len(lines)):
        line = lines[line_index].strip()
        if not line:
            continue
        row_fields = _split_run_together_deck_numbers(line)
        if len(row_fields) < 7:
            layer_row_stop_index = line_index
            break
        try:
            named_columns = [float(row_fields[column]) for column in range(7)]
            convective_flux = float(row_fields[7]) if len(row_fields) > 7 else 0.0
            convective_velocity = float(row_fields[8]) if len(row_fields) > 8 else 0.0
        except ValueError:
            layer_row_stop_index = line_index
            break
        layer_rows.append(named_columns + [convective_flux, convective_velocity])
        layer_row_stop_index = line_index + 1

    if not layer_rows:
        raise ValueError(f"No layer rows parsed from READ DECK6 in {source}")

    # Footer phase: PRADK (radiation-pressure seed), BEGIN (iteration
    # record), and END are kept verbatim for rewrites and PRADK parsing.
    for raw_line in lines[layer_row_stop_index:]:
        line = raw_line.strip()
        if line.startswith("PRADK"):
            metadata["surface_radiation_pressure_line"] = raw_line.rstrip()
        elif line.startswith("BEGIN"):
            metadata["begin_line"] = raw_line.rstrip()
        elif line.startswith("END"):
            metadata["end_line"] = raw_line.rstrip()

    layer_table = np.asarray(layer_rows, dtype=np.float64)
    return ModelAtmosphere(
        column_mass=layer_table[:, 0],
        temperature=layer_table[:, 1],
        gas_pressure=layer_table[:, 2],
        electron_density=layer_table[:, 3],
        rosseland_opacity=layer_table[:, 4],
        radiative_acceleration=layer_table[:, 5],
        microturbulence=layer_table[:, 6],
        convective_flux=layer_table[:, 7],
        convective_velocity=layer_table[:, 8],
        metadata=metadata,
        fixed_column_abundance_values=fixed_column_abundance_values,
    )


def read_atmosphere_deck(path: Path) -> ModelAtmosphere:
    """Read a pykurucz READ DECK6 ``.atm`` text deck.

    This is the TEXT-INTEROP path: it parses external/pykurucz decks (and the
    solver's own written comparison deck). It is NOT the internal product,
    which is the structured NPZ (``payne_zero_synthesis`` consumes the NPZ, not
    this text deck).
    """

    return parse_atmosphere_deck(
        path.read_text(encoding="utf-8", errors="replace"),
        source=str(path),
    )


def linear_elemental_abundances(model: ModelAtmosphere) -> np.ndarray:
    """Decode the fixed-column abundance block to 99 linear number fractions."""

    elemental_abundances = np.full(99, np.nan, dtype=np.float64)
    for atomic_number, stored_value in model.fixed_column_abundance_values.items():
        if not 1 <= int(atomic_number) <= 99:
            continue
        elemental_abundances[int(atomic_number) - 1] = (
            float(stored_value)
            if int(atomic_number) <= 2
            else float(np.power(10.0, stored_value))
        )
    missing = np.flatnonzero(~np.isfinite(elemental_abundances)) + 1
    if missing.size:
        shown = ", ".join(str(int(atomic_number)) for atomic_number in missing[:12])
        if missing.size > 12:
            shown += ", ..."
        raise ValueError(
            "fixed-column atmosphere does not carry a complete 99-element "
            f"abundance block; missing Z={shown}"
        )
    return elemental_abundances


def format_atmosphere_deck(model: ModelAtmosphere) -> str:
    """Format the exact fixed-column text used for quantization and interchange.

    Production calls this formatter and immediately reparses the returned text
    in memory. ``write_atmosphere_deck`` is the external-oracle file wrapper.
    """

    output: list[str] = []
    predeck = model.metadata.get("predeck_block")
    if predeck:
        output.extend(predeck.splitlines())
    else:
        effective_temperature = float(
            model.metadata.get("effective_temperature", "0.0")
        )
        log_surface_gravity = float(model.metadata.get("log_surface_gravity", "0.0"))
        title = model.metadata.get("title", "payne_zero_atmosphere")
        opacity_flags = model.metadata.get(
            "opacity_flags", "OPACITY IFOP 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 0 0 0"
        )
        output.append(
            f"TEFF   {effective_temperature:7.1f}  "
            f"GRAVITY  {log_surface_gravity:6.4f} LTE "
        )
        output.append(f"TITLE {title:<74}")
        output.append(f" {opacity_flags}")
        output.append(" CONVECTION ON   1.25 TURBULENCE OFF  0.00  0.00  0.00  0.00")
        hydrogen = model.fixed_column_abundance_values.get(1, 0.92)
        helium = model.fixed_column_abundance_values.get(2, 0.08)
        output.append(
            f"ABUNDANCE SCALE   1.00000 ABUNDANCE CHANGE 1 {hydrogen:.5f} 2 {helium:.5f}"
        )
        metals = sorted(
            (atomic_number, value)
            for atomic_number, value in model.fixed_column_abundance_values.items()
            if atomic_number >= 3
        )
        line = " ABUNDANCE CHANGE"
        for atomic_number, abundance in metals:
            fragment = f" {atomic_number:2d} {abundance:7.2f}"
            if len(line) + len(fragment) > 78:
                output.append(line)
                line = " ABUNDANCE CHANGE"
            line += fragment
        if metals:
            output.append(line)
        output.append(f"READ DECK6 {model.layers} RHOX,T,P,XNE,ABROSS,ACCRAD,VTURB")

    for depth_index in range(model.layers):
        output.append(
            f"{model.column_mass[depth_index]:14.8E} {model.temperature[depth_index]:8.1f}"
            f"{model.gas_pressure[depth_index]:10.3E}"
            f"{model.electron_density[depth_index]:10.3E}"
            f"{model.rosseland_opacity[depth_index]:10.3E}"
            f"{model.radiative_acceleration[depth_index]:10.3E}"
            f"{model.microturbulence[depth_index]:10.3E}"
            f"{model.convective_flux[depth_index]:10.3E}"
            f"{model.convective_velocity[depth_index]:10.3E}"
        )

    if surface_radiation_pressure_line := model.metadata.get(
        "surface_radiation_pressure_line"
    ):
        output.append(surface_radiation_pressure_line)
    output.append(
        model.metadata.get(
            "begin_line", "BEGIN                    ITERATION  1 COMPLETED"
        )
    )
    if end_line := model.metadata.get("end_line"):
        output.append(end_line)

    return "\n".join(output) + "\n"


def write_atmosphere_deck(model: ModelAtmosphere, path: Path) -> None:
    """Write an external fixed-column ``.atm`` interchange file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_atmosphere_deck(model), encoding="utf-8")
