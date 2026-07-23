"""Molecular-equilibrium input catalog."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


MAX_MOLECULES = 200
MAX_MOLECULAR_EQUATIONS = 35
MAX_MOLECULAR_COMPONENTS = 3 * MAX_MOLECULES

_MOLECULE_CODE_DIGIT_SCALES = np.array(
    [1.0e14, 1.0e12, 1.0e10, 1.0e8, 1.0e6, 1.0e4, 1.0e2, 1.0],
    dtype=np.float64,
)


@dataclass
class MolecularEquilibriumCatalog:
    """Parsed molecular-equilibrium records used by the atmosphere solver."""

    molecule_count: int
    equation_count: int
    component_count: int
    molecule_codes: np.ndarray
    equilibrium_coefficients: np.ndarray
    component_start_indices: np.ndarray
    component_equation_indices: np.ndarray
    equation_species_codes: np.ndarray
    species_to_equation_index: np.ndarray


def _fixed_width_float(line: str, start: int, stop: int) -> float:
    field = line[start:stop].strip()
    return float(field) if field else 0.0


def parse_molecular_equilibrium_record(raw_line: str) -> tuple[float, ...]:
    """Parse one legacy `READMOL` fixed-width boundary record."""

    line = raw_line.rstrip("\n\r")
    if not line:
        raise ValueError("blank molecular-equilibrium record")
    molecule_code = float(line[0 : min(18, len(line))].strip())
    return (
        molecule_code,
        _fixed_width_float(line, 18, 25),
        _fixed_width_float(line, 25, 36),
        _fixed_width_float(line, 36, 47),
        _fixed_width_float(line, 47, 58),
        _fixed_width_float(line, 58, 69),
        _fixed_width_float(line, 69, 80),
    )


def read_molecular_equilibrium_catalog(path: Path) -> MolecularEquilibriumCatalog:
    """Read molecular-equilibrium definitions.

    Canonical form is an ``.npz`` holding the parsed catalog arrays. The
    historical text catalog remains readable only for provenance tooling.
    """

    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(
            f"Molecular-equilibrium catalog not found: {source_path}"
        )
    if source_path.suffix == ".npz":
        with np.load(source_path, allow_pickle=False) as arrays:
            return MolecularEquilibriumCatalog(
                molecule_count=int(arrays["molecule_count"]),
                equation_count=int(arrays["equation_count"]),
                component_count=int(arrays["component_count"]),
                molecule_codes=np.asarray(arrays["molecule_codes"]),
                equilibrium_coefficients=np.asarray(arrays["equilibrium_coefficients"]),
                component_start_indices=np.asarray(arrays["component_start_indices"]),
                component_equation_indices=np.asarray(
                    arrays["component_equation_indices"]
                ),
                equation_species_codes=np.asarray(arrays["equation_species_codes"]),
                species_to_equation_index=np.asarray(
                    arrays["species_to_equation_index"]
                ),
            )

    molecule_codes = np.zeros(MAX_MOLECULES, dtype=np.float64)
    equilibrium_coefficients = np.zeros((7, MAX_MOLECULES), dtype=np.float64)
    component_start_indices = np.zeros(MAX_MOLECULES + 1, dtype=np.int32)
    component_species_codes = np.zeros(MAX_MOLECULAR_COMPONENTS, dtype=np.int32)
    component_equation_indices = np.zeros(MAX_MOLECULAR_COMPONENTS, dtype=np.int32)
    equation_species_codes = np.zeros(MAX_MOLECULAR_EQUATIONS, dtype=np.int32)
    species_to_equation_index = np.full(102, -1, dtype=np.int32)
    species_seen = np.zeros(102, dtype=np.int32)

    component_cursor = 0
    molecule_count = 0

    with source_path.open("r", encoding="ascii", errors="ignore") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if (
                not stripped
                or stripped.startswith("C")
                or stripped.startswith("c")
                or stripped.startswith("#")
            ):
                continue

            parsed = parse_molecular_equilibrium_record(raw_line)
            molecule_code = parsed[0]
            if molecule_code == 0.0:
                break
            if molecule_count >= MAX_MOLECULES:
                raise ValueError(f"Too many molecules: >{MAX_MOLECULES}")

            first_digit = -1
            for index, scale in enumerate(_MOLECULE_CODE_DIGIT_SCALES):
                if molecule_code >= scale:
                    first_digit = index
                    break
            if first_digit < 0:
                raise ValueError(f"Invalid molecule code: {molecule_code}")

            remainder = float(molecule_code)
            for scale in _MOLECULE_CODE_DIGIT_SCALES[first_digit:]:
                if component_cursor >= MAX_MOLECULAR_COMPONENTS:
                    raise ValueError(
                        f"Too many molecule components: >{MAX_MOLECULAR_COMPONENTS}"
                    )
                species_code = int(remainder / scale + 0.5)
                remainder -= float(species_code) * scale
                if species_code == 0:
                    species_code = 100
                species_seen[species_code] = 1
                component_species_codes[component_cursor] = species_code
                component_cursor += 1

            ion_count = int(remainder * 100.0 + 0.5)
            if ion_count >= 1:
                species_seen[100] = 1
                species_seen[101] = 1
                for _ in range(ion_count):
                    if component_cursor >= MAX_MOLECULAR_COMPONENTS:
                        raise ValueError(
                            f"Too many molecule components: >{MAX_MOLECULAR_COMPONENTS}"
                        )
                    component_species_codes[component_cursor] = 101
                    component_cursor += 1

            molecule_codes[molecule_count] = molecule_code
            equilibrium_coefficients[:6, molecule_count] = parsed[1:]
            molecule_count += 1
            component_start_indices[molecule_count] = component_cursor

    component_count = int(component_cursor)
    next_equation = 1
    for species_code in range(1, 101):
        if species_seen[species_code] == 0:
            continue
        next_equation += 1
        if next_equation > MAX_MOLECULAR_EQUATIONS:
            raise ValueError(
                f"Too many molecular equations: {next_equation} > {MAX_MOLECULAR_EQUATIONS}"
            )
        equation_index = next_equation - 1
        species_to_equation_index[species_code] = equation_index
        equation_species_codes[equation_index] = species_code

    equation_count = int(next_equation)
    species_to_equation_index[101] = equation_count
    for component_index in range(component_count):
        species_code = int(component_species_codes[component_index])
        component_equation_indices[component_index] = int(
            species_to_equation_index[species_code]
        )

    return MolecularEquilibriumCatalog(
        molecule_count=molecule_count,
        equation_count=equation_count,
        component_count=component_count,
        molecule_codes=molecule_codes,
        equilibrium_coefficients=equilibrium_coefficients,
        component_start_indices=component_start_indices,
        component_equation_indices=component_equation_indices,
        equation_species_codes=equation_species_codes,
        species_to_equation_index=species_to_equation_index,
    )


def find_default_molecular_equilibrium_catalog() -> Path | None:
    """Return the bundled molecular-equilibrium source catalog, if resolvable."""

    from .source_catalogs import SourceCatalogError, molecular_equilibrium_catalog_path

    try:
        return molecular_equilibrium_catalog_path()
    except (SourceCatalogError, FileNotFoundError):
        return None
