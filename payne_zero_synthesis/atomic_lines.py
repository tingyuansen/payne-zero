"""Atomic line-catalog parsing and compiled catalog loading.

The first-time (no cache) path parses the external fixed-width atomic source
catalog into a structure-of-arrays bundle. Normal synthesis reuses the compiled
catalog, so raw text parsing is a build/provenance path rather than a runtime
dependency.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import paths as runtime_paths
from .constants import CLASSICAL_LINE_STRENGTH_COEFFICIENT, LIGHT_SPEED_NM_PER_S

try:  # torch is optional for the numpy-only path (parity tests run CPU/fp64)
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# Atomic-catalog constants and tables.

_RECORD_LEN = 161  # External source catalog: 160 chars + '\n', uniform fixed-width

# Wavelength margins selected by each catalog line-size class, in nm.
_LINE_WINDOW_MARGINS_NM = np.array(
    [100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1], dtype=np.float64
)

# Periodic table (index = atomic number Z; 0 is a placeholder).  This ordering
# preserves the source-catalog species codes expected by the parity tests.
_ELEMENT_SYMBOLS = [
    "",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
]

# Cache versioning: bump when parsing, filtering, indexing, or sorting changes
# so stale package caches are rejected.
CACHE_SCHEMA = 1
CACHE_LOGIC_VERSION = 3


def _default_atomic_source_catalog() -> Path:
    return runtime_paths.source_catalog_path("lines", "atomic_source_lines_parsed.npz")


_IONIZATION_POTENTIAL_TABLE_PATH = (
    runtime_paths.SYNTHESIS_TABLE_DIR / "ionization_potential_lookup.npz"
)

_IONIZATION_POTENTIAL_TABLE: Optional[np.ndarray] = None


def _load_ionization_potential_table() -> np.ndarray:
    """Load the package ionization-potential lookup table [cm^-1]."""
    global _IONIZATION_POTENTIAL_TABLE
    if _IONIZATION_POTENTIAL_TABLE is not None:
        return _IONIZATION_POTENTIAL_TABLE
    _IONIZATION_POTENTIAL_TABLE = np.asarray(
        np.load(_IONIZATION_POTENTIAL_TABLE_PATH)["ionization_potential_cm"],
        dtype=np.float64,
    )
    return _IONIZATION_POTENTIAL_TABLE


# Geometric wavelength grid.


@dataclass(frozen=True)
class Grid:
    """Geometric wavelength grid used by line-center and wing indexing."""

    start_wavelength_nm: float
    end_wavelength_nm: float
    resolution: float  # R_grid (~3e5)

    @property
    def ratio(self) -> float:
        return 1.0 + 1.0 / self.resolution

    @property
    def log_spacing(self) -> float:
        return math.log(self.ratio)

    def build(self) -> np.ndarray:
        """Construct the geometric wavelength grid [nm] (float64)."""
        ratio = self.ratio
        log_spacing = math.log(ratio)
        first_grid_index = math.floor(math.log(self.start_wavelength_nm) / log_spacing)
        if math.exp(first_grid_index * log_spacing) < self.start_wavelength_nm:
            first_grid_index += 1
        first_wavelength = math.exp(first_grid_index * log_spacing)
        wavelength_values = []
        wavelength_nm = first_wavelength
        end_limit = self.end_wavelength_nm * (1.0 + 1e-9)
        while wavelength_nm <= end_limit:
            wavelength_values.append(wavelength_nm)
            wavelength_nm *= ratio
        return np.array(wavelength_values, dtype=np.float64)


def _center_indices(
    wavelength_grid: np.ndarray, line_wavelength_nm: np.ndarray
) -> np.ndarray:
    """Center-pixel index per line on the geometric wavelength grid.

    Returns -1 below the grid and grid.size above it; margin lines may still
    contribute through their wing anchors.
    """
    if wavelength_grid.size < 2:
        return np.zeros(line_wavelength_nm.size, dtype=np.int64)
    ratio = wavelength_grid[1] / wavelength_grid[0]
    log_spacing = np.log(ratio)
    grid_start_index = int(np.log(wavelength_grid[0]) / log_spacing + 0.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        nearest_log_index = (np.log(line_wavelength_nm) / log_spacing + 0.5).astype(
            np.int64
        )
        center_index = nearest_log_index - grid_start_index
    center_index[line_wavelength_nm < wavelength_grid[0]] = -1
    center_index[line_wavelength_nm > wavelength_grid[-1]] = wavelength_grid.size
    return center_index


# Default damping for surviving catalog records.


def _ionization_potential_index(
    atomic_number: np.ndarray, charge_offset: np.ndarray
) -> np.ndarray:
    """0-based index into the flat ionization-potential table."""
    table_index_1based = np.where(
        atomic_number <= 30,
        atomic_number * (atomic_number + 1) // 2 + charge_offset,
        atomic_number * 5 + 341 + charge_offset,
    )
    return table_index_1based - 1


def _lookup_ionization_potential_cm(table_index: np.ndarray) -> np.ndarray:
    ionization_potential_table = _load_ionization_potential_table()
    ionization_potential_cm = np.zeros(table_index.shape, dtype=np.float64)
    valid = (table_index >= 0) & (table_index < ionization_potential_table.size)
    ionization_potential_cm[valid] = ionization_potential_table[table_index[valid]]
    return ionization_potential_cm


def _default_stark_log(
    species_code: np.ndarray,
    lower_excitation_cm: np.ndarray,
    upper_excitation_cm: np.ndarray,
) -> np.ndarray:
    """Default Stark log10(gamma) per line, vectorized."""
    species_code = np.asarray(species_code, np.float64)
    atomic_number = (species_code + 1e-6).astype(np.int64)
    charge_offset = ((species_code - atomic_number) * 100.0 + 0.1).astype(np.int64)
    effective_charge = (charge_offset + 1).astype(np.float64)
    upper_level_excitation_cm = np.maximum(
        np.abs(lower_excitation_cm),
        np.abs(upper_excitation_cm),
    )

    effective_quantum_number_squared = np.full(species_code.shape, 25.0)
    table_index = _ionization_potential_index(atomic_number, charge_offset)
    ionization_potential_cm = _lookup_ionization_potential_cm(table_index)
    upper_level_gap_cm = ionization_potential_cm - upper_level_excitation_cm
    has_bound_upper_level = upper_level_gap_cm > 0
    effective_quantum_number_squared = np.where(
        has_bound_upper_level,
        109737.31
        * effective_charge
        * effective_charge
        / np.where(has_bound_upper_level, upper_level_gap_cm, 1.0),
        effective_quantum_number_squared,
    )

    stark_gamma = (
        1.0e-8
        * effective_quantum_number_squared
        * effective_quantum_number_squared
        * np.sqrt(effective_quantum_number_squared)
    )
    stark_log = np.log10(stark_gamma)
    stark_log = np.where(species_code >= 100.0, -5.0, stark_log)
    return stark_log


def _default_van_der_waals_log(
    species_code: np.ndarray,
    lower_excitation_cm: np.ndarray,
    upper_excitation_cm: np.ndarray,
) -> np.ndarray:
    """Default van der Waals log10(gamma) per line, vectorized."""
    species_code = np.asarray(species_code, np.float64)
    atomic_number = (species_code + 1e-6).astype(np.int64)
    charge_offset = ((species_code - atomic_number) * 100.0 + 0.1).astype(np.int64)
    effective_charge = (charge_offset + 1).astype(np.float64)

    lower_level_excitation_cm = np.minimum(
        np.abs(lower_excitation_cm),
        np.abs(upper_excitation_cm),
    )
    upper_level_excitation_cm = np.maximum(
        np.abs(lower_excitation_cm),
        np.abs(upper_excitation_cm),
    )

    table_index = _ionization_potential_index(atomic_number, charge_offset)
    ionization_potential_cm = _lookup_ionization_potential_cm(table_index)

    upper_effective_quantum_number_squared = np.full(species_code.shape, 25.0)
    has_bound_upper_level = (ionization_potential_cm > 0) & (
        (ionization_potential_cm - upper_level_excitation_cm) > 0
    )
    upper_level_gap_cm = np.where(
        has_bound_upper_level,
        ionization_potential_cm - upper_level_excitation_cm,
        1.0,
    )
    upper_effective_quantum_number_squared = np.where(
        has_bound_upper_level,
        109737.31 * effective_charge * effective_charge / upper_level_gap_cm,
        upper_effective_quantum_number_squared,
    )
    upper_effective_quantum_number_squared = np.minimum(
        upper_effective_quantum_number_squared,
        1000.0,
    )
    upper_radius_squared = (
        2.5 * (upper_effective_quantum_number_squared / effective_charge) ** 2
    )

    lower_effective_quantum_number_squared = np.full(species_code.shape, 25.0)
    has_bound_lower_level = (ionization_potential_cm > 0) & (
        (ionization_potential_cm - lower_level_excitation_cm) > 0
    )
    lower_level_gap_cm = np.where(
        has_bound_lower_level,
        ionization_potential_cm - lower_level_excitation_cm,
        1.0,
    )
    lower_effective_quantum_number_squared = np.where(
        has_bound_lower_level,
        109737.31 * effective_charge * effective_charge / lower_level_gap_cm,
        lower_effective_quantum_number_squared,
    )
    lower_effective_quantum_number_squared = np.minimum(
        lower_effective_quantum_number_squared,
        1000.0,
    )
    lower_radius_squared = (
        2.5 * (lower_effective_quantum_number_squared / effective_charge) ** 2
    )

    iso_electronic_sequence = atomic_number - effective_charge.astype(np.int64) + 1
    is_transition_metal_sequence = (iso_electronic_sequence > 20) & (
        iso_electronic_sequence < 29
    )
    upper_radius_squared = np.where(
        is_transition_metal_sequence,
        (45.0 - iso_electronic_sequence) / effective_charge,
        upper_radius_squared,
    )
    lower_radius_squared = np.where(
        is_transition_metal_sequence, 0.0, lower_radius_squared
    )

    upper_radius_squared = np.where(
        upper_radius_squared < lower_radius_squared,
        2.0 * lower_radius_squared,
        upper_radius_squared,
    )
    radius_squared_difference = upper_radius_squared - lower_radius_squared
    van_der_waals_gamma = np.where(
        radius_squared_difference > 0,
        4.5e-9
        * np.power(
            np.where(radius_squared_difference > 0, radius_squared_difference, 1.0),
            0.4,
        ),
        1.0e-9,
    )
    van_der_waals_log = np.log10(van_der_waals_gamma)

    van_der_waals_log = np.where(
        species_code >= 100.0, np.log10(1.0e-7 / effective_charge), van_der_waals_log
    )
    return van_der_waals_log


# Structure-of-arrays catalog bundle.


@dataclass
class LineCatalog:
    """Structure-of-arrays of per-line invariant fields.

    All arrays are length n_lines, ordered by ``sort`` (default: catalog parse
    order). Wavelengths are in nm, energies in cm^-1, and damping values use the
    normalized ``gamma_linear / (4*pi*nu)`` convention.
    """

    wavelength_nm: np.ndarray
    index_wavelength_nm: np.ndarray
    oscillator_strength: np.ndarray
    log_oscillator_strength: np.ndarray
    lower_excitation_cm: np.ndarray
    radiative_damping: np.ndarray
    stark_damping: np.ndarray
    van_der_waals_damping: np.ndarray
    raw_radiative_damping_log: np.ndarray
    raw_stark_damping_log: np.ndarray
    raw_van_der_waals_damping_log: np.ndarray
    ion_stage: np.ndarray
    atomic_number: np.ndarray
    species_code: np.ndarray
    line_size: np.ndarray  # raw line-size routing nibble
    line_type: np.ndarray  # 0 metal, -1 H, -2 D, -3 HeI, -4 He3, -6 HeII, 1 AUT, ...
    lower_principal_quantum_number: np.ndarray
    upper_principal_quantum_number: np.ndarray
    classical_line_strength: np.ndarray
    # bookkeeping for line-type partitioning (segment start offsets into the SoA)
    type_segments: dict  # {line_type: (start, stop)} valid only when sort != "catalog"
    grid: Grid
    sort: str

    def __len__(self) -> int:
        return int(self.wavelength_nm.size)

    _FIELDS = (
        "wavelength_nm",
        "index_wavelength_nm",
        "oscillator_strength",
        "log_oscillator_strength",
        "lower_excitation_cm",
        "radiative_damping",
        "stark_damping",
        "van_der_waals_damping",
        "raw_radiative_damping_log",
        "raw_stark_damping_log",
        "raw_van_der_waals_damping_log",
        "ion_stage",
        "atomic_number",
        "species_code",
        "line_size",
        "line_type",
        "lower_principal_quantum_number",
        "upper_principal_quantum_number",
        "classical_line_strength",
    )

    def to_npz(self, path: Path) -> None:
        """Write the SoA to a fast uncompressed .npz (zero-decompress reload)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(
            schema=CACHE_SCHEMA,
            logic_version=CACHE_LOGIC_VERSION,
            start_wavelength_nm=self.grid.start_wavelength_nm,
            end_wavelength_nm=self.grid.end_wavelength_nm,
            resolution=self.grid.resolution,
            sort=self.sort,
            type_segments={
                str(line_type): list(segment_bounds)
                for line_type, segment_bounds in self.type_segments.items()
            },
        )
        arrays = {field: getattr(self, field) for field in self._FIELDS}
        # store metadata as a json blob in a 0-d byte array
        arrays["__meta__"] = np.frombuffer(
            json.dumps(meta).encode("utf-8"), dtype=np.uint8
        )
        np.savez(path, **arrays)

    @classmethod
    def from_npz(cls, path: Path) -> "LineCatalog":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(bytes(data["__meta__"]).decode("utf-8"))
            start_wavelength_nm = meta["start_wavelength_nm"]
            end_wavelength_nm = meta["end_wavelength_nm"]
            grid = Grid(start_wavelength_nm, end_wavelength_nm, meta["resolution"])
            type_segments = {
                int(line_type): tuple(segment_bounds)
                for line_type, segment_bounds in meta["type_segments"].items()
            }
            fields = {}
            for field in cls._FIELDS:
                fields[field] = np.asarray(data[field])
        return cls(grid=grid, sort=meta["sort"], type_segments=type_segments, **fields)

    def to_torch(self, device=None, float_dtype=None, int_dtype=None) -> dict:
        """Return the SoA as torch tensors ready to push to ``device()``.

        Float fields default to fp32 (the precision-budget default, device.py);
        index/int fields to int64 (gather/scatter indices). Energies/wavelengths
        keep their f64 source values cast to the requested float dtype.
        """
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch not available")
        if device is None:
            from .device import device as _device

            device = _device()
        if float_dtype is None:
            from .device import DEFAULT_DTYPE

            float_dtype = DEFAULT_DTYPE
        if int_dtype is None:
            int_dtype = torch.int64
        float_fields = (
            "wavelength_nm",
            "index_wavelength_nm",
            "oscillator_strength",
            "log_oscillator_strength",
            "lower_excitation_cm",
            "radiative_damping",
            "stark_damping",
            "van_der_waals_damping",
            "raw_radiative_damping_log",
            "raw_stark_damping_log",
            "raw_van_der_waals_damping_log",
            "species_code",
            "classical_line_strength",
        )
        int_fields = (
            "ion_stage",
            "atomic_number",
            "line_size",
            "line_type",
            "lower_principal_quantum_number",
            "upper_principal_quantum_number",
        )
        tensor_fields = {}
        for field in float_fields:
            tensor_fields[field] = torch.as_tensor(
                getattr(self, field),
                dtype=float_dtype,
            ).to(device)
        for field in int_fields:
            tensor_fields[field] = torch.as_tensor(
                getattr(self, field),
                dtype=int_dtype,
            ).to(device)
        return tensor_fields


# Vectorized fixed-width parser.


def _parse_floats(block: np.ndarray) -> np.ndarray:
    """Parse a fixed-width byte column (shape [n, w]) into float64.

    Blank fields are physical zeros in the fixed-width catalog.  The parser
    strips and casts the whole column at once.
    """
    stripped_fields = np.char.strip(block.view(f"S{block.shape[1]}").ravel())
    parsed_values = np.zeros(stripped_fields.size, dtype=np.float64)
    nonempty_fields = stripped_fields != b""
    if nonempty_fields.any():
        parsed_values[nonempty_fields] = stripped_fields[nonempty_fields].astype(
            np.float64
        )
    return parsed_values


def _parse_ints(block: np.ndarray) -> np.ndarray:
    stripped_fields = np.char.strip(block.view(f"S{block.shape[1]}").ravel())
    parsed_values = np.zeros(stripped_fields.size, dtype=np.int64)
    nonempty_fields = stripped_fields != b""
    if nonempty_fields.any():
        candidate_fields = stripped_fields[nonempty_fields]
        numeric_fields = np.char.isdigit(np.char.lstrip(candidate_fields, b"-"))
        numeric_indices = np.nonzero(nonempty_fields)[0][numeric_fields]
        parsed_values[numeric_indices] = candidate_fields[numeric_fields].astype(
            np.int64
        )
    return parsed_values


_ATOMIC_PARSE_FIELDS = (
    "stored_wavelength_nm",
    "raw_log_oscillator_strength",
    "species_code",
    "first_energy_column_cm",
    "second_energy_column_cm",
    "radiative_damping_log",
    "stark_damping_log",
    "van_der_waals_damping_log",
    "lower_principal_quantum_number",
    "upper_principal_quantum_number",
    "primary_isotope_number",
    "primary_isotope_log_correction",
    "secondary_isotope_log_correction",
    "energy_shift_field",
    "isotope_shift_units",
    "line_size",
    "line_category_tag",
)


def _parse_atomic_source_catalog(path: Path) -> dict:
    """Vectorized fixed-width parse of the whole atomic source catalog.

    Returns the raw per-line fields BEFORE default-damping / normalization, so the
    window filter (which needs species/line-size metadata) can run first and
    defaults only fire on survivors. Column positions match the fixed-width
    source catalog.
    """

    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as arrays:
            return {name: np.asarray(arrays[name]) for name in _ATOMIC_PARSE_FIELDS}
    raw_bytes = np.fromfile(path, dtype=np.uint8)
    if raw_bytes.size % _RECORD_LEN != 0:
        # Fall back to ragged line handling only if the file is not uniform.
        return _parse_ragged_atomic_source_catalog(path)
    record_bytes = raw_bytes.reshape(-1, _RECORD_LEN)  # last column is '\n'

    def parse_float_field(start_byte: int, stop_byte: int) -> np.ndarray:
        return _parse_floats(record_bytes[:, start_byte:stop_byte])

    def parse_integer_field(start_byte: int, stop_byte: int) -> np.ndarray:
        return _parse_ints(record_bytes[:, start_byte:stop_byte])

    stored_wavelength_nm = parse_float_field(0, 11)
    raw_log_oscillator_strength = parse_float_field(11, 18)
    species_code = parse_float_field(18, 24)
    first_energy_column_cm = parse_float_field(24, 36)
    second_energy_column_cm = parse_float_field(52, 64)
    radiative_damping_log = parse_float_field(80, 86)
    stark_damping_log = parse_float_field(86, 92)
    van_der_waals_damping_log = parse_float_field(92, 98)
    lower_principal_quantum_number = parse_integer_field(102, 104)
    upper_principal_quantum_number = parse_integer_field(104, 106)
    primary_isotope_number = parse_integer_field(106, 109)
    primary_isotope_log_correction = parse_float_field(109, 115)
    secondary_isotope_log_correction = parse_float_field(118, 124)
    energy_shift_field = record_bytes[:, 124:134].view("S10").ravel()
    isotope_shift_units = parse_float_field(154, 160)
    line_size = parse_integer_field(140, 141)
    line_category_tag = np.char.strip(record_bytes[:, 141:144].view("S3").ravel())

    return dict(
        stored_wavelength_nm=stored_wavelength_nm,
        raw_log_oscillator_strength=raw_log_oscillator_strength,
        species_code=species_code,
        first_energy_column_cm=first_energy_column_cm,
        second_energy_column_cm=second_energy_column_cm,
        radiative_damping_log=radiative_damping_log,
        stark_damping_log=stark_damping_log,
        van_der_waals_damping_log=van_der_waals_damping_log,
        lower_principal_quantum_number=lower_principal_quantum_number,
        upper_principal_quantum_number=upper_principal_quantum_number,
        primary_isotope_number=primary_isotope_number,
        primary_isotope_log_correction=primary_isotope_log_correction,
        secondary_isotope_log_correction=secondary_isotope_log_correction,
        energy_shift_field=energy_shift_field,
        isotope_shift_units=isotope_shift_units,
        line_size=line_size,
        line_category_tag=line_category_tag,
    )


def _parse_ragged_atomic_source_catalog(
    path: Path,
) -> dict:  # pragma: no cover - rare path
    """Fallback for non-uniform-width catalogs (whitespace-tolerant via slicing).

    Pads/truncates each line to 160 chars then reuses the fixed-width parser.
    """
    with path.open("rb") as file_handle:
        source_lines = [line.rstrip(b"\n") for line in file_handle.readlines()]
    record_bytes = np.full((len(source_lines), _RECORD_LEN), ord(" "), dtype=np.uint8)
    for row_index, source_line in enumerate(source_lines):
        width = min(len(source_line), 160)
        record_bytes[row_index, :width] = np.frombuffer(
            source_line[:width], dtype=np.uint8
        )
    # write a temp uniform buffer and reuse the main path's column logic
    raw_bytes = record_bytes.reshape(-1)
    temp_path = path.parent / ".__engine_tmp_uniform.bin"
    raw_bytes.tofile(temp_path)
    try:
        parsed = _parse_atomic_source_catalog(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return parsed


def _parse_energy_shift_subfields(
    source_shift_field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Energy shifts from the two 5-byte source subfields as int*1e-3.

    Non-integer subfields fall back to zero through the digit mask below.
    """
    record_count = source_shift_field.size
    shift_fields = np.ascontiguousarray(source_shift_field).view("S10").ravel()
    # split into two contiguous 5-byte subcolumns
    shift_field_bytes = shift_fields.view(np.uint8).reshape(record_count, 10)
    lower_shift_field = np.ascontiguousarray(shift_field_bytes[:, 0:5])
    upper_shift_field = np.ascontiguousarray(shift_field_bytes[:, 5:10])

    def _parse_shift_column(block: np.ndarray) -> np.ndarray:
        stripped_fields = np.char.strip(block.view("S5").ravel())
        parsed_values = np.zeros(stripped_fields.size, dtype=np.float64)
        nonempty_fields = stripped_fields != b""
        if not nonempty_fields.any():
            return parsed_values
        candidate_fields = stripped_fields[nonempty_fields]
        numeric_fields = np.char.isdigit(np.char.lstrip(candidate_fields, b"+-"))
        numeric_indices = np.nonzero(nonempty_fields)[0][numeric_fields]
        if numeric_indices.size:
            parsed_values[numeric_indices] = (
                candidate_fields[numeric_fields].astype(np.int64).astype(np.float64)
            )
        return parsed_values

    first_energy_shift_cm = _parse_shift_column(lower_shift_field) * 1.0e-3
    second_energy_shift_cm = _parse_shift_column(upper_shift_field) * 1.0e-3
    return first_energy_shift_cm, second_energy_shift_cm


def _build_records(raw: dict, apply_iso_corr: bool = True) -> dict:
    """Apply unit conversions and default damping over the full raw catalog."""
    stored_wavelength_nm = raw["stored_wavelength_nm"]
    raw_log_oscillator_strength = raw["raw_log_oscillator_strength"]
    species_code = raw["species_code"]
    first_energy_column_cm = raw["first_energy_column_cm"]
    second_energy_column_cm = raw["second_energy_column_cm"]

    isotope_log_correction = (
        raw["primary_isotope_log_correction"] + raw["secondary_isotope_log_correction"]
        if apply_iso_corr
        else 0.0
    )
    log_oscillator_strength = raw_log_oscillator_strength + isotope_log_correction

    abs_first_energy_column_cm = np.abs(first_energy_column_cm)
    abs_second_energy_column_cm = np.abs(second_energy_column_cm)
    lower_excitation_cm = np.minimum(
        abs_first_energy_column_cm, abs_second_energy_column_cm
    )
    upper_excitation_cm = np.maximum(
        abs_first_energy_column_cm, abs_second_energy_column_cm
    )

    first_energy_shift_cm, second_energy_shift_cm = _parse_energy_shift_subfields(
        raw["energy_shift_field"]
    )
    shifted_energy_difference_cm = np.abs(
        (abs_second_energy_column_cm + second_energy_shift_cm)
        - (abs_first_energy_column_cm + first_energy_shift_cm)
    )
    isotope_shift_nm = raw["isotope_shift_units"] * 1.0e-4
    with np.errstate(divide="ignore", invalid="ignore"):
        wavelength_from_energy_nm = np.where(
            shifted_energy_difference_cm > 0,
            1.0e7
            / np.where(
                shifted_energy_difference_cm > 0, shifted_energy_difference_cm, 1.0
            )
            + isotope_shift_nm,
            0.0,
        )
    wavelength_nm = np.where(
        shifted_energy_difference_cm > 0,
        wavelength_from_energy_nm,
        stored_wavelength_nm,
    )
    index_wavelength_nm = wavelength_nm.copy()

    raw_radiative_damping_log = raw["radiative_damping_log"]
    raw_stark_damping_log = raw["stark_damping_log"]
    raw_van_der_waals_damping_log = raw["van_der_waals_damping_log"]

    radiative_damping = np.where(
        raw_radiative_damping_log != 0.0,
        np.power(10.0, raw_radiative_damping_log),
        0.0,
    )
    stark_damping = np.where(
        raw_stark_damping_log != 0.0,
        np.power(10.0, raw_stark_damping_log),
        0.0,
    )
    van_der_waals_damping = np.where(
        raw_van_der_waals_damping_log != 0.0,
        np.power(10.0, raw_van_der_waals_damping_log),
        0.0,
    )

    need_radiative_default = radiative_damping == 0.0
    radiative_damping = np.where(
        need_radiative_default,
        2.223e13 / (wavelength_nm**2),
        radiative_damping,
    )

    need_stark_default = stark_damping == 0.0
    if need_stark_default.any():
        stark_default_log = _default_stark_log(
            species_code[need_stark_default],
            lower_excitation_cm[need_stark_default],
            upper_excitation_cm[need_stark_default],
        )
        stark_damping = stark_damping.copy()
        stark_damping[need_stark_default] = np.where(
            stark_default_log != 0.0,
            np.power(10.0, stark_default_log),
            0.0,
        )

    need_van_der_waals_default = van_der_waals_damping == 0.0
    if need_van_der_waals_default.any():
        van_der_waals_default_log = _default_van_der_waals_log(
            species_code[need_van_der_waals_default],
            lower_excitation_cm[need_van_der_waals_default],
            upper_excitation_cm[need_van_der_waals_default],
        )
        van_der_waals_damping = van_der_waals_damping.copy()
        van_der_waals_damping[need_van_der_waals_default] = np.where(
            van_der_waals_default_log != 0.0,
            np.power(10.0, van_der_waals_default_log),
            0.0,
        )

    line_frequency_hz = LIGHT_SPEED_NM_PER_S / np.maximum(wavelength_nm, 1e-12)
    damping_normalization = 12.5664 * line_frequency_hz
    radiative_damping = radiative_damping / damping_normalization
    stark_damping = stark_damping / damping_normalization
    van_der_waals_damping = van_der_waals_damping / damping_normalization

    atomic_number = (species_code + 1e-6).astype(np.int64)
    ion_stage_fraction = species_code - atomic_number
    ion_stage = np.where(
        ion_stage_fraction > 1e-6,
        np.rint(ion_stage_fraction * 100.0).astype(np.int64) + 1,
        1,
    )

    line_category_tag = raw["line_category_tag"]
    primary_isotope_number = raw["primary_isotope_number"]
    line_type = np.zeros(species_code.shape, dtype=np.int64)
    line_type = np.where(line_category_tag == b"AUT", 1, line_type)
    line_type = np.where(line_category_tag == b"COR", 2, line_type)
    line_type = np.where(line_category_tag == b"PRD", 3, line_type)
    is_standard_line = (
        (line_category_tag != b"AUT")
        & (line_category_tag != b"COR")
        & (line_category_tag != b"PRD")
    )
    is_neutral_hydrogen = (atomic_number == 1) & (ion_stage == 1)
    is_neutral_helium = (atomic_number == 2) & (ion_stage == 1)
    is_ionized_helium = (atomic_number == 2) & (ion_stage == 2)
    line_type = np.where(
        is_standard_line & is_neutral_hydrogen & (primary_isotope_number == 2),
        -2,
        line_type,
    )
    line_type = np.where(
        is_standard_line & is_neutral_hydrogen & (primary_isotope_number != 2),
        -1,
        line_type,
    )
    line_type = np.where(
        is_standard_line & is_neutral_helium & (primary_isotope_number == 3),
        -4,
        line_type,
    )
    line_type = np.where(
        is_standard_line & is_neutral_helium & (primary_isotope_number != 3),
        -3,
        line_type,
    )
    line_type = np.where(is_standard_line & is_ionized_helium, -6, line_type)

    return dict(
        wavelength_nm=wavelength_nm,
        index_wavelength_nm=index_wavelength_nm,
        log_oscillator_strength=log_oscillator_strength,
        oscillator_strength=np.power(10.0, log_oscillator_strength),
        lower_excitation_cm=lower_excitation_cm,
        radiative_damping=radiative_damping,
        stark_damping=stark_damping,
        van_der_waals_damping=van_der_waals_damping,
        raw_radiative_damping_log=raw_radiative_damping_log,
        raw_stark_damping_log=raw_stark_damping_log,
        raw_van_der_waals_damping_log=raw_van_der_waals_damping_log,
        ion_stage=ion_stage,
        atomic_number=atomic_number,
        species_code=species_code,
        line_type=line_type,
        lower_principal_quantum_number=raw["lower_principal_quantum_number"].astype(
            np.int64
        ),
        upper_principal_quantum_number=raw["upper_principal_quantum_number"].astype(
            np.int64
        ),
        line_size=raw["line_size"].astype(np.int64),
    )


def _line_window_mask(
    catalog_records: dict,
    start_wavelength_nm: float,
    end_wavelength_nm: float,
) -> np.ndarray:
    """Per-line wavelength-window filter, vectorized."""
    wavelength_nm = catalog_records["wavelength_nm"]
    line_type = catalog_records["line_type"]
    line_size = catalog_records["line_size"]
    species_code = catalog_records["species_code"]
    atomic_number = catalog_records["atomic_number"]

    red_window_margin_scale = (
        1.0 if start_wavelength_nm <= 500.0 else start_wavelength_nm / 500.0
    )
    line_size_code = np.where(line_size > 0, line_size, 0).astype(np.int64)
    margin_class = np.minimum(8 - line_size_code, 7).astype(np.int64)
    is_hydrogen_element = atomic_number == 1
    uses_hydrogen_margin = (
        np.isclose(species_code, 1.0, rtol=0.0, atol=1e-6)
        | np.isin(line_type, np.array([-1, -2]))
        | is_hydrogen_element
    )
    margin_class = np.where(uses_hydrogen_margin, 1, margin_class)
    margin_class = np.clip(margin_class, 1, 7)
    margin_nm = _LINE_WINDOW_MARGINS_NM[margin_class - 1] * red_window_margin_scale
    return (wavelength_nm >= (start_wavelength_nm - margin_nm)) & (
        wavelength_nm <= (end_wavelength_nm + margin_nm)
    )


def _assemble_catalog(
    catalog_records: dict,
    window_mask: np.ndarray,
    grid_obj: Grid,
    wavelength_grid: np.ndarray,
    sort: str,
) -> LineCatalog:
    """Build the SoA from the masked full-catalog fields, with grid indices + sort."""
    selected = np.nonzero(window_mask)[0]
    wavelength_nm = catalog_records["wavelength_nm"][selected]
    index_wavelength_nm = catalog_records["index_wavelength_nm"][selected]
    log_oscillator_strength = catalog_records["log_oscillator_strength"][selected]
    oscillator_strength = catalog_records["oscillator_strength"][selected]
    lower_excitation_cm = catalog_records["lower_excitation_cm"][selected]
    radiative_damping = catalog_records["radiative_damping"][selected]
    stark_damping = catalog_records["stark_damping"][selected]
    van_der_waals_damping = catalog_records["van_der_waals_damping"][selected]
    raw_radiative_damping_log = catalog_records["raw_radiative_damping_log"][selected]
    raw_stark_damping_log = catalog_records["raw_stark_damping_log"][selected]
    raw_van_der_waals_damping_log = catalog_records["raw_van_der_waals_damping_log"][
        selected
    ]
    ion_stage = catalog_records["ion_stage"][selected]
    atomic_number = catalog_records["atomic_number"][selected]
    species_code = catalog_records["species_code"][selected]
    line_size = catalog_records["line_size"][selected]
    line_type = catalog_records["line_type"][selected]
    lower_principal_quantum_number = catalog_records["lower_principal_quantum_number"][
        selected
    ]
    upper_principal_quantum_number = catalog_records["upper_principal_quantum_number"][
        selected
    ]

    line_center_index_0based = _center_indices(wavelength_grid, index_wavelength_nm)

    line_frequency_hz = LIGHT_SPEED_NM_PER_S / np.maximum(index_wavelength_nm, 1e-30)
    classical_line_strength = (
        CLASSICAL_LINE_STRENGTH_COEFFICIENT * oscillator_strength / line_frequency_hz
    )

    type_segments: dict = {}
    if sort == "catalog":
        order = np.arange(wavelength_nm.size)
    elif sort == "type_center":
        # stable sort by (line_type, center index) -> contiguous type segments,
        # center-pixel order within each (scatter-target locality).
        order = np.lexsort((line_center_index_0based, line_type))
    else:
        raise ValueError(f"unknown sort={sort!r}")

    def apply_sort_order(field_values):
        return field_values[order]

    wavelength_nm, index_wavelength_nm = map(
        apply_sort_order,
        (wavelength_nm, index_wavelength_nm),
    )
    log_oscillator_strength, oscillator_strength = map(
        apply_sort_order,
        (log_oscillator_strength, oscillator_strength),
    )
    lower_excitation_cm = apply_sort_order(lower_excitation_cm)
    radiative_damping, stark_damping, van_der_waals_damping = map(
        apply_sort_order,
        (radiative_damping, stark_damping, van_der_waals_damping),
    )
    raw_radiative_damping_log, raw_stark_damping_log, raw_van_der_waals_damping_log = (
        map(
            apply_sort_order,
            (
                raw_radiative_damping_log,
                raw_stark_damping_log,
                raw_van_der_waals_damping_log,
            ),
        )
    )
    ion_stage, atomic_number, species_code, line_size, line_type = map(
        apply_sort_order,
        (ion_stage, atomic_number, species_code, line_size, line_type),
    )
    lower_principal_quantum_number, upper_principal_quantum_number = map(
        apply_sort_order,
        (lower_principal_quantum_number, upper_principal_quantum_number),
    )
    line_center_index_0based, classical_line_strength = map(
        apply_sort_order,
        (line_center_index_0based, classical_line_strength),
    )

    if sort == "type_center":
        types = np.unique(line_type)
        for line_type_value in types:
            where = np.nonzero(line_type == line_type_value)[0]
            type_segments[int(line_type_value)] = (int(where[0]), int(where[-1] + 1))

    return LineCatalog(
        wavelength_nm=wavelength_nm,
        index_wavelength_nm=index_wavelength_nm,
        oscillator_strength=oscillator_strength,
        log_oscillator_strength=log_oscillator_strength,
        lower_excitation_cm=lower_excitation_cm,
        radiative_damping=radiative_damping,
        stark_damping=stark_damping,
        van_der_waals_damping=van_der_waals_damping,
        raw_radiative_damping_log=raw_radiative_damping_log,
        raw_stark_damping_log=raw_stark_damping_log,
        raw_van_der_waals_damping_log=raw_van_der_waals_damping_log,
        ion_stage=ion_stage,
        atomic_number=atomic_number,
        species_code=species_code,
        line_size=line_size,
        line_type=line_type,
        lower_principal_quantum_number=lower_principal_quantum_number,
        upper_principal_quantum_number=upper_principal_quantum_number,
        classical_line_strength=classical_line_strength,
        type_segments=type_segments,
        grid=grid_obj,
        sort=sort,
    )


# Public API.


def parse_catalog(
    grid: Grid,
    catalog_path: Path | None = None,
    sort: str = "catalog",
    apply_iso_corr: bool = True,
) -> LineCatalog:
    """Parse the external atomic source catalog into the SoA bundle."""
    catalog_path = (
        Path(catalog_path)
        if catalog_path is not None
        else _default_atomic_source_catalog()
    )
    source_fields = _parse_atomic_source_catalog(catalog_path)
    catalog_records = _build_records(source_fields, apply_iso_corr=apply_iso_corr)
    window_mask = _line_window_mask(
        catalog_records,
        grid.start_wavelength_nm,
        grid.end_wavelength_nm,
    )
    wavelength_grid = grid.build()
    return _assemble_catalog(catalog_records, window_mask, grid, wavelength_grid, sort)


def _cache_key(catalog_path: Path, grid: Grid, sort: str, apply_iso_corr: bool) -> str:
    source_stat = catalog_path.stat()
    payload = {
        "schema": CACHE_SCHEMA,
        "logic_version": CACHE_LOGIC_VERSION,
        "source": str(catalog_path.resolve()),
        "size": int(source_stat.st_size),
        "mtime_ns": int(
            getattr(
                source_stat,
                "st_mtime_ns",
                int(source_stat.st_mtime * 1e9),
            )
        ),
        "start_wavelength_nm": float(grid.start_wavelength_nm),
        "end_wavelength_nm": float(grid.end_wavelength_nm),
        "resolution": float(grid.resolution),
        "sort": sort,
        "iso_corr": bool(apply_iso_corr),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:24]


def _default_cache_dir() -> Path:
    return runtime_paths.PACKAGE_CACHE_ROOT / "atomic_lines"


def load_catalog(
    window: tuple[float, float],
    grid: float | Grid,
    catalog_path: Path | None = None,
    cache_dir: Optional[Path] = None,
    sort: str = "catalog",
    apply_iso_corr: bool = True,
    rebuild: bool = False,
) -> LineCatalog:
    """Load the atomic line catalog for a wavelength window and grid."""
    start_wavelength_nm, end_wavelength_nm = float(window[0]), float(window[1])
    grid_obj = (
        grid
        if isinstance(grid, Grid)
        else Grid(start_wavelength_nm, end_wavelength_nm, float(grid))
    )
    # keep the grid window consistent with the requested window
    grid_obj = Grid(start_wavelength_nm, end_wavelength_nm, grid_obj.resolution)

    catalog_path = (
        Path(catalog_path)
        if catalog_path is not None
        else _default_atomic_source_catalog()
    )
    cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    key = _cache_key(catalog_path, grid_obj, sort, apply_iso_corr)
    cache_path = cache_dir / f"atomic_lines_{key}.npz"

    if not rebuild and cache_path.exists():
        try:
            return LineCatalog.from_npz(cache_path)
        except Exception:
            pass  # corrupt/stale -> reparse

    catalog = parse_catalog(
        grid_obj,
        catalog_path=catalog_path,
        sort=sort,
        apply_iso_corr=apply_iso_corr,
    )
    try:
        catalog.to_npz(cache_path)
    except Exception:  # pragma: no cover - caching is best-effort
        pass
    return catalog


__all__ = ["Grid", "LineCatalog", "parse_catalog", "load_catalog"]
