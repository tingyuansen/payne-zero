"""Binary line-catalog readers used by atmosphere line absorption."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SelectedLineCatalog:
    """Compact selected-line records written by line selection."""

    packed_wavelength_index: np.ndarray
    packed_species_slot: np.ndarray
    lower_excitation_index: np.ndarray
    log_strength_index: np.ndarray
    radiative_damping_index: np.ndarray
    stark_damping_index: np.ndarray
    van_der_waals_damping_index: np.ndarray

    @property
    def line_count(self) -> int:
        return int(self.packed_wavelength_index.size)


@dataclass(frozen=True)
class LineTransitionCatalog:
    """Detailed transition records consumed by the XLINOP line-opacity path."""

    vacuum_wavelength_nm: np.ndarray
    lower_excitation_cm: np.ndarray
    oscillator_strength: np.ndarray
    lower_hydrogen_level: np.ndarray
    upper_hydrogen_level: np.ndarray
    packed_species_slot: np.ndarray
    line_type: np.ndarray
    hydrogen_continuum_selector_index: np.ndarray
    continuum_species_slot: np.ndarray
    radiative_damping: np.ndarray
    stark_damping: np.ndarray
    van_der_waals_damping: np.ndarray
    packed_wavelength_index: np.ndarray
    line_limit: np.ndarray

    @property
    def line_count(self) -> int:
        return int(self.packed_wavelength_index.size)


def _empty_selected_catalog() -> SelectedLineCatalog:
    return SelectedLineCatalog(
        packed_wavelength_index=np.zeros(0, dtype=np.int32),
        packed_species_slot=np.zeros(0, dtype=np.int16),
        lower_excitation_index=np.zeros(0, dtype=np.int16),
        log_strength_index=np.zeros(0, dtype=np.int16),
        radiative_damping_index=np.zeros(0, dtype=np.int16),
        stark_damping_index=np.zeros(0, dtype=np.int16),
        van_der_waals_damping_index=np.zeros(0, dtype=np.int16),
    )


def _empty_transition_catalog() -> LineTransitionCatalog:
    return LineTransitionCatalog(
        vacuum_wavelength_nm=np.zeros(0, dtype=np.float64),
        lower_excitation_cm=np.zeros(0, dtype=np.float64),
        oscillator_strength=np.zeros(0, dtype=np.float64),
        lower_hydrogen_level=np.zeros(0, dtype=np.int32),
        upper_hydrogen_level=np.zeros(0, dtype=np.int32),
        packed_species_slot=np.zeros(0, dtype=np.int32),
        line_type=np.zeros(0, dtype=np.int32),
        hydrogen_continuum_selector_index=np.zeros(0, dtype=np.int32),
        continuum_species_slot=np.zeros(0, dtype=np.int32),
        radiative_damping=np.zeros(0, dtype=np.float64),
        stark_damping=np.zeros(0, dtype=np.float64),
        van_der_waals_damping=np.zeros(0, dtype=np.float64),
        packed_wavelength_index=np.zeros(0, dtype=np.int32),
        line_limit=np.zeros(0, dtype=np.int32),
    )


def _decode_selected_halves(words: np.ndarray, *, swap_pairs: bool) -> np.ndarray:
    halves = np.ascontiguousarray(words, dtype=np.int32).view(np.int16).reshape(-1, 6)
    if swap_pairs:
        return halves[:, [1, 0, 3, 2, 5, 4]]
    return halves


def _selected_halfword_score(decoded: np.ndarray) -> int:
    if decoded.shape[1] != 6:
        return 0
    score = 0
    for column in (decoded[:, 2], decoded[:, 3], decoded[:, 4], decoded[:, 5]):
        score += int(np.count_nonzero((column > 0) & (column < 32768)))
    return score


def decode_selected_line_words(
    words: np.ndarray, *, detect_swapped_layout: bool = True
) -> SelectedLineCatalog:
    """Decode in-memory `(N, 4)` int32 selected-line words into a `SelectedLineCatalog`.

    Identical decoding to reading the same words from a unit-12 file, so the atmosphere
    solver can consume freshly generated selection words directly, with no disk round-trip.

    ``detect_swapped_layout`` guards against byte-swapped external files by decoding both
    ways and scoring; freshly generated words are always native, so the generator passes
    ``False`` and skips the second (1e8-row) decode + score -- a large cool-star saving.
    """

    words = np.ascontiguousarray(words, dtype=np.int32).reshape(-1, 4)
    if words.shape[0] == 0:
        return _empty_selected_catalog()

    packed_fields = words[:, 1:4]
    decoded = _decode_selected_halves(packed_fields, swap_pairs=False)
    if detect_swapped_layout:
        swapped = _decode_selected_halves(packed_fields, swap_pairs=True)
        if _selected_halfword_score(swapped) > _selected_halfword_score(decoded):
            decoded = swapped

    return SelectedLineCatalog(
        packed_wavelength_index=words[:, 0].astype(np.int32, copy=False),
        packed_species_slot=decoded[:, 0].astype(np.int16, copy=False),
        lower_excitation_index=decoded[:, 1].astype(np.int16, copy=False),
        log_strength_index=decoded[:, 2].astype(np.int16, copy=False),
        radiative_damping_index=decoded[:, 3].astype(np.int16, copy=False),
        stark_damping_index=decoded[:, 4].astype(np.int16, copy=False),
        van_der_waals_damping_index=decoded[:, 5].astype(np.int16, copy=False),
    )


def read_selected_line_catalog(path: Path | str) -> SelectedLineCatalog:
    """Read compact selected-line records from the historical binary format."""

    raw_words = np.fromfile(Path(path), dtype=np.int32)
    if raw_words.size == 0:
        return _empty_selected_catalog()
    if raw_words.size % 4 != 0:
        raise ValueError(
            f"Invalid selected-line word count {raw_words.size}; expected a multiple of 4."
        )
    return decode_selected_line_words(raw_words.reshape(-1, 4))


def _decode_transition_records_60(raw: bytes) -> LineTransitionCatalog:
    record_count = len(raw) // 60
    if record_count == 0:
        return _empty_transition_catalog()

    dtype = np.dtype(
        [
            ("vacuum_wavelength_nm", "<f8"),
            ("lower_excitation_cm", "<f4"),
            ("oscillator_strength", "<f4"),
            ("lower_hydrogen_level", "<i4"),
            ("upper_hydrogen_level", "<i4"),
            ("packed_species_slot", "<i4"),
            ("line_type", "<i4"),
            ("hydrogen_continuum_selector_index", "<i4"),
            ("continuum_species_slot", "<i4"),
            ("radiative_damping", "<f4"),
            ("stark_damping", "<f4"),
            ("van_der_waals_damping", "<f4"),
            ("packed_wavelength_index", "<i4"),
            ("line_limit", "<i4"),
        ]
    )
    records = np.frombuffer(raw[: record_count * 60], dtype=dtype)
    return LineTransitionCatalog(
        vacuum_wavelength_nm=records["vacuum_wavelength_nm"].astype(np.float64),
        lower_excitation_cm=records["lower_excitation_cm"].astype(np.float64),
        oscillator_strength=records["oscillator_strength"].astype(np.float64),
        lower_hydrogen_level=records["lower_hydrogen_level"].astype(np.int32),
        upper_hydrogen_level=records["upper_hydrogen_level"].astype(np.int32),
        packed_species_slot=records["packed_species_slot"].astype(np.int32),
        line_type=records["line_type"].astype(np.int32),
        hydrogen_continuum_selector_index=records[
            "hydrogen_continuum_selector_index"
        ].astype(np.int32),
        continuum_species_slot=records["continuum_species_slot"].astype(np.int32),
        radiative_damping=records["radiative_damping"].astype(np.float64),
        stark_damping=records["stark_damping"].astype(np.float64),
        van_der_waals_damping=records["van_der_waals_damping"].astype(np.float64),
        packed_wavelength_index=records["packed_wavelength_index"].astype(np.int32),
        line_limit=records["line_limit"].astype(np.int32),
    )


def _decode_transition_records_56(words: np.ndarray) -> LineTransitionCatalog:
    if words.ndim != 2 or words.shape[1] != 14:
        raise ValueError("Transition record words must have shape (N, 14).")
    float_words = words.view(np.float32)
    return LineTransitionCatalog(
        vacuum_wavelength_nm=float_words[:, 0].astype(np.float64),
        lower_excitation_cm=float_words[:, 1].astype(np.float64),
        oscillator_strength=float_words[:, 2].astype(np.float64),
        lower_hydrogen_level=words[:, 3].astype(np.int32),
        upper_hydrogen_level=words[:, 4].astype(np.int32),
        packed_species_slot=words[:, 5].astype(np.int32),
        line_type=words[:, 6].astype(np.int32),
        hydrogen_continuum_selector_index=words[:, 7].astype(np.int32),
        continuum_species_slot=words[:, 8].astype(np.int32),
        radiative_damping=float_words[:, 9].astype(np.float64),
        stark_damping=float_words[:, 10].astype(np.float64),
        van_der_waals_damping=float_words[:, 11].astype(np.float64),
        packed_wavelength_index=words[:, 12].astype(np.int32),
        line_limit=words[:, 13].astype(np.int32),
    )


def _extract_fortran_records_60(raw: bytes) -> bytes:
    records: list[bytes] = []
    position = 0
    while position + 8 <= len(raw):
        record_size = int.from_bytes(
            raw[position : position + 4], "little", signed=True
        )
        position += 4
        if record_size != 60:
            raise ValueError(
                f"Unexpected transition record size {record_size}; expected 60."
            )
        if position + record_size + 4 > len(raw):
            raise ValueError("Transition record exceeds file bounds.")
        records.append(raw[position : position + record_size])
        position += record_size
        tail_size = int.from_bytes(raw[position : position + 4], "little", signed=True)
        position += 4
        if tail_size != record_size:
            raise ValueError("Mismatched transition record markers.")
    if position != len(raw):
        raise ValueError("Trailing bytes in transition catalog.")
    return b"".join(records)


def _extract_fortran_records_56(raw: bytes, *, endian: str) -> np.ndarray:
    records: list[np.ndarray] = []
    position = 0
    marker_dtype = np.dtype(f"{endian}i4")
    word_dtype = np.dtype(f"{endian}i4")
    while position + 8 <= len(raw):
        record_size = int(
            np.frombuffer(raw, dtype=marker_dtype, count=1, offset=position)[0]
        )
        position += 4
        if record_size != 56 or position + record_size + 4 > len(raw):
            raise ValueError("Invalid 56-byte transition record marker.")
        record = np.frombuffer(raw, dtype=word_dtype, count=14, offset=position).astype(
            np.int32,
            copy=False,
        )
        position += record_size
        tail_size = int(
            np.frombuffer(raw, dtype=marker_dtype, count=1, offset=position)[0]
        )
        position += 4
        if tail_size != record_size:
            raise ValueError("Mismatched transition record markers.")
        records.append(record)
    if position != len(raw):
        raise ValueError("Trailing bytes in transition catalog.")
    if not records:
        return np.zeros((0, 14), dtype=np.int32)
    return np.vstack(records)


def _transition_catalog_looks_physical(catalog: LineTransitionCatalog) -> bool:
    if catalog.line_count == 0:
        return True
    wavelengths = catalog.vacuum_wavelength_nm
    return bool(np.isfinite(wavelengths).all() and float(np.nanmax(wavelengths)) > 0.0)


def _decode_transition_catalog(raw: bytes) -> LineTransitionCatalog:
    try:
        catalog = _decode_transition_records_60(_extract_fortran_records_60(raw))
        if _transition_catalog_looks_physical(catalog):
            return catalog
    except Exception:
        pass

    if len(raw) % 60 == 0:
        catalog = _decode_transition_records_60(raw)
        if _transition_catalog_looks_physical(catalog):
            return catalog

    if len(raw) % 56 == 0:
        for endian in ("<", ">"):
            words = np.frombuffer(raw, dtype=np.dtype(f"{endian}i4")).reshape(-1, 14)
            catalog = _decode_transition_records_56(words.astype(np.int32, copy=False))
            if _transition_catalog_looks_physical(catalog):
                return catalog

    for endian in ("<", ">"):
        try:
            words = _extract_fortran_records_56(raw, endian=endian)
        except Exception:
            continue
        catalog = _decode_transition_records_56(words)
        if _transition_catalog_looks_physical(catalog):
            return catalog

    raise ValueError("Unable to decode detailed line-transition catalog.")


_TRANSITION_CATALOG_FIELDS = (
    "vacuum_wavelength_nm",
    "lower_excitation_cm",
    "oscillator_strength",
    "lower_hydrogen_level",
    "upper_hydrogen_level",
    "packed_species_slot",
    "line_type",
    "hydrogen_continuum_selector_index",
    "continuum_species_slot",
    "radiative_damping",
    "stark_damping",
    "van_der_waals_damping",
    "packed_wavelength_index",
    "line_limit",
)


def read_line_transition_catalog(path: Path | str) -> LineTransitionCatalog:
    """Read detailed transition records.

    Canonical form: an ``.npz`` holding the decoded field arrays. The legacy
    historical binary stream is still decoded for provenance tooling.
    """

    catalog_path = Path(path)
    if catalog_path.suffix == ".npz":
        with np.load(catalog_path, allow_pickle=False) as arrays:
            return LineTransitionCatalog(
                **{
                    name: np.asarray(arrays[name])
                    for name in _TRANSITION_CATALOG_FIELDS
                }
            )
    raw = catalog_path.read_bytes()
    if not raw:
        return _empty_transition_catalog()
    return _decode_transition_catalog(raw)
