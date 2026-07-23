"""Window compiler for molecular line sources.

Compiles the parsed molecular bands (`molecules/<band>_parsed.npz`, in
manifest order) and the packed TiO/H2O records into the per-window line
arrays the molecular kernel consumes. Per-line arithmetic deliberately
follows the original scalar order so compiled windows are byte-stable;
results cache outside the tree keyed by source fingerprints.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .constants import (
    CLASSICAL_LINE_STRENGTH_COEFFICIENT,
    LIGHT_SPEED_NM_PER_S,
    NATURAL_LOG_10,
)

logger = logging.getLogger(__name__)

# Molecular source-catalog cache.
_MOLECULAR_CACHE_SCHEMA = 3  # bump when the compiled output format changes


def _molecular_cache_key(
    source_paths: Sequence[Path],
    start_wavelength_nm: float,
    end_wavelength_nm: float,
    resolution: float,
    wavelength_mode: str,
    extra: str = "",
) -> str:
    """Deterministic hex digest for molecular compilation inputs."""
    cache_payload: dict = {
        "schema": _MOLECULAR_CACHE_SCHEMA,
        "start_wavelength_nm": float(start_wavelength_nm),
        "end_wavelength_nm": float(end_wavelength_nm),
        "resolution": float(resolution),
        "wavelength_mode": wavelength_mode,
        "extra": extra,
    }
    # Source identity is part of the cache key so edits invalidate old arrays.
    source_file_fingerprints = []
    for source_path in sorted(source_paths):
        try:
            source_stat = source_path.stat()
            source_file_fingerprints.append(
                (
                    str(source_path.resolve()),
                    int(source_stat.st_size),
                    int(
                        getattr(
                            source_stat,
                            "st_mtime_ns",
                            int(source_stat.st_mtime * 1e9),
                        )
                    ),
                )
            )
        except OSError:
            source_file_fingerprints.append((str(source_path), 0, 0))
    cache_payload["files"] = source_file_fingerprints
    cache_payload_bytes = json.dumps(cache_payload, sort_keys=True).encode()
    return hashlib.blake2b(cache_payload_bytes, digest_size=12).hexdigest()


def _molecular_cache_dir(source_paths: Sequence[Path]) -> Path:
    """Return the user-writable molecular parser cache directory.

    Kept toggle: PAYNE_ZERO_SYNTHESIS_MOLECULAR_SOURCE_CACHE_DIR relocates the
    molecular-parser cache (useful on read-only or shared deployments).
    """
    override = os.environ.get("PAYNE_ZERO_SYNTHESIS_MOLECULAR_SOURCE_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return (
        Path(
            os.environ.get(
                "PAYNE_ZERO_SYNTHESIS_CACHE_DIR",
                str(Path.home() / ".cache" / "payne-zero-synthesis"),
            )
        ).expanduser()
        / "molecular_source"
    )


def _save_molecular_cache(cache_path: Path, data: Dict[str, np.ndarray]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **data)
    logger.info(
        "Saved molecular cache: %s (%d lines)",
        cache_path,
        len(data.get("center_index_1based", [])),
    )


def _load_molecular_cache(cache_path: Path) -> Optional[Dict[str, np.ndarray]]:
    if not cache_path.exists():
        return None
    if (
        os.environ.get("PAYNE_ZERO_SYNTHESIS_DISABLE_MOLECULAR_SOURCE_CACHE", "0")
        == "1"
    ):
        return None
    try:
        data = np.load(cache_path)
        result = {
            "center_index_1based": np.asarray(
                data["center_index_1based"], dtype=np.int32
            ),
            "classical_line_strength": np.asarray(
                data["classical_line_strength"], dtype=np.float32
            ),
            "species_code": np.asarray(data["species_code"], dtype=np.int16),
            "lower_excitation_cm": np.asarray(
                data["lower_excitation_cm"], dtype=np.float32
            ),
            "radiative_damping": np.asarray(
                data["radiative_damping"], dtype=np.float32
            ),
            "stark_damping": np.asarray(data["stark_damping"], dtype=np.float32),
            "van_der_waals_damping": np.asarray(
                data["van_der_waals_damping"], dtype=np.float32
            ),
            "margin_class": np.asarray(data["margin_class"], dtype=np.int16),
        }
        logger.info(
            "Loaded molecular cache: %s (%d lines)",
            cache_path,
            len(result["center_index_1based"]),
        )
        return result
    except Exception as exc:
        logger.warning("Failed to load molecular cache %s: %s", cache_path, exc)
        return None


# Vacuum-to-air wavelength correction used by packed molecular binaries.
def _vacair(wl_vac_nm: float) -> float:
    """Return air wavelength (nm) given vacuum wavelength (nm)."""
    waven = 1.0e7 / wl_vac_nm  # wavenumber cm^-1
    return wl_vac_nm / (
        1.0000834213 + 2406030.0 / (1.30e10 - waven**2) + 15997.0 / (3.89e9 - waven**2)
    )


def _build_airshift_table(table_size: int = 100_000) -> np.ndarray:
    """Pre-build airshift[IWL] table where IWL = int(wl_nm * 10 + 0.5).
    Index 0..1999 are zero (UV); 2000..n-1 hold (air-vac) shifts in nm."""
    wavelength_shift_nm = np.zeros(table_size, dtype=np.float64)
    for table_index in range(2000, table_size):
        vacuum_wavelength_nm = table_index * 0.1
        wavelength_shift_nm[table_index] = (
            _vacair(vacuum_wavelength_nm) - vacuum_wavelength_nm
        )
    return wavelength_shift_nm


# Molecular species dispatch.
# Key: (source_code, isotope_index) -> species and isotope log-weight fields.
# Fixed-width text source columns are decoded in `_parse_text_line()`.
# Wavelengths are kept in nm internally so the pipeline uses one unit.
_MOL_DISPATCH: Dict[Tuple[int, int], Tuple[int, int, int, float, float]] = {
    # H2
    (240, 1): (240, 1, 1, 0.0, -5.0),
    # HD
    (240, 2): (240, 1, 2, 0.0, -4.469),
    # CH
    (106, 12): (246, 1, 12, 0.0, -0.005),
    (106, 13): (246, 1, 13, 0.0, -1.955),
    # NH
    (114, 14): (252, 1, 14, 0.0, -0.002),
    (114, 15): (252, 1, 15, 0.0, -2.444),
    # OH
    (108, 16): (258, 1, 16, 0.0, -0.001),
    (108, 18): (258, 1, 18, 0.0, -2.690),
    # CO
    (608, 12): (276, 12, 16, -0.005, -0.001),
    (608, 13): (276, 13, 16, -1.955, -0.001),
    (608, 16): (276, 12, 16, -0.005, -0.001),  # source-code synonym
    (608, 17): (276, 12, 17, -0.005, -3.398),
    (608, 18): (276, 12, 18, -0.005, -2.690),
    # CN
    (607, 12): (270, 12, 14, -0.005, -0.002),
    (607, 13): (270, 13, 14, -1.955, -0.002),
    (607, 15): (270, 12, 15, -0.005, -2.444),
    # C2  (CODE 606 -> C2)
    (606, 12): (264, 12, 12, -0.005, -0.005),
    (606, 13): (264, 12, 13, -0.005, -1.955),
    (606, 33): (264, 13, 13, -1.955, -1.955),
    # MgH
    (112, 24): (300, 1, 24, 0.0, -0.105),
    (112, 25): (300, 1, 25, 0.0, -0.996),
    (112, 26): (300, 1, 26, 0.0, -0.947),
    # SiH
    (114, 28): (312, 1, 28, 0.0, -0.035),  # SiH when isotope_index=28,29,30
    (114, 29): (312, 1, 29, 0.0, -1.331),
    (114, 30): (312, 1, 30, 0.0, -1.516),
    # NaH
    (123, 23): (492, 1, 23, 0.0, 0.0),
    # KH
    (119, 39): (498, 39, 1, -0.030, 0.0),
    (119, 41): (498, 41, 1, -1.172, 0.0),
    # CaH
    (120, 40): (342, 40, 1, -0.013, 0.0),
    (120, 42): (342, 42, 1, -2.189, 0.0),
    (120, 43): (342, 43, 1, -2.870, 0.0),
    (120, 44): (342, 44, 1, -1.681, 0.0),
    (120, 46): (342, 46, 1, -4.398, 0.0),
    (120, 48): (342, 48, 1, -2.728, 0.0),
    # TiO
    (816, 46): (366, 16, 46, 0.0, -1.101),
    (816, 47): (366, 16, 47, 0.0, -1.138),
    (816, 48): (366, 16, 48, 0.0, -0.131),
    (816, 49): (366, 16, 49, 0.0, -1.259),
    (816, 50): (366, 16, 50, 0.0, -1.272),
    # VO
    (816, 51): (372, 16, 51, 0.0, -0.001),
    # CrH
    (124, 50): (432, 50, 1, -1.362, 0.0),
    (124, 52): (432, 52, 1, -0.077, 0.0),
    (124, 53): (432, 53, 1, -1.022, 0.0),
    (124, 54): (432, 54, 1, -1.626, 0.0),
    # FeH
    (156, 54): (444, 54, 1, -1.237, 0.0),
    (156, 56): (444, 56, 1, -0.038, 0.0),
    (156, 57): (444, 57, 1, -1.658, 0.0),
    (156, 58): (444, 58, 1, -2.553, 0.0),
    # Accepted alternate FeH source code.
    (126, 54): (444, 54, 1, -1.237, 0.0),
    (126, 56): (444, 56, 1, -0.038, 0.0),
    (126, 57): (444, 57, 1, -1.658, 0.0),
    (126, 58): (444, 58, 1, -2.553, 0.0),
    # AlO
    (813, 16): (324, 27, 16, 0.0, -0.001),
    (813, 17): (324, 27, 17, 0.0, -3.398),
    (813, 18): (324, 27, 18, 0.0, -2.690),
    # CoO
    (827, 16): (576, 59, 16, 0.0, 0.0),
    (827, 17): (576, 59, 17, 0.0, -3.398),
    (827, 18): (576, 59, 18, 0.0, -2.690),
    # SiO
    (814, 16): (330, 28, 16, -0.035, -0.001),
    (814, 17): (330, 29, 16, -1.328, -0.001),
    (814, 18): (330, 30, 16, -1.510, -0.001),
    (814, 28): (330, 28, 16, -0.035, -0.001),
    (814, 29): (330, 29, 16, -1.328, -0.001),
    (814, 30): (330, 30, 16, -1.510, -0.001),
    # MgO
    (812, 24): (306, 24, 16, 0.0, -0.105),
}

# Principal isotope rows used when the source code alone identifies a molecule.
_MOL_CODE_ONLY_DISPATCH: Dict[int, Tuple[int, int, int, float, float]] = {
    101: (240, 1, 1, 0.0, -5.0),  # H2
    106: (246, 1, 12, 0.0, -0.005),  # CH
    107: (252, 1, 14, 0.0, -0.002),  # NH
    108: (258, 1, 16, 0.0, -0.001),  # OH
    608: (276, 12, 16, -0.005, -0.001),  # CO
    607: (270, 12, 14, -0.005, -0.002),  # CN
    606: (264, 12, 12, -0.005, -0.005),  # C2
    112: (300, 1, 24, 0.0, -0.105),  # MgH
    113: (306, 1, 27, 0.0, 0.0),  # AlH
    114: (312, 1, 28, 0.0, -0.035),  # SiH
    111: (492, 1, 11, 0.0, 0.0),  # NaH
    119: (498, 39, 1, -0.030, 0.0),  # KH
    120: (342, 40, 1, -0.013, 0.0),  # CaH
    123: (426, 1, 23, 0.0, 0.0),  # VH
    124: (432, 52, 1, -0.077, 0.0),  # CrH
    126: (444, 56, 1, -0.038, 0.0),  # FeH alternate code
    156: (444, 56, 1, -0.038, 0.0),  # FeH in fehfx.dat
    822: (366, 48, 16, 0.0, -0.131),  # TiO
    816: (348, 16, 32, 0.0, 0.0),  # SO
    813: (324, 27, 16, 0.0, -0.001),  # AlO
    823: (372, 51, 16, 0.0, 0.0),  # VO
    827: (576, 59, 16, 0.0, 0.0),  # CoO
    814: (330, 28, 16, -0.035, -0.001),  # SiO
    812: (318, 24, 16, 0.0, -0.105),  # MgO
}


def _dispatch_molecule(
    source_code: int,
    isotope_index: int,
) -> Optional[Tuple[int, int, int, float, float]]:
    """Return molecular species metadata for a source-catalog line."""
    row = _MOL_DISPATCH.get((source_code, isotope_index))
    if row is not None:
        return row
    return _MOL_CODE_ONLY_DISPATCH.get(source_code)


# Molecular text line compiler.
def _parse_field(line: str, start: int, end: int, field_kind: str):
    """Parse one fixed-width field from a molecular text line."""
    field = line[start:end] if len(line) >= end else line[start:]
    field = field.strip()
    if not field:
        return 0 if field_kind == "i" else (0.0 if field_kind == "f" else "")
    if field_kind == "f":
        try:
            return float(field)
        except ValueError:
            return 0.0
    if field_kind == "i":
        try:
            return int(field)
        except ValueError:
            return 0
    return field


def _parse_text_line(line: str):
    """Parse one fixed-width molecular source record.

    Returns the stored wavelength, line strength metadata, energy columns,
    species code, labels, isotope index, and scaled radiative damping.

    The runtime reads the converted ``molecular_band_lines.npz`` instead of
    text, so nothing calls this at synthesis time; it is kept as the
    executable record of the fixed-width column layout the conversion used
    (provenance for the raw text payloads preserved in the source tree).
    """
    stored_wavelength_nm = _parse_field(line, 0, 10, "f")
    log_oscillator_strength = _parse_field(line, 10, 17, "f")
    lower_j = _parse_field(line, 17, 22, "f")
    first_energy_cm = _parse_field(line, 22, 32, "f")
    upper_j = _parse_field(line, 32, 37, "f")
    second_energy_cm = _parse_field(line, 37, 48, "f")
    source_code = _parse_field(line, 48, 52, "i")
    lower_label = _parse_field(line, 52, 60, "s")
    upper_label = _parse_field(line, 60, 68, "s")
    isotope_index = _parse_field(line, 68, 70, "i")
    radiative_damping_log_scaled = _parse_field(line, 70, 74, "i")
    return (
        stored_wavelength_nm,
        log_oscillator_strength,
        lower_j,
        first_energy_cm,
        upper_j,
        second_energy_cm,
        source_code,
        lower_label,
        upper_label,
        isotope_index,
        radiative_damping_log_scaled,
    )


def _molecular_text_wavelength_nm(
    stored_wavelength_nm: float,
    first_energy_cm: float,
    second_energy_cm: float,
    use_energy_level_wavelengths: bool,
) -> float:
    """Return the wavelength in nm used for a molecular text-source line."""
    if use_energy_level_wavelengths:
        energy_difference_cm = abs(abs(second_energy_cm) - abs(first_energy_cm))
        if energy_difference_cm > 0.0:
            return 1.0e7 / energy_difference_cm
    return abs(stored_wavelength_nm)


def _iterate_parsed_band(arrays: "np.lib.npyio.NpzFile", band: str):
    """Yield per-line tuples for one band group of the combined NPZ (file order)."""

    def rows():
        stored_wavelength_nm = arrays[f"{band}/stored_wavelength_nm"]
        log_oscillator_strength = arrays[f"{band}/log_oscillator_strength"]
        first_energy_cm = arrays[f"{band}/first_energy_cm"]
        second_energy_cm = arrays[f"{band}/second_energy_cm"]
        source_code = arrays[f"{band}/source_code"]
        isotope_index = arrays[f"{band}/isotope_index"]
        radiative_damping_log_scaled = arrays[f"{band}/radiative_damping_log_scaled"]
        upper_label_is_ground_state = arrays[f"{band}/upper_label_is_ground_state"]
        for row in range(stored_wavelength_nm.size):
            yield (
                float(stored_wavelength_nm[row]),
                float(log_oscillator_strength[row]),
                float(first_energy_cm[row]),
                float(second_energy_cm[row]),
                int(source_code[row]),
                bool(upper_label_is_ground_state[row]),
                int(isotope_index[row]),
                float(radiative_damping_log_scaled[row]),
            )

    return rows()


def _band_dispatch_arrays(source_code: np.ndarray, isotope_index: np.ndarray):
    """Vectorized `_dispatch_molecule` over integer keys (exact lookups)."""
    packed_keys = source_code.astype(np.int64) * 10_000 + isotope_index.astype(np.int64)
    unique_keys, inverse = np.unique(packed_keys, return_inverse=True)
    species_by_key = np.zeros(unique_keys.size, dtype=np.int64)
    weight_primary_by_key = np.zeros(unique_keys.size, dtype=np.float64)
    weight_secondary_by_key = np.zeros(unique_keys.size, dtype=np.float64)
    valid_by_key = np.zeros(unique_keys.size, dtype=np.bool_)
    for key_index, packed in enumerate(unique_keys.tolist()):
        row = _dispatch_molecule(packed // 10_000, packed % 10_000)
        if row is None:
            continue
        species_by_key[key_index] = row[0]
        weight_primary_by_key[key_index] = row[3]
        weight_secondary_by_key[key_index] = row[4]
        valid_by_key[key_index] = True
    return (
        species_by_key[inverse],
        weight_primary_by_key[inverse],
        weight_secondary_by_key[inverse],
        valid_by_key[inverse],
    )


def _get_compiled_band_kernel():
    """Numba mirror of the per-line compile loop (libm-exact, byte-stable)."""
    global _COMPILED_BAND_KERNEL
    if _COMPILED_BAND_KERNEL is not None:
        return _COMPILED_BAND_KERNEL
    import numba

    @numba.njit(cache=True)
    def kernel(
        stored_wavelength_nm,
        log_oscillator_strength,
        first_energy_cm,
        second_energy_cm,
        dispatch_species,
        dispatch_weight_primary,
        dispatch_weight_secondary,
        dispatch_valid,
        upper_label_is_ground_state,
        radiative_damping_log_scaled,
        use_energy_level_wavelengths,
        include_predicted_lines,
        window_min_nm,
        window_max_nm,
        log_grid_ratio,
        grid_origin_index,
        out_center_index,
        out_strength,
        out_species,
        out_excitation,
        out_radiative,
        out_stark,
        out_van_der_waals_damping,
    ):
        count = 0
        for row in range(stored_wavelength_nm.shape[0]):
            stored = stored_wavelength_nm[row]
            if abs(stored) == 0.0:
                continue
            first_energy = first_energy_cm[row]
            second_energy = second_energy_cm[row]
            if not include_predicted_lines and (
                first_energy < 0.0 or second_energy < 0.0
            ):
                continue
            wavelength_nm = abs(stored)
            if use_energy_level_wavelengths:
                energy_difference_cm = abs(abs(second_energy) - abs(first_energy))
                if energy_difference_cm > 0.0:
                    wavelength_nm = 1.0e7 / energy_difference_cm
            if wavelength_nm < window_min_nm:
                continue
            if wavelength_nm > window_max_nm:
                break
            if use_energy_level_wavelengths:
                file_wavelength_nm = abs(stored)
                if file_wavelength_nm > window_max_nm + 10.0 or (
                    file_wavelength_nm > 0.0
                    and file_wavelength_nm < window_min_nm - 10.0
                ):
                    continue
            if not dispatch_valid[row]:
                continue
            oscillator_strength = math.exp(
                (
                    log_oscillator_strength[row]
                    + dispatch_weight_primary[row]
                    + dispatch_weight_secondary[row]
                )
                * NATURAL_LOG_10
            )
            lower_excitation_cm = min(abs(first_energy), abs(second_energy))
            grid_position = math.log(max(wavelength_nm, 1e-30)) / log_grid_ratio + 0.5
            center_index_1based = int(grid_position) - grid_origin_index + 1
            line_frequency_hz = LIGHT_SPEED_NM_PER_S / max(wavelength_nm, 1e-30)
            classical_line_strength = (
                CLASSICAL_LINE_STRENGTH_COEFFICIENT
                * oscillator_strength
                / line_frequency_hz
            )
            line_frequency_4pi = line_frequency_hz * 12.5664
            radiative_gamma = 10.0 ** (radiative_damping_log_scaled[row] * 0.01)
            stark_gamma = 3.0e-5
            van_der_waals_gamma = 1.0e-7
            if upper_label_is_ground_state[row]:
                stark_gamma = 3.0e-8
                van_der_waals_gamma = 1.0e-8
            out_center_index[count] = center_index_1based
            out_strength[count] = classical_line_strength
            out_species[count] = dispatch_species[row]
            out_excitation[count] = lower_excitation_cm
            out_radiative[count] = radiative_gamma / line_frequency_4pi
            out_stark[count] = stark_gamma / line_frequency_4pi
            out_van_der_waals_damping[count] = van_der_waals_gamma / line_frequency_4pi
            count += 1
        return count

    _COMPILED_BAND_KERNEL = kernel
    return kernel


_COMPILED_BAND_KERNEL = None


def compile_molecular_text(
    band_catalog_path: Path,
    band_names: Sequence[str],
    start_wavelength_nm: float,
    end_wavelength_nm: float,
    resolution: float,
    use_energy_level_wavelengths: bool = False,
    include_predicted_lines: bool = False,
) -> Dict[str, np.ndarray]:
    """Compile molecular bands from the combined band catalog, in the given order."""
    cache_key = _molecular_cache_key(
        [band_catalog_path],
        start_wavelength_nm,
        end_wavelength_nm,
        resolution,
        "energy_levels" if use_energy_level_wavelengths else "stored",
        extra=f"text_predicted_{include_predicted_lines}_bands_{'_'.join(band_names)}",
    )
    cache_dir = _molecular_cache_dir([band_catalog_path])
    cache_path = cache_dir / f"molecular_text_{cache_key}.npz"
    cached = _load_molecular_cache(cache_path)
    if cached is not None:
        return cached

    ratio = 1.0 + 1.0 / resolution
    log_grid_ratio = math.log(ratio)
    grid_origin_index = math.floor(math.log(start_wavelength_nm) / log_grid_ratio)
    if math.exp(grid_origin_index * log_grid_ratio) < start_wavelength_nm:
        grid_origin_index += 1

    window_max_nm = end_wavelength_nm + 0.1
    window_min_nm = start_wavelength_nm - 0.01

    center_indices_1based: List[int] = []
    classical_line_strengths: List[float] = []
    species_codes: List[int] = []
    lower_excitation_cm_values: List[float] = []
    radiative_damping_values: List[float] = []
    stark_damping_values: List[float] = []
    van_der_waals_damping_values: List[float] = []
    margin_classes: List[int] = []

    band_arrays = np.load(band_catalog_path, allow_pickle=False)
    # The compiled band kernel is the production path; the scalar per-band loop
    # below remains the fallback when numba is unavailable (and the parity oracle
    # used during porting).
    try:
        band_kernel = _get_compiled_band_kernel()
    except ImportError:
        band_kernel = None
    if band_kernel is not None:
        for band in band_names:
            stored = np.ascontiguousarray(band_arrays[f"{band}/stored_wavelength_nm"])
            n_rows = stored.shape[0]
            dispatch_species, weight_primary, weight_secondary, dispatch_valid = (
                _band_dispatch_arrays(
                    band_arrays[f"{band}/source_code"],
                    band_arrays[f"{band}/isotope_index"],
                )
            )
            out_center = np.empty(n_rows, dtype=np.int64)
            out_strength = np.empty(n_rows, dtype=np.float64)
            out_species = np.empty(n_rows, dtype=np.int64)
            out_excitation = np.empty(n_rows, dtype=np.float64)
            out_radiative = np.empty(n_rows, dtype=np.float64)
            out_stark = np.empty(n_rows, dtype=np.float64)
            out_van_der_waals_damping = np.empty(n_rows, dtype=np.float64)
            count = band_kernel(
                stored,
                np.ascontiguousarray(band_arrays[f"{band}/log_oscillator_strength"]),
                np.ascontiguousarray(band_arrays[f"{band}/first_energy_cm"]),
                np.ascontiguousarray(band_arrays[f"{band}/second_energy_cm"]),
                np.ascontiguousarray(dispatch_species),
                np.ascontiguousarray(weight_primary),
                np.ascontiguousarray(weight_secondary),
                np.ascontiguousarray(dispatch_valid),
                np.ascontiguousarray(
                    band_arrays[f"{band}/upper_label_is_ground_state"]
                ),
                np.ascontiguousarray(
                    band_arrays[f"{band}/radiative_damping_log_scaled"]
                ),
                use_energy_level_wavelengths,
                include_predicted_lines,
                window_min_nm,
                window_max_nm,
                log_grid_ratio,
                grid_origin_index,
                out_center,
                out_strength,
                out_species,
                out_excitation,
                out_radiative,
                out_stark,
                out_van_der_waals_damping,
            )
            center_indices_1based.append(out_center[:count])
            classical_line_strengths.append(out_strength[:count])
            species_codes.append(out_species[:count])
            lower_excitation_cm_values.append(out_excitation[:count])
            radiative_damping_values.append(out_radiative[:count])
            stark_damping_values.append(out_stark[:count])
            van_der_waals_damping_values.append(out_van_der_waals_damping[:count])
        total = sum(int(chunk.size) for chunk in center_indices_1based)
        result = {
            "center_index_1based": np.concatenate(center_indices_1based).astype(
                np.int32
            )
            if total
            else np.zeros(0, np.int32),
            "classical_line_strength": np.concatenate(classical_line_strengths).astype(
                np.float32
            )
            if total
            else np.zeros(0, np.float32),
            "species_code": np.concatenate(species_codes).astype(np.int16)
            if total
            else np.zeros(0, np.int16),
            "lower_excitation_cm": np.concatenate(lower_excitation_cm_values).astype(
                np.float32
            )
            if total
            else np.zeros(0, np.float32),
            "radiative_damping": np.concatenate(radiative_damping_values).astype(
                np.float32
            )
            if total
            else np.zeros(0, np.float32),
            "stark_damping": np.concatenate(stark_damping_values).astype(np.float32)
            if total
            else np.zeros(0, np.float32),
            "van_der_waals_damping": np.concatenate(
                van_der_waals_damping_values
            ).astype(np.float32)
            if total
            else np.zeros(0, np.float32),
            "margin_class": np.full(total, 7, dtype=np.int16),
        }
        _save_molecular_cache(cache_path, result)
        return result

    for band in band_names:
        parsed_rows = _iterate_parsed_band(band_arrays, band)
        for parsed_row in parsed_rows:
            (
                stored_wavelength_nm,
                log_oscillator_strength,
                first_energy_cm,
                second_energy_cm,
                source_code,
                upper_label_is_ground_state,
                isotope_index,
                radiative_damping_log_scaled,
            ) = parsed_row

            if abs(stored_wavelength_nm) == 0.0:
                continue

            if not include_predicted_lines and (
                first_energy_cm < 0.0 or second_energy_cm < 0.0
            ):
                continue

            wavelength_nm = _molecular_text_wavelength_nm(
                stored_wavelength_nm,
                first_energy_cm,
                second_energy_cm,
                use_energy_level_wavelengths,
            )
            if wavelength_nm < window_min_nm:
                continue
            if wavelength_nm > window_max_nm:
                # Files are sorted by wavelength; once past the window we can stop.
                break

            # Some energy-derived wavelengths move far from their stored
            # file position; keep the broad source-file guard used by the
            # validated line bundles.
            if use_energy_level_wavelengths:
                file_wavelength_nm = abs(stored_wavelength_nm)
                if file_wavelength_nm > window_max_nm + 10.0 or (
                    file_wavelength_nm > 0.0
                    and file_wavelength_nm < window_min_nm - 10.0
                ):
                    continue

            molecular_species = _dispatch_molecule(source_code, isotope_index)
            if molecular_species is None:
                continue

            (
                species_code,
                _isotope_1,
                _isotope_2,
                isotope_weight_log_primary,
                isotope_weight_log_secondary,
            ) = molecular_species
            oscillator_strength = math.exp(
                (
                    log_oscillator_strength
                    + isotope_weight_log_primary
                    + isotope_weight_log_secondary
                )
                * NATURAL_LOG_10
            )
            lower_excitation_cm = min(abs(first_energy_cm), abs(second_energy_cm))

            grid_position = math.log(max(wavelength_nm, 1e-30)) / log_grid_ratio + 0.5
            center_index_1based = int(grid_position) - grid_origin_index + 1

            line_frequency_hz = LIGHT_SPEED_NM_PER_S / max(wavelength_nm, 1e-30)
            classical_line_strength = (
                CLASSICAL_LINE_STRENGTH_COEFFICIENT
                * oscillator_strength
                / line_frequency_hz
            )

            line_frequency_4pi = line_frequency_hz * 12.5664
            radiative_gamma = 10.0 ** (radiative_damping_log_scaled * 0.01)
            stark_gamma = 3.0e-5
            van_der_waals_gamma = 1.0e-7
            if upper_label_is_ground_state:
                stark_gamma = 3.0e-8
                van_der_waals_gamma = 1.0e-8

            normalized_radiative_gamma = radiative_gamma / line_frequency_4pi
            normalized_stark_gamma = stark_gamma / line_frequency_4pi
            normalized_van_der_waals_gamma = van_der_waals_gamma / line_frequency_4pi

            center_indices_1based.append(center_index_1based)
            classical_line_strengths.append(classical_line_strength)
            species_codes.append(species_code)
            lower_excitation_cm_values.append(lower_excitation_cm)
            radiative_damping_values.append(normalized_radiative_gamma)
            stark_damping_values.append(normalized_stark_gamma)
            van_der_waals_damping_values.append(normalized_van_der_waals_gamma)
            margin_classes.append(7)

    result = _compiled_molecular_arrays(
        center_indices_1based,
        classical_line_strengths,
        species_codes,
        lower_excitation_cm_values,
        radiative_damping_values,
        stark_damping_values,
        van_der_waals_damping_values,
        margin_classes,
    )
    _save_molecular_cache(cache_path, result)
    return result


# TiO Schwenke binary compiler.
# Packed records store wavelength, isotope, lower energy, line strength, and
# damping-table indices in 16 bytes.
_TIO_FINE_LOG_GRID = math.log(1.0 + 1.0 / 2_000_000.0)
_TIO_ISOTOPE_FRACTIONS = [0.0793, 0.0728, 0.7394, 0.0551, 0.0534]
_TIO_SPECIES_CODE = 366
_PACKED_LOG_TABLE_SIZE = 32768
_PACKED_LOG_TABLE_OFFSET = 16384


def _build_packed_log_table() -> np.ndarray:
    """Build the 32768-entry log-to-linear lookup table used by Schwenke/H2O readers."""
    table_index = np.arange(_PACKED_LOG_TABLE_SIZE, dtype=np.float64)
    return (10.0 ** ((table_index - _PACKED_LOG_TABLE_OFFSET) * 0.001)).astype(
        np.float32
    )


_PACKED_LOG_TABLE: Optional[np.ndarray] = None


def _get_packed_log_table() -> np.ndarray:
    global _PACKED_LOG_TABLE
    if _PACKED_LOG_TABLE is None:
        _PACKED_LOG_TABLE = _build_packed_log_table()
    return _PACKED_LOG_TABLE


def _load_packed_molecular_records(path: Path, record_dtype: np.dtype) -> np.ndarray:
    """Memory-map packed molecular records (.npy canonical, raw .bin legacy)."""
    if path.suffix == ".npy":
        records = np.load(path, mmap_mode="r")
        if records.dtype != record_dtype:
            records = records.view(record_dtype)
        return records
    return np.memmap(path, mode="r", dtype=record_dtype)


def compile_tio_schwenke(
    bin_path: Path,
    start_wavelength_nm: float,
    end_wavelength_nm: float,
    resolution: float,
    use_vacuum_wavelengths: bool = False,
) -> Dict[str, np.ndarray]:
    """Compile the Schwenke TiO packed-binary line list."""
    cache_key = _molecular_cache_key(
        [bin_path],
        start_wavelength_nm,
        end_wavelength_nm,
        resolution,
        "vacuum" if use_vacuum_wavelengths else "air",
        extra="tio_schwenke",
    )
    cache_dir = _molecular_cache_dir([bin_path])
    cache_path = cache_dir / f"molecular_tio_{cache_key}.npz"
    cached = _load_molecular_cache(cache_path)
    if cached is not None:
        return cached

    packed_log_table = _get_packed_log_table()
    ratio = 1.0 + 1.0 / resolution
    log_grid_ratio = math.log(ratio)
    grid_origin_index = math.floor(math.log(start_wavelength_nm) / log_grid_ratio)
    if math.exp(grid_origin_index * log_grid_ratio) < start_wavelength_nm:
        grid_origin_index += 1

    record_dtype = np.dtype(
        [
            ("wavelength_code", "<i4"),
            ("isotope_species_code", "<i2"),
            ("lower_energy_code", "<i2"),
            ("log_oscillator_strength_code", "<i2"),
            ("radiative_damping_code", "<i2"),
            ("stark_damping_code", "<i2"),
            ("van_der_waals_damping_code", "<i2"),
        ]
    )
    try:
        packed_records = _load_packed_molecular_records(bin_path, record_dtype)
    except OSError:
        return _compiled_molecular_arrays([], [], [], [], [], [], [], [])

    if packed_records.size == 0:
        return _compiled_molecular_arrays([], [], [], [], [], [], [], [])

    vacuum_wavelength_nm = np.exp(
        packed_records["wavelength_code"].astype(np.float64) * _TIO_FINE_LOG_GRID
    )
    if not use_vacuum_wavelengths:
        airshift = _build_airshift_table(60_000)
        airshift_index = np.clip(
            (vacuum_wavelength_nm * 10.0 + 0.5).astype(np.int64),
            0,
            len(airshift) - 1,
        )
        synthesis_wavelength_nm = vacuum_wavelength_nm + airshift[airshift_index]
    else:
        synthesis_wavelength_nm = vacuum_wavelength_nm

    isotope_index_by_record = (
        np.abs(packed_records["isotope_species_code"].astype(np.int32)) - 8949
    )
    mask = (
        (synthesis_wavelength_nm >= start_wavelength_nm - 1.0)
        & (synthesis_wavelength_nm <= end_wavelength_nm + 1.0)
        & (isotope_index_by_record >= 1)
        & (isotope_index_by_record <= 5)
    )
    selected_count = int(mask.sum())
    if selected_count == 0:
        return _compiled_molecular_arrays([], [], [], [], [], [], [], [])

    wavelength_nm_selected = synthesis_wavelength_nm[mask]
    isotope_index_selected = isotope_index_by_record[mask]
    lower_energy_index = np.clip(
        packed_records["lower_energy_code"][mask].astype(np.int32),
        0,
        _PACKED_LOG_TABLE_SIZE - 1,
    )
    oscillator_strength_index = np.clip(
        packed_records["log_oscillator_strength_code"][mask].astype(np.int32),
        0,
        _PACKED_LOG_TABLE_SIZE - 1,
    )
    radiative_damping_index = np.clip(
        packed_records["radiative_damping_code"][mask].astype(np.int32),
        0,
        _PACKED_LOG_TABLE_SIZE - 1,
    )

    line_frequency_hz = LIGHT_SPEED_NM_PER_S / wavelength_nm_selected
    line_frequency_4pi = line_frequency_hz * 12.5664

    isotope_fraction = np.array(_TIO_ISOTOPE_FRACTIONS, dtype=np.float64)[
        isotope_index_selected - 1
    ]
    classical_line_strength = (
        0.01502
        * packed_log_table[oscillator_strength_index]
        / line_frequency_hz
        * isotope_fraction
    )
    lower_excitation_cm = packed_log_table[lower_energy_index].astype(np.float64)

    normalized_radiative_gamma = (
        packed_log_table[radiative_damping_index] / line_frequency_4pi
    )
    normalized_stark_gamma = float(packed_log_table[1]) / line_frequency_4pi
    normalized_van_der_waals_gamma = float(packed_log_table[9384]) / line_frequency_4pi

    grid_position = np.floor(
        np.log(np.maximum(wavelength_nm_selected, 1e-30)) / log_grid_ratio + 0.5
    ).astype(np.int32)
    center_indices_1based = grid_position - grid_origin_index + 1

    species_codes = np.full(selected_count, _TIO_SPECIES_CODE, dtype=np.int32)
    margin_classes = np.full(selected_count, 7, dtype=np.int16)

    result = _compiled_molecular_arrays(
        center_indices_1based,
        classical_line_strength,
        species_codes,
        lower_excitation_cm,
        normalized_radiative_gamma,
        normalized_stark_gamma,
        normalized_van_der_waals_gamma,
        margin_classes,
    )
    _save_molecular_cache(cache_path, result)
    return result


# H2O Partridge-Schwenke binary compiler.
# Packed records store wavelength, lower energy, and line strength. The signs
# of the energy and strength fields encode the isotopologue.
_H2O_FINE_LOG_GRID = math.log(1.0 + 1.0 / 2_000_000.0)
_H2O_ISOTOPE_FRACTIONS = [0.9976, 0.0004, 0.0020, 0.00001]
_H2O_SPECIES_CODE = 534


def compile_h2o_partridge(
    bin_path: Path,
    start_wavelength_nm: float,
    end_wavelength_nm: float,
    resolution: float,
    use_vacuum_wavelengths: bool = False,
) -> Dict[str, np.ndarray]:
    """Compile the Partridge-Schwenke H2O packed-binary line list."""
    cache_key = _molecular_cache_key(
        [bin_path],
        start_wavelength_nm,
        end_wavelength_nm,
        resolution,
        "vacuum" if use_vacuum_wavelengths else "air",
        extra="h2o_partridge",
    )
    cache_dir = _molecular_cache_dir([bin_path])
    cache_path = cache_dir / f"molecular_h2o_{cache_key}.npz"
    cached = _load_molecular_cache(cache_path)
    if cached is not None:
        return cached

    packed_log_table = _get_packed_log_table()
    ratio = 1.0 + 1.0 / resolution
    log_grid_ratio = math.log(ratio)
    grid_origin_index = math.floor(math.log(start_wavelength_nm) / log_grid_ratio)
    if math.exp(grid_origin_index * log_grid_ratio) < start_wavelength_nm:
        grid_origin_index += 1

    record_dtype = np.dtype(
        [
            ("wavelength_code", "<i4"),
            ("signed_lower_energy_code", "<i2"),
            ("signed_log_oscillator_strength_code", "<i2"),
        ]
    )
    try:
        packed_records = _load_packed_molecular_records(bin_path, record_dtype)
    except OSError:
        return _compiled_molecular_arrays([], [], [], [], [], [], [], [])

    if packed_records.size == 0:
        return _compiled_molecular_arrays([], [], [], [], [], [], [], [])

    vacuum_wavelength_nm = np.exp(
        packed_records["wavelength_code"].astype(np.float64) * _H2O_FINE_LOG_GRID
    )
    vacuum_frequency_hz = LIGHT_SPEED_NM_PER_S / vacuum_wavelength_nm

    if not use_vacuum_wavelengths:
        airshift = _build_airshift_table(100_000)
        airshift_index = np.clip(
            (vacuum_wavelength_nm * 10.0 + 0.5).astype(np.int64),
            0,
            len(airshift) - 1,
        )
        synthesis_wavelength_nm = vacuum_wavelength_nm + airshift[airshift_index]
    else:
        synthesis_wavelength_nm = vacuum_wavelength_nm

    mask = (synthesis_wavelength_nm >= start_wavelength_nm - 1.0) & (
        synthesis_wavelength_nm <= end_wavelength_nm + 1.0
    )
    selected_count = int(mask.sum())
    if selected_count == 0:
        return _compiled_molecular_arrays([], [], [], [], [], [], [], [])

    signed_lower_energy_code = packed_records["signed_lower_energy_code"][mask].astype(
        np.int32
    )
    signed_oscillator_strength_code = packed_records[
        "signed_log_oscillator_strength_code"
    ][mask].astype(np.int32)
    wavelength_nm_selected = synthesis_wavelength_nm[mask]
    frequency_for_strength_hz = vacuum_frequency_hz[mask]

    isotope_index = np.where(
        (signed_lower_energy_code > 0) & (signed_oscillator_strength_code > 0),
        0,
        np.where(
            signed_lower_energy_code > 0,
            1,
            np.where(signed_oscillator_strength_code > 0, 2, 3),
        ),
    )
    lower_excitation_cm = np.abs(signed_lower_energy_code).astype(np.float64)
    oscillator_strength_index = np.clip(
        np.abs(signed_oscillator_strength_code),
        0,
        _PACKED_LOG_TABLE_SIZE - 1,
    )

    isotope_fraction = np.array(_H2O_ISOTOPE_FRACTIONS, dtype=np.float64)[isotope_index]
    line_frequency_4pi = frequency_for_strength_hz * 12.5664

    classical_line_strength = (
        0.01502
        * packed_log_table[oscillator_strength_index]
        / frequency_for_strength_hz
        * isotope_fraction
    )

    radiative_gamma = 2.223e13 / np.maximum(wavelength_nm_selected, 1e-6) ** 2 * 0.001
    normalized_radiative_gamma = radiative_gamma / line_frequency_4pi
    normalized_stark_gamma = float(packed_log_table[1]) / line_frequency_4pi
    normalized_van_der_waals_gamma = float(packed_log_table[9384]) / line_frequency_4pi

    grid_position = np.floor(
        np.log(np.maximum(wavelength_nm_selected, 1e-30)) / log_grid_ratio + 0.5
    ).astype(np.int32)
    center_indices_1based = grid_position - grid_origin_index + 1

    species_codes = np.full(selected_count, _H2O_SPECIES_CODE, dtype=np.int32)
    margin_classes = np.full(selected_count, 7, dtype=np.int16)

    result = _compiled_molecular_arrays(
        center_indices_1based,
        classical_line_strength,
        species_codes,
        lower_excitation_cm,
        normalized_radiative_gamma,
        normalized_stark_gamma,
        normalized_van_der_waals_gamma,
        margin_classes,
    )
    _save_molecular_cache(cache_path, result)
    return result


# Internal helpers.
def _compiled_molecular_arrays(
    center_indices_1based,
    classical_line_strengths,
    species_codes,
    lower_excitation_cm_values,
    radiative_damping_values,
    stark_damping_values,
    van_der_waals_damping_values,
    margin_classes,
) -> Dict[str, np.ndarray]:
    return {
        "center_index_1based": np.asarray(center_indices_1based, dtype=np.int32),
        "classical_line_strength": np.asarray(
            classical_line_strengths, dtype=np.float32
        ),
        "species_code": np.asarray(species_codes, dtype=np.int16),
        "lower_excitation_cm": np.asarray(lower_excitation_cm_values, dtype=np.float32),
        "radiative_damping": np.asarray(radiative_damping_values, dtype=np.float32),
        "stark_damping": np.asarray(stark_damping_values, dtype=np.float32),
        "van_der_waals_damping": np.asarray(
            van_der_waals_damping_values, dtype=np.float32
        ),
        "margin_class": np.asarray(margin_classes, dtype=np.int16),
    }


__all__ = [
    "compile_molecular_text",
    "compile_tio_schwenke",
    "compile_h2o_partridge",
]
