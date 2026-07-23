# ruff: noqa: E402
"""Selected-line generation from resident source catalogs.

Reads the packed line-word catalogs from the data home (predicted,
observed, high-excitation, diatomic, TiO, water), applies the
strength/window selection driven by the current atmosphere labels, and
emits the packed selected-line words the opacity kernels consume. The
packed 4 x int32 word format is decoded only here.
"""

from __future__ import annotations

import weakref
from pathlib import Path

import numpy as np

from ._numba_cache import configure_numba_cache

configure_numba_cache()
try:
    import numba
except ImportError as exc:  # pragma: no cover - numba is a hard requirement
    raise ImportError(
        "numba is required: the compiled selected-line kernel is the sole "
        "production path for payne_zero_atmosphere."
    ) from exc

_NUMBA_AVAILABLE = True

from .constants import LIGHT_SPEED_NM_PER_S as _LIGHT_SPEED_NM_PER_SECOND
from .line_catalog import decode_selected_line_words
from .line_profile_math import build_selection_log_lookup


_LINE_SELECTION_STRENGTH_SCALE = 0.026538 / 1.77245
_PACKED_WAVELENGTH_STEP = np.log(1.0 + 1.0 / 2_000_000.0)
_TITANIUM_OXIDE_LOG_STRENGTH_OFFSETS = np.asarray(
    [-1101, -1138, -131, -1259, -1272],
    dtype=np.int32,
)

_DIATOMIC_MOLECULE_CODES = np.asarray(
    [
        8410,
        8411,
        8460,
        8461,
        8470,
        8471,
        8480,
        8481,
        8482,
        8510,
        8511,
        8512,
        8530,
        8531,
        8532,
        8580,
        8581,
        8582,
        8583,
        8584,
        8620,
        8621,
        8622,
        8623,
        8640,
        8641,
        8642,
        8643,
        8680,
        8681,
        8682,
        8690,
        8691,
        8692,
        8693,
        8700,
        8701,
        8702,
        8703,
        8704,
        8705,
        8890,
        8891,
        8892,
        8896,
        8960,
    ],
    dtype=np.int32,
)
_DIATOMIC_LOG_STRENGTH_OFFSETS = np.asarray(
    [
        0,
        -4469,
        -5,
        -1955,
        -2,
        -2444,
        -1,
        -3398,
        -2690,
        -105,
        -996,
        -947,
        -35,
        -1331,
        -1516,
        -13,
        -2189,
        -2870,
        -1681,
        -4398,
        -1362,
        -77,
        -1022,
        -1626,
        -1237,
        -38,
        -1658,
        -2553,
        -10,
        -1960,
        -3910,
        -7,
        -1957,
        -2449,
        -4399,
        -6,
        -1956,
        -3403,
        -5353,
        -2695,
        -4645,
        -36,
        -1332,
        -1517,
        -2725,
        -2,
    ],
    dtype=np.int32,
)
_DIATOMIC_CODE_LUT_MIN = int(np.min(_DIATOMIC_MOLECULE_CODES))
_DIATOMIC_CODE_LUT_MAX = int(np.max(_DIATOMIC_MOLECULE_CODES))
_DIATOMIC_CODE_TO_OFFSET_INDEX = np.zeros(
    _DIATOMIC_CODE_LUT_MAX - _DIATOMIC_CODE_LUT_MIN + 1,
    dtype=np.int32,
)
for _offset_index, _molecule_code in enumerate(_DIATOMIC_MOLECULE_CODES):
    _DIATOMIC_CODE_TO_OFFSET_INDEX[int(_molecule_code) - _DIATOMIC_CODE_LUT_MIN] = (
        _offset_index
    )


# Upper bound on the packed-wavelength code span tabulated by the fast
# bin-assignment lookup table (int32 entries; 2**25 entries = 128 MiB,
# covering every bundled catalog: the widest, the predicted atomic set,
# spans ~25.3M codes). Wider catalogs fall back to the original
# np.searchsorted expression, which is value-identical by construction.
_BIN_ASSIGNMENT_LUT_MAX_SPAN = 1 << 25

# Per-loaded-catalog derived-data cache used by the fast path. Keyed by the
# id() of the loaded catalog array, validated against a weak reference, and
# evicted by weakref callback when the catalog array is garbage collected:
# derived data lives exactly as long as the caller keeps the catalog alive.
# Only arrays that do NOT alias the catalog buffer are stored, so the cache
# never extends the catalog's lifetime. It stores no selection output, only
# input-derived arrays (halfword-widened criterion columns + bin assignment)
# that are value-identical on every recomputation — selection itself is
# recomputed in-memory each solve, with no disk cache. Catalog arrays are read-only inputs everywhere in
# this package; mutating one in place between selection calls would leave
# stale derived data (use a fresh array, as every reader here returns).
_CATALOG_DERIVED_CACHE: dict[int, tuple[weakref.ref, dict]] = {}


def _catalog_derived_state(catalog: np.ndarray) -> dict:
    """Return the mutable derived-data dict tied to this catalog object."""

    key = id(catalog)
    entry = _CATALOG_DERIVED_CACHE.get(key)
    if entry is not None and entry[0]() is catalog:
        return entry[1]
    state: dict = {}

    def _evict(reference: weakref.ref, _key: int = key) -> None:
        current = _CATALOG_DERIVED_CACHE.get(_key)
        if current is not None and current[0] is reference:
            del _CATALOG_DERIVED_CACHE[_key]

    _CATALOG_DERIVED_CACHE[key] = (weakref.ref(catalog, _evict), state)
    return state


def _standard_halfword_view(raw_words: np.ndarray) -> np.ndarray:
    """Zero-copy `(N, 6)` int16 view of the packed halfwords in words 1..3.

    `raw_words` must be C-contiguous `(N, 4)` int32 (callers normalize with
    `np.ascontiguousarray`). Viewing the whole row as eight halfwords and
    slicing columns 2..7 exposes exactly the bytes the original
    ``np.ascontiguousarray(words[:, 1:4]).view(np.int16).reshape(-1, 6)``
    copied, so every halfword value is identical.
    """

    return raw_words.view(np.int16)[:, 2:8]


def _standard_int32_columns(
    raw_words: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Species/excitation/strength halfwords widened to int32, cached per catalog."""

    state = _catalog_derived_state(raw_words)
    columns = state.get("standard_int32_columns")
    if columns is None:
        packed_halves = _standard_halfword_view(raw_words)
        columns = (
            packed_halves[:, 0].astype(np.int32),
            packed_halves[:, 1].astype(np.int32),
            packed_halves[:, 2].astype(np.int32),
        )
        state["standard_int32_columns"] = columns
    return columns


def _selection_lookup_float32(lookup: np.ndarray) -> np.ndarray:
    """float32 widening of the selection log lookup, cached per table object.

    Elementwise ``float64 -> float32`` conversion commutes with indexing, so
    ``lookup.astype(np.float32)[index]`` is bit-identical to the original
    ``lookup[index].astype(np.float32)``.
    """

    state = _catalog_derived_state(lookup)
    table = state.get("float32_lookup")
    if table is None:
        table = lookup.astype(np.float32)
        state["float32_lookup"] = table
    return table


def _bin_assignment_base(
    wavelength_codes: np.ndarray,
    packed_bins: np.ndarray,
) -> np.ndarray:
    """Clipped, monotonized packed-wavelength -> frequency-bin assignment.

    Value-identical to the original expression
    ``np.maximum.accumulate(np.clip(np.searchsorted(packed_bins,
    codes.astype(np.int64), side="right").astype(np.int32), 0,
    packed_bins.size - 1))``: the tabulated branch evaluates that same
    searchsorted for every integer code in the catalog's code span and
    gathers, which is exact because searchsorted is a pure elementwise
    function of the code value.

    The table is built only when the code span is no larger than the
    catalog itself (measured crossover: the LUT build costs ~span
    searchsorted probes plus an O(N) gather vs ~N probes for the fallback,
    so span > N catalogs are faster on the original expression) and within
    the memory cap. Both branches are byte-identical on every input.
    """

    codes = wavelength_codes.astype(np.int64)
    span = 0
    if codes.size > 0 and packed_bins.size > 0:
        code_min = int(codes.min())
        code_max = int(codes.max())
        span = code_max - code_min + 1
    if 0 < span <= min(codes.size, _BIN_ASSIGNMENT_LUT_MAX_SPAN):
        lut = np.searchsorted(
            packed_bins,
            np.arange(code_min, code_max + 1, dtype=np.int64),
            side="right",
        ).astype(np.int32)
        lut = np.clip(lut, 0, packed_bins.size - 1)
        clipped = lut[codes - np.int64(code_min)]
    else:
        raw_bins = np.searchsorted(packed_bins, codes, side="right").astype(np.int32)
        clipped = np.clip(raw_bins, 0, packed_bins.size - 1)
    return np.maximum.accumulate(clipped)


def _cached_bin_assignment_base(
    catalog: np.ndarray,
    wavelength_codes: np.ndarray,
    packed_bins: np.ndarray,
) -> np.ndarray:
    """Frequency-bin assignment computed once per (catalog, bin-edge grid) pair."""

    state = _catalog_derived_state(catalog)
    bins_key = ("bin_assignment_base", packed_bins.size, packed_bins.tobytes())
    base = state.get(bins_key)
    if base is None:
        base = _bin_assignment_base(wavelength_codes, packed_bins)
        state[bins_key] = base
    return base


def compute_doppler_population_ratio_max(
    *,
    partition_normalized_population_over_mass_density_and_fractional_doppler_width: np.ndarray,
    continuum_line_selection_threshold: np.ndarray,
) -> np.ndarray:
    """Return maximum line-strength-to-threshold ratio by species and bin."""

    doppler_population = np.asarray(
        partition_normalized_population_over_mass_density_and_fractional_doppler_width,
        dtype=np.float32,
    )
    continuum = np.asarray(continuum_line_selection_threshold, dtype=np.float32)
    if doppler_population.ndim != 2 or continuum.ndim != 2:
        raise ValueError(
            "population line-strength factors and continuum selection threshold "
            "must be 2-D"
        )
    if doppler_population.shape[0] != continuum.shape[0]:
        raise ValueError("population and continuum depth dimensions must match")

    depth_count, species_count = doppler_population.shape
    _, bin_count = continuum.shape
    ratio_max = np.zeros((species_count, bin_count), dtype=np.float32)
    for bin_index in range(bin_count):
        denominator = continuum[:, bin_index].reshape(depth_count, 1)
        ratio = np.divide(
            doppler_population,
            denominator,
            out=np.zeros_like(doppler_population, dtype=np.float32),
            where=denominator > np.float32(0.0),
        )
        ratio_max[:, bin_index] = np.max(ratio, axis=0)
    return ratio_max


# Process-resident cache of loaded catalogs, keyed by (resolved path, size, mtime).
# Returning the SAME array object across solves lets the object-keyed derived-column
# and bin-assignment caches below be reused, so a grid of models (solved in one
# process) reads + derives each ~1e8-line catalog only ONCE. The catalogs are static,
# so the (size, mtime) guard invalidates only if a file actually changes. Memory: the
# bundled catalogs total ~6 GB resident; for a single CLI solve this is the array the
# solve needs anyway (freed at process exit), and for an in-process grid it is the
# intended "read once, reuse" residency.
_CATALOG_READ_CACHE: dict = {}


def _cached_catalog_read(path: Path | str, loader) -> np.ndarray:
    resolved = Path(path).resolve()
    try:
        stat = resolved.stat()
    except OSError:
        return loader(path)
    key = (str(resolved), stat.st_size, stat.st_mtime_ns)
    cached = _CATALOG_READ_CACHE.get(key)
    if cached is None:
        cached = loader(path)
        _CATALOG_READ_CACHE[key] = cached
    return cached


def read_standard_line_catalog(path: Path | str) -> np.ndarray:
    """Read a packed selected-line word catalog into `(N, 4)` int32 words.

    Canonical form: an ``.npy`` file holding either a plain ``(N, 4)`` int32
    array or 16-byte structured records (viewed as packed words). The legacy
    raw ``.bin`` stream (fixed 16-byte records, no markers) is still parsed
    for provenance tooling. The load is process-resident (see
    ``_cached_catalog_read``), so a grid of models reads each catalog once.
    """

    return _cached_catalog_read(path, _read_standard_line_catalog_uncached)


def _read_standard_line_catalog_uncached(path: Path | str) -> np.ndarray:
    catalog_path = Path(path)
    if catalog_path.suffix == ".npy":
        # Catalogs above the 2 GB hosting limit are sharded as
        # ``<stem>_part1.npy`` .. ``_partN.npy`` (row-split, order preserved).
        if catalog_path.stem.endswith("_part1"):
            stem = catalog_path.stem[: -len("_part1")]
            part_paths = sorted(catalog_path.parent.glob(f"{stem}_part*.npy"))
            stored = np.concatenate([np.load(part) for part in part_paths])
        else:
            stored = np.load(catalog_path)
        if stored.dtype.fields is not None:
            return stored.view(np.int32).reshape(-1, 4)
        if stored.dtype != np.int32 or stored.ndim != 2 or stored.shape[1] != 4:
            raise ValueError(f"{path}: expected (N, 4) int32 packed line words")
        return stored
    raw_words = np.fromfile(catalog_path, dtype=np.int32)
    if raw_words.size % 4 != 0:
        raise ValueError(f"{path}: word count {raw_words.size} is not a multiple of 4.")
    return raw_words.reshape(-1, 4)


def read_diatomic_line_catalog(path: Path | str) -> np.ndarray:
    """Read the diatomic catalog into `(N, 4)` int32 words.

    Canonical form: an ``.npy`` file with the Fortran record markers already
    stripped. The legacy sequential-unformatted ``.bin`` stream is still
    parsed for provenance tooling. Process-resident like the standard reader.
    """

    return _cached_catalog_read(path, _read_diatomic_line_catalog_uncached)


def _read_diatomic_line_catalog_uncached(path: Path | str) -> np.ndarray:
    catalog_path = Path(path)
    if catalog_path.suffix == ".npy":
        stored = np.load(catalog_path)
        if stored.dtype != np.int32 or stored.ndim != 2 or stored.shape[1] != 4:
            raise ValueError(f"{path}: expected (N, 4) int32 packed line words")
        return stored
    raw_words = np.fromfile(catalog_path, dtype=np.int32)
    if raw_words.size % 6 != 0:
        raise ValueError(f"{path}: word count {raw_words.size} is not a multiple of 6.")
    records = raw_words.reshape(-1, 6)
    markers_ok = (records[:, 0] == 16) & (records[:, 5] == 16)
    if not np.all(markers_ok):
        bad_count = int(np.count_nonzero(~markers_ok))
        raise ValueError(f"{path}: {bad_count} diatomic records have bad markers")
    return np.ascontiguousarray(records[:, 1:5], dtype=np.int32)


def read_water_line_catalog(path: Path | str) -> np.ndarray:
    """Read the H2O catalog into wavelength/energy/strength code columns.

    Canonical form: an ``.npy`` file of 8-byte structured records
    (``wavelength_code`` i4, ``signed_lower_energy_code`` i2,
    ``signed_log_oscillator_strength_code`` i2). The legacy raw ``.bin`` stream is still
    parsed for provenance tooling. Process-resident like the standard reader.
    """

    return _cached_catalog_read(path, _read_water_line_catalog_uncached)


def _read_water_line_catalog_uncached(path: Path | str) -> np.ndarray:
    catalog_path = Path(path)
    if catalog_path.suffix == ".npy":
        stored = np.load(catalog_path)
        if stored.dtype.fields is None:
            raise ValueError(f"{path}: expected structured water-line records")
        return np.column_stack(
            (
                stored["wavelength_code"].astype(np.int32),
                stored["signed_lower_energy_code"].astype(np.int32),
                stored["signed_log_oscillator_strength_code"].astype(np.int32),
            )
        )
    raw_words = np.fromfile(catalog_path, dtype=np.int32)
    if raw_words.size % 2 != 0:
        raise ValueError(f"{path}: word count {raw_words.size} is not a multiple of 2.")
    words = raw_words.reshape(-1, 2)
    packed_pair = np.ascontiguousarray(words[:, 1]).view(np.int16).reshape(-1, 2)
    return np.column_stack(
        (
            words[:, 0].astype(np.int32),
            packed_pair[:, 0].astype(np.int32),
            packed_pair[:, 1].astype(np.int32),
        )
    )


def select_standard_line_words(
    words: np.ndarray,
    *,
    packed_continuum_wavelengths: np.ndarray,
    selection_log_lookup: np.ndarray,
    doppler_population_ratio_max: np.ndarray,
    deepest_hc_over_kt: float,
    frequency_per_bin: np.ndarray | None = None,
    minimum_population_ratio: float = 0.0,
    frequency_bin_floor: int = 0,
    log_strength_offsets: np.ndarray | None = None,
    population_slot_override: int | None = None,
    stark_damping_override: int | None = None,
    van_der_waals_damping_override: int | None = None,
) -> tuple[np.ndarray, int]:
    """Select standard fixed-record raw lines and return output words.

    This mirrors pykurucz's `_process_standard_catalog` for the unmodified
    atomic fixed-record catalogs (predicted, observed, and high-excitation).
    The optional log-strength offsets and Stark override cover the diatomic
    molecular family, which uses the same core selection criterion.

    Halfwords are widened once per loaded catalog object and the
    frequency-bin assignment is reused per (catalog, bin-edge grid) pair.
    """

    raw_words = np.ascontiguousarray(words, dtype=np.int32)
    if raw_words.ndim != 2 or raw_words.shape[1] != 4:
        raise ValueError("standard line words must have shape (N, 4)")
    if raw_words.size == 0:
        return raw_words.copy(), int(frequency_bin_floor)

    packed_bins = np.asarray(packed_continuum_wavelengths, dtype=np.int64)
    lookup = np.asarray(selection_log_lookup, dtype=np.float64)
    ratio_max = np.asarray(doppler_population_ratio_max, dtype=np.float32)
    if frequency_per_bin is None:
        wavelength_from_bin = np.exp(packed_bins * _PACKED_WAVELENGTH_STEP).astype(
            np.float64
        )
        wavelength_from_bin = np.where(
            wavelength_from_bin > 0.0, wavelength_from_bin, 1.0e-300
        )
        frequency_per_bin = (_LIGHT_SPEED_NM_PER_SECOND / wavelength_from_bin).astype(
            np.float32
        )
    frequencies = np.asarray(frequency_per_bin, dtype=np.float32)

    return _select_standard_line_words_fast(
        raw_words,
        packed_bins=packed_bins,
        lookup=lookup,
        ratio_max=ratio_max,
        frequencies=frequencies,
        deepest_hc_over_kt=deepest_hc_over_kt,
        minimum_population_ratio=minimum_population_ratio,
        frequency_bin_floor=frequency_bin_floor,
        log_strength_offsets=log_strength_offsets,
        population_slot_override=population_slot_override,
        stark_damping_override=stark_damping_override,
        van_der_waals_damping_override=van_der_waals_damping_override,
    )


@numba.njit(cache=True, nogil=True, parallel=True)
def _selection_mask_compiled(
    packed_species,
    bin_base,
    adjusted_log_strength,
    lower_excitation,
    ratio_max,
    frequencies,
    lookup_float32,
    boltzmann_table,
    scale,
    min_pop,
    frequency_bin_floor,
    species_slot_override,
    n_species,
    lookup_size,
):
    """Fused index-clipping AND keep-mask over the ~1e8 lines, parallel over lines.

    The integer index derivation (species slot, bin floor, strength/excitation lookup
    clips) and the float32 keep test are fused into one prange pass, so no ~1e8-element
    intermediate index arrays are materialized. Byte-identical to the numpy reference --
    the same integer clips and the same float32 operations in the same order. The
    per-line ``exp`` is NOT evaluated here; it is pre-tabulated in numpy over the strength
    lookup and gathered (``boltzmann_table``), so numba never calls its own ``exp`` (which
    can differ from numpy's by a ulp and flip a marginal line). ``fastmath`` stays off.
    """
    n = bin_base.shape[0]
    mask = np.empty(n, dtype=np.bool_)
    for i in numba.prange(n):
        # species slot: constant override, else |packed_species| // 10 - 1, clipped
        if species_slot_override >= 0:
            species = species_slot_override - 1
        else:
            species = abs(packed_species[i]) // 10 - 1
        if species < 0:
            species = 0
        elif species > n_species - 1:
            species = n_species - 1
        # frequency bin: running-max base with the constant floor applied
        frequency_bin = bin_base[i]
        if frequency_bin < frequency_bin_floor:
            frequency_bin = frequency_bin_floor
        # strength / excitation lookup indices (clipped)
        strength = adjusted_log_strength[i] - 1
        if strength < 0:
            strength = 0
        elif strength > lookup_size - 1:
            strength = lookup_size - 1
        excitation = lower_excitation[i] - 1
        if excitation < 0:
            excitation = 0
        elif excitation > lookup_size - 1:
            excitation = lookup_size - 1
        # keep test
        population_ratio = ratio_max[species, frequency_bin]
        denominator = frequencies[frequency_bin]
        if not (denominator > np.float32(0.0)):
            denominator = np.float32(1.0e-37)
        center_ratio = scale * lookup_float32[strength] * population_ratio / denominator
        boltzmann = boltzmann_table[excitation]
        mask[i] = (
            (population_ratio > min_pop)
            and (center_ratio >= np.float32(1.0))
            and (center_ratio * boltzmann >= np.float32(1.0))
        )
    return mask


def _selected_line_indices(
    packed_species,
    bin_base,
    adjusted_log_strength,
    lower_excitation,
    ratio_max,
    frequencies,
    lookup_float32,
    deepest_hc_over_kt,
    minimum_population_ratio,
    frequency_bin_floor,
    species_slot_override,
):
    """Indices of the selected lines. The compiled kernel derives the clipped indices
    and evaluates the keep test in one fused parallel pass over the raw per-line columns.

    The Boltzmann factor is tabulated in numpy over the strength lookup (bitwise) and
    gathered inside the kernel, keeping the result byte-identical to the numpy reference.
    ``species_slot_override`` is the constant population slot (>= 0) for the TiO / H3+
    families, or -1 to derive the slot from ``packed_species``.
    """
    hc_over_kt = np.float32(deepest_hc_over_kt)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        boltzmann_table = np.exp(-lookup_float32 * hc_over_kt).astype(np.float32)
    mask = _selection_mask_compiled(
        np.ascontiguousarray(packed_species, dtype=np.int32),
        np.ascontiguousarray(bin_base, dtype=np.int32),
        np.ascontiguousarray(adjusted_log_strength, dtype=np.int32),
        np.ascontiguousarray(lower_excitation, dtype=np.int32),
        np.ascontiguousarray(ratio_max, dtype=np.float32),
        np.ascontiguousarray(frequencies, dtype=np.float32),
        lookup_float32,
        boltzmann_table,
        np.float32(_LINE_SELECTION_STRENGTH_SCALE),
        np.float32(minimum_population_ratio),
        np.int32(frequency_bin_floor),
        np.int32(species_slot_override),
        np.int32(ratio_max.shape[0]),
        np.int32(lookup_float32.size),
    )
    return np.where(mask)[0]


@numba.njit(cache=True, nogil=True, parallel=True)
def _gather_selected_rows(words, indices):
    """Parallel row gather -- ``words[indices]`` copied in parallel (byte-identical).

    The output materialization (~1e8 selected rows) is memory-bandwidth-bound, so a
    parallel copy scales with cores/aggregate bandwidth where numpy's fancy index is
    single-threaded.
    """
    n = indices.shape[0]
    out = np.empty((n, 4), dtype=np.int32)
    for i in numba.prange(n):
        idx = indices[i]
        out[i, 0] = words[idx, 0]
        out[i, 1] = words[idx, 1]
        out[i, 2] = words[idx, 2]
        out[i, 3] = words[idx, 3]
    return out


def _select_standard_line_words_fast(
    raw_words: np.ndarray,
    *,
    packed_bins: np.ndarray,
    lookup: np.ndarray,
    ratio_max: np.ndarray,
    frequencies: np.ndarray,
    deepest_hc_over_kt: float,
    minimum_population_ratio: float,
    frequency_bin_floor: int,
    log_strength_offsets: np.ndarray | None,
    population_slot_override: int | None,
    stark_damping_override: int | None,
    van_der_waals_damping_override: int | None,
) -> tuple[np.ndarray, int]:
    """Select standard (atomic/diatomic/TiO) line words for one pass.

    The criterion halfwords are widened to int32 once per loaded catalog
    object (via a zero-copy int16 view; damping halfwords stay packed until
    output), the frequency-bin assignment is computed once per (catalog,
    bin-edge grid) pair, and the bin floor is applied after the running
    maximum, which commutes exactly for a constant integer floor.
    """

    packed_wavelength = raw_words[:, 0]
    packed_species, lower_excitation, log_strength = _standard_int32_columns(raw_words)
    lookup_float32 = _selection_lookup_float32(lookup)
    bin_base = _cached_bin_assignment_base(raw_words, packed_wavelength, packed_bins)

    # adjusted_log_strength is needed both for the keep test and the packed output.
    if log_strength_offsets is not None:
        adjusted_log_strength = np.maximum(
            log_strength + np.asarray(log_strength_offsets, dtype=np.int32),
            1,
        )
    else:
        adjusted_log_strength = log_strength

    # The keep test AND the per-line index clips (species slot, bin floor, strength /
    # excitation lookup) are fused into one compiled parallel pass over the raw columns,
    # so no ~1e8-element intermediate index arrays are built. Byte-identical to the numpy
    # reference for every catalog family.
    species_slot_override = (
        -1 if population_slot_override is None else int(population_slot_override)
    )
    selected_index = _selected_line_indices(
        packed_species,
        bin_base,
        adjusted_log_strength,
        lower_excitation,
        ratio_max,
        frequencies,
        lookup_float32,
        deepest_hc_over_kt,
        minimum_population_ratio,
        int(frequency_bin_floor),
        species_slot_override,
    )
    if (
        log_strength_offsets is None
        and stark_damping_override is None
        and van_der_waals_damping_override is None
    ):
        selected_words = _gather_selected_rows(raw_words, selected_index)
    else:
        packed_halves = _standard_halfword_view(raw_words)
        selected_words = np.zeros((selected_index.size, 4), dtype=np.int32)
        selected_words[:, 0] = packed_wavelength[selected_index].astype(np.int32)
        packed_output = selected_words[:, 1:4].view(np.int16).reshape(-1, 6)
        packed_output[:, 0] = packed_species[selected_index].astype(np.int16)
        packed_output[:, 1] = lower_excitation[selected_index].astype(np.int16)
        packed_output[:, 2] = adjusted_log_strength[selected_index].astype(np.int16)
        packed_output[:, 3] = packed_halves[selected_index, 3]
        if stark_damping_override is None:
            packed_output[:, 4] = packed_halves[selected_index, 4]
        else:
            packed_output[:, 4] = np.int16(stark_damping_override)
        if van_der_waals_damping_override is None:
            packed_output[:, 5] = packed_halves[selected_index, 5]
        else:
            packed_output[:, 5] = np.int16(van_der_waals_damping_override)
        selected_words = np.ascontiguousarray(selected_words, dtype=np.int32)
    # Next call's frequency-bin floor: bin_base is the running maximum (monotonic
    # non-decreasing), so its last element is the largest; apply this call's floor.
    frequency_bin_end = max(int(bin_base[-1]), int(frequency_bin_floor))
    return selected_words, frequency_bin_end


def _diatomic_log_strength_offsets(words: np.ndarray) -> np.ndarray:
    raw_words = np.ascontiguousarray(words, dtype=np.int32)
    packed_species = _standard_int32_columns(raw_words)[0]
    molecule_code = np.abs(packed_species.astype(np.int32))
    lut_index = np.clip(
        molecule_code - _DIATOMIC_CODE_LUT_MIN,
        0,
        _DIATOMIC_CODE_TO_OFFSET_INDEX.size - 1,
    )
    offset_index = _DIATOMIC_CODE_TO_OFFSET_INDEX[lut_index]
    return _DIATOMIC_LOG_STRENGTH_OFFSETS[offset_index]


def _titanium_oxide_log_strength_offsets(words: np.ndarray) -> np.ndarray:
    raw_words = np.ascontiguousarray(words, dtype=np.int32)
    packed_species = _standard_int32_columns(raw_words)[0]
    isotope_index = (np.abs(packed_species.astype(np.int32)) - 8949).clip(1, 5) - 1
    return _TITANIUM_OXIDE_LOG_STRENGTH_OFFSETS[isotope_index]


def select_water_line_words(
    records: np.ndarray,
    *,
    packed_continuum_wavelengths: np.ndarray,
    selection_log_lookup: np.ndarray,
    doppler_population_ratio_max: np.ndarray,
    deepest_hc_over_kt: float,
    frequency_per_bin: np.ndarray | None = None,
) -> np.ndarray:
    """Select H2O (water) records and return compact selected-line words."""

    raw_records = np.ascontiguousarray(records, dtype=np.int32)
    if raw_records.ndim != 2 or raw_records.shape[1] != 3:
        raise ValueError("water line records must have shape (N, 3)")
    if raw_records.size == 0:
        return np.zeros((0, 4), dtype=np.int32)

    packed_bins = np.asarray(packed_continuum_wavelengths, dtype=np.int64)
    lookup = np.asarray(selection_log_lookup, dtype=np.float64)
    ratio_max = np.asarray(doppler_population_ratio_max, dtype=np.float32)
    if ratio_max.shape[0] <= 939:
        raise ValueError("H2O selection requires population slot 940")
    if frequency_per_bin is None:
        wavelength_from_bin = np.exp(packed_bins * _PACKED_WAVELENGTH_STEP).astype(
            np.float64
        )
        wavelength_from_bin = np.where(
            wavelength_from_bin > 0.0, wavelength_from_bin, 1.0e-300
        )
        frequency_per_bin = (_LIGHT_SPEED_NM_PER_SECOND / wavelength_from_bin).astype(
            np.float32
        )
    frequencies = np.asarray(frequency_per_bin, dtype=np.float32)

    return _select_water_line_words_fast(
        raw_records,
        packed_bins=packed_bins,
        lookup=lookup,
        ratio_max=ratio_max,
        frequencies=frequencies,
        deepest_hc_over_kt=deepest_hc_over_kt,
    )


def _water_line_derived_columns(
    raw_records: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label-independent water columns, cached per loaded catalog object.

    Returns ``(packed_species, excitation_energy, adjusted_log_strength)``;
    the raw catalog columns are read through views instead of per-pass
    `astype` copies, which cannot change any value.
    """

    state = _catalog_derived_state(raw_records)
    derived = state.get("water_derived_columns")
    if derived is None:
        lower_excitation_raw = raw_records[:, 1]
        log_strength_raw = raw_records[:, 2]
        isotope = np.full(raw_records.shape[0], 4, dtype=np.int32)
        isotope = np.where(log_strength_raw > 0, 3, isotope)
        isotope = np.where(lower_excitation_raw > 0, 2, isotope)
        isotope = np.where(
            (lower_excitation_raw > 0) & (log_strength_raw > 0),
            1,
            isotope,
        )
        packed_species = -(9399 + isotope)
        excitation_energy = np.abs(lower_excitation_raw).astype(np.float32)
        isotope_log_strength_offsets = np.asarray(
            [-1, -3398, -2690, -5000], dtype=np.int32
        )
        adjusted_log_strength = np.maximum(
            np.abs(log_strength_raw).astype(np.int32)
            + isotope_log_strength_offsets[isotope - 1],
            1,
        )
        derived = (packed_species, excitation_energy, adjusted_log_strength)
        state["water_derived_columns"] = derived
    return derived


@numba.njit(cache=True, nogil=True, parallel=True)
def _water_selection_mask_compiled(
    frequency_bin,
    adjusted_log_strength,
    excitation_code,
    ratio_max,
    frequencies,
    lookup_float32,
    boltzmann_table,
    scale,
    water_species,
    lookup_size,
):
    """Fused keep-mask for the H2O family, parallel over lines.

    Same shape as the standard kernel but with the fixed H2O population slot and the
    ``population_ratio > 0`` floor. The Boltzmann factor is tabulated in numpy over the
    integer energy code (H2O energies are a 16-bit code, so the table is exact) and
    gathered, so numba never calls its own exp. Byte-identical to the numpy reference.
    """
    n = frequency_bin.shape[0]
    mask = np.empty(n, dtype=np.bool_)
    for i in numba.prange(n):
        fb = frequency_bin[i]
        population_ratio = ratio_max[water_species, fb]
        denominator = frequencies[fb]
        if not (denominator > np.float32(0.0)):
            denominator = np.float32(1.0e-37)
        strength = adjusted_log_strength[i] - 1
        if strength < 0:
            strength = 0
        elif strength > lookup_size - 1:
            strength = lookup_size - 1
        center_ratio = scale * lookup_float32[strength] * population_ratio / denominator
        boltzmann = boltzmann_table[excitation_code[i]]
        mask[i] = (
            (population_ratio > np.float32(0.0))
            and (center_ratio >= np.float32(1.0))
            and (center_ratio * boltzmann >= np.float32(1.0))
        )
    return mask


def _water_selected_line_indices(
    frequency_bin,
    adjusted_log_strength,
    excitation_energy,
    ratio_max,
    frequencies,
    lookup_float32,
    deepest_hc_over_kt,
):
    """Indices of the selected H2O lines. The per-line exp is tabulated in numpy over
    the integer energy code (bitwise) and gathered in the kernel, so the mask is
    byte-identical to the numpy reference."""
    hc_over_kt = np.float32(deepest_hc_over_kt)
    excitation_code = np.ascontiguousarray(excitation_energy).astype(np.int32)
    max_code = int(excitation_code.max()) if excitation_code.size else 0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        boltzmann_table = np.exp(
            -np.arange(max_code + 1, dtype=np.float32) * hc_over_kt
        ).astype(np.float32)
    mask = _water_selection_mask_compiled(
        np.ascontiguousarray(frequency_bin, dtype=np.int32),
        np.ascontiguousarray(adjusted_log_strength, dtype=np.int32),
        excitation_code,
        np.ascontiguousarray(ratio_max, dtype=np.float32),
        np.ascontiguousarray(frequencies, dtype=np.float32),
        lookup_float32,
        boltzmann_table,
        np.float32(_LINE_SELECTION_STRENGTH_SCALE),
        np.int32(939),
        np.int32(lookup_float32.size),
    )
    return np.where(mask)[0]


def _select_water_line_words_fast(
    raw_records: np.ndarray,
    *,
    packed_bins: np.ndarray,
    lookup: np.ndarray,
    ratio_max: np.ndarray,
    frequencies: np.ndarray,
    deepest_hc_over_kt: float,
) -> np.ndarray:
    """Select water line words for one pass.

    Reuses the label-independent derived columns and the per-(catalog,
    bin-edge grid) frequency-bin assignment.
    """

    packed_wavelength = raw_records[:, 0]
    (
        packed_species,
        excitation_energy,
        adjusted_log_strength,
    ) = _water_line_derived_columns(raw_records)
    lookup_float32 = _selection_lookup_float32(lookup)
    frequency_bin = _cached_bin_assignment_base(
        raw_records, packed_wavelength, packed_bins
    )

    # Fused parallel keep-mask (H2O population slot 939, population_ratio > 0 floor);
    # byte-identical to the numpy reference (exp tabulated over the integer energy code).
    selected_index = _water_selected_line_indices(
        frequency_bin,
        adjusted_log_strength,
        excitation_energy,
        ratio_max,
        frequencies,
        lookup_float32,
        deepest_hc_over_kt,
    )
    if selected_index.size == 0:
        return np.zeros((0, 4), dtype=np.int32)

    # Both packed columns are functions of small domains -- radiative damping of the
    # frequency bin (~344 bins), excitation of the integer energy code -- so tabulate the
    # log10 over those domains (numpy, byte-identical) and gather, instead of ~1e8 log10s.
    bin_frequency = frequencies.astype(np.float64)
    radiative_damping_by_bin = 2.474e-22 * (bin_frequency * bin_frequency) * 0.001
    packed_radiative_damping_by_bin = np.clip(
        (
            np.log10(np.maximum(radiative_damping_by_bin, 1.0e-300)) * 1000.0 + 16384.5
        ).astype(np.int64),
        1,
        32768,
    ).astype(np.int32)
    packed_radiative_damping = packed_radiative_damping_by_bin[
        frequency_bin[selected_index]
    ]

    excitation_code = np.ascontiguousarray(excitation_energy).astype(np.int32)
    max_code = int(excitation_code.max()) if excitation_code.size else 0
    energy_by_code = np.arange(max_code + 1, dtype=np.float64)
    packed_excitation_by_code = np.clip(
        (np.log10(np.maximum(energy_by_code, 1.0e-300)) * 1000.0 + 16384.5).astype(
            np.int64
        ),
        1,
        32768,
    ).astype(np.int32)
    packed_excitation = packed_excitation_by_code[excitation_code[selected_index]]

    selected_words = np.zeros((selected_index.size, 4), dtype=np.int32)
    selected_words[:, 0] = packed_wavelength[selected_index].astype(np.int32)
    packed_output = selected_words[:, 1:4].view(np.int16).reshape(-1, 6)
    packed_output[:, 0] = packed_species[selected_index].astype(np.int16)
    packed_output[:, 1] = packed_excitation.astype(np.int16)
    packed_output[:, 2] = adjusted_log_strength[selected_index].astype(np.int16)
    packed_output[:, 3] = packed_radiative_damping.astype(np.int16)
    packed_output[:, 4] = np.int16(1)
    packed_output[:, 5] = np.int16(9384)
    return np.ascontiguousarray(selected_words, dtype=np.int32)


def write_selected_line_words(path: Path | str, words: np.ndarray) -> None:
    """Write selected compact selected-line int32 words."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(words, dtype=np.int32).tofile(target)


def generate_selected_lines(
    *,
    partition_normalized_population_over_mass_density_and_fractional_doppler_width: np.ndarray,
    continuum_line_selection_threshold: np.ndarray,
    packed_continuum_wavelengths: np.ndarray,
    hc_over_kt: np.ndarray,
    selected_lines_output: Path | str | None = None,
    predicted_atomic_lines_path: Path | str | None = None,
    observed_atomic_lines_path: Path | str | None = None,
    high_excitation_lines_path: Path | str | None = None,
    diatomic_lines_path: Path | str | None = None,
    titanium_oxide_lines_path: Path | str | None = None,
    water_lines_path: Path | str | None = None,
    h3plus_lines_path: Path | str | None = None,
):
    """Generate the compact selected-line catalog from ported raw SELECTLINES catalogs.

    Returns the decoded ``SelectedLineCatalog`` in memory. If ``selected_lines_output`` is
    given the packed words are also written there (kept for tooling); the atmosphere solver
    leaves it ``None`` and consumes the returned catalog directly, with no disk round-trip.
    """

    ratio_max = compute_doppler_population_ratio_max(
        partition_normalized_population_over_mass_density_and_fractional_doppler_width=(
            partition_normalized_population_over_mass_density_and_fractional_doppler_width
        ),
        continuum_line_selection_threshold=continuum_line_selection_threshold,
    )
    packed_bins = np.asarray(packed_continuum_wavelengths, dtype=np.int64)
    wavelength_from_bin = np.exp(packed_bins * _PACKED_WAVELENGTH_STEP).astype(
        np.float64
    )
    wavelength_from_bin = np.where(
        wavelength_from_bin > 0.0, wavelength_from_bin, 1.0e-300
    )
    frequency_per_bin = (_LIGHT_SPEED_NM_PER_SECOND / wavelength_from_bin).astype(
        np.float32
    )
    lookup = build_selection_log_lookup()
    deepest_hckt = float(np.asarray(hc_over_kt, dtype=np.float64)[-1])

    selected_groups: list[np.ndarray] = []
    low_frequency_end = 0

    if (
        predicted_atomic_lines_path is not None
        and Path(predicted_atomic_lines_path).exists()
    ):
        selected, low_frequency_end = select_standard_line_words(
            read_standard_line_catalog(predicted_atomic_lines_path),
            packed_continuum_wavelengths=packed_bins,
            selection_log_lookup=lookup,
            doppler_population_ratio_max=ratio_max,
            deepest_hc_over_kt=deepest_hckt,
            frequency_per_bin=frequency_per_bin,
            minimum_population_ratio=1.0e-37,
            frequency_bin_floor=low_frequency_end,
        )
        selected_groups.append(selected)

    if (
        observed_atomic_lines_path is not None
        and Path(observed_atomic_lines_path).exists()
    ):
        selected, _ = select_standard_line_words(
            read_standard_line_catalog(observed_atomic_lines_path),
            packed_continuum_wavelengths=packed_bins,
            selection_log_lookup=lookup,
            doppler_population_ratio_max=ratio_max,
            deepest_hc_over_kt=deepest_hckt,
            frequency_per_bin=frequency_per_bin,
            minimum_population_ratio=1.0e-37,
        )
        selected_groups.append(selected)

    if (
        high_excitation_lines_path is not None
        and Path(high_excitation_lines_path).exists()
    ):
        selected, _ = select_standard_line_words(
            read_standard_line_catalog(high_excitation_lines_path),
            packed_continuum_wavelengths=packed_bins,
            selection_log_lookup=lookup,
            doppler_population_ratio_max=ratio_max,
            deepest_hc_over_kt=deepest_hckt,
            frequency_per_bin=frequency_per_bin,
        )
        selected_groups.append(selected)

    if diatomic_lines_path is not None and Path(diatomic_lines_path).exists():
        diatomic_words = read_diatomic_line_catalog(diatomic_lines_path)
        selected, _ = select_standard_line_words(
            diatomic_words,
            packed_continuum_wavelengths=packed_bins,
            selection_log_lookup=lookup,
            doppler_population_ratio_max=ratio_max,
            deepest_hc_over_kt=deepest_hckt,
            frequency_per_bin=frequency_per_bin,
            log_strength_offsets=_diatomic_log_strength_offsets(diatomic_words),
            stark_damping_override=1,
        )
        selected_groups.append(selected)

    if (
        titanium_oxide_lines_path is not None
        and Path(titanium_oxide_lines_path).exists()
    ):
        titanium_oxide_words = read_standard_line_catalog(titanium_oxide_lines_path)
        selected, _ = select_standard_line_words(
            titanium_oxide_words,
            packed_continuum_wavelengths=packed_bins,
            selection_log_lookup=lookup,
            doppler_population_ratio_max=ratio_max,
            deepest_hc_over_kt=deepest_hckt,
            frequency_per_bin=frequency_per_bin,
            log_strength_offsets=_titanium_oxide_log_strength_offsets(
                titanium_oxide_words
            ),
            population_slot_override=895,
            stark_damping_override=1,
            van_der_waals_damping_override=9384,
        )
        selected_groups.append(selected)

    if water_lines_path is not None and Path(water_lines_path).exists():
        selected = select_water_line_words(
            read_water_line_catalog(water_lines_path),
            packed_continuum_wavelengths=packed_bins,
            selection_log_lookup=lookup,
            doppler_population_ratio_max=ratio_max,
            deepest_hc_over_kt=deepest_hckt,
            frequency_per_bin=frequency_per_bin,
        )
        selected_groups.append(selected)

    if h3plus_lines_path is not None and Path(h3plus_lines_path).exists():
        h3plus_words = read_standard_line_catalog(h3plus_lines_path)
        selected, _ = select_standard_line_words(
            h3plus_words,
            packed_continuum_wavelengths=packed_bins,
            selection_log_lookup=lookup,
            doppler_population_ratio_max=ratio_max,
            deepest_hc_over_kt=deepest_hckt,
            frequency_per_bin=frequency_per_bin,
            log_strength_offsets=np.full(h3plus_words.shape[0], -1272, dtype=np.int32),
            population_slot_override=895,
            stark_damping_override=1,
            van_der_waals_damping_override=9384,
        )
        selected_groups.append(selected)

    output_words = (
        np.vstack(selected_groups)
        if selected_groups
        else np.zeros((0, 4), dtype=np.int32)
    )
    if selected_lines_output is not None:
        write_selected_line_words(selected_lines_output, output_words)
    # Freshly generated words are native-layout, so skip the swapped-layout probe.
    return decode_selected_line_words(output_words, detect_swapped_layout=False)
