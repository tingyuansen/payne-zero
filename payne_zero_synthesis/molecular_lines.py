"""GPU-resident molecular band-opacity kernel.

The molecular catalog is kept as structure-of-arrays, uploaded once, and
processed in on-device chunks.  Each line contributes a center opacity plus
red/blue Voigt wings into a depth-by-wavelength accumulator.  Molecular
equilibrium is solved elsewhere; this module consumes the molecular population
slot in the per-ion population cube.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np

try:  # torch is optional for the numpy-only loader path
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from . import paths as runtime_paths
from .constants import (
    ATOMIC_MASS_GRAM,
    BOLTZMANN_ERG_PER_K,
    LIGHT_SPEED_CM_PER_S,
    LIGHT_SPEED_NM_PER_S,
    PLANCK_ERG_SECOND,
)
from .device import ACCUMULATION_DTYPE, DEFAULT_DTYPE, device, to_dev
from .line_opacity import (
    LINE_CENTER_CUTOFF_RATIO,
    MAX_WING_PROFILE_STEPS,
    interpolate_harris_profile,
    highp_dtype,
)


# Per-species molecular masses [amu] keyed by the line-list species code. The
# masses set the molecular Doppler fraction and therefore the line amplitude.
# Species absent from this table use the conservative 20 amu default.
SPECIES_MASS_AMU = {
    240: 2.0,
    246: 13.0,
    258: 17.0,
    264: 24.0,
    270: 26.0,
    324: 43.0,
    342: 41.0,
    366: 64.0,
    372: 67.0,
    432: 52.0,
    492: 24.0,
    252: 15.0,  # NH
    276: 28.0,  # CO
    282: 28.0,  # N2
    288: 30.0,  # NO
    294: 32.0,  # O2
    300: 25.0,  # MgH      (giant 5000-5500 A band; was missing)
    306: 28.0,  # AlH
    312: 29.0,  # SiH
    318: 40.0,  # MgO
    330: 44.0,  # SiO
    336: 33.0,  # SH
    348: 48.0,  # SO
    354: 56.0,  # CaO
    360: 61.0,  # ScO
    378: 8.0,  # HeH
    384: 10.0,  # LiH
    390: 12.0,  # BeH
    396: 20.0,  # FH
    402: 32.0,  # PH
    408: 36.0,  # ClH
    414: 46.0,  # ScH
    420: 49.0,  # TiH
    426: 52.0,  # VH
    438: 56.0,  # MnH
    444: 57.0,  # FeH      (M-dwarf 9900-10500 A Wing-Ford band #43; was missing)
    450: 13.0,  # 13CH
    456: 15.0,  # 15NH
    462: 17.0,  # 18OH
    468: 25.0,  # MgH iso
    474: 28.0,  # AlH iso
    480: 29.0,  # SiH iso
    486: 41.0,  # CaH iso
    498: 40.0,  # KH
    504: 3.0,  # H3+
    510: 51.0,  # ClO
    516: 68.0,  # CrO
    522: 71.0,  # MnO
    528: 72.0,  # FeO
    534: 18.0,  # H2O
    540: 44.0,  # CO2
    546: 14.0,  # HCN
    552: 36.0,  # C3
    558: 60.0,  # CoH
    564: 59.0,  # NiH
    570: 64.0,  # CuH
    576: 75.0,  # CoO
    582: 74.0,  # NiO
    588: 79.0,  # CuO
    594: 28.0,  # 13CO
    780: 104.0,  # YO
    786: 107.0,  # ZrO
    792: 155.0,  # LaO
}

# Lines per dense cutoff chunk.  This bounds the temporary [depth, line] tensors
# while keeping the loop on resident device arrays.
CHUNK_LINES = 500_000

# Surviving depth-line pairs per wing-walk block.  This bounds the wider
# [pair, offset] tensors used by the Voigt wing deposit.
PAIR_CHUNK = 200_000


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
    return max(1, value)


def molecular_chunk_lines(default: int = CHUNK_LINES) -> int:
    """Line chunk size for the molecular dense cutoff stage."""
    return _env_positive_int("PAYNE_ZERO_SYNTHESIS_MOLECULAR_CHUNK_LINES", default)


# Bump the logic version when catalog-derived fields or cache schemas change.
CACHE_SCHEMA = 1
CACHE_LOGIC_VERSION = 2


@dataclass
class MolecularLineCatalog:
    """Structure-of-arrays of per-line invariant molecular fields."""

    center_index_1based: np.ndarray
    classical_line_strength: np.ndarray
    species_code: np.ndarray
    lower_excitation_cm: np.ndarray
    radiative_damping: np.ndarray
    stark_damping: np.ndarray
    van_der_waals_damping: np.ndarray
    center_index: np.ndarray
    species_population_column: np.ndarray
    wavelength_nm: np.ndarray
    log_grid_ratio: float
    grid_origin_index: int
    unique_species_codes: np.ndarray

    def __len__(self) -> int:
        return int(self.center_index_1based.size)

    _SERIALIZED_FIELDS = (
        "center_index_1based",
        "classical_line_strength",
        "species_code",
        "lower_excitation_cm",
        "radiative_damping",
        "stark_damping",
        "van_der_waals_damping",
    )

    def to_npz(self, path: Path) -> None:
        """Write the physical catalog fields; derived indices are rebuilt on load."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(
            schema=CACHE_SCHEMA,
            logic_version=CACHE_LOGIC_VERSION,
            log_grid_ratio=float(self.log_grid_ratio),
            grid_origin_index=int(self.grid_origin_index),
        )
        arrays = {
            field_name: getattr(self, field_name)
            for field_name in self._SERIALIZED_FIELDS
        }
        arrays["unique_species_codes"] = self.unique_species_codes
        arrays["__meta__"] = np.frombuffer(
            json.dumps(meta).encode("utf-8"), dtype=np.uint8
        )
        np.savez(path, **arrays)

    @classmethod
    def from_npz(cls, path: Path) -> "MolecularLineCatalog":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(bytes(data["__meta__"]).decode("utf-8"))
            mapping = {key: data[key] for key in data.files if key != "__meta__"}
            mapping["unique_species_codes"] = data["unique_species_codes"]
            mapping["log_grid_ratio"] = meta["log_grid_ratio"]
            mapping["grid_origin_index"] = meta["grid_origin_index"]
            return cls.from_mapping(mapping)

    @classmethod
    def from_mapping(cls, mapping: dict) -> "MolecularLineCatalog":
        """Build from modern molecular catalog arrays."""
        if "__meta__" in mapping and "log_grid_ratio" not in mapping:
            meta = json.loads(bytes(np.asarray(mapping["__meta__"])).decode("utf-8"))
            mapping = dict(mapping)
            mapping["log_grid_ratio"] = meta["log_grid_ratio"]
            mapping["grid_origin_index"] = meta["grid_origin_index"]

        fields = {}
        for field in cls._SERIALIZED_FIELDS:
            if field in mapping:
                fields[field] = np.asarray(mapping[field])

        missing_base_fields = {
            "center_index_1based",
            "classical_line_strength",
            "species_code",
            "lower_excitation_cm",
            "radiative_damping",
            "stark_damping",
            "van_der_waals_damping",
        } - set(fields)
        if missing_base_fields:
            missing = ", ".join(sorted(missing_base_fields))
            raise KeyError(f"Missing molecular catalog arrays: {missing}")

        log_grid_ratio = float(np.asarray(mapping["log_grid_ratio"]))
        grid_origin_index = int(np.asarray(mapping["grid_origin_index"]))

        if {
            "center_index",
            "species_population_column",
            "wavelength_nm",
        } - set(fields):
            indices = _precompute_indices(
                fields["center_index_1based"],
                fields["species_code"],
                log_grid_ratio,
                grid_origin_index,
            )
            fields.update(indices)

        if "unique_species_codes" in mapping:
            unique_species_codes = np.asarray(mapping["unique_species_codes"])
        else:
            unique_species_codes = np.unique(fields["species_code"]).astype(np.int32)

        return cls(
            log_grid_ratio=log_grid_ratio,
            grid_origin_index=grid_origin_index,
            unique_species_codes=unique_species_codes,
            **fields,
        )


def _precompute_indices(
    center_index_1based: np.ndarray,
    species_code: np.ndarray,
    log_grid_ratio: float,
    grid_origin_index: int,
) -> dict:
    """Compute catalog indices and reconstructed line wavelengths once."""
    center_index_1based_64 = center_index_1based.astype(np.int64)
    species_code_64 = species_code.astype(np.int64)
    center_index = (center_index_1based_64 - 1).astype(np.int32)
    species_population_column = (species_code_64 // 6 - 1).astype(np.int32)
    wavelength_nm = np.exp(
        (center_index_1based_64.astype(np.float64) - 1 + grid_origin_index)
        * log_grid_ratio
    ).astype(np.float32)
    return dict(
        center_index=center_index,
        species_population_column=species_population_column,
        wavelength_nm=wavelength_nm,
    )


def build_catalog_from_arrays(arrays: dict) -> MolecularLineCatalog:
    """Assemble a molecular catalog from modern compiled arrays."""
    return MolecularLineCatalog.from_mapping(arrays)


def _default_cache_dir() -> Path:
    return runtime_paths.PACKAGE_CACHE_ROOT / "molecular_indices"


def _cache_key(source_path: Path) -> str:
    source_stat = source_path.stat()
    payload = {
        "schema": CACHE_SCHEMA,
        "logic_version": CACHE_LOGIC_VERSION,
        "source": str(source_path.resolve()),
        "size": int(source_stat.st_size),
        "mtime_ns": int(
            getattr(source_stat, "st_mtime_ns", int(source_stat.st_mtime * 1e9))
        ),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:24]


def load_catalog(
    source_path: Path, cache_dir: Optional[Path] = None, rebuild: bool = False
) -> MolecularLineCatalog:
    """Load a molecular catalog, building the derived-index cache if needed."""
    source_path = Path(source_path)
    cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    key = _cache_key(source_path)
    cache_path = cache_dir / f"molecular_lines_{key}.npz"

    if not rebuild and cache_path.exists():
        try:
            return MolecularLineCatalog.from_npz(cache_path)
        except Exception:
            pass  # corrupt/stale -> reparse

    with np.load(source_path, allow_pickle=False) as data:
        compiled = {key: data[key] for key in data.files}
    catalog = build_catalog_from_arrays(compiled)
    try:
        catalog.to_npz(cache_path)
    except Exception:  # pragma: no cover - caching is best-effort
        pass
    return catalog


@dataclass
class MolecularLineInvariants:
    """GPU-resident static molecular data and numerical tables.

    These tensors depend only on the line list, synthesis grid, and Voigt tables.
    They can stay resident while atmospheres or abundances change.
    """

    # grid
    wavelength_grid: torch.Tensor
    n_wavelengths: int
    local_resolving_power: torch.Tensor
    # line static SoA (contiguous per field, resident)
    classical_line_strength: torch.Tensor
    lower_excitation_cm: torch.Tensor
    radiative_damping: torch.Tensor
    stark_damping: torch.Tensor
    van_der_waals_damping: torch.Tensor
    center_index: torch.Tensor
    line_species_index: torch.Tensor
    line_wavelength_nm: torch.Tensor
    # per-species bookkeeping
    species_code: torch.Tensor
    species_population_column: torch.Tensor
    species_mass_amu: torch.Tensor
    # numerical tables
    harris_profile_h0_table: torch.Tensor
    harris_profile_h1_table: torch.Tensor
    harris_profile_h2_table: torch.Tensor


def precompute_invariants(
    catalog: MolecularLineCatalog,
    wavelength_grid_nm,
    harris_profile_h0_table,
    harris_profile_h1_table,
    harris_profile_h2_table,
    runtime_device=None,
) -> MolecularLineInvariants:
    """Build resident tensors from a molecular catalog and synthesis grid."""
    compute_device = runtime_device if runtime_device is not None else device()
    work_dtype = highp_dtype(compute_device)
    wavelength_grid = np.asarray(wavelength_grid_nm, np.float64)
    n_wavelengths = wavelength_grid.size

    species_code_per_line = catalog.species_code.astype(np.int64)
    unique_species_codes = catalog.unique_species_codes.astype(np.int64)
    species_population_column = unique_species_codes // 6 - 1
    species_mass = np.array(
        [SPECIES_MASS_AMU.get(int(code), 20.0) for code in unique_species_codes],
        dtype=np.float64,
    )

    # Map each line species to a compact species table column.
    sorted_species_positions = np.argsort(unique_species_codes)
    line_species_index = sorted_species_positions[
        np.searchsorted(
            unique_species_codes,
            species_code_per_line,
            sorter=sorted_species_positions,
        )
    ]

    # Keep this differencing in float64; optical grids are close enough that fp32
    # loses resolving-power precision before the value is stored on device.
    local_resolving_power_array = _local_resolving_power_array(wavelength_grid)

    def to_device_tensor(array, dtype):
        return torch.as_tensor(array, dtype=dtype).to(compute_device)

    return MolecularLineInvariants(
        wavelength_grid=to_device_tensor(wavelength_grid, work_dtype),
        n_wavelengths=n_wavelengths,
        local_resolving_power=to_device_tensor(local_resolving_power_array, work_dtype),
        classical_line_strength=to_device_tensor(
            catalog.classical_line_strength, DEFAULT_DTYPE
        ),
        lower_excitation_cm=to_device_tensor(
            catalog.lower_excitation_cm.astype(np.float32),
            work_dtype,
        ),
        radiative_damping=to_device_tensor(catalog.radiative_damping, DEFAULT_DTYPE),
        stark_damping=to_device_tensor(catalog.stark_damping, DEFAULT_DTYPE),
        van_der_waals_damping=to_device_tensor(
            catalog.van_der_waals_damping, DEFAULT_DTYPE
        ),
        center_index=to_device_tensor(
            catalog.center_index.astype(np.int64), torch.int64
        ),
        line_species_index=to_device_tensor(line_species_index, torch.int64),
        line_wavelength_nm=to_device_tensor(
            catalog.wavelength_nm.astype(np.float32), work_dtype
        ),
        species_code=to_device_tensor(unique_species_codes, torch.int64),
        species_population_column=to_device_tensor(
            species_population_column, torch.int64
        ),
        species_mass_amu=to_device_tensor(species_mass, work_dtype),
        harris_profile_h0_table=to_device_tensor(harris_profile_h0_table, work_dtype),
        harris_profile_h1_table=to_device_tensor(harris_profile_h1_table, work_dtype),
        harris_profile_h2_table=to_device_tensor(harris_profile_h2_table, work_dtype),
    )


def species_population_doppler_ratio(invariants: MolecularLineInvariants, state: dict):
    """Return population/(mass_density * Doppler fraction) per depth and species."""
    compute_device = invariants.wavelength_grid.device
    work_dtype = invariants.wavelength_grid.dtype
    temperature = to_dev(state["temperature"], work_dtype, compute_device)
    mass_density = to_dev(state["mass_density"], work_dtype, compute_device)
    microturbulence = to_dev(state["microturbulence"], work_dtype, compute_device)
    partition_normalized_populations = to_dev(
        state["partition_normalized_populations"], work_dtype, compute_device
    )

    species_mass = invariants.species_mass_amu[None, :]
    thermal_velocity_squared = (
        2.0
        * BOLTZMANN_ERG_PER_K
        * temperature[:, None]
        / (species_mass * ATOMIC_MASS_GRAM)
    )
    doppler_fraction = (
        torch.sqrt(thermal_velocity_squared + (microturbulence[:, None]) ** 2)
        / LIGHT_SPEED_CM_PER_S
    )

    molecular_population_slot = partition_normalized_populations[:, 5, :]
    species_population = molecular_population_slot[
        :, invariants.species_population_column
    ]

    valid = (
        (species_population > 0) & (doppler_fraction > 0) & (mass_density[:, None] > 0)
    )
    density_safe = torch.where(
        mass_density[:, None] > 0,
        mass_density[:, None],
        torch.ones_like(mass_density[:, None]),
    )
    doppler_safe = torch.where(
        doppler_fraction > 0,
        doppler_fraction,
        torch.ones_like(doppler_fraction),
    )
    population_doppler_ratio = torch.where(
        valid,
        species_population / (density_safe * doppler_safe),
        torch.zeros_like(species_population),
    )
    return population_doppler_ratio, doppler_fraction


def _voigt(voigt_offset, damping_ratio, invariants):
    """Tabulated Voigt H(a, |v|)."""
    return interpolate_harris_profile(
        voigt_offset,
        damping_ratio,
        invariants.harris_profile_h0_table,
        invariants.harris_profile_h1_table,
        invariants.harris_profile_h2_table,
    )


def accumulate_molecular(
    invariants: MolecularLineInvariants,
    state: dict,
    apply_stim: bool = True,
    chunk_lines: Optional[int] = None,
) -> torch.Tensor:
    """Accumulate molecular band opacity in on-device line chunks."""
    compute_device = invariants.wavelength_grid.device
    work_dtype = invariants.wavelength_grid.dtype
    # Keep the validated invariant precision and fixed accumulation policy.
    # Lower-precision runtime overrides are intentionally unsupported.
    eval_dtype = work_dtype
    accumulator_dtype = ACCUMULATION_DTYPE
    n_depths = to_dev(state["temperature"], work_dtype, compute_device).shape[0]
    n_wavelengths = invariants.n_wavelengths

    electron_density = to_dev(state["electron_density"], DEFAULT_DTYPE, compute_device)
    hc_over_kt = to_dev(
        state["hc_over_kt"],
        eval_dtype,
        compute_device,
    )
    collision_density_proxy = to_dev(
        state["collision_density_proxy"],
        DEFAULT_DTYPE,
        compute_device,
    )
    continuum_opacity = to_dev(state["continuum_opacity"], eval_dtype, compute_device)

    population_doppler_ratio_by_species, doppler_fraction_by_species = (
        species_population_doppler_ratio(invariants, state)
    )
    population_doppler_ratio_by_species = population_doppler_ratio_by_species.to(
        eval_dtype
    )
    doppler_fraction_by_species = doppler_fraction_by_species.to(eval_dtype)

    chunk_invariants = invariants
    if eval_dtype != work_dtype:
        chunk_invariants = SimpleNamespace(
            wavelength_grid=invariants.wavelength_grid,
            harris_profile_h0_table=invariants.harris_profile_h0_table.to(eval_dtype),
            harris_profile_h1_table=invariants.harris_profile_h1_table.to(eval_dtype),
            harris_profile_h2_table=invariants.harris_profile_h2_table.to(eval_dtype),
            classical_line_strength=invariants.classical_line_strength,
            lower_excitation_cm=invariants.lower_excitation_cm.to(eval_dtype),
            radiative_damping=invariants.radiative_damping,
            stark_damping=invariants.stark_damping,
            van_der_waals_damping=invariants.van_der_waals_damping,
            center_index=invariants.center_index,
            line_species_index=invariants.line_species_index,
            line_wavelength_nm=invariants.line_wavelength_nm.to(eval_dtype),
        )

    molecular_opacity = torch.zeros(
        (n_depths, n_wavelengths),
        dtype=accumulator_dtype,
        device=compute_device,
    )

    local_resolving_power = invariants.local_resolving_power.to(eval_dtype)

    n_molecular_lines = invariants.center_index.numel()
    chunk_lines = (
        molecular_chunk_lines() if chunk_lines is None else max(1, int(chunk_lines))
    )
    for line_start in range(0, n_molecular_lines, chunk_lines):
        line_stop = min(line_start + chunk_lines, n_molecular_lines)
        _accumulate_chunk(
            molecular_opacity,
            chunk_invariants,
            line_start,
            line_stop,
            population_doppler_ratio_by_species,
            doppler_fraction_by_species,
            electron_density,
            hc_over_kt,
            collision_density_proxy,
            continuum_opacity,
            local_resolving_power,
            n_wavelengths,
            eval_dtype,
        )
    if molecular_opacity.dtype != ACCUMULATION_DTYPE and compute_device.type == "cuda":
        molecular_opacity = molecular_opacity.to(ACCUMULATION_DTYPE)

    if apply_stim:
        temperature = to_dev(state["temperature"], work_dtype, compute_device)
        frequency_grid_hz = (LIGHT_SPEED_NM_PER_S / invariants.wavelength_grid).to(
            work_dtype
        )
        planck_over_thermal_energy = (
            PLANCK_ERG_SECOND
            / (BOLTZMANN_ERG_PER_K * torch.clamp(temperature, min=1.0))
        ).to(work_dtype)
        stimulated_emission_factor = 1.0 - torch.exp(
            -frequency_grid_hz[None, :] * planck_over_thermal_energy[:, None]
        )
        molecular_opacity = (
            molecular_opacity.to(work_dtype) * stimulated_emission_factor
        ).to(ACCUMULATION_DTYPE)
    return molecular_opacity


def _local_resolving_power_array(wavelength_grid: np.ndarray) -> np.ndarray:
    """Per-pixel local resolving power, computed in float64."""
    wavelength_grid = np.asarray(wavelength_grid, np.float64)
    n_wavelengths = wavelength_grid.size
    if n_wavelengths == 1:
        return np.full(1, 300000.0, dtype=np.float64)
    local_resolving_power = np.empty(n_wavelengths, dtype=np.float64)
    local_resolving_power[:-1] = 1.0 / (
        wavelength_grid[1:] / wavelength_grid[:-1] - 1.0
    )
    local_resolving_power[-1] = 1.0 / (wavelength_grid[-1] / wavelength_grid[-2] - 1.0)
    return local_resolving_power


def _accumulate_chunk(
    molecular_opacity,
    invariants,
    line_start,
    line_stop,
    population_doppler_ratio_by_species,
    doppler_fraction_by_species,
    electron_density,
    hc_over_kt,
    collision_density_proxy,
    continuum_opacity,
    local_resolving_power,
    n_wavelengths,
    work_dtype,
):
    """Accumulate one on-device molecular line chunk."""
    classical_strength = invariants.classical_line_strength[line_start:line_stop].to(
        work_dtype
    )
    lower_excitation_cm = invariants.lower_excitation_cm[line_start:line_stop]
    radiative_damping = invariants.radiative_damping[line_start:line_stop].to(
        work_dtype
    )
    stark_damping = invariants.stark_damping[line_start:line_stop].to(work_dtype)
    van_der_waals_damping = invariants.van_der_waals_damping[line_start:line_stop].to(
        work_dtype
    )
    line_species_index = invariants.line_species_index[line_start:line_stop]
    center_index = invariants.center_index[line_start:line_stop]
    wavelength_nm = invariants.line_wavelength_nm[line_start:line_stop]

    population_doppler_ratio = population_doppler_ratio_by_species[
        :, line_species_index
    ]
    doppler_fraction = doppler_fraction_by_species[:, line_species_index]

    center_index_clamped = torch.clamp(center_index, 0, n_wavelengths - 1)
    opacity_floor = (
        LINE_CENTER_CUTOFF_RATIO * continuum_opacity[:, center_index_clamped]
    )

    pre_excitation_strength = classical_strength[None, :] * population_doppler_ratio
    excitation_factor = torch.exp(-lower_excitation_cm[None, :] * hc_over_kt[:, None])
    line_amplitude = pre_excitation_strength * excitation_factor
    damping_ratio_raw = (
        radiative_damping[None, :]
        + stark_damping[None, :] * electron_density[:, None].to(work_dtype)
        + van_der_waals_damping[None, :]
        * collision_density_proxy[:, None].to(work_dtype)
    )
    doppler_safe = torch.where(
        doppler_fraction > 0,
        doppler_fraction,
        torch.ones_like(doppler_fraction),
    )
    damping_ratio_raw = damping_ratio_raw / doppler_safe
    keep = (
        (population_doppler_ratio > 0)
        & (doppler_fraction > 0)
        & (wavelength_nm[None, :] > 0)
        & (pre_excitation_strength >= opacity_floor)
        & (line_amplitude > 0)
        & (line_amplitude >= opacity_floor)
        & (damping_ratio_raw >= 0)
    )
    if not bool(keep.any()):
        return

    # Flatten the surviving depth-line pairs before the wing walk.
    pair_positions = torch.nonzero(keep, as_tuple=False)
    depth_row_all = pair_positions[:, 0]
    line_column_all = pair_positions[:, 1]
    n_pairs = depth_row_all.numel()

    for pair_start in range(0, n_pairs, PAIR_CHUNK):
        pair_stop = min(pair_start + PAIR_CHUNK, n_pairs)
        depth_row = depth_row_all[pair_start:pair_stop]
        line_column = line_column_all[pair_start:pair_stop]

        pair_line_amplitude = line_amplitude[depth_row, line_column]
        damping_ratio = torch.clamp(
            damping_ratio_raw[depth_row, line_column], min=1e-12
        )
        pair_opacity_floor = opacity_floor[depth_row, line_column]
        pair_center_index = center_index[line_column]
        pair_doppler_fraction = doppler_fraction[depth_row, line_column]
        pair_resolving_power = local_resolving_power[center_index_clamped[line_column]]
        small_damping = damping_ratio < 0.2

        center_profile = torch.where(
            damping_ratio < 0.2,
            1.0 - 1.128 * damping_ratio,
            _voigt(torch.zeros_like(damping_ratio), damping_ratio, invariants),
        )
        center_opacity = pair_line_amplitude * center_profile
        center_on_grid = (pair_center_index >= 0) & (pair_center_index < n_wavelengths)
        _scatter_add_flat(
            molecular_opacity,
            depth_row[center_on_grid],
            torch.clamp(pair_center_index[center_on_grid], 0, n_wavelengths - 1),
            center_opacity[center_on_grid],
            n_wavelengths,
        )

        doppler_grid_width = pair_doppler_fraction * pair_resolving_power
        ten_doppler_steps = torch.clamp(
            (10.0 * doppler_grid_width).to(torch.int64),
            max=MAX_WING_PROFILE_STEPS,
        )
        table_step_scale = torch.where(
            doppler_grid_width > 0,
            200.0 / doppler_grid_width,
            torch.full_like(doppler_grid_width, 200.0),
        )
        voigt_coordinate_scale = torch.where(
            doppler_grid_width > 0,
            1.0 / doppler_grid_width,
            torch.full_like(doppler_grid_width, 1e-6),
        )

        pair = dict(
            depth_row=depth_row,
            center_index=pair_center_index,
            line_amplitude=pair_line_amplitude,
            damping_ratio=damping_ratio,
            small_damping=small_damping,
            table_step_scale=table_step_scale,
            voigt_coordinate_scale=voigt_coordinate_scale,
            ten_doppler_steps=ten_doppler_steps,
            opacity_floor=pair_opacity_floor,
        )

        max_ten_doppler_steps = int(ten_doppler_steps.max().item())
        early_cutoff = torch.zeros_like(ten_doppler_steps, dtype=torch.bool)
        profile_at_ten_doppler = torch.zeros_like(pair_line_amplitude)
        if max_ten_doppler_steps >= 1:
            early_cutoff, profile_at_ten_doppler = _near_wing(
                molecular_opacity,
                invariants,
                pair,
                n_wavelengths,
                max_ten_doppler_steps,
            )

        far_wing_active = (
            (~early_cutoff) & (ten_doppler_steps > 0) & (profile_at_ten_doppler > 0)
        )
        if bool(far_wing_active.any()):
            _far_wing(
                molecular_opacity,
                pair,
                profile_at_ten_doppler,
                far_wing_active,
                n_wavelengths,
            )


def _near_wing_profile(
    steps,
    line_amplitude,
    damping_ratio,
    small_damping,
    table_step_scale,
    voigt_coordinate_scale,
    invariants,
):
    """Evaluate the near-wing profile for a block of integer offsets."""
    step_axis = steps.view(1, -1)
    table_index_float = 0.5 + step_axis * table_step_scale[:, None]
    table_index = torch.clamp(
        table_index_float.to(torch.int64),
        0,
        invariants.harris_profile_h0_table.numel() - 1,
    )
    profile_value = line_amplitude[:, None] * (
        invariants.harris_profile_h0_table[table_index]
        + damping_ratio[:, None] * invariants.harris_profile_h1_table[table_index]
    )
    broad_damping_rows = torch.nonzero(~small_damping, as_tuple=False).squeeze(1)
    if broad_damping_rows.numel():
        voigt_coordinate = (
            step_axis * voigt_coordinate_scale[broad_damping_rows][:, None]
        )
        profile_value[broad_damping_rows] = line_amplitude[broad_damping_rows][
            :, None
        ] * _voigt(
            voigt_coordinate,
            damping_ratio[broad_damping_rows][:, None],
            invariants,
        )
    return profile_value


def _near_wing(
    molecular_opacity,
    invariants,
    pair,
    n_wavelengths,
    max_ten_doppler_steps,
):
    """Deposit near wings and return the far-wing seed for each pair."""
    compute_device = molecular_opacity.device
    depth_row = pair["depth_row"]
    center_index = pair["center_index"]
    line_amplitude = pair["line_amplitude"]
    damping_ratio = pair["damping_ratio"]
    small_damping = pair["small_damping"]
    table_step_scale = pair["table_step_scale"]
    voigt_coordinate_scale = pair["voigt_coordinate_scale"]
    ten_doppler_steps = pair["ten_doppler_steps"]
    opacity_floor = pair["opacity_floor"]
    offset_chunk = 256

    reach = ten_doppler_steps.clone()
    cutoff_seen = torch.zeros_like(ten_doppler_steps, dtype=torch.bool)
    profile_at_ten_doppler = torch.zeros_like(line_amplitude)
    step_start = 1
    while step_start <= max_ten_doppler_steps:
        step_stop = min(step_start + offset_chunk - 1, max_ten_doppler_steps)
        steps = torch.arange(step_start, step_stop + 1, device=compute_device)
        step_axis = steps.view(1, -1)
        within_near_wing = step_axis <= ten_doppler_steps[:, None]
        profile_value = _near_wing_profile(
            steps,
            line_amplitude,
            damping_ratio,
            small_damping,
            table_step_scale,
            voigt_coordinate_scale,
            invariants,
        )

        below_floor = within_near_wing & (profile_value < opacity_floor[:, None])
        crosses_in_block = below_floor.any(dim=1)
        first_cross_step = step_start + torch.argmax(below_floor.to(torch.int8), dim=1)
        first_crosses_here = crosses_in_block & (~cutoff_seen)
        reach = torch.where(first_crosses_here, first_cross_step, reach)

        block_stop_step = torch.where(
            first_crosses_here,
            first_cross_step,
            torch.full_like(first_cross_step, step_stop),
        )
        active = (~cutoff_seen)[:, None] & (step_axis <= block_stop_step[:, None])
        deposit = within_near_wing & active
        red_column = center_index[:, None] + step_axis
        blue_column = center_index[:, None] - step_axis
        depth_rows = depth_row[:, None].expand(-1, steps.numel())
        red_on_grid = deposit & (red_column >= 0) & (red_column < n_wavelengths)
        blue_on_grid = deposit & (blue_column >= 0) & (blue_column < n_wavelengths)
        _scatter_masked(
            molecular_opacity,
            depth_rows,
            red_column,
            profile_value,
            red_on_grid,
            n_wavelengths,
        )
        _scatter_masked(
            molecular_opacity,
            depth_rows,
            blue_column,
            profile_value,
            blue_on_grid,
            n_wavelengths,
        )

        at_ten_doppler = (
            (step_axis == ten_doppler_steps[:, None])
            & within_near_wing
            & (~cutoff_seen[:, None])
        )
        no_cross_to_ten_doppler = ~(
            below_floor & (step_axis <= ten_doppler_steps[:, None])
        ).any(dim=1)
        set_seed = at_ten_doppler.any(dim=1) & no_cross_to_ten_doppler
        profile_at_ten_doppler = torch.where(
            set_seed,
            (profile_value * at_ten_doppler).sum(dim=1),
            profile_at_ten_doppler,
        )

        cutoff_seen = cutoff_seen | crosses_in_block
        step_start = step_stop + 1

    return cutoff_seen, profile_at_ten_doppler


def _far_wing(
    molecular_opacity,
    pair,
    profile_at_ten_doppler,
    far_wing_active,
    n_wavelengths,
):
    """Deposit far wings with the same irreversible edge break as the reference."""
    compute_device = molecular_opacity.device
    selected = torch.nonzero(far_wing_active, as_tuple=False).squeeze(1)
    if selected.numel() == 0:
        return
    depth_row = pair["depth_row"][selected]
    center_index = pair["center_index"][selected]
    ten_doppler_steps = pair["ten_doppler_steps"][selected]
    opacity_floor = pair["opacity_floor"][selected]
    far_wing_scale = (
        profile_at_ten_doppler[selected]
        * ten_doppler_steps.to(profile_at_ten_doppler.dtype) ** 2
    )

    has_floor = opacity_floor > 0
    opacity_floor_safe = torch.where(
        has_floor, opacity_floor, torch.ones_like(opacity_floor)
    )
    max_step = torch.where(
        has_floor,
        torch.clamp(
            (torch.sqrt(far_wing_scale / opacity_floor_safe) + 1.0).to(torch.int64),
            max=MAX_WING_PROFILE_STEPS,
        ),
        torch.full_like(ten_doppler_steps, MAX_WING_PROFILE_STEPS),
    )
    farthest_step = int(max_step.max().item())
    if farthest_step < 1:
        return

    alive = torch.ones_like(ten_doppler_steps, dtype=torch.bool)
    offset_chunk = 256
    step_start = 1
    while step_start <= farthest_step:
        step_stop = min(step_start + offset_chunk - 1, farthest_step)
        steps = torch.arange(step_start, step_stop + 1, device=compute_device)
        step_axis = steps.view(1, -1)
        beyond_near_wing = step_axis > ten_doppler_steps[:, None]
        within = (step_axis <= max_step[:, None]) & alive[:, None] & beyond_near_wing
        step_float = step_axis.to(far_wing_scale.dtype)
        profile_value = far_wing_scale[:, None] / (step_float * step_float)
        red_column = center_index[:, None] + step_axis
        blue_column = center_index[:, None] - step_axis
        red_on_grid = (red_column >= 0) & (red_column < n_wavelengths)
        blue_on_grid = (blue_column >= 0) & (blue_column < n_wavelengths)

        neither_on_grid = within & ~(red_on_grid | blue_on_grid)
        n_offsets = steps.numel()
        first_kill_offset = torch.where(
            neither_on_grid.any(dim=1),
            torch.argmax(neither_on_grid.to(torch.int8), dim=1),
            torch.full((neither_on_grid.shape[0],), n_offsets, device=compute_device),
        )
        offset_axis = torch.arange(n_offsets, device=compute_device).view(1, -1)
        before_first_kill = offset_axis <= first_kill_offset[:, None]
        deposit = within & before_first_kill
        depth_rows = depth_row[:, None].expand(-1, n_offsets)
        red_ok = deposit & red_on_grid
        blue_ok = deposit & blue_on_grid
        _scatter_masked(
            molecular_opacity,
            depth_rows,
            red_column,
            profile_value,
            red_ok,
            n_wavelengths,
        )
        _scatter_masked(
            molecular_opacity,
            depth_rows,
            blue_column,
            profile_value,
            blue_ok,
            n_wavelengths,
        )

        alive = alive & (~neither_on_grid.any(dim=1)) & (max_step > step_stop)
        step_start = step_stop + 1
        if not bool(alive.any()):
            break


def _scatter_add_flat(molecular_opacity, depth_rows, columns, values, n_wavelengths):
    """Add flat depth-column contributions into the opacity accumulator."""
    if depth_rows.numel() == 0:
        return
    flat_index = depth_rows.to(torch.int64) * n_wavelengths + columns.to(torch.int64)
    molecular_opacity.view(-1).index_put_(
        (flat_index,),
        values.to(molecular_opacity.dtype),
        accumulate=True,
    )


def _scatter_masked(
    molecular_opacity,
    depth_rows_2d,
    columns_2d,
    values_2d,
    mask_2d,
    n_wavelengths,
):
    """Flatten a masked pair-offset block before scatter-add."""
    if not bool(mask_2d.any()):
        return
    mask = mask_2d.reshape(-1)
    depth_rows = depth_rows_2d.reshape(-1)[mask]
    columns = columns_2d.reshape(-1)[mask]
    values = values_2d.reshape(-1)[mask]
    _scatter_add_flat(molecular_opacity, depth_rows, columns, values, n_wavelengths)


__all__ = [
    "MolecularLineCatalog",
    "MolecularLineInvariants",
    "load_catalog",
    "build_catalog_from_arrays",
    "precompute_invariants",
    "species_population_doppler_ratio",
    "accumulate_molecular",
    "SPECIES_MASS_AMU",
]
