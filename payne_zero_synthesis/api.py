"""Public structured-atmosphere spectrum-synthesis API.

`synthesize` consumes the native NPZ product. `build_structured_atmosphere`
and `save_structured_atmosphere` construct that product from solver columns.
Everything else in the package is engine machinery behind these calls.

Repeated calls over the same (window, resolution, device, dtype) reuse the
device-resident window invariants through an in-process cache keyed only by
those physical inputs plus catalog/table file identity; disable it with
PAYNE_ZERO_SYNTHESIS_DISABLE_INVARIANT_CACHE=1 (identical outputs, slower) or
drop it with `clear_window_invariant_cache()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Mapping
import warnings

import numpy as np
import torch

from . import synthesis as _synthesis_engine
from .atmosphere import (
    ATMOSPHERE_PRODUCT_METADATA_SCHEMA_VERSION,
    ATMOSPHERE_SCHEMA_VERSION,
    load_atmosphere_npz,
    validate_atmosphere_npz,
)
from .pipeline import (  # noqa: F401 - public re-exports
    clear_window_invariant_cache,
    window_invariant_cache_enabled,
)


SPEED_OF_LIGHT_NM_S = 2.99792458e17
FOUR_PI = 4.0 * np.pi


@dataclass(frozen=True)
class Spectrum:
    """Spectrum and timing on a wavelength grid.

    ``flux_total`` and ``flux_continuum`` are spectral flux densities per
    nanometer. ``normalized_flux`` is their dimensionless ratio.
    """

    wavelength_nm: np.ndarray
    flux_total: np.ndarray
    flux_continuum: np.ndarray
    normalized_flux: np.ndarray
    seconds: float

    def _npz_payload(self) -> dict[str, np.ndarray]:
        return {
            "wavelength_nm": self.wavelength_nm,
            "flux_total": self.flux_total,
            "flux_continuum": self.flux_continuum,
            "normalized_flux": self.normalized_flux,
            "seconds": np.asarray([self.seconds], np.float64),
        }

    def save_npz(self, path: str | Path) -> None:
        output_path = Path(path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **self._npz_payload())


@dataclass(frozen=True)
class ForwardTimings:
    """Wall times for the stages of a label-driven spectrum calculation."""

    initializer_seconds: float
    population_bridge_seconds: float
    synthesis_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class InitializedAtmosphere:
    """Population-bridged atmosphere predicted from stellar labels.

    This product is ready for synthesis or line-list calibration without an
    intermediate file.  It is an initializer prediction rather than a
    physically converged atmosphere, which is made explicit by the two safety
    fields.
    """

    structured_atmosphere: dict[str, np.ndarray] = field(repr=False)
    initializer_family: str
    labels: dict[str, object]
    provenance: dict[str, object]
    timings: ForwardTimings
    atmosphere_converged: bool = field(default=False, init=False)
    atmosphere_closure_required: bool = field(default=True, init=False)

    def save_npz(self, path: str | Path) -> tuple[str, ...]:
        """Save a reusable atmosphere marked as an unconverged initializer."""

        detailed_metadata = {
            "atmosphere_product_role": "learned_initializer_prediction",
            "initializer_family": self.initializer_family,
            "labels": self.labels,
            "provenance": self.provenance,
            "timings": {
                "initializer_seconds": self.timings.initializer_seconds,
                "population_bridge_seconds": self.timings.population_bridge_seconds,
                "total_seconds": self.timings.total_seconds,
            },
            "atmosphere_converged": self.atmosphere_converged,
            "atmosphere_closure_required": self.atmosphere_closure_required,
        }
        product_arrays = {
            "atmosphere_product_metadata_schema": np.asarray(
                [ATMOSPHERE_PRODUCT_METADATA_SCHEMA_VERSION], np.int32
            ),
            "atmosphere_product_role": np.asarray(
                "learned_initializer_prediction"
            ),
            "atmosphere_converged": np.asarray(
                [self.atmosphere_converged], np.bool_
            ),
            "atmosphere_closure_required": np.asarray(
                [self.atmosphere_closure_required], np.bool_
            ),
            "initializer_family": np.asarray(self.initializer_family),
            "atmosphere_metadata_json": np.asarray(
                json.dumps(detailed_metadata, sort_keys=True)
            ),
        }
        return _save_structured_atmosphere(
            self.structured_atmosphere,
            path,
            product_arrays=product_arrays,
        )


@dataclass(frozen=True)
class LabelSpectrum(Spectrum):
    """Spectrum synthesized from a learned atmosphere initializer."""

    initializer_family: str
    labels: dict[str, object]
    provenance: dict[str, object]
    timings: ForwardTimings
    initialized_atmosphere: InitializedAtmosphere = field(repr=False)
    atmosphere_converged: bool = field(default=False, init=False)
    atmosphere_closure_required: bool = field(default=True, init=False)

    def _npz_payload(self) -> dict[str, np.ndarray]:
        payload = super()._npz_payload()
        payload.update(
            {
                "initializer_family": np.asarray(self.initializer_family),
                "atmosphere_converged": np.asarray(
                    [self.atmosphere_converged], np.bool_
                ),
                "atmosphere_closure_required": np.asarray(
                    [self.atmosphere_closure_required], np.bool_
                ),
                "initializer_seconds": np.asarray(
                    [self.timings.initializer_seconds], np.float64
                ),
                "population_bridge_seconds": np.asarray(
                    [self.timings.population_bridge_seconds], np.float64
                ),
                "synthesis_seconds": np.asarray(
                    [self.timings.synthesis_seconds], np.float64
                ),
                "metadata_json": np.asarray(
                    json.dumps(
                        {
                            "initializer_family": self.initializer_family,
                            "labels": self.labels,
                            "provenance": self.provenance,
                            "atmosphere_converged": self.atmosphere_converged,
                            "atmosphere_closure_required": (
                                self.atmosphere_closure_required
                            ),
                        },
                        sort_keys=True,
                    )
                ),
            }
        )
        return payload


def _torch_dtype(name: str | torch.dtype | None) -> torch.dtype | None:
    if name is None or name == "auto":
        return None
    if isinstance(name, torch.dtype):
        return name
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype {name!r}")


def _runtime_device(
    name: str | torch.device | None,
) -> str | torch.device | None:
    """Translate the public ``auto`` spelling to the engine's default policy."""

    if name is None or name == "auto":
        return None
    if isinstance(name, torch.device):
        return name
    if name in {"cpu", "cuda", "mps"}:
        return name
    raise ValueError(f"unsupported device {name!r}")


def _resolved_r_grid(
    *,
    r_grid: float | None,
    resolution: float | None,
) -> float:
    if r_grid is not None and resolution is not None:
        if float(r_grid) != float(resolution):
            raise ValueError("r_grid and resolution specify different values")
    value = (
        20_000.0
        if r_grid is None and resolution is None
        else (r_grid if r_grid is not None else resolution)
    )
    resolved = float(value)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError("r_grid must be finite and positive")
    return resolved


@lru_cache(maxsize=8)
def _file_sha256(path_text: str) -> str:
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _initializer_checkpoint_provenance(
    family: str,
    *,
    five_label_path: str | Path | None,
    cno8_path: str | Path | None,
) -> dict[str, object]:
    from payne_zero_atmosphere.warm_start import (
        CNO8_FAMILY,
        DEFAULT_CNO8_WEIGHTS_PATH,
        DEFAULT_FIVE_LABEL_WEIGHTS_PATH,
    )

    checkpoint = (
        Path(
            (five_label_path or DEFAULT_FIVE_LABEL_WEIGHTS_PATH)
            if family != CNO8_FAMILY
            else (cno8_path or DEFAULT_CNO8_WEIGHTS_PATH)
        )
        .expanduser()
        .resolve()
    )
    return {
        "role": "learned_atmosphere_initializer",
        "initializer_family": family,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _file_sha256(str(checkpoint)),
        "is_final_atmosphere": False,
        "atmosphere_closure_required": True,
    }


def _normalize_sparse_direct_xh(
    abundances: Mapping[int | str, float],
) -> dict[int, float]:
    from payne_zero_atmosphere.warm_start import ATOMIC_NUMBER_BY_SYMBOL

    normalized: dict[int, float] = {}
    for key, value in abundances.items():
        if isinstance(key, (bool, np.bool_)):
            raise ValueError("direct abundance keys must be symbols or atomic numbers")
        if isinstance(key, str):
            text = key.strip().lower()
            if text not in ATOMIC_NUMBER_BY_SYMBOL:
                raise ValueError(f"unknown element symbol {key!r}")
            atomic_number = int(ATOMIC_NUMBER_BY_SYMBOL[text])
        else:
            try:
                atomic_number = int(key)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "direct abundance keys must be symbols or atomic numbers"
                ) from error
            if atomic_number != key:
                raise ValueError(f"abundance key {key!r} is not an atomic number")
        abundance = float(value)
        if not np.isfinite(abundance):
            raise ValueError(f"[Z={atomic_number}/H] must be finite")
        if atomic_number in normalized:
            raise ValueError(f"duplicate direct abundance for Z={atomic_number}")
        normalized[atomic_number] = abundance
    return normalized


def _surface_flux_per_wavelength_nm(
    wavelength_nm: np.ndarray,
    eddington_flux_per_frequency: np.ndarray,
) -> np.ndarray:
    """Convert Eddington ``H_nu`` to physical surface ``F_lambda`` per nm."""

    wavelength_nm_array = np.asarray(wavelength_nm, np.float64)
    if np.any(wavelength_nm_array <= 0.0):
        raise ValueError("wavelength_nm must be strictly positive")
    return (
        FOUR_PI
        * np.asarray(eddington_flux_per_frequency, np.float64)
        * SPEED_OF_LIGHT_NM_S
        / np.square(wavelength_nm_array)
    )


def _wrap(result, seconds: float) -> Spectrum:
    wavelength_nm = np.asarray(result.wavelength_nm, np.float64)
    return Spectrum(
        wavelength_nm=wavelength_nm,
        flux_total=_surface_flux_per_wavelength_nm(
            wavelength_nm, result.eddington_flux_total_per_frequency
        ),
        flux_continuum=_surface_flux_per_wavelength_nm(
            wavelength_nm, result.eddington_flux_continuum_per_frequency
        ),
        normalized_flux=np.asarray(result.normalized_flux, np.float64),
        seconds=float(seconds),
    )


def build_structured_atmosphere(
    *,
    temperature,
    column_mass,
    gas_pressure,
    electron_density,
    elemental_abundances,
    mean_nuclear_mass_amu: float | None = None,
    microturbulence=None,
    mass_density=None,
    molecular_lines: bool = True,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    eos_tolerance: float = 1.0e-5,
) -> dict[str, np.ndarray]:
    """Build the native atmosphere mapping from solver depth columns.

    Use this when an atmosphere solver already has converged thermodynamic
    columns and needs to hand them to Payne Zero synthesis without writing an
    intermediate text deck. The returned mapping uses the public field names
    documented in ``atmosphere_schema.json``.

    ``temperature`` [K], ``column_mass`` [g cm^-2], ``gas_pressure``
    [dyne cm^-2], and ``electron_density`` [cm^-3] must be finite
    one-dimensional arrays with the same outer-to-inner depth ordering;
    ``column_mass`` must increase with index. ``elemental_abundances`` has
    shape ``(99,)`` and stores linear number fractions for atomic numbers
    1 through 99 at indices 0 through 98. It is not a log-epsilon or ``[X/H]``
    array. ``microturbulence`` may be a scalar or one value per layer in
    cm s^-1. ``mass_density``, when supplied, is one value per layer in
    g cm^-3.
    """

    return _synthesis_engine.build_structured_atmosphere_from_columns(
        temperature=temperature,
        column_mass=column_mass,
        gas_pressure=gas_pressure,
        electron_density=electron_density,
        elemental_abundances=elemental_abundances,
        mean_nuclear_mass_amu=mean_nuclear_mass_amu,
        microturbulence=microturbulence,
        mass_density=mass_density,
        device=_runtime_device(device),
        dtype=_torch_dtype(dtype),
        molecular_lines=molecular_lines,
        eos_tolerance=eos_tolerance,
    )


def _save_structured_atmosphere(
    atmosphere: dict[str, np.ndarray],
    path: str | Path,
    *,
    product_arrays: Mapping[str, np.ndarray] | None = None,
) -> tuple[str, ...]:
    """Write and validate a native atmosphere with optional product metadata."""

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(value) for name, value in atmosphere.items()}
    extension = {
        name: np.asarray(value) for name, value in (product_arrays or {}).items()
    }
    overlap = sorted(set(arrays).intersection(extension))
    if overlap:
        raise ValueError(
            "atmosphere product metadata collides with physical arrays: "
            + ", ".join(overlap)
        )
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    output_path.unlink(missing_ok=True)
    temporary_path.unlink(missing_ok=True)
    try:
        with temporary_path.open("wb") as handle:
            np.savez(handle, **arrays)
        canonical_arrays = load_atmosphere_npz(temporary_path)
        canonical_arrays["atmosphere_schema_version"] = np.asarray(
            [ATMOSPHERE_SCHEMA_VERSION], dtype=np.int32
        )
        canonical_arrays.update(extension)
        with temporary_path.open("wb") as handle:
            np.savez(handle, **canonical_arrays)
        names = validate_atmosphere_npz(temporary_path)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return names


def save_structured_atmosphere(
    atmosphere: dict[str, np.ndarray],
    path: str | Path,
) -> tuple[str, ...]:
    """Write and validate a native structured atmosphere NPZ."""

    return _save_structured_atmosphere(atmosphere, path)


def synthesize(
    atmosphere_npz: str | Path | Mapping[str, np.ndarray] | InitializedAtmosphere,
    *,
    wavelength_start_nm: float = 400.0,
    wavelength_end_nm: float = 900.0,
    resolution: float = 20000.0,
    molecular_lines: bool = True,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    spectral_operator=None,
) -> Spectrum:
    """Synthesize from a native atmosphere file or in-memory atmosphere.

    ``spectral_operator`` is an optional prepared device-resident operator that
    transforms wavelength-density total and continuum flux before their one
    final host transfer. It must expose a one-dimensional
    ``output_wavelength_nm`` array and a
    ``convolve_fluxes(total_flux, continuum_flux)`` method. The method receives
    two one-dimensional Torch tensors on the synthesis device and returns
    projected total, continuum, and normalized-flux tensors of the same length
    as ``output_wavelength_nm``. An :class:`InitializedAtmosphere` can be passed
    directly, so label-driven synthesis and line-list calibration can share
    exactly the same population-bridged state without a temporary NPZ.
    """

    if isinstance(atmosphere_npz, InitializedAtmosphere):
        atmosphere = atmosphere_npz.structured_atmosphere
    elif isinstance(atmosphere_npz, Mapping):
        atmosphere = dict(atmosphere_npz)
    else:
        atmosphere = load_atmosphere_npz(atmosphere_npz)
    result, seconds = _synthesis_engine.synthesize_structured_atmosphere(
        atmosphere,
        wavelength_start_nm=wavelength_start_nm,
        wavelength_end_nm=wavelength_end_nm,
        resolution=resolution,
        molecular_lines=molecular_lines,
        device=_runtime_device(device),
        dtype=_torch_dtype(dtype),
        spectral_operator=spectral_operator,
    )
    return _wrap(result, seconds)


def initialize_atmosphere_from_labels(
    *,
    effective_temperature: float,
    log_surface_gravity: float,
    metallicity: float = 0.0,
    fe_over_h: float | None = None,
    alpha_enhancement: float = 0.0,
    microturbulence_km_s: float = 2.0,
    c_over_m: float | None = None,
    n_over_m: float | None = None,
    o_over_m: float | None = None,
    x_over_h: Mapping[int | str, float] | None = None,
    initializer_family: str = "auto",
    molecular_lines: bool = True,
    device: str | torch.device | None = "auto",
    dtype: str | torch.dtype | None = "auto",
    five_label_path: str | Path | None = None,
    cno8_path: str | Path | None = None,
) -> InitializedAtmosphere:
    """Predict and population-bridge an atmosphere from stellar labels.

    ``initializer_family="auto"`` uses the five-label initializer unless an
    independent C, N, or O coordinate is supplied, in which case it selects
    the eight-label initializer.  Supplying ``x_over_h`` selects the
    direct-abundance initializer.  Direct-abundance values are sparse
    individual ``[X/H]`` coordinates keyed by symbol or atomic number.
    Unspecified elements inherit ``fe_over_h``. Bulk ``metallicity`` and
    ``alpha_enhancement`` coordinates do not apply in direct-abundance mode.

    The returned structure is suitable for fast synthesis and optimization.
    It has not passed the iterative physical-atmosphere convergence test, so
    ``atmosphere_converged`` is false and ``atmosphere_closure_required`` is
    true by construction.
    """

    from payne_zero_atmosphere import (
        CNO8_FAMILY,
        FIVE_LABEL_FAMILY,
        emulator_warm_start_model,
        linear_elemental_abundances,
        select_warm_start_family,
    )
    from payne_zero_atmosphere.direct_abundance import (
        DIRECT_XH_ATOMIC_NUMBERS,
        DIRECT_XH_FAMILY,
        build_direct_abundance_optimizer_surrogate,
    )

    family_aliases = {
        "auto": "auto",
        "five_label": FIVE_LABEL_FAMILY,
        "5d": FIVE_LABEL_FAMILY,
        "cno8": CNO8_FAMILY,
        "8d": CNO8_FAMILY,
        "direct_abundance": DIRECT_XH_FAMILY,
        "direct-abundance": DIRECT_XH_FAMILY,
    }
    try:
        requested_family = family_aliases[str(initializer_family).lower()]
    except KeyError as error:
        raise ValueError(
            "initializer_family must be auto, five_label, cno8, or direct_abundance"
        ) from error

    direct_requested = x_over_h is not None or fe_over_h is not None
    if requested_family == "auto":
        if direct_requested:
            family = DIRECT_XH_FAMILY
        else:
            family = select_warm_start_family(
                carbon_enhancement=c_over_m,
                nitrogen_enhancement=n_over_m,
                oxygen_enhancement=o_over_m,
            )
    else:
        family = requested_family

    if family == DIRECT_XH_FAMILY:
        if (
            any(
                value is not None
                for value in (
                    c_over_m,
                    n_over_m,
                    o_over_m,
                )
            )
            or float(alpha_enhancement) != 0.0
            or float(metallicity) != 0.0
        ):
            raise ValueError(
                "x_over_h uses individual [X/H] coordinates; do not also "
                "supply bulk metallicity, alpha, or CNO enhancement coordinates"
            )
    elif direct_requested:
        raise ValueError(
            "x_over_h requires initializer_family='auto' or 'direct_abundance'"
        )
    elif family == FIVE_LABEL_FAMILY and any(
        value is not None
        for value in (
            c_over_m,
            n_over_m,
            o_over_m,
        )
    ):
        raise ValueError("independent CNO coordinates require the cno8 initializer")
    elif fe_over_h is not None:
        raise ValueError("fe_over_h applies only to direct_abundance")

    total_start = time.perf_counter()
    initializer_start = time.perf_counter()
    label_record: dict[str, object] = {
        "effective_temperature": float(effective_temperature),
        "log_surface_gravity": float(log_surface_gravity),
        "microturbulence_km_s": float(microturbulence_km_s),
    }
    if family == DIRECT_XH_FAMILY:
        sparse_xh = _normalize_sparse_direct_xh(x_over_h or {})
        if fe_over_h is None and 26 not in sparse_xh:
            raise ValueError(
                "direct_abundance requires fe_over_h or an Fe entry in x_over_h"
            )
        iron = float(sparse_xh[26] if fe_over_h is None else fe_over_h)
        if 26 in sparse_xh and not np.isclose(
            sparse_xh[26], iron, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(
                "Fe in x_over_h disagrees with the fe_over_h baseline"
            )
        complete_xh = {
            atomic_number: iron for atomic_number in DIRECT_XH_ATOMIC_NUMBERS
        }
        complete_xh.update(sparse_xh)
        # The lower-level atmosphere module preserves the historical
        # opt-in warning on its specialist surrogate entry point. This public
        # wrapper already marks every initialized result as unconverged and
        # requiring a physical atmosphere check, so repeating that warning on
        # every spectrum evaluation would obscure ordinary optimizer use.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"experimental direct-\[X/H\] optimizer surrogate:.*",
                category=RuntimeWarning,
            )
            surrogate = build_direct_abundance_optimizer_surrogate(
                effective_temperature=effective_temperature,
                log_surface_gravity=log_surface_gravity,
                microturbulence_km_s=microturbulence_km_s,
                abundance_by_atomic_number=complete_xh,
                enable_experimental_optimizer_surrogate=True,
                device="cpu",
            )
        model_atmosphere = surrogate.optimizer_atmosphere
        provenance = surrogate.provenance()
        provenance.update(
            {
                "initializer_family": DIRECT_XH_FAMILY,
                "input_abundance_semantics": "[X/H]",
                "unspecified_elements_inherit": "[Fe/H]",
                "iron_abundance_relative_to_hydrogen": iron,
                "requested_sparse_abundances": {
                    str(atomic_number): value
                    for atomic_number, value in sorted(sparse_xh.items())
                },
                "atmosphere_closure_required": True,
            }
        )
        label_record.update(
            {
                "iron_abundance_relative_to_hydrogen": iron,
                "x_over_h": {
                    str(atomic_number): value
                    for atomic_number, value in sorted(sparse_xh.items())
                },
            }
        )
    else:
        if family == CNO8_FAMILY:
            c_over_m = 0.0 if c_over_m is None else c_over_m
            n_over_m = 0.0 if n_over_m is None else n_over_m
            o_over_m = alpha_enhancement if o_over_m is None else o_over_m
        model_atmosphere, _deck = emulator_warm_start_model(
            effective_temperature=effective_temperature,
            log_surface_gravity=log_surface_gravity,
            metallicity=metallicity,
            alpha_enhancement=alpha_enhancement,
            microturbulence_km_s=microturbulence_km_s,
            carbon_enhancement=c_over_m,
            nitrogen_enhancement=n_over_m,
            oxygen_enhancement=o_over_m,
            device="cpu",
            five_label_path=None if five_label_path is None else Path(five_label_path),
            cno8_path=None if cno8_path is None else Path(cno8_path),
        )
        provenance = _initializer_checkpoint_provenance(
            family,
            five_label_path=five_label_path,
            cno8_path=cno8_path,
        )
        label_record.update(
            {
                "metallicity": float(metallicity),
                "alpha_enhancement": float(alpha_enhancement),
            }
        )
        if family == CNO8_FAMILY:
            label_record.update(
                {
                    "c_over_m": float(c_over_m),
                    "n_over_m": float(n_over_m),
                    "o_over_m": float(o_over_m),
                }
            )
    initializer_seconds = time.perf_counter() - initializer_start

    bridge_start = time.perf_counter()
    elemental_abundances = linear_elemental_abundances(model_atmosphere)
    structured = build_structured_atmosphere(
        temperature=model_atmosphere.temperature,
        column_mass=model_atmosphere.column_mass,
        gas_pressure=model_atmosphere.gas_pressure,
        electron_density=model_atmosphere.electron_density,
        elemental_abundances=elemental_abundances,
        microturbulence=model_atmosphere.microturbulence,
        molecular_lines=molecular_lines,
        device=device,
        dtype=dtype,
    )
    bridge_seconds = time.perf_counter() - bridge_start
    total_seconds = time.perf_counter() - total_start
    timings = ForwardTimings(
        initializer_seconds=float(initializer_seconds),
        population_bridge_seconds=float(bridge_seconds),
        synthesis_seconds=0.0,
        total_seconds=float(total_seconds),
    )
    return InitializedAtmosphere(
        structured_atmosphere=structured,
        initializer_family=family,
        labels=label_record,
        provenance=provenance,
        timings=timings,
    )


def synthesize_from_labels(
    *,
    effective_temperature: float,
    log_surface_gravity: float,
    metallicity: float = 0.0,
    fe_over_h: float | None = None,
    alpha_enhancement: float = 0.0,
    microturbulence_km_s: float = 2.0,
    c_over_m: float | None = None,
    n_over_m: float | None = None,
    o_over_m: float | None = None,
    x_over_h: Mapping[int | str, float] | None = None,
    initializer_family: str = "auto",
    wavelength_start_nm: float = 400.0,
    wavelength_end_nm: float = 900.0,
    r_grid: float | None = None,
    resolution: float | None = None,
    molecular_lines: bool = True,
    device: str | torch.device | None = "auto",
    dtype: str | torch.dtype | None = "auto",
    spectral_operator=None,
    five_label_path: str | Path | None = None,
    cno8_path: str | Path | None = None,
) -> LabelSpectrum:
    """Predict an atmosphere from labels and synthesize its spectrum.

    The wavelength range, intrinsic logarithmic sampling ``r_grid``, device,
    and dtype are caller controlled.  ``resolution`` is accepted as a
    backward-friendly alias for ``r_grid``.
    """

    total_start = time.perf_counter()
    initialized = initialize_atmosphere_from_labels(
        effective_temperature=effective_temperature,
        log_surface_gravity=log_surface_gravity,
        metallicity=metallicity,
        fe_over_h=fe_over_h,
        alpha_enhancement=alpha_enhancement,
        microturbulence_km_s=microturbulence_km_s,
        c_over_m=c_over_m,
        n_over_m=n_over_m,
        o_over_m=o_over_m,
        x_over_h=x_over_h,
        initializer_family=initializer_family,
        molecular_lines=molecular_lines,
        device=device,
        dtype=dtype,
        five_label_path=five_label_path,
        cno8_path=cno8_path,
    )
    resolved_r_grid = _resolved_r_grid(r_grid=r_grid, resolution=resolution)
    result, synthesis_seconds = _synthesis_engine.synthesize_structured_atmosphere(
        initialized.structured_atmosphere,
        wavelength_start_nm=wavelength_start_nm,
        wavelength_end_nm=wavelength_end_nm,
        resolution=resolved_r_grid,
        molecular_lines=molecular_lines,
        device=_runtime_device(device),
        dtype=_torch_dtype(dtype),
        spectral_operator=spectral_operator,
    )
    total_seconds = time.perf_counter() - total_start
    wrapped = _wrap(result, total_seconds)
    timings = ForwardTimings(
        initializer_seconds=initialized.timings.initializer_seconds,
        population_bridge_seconds=initialized.timings.population_bridge_seconds,
        synthesis_seconds=float(synthesis_seconds),
        total_seconds=float(total_seconds),
    )
    return LabelSpectrum(
        wavelength_nm=wrapped.wavelength_nm,
        flux_total=wrapped.flux_total,
        flux_continuum=wrapped.flux_continuum,
        normalized_flux=wrapped.normalized_flux,
        seconds=wrapped.seconds,
        initializer_family=initialized.initializer_family,
        labels=initialized.labels,
        provenance=initialized.provenance,
        timings=timings,
        initialized_atmosphere=initialized,
    )
