"""Reusable atmosphere-emulator and synthesis forward model for APOGEE."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import time

import numpy as np
import torch

from .lsf import APOGEEDR14LSF
from .spectral_nuisance import APOGEESpectralNuisance


LABEL_NAMES = (
    "effective_temperature",
    "log_surface_gravity",
    "metallicity",
    "alpha_enhancement",
    "microturbulence_km_s",
)
LABEL_LOWER = np.asarray([4000.0, 0.7, -2.5, -0.1, 0.5], np.float64)
LABEL_UPPER = np.asarray([10500.0, 5.3, 0.5, 0.5, 4.0], np.float64)
WAVELENGTH_START_NM = 1500.0
WAVELENGTH_END_NM = 1700.0
APOGEE_LSF_TABULATION_RESOLUTION = 1.0 / (10.0 ** (2.0e-6) - 1.0)
APOGEE_SYNTHESIS_R_GRID = 300_000.0
# Compatibility name retained for callers of the pre-v1.3 API.
APOGEE_SYNTHESIS_RESOLUTION = APOGEE_SYNTHESIS_R_GRID
APOGEE_SYNTHESIS_WAVELENGTH_START_NM = 1515.0662727315726
APOGEE_SYNTHESIS_WAVELENGTH_END_NM = 1700.0
OPTIMIZER_SURROGATE_STRUCTURED_CACHE_SIZE = 8


def _override_synthesis_abundances(
    elemental_abundances: np.ndarray,
    *,
    alpha_enhancement: float,
    element_enhancements: dict[int, float] | None,
    carbon_enhancement: float | None,
    nitrogen_enhancement: float | None,
    oxygen_enhancement: float | None,
) -> np.ndarray:
    """Apply absolute ``[X/M]`` values to the fixed-structure synthesis mixture.

    The atmosphere initializer still sees its declared bulk labels.  Each
    override replaces the corresponding synthesis abundance relative to that
    baseline before the population bridge is rebuilt.  This makes, for
    example, magnesium independent of the bulk-alpha *synthesis* response
    without claiming an independently recomputed magnesium atmosphere.
    """

    if not element_enhancements:
        return elemental_abundances
    from payne_zero_atmosphere import ALPHA_ELEMENT_ATOMIC_NUMBERS

    abundances = np.asarray(elemental_abundances, np.float64).copy()
    original_sum = float(np.sum(abundances))
    cno_enhancements = {
        6: 0.0 if carbon_enhancement is None else float(carbon_enhancement),
        7: 0.0 if nitrogen_enhancement is None else float(nitrogen_enhancement),
        8: (
            float(alpha_enhancement)
            if oxygen_enhancement is None
            else float(oxygen_enhancement)
        ),
    }
    for atomic_number, requested_enhancement in sorted(element_enhancements.items()):
        atomic_number = int(atomic_number)
        requested_enhancement = float(requested_enhancement)
        if not 3 <= atomic_number <= abundances.size:
            raise ValueError("synthesis abundance overrides require 3 <= Z <= 99")
        if not np.isfinite(requested_enhancement):
            raise ValueError("synthesis element enhancements must be finite")
        baseline_enhancement = cno_enhancements.get(
            atomic_number,
            (
                float(alpha_enhancement)
                if atomic_number in ALPHA_ELEMENT_ATOMIC_NUMBERS
                else 0.0
            ),
        )
        abundances[atomic_number - 1] *= np.power(
            10.0, requested_enhancement - baseline_enhancement
        )
    abundance_sum = float(np.sum(abundances))
    if not np.isfinite(abundance_sum) or abundance_sum <= 0.0:
        raise ValueError("synthesis abundance overrides produced an invalid mixture")
    abundances *= original_sum / abundance_sum
    return abundances


def _device_and_dtype(device_request: str, dtype_request: str) -> tuple[str, str]:
    """Resolve the requested Torch device and synthesis dtype."""

    if not isinstance(device_request, str) or device_request not in {
        "auto",
        "cpu",
        "mps",
        "cuda",
    }:
        raise ValueError("device must be auto, cpu, mps, or cuda")
    if not isinstance(dtype_request, str) or dtype_request not in {
        "auto",
        "float32",
        "float64",
    }:
        raise ValueError("dtype must be auto, float32, or float64")
    if device_request == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_request
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    dtype = (
        ("float64" if device == "cpu" else "float32")
        if dtype_request == "auto"
        else dtype_request
    )
    if device == "mps" and dtype == "float64":
        raise ValueError("MPS synthesis requires float32")
    return device, dtype


def _torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "float64":
        return torch.float64
    if dtype == "float32":
        return torch.float32
    raise ValueError("dtype must be float32 or float64")


def _resolve_synthesis_r_grid(
    synthesis_r_grid: float | None = None,
    synthesis_resolution: float | None = None,
) -> float:
    """Resolve the preferred R_grid name and its historical API alias."""

    if synthesis_r_grid is not None and synthesis_resolution is not None:
        if float(synthesis_r_grid) != float(synthesis_resolution):
            raise ValueError(
                "synthesis_r_grid and legacy synthesis_resolution disagree"
            )
    value = (
        synthesis_r_grid
        if synthesis_r_grid is not None
        else synthesis_resolution
        if synthesis_resolution is not None
        else APOGEE_SYNTHESIS_R_GRID
    )
    resolved = float(value)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError("synthesis_r_grid must be finite and positive")
    return resolved


def _labels_dict(values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(LABEL_NAMES, values, strict=True)}


def _synthesis_window(
    apogee_dr14_lsf: bool,
    synthesis_r_grid: float = APOGEE_SYNTHESIS_R_GRID,
) -> tuple[float, float]:
    if apogee_dr14_lsf:
        blue_edge = (
            APOGEE_SYNTHESIS_WAVELENGTH_START_NM
            if synthesis_r_grid == APOGEE_SYNTHESIS_R_GRID
            else 1515.0
        )
        return blue_edge, APOGEE_SYNTHESIS_WAVELENGTH_END_NM
    return WAVELENGTH_START_NM, WAVELENGTH_END_NM


class FastForwardModel:
    """Map five stellar labels to an observed-grid normalized APOGEE spectrum.

    The model combines the atmosphere initializer, population bridge, synthesis
    engine, measured LSF, and optional kinematic nuisance projection. Survey
    acquisition and catalog selection remain outside this installed module.
    """

    def __init__(
        self,
        *,
        device: str,
        dtype: str,
        synthesis_r_grid: float | None = None,
        resolution: float | None = None,
        apogee_dr14_lsf: bool = False,
        fit_spectral_nuisance: bool = False,
        atomic_calibration_path: str | Path | None = None,
        apogee_lsf_path: str | Path | None = None,
    ) -> None:
        self.device, self.dtype = _device_and_dtype(device, dtype)
        self.synthesis_r_grid = _resolve_synthesis_r_grid(
            synthesis_r_grid, resolution
        )
        # Attribute alias retained for low-level callers of the historical API.
        self.resolution = self.synthesis_r_grid
        self.apogee_dr14_lsf = bool(apogee_dr14_lsf)
        self.fit_spectral_nuisance = bool(fit_spectral_nuisance)
        self.atomic_calibration_path = (
            None
            if atomic_calibration_path is None
            else Path(atomic_calibration_path).expanduser().resolve()
        )
        self.apogee_lsf_path = (
            None
            if apogee_lsf_path is None
            else Path(apogee_lsf_path).expanduser().resolve()
        )
        if self.fit_spectral_nuisance and not self.apogee_dr14_lsf:
            raise ValueError("spectral nuisance fitting requires the APOGEE LSF grid")
        self.wavelength_start_nm, self.wavelength_end_nm = _synthesis_window(
            self.apogee_dr14_lsf, self.synthesis_r_grid
        )
        self.spectral_operator: APOGEEDR14LSF | APOGEESpectralNuisance | None = None
        self.spectral_operator_setup_seconds = 0.0
        self.wavelength_nm: np.ndarray | None = None
        self.window_build_profile: dict[str, float] = {}
        self.window_invariants = None
        self.atomic_calibration_metadata: dict[str, object] | None = None
        self.resident_invariant_cache_hit_seconds = float("nan")
        self.resident_invariant_identity_reused = False
        self._optimizer_surrogate_structured_cache: OrderedDict[
            str, tuple[np.ndarray, dict]
        ] = OrderedDict()
        self.optimizer_surrogate_structured_cache_hits = 0
        self.optimizer_surrogate_structured_cache_misses = 0
        self.last_optimizer_surrogate_provenance: dict[str, object] | None = None
        bridge_setup_start = time.perf_counter()
        from payne_zero_synthesis import equation_of_state as synthesis_eos
        from payne_zero_synthesis import molecular_equilibrium as synthesis_molecular
        from payne_zero_synthesis import pipeline as synthesis_pipeline

        self._synthesis_pipeline = synthesis_pipeline
        self.eos_tables = synthesis_eos.EOSTables.from_npz(
            device=torch.device(self.device), dtype=_torch_dtype(self.dtype)
        )
        self.atomic_masses = synthesis_pipeline.load_atomic_masses()
        self.molecular_species_codes = (
            synthesis_molecular.supported_molecular_species_codes()
        )
        self.molecules_path = synthesis_molecular._default_molecule_table()
        self.bridge_setup_seconds = time.perf_counter() - bridge_setup_start

    def prepare_window(self) -> float:
        """Build process-resident synthesis invariants outside optimizer timing."""

        from payne_zero_synthesis.pipeline import window_invariants_for

        start = time.perf_counter()
        base_bundle = window_invariants_for(
            wl_start_nm=self.wavelength_start_nm,
            wl_end_nm=self.wavelength_end_nm,
            resolution=self.synthesis_r_grid,
            molecular_lines=True,
            runtime_device=torch.device(self.device),
            work_dtype=_torch_dtype(self.dtype),
        )
        cache_check_start = time.perf_counter()
        cached_bundle = window_invariants_for(
            wl_start_nm=self.wavelength_start_nm,
            wl_end_nm=self.wavelength_end_nm,
            resolution=self.synthesis_r_grid,
            molecular_lines=True,
            runtime_device=torch.device(self.device),
            work_dtype=_torch_dtype(self.dtype),
        )
        self.resident_invariant_cache_hit_seconds = (
            time.perf_counter() - cache_check_start
        )
        self.resident_invariant_identity_reused = cached_bundle is base_bundle
        if not self.resident_invariant_identity_reused:
            raise RuntimeError(
                "window invariant cache did not reuse the resident bundle"
            )
        bundle = base_bundle
        if self.atomic_calibration_path is not None:
            from .atomic_calibration import calibrated_window_invariants

            bundle, self.atomic_calibration_metadata = calibrated_window_invariants(
                base_bundle, self.atomic_calibration_path
            )
        self.window_invariants = bundle
        self.window_build_profile = {
            str(name): float(seconds)
            for name, seconds in base_bundle.build_profile.items()
        }
        self.wavelength_nm = np.asarray(bundle.wavelength_nm, np.float64)
        if self.apogee_dr14_lsf:
            operator_start = time.perf_counter()
            operator_class = (
                APOGEESpectralNuisance if self.fit_spectral_nuisance else APOGEEDR14LSF
            )
            operator_dtype = _torch_dtype(self.dtype)
            if self.apogee_lsf_path is None:
                self.spectral_operator = operator_class(
                    self.wavelength_nm,
                    device=self.device,
                    dtype=operator_dtype,
                )
            else:
                self.spectral_operator = operator_class(
                    self.wavelength_nm,
                    device=self.device,
                    dtype=operator_dtype,
                    asset_path=self.apogee_lsf_path,
                )
            self.spectral_operator.prepare()
            self.spectral_operator_setup_seconds = time.perf_counter() - operator_start
            self.wavelength_nm = self.spectral_operator.output_wavelength_nm
        return time.perf_counter() - start

    def _synthesize_model_atmosphere(
        self,
        model_atmosphere,
        *,
        alpha_enhancement: float,
        carbon_enhancement: float | None,
        nitrogen_enhancement: float | None,
        oxygen_enhancement: float | None,
        synthesis_element_enhancements: dict[int, float] | None,
        emulator_seconds: float,
        total_start: float,
        structured: dict | None = None,
        structured_cache_key: str | None = None,
        structured_cache_abundance: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """Bridge one explicit atmosphere and synthesize it on the active grid."""

        from payne_zero_atmosphere import linear_elemental_abundances
        from payne_zero_synthesis import synthesis as synthesis_engine

        bridge_start = time.perf_counter()
        structured_cache_reused = False
        if structured is None and structured_cache_key is not None:
            cached = self._optimizer_surrogate_structured_cache.get(
                structured_cache_key
            )
            if cached is not None:
                cached_abundance, structured = cached
                if structured_cache_abundance is None or not np.array_equal(
                    cached_abundance, structured_cache_abundance
                ):
                    raise RuntimeError(
                        "optimizer surrogate structured-cache identity collision"
                    )
                self._optimizer_surrogate_structured_cache.move_to_end(
                    structured_cache_key
                )
                self.optimizer_surrogate_structured_cache_hits += 1
                structured_cache_reused = True
        if structured is None:
            elemental_abundances = linear_elemental_abundances(model_atmosphere)
            elemental_abundances = _override_synthesis_abundances(
                elemental_abundances,
                alpha_enhancement=alpha_enhancement,
                element_enhancements=synthesis_element_enhancements,
                carbon_enhancement=carbon_enhancement,
                nitrogen_enhancement=nitrogen_enhancement,
                oxygen_enhancement=oxygen_enhancement,
            )
            structured = self._synthesis_pipeline.build_structured_atmosphere_from_columns(
                temperature=model_atmosphere.temperature,
                column_mass=model_atmosphere.column_mass,
                gas_pressure=model_atmosphere.gas_pressure,
                electron_density=model_atmosphere.electron_density,
                elemental_abundances=elemental_abundances,
                mean_nuclear_mass_amu=synthesis_engine.compute_mean_nuclear_mass_amu(
                    elemental_abundances
                ),
                microturbulence=model_atmosphere.microturbulence,
                eos_tables=self.eos_tables,
                electron_density_seed=model_atmosphere.electron_density,
                tol=1.0e-5,
                atomic_masses=self.atomic_masses,
                mass_density=None,
                molecular_species_codes=self.molecular_species_codes,
                molecules_path=self.molecules_path,
            )
            if structured_cache_key is not None:
                if structured_cache_abundance is None:
                    raise RuntimeError(
                        "optimizer surrogate cache requires its realized mixture"
                    )
                self._optimizer_surrogate_structured_cache[structured_cache_key] = (
                    np.asarray(structured_cache_abundance, np.float64).copy(),
                    structured,
                )
                self._optimizer_surrogate_structured_cache.move_to_end(
                    structured_cache_key
                )
                while (
                    len(self._optimizer_surrogate_structured_cache)
                    > OPTIMIZER_SURROGATE_STRUCTURED_CACHE_SIZE
                ):
                    self._optimizer_surrogate_structured_cache.popitem(last=False)
                self.optimizer_surrogate_structured_cache_misses += 1
        bridge_seconds = time.perf_counter() - bridge_start

        result, synthesis_seconds = synthesis_engine.synthesize_structured_atmosphere(
            structured,
            wavelength_start_nm=self.wavelength_start_nm,
            wavelength_end_nm=self.wavelength_end_nm,
            resolution=self.synthesis_r_grid,
            molecular_lines=True,
            device=self.device,
            dtype=_torch_dtype(self.dtype),
            spectral_operator=self.spectral_operator,
            window_invariants=self.window_invariants,
        )
        instrument_lsf_seconds = float(result.spectral_operator_seconds)
        kinematic_projection_seconds = 0.0
        if isinstance(self.spectral_operator, APOGEESpectralNuisance):
            instrument_lsf_seconds = float(self.spectral_operator.lsf.last_seconds)
            kinematic_projection_seconds = float(
                self.spectral_operator.last_kinematic_seconds
            )
        timings = {
            "emulator_seconds": float(emulator_seconds),
            "population_bridge_seconds": bridge_seconds,
            "synthesis_seconds": float(synthesis_seconds),
            "physics_synthesis_seconds": float(synthesis_seconds)
            - instrument_lsf_seconds,
            "instrument_lsf_seconds": instrument_lsf_seconds,
            "kinematic_projection_seconds": kinematic_projection_seconds,
            "total_forward_seconds": time.perf_counter() - total_start,
            "synthesis_r_grid": self.synthesis_r_grid,
            "resolution_grid": self.synthesis_r_grid,
        }
        if structured_cache_key is not None:
            timings["optimizer_surrogate_structured_cache_reused"] = float(
                structured_cache_reused
            )
        return (
            np.asarray(result.wavelength_nm, np.float64),
            np.asarray(result.normalized_flux, np.float64),
            timings,
        )

    def evaluate(
        self,
        label_values: np.ndarray,
        *,
        residual_rv_km_s: float = 0.0,
        vmacro_km_s: float = 0.0,
        carbon_enhancement: float | None = None,
        nitrogen_enhancement: float | None = None,
        oxygen_enhancement: float | None = None,
        synthesis_element_enhancements: dict[int, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """Evaluate the full forward model at one stellar/nuisance point.

        ``synthesis_element_enhancements`` maps atomic number to an absolute
        ``[X/M]`` used only after the baseline atmosphere has been decoded.
        """

        from payne_zero_atmosphere import (
            emulator_warm_start_model,
        )

        labels = _labels_dict(np.asarray(label_values, np.float64))
        total_start = time.perf_counter()
        if isinstance(self.spectral_operator, APOGEESpectralNuisance):
            self.spectral_operator.set_parameters(
                residual_rv_km_s=residual_rv_km_s,
                vmacro_km_s=vmacro_km_s,
            )

        emulator_start = time.perf_counter()
        model_atmosphere, _ = emulator_warm_start_model(
            effective_temperature=labels["effective_temperature"],
            log_surface_gravity=labels["log_surface_gravity"],
            metallicity=labels["metallicity"],
            alpha_enhancement=labels["alpha_enhancement"],
            microturbulence_km_s=labels["microturbulence_km_s"],
            carbon_enhancement=carbon_enhancement,
            nitrogen_enhancement=nitrogen_enhancement,
            oxygen_enhancement=oxygen_enhancement,
            device="cpu",
        )
        emulator_seconds = time.perf_counter() - emulator_start
        return self._synthesize_model_atmosphere(
            model_atmosphere,
            alpha_enhancement=labels["alpha_enhancement"],
            carbon_enhancement=carbon_enhancement,
            nitrogen_enhancement=nitrogen_enhancement,
            oxygen_enhancement=oxygen_enhancement,
            synthesis_element_enhancements=synthesis_element_enhancements,
            emulator_seconds=emulator_seconds,
            total_start=total_start,
        )

    def evaluate_optimizer_surrogate(
        self,
        surrogate,
        *,
        residual_rv_km_s: float = 0.0,
        vmacro_km_s: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """Synthesize one explicit direct-[X/H] optimizer-only surrogate.

        The surrogate atmosphere and the synthesis population bridge must carry
        the identical realized, quantized 97-slot mixture.  This method never
        promotes the decoded structure to a final atmosphere; callers must run
        and record the mandatory converged exact closure at the selected point.
        """

        from payne_zero_atmosphere.direct_abundance import (
            DirectAbundanceOptimizerSurrogate,
            direct_abundance_mixture_sha256,
        )
        from payne_zero_atmosphere.warm_start import (
            SOLAR_METAL_LOG_ABUNDANCES_3_TO_99,
        )

        if not isinstance(surrogate, DirectAbundanceOptimizerSurrogate):
            raise TypeError(
                "evaluate_optimizer_surrogate requires an explicit "
                "DirectAbundanceOptimizerSurrogate product"
            )
        if (
            surrogate.exact_closure_required is not True
            or surrogate.is_final_atmosphere is not False
        ):
            raise RuntimeError(
                "direct-[X/H] optimizer surrogate safety flags are invalid"
            )
        atmosphere = surrogate.optimizer_atmosphere
        atmosphere_realized = np.asarray(
            [
                atmosphere.fixed_column_abundance_values[atomic_number]
                - SOLAR_METAL_LOG_ABUNDANCES_3_TO_99[atomic_number - 3]
                for atomic_number in range(3, 100)
            ],
            np.float64,
        )
        if not np.allclose(
            atmosphere_realized,
            surrogate.realized_abundance_vector,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                "optimizer surrogate atmosphere and synthesis mixture differ"
            )
        if (
            direct_abundance_mixture_sha256(atmosphere_realized)
            != surrogate.realized_mixture_sha256
        ):
            raise RuntimeError("optimizer surrogate realized-mixture hash is invalid")

        total_start = time.perf_counter()
        if isinstance(self.spectral_operator, APOGEESpectralNuisance):
            self.spectral_operator.set_parameters(
                residual_rv_km_s=residual_rv_km_s,
                vmacro_km_s=vmacro_km_s,
            )
        result = self._synthesize_model_atmosphere(
            atmosphere,
            alpha_enhancement=0.0,
            carbon_enhancement=None,
            nitrogen_enhancement=None,
            oxygen_enhancement=None,
            synthesis_element_enhancements=None,
            emulator_seconds=0.0,
            total_start=total_start,
            structured_cache_key=surrogate.surrogate_identity_sha256,
            structured_cache_abundance=surrogate.realized_abundance_vector,
        )
        provenance = surrogate.provenance()
        provenance.update(
            {
                "synthesis_realized_mixture_sha256": (
                    surrogate.realized_mixture_sha256
                ),
                "atmosphere_and_synthesis_mixture_identical": True,
                "structured_cache_reused": bool(
                    result[2]["optimizer_surrogate_structured_cache_reused"]
                ),
                "structured_cache_hits": (
                    self.optimizer_surrogate_structured_cache_hits
                ),
                "structured_cache_misses": (
                    self.optimizer_surrogate_structured_cache_misses
                ),
            }
        )
        self.last_optimizer_surrogate_provenance = provenance
        return result

    def project_cached_nuisance(
        self,
        *,
        residual_rv_km_s: float,
        vmacro_km_s: float,
    ) -> torch.Tensor:
        """Project cached native flux at new kinematics without re-synthesis."""

        if not isinstance(self.spectral_operator, APOGEESpectralNuisance):
            raise RuntimeError("this forward model has no spectral nuisance cache")
        return self.spectral_operator.project_cached(
            residual_rv_km_s=residual_rv_km_s,
            vmacro_km_s=vmacro_km_s,
        )[2]
