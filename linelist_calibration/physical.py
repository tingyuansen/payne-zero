"""Differentiable fixed-atmosphere synthesis for atomic line calibration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_functional

from payne_zero_synthesis import line_opacity, radiative_transfer
from payne_zero_synthesis.device import resolve_runtime
from payne_zero_synthesis import paths as synthesis_paths
from payne_zero_synthesis.pipeline import SynthesisPipeline

from .catalog import (
    ATOMIC_CALIBRATION_SCHEMA_VERSION,
    canonical_atomic_row_identities,
    validate_atomic_calibration,
    write_substituted_catalog as write_corrected_catalog,
)


LIGHT_SPEED_KM_S = 299_792.458
_PARAMETER_FAMILIES = ("loggf", "vdw", "radiative", "stark")


@dataclass(frozen=True)
class AtomicTransition:
    """One physical transition group selected from the synthesis catalog.

    The catalog row nearest ``wavelength_nm`` seeds the group. Components of
    the same element and ion stage are linked when their lower excitation and
    wavelength agree within the declared tolerances.
    """

    atomic_number: int
    ion_stage: int
    wavelength_nm: float
    name: str | None = None
    search_tolerance_nm: float = 0.008
    component_wavelength_tolerance_nm: float = 0.035
    component_excitation_tolerance_cm: float = 2.0


@dataclass(frozen=True)
class ResolvedAtomicTransition:
    """Catalog identity of one transition group used by a calibration model."""

    name: str
    atomic_number: int
    ion_stage: int
    wavelength_nm: float
    catalog_indices: tuple[int, ...]


def gaussian_velocity_kernel(
    velocity_step_km_s: float,
    sigma_km_s: float,
) -> np.ndarray:
    """Return a normalized odd Gaussian kernel sampled in velocity bins."""

    velocity_step = float(velocity_step_km_s)
    sigma = float(sigma_km_s)
    if not np.isfinite(velocity_step) or velocity_step <= 0.0:
        raise ValueError("velocity_step_km_s must be finite and positive")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma_km_s must be finite and positive")

    fine_step = velocity_step / 20.0
    extent = 6.0 * sigma
    half = int(np.ceil(extent / fine_step))
    velocity = np.arange(-half, half + 1, dtype=np.float64) * fine_step
    profile = np.exp(-0.5 * np.square(velocity / sigma))
    profile /= np.sum(profile)

    half_bins = int(np.ceil(extent / velocity_step)) + 1
    centers = np.arange(-half_bins, half_bins + 1, dtype=np.float64) * velocity_step
    edges = np.concatenate(
        (
            centers - 0.5 * velocity_step,
            np.asarray([centers[-1] + 0.5 * velocity_step]),
        )
    )
    binned, _ = np.histogram(velocity, bins=edges, weights=profile)
    keep = binned > np.max(binned) * 1.0e-8
    first, last = np.flatnonzero(keep)[[0, -1]]
    center = binned.size // 2
    radius = max(center - first, last - center)
    binned = binned[center - radius : center + radius + 1]
    binned = 0.5 * (binned + binned[::-1])
    return binned / np.sum(binned)


class SynthesisLineCalibrationModel:
    """A differentiable atomic calibration model for one fixed atmosphere.

    The expensive continuum and unchanged line opacity are evaluated once.
    Each call replaces only the selected transition opacity, solves radiative
    transfer, applies an optional fixed broadening kernel, and samples an
    optional observed wavelength grid. Corrections are additive dex offsets.
    """

    def __init__(
        self,
        atmosphere_path: str | Path,
        *,
        wavelength_start_nm: float,
        wavelength_end_nm: float,
        resolution: float,
        transitions: Sequence[AtomicTransition],
        observed_wavelength_nm: np.ndarray | None = None,
        radial_velocity_km_s: float = 0.0,
        broadening_kernel: np.ndarray | None = None,
        gaussian_broadening_sigma_km_s: float | None = None,
        molecular_lines: bool = True,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
    ) -> None:
        requested_dtype = self._torch_dtype(dtype)
        requested_device = None if device is None or device == "auto" else device
        self.device, self.dtype = resolve_runtime(requested_device, requested_dtype)
        if not transitions:
            raise ValueError("at least one atomic transition is required")
        self.pipeline = SynthesisPipeline(
            atmosphere_path,
            wl_start_nm=float(wavelength_start_nm),
            wl_end_nm=float(wavelength_end_nm),
            resolution=float(resolution),
            molecular_lines=bool(molecular_lines),
            device=self.device,
            dtype=self.dtype,
        )
        baseline = self.pipeline.run(keep_slabs=True)
        self.native_wavelength_nm = np.asarray(
            self.pipeline.wavelength_nm, np.float64
        )
        self.native_velocity_step_km_s = float(
            LIGHT_SPEED_KM_S
            * np.log(self.native_wavelength_nm[1] / self.native_wavelength_nm[0])
        )

        catalog = self.pipeline._atomic_kernel_catalog
        self._atomic_catalog = {
            name: np.asarray(value).copy()
            for name, value in catalog.items()
            if isinstance(value, np.ndarray)
        }
        self.transitions, selected_indices, group_by_line = self._resolve_transitions(
            catalog, transitions
        )
        self._selected_catalog_indices = selected_indices.copy()
        selected_catalog = SynthesisPipeline._slice_atomic_catalog(
            catalog, selected_indices
        )
        self._selected_invariants = line_opacity.precompute_invariants(
            selected_catalog,
            self.native_wavelength_nm,
            runtime_device=self.device,
        )
        if (
            self._selected_invariants.metal_classical_strength.numel()
            != selected_indices.size
        ):
            raise ValueError(
                "selected transition contains a line type unsupported by the "
                "ordinary atomic calibration path"
            )
        self._group_by_line = torch.as_tensor(
            group_by_line, dtype=torch.int64, device=self.device
        )

        self._continuum_absorption = torch.as_tensor(
            baseline.continuum_absorption, dtype=self.dtype, device=self.device
        )
        self._continuum_scattering = torch.as_tensor(
            baseline.continuum_scattering, dtype=self.dtype, device=self.device
        )
        continuum_total = self._continuum_absorption + self._continuum_scattering
        self._atomic_state = {
            "partition_normalized_populations": (
                self.pipeline._partition_normalized_populations
            ),
            "fractional_doppler_widths": self.pipeline._fractional_doppler_widths,
            "mass_density": self.pipeline._mass_density,
            "electron_density": self.pipeline._electron_density,
            "temperature": self.pipeline._temperature,
            "hc_over_kt": self.pipeline._hc_over_kt,
            "collision_density_proxy": self.pipeline._collision_density_proxy,
            "continuum_opacity": continuum_total,
            "helium_core_weight_grid": None,
            "helium_tail_weight_grid": None,
        }
        zeros = torch.zeros(
            len(self.transitions), dtype=self.dtype, device=self.device
        )
        with torch.no_grad():
            selected_baseline = self._selected_opacity(
                loggf=zeros,
                vdw=None,
                radiative=None,
                stark=None,
            )
        full_line = torch.as_tensor(
            baseline.line_mass_absorption_coefficient,
            dtype=selected_baseline.dtype,
            device=self.device,
        )
        self._fixed_line_opacity = full_line - selected_baseline
        self._line_source = torch.as_tensor(
            baseline.line_source, dtype=self.dtype, device=self.device
        )
        self._line_scattering = torch.zeros_like(self._line_source)
        if broadening_kernel is not None and gaussian_broadening_sigma_km_s is not None:
            raise ValueError(
                "provide broadening_kernel or gaussian_broadening_sigma_km_s, not both"
            )
        if gaussian_broadening_sigma_km_s is not None:
            broadening_kernel = gaussian_velocity_kernel(
                self.native_velocity_step_km_s,
                gaussian_broadening_sigma_km_s,
            )
        self._broadening_kernel = self._validate_kernel(broadening_kernel)
        (
            self.output_wavelength_nm,
            self._sample_left,
            self._sample_right,
            self._sample_fraction,
        ) = self._prepare_sampling(
            observed_wavelength_nm,
            radial_velocity_km_s=float(radial_velocity_km_s),
        )

    @staticmethod
    def _torch_dtype(value: str | torch.dtype | None) -> torch.dtype | None:
        if value is None or value == "auto":
            return None
        if isinstance(value, torch.dtype):
            return value
        if value == "float32":
            return torch.float32
        if value == "float64":
            return torch.float64
        raise ValueError("dtype must be float32 or float64")

    @staticmethod
    def _resolve_transitions(
        catalog: dict,
        requested: Sequence[AtomicTransition],
    ) -> tuple[tuple[ResolvedAtomicTransition, ...], np.ndarray, np.ndarray]:
        wavelength = np.asarray(catalog["wavelength_nm"], np.float64)
        excitation = np.asarray(catalog["lower_excitation_cm"], np.float64)
        atomic_number = np.asarray(catalog["atomic_number"], np.int64)
        ion_stage = np.asarray(catalog["ion_stage"], np.int64)
        line_type = np.asarray(catalog["line_type"], np.int64)
        ordinary = np.isin(line_type, (0, 3))

        resolved: list[ResolvedAtomicTransition] = []
        used: set[int] = set()
        for request in requested:
            possible = np.flatnonzero(
                ordinary
                & (atomic_number == int(request.atomic_number))
                & (ion_stage == int(request.ion_stage))
                & (
                    np.abs(wavelength - float(request.wavelength_nm))
                    <= float(request.search_tolerance_nm)
                )
            )
            if possible.size == 0:
                raise ValueError(
                    "no ordinary atomic transition matches "
                    f"Z={request.atomic_number}, ion={request.ion_stage}, "
                    f"wavelength={request.wavelength_nm:.6f} nm"
                )
            seed = int(
                possible[
                    np.argmin(
                        np.abs(wavelength[possible] - float(request.wavelength_nm))
                    )
                ]
            )
            components = np.flatnonzero(
                ordinary
                & (atomic_number == atomic_number[seed])
                & (ion_stage == ion_stage[seed])
                & (
                    np.abs(excitation - excitation[seed])
                    <= float(request.component_excitation_tolerance_cm)
                )
                & (
                    np.abs(wavelength - wavelength[seed])
                    <= float(request.component_wavelength_tolerance_nm)
                )
            )
            overlap = used.intersection(int(index) for index in components)
            if overlap:
                raise ValueError("atomic transition selections overlap")
            component_tuple = tuple(int(index) for index in components)
            used.update(component_tuple)
            resolved.append(
                ResolvedAtomicTransition(
                    name=(
                        request.name
                        or (
                            f"Z{int(atomic_number[seed])} "
                            f"ion {int(ion_stage[seed])} "
                            f"{wavelength[seed]:.6f} nm"
                        )
                    ),
                    atomic_number=int(atomic_number[seed]),
                    ion_stage=int(ion_stage[seed]),
                    wavelength_nm=float(wavelength[seed]),
                    catalog_indices=component_tuple,
                )
            )

        selected_indices = np.asarray(sorted(used), np.int64)
        position = {
            int(catalog_index): local_index
            for local_index, catalog_index in enumerate(selected_indices)
        }
        group_by_line = np.empty(selected_indices.size, np.int64)
        for group_index, transition in enumerate(resolved):
            for catalog_index in transition.catalog_indices:
                group_by_line[position[catalog_index]] = group_index
        return tuple(resolved), selected_indices, group_by_line

    def _validate_kernel(
        self, broadening_kernel: np.ndarray | None
    ) -> torch.Tensor | None:
        if broadening_kernel is None:
            return None
        kernel = np.asarray(broadening_kernel, np.float64)
        if (
            kernel.ndim != 1
            or kernel.size == 0
            or kernel.size % 2 != 1
            or not np.all(np.isfinite(kernel))
            or np.any(kernel < 0.0)
            or np.sum(kernel) <= 0.0
        ):
            raise ValueError(
                "broadening_kernel must be a finite, nonnegative, odd vector"
            )
        kernel = kernel / np.sum(kernel)
        return torch.as_tensor(kernel, dtype=self.dtype, device=self.device)

    def _prepare_sampling(
        self,
        observed_wavelength_nm: np.ndarray | None,
        *,
        radial_velocity_km_s: float,
    ) -> tuple[
        np.ndarray,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if observed_wavelength_nm is None:
            return self.native_wavelength_nm.copy(), None, None, None
        observed = np.asarray(observed_wavelength_nm, np.float64)
        if (
            observed.ndim != 1
            or observed.size == 0
            or not np.all(np.isfinite(observed))
            or np.any(np.diff(observed) <= 0.0)
        ):
            raise ValueError(
                "observed_wavelength_nm must be a finite increasing vector"
            )
        rest = observed / (1.0 + radial_velocity_km_s / LIGHT_SPEED_KM_S)
        right = np.searchsorted(self.native_wavelength_nm, rest)
        if np.any(right <= 0) or np.any(right >= self.native_wavelength_nm.size):
            raise ValueError(
                "observed wavelengths after velocity correction lie outside "
                "the synthesis window"
            )
        left = right - 1
        fraction = (
            (rest - self.native_wavelength_nm[left])
            / (self.native_wavelength_nm[right] - self.native_wavelength_nm[left])
        )
        return (
            observed.copy(),
            torch.as_tensor(left, dtype=torch.int64, device=self.device),
            torch.as_tensor(right, dtype=torch.int64, device=self.device),
            torch.as_tensor(fraction, dtype=self.dtype, device=self.device),
        )

    def _selected_opacity(
        self,
        *,
        loggf: torch.Tensor,
        vdw: torch.Tensor | None,
        radiative: torch.Tensor | None,
        stark: torch.Tensor | None,
    ) -> torch.Tensor:
        line_loggf = loggf[self._group_by_line]
        modified = replace(
            self._selected_invariants,
            metal_classical_strength=(
                self._selected_invariants.metal_classical_strength
                * torch.exp(line_loggf * np.log(10.0))
            ),
        )
        for correction, field in (
            (vdw, "metal_van_der_waals_damping"),
            (radiative, "metal_radiative_damping"),
            (stark, "metal_stark_damping"),
        ):
            if correction is not None:
                line_correction = correction[self._group_by_line]
                modified = replace(
                    modified,
                    **{
                        field: (
                            getattr(modified, field)
                            * torch.exp(line_correction * np.log(10.0))
                        )
                    },
                )
        return line_opacity.accumulate_atomic(
            modified,
            self._atomic_state,
            do_metal=True,
            do_helium=False,
            apply_stim=True,
        )

    def _broaden(
        self, total: torch.Tensor, continuum: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._broadening_kernel is None:
            return total, continuum
        pad = self._broadening_kernel.numel() // 2
        pair = torch.stack((total, continuum), dim=0).unsqueeze(1)
        padded = torch_functional.pad(pair, (pad, pad), mode="reflect")
        broadened = torch_functional.conv1d(
            padded, self._broadening_kernel.reshape(1, 1, -1)
        )[:, 0]
        return broadened[0], broadened[1]

    def spectrum(
        self,
        corrections_dex: torch.Tensor,
        *,
        parameter_families: Sequence[str] = ("loggf",),
    ) -> torch.Tensor:
        """Return normalized flux while retaining the PyTorch gradient graph.

        The flat parameter vector is ordered by family, then transition group.
        For two groups and ``("loggf", "vdw")``, its order is both log(gf)
        corrections followed by both van der Waals corrections.
        """

        families = tuple(parameter_families)
        if (
            not families
            or len(set(families)) != len(families)
            or any(family not in _PARAMETER_FAMILIES for family in families)
        ):
            raise ValueError(
                "parameter_families must be unique members of "
                f"{_PARAMETER_FAMILIES}"
            )
        expected = len(families) * len(self.transitions)
        if corrections_dex.ndim != 1 or corrections_dex.numel() != expected:
            raise ValueError(
                f"corrections_dex must contain {expected} values for "
                f"{len(self.transitions)} transition groups and {len(families)} families"
            )
        values = corrections_dex.reshape(len(families), len(self.transitions))
        by_family = dict(zip(families, values, strict=True))
        zeros = torch.zeros(
            len(self.transitions),
            dtype=corrections_dex.dtype,
            device=corrections_dex.device,
        )
        selected = self._selected_opacity(
            loggf=by_family.get("loggf", zeros),
            vdw=by_family.get("vdw"),
            radiative=by_family.get("radiative"),
            stark=by_family.get("stark"),
        )
        line = self._fixed_line_opacity + selected
        total, continuum, _ = radiative_transfer.solve_spectrum(
            self._continuum_absorption,
            self._continuum_scattering,
            line.to(self.dtype),
            self._line_scattering,
            self._line_source,
            self.pipeline.column_mass,
            self.pipeline.transfer_tables,
            assert_no_saturated_core=False,
        )
        total, continuum = self._broaden(total, continuum)
        normalized = total / continuum
        if self._sample_left is None:
            return normalized
        return (
            normalized[self._sample_left] * (1.0 - self._sample_fraction)
            + normalized[self._sample_right] * self._sample_fraction
        )

    def callback(
        self,
        parameter_families: Sequence[str] = ("loggf",),
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return the callback expected by ``calibrate_line_parameters``."""

        families = tuple(parameter_families)

        def forward(corrections_dex: torch.Tensor) -> torch.Tensor:
            return self.spectrum(
                corrections_dex,
                parameter_families=families,
            )

        return forward

    def baseline_flux(
        self,
        parameter_families: Sequence[str] = ("loggf",),
    ) -> np.ndarray:
        """Return the zero-correction model flux as a host NumPy array."""

        count = len(tuple(parameter_families)) * len(self.transitions)
        zeros = torch.zeros(count, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            return self.spectrum(
                zeros, parameter_families=parameter_families
            ).detach().cpu().numpy().astype(np.float64)

    def write_atomic_calibration_overlay(
        self,
        corrections_dex: np.ndarray | torch.Tensor,
        output_path: str | Path,
        *,
        parameter_families: Sequence[str] = ("loggf",),
        calibration_name: str = "payne_zero_atomic_calibration",
        source_catalog_path: str | Path | None = None,
        substituted_catalog_path: str | Path | None = None,
    ) -> dict[str, object]:
        """Write fitted values as a source-bound schema-4 atomic overlay.

        Selected physical transition groups retain their fitted corrections.
        Other catalog rows inside the minimal rectangular schema scope receive
        zero corrections so the resulting overlay satisfies complete-coverage
        validation. The optional substituted catalog is a corrected copy of
        the active synthesis-window catalog; the source catalog is untouched.
        """

        families = tuple(parameter_families)
        if (
            not families
            or len(set(families)) != len(families)
            or any(family not in _PARAMETER_FAMILIES for family in families)
        ):
            raise ValueError(
                "parameter_families must be unique members of "
                f"{_PARAMETER_FAMILIES}"
            )
        values = (
            corrections_dex.detach().cpu().numpy()
            if isinstance(corrections_dex, torch.Tensor)
            else np.asarray(corrections_dex)
        )
        expected = len(families) * len(self.transitions)
        values = np.asarray(values, np.float64)
        if values.shape != (expected,) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"corrections_dex must contain {expected} finite values"
            )
        if not calibration_name.strip():
            raise ValueError("calibration_name must be non-empty")

        source_path = (
            Path(source_catalog_path).expanduser().resolve()
            if source_catalog_path is not None
            else synthesis_paths.source_catalog_path(
                "lines", "atomic_source_lines_parsed.npz"
            ).resolve()
        )
        if not source_path.is_file():
            raise FileNotFoundError(f"atomic source catalog not found: {source_path}")

        catalog = self._atomic_catalog
        selected = self._selected_catalog_indices
        wavelength = np.asarray(catalog["wavelength_nm"], np.float64)
        atomic_number = np.asarray(catalog["atomic_number"], np.int64)
        ion_stage = np.asarray(catalog["ion_stage"], np.int64)
        line_type = np.asarray(catalog["line_type"], np.int64)
        scope_wavelength = np.asarray(
            [np.min(wavelength[selected]), np.max(wavelength[selected])],
            np.float64,
        )
        scope_atomic_number = np.unique(atomic_number[selected])
        scope_ion_stage = np.unique(ion_stage[selected])
        scope_line_type = np.unique(line_type[selected])
        scope_mask = (
            (wavelength >= scope_wavelength[0])
            & (wavelength <= scope_wavelength[1])
            & np.isin(atomic_number, scope_atomic_number)
            & np.isin(ion_stage, scope_ion_stage)
            & np.isin(line_type, scope_line_type)
        )
        component_rows = np.flatnonzero(scope_mask)
        selected_group = {
            int(row): int(group)
            for row, group in zip(
                selected.tolist(),
                self._group_by_line.detach().cpu().numpy().tolist(),
                strict=True,
            )
        }

        component_group = np.empty(component_rows.size, np.int64)
        next_group = len(self.transitions)
        for component_index, row in enumerate(component_rows.tolist()):
            group = selected_group.get(int(row))
            if group is None:
                group = next_group
                next_group += 1
            component_group[component_index] = group
        group_count = next_group

        corrections = {
            family: np.zeros(group_count, np.float64)
            for family in _PARAMETER_FAMILIES
        }
        selected_values = values.reshape(len(families), len(self.transitions))
        for family, family_values in zip(families, selected_values, strict=True):
            corrections[family][: len(self.transitions)] = family_values

        signatures, ordinals = canonical_atomic_row_identities(catalog)
        component_signatures = signatures[component_rows]
        component_ordinals = ordinals[component_rows]
        group_keys: list[str] = []
        for group in range(group_count):
            digest = hashlib.sha256(
                b"payne-zero-atomic-calibration-group-v1\0"
            )
            members = np.flatnonzero(component_group == group)
            for member in members.tolist():
                digest.update(component_signatures[member].encode("ascii"))
                digest.update(int(component_ordinals[member]).to_bytes(8, "little"))
            group_keys.append(digest.hexdigest())
        if len(set(group_keys)) != group_count:
            raise RuntimeError("generated atomic calibration group keys are not unique")

        source_digest = hashlib.sha256()
        with source_path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                source_digest.update(block)

        destination = Path(output_path).expanduser().resolve()
        if destination.suffix != ".npz":
            destination = Path(f"{destination}.npz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            schema=np.asarray(ATOMIC_CALIBRATION_SCHEMA_VERSION, np.int64),
            calibration_name=np.asarray(calibration_name.strip()),
            source_catalog_sha256=np.asarray(source_digest.hexdigest()),
            key=np.asarray(group_keys, dtype="U64"),
            component_count=np.bincount(
                component_group, minlength=group_count
            ).astype(np.int64),
            delta_loggf_dex=corrections["loggf"],
            delta_log_vdw_dex=corrections["vdw"],
            delta_log_radiative_dex=corrections["radiative"],
            delta_log_stark_dex=corrections["stark"],
            component_group_index=component_group,
            component_row_signature_sha256=component_signatures,
            component_occurrence_ordinal=component_ordinals,
            scope_wavelength_nm=scope_wavelength,
            scope_atomic_number=scope_atomic_number,
            scope_line_type=scope_line_type,
            scope_ion_stage=scope_ion_stage,
        )
        inventory = validate_atomic_calibration(destination)
        result: dict[str, object] = {
            **inventory,
            "output_path": str(destination),
            "overlay_path": str(destination),
            "selected_transition_count": len(self.transitions),
            "zero_passthrough_group_count": group_count - len(self.transitions),
        }
        if substituted_catalog_path is not None:
            application = write_corrected_catalog(
                catalog,
                destination,
                substituted_catalog_path,
                source_catalog_path=source_path,
            )
            result.update(
                {
                    "substituted_catalog_path": application["output_path"],
                    "substituted_catalog_metadata_path": application["metadata_path"],
                    "substituted_catalog_matched_rows": application[
                        "matched_catalog_rows"
                    ],
                }
            )
        return result


__all__ = [
    "AtomicTransition",
    "ResolvedAtomicTransition",
    "SynthesisLineCalibrationModel",
    "gaussian_velocity_kernel",
]
