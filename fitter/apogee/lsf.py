"""Device-resident representative APOGEE DR14 combo-LSF operator.

The packaged asset is a calibration-derived all-slit mean, not the per-star
four-library ASPCAP selection.  It averages the six representative fibers used
by the reference ``jobovy/apogee`` ``fiber="combo"`` implementation.

Payne Zero's ``resolution`` controls logarithmic grid sampling; it does not add
an instrumental Gaussian.  This operator therefore folds linear resampling
from that finite synthesis grid into the measured LSF exactly, rather than
subtracting two nominal resolving powers in quadrature.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch


FITTER_ROOT = Path(__file__).resolve().parent
DEFAULT_ASSET = FITTER_ROOT / "data" / "apogee_dr14_combo_lsf.npz"


def _banded_apply(
    flux_batch: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Apply row-dependent compact weights to ``[batch, wavelength]`` flux."""

    return (flux_batch[:, indices] * weights.unsqueeze(0)).sum(dim=-1)


class APOGEEDR14LSF:
    """Finite-grid-aware, wavelength-dependent DR14 combo-LSF operator.

    Construction happens once. The interpolation and LSF weights and indices
    then remain resident on the selected device for every synthesis call.
    """

    name = "apogee_dr14_r8_combo_mean"

    def __init__(
        self,
        input_wavelength_nm: np.ndarray,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
        asset_path: str | Path = DEFAULT_ASSET,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.asset_path = Path(asset_path)
        self.last_seconds = 0.0

        setup_start = time.perf_counter()
        wavelength = np.asarray(input_wavelength_nm, np.float64)
        if wavelength.ndim != 1 or wavelength.size < 2:
            raise ValueError("input_wavelength_nm must be a one-dimensional grid")
        if not np.all(np.diff(wavelength) > 0.0):
            raise ValueError("input_wavelength_nm must be strictly increasing")
        self.input_wavelength_nm = wavelength

        with np.load(self.asset_path, allow_pickle=False) as asset:
            self.output_wavelength_nm = np.asarray(asset["wavelength_nm"], np.float64)
            apstar_pixel = np.asarray(asset["apstar_pixel"], np.int64)
            tap_offset = np.asarray(asset["tap_offset_oversampled_pixel"], np.int64)
            native_weights = np.asarray(asset["kernel_weights"], np.float64)
            self.asset_metadata = json.loads(str(asset["metadata_json"]))
        self.apstar_pixel = apstar_pixel

        if self.asset_metadata.get("asset_schema_version") != 1:
            raise ValueError(f"unsupported APOGEE LSF asset: {self.asset_path}")
        if self.output_wavelength_nm.shape != apstar_pixel.shape:
            raise ValueError("LSF output wavelengths and pixels differ in length")
        if not np.all(np.diff(self.output_wavelength_nm) > 0.0):
            raise ValueError("LSF output wavelengths must be strictly increasing")
        if not np.all(np.diff(apstar_pixel) > 0):
            raise ValueError("LSF apStar pixels must be strictly increasing")
        if (
            not np.all(np.isfinite(native_weights))
            or np.any(native_weights < 0.0)
            or np.any(np.sum(native_weights, axis=1) <= 0.0)
        ):
            raise ValueError("LSF kernel weights must be finite and nonnegative")

        indices, weights = self._fold_linear_resampling(
            wavelength,
            apstar_pixel,
            tap_offset,
            native_weights,
        )
        self.maximum_banded_taps = int(indices.shape[1])
        self.nonzero_weights = int(np.count_nonzero(weights))
        self._indices = torch.as_tensor(indices, dtype=torch.int64, device=self.device)
        self._weights = torch.as_tensor(weights, dtype=self.dtype, device=self.device)
        self.implementation = "banded_eager"
        self.setup_seconds = time.perf_counter() - setup_start
        self.benchmark_seconds_per_flux_pair = float("nan")

    @staticmethod
    def _fold_linear_resampling(
        input_wavelength_nm: np.ndarray,
        apstar_pixel: np.ndarray,
        tap_offset: np.ndarray,
        native_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fold synthesis-grid interpolation into compact row weights once."""

        if native_weights.shape != (apstar_pixel.size, tap_offset.size):
            raise ValueError("LSF asset arrays have inconsistent shapes")
        oversampled_index = 3 * apstar_pixel[:, None] + tap_offset[None, :]
        tap_wavelength_nm = 0.1 * np.power(10.0, 4.179 + oversampled_index * 2.0e-6)
        right = np.searchsorted(input_wavelength_nm, tap_wavelength_nm, side="right")
        if np.any(right == 0) or np.any(right == input_wavelength_nm.size):
            raise ValueError(
                "synthesis wavelength grid does not cover every APOGEE LSF tap"
            )
        left = right - 1
        fraction_right = (tap_wavelength_nm - input_wavelength_nm[left]) / (
            input_wavelength_nm[right] - input_wavelength_nm[left]
        )
        raw_indices = np.concatenate((left, right), axis=1)
        raw_weights = np.concatenate(
            (
                native_weights * (1.0 - fraction_right),
                native_weights * fraction_right,
            ),
            axis=1,
        )

        # Adjacent subpixel taps usually interpolate from the same synthesis
        # samples.  Merge those duplicates now so the repeated GPU kernel reads
        # only the minimal number of input values.
        merged_indices: list[np.ndarray] = []
        merged_weights: list[np.ndarray] = []
        maximum_taps = 0
        for row_indices, row_weights in zip(raw_indices, raw_weights, strict=True):
            unique, inverse = np.unique(row_indices, return_inverse=True)
            combined = np.zeros(unique.size, np.float64)
            np.add.at(combined, inverse, row_weights)
            keep = combined > np.finfo(np.float32).tiny
            unique = unique[keep]
            combined = combined[keep]
            combined /= np.sum(combined)
            merged_indices.append(unique)
            merged_weights.append(combined)
            maximum_taps = max(maximum_taps, unique.size)

        indices = np.zeros((apstar_pixel.size, maximum_taps), np.int64)
        weights = np.zeros((apstar_pixel.size, maximum_taps), np.float64)
        for row, (row_indices, row_weights) in enumerate(
            zip(merged_indices, merged_weights, strict=True)
        ):
            count = row_indices.size
            indices[row, :count] = row_indices
            weights[row, :count] = row_weights
        return indices, weights

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def prepare(self, *, repeats: int = 30) -> float:
        """Warm and time the production banded operator once."""

        if repeats < 1:
            raise ValueError("repeats must be positive")
        prepare_start = time.perf_counter()
        dummy = torch.linspace(
            0.7,
            1.1,
            self.input_wavelength_nm.size,
            dtype=self.dtype,
            device=self.device,
        ).repeat(2, 1)
        for _ in range(3):
            _banded_apply(dummy, self._indices, self._weights)
        self._synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            _banded_apply(dummy, self._indices, self._weights)
        self._synchronize()
        self.benchmark_seconds_per_flux_pair = (time.perf_counter() - start) / repeats
        return time.perf_counter() - prepare_start

    def convolve_fluxes(
        self,
        total_flux: torch.Tensor,
        continuum_flux: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convolve total/continuum together and return their normalized ratio."""

        if total_flux.shape != continuum_flux.shape:
            raise ValueError("total and continuum flux shapes differ")
        if total_flux.ndim != 1:
            raise ValueError("instrument operator expects one spectrum at a time")
        self._synchronize()
        start = time.perf_counter()
        batch = torch.stack((total_flux, continuum_flux), dim=0)
        convolved = _banded_apply(batch, self._indices, self._weights)
        normalized = convolved[0] / convolved[1]
        self._synchronize()
        self.last_seconds = time.perf_counter() - start
        return convolved[0], convolved[1], normalized

    def metadata(self) -> dict[str, object]:
        """Serializable setup, correctness, and steady-state benchmark details."""

        return {
            "name": self.name,
            "asset_path": str(self.asset_path),
            "asset_metadata": self.asset_metadata,
            "input_pixels": int(self.input_wavelength_nm.size),
            "output_pixels": int(self.output_wavelength_nm.size),
            "maximum_banded_taps": self.maximum_banded_taps,
            "nonzero_weights": self.nonzero_weights,
            "implementation": self.implementation,
            "setup_seconds": self.setup_seconds,
            "benchmark_seconds_per_flux_pair": (self.benchmark_seconds_per_flux_pair),
        }
