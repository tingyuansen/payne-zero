"""Compact array-based interface to the existing APOGEE optimizer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .forward_model import _device_and_dtype, _resolve_synthesis_r_grid
from .lsf import DEFAULT_ASSET
from .optimizer import (
    INITIAL_LABEL_MODES,
    INITIAL_RV_MODES,
    NormalizedSpectrumInput,
    run_one_star,
)


def fit_apogee_spectrum(
    result_dir: str | Path,
    *,
    object_id: str,
    wavelength_nm: np.ndarray,
    normalized_flux: np.ndarray,
    inverse_variance: np.ndarray,
    reference_labels: np.ndarray,
    reference_vmacro_km_s: float,
    good_pixel_mask: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    c_over_m: float | None = None,
    n_over_m: float | None = None,
    o_over_m: float | None = None,
    carbon_enhancement: float | None = None,
    nitrogen_enhancement: float | None = None,
    oxygen_enhancement: float | None = None,
    atomic_calibration_path: str | Path | None = None,
    device: str = "auto",
    dtype: str = "auto",
    torch_threads: int | None = None,
    synthesis_r_grid: float | None = None,
    synthesis_resolution: float | None = None,
    fresh_jacobian_rounds: int = 1,
    continuum_order: int = 2,
    fit_cno8: bool = False,
    initial_label_mode: str = "reference",
    initial_rv_mode: str = "rest_frame",
    compact_trace: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Fit one normalized APOGEE spectrum and write its complete fit summary.

    Spectrum arrays have shape ``(7514,)`` on the retained apStar grid.
    ``wavelength_nm`` is in nm, ``normalized_flux`` is dimensionless, and
    ``inverse_variance`` is its inverse variance. ``good_pixel_mask=True``
    requests inclusion; finite flux, finite inverse variance, and positive
    inverse variance are also required. Inputs are converted to ``float64`` and
    the mask to Boolean.

    ``reference_labels`` has shape ``(5,)`` and contains ``(Teff, logg, [M/H],
    [alpha/M], vmicro)`` in K, dex, dex, dex, and km/s.
    ``reference_vmacro_km_s`` is in km/s. The reference
    values initialize the optimizer by default and remain an external
    comparison point; they are not treated as truth. Set
    ``initial_label_mode="controlled_offset"`` only to reproduce the displaced
    start used by the controlled recovery experiment. APOGEE spectra are
    normally already rest-framed, so ``initial_rv_mode="rest_frame"`` starts
    the residual velocity at zero without calculating a CCF. Use
    ``initial_rv_mode="coarse_ccf"`` when the input needs a coarse velocity
    initialization. ``synthesis_r_grid`` is the intrinsic logarithmic
    model-grid density before LSF convolution, not the APOGEE instrumental
    resolving power; it defaults to 300,000.
    ``synthesis_resolution`` is a compatibility
    alias and must not be supplied with a different value.
    When ``fit_cno8`` is true, ``c_over_m``, ``n_over_m``, and ``o_over_m`` are
    required starting coordinates and the fit order extends to ``[C/M]``,
    ``[N/M]``, and ``[O/M]`` after the five reference labels.
    """

    preferred_cno = (c_over_m, n_over_m, o_over_m)
    legacy_cno = (carbon_enhancement, nitrogen_enhancement, oxygen_enhancement)
    resolved_cno: list[float | None] = []
    for name, preferred, legacy in zip(
        ("[C/M]", "[N/M]", "[O/M]"), preferred_cno, legacy_cno, strict=True
    ):
        if preferred is not None and legacy is not None:
            raise ValueError(f"supply {name} through only one Python keyword")
        selected = preferred if preferred is not None else legacy
        if selected is not None and not np.isfinite(float(selected)):
            raise ValueError(f"the starting {name} value must be finite")
        resolved_cno.append(selected)
    carbon_enhancement, nitrogen_enhancement, oxygen_enhancement = resolved_cno

    labels = np.asarray(reference_labels, np.float64)
    if labels.shape != (5,):
        raise ValueError("reference_labels must contain exactly five values")
    if not np.all(np.isfinite(labels)):
        raise ValueError("reference_labels must be finite")
    reference_vmacro = float(reference_vmacro_km_s)
    if not np.isfinite(reference_vmacro):
        raise ValueError("reference_vmacro_km_s must be finite")
    flux = np.asarray(normalized_flux, np.float64)
    wavelength = np.asarray(wavelength_nm, np.float64)
    ivar = np.asarray(inverse_variance, np.float64)
    if good_pixel_mask is None:
        mask = np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0.0)
    else:
        mask = np.asarray(good_pixel_mask, bool)
    if not (wavelength.shape == flux.shape == ivar.shape == mask.shape):
        raise ValueError("wavelength, flux, inverse variance, and mask shapes differ")
    with np.load(DEFAULT_ASSET, allow_pickle=False) as asset:
        retained_wavelength = np.asarray(asset["wavelength_nm"], np.float64)
    if wavelength.shape != retained_wavelength.shape or not np.allclose(
        wavelength, retained_wavelength, rtol=0.0, atol=5.0e-10
    ):
        raise ValueError(
            "wavelength_nm must match the retained-pixel grid in the packaged "
            "APOGEE DR14 LSF asset"
        )
    if continuum_order < 0 or continuum_order > 4:
        raise ValueError("continuum_order must be between zero and four")
    if fresh_jacobian_rounds < 0:
        raise ValueError("fresh_jacobian_rounds must be nonnegative")
    if initial_label_mode not in INITIAL_LABEL_MODES:
        raise ValueError(f"initial_label_mode must be one of {INITIAL_LABEL_MODES}")
    if initial_rv_mode not in INITIAL_RV_MODES:
        raise ValueError(f"initial_rv_mode must be one of {INITIAL_RV_MODES}")

    resolved_r_grid = _resolve_synthesis_r_grid(synthesis_r_grid, synthesis_resolution)
    resolved_device, resolved_dtype = _device_and_dtype(device, dtype)
    if resolved_device == "cpu" and torch_threads is not None:
        if torch_threads < 1:
            raise ValueError("torch_threads must be positive")
        torch.set_num_threads(torch_threads)

    calibration = (
        None
        if atomic_calibration_path is None
        else Path(atomic_calibration_path).expanduser().resolve()
    )
    spectrum = NormalizedSpectrumInput(
        object_id=str(object_id),
        wavelength_nm=wavelength,
        observed_flux=flux,
        inverse_variance=ivar,
        good_pixel_mask=mask,
        catalog_labels=labels,
        catalog_vmacro_km_s=reference_vmacro,
        metadata={**(metadata or {}), "object_id": str(object_id)},
        data_mode="apogee_normalized_spectrum",
        reference_is_truth=False,
        carbon_enhancement=carbon_enhancement,
        nitrogen_enhancement=nitrogen_enhancement,
        oxygen_enhancement=oxygen_enhancement,
        atomic_calibration_path=calibration,
    )
    return run_one_star(
        Path(result_dir).expanduser().resolve(),
        device=resolved_device,
        dtype=resolved_dtype,
        synthesis_r_grid=resolved_r_grid,
        fresh_jacobian_rounds=fresh_jacobian_rounds,
        force=force,
        spectrum_input=spectrum,
        continuum_order=continuum_order,
        fit_cno8=fit_cno8,
        initial_label_mode=initial_label_mode,
        initial_rv_mode=initial_rv_mode,
        compact_trace=compact_trace,
    )
