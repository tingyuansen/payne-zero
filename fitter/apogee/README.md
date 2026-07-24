# APOGEE normalized-spectrum fitting

This package is the reusable boundary of the APOGEE fitter. It applies the packaged representative DR14 all-slit mean line-spread function after macroscopic broadening and a residual Doppler shift, profiles one Legendre continuum per retained detector chip, and runs a physical-synthesis optimizer on the retained APOGEE pixel grid. The bundled kernel is not an official per-star ASPCAP LSF. Analyses that require a visit-, fiber-, or star-specific LSF should supply their own instrument operator through `fitter.normalized`. This package does not acquire survey files, select catalog samples, or schedule cluster jobs.

`fit_apogee_spectrum` and `payne-zero-fit-apogee` run the fast search with the learned atmosphere initializer followed by physical spectral synthesis. They do not automatically solve the final atmosphere to convergence. A complete physical-atmosphere fit must add the converged-model callback through `fitter.refine_with_physical_atmosphere`; that correction remains explicit so the output is not mistaken for a converged atmosphere result.

The installed array API and command expose exactly two stellar-label families: the five standard labels and the eight-label CNO-aware family selected with `fit_cno8=True`. They do not expose the optional released direct-[X/H] initializer, whose validation is more limited and which requires a converged physical solve through its explicit atmosphere API.

The array API example below fits the public DR14 spectrum bundled with the tutorial. It uses a reduced synthesis-grid density so the example remains practical on CPU and laptop GPUs.

```python
import json
import numpy as np
from pathlib import Path
from fitter.apogee import fit_apogee_spectrum

data_path = Path("examples/data/apogee_dr14_example.npz")
metadata = json.loads(Path("examples/data/apogee_dr14_example.json").read_text())
reference_labels = np.array([
    metadata["effective_temperature"],
    metadata["log_surface_gravity"],
    metadata["metallicity"],
    metadata["alpha_enhancement"],
    metadata["microturbulence_km_s"],
])

with np.load(data_path, allow_pickle=False) as spectrum:
    summary = fit_apogee_spectrum(
        "tutorial_output/apogee_readme_fit",
        object_id=metadata["object_id"],
        wavelength_nm=spectrum["wavelength_nm"],
        normalized_flux=spectrum["normalized_flux"],
        inverse_variance=spectrum["inverse_variance"],
        good_pixel_mask=spectrum["good_pixel_mask"],
        reference_labels=reference_labels,
        reference_vmacro_km_s=metadata["macroscopic_broadening_km_s"],
        device="auto",
        dtype="auto",
        synthesis_r_grid=20_000.0,
        atomic_calibration_path=(
            "linelist_calibration/data/sun_arcturus_fts_hband_shared.npz"
        ),
        fresh_jacobian_rounds=0,
        force=True,
    )
```

The label order is `Teff`, `logg`, `[M/H]`, `[alpha/M]`, and microturbulence. The wavelength, flux, weights, and mask must represent the 7,514 retained apStar pixels described by the bundled LSF asset. Reference labels initialize the established optimizer and remain an external comparison point, not truth. The default `initial_label_mode="reference"` uses those values directly. `initial_label_mode="controlled_offset"` is available only to reproduce the fixed displaced start used by the controlled recovery experiment. Combined APOGEE spectra are already approximately rest-framed, so the default `initial_rv_mode="rest_frame"` starts the residual velocity at zero and does not calculate an unused CCF. For spectra that still need a coarse velocity start, set `initial_rv_mode="coarse_ccf"`. The optional synthesis density is `R_grid=300,000` by default and is applied before the instrumental LSF. The historical Python keyword `synthesis_resolution=` remains an alias for `synthesis_r_grid=`.

Set `fit_cno8=True` together with finite `c_over_m`, `n_over_m`, and `o_over_m` starts to fit the ordered stellar coordinates `Teff`, `logg`, `[M/H]`, `[alpha/M]`, microturbulence, `[C/M]`, `[N/M]`, and `[O/M]`. The residual velocity and macroscopic broadening remain separate nuisance coordinates. The command-line equivalents are `--fit-cno8`, `--c-over-m`, `--n-over-m`, and `--o-over-m`; all three abundance starts are required when the CNO8 fit is enabled.

`device="auto"` selects CUDA, then Apple Metal, then CPU. In this APOGEE fitter, `dtype="auto"` uses `float32` synthesis on CUDA or Metal and `float64` on CPU; this accelerator-oriented policy is intentionally distinct from the core synthesis API's `float64` CUDA default. Explicit `float32` and `float64` are accepted on CUDA/CPU, while Metal requires `float32`. Invalid device or dtype names fail before model setup.

## Optional atomic calibration

`atomic_calibration_path=` is optional and leaves the source catalog unchanged. New public products use correction-only schema 4 with:

- scalar integer `schema=4` and scalar string `calibration_name`;
- unique opaque SHA-256 group identifiers in `key`, plus finite group vectors `delta_loggf_dex`, `delta_log_vdw_dex`, `delta_log_radiative_dex`, and `delta_log_stark_dex`;
- integer `component_group_index`, which maps each component to one group;
- `component_row_signature_sha256` and integer `component_occurrence_ordinal`, which identify exact rows and duplicate occurrences without publishing the source values;
- coarse wavelength, atomic-number, line-type, and ion-stage scope vectors; and
- a required scalar `source_catalog_sha256`. Optional parameter, evidence, and grouping digests bind the retained provenance.

Before applying schema 4, the fitter resolves the exact user-local source catalog through the synthesis data path and verifies its complete file digest. It then recomputes each opaque row identity and projects the corrections onto a private catalog copy. A full-window overlay may contain rows not retained by an active synthesis window, but every in-scope active row must match. A source-hash mismatch, missing row, incomplete duplicate group, or incomplete active-row coverage fails before synthesis.

`fitter.apogee.validate_atomic_calibration(path)` checks the standalone format without constructing a forward model. Schemas 2 and 3 and unversioned products remain readable for legacy compatibility, including their cleartext identities and optional absolute values, but schema 4 is the public format. The generic `linelist_calibration.write_substituted_catalog` helper writes a complete calibrated catalog to a new path and requires the exact source path for schema 4.

The equivalent command is:

```bash
payne-zero-fit-apogee examples/data/apogee_dr14_example.npz tutorial_output/apogee_readme_cli \
  --object-id 2M08002084+4044415 \
  --reference-labels 4858.537 2.426797 -0.3255714 0.2527028 1.278733 \
  --reference-vmacro 3.617606 \
  --synthesis-r-grid 20000 \
  --fresh-jacobian-rounds 0 \
  --atomic-calibration linelist_calibration/data/sun_arcturus_fts_hband_shared.npz \
  --force
```

Add `--initial-rv-mode coarse-ccf` when the prepared spectrum is not already rest-framed. `--initial-label-mode controlled-offset` reproduces the controlled recovery start and is not the normal fitting default.

The NPZ requires `wavelength_nm`, `normalized_flux`, and `inverse_variance`; `good_pixel_mask` is optional and otherwise derived from finite positive weights. The command writes the same summary and optimization trace as the Python API. Use `fitter.normalized` instead when the spectrum is not on the APOGEE retained-pixel grid or needs a different instrument model.
