# Differentiable line-list calibration

`linelist_calibration` fits continuous atomic parameters through differentiable Payne Zero synthesis. The physical model holds a converged atmosphere fixed, recalculates the selected line opacity and transfer solution, and compares the broadened result with a standard-star spectrum. The generic optimizer can combine several stars, normalization, and instrument responses in one PyTorch callback.

From a checkout, run `./install.sh` at the repository root before using the interface.

## Real solar FTS example

The bundled example calibrates the oscillator strength of one Fe I transition against a small excerpt of the Livingston and Wallace solar Fourier-transform-spectrometer (FTS) atlas. It uses a structured solar atmosphere, the active atomic catalog, the physical opacity and transfer calculation, the registered velocity, and the atlas broadening:

```bash
python -m linelist_calibration.examples.fit_solar_fts_line \
  --output-dir results/solar-fts-line-calibration
```

The example changes one `log(gf)` value and writes the optimizer trace, the observed, baseline, and calibrated spectra, and a source-bound schema-4 overlay that can be applied in later fits. On the bundled line, the fitted correction is about `+0.80 dex`, close to the value retained by the many-line Sun–Arcturus calibration.

The same calculation can be assembled explicitly:

```python
import torch

from linelist_calibration import calibrate_line_parameters
from linelist_calibration.examples.fit_solar_fts_line import build_example

data, configuration, physical_model, atlas_metadata = build_example(device="auto")
model = physical_model.callback(("loggf",))
result = calibrate_line_parameters(
    data,
    configuration,
    model,
)
result.save("results/solar-fts-line-calibration")
physical_model.write_atomic_calibration_overlay(
    result.values,
    "results/solar-fts-line-calibration/solar_fe_i_overlay.npz",
    parameter_families=("loggf",),
    calibration_name="my_solar_fe_i_calibration",
)

calibrated_flux = model(
    torch.as_tensor(
        result.values,
        device=physical_model.device,
        dtype=physical_model.dtype,
    )
).detach().cpu().numpy()
```

`CalibrationData` stores the observed flux and nonnegative quality weights. `CalibrationConfiguration` stores the initial values, bounds, names, tolerances, device, and data type. `SynthesisLineCalibrationModel` resolves each requested physical transition against the active catalog, evaluates unchanged opacity once, and keeps the repeated calculation on the selected device. The returned corrections are additive dex offsets.

## Use another standard star or instrument

A Gaia benchmark star, another FTS atlas, or a survey standard uses the same boundary. Supply its converged structured atmosphere, observed wavelength and normalized-flux arrays, weights, velocity registration, broadening, and the transitions to calibrate:

```python
import numpy as np

from linelist_calibration import (
    AtomicTransition,
    CalibrationConfiguration,
    CalibrationData,
    SynthesisLineCalibrationModel,
    calibrate_line_parameters,
)

observation = CalibrationData(
    flux=observed_normalized_flux,
    weight=np.where(good_pixel_mask, inverse_variance, 0.0),
)
physical_model = SynthesisLineCalibrationModel(
    "my_standard_star_atmosphere.npz",
    wavelength_start_nm=window_start_nm,
    wavelength_end_nm=window_end_nm,
    resolution=300_000,
    transitions=(
        AtomicTransition(26, 1, 1568.18024, name="Fe I 1568.180 nm"),
        AtomicTransition(14, 1, 1589.275, name="Si I 1589.275 nm"),
    ),
    observed_wavelength_nm=observed_wavelength_nm,
    radial_velocity_km_s=registered_velocity_km_s,
    gaussian_broadening_sigma_km_s=broadening_sigma_km_s,
    device="cuda",
    dtype="float32",
)
configuration = CalibrationConfiguration(
    initial=np.zeros(2),
    lower=np.full(2, -1.0),
    upper=np.full(2, 1.0),
    names=("Fe I delta log(gf)", "Si I delta log(gf)"),
    maximum_iterations=30,
    device=str(physical_model.device),
    dtype="float32",
)
result = calibrate_line_parameters(
    observation,
    configuration,
    physical_model.callback(("loggf",)),
)
export = physical_model.write_atomic_calibration_overlay(
    result.values,
    "results/my_standard_star_overlay.npz",
    parameter_families=("loggf",),
    calibration_name="my_standard_star",
)
```

`write_atomic_calibration_overlay(...)` converts the optimized vector into the same correction-only schema-4 format used by the bundled calibrations. It binds the overlay to the exact source catalog by SHA-256 and includes zero-correction pass-through groups when the rectangular schema scope contains neighboring rows. Pass the resulting path as `atomic_calibration_path=` to the APOGEE fitter. Set `substituted_catalog_path=` when a corrected copy of the active synthesis-window catalog is also useful; the source catalog is never modified.

The native synthesis window must extend beyond the observed pixels so velocity registration and broadening have context. Use `broadening_kernel=` instead of the Gaussian width for a measured shift-invariant profile. `SynthesisLineCalibrationModel` does not apply a wavelength-dependent line-spread function to total and continuum flux separately; use the survey fitter instrument operator for that case.

For a joint calibration, construct one `SynthesisLineCalibrationModel` per star with the same ordered transitions, concatenate their observed flux and weights, and return `torch.cat([model_a(theta), model_b(theta)])` from the shared callback. The same correction vector then has to explain both atmospheres. Parameter families are ordered by family and then transition, so `callback(("loggf", "vdw", "radiative", "stark"))` fits all four atomic quantities for every group.

The optimizer minimizes `sum(weight * (model - observed)**2) / sum(weight)`. Weights must be finite and nonnegative, and their meaning should be declared by the application. Samples with zero weight or non-finite observed flux are excluded. The result stores the bounded physical parameters, objective history, and per-evaluation wall times. `result.save(...)` writes `calibrated_parameters.npz` and `calibration_summary.json`.

The analytic two-star example remains as an optimizer-only installation check:

```bash
python -m linelist_calibration.examples.fit_synthetic_standard_stars \
  --output-dir results/line-calibration-smoke
```

## Correction-only atomic overlays

Schema 4 is the public line-list calibration format. It releases four fitted dex corrections per physical transition group without copying catalog rows or absolute source parameters:

| Overlay field | Applied catalog operation |
|---|---|
| `delta_loggf_dex` | add to $\log_{10}(gf)$ |
| `delta_log_vdw_dex` | multiply van der Waals damping by $10^{\Delta}$ |
| `delta_log_radiative_dex` | multiply radiative damping by $10^{\Delta}$ |
| `delta_log_stark_dex` | multiply Stark damping by $10^{\Delta}$ |

`component_group_index` maps blended catalog components to their shared correction. Each overlay is bound to the exact source catalog and identifies duplicate rows unambiguously. Application validates the source and every in-scope transition before applying the fitted deltas to a private copy.

```python
import numpy as np

from linelist_calibration import (
    apply_atomic_calibration,
    bundled_atomic_calibration,
    validate_atomic_calibration,
)
from payne_zero_synthesis import paths as synthesis_paths
from payne_zero_synthesis.atomic_lines import load_catalog

source_catalog = synthesis_paths.source_catalog_path(
    "lines", "atomic_source_lines_parsed.npz"
)
parsed_catalog = load_catalog(
    (1500.0, 1700.0), 300_000.0, catalog_path=source_catalog
)
catalog = {
    name: value
    for name, value in vars(parsed_catalog).items()
    if isinstance(value, np.ndarray)
}
overlay = bundled_atomic_calibration("sun_arcturus_fts_hband_shared")
inventory = validate_atomic_calibration(overlay)
corrected, metadata = apply_atomic_calibration(
    catalog,
    overlay,
    source_catalog_path=source_catalog,
)
```

The retained calibrations use Fourier transform spectrometer (FTS) atlases and have three scientific roles:

| Bundle | Intended use |
|---|---|
| `sun_arcturus_fts_hband_shared` | Shared Sun–Arcturus correction used by the survey demonstration |
| `sun_fts_hband` | Independent solar calibration |
| `arcturus_fts_hband_joint_epochs` | Independent joint-epoch Arcturus calibration |

The retained provenance corrections have the following meanings:

| Overlay field | Catalog correction | Principal spectral effect |
|---|---|---|
| `delta_loggf_dex` | Additive correction to log oscillator strength | Scales integrated line opacity and line strength |
| `delta_log_vdw_dex` | Multiplicative dex correction to van der Waals damping | Changes pressure-broadened wings from neutral collisions |
| `delta_log_radiative_dex` | Multiplicative dex correction to radiative damping | Changes natural broadening from the transition lifetime |
| `delta_log_stark_dex` | Multiplicative dex correction to Stark damping | Changes pressure broadening from charged-particle collisions |

Application returns a private corrected copy and never changes the source catalog. The default `catalog_scope="complete"` validates the full declared scope. Use `catalog_scope="selected_window"` only when synthesis has intentionally loaded a wavelength-selected subset; every active calibratable row must still match. The packaged standard-star overlays use schema 4, while older formats remain readable for compatibility.

The standard-star corrections are effective astrophysical calibrations rather than laboratory atomic measurements. `write_substituted_catalog(...)` writes a complete `.npz` catalog plus a `.npz.json` metadata sidecar, appending `.npz` when the requested destination does not already have that suffix.
