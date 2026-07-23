# Differentiable line-list calibration

`linelist_calibration.optimize` provides bounded LBFGS optimization for any
differentiable physical line-parameter vector. The user supplies a Torch
forward callback that returns flux with the same shape as the observed data;
the callback can include one or many spectra, normalization, broadening, an
instrument response, and any desired mapping from parameters to atomic data.

From a checkout, run `./install.sh` at the repository root before using the
interface.

## Python interface

The callback-oriented example below is schematic. The observed arrays and the
differentiable forward callback are supplied by the user's calibration data
and synthesis calculation.

```python
import numpy as np
from linelist_calibration import (
    CalibrationConfiguration,
    CalibrationData,
    calibrate_line_parameters,
)

data = CalibrationData(observed_flux, objective_weight)
configuration = CalibrationConfiguration(
    initial=np.zeros(parameter_count),
    lower=np.full(parameter_count, -1.5),
    upper=np.full(parameter_count, 1.5),
    maximum_iterations=100,
    device="cuda",
    dtype="float64",
)

def differentiable_spectrum(parameters):
    return model_flux  # Torch tensor, same shape as observed_flux

result = calibrate_line_parameters(data, configuration, differentiable_spectrum)
result.save("results/my_line_list")
```

The optimizer minimizes
`sum(weight * (model - observed)**2) / sum(weight)`. Weights need not be formal
inverse variances, but they must be finite and nonnegative and their meaning
must be declared by the application. Samples with zero weight or non-finite
observed flux are excluded. The result stores the bounded physical parameters,
the full objective history, and per-evaluation wall times that include the
callback, objective, and gradient work while excluding result I/O and plotting.
`result.save(...)` writes `calibrated_parameters.npz` and
`calibration_summary.json` below the requested directory.

## Working Sun/Arcturus example

The reusable optimizer and overlay operations are Python APIs. The module
command below is a self-contained example CLI, not an atlas- or survey-specific
calibration command.

The self-contained example jointly fits two synthetic line-strength
corrections to analytic Sun and Arcturus profiles:

```bash
python -m linelist_calibration.examples.fit_synthetic_standard_stars \
  --output-dir results/standard-star-example
```

The analytic profiles stand in for a differentiable synthesis callback, so the
example requires no atlas download or precomputed atmosphere. Replace the
callback with the differentiable forward model appropriate to your data.

## Correction-only atomic overlays

Schema 4 is the public line-list calibration format. It releases four fitted
dex corrections per physical transition group without copying catalog rows or
absolute source parameters:

| Overlay field | Applied catalog operation |
|---|---|
| `delta_loggf_dex` | add to $\log_{10}(gf)$ |
| `delta_log_vdw_dex` | multiply van der Waals damping by $10^{\Delta}$ |
| `delta_log_radiative_dex` | multiply radiative damping by $10^{\Delta}$ |
| `delta_log_stark_dex` | multiply Stark damping by $10^{\Delta}$ |

`component_group_index` maps each calibrated component to its correction group.
The component itself is represented only by a canonical SHA-256 row signature
and a zero-based occurrence ordinal for exact duplicate rows. The signature
serializes the documented identity fields into fixed little-endian int64 and
float64 values before hashing. It is therefore stable across NumPy dtype widths
and host byte order without publishing those values.

Every schema-4 overlay requires `source_catalog_sha256`. Application hashes the
exact user-local base-catalog file and fails before changing any value if the
digest differs. It then recomputes the opaque row identities from the supplied
parsed catalog, requires complete coverage of the declared wavelength, element,
ion, and line-type scope, and applies only the fitted deltas.

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

The retained calibrations have three scientific roles:

| Bundle | Intended use |
|---|---|
| `sun_arcturus_fts_hband_shared` | Shared Sun--Arcturus correction used by the survey demonstration |
| `sun_fts_hband` | Independent solar calibration |
| `arcturus_fts_hband_joint_epochs` | Independent joint-epoch Arcturus calibration |

The retained provenance corrections have the following meanings:

| Overlay field | Catalog correction | Principal spectral effect |
|---|---|---|
| `delta_loggf_dex` | Additive correction to log oscillator strength | Scales integrated line opacity and line strength |
| `delta_log_vdw_dex` | Multiplicative dex correction to van der Waals damping | Changes pressure-broadened wings from neutral collisions |
| `delta_log_radiative_dex` | Multiplicative dex correction to radiative damping | Changes natural broadening from the transition lifetime |
| `delta_log_stark_dex` | Multiplicative dex correction to Stark damping | Changes pressure broadening from charged-particle collisions |

Application returns a private corrected copy and never changes the source
catalog. The default `catalog_scope="complete"` requires
every overlay component to match an exact supplied-catalog row and requires the
overlay to cover every calibratable row in its declared wavelength and element
range. `write_substituted_catalog(...)` uses this strict contract and also
requires `source_catalog_path` for schema 4.

An active synthesis catalog may be a selected subset of the catalog represented
by a portable overlay. Pass `catalog_scope="selected_window"` only for this
case. Overlay components absent from the selected catalog are reported in the
returned metadata, while every calibratable selected row still requires an
exact match. Identity mismatches, incomplete selected-row coverage, and
ambiguous partial duplicate groups fail rather than inheriting an arbitrary
correction. The APOGEE fitter uses this selected-window contract internally.

Schemas 2 and 3 and unversioned legacy overlays remain readable for backward
compatibility. The packaged standard-star overlays use schema 4.

The standard-star corrections are effective astrophysical calibrations rather
than laboratory atomic measurements.
`write_substituted_catalog(...)` writes a complete `.npz` catalog plus a
`.npz.json` metadata sidecar, appending `.npz` when the requested destination
does not already have that suffix.
