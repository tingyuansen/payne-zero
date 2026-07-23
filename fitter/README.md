# Normalized-spectrum fitting

`fitter.normalized` is an instrument-agnostic weighted fitter. It operates on
the observed wavelength grid, supports arbitrary nonlinear parameters, profiles
a linear multiplicative continuum at every trial, and stores every accepted
parameter vector, spectrum, and objective together with the runtime of every
model or Jacobian callback. The user supplies the forward callback, so
synthesis, LSF convolution, Doppler shifting, and resampling remain explicit
rather than being silently assumed.

From a checkout, run `./install.sh` at the repository root before using the
interface.

Run a complete deterministic example before connecting a synthesis callback:

```bash
python -m fitter.examples.fit_normalized --out results/toy_normalized_fit
```

It recovers two absorption-line strengths while profiling a linear continuum
and writes the same summary and trace products as a user-supplied model.

## Python interface

The callback-oriented example below is schematic. The observed arrays come
from the user's reduced spectrum, and the callback must return model flux on
that same wavelength grid.

```python
import numpy as np
from fitter import FitConfiguration, NormalizedSpectrum, fit_normalized_spectrum

spectrum = NormalizedSpectrum(wavelength, normalized_flux, inverse_variance, mask)
configuration = FitConfiguration(
    names=("teff", "logg", "metallicity"),
    initial=np.array([4800.0, 2.5, -0.2]),
    lower=np.array([4000.0, 0.7, -2.5]),
    upper=np.array([6000.0, 4.0, 0.5]),
    derivative_steps=np.array([50.0, 0.05, 0.05]),
    trust_half_width=np.array([300.0, 0.3, 0.3]),
)

def observed_grid_model(parameters):
    # Return normalized model flux on `wavelength`, including the instrument
    # response and any required wavelength projection.
    return model_flux

result = fit_normalized_spectrum(
    spectrum,
    configuration,
    observed_grid_model,
    continuum_basis=continuum_basis,
)
result.save("results/my_star")
```

Rotation can be applied before an instrument response with the optional
device-resident Gray-profile operator:

```python
import torch
from fitter import RotationalBroadening

rotation = RotationalBroadening(
    native_log_wavelength_nm,
    maximum_vsini_km_s=100.0,
    limb_darkening=0.6,
    device=native_flux.device,
    dtype=native_flux.dtype,
)
rotated_flux = rotation(native_flux, vsini_km_s=12.0)
```

Construction validates the uniform log-wavelength grid and caches its velocity
geometry on CPU, CUDA, or Apple Metal. Repeated application has no NumPy or
host-device round trip. Flux derivatives remain in the Torch computational
graph. The scalar `vsini_km_s` interface is intended for finite-difference or
other bounded optimization and is not differentiable with respect to
`vsini_km_s` itself. The APOGEE reference fit does not enable this optional
operator.

Pass `jacobian=` when an analytic or autodifferentiable Jacobian of the
continuum-profiled observed-grid model is available; otherwise the fitter uses
bounded one-sided differences.

## Converged physical-atmosphere refinement

The fast fit can be checked and refined with a callback that solves a physical
atmosphere to convergence and synthesizes its observed-grid spectrum:

```python
from fitter import PhysicalAtmosphereConfiguration, refine_with_physical_atmosphere

physical_result = refine_with_physical_atmosphere(
    spectrum,
    configuration,
    result,
    observed_grid_model,
    converged_atmosphere_model,
    PhysicalAtmosphereConfiguration(
        maximum_discrepancy_rms=2.0e-3,
        maximum_objective_degradation=0.1,
        minimum_predicted_objective_improvement=1.0e-3,
        maximum_physical_evaluations=4,
        correction_derivative_steps=np.array([25.0, 0.03, 0.03]),
        correction_trust_half_width=np.array([150.0, 0.15, 0.15]),
    ),
    continuum_basis=continuum_basis,
)
physical_result.save("results/my_star")
```

The first two thresholds gate the profiled flux discrepancy and the increase in
mean weighted squared residual. If either gate fails, a fresh local fast-model
Jacobian proposes a bounded correction while holding the local
physical-minus-fast discrepancy fixed. Another atmosphere is solved only when
that proposal has the declared predicted gain, and it is accepted only when
the converged physical model improves the objective. The evaluation limit is a
user-selected safety cap, not a fixed requirement of the method. Both callbacks
use the same continuum profiling, and the saved trace includes every physical
check, correction proposal, outcome, resolved numerical control, and runtime
without plotting. Omit the two correction-scale arrays to derive them from
`FitConfiguration`.

`PhysicalAtmosphereResult` reports `physical_fit_stationary` and
`fast_physical_gates_passed` separately. The first means that the local
discrepancy-corrected physical fit reached the declared step or predicted-gain
threshold. The second means that the accepted fast and converged-atmosphere
spectra pass both agreement gates. `successful` is true when either valid
stopping route succeeds. Timing fields do not overlap. `correction_seconds`
contains only Jacobian assembly, continuum profiling, and linear-algebra
overhead outside the separately recorded callbacks. Each correction also
records its selected backtracking fraction and actual step norm in trust-scaled
and parameter units.

## APOGEE interface

[`apogee/`](apogee/) packages a representative DR14 all-slit mean LSF,
residual-RV and broadening operators, chip-wise continuum profiling,
atomic-calibration projection, initializer-atmosphere synthesis forward model,
and the established fast optimizer. It accepts prepared normalized apStar
arrays without downloading survey data or assuming a survey catalog. The
convenience entry point does not automatically run the converged-atmosphere
refinement described above. See its README for the array API, LSF scope, and
`payne-zero-fit-apogee` command.

This installed APOGEE interface exposes exactly two stellar-label families:
the five standard labels and the eight-label CNO family. The optional
direct-[X/H] initializer is not exposed by `fit_apogee_spectrum` or
`payne-zero-fit-apogee`.
