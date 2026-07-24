# Normalized-spectrum fitting

`fitter.normalized` is an instrument-agnostic weighted fitter. It operates on the observed wavelength grid, supports arbitrary nonlinear parameters, profiles a linear multiplicative continuum at every trial, and stores every accepted parameter vector, spectrum, and objective together with the runtime of every model or Jacobian callback. The user supplies the forward callback, so synthesis, line-spread-function (LSF) convolution, Doppler shifting, and resampling remain explicit rather than being silently assumed.

From a checkout, run `./install.sh` at the repository root before using the interface.

The examples use fast synthesis from labels for repeated trial spectra. Use the converged-atmosphere refinement when the retained result must be synthesized from a physically converged atmosphere.

## Fit a spectrum from any instrument

The generic path has four explicit pieces: the observed arrays, a Payne Zero label-to-spectrum callback, an instrument operator, and the fit configuration. The example below assumes that `wavelength_nm`, `flux`, `inverse_variance`, and `good_pixel_mask` came from a survey reduction or a standard-star atlas.

`NormalizedSpectrum` accepts NumPy-compatible one-dimensional arrays of the same length. `wavelength` must be finite and strictly increasing. It is in nanometers when used with Payne Zero synthesis and the operators below. `flux` is dimensionless normalized flux, and `inverse_variance` is its inverse variance. A `True` mask entry requests that pixel; validation also requires finite flux, finite inverse variance, and positive inverse variance. At least two valid pixels are required. Inputs are converted to `float64` and the mask to Boolean.

First synthesize a native reference grid slightly wider than the observed interval. Then construct a constant-resolution projection. The output wavelength need not be uniform:

```python
import numpy as np
import torch

from fitter import ObservedSpectrumOperator
from payne_zero_synthesis import synthesize_from_labels

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float32 if device != "cpu" else torch.float64
dtype_name = "float32" if torch_dtype == torch.float32 else "float64"

native = synthesize_from_labels(
    effective_temperature=4800,
    log_surface_gravity=2.5,
    metallicity=-0.3,
    alpha_enhancement=0.2,
    microturbulence_km_s=1.5,
    wavelength_start_nm=0.9995 * wavelength_nm.min(),
    wavelength_end_nm=1.0005 * wavelength_nm.max(),
    r_grid=100_000,
    device=device,
    dtype=dtype_name,
)
instrument = ObservedSpectrumOperator(
    native.wavelength_nm,
    wavelength_nm,
    resolving_power=28_000,
    device=device,
    dtype=torch_dtype,
)
instrument.set_parameters(
    radial_velocity_km_s=0.0,
    broadening_sigma_km_s=4.0,
)
```

`ObservedSpectrumOperator` applies residual velocity, Gaussian broadening, an LSF, and linear resampling on the selected device. Use either `resolving_power=` for a constant-resolution Gaussian LSF or `lsf_kernel=` for one sampled shift-invariant LSF. `lsf_kernel` must be a one-dimensional NumPy-compatible array with an odd number of finite, nonnegative, dimensionless response weights. Its samples are spaced by one native log-wavelength pixel, and its middle entry is the zero-offset sample. The values may have any positive sum because the operator normalizes them internally. For example:

```python
lsf_kernel = np.array([0.02, 0.12, 0.32, 0.44, 0.32, 0.12, 0.02])
instrument = ObservedSpectrumOperator(
    native.wavelength_nm,
    wavelength_nm,
    lsf_kernel=lsf_kernel,
    device=device,
    dtype=torch_dtype,
)
```

The sampled kernel is applied on the native grid before interpolation to `output_wavelength_nm`. It therefore describes offsets in native pixels, not wavelength units, velocity units, or observed-pixel indices. A two-dimensional array of per-pixel kernels is not accepted by this argument. The input wavelength must contain at least three positive points uniformly spaced in log wavelength. The output wavelength must be positive, strictly increasing, and contained in the input interval. Both are in nanometers.

A wavelength-dependent, fiber-specific, or detector-specific response can instead implement the same protocol by providing an `output_wavelength_nm` array and a `convolve_fluxes(total_flux, continuum_flux)` method. The method receives one-dimensional total- and continuum-flux tensors on the native grid, on the operator device and with its Torch dtype. It returns total flux, continuum flux, and their normalized ratio on `output_wavelength_nm`. Optional `name` and `last_seconds` attributes are retained in synthesis metadata. The APOGEE adapter is one such implementation.

Next wrap the reduced spectrum and define the model parameters. The fitter uses finite differences unless a Jacobian callback is supplied:

```python
from fitter import FitConfiguration, NormalizedSpectrum, fit_normalized_spectrum

observation = NormalizedSpectrum(
    wavelength=wavelength_nm,
    flux=flux,
    inverse_variance=inverse_variance,
    mask=good_pixel_mask,
)
configuration = FitConfiguration(
    names=("Teff", "logg", "M_H", "alpha_M", "microturbulence"),
    initial=np.array([4800.0, 2.5, -0.3, 0.2, 1.5]),
    lower=np.array([4000.0, 0.7, -2.5, -0.1, 0.5]),
    upper=np.array([6000.0, 5.0, 0.5, 0.5, 4.0]),
    derivative_steps=np.array([50.0, 0.05, 0.05, 0.025, 0.10]),
    trust_half_width=np.array([150.0, 0.20, 0.15, 0.10, 0.40]),
    maximum_iterations=8,
)

def model(parameters):
    teff, logg, metallicity, alpha, micro = parameters
    return synthesize_from_labels(
        effective_temperature=teff,
        log_surface_gravity=logg,
        metallicity=metallicity,
        alpha_enhancement=alpha,
        microturbulence_km_s=micro,
        wavelength_start_nm=native.wavelength_nm[0],
        wavelength_end_nm=native.wavelength_nm[-1],
        r_grid=100_000,
        device=device,
        dtype=dtype_name,
        spectral_operator=instrument,
    ).normalized_flux

coordinate = 2.0 * (wavelength_nm - wavelength_nm.min()) / np.ptp(wavelength_nm) - 1.0
continuum_basis = np.polynomial.legendre.legvander(coordinate, 2)
result = fit_normalized_spectrum(
    observation,
    configuration,
    model,
    continuum_basis=continuum_basis,
)
result.save("results/my_spectrum")
```

If the configuration contains `P` names and the spectrum contains `N` pixels, every configuration vector has shape `(P,)` in `names` order. The model callback receives a copy of that vector and returns finite normalized flux with shape `(N,)`. An optional Jacobian callback returns shape `(N, P)`, with rows in observed-pixel order and columns in parameter-name order.

The optional continuum basis has shape `(N, K)`. The fitter multiplies each column by the physical model and profiles its `K` coefficients by weighted linear least squares at every trial. It is not part of the nonlinear parameter vector. When an explicit Jacobian is supplied with a continuum basis, it must describe the continuum-profiled model. To fit CNO-sensitive stars, add `c_over_m`, `n_over_m`, and `o_over_m` to the callback and configuration. To fit individual abundances, pass `fe_over_h`, a sparse `x_over_h` mapping, and `initializer_family="direct_abundance"`; omitted elements inherit `[Fe/H]`.

For a quick fitter-only installation check, run:

```bash
python -m fitter.examples.fit_normalized --out results/toy_normalized_fit
```

This deterministic check uses analytic absorption profiles so it runs without synthesis data.

Rotation can be applied before an instrument response with the optional device-resident Gray-profile operator:

```python
import torch
import numpy as np
from fitter import RotationalBroadening

native_wavelength_nm = np.exp(np.linspace(np.log(500.0), np.log(501.0), 2048))
native_flux = torch.ones(2048, dtype=torch.float64)
rotation = RotationalBroadening(
    native_wavelength_nm,
    maximum_vsini_km_s=100.0,
    limb_darkening=0.6,
    device=native_flux.device,
    dtype=native_flux.dtype,
)
rotated_flux = rotation(native_flux, vsini_km_s=12.0)
```

Construction validates the uniform log-wavelength grid and caches its velocity geometry on CPU, CUDA, or Apple Metal. Repeated application has no NumPy or host-device round trip. Flux derivatives remain in the Torch computational graph. The scalar `vsini_km_s` interface is intended for finite-difference or other bounded optimization and is not differentiable with respect to `vsini_km_s` itself. The APOGEE reference fit does not enable this optional operator.

Pass `jacobian=` when an analytic or autodifferentiable Jacobian of the continuum-profiled observed-grid model is available; otherwise the fitter uses bounded one-sided differences.

## Converged physical-atmosphere refinement

The fast fit can be checked and refined with a callback that solves a physical atmosphere to convergence and synthesizes its observed-grid spectrum:

```python
from tempfile import TemporaryDirectory

from payne_zero_atmosphere import solve_structured_atmosphere
from payne_zero_synthesis import synthesize
from fitter import PhysicalAtmosphereConfiguration, refine_with_physical_atmosphere

def physical_model(parameters):
    teff, logg, metallicity, alpha, micro = parameters
    with TemporaryDirectory() as directory:
        atmosphere_path = solve_structured_atmosphere(
            effective_temperature=teff,
            log_surface_gravity=logg,
            metallicity=metallicity,
            alpha_enhancement=alpha,
            microturbulence_km_s=micro,
            out_dir=directory,
        )
        return synthesize(
            atmosphere_path,
            wavelength_start_nm=native.wavelength_nm[0],
            wavelength_end_nm=native.wavelength_nm[-1],
            resolution=100_000,
            device=device,
            dtype=dtype_name,
            spectral_operator=instrument,
        ).normalized_flux

physical_result = refine_with_physical_atmosphere(
    observation,
    configuration,
    result,
    model,
    physical_model,
    PhysicalAtmosphereConfiguration(
        maximum_discrepancy_rms=2.0e-3,
        maximum_objective_degradation=0.1,
        minimum_predicted_objective_improvement=1.0e-3,
        maximum_physical_evaluations=4,
        correction_derivative_steps=np.array([50.0, 0.05, 0.05, 0.025, 0.10]),
        correction_trust_half_width=np.array([75.0, 0.10, 0.08, 0.06, 0.25]),
    ),
    continuum_basis=continuum_basis,
)
physical_result.save("results/my_spectrum")
```

In a physical application, the second model callback runs `solve_structured_atmosphere` at the proposed labels and passes its output to `synthesize` with the same `instrument` object. The first two thresholds gate the profiled flux discrepancy and the increase in mean weighted squared residual. If either gate fails, a fresh local fast-model Jacobian proposes a bounded correction while holding the local physical-minus-fast discrepancy fixed. Another atmosphere is solved only when that proposal has the declared predicted gain, and it is accepted only when the converged physical model improves the objective. The evaluation limit is a user-selected safety cap, not a fixed requirement of the method. Both callbacks use the same continuum profiling, and the saved trace includes every physical check, correction proposal, outcome, resolved numerical control, and runtime without plotting. Omit the two correction-scale arrays to derive them from `FitConfiguration`.

`PhysicalAtmosphereResult` reports `physical_fit_stationary` and `fast_physical_gates_passed` separately. The first means that the local discrepancy-corrected physical fit reached the declared step or predicted-gain threshold. The second means that the accepted fast and converged-atmosphere spectra pass both agreement gates. `successful` is true when either valid stopping route succeeds. Timing fields do not overlap. `correction_seconds` contains only Jacobian assembly, continuum profiling, and linear-algebra overhead outside the separately recorded callbacks. Each correction also records its selected backtracking fraction and actual step norm in trust-scaled and parameter units.

## APOGEE interface

[`apogee/`](apogee/) packages a representative Apache Point Observatory Galactic Evolution Experiment (APOGEE) Data Release 14 all-slit mean line-spread function, residual radial-velocity and broadening operators, chip-wise continuum profiling, atomic-calibration projection, an initialized-atmosphere synthesis model, and the established fast optimizer. It accepts prepared normalized apStar arrays without downloading survey data or assuming a survey catalog. The convenience entry point does not automatically run the converged-atmosphere refinement described above. See its README for the Python array interface, instrument-kernel scope, and `payne-zero-fit-apogee` command.

This installed APOGEE interface exposes exactly two stellar-label families: the five standard labels and the eight-label family with independent carbon, nitrogen, and oxygen coordinates. The optional direct-[X/H] initializer is not exposed by `fit_apogee_spectrum` or `payne-zero-fit-apogee`.
