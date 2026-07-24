# Payne Zero Synthesis

`payne_zero_synthesis` calculates an emergent spectrum either directly from stellar labels or from an existing structured atmosphere. The label interface uses a released atmosphere initializer and population bridge in memory. The archive interface accepts the structured NumPy `.npz` product written by `payne_zero_atmosphere`; historical text atmosphere decks are not synthesis inputs.

The implementation uses PyTorch and selects NVIDIA CUDA, then Apple Metal, then a central processing unit (CPU) when no device is specified. CUDA and Metal accelerate broad wavelength windows, while CPU execution remains available for deterministic verification or systems without a suitable graphics processor. If the numerical data type is also omitted, the Python interface uses 32-bit floating point on Metal and 64-bit floating point on CUDA or CPU.

## Choose a workflow

| workflow | entry point | choose it when |
| --- | --- | --- |
| fast synthesis from labels | `payne-zero-synthesis` without an atmosphere path, or `synthesize_from_labels(...)` | exploring labels or evaluating many nearby spectra |
| converged atmosphere and synthesis | `payne-zero-atmosphere`, then `payne-zero-synthesis ATMOSPHERE` or `synthesize(...)` | retaining the atmosphere or requiring a final spectrum from a physically converged structure |

Both workflows use the same synthesis kernels. The fast path predicts and population-bridges an initialized atmosphere but does not run the iterative atmosphere solver.

## Command-line interface

### Fast synthesis from labels

Five-label synthesis is the default:

```bash
payne-zero-synthesis \
  --effective-temperature 5777 --log-surface-gravity 4.44 \
  --metallicity 0.0 --alpha-enhancement 0.0 \
  --microturbulence-km-s 1.0 \
  --wl-start-nm 500 --wl-end-nm 510 --r-grid 20000 \
  --out sun_spectrum.npz
```

Add `--c-over-m`, `--n-over-m`, and `--o-over-m` for the eight-label CNO initializer:

```bash
payne-zero-synthesis \
  --effective-temperature 4600 --log-surface-gravity 2.2 \
  --metallicity -0.4 --alpha-enhancement 0.2 \
  --c-over-m -0.25 --n-over-m 0.35 --o-over-m 0.15 \
  --wl-start-nm 1500 --wl-end-nm 1700 --r-grid 300000 \
  --out cno_giant_spectrum.npz
```

To control individual abundances, install the optional direct-abundance asset and give an iron baseline plus any `[X/H]` values:

```bash
payne-zero-synthesis \
  --effective-temperature 4600 --log-surface-gravity 2.2 \
  --initializer direct-abundance \
  --fe-over-h -0.4 --c-over-h -0.65 --n-over-h -0.05 --mg-over-h -0.15 \
  --wl-start-nm 1500 --wl-end-nm 1700 --r-grid 300000 \
  --out direct_abundance_spectrum.npz
```

Every element represented by the direct initializer has an `--x-over-h` flag such as `--mg-over-h`. The repeatable generic spelling `--abundance Mg:-0.15` and a JSON `--abundance-file` are also accepted. Unspecified elements inherit `[Fe/H]`.

### Synthesis from a converged atmosphere

Pass the structured product from `payne-zero-atmosphere` as the positional argument:

```bash
payne-zero-synthesis atmosphere.npz \
  --wl-start-nm 400 --wl-end-nm 900 --r-grid 20000 \
  --out spectrum.npz
```

| argument | default | meaning |
| --- | --- | --- |
| `atmosphere` | omitted | optional structured-atmosphere archive; omit for label synthesis |
| `--effective-temperature`, `--log-surface-gravity` | required in label mode | basic stellar parameters |
| `--metallicity`, `--alpha-enhancement` | `0`, `0` | `[M/H]` and `[alpha/M]` in five- or eight-label mode |
| `--c-over-m`, `--n-over-m`, `--o-over-m` | omitted | independent CNO coordinates; select eight-label mode |
| `--initializer direct-abundance`, `--x-over-h` | omitted | individual `[X/H]` mode |
| `--save-initialized-atmosphere` | omitted | retain the initialized atmosphere with explicit role and provenance metadata |
| `--wl-start-nm`, `--wl-end-nm` | `400`, `900` | wavelength bounds [nm] |
| `--r-grid`, `--resolution` | `20000` | logarithmic intrinsic-grid density, `R_grid` |
| `--device` | best available | NVIDIA CUDA, Apple Metal, or CPU |
| `--dtype` | device-aware | `float32` on Metal; `float64` on CUDA/CPU |
| `--no-molecular-lines` | off | omit molecular line opacity |
| `--validate-only` | off | validate a positional atmosphere archive and exit |

Every spectrum NumPy archive contains:

- `wavelength_nm`;
- `flux_total`, the total surface `F_lambda` spectral flux density per nanometer;
- `flux_continuum`, the continuum surface `F_lambda` spectral flux density per nanometer;
- `normalized_flux = flux_total / flux_continuum`;
- `seconds`, the complete wall time represented by that product.

For archive-based `synthesize(...)`, `seconds` is the synthesis wall time. Label-driven products also contain the initializer family, stage timings, and metadata that identifies the atmosphere as initialized; their `seconds` value is the sum of initialization, population bridging, and synthesis. Use `synthesis_seconds` when only the spectral calculation is needed.

Transfer is evaluated internally as Eddington `H_nu`. The public interface applies `F = 4 pi H` and the exact frequency-to-wavelength Jacobian needed to return both surface-flux arrays per nanometer.

## Python interface

### Fast synthesis from labels

Use `synthesize_from_labels` for repeated label-driven calculations:

```python
from payne_zero_synthesis import synthesize_from_labels

five_label = synthesize_from_labels(
    effective_temperature=4750,
    log_surface_gravity=2.5,
    metallicity=-0.3,
    alpha_enhancement=0.15,
    microturbulence_km_s=1.5,
    wavelength_start_nm=500,
    wavelength_end_nm=510,
    r_grid=20_000,
    device="auto",
    dtype="auto",
)

eight_label = synthesize_from_labels(
    effective_temperature=4750,
    log_surface_gravity=2.5,
    metallicity=-0.3,
    alpha_enhancement=0.15,
    c_over_m=-0.2,
    n_over_m=0.3,
    o_over_m=0.1,
    wavelength_start_nm=500,
    wavelength_end_nm=510,
    r_grid=20_000,
    device="auto",
)

direct = synthesize_from_labels(
    effective_temperature=4750,
    log_surface_gravity=2.5,
    fe_over_h=-0.3,
    x_over_h={"C": -0.5, "N": 0.0, "Mg": -0.1},
    initializer_family="direct_abundance",
    wavelength_start_nm=500,
    wavelength_end_nm=510,
    r_grid=20_000,
    device="auto",
)
```

The result contains the spectrum, initializer family, input labels, stage timings, and the in-memory initialized atmosphere. Pass `result.initialized_atmosphere` to `synthesize` to reuse that population state over another wavelength interval. Saving it with `result.initialized_atmosphere.save_npz(...)` preserves its initializer role, family, and provenance alongside the physical arrays. `load_atmosphere_product_metadata(...)` reads this metadata when a workflow needs to distinguish the saved initializer from a converged physical atmosphere.

### Solve a converged atmosphere and synthesize

Use the atmosphere solver first when the retained spectrum must use a physically converged structure:

```python
from payne_zero_atmosphere import solve_structured_atmosphere
from payne_zero_synthesis import synthesize

atmosphere_path = solve_structured_atmosphere(
    effective_temperature=4800,
    log_surface_gravity=2.5,
    metallicity=-0.5,
    alpha_enhancement=0.3,
    microturbulence_km_s=1.8,
    c_over_m=0.1,
    n_over_m=0.2,
    o_over_m=0.1,
    out_dir="runs/giant",
)
spectrum = synthesize(
    atmosphere_path,
    wavelength_start_nm=1500,
    wavelength_end_nm=1700,
    resolution=300_000,
    device="auto",
    dtype="auto",
)
```

In `synthesize_from_labels`, the older Python keyword `resolution` is an alias for `r_grid = lambda / Delta lambda` between adjacent intrinsic model samples. The archive-based `synthesize` function retains `resolution` as its keyword for the same intrinsic sampling density. Instrumental resolving power is applied separately through a spectral operator. [`fitter.ObservedSpectrumOperator`](../fitter/README.md) supplies constant-resolution or sampled-kernel projection to arbitrary observed pixels, and the APOGEE adapter supplies a measured wavelength-dependent LSF.

`build_structured_atmosphere` and `save_structured_atmosphere` are also public for callers that already hold physical atmosphere columns in memory.

## Atmosphere Contract

[`atmosphere_schema.json`](atmosphere_schema.json) is the machine-readable schema. Version 4 distinguishes two quantities that must not be interchanged:

- `ion_stage_populations`: actual ion-stage number densities [cm^-3];
- `partition_normalized_populations`: the same populations divided by their partition functions [cm^-3 per partition function].

The former controls charge-weighted free-free opacity. The latter controls bound-state opacity. Dedicated hydrogen, helium, metal, and molecular columns use equally explicit names. New products contain canonical names only. The loader accepts schema versions 1 through 3 as a read-only compatibility boundary and reconstructs the actual ion-stage population cube only for pre-version-3 files.

## Performance

Synthesis is parallel over wavelength. At `R_grid = 300,000`, a warm 300–1000 nm spectrum takes about 14–21 s on an H100 across the retained stellar controls. A solar 1500–1700 nm spectrum takes 1.4 s on H100 and 3.1 s on A100 or V100. The public interface evaluates one atmosphere per call.

Window-invariant line and transfer data are cached in process. Derived caches may be deleted without changing the physics.

Prepare all persistent caches for a wavelength window before timing:

```bash
python -m payne_zero_synthesis.prewarm \
  --wavelength-start-nm 400 --wavelength-end-nm 900 --r-grid 20000
```

Source checkouts store prepared windows under `.cache/payne-zero/synthesis/`. The molecular source parser uses `~/.cache/payne-zero-synthesis/` unless `PAYNE_ZERO_SYNTHESIS_CACHE_DIR` is set. Prewarm builds the window-specific atomic and molecular products, which are independent of stellar labels. Long-lived workers also retain the wavelength grid, catalog mappings, profile tables, and device tensors in memory so nearby optimizer evaluations do not repeat setup.

| environment variable | effect |
| --- | --- |
| `PAYNE_ZERO_DATA_ROOT` | relocate `source_data_files/` |
| `PAYNE_ZERO_SOURCE_CATALOG_ROOT` | override the shared full source-catalog tree |
| `PAYNE_ZERO_SYNTHESIS_SOURCE_CATALOG_ROOT` | override the source-catalog tree |
| `PAYNE_ZERO_SYNTHESIS_DISABLE_INVARIANT_CACHE=1` | disable the in-process invariant cache |
| `PAYNE_ZERO_SYNTHESIS_CACHE_DIR` | relocate derived synthesis caches |

Device and numerical data type are function arguments, not environment settings.

## Modules

| module | responsibility |
| --- | --- |
| `api.py` | public `Spectrum`, synthesis, build, and save functions |
| `atmosphere.py` | canonical NumPy-archive loading, validation, and compatibility upgrade |
| `pipeline.py`, `synthesis.py` | synthesis orchestration |
| `equation_of_state.py` | partition functions, ionization, and population state |
| `continuum.py` | continuum absorption and scattering |
| `atomic_lines.py`, `molecular_lines.py` | line-catalog preparation |
| `hydrogen_lines.py`, `line_opacity.py` | line-opacity kernels |
| `radiative_transfer.py` | emergent total and continuum flux |
| `constants.py`, `device.py`, `paths.py` | constants, execution policy, and data paths |
