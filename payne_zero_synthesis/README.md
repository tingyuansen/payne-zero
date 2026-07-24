# Payne Zero Synthesis

`payne_zero_synthesis` turns a solved model atmosphere into an emergent spectrum. The public input is the structured-atmosphere NumPy `.npz` archive produced by `payne_zero_atmosphere`; text atmosphere decks are not a synthesis input.

The implementation uses PyTorch and selects NVIDIA CUDA, then Apple Metal, then a central processing unit (CPU) when no device is specified. CUDA and Metal accelerate broad wavelength windows, while CPU execution remains available for deterministic verification or systems without a suitable graphics processor. If the numerical data type is also omitted, the Python interface uses 32-bit floating point on Metal and 64-bit floating point on CUDA or CPU.

## Command-line interface

```bash
python -m payne_zero_synthesis.cli atmosphere.npz \
  --out spectrum.npz \
  --wl-start-nm 400 --wl-end-nm 900 \
  --r-grid 20000
```

| argument | default | meaning |
| --- | --- | --- |
| `atmosphere` | required | structured-atmosphere NumPy archive |
| `--out` | required unless `--validate-only` | output spectrum NumPy archive |
| `--validate-only` | off | validate the atmosphere and exit |
| `--wl-start-nm`, `--wl-end-nm` | `400`, `900` | wavelength bounds [nm] |
| `--r-grid`, `--resolution` | `20000` | logarithmic wavelength-grid density, `R_grid` |
| `--device` | best available | NVIDIA CUDA (`cuda`), Apple Metal Performance Shaders (`mps`), or CPU (`cpu`) |
| `--dtype` | device-aware | `float32` on Metal; `float64` on CUDA/CPU |
| `--no-molecular-lines` | off | omit molecular line opacity |

The spectrum NumPy archive contains:

- `wavelength_nm`;
- `flux_total`, the total surface `F_lambda` spectral flux density per nanometer;
- `flux_continuum`, the continuum surface `F_lambda` spectral flux density per nanometer;
- `normalized_flux = flux_total / flux_continuum`;
- `seconds`, the synthesis wall time.

Transfer is evaluated internally as Eddington `H_nu`. The public interface applies `F = 4 pi H` and the exact frequency-to-wavelength Jacobian needed to return both surface-flux arrays per nanometer.

## Python interface

```python
from payne_zero_synthesis import synthesize

spectrum = synthesize(
    "payne_zero_structured_atmosphere.npz",
    wavelength_start_nm=400.0,
    wavelength_end_nm=900.0,
    resolution=20_000.0,
    molecular_lines=True,
    device="auto",
    dtype="auto",
)
spectrum.save_npz("sun_spectrum.npz")
```

The historical Python argument `resolution` denotes `R_grid = lambda / Delta lambda` for adjacent model samples. Instrumental resolving power is applied separately by an instrument model such as the public Apache Point Observatory Galactic Evolution Experiment (APOGEE) line-spread-function operator.

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
