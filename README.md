# Payne Zero

Payne Zero calculates one-dimensional LTE stellar atmospheres and synthetic
spectra with the Kurucz ATLAS12 and SYNTHE physics. The implementation uses
compiled multicore CPU kernels for atmosphere iteration and PyTorch kernels on
NVIDIA or Apple GPUs for spectral synthesis. The converged atmosphere is
passed directly to synthesis as a structured NPZ file.

The repository also provides general interfaces for normalized-spectrum
fitting and differentiable line-list calibration.

## Components

| directory | purpose |
| --- | --- |
| [`payne_zero_atmosphere/`](payne_zero_atmosphere/README.md) | model-atmosphere initialization and converged physical solve |
| [`payne_zero_synthesis/`](payne_zero_synthesis/README.md) | CPU, CUDA, and Metal spectral synthesis |
| [`fitter/`](fitter/README.md) | weighted normalized-spectrum fitting |
| [`linelist_calibration/`](linelist_calibration/README.md) | differentiable calibration of atomic line parameters |
| [`source_data_files/`](source_data_files/README.md) | hash-verified runtime catalogs, tables, and atmosphere initializers |

## Installation

Python 3.11 or newer and Git LFS are required. Clone normally so Git LFS
downloads the runtime arrays, then run the installer:

```bash
git clone https://github.com/tingyuansen/payne-zero.git
cd payne-zero
./install.sh
```

The installer verifies the runtime files, installs the Python packages, and
builds persistent caches in
`.cache/payne-zero/`. A clean installation can spend 10--20 minutes compiling.
Later runs reuse these caches.

The optional direct-abundance initializer is installed only when requested:

```bash
PAYNE_ZERO_INCLUDE_DIRECT_XH=1 ./install.sh
```

## Atmosphere modes

All three modes predict only an initial depth structure. The physical solver
then iterates at the requested labels and abundances, and writes a result only
after convergence.

| mode | public coordinates | use |
| --- | --- | --- |
| five-label | `Teff`, `logg`, `[M/H]`, `[alpha/M]`, microturbulence | default ordinary-star atmosphere |
| eight-label | five-label set plus `[C/M]`, `[N/M]`, `[O/M]` | CNO-sensitive mixtures, including evolved giants |
| direct abundance | `Teff`, `logg`, microturbulence, `[Fe/H]`, and any individual `[X/H]` values | explicit-mixture initializer; requires the optional asset and a converged physical solve |

The five-label initializer is selected by default. Supplying any C, N, or O
coordinate selects the eight-label initializer. Direct abundance is selected
explicitly in either interface. Every element is available as an individual
`X_over_h` coordinate. Unspecified metals inherit `[Fe/H]`; internally the
complete mixture is re-expressed as `[Fe/H]` and 80 `[X/Fe]` coordinates.

The command-line and Python names map to the scientific coordinates as follows:

| coordinate | command line | Python keyword | mode |
| --- | --- | --- | --- |
| `Teff` | `--effective-temperature` | `effective_temperature` | all |
| `logg` | `--log-surface-gravity` | `log_surface_gravity` | all |
| `[M/H]` | `--metallicity` | `metallicity` | 5D and 8D |
| `[alpha/M]` | `--alpha-enhancement` | `alpha_enhancement` | 5D and 8D |
| microturbulence | `--microturbulence-km-s` | `microturbulence_km_s` | all |
| `[C/M]` | `--c-over-m` | `c_over_m` | 8D |
| `[N/M]` | `--n-over-m` | `n_over_m` | 8D |
| `[O/M]` | `--o-over-m` | `o_over_m` | 8D |
| individual `[X/H]` | `--x-over-h` such as `--mg-over-h` | `x_over_h` such as `mg_over_h` | direct abundance |

The common five- and eight-label support is approximately 4,000--10,500 K in
effective temperature, 0.7--5.3 in `logg`, -2.5--0.5 in `[M/H]`, -0.1--0.5 in
`[alpha/M]`, and 0.5--4.0 km s^-1 in microturbulence. The CNO coordinates span
about -0.5--0.5 dex relative to the base metal mixture. The direct-abundance
interface uses the same stellar range and accepts -0.5--0.5 in each `[X/Fe]`.
Exact contracts are documented in the atmosphere README.
The complete initializer training corpora are available as an optional
[v1.3 data bundle](source_data_files/atmosphere_emulator/TRAINING_CORPORA.md);
they are not downloaded for ordinary installation.

## Basic workflow

Solve an atmosphere:

```bash
python -m payne_zero_atmosphere \
  --effective-temperature 5777 \
  --log-surface-gravity 4.44 \
  --out runs/sun
```

This writes `runs/sun/payne_zero_structured_atmosphere.npz`. Synthesize a
spectrum from that product:

```bash
python -m payne_zero_synthesis.cli \
  runs/sun/payne_zero_structured_atmosphere.npz \
  --out runs/sun/spectrum.npz \
  --wl-start-nm 400 --wl-end-nm 900 --r-grid 20000
```

The synthesis device defaults to CUDA, then Metal, then CPU. `--r-grid` is the
sampling density of the intrinsic model grid, not the resolving power of an
instrument.

The corresponding Python interface is:

```python
from payne_zero_atmosphere import solve_structured_atmosphere
from payne_zero_synthesis import synthesize

atmosphere_path = solve_structured_atmosphere(
    effective_temperature=4800,
    log_surface_gravity=2.5,
    metallicity=-0.5,
    alpha_enhancement=0.3,
    c_over_m=0.1,
    n_over_m=0.2,
    o_over_m=0.1,
    out_dir="runs/giant",
)

spectrum = synthesize(
    atmosphere_path,
    wavelength_start_nm=1500,
    wavelength_end_nm=1700,
    resolution=300000,
    device="cuda",
)
```

Supplying the CNO coordinates in this example selects the eight-label
initializer. The atmosphere README gives complete CLI and Python examples for
the 81-element initializer.

## Performance

The following warm measurements exclude installation and first-use cache
construction. They are guides rather than hardware-independent guarantees.

| calculation | hardware and setup | measured wall time |
| --- | --- | ---: |
| one atmosphere iteration | 16 AMD EPYC CPU threads | 2--5 s across hot-dwarf, solar, red-giant, and K-dwarf controls |
| three atmosphere iterations | same CPU setup | 6--16 s |
| solar spectrum, 1500--1700 nm, `R_grid=300000` | one H100 | 1.4 s |
| solar spectrum, same setup | one A100 or V100 | 3.1 s |
| 300--1000 nm spectrum, `R_grid=300000` | one H100 | about 14--21 s across the four controls |
| APOGEE normalized-spectrum synthesis search | one H100 | median 43 s per star in the retained survey experiment |
| 101,124-parameter standard-star line calibration | one H100 | about 1--10 min, depending on the target and convergence path |

Atmosphere iteration is CPU-oriented because its ordered outer iteration does
not benefit from the available GPU implementation. Synthesis is parallel over
wavelength and line profiles, so CUDA is the preferred production path. Metal
is supported on Apple Silicon for local work, and both packages retain CPU
fallbacks.

Atmosphere kernels and prepared synthesis windows default to
`.cache/payne-zero/` in a source checkout. The molecular source parser uses
`~/.cache/payne-zero-synthesis/` unless `PAYNE_ZERO_SYNTHESIS_CACHE_DIR` is
set. Set that variable and `PAYNE_ZERO_NUMBA_CACHE_DIR` to relocate the
persistent caches.

## Fitting and line calibration

[`fitter/`](fitter/README.md) fits normalized spectra with inverse-variance
weights, bounds, optional Jacobians, profiled linear continua, trust regions,
and complete parameter and spectrum traces. A separate callback can replace
the initialized atmosphere with a converged physical atmosphere and refine the
candidate when needed. Instrument-specific operations remain in adapters such
as `fitter/apogee/`.

[`linelist_calibration/`](linelist_calibration/README.md) optimizes continuous
atomic parameters through a user-supplied differentiable synthesis callback.
It supports multiple spectra and bounded joint optimization. The unchanged
Kurucz line catalog remains the default; optional Sun--Arcturus overlays are
provided separately with their provenance.

## Products and conventions

`payne_zero_structured_atmosphere.npz` is the interchange product between the
two physical stages. Its machine-readable schema is
[`payne_zero_synthesis/atmosphere_schema.json`](payne_zero_synthesis/atmosphere_schema.json).
The spectrum product contains wavelength, total and continuum surface
`F_lambda` per nanometer, normalized flux, and runtime metadata.

The abundance reference is AGSS09. `[M/H]` changes all metals, `[alpha/M]`
adds a common offset to O, Ne, Mg, Si, S, Ca, and Ti, and explicit CNO values
replace the corresponding offsets in the eight-label mode. Detailed abundance
and file-format conventions are in the atmosphere and synthesis READMEs.

## License

Payne Zero-authored code is released under the
[BSD 3-Clause License](LICENSE).

## Citation

If Payne Zero contributes to a publication, please cite Ting & Kim,
*The Payne Zero Project I: Stellar Spectra from Physical Models in Seconds*,
submitted to the *Open Journal of Astrophysics*. The arXiv identifier will be
added when it is assigned.

```bibtex
@unpublished{TingKim2026PayneZero,
  author = {Ting, Yuan-Sen and Kim, Elliot M.},
  title = {The Payne Zero Project I: Stellar Spectra from Physical Models in Seconds},
  year = {2026},
  note = {Submitted to the Open Journal of Astrophysics},
  url = {https://github.com/tingyuansen/payne-zero}
}
```
