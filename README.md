# Payne Zero

Payne Zero calculates one-dimensional local thermodynamic equilibrium (LTE) stellar atmospheres and synthetic spectra. It is a modern reimplementation of the physical calculation rather than a wrapper around the historical programs. Atmosphere iteration uses compiled kernels on multicore central processing units (CPUs), while spectral synthesis uses PyTorch kernels on NVIDIA or Apple graphics processing units (GPUs). The converged atmosphere passes directly to synthesis as a structured NumPy `.npz` archive.

The repository also provides general interfaces for normalized-spectrum fitting and differentiable line-list calibration.

## Components

| directory | purpose |
| --- | --- |
| [`payne_zero_atmosphere/`](payne_zero_atmosphere/README.md) | model-atmosphere initialization and converged physical solve |
| [`payne_zero_synthesis/`](payne_zero_synthesis/README.md) | CPU, CUDA, and Metal spectral synthesis |
| [`fitter/`](fitter/README.md) | weighted normalized-spectrum fitting |
| [`linelist_calibration/`](linelist_calibration/README.md) | differentiable calibration of atomic line parameters |
| [`source_data_files/`](source_data_files/README.md) | hash-verified runtime catalogs, tables, and atmosphere initializers |
| [`payne_zero_tutorial.ipynb`](payne_zero_tutorial.ipynb) | step-by-step synthesis, atmosphere, calibration, mock-fitting, and APOGEE tutorial |

## Installation

Python 3.11 or newer and Git Large File Storage (Git LFS) are required. Clone normally so Git LFS downloads the runtime arrays, then run the installer:

```bash
git clone https://github.com/tingyuansen/payne-zero.git
cd payne-zero
./install.sh
```

The installer verifies the runtime files, installs the Python packages, and builds persistent caches in `.cache/payne-zero/`. A clean installation can spend 10–20 minutes compiling. Later runs reuse these caches.

The optional direct-abundance initializer is installed only when requested:

```bash
PAYNE_ZERO_INCLUDE_DIRECT_ABUNDANCE=1 ./install.sh
```

Install the plotting and Jupyter dependencies before running the tutorial:

```bash
python -m pip install -e ".[tutorial]"
jupyter lab payne_zero_tutorial.ipynb
```

## Choose a workflow

Payne Zero exposes two paths that use the same synthesis calculation:

1. **Fast synthesis from labels.** `payne-zero-synthesis` or `synthesize_from_labels(...)` predicts an initialized atmosphere and immediately synthesizes a spectrum. Choose this path for exploration, repeated optimizer evaluations, and other work that needs many nearby spectra.
2. **Converged atmosphere followed by synthesis.** `payne-zero-atmosphere` iterates the physical atmosphere at the requested mixture, then `payne-zero-synthesis` or `synthesize(...)` calculates its spectrum. Choose this path when the atmosphere itself is a retained result or when the final spectrum must use a physically converged atmosphere.

The initializer is a starting model in both paths. The first path stops after prediction and population reconstruction. The second continues through the iterative atmosphere solve.

## Atmosphere modes

The three initializer families turn stellar labels into a complete depth structure. That prediction can be used immediately for fast synthesis and repeated fitting. When a physically converged atmosphere is required, the atmosphere solver uses the same prediction as its starting state and iterates at the requested mixture. Here `Teff` denotes effective temperature, `logg` denotes the base-10 logarithm of surface gravity, and CNO denotes carbon, nitrogen, and oxygen.

| mode | public coordinates | use |
| --- | --- | --- |
| five-label | `Teff`, `logg`, `[M/H]`, `[alpha/M]`, microturbulence | default ordinary-star atmosphere |
| eight-label | five-label set plus `[C/M]`, `[N/M]`, `[O/M]` | carbon-, nitrogen-, and oxygen-sensitive mixtures, including evolved giants |
| direct abundance | `Teff`, `logg`, microturbulence, `[Fe/H]`, and any individual `[X/H]` values | explicit abundance mixtures; requires the optional initializer asset |

The five-label initializer is selected by default. Supplying any C, N, or O coordinate selects the eight-label initializer. Direct abundance is selected explicitly in either interface. Every supported element is available as an individual `X_over_h` coordinate. Unspecified metals inherit `[Fe/H]`; internally the complete mixture is re-expressed as `[Fe/H]` and 80 `[X/Fe]` coordinates.

The command-line and Python names map to the scientific coordinates as follows:

| coordinate | command line | Python keyword | mode |
| --- | --- | --- | --- |
| `Teff` | `--effective-temperature` | `effective_temperature` | all |
| `logg` | `--log-surface-gravity` | `log_surface_gravity` | all |
| `[M/H]` | `--metallicity` | `metallicity` | five-label and eight-label |
| `[alpha/M]` | `--alpha-enhancement` | `alpha_enhancement` | five-label and eight-label |
| microturbulence | `--microturbulence-km-s` | `microturbulence_km_s` | all |
| `[C/M]` | `--c-over-m` | `c_over_m` | eight-label |
| `[N/M]` | `--n-over-m` | `n_over_m` | eight-label |
| `[O/M]` | `--o-over-m` | `o_over_m` | eight-label |
| `[Fe/H]` | `--fe-over-h` | `fe_over_h` | direct abundance |
| individual `[X/H]` | element flags such as `--mg-over-h` | `x_over_h` mapping such as `{"Mg": -0.2}` | direct abundance |

The common five- and eight-label support is approximately 4,000–10,500 K in effective temperature, 0.7–5.3 in `logg`, −2.5–0.5 in `[M/H]`, −0.1–0.5 in `[alpha/M]`, and 0.5–4.0 km s⁻¹ in microturbulence. The CNO coordinates span about −0.5–0.5 dex relative to the base metal mixture. The direct-abundance interface uses the same stellar range and accepts −0.5–0.5 in each `[X/Fe]`. Exact contracts are documented in the atmosphere README. The complete initializer training corpora are available as an optional [v1.3 data bundle](source_data_files/atmosphere_emulator/TRAINING_CORPORA.md); they are not downloaded for ordinary installation.

## 1. Fast synthesis from stellar labels

The common entry point needs stellar labels, a wavelength interval, and the sampling density of the intrinsic spectrum. It predicts the atmosphere, builds its chemical populations, and synthesizes without an intermediate file or iterative atmosphere solve:

```bash
payne-zero-synthesis \
  --effective-temperature 5777 \
  --log-surface-gravity 4.44 \
  --metallicity 0.0 \
  --alpha-enhancement 0.0 \
  --microturbulence-km-s 1.0 \
  --wl-start-nm 500 --wl-end-nm 510 --r-grid 20000 \
  --out runs/sun_fast_spectrum.npz
```

Supplying independent CNO coordinates selects the eight-label initializer:

```bash
payne-zero-synthesis \
  --effective-temperature 4600 --log-surface-gravity 2.2 \
  --metallicity -0.4 --alpha-enhancement 0.2 \
  --c-over-m -0.25 --n-over-m 0.35 --o-over-m 0.15 \
  --wl-start-nm 1500 --wl-end-nm 1700 --r-grid 300000 \
  --out runs/cno_giant_fast_spectrum.npz
```

The optional direct-abundance initializer accepts each element on the usual `[X/H]` scale. Unspecified elements inherit `[Fe/H]`:

```bash
payne-zero-synthesis \
  --effective-temperature 4600 --log-surface-gravity 2.2 \
  --initializer direct-abundance \
  --fe-over-h -0.4 --c-over-h -0.65 --n-over-h -0.05 --mg-over-h -0.15 \
  --wl-start-nm 1500 --wl-end-nm 1700 --r-grid 300000 \
  --out runs/direct_abundance_fast_spectrum.npz
```

CUDA is selected first, then Apple Metal, then CPU. `--r-grid` controls adjacent samples in the intrinsic logarithmic wavelength grid; it is not the resolving power of an instrument.

The same three cases use one Python function:

```python
from payne_zero_synthesis import synthesize_from_labels

five_label = synthesize_from_labels(
    effective_temperature=5777,
    log_surface_gravity=4.44,
    metallicity=0.0,
    alpha_enhancement=0.0,
    microturbulence_km_s=1.0,
    wavelength_start_nm=500,
    wavelength_end_nm=510,
    r_grid=20_000,
    device="auto",
)

eight_label = synthesize_from_labels(
    effective_temperature=4600,
    log_surface_gravity=2.2,
    metallicity=-0.4,
    alpha_enhancement=0.2,
    c_over_m=-0.25,
    n_over_m=0.35,
    o_over_m=0.15,
    wavelength_start_nm=1500,
    wavelength_end_nm=1700,
    r_grid=300_000,
    device="auto",
)

direct = synthesize_from_labels(
    effective_temperature=4600,
    log_surface_gravity=2.2,
    fe_over_h=-0.4,
    x_over_h={"C": -0.65, "N": -0.05, "Mg": -0.15},
    initializer_family="direct_abundance",
    wavelength_start_nm=1500,
    wavelength_end_nm=1700,
    r_grid=300_000,
    device="auto",
)
```

These spectra use initialized atmospheres. Use them for repeated forward evaluations. Use the converged workflow below for a retained physical atmosphere or a final spectrum that requires atmospheric consistency.

## 2. Solve a converged atmosphere and synthesize

For a converged atmosphere at the requested mixture, run the physical solver and synthesize its structured product:

```bash
payne-zero-atmosphere \
  --effective-temperature 4600 --log-surface-gravity 2.2 \
  --metallicity -0.4 --alpha-enhancement 0.2 \
  --c-over-m -0.25 --n-over-m 0.35 --o-over-m 0.15 \
  --out runs/cno_giant

payne-zero-synthesis \
  runs/cno_giant/payne_zero_structured_atmosphere.npz \
  --wl-start-nm 1500 --wl-end-nm 1700 --r-grid 300000 \
  --out runs/cno_giant/converged_spectrum.npz
```

The first command performs the iterative physical calculation. The second accepts the resulting archive through the same synthesis interface used for any solver that writes the documented structured schema. The [tutorial](payne_zero_tutorial.ipynb) develops both workflows before applying the forward model to fitting and line calibration.

## Performance

The following warm measurements exclude installation and first-use cache construction. They are guides rather than hardware-independent guarantees.

| calculation | hardware and setup | measured wall time |
| --- | --- | ---: |
| one atmosphere iteration | 16 AMD EPYC CPU threads | 2–5 s across hot-dwarf, solar, red-giant, and K-dwarf controls |
| three atmosphere iterations | same CPU setup | 6–16 s |
| solar spectrum, 1500–1700 nm, `R_grid=300000` | one H100 | 1.4 s |
| solar spectrum, same setup | one A100 or V100 | 3.1 s |
| 300–1000 nm spectrum, `R_grid=300000` | one H100 | about 14–21 s across the four controls |
| APOGEE normalized-spectrum synthesis search | one H100 | median 43 s per star in the retained survey experiment |
| 101,124-parameter standard-star line calibration | one H100 | about 1–10 min, depending on the target and convergence path |

Atmosphere iteration is CPU-oriented because its ordered outer iteration does not benefit from the available GPU implementation. Synthesis is parallel over wavelength and line profiles, so CUDA is the preferred production path. Metal is supported on Apple Silicon for local work, and both packages retain CPU fallbacks.

Atmosphere kernels and prepared synthesis windows default to `.cache/payne-zero/` in a source checkout. The molecular source parser uses `~/.cache/payne-zero-synthesis/` unless `PAYNE_ZERO_SYNTHESIS_CACHE_DIR` is set. Set that variable and `PAYNE_ZERO_NUMBA_CACHE_DIR` to relocate the persistent caches.

## Fitting and line calibration

[`fitter/`](fitter/README.md) fits normalized spectra with inverse-variance weights, bounds, optional Jacobians, profiled linear continua, trust regions, and complete parameter and spectrum traces. `ObservedSpectrumOperator` maps a native Payne Zero spectrum to any increasing observed wavelength array with velocity, broadening, a constant-resolution or sampled LSF, and resampling. A custom wavelength-dependent response implements the same spectral-operator protocol; [`fitter/apogee/`](fitter/apogee/README.md) is the included example. A separate callback can replace the initialized atmosphere with a converged physical atmosphere and refine the candidate when needed.

[`linelist_calibration/`](linelist_calibration/README.md) optimizes oscillator strengths and damping parameters through differentiable physical synthesis. Its runnable example reads a real solar FTS excerpt, resolves the requested transition in the active catalog, fits it, and compares the baseline and calibrated profiles. The same interface accepts another standard star, observed wavelength grid, broadening kernel, or joint collection of stars. The unchanged source catalog remains the default; optional Sun–Arcturus correction overlays are provided separately.

## Products and conventions

`payne_zero_structured_atmosphere.npz` is the NumPy archive exchanged between the two physical stages. Its machine-readable schema is [`payne_zero_synthesis/atmosphere_schema.json`](payne_zero_synthesis/atmosphere_schema.json). An initialized atmosphere saved for reuse carries explicit role and provenance metadata and is not a substitute for the solver's converged product. The spectrum product contains wavelength, total and continuum surface `F_lambda` per nanometer, normalized flux, and runtime metadata.

The reference solar mixture is the photospheric composition of Asplund, Grevesse, Sauval, and Scott (2009), commonly abbreviated AGSS09. `[M/H]` changes all metals, `[alpha/M]` adds a common offset to O, Ne, Mg, Si, S, Ca, and Ti, and explicit carbon, nitrogen, and oxygen values replace the corresponding offsets in the eight-label mode. Detailed abundance and file-format conventions are in the atmosphere and synthesis READMEs.

## License

Payne Zero-authored code is released under the [BSD 3-Clause License](LICENSE).

## Citation

If Payne Zero contributes to a publication, please cite Ting & Kim, *The Payne Zero Project I: Stellar Spectra from Physical Models in Seconds*. Replace the placeholder arXiv identifier below when the record is available.

```bibtex
@article{TingKim2026PayneZero,
  author = {Ting, Yuan-Sen and Kim, Elliot M.},
  title = {The Payne Zero Project I: Stellar Spectra from Physical Models in Seconds},
  journal = {arXiv e-prints},
  year = {2026},
  eprint = {xxxx.xxxxx},
  archivePrefix = {arXiv},
  primaryClass = {astro-ph.IM}
}
```
