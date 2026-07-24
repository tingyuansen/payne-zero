# Payne Zero Atmosphere

`payne_zero_atmosphere` predicts a complete starting structure and iterates it to a converged one-dimensional local thermodynamic equilibrium (LTE) model atmosphere. The solver runs on multicore central processing units (CPUs) with compiled Numba kernels. Independent stars are the natural unit of parallel work.

The public product is a structured-atmosphere NumPy `.npz` archive consumed directly by `payne_zero_synthesis`. Text atmosphere decks are retained only as a fixed-column compatibility boundary for the independent pykurucz reference and for the in-memory quantization used by the certified solver path.

## Command-line interface

```bash
python -m payne_zero_atmosphere \
  --effective-temperature 5777 \
  --log-surface-gravity 4.44 \
  --out runs/sun
```

This writes `runs/sun/payne_zero_structured_atmosphere.npz` only after the exact solver satisfies its convergence criterion.

An eight-coordinate giant request is explicit:

```bash
python -m payne_zero_atmosphere \
  --effective-temperature 4800 --log-surface-gravity 2.5 \
  --metallicity -0.5 --alpha-enhancement 0.3 \
  --microturbulence-km-s 1.8 \
  --c-over-m 0.1 --n-over-m 0.2 --o-over-m 0.1 \
  --out runs/giant
```

| argument | default | physical meaning |
| --- | --- | --- |
| `--effective-temperature` | required | effective temperature [K] |
| `--log-surface-gravity` | required | log10 surface gravity [cgs] |
| `--metallicity` | `0` | [M/H] |
| `--alpha-enhancement` | `0` | [alpha/M] |
| `--microturbulence-km-s` | `2` | microturbulent velocity [km s⁻¹] |
| `--c-over-m` | unset | [C/M] |
| `--n-over-m` | unset | [N/M] |
| `--o-over-m` | unset | [O/M] |
| `--abundance N:+x` | unset | repeatable exact-solver `[X/H]` override by atomic number or symbol, for example `--abundance Fe:+0.3` |
| `--initializer` | `auto` | `auto` selects the five-label or eight-label initializer; `direct-abundance` selects the 81-element initializer |
| `--abundance-file` | unset | JSON mapping of element symbols or atomic numbers to `[X/H]`; unspecified elements inherit Fe |
| `--iterations-per-trial` | `15` | exact iterations allowed per initializer |
| `--max-trials` | `2` | deterministic nearby initializer trials |
| `--initializer-seed` | `20260713` | deterministic trial seed |
| `--initializer-jitter-scale` | `0.01` | nearby offset as a fraction of label span |

The production solve enables molecules, convection, and the full source-line catalogs. It requires at least three iterations, one consecutive converged iteration, and a maximum deep-layer relative temperature change of `5e-4`. The requested stellar labels never change during a nearby-initializer retry.

## Initializers

Every initializer predicts the same six atmosphere fields. It changes the starting point, not the requested physical model or the convergence test. CNO denotes carbon, nitrogen, and oxygen.

| mode | public coordinates | selection |
| --- | --- | --- |
| five-label | `Teff`, `logg`, `[M/H]`, `[alpha/M]`, microturbulence | default |
| eight-label | five-label set plus `[C/M]`, `[N/M]`, `[O/M]` | automatic when any carbon, nitrogen, or oxygen coordinate is supplied |
| direct abundance | `Teff`, `logg`, microturbulence, `[Fe/H]`, and any individual `[X/H]` values | explicit command-line or Python selection |

The five- and eight-label initializers are installed by default. An out-of-support query warns and clips only the initializer input to its support boundary. The physical solve still uses the exact requested labels and writes the structured product only after convergence.

### Direct-abundance initializer

The optional direct-abundance initializer exposes every supported element as an individual coordinate, such as `fe_over_h`, `mg_over_h`, or `c_over_h`. When used through this atmosphere interface, its decoded structure is followed by the physical solve before synthesis. The synthesis interface can use the same initialized structure for a fast optimizer model and marks that product as unconverged.

From the repository root, install its optional checkpoint with

```bash
PAYNE_ZERO_INCLUDE_DIRECT_ABUNDANCE=1 ./install.sh
```

Supply `[Fe/H]` and any elements that differ from it. Unspecified metals inherit `[Fe/H]`, producing the complete mixture used by the atmosphere and synthesis calculations. The support is 4,000–10,500 K, `logg` 0.7–5.3, microturbulence 0.5–4.0 km s⁻¹, `[Fe/H]` −2.5–0.5, and each `[X/Fe]` −0.5–0.5. The public input remains 81 `[X/H]` values; `[Fe/H]` and `[X/Fe]` are only the network's internal coordinates. Inputs are quantized to the solver's 0.01 dex abundance precision.

The 81 available elements are Li–Mo, Ru–Nd, Sm–Bi, Th, and U.

The command-line interface provides one option per element:

```bash
python -m payne_zero_atmosphere \
  --effective-temperature 4800 --log-surface-gravity 2.5 \
  --microturbulence-km-s 1.8 \
  --fe-over-h -0.2 --c-over-h 0.2 --mg-over-h 0.1 \
  --out runs/direct-abundance
```

Supplying any `--x-over-h` option selects the direct-abundance initializer. A JSON object supplied through `--abundance-file` is convenient for many coordinates; its keys may be symbols or atomic numbers.

### Validated initializer coverage

The common ordinary-star support is approximately:

| label | range |
| --- | --- |
| effective temperature | 4,000–10,500 K |
| log surface gravity | 0.7–5.3 dex (cgs) |
| metallicity | -2.5 to +0.5 [M/H] |
| alpha enhancement | -0.1 to +0.5 [alpha/M] |
| microturbulence | 0.5–4.0 km s⁻¹ |

The three CNO coordinates cover approximately `-0.5 <= [X/M] <= 0.5`. Exact asset hashes, serialized compatibility keys, and training-corpus provenance are recorded in [`release_manifest.json`](../source_data_files/atmosphere_emulator/release_manifest.json).

## Abundances

The reference solar mixture is the photospheric composition by number from Asplund, Grevesse, Sauval, and Scott (2009), commonly abbreviated AGSS09, with helium fixed at `0.078370`. The public abundance coordinates are:

- `[M/H]`: applied to every metal (`Z >= 3`);
- `[alpha/M]`: applied in addition to O, Ne, Mg, Si, S, Ca, and Ti;
- `[C/M]`, `[N/M]`, `[O/M]`: learned eight-label coordinates, with explicit oxygen replacing the alpha-scaled oxygen offset;
- `[X/H]`: an advanced exact-solver override relative to solar. Arbitrary per-element abundances are not learned coordinates of either validated default initializer. The separately installed direct-abundance initializer learns an 81-element starting structure but never replaces the exact solve.

Hydrogen is renormalized after helium and metals are assigned. The user-facing name `metallicity` always means [M/H], not [Fe/H].

## Python interface

```python
from payne_zero_atmosphere import solve_structured_atmosphere

path = solve_structured_atmosphere(
    effective_temperature=4800.0,  # Teff [K]
    log_surface_gravity=2.5,       # logg [cgs]
    out_dir="runs/giant",
    metallicity=-0.5,              # [M/H]
    alpha_enhancement=0.3,         # [alpha/M]
    microturbulence_km_s=1.8,
    c_over_m=0.1,                  # [C/M]; adding C, N, or O selects eight-label
    n_over_m=0.2,                  # [N/M]
    o_over_m=0.1,                  # [O/M]
)
```

Omit the three carbon, nitrogen, and oxygen keywords for the five-label initializer. The direct-abundance Python interface uses the same high-level solver:

```python
from payne_zero_atmosphere import solve_structured_atmosphere
path = solve_structured_atmosphere(
    effective_temperature=4800.0,
    log_surface_gravity=2.5,
    microturbulence_km_s=1.8,
    fe_over_h=-0.2,
    c_over_h=0.2,
    mg_over_h=0.1,
    out_dir="runs/direct-abundance",
)
```

Any of the 81 element names can be supplied independently. The lower-level `abundance_by_atomic_number` mapping remains available for generated mixtures.

Advanced callers may supply an in-memory `ModelAtmosphere` directly:

```python
from payne_zero_atmosphere import (
    AtmosphereConfig,
    AtmosphereInput,
    AtmosphereOutput,
    emulator_warm_start_model,
    run_atmosphere_model,
)
from payne_zero_atmosphere.source_catalogs import (
    molecular_equilibrium_catalog_path,
    source_line_paths,
)

initial_atmosphere, _ = emulator_warm_start_model(
    effective_temperature=5777.0,
    log_surface_gravity=4.44,
)
result = run_atmosphere_model(
    AtmosphereConfig(
        inputs=AtmosphereInput(
            initial_atmosphere=initial_atmosphere,
            molecules_path=molecular_equilibrium_catalog_path(),
            **source_line_paths(),
        ),
        outputs=AtmosphereOutput(
            structured_atmosphere_path="runs/sun/payne_zero_structured_atmosphere.npz"
        ),
        iterations=30,
        enable_molecules=True,
        enable_convection=True,
        enable_convergence_stop=True,
        minimum_iterations_before_convergence=3,
        maximum_deep_layer_relative_temperature_change=5e-4,
    )
)
assert result.converged
```

The production default requires at least three physical iterations before the convergence test can stop the solve.

`AtmosphereInput.initial_atmosphere` accepts a `ModelAtmosphere`, not a path. `read_atmosphere_deck` is an explicit external-reference boundary that returns such a model when a historical text atmosphere must be compared.

## Structured Atmosphere Schema

The machine-readable contract is [`payne_zero_synthesis/atmosphere_schema.json`](../payne_zero_synthesis/atmosphere_schema.json). Schema version 4 makes the population semantics explicit:

| field | unit | meaning |
| --- | --- | --- |
| `temperature` | K | layer temperature |
| `column_mass` | g cm⁻² | mass above unit surface area |
| `gas_pressure` | dyne cm⁻² | gas pressure |
| `electron_density` | cm⁻³ | electron number density |
| `mass_density` | g cm⁻³ | mass density |
| `microturbulence` | cm s⁻¹ | microturbulent velocity |
| `ion_stage_populations` | cm⁻³ | actual ion-stage number-density cube |
| `partition_normalized_populations` | cm⁻³ per partition function | ion-stage populations divided by partition functions |
| `fractional_doppler_widths` | v/c | Doppler-width cube |
| `hydrogen_neutral_population`, `hydrogen_ionized_population` | cm⁻³ | H I and H II number densities |
| `helium_neutral_population`, `helium_singly_ionized_population` | cm⁻³ | He I and He II number densities |
| `molecular_hydrogen_population` | cm⁻³ | H2 number density |
| element-specific `*_partition_normalized_*` fields | cm⁻³ per partition function | dedicated bound-opacity inputs |
| `elemental_abundances` | relative number fraction | elements Z=1..99 |
| `continuum_edge_*` | Hz or nm | continuum sampling edges and intervals |

Actual populations and partition-normalized populations are distinct physical quantities. Free-free opacity uses the actual ion-stage cube; bound opacity uses partition-normalized populations. Schema versions 1 through 3 are accepted only by the synthesis loader as a read-only compatibility boundary; pre-version-3 population cubes are reconstructed before calculation.

## Fixed-Column Quantization

Warm starts and converged columns pass through the reference implementation's fixed-digit column format in memory. This is a numerical compatibility rule: the finite digits alter the exact solver trajectory. No text atmosphere is a production artifact, and synthesis never parses one. `atmosphere_io.py` owns this narrow external/quantization boundary.

## Execution

Compiled Numba kernels use the configured CPU thread pool. Line selection is computed once from the resident catalogs for each atmosphere solve and reused across its iterations.

Run the one-time installation prewarm before timing or producing a grid:

```bash
python -m payne_zero_atmosphere.prewarm \
  --out-dir .cache/payne-zero/prewarm-atmosphere
```

Compiled artifacts default to `.cache/payne-zero/numba-atmosphere/`. A matching later prewarm is a no-op. Re-run it after changing Python, Numba, CPU class, kernel source, or runtime catalogs. The prewarm covers hot, solar, giant, and atomic-only branches with the complete source catalogs, including TiO and H2O. It is installation work and is excluded from atmosphere timings.

| environment variable | effect |
| --- | --- |
| `NUMBA_NUM_THREADS` | cap the Numba thread pool; default uses available CPUs |
| `PAYNE_ZERO_NUMBA_CACHE_DIR` | persistent Payne Zero cache override; also applied when Numba was imported first |
| `NUMBA_CACHE_DIR` | standard Numba override; takes precedence when already set |
| `PAYNE_ZERO_DATA_ROOT` | relocate `source_data_files/` |
| `PAYNE_ZERO_SOURCE_CATALOG_ROOT` | override the shared full source-catalog tree |
| `PAYNE_ZERO_ATMOSPHERE_PROGRESS=1` | print per-iteration convergence progress |

## Modules

| module | responsibility |
| --- | --- |
| `cli.py` | production solve entry point |
| `warm_start.py` | family dispatch, checkpoint validation, decoding, retries |
| `runner.py` | exact iteration and product publication |
| `config.py`, `run_setup.py` | structured configuration and validated setup |
| `atmosphere_io.py` | fixed-column compatibility I/O |
| `equation_of_state.py`, `molecular_equilibrium.py` | ionization and molecular state |
| `continuum_opacity.py`, `line_opacity.py` | continuum and line opacity |
| `line_selection.py`, `line_catalog.py` | resident line selection and catalogs |
| `radiative_transfer.py`, `transfer_kernels.py` | transfer operators and compiled accumulation |
| `temperature_correction.py`, `convection.py`, `convergence.py` | structure iteration |
| `hydrostatic.py`, `radiative_pressure.py`, `rosseland_mean.py` | pressure and mean-opacity updates |
| `specific_internal_energy.py`, `microturbulence.py`, `doppler.py` | physical auxiliary fields |
| `runtime_state.py`, `population_layout.py` | packed runtime state and slot layout |
| `synthesis_bridge.py` | structured schema v4 handoff and validation |
