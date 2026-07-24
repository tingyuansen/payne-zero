# Source catalogs

This tree contains the full-range atomic and molecular inputs from which both packages prepare their wavelength- and label-dependent working sets. The files are physics inputs, not reference spectra or cached model outputs.

## Atomic and atmosphere inputs

| file | use |
| --- | --- |
| `lines/predicted_atomic_lines_part{1,2,3}.npy` | one packed predicted-line catalog split into three storage shards |
| `lines/observed_atomic_lines.npy` | packed observed atomic lines |
| `lines/high_excitation_lines.npy` | packed high-excitation lines |
| `lines/diatomic_lines.npy` | packed diatomic lines used by atmosphere opacity selection |
| `lines/detailed_transition_lines.npz` | decoded wavelength, excitation, oscillator-strength, damping, and hydrogen-level fields |
| `lines/atomic_source_lines_parsed.npz` | full decoded atomic source list used by synthesis |
| `lines/molecular_equilibrium_atmosphere.npz` | 170-species atmosphere equation-of-state catalog |
| `lines/molecular_equilibrium_synthesis.npz` | 190-species synthesis equation-of-state catalog |

The two molecular-equilibrium files belong to different physical stages and are not interchangeable. The predicted catalog shards are reassembled in row order and do not change the calculation.

## Molecular synthesis inputs

| file | use |
| --- | --- |
| `molecules/manifest.json` | explicit molecular compile order |
| `molecules/molecular_band_lines.npz` | 32 molecular band systems stored as per-band array groups |
| `molecules/titanium_oxide_lines.npy` | Schwenke TiO packed transitions |
| `molecules/water_lines.npy` | Partridge--Schwenke H2O packed transitions |

The molecular compiler follows `manifest.json`; it does not discover inputs by filename globbing.

## Location

Both packages resolve this tree through `PAYNE_ZERO_DATA_ROOT`. `PAYNE_ZERO_SOURCE_CATALOG_ROOT` overrides the source-catalog directory, and `PAYNE_ZERO_SYNTHESIS_SOURCE_CATALOG_ROOT` provides the synthesis-specific override. The installer verifies the catalog tree before use.
