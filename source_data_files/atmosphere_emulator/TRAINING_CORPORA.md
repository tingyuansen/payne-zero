# Atmosphere training corpora

The complete atmosphere corpora used to train and validate the three initializers are distributed as the optional [`v1.3` training-data release](https://github.com/tingyuansen/payne-zero/releases/tag/v1.3). They are not required to run Payne Zero. The release bundle expands into three separate families:

| family | atmospheres | labels | files |
| --- | ---: | --- | --- |
| five-label | 52,199 | `Teff`, `logg`, `[M/H]`, `[alpha/M]`, microturbulence | one NPZ |
| CNO8 | 53,824 | five-label set plus `[C/M]`, `[N/M]`, `[O/M]` | one NPZ |
| direct X/H | 82,016 | `Teff`, `logg`, microturbulence, and complete elemental mixtures | six immutable NPZ shards |

These are the complete atmosphere-training datasets used by the paper. The runtime checkpoints remain in the normal repository data. Spectra, timing measurements, and figure products are separate research outputs rather than atmosphere-training corpora.

## Common atmosphere arrays

Every corpus stores converged profiles on the same 80-layer Rosseland-depth grid.

| key | shape | meaning |
| --- | --- | --- |
| `atmosphere_profiles` | `(N, 80, 6)` | converged atmosphere fields |
| `standard_rosseland_optical_depth` | `(N, 80)` | common depth coordinate |
| `target_fields` | `(6,)` | field order for the last profile axis |
| `iterations_to_convergence` | `(N,)` | physical-solver iteration count |

The six fields are `column_mass`, `temperature`, `gas_pressure`, `electron_density`, `rosseland_opacity`, and `radiative_acceleration`.

## Five-label and CNO8 labels

The files `five_label/strict_truth_52199.npz` and `cno8/strict_truth_53824.npz` store each atmosphere's labels and provenance as JSON in `labels_json`. Load one row with:

```python
import json
import numpy as np

with np.load("five_label/strict_truth_52199.npz", allow_pickle=False) as data:
    labels = json.loads(str(data["labels_json"][0]))
    fields = data["target_fields"].tolist()
    atmosphere = data["atmosphere_profiles"][0]
```

The CNO8 file also provides `label_fields`, `identity_sha256`, `parent_group_ids`, `record_kinds`, and split-role metadata. The five-label file provides `slugs`, acquisition-ledger identity, and `depth_grid_verified`.

## Direct-X/H labels

The direct-abundance corpus retains six shards because each shard is bound to its original generation campaign and SHA-256 identity. All shards use the same arrays:

| key | shape | meaning |
| --- | --- | --- |
| `stellar_labels` | `(N, 3)` | values ordered by `stellar_label_fields` |
| `stellar_label_fields` | `(3,)` | `effective_temperature`, `log_surface_gravity`, `microturbulence_km_s` |
| `abundance_vectors` | `(N, 97)` | `[X/H]` for atomic numbers 3 through 99 |
| `atomic_numbers` | `(97,)` | atomic number for each abundance column |
| `element_order` | `(97,)` | element symbol for each abundance column |
| `identity_sha256` | `(N,)` | immutable atmosphere-mixture identity |
| `roles` | `(N,)` | fixed training or validation role |
| `source_families` | `(N,)` | generation-campaign family |

The runtime initializer exposes the 81 elements with finite adopted solar references. The remaining 16 solver slots are retained in `abundance_vectors` so every stored atmosphere carries its complete mixture. `metadata_json` preserves generation provenance, including historical scratch paths, and is not required when loading or retraining from the numerical arrays.

## Verification

After extracting the bundle, run:

```bash
cd payne-zero-v1.3-atmosphere-training-corpora
shasum -a 256 -c SHA256SUMS
```

The included `manifest.json` records every file name, byte count, row count, array shape, and SHA-256 digest. The archive preserves the exact corpus files used by the paper; it does not recompute or concatenate them.
