# Runtime data

This directory is the shared data root for the atmosphere and synthesis packages. Set `PAYNE_ZERO_DATA_ROOT` to use another location.

| directory | contents |
| --- | --- |
| `atmosphere_tables/` | equation-of-state, opacity, line-profile, and transfer tables used by the atmosphere solver |
| `atmosphere_emulator/` | five-label, eight-label, and optional direct-abundance initializer assets |
| `synthesis_tables/` | invariant tables used by spectral synthesis |
| `source_catalogs/` | atomic and molecular source catalogs used to prepare wavelength windows |

The runtime arrays are distributed through Git LFS. `install.sh` verifies every file against `runtime_data_manifest.json`, installs the two default initializer checkpoints, and builds persistent caches. Set `PAYNE_ZERO_INCLUDE_DIRECT_ABUNDANCE=1` to install the optional direct-abundance checkpoint.

At runtime, `PAYNE_ZERO_DATA_ROOT` can relocate the complete data tree and `PAYNE_ZERO_SOURCE_CATALOG_ROOT` can override only the source catalogs. Initializer training corpora are not runtime inputs. The complete 52,199-row five-label, 53,824-row CNO8, and 82,016-row direct-abundance corpora are available as an optional, hash-verified [release bundle](atmosphere_emulator/TRAINING_CORPORA.md). Other research evidence remains outside the runtime software.

Each subdirectory README documents its stored arrays and physical meanings. The structured atmosphere exchanged by the two packages follows [`payne_zero_synthesis/atmosphere_schema.json`](../payne_zero_synthesis/atmosphere_schema.json).
