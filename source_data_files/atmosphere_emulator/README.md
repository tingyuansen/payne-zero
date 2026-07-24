# Atmosphere initializer assets

This directory stores neural-network initializers and optional training evidence. An initializer supplies only the starting structure; the physical atmosphere solver decides convergence and produces the atmosphere used by synthesis.

## Families

| directory | coordinates | runtime status |
| --- | --- | --- |
| `five_label/` | `Teff`, `logg`, `[M/H]`, `[alpha/M]`, microturbulence | installed by default |
| `cno8/` | five-label set plus independent carbon, nitrogen, and oxygen coordinates `[C/M]`, `[N/M]`, and `[O/M]` | installed by default |
| `direct_abundance/` | `Teff`, `logg`, microturbulence, and 81 `[X/H]` values | optional direct-abundance initializer |

The ordinary dispatcher selects the five-label family unless a C, N, or O coordinate is supplied. Direct abundance is a separate explicit interface and is not selected automatically.

All families decode the same six fields on 80 depth layers: column mass, temperature, gas pressure, electron density, Rosseland opacity, and radiative acceleration. The five- and eight-label models use 160-component principal-component-analysis (PCA) representations followed by neural label maps.

## Coverage

| coordinate | supported range |
| --- | --- |
| effective temperature | 4,000–10,500 K |
| `logg` | 0.7–5.3 |
| `[M/H]` or direct `[Fe/H]` | −2.5–0.5 |
| `[alpha/M]` | −0.1–0.5 |
| microturbulence | 0.5–4.0 km s⁻¹ |
| each carbon, nitrogen, or oxygen `[X/M]`, or direct `[X/Fe]` | −0.5–0.5 |

The five-label model was trained from 52,199 converged atmospheres, the eight-label model from 53,824 independently solved carbon-nitrogen-oxygen atmospheres, and the direct-abundance model from 82,016 independently varied complete mixtures. The training corpora are not runtime inputs. They are available as a separate, hash-verified [`v1.3` release bundle](TRAINING_CORPORA.md) so the runtime checkout remains compact.

## Installation and contracts

From the repository root, `install.sh` installs the five- and eight-label checkpoints. Include the direct-abundance checkpoint with:

```bash
PAYNE_ZERO_INCLUDE_DIRECT_ABUNDANCE=1 ./install.sh
```

`release_manifest.json` documents the default family labels, support, checkpoint identities, and training splits. `direct_abundance/manifest.json` defines the complete abundance order and mandatory physical-solve policy, and is validated when the optional initializer is loaded.
