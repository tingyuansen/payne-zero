# Atmosphere physics tables

These packed NumPy tables are invariant inputs to `payne_zero_atmosphere`. They are runtime physics data, not stored model answers.

| file | contents |
| --- | --- |
| `continuum_level_tables.npz` | explicit level energies and statistical weights used by continuum opacity |
| `continuum_opacity_tables.npz` | bound-free, free-free, scattering, and collision tables |
| `hydrogen_line_profile_tables.npz` | hydrogen Stark and profile interpolation data |
| `ionization_potential_tables.npz` | ionization potentials used by the equation of state |
| `iron_group_partition_tables.npz` | iron-group partition functions |
| `isotope_tables.npz` | major-isotope masses |
| `karzas_latter_tables.npz` | hydrogenic bound-free Gaunt factors |
| `line_opacity_tables.npz` | Voigt and hydrogen profile interpolation tables |
| `molecular_equilibrium_tables.npz` | molecular-equilibrium coefficients and H2 partition data |
| `packed_level_metadata.npz` | packed equation-of-state level metadata |
| `radiative_transfer_tables.npz` | transfer quadrature operators |
| `special_partition_tables.npz` | explicit light-element partition functions |

The loader modules validate required keys, shapes, and dtypes. Physical array names are the public schema; inspect an NPZ with `numpy.load(path).files` when working on a table-specific extension.
