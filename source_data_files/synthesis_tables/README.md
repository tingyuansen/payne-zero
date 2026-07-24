# Synthesis physics tables

These packed NumPy tables are invariant inputs to `payne_zero_synthesis`. They are runtime physics data, not stored spectra.

| file | contents |
| --- | --- |
| `atomic_masses.npz` | atomic masses used in Doppler widths |
| `continuum_edge_grid.npz` | continuum interpolation edges and samples |
| `continuum_tables.npz` | continuum opacity and scattering tables |
| `ionization_potential_lookup.npz` | ionization potentials used by line damping defaults |
| `line_profile_tables.npz` | Voigt, hydrogen Stark, and specialized profile data |
| `partition_saha_inputs.npz` | partition functions and Saha inputs |
| `transfer_tables.npz` | emergent-flux quadrature operators |

The loader modules validate required keys, shapes, and dtypes. Physical array names are the public schema; inspect an NPZ with `numpy.load(path).files` when working on a table-specific extension.
