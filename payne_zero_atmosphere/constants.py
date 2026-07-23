"""Shared physical constants for the atmosphere package (single source).

Two tiers coexist deliberately, mirroring the convention in
``payne_zero_synthesis/constants.py`` (which stays a separate, parity-pinned
set — the atmosphere package must never be imported by synthesis, and stage
separation of constants is deliberate):

* ``*_EXACT`` constants match CODATA 2018, written as literals so no library
  update can silently shift a certified number.
* ``*_REFERENCE`` constants are rounded values in the validated reference
  reference implementation bakes into specific formulas.  Substituting the
  exact values there changes certified digits, so differing literals stay
  pinned under distinct names — never unify an ``_EXACT`` with a
  ``_REFERENCE`` value.

Several of these constants are read inside numba-compiled kernels as
module-level globals; plain float module attributes imported by name bake
identically into the compiled kernels.
"""

# Exact (CODATA 2018) tier.
LIGHT_SPEED_CM_PER_S_EXACT = 2.99792458e10
LIGHT_SPEED_NM_PER_S = 2.99792458e17
LIGHT_SPEED_ANGSTROM_PER_S = 2.99792458e18
PLANCK_ERG_SECOND_EXACT = 6.62607015e-27
BOLTZMANN_ERG_PER_K_EXACT = 1.380649e-16

# Rounded reference tier (parity-pinned literals).
LIGHT_SPEED_CM_PER_S_REFERENCE = 2.997925e10
PLANCK_ERG_SECOND_REFERENCE = 6.6256e-27
BOLTZMANN_ERG_PER_K_REFERENCE = 1.38054e-16
BOLTZMANN_EV_PER_K_REFERENCE = 8.6171e-5
ATOMIC_MASS_GRAM_REFERENCE = 1.660e-24

# CRITICAL FENCE: this is the atmosphere-stage eV -> wavenumber
# conversion.  Its synthesis counterpart,
# ``payne_zero_synthesis.constants.LINE_CATALOG_WAVENUMBER_PER_EV``
# (= 8065.54429, the raw line-catalog convention), is DELIBERATELY different
# per stage.  The two must never be merged or "corrected" to agree.
WAVENUMBER_PER_EV_REFERENCE = 8065.479

REFERENCE_NATURAL_LOG_10 = 2.30258509299405
