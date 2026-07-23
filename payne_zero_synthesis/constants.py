"""Physical constants used by the synthesis kernels.

Two tiers coexist deliberately:

* The exact constants below match CODATA 2018 — the same values
  ``astropy.constants`` carries — written as literals so the synthesis has no
  astropy dependency and no risk of a library update silently shifting a
  validated number.
* The ``REFERENCE_*`` constants are the rounded values the validated
  reference implementation bakes into specific formulas (Saha, wavenumber
  conversion, profile masses). Substituting the exact values there changes
  validated digits, so they stay pinned and separate.

Cross-package note: ``payne_zero_atmosphere/constants.py`` keeps its own
parity-pinned set. The stage separation is deliberate — this module must stay
importable without the atmosphere package, and each stage's constants are
certified independently, so do not merge or cross-import the two sets.
"""

import math

LIGHT_SPEED_CM_PER_S = 2.99792458e10
LIGHT_SPEED_NM_PER_S = 2.99792458e17
LIGHT_SPEED_ANGSTROM_PER_S = 2.99792458e18

PLANCK_ERG_SECOND = 6.62607015e-27
BOLTZMANN_ERG_PER_K = 1.380649e-16
BOLTZMANN_EV_PER_K = 8.617333262e-5
ATOMIC_MASS_GRAM = 1.66053906660e-24
NATURAL_LOG_10 = math.log(10.0)

CLASSICAL_LINE_STRENGTH_COEFFICIENT = 0.026538 / 1.77245

# Rounded reference constants preserve the validated synthesis arithmetic. Keep
# them separate from the exact constants so parity-sensitive code is explicit.
REFERENCE_PLANCK_ERG_SECOND = 6.6256e-27
REFERENCE_BOLTZMANN_ERG_PER_K = 1.38054e-16
REFERENCE_BOLTZMANN_EV_PER_K = 8.6171e-5
REFERENCE_ATOMIC_MASS_GRAM = 1.660e-24
REFERENCE_WAVENUMBER_PER_EV = 8065.479
REFERENCE_SAHA_COEFFICIENT = 2.4148e15
REFERENCE_NATURAL_LOG_10 = 2.30258509299405
HYDROGEN_PROFILE_ATOMIC_MASS_GRAM = 1.66054e-24

# Raw line-catalog tools use a separate source-catalog conversion convention.
LINE_CATALOG_WAVENUMBER_PER_EV = 8065.54429
