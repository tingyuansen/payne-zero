"""Payne Zero-owned warm-start atmospheres.

The bundled five-label emulator supplies all six layer fields needed by the
exact atmosphere solver. This module owns the boundary between that prediction
and the solver: abundance scaling, checkpoint validation, deterministic nearby
initializer trials, and the quantized warm-start formatter/writer.

`emulator_warm_start_model` is the primary entry point: it returns the
in-memory `ModelAtmosphere` the solver consumes plus the matching deck text.
The prediction is always format-then-parsed through the deck text (in
memory) because the fixed-digit deck format quantizes the values and the
certified baselines converged through that quantization.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Mapping
import warnings

import numpy as np

from .atmosphere_io import ModelAtmosphere, parse_atmosphere_deck
from .data_files import atmosphere_emulator_dir


DEFAULT_EMULATOR_ASSET_DIR = atmosphere_emulator_dir()

FIVE_LABEL_FAMILY = "five_label"
CNO8_FAMILY = "cno8"
WARM_START_FAMILIES = (FIVE_LABEL_FAMILY, CNO8_FAMILY)

DEFAULT_FIVE_LABEL_WEIGHTS_PATH = (
    DEFAULT_EMULATOR_ASSET_DIR / FIVE_LABEL_FAMILY / "checkpoint.pt"
)
DEFAULT_CNO8_WEIGHTS_PATH = DEFAULT_EMULATOR_ASSET_DIR / CNO8_FAMILY / "checkpoint.pt"

FIVE_LABEL_CHECKPOINT_FORMAT = "payne_zero_complete_atmosphere_latent_v2"
CNO8_CHECKPOINT_FORMAT = "payne_zero_cno8_complete_atmosphere_latent_v3"
FIVE_LABEL_CHECKPOINT_FEATURE_FIELDS = (
    "temperature_ratio_5040_k_over_temperature",
    "log10_surface_gravity_cgs",
    "metallicity",
    "alpha_enhancement",
    "microturbulence_km_s",
)
CNO8_CHECKPOINT_FEATURE_FIELDS = (
    *FIVE_LABEL_CHECKPOINT_FEATURE_FIELDS,
    "carbon_enhancement",
    "nitrogen_enhancement",
    "oxygen_enhancement",
)
CNO_ENHANCEMENT_FIELDS = (
    "carbon_enhancement",
    "nitrogen_enhancement",
    "oxygen_enhancement",
)
CNO_ATOMIC_NUMBERS = {
    "carbon_enhancement": 6,
    "nitrogen_enhancement": 7,
    "oxygen_enhancement": 8,
}
INITIALIZER_COORDINATE_FIELDS = (
    "log10_column_mass_increment",
    "log10_temperature_relative_to_grey",
    "log10_gas_pressure",
    "log10_electron_density",
    "log10_rosseland_opacity",
    "asinh_radiative_acceleration",
)
INITIALIZER_OUTPUT_FIELDS = (
    "column_mass",
    "temperature",
    "gas_pressure",
    "electron_density",
    "rosseland_opacity",
    "radiative_acceleration",
)
INITIALIZER_STANDARD_ROSSELAND_OPTICAL_DEPTH = 10.0 ** (
    -6.875 + np.arange(80, dtype=np.float64) * 0.125
)
INITIALIZER_BOUND_TOLERANCE_FRACTION = 1.0e-4

HELIUM_NUMBER_FRACTION = 0.078370
ALPHA_ELEMENT_ATOMIC_NUMBERS = (8, 10, 12, 14, 16, 20, 22)

SOLAR_METAL_LOG_ABUNDANCES_3_TO_99 = np.array(
    [
        -10.99,
        -10.66,
        -9.34,
        -3.61,
        -4.21,
        -3.35,
        -7.48,
        -4.11,
        -5.80,
        -4.44,
        -5.59,
        -4.53,
        -6.63,
        -4.92,
        -6.54,
        -5.64,
        -7.01,
        -5.70,
        -8.89,
        -7.09,
        -8.11,
        -6.40,
        -6.61,
        -4.54,
        -7.05,
        -5.82,
        -7.85,
        -7.48,
        -9.00,
        -8.39,
        -9.74,
        -8.70,
        -9.50,
        -8.79,
        -9.52,
        -9.17,
        -9.83,
        -9.46,
        -10.58,
        -10.16,
        -20.00,
        -10.29,
        -11.13,
        -10.47,
        -11.10,
        -10.33,
        -11.24,
        -10.00,
        -11.03,
        -9.86,
        -10.49,
        -9.80,
        -10.96,
        -9.86,
        -10.94,
        -10.46,
        -11.32,
        -10.62,
        -20.00,
        -11.08,
        -11.52,
        -10.97,
        -11.74,
        -10.94,
        -11.56,
        -11.12,
        -11.94,
        -11.20,
        -11.94,
        -11.19,
        -12.16,
        -11.19,
        -11.78,
        -10.64,
        -10.66,
        -10.42,
        -11.12,
        -10.87,
        -11.14,
        -10.29,
        -11.39,
        -20.00,
        -20.00,
        -20.00,
        -20.00,
        -20.00,
        -20.00,
        -12.02,
        -20.00,
        -12.58,
        -20.00,
        -20.00,
        -20.00,
        -20.00,
        -20.00,
        -20.00,
        -20.00,
    ],
    dtype=np.float64,
)

ELEMENT_SYMBOLS: dict[int, str] = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    37: "Rb",
    38: "Sr",
    39: "Y",
    40: "Zr",
    41: "Nb",
    42: "Mo",
    43: "Tc",
    44: "Ru",
    45: "Rh",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    49: "In",
    50: "Sn",
    51: "Sb",
    52: "Te",
    53: "I",
    54: "Xe",
    55: "Cs",
    56: "Ba",
    57: "La",
    58: "Ce",
    59: "Pr",
    60: "Nd",
    61: "Pm",
    62: "Sm",
    63: "Eu",
    64: "Gd",
    65: "Tb",
    66: "Dy",
    67: "Ho",
    68: "Er",
    69: "Tm",
    70: "Yb",
    71: "Lu",
    72: "Hf",
    73: "Ta",
    74: "W",
    75: "Re",
    76: "Os",
    77: "Ir",
    78: "Pt",
    79: "Au",
    80: "Hg",
    81: "Tl",
    82: "Pb",
    83: "Bi",
    84: "Po",
    85: "At",
    86: "Rn",
    87: "Fr",
    88: "Ra",
    89: "Ac",
    90: "Th",
    91: "Pa",
    92: "U",
    93: "NP",
    94: "Pu",
    95: "Am",
    96: "Cm",
    97: "Bk",
    98: "Cf",
    99: "Es",
}
ATOMIC_NUMBER_BY_SYMBOL = {symbol.lower(): z for z, symbol in ELEMENT_SYMBOLS.items()}


def compute_metal_log_number_abundances(
    metallicity: float = 0.0,
    alpha_enhancement: float = 0.0,
    absolute_abundance_offsets: Mapping[int, float] | None = None,
) -> np.ndarray:
    """Return log10 number abundances for elements Z=3..99.

    The first two scalar offsets match the existing production convention:
    apply [M/H] to all metals, then apply [alpha/M] to O, Ne, Mg, Si, S, Ca,
    and Ti.  Individual element offsets are absolute bracket offsets relative
    to solar and override the scalar scaling for that element.
    """

    abundances = SOLAR_METAL_LOG_ABUNDANCES_3_TO_99.copy() + float(metallicity)
    for atomic_number in ALPHA_ELEMENT_ATOMIC_NUMBERS:
        abundances[atomic_number - 3] += float(alpha_enhancement)
    if absolute_abundance_offsets:
        for atomic_number, offset in absolute_abundance_offsets.items():
            if 3 <= int(atomic_number) <= 99:
                abundances[int(atomic_number) - 3] = SOLAR_METAL_LOG_ABUNDANCES_3_TO_99[
                    int(atomic_number) - 3
                ] + float(offset)
    return abundances


def compute_hydrogen_fraction(
    metallicity: float = 0.0,
    alpha_enhancement: float = 0.0,
    absolute_abundance_offsets: Mapping[int, float] | None = None,
) -> float:
    """Return the H number fraction after He and metals are assigned."""

    metal_log_number_abundances = compute_metal_log_number_abundances(
        metallicity=metallicity,
        alpha_enhancement=alpha_enhancement,
        absolute_abundance_offsets=absolute_abundance_offsets,
    )
    return float(
        1.0 - HELIUM_NUMBER_FRACTION - np.sum(10.0**metal_log_number_abundances)
    )


def parse_abundance_offset(text: str) -> tuple[int, float]:
    """Parse ``Fe:+0.3`` or ``26:+0.3`` into ``(26, 0.3)``."""

    fields = text.split(":")
    if len(fields) != 2:
        raise ValueError(f"abundance offset must look like 'Fe:+0.3', got {text!r}")
    element_text, offset_text = fields[0].strip(), fields[1].strip()
    if element_text.isdigit():
        atomic_number = int(element_text)
    else:
        try:
            atomic_number = ATOMIC_NUMBER_BY_SYMBOL[element_text.lower()]
        except KeyError as exc:
            raise ValueError(f"unknown element symbol {element_text!r}") from exc
    if not 3 <= atomic_number <= 99:
        raise ValueError(
            f"element Z={atomic_number} is outside the supported 3..99 range"
        )
    return atomic_number, float(offset_text)


def select_warm_start_family(
    *,
    carbon_enhancement: float | None = None,
    nitrogen_enhancement: float | None = None,
    oxygen_enhancement: float | None = None,
    absolute_abundance_offsets: Mapping[int, float] | None = None,
) -> str:
    """Select the only production family consistent with the public labels.

    The five-label workhorse remains the default. Any explicit C, N, or O
    coordinate, including an absolute ``[X/H]`` entry in
    ``absolute_abundance_offsets``, selects the eight-label CNO-aware
    initializer. There is no fallback between the two families.
    """

    explicit_relative = (
        carbon_enhancement,
        nitrogen_enhancement,
        oxygen_enhancement,
    )
    explicit_absolute = {
        int(atomic_number) for atomic_number in (absolute_abundance_offsets or {})
    }
    if any(value is not None for value in explicit_relative) or explicit_absolute & {
        6,
        7,
        8,
    }:
        return CNO8_FAMILY
    return FIVE_LABEL_FAMILY


def resolve_cno8_labels(
    *,
    metallicity: float,
    alpha_enhancement: float,
    carbon_enhancement: float | None = None,
    nitrogen_enhancement: float | None = None,
    oxygen_enhancement: float | None = None,
    absolute_abundance_offsets: Mapping[int, float] | None = None,
) -> dict[str, float]:
    """Resolve partial CNO input into the canonical relative coordinates.

    Missing carbon and nitrogen default to ``[X/M]=0``. Missing oxygen follows
    the bulk alpha enhancement. Absolute CNO entries in ``absolute_abundance_offsets``
    are converted to relative coordinates and must agree with any duplicate
    explicit relative value.
    """

    metal = float(metallicity)
    alpha = float(alpha_enhancement)
    requested = {
        "carbon_enhancement": carbon_enhancement,
        "nitrogen_enhancement": nitrogen_enhancement,
        "oxygen_enhancement": oxygen_enhancement,
    }
    defaults = {
        "carbon_enhancement": 0.0,
        "nitrogen_enhancement": 0.0,
        "oxygen_enhancement": alpha,
    }
    absolute = {
        int(atomic_number): float(value)
        for atomic_number, value in (absolute_abundance_offsets or {}).items()
    }
    resolved: dict[str, float] = {}
    for field, atomic_number in CNO_ATOMIC_NUMBERS.items():
        relative = requested[field]
        from_absolute = (
            absolute[atomic_number] - metal if atomic_number in absolute else None
        )
        if (
            relative is not None
            and from_absolute is not None
            and not np.isclose(
                float(relative),
                from_absolute,
                rtol=0.0,
                atol=32.0 * np.finfo(np.float64).eps,
            )
        ):
            symbol = ELEMENT_SYMBOLS[atomic_number]
            raise ValueError(
                f"[{symbol}/M] conflicts with the absolute [{symbol}/H] override"
            )
        value = (
            float(relative)
            if relative is not None
            else from_absolute
            if from_absolute is not None
            else defaults[field]
        )
        resolved[field] = float(value)
    if not np.all(np.isfinite([metal, alpha, *resolved.values()])):
        raise ValueError("CNO abundance labels must be finite")
    return resolved


def cno8_absolute_abundance_offsets(
    *,
    metallicity: float,
    cno_labels: Mapping[str, float],
) -> dict[int, float]:
    """Return absolute ``[C/H]``, ``[N/H]``, and ``[O/H]`` solver offsets."""

    metal = float(metallicity)
    return {
        atomic_number: metal + float(cno_labels[field])
        for field, atomic_number in CNO_ATOMIC_NUMBERS.items()
    }


def atmosphere_prediction_to_layer_table(
    prediction: Mapping[str, np.ndarray],
    *,
    microturbulence_km_s: float = 2.0,
) -> np.ndarray:
    """Pack emulator output columns into the 9-column READ DECK6 layer table."""

    column_mass = np.asarray(prediction["column_mass"], dtype=np.float64)
    layer_count = column_mass.size
    layer_table = np.zeros((layer_count, 9), dtype=np.float64)
    layer_table[:, 0] = column_mass
    layer_table[:, 1] = np.asarray(prediction["temperature"], dtype=np.float64)
    layer_table[:, 2] = np.asarray(prediction["gas_pressure"], dtype=np.float64)
    layer_table[:, 3] = np.asarray(prediction["electron_density"], dtype=np.float64)
    layer_table[:, 4] = np.asarray(prediction["rosseland_opacity"], dtype=np.float64)
    layer_table[:, 5] = np.asarray(
        prediction["radiative_acceleration"], dtype=np.float64
    )
    layer_table[:, 6] = float(microturbulence_km_s) * 1.0e5
    return layer_table


def format_warm_start_deck(
    *,
    effective_temperature: float,
    log_surface_gravity: float,
    layer_table: np.ndarray,
    metallicity: float = 0.0,
    alpha_enhancement: float = 0.0,
    absolute_abundance_offsets: Mapping[int, float] | None = None,
    title: str | None = None,
) -> str:
    """Format a fixed-width READ DECK6 warm-start atmosphere as deck text.

    The fixed-digit deck format QUANTIZES the emulator prediction, and the
    certified baselines converged through exactly that quantization.  Every
    consumer of a warm start — in-memory or on disk — must therefore go
    through this formatter (and the shared deck parser) so the solver input
    is identical either way.
    """

    table = np.asarray(layer_table, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] != 9:
        raise ValueError("layer_table must have shape (n_layers, 9)")

    metal_log_number_abundances = compute_metal_log_number_abundances(
        metallicity=metallicity,
        alpha_enhancement=alpha_enhancement,
        absolute_abundance_offsets=absolute_abundance_offsets,
    )
    hydrogen_fraction = compute_hydrogen_fraction(
        metallicity=metallicity,
        alpha_enhancement=alpha_enhancement,
        absolute_abundance_offsets=absolute_abundance_offsets,
    )

    title_text = title if title is not None else "Payne Zero emulator warm start"
    lines: list[str] = [
        f"TEFF   {effective_temperature:.0f}.  GRAVITY {log_surface_gravity:7.4f} LTE ",
        f"TITLE {title_text:<80}",
        " OPACITY IFOP 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 0 0 0",
        " CONVECTION ON   1.25 TURBULENCE OFF  0.00  0.00  0.00  0.00",
        (
            "ABUNDANCE SCALE   1.00000 ABUNDANCE CHANGE 1 "
            f"{hydrogen_fraction:.5f} 2 {HELIUM_NUMBER_FRACTION:.5f}"
        ),
    ]

    for start_atomic_number in range(3, 100, 6):
        end_atomic_number = min(start_atomic_number + 5, 99)
        parts = [" ABUNDANCE CHANGE"]
        for atomic_number in range(start_atomic_number, end_atomic_number + 1):
            abundance = metal_log_number_abundances[atomic_number - 3]
            if abundance > -10.0:
                parts.append(f" {atomic_number:2d}  {abundance:5.2f}")
            else:
                parts.append(f" {atomic_number:2d} {abundance:6.2f}")
        lines.append("".join(parts))

    lines.append(" ABUNDANCE TABLE")
    lines.append(
        f"    1H   {hydrogen_fraction:.6f}       2He  {HELIUM_NUMBER_FRACTION:.6f}"
    )
    for start_atomic_number in range(3, 100, 5):
        end_atomic_number = min(start_atomic_number + 4, 99)
        parts = []
        for atomic_number in range(start_atomic_number, end_atomic_number + 1):
            symbol = ELEMENT_SYMBOLS[atomic_number]
            abundance = metal_log_number_abundances[atomic_number - 3]
            if abundance > -10.0:
                padding = " " * (3 - len(symbol))
                parts.append(
                    f"{atomic_number:5d}{symbol}{padding}{abundance:6.3f} 0.000"
                )
            else:
                padding = " " * (3 - len(symbol) - 1)
                parts.append(
                    f"{atomic_number:5d}{symbol}{padding}{abundance:7.3f} 0.000"
                )
        lines.append("".join(parts))

    lines.append(f"READ DECK6 {table.shape[0]} RHOX,T,P,XNE,ABROSS,ACCRAD,VTURB")
    for row in table:
        row_fields = [f" {row[0]:.8E}", f"   {row[1]:.1f}"]
        for column_index in range(2, 6):
            row_fields.append(f" {row[column_index]:.3E}")
        row_fields.append(f" {row[6]:.3E}")
        row_fields.append(f" {row[7]:.3E} {row[8]:.3E}")
        lines.append("".join(row_fields))

    lines.append("PRADK 5.0000E-01")
    lines.append("BEGIN                    ITERATION  15 COMPLETED")

    return "\n".join(lines)


class AtmosphereInitializer:
    """Complete-state PCA/latent warm-start emulator."""

    def __init__(self, checkpoint: Mapping, *, device: str):
        import torch.nn as nn

        _validate_initializer_checkpoint(checkpoint)
        self.checkpoint = checkpoint
        self.device = device
        self.family = _checkpoint_family(checkpoint)
        self.checkpoint_feature_fields = _checkpoint_feature_fields(checkpoint)
        self.standard_rosseland_optical_depth = np.asarray(
            checkpoint["coordinates"]["standard_rosseland_optical_depth"],
            dtype=np.float64,
        ).copy()
        config = checkpoint["model"]["config"]
        layers: list[nn.Module] = []
        dimension = int(config["input_dim"])
        for _ in range(int(config["hidden_layers"])):
            layers.extend((nn.Linear(dimension, int(config["width"])), nn.SiLU()))
            dimension = int(config["width"])
        layers.append(nn.Linear(dimension, int(config["output_dim"])))
        self.model = nn.Sequential(*layers).to(device)
        self.model.load_state_dict(checkpoint["model"]["state_dict"], strict=True)
        self.model.eval()

    def predict(
        self,
        *,
        effective_temperature: float,
        log_surface_gravity: float,
        metallicity: float,
        alpha_enhancement: float,
        microturbulence_km_s: float,
        carbon_enhancement: float | None = None,
        nitrogen_enhancement: float | None = None,
        oxygen_enhancement: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Decode one complete six-field atmosphere on the fixed depth grid."""

        import torch

        features = _initializer_checkpoint_features(
            effective_temperature=effective_temperature,
            log_surface_gravity=log_surface_gravity,
            metallicity=metallicity,
            alpha_enhancement=alpha_enhancement,
            microturbulence_km_s=microturbulence_km_s,
            carbon_enhancement=carbon_enhancement,
            nitrogen_enhancement=nitrogen_enhancement,
            oxygen_enhancement=oxygen_enhancement,
            checkpoint_feature_fields=self.checkpoint_feature_fields,
        )
        _require_initializer_bounds(
            features,
            self.checkpoint["labels"]["bounds"],
            checkpoint_feature_fields=self.checkpoint_feature_fields,
        )
        label_mean = np.asarray(self.checkpoint["labels"]["mean"], dtype=np.float64)
        label_std = np.asarray(self.checkpoint["labels"]["std"], dtype=np.float64)
        model_input = (features - label_mean) / label_std
        with torch.no_grad():
            standardized_coefficients = (
                self.model(
                    torch.as_tensor(
                        model_input[None, :], dtype=torch.float32, device=self.device
                    )
                )
                .detach()
                .cpu()
                .numpy()
            )

        pca = self.checkpoint["pca"]
        coefficients = standardized_coefficients * np.asarray(
            pca["coefficient_std"], dtype=np.float64
        ) + np.asarray(pca["coefficient_mean"], dtype=np.float64)
        standardized_coordinates = coefficients @ np.asarray(
            pca["basis"], dtype=np.float64
        )
        flattened = standardized_coordinates * np.asarray(
            pca["coordinate_std"], dtype=np.float64
        ) + np.asarray(pca["coordinate_mean"], dtype=np.float64)
        coordinates = flattened.reshape(80, len(INITIALIZER_COORDINATE_FIELDS))
        acceleration_scale = float(self.checkpoint["coordinates"]["acceleration_scale"])
        grey_temperature = (
            float(effective_temperature)
            * (0.75 * (self.standard_rosseland_optical_depth + 2.0 / 3.0)) ** 0.25
        )
        decoded = np.empty((80, 6), dtype=np.float64)
        decoded[:, 0] = np.cumsum(10.0 ** np.clip(coordinates[:, 0], -30.0, 30.0))
        decoded[:, 1] = grey_temperature * 10.0 ** np.clip(coordinates[:, 1], -3.0, 3.0)
        decoded[:, 2:5] = 10.0 ** np.clip(coordinates[:, 2:5], -30.0, 30.0)
        decoded[:, 5] = acceleration_scale * np.sinh(
            np.clip(coordinates[:, 5], -20.0, 20.0)
        )
        if not np.all(np.isfinite(decoded)) or np.any(decoded[:, :5] <= 0.0):
            raise RuntimeError("atmosphere initializer decoded an invalid atmosphere")
        if np.any(np.diff(decoded[:, 0]) <= 0.0):
            raise RuntimeError(
                "atmosphere initializer decoded nonmonotonic column mass"
            )
        return {
            field: decoded[:, index]
            for index, field in enumerate(INITIALIZER_OUTPUT_FIELDS)
        }

    def predict_layer_table(self, **labels: float) -> np.ndarray:
        """Return the standard nine-column warm-start table."""

        prediction = self.predict(**labels)
        return atmosphere_prediction_to_layer_table(
            prediction,
            microturbulence_km_s=float(labels["microturbulence_km_s"]),
        )


def _initializer_checkpoint_features(
    *,
    effective_temperature: float,
    log_surface_gravity: float,
    metallicity: float,
    alpha_enhancement: float,
    microturbulence_km_s: float,
    carbon_enhancement: float | None = None,
    nitrogen_enhancement: float | None = None,
    oxygen_enhancement: float | None = None,
    checkpoint_feature_fields: tuple[str, ...] = FIVE_LABEL_CHECKPOINT_FEATURE_FIELDS,
) -> np.ndarray:
    if not np.isfinite(effective_temperature) or effective_temperature <= 0.0:
        raise ValueError("effective temperature must be finite and positive")
    values = [
        5040.0 / float(effective_temperature),
        log_surface_gravity,
        metallicity,
        alpha_enhancement,
        microturbulence_km_s,
    ]
    if checkpoint_feature_fields == CNO8_CHECKPOINT_FEATURE_FIELDS:
        cno_values = (
            carbon_enhancement,
            nitrogen_enhancement,
            oxygen_enhancement,
        )
        if any(value is None for value in cno_values):
            missing = [
                field
                for field, value in zip(CNO_ENHANCEMENT_FIELDS, cno_values, strict=True)
                if value is None
            ]
            raise ValueError(
                "eight-label CNO-aware input is missing fields: " + ", ".join(missing)
            )
        values.extend(float(value) for value in cno_values if value is not None)
    elif checkpoint_feature_fields != FIVE_LABEL_CHECKPOINT_FEATURE_FIELDS:
        raise ValueError(
            f"unsupported checkpoint features {checkpoint_feature_fields!r}"
        )
    features = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(features)):
        raise ValueError("warm-start labels must be finite")
    return features


def _require_initializer_bounds(
    features: np.ndarray,
    bounds: Mapping[str, object],
    *,
    checkpoint_feature_fields: tuple[str, ...] = FIVE_LABEL_CHECKPOINT_FEATURE_FIELDS,
) -> None:
    violations = []
    for index, field in enumerate(checkpoint_feature_fields):
        lower, upper = (float(value) for value in bounds[field])
        value = float(features[index])
        tolerance = INITIALIZER_BOUND_TOLERANCE_FRACTION * max(1.0, upper - lower)
        if value < lower - tolerance or value > upper + tolerance:
            violations.append(f"{field}={value:g} not in [{lower:g}, {upper:g}]")
    if violations:
        raise ValueError(
            "label is outside atmosphere initializer support: " + ", ".join(violations)
        )


def warm_start_supported(
    *,
    effective_temperature: float,
    log_surface_gravity: float,
    metallicity: float = 0.0,
    alpha_enhancement: float = 0.0,
    microturbulence_km_s: float = 2.0,
    carbon_enhancement: float | None = None,
    nitrogen_enhancement: float | None = None,
    oxygen_enhancement: float | None = None,
    absolute_abundance_offsets: Mapping[int, float] | None = None,
    checkpoint_path: Path | None = None,
    device: str = "cpu",
) -> bool:
    """Return whether the deterministically selected family covers the label."""

    try:
        family = select_warm_start_family(
            carbon_enhancement=carbon_enhancement,
            nitrogen_enhancement=nitrogen_enhancement,
            oxygen_enhancement=oxygen_enhancement,
            absolute_abundance_offsets=absolute_abundance_offsets,
        )
        cno_labels = (
            resolve_cno8_labels(
                metallicity=metallicity,
                alpha_enhancement=alpha_enhancement,
                carbon_enhancement=carbon_enhancement,
                nitrogen_enhancement=nitrogen_enhancement,
                oxygen_enhancement=oxygen_enhancement,
                absolute_abundance_offsets=absolute_abundance_offsets,
            )
            if family == CNO8_FAMILY
            else {}
        )
        selected_path = checkpoint_path or (
            DEFAULT_CNO8_WEIGHTS_PATH
            if family == CNO8_FAMILY
            else DEFAULT_FIVE_LABEL_WEIGHTS_PATH
        )
        emulator = load_atmosphere_initializer(
            checkpoint_path=selected_path,
            device=device,
        )
        emulator_family = getattr(emulator, "family", family)
        checkpoint_feature_fields = getattr(
            emulator,
            "checkpoint_feature_fields",
            CNO8_CHECKPOINT_FEATURE_FIELDS
            if family == CNO8_FAMILY
            else FIVE_LABEL_CHECKPOINT_FEATURE_FIELDS,
        )
        if emulator_family != family:
            raise ValueError("warm-start checkpoint belongs to the wrong family")
        features = _initializer_checkpoint_features(
            effective_temperature=effective_temperature,
            log_surface_gravity=log_surface_gravity,
            metallicity=metallicity,
            alpha_enhancement=alpha_enhancement,
            microturbulence_km_s=microturbulence_km_s,
            **cno_labels,
            checkpoint_feature_fields=checkpoint_feature_fields,
        )
        _require_initializer_bounds(
            features,
            emulator.checkpoint["labels"]["bounds"],
            checkpoint_feature_fields=checkpoint_feature_fields,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return False
    return True


def _checkpoint_family(checkpoint: Mapping) -> str:
    checkpoint_format = checkpoint.get("format")
    if checkpoint_format == FIVE_LABEL_CHECKPOINT_FORMAT:
        return FIVE_LABEL_FAMILY
    if (
        checkpoint_format == CNO8_CHECKPOINT_FORMAT
        and checkpoint.get("family") == CNO8_FAMILY
    ):
        return CNO8_FAMILY
    raise ValueError(
        "atmosphere initializer checkpoint format or family is incompatible"
    )


def _checkpoint_feature_fields(checkpoint: Mapping) -> tuple[str, ...]:
    return (
        FIVE_LABEL_CHECKPOINT_FEATURE_FIELDS
        if _checkpoint_family(checkpoint) == FIVE_LABEL_FAMILY
        else CNO8_CHECKPOINT_FEATURE_FIELDS
    )


def _validate_initializer_checkpoint(checkpoint: Mapping) -> None:
    checkpoint_feature_fields = _checkpoint_feature_fields(checkpoint)
    schema_version = checkpoint.get("schema_version", -1)
    if (
        not isinstance(schema_version, (int, np.integer))
        or isinstance(schema_version, (bool, np.bool_))
        or int(schema_version) != 1
    ):
        raise ValueError("atmosphere initializer checkpoint schema is incompatible")
    labels = checkpoint.get("labels", {})
    coordinates = checkpoint.get("coordinates", {})
    pca = checkpoint.get("pca", {})
    model = checkpoint.get("model", {})
    if tuple(labels.get("fields", ())) != checkpoint_feature_fields:
        raise ValueError("atmosphere initializer checkpoint labels are incompatible")
    label_mean = np.asarray(labels.get("mean"), dtype=np.float64)
    label_std = np.asarray(labels.get("std"), dtype=np.float64)
    if (
        label_mean.shape != (len(checkpoint_feature_fields),)
        or label_std.shape != label_mean.shape
        or not np.all(np.isfinite(label_mean))
        or np.any(~np.isfinite(label_std) | (label_std <= 0.0))
    ):
        raise ValueError(
            "atmosphere initializer checkpoint label normalization is invalid"
        )
    bounds = labels.get("bounds", {})
    for field in checkpoint_feature_fields:
        values = np.asarray(bounds.get(field), dtype=np.float64)
        if (
            values.shape != (2,)
            or not np.all(np.isfinite(values))
            or values[0] > values[1]
        ):
            raise ValueError(
                f"atmosphere initializer checkpoint bound {field} is invalid"
            )
    if tuple(coordinates.get("fields", ())) != INITIALIZER_COORDINATE_FIELDS:
        raise ValueError(
            "atmosphere initializer checkpoint coordinates are incompatible"
        )
    if tuple(coordinates.get("target_fields", ())) != INITIALIZER_OUTPUT_FIELDS:
        raise ValueError("atmosphere initializer checkpoint targets are incompatible")
    acceleration_scale = float(coordinates.get("acceleration_scale", 0.0))
    if not np.isfinite(acceleration_scale) or acceleration_scale <= 0.0:
        raise ValueError(
            "atmosphere initializer checkpoint acceleration scale is invalid"
        )
    standard_rosseland_optical_depth = np.asarray(
        coordinates.get("standard_rosseland_optical_depth"), dtype=np.float64
    )
    if (
        standard_rosseland_optical_depth.shape
        != INITIALIZER_STANDARD_ROSSELAND_OPTICAL_DEPTH.shape
        or not np.all(np.isfinite(standard_rosseland_optical_depth))
        or np.any(standard_rosseland_optical_depth <= 0.0)
        or np.any(np.diff(standard_rosseland_optical_depth) <= 0.0)
        or not np.allclose(
            standard_rosseland_optical_depth,
            INITIALIZER_STANDARD_ROSSELAND_OPTICAL_DEPTH,
            rtol=8.0 * np.finfo(np.float64).eps,
            atol=0.0,
        )
    ):
        raise ValueError("atmosphere initializer checkpoint depth grid is incompatible")
    components = int(pca.get("components", 0))
    flattened_size = 80 * len(INITIALIZER_COORDINATE_FIELDS)
    required_shapes = {
        "coordinate_mean": (flattened_size,),
        "coordinate_std": (flattened_size,),
        "basis": (components, flattened_size),
        "coefficient_mean": (components,),
        "coefficient_std": (components,),
    }
    for key, shape in required_shapes.items():
        values = np.asarray(pca.get(key), dtype=np.float64)
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(
                f"atmosphere initializer checkpoint PCA field {key} is invalid"
            )
        if key.endswith("std") and np.any(values <= 0.0):
            raise ValueError(
                f"atmosphere initializer checkpoint PCA field {key} must be positive"
            )
    config = model.get("config", {})
    if int(config.get("input_dim", -1)) != len(checkpoint_feature_fields):
        raise ValueError(
            "atmosphere initializer checkpoint input dimension is incompatible"
        )
    if int(config.get("output_dim", -1)) != components:
        raise ValueError(
            "atmosphere initializer checkpoint output dimension is incompatible"
        )
    if int(config.get("width", 0)) < 1 or int(config.get("hidden_layers", -1)) < 0:
        raise ValueError("atmosphere initializer checkpoint architecture is invalid")
    if not isinstance(model.get("state_dict"), Mapping):
        raise ValueError("atmosphere initializer checkpoint model state is missing")


@lru_cache(maxsize=4)
def _load_atmosphere_initializer_cached(
    checkpoint_path: str,
    device: str,
) -> AtmosphereInitializer:
    import torch

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("atmosphere initializer checkpoint does not contain a mapping")
    return AtmosphereInitializer(checkpoint, device=device)


def load_atmosphere_initializer(
    *,
    checkpoint_path: Path | None = None,
    device: str = "cpu",
) -> AtmosphereInitializer:
    """Load the bundled complete-state initializer."""

    path = Path(checkpoint_path or DEFAULT_FIVE_LABEL_WEIGHTS_PATH).resolve()
    if not path.exists():
        raise FileNotFoundError(f"atmosphere initializer checkpoint not found: {path}")
    return _load_atmosphere_initializer_cached(str(path), device)


def deterministic_initializer_labels(
    *,
    effective_temperature: float,
    log_surface_gravity: float,
    metallicity: float = 0.0,
    alpha_enhancement: float = 0.0,
    microturbulence_km_s: float = 2.0,
    carbon_enhancement: float | None = None,
    nitrogen_enhancement: float | None = None,
    oxygen_enhancement: float | None = None,
    absolute_abundance_offsets: Mapping[int, float] | None = None,
    max_trials: int = 1,
    seed: int = 20260713,
    jitter_scale: float = 0.01,
    checkpoint_path: Path | None = None,
    device: str = "cpu",
) -> tuple[dict[str, float] | None, ...]:
    """Return the exact-label initializer followed by reproducible neighbors.

    Jitter is measured as a fraction of each checkpoint label width.  The
    returned labels are used only to predict layer columns; exact solver
    metadata and chemistry remain the requested target values.
    """

    if int(max_trials) < 1:
        raise ValueError("max_trials must be positive")
    if not np.isfinite(jitter_scale) or float(jitter_scale) < 0.0:
        raise ValueError("jitter_scale must be finite and non-negative")
    if int(max_trials) > 1 and float(jitter_scale) == 0.0:
        raise ValueError("jitter_scale must be positive when retries are enabled")

    family = select_warm_start_family(
        carbon_enhancement=carbon_enhancement,
        nitrogen_enhancement=nitrogen_enhancement,
        oxygen_enhancement=oxygen_enhancement,
        absolute_abundance_offsets=absolute_abundance_offsets,
    )
    cno_labels = (
        resolve_cno8_labels(
            metallicity=metallicity,
            alpha_enhancement=alpha_enhancement,
            carbon_enhancement=carbon_enhancement,
            nitrogen_enhancement=nitrogen_enhancement,
            oxygen_enhancement=oxygen_enhancement,
            absolute_abundance_offsets=absolute_abundance_offsets,
        )
        if family == CNO8_FAMILY
        else {}
    )
    selected_path = checkpoint_path or (
        DEFAULT_CNO8_WEIGHTS_PATH
        if family == CNO8_FAMILY
        else DEFAULT_FIVE_LABEL_WEIGHTS_PATH
    )
    emulator = load_atmosphere_initializer(
        checkpoint_path=selected_path,
        device=device,
    )
    emulator_family = getattr(emulator, "family", family)
    checkpoint_feature_fields = getattr(
        emulator,
        "checkpoint_feature_fields",
        CNO8_CHECKPOINT_FEATURE_FIELDS
        if family == CNO8_FAMILY
        else FIVE_LABEL_CHECKPOINT_FEATURE_FIELDS,
    )
    if emulator_family != family:
        raise ValueError("warm-start checkpoint belongs to the wrong family")
    target = _initializer_checkpoint_features(
        effective_temperature=effective_temperature,
        log_surface_gravity=log_surface_gravity,
        metallicity=metallicity,
        alpha_enhancement=alpha_enhancement,
        microturbulence_km_s=microturbulence_km_s,
        **cno_labels,
        checkpoint_feature_fields=checkpoint_feature_fields,
    )
    bounds = emulator.checkpoint["labels"]["bounds"]
    limits = np.asarray(
        [bounds[field] for field in checkpoint_feature_fields],
        dtype=np.float64,
    )
    lower = limits[:, 0]
    upper = limits[:, 1]
    widths = upper - lower
    projected_target = np.clip(target, lower, upper)
    canonical = json.dumps(
        {
            "physical_label": {
                "effective_temperature": float(effective_temperature),
                "log10_surface_gravity_cgs": float(log_surface_gravity),
                "metallicity": float(metallicity),
                "alpha_enhancement": float(alpha_enhancement),
                "microturbulence_km_s": float(microturbulence_km_s),
                **{
                    checkpoint_field: float(cno_labels[physical_name])
                    for checkpoint_field, physical_name in zip(
                        CNO8_CHECKPOINT_FEATURE_FIELDS[5:],
                        CNO_ENHANCEMENT_FIELDS,
                        strict=True,
                    )
                    if physical_name in cno_labels
                },
                "abundance_offsets": [
                    [int(atomic_number), float(value)]
                    for atomic_number, value in sorted(
                        (absolute_abundance_offsets or {}).items()
                    )
                ],
            },
            "seed": int(seed),
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    def label_from_features(features: np.ndarray) -> dict[str, float]:
        label = {
            "effective_temperature": float(5040.0 / features[0]),
            "log_surface_gravity": float(features[1]),
            "metallicity": float(features[2]),
            "alpha_enhancement": float(features[3]),
            "microturbulence_km_s": float(features[4]),
        }
        if family == CNO8_FAMILY:
            label.update(
                {
                    field: float(features[index])
                    for index, field in enumerate(CNO_ENHANCEMENT_FIELDS, start=5)
                }
            )
        return label

    target_was_projected = not np.array_equal(projected_target, target)
    if target_was_projected:
        changed_fields = [
            field
            for field, requested, projected in zip(
                checkpoint_feature_fields,
                target,
                projected_target,
                strict=True,
            )
            if requested != projected
        ]
        warnings.warn(
            "requested labels exceed initializer support; using the nearest "
            "supported initializer for "
            + ", ".join(changed_fields)
            + " while retaining the exact requested labels in the atmosphere solve",
            RuntimeWarning,
            stacklevel=2,
        )
    initializers: list[dict[str, float] | None] = [
        label_from_features(projected_target) if target_was_projected else None
    ]
    for trial_index in range(1, int(max_trials)):
        direction = []
        for field in checkpoint_feature_fields:
            digest = hashlib.sha256(
                f"{canonical}:{trial_index}:{field}".encode("utf-8")
            ).digest()
            unit = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
            direction.append(2.0 * unit - 1.0)
        features = np.clip(
            projected_target + float(jitter_scale) * widths * np.asarray(direction),
            lower,
            upper,
        )
        initializers.append(label_from_features(features))
    return tuple(initializers)


def emulator_warm_start_model(
    *,
    effective_temperature: float,
    log_surface_gravity: float,
    metallicity: float = 0.0,
    alpha_enhancement: float = 0.0,
    microturbulence_km_s: float = 2.0,
    carbon_enhancement: float | None = None,
    nitrogen_enhancement: float | None = None,
    oxygen_enhancement: float | None = None,
    absolute_abundance_offsets: Mapping[int, float] | None = None,
    device: str = "cpu",
    five_label_path: Path | None = None,
    cno8_path: Path | None = None,
    initializer_label: Mapping[str, float] | None = None,
    title: str | None = None,
) -> tuple[ModelAtmosphere, str]:
    """Predict an emulator warm start and return ``(atmosphere, deck_text)``.

    The returned `ModelAtmosphere` is the primary in-memory product: it is
    obtained by formatting the prediction with `format_warm_start_deck` and
    parsing that text with the shared deck parser, entirely in memory.  This
    format-then-parse step is deliberate, not an inefficiency — the deck
    format's finite digits quantize the prediction, and the certified
    baselines converged through that quantization.  ``deck_text`` is the
    matching provenance deck; writing it to disk and re-reading it yields a
    byte-identical parse.

    ``initializer_label`` may evaluate the emulator at a nearby point for a
    deterministic multi-start solve.  It changes only the predicted layer
    structure.  The returned deck always retains the requested stellar
    labels, abundances, and microturbulence so the exact solve targets the
    requested atmosphere.
    """

    family = select_warm_start_family(
        carbon_enhancement=carbon_enhancement,
        nitrogen_enhancement=nitrogen_enhancement,
        oxygen_enhancement=oxygen_enhancement,
        absolute_abundance_offsets=absolute_abundance_offsets,
    )
    cno_labels = (
        resolve_cno8_labels(
            metallicity=metallicity,
            alpha_enhancement=alpha_enhancement,
            carbon_enhancement=carbon_enhancement,
            nitrogen_enhancement=nitrogen_enhancement,
            oxygen_enhancement=oxygen_enhancement,
            absolute_abundance_offsets=absolute_abundance_offsets,
        )
        if family == CNO8_FAMILY
        else {}
    )
    target_label = {
        "effective_temperature": float(effective_temperature),
        "log_surface_gravity": float(log_surface_gravity),
        "metallicity": float(metallicity),
        "alpha_enhancement": float(alpha_enhancement),
        "microturbulence_km_s": float(microturbulence_km_s),
        **cno_labels,
    }
    prediction_label = dict(target_label)
    if initializer_label is not None:
        unknown = sorted(set(initializer_label) - set(target_label))
        if unknown:
            raise ValueError(
                "initializer_label contains unsupported fields: " + ", ".join(unknown)
            )
        prediction_label.update(
            {key: float(value) for key, value in initializer_label.items()}
        )
    if family == FIVE_LABEL_FAMILY and cno8_path is not None:
        raise ValueError("cno8_path was supplied for a five-label warm start")
    if family == CNO8_FAMILY and five_label_path is not None:
        raise ValueError(
            "five_label_path was supplied for an eight-label CNO-aware warm start"
        )
    selected_path = Path(
        (five_label_path or DEFAULT_FIVE_LABEL_WEIGHTS_PATH)
        if family == FIVE_LABEL_FAMILY
        else (cno8_path or DEFAULT_CNO8_WEIGHTS_PATH)
    )
    if not selected_path.exists():
        raise FileNotFoundError(
            f"{family} emulator checkpoint not found: {selected_path.resolve()}"
        )
    emulator = load_atmosphere_initializer(
        checkpoint_path=selected_path,
        device=device,
    )
    emulator_family = getattr(emulator, "family", family)
    checkpoint_feature_fields = getattr(
        emulator,
        "checkpoint_feature_fields",
        CNO8_CHECKPOINT_FEATURE_FIELDS
        if family == CNO8_FAMILY
        else FIVE_LABEL_CHECKPOINT_FEATURE_FIELDS,
    )
    if emulator_family != family:
        raise ValueError("warm-start checkpoint belongs to the wrong family")
    target_features = _initializer_checkpoint_features(
        **target_label,
        checkpoint_feature_fields=checkpoint_feature_fields,
    )
    prediction_features = _initializer_checkpoint_features(
        **prediction_label,
        checkpoint_feature_fields=checkpoint_feature_fields,
    )
    _require_initializer_bounds(
        prediction_features,
        emulator.checkpoint["labels"]["bounds"],
        checkpoint_feature_fields=checkpoint_feature_fields,
    )
    if initializer_label is None:
        _require_initializer_bounds(
            target_features,
            emulator.checkpoint["labels"]["bounds"],
            checkpoint_feature_fields=checkpoint_feature_fields,
        )
    layer_table = emulator.predict_layer_table(**prediction_label)
    default_title = f"Payne Zero {family.replace('_', '-')} warm start"

    layer_table = np.asarray(layer_table, dtype=np.float64).copy()
    layer_table[:, 6] = target_label["microturbulence_km_s"] * 1.0e5

    bad_temperature = np.any(layer_table[:, 1] <= 0.0) or np.any(
        ~np.isfinite(layer_table[:, 1])
    )
    bad_column_mass = np.any(layer_table[:, 0] <= 0.0) or np.any(
        ~np.isfinite(layer_table[:, 0])
    )
    if bad_temperature or bad_column_mass:
        raise RuntimeError(
            "warm-start emulator produced non-positive or non-finite "
            "temperature or column-mass values"
        )

    target_offsets = dict(absolute_abundance_offsets or {})
    if family == CNO8_FAMILY:
        target_offsets.update(
            cno8_absolute_abundance_offsets(
                metallicity=metallicity,
                cno_labels=cno_labels,
            )
        )

    deck_text = format_warm_start_deck(
        effective_temperature=effective_temperature,
        log_surface_gravity=log_surface_gravity,
        layer_table=layer_table,
        metallicity=metallicity,
        alpha_enhancement=alpha_enhancement,
        absolute_abundance_offsets=target_offsets,
        title=title or default_title,
    )
    return parse_atmosphere_deck(deck_text, source="<emulator warm start>"), deck_text
