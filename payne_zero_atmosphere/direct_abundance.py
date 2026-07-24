"""Experimental 81-abundance warm-start initializer.

This module is deliberately separate from the released five-label and CNO
initializers.  Calling it requires an explicit opt-in.  The public decoded
product is only a provenance-marked deck; the public atmosphere route returns
only after a subsequent exact solve converges.  The decoded profile is not a
converged model atmosphere and must never be used directly for synthesis.

The current checkpoint did not pass the frozen direct-[X/H] release gate.  It
is retained as an experimental starting point because the exact solver remains
the final authority.  Nothing in this module changes automatic warm-start
selection in :mod:`payne_zero_atmosphere.warm_start`.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field, replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Mapping
import warnings

import numpy as np

from .atmosphere_io import ModelAtmosphere, parse_atmosphere_deck
from .config import AtmosphereConfig
from .data_files import atmosphere_emulator_dir
from .warm_start import (
    ELEMENT_SYMBOLS,
    INITIALIZER_OUTPUT_FIELDS,
    INITIALIZER_STANDARD_ROSSELAND_OPTICAL_DEPTH,
    SOLAR_METAL_LOG_ABUNDANCES_3_TO_99,
    atmosphere_prediction_to_layer_table,
    format_warm_start_deck,
)

if TYPE_CHECKING:
    from .runner import AtmosphereRunResult


DIRECT_XH_FAMILY = "direct_abundance"
DIRECT_XH_EXPERIMENTAL_CHECKPOINT_FORMAT = (
    "payne_zero_direct_xh_complete_atmosphere_standalone_experimental_v1"
)
# Retained for the release-packaging reproducibility script.  The current
# preview consumes the experimental format above and is not a release asset.
DIRECT_XH_STANDALONE_CHECKPOINT_FORMAT = (
    "payne_zero_direct_xh_standalone_initializer_v1"
)
DIRECT_XH_ATOMIC_NUMBERS = tuple(
    atomic_number
    for atomic_number in range(3, 100)
    if SOLAR_METAL_LOG_ABUNDANCES_3_TO_99[atomic_number - 3] > -19.0
)
DIRECT_XH_SENTINEL_ATOMIC_NUMBERS = tuple(
    atomic_number
    for atomic_number in range(3, 100)
    if atomic_number not in DIRECT_XH_ATOMIC_NUMBERS
)
DIRECT_XH_FEATURE_FIELDS = (
    "temperature_ratio_5040_k_over_temperature",
    "log_surface_gravity",
    "microturbulence_km_s",
    "iron_abundance_relative_to_hydrogen",
) + tuple(
    f"element_{atomic_number}_abundance_relative_to_iron"
    for atomic_number in DIRECT_XH_ATOMIC_NUMBERS
    if atomic_number != 26
)
DIRECT_XH_SUPPORT = {
    "effective_temperature": (4000.0, 10500.0),
    "log_surface_gravity": (0.7, 5.3),
    "microturbulence_km_s": (0.5, 4.0),
    "iron_abundance_relative_to_hydrogen": (-2.5, 0.5),
    "element_abundance_relative_to_iron": (-0.5, 0.5),
}
DIRECT_XH_ABUNDANCE_QUANTIZATION_DEX = 0.01
DIRECT_XH_REFERENCE_PAYLOAD_SHA256 = (
    "60920eccff0ecfb40a38114905e2ba24b67bdcdab8700c46ee1e673325a6d3f5"
)
DIRECT_XH_EXPECTED_CHECKPOINT_SHA256 = (
    "1b8e1db1514956dfbf890eb5ae96e01bd918acfc86be538b6b77230332104243"
)
DIRECT_XH_EXPECTED_MANIFEST_SHA256 = (
    "fb59da5e6bd3f8fcba06e0c4c284137e90aab5c4e93165daa74d8ce2ae268710"
)
DEFAULT_DIRECT_XH_ASSET_DIR = atmosphere_emulator_dir() / DIRECT_XH_FAMILY
DEFAULT_DIRECT_XH_CHECKPOINT_PATH = DEFAULT_DIRECT_XH_ASSET_DIR / "checkpoint.pt"
DEFAULT_DIRECT_XH_MANIFEST_PATH = DEFAULT_DIRECT_XH_ASSET_DIR / "manifest.json"

EXPERIMENTAL_DIRECT_XH_WARNING = (
    "experimental direct-[X/H] initializer: the decoded profile is only a "
    "starting structure; a converged exact atmosphere solve is mandatory"
)
EXPERIMENTAL_DIRECT_XH_OPTIMIZER_SURROGATE_WARNING = (
    "experimental direct-[X/H] optimizer surrogate: this decoded structure is "
    "not a converged atmosphere and may be used only inside an optimizer; a "
    "converged exact atmosphere solve is mandatory before reporting a result"
)


@dataclass(frozen=True)
class DirectAbundanceCheckpointProvenance:
    """Identity of the checkpoint used to build one starting structure."""

    path: Path
    sha256: str
    manifest_path: Path
    manifest_sha256: str
    release_gate_passed: bool


@dataclass(frozen=True)
class DirectAbundanceOptimizerSurrogate:
    """Explicit optimizer-only decoded structure with immutable safety flags.

    This product exists only for abundance-search acceleration.  It is not a
    model-atmosphere result, and its fixed ``exact_closure_required`` and
    ``is_final_atmosphere`` fields prevent callers from relabeling it as one.
    """

    optimizer_atmosphere: ModelAtmosphere = dataclass_field(repr=False)
    effective_temperature: float
    log_surface_gravity: float
    microturbulence_km_s: float
    realized_abundance_vector: np.ndarray = dataclass_field(repr=False)
    realized_mixture_sha256: str
    deck_sha256: str
    surrogate_identity_sha256: str
    checkpoint: DirectAbundanceCheckpointProvenance
    role: str = dataclass_field(
        default="experimental_direct_xh_optimizer_surrogate", init=False
    )
    exact_closure_required: bool = dataclass_field(default=True, init=False)
    is_final_atmosphere: bool = dataclass_field(default=False, init=False)

    def __post_init__(self) -> None:
        abundance = np.asarray(self.realized_abundance_vector, np.float64).copy()
        if abundance.shape != (97,) or not np.all(np.isfinite(abundance)):
            raise ValueError(
                "optimizer surrogate realized abundance vector must contain "
                "97 finite values"
            )
        abundance.setflags(write=False)
        object.__setattr__(self, "realized_abundance_vector", abundance)
        observed_mixture_sha256 = direct_abundance_mixture_sha256(abundance)
        if self.realized_mixture_sha256 != observed_mixture_sha256:
            raise ValueError("optimizer surrogate realized-mixture identity is invalid")
        atmosphere_realized = np.asarray(
            [
                self.optimizer_atmosphere.fixed_column_abundance_values[z]
                - SOLAR_METAL_LOG_ABUNDANCES_3_TO_99[z - 3]
                for z in range(3, 100)
            ],
            np.float64,
        )
        if not np.allclose(atmosphere_realized, abundance, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "optimizer surrogate atmosphere and realized mixture differ"
            )
        expected_identity = _canonical_payload_sha256(
            {
                "schema": 1,
                "role": self.role,
                "exact_closure_required": True,
                "is_final_atmosphere": False,
                "effective_temperature": float(self.effective_temperature),
                "log_surface_gravity": float(self.log_surface_gravity),
                "microturbulence_km_s": float(self.microturbulence_km_s),
                "realized_mixture_sha256": self.realized_mixture_sha256,
                "deck_sha256": self.deck_sha256,
                "checkpoint_sha256": self.checkpoint.sha256,
                "manifest_sha256": self.checkpoint.manifest_sha256,
                "checkpoint_release_gate_passed": self.checkpoint.release_gate_passed,
            }
        )
        if self.surrogate_identity_sha256 != expected_identity:
            raise ValueError("optimizer surrogate provenance identity is invalid")

    def provenance(self) -> dict[str, object]:
        """Return the JSON-ready identity required beside optimizer results."""

        return {
            "role": self.role,
            "exact_closure_required": self.exact_closure_required,
            "is_final_atmosphere": self.is_final_atmosphere,
            "effective_temperature": self.effective_temperature,
            "log_surface_gravity": self.log_surface_gravity,
            "microturbulence_km_s": self.microturbulence_km_s,
            "realized_mixture_sha256": self.realized_mixture_sha256,
            "deck_sha256": self.deck_sha256,
            "surrogate_identity_sha256": self.surrogate_identity_sha256,
            "checkpoint_path": str(self.checkpoint.path),
            "checkpoint_sha256": self.checkpoint.sha256,
            "manifest_path": str(self.checkpoint.manifest_path),
            "manifest_sha256": self.checkpoint.manifest_sha256,
            "checkpoint_release_gate_passed": self.checkpoint.release_gate_passed,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_payload_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def direct_abundance_mixture_sha256(abundance_vector: np.ndarray) -> str:
    """Return the canonical identity of a realized quantized 97-slot mixture."""

    values = np.asarray(abundance_vector, np.float64)
    if values.shape != (97,) or not np.all(np.isfinite(values)):
        raise ValueError("realized direct-[X/H] mixture must contain 97 finite values")
    values = np.asarray(
        np.round(values / DIRECT_XH_ABUNDANCE_QUANTIZATION_DEX)
        * DIRECT_XH_ABUNDANCE_QUANTIZATION_DEX,
        dtype="<f8",
    )
    # IEEE-754 signed zero is physically and numerically the same abundance.
    # Text deck round-trips may change ``-0.00`` to ``+0.00`` (or conversely),
    # so canonicalize the sign before computing the provenance identity.
    values[values == 0.0] = 0.0
    digest = hashlib.sha256()
    digest.update(b"payne-zero-direct-xh-realized-mixture-v1\0")
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _require_opt_in(enable_experimental: bool) -> None:
    if enable_experimental is not True:
        raise RuntimeError(
            "direct-[X/H] initialization is experimental; pass "
            "enable_experimental=True explicitly"
        )


def _require_optimizer_surrogate_opt_in(enable: bool) -> None:
    if enable is not True:
        raise RuntimeError(
            "direct-[X/H] optimizer-surrogate use is experimental; pass "
            "enable_experimental_optimizer_surrogate=True explicitly"
        )


def _finite_scalar(name: str, value: float) -> float:
    converted = float(value)
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _within(name: str, value: float, bounds: tuple[float, float]) -> None:
    if not bounds[0] <= value <= bounds[1]:
        raise ValueError(f"{name}={value:g} is outside {bounds}")


def complete_direct_abundance_vector(
    abundance_by_atomic_number: Mapping[int, float],
) -> np.ndarray:
    """Return the quantized 97-slot solver vector from 81 public ``[X/H]`` labels.

    Every finite-solar-reference abundance is required.  The remaining 16
    solver slots have no public solar reference and inherit ``[Fe/H]``.  This
    matches the frozen training and solver-handoff contract; there is no
    projection, implicit solar default, or lower-dimensional fallback.
    """

    values: dict[int, float] = {}
    for key, value in abundance_by_atomic_number.items():
        if isinstance(key, (bool, np.bool_)):
            raise ValueError("abundance keys must be atomic numbers")
        try:
            atomic_number = int(key)
        except (TypeError, ValueError) as error:
            raise ValueError("abundance keys must be atomic numbers") from error
        if atomic_number != key:
            raise ValueError(f"abundance key {key!r} is not an integer atomic number")
        values[atomic_number] = _finite_scalar(f"[Z={atomic_number}/H]", value)

    public = set(DIRECT_XH_ATOMIC_NUMBERS)
    unknown = sorted(set(values) - public)
    missing = sorted(public - set(values))
    if unknown:
        raise ValueError(f"unsupported public abundance atomic numbers: {unknown}")
    if missing:
        symbols = ", ".join(ELEMENT_SYMBOLS[z] for z in missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise ValueError(
            "direct-[X/H] input requires all 81 public abundances; missing "
            f"{symbols}{suffix}"
        )

    iron = values[26]
    vector = np.full(97, iron, dtype=np.float64)
    for atomic_number, value in values.items():
        vector[atomic_number - 3] = value
    vector = (
        np.round(vector / DIRECT_XH_ABUNDANCE_QUANTIZATION_DEX)
        * DIRECT_XH_ABUNDANCE_QUANTIZATION_DEX
    )
    # A quantized abundance of zero has one physical representation.  Normalize
    # IEEE-754 signed zero here so every downstream consumer, including legacy
    # provenance validators, sees the same serialized 97-slot mixture.
    vector[vector == 0.0] = 0.0
    iron = float(vector[26 - 3])
    _within(
        "[Fe/H]",
        iron,
        DIRECT_XH_SUPPORT["iron_abundance_relative_to_hydrogen"],
    )
    # Both the element and iron abundances lie on the frozen centidex lattice.
    # Compare their difference in integer lattice units: subtracting their
    # binary floating-point representations can otherwise turn an inclusive
    # +0.50 boundary into 0.5000000000000001 and reject a valid mixture.
    relative_units = np.rint(
        (vector - iron) / DIRECT_XH_ABUNDANCE_QUANTIZATION_DEX
    ).astype(np.int64)
    lower, upper = DIRECT_XH_SUPPORT["element_abundance_relative_to_iron"]
    lower_units = int(np.rint(lower / DIRECT_XH_ABUNDANCE_QUANTIZATION_DEX))
    upper_units = int(np.rint(upper / DIRECT_XH_ABUNDANCE_QUANTIZATION_DEX))
    bad = np.flatnonzero(
        (relative_units < lower_units) | (relative_units > upper_units)
    )
    if bad.size:
        symbols = ", ".join(ELEMENT_SYMBOLS[index + 3] for index in bad[:8])
        suffix = " ..." if bad.size > 8 else ""
        raise ValueError(f"[X/Fe] is outside {(lower, upper)} for {symbols}{suffix}")
    return vector


def retained_direct_abundance_mixture(
    *,
    iron_abundance_relative_to_hydrogen: float,
    retained_abundance_relative_to_iron_by_atomic_number: Mapping[int, float],
) -> tuple[dict[int, float], np.ndarray]:
    """Map retained coordinates to public 81-slot and realized 97-slot ``[X/H]``.

    All unselected public metals inherit ``[Fe/H]``.  A retained non-iron
    coordinate supplies ``[X/H] = [Fe/H] + [X/Fe]``.  The returned mapping is
    the complete 81-label public initializer input; the returned vector is the
    authoritative 0.01-dex-quantized solver/synthesis mixture.  Iron is a
    separate coordinate and therefore must not appear in the retained mapping.
    """

    iron = _finite_scalar(
        "iron_abundance_relative_to_hydrogen",
        iron_abundance_relative_to_hydrogen,
    )
    retained: dict[int, float] = {}
    for key, value in retained_abundance_relative_to_iron_by_atomic_number.items():
        if isinstance(key, (bool, np.bool_)):
            raise ValueError("retained abundance keys must be atomic numbers")
        try:
            atomic_number = int(key)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "retained abundance keys must be atomic numbers"
            ) from error
        if atomic_number != key:
            raise ValueError(
                f"retained abundance key {key!r} is not an integer atomic number"
            )
        if atomic_number == 26:
            raise ValueError("iron is supplied through the separate [Fe/H] coordinate")
        if atomic_number not in DIRECT_XH_ATOMIC_NUMBERS:
            raise ValueError(
                f"Z={atomic_number} is not a public direct-[X/H] abundance label"
            )
        retained[atomic_number] = _finite_scalar(
            f"[Z={atomic_number}/Fe]", value
        )

    public_xh = {atomic_number: iron for atomic_number in DIRECT_XH_ATOMIC_NUMBERS}
    for atomic_number, relative in retained.items():
        public_xh[atomic_number] = iron + relative
    realized = complete_direct_abundance_vector(public_xh)
    return public_xh, realized


def _feature_vector(
    *,
    effective_temperature: float,
    log_surface_gravity: float,
    microturbulence_km_s: float,
    abundance_vector: np.ndarray,
) -> np.ndarray:
    temperature = _finite_scalar("effective_temperature", effective_temperature)
    gravity = _finite_scalar("log_surface_gravity", log_surface_gravity)
    microturbulence = _finite_scalar(
        "microturbulence_km_s", microturbulence_km_s
    )
    _within("effective_temperature", temperature, DIRECT_XH_SUPPORT["effective_temperature"])
    _within("log_surface_gravity", gravity, DIRECT_XH_SUPPORT["log_surface_gravity"])
    _within(
        "microturbulence_km_s",
        microturbulence,
        DIRECT_XH_SUPPORT["microturbulence_km_s"],
    )
    iron = float(abundance_vector[26 - 3])
    non_iron_indices = np.asarray(
        [atomic_number - 3 for atomic_number in DIRECT_XH_ATOMIC_NUMBERS if atomic_number != 26],
        dtype=np.int64,
    )
    features = np.concatenate(
        (
            np.asarray([5040.0 / temperature, gravity, microturbulence, iron]),
            abundance_vector[non_iron_indices] - iron,
        )
    )
    if features.shape != (84,) or not np.all(np.isfinite(features)):
        raise RuntimeError("direct-[X/H] feature construction violated the 84-input contract")
    return features


def _set_encoded_model(config: Mapping[str, object]):
    import torch
    import torch.nn as nn

    input_dim = int(config["input_dim"])
    output_dim = int(config["output_dim"])
    width = int(config["width"])
    hidden_layers = int(config["hidden_layers"])
    latent = int(config["abundance_latent_dim"])

    class SetEncodedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            state_dim = 4
            element_count = input_dim - state_dim
            embedding_dim = min(64, max(16, latent // 2))
            token_width = min(384, max(128, width // 8))
            self.element_embedding = nn.Parameter(
                torch.empty(element_count, embedding_dim)
            )
            nn.init.normal_(self.element_embedding, mean=0.0, std=0.02)
            self.response_law = nn.Sequential(
                nn.Linear(state_dim + embedding_dim + 2, token_width),
                nn.SiLU(),
                nn.Linear(token_width, 2 * latent),
            )
            layers: list[nn.Module] = []
            dimension = state_dim + latent
            for _ in range(hidden_layers):
                layers.extend((nn.Linear(dimension, width), nn.SiLU()))
                dimension = width
            layers.append(nn.Linear(dimension, output_dim))
            self.decoder = nn.Sequential(*layers)

        def forward(self, features):
            state = features[:, :4]
            relative = features[:, 4:]
            batch_size, element_count = relative.shape
            expanded_state = state[:, None, :].expand(batch_size, element_count, 4)
            embedding = self.element_embedding[None, :, :].expand(
                batch_size, element_count, self.element_embedding.shape[1]
            )
            amplitude = relative[:, :, None]
            token_input = torch.cat(
                (expanded_state, embedding, amplitude, amplitude.square()), dim=2
            )
            law = self.response_law(token_input).reshape(
                batch_size, element_count, 2, latent
            )
            response = amplitude * law[:, :, 0, :] + amplitude.square() * law[:, :, 1, :]
            return self.decoder(torch.cat((state, response.sum(dim=1)), dim=1))

    return SetEncodedModel()


def _validate_checkpoint(checkpoint: Mapping[str, object]) -> None:
    if (
        checkpoint.get("format") != DIRECT_XH_EXPERIMENTAL_CHECKPOINT_FORMAT
        or checkpoint.get("schema_version") != 1
        or checkpoint.get("family") != "explicit_xh_standalone"
    ):
        raise ValueError("direct-[X/H] checkpoint format, schema, or family is incompatible")
    labels = checkpoint.get("labels", {})
    if not isinstance(labels, Mapping):
        raise ValueError("direct-[X/H] checkpoint labels are missing")
    if tuple(labels.get("feature_fields", ())) != DIRECT_XH_FEATURE_FIELDS:
        raise ValueError("direct-[X/H] checkpoint changed the frozen 84-feature order")
    if tuple(labels.get("atomic_numbers", ())) != tuple(range(3, 100)):
        raise ValueError("direct-[X/H] checkpoint changed the 97-slot solver order")
    if tuple(labels.get("model_atomic_numbers", ())) != DIRECT_XH_ATOMIC_NUMBERS:
        raise ValueError("direct-[X/H] checkpoint changed the 81 public abundances")
    if labels.get("support") != DIRECT_XH_SUPPORT:
        raise ValueError("direct-[X/H] checkpoint changed the frozen support")
    mean = np.asarray(labels.get("feature_mean"), dtype=np.float64)
    std = np.asarray(labels.get("feature_std"), dtype=np.float64)
    if (
        mean.shape != (84,)
        or std.shape != (84,)
        or not np.all(np.isfinite(mean))
        or np.any(~np.isfinite(std) | (std <= 0.0))
    ):
        raise ValueError("direct-[X/H] checkpoint feature scaling is invalid")

    model = checkpoint.get("model", {})
    if not isinstance(model, Mapping):
        raise ValueError("direct-[X/H] checkpoint model is missing")
    config = model.get("config", {})
    if not isinstance(config, Mapping):
        raise ValueError("direct-[X/H] checkpoint model config is missing")
    if (
        config.get("architecture") != "set_encoded"
        or int(config.get("input_dim", -1)) != 84
        or int(config.get("output_dim", -1)) != 160
        or int(config.get("width", -1)) != 2048
        or int(config.get("hidden_layers", -1)) != 6
        or int(config.get("abundance_latent_dim", -1)) != 128
    ):
        raise ValueError("direct-[X/H] checkpoint architecture is incompatible")
    if not isinstance(model.get("state_dict"), Mapping):
        raise ValueError("direct-[X/H] checkpoint model state is missing")

    coordinates = checkpoint.get("coordinates", {})
    if tuple(coordinates.get("fields", ())) != INITIALIZER_OUTPUT_FIELDS:
        raise ValueError("direct-[X/H] checkpoint coordinate fields are incompatible")
    depth = np.asarray(
        coordinates.get("standard_rosseland_optical_depth"), dtype=np.float64
    )
    if (
        depth.shape != INITIALIZER_STANDARD_ROSSELAND_OPTICAL_DEPTH.shape
        or not np.allclose(
            depth,
            INITIALIZER_STANDARD_ROSSELAND_OPTICAL_DEPTH,
            rtol=8.0 * np.finfo(np.float64).eps,
            atol=0.0,
        )
    ):
        raise ValueError("direct-[X/H] checkpoint depth grid is incompatible")
    acceleration_scale = float(coordinates.get("acceleration_scale", np.nan))
    if not np.isfinite(acceleration_scale) or acceleration_scale <= 0.0:
        raise ValueError("direct-[X/H] acceleration scale is invalid")

    pca = checkpoint.get("pca", {})
    shapes = {
        "coordinate_mean": (480,),
        "coordinate_std": (480,),
        "basis": (160, 480),
        "coefficient_mean": (160,),
        "coefficient_std": (160,),
    }
    for field, shape in shapes.items():
        values = np.asarray(pca.get(field), dtype=np.float64)
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(f"direct-[X/H] PCA field {field} is invalid")
        if field.endswith("std") and np.any(values <= 0.0):
            raise ValueError(f"direct-[X/H] PCA field {field} must be positive")


def _checkpoint_provenance(
    checkpoint_path: Path, manifest_path: Path
) -> DirectAbundanceCheckpointProvenance:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "experimental direct-[X/H] checkpoint not found: "
            f"{checkpoint_path}; install or provide the hash-verified experimental asset"
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"experimental direct-[X/H] manifest not found: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "experimental direct-[X/H] manifest is unreadable or invalid JSON"
        ) from error
    if not isinstance(manifest, Mapping):
        raise ValueError("experimental direct-[X/H] manifest must contain an object")
    if manifest.get("format") != "payne_zero_direct_xh_experimental_asset_v1":
        raise ValueError("experimental direct-[X/H] manifest format is incompatible")
    manifest_sha256 = _sha256(manifest_path)
    if manifest_sha256 != DIRECT_XH_EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            "experimental direct-[X/H] manifest hash does not match its frozen provenance"
        )
    training = manifest.get("training", {})
    split_total = sum(
        int(training.get(field, -1000000))
        for field in (
            "train_records",
            "fit_validation_records",
            "internal_check_records_unused_for_fit",
            "unused_external_gate_records",
        )
    )
    unused_external = manifest.get("unused_external_gate_rows", {})
    standalone_external = manifest.get("standalone_external_panel", {})
    if (
        manifest.get("status") != "experimental_opt_in_only"
        or manifest.get("automatic_dispatch") is not False
        or manifest.get("decoded_profile_is_final_atmosphere") is not False
        or manifest.get("exact_solver_after_decode_is_mandatory") is not True
        or manifest.get("release_gate", {}).get("passed") is not False
        or training.get("independent_direct_xh_truth_only") is not True
        or split_total != int(training.get("total_records", -1))
        or any(
            int(training.get(field, -1)) != 0
            for field in (
                "five_label_records",
                "cno8_records",
                "matched_cno_records",
                "baseline_control_records",
            )
        )
        or int(unused_external.get("records", -1))
        != int(training.get("unused_external_gate_records", -2))
        or unused_external.get("used_for_fit") is not False
        or unused_external.get("identity_manifest_packaged") is not False
        or standalone_external.get("status") != "reserved_unmaterialized"
        or int(standalone_external.get("reserved_parent_groups", -1)) != 24
        or standalone_external.get("identity_manifest_packaged") is not False
        or standalone_external.get("evaluated") is not False
    ):
        raise RuntimeError(
            "experimental direct-[X/H] manifest changed its frozen safety contract"
        )
    record = manifest.get("checkpoint", {})
    if not isinstance(record, Mapping):
        raise ValueError("experimental direct-[X/H] checkpoint record is missing")
    expected_record = {
        "path": "checkpoint.pt",
        "format": DIRECT_XH_EXPERIMENTAL_CHECKPOINT_FORMAT,
        "schema_version": 1,
        "architecture": "set_encoded",
        "input_dim": 84,
        "output_dim": 160,
        "width": 2048,
        "hidden_layers": 6,
        "abundance_latent_dim": 128,
    }
    if any(record.get(key) != value for key, value in expected_record.items()):
        raise ValueError(
            "experimental direct-[X/H] manifest checkpoint schema is incompatible"
        )
    expected = record.get("sha256")
    if expected != DIRECT_XH_EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "experimental direct-[X/H] checkpoint hash does not match its frozen provenance"
        )
    if int(record.get("bytes", -1)) != checkpoint_path.stat().st_size:
        raise RuntimeError("experimental direct-[X/H] checkpoint size changed")
    observed = _sha256(checkpoint_path)
    if observed != expected:
        raise RuntimeError(
            "experimental direct-[X/H] checkpoint hash does not match its frozen provenance"
        )
    return DirectAbundanceCheckpointProvenance(
        path=checkpoint_path,
        sha256=observed,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        release_gate_passed=bool(manifest.get("release_gate", {}).get("passed")),
    )


class DirectAbundanceInitializer:
    """Hash-bound experimental 84-input initializer."""

    def __init__(
        self,
        checkpoint: Mapping[str, object],
        *,
        device: str,
        provenance: DirectAbundanceCheckpointProvenance,
    ) -> None:
        _validate_checkpoint(checkpoint)
        self.checkpoint = checkpoint
        self.device = device
        self.provenance = provenance
        self.model = _set_encoded_model(checkpoint["model"]["config"]).to(device)
        self.model.load_state_dict(checkpoint["model"]["state_dict"], strict=True)
        self.model.eval()

    def predict(
        self,
        *,
        effective_temperature: float,
        log_surface_gravity: float,
        microturbulence_km_s: float,
        abundance_by_atomic_number: Mapping[int, float],
    ) -> dict[str, np.ndarray]:
        """Decode a starting profile; this is not a model atmosphere product."""

        import torch

        abundance = complete_direct_abundance_vector(abundance_by_atomic_number)
        feature = _feature_vector(
            effective_temperature=effective_temperature,
            log_surface_gravity=log_surface_gravity,
            microturbulence_km_s=microturbulence_km_s,
            abundance_vector=abundance,
        )
        labels = self.checkpoint["labels"]
        standardized = (feature - np.asarray(labels["feature_mean"])) / np.asarray(
            labels["feature_std"]
        )
        with torch.no_grad():
            prediction = (
                self.model(
                    torch.as_tensor(
                        standardized[None, :], dtype=torch.float32, device=self.device
                    )
                )
                .detach()
                .cpu()
                .numpy()[0]
            )
        pca = self.checkpoint["pca"]
        coefficient = prediction * np.asarray(pca["coefficient_std"]) + np.asarray(
            pca["coefficient_mean"]
        )
        coordinate = (coefficient @ np.asarray(pca["basis"])) * np.asarray(
            pca["coordinate_std"]
        ) + np.asarray(pca["coordinate_mean"])
        coordinate = coordinate.reshape(80, 6)
        depth = np.asarray(
            self.checkpoint["coordinates"]["standard_rosseland_optical_depth"]
        )
        grey = float(effective_temperature) * (0.75 * (depth + 2.0 / 3.0)) ** 0.25
        profile = np.empty((80, 6), dtype=np.float64)
        profile[:, 0] = np.cumsum(10.0 ** np.clip(coordinate[:, 0], -30.0, 30.0))
        profile[:, 1] = grey * 10.0 ** np.clip(coordinate[:, 1], -3.0, 3.0)
        profile[:, 2:5] = 10.0 ** np.clip(coordinate[:, 2:5], -30.0, 30.0)
        profile[:, 5] = float(
            self.checkpoint["coordinates"]["acceleration_scale"]
        ) * np.sinh(np.clip(coordinate[:, 5], -20.0, 20.0))
        if (
            not np.all(np.isfinite(profile))
            or np.any(profile[:, :5] <= 0.0)
            or np.any(np.diff(profile[:, 0]) <= 0.0)
        ):
            raise RuntimeError("direct-[X/H] initializer decoded a nonphysical start")
        return {
            field: profile[:, index]
            for index, field in enumerate(INITIALIZER_OUTPUT_FIELDS)
        }


@lru_cache(maxsize=4)
def _load_direct_abundance_initializer_cached(
    checkpoint_path: str, manifest_path: str, device: str
) -> DirectAbundanceInitializer:
    import torch

    checkpoint_file = Path(checkpoint_path)
    manifest_file = Path(manifest_path)
    provenance = _checkpoint_provenance(checkpoint_file, manifest_file)
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("direct-[X/H] checkpoint does not contain a mapping")
    return DirectAbundanceInitializer(
        checkpoint, device=device, provenance=provenance
    )


def load_direct_abundance_initializer(
    *,
    enable_experimental: bool = False,
    checkpoint_path: Path | str | None = None,
    manifest_path: Path | str | None = None,
    device: str = "cpu",
) -> DirectAbundanceInitializer:
    """Load the experimental initializer after an explicit opt-in and hash gate."""

    _require_opt_in(enable_experimental)
    checkpoint = Path(checkpoint_path or DEFAULT_DIRECT_XH_CHECKPOINT_PATH).resolve()
    manifest = Path(manifest_path or DEFAULT_DIRECT_XH_MANIFEST_PATH).resolve()
    warnings.warn(EXPERIMENTAL_DIRECT_XH_WARNING, RuntimeWarning, stacklevel=2)
    return _load_direct_abundance_initializer_cached(
        str(checkpoint), str(manifest), str(device)
    )


def _decode_direct_abundance_start(
    *,
    initializer: DirectAbundanceInitializer,
    effective_temperature: float,
    log_surface_gravity: float,
    microturbulence_km_s: float,
    abundance_vector: np.ndarray,
    title: str,
    source: str,
) -> tuple[ModelAtmosphere, str]:
    """Decode and deck-quantize one experimental starting structure."""

    abundance = np.asarray(abundance_vector, np.float64)
    if abundance.shape != (97,) or not np.all(np.isfinite(abundance)):
        raise ValueError("direct-[X/H] start requires a complete realized mixture")
    public_xh = {
        atomic_number: float(abundance[atomic_number - 3])
        for atomic_number in DIRECT_XH_ATOMIC_NUMBERS
    }
    prediction = initializer.predict(
        effective_temperature=effective_temperature,
        log_surface_gravity=log_surface_gravity,
        microturbulence_km_s=microturbulence_km_s,
        abundance_by_atomic_number=public_xh,
    )
    layer_table = atmosphere_prediction_to_layer_table(
        prediction, microturbulence_km_s=microturbulence_km_s
    )
    offsets = {
        atomic_number: float(abundance[atomic_number - 3])
        for atomic_number in range(3, 100)
    }
    deck = format_warm_start_deck(
        effective_temperature=effective_temperature,
        log_surface_gravity=log_surface_gravity,
        layer_table=layer_table,
        metallicity=0.0,
        alpha_enhancement=0.0,
        absolute_abundance_offsets=offsets,
        title=title,
    )
    atmosphere = parse_atmosphere_deck(deck, source=source)
    realized = np.asarray(
        [
            atmosphere.fixed_column_abundance_values[z]
            - SOLAR_METAL_LOG_ABUNDANCES_3_TO_99[z - 3]
            for z in range(3, 100)
        ]
    )
    if not np.allclose(realized, abundance, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("direct-[X/H] deck changed the requested quantized mixture")
    return atmosphere, deck


def _direct_abundance_warm_start_model(
    *,
    effective_temperature: float,
    log_surface_gravity: float,
    microturbulence_km_s: float,
    abundance_by_atomic_number: Mapping[int, float],
    enable_experimental: bool = False,
    checkpoint_path: Path | str | None = None,
    manifest_path: Path | str | None = None,
    device: str = "cpu",
) -> tuple[ModelAtmosphere, str]:
    """Build an internal experimental starting structure and quantized deck.

    This private boundary exists so the public API cannot return a decoded
    ``ModelAtmosphere`` before a converged exact solve.  Public callers may
    request the deck alone or use :func:`run_direct_abundance_atmosphere`.
    """

    initializer = load_direct_abundance_initializer(
        enable_experimental=enable_experimental,
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        device=device,
    )
    abundance = complete_direct_abundance_vector(abundance_by_atomic_number)
    return _decode_direct_abundance_start(
        initializer=initializer,
        effective_temperature=effective_temperature,
        log_surface_gravity=log_surface_gravity,
        microturbulence_km_s=microturbulence_km_s,
        abundance_vector=abundance,
        title="EXPERIMENTAL direct-XH start; exact solve required",
        source="experimental-direct-xh-warm-start",
    )


@lru_cache(maxsize=64)
def _build_direct_abundance_optimizer_surrogate_cached(
    checkpoint_path: str,
    manifest_path: str,
    device: str,
    effective_temperature: float,
    log_surface_gravity: float,
    microturbulence_km_s: float,
    realized_abundance_values: tuple[float, ...],
) -> DirectAbundanceOptimizerSurrogate:
    initializer = _load_direct_abundance_initializer_cached(
        checkpoint_path, manifest_path, device
    )
    realized = np.asarray(realized_abundance_values, np.float64)
    atmosphere, deck = _decode_direct_abundance_start(
        initializer=initializer,
        effective_temperature=effective_temperature,
        log_surface_gravity=log_surface_gravity,
        microturbulence_km_s=microturbulence_km_s,
        abundance_vector=realized,
        title=(
            "EXPERIMENTAL direct-XH optimizer surrogate; exact closure required"
        ),
        source="experimental-direct-xh-optimizer-surrogate",
    )
    mixture_sha256 = direct_abundance_mixture_sha256(realized)
    deck_sha256 = hashlib.sha256(deck.encode("utf-8")).hexdigest()
    identity_sha256 = _canonical_payload_sha256(
        {
            "schema": 1,
            "role": "experimental_direct_xh_optimizer_surrogate",
            "exact_closure_required": True,
            "is_final_atmosphere": False,
            "effective_temperature": float(effective_temperature),
            "log_surface_gravity": float(log_surface_gravity),
            "microturbulence_km_s": float(microturbulence_km_s),
            "realized_mixture_sha256": mixture_sha256,
            "deck_sha256": deck_sha256,
            "checkpoint_sha256": initializer.provenance.sha256,
            "manifest_sha256": initializer.provenance.manifest_sha256,
            "checkpoint_release_gate_passed": (
                initializer.provenance.release_gate_passed
            ),
        }
    )
    return DirectAbundanceOptimizerSurrogate(
        optimizer_atmosphere=atmosphere,
        effective_temperature=float(effective_temperature),
        log_surface_gravity=float(log_surface_gravity),
        microturbulence_km_s=float(microturbulence_km_s),
        realized_abundance_vector=realized,
        realized_mixture_sha256=mixture_sha256,
        deck_sha256=deck_sha256,
        surrogate_identity_sha256=identity_sha256,
        checkpoint=initializer.provenance,
    )


def build_direct_abundance_optimizer_surrogate(
    *,
    effective_temperature: float,
    log_surface_gravity: float,
    microturbulence_km_s: float,
    abundance_by_atomic_number: Mapping[int, float],
    enable_experimental_optimizer_surrogate: bool = False,
    checkpoint_path: Path | str | None = None,
    manifest_path: Path | str | None = None,
    device: str = "cpu",
) -> DirectAbundanceOptimizerSurrogate:
    """Return an explicit decoded optimizer surrogate, never a final atmosphere.

    The complete public 81-label input is validated and quantized before it is
    used as the cache key.  Consequently, repeated optimizer coordinates that
    realize the same stellar state and 97-slot mixture reuse one decoded
    structure.  Every consumer must retain the product's mandatory exact-
    closure provenance and run a converged exact atmosphere at the selected
    mixture before reporting a physical result.
    """

    _require_optimizer_surrogate_opt_in(enable_experimental_optimizer_surrogate)
    checkpoint = Path(checkpoint_path or DEFAULT_DIRECT_XH_CHECKPOINT_PATH).resolve()
    manifest = Path(manifest_path or DEFAULT_DIRECT_XH_MANIFEST_PATH).resolve()
    warnings.warn(
        EXPERIMENTAL_DIRECT_XH_OPTIMIZER_SURROGATE_WARNING,
        RuntimeWarning,
        stacklevel=2,
    )
    realized = complete_direct_abundance_vector(abundance_by_atomic_number)
    return _build_direct_abundance_optimizer_surrogate_cached(
        str(checkpoint),
        str(manifest),
        str(device),
        _finite_scalar("effective_temperature", effective_temperature),
        _finite_scalar("log_surface_gravity", log_surface_gravity),
        _finite_scalar("microturbulence_km_s", microturbulence_km_s),
        tuple(float(value) for value in realized),
    )


def direct_abundance_warm_start_deck(
    *,
    effective_temperature: float,
    log_surface_gravity: float,
    microturbulence_km_s: float,
    abundance_by_atomic_number: Mapping[int, float],
    enable_experimental: bool = False,
    checkpoint_path: Path | str | None = None,
    manifest_path: Path | str | None = None,
    device: str = "cpu",
) -> str:
    """Return only the provenance-marked deck for an experimental exact start.

    The deck is not a converged atmosphere product.  This function deliberately
    does not expose its parsed ``ModelAtmosphere``; use
    :func:`run_direct_abundance_atmosphere` to obtain an atmosphere result.
    """

    _atmosphere, deck = _direct_abundance_warm_start_model(
        effective_temperature=effective_temperature,
        log_surface_gravity=log_surface_gravity,
        microturbulence_km_s=microturbulence_km_s,
        abundance_by_atomic_number=abundance_by_atomic_number,
        enable_experimental=enable_experimental,
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        device=device,
    )
    return deck


def run_direct_abundance_atmosphere(
    *,
    exact_config: AtmosphereConfig,
    effective_temperature: float,
    log_surface_gravity: float,
    microturbulence_km_s: float,
    abundance_by_atomic_number: Mapping[int, float],
    enable_experimental: bool = False,
    checkpoint_path: Path | str | None = None,
    manifest_path: Path | str | None = None,
    device: str = "cpu",
) -> AtmosphereRunResult:
    """Return an atmosphere only after the exact solver reports convergence.

    ``exact_config.inputs.initial_atmosphere`` is replaced by the decoded
    direct-abundance start.  All catalog, physics, convergence, and output
    settings otherwise come from ``exact_config``.  A terminal nonconverged
    state is never returned through this experimental public API.
    """

    _require_opt_in(enable_experimental)
    if not exact_config.enable_convergence_stop:
        raise ValueError(
            "experimental direct-[X/H] routing requires enable_convergence_stop=True"
        )
    minimum_required_iterations = (
        int(exact_config.minimum_iterations_before_convergence)
        + int(exact_config.required_consecutive_converged_iterations)
        - 1
    )
    if int(exact_config.iterations) < minimum_required_iterations:
        raise ValueError(
            "experimental direct-[X/H] routing requires enough exact iterations "
            "to satisfy its convergence contract"
        )

    starting_atmosphere, _deck = _direct_abundance_warm_start_model(
        effective_temperature=effective_temperature,
        log_surface_gravity=log_surface_gravity,
        microturbulence_km_s=microturbulence_km_s,
        abundance_by_atomic_number=abundance_by_atomic_number,
        enable_experimental=enable_experimental,
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        device=device,
    )
    exact_inputs = replace(
        exact_config.inputs,
        initial_atmosphere=starting_atmosphere,
    )
    from .runner import run_atmosphere_model

    result = run_atmosphere_model(replace(exact_config, inputs=exact_inputs))
    if not result.converged:
        raise RuntimeError(
            "mandatory exact atmosphere solve from the experimental direct-[X/H] "
            "start did not converge"
        )

    requested = complete_direct_abundance_vector(abundance_by_atomic_number)
    realized = np.asarray(
        [
            result.atmosphere.fixed_column_abundance_values[atomic_number]
            - SOLAR_METAL_LOG_ABUNDANCES_3_TO_99[atomic_number - 3]
            for atomic_number in range(3, 100)
        ],
        dtype=np.float64,
    )
    if not np.allclose(realized, requested, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(
            "converged direct-[X/H] exact solve changed the requested quantized mixture"
        )
    return result


__all__ = [
    "DEFAULT_DIRECT_XH_CHECKPOINT_PATH",
    "DEFAULT_DIRECT_XH_MANIFEST_PATH",
    "DIRECT_XH_ABUNDANCE_QUANTIZATION_DEX",
    "DIRECT_XH_ATOMIC_NUMBERS",
    "DIRECT_XH_EXPECTED_CHECKPOINT_SHA256",
    "DIRECT_XH_EXPECTED_MANIFEST_SHA256",
    "DIRECT_XH_EXPERIMENTAL_CHECKPOINT_FORMAT",
    "DIRECT_XH_FAMILY",
    "DIRECT_XH_FEATURE_FIELDS",
    "DIRECT_XH_REFERENCE_PAYLOAD_SHA256",
    "DIRECT_XH_SENTINEL_ATOMIC_NUMBERS",
    "DIRECT_XH_STANDALONE_CHECKPOINT_FORMAT",
    "DIRECT_XH_SUPPORT",
    "DirectAbundanceCheckpointProvenance",
    "DirectAbundanceInitializer",
    "DirectAbundanceOptimizerSurrogate",
    "build_direct_abundance_optimizer_surrogate",
    "complete_direct_abundance_vector",
    "direct_abundance_mixture_sha256",
    "direct_abundance_warm_start_deck",
    "load_direct_abundance_initializer",
    "retained_direct_abundance_mixture",
    "run_direct_abundance_atmosphere",
]
