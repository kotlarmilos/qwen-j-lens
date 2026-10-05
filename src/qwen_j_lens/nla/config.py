"""TOML configuration for the fixed-symbol autoencoder experiment."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class NLAModelConfig:
    model_id: str
    revision: str | None
    local_files_only: bool
    dtype: str
    device_map: str | None
    enable_thinking: bool


@dataclass(frozen=True)
class NLAActivationConfig:
    layer: int
    corpus_path: Path
    max_prompts: int
    batch_size: int


@dataclass(frozen=True)
class NLAAutoencoderConfig:
    code_length: int
    code_vocab: int
    width: int
    embedding_dim: int
    temperature: float


@dataclass(frozen=True)
class NLATrainConfig:
    steps: int
    lr: float
    batch_size: int
    entropy_weight: float


@dataclass(frozen=True)
class NLAFaithfulnessConfig:
    test_fraction: float
    reader_steps: int
    reader_lr: float
    reader_width: int
    reader_embedding_dim: int
    gap_threshold: float
    retained_threshold: float


@dataclass(frozen=True)
class NLARunConfig:
    name: str
    seed: int
    model: NLAModelConfig
    activations: NLAActivationConfig
    autoencoder: NLAAutoencoderConfig
    train: NLATrainConfig
    faithfulness: NLAFaithfulnessConfig


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    child = value.get(key)
    if not isinstance(child, Mapping):
        raise ValueError(f"Missing [{key}] table")
    return child


def _positive_int(value: Mapping[str, Any], key: str, default: int | None = None) -> int:
    raw = value.get(key, default)
    if raw is None:
        raise ValueError(f"{key} is required")
    number = int(raw)
    if number <= 0:
        raise ValueError(f"{key} must be positive")
    return number


def _positive_float(value: Mapping[str, Any], key: str, default: float | None = None) -> float:
    raw = value.get(key, default)
    if raw is None:
        raise ValueError(f"{key} is required")
    number = float(raw)
    if number <= 0.0:
        raise ValueError(f"{key} must be positive")
    return number


def load_nla_config(path: str | Path) -> NLARunConfig:
    """Read and validate a fixed-symbol autoencoder run configuration."""
    config_path = Path(path)
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)
    model = _mapping(raw, "model")
    activations = _mapping(raw, "activations")
    autoencoder = _mapping(raw, "autoencoder")
    train = _mapping(raw, "train")
    faithfulness = _mapping(raw, "faithfulness")

    corpus = Path(str(activations["corpus_path"]))
    if not corpus.is_absolute():
        corpus = (config_path.parent.parent / corpus).resolve()

    layer = int(activations.get("layer", -1))
    if layer < 0:
        raise ValueError("activations.layer must be nonnegative")

    test_fraction = float(faithfulness.get("test_fraction", 0.5))
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("faithfulness.test_fraction must be strictly between 0 and 1")

    gap_threshold = float(faithfulness.get("gap_threshold", 0.2))
    retained_threshold = float(faithfulness.get("retained_threshold", 0.5))
    if gap_threshold < 0.0:
        raise ValueError("faithfulness.gap_threshold must not be negative")
    if not 0.0 <= retained_threshold <= 1.0:
        raise ValueError("faithfulness.retained_threshold must be within [0, 1]")

    return NLARunConfig(
        name=str(raw.get("name", config_path.stem)),
        seed=int(raw.get("seed", 0)),
        model=NLAModelConfig(
            model_id=str(model["id"]),
            revision=(str(model["revision"]) if model.get("revision") is not None else None),
            local_files_only=bool(model.get("local_files_only", True)),
            dtype=str(model.get("dtype", "auto")),
            device_map=(str(model["device_map"]) if model.get("device_map") is not None else None),
            enable_thinking=bool(model.get("enable_thinking", False)),
        ),
        activations=NLAActivationConfig(
            layer=layer,
            corpus_path=corpus,
            max_prompts=_positive_int(activations, "max_prompts"),
            batch_size=_positive_int(activations, "batch_size", 1),
        ),
        autoencoder=NLAAutoencoderConfig(
            code_length=_positive_int(autoencoder, "code_length"),
            code_vocab=_positive_int(autoencoder, "code_vocab"),
            width=_positive_int(autoencoder, "width"),
            embedding_dim=_positive_int(autoencoder, "embedding_dim"),
            temperature=_positive_float(autoencoder, "temperature", 1.0),
        ),
        train=NLATrainConfig(
            steps=_positive_int(train, "steps"),
            lr=_positive_float(train, "lr"),
            batch_size=_positive_int(train, "batch_size"),
            entropy_weight=float(train.get("entropy_weight", 0.0)),
        ),
        faithfulness=NLAFaithfulnessConfig(
            test_fraction=test_fraction,
            reader_steps=_positive_int(faithfulness, "reader_steps"),
            reader_lr=_positive_float(faithfulness, "reader_lr"),
            reader_width=_positive_int(faithfulness, "reader_width"),
            reader_embedding_dim=_positive_int(faithfulness, "reader_embedding_dim"),
            gap_threshold=gap_threshold,
            retained_threshold=retained_threshold,
        ),
    )
