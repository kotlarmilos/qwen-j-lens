"""TOML configuration loading and validation."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    revision: str | None
    local_files_only: bool
    dtype: str
    device_map: str | None
    enable_thinking: bool


@dataclass(frozen=True)
class EstimationConfig:
    layers: tuple[int, ...]
    calibration_path: Path
    controlled_tokens: tuple[str, ...]
    include_future: bool
    max_prompts: int
    batch_size: int
    vjp_chunk_size: int


@dataclass(frozen=True)
class EvaluationConfig:
    prompts_path: Path
    seeds: tuple[int, ...]
    max_new_tokens: int
    layer: int
    temperature: float


@dataclass(frozen=True)
class RunConfig:
    name: str
    seed: int
    model: ModelConfig
    estimation: EstimationConfig
    evaluation: EvaluationConfig


def load_config(path: str | Path) -> RunConfig:
    """Read and validate a project run configuration."""
    config_path = Path(path)
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)
    model = _mapping(raw, "model")
    estimation = _mapping(raw, "estimation")
    evaluation = _mapping(raw, "evaluation")
    layers = tuple(int(layer) for layer in estimation.get("layers", ()))
    tokens = tuple(str(token) for token in estimation.get("controlled_tokens", ()))
    if not layers or min(layers) < 0:
        raise ValueError("estimation.layers must contain nonnegative indices")
    if not tokens:
        raise ValueError("estimation.controlled_tokens must not be empty")
    max_prompts = int(estimation.get("max_prompts", 0))
    if max_prompts <= 0:
        raise ValueError("estimation.max_prompts must be positive")
    batch_size = int(estimation.get("batch_size", 1))
    if batch_size <= 0:
        raise ValueError("estimation.batch_size must be positive")
    vjp_chunk_size = int(estimation.get("vjp_chunk_size", 32))
    if vjp_chunk_size <= 0:
        raise ValueError("estimation.vjp_chunk_size must be positive")
    calibration = Path(str(estimation["calibration_path"]))
    if not calibration.is_absolute():
        calibration = (config_path.parent.parent / calibration).resolve()
    evaluation_path = Path(str(evaluation["prompts_path"]))
    if not evaluation_path.is_absolute():
        evaluation_path = (config_path.parent.parent / evaluation_path).resolve()
    seeds = tuple(int(seed) for seed in evaluation.get("seeds", ()))
    if not seeds:
        raise ValueError("evaluation.seeds must not be empty")
    max_new_tokens = int(evaluation.get("max_new_tokens", 0))
    if max_new_tokens <= 0:
        raise ValueError("evaluation.max_new_tokens must be positive")
    temperature = float(evaluation.get("temperature", 0.0))
    if temperature < 0:
        raise ValueError("evaluation.temperature must not be negative")
    evaluation_layer = int(evaluation["layer"])
    if evaluation_layer not in layers:
        raise ValueError("evaluation.layer must be included in estimation.layers")
    return RunConfig(
        name=str(raw.get("name", config_path.stem)),
        seed=int(raw.get("seed", 0)),
        model=ModelConfig(
            model_id=str(model["id"]),
            revision=(
                str(model["revision"]) if model.get("revision") is not None else None
            ),
            local_files_only=bool(model.get("local_files_only", True)),
            dtype=str(model.get("dtype", "auto")),
            device_map=(
                str(model["device_map"]) if model.get("device_map") is not None else None
            ),
            enable_thinking=bool(model.get("enable_thinking", False)),
        ),
        estimation=EstimationConfig(
            layers=layers,
            calibration_path=calibration,
            controlled_tokens=tokens,
            include_future=bool(estimation.get("include_future", True)),
            max_prompts=max_prompts,
            batch_size=batch_size,
            vjp_chunk_size=vjp_chunk_size,
        ),
        evaluation=EvaluationConfig(
            prompts_path=evaluation_path,
            seeds=seeds,
            max_new_tokens=max_new_tokens,
            layer=evaluation_layer,
            temperature=temperature,
        ),
    )


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    child = value.get(key)
    if not isinstance(child, Mapping):
        raise ValueError(f"Missing [{key}] table")
    return child
