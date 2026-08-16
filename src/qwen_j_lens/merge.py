"""Merge independently computed calibration shards without loading a model."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from qwen_j_lens.artifacts import ArtifactStore
from qwen_j_lens.config import RunConfig


@dataclass(frozen=True)
class EstimateShard:
    root: Path
    values: torch.Tensor
    offset: int
    prompt_count: int
    context_count: int
    pair_count: int


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_shard(root: Path, config: RunConfig, layer: int) -> EstimateShard:
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    if run.get("model_id") != config.model.model_id:
        raise ValueError(f"Model ID mismatch in shard `{root}`")
    if run.get("model_revision") != config.model.revision:
        raise ValueError(f"Model revision mismatch in shard `{root}`")
    shard = run.get("prompt_shard")
    if not isinstance(shard, dict):
        raise ValueError(f"Prompt shard metadata is missing in `{root}`")
    layers = [
        value
        for value in run.get("layers", [])
        if isinstance(value, dict) and value.get("layer") == layer
    ]
    if len(layers) != 1:
        raise ValueError(f"Shard `{root}` does not contain exactly one layer {layer}")
    layer_metadata = layers[0]
    relative = str(layer_metadata["controlled_token_vectors"])
    vector_path = root / relative
    records = json.loads((root / "records.json").read_text(encoding="utf-8"))
    matching_records = [
        record
        for record in records
        if isinstance(record, dict) and record.get("path") == relative
    ]
    if len(matching_records) != 1:
        raise ValueError(f"Vector record is missing or duplicated in `{root}`")
    if _sha256(vector_path) != matching_records[0].get("sha256"):
        raise ValueError(f"Vector SHA-256 mismatch in shard `{root}`")
    values = torch.from_numpy(np.load(vector_path, allow_pickle=False)).to(
        torch.float64
    )
    expected = (len(config.estimation.controlled_tokens),)
    if values.ndim != 2 or values.shape[0] != expected[0]:
        raise ValueError(f"Unexpected controlled-token vector shape in `{root}`")
    return EstimateShard(
        root=root,
        values=values,
        offset=int(shard["offset"]),
        prompt_count=int(shard["count"]),
        context_count=int(layer_metadata["context_count"]),
        pair_count=int(layer_metadata["pair_count"]),
    )


def merge_estimate_shards(
    config: RunConfig,
    inputs: Sequence[Path],
    output: Path,
    *,
    layer: int,
) -> None:
    """Merge a complete, nonoverlapping prompt partition by context count."""
    if layer not in config.estimation.layers:
        raise ValueError("Merged layer must be declared in estimation.layers")
    if not inputs:
        raise ValueError("At least one estimate shard is required")
    shards = sorted(
        (_load_shard(path, config, layer) for path in inputs),
        key=lambda shard: shard.offset,
    )
    expected_offset = 0
    for shard in shards:
        if shard.offset != expected_offset:
            raise ValueError(
                "Prompt shards must form a contiguous partition beginning at offset 0"
            )
        if shard.prompt_count <= 0 or shard.context_count <= 0:
            raise ValueError("Prompt and context counts must be positive")
        expected_offset += shard.prompt_count
    if expected_offset != config.estimation.max_prompts:
        raise ValueError(
            f"Prompt shards cover {expected_offset} prompts, expected "
            f"{config.estimation.max_prompts}"
        )
    shape = shards[0].values.shape
    if any(shard.values.shape != shape for shard in shards):
        raise ValueError("Estimate shards have inconsistent vector shapes")

    context_count = sum(shard.context_count for shard in shards)
    weighted = sum(
        (shard.values * shard.context_count for shard in shards),
        start=torch.zeros_like(shards[0].values),
    )
    merged = (weighted / context_count).to(torch.float32)
    pair_count = sum(shard.pair_count for shard in shards)
    store = ArtifactStore(output)
    vector_record = store.write_array(
        f"controlled-token-vectors-layer-{layer:03d}.npy",
        merged,
        metadata={
            "construction": "context-weighted merge of selected-token VJP shards",
            "context_count": context_count,
            "layer": layer,
            "pair_count": pair_count,
            "tokens": list(config.estimation.controlled_tokens),
        },
    )
    run_record = store.write_json(
        "run.json",
        {
            "config": config.name,
            "layers": [
                {
                    "context_count": context_count,
                    "controlled_token_vectors": vector_record.path,
                    "layer": layer,
                    "pair_count": pair_count,
                }
            ],
            "merged_shards": [
                {
                    "context_count": shard.context_count,
                    "offset": shard.offset,
                    "pair_count": shard.pair_count,
                    "path": str(shard.root),
                    "prompt_count": shard.prompt_count,
                }
                for shard in shards
            ],
            "model_id": config.model.model_id,
            "model_revision": config.model.revision,
            "prompt_shard": {
                "count": expected_offset,
                "offset": 0,
            },
        },
    )
    store.write_records("records.json", [vector_record, run_record])
