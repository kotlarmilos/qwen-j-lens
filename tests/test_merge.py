from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from qwen_j_lens.artifacts import ArtifactStore
from qwen_j_lens.config import load_config
from qwen_j_lens.merge import merge_estimate_shards


def _write_shard(
    root: Path,
    *,
    value: float,
    offset: int,
    count: int,
) -> None:
    config = load_config("configs/smoke.toml")
    store = ArtifactStore(root)
    vector = store.write_array(
        "controlled-token-vectors-layer-024.npy",
        torch.full((len(config.estimation.controlled_tokens), 3), value),
    )
    run = store.write_json(
        "run.json",
        {
            "layers": [
                {
                    "context_count": count,
                    "controlled_token_vectors": vector.path,
                    "layer": 24,
                    "pair_count": count * 10,
                }
            ],
            "model_id": config.model.model_id,
            "model_revision": config.model.revision,
            "prompt_shard": {"count": count, "offset": offset},
        },
    )
    store.write_records("records.json", [vector, run])


def test_merge_estimate_shards_uses_context_weighting(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    output = tmp_path / "merged"
    _write_shard(first, value=1.0, offset=0, count=3)
    _write_shard(second, value=4.0, offset=3, count=5)

    merge_estimate_shards(
        load_config("configs/smoke.toml"),
        [second, first],
        output,
        layer=24,
    )

    values = np.load(
        output / "controlled-token-vectors-layer-024.npy",
        allow_pickle=False,
    )
    assert values == pytest.approx(2.875)
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["layers"][0]["context_count"] == 8
    assert run["layers"][0]["pair_count"] == 80
    assert set(run["source_code"]) == {"git_revision", "git_status"}
