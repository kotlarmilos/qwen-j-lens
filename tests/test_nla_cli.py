from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qwen_j_lens.nla.config import (
    NLAActivationConfig,
    NLAAutoencoderConfig,
    NLAFaithfulnessConfig,
    NLAModelConfig,
    NLARunConfig,
    NLATrainConfig,
    load_nla_config,
)
from qwen_j_lens.nla_cli import execute_nla
from toy import ToyAdapter

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_load_nla_smoke_config_resolves_corpus() -> None:
    config = load_nla_config(REPO_ROOT / "configs" / "nla-smoke.toml")

    assert config.model.model_id == "Qwen/Qwen3-4B-Instruct-2507"
    assert config.activations.layer == 18
    assert config.autoencoder.code_length == 4
    assert config.activations.corpus_path.is_file()


def _toy_config(corpus: Path) -> NLARunConfig:
    return NLARunConfig(
        name="toy-nla",
        seed=7,
        model=NLAModelConfig(
            model_id="local/toy",
            revision=None,
            local_files_only=True,
            dtype="float32",
            device_map=None,
            enable_thinking=False,
        ),
        activations=NLAActivationConfig(
            layer=2, corpus_path=corpus, max_prompts=8, batch_size=2
        ),
        autoencoder=NLAAutoencoderConfig(
            code_length=2, code_vocab=8, width=16, embedding_dim=4, temperature=1.0
        ),
        train=NLATrainConfig(steps=40, lr=0.05, batch_size=16, entropy_weight=0.01),
        faithfulness=NLAFaithfulnessConfig(
            test_fraction=0.5,
            reader_steps=30,
            reader_lr=0.05,
            reader_width=16,
            reader_embedding_dim=4,
            gap_threshold=0.2,
            retained_threshold=0.5,
        ),
    )


def test_execute_nla_writes_deterministic_artifacts(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    lines = [
        {"id": f"c{index}", "text": f"a careful test sentence number {index}"}
        for index in range(6)
    ]
    corpus.write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8"
    )
    config = _toy_config(corpus)
    output = tmp_path / "run"

    execute_nla(ToyAdapter(width=4, vocab_size=9, layers=3), config, output, specific_targets=None)

    records = json.loads((output / "records.json").read_text())
    paths = {record["path"] for record in records}
    assert {"codes.npy", "training-history.json", "faithfulness.json", "run.json"} <= paths

    faithfulness = json.loads((output / "faithfulness.json").read_text())
    assert "independent_reader" in faithfulness
    assert "gap" in faithfulness["independent_reader"]
    assert "code_usage" in faithfulness

    codes = np.load(output / "codes.npy", allow_pickle=False)
    assert codes.ndim == 2
    assert codes.shape[1] == config.autoencoder.code_length
