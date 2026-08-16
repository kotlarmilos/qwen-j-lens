from __future__ import annotations

import json
from pathlib import Path

from qwen_j_lens.cross_method import (
    CrossMethodConfig,
    concept_agreement,
    execute_cross_method,
    jlens_linear_readout,
    load_cross_config,
    pearson_r,
    spearman_rho,
)
from qwen_j_lens.nla.config import (
    NLAAutoencoderConfig,
    NLAModelConfig,
    NLATrainConfig,
)
import torch
from toy import ToyAdapter

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_load_cross_config_resolves_corpus() -> None:
    config = load_cross_config(REPO_ROOT / "configs" / "cross-4b-dev.toml")

    assert config.model.model_id == "Qwen/Qwen3-4B-Instruct-2507"
    assert config.hidden_state_index == 18
    assert config.hidden_state_index - 1 == 17
    assert len(config.tokens) == 13
    assert config.corpus_path.is_file()
    assert 0.0 < config.test_fraction < 1.0


def test_spearman_matches_pearson_on_monotone_data() -> None:
    first = [0.1, 0.2, 0.3, 0.4]
    second = [1.0, 4.0, 9.0, 16.0]
    assert abs(spearman_rho(first, second) - 1.0) < 1e-9
    assert abs(pearson_r([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) - 1.0) < 1e-9
    assert abs(spearman_rho([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) + 1.0) < 1e-9


def test_jlens_linear_readout_shape() -> None:
    activations = torch.randn(5, 4)
    lens = torch.randn(3, 4)
    readout = jlens_linear_readout(activations, lens)
    assert readout.shape == (5, 3)
    expected = activations @ lens.transpose(0, 1)
    assert torch.allclose(readout, expected)


def test_concept_agreement_recovers_planted_signal() -> None:
    torch.manual_seed(0)
    rows, hidden = 40, 6
    activations = torch.randn(rows, hidden)
    lens = torch.eye(2, hidden)
    logits = activations[:, :2].clone()
    reconstructions = activations.clone()
    train_index = torch.arange(0, 20)
    test_index = torch.arange(20, 40)

    report = concept_agreement(
        activations,
        logits,
        lens,
        reconstructions,
        tokens=("a", "b"),
        token_ids=(0, 1),
        train_index=train_index,
        test_index=test_index,
        hidden_state_index=2,
        jlens_capture_layer=1,
        reconstruction_error=0.0,
        pair_count=10,
        context_count=2,
        fidelity_threshold=0.5,
        retention_threshold=0.5,
    )

    assert report.mean_jlens_fidelity > 0.9
    assert report.mean_nla_retention > 0.9
    assert report.quadrant_counts["both_recoverable"] == 2
    assert len(report.concepts) == 2


def _toy_config(corpus: Path) -> CrossMethodConfig:
    return CrossMethodConfig(
        name="toy-cross",
        seed=7,
        model=NLAModelConfig(
            model_id="local/toy",
            revision=None,
            local_files_only=True,
            dtype="float32",
            device_map=None,
            enable_thinking=False,
        ),
        hidden_state_index=2,
        corpus_path=corpus,
        max_prompts=8,
        batch_size=2,
        tokens=("a", "b", "c", "d"),
        include_future=True,
        vjp_chunk_size=8,
        autoencoder=NLAAutoencoderConfig(
            code_length=2, code_vocab=8, width=16, embedding_dim=4, temperature=1.0
        ),
        train=NLATrainConfig(steps=30, lr=0.05, batch_size=16, entropy_weight=0.01),
        test_fraction=0.5,
        fidelity_threshold=0.5,
        retention_threshold=0.5,
    )


def test_execute_cross_method_writes_agreement_artifacts(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    lines = [
        {"id": f"c{index}", "text": f"a careful test sentence number {index} here"}
        for index in range(6)
    ]
    corpus.write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8"
    )
    config = _toy_config(corpus)
    output = tmp_path / "run"

    report = execute_cross_method(
        ToyAdapter(width=4, vocab_size=9, layers=3), config, output
    )

    assert report.hidden_state_index == 2
    assert report.jlens_capture_layer == 1
    assert report.activation_rows == report.train_rows + report.test_rows
    assert len(report.concepts) == 4
    assert set(report.quadrant_counts) == {
        "both_recoverable",
        "jlens_only",
        "nla_only",
        "neither",
    }
    assert sum(report.quadrant_counts.values()) == 4

    records = json.loads((output / "records.json").read_text())
    paths = {record["path"] for record in records}
    assert {"agreement.json", "training-history.json", "run.json"} <= paths

    agreement = json.loads((output / "agreement.json").read_text())
    assert agreement["hidden_state_index"] == 2
    assert agreement["jlens_capture_layer"] == 1
    assert "spearman_rho" in agreement["agreement"]
    assert "pearson_r" in agreement["agreement"]
    assert len(agreement["concepts"]) == 4
    for concept in agreement["concepts"]:
        assert 0.0 <= concept["jlens_fidelity"]
        assert concept["quadrant"] in {
            "both_recoverable",
            "jlens_only",
            "nla_only",
            "neither",
        }

    run = json.loads((output / "run.json").read_text())
    assert run["model_id"] == "local/toy"
    assert run["config"]["jlens_capture_layer"] == 1
