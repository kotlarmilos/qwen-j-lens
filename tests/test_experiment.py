from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from collections.abc import Iterator

import numpy as np
import pytest
import torch

from qwen_j_lens.artifacts import ArtifactStore, validate_artifact_tree
from qwen_j_lens.capture import capture_layer_activations
from qwen_j_lens.cli import main as jlens_main
from qwen_j_lens.config import load_config
from qwen_j_lens.cross_method import load_cross_config
from qwen_j_lens.evaluation import EvaluationExample, evaluate_examples
from qwen_j_lens.jacobian import (
    estimate_residual_jacobian,
    estimate_token_vjps,
    token_vectors_from_jacobian,
)
from qwen_j_lens.merge import merge_estimate_shards
from qwen_j_lens.nla.activations import ActivationBatch, collect_layer_activations
from qwen_j_lens.nla.autoencoder import reconstruction_error
from qwen_j_lens.nla.capacity import linear_reference, trivial_baselines
from qwen_j_lens.nla.config import load_nla_config
from qwen_j_lens.nla.faithfulness import (
    FaithfulnessReport,
    code_usage,
    independent_reader_gap,
)
from toy import ScalarCausalAdapter, ToyAdapter


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def scratch_path(request: pytest.FixtureRequest) -> Iterator[Path]:
    root = REPO_ROOT / ".test-scratch"
    path = root / request.node.name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)
        if root.exists() and not any(root.iterdir()):
            root.rmdir()


def test_block_and_hidden_state_indices_are_explicit() -> None:
    adapter = ToyAdapter(width=4, vocab_size=9, layers=3)
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)
    _, captured = capture_layer_activations(
        adapter,
        input_ids,
        attention_mask,
        layer_indices=[1],
    )
    rows = collect_layer_activations(
        adapter,
        [ActivationBatch(input_ids=input_ids, attention_mask=attention_mask)],
        layer_index=2,
    )

    torch.testing.assert_close(captured[1].reshape(3, 4), rows)

    jlens = load_config(REPO_ROOT / "configs" / "qwen-publication.toml")
    symbols = load_nla_config(REPO_ROOT / "configs" / "nla-publication.toml")
    cross = load_cross_config(REPO_ROOT / "configs" / "cross-4b-dev.toml")
    assert jlens.evaluation.layer == 24
    assert symbols.activations.layer == 24
    assert jlens.evaluation.layer + 1 == 25
    assert jlens.evaluation.layer + 1 != symbols.activations.layer
    assert cross.hidden_state_index - 1 == 17


def test_selected_token_vjps_preserve_pair_and_context_weighting() -> None:
    scalar = ScalarCausalAdapter()
    estimate = estimate_token_vjps(
        scalar,
        [
            (torch.tensor([[1, 2]]), torch.ones((1, 2), dtype=torch.long)),
            (torch.tensor([[3]]), torch.ones((1, 1), dtype=torch.long)),
        ],
        layer_index=0,
        token_ids=[0, 1],
    )

    assert estimate.pair_count == 4
    assert estimate.context_count == 2
    assert estimate.values[:, 0].tolist() == pytest.approx(
        [13.0 / 3.0, 26.0 / 3.0]
    )

    adapter = ToyAdapter(width=3, vocab_size=7, layers=2)
    batch = [
        (torch.tensor([[1, 2]]), torch.ones((1, 2), dtype=torch.long))
    ]
    full = estimate_residual_jacobian(
        adapter,
        batch,
        layer_index=0,
        vjp_chunk_size=2,
    )
    selected = estimate_token_vjps(
        adapter,
        batch,
        layer_index=0,
        token_ids=[1, 5],
        vjp_chunk_size=1,
    )
    torch.testing.assert_close(
        selected.values,
        token_vectors_from_jacobian(adapter, full.values, token_ids=[1, 5]),
    )


def _write_shard(
    root: Path,
    *,
    value: float,
    offset: int,
    count: int,
    pair_count: int,
) -> None:
    config = load_config(REPO_ROOT / "configs" / "qwen-publication.toml")
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
                    "pair_count": pair_count,
                }
            ],
            "model_id": config.model.model_id,
            "model_revision": config.model.revision,
            "prompt_shard": {"count": count, "offset": offset},
        },
    )
    store.write_records("records.json", [vector, run])


def test_merge_uses_context_count_not_pair_count(scratch_path: Path) -> None:
    first = scratch_path / "first"
    second = scratch_path / "second"
    output = scratch_path / "merged"
    _write_shard(first, value=1.0, offset=0, count=300, pair_count=1000)
    _write_shard(second, value=4.0, offset=300, count=700, pair_count=1)

    merge_estimate_shards(
        load_config(REPO_ROOT / "configs" / "qwen-publication.toml"),
        [second, first],
        output,
        layer=24,
    )

    values = np.load(
        output / "controlled-token-vectors-layer-024.npy",
        allow_pickle=False,
    )
    assert values == pytest.approx(3.1)
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["layers"][0]["context_count"] == 1000
    assert run["layers"][0]["pair_count"] == 1001


def test_one_same_layer_edit_changes_the_recorded_coordinate() -> None:
    adapter = ToyAdapter()
    vectors = {" blue": torch.ones(adapter.hidden_size)}
    example = EvaluationExample.from_mapping(
        {
            "id": "edit-readback",
            "property": "internal_reasoning",
            "prompt": "Return one token.",
            "target_concepts": ["blue"],
            "measurement_steps": [0],
            "intervention": {
                "kind": "steer",
                "target": " blue",
                "coefficient": 4.0,
            },
        }
    )

    result = evaluate_examples(
        adapter,
        [example],
        vectors,
        layer_index=1,
        seeds=[7],
        max_new_tokens=1,
    )[0]

    assert result.baseline_target_score == result.baseline_trajectory[0]
    assert result.intervention_target_score == result.intervention_trajectory[0]
    assert result.intervention_target_score != pytest.approx(
        result.baseline_target_score
    )


def test_centered_error_trivial_baselines_and_pca_are_consistent() -> None:
    generator = torch.Generator().manual_seed(11)
    coefficients = torch.randn((80, 1), generator=generator)
    direction = torch.randn((1, 6), generator=generator)
    activations = coefficients @ direction + 5.0
    mean = activations.mean(dim=0, keepdim=True).expand_as(activations)
    baselines = trivial_baselines(activations)
    reference = linear_reference(activations, [0, 1])

    assert float(reconstruction_error(mean, activations)) == pytest.approx(1.0)
    assert baselines.mean_predictor == pytest.approx(1.0)
    assert baselines.zero_predictor == pytest.approx(
        float(reconstruction_error(torch.zeros_like(activations), activations))
    )
    assert reference[0].relative_error == pytest.approx(1.0, abs=1e-6)
    assert reference[1].relative_error < 1e-6


def test_fixed_symbol_usage_does_not_create_a_faithfulness_label() -> None:
    generator = torch.Generator().manual_seed(1)
    count = 40
    activations = torch.randn((count, 6), generator=generator)
    codes = torch.stack(
        (torch.arange(count), torch.zeros(count, dtype=torch.long)), dim=1
    )
    reader = independent_reader_gap(
        codes,
        activations,
        train_index=torch.arange(0, count // 2),
        test_index=torch.arange(count // 2, count),
        code_vocab=count,
        steps=300,
        seed=0,
    )
    usage = code_usage(codes, code_vocab=count)
    report = FaithfulnessReport(
        reconstruction_error=1.005,
        independent_reader=reader,
        gist_versus_specific={
            "original_specific_r2": 0.8,
            "reconstructed_specific_r2": 0.8,
            "specific_retained_fraction": 1.0,
        },
        code_usage=usage,
        gap_threshold=0.2,
        retained_threshold=0.5,
    ).to_mapping()

    assert reader["gap"] == pytest.approx(
        reader["independent_test_error"] - reader["co_adapted_test_error"]
    )
    assert reader["gap"] > 0.3
    assert usage["distinct_code_fraction"] == pytest.approx(1.0)
    assert "faithful" not in report["flags"]
    assert "do not establish faithfulness" in report["interpretation"]


def test_validate_config_cli_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    result = jlens_main(
        [
            "validate-config",
            "--config",
            str(REPO_ROOT / "configs" / "qwen-publication.toml"),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["controlled_token_count"] == 13
    assert payload["layers"] == [24]
    assert payload["evaluation_layer"] == 24
    assert payload["calibration_prompt_count"] == 1000


def test_repository_evidence_provenance_and_hashes_are_consistent() -> None:
    complete, message = validate_artifact_tree(REPO_ROOT / "artifacts")
    assert complete, message

    manifest = json.loads(
        (REPO_ROOT / "artifacts/manifest.json").read_text(encoding="utf-8")
    )
    for entry in manifest["inputs"]:
        path = REPO_ROOT / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]

    release = manifest["release"]
    assert release["base_commit"] == "10c04c7"
    assert release["hardware"] == "NVIDIA A100 80 GB PCIe"
    assert (
        release["model_revision"]
        == "b968826d9c46dd6066d109eabc6255188de91218"
    )

    jlens = manifest["runs"]["controlled_subset_j_lens"]
    symbols = manifest["runs"]["fixed_symbol_autoencoder"]
    assert jlens["layer"] + 1 == 25
    assert symbols["reconstruction_error"] == pytest.approx(1.0051531791687012)
    assert symbols["mean_predictor"] == 1.0
    assert symbols["unique_code_strings"] == 2

    reader = json.loads(
        (
            REPO_ROOT / "artifacts/colab/nla-qwen3-8b/faithfulness.json"
        ).read_text(encoding="utf-8")
    )["independent_reader"]
    assert reader["co_adapted_test_error"] > 0.99
    assert reader["independent_test_error"] > 0.99
    assert abs(reader["gap"]) < 1e-4

    rows = [
        json.loads(line)
        for line in (
            REPO_ROOT / "artifacts/colab/qwen3-8b-layer-24-evaluate/results.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    verbal = [row for row in rows if row["property"] == "verbal_report"]
    selectivity = [row for row in rows if row["property"] == "selectivity"]
    assert {row["baseline_target_score"] for row in verbal} == {0.0}
    assert {row["intervention_target_score"] for row in verbal} == {0.0}
    assert {row["baseline_target_score"] for row in selectivity} == {1.0}
    assert {row["intervention_target_score"] for row in selectivity} == {1.0}

    diagnostic = json.loads(
        (
            REPO_ROOT
            / "artifacts/qwen3-4b-bit-budget-layer18/bit-budget.json"
        ).read_text(encoding="utf-8")
    )
    assert diagnostic["baselines"]["mean_predictor"] == 1.0
    assert diagnostic["linear_reference"][0]["relative_error"] == 1.0
    assert "in-sample" in diagnostic["method"]


def _artifact_tree(root: Path) -> None:
    run = ArtifactStore(root / "model-run")
    records = [
        run.write_array("layer.npy", np.eye(2, dtype=np.float32)),
        run.write_json("run.json", {"model": "toy"}),
    ]
    records_record = run.write_records("records.json", records)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "required_evidence": [
                    {
                        "path": "model-run/records.json",
                        "sha256": records_record.sha256,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_artifact_validation_rejects_corruption_and_path_escape(
    scratch_path: Path,
) -> None:
    _artifact_tree(scratch_path)
    complete, message = validate_artifact_tree(scratch_path)
    assert complete, message

    array = scratch_path / "model-run/layer.npy"
    original = array.read_bytes()
    array.write_bytes(b"corrupt")
    complete, message = validate_artifact_tree(scratch_path)
    assert not complete
    assert "SHA-256 mismatch" in message

    array.write_bytes(original)
    records_path = scratch_path / "model-run/records.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    records[0]["path"] = "../../escape.npy"
    records[0]["sha256"] = hashlib.sha256(b"escape").hexdigest()
    records_path.write_text(json.dumps(records), encoding="utf-8")
    complete, message = validate_artifact_tree(scratch_path)
    assert not complete
    assert "escapes the root" in message
