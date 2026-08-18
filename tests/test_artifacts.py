from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qwen_j_lens.artifacts import ArtifactStore, canonical_json_bytes, portable_path


def test_canonical_json_is_deterministic() -> None:
    left = canonical_json_bytes({"z": [2, 1], "a": {"q": True}})
    right = canonical_json_bytes({"a": {"q": True}, "z": [2, 1]})

    assert left == right
    assert left == b'{"a":{"q":true},"z":[2,1]}\n'


def test_portable_path_does_not_embed_workspace_prefix(tmp_path: Path) -> None:
    within_repository = Path.cwd() / "data" / "prompts.jsonl"
    outside_repository = tmp_path / "private" / "prompts.jsonl"

    assert portable_path(within_repository) == "data/prompts.jsonl"
    assert portable_path(outside_repository) == "prompts.jsonl"


def test_npy_storage_is_deterministic_and_verified(tmp_path) -> None:
    first = ArtifactStore(tmp_path / "first")
    second = ArtifactStore(tmp_path / "second")
    array = np.arange(6, dtype=np.float32).reshape(2, 3)

    first_record = first.write_array("layer.npy", array, metadata={"layer": 1})
    second_record = second.write_array("layer.npy", array, metadata={"layer": 1})

    assert first_record.sha256 == second_record.sha256
    assert (tmp_path / "first/layer.npy").read_bytes() == (
        tmp_path / "second/layer.npy"
    ).read_bytes()
    assert np.array_equal(first.read_array(first_record), array)


def test_repository_manifest_separates_tested_and_pending() -> None:
    manifest = json.loads(
        open("artifacts/manifest.json", encoding="utf-8").read()
    )

    assert manifest["locally_tested"]["network_required"] is False
    assert len(manifest["pending_qwen_experiments"]) == 4
    methods = {entry["method"] for entry in manifest["pending_qwen_experiments"]}
    assert methods == {"j-lens", "natural-language-autoencoder"}
    assert "pending" in manifest["claim_status"]
    assert "not publication measurements" in manifest["claim_status"]


def test_publication_runs_declare_a_finding_and_a_caveat() -> None:
    """Every publication-scale run must state what it shows and what bounds it."""
    manifest = json.loads(
        open("artifacts/manifest.json", encoding="utf-8").read()
    )

    runs = manifest["publication_scale_runs"]["runs"]

    assert runs
    for run in runs:
        assert run["finding"].strip()
        assert run["caveat"].strip()
        assert Path(run["artifact_dir"]).is_dir()
        assert Path(run["records"]).is_file()


def test_capacity_run_records_a_trivial_predictor_baseline() -> None:
    """The capacity diagnostic is only meaningful next to its baselines."""
    manifest = json.loads(
        open("artifacts/manifest.json", encoding="utf-8").read()
    )

    capacity = next(
        run
        for run in manifest["publication_scale_runs"]["runs"]
        if run["method"] == "autoencoder-capacity-diagnostic"
    )

    assert capacity["baselines"]["zero_predictor"] == 1.0
    assert 0.0 < capacity["baselines"]["mean_predictor"] < 1.0
    assert capacity["beats_mean_predictor"] is False
    assert capacity["row_count_sweep"]
