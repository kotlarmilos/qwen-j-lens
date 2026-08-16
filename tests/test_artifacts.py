from __future__ import annotations

import json

import numpy as np

from qwen_j_lens.artifacts import ArtifactStore, canonical_json_bytes


def test_canonical_json_is_deterministic() -> None:
    left = canonical_json_bytes({"z": [2, 1], "a": {"q": True}})
    right = canonical_json_bytes({"a": {"q": True}, "z": [2, 1]})

    assert left == right
    assert left == b'{"a":{"q":true},"z":[2,1]}\n'


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
    assert "No Qwen experiment results" in manifest["claim_status"]
