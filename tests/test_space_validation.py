from __future__ import annotations

import json

import numpy as np

from qwen_j_lens.artifacts import ArtifactStore
from space.artifact_validation import validate_artifacts


def _complete_tree(root) -> None:
    run = ArtifactStore(root / "model-run")
    records = [
        run.write_array("layer.npy", np.eye(2, dtype=np.float32)),
        run.write_json("run.json", {"model": "toy"}),
    ]
    run.write_records("records.json", records)
    (root / "evaluation").mkdir()
    (root / "evaluation/summary.json").write_text('{"ok":true}\n')
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "required_uploaded_artifacts": [
                    "model-run/records.json",
                    "evaluation/summary.json",
                ]
            }
        )
    )


def test_space_recursively_verifies_referenced_arrays(tmp_path) -> None:
    _complete_tree(tmp_path)

    complete, message, payload = validate_artifacts(tmp_path)

    assert complete is True
    assert "SHA-256 records are valid" in message
    assert payload == {"ok": True}


def test_space_rejects_corrupt_and_missing_referenced_arrays(tmp_path) -> None:
    _complete_tree(tmp_path)
    (tmp_path / "model-run/layer.npy").write_bytes(b"corrupt")

    complete, message, _ = validate_artifacts(tmp_path)

    assert complete is False
    assert "SHA-256 mismatch" in message

    (tmp_path / "model-run/layer.npy").unlink()
    complete, message, _ = validate_artifacts(tmp_path)
    assert complete is False
    assert "Missing referenced artifact" in message
