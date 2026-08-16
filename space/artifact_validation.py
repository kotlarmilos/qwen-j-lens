"""Recursive artifact validation used by the Space."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _safe_path(root: Path, parent: Path, relative: str) -> Path:
    candidate = (parent / relative).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"Artifact path escapes the upload root: {relative}")
    return candidate


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_artifacts(root: Path) -> tuple[bool, str, dict[str, Any]]:
    """Validate manifest requirements and every recursively discovered record set."""
    root = root.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return (
            False,
            f"Missing manifest `{manifest_path}`. Upload the repository artifact manifest.",
            {},
        )
    try:
        manifest = _load_json(manifest_path)
    except (OSError, json.JSONDecodeError) as error:
        return False, f"Manifest is invalid: {error}", {}

    errors: list[str] = []
    for relative in manifest.get("required_uploaded_artifacts", []):
        try:
            path = _safe_path(root, root, str(relative))
        except ValueError as error:
            errors.append(str(error))
            continue
        if not path.is_file():
            errors.append(f"Missing required file `{relative}`")

    record_paths = sorted(root.rglob("records.json"))
    if not record_paths:
        errors.append("No recursively discoverable `records.json` files were uploaded")

    referenced_arrays = 0
    for records_path in record_paths:
        try:
            records = _load_json(records_path)
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"Invalid record file `{records_path.relative_to(root)}`: {error}")
            continue
        if not isinstance(records, list):
            errors.append(f"Record file `{records_path.relative_to(root)}` must be a list")
            continue
        for offset, record in enumerate(records):
            if not isinstance(record, dict):
                errors.append(
                    f"Record {offset} in `{records_path.relative_to(root)}` is not an object"
                )
                continue
            relative = record.get("path")
            expected = record.get("sha256")
            kind = record.get("kind")
            if not isinstance(relative, str) or not isinstance(expected, str):
                errors.append(
                    f"Record {offset} in `{records_path.relative_to(root)}` lacks path or SHA-256"
                )
                continue
            try:
                artifact_path = _safe_path(root, records_path.parent, relative)
            except ValueError as error:
                errors.append(str(error))
                continue
            if kind == "npy":
                referenced_arrays += 1
            if not artifact_path.is_file():
                errors.append(
                    f"Missing referenced artifact `{artifact_path.relative_to(root)}`"
                )
                continue
            actual = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if actual != expected:
                errors.append(
                    f"SHA-256 mismatch for `{artifact_path.relative_to(root)}`"
                )

    if referenced_arrays == 0:
        errors.append("No NPY arrays are referenced by uploaded record files")
    if errors:
        lines = "\n".join(f"- {error}" for error in errors)
        return (
            False,
            "Artifacts are incomplete or invalid. No result is displayed.\n\n"
            f"Validation failures\n\n{lines}",
            manifest,
        )

    summary_path = root / "evaluation/summary.json"
    try:
        summary = _load_json(summary_path)
    except (OSError, json.JSONDecodeError) as error:
        return False, f"Evaluation summary is invalid: {error}", manifest
    return True, "All required artifacts and SHA-256 records are valid.", summary
