"""Deterministic artifact metadata and array storage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
from torch import Tensor

from qwen_j_lens.errors import ArtifactError


def canonical_json_bytes(value: object) -> bytes:
    """Serialize JSON with stable key ordering and no insignificant whitespace."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def portable_path(path: str | Path) -> str:
    """Represent local artifact inputs without embedding the host workspace path."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.name


def capture_source_state() -> dict[str, str]:
    """Capture the tracked Git source state before an artifact-producing run."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", ".", ":(exclude)artifacts"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except FileNotFoundError:
        return {"git_revision": "unavailable", "git_status": "git executable unavailable"}
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or "Git source state unavailable"
        return {"git_revision": "unavailable", "git_status": detail}
    return {
        "git_revision": revision,
        "git_status": "clean" if not status else status,
    }


@dataclass(frozen=True)
class ArtifactRecord:
    path: str
    sha256: str
    kind: str
    metadata: Mapping[str, Any]


class ArtifactStore:
    """Store numerical arrays as stable NPY files plus canonical JSON metadata."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(self, relative_path: str, value: object) -> ArtifactRecord:
        data = canonical_json_bytes(value)
        return self.write_bytes(relative_path, data, kind="json")

    def write_bytes(
        self, relative_path: str, data: bytes, *, kind: str
    ) -> ArtifactRecord:
        """Write deterministic bytes and return their checksum record."""
        path = self._path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return ArtifactRecord(
            path=relative_path,
            sha256=hashlib.sha256(data).hexdigest(),
            kind=kind,
            metadata={},
        )

    def write_array(
        self,
        relative_path: str,
        value: Tensor | np.ndarray,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        if not relative_path.endswith(".npy"):
            raise ValueError("Numerical artifact paths must end in .npy")
        array = value.detach().cpu().numpy() if isinstance(value, Tensor) else np.asarray(value)
        path = self._path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.lib.format.write_array(stream, array, allow_pickle=False)
        data = path.read_bytes()
        return ArtifactRecord(
            path=relative_path,
            sha256=hashlib.sha256(data).hexdigest(),
            kind="npy",
            metadata=dict(metadata or {}),
        )

    def read_array(self, record: ArtifactRecord) -> np.ndarray:
        path = self._path(record.path)
        if not path.is_file():
            raise ArtifactError(f"Artifact is missing: {record.path}")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != record.sha256:
            raise ArtifactError(f"Artifact checksum mismatch: {record.path}")
        return np.load(path, allow_pickle=False)

    def write_records(
        self, relative_path: str, records: list[ArtifactRecord]
    ) -> ArtifactRecord:
        return self.write_json(relative_path, [asdict(record) for record in records])

    def _path(self, relative_path: str) -> Path:
        path = (self.root / relative_path).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError("Artifact path escapes the store root")
        return path


def _safe_artifact_path(root: Path, parent: Path, relative: str) -> Path:
    path = (parent / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Artifact path escapes the root: {relative}")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifact_tree(root: str | Path) -> tuple[bool, str]:
    """Verify the release manifest and every recursively discovered record file."""
    artifact_root = Path(root).resolve()
    manifest_path = artifact_root / "manifest.json"
    if not manifest_path.is_file():
        return False, f"Missing artifact manifest: {manifest_path}"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, f"Invalid artifact manifest: {error}"

    errors: list[str] = []
    for entry in manifest.get("required_evidence", []):
        if not isinstance(entry, dict):
            errors.append("Manifest required_evidence entries must be objects")
            continue
        relative = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            errors.append("Manifest evidence entries require path and sha256")
            continue
        try:
            path = _safe_artifact_path(artifact_root, artifact_root, relative)
        except ValueError as error:
            errors.append(str(error))
            continue
        if not path.is_file():
            errors.append(f"Missing required evidence: {relative}")
            continue
        if _file_sha256(path) != expected:
            errors.append(f"SHA-256 mismatch for required evidence: {relative}")

    records_paths = sorted(artifact_root.rglob("records.json"))
    if not records_paths:
        errors.append("No artifact records.json files were found")
    for records_path in records_paths:
        try:
            records = json.loads(records_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(
                f"Invalid record file {records_path.relative_to(artifact_root)}: {error}"
            )
            continue
        if not isinstance(records, list):
            errors.append(
                f"Record file {records_path.relative_to(artifact_root)} must be a list"
            )
            continue
        for offset, record in enumerate(records):
            if not isinstance(record, dict):
                errors.append(
                    f"Record {offset} in {records_path.relative_to(artifact_root)} "
                    "must be an object"
                )
                continue
            relative = record.get("path")
            expected = record.get("sha256")
            if not isinstance(relative, str) or not isinstance(expected, str):
                errors.append(
                    f"Record {offset} in {records_path.relative_to(artifact_root)} "
                    "requires path and sha256"
                )
                continue
            try:
                path = _safe_artifact_path(
                    artifact_root, records_path.parent, relative
                )
            except ValueError as error:
                errors.append(str(error))
                continue
            if not path.is_file():
                errors.append(
                    f"Missing recorded artifact: {path.relative_to(artifact_root)}"
                )
                continue
            if _file_sha256(path) != expected:
                errors.append(
                    f"SHA-256 mismatch for recorded artifact: "
                    f"{path.relative_to(artifact_root)}"
                )

    if errors:
        return False, "\n".join(errors)
    return True, "All required evidence and artifact records are valid."
