#!/usr/bin/env python3
"""Build a pinned, deterministic WikiText calibration sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence

DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATASET_LICENSE = "CC BY-SA 3.0 and GFDL"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _sample_key(seed: int, index: int, text: str) -> str:
    value = f"{seed}\0{index}\0{text}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def build_corpus(
    rows: Sequence[dict[str, Any]],
    *,
    count: int,
    seed: int,
    min_characters: int = 160,
    max_characters: int = 1600,
) -> list[dict[str, object]]:
    """Select deterministic prose contexts from a materialized WikiText split."""
    candidates = []
    for source_index, row in enumerate(rows):
        text = _normalize(str(row["text"]))
        if (
            len(text) < min_characters
            or text.startswith("=")
            or text.endswith("=")
        ):
            continue
        candidates.append(
            (
                _sample_key(seed, source_index, text),
                source_index,
                text[:max_characters].rstrip(),
            )
        )
    if len(candidates) < count:
        raise ValueError(
            f"Only {len(candidates)} eligible WikiText rows remain, need {count}"
        )
    candidates.sort()
    return [
        {
            "id": f"wikitext-{offset:04d}",
            "text": text,
            "provenance": {
                "dataset": DATASET_ID,
                "config": DATASET_CONFIG,
                "revision": DATASET_REVISION,
                "split": "train",
                "source_index": source_index,
                "license": DATASET_LICENSE,
            },
        }
        for offset, (_key, source_index, text) in enumerate(candidates[:count])
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/publication_calibration_prompts.jsonl"),
    )
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify the existing corpus and metadata without loading the dataset.",
    )
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")

    metadata_path = args.output.with_suffix(".metadata.json")
    if args.verify_only:
        if not args.output.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("Publication corpus or metadata is missing")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "config": DATASET_CONFIG,
            "count": args.count,
            "dataset": DATASET_ID,
            "revision": DATASET_REVISION,
            "seed": args.seed,
            "split": "train",
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"Corpus metadata mismatch for {key}")
        actual = _sha256(args.output)
        if metadata.get("output_sha256") != actual:
            raise ValueError("Publication corpus SHA-256 does not match metadata")
        print(actual)
        return 0

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit(
            "Install corpus dependencies with `pip install -e '.[data]'`."
        ) from error

    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split="train",
        revision=DATASET_REVISION,
    )
    records = build_corpus(dataset, count=args.count, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    args.output.write_text(encoded, encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "config": DATASET_CONFIG,
                "count": len(records),
                "dataset": DATASET_ID,
                "license": DATASET_LICENSE,
                "output": str(args.output),
                "output_sha256": _sha256(args.output),
                "revision": DATASET_REVISION,
                "seed": args.seed,
                "split": "train",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
