"""Command-line entry point for the natural-language autoencoder path."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from qwen_j_lens.adapter import ModelAdapter
from qwen_j_lens.artifacts import ArtifactStore
from qwen_j_lens.nla.activations import ActivationBatch, collect_layer_activations
from qwen_j_lens.nla.config import NLARunConfig, load_nla_config
from qwen_j_lens.nla.faithfulness import (
    FaithfulnessReport,
    code_usage,
    gist_versus_specific,
    independent_reader_gap,
)
from qwen_j_lens.nla.train import train_autoencoder
from qwen_j_lens.qwen import QwenAdapter


def _corpus_prompts(path: Path, limit: int) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Activation corpus is missing: {path}")
    prompts = [
        str(json.loads(line)["text"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = prompts[:limit]
    if not selected:
        raise ValueError("Activation corpus contains no prompts")
    return selected


def _summarize_config(config: NLARunConfig) -> dict[str, object]:
    prompts = _corpus_prompts(config.activations.corpus_path, config.activations.max_prompts)
    return {
        "name": config.name,
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "layer": config.activations.layer,
        "corpus_prompt_count": len(prompts),
        "code_length": config.autoencoder.code_length,
        "code_vocab": config.autoencoder.code_vocab,
        "train_steps": config.train.steps,
        "gap_threshold": config.faithfulness.gap_threshold,
        "retained_threshold": config.faithfulness.retained_threshold,
    }


def _split_indices(count: int, test_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    if count < 4:
        raise ValueError("At least four activations are required to audit a split")
    test_size = max(1, int(round(count * test_fraction)))
    test_size = min(test_size, count - 1)
    train_index = torch.arange(0, count - test_size)
    test_index = torch.arange(count - test_size, count)
    return train_index, test_index


def run_nla(
    config: NLARunConfig,
    output: Path,
    *,
    allow_download: bool,
    specific_targets: Path | None,
) -> None:
    """Load the configured model, then run the autoencoder audit pipeline."""
    adapter = QwenAdapter.from_pretrained(
        config.model.model_id,
        local_files_only=False if allow_download else True,
        revision=config.model.revision,
        device_map=config.model.device_map,
        dtype=config.model.dtype,
        enable_thinking=config.model.enable_thinking,
    )
    execute_nla(adapter, config, output, specific_targets=specific_targets)


def execute_nla(
    adapter: ModelAdapter,
    config: NLARunConfig,
    output: Path,
    *,
    specific_targets: Path | None,
) -> None:
    """Collect activations, train the autoencoder, and write the audit.

    The adapter is injected so the whole artifact-producing pipeline runs offline
    against the tiny local transformer during tests and against an open-weight
    model on an accelerator.
    """
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    prompts = _corpus_prompts(config.activations.corpus_path, config.activations.max_prompts)
    batches = []
    for offset in range(0, len(prompts), config.activations.batch_size):
        input_ids, attention_mask = adapter.tokenize_raw(
            prompts[offset : offset + config.activations.batch_size]
        )
        batches.append(ActivationBatch(input_ids=input_ids, attention_mask=attention_mask))
    activations = collect_layer_activations(
        adapter, batches, layer_index=config.activations.layer
    )

    result = train_autoencoder(
        activations,
        code_length=config.autoencoder.code_length,
        code_vocab=config.autoencoder.code_vocab,
        width=config.autoencoder.width,
        embedding_dim=config.autoencoder.embedding_dim,
        temperature=config.autoencoder.temperature,
        steps=config.train.steps,
        lr=config.train.lr,
        batch_size=config.train.batch_size,
        entropy_weight=config.train.entropy_weight,
        seed=config.seed,
    )
    with torch.no_grad():
        sample = result.model(activations)
    codes = sample.code_indices
    reconstruction = sample.reconstruction

    train_index, test_index = _split_indices(
        activations.shape[0], config.faithfulness.test_fraction
    )
    gap = independent_reader_gap(
        codes,
        activations,
        train_index=train_index,
        test_index=test_index,
        code_vocab=config.autoencoder.code_vocab,
        width=config.faithfulness.reader_width,
        embedding_dim=config.faithfulness.reader_embedding_dim,
        steps=config.faithfulness.reader_steps,
        lr=config.faithfulness.reader_lr,
        seed=config.seed,
    )
    usage = code_usage(codes, config.autoencoder.code_vocab)

    faithfulness: dict[str, object] = {
        "reconstruction_error": result.final_relative_error,
        "independent_reader": gap,
        "code_usage": usage,
    }
    if specific_targets is not None:
        targets = torch.from_numpy(np.load(specific_targets, allow_pickle=False)).to(
            dtype=torch.float32
        )
        if targets.shape[0] != activations.shape[0]:
            raise ValueError("specific-targets rows must match the activation count")
        gist = gist_versus_specific(
            activations,
            reconstruction,
            targets,
            train_index=train_index,
            test_index=test_index,
        )
        report = FaithfulnessReport(
            reconstruction_error=result.final_relative_error,
            independent_reader=gap,
            gist_versus_specific=gist,
            code_usage=usage,
            gap_threshold=config.faithfulness.gap_threshold,
            retained_threshold=config.faithfulness.retained_threshold,
        )
        faithfulness = report.to_mapping()

    store = ArtifactStore(output)
    code_record = store.write_array(
        "codes.npy",
        codes.to(dtype=torch.int64),
        metadata={
            "layer": config.activations.layer,
            "code_length": config.autoencoder.code_length,
            "code_vocab": config.autoencoder.code_vocab,
            "row_count": int(codes.shape[0]),
        },
    )
    history_record = store.write_json("training-history.json", result.history)
    faithfulness_record = store.write_json("faithfulness.json", faithfulness)
    run_record = store.write_json(
        "run.json",
        {
            "config": {
                "name": config.name,
                "seed": config.seed,
                "model": asdict(config.model),
                "activations": {
                    **asdict(config.activations),
                    "corpus_path": str(config.activations.corpus_path),
                },
                "autoencoder": asdict(config.autoencoder),
                "train": asdict(config.train),
                "faithfulness": asdict(config.faithfulness),
            },
            "model_id": adapter.model_id,
            "activation_rows": int(activations.shape[0]),
            "specific_targets": None if specific_targets is None else str(specific_targets),
            "method": (
                "verbalize activations to a discrete code, reconstruct, and audit "
                "faithfulness with an independent reader and a specific-attribute probe"
            ),
            "source": "https://transformer-circuits.pub/2026/nla/index.html",
            "critique": "arXiv:2607.20379",
        },
    )
    store.write_records(
        "records.json",
        [code_record, history_record, faithfulness_record, run_record],
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="qwen-nla")
    commands = root.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-nla-config")
    validate.add_argument("--config", type=Path, required=True)
    run = commands.add_parser("nla-run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--specific-targets",
        type=Path,
        help="Optional NPY of per-activation attribute labels for the gist probe.",
    )
    run.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit Hugging Face downloads. Omit this flag for cache-only loading.",
    )
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_nla_config(args.config)
    if args.command == "validate-nla-config":
        print(json.dumps(_summarize_config(config), indent=2, sort_keys=True))
        return 0
    if args.command == "nla-run":
        run_nla(
            config,
            args.output,
            allow_download=args.allow_download,
            specific_targets=args.specific_targets,
        )
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
