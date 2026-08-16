"""Measure the bit-budget frontier and the optimization stability of the code.

The natural-language autoencoder reports a relative reconstruction error, and a
low value is easy to over-read. This script separates three explanations for any
value it produces. The first is capacity, which the bit-budget sweep varies. The
second is the decoder, which the width control varies at a fixed bit budget. The
third is optimization luck, which the seed controls vary with everything else
held fixed. A principal-component reference converted to bits gives a continuous
code to compare against at a matched budget.

Activations are collected once from the real model and cached, because the
collection stage dominates the runtime on a CPU host and every sweep reads the
same rows.

Usage:
    python scripts/bit_budget_study.py --config configs/cross-4b-dev.toml \\
        --corpus data/publication_calibration_prompts.jsonl --prompts 64 \\
        --cache .cache/activations-4b-layer18.pt \\
        --output artifacts/qwen3-4b-bit-budget-layer18
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

from qwen_j_lens.cross_method import (
    _corpus_prompts,
    collect_activation_signals,
    load_cross_config,
)
from qwen_j_lens.nla.activations import ActivationBatch
from qwen_j_lens.nla.capacity import (
    bit_budget_sweep,
    code_bits,
    linear_reference,
    seed_stability,
    trivial_baselines,
)
from qwen_j_lens.qwen import QwenAdapter

# The primary sweep holds the symbol vocabulary and the decoder width fixed so
# that the only axis is the number of code slots, which is the bit budget.
PRIMARY_SPECS = [(length, 64, 1024) for length in (4, 8, 16, 32, 64, 128, 256)]

# Four shapes that all cost ninety-six bits. If bits were the binding constraint
# these would agree, so any disagreement rules capacity out as the explanation.
EQUAL_BIT_SPECS = [(16, 64, 1024), (24, 16, 1024), (48, 4, 1024), (12, 256, 1024)]

# One bit budget across three decoder widths, which separates the code from the
# network that reads it.
WIDTH_SPECS = [(32, 64, 256), (32, 64, 1024), (32, 64, 4096)]

# A rank-k float32 linear code costs thirty-two bits per component, which puts
# the continuous reference on the same axis as the discrete one.
BITS_PER_COMPONENT = 32
REFERENCE_RANKS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# Configurations carried into the seed controls, chosen to span the frontier.
SEED_CASES = {
    "published_24_bits": (4, 64, 256),
    "wider_96_bits": (16, 64, 1024),
    "wider_192_bits": (32, 64, 1024),
    "wide_decoder_192_bits": (32, 64, 4096),
}


def collect(config_path: Path, corpus: Path, prompts: int, cache: Path) -> dict:
    """Collect activation rows from the real model, or reuse the cache."""
    if cache.exists():
        return torch.load(cache, map_location="cpu")
    cfg = load_cross_config(str(config_path))
    adapter = QwenAdapter.from_pretrained(
        cfg.model.model_id,
        revision=cfg.model.revision,
        dtype=cfg.model.dtype,
        device_map=cfg.model.device_map,
        local_files_only=True,
    )
    token_ids = [adapter.single_token_id(token) for token in cfg.tokens]
    selected = _corpus_prompts(corpus, prompts)
    rows: list[Tensor] = []
    logits: list[Tensor] = []
    started = time.time()
    for index, prompt in enumerate(selected, start=1):
        input_ids, attention_mask = adapter.tokenize_raw([prompt])
        batch = ActivationBatch(input_ids=input_ids, attention_mask=attention_mask)
        activation, logit = collect_activation_signals(
            adapter,
            [batch],
            nla_layer=cfg.hidden_state_index,
            token_ids=token_ids,
        )
        rows.append(activation)
        logits.append(logit)
        if index % 8 == 0:
            total = sum(int(item.shape[0]) for item in rows)
            elapsed = (time.time() - started) / 60.0
            print(f"{index}/{len(selected)} prompts, {total} rows, {elapsed:.1f} min")
    payload = {
        "acts": torch.cat(rows),
        "logits": torch.cat(logits),
        "tokens": cfg.tokens,
        "model_id": cfg.model.model_id,
        "revision": cfg.model.revision,
        "hidden_state_index": cfg.hidden_state_index,
        "prompts": len(selected),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cross-4b-dev.toml"))
    parser.add_argument(
        "--corpus", type=Path, default=Path("data/publication_calibration_prompts.jsonl")
    )
    parser.add_argument("--prompts", type=int, default=64)
    parser.add_argument(
        "--cache", type=Path, default=Path(".cache/activations-4b-layer18.pt")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/qwen3-4b-bit-budget-layer18")
    )
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--stability-seeds", type=int, default=20)
    args = parser.parse_args()

    payload = collect(args.config, args.corpus, args.prompts, args.cache)
    activations = payload["acts"].to(torch.float32)
    print(f"rows={activations.shape[0]} hidden={activations.shape[1]}")

    train = dict(
        embedding_dim=32,
        temperature=1.0,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
    )
    baselines = trivial_baselines(activations)
    # Anything at or above the mean predictor has learned nothing per row, so it
    # is the threshold that separates a collapsed fit from an informative one.
    collapse_threshold = baselines.mean_predictor
    print(f"mean_predictor={collapse_threshold:.4f}")

    reference = [
        {**asdict(point), "bits": BITS_PER_COMPONENT * point.rank}
        for point in linear_reference(activations, REFERENCE_RANKS)
    ]

    def sweep(specs: list[tuple[int, int, int]], entropy_weight: float) -> list[dict]:
        points = bit_budget_sweep(
            activations, specs, entropy_weight=entropy_weight, seed=args.seed, **train
        )
        for point in points:
            print(
                f"  len={point.code_length:4d} vocab={point.code_vocab:4d} "
                f"width={point.width:5d} bits={point.bits:7.1f} "
                f"err={point.relative_error:.4f} codes={point.distinct_codes}"
            )
        return [asdict(point) for point in points]

    seeds = list(range(args.stability_seeds))
    stability: dict[str, dict] = {}
    for name, (length, vocab, width) in SEED_CASES.items():
        result = seed_stability(
            activations,
            seeds,
            code_length=length,
            code_vocab=vocab,
            width=width,
            entropy_weight=0.01,
            **train,
        )
        escapes = [e for e in result.relative_errors if e < collapse_threshold]
        stability[name] = {
            "code_length": length,
            "code_vocab": vocab,
            "width": width,
            "bits": code_bits(length, vocab),
            "escape_count": len(escapes),
            "seed_count": len(seeds),
            "escape_rate": len(escapes) / len(seeds),
            "best_escaped_error": min(escapes) if escapes else None,
            **asdict(result),
        }
        print(
            f"{name}: escapes={len(escapes)}/{len(seeds)} best={result.best:.4f} "
            f"median={result.median:.4f} spread={result.spread:.4f}"
        )

    report = {
        "method": (
            "bit-budget frontier and optimization-stability control for the "
            "discrete natural-language code, measured on real activations with a "
            "principal-component reference converted to bits at thirty-two bits "
            "per retained component"
        ),
        "critique": "arXiv:2607.20379",
        "model_id": payload["model_id"],
        "model_revision": payload["revision"],
        "hidden_state_index": payload["hidden_state_index"],
        "calibration_prompts": int(payload["prompts"]),
        "activation_rows": int(activations.shape[0]),
        "hidden_size": int(activations.shape[1]),
        "device": "cpu",
        "dtype": "bfloat16 weights, float32 analysis",
        "platform": platform.platform(),
        "train": train,
        "seed": args.seed,
        "published_code_bits": code_bits(4, 64),
        "baselines": asdict(baselines),
        "collapse_threshold": collapse_threshold,
        "bits_per_linear_component": BITS_PER_COMPONENT,
        "linear_reference": reference,
        "bit_budget_sweep_entropy_0_01": sweep(PRIMARY_SPECS, 0.01),
        "bit_budget_sweep_entropy_0_10": sweep(PRIMARY_SPECS, 0.10),
        "equal_bit_shape_control": sweep(EQUAL_BIT_SPECS, 0.01),
        "decoder_width_control": sweep(WIDTH_SPECS, 0.01),
        "seed_stability": stability,
        "command": (
            "python scripts/bit_budget_study.py --config configs/cross-4b-dev.toml "
            "--corpus data/publication_calibration_prompts.jsonl --prompts 64"
        ),
        "working_directory": ".",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "bit-budget.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"WROTE {target}")


if __name__ == "__main__":
    main()
