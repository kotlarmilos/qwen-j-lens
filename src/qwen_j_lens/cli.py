"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from qwen_j_lens.artifacts import (
    ArtifactStore,
    capture_source_state,
    portable_path,
    validate_artifact_tree,
)
from qwen_j_lens.config import RunConfig, load_config
from qwen_j_lens.jacobian import estimate_token_vjps
from qwen_j_lens.merge import merge_estimate_shards
from qwen_j_lens.evaluation import (
    evaluate_examples,
    load_examples,
    summarize_results,
    validate_result_mapping,
)
from qwen_j_lens.qwen import QwenAdapter, load_qwen_tokenizer


def _prompts(path: Path, limit: int, *, offset: int = 0) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Calibration corpus is missing: {path}")
    prompts = [
        str(json.loads(line)["text"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts:
        raise ValueError("Calibration corpus contains no prompts")
    if offset < 0:
        raise ValueError("Calibration prompt offset must be nonnegative")
    selected = prompts[offset : offset + limit]
    if not selected:
        raise ValueError(
            f"Calibration prompt offset {offset} is outside a {len(prompts)}-prompt corpus"
        )
    return selected


def _validate(config: RunConfig) -> dict[str, object]:
    prompts = _prompts(config.estimation.calibration_path, config.estimation.max_prompts)
    return {
        "name": config.name,
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "local_files_only": config.model.local_files_only,
        "enable_thinking": config.model.enable_thinking,
        "layers": list(config.estimation.layers),
        "controlled_token_count": len(config.estimation.controlled_tokens),
        "calibration_prompt_count": len(prompts),
        "evaluation_prompt_path": str(config.evaluation.prompts_path),
        "evaluation_layer": config.evaluation.layer,
    }


def run_estimation(
    config: RunConfig,
    output: Path,
    *,
    allow_download: bool = False,
    layers: Sequence[int] | None = None,
    prompt_offset: int = 0,
    max_prompts: int | None = None,
) -> None:
    """Estimate controlled rows of W_U J_l directly with selected-token VJPs."""
    source = capture_source_state()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    adapter = QwenAdapter.from_pretrained(
        config.model.model_id,
        local_files_only=False if allow_download else True,
        revision=config.model.revision,
        device_map=config.model.device_map,
        dtype=config.model.dtype,
        enable_thinking=config.model.enable_thinking,
    )
    prompt_limit = (
        config.estimation.max_prompts if max_prompts is None else max_prompts
    )
    if prompt_limit <= 0 or prompt_limit > config.estimation.max_prompts:
        raise ValueError(
            "max_prompts must be positive and no larger than the configured limit"
        )
    prompts = _prompts(
        config.estimation.calibration_path,
        prompt_limit,
        offset=prompt_offset,
    )
    token_ids = [
        adapter.single_token_id(token)
        for token in config.estimation.controlled_tokens
    ]
    store = ArtifactStore(output)
    selected_layers = (
        config.estimation.layers if layers is None else tuple(dict.fromkeys(layers))
    )
    if not selected_layers:
        raise ValueError("At least one estimation layer is required")
    if any(layer not in config.estimation.layers for layer in selected_layers):
        raise ValueError("Selected layers must be declared in estimation.layers")
    records = []
    layer_metadata = []
    for layer in selected_layers:
        if layer >= len(adapter.layers):
            raise IndexError(f"Layer {layer} does not exist in {adapter.model_id}")
        batches = []
        for offset in range(0, len(prompts), config.estimation.batch_size):
            input_ids, attention_mask = adapter.tokenize_raw(
                prompts[offset : offset + config.estimation.batch_size]
            )
            batches.append((input_ids, attention_mask))
        estimate = estimate_token_vjps(
            adapter,
            batches,
            layer_index=layer,
            token_ids=token_ids,
            include_future=config.estimation.include_future,
            vjp_chunk_size=config.estimation.vjp_chunk_size,
        )
        vector_record = store.write_array(
            f"controlled-token-vectors-layer-{layer:03d}.npy",
            estimate.values,
            metadata={
                "layer": layer,
                "token_ids": token_ids,
                "tokens": list(config.estimation.controlled_tokens),
                "construction": "selected rows of W_U J_l estimated directly by VJP",
                "pair_count": estimate.pair_count,
                "context_count": estimate.context_count,
            },
        )
        records.append(vector_record)
        layer_metadata.append(
            {
                "layer": layer,
                "controlled_token_vectors": vector_record.path,
                "pair_count": estimate.pair_count,
                "context_count": estimate.context_count,
            }
        )
    records.append(
        store.write_json(
            "run.json",
            {
                "config": {
                    "name": config.name,
                    "seed": config.seed,
                    "model": asdict(config.model),
                    "estimation": {
                        **asdict(config.estimation),
                        "calibration_path": portable_path(
                            config.estimation.calibration_path
                        ),
                    },
                    "evaluation": {
                        **asdict(config.evaluation),
                        "prompts_path": portable_path(
                            config.evaluation.prompts_path
                        ),
                    },
                },
                "model_id": adapter.model_id,
                "model_revision": config.model.revision,
                "source_code": source,
                "prompt_shard": {
                    "offset": prompt_offset,
                    "count": len(prompts),
                },
                "layers": layer_metadata,
                "method": (
                    "corpus-averaged selected-token VJPs from the final "
                    "pre-normalization residual"
                ),
                "controlled_token_vectors": (
                    "selected rows of W_U J_l without materializing full J_l"
                ),
                "final_normalization": "applied only after J_l h_l during lens readout",
                "source": "https://transformer-circuits.pub/2026/workspace/index.html",
            },
        )
    )
    store.write_records("records.json", records)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="qwen-j-lens")
    commands = root.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-config")
    validate.add_argument("--config", type=Path, required=True)
    estimate = commands.add_parser("estimate")
    estimate.add_argument("--config", type=Path, required=True)
    estimate.add_argument("--output", type=Path, required=True)
    estimate.add_argument(
        "--layer",
        type=int,
        action="append",
        help="Estimate one declared layer. Repeat to select multiple layers.",
    )
    estimate.add_argument("--prompt-offset", type=int, default=0)
    estimate.add_argument(
        "--max-prompts",
        type=int,
        help="Process one shard no larger than estimation.max_prompts.",
    )
    estimate.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit Hugging Face downloads. Omit this flag for cache-only loading.",
    )
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--lens-vectors", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument(
        "--layer",
        type=int,
        help="Evaluate one declared layer instead of evaluation.layer.",
    )
    evaluate.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit Hugging Face downloads. Omit this flag for cache-only loading.",
    )
    tokens = commands.add_parser("validate-tokens")
    tokens.add_argument("--config", type=Path, required=True)
    tokens.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit tokenizer downloads. Omit this flag for cache-only loading.",
    )
    merge = commands.add_parser("merge-estimates")
    merge.add_argument("--config", type=Path, required=True)
    merge.add_argument("--input", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--layer", type=int, required=True)
    verify = commands.add_parser("verify-artifacts")
    verify.add_argument("--root", type=Path, default=Path("artifacts"))
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "verify-artifacts":
        valid, message = validate_artifact_tree(args.root)
        if not valid:
            raise ValueError(message)
        print(message)
        return 0

    config = load_config(args.config)
    if args.command == "validate-config":
        print(json.dumps(_validate(config), indent=2, sort_keys=True))
        return 0
    if args.command == "validate-tokens":
        tokenizer = load_qwen_tokenizer(
            config.model.model_id,
            local_files_only=False if args.allow_download else True,
            revision=config.model.revision,
        )
        token_rows = []
        for token in config.estimation.controlled_tokens:
            token_ids = tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(
                    f"Controlled token `{token}` encodes to {len(token_ids)} tokens, "
                    "expected one"
                )
            token_rows.append({"token": token, "token_id": int(token_ids[0])})
        print(
            json.dumps(
                {
                    "model_id": config.model.model_id,
                    "revision": config.model.revision,
                    "controlled_tokens": token_rows,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "merge-estimates":
        merge_estimate_shards(
            config,
            args.input,
            args.output,
            layer=args.layer,
        )
        return 0
    if args.command == "estimate":
        run_estimation(
            config,
            args.output,
            allow_download=args.allow_download,
            layers=args.layer,
            prompt_offset=args.prompt_offset,
            max_prompts=args.max_prompts,
        )
        return 0
    if args.command == "evaluate":
        source = capture_source_state()
        adapter = QwenAdapter.from_pretrained(
            config.model.model_id,
            local_files_only=False if args.allow_download else True,
            revision=config.model.revision,
            device_map=config.model.device_map,
            dtype=config.model.dtype,
            enable_thinking=config.model.enable_thinking,
        )
        vectors = torch.from_numpy(np.load(args.lens_vectors, allow_pickle=False))
        if vectors.shape[0] != len(config.estimation.controlled_tokens):
            raise ValueError("Lens vector rows do not match configured controlled tokens")
        if vectors.ndim != 2 or vectors.shape[1] != adapter.hidden_size:
            raise ValueError("Lens vector width does not match the configured model")
        layer_index = config.evaluation.layer if args.layer is None else args.layer
        if layer_index not in config.estimation.layers:
            raise ValueError("Evaluation layer must be declared in estimation.layers")
        examples = load_examples(config.evaluation.prompts_path)
        vector_map = {
            token: vectors[index]
            for index, token in enumerate(config.estimation.controlled_tokens)
        }
        jacobian_results = evaluate_examples(
            adapter,
            examples,
            vector_map,
            layer_index=layer_index,
            seeds=config.evaluation.seeds,
            max_new_tokens=config.evaluation.max_new_tokens,
            temperature=config.evaluation.temperature,
            lens="jacobian",
        )
        token_ids = [
            adapter.single_token_id(token)
            for token in config.estimation.controlled_tokens
        ]
        logit_vectors = adapter.unembedding_weight(token_ids).detach().to(
            dtype=torch.float32, device="cpu"
        )
        logit_results = evaluate_examples(
            adapter,
            examples,
            {
                token: logit_vectors[index]
                for index, token in enumerate(config.estimation.controlled_tokens)
            },
            layer_index=layer_index,
            seeds=config.evaluation.seeds,
            max_new_tokens=config.evaluation.max_new_tokens,
            temperature=config.evaluation.temperature,
            lens="logit",
        )
        results = jacobian_results + logit_results
        store = ArtifactStore(args.output)
        mappings = [result.to_mapping() for result in results]
        for mapping in mappings:
            validate_result_mapping(mapping)
        lines = [
            json.dumps(mapping, sort_keys=True, ensure_ascii=False)
            for mapping in mappings
        ]
        result_record = store.write_bytes(
            "results.jsonl",
            ("\n".join(lines) + "\n").encode("utf-8"),
            kind="jsonl",
        )
        summary_record = store.write_json(
            "summary.json", summarize_results(results)
        )
        run_record = store.write_json(
            "evaluation-run.json",
            {
                "config": config.name,
                "model_id": config.model.model_id,
                "model_revision": config.model.revision,
                "layer": layer_index,
                "jacobian_vectors": portable_path(args.lens_vectors),
                "source_code": source,
                "baselines": ["jacobian", "logit"],
                "enable_thinking": config.model.enable_thinking,
                "kv_cache": True,
            },
        )
        store.write_records(
            "records.json", [result_record, summary_record, run_record]
        )
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
