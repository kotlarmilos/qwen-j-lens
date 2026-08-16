"""Cross-method agreement between the J-lens and the natural-language autoencoder.

The J-lens paper and the natural-language autoencoder paper each localize a
concept from the same residual stream, but they ask different questions. The
J-lens asks whether a concept is present as a clean linear direction that reads
the model's own next-token logit. The autoencoder asks whether a concept
survives a discrete verbalize-and-reconstruct round trip. Neither paper measures
whether the two methods agree on which concepts are recoverable at a matched
layer. This module runs both methods on the same physical residual stream and
reports per-concept agreement.

For a controlled token ``t`` the ground-truth signal at each attended position is
the model's own next-token logit for ``t``. The J-lens fidelity is the held-out
coefficient of determination of that logit from the single stored J-lens
direction. The autoencoder retention is the fraction of that logit's linear
recoverability that survives the reconstruction, reusing the same gist probe the
faithfulness module already defines. The two scores measure the two properties
the J-lens article names, so a per-concept scatter separates concepts that both
methods recover, concepts that only one method recovers, and concepts that
neither recovers at that layer.

A concept that scores high on the J-lens and low on the autoencoder is written in
output coordinates that a description bottleneck drops. A concept that scores low
on the J-lens and high on the autoencoder is carried in a form the reconstruction
keeps without a clean linear readout. Both cases are informative, and neither is
visible from one method alone.
"""

from __future__ import annotations

import argparse
import json
import random
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from qwen_j_lens.adapter import ModelAdapter
from qwen_j_lens.artifacts import ArtifactStore
from qwen_j_lens.jacobian import estimate_token_vjps
from qwen_j_lens.nla.activations import ActivationBatch
from qwen_j_lens.nla.config import NLAAutoencoderConfig, NLAModelConfig, NLATrainConfig
from qwen_j_lens.nla.faithfulness import gist_versus_specific
from qwen_j_lens.nla.train import train_autoencoder
from qwen_j_lens.qwen import QwenAdapter


@dataclass(frozen=True)
class CrossMethodConfig:
    """Configuration for a matched-layer cross-method agreement run."""

    name: str
    seed: int
    model: NLAModelConfig
    hidden_state_index: int
    corpus_path: Path
    max_prompts: int
    batch_size: int
    tokens: tuple[str, ...]
    include_future: bool
    vjp_chunk_size: int
    autoencoder: NLAAutoencoderConfig
    train: NLATrainConfig
    test_fraction: float
    fidelity_threshold: float
    retention_threshold: float


@dataclass(frozen=True)
class ConceptAgreement:
    """Per-concept scores from both methods at one matched layer."""

    token: str
    token_id: int
    jlens_fidelity: float
    nla_original_r2: float
    nla_reconstructed_r2: float
    nla_retention: float
    quadrant: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "token_id": self.token_id,
            "jlens_fidelity": self.jlens_fidelity,
            "nla_original_r2": self.nla_original_r2,
            "nla_reconstructed_r2": self.nla_reconstructed_r2,
            "nla_retention": self.nla_retention,
            "quadrant": self.quadrant,
        }


@dataclass(frozen=True)
class AgreementReport:
    """Aggregated cross-method agreement over the controlled concept bank."""

    hidden_state_index: int
    jlens_capture_layer: int
    activation_rows: int
    train_rows: int
    test_rows: int
    reconstruction_error: float
    pair_count: int
    context_count: int
    fidelity_threshold: float
    retention_threshold: float
    mean_jlens_fidelity: float
    mean_nla_retention: float
    spearman_rho: float
    pearson_r: float
    quadrant_counts: Mapping[str, int]
    concepts: tuple[ConceptAgreement, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "hidden_state_index": self.hidden_state_index,
            "jlens_capture_layer": self.jlens_capture_layer,
            "activation_rows": self.activation_rows,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "reconstruction_error": self.reconstruction_error,
            "pair_count": self.pair_count,
            "context_count": self.context_count,
            "thresholds": {
                "fidelity": self.fidelity_threshold,
                "retention": self.retention_threshold,
            },
            "mean_jlens_fidelity": self.mean_jlens_fidelity,
            "mean_nla_retention": self.mean_nla_retention,
            "agreement": {
                "spearman_rho": self.spearman_rho,
                "pearson_r": self.pearson_r,
            },
            "quadrant_counts": dict(self.quadrant_counts),
            "concepts": [concept.to_mapping() for concept in self.concepts],
            "method": (
                "matched-layer per-concept agreement between the linear J-lens "
                "readout and the natural-language autoencoder gist probe"
            ),
        }


QUADRANTS = (
    "both_recoverable",
    "jlens_only",
    "nla_only",
    "neither",
)


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


def _split_indices(count: int, test_fraction: float) -> tuple[Tensor, Tensor]:
    if count < 4:
        raise ValueError("At least four activations are required to audit a split")
    test_size = max(1, int(round(count * test_fraction)))
    test_size = min(test_size, count - 1)
    train_index = torch.arange(0, count - test_size)
    test_index = torch.arange(count - test_size, count)
    return train_index, test_index


def collect_activation_signals(
    adapter: ModelAdapter,
    batches: Sequence[ActivationBatch],
    *,
    nla_layer: int,
    token_ids: Sequence[int],
) -> tuple[Tensor, Tensor]:
    """Return activations and model concept logits at ``hidden_states[nla_layer]``.

    Each attended position contributes one activation row and one row of
    next-token logits restricted to ``token_ids``. Padding positions never enter
    the returned tensors, so the two outputs share row order.
    """
    ids = torch.as_tensor([int(value) for value in token_ids], dtype=torch.long)
    activation_rows: list[Tensor] = []
    logit_rows: list[Tensor] = []
    for batch in batches:
        if batch.input_ids.ndim != 2:
            raise ValueError("Activation batches must be two-dimensional token grids")
        if batch.attention_mask.shape != batch.input_ids.shape:
            raise ValueError("Attention mask shape must match the input ids")
        with torch.no_grad():
            run = adapter.forward(
                batch.input_ids,
                batch.attention_mask,
                output_hidden_states=True,
            )
        if run.hidden_states is None:
            raise ValueError("Adapter did not return hidden states")
        if nla_layer < 0 or nla_layer >= len(run.hidden_states):
            raise IndexError(
                f"Layer {nla_layer} is not present in the returned hidden states"
            )
        hidden = run.hidden_states[nla_layer]
        concept_logits = run.logits.index_select(-1, ids.to(run.logits.device))
        mask = batch.attention_mask.to(dtype=torch.bool)
        selected_hidden = hidden[mask]
        selected_logits = concept_logits[mask]
        if selected_hidden.numel() > 0:
            activation_rows.append(
                selected_hidden.detach().to(device="cpu", dtype=torch.float32)
            )
            logit_rows.append(
                selected_logits.detach().to(device="cpu", dtype=torch.float32)
            )
    if not activation_rows:
        raise ValueError("No attended positions were found across the batches")
    # Rows move to CPU so the downstream autoencoder training and agreement
    # scoring run on one device regardless of where the model is resident. The
    # autoencoder draws batches from a CPU generator, so keeping the analysis on
    # CPU also keeps an accelerator run comparable with a CPU run.
    return torch.cat(activation_rows, dim=0), torch.cat(logit_rows, dim=0)


def jlens_linear_readout(activations: Tensor, lens_vectors: Tensor) -> Tensor:
    """Compute the linear J-lens score ``s_t = <v_t, h>`` for every concept.

    The result has shape ``[rows, concepts]``. Column ``t`` is the linear J-lens
    readout for concept ``t`` before the final normalization, matching the stored
    controlled-token direction ``v_t`` which is a row of ``W_U J_l``.
    """
    return activations.to(torch.float32) @ lens_vectors.to(torch.float32).transpose(0, 1)


def _probe_r2(
    inputs_train: Tensor,
    targets_train: Tensor,
    inputs_test: Tensor,
    targets_test: Tensor,
) -> float:
    """Held-out coefficient of determination for a least-squares linear probe.

    This matches ``qwen_j_lens.nla.faithfulness`` so the J-lens fidelity and the
    autoencoder gist probe are scored on the same footing.
    """
    ones_train = torch.ones((inputs_train.shape[0], 1))
    ones_test = torch.ones((inputs_test.shape[0], 1))
    design_train = torch.cat((inputs_train, ones_train), dim=1)
    design_test = torch.cat((inputs_test, ones_test), dim=1)
    solution = torch.linalg.lstsq(design_train, targets_train).solution
    prediction = design_test @ solution
    residual = (targets_test - prediction).pow(2).sum()
    total = (targets_test - targets_test.mean(dim=0, keepdim=True)).pow(2).sum()
    if float(total) <= 0.0:
        return 0.0
    return float(1.0 - residual / total)


def _ranks(values: Sequence[float]) -> list[float]:
    """Average ranks with tie handling, used for the Spearman correlation."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(values):
        end = position
        while (
            end + 1 < len(values)
            and values[order[end + 1]] == values[order[position]]
        ):
            end += 1
        average = (position + end) / 2.0
        for index in range(position, end + 1):
            ranks[order[index]] = average
        position = end + 1
    return ranks


def _pearson(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second):
        raise ValueError("Correlation inputs must have equal length")
    if len(first) < 2:
        return 0.0
    a = torch.tensor(first, dtype=torch.float64)
    b = torch.tensor(second, dtype=torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denominator = a.norm() * b.norm()
    if float(denominator) <= 0.0:
        return 0.0
    return float((a @ b) / denominator)


def spearman_rho(first: Sequence[float], second: Sequence[float]) -> float:
    """Spearman rank correlation implemented as the Pearson correlation of ranks."""
    return _pearson(_ranks(list(first)), _ranks(list(second)))


def pearson_r(first: Sequence[float], second: Sequence[float]) -> float:
    """Pearson correlation between two equal-length score sequences."""
    return _pearson(list(first), list(second))


def _quadrant(
    fidelity: float,
    retention: float,
    *,
    fidelity_threshold: float,
    retention_threshold: float,
) -> str:
    high_fidelity = fidelity >= fidelity_threshold
    high_retention = retention >= retention_threshold
    if high_fidelity and high_retention:
        return "both_recoverable"
    if high_fidelity and not high_retention:
        return "jlens_only"
    if not high_fidelity and high_retention:
        return "nla_only"
    return "neither"


def concept_agreement(
    activations: Tensor,
    concept_logits: Tensor,
    lens_vectors: Tensor,
    reconstructions: Tensor,
    *,
    tokens: Sequence[str],
    token_ids: Sequence[int],
    train_index: Tensor,
    test_index: Tensor,
    hidden_state_index: int,
    jlens_capture_layer: int,
    reconstruction_error: float,
    pair_count: int,
    context_count: int,
    fidelity_threshold: float,
    retention_threshold: float,
) -> AgreementReport:
    """Score both methods per concept and summarize their agreement.

    ``activations`` and ``reconstructions`` are the original and round-tripped
    residual rows at the matched layer. ``concept_logits`` holds the model's own
    next-token logit for each concept at every row. ``lens_vectors`` are the
    stored J-lens directions for the same concepts at the same layer.
    """
    if activations.ndim != 2 or reconstructions.shape != activations.shape:
        raise ValueError("Activations and reconstructions must share a 2-D shape")
    concept_count = len(tokens)
    if concept_count != len(token_ids):
        raise ValueError("tokens and token_ids must have equal length")
    if concept_logits.shape != (activations.shape[0], concept_count):
        raise ValueError("concept_logits must have shape [rows, concepts]")
    if lens_vectors.shape != (concept_count, activations.shape[1]):
        raise ValueError("lens_vectors must have shape [concepts, hidden]")

    readout = jlens_linear_readout(activations, lens_vectors)
    concepts: list[ConceptAgreement] = []
    fidelities: list[float] = []
    retentions: list[float] = []
    quadrant_counts = {name: 0 for name in QUADRANTS}
    for index, token in enumerate(tokens):
        target = concept_logits[:, index : index + 1]
        feature = readout[:, index : index + 1]
        raw_fidelity = _probe_r2(
            feature.index_select(0, train_index),
            target.index_select(0, train_index),
            feature.index_select(0, test_index),
            target.index_select(0, test_index),
        )
        fidelity = max(0.0, raw_fidelity)
        gist = gist_versus_specific(
            activations,
            reconstructions,
            target.squeeze(-1),
            train_index=train_index,
            test_index=test_index,
        )
        retention = max(0.0, gist["specific_retained_fraction"])
        quadrant = _quadrant(
            fidelity,
            retention,
            fidelity_threshold=fidelity_threshold,
            retention_threshold=retention_threshold,
        )
        quadrant_counts[quadrant] += 1
        fidelities.append(fidelity)
        retentions.append(retention)
        concepts.append(
            ConceptAgreement(
                token=token,
                token_id=int(token_ids[index]),
                jlens_fidelity=fidelity,
                nla_original_r2=gist["original_specific_r2"],
                nla_reconstructed_r2=gist["reconstructed_specific_r2"],
                nla_retention=retention,
                quadrant=quadrant,
            )
        )

    mean_fidelity = float(sum(fidelities) / concept_count)
    mean_retention = float(sum(retentions) / concept_count)
    return AgreementReport(
        hidden_state_index=hidden_state_index,
        jlens_capture_layer=jlens_capture_layer,
        activation_rows=int(activations.shape[0]),
        train_rows=int(train_index.shape[0]),
        test_rows=int(test_index.shape[0]),
        reconstruction_error=float(reconstruction_error),
        pair_count=int(pair_count),
        context_count=int(context_count),
        fidelity_threshold=fidelity_threshold,
        retention_threshold=retention_threshold,
        mean_jlens_fidelity=mean_fidelity,
        mean_nla_retention=mean_retention,
        spearman_rho=spearman_rho(fidelities, retentions),
        pearson_r=pearson_r(fidelities, retentions),
        quadrant_counts=quadrant_counts,
        concepts=tuple(concepts),
    )


def execute_cross_method(
    adapter: ModelAdapter,
    config: CrossMethodConfig,
    output: Path,
) -> AgreementReport:
    """Run both methods on the injected adapter and write the agreement artifact.

    The adapter is injected so the whole pipeline runs offline against the tiny
    local transformer during tests and against an open-weight model on hardware.
    """
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.hidden_state_index < 1:
        raise ValueError(
            "hidden_state_index must be at least 1 so a J-lens capture layer exists"
        )
    capture_layer = config.hidden_state_index - 1
    prompts = _corpus_prompts(config.corpus_path, config.max_prompts)
    activation_batches: list[ActivationBatch] = []
    vjp_batches: list[tuple[Tensor, Tensor]] = []
    for offset in range(0, len(prompts), config.batch_size):
        input_ids, attention_mask = adapter.tokenize_raw(
            prompts[offset : offset + config.batch_size]
        )
        activation_batches.append(
            ActivationBatch(input_ids=input_ids, attention_mask=attention_mask)
        )
        vjp_batches.append((input_ids, attention_mask))
    token_ids = [adapter.single_token_id(token) for token in config.tokens]

    estimate = estimate_token_vjps(
        adapter,
        vjp_batches,
        layer_index=capture_layer,
        token_ids=token_ids,
        include_future=config.include_future,
        vjp_chunk_size=config.vjp_chunk_size,
    )
    lens_vectors = estimate.values.to(device="cpu", dtype=torch.float32)

    activations, concept_logits = collect_activation_signals(
        adapter,
        activation_batches,
        nla_layer=config.hidden_state_index,
        token_ids=token_ids,
    )

    trained = train_autoencoder(
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
        reconstructions = trained.model(activations).reconstruction

    train_index, test_index = _split_indices(
        activations.shape[0], config.test_fraction
    )
    report = concept_agreement(
        activations,
        concept_logits,
        lens_vectors,
        reconstructions,
        tokens=config.tokens,
        token_ids=token_ids,
        train_index=train_index,
        test_index=test_index,
        hidden_state_index=config.hidden_state_index,
        jlens_capture_layer=capture_layer,
        reconstruction_error=trained.final_relative_error,
        pair_count=estimate.pair_count,
        context_count=estimate.context_count,
        fidelity_threshold=config.fidelity_threshold,
        retention_threshold=config.retention_threshold,
    )

    store = ArtifactStore(output)
    agreement_record = store.write_json("agreement.json", report.to_mapping())
    history_record = store.write_json("training-history.json", trained.history)
    run_record = store.write_json(
        "run.json",
        {
            "config": {
                "name": config.name,
                "seed": config.seed,
                "model": asdict(config.model),
                "hidden_state_index": config.hidden_state_index,
                "jlens_capture_layer": capture_layer,
                "corpus_path": str(config.corpus_path),
                "max_prompts": config.max_prompts,
                "batch_size": config.batch_size,
                "tokens": list(config.tokens),
                "include_future": config.include_future,
                "vjp_chunk_size": config.vjp_chunk_size,
                "autoencoder": asdict(config.autoencoder),
                "train": asdict(config.train),
                "test_fraction": config.test_fraction,
                "fidelity_threshold": config.fidelity_threshold,
                "retention_threshold": config.retention_threshold,
            },
            "model_id": adapter.model_id,
            "activation_rows": report.activation_rows,
            "method": (
                "estimate matched-layer J-lens directions and autoencoder "
                "reconstructions, then compare per-concept recoverability of the "
                "model's own next-token logit"
            ),
            "jlens_source": "https://transformer-circuits.pub/2026/workspace/index.html",
            "nla_source": "https://transformer-circuits.pub/2026/nla/index.html",
        },
    )
    store.write_records(
        "records.json", [agreement_record, history_record, run_record]
    )
    return report


def run_cross_method(
    config: CrossMethodConfig,
    output: Path,
    *,
    allow_download: bool,
) -> AgreementReport:
    """Load the configured model, then run the cross-method agreement pipeline."""
    adapter = QwenAdapter.from_pretrained(
        config.model.model_id,
        local_files_only=False if allow_download else True,
        revision=config.model.revision,
        device_map=config.model.device_map,
        dtype=config.model.dtype,
        enable_thinking=config.model.enable_thinking,
    )
    return execute_cross_method(adapter, config, output)


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    child = value.get(key)
    if not isinstance(child, Mapping):
        raise ValueError(f"Missing [{key}] table")
    return child


def _positive_int(value: Mapping[str, Any], key: str, default: int | None = None) -> int:
    raw = value.get(key, default)
    if raw is None:
        raise ValueError(f"{key} is required")
    number = int(raw)
    if number <= 0:
        raise ValueError(f"{key} must be positive")
    return number


def _positive_float(value: Mapping[str, Any], key: str, default: float | None = None) -> float:
    raw = value.get(key, default)
    if raw is None:
        raise ValueError(f"{key} is required")
    number = float(raw)
    if number <= 0.0:
        raise ValueError(f"{key} must be positive")
    return number


def _unit_float(value: Mapping[str, Any], key: str, default: float) -> float:
    number = float(value.get(key, default))
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{key} must be within [0, 1]")
    return number


def load_cross_config(path: str | Path) -> CrossMethodConfig:
    """Read and validate a cross-method agreement configuration."""
    config_path = Path(path)
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)
    model = _mapping(raw, "model")
    activations = _mapping(raw, "activations")
    concepts = _mapping(raw, "concepts")
    jlens = _mapping(raw, "jlens")
    autoencoder = _mapping(raw, "autoencoder")
    train = _mapping(raw, "train")
    agreement = _mapping(raw, "agreement")

    corpus = Path(str(activations["corpus_path"]))
    if not corpus.is_absolute():
        corpus = (config_path.parent.parent / corpus).resolve()

    hidden_state_index = int(activations.get("hidden_state_index", -1))
    if hidden_state_index < 1:
        raise ValueError("activations.hidden_state_index must be at least 1")

    tokens = tuple(str(token) for token in concepts.get("tokens", ()))
    if not tokens:
        raise ValueError("concepts.tokens must contain at least one token")

    test_fraction = float(agreement.get("test_fraction", 0.5))
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("agreement.test_fraction must be strictly between 0 and 1")

    return CrossMethodConfig(
        name=str(raw.get("name", config_path.stem)),
        seed=int(raw.get("seed", 0)),
        model=NLAModelConfig(
            model_id=str(model["id"]),
            revision=(str(model["revision"]) if model.get("revision") is not None else None),
            local_files_only=bool(model.get("local_files_only", True)),
            dtype=str(model.get("dtype", "auto")),
            device_map=(str(model["device_map"]) if model.get("device_map") is not None else None),
            enable_thinking=bool(model.get("enable_thinking", False)),
        ),
        hidden_state_index=hidden_state_index,
        corpus_path=corpus,
        max_prompts=_positive_int(activations, "max_prompts"),
        batch_size=_positive_int(activations, "batch_size", 1),
        tokens=tokens,
        include_future=bool(jlens.get("include_future", True)),
        vjp_chunk_size=_positive_int(jlens, "vjp_chunk_size", 32),
        autoencoder=NLAAutoencoderConfig(
            code_length=_positive_int(autoencoder, "code_length"),
            code_vocab=_positive_int(autoencoder, "code_vocab"),
            width=_positive_int(autoencoder, "width"),
            embedding_dim=_positive_int(autoencoder, "embedding_dim"),
            temperature=_positive_float(autoencoder, "temperature", 1.0),
        ),
        train=NLATrainConfig(
            steps=_positive_int(train, "steps"),
            lr=_positive_float(train, "lr"),
            batch_size=_positive_int(train, "batch_size"),
            entropy_weight=float(train.get("entropy_weight", 0.0)),
        ),
        test_fraction=test_fraction,
        fidelity_threshold=_unit_float(agreement, "fidelity_threshold", 0.5),
        retention_threshold=_unit_float(agreement, "retention_threshold", 0.5),
    )


def _summarize_config(config: CrossMethodConfig) -> dict[str, object]:
    prompts = _corpus_prompts(config.corpus_path, config.max_prompts)
    return {
        "name": config.name,
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "hidden_state_index": config.hidden_state_index,
        "jlens_capture_layer": config.hidden_state_index - 1,
        "controlled_token_count": len(config.tokens),
        "corpus_prompt_count": len(prompts),
        "fidelity_threshold": config.fidelity_threshold,
        "retention_threshold": config.retention_threshold,
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="qwen-cross")
    commands = root.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-cross-config")
    validate.add_argument("--config", type=Path, required=True)
    run = commands.add_parser("cross-run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit Hugging Face downloads. Omit this flag for cache-only loading.",
    )
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_cross_config(args.config)
    if args.command == "validate-cross-config":
        print(json.dumps(_summarize_config(config), indent=2, sort_keys=True))
        return 0
    if args.command == "cross-run":
        report = run_cross_method(
            config,
            args.output,
            allow_download=args.allow_download,
        )
        print(
            json.dumps(
                {
                    "hidden_state_index": report.hidden_state_index,
                    "activation_rows": report.activation_rows,
                    "mean_jlens_fidelity": report.mean_jlens_fidelity,
                    "mean_nla_retention": report.mean_nla_retention,
                    "spearman_rho": report.spearman_rho,
                    "quadrant_counts": dict(report.quadrant_counts),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
