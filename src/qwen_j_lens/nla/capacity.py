"""Capacity diagnostics for the discrete natural-language autoencoder code.

The autoencoder compresses a residual-stream row into `code_length` discrete
slots drawn from a vocabulary of `code_vocab` symbols. The reported metric is a
relative reconstruction error normalized by the mean squared norm of the target,
so a value near zero reads as faithful compression and a value near one reads as
a code that carries no more than a constant predictor would.

That reading is only sound when two conditions hold. The activation sample has
to be diverse enough that reconstructing it requires real information, and the
discrete bottleneck has to use more than a handful of its available codes. This
module measures both conditions directly so a reconstruction number can be
interpreted rather than taken at face value.

The three diagnostics are a pair of trivial predictor baselines, a sweep over
activation-row count at fixed hyperparameters, and a sweep over the entropy
bonus that reports how many distinct codes the trained bottleneck actually uses.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import torch
from torch import Tensor

from qwen_j_lens.nla.train import train_autoencoder


@dataclass(frozen=True)
class CodeUsage:
    """How much of the discrete code space a trained bottleneck occupies."""

    distinct_codes: int
    available_codes: int
    code_entropy: float

    @property
    def occupancy(self) -> float:
        """Fraction of the available code space that the bottleneck uses."""
        return self.distinct_codes / self.available_codes


@dataclass(frozen=True)
class TrivialBaselines:
    """Reconstruction error of predictors that carry no per-row information.

    The zero predictor is exactly one by construction because the metric divides
    by the mean squared norm. The mean predictor is the honest floor, since any
    code that scores above it has learned less than a single constant vector.
    """

    zero_predictor: float
    mean_predictor: float
    constant_energy_fraction: float


@dataclass(frozen=True)
class SweepPoint:
    """One measured configuration in a capacity sweep."""

    rows: int
    entropy_weight: float
    relative_error: float
    distinct_codes: int
    code_entropy: float


def trivial_baselines(activations: Tensor) -> TrivialBaselines:
    """Measure the reconstruction error of the zero and mean predictors.

    A trained code that does not score below `mean_predictor` has extracted no
    per-row information, whatever its absolute value looks like.
    """
    if activations.ndim != 2:
        raise ValueError("Activations must be a two-dimensional row matrix")
    if activations.shape[0] == 0:
        raise ValueError("Activations must contain at least one row")
    target = activations.to(torch.float32)
    energy = target.pow(2).sum(dim=-1).mean()
    if float(energy) == 0.0:
        raise ValueError("Activations carry no energy, so the metric is undefined")
    centered = target - target.mean(dim=0, keepdim=True)
    mean_error = float(centered.pow(2).sum(dim=-1).mean() / energy)
    return TrivialBaselines(
        zero_predictor=1.0,
        mean_predictor=mean_error,
        constant_energy_fraction=1.0 - mean_error,
    )


def count_code_usage(
    model: object,
    activations: Tensor,
    *,
    code_length: int,
    code_vocab: int,
) -> CodeUsage:
    """Count how many distinct discrete codes the model assigns to the rows.

    A collapsed bottleneck assigns nearly every row the same code. When that
    happens the reconstruction cannot depend on the input, so a low error means
    the rows were already close to one another.
    """
    with torch.no_grad():
        sample = model(activations.to(torch.float32))  # type: ignore[operator]
        indices = sample.code_onehot.argmax(dim=-1)
        distinct = {tuple(int(v) for v in row) for row in indices}
        marginal = sample.code_onehot.mean(dim=0).clamp_min(1e-9)
        entropy = float((-(marginal * marginal.log()).sum(dim=-1)).mean())
    return CodeUsage(
        distinct_codes=len(distinct),
        available_codes=code_vocab**code_length,
        code_entropy=entropy,
    )


def _fit(
    activations: Tensor,
    *,
    code_length: int,
    code_vocab: int,
    width: int,
    embedding_dim: int,
    temperature: float,
    steps: int,
    lr: float,
    batch_size: int,
    entropy_weight: float,
    seed: int,
) -> tuple[float, CodeUsage]:
    """Train one autoencoder and return its error together with its code usage."""
    result = train_autoencoder(
        activations,
        code_length=code_length,
        code_vocab=code_vocab,
        width=width,
        embedding_dim=embedding_dim,
        temperature=temperature,
        steps=steps,
        lr=lr,
        batch_size=min(batch_size, activations.shape[0]),
        entropy_weight=entropy_weight,
        seed=seed,
    )
    usage = count_code_usage(
        result.model,
        activations,
        code_length=code_length,
        code_vocab=code_vocab,
    )
    return result.final_relative_error, usage


def row_count_sweep(
    activations: Tensor,
    row_counts: Sequence[int],
    *,
    code_length: int,
    code_vocab: int,
    width: int,
    embedding_dim: int,
    temperature: float,
    steps: int,
    lr: float,
    batch_size: int,
    entropy_weight: float,
    seed: int,
) -> list[SweepPoint]:
    """Refit the autoencoder on growing prefixes of one activation matrix.

    Model, layer, and hyperparameters are held fixed, so the only variable is how
    many rows the code has to cover. A low error that appears only at small row
    counts is a redundancy artifact rather than evidence of compression.
    """
    if activations.ndim != 2:
        raise ValueError("Activations must be a two-dimensional row matrix")
    points: list[SweepPoint] = []
    for count in row_counts:
        if count <= 0:
            raise ValueError("Row counts must be positive")
        if count > activations.shape[0]:
            continue
        error, usage = _fit(
            activations[:count],
            code_length=code_length,
            code_vocab=code_vocab,
            width=width,
            embedding_dim=embedding_dim,
            temperature=temperature,
            steps=steps,
            lr=lr,
            batch_size=batch_size,
            entropy_weight=entropy_weight,
            seed=seed,
        )
        points.append(
            SweepPoint(
                rows=count,
                entropy_weight=entropy_weight,
                relative_error=error,
                distinct_codes=usage.distinct_codes,
                code_entropy=usage.code_entropy,
            )
        )
    return points


def entropy_weight_sweep(
    activations: Tensor,
    entropy_weights: Sequence[float],
    *,
    code_length: int,
    code_vocab: int,
    width: int,
    embedding_dim: int,
    temperature: float,
    steps: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> list[SweepPoint]:
    """Refit the autoencoder across entropy bonuses on a fixed row set.

    The entropy bonus is the only term that opposes code collapse. Sweeping it
    separates a bottleneck that cannot represent the data from one that simply
    never learned to use its symbols.
    """
    if activations.ndim != 2:
        raise ValueError("Activations must be a two-dimensional row matrix")
    points: list[SweepPoint] = []
    for weight in entropy_weights:
        if weight < 0.0:
            raise ValueError("Entropy weights must not be negative")
        error, usage = _fit(
            activations,
            code_length=code_length,
            code_vocab=code_vocab,
            width=width,
            embedding_dim=embedding_dim,
            temperature=temperature,
            steps=steps,
            lr=lr,
            batch_size=batch_size,
            entropy_weight=weight,
            seed=seed,
        )
        points.append(
            SweepPoint(
                rows=activations.shape[0],
                entropy_weight=weight,
                relative_error=error,
                distinct_codes=usage.distinct_codes,
                code_entropy=usage.code_entropy,
            )
        )
    return points


def capacity_report(
    activations: Tensor,
    *,
    row_counts: Sequence[int],
    entropy_weights: Sequence[float],
    code_length: int,
    code_vocab: int,
    width: int,
    embedding_dim: int,
    temperature: float,
    steps: int,
    lr: float,
    batch_size: int,
    entropy_weight: float,
    seed: int,
) -> dict[str, object]:
    """Run every capacity diagnostic and return one serializable report."""
    baselines = trivial_baselines(activations)
    rows = row_count_sweep(
        activations,
        row_counts,
        code_length=code_length,
        code_vocab=code_vocab,
        width=width,
        embedding_dim=embedding_dim,
        temperature=temperature,
        steps=steps,
        lr=lr,
        batch_size=batch_size,
        entropy_weight=entropy_weight,
        seed=seed,
    )
    weights = entropy_weight_sweep(
        activations,
        entropy_weights,
        code_length=code_length,
        code_vocab=code_vocab,
        width=width,
        embedding_dim=embedding_dim,
        temperature=temperature,
        steps=steps,
        lr=lr,
        batch_size=batch_size,
        seed=seed,
    )
    best = min(weights, key=lambda point: point.relative_error) if weights else None
    return {
        "method": (
            "capacity diagnostics for the discrete autoencoder code: trivial "
            "predictor baselines, an activation-row sweep at fixed "
            "hyperparameters, and an entropy-bonus sweep with code occupancy"
        ),
        "activation_rows": int(activations.shape[0]),
        "hidden_size": int(activations.shape[1]),
        "available_codes": code_vocab**code_length,
        "baselines": asdict(baselines),
        "row_count_sweep": [asdict(point) for point in rows],
        "entropy_weight_sweep": [asdict(point) for point in weights],
        "best_entropy_weight": asdict(best) if best is not None else None,
        "beats_mean_predictor": (
            bool(best.relative_error < baselines.mean_predictor)
            if best is not None
            else False
        ),
    }
