"""Seed and code-shape diagnostics for the fixed-symbol autoencoder.

The autoencoder compresses a residual-stream row into `code_length` discrete
slots drawn from a vocabulary of `code_vocab` symbols. The reported metric is a
relative reconstruction error normalized by centered activation energy. A value
near zero is low reconstruction error on the evaluated rows. A value at or above
one does not outperform the constant-mean predictor on aggregate.

The retained diagnostic compares trivial predictors, an in-sample linear
reference, code shapes, decoder widths, and random seeds. It measures fitting
instability for the tested configurations. It does not identify an information
capacity limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import math
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

    The mean predictor is exactly one because the metric divides by centered
    activation energy. The zero predictor is measured on that same denominator
    and can be larger when the activation mean carries energy.
    """

    zero_predictor: float
    mean_predictor: float
    constant_energy_fraction: float


@dataclass(frozen=True)
class BitBudgetPoint:
    """One measured configuration in a code bit-budget sweep."""

    code_length: int
    code_vocab: int
    width: int
    bits: float
    relative_error: float
    distinct_codes: int
    code_entropy: float


@dataclass(frozen=True)
class LinearReferencePoint:
    """Relative error of a rank-limited linear reconstruction."""

    rank: int
    relative_error: float


@dataclass(frozen=True)
class SeedStability:
    """Spread of one fixed configuration refit under several random seeds.

    Large within-configuration spread limits any attribution of differences
    between configurations to nominal code bits.
    """

    seeds: tuple[int, ...]
    relative_errors: tuple[float, ...]
    distinct_codes: tuple[int, ...]
    best: float
    worst: float
    median: float
    spread: float


def code_bits(code_length: int, code_vocab: int) -> float:
    """Information capacity of the discrete code in bits.

    A code of `code_length` slots over `code_vocab` symbols can express
    `code_vocab ** code_length` distinct messages, which is
    `code_length * log2(code_vocab)` bits. This is an upper bound that assumes
    the bottleneck uses its whole code space, so a collapsed code carries less.
    """
    if code_length <= 0 or code_vocab <= 0:
        raise ValueError("Code dimensions must be positive")
    return float(code_length) * math.log2(float(code_vocab))


def linear_reference(
    activations: Tensor, ranks: Sequence[int]
) -> list[LinearReferencePoint]:
    """Relative error of reconstructing rows from their top principal components.

    Rank zero keeps only the mean and reproduces the constant-mean baseline.
    Higher ranks provide an in-sample reconstruction reference because the PCA
    basis is fitted and scored on the same matrix. This function does not count
    storage for the shared basis or mean.
    """
    if activations.ndim != 2:
        raise ValueError("Activations must be a two-dimensional row matrix")
    target = activations.to(torch.float32)
    mean = target.mean(dim=0, keepdim=True)
    centered = target - mean
    energy = centered.pow(2).sum(dim=-1).mean()
    if float(energy) == 0.0:
        raise ValueError(
            "Activations carry no centered energy, so the metric is undefined"
        )
    covariance = (centered.T @ centered) / centered.shape[0]
    _, eigenvectors = torch.linalg.eigh(covariance)
    ordered = eigenvectors.flip(-1)
    points: list[LinearReferencePoint] = []
    for rank in ranks:
        if rank < 0:
            raise ValueError("Ranks must not be negative")
        usable = min(rank, ordered.shape[1])
        if usable == 0:
            residual = centered
        else:
            basis = ordered[:, :usable]
            residual = centered - (centered @ basis) @ basis.T
        error = float(residual.pow(2).sum(dim=-1).mean() / energy)
        points.append(LinearReferencePoint(rank=usable, relative_error=error))
    return points


def trivial_baselines(activations: Tensor) -> TrivialBaselines:
    """Measure the reconstruction error of the zero and mean predictors.

    A trained code that does not score below `mean_predictor` does not improve
    on the constant-mean predictor on aggregate.
    """
    if activations.ndim != 2:
        raise ValueError("Activations must be a two-dimensional row matrix")
    if activations.shape[0] == 0:
        raise ValueError("Activations must contain at least one row")
    target = activations.to(torch.float32)
    mean = target.mean(dim=0, keepdim=True)
    centered = target - mean
    centered_energy = centered.pow(2).sum(dim=-1).mean()
    if float(centered_energy) == 0.0:
        raise ValueError(
            "Activations carry no centered energy, so the metric is undefined"
        )
    uncentered_energy = target.pow(2).sum(dim=-1).mean()
    return TrivialBaselines(
        zero_predictor=float(uncentered_energy / centered_energy),
        mean_predictor=1.0,
        constant_energy_fraction=float(
            1.0 - centered_energy / uncentered_energy
        ),
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


def bit_budget_sweep(
    activations: Tensor,
    specs: Sequence[tuple[int, int, int]],
    *,
    embedding_dim: int,
    temperature: float,
    steps: int,
    lr: float,
    batch_size: int,
    entropy_weight: float,
    seed: int,
) -> list[BitBudgetPoint]:
    """Refit the autoencoder across code shapes and decoder widths.

    Each spec is a `(code_length, code_vocab, width)` triple. The sweep records
    fitted outcomes across nominal code shapes and decoder widths. It is not an
    optimal capacity frontier.
    """
    if activations.ndim != 2:
        raise ValueError("Activations must be a two-dimensional row matrix")
    points: list[BitBudgetPoint] = []
    for code_length, code_vocab, width in specs:
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
            entropy_weight=entropy_weight,
            seed=seed,
        )
        points.append(
            BitBudgetPoint(
                code_length=code_length,
                code_vocab=code_vocab,
                width=width,
                bits=code_bits(code_length, code_vocab),
                relative_error=error,
                distinct_codes=usage.distinct_codes,
                code_entropy=usage.code_entropy,
            )
        )
    return points


def seed_stability(
    activations: Tensor,
    seeds: Sequence[int],
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
) -> SeedStability:
    """Refit one configuration under several seeds and report the spread.

    Everything except the seed is held fixed, so the resulting range measures
    sensitivity to initialization and batch order under one training recipe.
    """
    if activations.ndim != 2:
        raise ValueError("Activations must be a two-dimensional row matrix")
    if not seeds:
        raise ValueError("Seed stability needs at least one seed")
    errors: list[float] = []
    codes: list[int] = []
    for seed in seeds:
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
            entropy_weight=entropy_weight,
            seed=seed,
        )
        errors.append(error)
        codes.append(usage.distinct_codes)
    ordered = sorted(errors)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        median = ordered[middle]
    else:
        median = 0.5 * (ordered[middle - 1] + ordered[middle])
    return SeedStability(
        seeds=tuple(int(seed) for seed in seeds),
        relative_errors=tuple(errors),
        distinct_codes=tuple(codes),
        best=ordered[0],
        worst=ordered[-1],
        median=median,
        spread=ordered[-1] - ordered[0],
    )
