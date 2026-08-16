"""Faithfulness audits for verbalized activation codes.

Reconstruction quality alone does not prove that a natural-language code is a
good explanation. The ``RECAP`` critique, *Train the Model, Not the Reader*,
shows two ways a high-reconstruction code can still mislead a human reader. The
first is a co-adapted private code: the reconstructor and verbalizer agree on a
scheme that a fresh, independent reader cannot decode. The second is a gist
code: the text captures coarse structure but drops the specific fact the
activation carried. This module measures both effects with deterministic linear
algebra and small held-out reconstructors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor

from qwen_j_lens.nla.autoencoder import Reconstructor, reconstruction_error


def _one_hot(codes: Tensor, code_vocab: int) -> Tensor:
    if codes.ndim != 2:
        raise ValueError("Codes must be a two-dimensional index matrix")
    if int(codes.max()) >= code_vocab or int(codes.min()) < 0:
        raise ValueError("Code indices fall outside the declared vocabulary")
    return torch.zeros(
        (*codes.shape, code_vocab), dtype=torch.float32
    ).scatter_(-1, codes.unsqueeze(-1), 1.0)


def _train_reader(
    code_onehot: Tensor,
    targets: Tensor,
    *,
    width: int,
    embedding_dim: int,
    steps: int,
    lr: float,
    seed: int,
) -> Reconstructor:
    """Fit a reconstructor that maps fixed codes back to target activations."""
    torch.manual_seed(seed)
    reader = Reconstructor(
        hidden_size=targets.shape[1],
        code_length=code_onehot.shape[1],
        code_vocab=code_onehot.shape[2],
        width=width,
        embedding_dim=embedding_dim,
    )
    optimizer = torch.optim.Adam(reader.parameters(), lr=lr)
    for _ in range(steps):
        prediction = reader(code_onehot)
        loss = reconstruction_error(prediction, targets, relative=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return reader


def independent_reader_gap(
    codes: Tensor,
    activations: Tensor,
    *,
    train_index: Tensor,
    test_index: Tensor,
    code_vocab: int,
    width: int = 32,
    embedding_dim: int = 8,
    steps: int = 200,
    lr: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Compare a co-adapted reader against an independent held-out reader.

    The co-adapted reader trains on every code, including the held-out ones, so
    it can memorize an example-specific scheme. The independent reader trains
    only on the training split and is scored on the held-out split. When the
    code is a generalizable description the two test errors are close. When the
    code is a private, example-specific scheme the independent reader fails and
    the gap grows.
    """
    code_onehot = _one_hot(codes, code_vocab)
    train_targets = activations.index_select(0, train_index)
    test_targets = activations.index_select(0, test_index)
    train_codes = code_onehot.index_select(0, train_index)
    test_codes = code_onehot.index_select(0, test_index)

    co_adapted = _train_reader(
        code_onehot,
        activations,
        width=width,
        embedding_dim=embedding_dim,
        steps=steps,
        lr=lr,
        seed=seed,
    )
    independent = _train_reader(
        train_codes,
        train_targets,
        width=width,
        embedding_dim=embedding_dim,
        steps=steps,
        lr=lr,
        seed=seed + 1,
    )
    with torch.no_grad():
        co_error = float(
            reconstruction_error(co_adapted(test_codes), test_targets, relative=True)
        )
        independent_error = float(
            reconstruction_error(independent(test_codes), test_targets, relative=True)
        )
    return {
        "co_adapted_test_error": co_error,
        "independent_test_error": independent_error,
        "gap": independent_error - co_error,
    }


def _linear_probe_r2(
    inputs_train: Tensor,
    targets_train: Tensor,
    inputs_test: Tensor,
    targets_test: Tensor,
) -> float:
    """Coefficient of determination for a least-squares linear probe."""
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


def gist_versus_specific(
    activations: Tensor,
    reconstructions: Tensor,
    specific_targets: Tensor,
    *,
    train_index: Tensor,
    test_index: Tensor,
) -> dict[str, float]:
    """Test whether reconstructions keep a specific attribute or only the gist.

    A linear probe is fit to recover ``specific_targets`` from the original
    activations and again from the reconstructions. When the reconstruction
    preserves the specific attribute both probes reach a similar score. When the
    code keeps only the gist the reconstruction probe drops sharply while the
    original probe stays high.
    """
    if specific_targets.ndim == 1:
        specific_targets = specific_targets.unsqueeze(-1)
    original_r2 = _linear_probe_r2(
        activations.index_select(0, train_index),
        specific_targets.index_select(0, train_index),
        activations.index_select(0, test_index),
        specific_targets.index_select(0, test_index),
    )
    reconstructed_r2 = _linear_probe_r2(
        reconstructions.index_select(0, train_index),
        specific_targets.index_select(0, train_index),
        reconstructions.index_select(0, test_index),
        specific_targets.index_select(0, test_index),
    )
    retained = 0.0 if original_r2 <= 0.0 else reconstructed_r2 / original_r2
    return {
        "original_specific_r2": original_r2,
        "reconstructed_specific_r2": reconstructed_r2,
        "specific_retained_fraction": retained,
    }


def code_usage(codes: Tensor, code_vocab: int) -> dict[str, float]:
    """Summarize how a code population uses its symbol vocabulary.

    ``distinct_code_fraction`` near one means codes are close to example-specific,
    which is the memorization signature the independent-reader gap tests for.
    """
    code_onehot = _one_hot(codes, code_vocab)
    per_slot_marginal = code_onehot.mean(dim=0).clamp_min(1e-9)
    per_slot_entropy = -(per_slot_marginal * per_slot_marginal.log()).sum(dim=-1)
    used = (code_onehot.sum(dim=0) > 0).any(dim=0).float().mean()
    distinct = torch.unique(codes, dim=0).shape[0]
    return {
        "mean_slot_entropy": float(per_slot_entropy.mean()),
        "vocabulary_fraction_used": float(used),
        "distinct_code_fraction": float(distinct / codes.shape[0]),
    }


@dataclass(frozen=True)
class FaithfulnessReport:
    """Aggregated faithfulness measurements with threshold-based flags."""

    reconstruction_error: float
    independent_reader: Mapping[str, float]
    gist_versus_specific: Mapping[str, float]
    code_usage: Mapping[str, float]
    gap_threshold: float
    retained_threshold: float

    @property
    def private_code_suspected(self) -> bool:
        return self.independent_reader["gap"] > self.gap_threshold

    @property
    def gist_only_suspected(self) -> bool:
        return (
            self.gist_versus_specific["specific_retained_fraction"]
            < self.retained_threshold
        )

    @property
    def faithful(self) -> bool:
        return not self.private_code_suspected and not self.gist_only_suspected

    def to_mapping(self) -> dict[str, Any]:
        return {
            "reconstruction_error": self.reconstruction_error,
            "independent_reader": dict(self.independent_reader),
            "gist_versus_specific": dict(self.gist_versus_specific),
            "code_usage": dict(self.code_usage),
            "thresholds": {
                "gap": self.gap_threshold,
                "specific_retained": self.retained_threshold,
            },
            "flags": {
                "private_code_suspected": self.private_code_suspected,
                "gist_only_suspected": self.gist_only_suspected,
                "faithful": self.faithful,
            },
        }
