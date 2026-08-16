from __future__ import annotations

import pytest
import torch

from qwen_j_lens.jacobian import (
    estimate_residual_jacobian,
    estimate_token_vjps,
    token_vectors_from_jacobian,
)
from toy import ScalarCausalAdapter, ToyAdapter


def test_token_vjp_shape_for_toy_transformer() -> None:
    adapter = ToyAdapter(width=3, vocab_size=7, layers=2)
    estimate = estimate_token_vjps(
        adapter,
        [(torch.tensor([[1, 2]]), torch.ones((1, 2), dtype=torch.long))],
        layer_index=0,
        token_ids=[1, 5],
    )

    assert estimate.values.shape == (2, 3)
    assert estimate.pair_count == 3
    assert torch.isfinite(estimate.values).all()


def test_pair_and_corpus_average_is_exact() -> None:
    adapter = ScalarCausalAdapter()
    batches = [
        (torch.tensor([[1, 2]]), torch.ones((1, 2), dtype=torch.long)),
        (torch.tensor([[3]]), torch.ones((1, 1), dtype=torch.long)),
    ]
    estimate = estimate_token_vjps(
        adapter, batches, layer_index=0, token_ids=[0, 1]
    )

    # Context means for W_U J are 8/3 and 6, then averaged equally.
    # ScalarCausalAdapter has a 5x final norm, so differentiating final logits
    # would incorrectly multiply both values by five.
    assert estimate.pair_count == 4
    assert estimate.context_count == 2
    assert estimate.values[:, 0].tolist() == pytest.approx([13.0 / 3.0, 26.0 / 3.0])


def test_present_only_excludes_future_pairs() -> None:
    adapter = ScalarCausalAdapter()
    estimate = estimate_token_vjps(
        adapter,
        [(torch.tensor([[1, 2]]), None)],
        layer_index=0,
        token_ids=[0],
        include_future=False,
    )

    assert estimate.pair_count == 2
    assert estimate.values.item() == pytest.approx(3.0)


def test_full_residual_jacobian_shape_and_average() -> None:
    adapter = ScalarCausalAdapter()
    estimate = estimate_residual_jacobian(
        adapter,
        [(torch.tensor([[1, 2]]), torch.ones((1, 2), dtype=torch.long))],
        layer_index=0,
    )

    assert estimate.values.shape == (1, 1)
    assert estimate.values.item() == pytest.approx(8.0 / 3.0)


def test_full_residual_jacobian_uses_batched_vjp_chunks() -> None:
    adapter = ToyAdapter(width=3, vocab_size=7, layers=2)
    estimate = estimate_residual_jacobian(
        adapter,
        [(torch.tensor([[1, 2]]), torch.ones((1, 2), dtype=torch.long))],
        layer_index=0,
        vjp_chunk_size=2,
    )

    assert estimate.values.shape == (3, 3)
    assert torch.isfinite(estimate.values).all()


def test_token_vectors_are_rows_of_unembedding_times_jacobian() -> None:
    adapter = ScalarCausalAdapter()

    vectors = token_vectors_from_jacobian(
        adapter, torch.tensor([[3.0]]), token_ids=[0, 1]
    )

    assert vectors.tolist() == [[3.0], [6.0]]


def test_selected_token_vjps_match_rows_from_full_jacobian() -> None:
    adapter = ToyAdapter(width=3, vocab_size=7, layers=2)
    batch = [(torch.tensor([[1, 2]]), torch.ones((1, 2), dtype=torch.long))]
    full = estimate_residual_jacobian(
        adapter,
        batch,
        layer_index=0,
        vjp_chunk_size=2,
    )
    selected = estimate_token_vjps(
        adapter,
        batch,
        layer_index=0,
        token_ids=[1, 5],
        vjp_chunk_size=1,
    )

    torch.testing.assert_close(
        selected.values,
        token_vectors_from_jacobian(adapter, full.values, token_ids=[1, 5]),
    )
