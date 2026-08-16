from __future__ import annotations

import torch

from qwen_j_lens.nla.activations import ActivationBatch, collect_layer_activations
from toy import ToyAdapter


def _batch(rows: list[list[int]], masks: list[list[int]]) -> ActivationBatch:
    return ActivationBatch(
        input_ids=torch.tensor(rows, dtype=torch.long),
        attention_mask=torch.tensor(masks, dtype=torch.long),
    )


def test_collect_returns_one_row_per_attended_token() -> None:
    adapter = ToyAdapter(width=4, vocab_size=9, layers=3)
    batch = _batch([[1, 2, 3], [4, 5, 0]], [[1, 1, 1], [1, 1, 0]])

    activations = collect_layer_activations(adapter, [batch], layer_index=1)

    assert activations.shape == (5, 4)
    assert torch.isfinite(activations).all()


def test_collect_reads_requested_layer() -> None:
    adapter = ToyAdapter(width=4, vocab_size=9, layers=3)
    batch = _batch([[1, 2]], [[1, 1]])

    run = adapter.forward(batch.input_ids, batch.attention_mask, output_hidden_states=True)
    embedding_rows = collect_layer_activations(adapter, [batch], layer_index=0)

    assert torch.allclose(embedding_rows, run.hidden_states[0].reshape(2, 4))


def test_collect_rejects_out_of_range_layer() -> None:
    adapter = ToyAdapter(width=4, vocab_size=9, layers=3)
    batch = _batch([[1, 2]], [[1, 1]])

    try:
        collect_layer_activations(adapter, [batch], layer_index=99)
    except IndexError:
        return
    raise AssertionError("expected IndexError for an out-of-range layer")
