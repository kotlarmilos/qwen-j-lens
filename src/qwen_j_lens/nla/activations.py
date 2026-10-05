"""Collect residual-stream activations from any model adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor

from qwen_j_lens.adapter import ModelAdapter


@dataclass(frozen=True)
class ActivationBatch:
    """A padded tokenized batch used as an activation source."""

    input_ids: Tensor
    attention_mask: Tensor


def _validate_layer(adapter: ModelAdapter, layer_index: int) -> None:
    depth = len(adapter.layers)
    if layer_index < 0 or layer_index > depth:
        raise IndexError(
            f"Layer {layer_index} is outside the residual stack of depth {depth}"
        )


def collect_layer_activations(
    adapter: ModelAdapter,
    batches: Iterable[ActivationBatch],
    *,
    layer_index: int,
) -> Tensor:
    """Return one activation row per attended token at ``layer_index``.

    The activation is read from ``ModelRun.hidden_states`` where index ``0`` is
    the embedding output and index ``k >= 1`` is the residual stream after block
    ``k - 1``. Only positions with an attention mask of one contribute a row, so
    padding never enters the autoencoder corpus.
    """
    _validate_layer(adapter, layer_index)
    rows: list[Tensor] = []
    for batch in batches:
        if batch.input_ids.ndim != 2:
            raise ValueError("Activation batches must be two-dimensional token grids")
        if batch.attention_mask.shape != batch.input_ids.shape:
            raise ValueError("Attention mask shape must match the input ids")
        run = adapter.forward(
            batch.input_ids,
            batch.attention_mask,
            output_hidden_states=True,
        )
        if run.hidden_states is None:
            raise ValueError("Adapter did not return hidden states")
        if layer_index >= len(run.hidden_states):
            raise IndexError(
                f"Layer {layer_index} is not present in the returned hidden states"
            )
        hidden = run.hidden_states[layer_index]
        mask = batch.attention_mask.to(dtype=torch.bool)
        selected = hidden[mask]
        if selected.numel() > 0:
            rows.append(selected.detach().to(dtype=torch.float32))
    if not rows:
        raise ValueError("No attended positions were found across the batches")
    return torch.cat(rows, dim=0)
