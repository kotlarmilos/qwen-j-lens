"""Logit-lens baseline."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from qwen_j_lens.adapter import ModelAdapter


def logit_lens(
    adapter: ModelAdapter,
    hidden: Tensor,
    *,
    token_ids: Sequence[int] | None = None,
) -> Tensor:
    """Apply final normalization and unembedding directly to an intermediate state."""
    logits = adapter.unembed(adapter.normalize(hidden))
    if token_ids is None:
        return logits
    indices = torch.as_tensor(token_ids, device=logits.device, dtype=torch.long)
    return logits.index_select(-1, indices)


def jacobian_lens(
    adapter: ModelAdapter,
    hidden: Tensor,
    jacobian: Tensor,
    *,
    token_ids: Sequence[int] | None = None,
) -> Tensor:
    """Apply ``W_U norm(J_l h_l)`` using row-major activation tensors."""
    expected = (adapter.hidden_size, adapter.hidden_size)
    if tuple(jacobian.shape) != expected:
        raise ValueError(f"jacobian must have shape {expected}")
    transformed = hidden.to(torch.float32) @ jacobian.to(
        device=hidden.device, dtype=torch.float32
    ).T
    logits = adapter.unembed(adapter.normalize(transformed.to(hidden.dtype)))
    if token_ids is None:
        return logits
    indices = torch.as_tensor(token_ids, device=logits.device, dtype=torch.long)
    return logits.index_select(-1, indices)


def top_tokens(logits: Tensor, k: int = 10) -> tuple[Tensor, Tensor]:
    """Return descending top-k values and token IDs."""
    if k <= 0 or k > logits.shape[-1]:
        raise ValueError(f"k must be in [1, {logits.shape[-1]}]")
    return torch.topk(logits, k=k, dim=-1)
