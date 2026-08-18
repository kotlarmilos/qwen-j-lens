"""Deterministic training loop for the natural-language autoencoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from qwen_j_lens.nla.autoencoder import (
    NaturalLanguageAutoencoder,
    reconstruction_error,
)


@dataclass(frozen=True)
class TrainResult:
    """Outcome of one autoencoder training run."""

    model: NaturalLanguageAutoencoder
    history: list[dict[str, float]]
    final_relative_error: float
    final_code_entropy: float


def _code_entropy(code_onehot: Tensor) -> Tensor:
    """Mean per-slot entropy of the batch-marginal symbol distribution.

    A high value means the code spreads across its symbol vocabulary. A value
    near zero means the code has collapsed onto a single symbol and carries no
    information, which the reconstructor cannot use.
    """
    marginal = code_onehot.mean(dim=0)
    marginal = marginal.clamp_min(1e-9)
    per_slot = -(marginal * marginal.log()).sum(dim=-1)
    return per_slot.mean()


def train_autoencoder(
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
) -> TrainResult:
    """Train a verbalizer and reconstructor to minimize reconstruction error.

    The objective is reconstruction error minus a code-usage entropy bonus. The
    entropy term keeps the discrete bottleneck from collapsing onto one symbol in
    the small offline setting; it does not change the reconstruction target.
    """
    if activations.ndim != 2:
        raise ValueError("Activations must be a two-dimensional row matrix")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if entropy_weight < 0.0:
        raise ValueError("entropy_weight must not be negative")
    torch.manual_seed(seed)
    generator = torch.Generator(device=activations.device).manual_seed(seed)
    rows = activations.shape[0]
    effective_batch = min(batch_size, rows)
    model = NaturalLanguageAutoencoder(
        hidden_size=activations.shape[1],
        code_length=code_length,
        code_vocab=code_vocab,
        width=width,
        embedding_dim=embedding_dim,
        temperature=temperature,
    ).to(activations.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    for step in range(steps):
        indices = torch.randint(
            rows,
            (effective_batch,),
            generator=generator,
            device=activations.device,
        )
        batch = activations.index_select(0, indices)
        sample = model(batch)
        error = reconstruction_error(sample.reconstruction, batch, relative=True)
        entropy = _code_entropy(sample.code_onehot)
        loss = error - entropy_weight * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(
            {
                "step": float(step),
                "relative_error": float(error.detach()),
                "code_entropy": float(entropy.detach()),
                "loss": float(loss.detach()),
            }
        )
    with torch.no_grad():
        final = model(activations)
        final_error = float(
            reconstruction_error(final.reconstruction, activations, relative=True)
        )
        final_entropy = float(_code_entropy(final.code_onehot))
    return TrainResult(
        model=model,
        history=history,
        final_relative_error=final_error,
        final_code_entropy=final_entropy,
    )
