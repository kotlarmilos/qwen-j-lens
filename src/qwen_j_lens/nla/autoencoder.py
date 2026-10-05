"""Fixed-symbol autoencoder over model activations.

The encoder maps an activation to a short sequence of IDs from a fixed symbol
vocabulary. The reconstructor maps that sequence back to an activation. This is
a discrete reconstruction surrogate, not a natural-language verbalizer or a
human-readable explanation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class CodeSample:
    """One forward pass through the autoencoder."""

    reconstruction: Tensor
    code_indices: Tensor
    code_onehot: Tensor
    code_logits: Tensor


def straight_through_code(logits: Tensor, *, temperature: float) -> tuple[Tensor, Tensor]:
    """Discretize per-slot logits with a straight-through hard argmax.

    The forward pass returns a one-hot selection so the bottleneck is genuinely
    discrete. The backward pass flows through the softmax so the encoder still
    receives gradients. Returns the one-hot codes and their integer indices.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    soft = torch.softmax(logits / temperature, dim=-1)
    indices = soft.argmax(dim=-1)
    hard = torch.zeros_like(soft).scatter_(-1, indices.unsqueeze(-1), 1.0)
    code = hard + soft - soft.detach()
    return code, indices


class SymbolEncoder(nn.Module):
    """Map an activation to per-slot symbol logits."""

    def __init__(
        self,
        *,
        hidden_size: int,
        code_length: int,
        code_vocab: int,
        width: int,
    ) -> None:
        super().__init__()
        if min(hidden_size, code_length, code_vocab, width) <= 0:
            raise ValueError("Symbol encoder dimensions must be positive")
        self.code_length = code_length
        self.code_vocab = code_vocab
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, width),
            nn.GELU(),
            nn.Linear(width, code_length * code_vocab),
        )

    def forward(self, activations: Tensor) -> Tensor:
        logits = self.encoder(activations)
        return logits.reshape(*activations.shape[:-1], self.code_length, self.code_vocab)


class Reconstructor(nn.Module):
    """Map a symbol sequence back to an activation."""

    def __init__(
        self,
        *,
        hidden_size: int,
        code_length: int,
        code_vocab: int,
        width: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        if min(hidden_size, code_length, code_vocab, width, embedding_dim) <= 0:
            raise ValueError("Reconstructor dimensions must be positive")
        self.code_length = code_length
        self.code_vocab = code_vocab
        self.slot_embedding = nn.Parameter(
            torch.zeros(code_length, code_vocab, embedding_dim)
        )
        nn.init.normal_(self.slot_embedding, std=0.02)
        self.decoder = nn.Sequential(
            nn.Linear(code_length * embedding_dim, width),
            nn.GELU(),
            nn.Linear(width, hidden_size),
        )

    def forward(self, code_onehot: Tensor) -> Tensor:
        if code_onehot.shape[-2:] != (self.code_length, self.code_vocab):
            raise ValueError("Code shape does not match the reconstructor vocabulary")
        embedded = torch.einsum("...lv,lve->...le", code_onehot, self.slot_embedding)
        flattened = embedded.reshape(*embedded.shape[:-2], -1)
        return self.decoder(flattened)


class FixedSymbolAutoencoder(nn.Module):
    """Compose a symbol encoder and reconstructor around a discrete code."""

    def __init__(
        self,
        *,
        hidden_size: int,
        code_length: int,
        code_vocab: int,
        width: int,
        embedding_dim: int,
        temperature: float,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.encoder = SymbolEncoder(
            hidden_size=hidden_size,
            code_length=code_length,
            code_vocab=code_vocab,
            width=width,
        )
        self.reconstructor = Reconstructor(
            hidden_size=hidden_size,
            code_length=code_length,
            code_vocab=code_vocab,
            width=width,
            embedding_dim=embedding_dim,
        )

    def forward(self, activations: Tensor) -> CodeSample:
        logits = self.encoder(activations)
        code_onehot, indices = straight_through_code(
            logits, temperature=self.temperature
        )
        reconstruction = self.reconstructor(code_onehot)
        return CodeSample(
            reconstruction=reconstruction,
            code_indices=indices,
            code_onehot=code_onehot,
            code_logits=logits,
        )


def reconstruction_error(
    reconstruction: Tensor,
    target: Tensor,
    *,
    relative: bool = True,
) -> Tensor:
    """Mean squared reconstruction error, optionally normalized by centered energy.

    The relative form divides by the mean squared norm after subtracting the
    activation mean. A value near zero means the code preserves the activation;
    the constant-mean predictor is exactly one, while the zero predictor can be
    larger than one when the activation mean carries energy.
    """
    if reconstruction.shape != target.shape:
        raise ValueError("Reconstruction and target shapes must match")
    squared = (reconstruction - target).pow(2).sum(dim=-1)
    error = squared.mean()
    if not relative:
        return error
    centered = target - target.mean(dim=0, keepdim=True)
    energy = centered.pow(2).sum(dim=-1).mean()
    if energy <= 0.0:
        raise ValueError(
            "Relative reconstruction error requires nonzero centered target energy"
        )
    return error / energy
