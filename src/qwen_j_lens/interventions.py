"""Controlled-vector projection, ablation, steering, and coordinate patching."""

from __future__ import annotations

import torch
from torch import Tensor


def _matrix(vectors: Tensor, hidden_width: int) -> Tensor:
    if vectors.ndim != 2 or vectors.shape[1] != hidden_width:
        raise ValueError(
            f"vectors must have shape [count, {hidden_width}], got {tuple(vectors.shape)}"
        )
    if vectors.shape[0] == 0:
        raise ValueError("At least one vector is required")
    return vectors


def project_onto_span(hidden: Tensor, vectors: Tensor, *, rcond: float = 1e-6) -> Tensor:
    """Orthogonally project residual vectors onto the span of row vectors."""
    basis = _matrix(vectors, hidden.shape[-1]).to(
        device=hidden.device, dtype=torch.float32
    )
    working = hidden.to(torch.float32)
    coefficients = working @ torch.linalg.pinv(basis, rtol=rcond)
    return (coefficients @ basis).to(hidden.dtype)


def ablate_controlled_span(
    hidden: Tensor, vectors: Tensor, *, strength: float = 1.0
) -> Tensor:
    """Remove some or all of the span of an explicitly controlled vector bank."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    return hidden - strength * project_onto_span(hidden, vectors)


def steer(hidden: Tensor, vector: Tensor, *, coefficient: float) -> Tensor:
    """Add a J-lens direction with an explicit coefficient."""
    if vector.ndim != 1 or vector.shape[0] != hidden.shape[-1]:
        raise ValueError("vector must match the residual width")
    return hidden + coefficient * vector.to(hidden)


def swap_coordinates(
    hidden: Tensor,
    source_vector: Tensor,
    target_vector: Tensor,
    *,
    strength: float = 1.0,
) -> Tensor:
    """Swap source and target pseudoinverse coordinates in their shared span."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    vectors = torch.stack((source_vector, target_vector)).to(
        device=hidden.device, dtype=torch.float32
    )
    _matrix(vectors, hidden.shape[-1])
    working = hidden.to(torch.float32)
    coordinates = working @ torch.linalg.pinv(vectors)
    swapped = coordinates.flip(-1)
    delta = ((swapped - coordinates) @ vectors).to(hidden.dtype)
    return hidden + strength * delta


def sparse_nonnegative_projection(
    hidden: Tensor,
    vectors: Tensor,
    *,
    k: int = 25,
) -> tuple[Tensor, Tensor]:
    """Approximate a sparse nonnegative component in a supplied vector bank.

    Candidate vectors are selected by positive normalized correlation. A
    nonnegative least-squares approximation is then formed by clamping the
    unconstrained least-squares coefficients. This is a compact baseline, not
    an implementation of the source paper's gradient-pursuit solver.
    """
    basis = _matrix(vectors, hidden.shape[-1]).to(
        device=hidden.device, dtype=torch.float32
    )
    if hidden.ndim != 1:
        raise ValueError("sparse_nonnegative_projection expects one residual vector")
    if k <= 0:
        raise ValueError("k must be positive")
    normalized = basis / basis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    working = hidden.to(torch.float32)
    scores = normalized @ working
    selected = torch.topk(scores, k=min(k, basis.shape[0])).indices
    active = basis.index_select(0, selected)
    coefficients = (working @ torch.linalg.pinv(active)).clamp_min(0)
    projection = (coefficients @ active).to(hidden.dtype)
    dense_coefficients = torch.zeros(
        basis.shape[0], dtype=torch.float32, device=hidden.device
    )
    dense_coefficients[selected] = coefficients
    return projection, dense_coefficients
