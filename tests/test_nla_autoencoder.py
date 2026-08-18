from __future__ import annotations

import torch
import pytest

from qwen_j_lens.nla.autoencoder import (
    NaturalLanguageAutoencoder,
    straight_through_code,
    reconstruction_error,
)
from qwen_j_lens.nla.train import train_autoencoder


def _clustered_activations(
    clusters: int, per_cluster: int, hidden: int, seed: int
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    centers = torch.randn((clusters, hidden), generator=generator) * 3.0
    rows = []
    for index in range(clusters * per_cluster):
        cluster = index % clusters
        noise = torch.randn((hidden,), generator=generator) * 0.05
        rows.append(centers[cluster] + noise)
    return torch.stack(rows, dim=0)


def test_straight_through_code_is_one_hot_with_gradient() -> None:
    logits = torch.randn((4, 3, 5), requires_grad=True)
    code, indices = straight_through_code(logits, temperature=1.0)

    assert code.shape == (4, 3, 5)
    assert torch.allclose(code.sum(dim=-1), torch.ones((4, 3)))
    assert indices.shape == (4, 3)
    code.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_autoencoder_forward_shapes() -> None:
    torch.manual_seed(0)
    model = NaturalLanguageAutoencoder(
        hidden_size=8,
        code_length=2,
        code_vocab=6,
        width=16,
        embedding_dim=4,
        temperature=1.0,
    )
    activations = torch.randn((10, 8))
    sample = model(activations)

    assert sample.reconstruction.shape == (10, 8)
    assert sample.code_indices.shape == (10, 2)
    assert sample.code_onehot.shape == (10, 2, 6)


def test_training_reduces_reconstruction_error_on_clusters() -> None:
    activations = _clustered_activations(clusters=6, per_cluster=20, hidden=8, seed=3)
    result = train_autoencoder(
        activations,
        code_length=1,
        code_vocab=8,
        width=32,
        embedding_dim=8,
        temperature=1.0,
        steps=400,
        lr=0.05,
        batch_size=64,
        entropy_weight=0.01,
        seed=5,
    )

    assert result.history[-1]["relative_error"] < result.history[0]["relative_error"]
    assert result.final_relative_error < 0.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_training_keeps_cuda_activations_and_model_on_one_device() -> None:
    activations = torch.randn((16, 8), device="cuda")
    result = train_autoencoder(
        activations,
        code_length=1,
        code_vocab=4,
        width=8,
        embedding_dim=4,
        temperature=1.0,
        steps=2,
        lr=0.01,
        batch_size=8,
        entropy_weight=0.01,
        seed=5,
    )

    assert next(result.model.parameters()).device.type == "cuda"


def test_reconstruction_error_relative_reference() -> None:
    target = torch.randn((20, 4))
    perfect = reconstruction_error(target, target, relative=True)
    mean_only = reconstruction_error(
        target.mean(dim=0, keepdim=True).expand_as(target), target, relative=True
    )

    assert float(perfect) < 1e-6
    assert abs(float(mean_only) - 1.0) < 1e-5
