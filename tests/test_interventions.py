from __future__ import annotations

import torch

from qwen_j_lens.interventions import (
    ablate_controlled_span,
    project_onto_span,
    sparse_nonnegative_projection,
    steer,
    swap_coordinates,
)


def test_projection_and_controlled_ablation() -> None:
    hidden = torch.tensor([2.0, 3.0, 4.0])
    vectors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    projected = project_onto_span(hidden, vectors)
    suppressed = ablate_controlled_span(hidden, vectors)

    assert torch.allclose(projected, torch.tensor([2.0, 3.0, 0.0]))
    assert torch.allclose(suppressed, torch.tensor([0.0, 0.0, 4.0]))
    assert torch.allclose(projected @ suppressed, torch.tensor(0.0))


def test_steering_and_coordinate_swap() -> None:
    hidden = torch.tensor([2.0, 3.0, 4.0])
    source = torch.tensor([1.0, 0.0, 0.0])
    target = torch.tensor([0.0, 1.0, 0.0])

    assert torch.allclose(
        steer(hidden, target, coefficient=2.0), torch.tensor([2.0, 5.0, 4.0])
    )
    assert torch.allclose(
        swap_coordinates(hidden, source, target), torch.tensor([3.0, 2.0, 4.0])
    )


def test_sparse_nonnegative_projection_respects_k() -> None:
    hidden = torch.tensor([2.0, 1.0, -4.0])
    vectors = torch.eye(3)
    projected, coefficients = sparse_nonnegative_projection(hidden, vectors, k=2)

    assert torch.allclose(projected, torch.tensor([2.0, 1.0, 0.0]))
    assert torch.count_nonzero(coefficients) == 2


def test_projection_supports_bfloat16_activations() -> None:
    hidden = torch.tensor([2.0, 3.0], dtype=torch.bfloat16)
    vectors = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    projected = project_onto_span(hidden, vectors)

    assert projected.dtype == torch.bfloat16
    assert torch.equal(projected, torch.tensor([2.0, 0.0], dtype=torch.bfloat16))
