from __future__ import annotations

import torch

from qwen_j_lens.capture import capture_layer_activations, replace_layer_activation
from qwen_j_lens.logit_lens import jacobian_lens, logit_lens, top_tokens
from toy import ScalarCausalAdapter
from toy import ToyAdapter


def test_layer_capture_has_expected_shapes_and_values() -> None:
    adapter = ToyAdapter()
    input_ids = torch.tensor([[1, 2, 3]])
    run, captured = capture_layer_activations(
        adapter, input_ids, layer_indices=[0, 2]
    )

    assert run.logits.shape == (1, 3, adapter.vocab_size)
    assert set(captured) == {0, 2}
    assert captured[0].shape == (1, 3, adapter.hidden_size)
    assert not captured[0].requires_grad


def test_replacement_changes_output_and_removes_hook() -> None:
    adapter = ToyAdapter()
    input_ids = torch.tensor([[1, 2]])
    baseline = adapter.forward(input_ids).logits
    replacement = torch.zeros((1, 2, adapter.hidden_size))
    with replace_layer_activation(adapter.layers[1], replacement):
        changed = adapter.forward(input_ids).logits
    restored = adapter.forward(input_ids).logits

    assert not torch.allclose(changed, baseline)
    assert torch.allclose(restored, baseline)


def test_logit_lens_supports_controlled_subset() -> None:
    adapter = ToyAdapter()
    _, captured = capture_layer_activations(
        adapter, torch.tensor([[1, 2]]), layer_indices=[0]
    )
    logits = logit_lens(adapter, captured[0], token_ids=[1, 4, 7])
    values, indices = top_tokens(logits, k=2)

    assert logits.shape == (1, 2, 3)
    assert values.shape == indices.shape == (1, 2, 2)


def test_jacobian_lens_applies_nonidentity_norm_after_jacobian() -> None:
    adapter = ScalarCausalAdapter()
    hidden = torch.tensor([[[3.0]]])
    jacobian = torch.tensor([[2.0]])

    logits = jacobian_lens(adapter, hidden, jacobian)

    assert logits.tolist() == [[[30.0, 60.0]]]
