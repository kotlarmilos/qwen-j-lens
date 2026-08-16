"""Residual-stream capture and replacement hooks."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import torch
from torch import Tensor, nn

from qwen_j_lens.adapter import ModelAdapter, ModelRun


def _output_tensor(output: object) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError("Layer output must be a Tensor or a tuple/list beginning with a Tensor")


def _replace_output(output: object, replacement: Tensor) -> object:
    if isinstance(output, Tensor):
        return replacement
    if isinstance(output, tuple):
        return (replacement, *output[1:])
    if isinstance(output, list):
        return [replacement, *output[1:]]
    raise TypeError("Layer output must be a Tensor or a tuple/list beginning with a Tensor")


def resolve_layers(layers: Sequence[nn.Module], indices: Sequence[int] | None) -> list[int]:
    """Validate and normalize requested layer indices."""
    selected = list(range(len(layers))) if indices is None else list(indices)
    if not selected:
        raise ValueError("At least one layer is required")
    if len(set(selected)) != len(selected):
        raise ValueError("Layer indices must be unique")
    if any(index < 0 or index >= len(layers) for index in selected):
        raise IndexError(f"Layer index outside [0, {len(layers) - 1}]")
    return selected


def capture_layer_activations(
    adapter: ModelAdapter,
    input_ids: Tensor,
    attention_mask: Tensor | None = None,
    *,
    layer_indices: Sequence[int] | None = None,
    detach: bool = True,
) -> tuple[ModelRun, dict[int, Tensor]]:
    """Capture each selected block's output residual stream."""
    selected = resolve_layers(adapter.layers, layer_indices)
    activations: dict[int, Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for index in selected:
        def save_output(
            _module: nn.Module,
            _inputs: tuple[object, ...],
            output: object,
            *,
            layer_index: int = index,
        ) -> None:
            tensor = _output_tensor(output)
            activations[layer_index] = tensor.detach().clone() if detach else tensor

        handles.append(adapter.layers[index].register_forward_hook(save_output))

    try:
        run = adapter.forward(input_ids, attention_mask)
    finally:
        for handle in handles:
            handle.remove()
    if set(activations) != set(selected):
        missing = sorted(set(selected) - set(activations))
        raise RuntimeError(f"Requested layers did not execute: {missing}")
    return run, activations


@contextmanager
def replace_layer_activation(
    layer: nn.Module,
    replacement: Tensor | Callable[[Tensor], Tensor],
) -> Iterator[None]:
    """Replace a block output while preserving its container type."""

    def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> object:
        original = _output_tensor(output)
        value = replacement(original) if callable(replacement) else replacement
        if value.shape != original.shape:
            raise ValueError(
                f"Replacement shape {tuple(value.shape)} does not match {tuple(original.shape)}"
            )
        return _replace_output(output, value)

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@dataclass(frozen=True)
class PositionIntervention:
    """Apply a residual edit at selected sequence positions."""

    positions: tuple[int, ...]
    edit: Callable[[Tensor], Tensor]

    def __call__(self, hidden: Tensor) -> Tensor:
        result = hidden.clone()
        sequence_length = hidden.shape[-2]
        for position in self.positions:
            normalized = position if position >= 0 else sequence_length + position
            if normalized < 0 or normalized >= sequence_length:
                raise IndexError(f"Position {position} outside sequence length {sequence_length}")
            result[..., normalized, :] = self.edit(hidden[..., normalized, :])
        return result

