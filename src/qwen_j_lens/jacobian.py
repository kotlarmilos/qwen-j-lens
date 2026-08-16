"""Corpus-averaged Jacobian and token VJP estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor

from qwen_j_lens.adapter import ModelAdapter
from qwen_j_lens.capture import capture_layer_activations, replace_layer_activation


@dataclass(frozen=True)
class JacobianEstimate:
    """A corpus average and the number of source-target pairs in that average."""

    values: Tensor
    pair_count: int
    context_count: int
    layer_index: int
    token_ids: tuple[int, ...] | None = None


def _valid_positions(mask: Tensor | None, batch: int, sequence: int) -> list[list[int]]:
    if mask is None:
        return [list(range(sequence)) for _ in range(batch)]
    if mask.shape != (batch, sequence):
        raise ValueError("attention_mask must match the first two input dimensions")
    return [
        torch.nonzero(mask[row].bool(), as_tuple=False).flatten().tolist()
        for row in range(batch)
    ]


def _leaf_replacement(
    adapter: ModelAdapter,
    input_ids: Tensor,
    attention_mask: Tensor | None,
    layer_index: int,
) -> Tensor:
    _, captured = capture_layer_activations(
        adapter,
        input_ids,
        attention_mask,
        layer_indices=[layer_index],
    )
    return captured[layer_index].detach().requires_grad_(True)


def estimate_token_vjps(
    adapter: ModelAdapter,
    batches: Iterable[tuple[Tensor, Tensor | None]],
    *,
    layer_index: int,
    token_ids: Sequence[int],
    include_future: bool = True,
    vjp_chunk_size: int = 32,
) -> JacobianEstimate:
    """Estimate selected rows of ``W_U J_l`` without materializing all of ``J_l``.

    Each source-target position pair receives equal weight. With ``include_future``
    enabled, all valid ``target >= source`` pairs are included, matching the
    present-and-future averaging in the J-lens method. The output has shape
    ``[len(token_ids), hidden_size]``. Gradients originate from the linear
    unembedding of the final pre-normalization residual. The model's final
    normalization is deliberately outside this estimate.
    """
    ids = tuple(int(token_id) for token_id in token_ids)
    if not ids:
        raise ValueError("token_ids must not be empty")
    if any(token_id < 0 or token_id >= adapter.vocab_size for token_id in ids):
        raise ValueError("token_ids contains an out-of-vocabulary ID")
    if vjp_chunk_size <= 0:
        raise ValueError("vjp_chunk_size must be positive")
    total: Tensor | None = None
    pair_count = 0
    context_count = 0

    for input_ids, attention_mask in batches:
        replacement = _leaf_replacement(
            adapter, input_ids, attention_mask, layer_index
        )
        with replace_layer_activation(adapter.layers[layer_index], replacement):
            run = adapter.forward(
                input_ids, attention_mask, output_hidden_states=True
            )
        batch_size, sequence_length, _ = replacement.shape
        positions = _valid_positions(
            attention_mask, batch_size, sequence_length
        )
        token_weights = adapter.unembedding_weight(ids).to(
            device=run.final_pre_norm.device,
            dtype=run.final_pre_norm.dtype,
        )
        token_readout = run.final_pre_norm @ token_weights.transpose(0, 1)
        accumulation_dtype = (
            torch.float64 if replacement.dtype == torch.float64 else torch.float32
        )
        batch_total = torch.zeros(
            len(ids),
            adapter.hidden_size,
            dtype=accumulation_dtype,
            device="cpu",
        )

        targets = [
            (row, target)
            for row, valid in enumerate(positions)
            for target in valid
        ]
        if include_future:
            weighted_rows = []
            for row, valid in enumerate(positions):
                row_pair_count = len(valid) * (len(valid) + 1) // 2
                if row_pair_count:
                    source_indices = torch.as_tensor(
                        valid, device=replacement.device, dtype=torch.long
                    )
                    weighted_rows.append(
                        token_readout[row]
                        .index_select(0, source_indices)
                        .sum(dim=0)
                        / row_pair_count
                    )
            if not weighted_rows:
                continue
            weighted_readout = torch.stack(weighted_rows).sum(dim=0)
            chunk_starts = tuple(range(0, len(ids), vjp_chunk_size))
            for chunk_offset, start in enumerate(chunk_starts):
                stop = min(start + vjp_chunk_size, len(ids))
                width = stop - start
                gradient = torch.autograd.grad(
                    weighted_readout[start:stop],
                    replacement,
                    grad_outputs=torch.eye(
                        width,
                        device=weighted_readout.device,
                        dtype=weighted_readout.dtype,
                    ),
                    is_grads_batched=True,
                    retain_graph=chunk_offset < len(chunk_starts) - 1,
                    create_graph=False,
                )[0]
                for row, valid in enumerate(positions):
                    source_indices = torch.as_tensor(
                        valid, device=replacement.device, dtype=torch.long
                    )
                    batch_total[start:stop] += (
                        gradient[:, row]
                        .index_select(1, source_indices)
                        .sum(dim=1)
                        .to(dtype=accumulation_dtype, device="cpu")
                    )
        else:
            row_totals = torch.zeros(
                batch_size,
                len(ids),
                adapter.hidden_size,
                dtype=accumulation_dtype,
                device=replacement.device,
            )
            for token_offset, token_id in enumerate(ids):
                for target_offset, (row, target) in enumerate(targets):
                    retain = (
                        token_offset < len(ids) - 1
                        or target_offset < len(targets) - 1
                    )
                    gradient = torch.autograd.grad(
                        token_readout[row, target, token_offset],
                        replacement,
                        retain_graph=retain,
                        create_graph=False,
                    )[0]
                    row_totals[row, token_offset] += gradient[row, target].to(
                        accumulation_dtype
                    )
            for row, valid in enumerate(positions):
                if valid:
                    batch_total += (row_totals[row] / len(valid)).detach().cpu()
        valid_contexts = sum(bool(valid) for valid in positions)
        if valid_contexts:
            total = batch_total if total is None else total + batch_total
            context_count += valid_contexts
        pair_count += sum(
            len(valid) * (len(valid) + 1) // 2 if include_future else len(valid)
            for valid in positions
        )

    if total is None or pair_count == 0 or context_count == 0:
        raise ValueError("No valid source-target pairs were supplied")
    return JacobianEstimate(
        values=(total / context_count).detach().cpu(),
        pair_count=pair_count,
        context_count=context_count,
        layer_index=layer_index,
        token_ids=ids,
    )


def estimate_residual_jacobian(
    adapter: ModelAdapter,
    batches: Iterable[tuple[Tensor, Tensor | None]],
    *,
    layer_index: int,
    include_future: bool = True,
    vjp_chunk_size: int = 32,
) -> JacobianEstimate:
    """Estimate the full average final-pre-normalization Jacobian for small models.

    Complexity is quadratic in hidden width and sequence length. Publication
    runs should prefer ``estimate_token_vjps`` over a controlled token subset.
    """
    if vjp_chunk_size <= 0:
        raise ValueError("vjp_chunk_size must be positive")
    total: Tensor | None = None
    pair_count = 0
    context_count = 0
    for input_ids, attention_mask in batches:
        replacement = _leaf_replacement(
            adapter, input_ids, attention_mask, layer_index
        )
        with replace_layer_activation(adapter.layers[layer_index], replacement):
            run = adapter.forward(
                input_ids, attention_mask, output_hidden_states=True
            )
        final_hidden = run.final_pre_norm
        batch_size, sequence_length, output_width = final_hidden.shape
        positions = _valid_positions(attention_mask, batch_size, sequence_length)
        accumulation_dtype = (
            torch.float64 if replacement.dtype == torch.float64 else torch.float32
        )
        row_totals = torch.zeros(
            batch_size,
            output_width,
            adapter.hidden_size,
            dtype=accumulation_dtype,
            device=replacement.device,
        )
        targets = [
            (row, target)
            for row, valid in enumerate(positions)
            for target in valid
        ]
        if include_future:
            chunks_per_row = (output_width + vjp_chunk_size - 1) // vjp_chunk_size
            call_count = chunks_per_row * len(positions)
            call_offset = 0
            identity = torch.eye(
                output_width,
                device=final_hidden.device,
                dtype=final_hidden.dtype,
            )
            for row, valid in enumerate(positions):
                source_indices = torch.as_tensor(
                    valid, device=replacement.device, dtype=torch.long
                )
                summed_final = final_hidden[row, source_indices].sum(dim=0)
                for start in range(0, output_width, vjp_chunk_size):
                    stop = min(start + vjp_chunk_size, output_width)
                    call_offset += 1
                    gradient = torch.autograd.grad(
                        summed_final,
                        replacement,
                        grad_outputs=identity[start:stop],
                        is_grads_batched=True,
                        retain_graph=call_offset < call_count,
                    )[0]
                    row_totals[row, start:stop] += (
                        gradient[:, row]
                        .index_select(1, source_indices)
                        .sum(dim=1)
                        .to(accumulation_dtype)
                    )
        else:
            chunks_per_target = (
                output_width + vjp_chunk_size - 1
            ) // vjp_chunk_size
            call_count = chunks_per_target * len(targets)
            call_offset = 0
            identity = torch.eye(
                output_width,
                device=final_hidden.device,
                dtype=final_hidden.dtype,
            )
            for row, target in targets:
                for start in range(0, output_width, vjp_chunk_size):
                    stop = min(start + vjp_chunk_size, output_width)
                    call_offset += 1
                    gradient = torch.autograd.grad(
                        final_hidden[row, target],
                        replacement,
                        grad_outputs=identity[start:stop],
                        is_grads_batched=True,
                        retain_graph=call_offset < call_count,
                    )[0]
                    row_totals[row, start:stop] += gradient[
                        :, row, target
                    ].to(
                        accumulation_dtype
                    )
        for row, valid in enumerate(positions):
            row_pair_count = sum(
                1
                for target in valid
                for source in valid
                if source == target or (include_future and source <= target)
            )
            if row_pair_count:
                row_average = row_totals[row] / row_pair_count
                total = row_average if total is None else total + row_average
                context_count += 1
        pair_count += sum(
            len(valid) * (len(valid) + 1) // 2 if include_future else len(valid)
            for valid in positions
        )
    if total is None or pair_count == 0 or context_count == 0:
        raise ValueError("No valid source-target pairs were supplied")
    return JacobianEstimate(
        values=(total / context_count).detach().cpu(),
        pair_count=pair_count,
        context_count=context_count,
        layer_index=layer_index,
    )


def token_vectors_from_jacobian(
    adapter: ModelAdapter,
    jacobian: Tensor,
    *,
    token_ids: Sequence[int] | None = None,
) -> Tensor:
    """Construct J-lens vectors as rows of ``W_U J_l``."""
    expected = (adapter.hidden_size, adapter.hidden_size)
    if tuple(jacobian.shape) != expected:
        raise ValueError(f"jacobian must have shape {expected}")
    weight = adapter.unembedding_weight(token_ids).to(
        device=jacobian.device, dtype=torch.float32
    )
    return weight @ jacobian.to(torch.float32)
