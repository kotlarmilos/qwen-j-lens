"""Model abstraction used by lensing and intervention code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelRun:
    """Normalized output from a causal language model."""

    logits: Tensor
    final_pre_norm: Tensor
    hidden_states: tuple[Tensor, ...] | None = None


@runtime_checkable
class CachedForwardAdapter(Protocol):
    """Optional incremental-generation interface."""

    def forward_cached(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        cache: object | None,
    ) -> tuple[ModelRun, object]:
        """Run one generation step and return the updated model cache."""


@runtime_checkable
class ModelAdapter(Protocol):
    """Minimal causal-transformer interface required by this project."""

    @property
    def model_id(self) -> str:
        """Stable model identifier."""

    @property
    def layers(self) -> Sequence[nn.Module]:
        """Transformer blocks in execution order."""

    @property
    def hidden_size(self) -> int:
        """Residual-stream width."""

    @property
    def vocab_size(self) -> int:
        """Output vocabulary size."""

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        output_hidden_states: bool = False,
    ) -> ModelRun:
        """Run the model without sampling."""

    def normalize(self, hidden: Tensor) -> Tensor:
        """Apply the model's final residual normalization."""

    def unembed(self, hidden: Tensor) -> Tensor:
        """Map normalized residual vectors to vocabulary logits."""

    def unembedding_weight(self, token_ids: Sequence[int] | None = None) -> Tensor:
        """Return rows of the output unembedding matrix."""

    def tokenize_raw(self, prompts: Sequence[str]) -> tuple[Tensor, Tensor]:
        """Tokenize raw calibration text without a chat template."""

    def tokenize_chat(self, prompt: str) -> tuple[Tensor, Tensor]:
        """Tokenize one user turn with an assistant generation prompt."""

    def token_strings(self, token_ids: Sequence[int]) -> list[str]:
        """Decode individual vocabulary entries."""

    def single_token_id(self, text: str) -> int:
        """Encode text that must correspond to exactly one vocabulary token."""

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode a generated token sequence."""

    @property
    def stop_token_ids(self) -> tuple[int, ...]:
        """Token IDs that terminate generation."""
