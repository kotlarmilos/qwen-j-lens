"""Tiny local adapters used only by offline tests."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from qwen_j_lens.adapter import ModelRun


class ToyBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(width, width, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        causal_mean = hidden.cumsum(dim=1) / torch.arange(
            1, hidden.shape[1] + 1, device=hidden.device
        ).view(1, -1, 1)
        return hidden + torch.tanh(self.projection(hidden) + 0.1 * causal_mean)


class ToyAdapter:
    def __init__(self, *, width: int = 4, vocab_size: int = 9, layers: int = 3) -> None:
        torch.manual_seed(7)
        self.embedding = nn.Embedding(vocab_size, width)
        self._layers = nn.ModuleList(ToyBlock(width) for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size, bias=False)
        self._width = width
        self._vocab_size = vocab_size
        self.chat_calls = 0

    @property
    def model_id(self) -> str:
        return "local/toy"

    @property
    def layers(self) -> Sequence[nn.Module]:
        return self._layers

    @property
    def hidden_size(self) -> int:
        return self._width

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        output_hidden_states: bool = False,
    ) -> ModelRun:
        hidden = self.embedding(input_ids)
        states = [hidden]
        for layer in self._layers:
            hidden = layer(hidden)
            states.append(hidden)
        final = self.normalize(hidden)
        return ModelRun(
            logits=self.unembed(final),
            final_pre_norm=hidden,
            hidden_states=tuple(states + [final]) if output_hidden_states else None,
        )

    def normalize(self, hidden: Tensor) -> Tensor:
        return self.norm(hidden)

    def unembed(self, hidden: Tensor) -> Tensor:
        return self.head(hidden)

    def unembedding_weight(self, token_ids: Sequence[int] | None = None) -> Tensor:
        if token_ids is None:
            return self.head.weight
        return self.head.weight.index_select(
            0, torch.tensor(token_ids, dtype=torch.long)
        )

    def tokenize_raw(self, prompts: Sequence[str]) -> tuple[Tensor, Tensor]:
        rows = [[ord(character) % self.vocab_size for character in prompt] for prompt in prompts]
        width = max(map(len, rows))
        ids = torch.zeros((len(rows), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(rows):
            ids[index, : len(row)] = torch.tensor(row)
            mask[index, : len(row)] = 1
        return ids, mask

    def tokenize_chat(self, prompt: str) -> tuple[Tensor, Tensor]:
        self.chat_calls += 1
        ids, mask = self.tokenize_raw([prompt])
        prefix = torch.tensor([[self.vocab_size - 1]], dtype=torch.long)
        return torch.cat((prefix, ids), dim=1), torch.cat(
            (torch.ones_like(prefix), mask), dim=1
        )

    def token_strings(self, token_ids: Sequence[int]) -> list[str]:
        return [f"token-{token_id}" for token_id in token_ids]

    def single_token_id(self, text: str) -> int:
        return ord(text[-1]) % self.vocab_size

    def decode(self, token_ids: Sequence[int]) -> str:
        return " ".join(self.token_strings(token_ids))

    @property
    def stop_token_ids(self) -> tuple[int, ...]:
        return ()


class ScalarCausalAdapter:
    """Analytic adapter where final[t] is sum(hidden[:t+1] squared)."""

    def __init__(self) -> None:
        self.embedding = nn.Embedding(8, 1)
        with torch.no_grad():
            self.embedding.weight[:, 0] = torch.arange(8, dtype=torch.float32)
        self.identity = nn.Identity()

    @property
    def model_id(self) -> str:
        return "local/scalar"

    @property
    def layers(self) -> Sequence[nn.Module]:
        return [self.identity]

    @property
    def hidden_size(self) -> int:
        return 1

    @property
    def vocab_size(self) -> int:
        return 2

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        output_hidden_states: bool = False,
    ) -> ModelRun:
        hidden = self.identity(self.embedding(input_ids))
        final = hidden.square().cumsum(dim=1)
        logits = self.unembed(self.normalize(final))
        return ModelRun(
            logits=logits,
            final_pre_norm=final,
            hidden_states=(hidden, final, self.normalize(final))
            if output_hidden_states
            else None,
        )

    def normalize(self, hidden: Tensor) -> Tensor:
        return 5 * hidden

    def unembed(self, hidden: Tensor) -> Tensor:
        return torch.cat((hidden, 2 * hidden), dim=-1)

    def unembedding_weight(self, token_ids: Sequence[int] | None = None) -> Tensor:
        weight = torch.tensor([[1.0], [2.0]])
        if token_ids is None:
            return weight
        return weight.index_select(0, torch.tensor(token_ids, dtype=torch.long))

    def tokenize_raw(self, prompts: Sequence[str]) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def tokenize_chat(self, prompt: str) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def token_strings(self, token_ids: Sequence[int]) -> list[str]:
        return [str(value) for value in token_ids]

    def single_token_id(self, text: str) -> int:
        return int(text)

    def decode(self, token_ids: Sequence[int]) -> str:
        return " ".join(str(value) for value in token_ids)

    @property
    def stop_token_ids(self) -> tuple[int, ...]:
        return ()
