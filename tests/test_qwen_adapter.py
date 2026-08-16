from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import Tensor, nn

from qwen_j_lens.qwen import QwenAdapter


class ScaleNorm(nn.Module):
    def forward(self, hidden: Tensor) -> Tensor:
        return 3 * hidden


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity()])
        self.norm = ScaleNorm()


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeBackbone()
        self.embedding = nn.Embedding(10, 2)
        self.lm_head = nn.Linear(2, 10, bias=False)
        self.config = SimpleNamespace(hidden_size=2, vocab_size=10)
        self.generation_config = SimpleNamespace(eos_token_id=[7, 8])

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        output_hidden_states: bool,
        use_cache: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        hidden = self.embedding(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        normalized = self.model.norm(hidden)
        return SimpleNamespace(
            logits=self.lm_head(normalized),
            hidden_states=(hidden, normalized) if output_hidden_states else None,
        )


class FakeTokenizer:
    def __init__(self) -> None:
        self.raw_calls = 0
        self.chat_calls: list[dict[str, object]] = []

    def __call__(self, prompts: list[str], **kwargs: object) -> dict[str, Tensor]:
        self.raw_calls += 1
        return {
            "input_ids": torch.tensor([[1, 2]] * len(prompts)),
            "attention_mask": torch.ones((len(prompts), 2), dtype=torch.long),
        }

    def apply_chat_template(
        self, messages: list[dict[str, str]], **kwargs: object
    ) -> dict[str, Tensor]:
        self.chat_calls.append(dict(kwargs))
        return {
            "input_ids": torch.tensor([[3, 4, 5]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return [len(text) % 10]

    def decode(
        self, token_ids: list[int], *, skip_special_tokens: bool = False
    ) -> str:
        return ",".join(str(token_id) for token_id in token_ids)


def test_qwen_captures_pre_norm_and_separates_tokenizers() -> None:
    model = FakeModel()
    tokenizer = FakeTokenizer()
    adapter = QwenAdapter(model, tokenizer, "fake/qwen")

    raw_ids, _ = adapter.tokenize_raw(["calibration"])
    chat_ids, _ = adapter.tokenize_chat("evaluation")
    run = adapter.forward(chat_ids, output_hidden_states=True)

    expected_pre_norm = model.embedding(chat_ids)
    assert torch.allclose(run.final_pre_norm, expected_pre_norm)
    assert torch.allclose(run.logits, model.lm_head(3 * expected_pre_norm))
    assert raw_ids.shape == (1, 2)
    assert chat_ids.shape == (1, 3)
    assert tokenizer.raw_calls == 1
    assert tokenizer.chat_calls == [
        {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
            "enable_thinking": False,
        }
    ]
    assert adapter.stop_token_ids == (7, 8)
