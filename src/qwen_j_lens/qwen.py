"""Hugging Face adapter for dense Qwen causal language models."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor, nn

from qwen_j_lens.adapter import ModelRun
from qwen_j_lens.errors import MissingModelError


def load_qwen_tokenizer(
    model_id: str,
    *,
    local_files_only: bool = True,
    revision: str | None = None,
) -> Any:
    """Load only the tokenizer so controlled tokens can be checked cheaply."""
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise MissingModelError(
            "Qwen support is not installed. Install the optional dependency with "
            "`pip install -e '.[qwen]'`."
        ) from error
    try:
        tokenizer_kwargs: dict[str, object] = {
            "local_files_only": local_files_only
        }
        if revision is not None:
            tokenizer_kwargs["revision"] = revision
        return AutoTokenizer.from_pretrained(
            model_id,
            **tokenizer_kwargs,
        )
    except (OSError, ValueError) as error:
        mode = "local Hugging Face cache" if local_files_only else "configured source"
        raise MissingModelError(
            f"Tokenizer `{model_id}` is unavailable in the {mode}. No download was "
            "attempted when local_files_only=true."
        ) from error


class QwenAdapter:
    """Adapt a Hugging Face Qwen model and tokenizer to ``ModelAdapter``."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        model_id: str,
        *,
        enable_thinking: bool = False,
    ) -> None:
        backbone = getattr(model, "model", None)
        layers = getattr(backbone, "layers", None)
        norm = getattr(backbone, "norm", None)
        lm_head = getattr(model, "lm_head", None)
        config = getattr(model, "config", None)
        if layers is None or norm is None or lm_head is None or config is None:
            raise TypeError("Expected a Hugging Face Qwen causal LM with model.layers and lm_head")
        self._model = model
        self._tokenizer = tokenizer
        self._model_id = model_id
        self._layers: Sequence[nn.Module] = layers
        self._norm: nn.Module = norm
        self._lm_head: nn.Module = lm_head
        self._hidden_size = int(config.hidden_size)
        self._vocab_size = int(config.vocab_size)
        self._enable_thinking = enable_thinking

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        local_files_only: bool = True,
        revision: str | None = None,
        device_map: str | None = None,
        dtype: str = "auto",
        enable_thinking: bool = False,
    ) -> QwenAdapter:
        """Load Qwen through Transformers, offline by default."""
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as error:
            raise MissingModelError(
                "Qwen support is not installed. Install the optional dependency with "
                "`pip install -e '.[qwen]'`."
            ) from error

        torch_dtype: str | torch.dtype
        torch_dtype = dtype
        if dtype != "auto":
            parsed = getattr(torch, dtype, None)
            if not isinstance(parsed, torch.dtype):
                raise ValueError(f"Unknown torch dtype: {dtype}")
            torch_dtype = parsed
        try:
            tokenizer = load_qwen_tokenizer(
                model_id,
                local_files_only=local_files_only,
                revision=revision,
            )
            model_kwargs: dict[str, object] = {
                "local_files_only": local_files_only,
                "device_map": device_map,
                "torch_dtype": torch_dtype,
            }
            if revision is not None:
                model_kwargs["revision"] = revision
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        except (OSError, ValueError) as error:
            mode = "local Hugging Face cache" if local_files_only else "configured source"
            raise MissingModelError(
                f"Model `{model_id}` is unavailable in the {mode}. No download was attempted "
                "when local_files_only=true."
            ) from error
        model.eval()
        model.requires_grad_(False)
        return cls(
            model,
            tokenizer,
            model_id,
            enable_thinking=enable_thinking,
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def layers(self) -> Sequence[nn.Module]:
        return self._layers

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

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
        run, _cache = self._run_model(
            input_ids,
            attention_mask,
            output_hidden_states=output_hidden_states,
            use_cache=False,
            cache=None,
        )
        return run

    def forward_cached(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        cache: object | None,
    ) -> tuple[ModelRun, object]:
        """Run an incremental generation step with the model's native KV cache."""
        run, updated_cache = self._run_model(
            input_ids,
            attention_mask,
            output_hidden_states=False,
            use_cache=True,
            cache=cache,
        )
        if updated_cache is None:
            raise RuntimeError("Qwen did not return a cache for incremental generation")
        return run, updated_cache

    def _run_model(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        *,
        output_hidden_states: bool,
        use_cache: bool,
        cache: object | None,
    ) -> tuple[ModelRun, object | None]:
        pre_norm: list[Tensor] = []

        def capture_pre_norm(
            _module: nn.Module, inputs: tuple[object, ...]
        ) -> None:
            if not inputs or not isinstance(inputs[0], Tensor):
                raise TypeError("Qwen final norm did not receive a residual tensor")
            pre_norm.append(inputs[0])

        handle = self._norm.register_forward_pre_hook(capture_pre_norm)
        try:
            model_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "output_hidden_states": output_hidden_states,
                "use_cache": use_cache,
                "return_dict": True,
            }
            if use_cache:
                model_kwargs["past_key_values"] = cache
            output = self._model(**model_kwargs)
        finally:
            handle.remove()
        if len(pre_norm) != 1:
            raise RuntimeError("Qwen final pre-normalization residual was not captured")
        hidden_states = (
            tuple(output.hidden_states)
            if output_hidden_states and output.hidden_states is not None
            else None
        )
        return (
            ModelRun(
                logits=output.logits,
                final_pre_norm=pre_norm[0],
                hidden_states=hidden_states,
            ),
            getattr(output, "past_key_values", None),
        )

    def normalize(self, hidden: Tensor) -> Tensor:
        return self._norm(hidden)

    def unembed(self, hidden: Tensor) -> Tensor:
        return self._lm_head(hidden)

    def unembedding_weight(self, token_ids: Sequence[int] | None = None) -> Tensor:
        weight = self._lm_head.weight
        if token_ids is None:
            return weight
        indices = torch.as_tensor(token_ids, device=weight.device, dtype=torch.long)
        return weight.index_select(0, indices)

    def tokenize_raw(self, prompts: Sequence[str]) -> tuple[Tensor, Tensor]:
        batch = self._tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )
        device = next(self._model.parameters()).device
        return batch["input_ids"].to(device), batch["attention_mask"].to(device)

    def tokenize_chat(self, prompt: str) -> tuple[Tensor, Tensor]:
        batch = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=self._enable_thinking,
        )
        device = next(self._model.parameters()).device
        return batch["input_ids"].to(device), batch["attention_mask"].to(device)

    def token_strings(self, token_ids: Sequence[int]) -> list[str]:
        return [self._tokenizer.decode([token_id]) for token_id in token_ids]

    def single_token_id(self, text: str) -> int:
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"Controlled token `{text}` encodes to {len(token_ids)} tokens, expected one"
            )
        return int(token_ids[0])

    def decode(self, token_ids: Sequence[int]) -> str:
        return str(self._tokenizer.decode(list(token_ids), skip_special_tokens=True))

    @property
    def stop_token_ids(self) -> tuple[int, ...]:
        generation_config = getattr(self._model, "generation_config", None)
        value = getattr(generation_config, "eos_token_id", None)
        if value is None:
            return ()
        if isinstance(value, int):
            return (value,)
        return tuple(int(token_id) for token_id in value)
