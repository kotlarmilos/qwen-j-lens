from __future__ import annotations

import sys
import types

import pytest

from qwen_j_lens.errors import MissingModelError
from qwen_j_lens.qwen import QwenAdapter


def test_missing_model_error_is_offline_and_actionable(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class Missing:
        @classmethod
        def from_pretrained(cls, _model_id: str, **kwargs: object) -> None:
            calls.append(dict(kwargs))
            raise OSError("not cached")

    fake = types.ModuleType("transformers")
    fake.AutoTokenizer = Missing
    fake.AutoModelForCausalLM = Missing
    monkeypatch.setitem(sys.modules, "transformers", fake)

    with pytest.raises(MissingModelError, match="No download was attempted"):
        QwenAdapter.from_pretrained("Qwen/definitely-missing", local_files_only=True)

    assert calls == [{"local_files_only": True}]
