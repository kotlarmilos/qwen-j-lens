from __future__ import annotations

import pytest
import torch

from qwen_j_lens.adapter import ModelRun
from qwen_j_lens.evaluation import (
    EVALUATION_RESULT_SCHEMA,
    EvaluationExample,
    EvaluationResult,
    concept_score,
    evaluate_examples,
    generate_with_trajectory,
    load_examples,
    summarize_results,
    validate_result_mapping,
)
from toy import ToyAdapter


def test_evaluation_schema_and_selectivity_summary() -> None:
    example = EvaluationExample.from_mapping(
        {
            "id": "x",
            "property": "selectivity",
            "prompt": "target",
            "target_concepts": ["answer"],
            "control_prompt": "control",
        }
    )
    result = EvaluationResult(
        example_id=example.id,
        property=example.property,
        lens="jacobian",
        seed=1,
        baseline_text="a",
        intervention_text="b",
        baseline_target_score=0.2,
        intervention_target_score=0.8,
        baseline_control_score=0.7,
        intervention_control_score=0.6,
    )

    summary = summarize_results([result])["jacobian"]["selectivity"]
    assert result.target_effect == pytest.approx(0.6)
    assert result.control_effect == pytest.approx(-0.1)
    assert result.target_damage == pytest.approx(0.6)
    assert result.control_damage == pytest.approx(0.1)
    assert summary["mean_selectivity_gap"] == pytest.approx(0.5)

    reversed_result = EvaluationResult(
        example_id="y",
        property="selectivity",
        lens="jacobian",
        seed=1,
        baseline_text="a",
        intervention_text="b",
        baseline_target_score=0.8,
        intervention_target_score=0.2,
        baseline_control_score=0.6,
        intervention_control_score=0.7,
    )
    assert reversed_result.target_damage == pytest.approx(result.target_damage)
    assert reversed_result.control_damage == pytest.approx(result.control_damage)


def test_registered_examples_load_and_concept_matching_is_bounded() -> None:
    examples = load_examples("data/evaluation_prompts.jsonl")

    assert {example.property for example in examples} == {
        "verbal_report",
        "directed_modulation",
        "internal_reasoning",
        "selectivity",
    }
    assert concept_score("Dog, not dogmatic.", ["dog"]) == 1.0
    assert concept_score("dogmatic", ["dog"]) == 0.0


def test_generated_mapping_matches_declared_schema_properties() -> None:
    result = EvaluationResult(
        example_id="schema",
        property="internal_reasoning",
        lens="jacobian",
        seed=3,
        baseline_text="C",
        intervention_text="B",
        baseline_target_score=0.5,
        intervention_target_score=0.2,
        baseline_trajectory=(0.4, 0.6),
        intervention_trajectory=(0.1, 0.3),
        measurement_steps=(0, 1),
    )
    mapping = result.to_mapping()

    assert set(mapping) == set(EVALUATION_RESULT_SCHEMA["properties"])
    assert set(EVALUATION_RESULT_SCHEMA["required"]) == set(mapping)
    validate_result_mapping(mapping)


class StopToyAdapter(ToyAdapter):
    @property
    def stop_token_ids(self) -> tuple[int, ...]:
        return (4, 5)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        output_hidden_states: bool = False,
    ) -> ModelRun:
        run = super().forward(
            input_ids,
            attention_mask,
            output_hidden_states=output_hidden_states,
        )
        logits = torch.zeros_like(run.logits)
        logits[..., 4] = 10
        return ModelRun(
            logits=logits,
            final_pre_norm=run.final_pre_norm,
            hidden_states=run.hidden_states,
        )


def test_generation_uses_chat_template_and_captures_surface_trajectory() -> None:
    adapter = StopToyAdapter()
    vectors = {" blue": torch.ones(adapter.hidden_size)}

    trace = generate_with_trajectory(
        adapter,
        "Count while holding blue in mind.",
        max_new_tokens=5,
        temperature=0.0,
        layer_index=1,
        probe_concepts=("blue",),
        vectors=vectors,
    )

    assert adapter.chat_calls == 1
    assert trace.token_ids == (4,)
    assert len(trace.activation_trajectory) == 1


def test_internal_reasoning_scores_preregistered_activation_not_keyword() -> None:
    adapter = StopToyAdapter()
    vectors = {" blue": torch.ones(adapter.hidden_size)}
    example = EvaluationExample.from_mapping(
        {
            "id": "reasoning-trace",
            "property": "internal_reasoning",
            "prompt": "Reason silently, then answer.",
            "target_concepts": ["blue"],
            "measurement_steps": [0],
            "intervention": {
                "kind": "steer",
                "target": " blue",
                "coefficient": 2.0,
            },
        }
    )

    result = evaluate_examples(
        adapter,
        [example],
        vectors,
        layer_index=1,
        seeds=[1],
        max_new_tokens=1,
    )[0]

    assert concept_score(result.baseline_text, ["blue"]) == 0.0
    assert result.baseline_target_score == result.baseline_trajectory[0]
    assert result.intervention_target_score == result.intervention_trajectory[0]
    assert result.intervention_target_score != result.baseline_target_score


class CachedToyAdapter(StopToyAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.cached_input_lengths: list[int] = []
        self.cached_calls = 0

    def forward_cached(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: object | None,
    ) -> tuple[ModelRun, object]:
        self.cached_input_lengths.append(input_ids.shape[1])
        run = super().forward(input_ids, attention_mask)
        logits = torch.zeros_like(run.logits)
        logits[..., 1 if self.cached_calls == 0 else 4] = 10
        self.cached_calls += 1
        return (
            ModelRun(
                logits=logits,
                final_pre_norm=run.final_pre_norm,
                hidden_states=run.hidden_states,
            ),
            object(),
        )


def test_generation_uses_incremental_cache_when_adapter_supports_it() -> None:
    adapter = CachedToyAdapter()

    trace = generate_with_trajectory(
        adapter,
        "Use the cache.",
        max_new_tokens=3,
        temperature=0.0,
        layer_index=1,
        probe_concepts=(),
        vectors={},
    )

    assert trace.token_ids == (1, 4)
    assert adapter.cached_input_lengths[1:] == [1]
