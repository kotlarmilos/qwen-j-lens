"""Evaluation schemas and trajectory-based functional workspace protocols."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence, cast

import torch
from torch import Tensor

from qwen_j_lens.adapter import CachedForwardAdapter, ModelAdapter
from qwen_j_lens.capture import replace_layer_activation
from qwen_j_lens.interventions import (
    ablate_controlled_span,
    steer,
    swap_coordinates,
)

WorkspaceProperty = Literal[
    "verbal_report",
    "directed_modulation",
    "internal_reasoning",
    "selectivity",
]
LensKind = Literal["jacobian", "logit"]

WORKSPACE_PROPERTIES: tuple[WorkspaceProperty, ...] = (
    "verbal_report",
    "directed_modulation",
    "internal_reasoning",
    "selectivity",
)


@dataclass(frozen=True)
class EvaluationExample:
    """A preregistered prompt, probe trajectory, and causal comparison."""

    id: str
    property: WorkspaceProperty
    prompt: str
    target_concepts: tuple[str, ...]
    control_prompt: str | None = None
    control_concepts: tuple[str, ...] = ()
    measurement_steps: tuple[int, ...] = ()
    intervention: Mapping[str, Any] | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EvaluationExample:
        property_name = str(value["property"])
        if property_name not in WORKSPACE_PROPERTIES:
            raise ValueError(f"Unknown workspace property: {property_name}")
        measurement_steps = tuple(
            int(step) for step in value.get("measurement_steps", ())
        )
        if any(step < 0 for step in measurement_steps):
            raise ValueError("measurement_steps must contain nonnegative generation steps")
        if property_name == "internal_reasoning" and not measurement_steps:
            raise ValueError("Internal reasoning examples require measurement_steps")
        return cls(
            id=str(value["id"]),
            property=cast(WorkspaceProperty, property_name),
            prompt=str(value["prompt"]),
            target_concepts=tuple(str(item) for item in value["target_concepts"]),
            control_prompt=(
                str(value["control_prompt"]) if value.get("control_prompt") else None
            ),
            control_concepts=tuple(
                str(item) for item in value.get("control_concepts", ())
            ),
            measurement_steps=measurement_steps,
            intervention=value.get("intervention"),
        )


@dataclass(frozen=True)
class GenerationTrace:
    text: str
    token_ids: tuple[int, ...]
    activation_trajectory: tuple[float, ...]


@dataclass(frozen=True)
class EvaluationResult:
    """One paired measurement with raw activation trajectories."""

    example_id: str
    property: WorkspaceProperty
    lens: LensKind
    seed: int
    baseline_text: str
    intervention_text: str
    baseline_target_score: float
    intervention_target_score: float
    baseline_trajectory: tuple[float, ...] = ()
    intervention_trajectory: tuple[float, ...] = ()
    measurement_steps: tuple[int, ...] = ()
    baseline_control_score: float | None = None
    intervention_control_score: float | None = None

    @property
    def target_effect(self) -> float:
        return self.intervention_target_score - self.baseline_target_score

    @property
    def control_effect(self) -> float | None:
        if self.baseline_control_score is None or self.intervention_control_score is None:
            return None
        return self.intervention_control_score - self.baseline_control_score

    @property
    def target_damage(self) -> float:
        return abs(self.target_effect)

    @property
    def control_damage(self) -> float | None:
        effect = self.control_effect
        return abs(effect) if effect is not None else None

    @property
    def selectivity_gap(self) -> float | None:
        damage = self.control_damage
        return self.target_damage - damage if damage is not None else None

    def to_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            {
                "target_effect": self.target_effect,
                "control_effect": self.control_effect,
                "target_damage": self.target_damage,
                "control_damage": self.control_damage,
                "selectivity_gap": self.selectivity_gap,
            }
        )
        return value


def summarize_results(
    results: Sequence[EvaluationResult],
) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate signed effects and consistently absolute damage metrics."""
    groups: dict[tuple[str, str], list[EvaluationResult]] = {}
    for result in results:
        groups.setdefault((result.lens, result.property), []).append(result)
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for (lens, property_name), group in sorted(groups.items()):
        target_effects = [result.target_effect for result in group]
        values = {
            "n": float(len(group)),
            "mean_target_effect": sum(target_effects) / len(target_effects),
            "mean_target_damage": sum(result.target_damage for result in group)
            / len(group),
        }
        controlled = [result for result in group if result.control_damage is not None]
        if controlled:
            control_effects = [
                cast(float, result.control_effect) for result in controlled
            ]
            control_damages = [
                cast(float, result.control_damage) for result in controlled
            ]
            values["mean_control_effect"] = sum(control_effects) / len(controlled)
            values["mean_control_damage"] = sum(control_damages) / len(controlled)
            values["mean_selectivity_gap"] = (
                values["mean_target_damage"] - values["mean_control_damage"]
            )
        summary.setdefault(lens, {})[property_name] = values
    return summary


def load_examples(path: str | Path) -> list[EvaluationExample]:
    """Load and validate JSONL prompt records."""
    examples = [
        EvaluationExample.from_mapping(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not examples:
        raise ValueError("Evaluation prompt file is empty")
    if len({example.id for example in examples}) != len(examples):
        raise ValueError("Evaluation example IDs must be unique")
    return examples


def concept_score(text: str, concepts: Sequence[str]) -> float:
    """Return the fraction of declared concepts present as case-insensitive terms."""
    if not concepts:
        return 0.0
    normalized = text.casefold()
    matches = [
        bool(
            re.search(
                rf"(?<!\w){re.escape(concept.strip().casefold())}(?!\w)",
                normalized,
            )
        )
        for concept in concepts
    ]
    return sum(matches) / len(matches)


def _vector_lookup(vectors: Mapping[str, Tensor], name: str) -> Tensor:
    exact = vectors.get(name)
    if exact is not None:
        return exact
    normalized = {key.strip().casefold(): value for key, value in vectors.items()}
    try:
        return normalized[name.strip().casefold()]
    except KeyError as error:
        raise ValueError(f"Controlled token `{name}` is absent from the vector bank") from error


def _unit_vector(vector: Tensor) -> Tensor:
    norm = vector.to(torch.float32).norm()
    if norm <= 1e-12:
        raise ValueError("Controlled token vectors must have nonzero norm")
    return vector / norm.to(vector)


def _probe_score(
    hidden: Tensor, concepts: Sequence[str], vectors: Mapping[str, Tensor]
) -> float:
    scores = []
    for concept in concepts:
        vector = _vector_lookup(vectors, concept).to(hidden)
        scores.append(
            float(torch.nn.functional.cosine_similarity(hidden, vector, dim=0).item())
        )
    return sum(scores) / len(scores)


def generate_with_trajectory(
    adapter: ModelAdapter,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    layer_index: int,
    probe_concepts: Sequence[str],
    vectors: Mapping[str, Tensor],
    edit: Callable[[Tensor], Tensor] | None = None,
) -> GenerationTrace:
    """Generate from a chat template and capture one probe score per surface token."""
    input_ids, attention_mask = adapter.tokenize_chat(prompt)
    cached = isinstance(adapter, CachedForwardAdapter)
    step_input_ids = input_ids
    cache: object | None = None
    generated: list[int] = []
    trajectory: list[float] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            def capture_and_edit(hidden: Tensor) -> Tensor:
                edited = edit(hidden) if edit is not None else hidden
                if probe_concepts:
                    trajectory.append(
                        _probe_score(edited[0, -1], probe_concepts, vectors)
                    )
                return edited

            with replace_layer_activation(
                adapter.layers[layer_index], capture_and_edit
            ):
                if cached:
                    run, cache = adapter.forward_cached(
                        step_input_ids, attention_mask, cache
                    )
                else:
                    run = adapter.forward(input_ids, attention_mask)
            next_logits = run.logits[0, -1]
            if temperature == 0:
                next_token = int(next_logits.argmax().item())
            else:
                probabilities = torch.softmax(next_logits / temperature, dim=-1)
                next_token = int(torch.multinomial(probabilities, 1).item())
            generated.append(next_token)
            if next_token in adapter.stop_token_ids:
                break
            next_input = torch.tensor([[next_token]], device=input_ids.device)
            if cached:
                step_input_ids = next_input
            else:
                input_ids = torch.cat((input_ids, next_input), dim=1)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        (1, 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ),
                dim=1,
            )
    return GenerationTrace(
        text=adapter.decode(generated),
        token_ids=tuple(generated),
        activation_trajectory=tuple(trajectory),
    )


def build_intervention(
    vectors: Mapping[str, Tensor], specification: Mapping[str, Any]
) -> Callable[[Tensor], Tensor]:
    """Build an edit over the explicitly supplied controlled token vectors."""
    kind = str(specification["kind"])
    if kind == "steer":
        vector = _unit_vector(
            _vector_lookup(vectors, str(specification["target"]))
        )
        coefficient = float(specification.get("coefficient", 1.0))
        return lambda hidden: steer(hidden, vector, coefficient=coefficient)
    if kind == "coordinate_swap":
        source = _unit_vector(
            _vector_lookup(vectors, str(specification["source"]))
        )
        target = _unit_vector(
            _vector_lookup(vectors, str(specification["target"]))
        )
        strength = float(specification.get("strength", 1.0))
        return lambda hidden: swap_coordinates(
            hidden, source, target, strength=strength
        )
    if kind == "controlled_subset_ablation":
        names = tuple(str(name) for name in specification.get("tokens", vectors))
        bank = torch.stack(tuple(_vector_lookup(vectors, name) for name in names))
        strength = float(specification.get("strength", 1.0))
        return lambda hidden: ablate_controlled_span(
            hidden, bank, strength=strength
        )
    raise ValueError(f"Unknown intervention kind: {kind}")


def _trajectory_score(trace: GenerationTrace, steps: Sequence[int]) -> float:
    selected = tuple(steps) if steps else tuple(range(len(trace.activation_trajectory)))
    if not selected:
        raise ValueError("No activation trajectory steps were captured")
    if max(selected) >= len(trace.activation_trajectory):
        raise ValueError(
            "A preregistered measurement step exceeds the generated trajectory"
        )
    return sum(trace.activation_trajectory[step] for step in selected) / len(selected)


def evaluate_examples(
    adapter: ModelAdapter,
    examples: Sequence[EvaluationExample],
    vectors: Mapping[str, Tensor],
    *,
    layer_index: int,
    seeds: Sequence[int],
    max_new_tokens: int,
    temperature: float = 0.0,
    lens: LensKind = "jacobian",
) -> list[EvaluationResult]:
    """Run paired chat evaluations with generation-time activation trajectories."""
    results: list[EvaluationResult] = []
    for seed in seeds:
        for example in examples:
            probe = (
                example.target_concepts
                if example.property in {"directed_modulation", "internal_reasoning"}
                else ()
            )
            edit = (
                build_intervention(vectors, example.intervention)
                if example.intervention
                else None
            )
            baseline_prompt = (
                example.control_prompt
                if example.property == "directed_modulation"
                else example.prompt
            )
            if baseline_prompt is None:
                raise ValueError(
                    f"Directed modulation example `{example.id}` needs a control prompt"
                )
            torch.manual_seed(seed)
            baseline = generate_with_trajectory(
                adapter,
                baseline_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                layer_index=layer_index,
                probe_concepts=probe,
                vectors=vectors,
            )
            torch.manual_seed(seed)
            intervention = generate_with_trajectory(
                adapter,
                example.prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                layer_index=layer_index,
                probe_concepts=probe,
                vectors=vectors,
                edit=edit,
            )

            if probe:
                baseline_target_score = _trajectory_score(
                    baseline, example.measurement_steps
                )
                intervention_target_score = _trajectory_score(
                    intervention, example.measurement_steps
                )
            else:
                baseline_target_score = concept_score(
                    baseline.text, example.target_concepts
                )
                intervention_target_score = concept_score(
                    intervention.text, example.target_concepts
                )

            baseline_control_score: float | None = None
            intervention_control_score: float | None = None
            if example.control_prompt and example.property != "directed_modulation":
                control_concepts = example.control_concepts or example.target_concepts
                torch.manual_seed(seed + 1)
                baseline_control = generate_with_trajectory(
                    adapter,
                    example.control_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    layer_index=layer_index,
                    probe_concepts=(),
                    vectors=vectors,
                )
                torch.manual_seed(seed + 1)
                intervention_control = generate_with_trajectory(
                    adapter,
                    example.control_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    layer_index=layer_index,
                    probe_concepts=(),
                    vectors=vectors,
                    edit=edit,
                )
                baseline_control_score = concept_score(
                    baseline_control.text, control_concepts
                )
                intervention_control_score = concept_score(
                    intervention_control.text, control_concepts
                )

            results.append(
                EvaluationResult(
                    example_id=example.id,
                    property=example.property,
                    lens=lens,
                    seed=seed,
                    baseline_text=baseline.text,
                    intervention_text=intervention.text,
                    baseline_target_score=baseline_target_score,
                    intervention_target_score=intervention_target_score,
                    baseline_trajectory=baseline.activation_trajectory,
                    intervention_trajectory=intervention.activation_trajectory,
                    measurement_steps=example.measurement_steps,
                    baseline_control_score=baseline_control_score,
                    intervention_control_score=intervention_control_score,
                )
            )
    return results


EVALUATION_EXAMPLE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Qwen J-lens evaluation example",
    "type": "object",
    "additionalProperties": False,
    "required": ["id", "property", "prompt", "target_concepts"],
    "properties": {
        "id": {"type": "string", "minLength": 1},
        "property": {"enum": list(WORKSPACE_PROPERTIES)},
        "prompt": {"type": "string", "minLength": 1},
        "target_concepts": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "control_prompt": {"type": ["string", "null"]},
        "control_concepts": {"type": "array", "items": {"type": "string"}},
        "measurement_steps": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
        },
        "intervention": {"type": ["object", "null"]},
    },
}


EVALUATION_RESULT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Qwen J-lens evaluation result",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "example_id",
        "property",
        "lens",
        "seed",
        "baseline_text",
        "intervention_text",
        "baseline_target_score",
        "intervention_target_score",
        "baseline_trajectory",
        "intervention_trajectory",
        "measurement_steps",
        "baseline_control_score",
        "intervention_control_score",
        "target_effect",
        "control_effect",
        "target_damage",
        "control_damage",
        "selectivity_gap",
    ],
    "properties": {
        "example_id": {"type": "string"},
        "property": {"enum": list(WORKSPACE_PROPERTIES)},
        "lens": {"enum": ["jacobian", "logit"]},
        "seed": {"type": "integer"},
        "baseline_text": {"type": "string"},
        "intervention_text": {"type": "string"},
        "baseline_target_score": {"type": "number"},
        "intervention_target_score": {"type": "number"},
        "baseline_trajectory": {"type": "array", "items": {"type": "number"}},
        "intervention_trajectory": {"type": "array", "items": {"type": "number"}},
        "measurement_steps": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
        },
        "baseline_control_score": {"type": ["number", "null"]},
        "intervention_control_score": {"type": ["number", "null"]},
        "target_effect": {"type": "number"},
        "control_effect": {"type": ["number", "null"]},
        "target_damage": {"type": "number", "minimum": 0},
        "control_damage": {"type": ["number", "null"], "minimum": 0},
        "selectivity_gap": {"type": ["number", "null"]},
    },
}


def validate_result_mapping(value: Mapping[str, Any]) -> None:
    """Validate generated rows against the in-package result schema."""
    required = set(EVALUATION_RESULT_SCHEMA["required"])
    properties = set(EVALUATION_RESULT_SCHEMA["properties"])
    keys = set(value)
    if keys != properties or not required.issubset(keys):
        raise ValueError(
            f"Evaluation result keys differ from schema: missing={sorted(required - keys)}, "
            f"extra={sorted(keys - properties)}"
        )
    if value["property"] not in WORKSPACE_PROPERTIES:
        raise ValueError("Evaluation result has an unknown property")
    if value["lens"] not in {"jacobian", "logit"}:
        raise ValueError("Evaluation result has an unknown lens")
    if not isinstance(value["example_id"], str) or not value["example_id"]:
        raise ValueError("Evaluation result `example_id` must be a nonempty string")
    if not isinstance(value["seed"], int):
        raise ValueError("Evaluation result `seed` must be an integer")
    if not isinstance(value["baseline_text"], str) or not isinstance(
        value["intervention_text"], str
    ):
        raise ValueError("Evaluation result texts must be strings")
    for key in (
        "baseline_target_score",
        "intervention_target_score",
        "target_effect",
        "target_damage",
    ):
        if not isinstance(value[key], (int, float)):
            raise ValueError(f"Evaluation result `{key}` must be numeric")
    for key in (
        "baseline_control_score",
        "intervention_control_score",
        "control_effect",
        "control_damage",
        "selectivity_gap",
    ):
        if value[key] is not None and not isinstance(value[key], (int, float)):
            raise ValueError(f"Evaluation result `{key}` must be numeric or null")
    for key in ("baseline_trajectory", "intervention_trajectory"):
        if not isinstance(value[key], (list, tuple)) or not all(
            isinstance(item, (int, float)) for item in value[key]
        ):
            raise ValueError(f"Evaluation result `{key}` must be a numeric array")
    if not isinstance(value["measurement_steps"], (list, tuple)) or not all(
        isinstance(item, int) and item >= 0 for item in value["measurement_steps"]
    ):
        raise ValueError("Evaluation result `measurement_steps` must be nonnegative integers")
    expected_target_effect = (
        value["intervention_target_score"] - value["baseline_target_score"]
    )
    if abs(value["target_effect"] - expected_target_effect) > 1e-9:
        raise ValueError("Evaluation result target effect is inconsistent")
    if abs(value["target_damage"] - abs(value["target_effect"])) > 1e-9:
        raise ValueError("Evaluation result target damage is inconsistent")
    if value["control_effect"] is None:
        if value["control_damage"] is not None or value["selectivity_gap"] is not None:
            raise ValueError("Control-derived metrics must all be null together")
    else:
        if (
            value["baseline_control_score"] is None
            or value["intervention_control_score"] is None
        ):
            raise ValueError("Control effects require both control scores")
        expected_control_effect = (
            value["intervention_control_score"] - value["baseline_control_score"]
        )
        if abs(value["control_effect"] - expected_control_effect) > 1e-9:
            raise ValueError("Evaluation result control effect is inconsistent")
        if abs(value["control_damage"] - abs(value["control_effect"])) > 1e-9:
            raise ValueError("Evaluation result control damage is inconsistent")
        expected_gap = value["target_damage"] - value["control_damage"]
        if abs(value["selectivity_gap"] - expected_gap) > 1e-9:
            raise ValueError("Evaluation result selectivity gap is inconsistent")
