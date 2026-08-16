"""Build the deterministic 1,000-prompt controlled calibration corpus."""

from __future__ import annotations

import json
from pathlib import Path

ROLES = (
    "engineer",
    "teacher",
    "biologist",
    "historian",
    "cook",
    "mechanic",
    "librarian",
    "architect",
    "nurse",
    "musician",
)
ACTIONS = (
    "compares two explanations",
    "checks the available evidence",
    "writes a concise summary",
    "finds a boundary condition",
    "organizes a sequence of steps",
    "reviews an unexpected result",
    "explains a difficult tradeoff",
    "tests an initial assumption",
    "records a careful observation",
    "chooses between two plans",
)
CONTEXTS = (
    "before making a decision",
    "while keeping uncertainty visible",
    "without adding unsupported details",
    "and then asks what could fail",
    "using plain language for the result",
    "after separating facts from guesses",
    "while preserving the original constraints",
    "and checks the answer one more time",
    "before sharing the work with a colleague",
    "and notes which evidence would change the conclusion",
)


def main() -> None:
    destination = Path(__file__).resolve().parents[1] / "data/publication_calibration_prompts.jsonl"
    rows = []
    for role_index, role in enumerate(ROLES):
        for action_index, action in enumerate(ACTIONS):
            for context_index, context in enumerate(CONTEXTS):
                identifier = role_index * 100 + action_index * 10 + context_index
                rows.append(
                    json.dumps(
                        {
                            "id": f"publication-{identifier:04d}",
                            "text": f"The {role} {action} {context}.",
                            "provenance": "deterministic synthetic controlled calibration",
                        },
                        sort_keys=True,
                    )
                )
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
