"""Artifact-only Gradio viewer."""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

try:
    from artifact_validation import validate_artifacts
except ImportError:
    from space.artifact_validation import validate_artifacts

ROOT = Path(os.environ.get("QWEN_J_LENS_ARTIFACTS", "artifacts"))


def status() -> tuple[str, object]:
    _complete, message, payload = validate_artifacts(ROOT)
    return message, payload


with gr.Blocks(title="Qwen J-lens Artifact Viewer") as demo:
    gr.Markdown("# Qwen J-lens Artifact Viewer")
    gr.Markdown(
        "This viewer requires uploaded artifacts and never downloads model weights."
    )
    message = gr.Markdown()
    payload = gr.JSON(label="Verified manifest or evaluation summary")
    refresh = gr.Button("Check artifacts")
    refresh.click(status, outputs=[message, payload])
    demo.load(status, outputs=[message, payload])


if __name__ == "__main__":
    demo.launch()
