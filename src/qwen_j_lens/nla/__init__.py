"""Natural Language Autoencoder machinery for open-weight interpretability.

This package reproduces the structure of Anthropic's Natural Language
Autoencoders on the same ``ModelAdapter`` abstraction used by the J-lens code,
and adds a faithfulness audit that follows the ``RECAP`` critique. The machinery
is exercised offline against a tiny local transformer. Real open-weight runs
defer to an accelerator, exactly as the J-lens path does.
"""

from __future__ import annotations

from qwen_j_lens.nla.activations import ActivationBatch, collect_layer_activations
from qwen_j_lens.nla.autoencoder import (
    NaturalLanguageAutoencoder,
    Reconstructor,
    Verbalizer,
    reconstruction_error,
)
from qwen_j_lens.nla.faithfulness import (
    FaithfulnessReport,
    code_usage,
    gist_versus_specific,
    independent_reader_gap,
)
from qwen_j_lens.nla.train import TrainResult, train_autoencoder

__all__ = [
    "ActivationBatch",
    "collect_layer_activations",
    "NaturalLanguageAutoencoder",
    "Reconstructor",
    "Verbalizer",
    "reconstruction_error",
    "FaithfulnessReport",
    "code_usage",
    "gist_versus_specific",
    "independent_reader_gap",
    "TrainResult",
    "train_autoencoder",
]
