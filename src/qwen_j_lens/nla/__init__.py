"""Fixed-symbol autoencoder experiments for open-weight interpretability.

This package implements a discrete reconstruction surrogate on the same
``ModelAdapter`` abstraction used by the selected-token J-lens experiment. It
does not implement a language-model verbalizer or reproduce the source Natural
Language Autoencoder method.
"""

from __future__ import annotations

from qwen_j_lens.nla.activations import ActivationBatch, collect_layer_activations
from qwen_j_lens.nla.autoencoder import (
    FixedSymbolAutoencoder,
    Reconstructor,
    SymbolEncoder,
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
    "FixedSymbolAutoencoder",
    "Reconstructor",
    "SymbolEncoder",
    "reconstruction_error",
    "FaithfulnessReport",
    "code_usage",
    "gist_versus_specific",
    "independent_reader_gap",
    "TrainResult",
    "train_autoencoder",
]
