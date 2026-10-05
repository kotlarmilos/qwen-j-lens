# Qwen3-8B selected-token J-lens and fixed-symbol autoencoder

This repository contains one measured Qwen3-8B interpretability study. It tests
13 selected-token J-lens directions and a fixed-symbol autoencoder on the pinned
model revision `b968826d9c46dd6066d109eabc6255188de91218`.

The study ran on an NVIDIA A100 80 GB PCIe. The J-lens result is limited to a
controlled token subset and unedited matched-prompt contrasts. The autoencoder
collapses to two code strings and reaches centered relative reconstruction error
`1.00515`. It does not beat the constant-mean predictor at `1.0`.

The experiment does not establish a reasoning effect, vocabulary-wide J-space,
faithful natural-language explanations, or a zero-predictor comparison. No
labeled specific-target probe was run.

Read `article.md` for the complete method, result, and limits.

## Evidence

The retained evidence is indexed by `artifacts/manifest.json`.

- `artifacts/colab/qwen3-8b-layer-24-merged` contains the merged 13 by 4,096
  direction array.
- `artifacts/colab/qwen3-8b-layer-24-evaluate` contains 80 matched evaluation
  rows and the summary.
- `artifacts/colab/nla-qwen3-8b` contains the fixed-symbol codes, training
  history, reader measurements, and run record.
- `artifacts/qwen3-4b-bit-budget-layer18` contains the retained seed-stability
  diagnostic used to interpret the collapse.

Qwen weights are not included.

## Layer indexing

The primary J-lens hooks block index 24. Its output is hidden-state index 25.
The primary autoencoder reads hidden-state index 24, which is the output after
block 23. These are adjacent residual states and are not a matched-layer
comparison.

`src/qwen_j_lens/cross_method.py` keeps the separate aligned behavior. For
hidden-state index `m`, it captures the J-lens at block index `m - 1`.

## Install and validate

Use Python 3.11 or newer.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
python -m pytest
qwen-j-lens verify-artifacts --root artifacts
python scripts/prepare_publication_corpus.py --verify-only
```

Install the model stack only on the accelerator host.

```bash
pip install -e '.[qwen,test,data]'
```

Model downloads remain opt-in. The loaders use local files unless a command
receives `--allow-download`.

## Reproduction

The exact publication configurations are listed below.

- `configs/qwen-publication.toml`
- `configs/nla-publication.toml`

The wrappers print the measured job plans by default.

```bash
./scripts/colab_publication.sh
./scripts/colab_nla.sh
```

The calibration corpus can be rebuilt from the pinned WikiText revision.

```bash
python scripts/prepare_publication_corpus.py
```

## Main limits

The J-lens direction bank contains 13 tokens rather than a broad vocabulary
frame. Directed modulation compares two unedited prompts. The internal
reasoning protocol reads a coordinate changed at the same layer. The
verbal-report and selectivity scores saturate and do not measure an effect.

The autoencoder uses fixed symbol IDs rather than natural language. Its two
readers both fail at about `0.999`, so their near-zero gap is not evidence of
faithfulness. The retained primary artifacts do not contain the target matrix
needed for a zero-predictor calculation.

The measured artifacts are also published at
`https://huggingface.co/kotlarmilos/qwen-j-lens`.
