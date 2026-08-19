# Reading a residual stream two ways: J-lens and Natural Language Autoencoders

This repository is an offline-first release candidate for two interpretability
methods on open-weight Qwen models. The first tests the functional workspace
properties described in
[Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html).
The second reproduces the structure of
[Natural Language Autoencoders](https://transformer-circuits.pub/2026/nla/index.html)
and adds the faithfulness audit motivated by the critique
[Train the Model, Not the Reader](https://arxiv.org/abs/2607.20379).
It targets Qwen through Hugging Face Transformers and keeps model downloads
opt-in. It contains checksum-verified Qwen3-8B result artifacts but no model
weights.

The J-lens path tests verbal report, directed modulation, internal reasoning,
and selectivity. The autoencoder path verbalizes an activation into a discrete
code, reconstructs the activation, and audits whether the code is a faithful
description. These are functional properties. Positive evidence would not
establish consciousness or subjective experience.

## Repository state

`artifacts/manifest.json` is the claim boundary. It distinguishes unit-tested
local machinery, development runs, publication measurements, and bounded negative
results.

The implementation includes:

- a typed model adapter protocol and a Hugging Face Qwen adapter
- residual activation capture and replacement hooks
- a logit-lens baseline with final normalization
- selected-token VJPs from the final pre-normalization residual
- controlled token vectors estimated directly as selected rows of `W_U J_l`
- matched logit-lens vector evaluations at every tested layer
- controlled-subset projection, ablation, steering, and coordinate swaps
- cached chat generation with activation trajectories and explicit thinking-mode control
- evaluation input and output JSON schemas
- deterministic NPY and canonical JSON artifact storage
- a natural-language autoencoder with a straight-through discrete code
- an independent-reader faithfulness gap and a gist-versus-specific probe
- an artifact-only Hugging Face Space

## Install

Use Python 3.11 or newer.

```bash
python3.13 -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
pytest
```

Install the heavyweight model stack only on the accelerator host.

```bash
pip install -e '.[qwen,test]'
```

The default loader passes `local_files_only=True`. The CLI downloads only when
the operator explicitly supplies `--allow-download`.

## Configurations

`configs/smoke.toml` uses the verified public identifier
`Qwen/Qwen3-4B-Instruct-2507` for development. The primary publication
configuration uses `Qwen/Qwen3-8B`. Both configurations pin an exact model
revision and disable thinking mode so the paired protocol has the same chat
semantics on both models.

Validate configuration and prompt paths without loading a model.

```bash
qwen-j-lens validate-config --config configs/smoke.toml
qwen-j-lens validate-config --config configs/qwen-publication.toml
qwen-nla validate-nla-config --config configs/nla-smoke.toml
qwen-nla validate-nla-config --config configs/nla-publication.toml
```

## Method

For source layer `l`, the primary source averages
`d h_final_pre_norm[t'] / d h_l[t]` over prompts and all valid `t' >= t`.
This gives `J_l` before the model's final RMSNorm. The full matrix is practical
on the toy model but not a sensible publication path for Qwen. The Qwen command
therefore differentiates only the declared token readouts and stores those
selected rows of `W_U J_l` directly. It does not materialize the hidden-width
square matrix. Lens readout applies the model's final normalization only after
`J_l h_l`. Every valid source-target pair receives equal weight.

The logit lens applies the final normalization and unembedding directly to an
intermediate residual stream. That is equivalent to replacing the average
Jacobian with the identity. The J-lens corrects layer-dependent
representational changes and should be compared against this baseline at every
tested layer. The generation-time trajectory probe is different. It scores
cosine similarity against raw Jacobian or unembedding vector banks without final
normalization, so those trajectory magnitudes are not calibrated lens
probabilities.

The source paper defines J-space using a broad overcomplete vocabulary bank and
sparse nonnegative combinations. This release estimates only 13 declared token
vectors. Its ablation is therefore named controlled-subset ablation. It does
not estimate vocabulary-wide J-space occupancy, top-k J-space contents, or the
paper's gradient-pursuit decomposition.

Calibration text uses raw tokenizer input. Evaluation uses
`apply_chat_template` with `add_generation_prompt=True` and stops on every EOS
ID declared by the model generation configuration. Directed modulation records
the target-vector trajectory while the surface task is generated. Internal
reasoning scores preregistered generation steps from those trajectories rather
than using output keywords as the primary score.

The primary source averages over 1,000 pretraining-like prompts. The publication
corpus here is a deterministic 1,000-context sample from the pinned
`Salesforce/wikitext` `wikitext-2-raw-v1` training split. It is real,
pretraining-like text, but it is narrower than the source distribution and
remains a methodological deviation. Rebuild it with
`python scripts/prepare_publication_corpus.py`.

## Natural language autoencoder

The autoencoder path reads the same residual stream and asks a different
question. A verbalizer maps an activation to a short discrete code, and a
reconstructor maps the code back to an activation. Reconstruction error measures
how much the code left out. The offline machinery uses a fixed symbol vocabulary
in place of the words a language-model verbalizer would emit, which keeps the
loop deterministic and runnable without an accelerator. The publication run uses
the same fixed-symbol surrogate on the accelerator. It is not a reproduction of a
language-model verbalizer that emits natural-language text.

Reconstruction alone is not proof. The audit adds two checks motivated by the
critique. The independent-reader gap trains a second reconstructor on a held-out
split and scores it on unseen codes; a large gap exposes a co-adapted private
code that no independent reader can decode. The gist-versus-specific probe fits a
linear probe for a labeled attribute on the original activations and on the
reconstructions; a large drop exposes a code that kept only the coarse gist. Both
audits return numbers, thresholds, and flags.

`configs/nla-smoke.toml` develops on `Qwen/Qwen3-4B-Instruct-2507`.
`configs/nla-publication.toml` reads the same preregistered layer 24 on
`Qwen/Qwen3-8B` that the J-lens evaluation uses, so the two methods can be
compared at one location.

## Measured Qwen3-8B runs

The primary configuration ran at layer 24 on one A100 PCIe 80 GB GPU. It uses ten
100-context shards followed by a checksum-validated, context-weighted merge. The
merged estimate covers 1,000 contexts and 15,620,093 position pairs. Five-seed
evaluation records one small matched-prompt trajectory contrast for directed
modulation and confirms that subset ablation and coordinate swapping change the
coordinates they directly manipulate. Because the probe reads the edited vector at
the same layer, this is an intervention-plumbing check rather than a downstream
reasoning effect. The verbal-report metric is floored at zero and selectivity is
ceilinged at one, with identical generated text across both arms, so neither text
protocol measures an effect. This is a controlled 13-token subset, not
vocabulary-wide J-space.

```bash
./scripts/colab_publication.sh
```

The default `JOB=plan` prints the exact estimation and evaluation commands. The
measured arrays, raw rows, summaries, source revisions, and hashes are under
`artifacts/colab/`.

The autoencoder path has its own wrapper. Its 1,000-prompt run collects 159,325
layer-24 activation rows and trains for 4,000 steps. The configured fixed-symbol
code collapses to two strings, reaches relative reconstruction error 1.005, and
does not beat the zero predictor. Both the co-adapted and independent readers fail
at about 0.999, so their near-zero gap is not evidence of faithfulness.

```bash
./scripts/colab_nla.sh
```

## Artifact integrity

Numerical arrays use uncompressed NPY so identical arrays produce identical
bytes. Metadata uses sorted canonical JSON. Every artifact record carries a
SHA-256 digest. The Space recursively discovers record files, requires every
referenced array, verifies SHA-256 values, and displays no result after any
validation failure.

## Project links

The local article is [`article.md`](article.md). These are the expected
publication locations. Nothing has been uploaded by this release candidate.

- GitHub repository at https://github.com/kotlarmilos/qwen-j-lens
- Hugging Face model artifact at https://huggingface.co/kotlarmilos/qwen-j-lens
- Hugging Face dataset at https://huggingface.co/datasets/kotlarmilos/qwen-j-lens-evaluation
- Hugging Face Space at https://huggingface.co/spaces/kotlarmilos/qwen-j-lens

## Citation and license

The methodological sources are:

Wes Gurnee and collaborators. *Verbalizable Representations Form a Global
Workspace in Language Models*. Transformer Circuits, July 2026.
https://transformer-circuits.pub/2026/workspace/index.html

*Natural Language Autoencoders*. Transformer Circuits, 2026.
https://transformer-circuits.pub/2026/nla/index.html

*Train the Model, Not the Reader*. arXiv:2607.20379.
https://arxiv.org/abs/2607.20379

Code and project text are available under the MIT License. Qwen models retain
their own licenses and terms.
