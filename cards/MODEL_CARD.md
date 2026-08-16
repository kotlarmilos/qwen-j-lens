---
license: mit
library_name: transformers
tags:
  - interpretability
  - jacobian-lens
  - natural-language-autoencoder
  - qwen
  - mechanistic-interpretability
---

# Qwen interpretability artifact card

This repository contains analysis code and, after accelerator runs, derived
artifacts for two interpretability methods. The first is the J-lens. The second
is a natural-language autoencoder with a faithfulness audit. It does not
redistribute Qwen weights and it is not a new trained model.

## Intended use

The J-lens artifacts support controlled research on whether an open-weight Qwen
instruct model exhibits functional workspace properties described by Gurnee and
collaborators in July 2026. The autoencoder artifacts support research on whether
a natural-language description of an activation is faithful, following the
Natural Language Autoencoders method and the critique in *Train the Model, Not
the Reader*. Intended users are interpretability researchers who can inspect
causal interventions, matched controls, and reconstruction audits.

## Base models

Development uses `Qwen/Qwen3-4B-Instruct-2507`. The primary planned run uses
`Qwen/Qwen3-8B`. Both artifacts pin an upstream revision and disable thinking
mode during the paired evaluation. Refer to each upstream model card for its
license, limitations, and intended use.

## Artifact format

Each `controlled-token-vectors-layer-NNN.npy` array has shape
`[controlled_token_count, hidden_size]`. Its rows are estimated directly with
selected-token VJPs from the final pre-normalization residual. `records.json`
binds arrays to layer indices, token IDs, token strings, source-target pair
counts, and SHA-256 digests. The Qwen path does not materialize the full
hidden-width square Jacobian. Final RMSNorm is applied after `J_l h_l` during
lens readout. `run.json` records the model and method. The primary estimate is
partitioned into ten 100-context shards. The merge command verifies every array
hash and requires contiguous coverage of contexts 0 through 999 before writing
the layer-24 vectors.

The autoencoder run writes `codes.npy` with shape `[activation_rows, code_length]`,
a `training-history.json` trace, and a `faithfulness.json` summary. The summary
holds the relative reconstruction error, the independent-reader gap, the
gist-versus-specific probe scores when a labeled attribute is supplied, and the
code-usage statistics. `records.json` binds every array and JSON file to a
SHA-256 digest, and `run.json` records the model, layer, and method.

## Current status

Only tiny local toy-model paths are tested in this release candidate. The
manifest marks both Qwen runs as pending. There are no empirical reproduction
results to report.

## Limitations

The 13-token bank omits almost all vocabulary concepts and all selected
multi-token concepts. Controlled-subset ablation is not vocabulary-wide
J-space suppression, occupancy measurement, or top-k decomposition. The sparse
nonnegative utility is not the source paper's gradient-pursuit implementation.
Evaluation uses Qwen's chat template and records generation-time activation
trajectories with incremental KV caching. The same protocol is repeated with
logit-lens directions. Intervention effects can still reflect model-specific
tokenization, prompting, or generation behavior. Functional workspace evidence
does not establish consciousness.

The autoencoder machinery uses a discrete symbol code in place of real words in
the offline setting, so the local tests measure the audit mechanics rather than
the readability of natural-language descriptions. A low reconstruction error is
not sufficient evidence of a faithful description. The independent-reader gap and
the gist-versus-specific probe are the checks that separate a faithful code from
a co-adapted private code or a gist-only code.

The straight-through discrete bottleneck is unstable across random seeds. Refitting
one fixed configuration twenty times on real Qwen3-4B layer-18 activations shows the
fit collapses onto a near-constant code on most seeds, with a within-configuration
error spread up to 0.9510 that exceeds every difference between the configurations
tested. A single-seed reconstruction error from this bottleneck is a draw from a
bimodal distribution rather than a measurement, so it should be reported over several
seeds together with a trivial-predictor baseline. The evidence is in
`artifacts/qwen3-4b-bit-budget-layer18` and the diagnostic lives in
`src/qwen_j_lens/nla/capacity.py`.

## Project links

The project article is
[`article.md`](https://github.com/kotlarmilos/qwen-j-lens/blob/main/article.md).
Expected publication locations are the
[GitHub repository](https://github.com/kotlarmilos/qwen-j-lens),
[model artifact](https://huggingface.co/kotlarmilos/qwen-j-lens),
[evaluation dataset](https://huggingface.co/datasets/kotlarmilos/qwen-j-lens-evaluation),
and [Space](https://huggingface.co/spaces/kotlarmilos/qwen-j-lens).
They are not published by this release candidate.

## Source

Wes Gurnee and collaborators. *Verbalizable Representations Form a Global
Workspace in Language Models*. Transformer Circuits, July 2026.
https://transformer-circuits.pub/2026/workspace/index.html

*Natural Language Autoencoders*. Transformer Circuits, 2026.
https://transformer-circuits.pub/2026/nla/index.html

*Train the Model, Not the Reader*. arXiv:2607.20379.
https://arxiv.org/abs/2607.20379
