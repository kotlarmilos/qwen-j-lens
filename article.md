---
title: "Testing selected-token J-lens directions and a fixed-symbol autoencoder on Qwen3-8B"
thumbnail: /blog/assets/qwen-j-lens-fixed-symbol/thumbnail.png
authors:
- user: kotlarmilos
---

# Testing selected-token J-lens directions and a fixed-symbol autoencoder on Qwen3-8B

## TL;DR

I test two small interpretability methods on a pinned Qwen3-8B model. The first
estimates 13 selected-token J-lens directions. The second compresses residual
activations into fixed symbol sequences and reconstructs them.

The study ran on an NVIDIA A100 80 GB PCIe with model revision
`b968826d9c46dd6066d109eabc6255188de91218`.

The J-lens path produces a small contrast between two unedited prompts and
confirms that coordinate edits change the coordinate read back at the same
layer. This does not establish a change in reasoning. The fixed-symbol
autoencoder collapses to two code strings and reaches centered relative
reconstruction error `1.00515`. It does not beat the constant-mean predictor at
`1.0`.

The useful result is the failure diagnosis. A direct readback can look like an
intervention result when it only checks the edited coordinate. A small reader
gap can look reassuring when both readers fail. A large nominal code space can
still collapse to almost no used codes.

## Scope

The J-lens implementation estimates directions for 13 declared output tokens.
It does not estimate a vocabulary-wide J-space or reproduce the source method
in full.

The autoencoder emits fixed symbol IDs. It does not emit natural-language
descriptions. Its result applies to this surrogate and training recipe.

The main release claim is limited to the following work.

- one controlled 13-token Qwen3-8B J-lens study
- one Qwen3-8B fixed-symbol autoencoder study
- one smaller seed-stability diagnostic used to interpret the collapse

No labeled specific-target probe was run. Nothing here establishes faithful
natural-language explanations or a reasoning effect.

## Inputs and provenance

Both primary methods use `Qwen/Qwen3-8B` at revision
`b968826d9c46dd6066d109eabc6255188de91218`.

The calibration input is a deterministic 1,000-context sample from
`Salesforce/wikitext` with configuration `wikitext-2-raw-v1`. The dataset
revision is `b08601e04326c79dfdd32d625aee71d232d685c3`.

The corpus SHA-256 is shown below.

```text
7919f2c32e2fa0da9ab79690b00844d5d7ce6d52f76a2c8ca6dc010e8b891e3c
```

The publication environment records one NVIDIA A100 80 GB PCIe, CUDA 12.4,
PyTorch 2.4.1, and Transformers 5.13.1. Qwen weights remain outside Git.

## Selected-token J-lens method

Let `h_l` be the residual activation at a chosen block and let
`h_final_pre_norm` be the final residual before RMS normalization. The local
downstream map has the following Jacobian.

```text
J_l = d h_final_pre_norm / d h_l
```

A full hidden-width square Jacobian is unnecessary for 13 tokens. For an
unembedding row `w_t`, the required direction is shown below.

```text
w_t^T J_l = d (w_t . h_final_pre_norm) / d h_l
```

The implementation estimates these rows with selected-token vector-Jacobian
products. It averages valid source and target position pairs inside each
context, then averages contexts equally. Ten 100-context shards are merged with
context-count weighting and contiguous-range checks.

The final merged array has shape 13 by 4,096 and covers these measurements.

| Measurement | Value |
| --- | ---: |
| Calibration contexts | 1,000 |
| Position pairs | 15,620,093 |
| Selected token directions | 13 |
| Estimation shards | 10 |

The stored direction rows are selected rows of `W_U J_l`. The trajectory
evaluation scores cosine similarity against raw direction banks. It does not
apply the full normalized lens readout and does not produce calibrated token
probabilities.

## J-lens evaluation

The paired evaluation contains eight prompts, five seeds, and two vector banks.
The result file contains 80 raw rows with generated text and activation
trajectories.

| Protocol | Jacobian signed mean | Raw logit signed mean | Interpretation |
| --- | ---: | ---: | --- |
| Directed modulation | 0.00791 | 0.00151 | Unedited matched-prompt contrast |
| Internal reasoning | -0.00277 | -0.00424 | Same-layer edit readback |
| Verbal report | 0.00000 | 0.00000 | Score is floored in both arms |
| Selectivity | 0.00000 | 0.00000 | Score is ceilinged in both arms |

Directed modulation compares two different unedited prompts. It is not a causal
intervention. The Jacobian and raw logit banks also use different scales, so
their values are not interchangeable effect sizes.

The internal reasoning rows apply an ablation or coordinate swap and then read
the edited coordinate while the hook is active. The Jacobian mean absolute
change is `0.00839`. This confirms the edit arithmetic. It does not show that
the change propagated into later reasoning.

The text scorers provide no useful result. Verbal-report scores are zero in both
arms. Selectivity scores are one in both arms. The paired generated text is
identical for those protocols. There is no positive control showing that the
chosen edit strength should change decoded text.

Five seeds do not provide five independent informative outcomes. They produce
only four distinct directed-modulation values and three distinct edit-readback
values per bank.

## Fixed-symbol autoencoder method

The autoencoder maps each residual row to eight symbol slots. Each slot chooses
one of 256 fixed IDs. A reconstructor maps the selected IDs back to a residual
vector.

The forward pass uses hard symbol choices and a straight-through gradient. The
configured code has 64 nominal bits per row, but nominal capacity does not show
how many code strings training actually uses.

Relative reconstruction error uses centered activation energy.

```text
mean(||reconstruction - target||^2)
------------------------------------------------
mean(||target - mean(target)||^2)
```

The constant-mean predictor is exactly `1.0` on this scale. The zero predictor
depends on the activation mean and must be computed from the same retained
target matrix.

The main artifacts retain the codes and measured relative errors. They do not
retain the target activations or reconstructions. They therefore do not support
a zero-predictor comparison.

## Autoencoder result

The run collects 159,325 hidden-state rows and trains for 4,000 steps with one
seed.

| Measurement | Result |
| --- | ---: |
| Centered relative reconstruction error | 1.005153 |
| Constant-mean predictor | 1.000000 |
| Unique code strings | 2 |
| Symbols used | 8 of 256 |
| Mean slot entropy | 0.029588 |
| Co-adapted reader error | 0.998870 |
| Independent reader error | 0.998877 |
| Reader gap | 0.0000069 |

The autoencoder does not beat the constant-mean predictor. It uses two code
strings over 159,325 rows. The configured 64-bit space is therefore mostly
unused.

The reader gap is near zero because both readers fail at nearly the same level.
It is not evidence that the code is faithful. The main run supplies no labeled
specific targets, so the attribute-retention probe is not run.

## Layer indexing

The primary configurations both contain the number 24, but they refer to
different states.

The J-lens hooks block index 24. Its output is `hidden_states[25]`. The
autoencoder reads `hidden_states[24]`, which is the output after block 23. The
primary runs therefore use adjacent residual states and are not a matched-layer
comparison.

The separate aligned runner keeps one explicit rule. If the autoencoder reads
hidden-state index `m`, the J-lens captures block index `m - 1`. The retained
test checks this mapping. The aligned runner is not used to turn the primary
single-method results into a matched comparison.

## Collapse diagnostic

A smaller Qwen3-4B diagnostic repeats fixed-symbol fits on 9,877 activation rows
with twenty seeds per selected configuration. This is not a publication result
for Qwen3-8B. It helps distinguish code capacity from fitting instability.

| Code shape | Nominal bits | Seeds below 1.0 | Best error | Median error |
| --- | ---: | ---: | ---: | ---: |
| 4 by 64 | 24 | 2 of 20 | 0.12996 | 1.00662 |
| 16 by 64 | 96 | 3 of 20 | 0.06161 | 1.00683 |
| 32 by 64 | 192 | 8 of 20 | 0.06667 | 1.00630 |
| 32 by 64 with wider decoder | 192 | 9 of 20 | 0.05670 | 1.00648 |

Most fits fail to beat the constant mean. Some low-bit fits succeed, while some
higher-bit fits collapse. Nominal bit count alone does not explain the result.

A one-component PCA reconstruction reaches centered error `0.01054` on the same
matrix. It is an in-sample reference. Its coefficient count excludes the shared
basis and mean, so it is not a matched storage comparison.

The diagnostic supports a narrow conclusion. The tested discrete bottleneck is
sensitive to initialization and training. It does not establish an information
capacity limit.

## Reproduction and verification

The exact primary configurations are listed below.

- `configs/qwen-publication.toml`
- `configs/nla-publication.toml`

The input builder pins the WikiText dataset revision and verifies the local
corpus hash.

```bash
python scripts/prepare_publication_corpus.py --verify-only
```

The publication wrappers print the measured accelerator job plans.

```bash
./scripts/colab_publication.sh
./scripts/colab_nla.sh
```

The retained artifacts and every nested record hash can be checked without a
model download.

```bash
qwen-j-lens verify-artifacts --root artifacts
```

The focused test suite covers selected-token Jacobian logic, centered
reconstruction, reader-gap arithmetic, layer-state alignment, shard weighting,
artifact corruption, path escape rejection, provenance hashes, and a
ToyAdapter intervention path.

## Conclusion

The selected-token J-lens code estimates the intended 13 directions and performs
the configured edits. The measurements do not establish a downstream reasoning
effect.

The fixed-symbol autoencoder fails its basic centered reconstruction check and
collapses to two code strings. Its near-zero reader gap does not rescue the
result because both readers fail.

The release preserves these bounded findings rather than treating saturated
scores, same-layer readbacks, or collapsed codes as positive interpretability
evidence.

## References

1. Anthropic. *Verbalizable Representations Form a Global Workspace in Language
   Models*. Transformer Circuits, 2026.
2. Anthropic. *Natural Language Autoencoders*. Transformer Circuits, 2026.
3. *Train the Model, Not the Reader*. arXiv:2607.20379.

The code and measured evidence are available in
[qwen-j-lens on GitHub](https://github.com/kotlarmilos/qwen-j-lens)
and
[qwen-j-lens on Hugging Face](https://huggingface.co/kotlarmilos/qwen-j-lens).
