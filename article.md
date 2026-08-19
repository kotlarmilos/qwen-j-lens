---
title: "Reading a residual stream two ways: J-lens and Natural Language Autoencoders"
thumbnail: /blog/assets/jlens-vs-nla/thumbnail.png
authors:
- user: kotlarmilos
---

# Reading a residual stream two ways: J-lens and Natural Language Autoencoders

## TL;DR

I want to know what a hidden activation in a language model means, and I want a
method that tells me when my answer is wrong. This repository studies two ways to
read the residual stream of an open-weight model and holds both to the same
standard. The first is Anthropic's J-lens, which reads an activation as a set of
directions the model itself uses to write tokens. The second is Anthropic's Natural
Language Autoencoder, which verbalizes an activation into text and then checks
whether that text carries the activation back. The two methods answer different
questions, and comparing them is the point of this project. On Qwen3-8B, the
controlled J-lens subset produces a small matched-prompt trajectory contrast and a
small direct edit-readback sanity check, while the two text metrics saturate and
cannot measure an effect. The fixed-symbol autoencoder collapses to two code strings
and fails to reconstruct better than zero, so its tiny independent-reader gap is not
evidence of faithfulness.

## The problem

The important word is experiment. I first tested the machinery against a tiny local
transformer, then ran the preregistered study on the pinned Qwen3-8B weights. The
repository includes the raw evaluation rows, numerical arrays, audit summaries,
source revisions, and hashes. A transformer carries information forward through its
residual stream. Pick an activation at layer `l` and token position `t`. Both methods
take that vector as their subject. They disagree about how to turn it into something
a person can read.

## Intuition

### The J-lens reads in output coordinates

A J-lens reads the activation in the model's own output coordinates. A small change
to the activation can affect the final residual state before the final RMSNorm at
every later position `t'`. The local first-order map is a Jacobian. The paper
averages that map across source positions, future target positions, and a broad
corpus of prompts. A high score says that an activation is aligned with a direction
that tends to make the model emit that token across contexts and future positions.

### The logit lens stays in the room

The logit lens applies final normalization and unembedding directly at an
intermediate layer. It is the special case where the Jacobian is the identity. That
shortcut is useful because residual connections keep coordinate systems somewhat
aligned, especially near the output. It is also exactly why the logit lens is a good
baseline rather than a substitute. Earlier layers can use representations that
downstream blocks rotate, gate, or otherwise transform before unembedding. The
J-lens estimates that downstream linearized transformation instead of assuming it
away. Every Qwen result therefore needs a layer-matched logit-lens comparison.

### The autoencoder verbalizes instead of projecting

The Natural Language Autoencoder asks a different question. It does not read the
activation in output coordinates. It writes a natural-language description of the
activation and then checks whether the description is enough to rebuild the
activation. A verbalizer turns an activation into text. A reconstructor turns that
text back into an activation. When the round trip returns to where it started, the
text is treated as a faithful explanation of what the activation carried. Text is
the bottleneck code. A description that omits what the activation encoded cannot
drive the reconstructor back to the original vector, so reconstruction error becomes
a measurable proxy for how much the description left out.

## The math

### The J-lens Jacobian and readout

The averaged Jacobian is

`J_l = E[d h_final_pre_norm[t'] / d h_l[t]]` for `t' >= t`

A Jacobian from one prompt mixes a general disposition to verbalize a concept with
whatever the model happens to be doing in that prompt. A corpus average is intended
to preserve the general disposition and wash out context-specific routing. After the
average map, the model's own final normalization and unembedding turn an
intermediate activation into token scores.

`lens(h_l) = softmax(W_U norm(J_l h_l))`

The token-indexed J-lens directions are the rows of `W_U J_l`. RMSNorm is not inside
that Jacobian or those vectors. It is applied after `J_l h_l` when the lens produces
a readout.

A full hidden-width by hidden-width Jacobian is straightforward on the toy model and
wasteful on a four or eight billion parameter model when the experiment controls
only thirteen token directions. The Qwen path computes selected-token VJPs instead.
Each batched VJP starts from the declared unembedding rows applied to the final
pre-normalization residual and differentiates those scores with respect to the
selected layer activation. This produces the required rows of `W_U J_l` directly
without storing the hidden-width square matrix. The estimator averages every valid
source and target pair with equal weight, then averages across calibration prompts.

### The autoencoder round trip

`h_l  ->  verbalizer  ->  text code  ->  reconstructor  ->  h_hat_l`

This implementation replaces the text code with a short sequence of discrete
symbols over a fixed vocabulary. Each symbol is only a surrogate for a word. The
verbalizer is a small network from the activation to per-slot symbol logits. A
straight-through argmax makes the code discrete on the forward pass while still
passing gradients to the verbalizer on the backward pass. The reconstructor embeds
the symbols and maps them back to the residual width. The accelerator run uses this
same fixed-symbol surrogate. It does not become a language-model text decoder, so
the autoencoder result below is not a reproduction of natural-language
verbalization.

### Reconstruction is necessary, not sufficient

The critique paper, *Train the Model, Not the Reader*, shows two ways a
high-reconstruction code can still mislead a human reader, and this repository
implements an audit for each. The first failure is a co-adapted private code. The
verbalizer and reconstructor can agree on a scheme that reconstructs well and yet
means nothing to anyone who did not train alongside them. The audit trains a second
reconstructor, an independent reader, on a held-out split of the same codes and
scores it on examples it never saw during training. When the code is a generalizable
description the gap is small. When the code is an example-specific private scheme the
gap grows. The gap, not the raw reconstruction error, is the faithfulness signal.
The second failure is a gist code. The audit fits a linear probe for a labeled
attribute on the original activations and again on the reconstructions. When the
reconstruction keeps only the gist the probe on reconstructions falls while the
probe on originals stays high. A third measurement reports how the code population
uses its vocabulary, because a code that assigns a distinct symbol string to every
example is memorizing rather than describing.

## In the code

The J-lens VJP estimator lives in `src/qwen_j_lens/jacobian.py`, the logit-lens
baseline in `src/qwen_j_lens/logit_lens.py`, activation capture in
`src/qwen_j_lens/capture.py`, the paired interventions in
`src/qwen_j_lens/interventions.py`, and the evaluation protocol in
`src/qwen_j_lens/evaluation.py`. The open-weight adapter and the Qwen loader live in
`src/qwen_j_lens/adapter.py` and `src/qwen_j_lens/qwen.py`. The autoencoder modules
live under `src/qwen_j_lens/nla/`. Activation collection is in `activations.py`, the
verbalizer, reconstructor, and straight-through code in `autoencoder.py`,
deterministic training in `train.py`, and the two faithfulness audits in
`faithfulness.py`. The J-lens command line is `src/qwen_j_lens/cli.py` and the
autoencoder command line is `src/qwen_j_lens/nla_cli.py`. `LEARNING.md` derives the
VJP method and the autoencoder audits from first principles with exact line
references.

The evaluation registers four workspace properties from the paper as paired causal
tests. Verbal report asks whether a concept present in J-space predicts what the
model reports and whether swapping concept coordinates redirects the report.
Directed modulation asks whether an instruction to hold a concept in mind raises
that concept's J-lens activation while surface output remains occupied with another
task. Internal reasoning asks whether intermediate answers appear before the final
response and whether targeted swaps or controlled-subset ablation redirect the
conclusion. Selectivity asks whether the same intervention damages deliberate
reasoning more than routine copying, grammar, and basic fluency. Each needs an
untreated baseline, a targeted intervention, matched controls, fixed seeds, raw
generations, and effect sizes, and each runs once with J-lens directions and once
with layer-matched logit-lens directions. Keyword scores help with triage, but
manual review remains necessary.

## Results and honesty boundary

The paper treats a broad vocabulary bank of J-lens directions as a sparse frame and
calls the active region J-space. This repository estimates only thirteen declared
directions, so it supports controlled-subset projection, ablation, steering, and
coordinate swaps, and it cannot support vocabulary-wide occupancy or top-k J-space
claims. Token strings, token IDs, layer indices, pair counts, model identifiers,
seeds, and array hashes all travel with the artifact. A token that does not encode
to exactly one Qwen vocabulary entry fails validation instead of silently changing
the question.

The local tests cover both methods and nothing about Qwen behavior. For the J-lens
they cover capture hooks, gradient shapes, exact averaging on an analytic toy model,
projection algebra, coordinate swaps, deterministic serialization, and offline model
errors. For the autoencoder they cover activation collection, the straight-through
code, deterministic training that lowers reconstruction error on a clustered
dataset, and the two faithfulness audits on constructed data where the right answer
is known. The independent-reader gap is exercised on memorized codes, where it must
open, and on generalizable cluster codes, where it must stay closed. The gist probe
is exercised on a reconstruction that destroys a specific attribute and on one that
preserves it.

Running both methods on the same preregistered layer 24 of the same open-weight
model is what makes the comparison possible. A concept that is present as a clean
J-lens direction and also survives the reconstruction round trip is a strong result.
A concept that shows up in one method and not the other is the more instructive case,
because it tells me whether the information is written in output coordinates,
carried in a form only a description can capture, or an artifact of one method's
machinery.

The artifact manifest makes the boundary machine-readable. The Space reads uploaded
artifacts only. It recursively parses record files, requires their referenced
arrays, and verifies every SHA-256 value. When any Qwen array or audit summary is
absent or corrupt, it displays no result. Neither method here can establish
subjective experience. The J-lens paper is careful to separate functional access
from phenomenal consciousness, and an open-weight replication should preserve that
distinction for both methods.

## The single-method Qwen3-8B result

The primary run used the pinned Qwen3-8B revision on one A100 PCIe 80 GB GPU. Ten
checksum-validated shards estimate thirteen controlled token directions at layer 24
from 1,000 WikiText contexts and 15,620,093 valid source-target position pairs. A
context-weighted merge produces one `13 x 4096` vector array. The held-out protocol
then evaluates eight paired prompts under five seeds with both the Jacobian and
layer-matched logit vector banks, producing eighty raw rows.

The trajectory metrics are cosine-similarity differences against those raw vector
banks. They are not token probabilities, and final RMSNorm is not applied in this
trajectory probe. Magnitudes from the two banks are therefore not directly
comparable as calibrated readout scores.

| Property | Comparison | Jacobian signed mean | Jacobian absolute damage | Logit signed mean | Logit absolute damage |
| --- | --- | ---: | ---: | ---: | ---: |
| Directed modulation | Matched prompts, no intervention | 0.00791 | 0.00791 | 0.00151 | 0.00296 |
| Internal-reasoning edit readback | Subset ablation and coordinate swap | -0.00277 | 0.00839 | -0.00424 | 0.00424 |

Directed modulation compares two different unedited prompts. It is a matched-prompt
contrast, not a causal lens intervention. The internal-reasoning rows are not a
causal reasoning result either. `reasoning-route` removes the full controlled span,
including the probed `C` direction, and `reasoning-plan` directly swaps the probed
`C` coordinate toward `B`. The probe then reads the edited vector at the same layer
while the hook remains active. It therefore checks that ablation and swapping alter
the coordinate they directly manipulate. It does not measure downstream propagation
or a change in reasoning.

The signed Jacobian mean also hides opposing edit readbacks: `reasoning-route` is
`+0.00562` under every seed, while `reasoning-plan` ranges from `-0.00903` to
`-0.01257`. The absolute-damage column makes that cancellation visible, but neither
quantity should be interpreted as a behavioral effect.

Five seeds do not provide five distinct outcomes here. Repeated generations collide,
leaving four distinct directed-modulation values and three distinct
internal-reasoning values per lens. These are small protocol observations, not
precise estimates.

The text metrics do not produce usable measurements. Every verbal-report score is
zero in both arms, every selectivity target and control score is one in both arms,
and every baseline and intervention generation is byte-identical. Verbal report is
at the floor, selectivity is at the ceiling, and no blinded manual review was run.
The controlled directions also have small cosine alignment with the layer-24
residual, so the configured edit strength provides no positive control that should
have changed decoded text. Reporting the arithmetic difference as a measured zero
would turn a null protocol into a null result.

More importantly, thirteen declared token directions are not the source paper's
broad overcomplete vocabulary bank. This run supports one matched-prompt trajectory
contrast and confirms that the intervention code changes the coordinate it directly
edits. It does not support a claim about downstream reasoning, vocabulary-wide
J-space, or a full functional workspace.

The autoencoder reads the same layer over the same 1,000 contexts, yielding 159,325
activation rows. Its code has eight slots over 256 symbols, the training schedule is
4,000 steps, and the full run uses one preregistered seed. It collapses.

| Autoencoder measurement | Result |
| --- | ---: |
| Relative reconstruction error | 1.00515 |
| Unique code strings | 2 |
| Symbols used | 8 of 256 |
| Mean per-slot entropy | 0.02959 |
| Co-adapted reader error | 0.99887 |
| Independent reader error | 0.99888 |
| Reader gap | 0.0000069 |

The zero predictor has relative error 1.0 by construction, so 1.005 means the code
does not reconstruct useful residual information. The independent-reader gap is
near zero only because both readers fail at the same level. Calling that
faithfulness would invert the audit. No labeled specific targets were supplied, so
the gist-versus-specific probe was not run. This result diagnoses one fixed-symbol
architecture and one seed. It does not show that the source paper's
natural-language verbalizer fails.

A development-scale run on the real `Qwen3-4B-Instruct-2507` weights now exists. It
ran on a single Apple Silicon machine, resident on CPU in bfloat16, fully offline
against the local Hugging Face cache. The J-lens estimate produced thirteen
controlled-token directions of width 2560 at layer 24 from two calibration prompts
and 291 position pairs. The layer-24 evaluation generated real continuations under
both the Jacobian lens and the logit lens and recorded small development-scale
causal effects, which is the honest outcome for directions estimated from only two
prompts. The autoencoder produced a reconstruction error of 0.023 and an
independent-reader gap of 0.0023 over 128 layer-18 activation rows. These artifacts
live under `local_development_runs` in the manifest and are labeled as
development-scale, not publication measurements. They demonstrate that the pipeline
runs end to end on the real model, and they do not stand in for a corpus-scale
J-space estimate or a preregistered 8B comparison.

A cross-method extension now measures something neither source paper reports,
whether the two methods agree on which concepts they recover from the same physical
residual stream. The J-lens and the autoencoder read different indices of the
hidden-state stack by default, so the runner aligns them. It captures the J-lens
directions at block 17 and reads the autoencoder activations at hidden state 18,
which are the same residual stream, and it scores both from the same four
calibration prompts. For each concept the ground truth is the model's own
next-token logit at every attended position. The J-lens fidelity is the held-out
coefficient of determination of that logit from the single stored direction. The
autoencoder retention is the fraction of that logit's linear recoverability that
survives the reconstruction, reusing the gist probe the faithfulness module already
defines. This is the difference between reproducing a paper and extending one,
because the matched-layer agreement question is only askable once both methods run
on the same open-weight residual stream.

The measured outcome is a clean negative result. Across sixty-four activation rows,
thirty-two of them held out over 547 position pairs, all thirteen controlled
concepts fall in the neither quadrant at the 0.5 thresholds. The autoencoder
reconstructs the residual with an error of 0.006 and still retains almost none of
any concept's next-token logit, which is the gist-code signature the faithfulness
audit was built to detect, now shown on the real weights across the whole concept
bank. The single J-lens direction, read without the final normalization, predicts
the model's own next-token logit weakly at this scale. The lesson is that a low
reconstruction error does not imply concept faithfulness, and that a clean linear
direction and a discrete description can both miss the specific attribute a residual
carries. The scale-up described below revises how much that 0.006 was ever worth.

Two honesty limits bound this result. The J-lens fidelity uses one stored direction
with no final normalization while the autoencoder retention uses a multivariate
probe, so the two axes differ in absolute scale and only their per-concept ordering
is directly comparable. The agreement statistics, a Spearman rank correlation of
0.53 and a Pearson correlation of 0.91, come from thirteen concepts and thirty-two
held-out rows, and the Pearson value is dominated by a single concept, so both are
development-scale signals rather than publication measurements. The comparison still
earns its place because even this negative result separates reconstruction quality
from concept faithfulness in a way that neither method shows alone.

## What broke at publication scale

Running the identical study on Qwen3-8B is where the interesting result appears.
Every method block in `configs/cross-8b-publication.toml` is copied verbatim from
the 4B development configuration, so model scale, capture layer, corpus size, and
device placement are the only variables. The run captured layer 24 of 36 across 256
calibration prompts on one RTX A6000, producing 39,697 activation rows and 3,752,262
position pairs, against 64 rows and 547 pairs at development scale.

The autoencoder did not scale. Relative reconstruction error came out at 1.008
against 0.006 at development scale, every concept's retention was exactly 0.0, and
the reported Spearman and Pearson correlations were both exactly 0.0. Those zeros
are the tell. A correlation of exactly zero across thirteen concepts is not a
measurement, it is what a constant vector returns, so the run had to be diagnosed
rather than reported.

The natural reading is that the larger model is harder. That reading is wrong, and
`src/qwen_j_lens/nla/capacity.py` exists to show why. It measures three things the
original metric leaves implicit. The first is a pair of trivial baselines. Relative
error divides by the mean squared norm of the target, so a zero predictor scores
exactly 1.0 by construction and a constant mean predictor scores 0.950 on this
sample. Any code that does not beat 0.950 has learned less than a single fixed
vector. The trained code scored 1.008, so it learned nothing at all.

The second is a sweep over activation-row count on one fixed activation matrix, with
model, layer, and every hyperparameter held constant. The only variable is how many
rows the code has to cover.

| Activation rows | Relative error | Distinct codes used |
| --- | --- | --- |
| 64 | 0.0147 | 4 |
| 128 | 1.0077 | 2 |
| 256 | 1.0021 | 4 |
| 1024 | 1.0029 | 8 |
| 4096 | 1.0071 | 1 |
| 16384 | 1.0064 | 4 |
| 39697 | 1.0084 | 1 |

At sixty-four rows the same 8B activations reproduce the development-scale result of
near-zero error. At every larger count the error returns to the trivial baseline.
The low number was never a property of the model or of the method. It was a property
of the sample size, because sixty-four rows drawn from a handful of prompts are
close enough to one another that four codes cover them.

The third diagnostic explains why the codes never help. The bottleneck has four
slots over a vocabulary of sixty-four symbols, so 16,777,216 codes are available.
The trained bottleneck used between one and eight of them at every row count. It had
collapsed. Sweeping the entropy bonus that opposes collapse, across 0.01, 0.1, 0.5,
and 2.0, moved the best relative error only to 1.003, still short of the 0.950 that a
constant vector achieves.

The first explanation I reached for was capacity. A discrete code of four slots over
sixty-four symbols carries at most twenty-four bits, and twenty-four bits look far
too few to preserve a 4096-dimensional residual stream. Part of that correction
stands. The development-scale reconstruction error of 0.006 measured redundancy in a
tiny activation sample rather than faithful compression, and reporting it without the
row-count control would have been a mistake. The capacity explanation itself does not
stand, and the next section is the control that rejects it. Either way the earlier
cross-method finding is strengthened rather than weakened, since the concepts fell in
the neither quadrant at both scales, and the publication-scale run shows the
autoencoder side of that comparison was never carrying information to begin with.

## The bit budget is not what binds

Testing a capacity claim needs a frontier rather than a single point, so
`scripts/bit_budget_study.py` sweeps the code shape and compares it against a
continuous code at a matched budget. It runs on real Qwen3-4B-Instruct-2507 layer-18
activations, 9,877 rows over 64 calibration prompts, on this laptop with no
accelerator. Working at 4B is deliberate, since the 0.006 claim was made at 4B and
the fairest place to test a claim is the setting that produced it.

The reference comes first. A rank-k principal-component reconstruction is a
continuous code costing thirty-two bits per component, which puts it on the same axis
as the discrete code.

| Retained components | Bits | Relative error |
| --- | --- | --- |
| 0 | 0 | 0.9851 |
| 1 | 32 | 0.0104 |
| 2 | 64 | 0.0068 |
| 8 | 256 | 0.0059 |
| 64 | 2048 | 0.0039 |
| 512 | 16384 | 0.0011 |

One principal direction, thirty-two bits, takes the error from 0.9851 to 0.0104. The
information is not the problem. These activations are close to rank one in energy, so
a twenty-four bit code has enough room in principle to place a row along a single
dominant direction.

The discrete sweep uses the published recipe, entropy weight 0.01 and decoder width
1024.

| Code shape | Bits | Relative error | Distinct codes |
| --- | --- | --- | --- |
| 4 x 64 | 24 | 1.0068 | 1 |
| 8 x 64 | 48 | 1.0068 | 1 |
| 16 x 64 | 96 | 0.4039 | 3 |
| 32 x 64 | 192 | 1.0068 | 1 |
| 64 x 64 | 384 | 1.0067 | 5 |
| 128 x 64 | 768 | 0.0899 | 12 |
| 256 x 64 | 1536 | 0.2692 | 7 |

The curve is not monotone. Ninety-six bits beat both one hundred and ninety-two and
three hundred and eighty-four. A capacity frontier cannot do that, so something other
than capacity is moving these numbers.

Two controls narrow it down. Four different shapes that all cost exactly ninety-six
bits give errors of 0.4039, 0.1372, 0.0913, and 0.1248, so bits are not a sufficient
statistic for the outcome. Holding the budget at one hundred and ninety-two bits and
varying only the decoder width gives 1.0066 at width 256, 1.0068 at width 1024, and
0.0797 at width 4096, so the decoder matters without explaining the pattern either.

The decisive control is the seed. Everything is held fixed except initialization and
batch order, and each configuration is refit twenty times. A fit counts as escaping
collapse when it beats the 0.9851 mean predictor.

| Configuration | Bits | Escaped | Best escaped error | Median | Spread |
| --- | --- | --- | --- | --- | --- |
| 4 x 64, width 256 | 24 | 2 of 20 | 0.1300 | 1.0066 | 0.8771 |
| 16 x 64, width 1024 | 96 | 3 of 20 | 0.0616 | 1.0068 | 0.9503 |
| 32 x 64, width 1024 | 192 | 8 of 20 | 0.0667 | 1.0063 | 0.9403 |
| 32 x 64, width 4096 | 192 | 9 of 20 | 0.0567 | 1.0065 | 0.9510 |

This settles it. The published twenty-four bit shape does reach 0.1300, on two seeds
out of twenty, so twenty-four bits are not the barrier and the counting argument was
wrong. What the bit budget buys is the probability of escaping collapse, which rises
from two in twenty at twenty-four bits to eight in twenty at one hundred and
ninety-two. What it does not buy is accuracy once escaped, which sits between 0.057
and 0.130 across every budget tested. Quadrupling the decoder width adds one escape
out of twenty.

The spread within a single configuration reaches 0.95, which is larger than any
difference between configurations. A bit-budget curve read without this control is a
scatter plot of an unstable optimization presented as a frontier. The 1.008 that the
8B run produced is close to the median outcome, and that run drew one seed.

Two conclusions survive. The straight-through discrete bottleneck used here collapses
on most seeds at every budget tested, so any single-seed reconstruction error from it
is close to uninformative and belongs in a report as a distribution over seeds rather
than as a number. And its best result anywhere in the sweep, 0.0567 at one hundred
and ninety-two bits, stays more than five times worse than one principal component at
thirty-two bits. On these activations the discrete natural-language code buys nothing
over a single linear direction and costs a great deal of stability.

I was wrong twice on the way here. The first reading was that 0.006 showed
compression, and the row-count sweep rejected it. The second reading was that
twenty-four bits were too few, and the seed sweep rejected that too. Both corrections
came from controls that cost minutes on a laptop, which is the argument for running
them before publishing a number rather than after.

The limits are specific. Twenty seeds estimate an escape rate to about one part in
twenty, so the gap between two in twenty and three in twenty is not resolved. The
sweeps use the published six-hundred step schedule, so a longer schedule or a
temperature anneal might raise the escape rate at every budget. Everything in this
section is measured at 4B layer 18 on a CPU. The 8B single-method run above uses a
different eight-slot code and one seed, and its collapse is consistent with the
unstable regime rather than an independent estimate of escape probability.

The completed evidence supports two narrow conclusions. Selected Qwen3-8B J-lens
directions produce one small matched-prompt trajectory contrast and one small
direct edit-readback sanity check, while the text-score protocols saturate and
remain uninterpretable. This does not establish a downstream causal effect or recover
the broader workspace result. The configured fixed-symbol autoencoder collapses and
provides no faithful description. Both are useful boundaries for work that extends
these methods, and neither is a full reproduction of its source paper.

Expected publication locations, not yet live:

- [GitHub repository](https://github.com/kotlarmilos/qwen-j-lens)
- [Hugging Face model](https://huggingface.co/kotlarmilos/qwen-j-lens)
- [Hugging Face dataset](https://huggingface.co/datasets/kotlarmilos/qwen-j-lens-evaluation)
- [Hugging Face Space](https://huggingface.co/spaces/kotlarmilos/qwen-j-lens)

## References

- Anthropic, [Functional workspace and the J-lens](https://transformer-circuits.pub/2026/workspace/index.html)
- Anthropic, [Natural Language Autoencoders](https://transformer-circuits.pub/2026/nla/index.html)
- Anthropic, [Natural language autoencoders research page](https://www.anthropic.com/research/natural-language-autoencoders)
- [Reference natural language autoencoder implementation](https://github.com/kitft/natural_language_autoencoders)
- *Train the Model, Not the Reader*, [arXiv:2607.20379](https://arxiv.org/abs/2607.20379)
