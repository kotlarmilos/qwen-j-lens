# Building up the J-lens and the Natural Language Autoencoder from scratch

This note derives the two interpretability methods in this repository from first
principles and ties each step to the code. The first method reads an activation in
the model's own output coordinates. The second writes a description of the
activation and checks whether the description carries it back. Both reduce to a
few equations you can rederive by hand.

## The object both methods study

A transformer moves information forward through a residual stream. Write `h_l` for
the activation at layer `l` and a token position. Everything below is a way to
answer one question, which is what `h_l` encodes about what the model will do next.

The final logits come from the last residual state through a normalization and an
unembedding. Write `norm` for the final RMSNorm and `W_U` for the unembedding
matrix. If we somehow had a residual vector `r`, the model turns it into token
scores by `W_U norm(r)`. The methods differ in how they connect `h_l` at an
interior layer to that final read-out.

## The logit lens, the honest baseline

The simplest guess applies the final read-out directly to the interior
activation.

```
logit_lens(h_l) = softmax( W_U norm(h_l) )
```

This works better than it has any right to because residual connections keep the
interior coordinate system loosely aligned with the output one, especially near
the last layers. It also fails in a predictable way. A middle layer can hold a
representation that later blocks rotate or gate before the unembedding ever sees
it, and the logit lens assumes that downstream transformation away. `logit_lens.py`
implements exactly this baseline, and it is the control that every J-lens number
must be compared against.

## The J-lens, a linearized downstream map

The J-lens keeps the downstream transformation instead of discarding it. Define
the function that the rest of the network applies from layer `l` to the final
pre-normalization residual.

```
f(h_l) = h_final_pre_norm
```

A small change to `h_l` produces a change in the final residual, and to first
order that change is linear.

```
delta h_final_pre_norm  ~=  J_l * delta h_l,    J_l = d h_final_pre_norm / d h_l
```

`J_l` is the Jacobian of the downstream map evaluated at the activation. Because a
single prompt entangles the general disposition to express a concept with the
particular routing of that prompt, the method averages the Jacobian over source
positions `t`, over future target positions `t' >= t`, and over a corpus, with
equal weight on every valid pair.

```
J_l = E[ d h_final_pre_norm[t'] / d h_l[t] ]   for t' >= t
```

The read-out then applies the model's own final normalization and unembedding,
and only after the Jacobian has acted.

```
lens(h_l) = softmax( W_U norm( J_l h_l ) )
```

The token-indexed directions of the lens are the rows of `W_U J_l`. The
normalization sits outside those directions, which is a detail the code guards
carefully.

### Why the code differentiates instead of building the matrix

The full Jacobian `J_l` is a hidden-width by hidden-width matrix. On a small model
you can form it, and `estimate_residual_jacobian` in `jacobian.py` does exactly
that for tests. On a multi-billion parameter model it is wasteful when an
experiment controls only a handful of token directions.

The trick is that a single row of `W_U J_l` is a gradient. Let `w_tau` be the
unembedding row for one controlled token `tau`. The score the lens assigns to that
token, before normalization, is built from `w_tau . (J_l h_l)`, and the row we
actually need is `w_tau^T J_l`. Now use the definition of `J_l`.

```
w_tau^T J_l  =  w_tau^T ( d h_final_pre_norm / d h_l )
             =  d ( w_tau . h_final_pre_norm ) / d h_l
```

The right-hand side is the gradient of one scalar, the score `w_tau .
h_final_pre_norm`, with respect to the layer activation. That is a
vector-Jacobian product, and it returns a full row without ever forming the square
matrix. The cost scales with the number of controlled tokens, not with the hidden
width. `estimate_token_vjps` in `jacobian.py` computes these rows, averaging every
valid source and target pair and then averaging across calibration prompts.

### The normalization-order trap

The averaged Jacobian is taken of the pre-normalization residual, and the final
normalization is applied afterward in the read-out. If you instead differentiated
the post-normalization logits, the Jacobian of the final norm would fold into the
result and rescale it. The test adapter `ScalarCausalAdapter` in `tests/toy.py`
has a deliberate five-times final normalization, and the averaging test checks
that the estimated values are not multiplied by five, which is what pins the
normalization to the correct side of the derivative.

### What thirteen directions can and cannot claim

The Qwen configurations declare thirteen controlled tokens. That is enough for
controlled-subset projection, ablation, steering, and coordinate swaps, and it is
not enough to claim vocabulary-wide occupancy or a top-k decomposition of the
sparse frame the source paper calls J-space. The code and the artifact card name
the subset explicitly so the smaller claim is not mistaken for the larger one.

## The Natural Language Autoencoder, a different question

The autoencoder does not read `h_l` in output coordinates. It writes a description
and tests whether the description is enough to rebuild the activation. Two learned
maps make this precise. A verbalizer sends the activation to a code, and a
reconstructor sends the code back to an activation.

```
h_l  --V-->  code c  --R-->  h_hat_l
```

The training signal is reconstruction error. If the code omits what the activation
encoded, no reconstructor can recover `h_l` from it, so a small reconstruction
error is evidence that the description retained the content. `reconstruction_error`
in `nla/autoencoder.py` returns the squared error normalized by the centered energy
of the target, so a value near zero means the code preserves the activation and a
value near one means it carries no more than the mean activation would.

### Making the bottleneck discrete and still trainable

A natural-language code is a sequence of discrete symbols, and discrete choices
have no useful gradient. `straight_through_code` in `nla/autoencoder.py` resolves
this with a straight-through estimator. The forward pass emits a hard one-hot
selection so the bottleneck is genuinely discrete, and the backward pass flows
through the softmax so the verbalizer still learns.

```
soft = softmax(logits / temperature)
hard = onehot(argmax(soft))
code = hard + soft - soft.detach()
```

The value used downstream equals `hard`, because `soft - soft.detach()` is zero in
the forward pass, while the gradient equals the gradient of `soft`, because
`hard` and `soft.detach()` are constants to autograd. In the offline machinery the
symbols stand in for the words a language-model verbalizer would emit, which keeps
the loop deterministic and cheap. On an accelerator the verbalizer becomes a text
decoder and the symbols become real words, and the structure is unchanged.

### Keeping the code from collapsing

A discrete code can collapse onto a single symbol and carry nothing. `train.py`
adds a code-usage entropy bonus computed by `_code_entropy`, which takes the
batch-marginal distribution over symbols per slot and rewards spreading mass
across the vocabulary. The objective is reconstruction error minus a weighted
entropy, so the training minimizes reconstruction while the entropy term keeps the
code alive.

```
loss = relative_reconstruction_error - entropy_weight * mean_slot_entropy
```

## Why reconstruction is necessary and not sufficient

A low reconstruction error is easy to over-read. The critique paper, *Train the
Model, Not the Reader*, shows two ways a code can reconstruct well and still lie to
a human reader. The repository implements an audit for each, in `nla/faithfulness.py`.

### The independent-reader gap

The first failure is a co-adapted private code. The verbalizer and reconstructor
can agree on a scheme that reconstructs perfectly and means nothing to anyone who
did not train alongside them, because the real signal rides in a private
correlation rather than in the apparent words.

The audit turns this into a generalization test on the code-to-activation map.
`independent_reader_gap` trains two reconstructors. The co-adapted reader trains on
every code, including the held-out ones, so it can memorize an example-specific
scheme. The independent reader trains only on a training split and is scored on
codes it never saw.

```
gap = independent_test_error - co_adapted_test_error
```

If the code is a generalizable description, the independent reader reconstructs the
held-out activations about as well as the co-adapted one and the gap is near zero.
If the code is example-specific, the independent reader has never seen the
held-out symbols, its reconstructor rows for those symbols stay at their
initialization, and the gap grows. The gap, not the raw reconstruction error, is
the faithfulness signal. The tests exercise both ends, a memorized code where the
gap must open and a clustered code where it must stay closed.

### The gist-versus-specific probe

The second failure is a gist code. The description captures coarse structure and
drops the specific fact the activation carried. `gist_versus_specific` fits a
least-squares linear probe for a labeled attribute, first on the original
activations and then on the reconstructions, and compares the coefficient of
determination on a held-out split.

```
retained = reconstructed_specific_r2 / original_specific_r2
```

If the reconstruction keeps the specific attribute, both probes reach a similar
score and `retained` is near one. If it keeps only the gist, the probe on the
reconstructions falls while the probe on the originals stays high, and `retained`
drops. `_linear_probe_r2` builds the probe with a bias column and `torch.linalg.lstsq`
so the measurement is deterministic. A third measurement, `code_usage`, reports
how many distinct codes appear per example, because a code that assigns a distinct
symbol string to every example is memorizing rather than describing, which is the
same signature the gap is built to catch.

`FaithfulnessReport` collects these numbers, compares them against a gap threshold
and a retained threshold, and turns them into a private-code flag, a gist-only
flag, and an overall faithful flag.

## Reading the two methods against each other

Both methods take a residual activation and make it legible, and both replace a
plausible story with a check. They differ in coordinates and in the direction of
the test. The J-lens reads the activation in the model's own output vocabulary and
validates the reading with paired causal interventions on the running model, and
it is linear and cheap but can only speak in single-token directions. The
autoencoder writes a free-text description and validates it by reconstruction and
by the two audits, and it can express a richer description but introduces two
trained networks whose agreement must itself be audited. A concept that appears as
a clean J-lens direction and also survives the reconstruction round trip is a
strong result, and a concept that appears in one method and not the other is the
more instructive case, which is why the autoencoder configuration reads the same
preregistered layer 24 that the J-lens evaluation uses.

### Measuring cross-method agreement at a matched layer

The instructive case only becomes a measurement once both methods read the same
physical activation, and getting that alignment right takes one careful index. In
`jacobian.py` the call `estimate_token_vjps(layer_index=L)` hooks `adapter.layers[L]`,
whose output is the residual stream the model exposes as `hidden_states[L+1]`. In
`nla/activations.py` the call `collect_layer_activations(layer_index=m)` reads
`hidden_states[m]` directly, where index 0 is the embedding output. To score both
methods on the residual stream at `hidden_states[m]`, the runner in `cross_method.py`
captures the J-lens directions at block `m - 1` and reads the autoencoder
activations at `m`. The configuration sets `hidden_state_index = m` and the code
derives `jlens_capture_layer = m - 1`, so the two studies never drift onto adjacent
layers.

Both scores predict the same ground truth, the model's own next-token logit for a
controlled token at every attended position, which the runner reads from
`run.logits` at the matched forward pass. The J-lens fidelity for a concept is the
held-out coefficient of determination of that logit from the single scalar
`s_t = <v_t, h>`, where `v_t` is the stored direction and `h` is the activation. The
autoencoder retention is `specific_retained_fraction` from `gist_versus_specific`,
the fraction of that logit's linear recoverability that survives the reconstruction.
A concept scores high on the first axis when it lives as a clean linear direction in
output coordinates, and high on the second axis when the discrete description keeps
the specific fact. Sorting the thirteen concepts into `both_recoverable`,
`jlens_only`, `nla_only`, and `neither` at a shared threshold turns the qualitative
taxonomy into four counts.

The real 4B run returns a clean negative result, and reading it teaches the point
better than a success would. The autoencoder drives reconstruction error to 0.006
yet almost no concept keeps its next-token logit through the round trip, so every
concept lands in `neither`. This is the gist-code failure the audit was built to
name, now visible on real weights across the whole bank rather than on a constructed
example. The single J-lens direction, read without the final normalization that the
faithful readout would apply, is a weak predictor at development scale. The honest
caveat is that the two axes are not symmetric, since the J-lens fidelity uses one
direction and the autoencoder retention uses a multivariate probe, so their absolute
values are not comparable and only their ordering across concepts is. Even so, the
comparison shows that a low reconstruction error and a clean linear direction can
each miss the specific attribute a residual carries, which is a claim neither method
can make on its own.

Everything here is tested against the tiny local transformer in `tests/toy.py`.
The open-weight Qwen runs for both methods remain pending accelerator jobs, and no
reproduction number is claimed until those artifacts exist and are reviewed.
