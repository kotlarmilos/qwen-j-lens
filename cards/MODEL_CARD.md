---
library_name: transformers
tags:
  - interpretability
  - jacobian-lens
  - discrete-autoencoder
  - qwen
---

# Qwen3-8B interpretability artifacts

This artifact repository contains one measured Qwen3-8B study of 13
selected-token J-lens directions and a fixed-symbol autoencoder. It does not
contain model weights or a new language model.

## Main result

The run uses `Qwen/Qwen3-8B` at revision
`b968826d9c46dd6066d109eabc6255188de91218` on an NVIDIA A100 80 GB PCIe.

The J-lens estimate covers 1,000 contexts and 15,620,093 valid position pairs.
Its evaluation supports only a small unedited matched-prompt contrast and a
same-layer edit readback. It does not establish a reasoning effect.

The fixed-symbol autoencoder uses two code strings over 159,325 activation
rows. Its centered relative reconstruction error is `1.00515`, which does not
beat the constant-mean predictor at `1.0`. Both readers fail at about `0.999`.
Their near-zero gap is not faithfulness evidence.

No labeled specific-target probe was run. The retained primary artifacts do not
support a zero-predictor comparison.

## Layer indexing

The primary J-lens hooks block index 24 and reads the state exposed as
`hidden_states[25]`. The autoencoder reads `hidden_states[24]`. The primary
runs use adjacent residual states and are not a matched-layer comparison.

The separate cross-method runner aligns a hidden-state index `m` with block
index `m - 1`.

## Files

- `artifacts/colab/qwen3-8b-layer-24-merged`
- `artifacts/colab/qwen3-8b-layer-24-evaluate`
- `artifacts/colab/nla-qwen3-8b`
- `artifacts/qwen3-4b-bit-budget-layer18`
- `artifacts/manifest.json`
- `configs/qwen-publication.toml`
- `configs/nla-publication.toml`

Every retained record file contains SHA-256 values for its referenced arrays
and JSON files. Run the local verifier with this command.

```bash
qwen-j-lens verify-artifacts --root artifacts
```

## Limits

The token bank omits almost the entire vocabulary. The trajectory scores are
raw cosine measurements rather than calibrated lens probabilities. Directed
modulation compares unedited prompts. The internal reasoning protocol reads a
coordinate changed at the same layer. The text scorers saturate.

The autoencoder emits fixed IDs rather than natural language. Its result applies
to this surrogate and training recipe. The smaller Qwen3-4B seed diagnostic
shows fitting instability and does not establish a capacity limit.

The complete explanation is in `article.md`.

The implementation is available at
`https://github.com/kotlarmilos/qwen-j-lens`.
