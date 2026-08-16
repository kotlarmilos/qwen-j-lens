#!/usr/bin/env bash
set -euo pipefail

# Run one natural-language autoencoder job per accelerator runtime. Each job
# collects residual-stream activations from an open-weight Qwen model at one
# layer, trains the verbalizer and reconstructor, and writes the faithfulness
# audit. Results persist under artifacts/colab so they can be downloaded and
# reviewed before any publication claim is made.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
JOB="${JOB:-plan}"
CONFIG="${CONFIG:-configs/nla-smoke.toml}"
SPECIFIC_TARGETS="${SPECIFIC_TARGETS:-}"
ALLOW_DOWNLOAD="${ALLOW_DOWNLOAD:-1}"
OUTPUT="${OUTPUT:-artifacts/colab/$(basename "$CONFIG" .toml)}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$ROOT"

if [[ "$JOB" == "plan" ]]; then
  cat <<'EOF'
Development run on Qwen3-4B:
  JOB=run CONFIG=configs/nla-smoke.toml \
    OUTPUT=artifacts/colab/nla-qwen3-4b \
    ./scripts/colab_nla.sh

Primary run on Qwen3-8B at layer 24:
  JOB=run CONFIG=configs/nla-publication.toml \
    OUTPUT=artifacts/colab/nla-qwen3-8b \
    ./scripts/colab_nla.sh

Optional gist probe. Supply an NPY of per-activation attribute labels aligned to
the collected activation rows:
  JOB=run CONFIG=configs/nla-publication.toml \
    SPECIFIC_TARGETS=data/specific_targets.npy \
    OUTPUT=artifacts/colab/nla-qwen3-8b \
    ./scripts/colab_nla.sh

Download the OUTPUT directory and review faithfulness.json before treating any
reconstruction number as a result.
EOF
  exit 0
fi

if [[ "$JOB" != "run" ]]; then
  echo "Unknown JOB: $JOB" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES

ARGS=(--config "$CONFIG" --output "$OUTPUT")
if [[ "$ALLOW_DOWNLOAD" == "1" ]]; then
  ARGS+=(--allow-download)
fi
if [[ -n "$SPECIFIC_TARGETS" ]]; then
  ARGS+=(--specific-targets "$SPECIFIC_TARGETS")
fi

"$PYTHON" -m qwen_j_lens.nla_cli validate-nla-config --config "$CONFIG"
"$PYTHON" -m qwen_j_lens.nla_cli nla-run "${ARGS[@]}"
