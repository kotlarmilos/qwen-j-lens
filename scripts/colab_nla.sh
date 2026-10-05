#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
JOB="${JOB:-plan}"
CONFIG="${CONFIG:-configs/nla-publication.toml}"
ALLOW_DOWNLOAD="${ALLOW_DOWNLOAD:-1}"
OUTPUT="${OUTPUT:-artifacts/colab/nla-qwen3-8b}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$ROOT"

if [[ "$JOB" == "plan" ]]; then
  cat <<'EOF'
Run the fixed-symbol Qwen3-8B autoencoder job.

  JOB=run OUTPUT=artifacts/colab/nla-qwen3-8b ./scripts/colab_nla.sh

The measured run has no labeled specific targets. Download the output directory
and review faithfulness.json with the centered constant-mean baseline.
EOF
  exit 0
fi

if [[ "$JOB" != "run" ]]; then
  printf 'Unknown JOB=%s. Expected plan or run.\n' "$JOB" >&2
  exit 2
fi

ARGS=(--config "$CONFIG" --output "$OUTPUT")
if [[ "$ALLOW_DOWNLOAD" == "1" ]]; then
  ARGS+=(--allow-download)
elif [[ "$ALLOW_DOWNLOAD" != "0" ]]; then
  printf 'ALLOW_DOWNLOAD must be 0 or 1.\n' >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES
"$PYTHON" -m qwen_j_lens.nla_cli validate-nla-config --config "$CONFIG"
"$PYTHON" -m qwen_j_lens.nla_cli nla-run "${ARGS[@]}"
