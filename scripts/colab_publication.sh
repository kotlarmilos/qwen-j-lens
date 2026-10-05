#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
JOB="${JOB:-plan}"
CONFIG="${CONFIG:-configs/qwen-publication.toml}"
LAYER="${LAYER:-24}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
PROMPT_COUNT="${PROMPT_COUNT:-}"
ALLOW_DOWNLOAD="${ALLOW_DOWNLOAD:-1}"
PREPARE_CORPUS="${PREPARE_CORPUS:-1}"
OUTPUT="${OUTPUT:-artifacts/colab/qwen3-8b-layer-${LAYER}-${JOB}-offset-${PROMPT_OFFSET}}"
LENS_VECTORS="${LENS_VECTORS:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$ROOT"

if [[ "$JOB" == "plan" ]]; then
  cat <<'EOF'
Run one 100-context estimation shard per accelerator runtime.

  JOB=estimate PROMPT_OFFSET=0 PROMPT_COUNT=100 \
    OUTPUT=artifacts/colab/qwen3-8b-layer-24-shard-0000 \
    ./scripts/colab_publication.sh

Repeat with offsets 100 through 900. Merge the ten downloaded directories.

  python -m qwen_j_lens.cli merge-estimates \
    --config configs/qwen-publication.toml --layer 24 \
    --input artifacts/colab/qwen3-8b-layer-24-shard-0000 \
    --input artifacts/colab/qwen3-8b-layer-24-shard-0100 \
    --input artifacts/colab/qwen3-8b-layer-24-shard-0200 \
    --input artifacts/colab/qwen3-8b-layer-24-shard-0300 \
    --input artifacts/colab/qwen3-8b-layer-24-shard-0400 \
    --input artifacts/colab/qwen3-8b-layer-24-shard-0500 \
    --input artifacts/colab/qwen3-8b-layer-24-shard-0600 \
    --input artifacts/colab/qwen3-8b-layer-24-shard-0700 \
    --input artifacts/colab/qwen3-8b-layer-24-shard-0800 \
    --input artifacts/colab/qwen3-8b-layer-24-shard-0900 \
    --output artifacts/colab/qwen3-8b-layer-24-merged

Run evaluation with the merged direction array.

  JOB=evaluate \
    LENS_VECTORS=artifacts/colab/qwen3-8b-layer-24-merged/controlled-token-vectors-layer-024.npy \
    OUTPUT=artifacts/colab/qwen3-8b-layer-24-evaluate \
    ./scripts/colab_publication.sh

Copy each output directory to persistent storage before ending the runtime.
EOF
  exit 0
fi

if [[ "$JOB" != "estimate" && "$JOB" != "evaluate" ]]; then
  printf 'Unknown JOB=%s. Expected plan, estimate, or evaluate.\n' "$JOB" >&2
  exit 2
fi

"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -e '.[qwen,test,data]'
"$PYTHON" -m qwen_j_lens.cli validate-config --config "$CONFIG"

DOWNLOAD_FLAG=()
if [[ "$ALLOW_DOWNLOAD" == "1" ]]; then
  DOWNLOAD_FLAG+=(--allow-download)
elif [[ "$ALLOW_DOWNLOAD" != "0" ]]; then
  printf 'ALLOW_DOWNLOAD must be 0 or 1.\n' >&2
  exit 2
fi

"$PYTHON" -m qwen_j_lens.cli validate-tokens \
  --config "$CONFIG" \
  "${DOWNLOAD_FLAG[@]}"

if [[ "$PREPARE_CORPUS" == "1" ]]; then
  "$PYTHON" scripts/prepare_publication_corpus.py
elif [[ "$PREPARE_CORPUS" != "0" ]]; then
  printf 'PREPARE_CORPUS must be 0 or 1.\n' >&2
  exit 2
fi

if [[ "$JOB" == "estimate" ]]; then
  PROMPT_ARGS=(--prompt-offset "$PROMPT_OFFSET")
  if [[ -n "$PROMPT_COUNT" ]]; then
    PROMPT_ARGS+=(--max-prompts "$PROMPT_COUNT")
  fi
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON" -m qwen_j_lens.cli estimate \
    --config "$CONFIG" \
    --layer "$LAYER" \
    "${PROMPT_ARGS[@]}" \
    --output "$OUTPUT" \
    "${DOWNLOAD_FLAG[@]}"
else
  if [[ -z "$LENS_VECTORS" || ! -f "$LENS_VECTORS" ]]; then
    printf 'JOB=evaluate requires an existing LENS_VECTORS file.\n' >&2
    exit 2
  fi
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON" -m qwen_j_lens.cli evaluate \
    --config "$CONFIG" \
    --layer "$LAYER" \
    --lens-vectors "$LENS_VECTORS" \
    --output "$OUTPUT" \
    "${DOWNLOAD_FLAG[@]}"
fi

printf 'Completed %s. Persist this directory before ending the runtime: %s\n' \
  "$JOB" "$OUTPUT"
