#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
JOB="${JOB:-plan}"
CONFIG="${CONFIG:-configs/smoke.toml}"
LAYER="${LAYER:-24}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
PROMPT_COUNT="${PROMPT_COUNT:-}"
ALLOW_DOWNLOAD="${ALLOW_DOWNLOAD:-1}"
PREPARE_CORPUS="${PREPARE_CORPUS:-1}"
OUTPUT="${OUTPUT:-artifacts/colab/$(basename "$CONFIG" .toml)-layer-${LAYER}-${JOB}-offset-${PROMPT_OFFSET}}"
LENS_VECTORS="${LENS_VECTORS:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$ROOT"

if [[ "$JOB" == "plan" ]]; then
  cat <<'EOF'
Run exactly one model, layer, and phase per accelerator runtime.

Development estimation:
  JOB=estimate CONFIG=configs/smoke.toml LAYER=24 \
    OUTPUT=artifacts/colab/qwen3-4b-layer-24-estimate \
    ./scripts/colab_publication.sh

Development evaluation with the downloaded estimation artifact:
  JOB=evaluate CONFIG=configs/smoke.toml LAYER=24 \
    LENS_VECTORS=artifacts/colab/qwen3-4b-layer-24-estimate/controlled-token-vectors-layer-024.npy \
    OUTPUT=artifacts/colab/qwen3-4b-layer-24-evaluate \
    ./scripts/colab_publication.sh

Primary estimation, one 100-context shard per runtime:
  JOB=estimate CONFIG=configs/qwen-publication.toml LAYER=24 \
    PROMPT_OFFSET=0 PROMPT_COUNT=100 \
    OUTPUT=artifacts/colab/qwen3-8b-layer-24-shard-0000 \
    ./scripts/colab_publication.sh

Repeat with offsets 100 through 900. After downloading all ten directories,
merge them locally:
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

Primary evaluation with the merged vectors:
  JOB=evaluate CONFIG=configs/qwen-publication.toml LAYER=24 \
    LENS_VECTORS=artifacts/colab/qwen3-8b-layer-24-merged/controlled-token-vectors-layer-024.npy \
    OUTPUT=artifacts/colab/qwen3-8b-layer-24-evaluate \
    ./scripts/colab_publication.sh

Set ALLOW_DOWNLOAD=0 only when the pinned model revision is already cached.
Copy OUTPUT to persistent storage or download it before ending the runtime.
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

if [[ "$CONFIG" == "configs/qwen-publication.toml" && "$PREPARE_CORPUS" == "1" ]]; then
  "$PYTHON" scripts/prepare_publication_corpus.py
elif [[ "$PREPARE_CORPUS" != "0" && "$PREPARE_CORPUS" != "1" ]]; then
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
    printf 'JOB=evaluate requires LENS_VECTORS to name a downloaded NPY artifact.\n' >&2
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
