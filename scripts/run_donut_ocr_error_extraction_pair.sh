#!/usr/bin/env bash
set -euo pipefail

# Build or run matching Donut OCR error extraction jobs for train and val.
#
# Required env vars:
#   CKPT, RUN, TRAIN_MANIFEST, VAL_MANIFEST
#
# Common optional env vars:
#   PB_ROOT=/home/nico/Code/PechaBridge
#   DATASET_NAME=openpecha_ocr_lines
#   CER_THRESHOLD=-1
#   DATASET_IMAGE_MODE=reference  # copy | reference | symlink
#   LOG_STAGE_TIMINGS=1  # log per-batch load/generate/decode/write timings
#   RANDOM_SAMPLE=2560  # random sample this many rows from each manifest into temp files
#   RANDOM_SEED=42
#   CUDA_VISIBLE_DEVICES=0
#   RUN_EXTRACT=1  # execute commands; otherwise only print them

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required env var: ${name}" >&2
    exit 1
  fi
}

trim_value() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

slugify() {
  local value="$1"
  value="${value//[^[:alnum:]_.-]/_}"
  value="${value##_}"
  value="${value%%_}"
  [[ -n "$value" ]] || value="unknown"
  printf '%s' "$value"
}

threshold_slug() {
  local threshold="$1"
  if [[ "$threshold" == -* ]]; then
    printf 'all'
    return
  fi
  local normalized="${threshold//./p}"
  printf 'gt%s' "$normalized"
}

make_random_manifest() {
  local split="$1"
  local src="$2"
  local n="$3"
  local seed="$4"
  local dst="$5"

  python - "$src" "$dst" "$n" "$seed" "$split" <<'PY'
import random
import sys
from pathlib import Path

src = Path(sys.argv[1].strip()).expanduser()
dst = Path(sys.argv[2].strip()).expanduser()
n = int(sys.argv[3])
seed = int(sys.argv[4])
split = str(sys.argv[5])

rows = []
with src.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(line)

rng = random.Random(seed)
if n > 0 and n < len(rows):
    selected = rng.sample(rows, n)
else:
    selected = list(rows)

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as out:
    out.writelines(selected)

print(f"random_manifest split={split} source={src} rows_total={len(rows)} rows_written={len(selected)} seed={seed} output={dst}", file=sys.stderr)
PY

  printf '%s' "$dst"
}

run_one() {
  local split="$1"
  local manifest="$2"
  local output_dir="$3"

  local cmd=(
    python cli.py extract-donut-ocr-errors
    --checkpoint "$CKPT"
    --manifest "$manifest"
    --output_dataset_dir "$output_dir"
    --cer_threshold "$CER_THRESHOLD"
    --tokenizer_path "$RUN/tokenizer"
    --image_processor_path "$RUN/image_processor"
    --image_preprocess_pipeline "$IMAGE_PREPROCESS_PIPELINE"
    --enable_fixed_resize
    --target_height "$TARGET_HEIGHT"
    --target_width "$TARGET_WIDTH"
    --max_target_length "$MAX_TARGET_LENGTH"
    --generation_max_length "$GENERATION_MAX_LENGTH"
    --batch_size "$BATCH_SIZE"
    --device "$DEVICE"
    --num_workers "$NUM_WORKERS"
    --dataset_image_mode "$DATASET_IMAGE_MODE"
  )

  if [[ "${INCLUDE_GOOGLE_BOOKS}" == "1" ]]; then
    cmd+=(--include_google_books)
  fi
  if [[ -n "${SOURCE_DATASETS}" ]]; then
    cmd+=(--source_datasets "$SOURCE_DATASETS")
  fi
  if [[ -n "${EXCLUDE_SOURCE_DATASETS}" ]]; then
    cmd+=(--exclude_source_datasets "$EXCLUDE_SOURCE_DATASETS")
  fi
  if [[ "${MAX_SAMPLES}" != "0" ]]; then
    cmd+=(--max_samples "$MAX_SAMPLES")
  fi
  if [[ "${LIMIT_ERRORS}" != "0" ]]; then
    cmd+=(--limit_errors "$LIMIT_ERRORS")
  fi
  if [[ "${LOG_STAGE_TIMINGS}" == "1" ]]; then
    cmd+=(--log_stage_timings --stage_timing_every_n "$STAGE_TIMING_EVERY_N" --stage_timing_warmup_batches "$STAGE_TIMING_WARMUP_BATCHES")
  fi
  if [[ "$split" == "val" ]]; then
    cmd+=(--val_fraction 0)
  elif [[ "${VAL_FRACTION}" != "0" ]]; then
    cmd+=(--val_fraction "$VAL_FRACTION")
  fi

  printf '\n# %s extraction -> %s\n' "$split" "$output_dir"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
  printf '%q ' "${cmd[@]}"
  printf '\n'

  if [[ "${RUN_EXTRACT}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "${cmd[@]}"
  fi
}

require_env CKPT
require_env RUN
require_env TRAIN_MANIFEST
require_env VAL_MANIFEST

CKPT="$(trim_value "$CKPT")"
RUN="$(trim_value "$RUN")"
TRAIN_MANIFEST="$(trim_value "$TRAIN_MANIFEST")"
VAL_MANIFEST="$(trim_value "$VAL_MANIFEST")"

PB_ROOT="${PB_ROOT:-/home/nico/Code/PechaBridge}"
DATASET_NAME="$(slugify "${DATASET_NAME:-openpecha_ocr_lines}")"
CHECKPOINT_NAME="$(slugify "$(basename "$CKPT")")"
CER_THRESHOLD="${CER_THRESHOLD:--1}"
THRESHOLD_SLUG="$(threshold_slug "$CER_THRESHOLD")"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PB_ROOT/datasets}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

IMAGE_PREPROCESS_PIPELINE="${IMAGE_PREPROCESS_PIPELINE:-gray}"
TARGET_HEIGHT="${TARGET_HEIGHT:-256}"
TARGET_WIDTH="${TARGET_WIDTH:-1024}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-160}"
GENERATION_MAX_LENGTH="${GENERATION_MAX_LENGTH:-160}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-cuda:0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DATASET_IMAGE_MODE="${DATASET_IMAGE_MODE:-reference}"
INCLUDE_GOOGLE_BOOKS="${INCLUDE_GOOGLE_BOOKS:-1}"
SOURCE_DATASETS="${SOURCE_DATASETS:-}"
EXCLUDE_SOURCE_DATASETS="${EXCLUDE_SOURCE_DATASETS:-}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
LIMIT_ERRORS="${LIMIT_ERRORS:-0}"
VAL_FRACTION="${VAL_FRACTION:-0}"
LOG_STAGE_TIMINGS="${LOG_STAGE_TIMINGS:-0}"
STAGE_TIMING_EVERY_N="${STAGE_TIMING_EVERY_N:-1}"
STAGE_TIMING_WARMUP_BATCHES="${STAGE_TIMING_WARMUP_BATCHES:-0}"
RANDOM_SAMPLE="${RANDOM_SAMPLE:-0}"
RANDOM_SEED="${RANDOM_SEED:-42}"
RUN_EXTRACT="${RUN_EXTRACT:-0}"

SAMPLE_SLUG=""
if [[ "$RANDOM_SAMPLE" != "0" ]]; then
  SAMPLE_SLUG="_sample${RANDOM_SAMPLE}_seed${RANDOM_SEED}"
fi

TRAIN_OUT="$OUTPUT_ROOT/donut_error_extract_${DATASET_NAME}_train_${CHECKPOINT_NAME}_${THRESHOLD_SLUG}${SAMPLE_SLUG}"
VAL_OUT="$OUTPUT_ROOT/donut_error_extract_${DATASET_NAME}_val_${CHECKPOINT_NAME}_${THRESHOLD_SLUG}${SAMPLE_SLUG}"

cleanup_tmp_manifests() {
  if [[ "$RANDOM_SAMPLE" != "0" ]]; then
    [[ "$EFFECTIVE_TRAIN_MANIFEST" != "$TRAIN_MANIFEST" ]] && rm -f "$EFFECTIVE_TRAIN_MANIFEST"
    [[ "$EFFECTIVE_VAL_MANIFEST" != "$VAL_MANIFEST" ]] && rm -f "$EFFECTIVE_VAL_MANIFEST"
  fi
}
trap cleanup_tmp_manifests EXIT

EFFECTIVE_TRAIN_MANIFEST="$TRAIN_MANIFEST"
EFFECTIVE_VAL_MANIFEST="$VAL_MANIFEST"
if [[ "$RANDOM_SAMPLE" != "0" ]]; then
  EFFECTIVE_TRAIN_MANIFEST="$(mktemp "$(dirname "$TRAIN_MANIFEST")/.donut_error_extract_train_random_${RANDOM_SAMPLE}_seed_${RANDOM_SEED}.jsonl.XXXXXX")"
  EFFECTIVE_VAL_MANIFEST="$(mktemp "$(dirname "$VAL_MANIFEST")/.donut_error_extract_val_random_${RANDOM_SAMPLE}_seed_${RANDOM_SEED}.jsonl.XXXXXX")"
  make_random_manifest train "$TRAIN_MANIFEST" "$RANDOM_SAMPLE" "$RANDOM_SEED" "$EFFECTIVE_TRAIN_MANIFEST" >/dev/null
  make_random_manifest val "$VAL_MANIFEST" "$RANDOM_SAMPLE" "$RANDOM_SEED" "$EFFECTIVE_VAL_MANIFEST" >/dev/null
fi

run_one train "$EFFECTIVE_TRAIN_MANIFEST" "$TRAIN_OUT"
run_one val "$EFFECTIVE_VAL_MANIFEST" "$VAL_OUT"
