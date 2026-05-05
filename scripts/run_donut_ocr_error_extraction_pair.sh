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
#   CUDA_VISIBLE_DEVICES=0
#   RUN_EXTRACT=1  # execute commands; otherwise only print them

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required env var: ${name}" >&2
    exit 1
  fi
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
RUN_EXTRACT="${RUN_EXTRACT:-0}"

TRAIN_OUT="$OUTPUT_ROOT/donut_error_extract_${DATASET_NAME}_train_${CHECKPOINT_NAME}_${THRESHOLD_SLUG}"
VAL_OUT="$OUTPUT_ROOT/donut_error_extract_${DATASET_NAME}_val_${CHECKPOINT_NAME}_${THRESHOLD_SLUG}"

run_one train "$TRAIN_MANIFEST" "$TRAIN_OUT"
run_one val "$VAL_MANIFEST" "$VAL_OUT"
