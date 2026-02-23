# DONUT / TroCR OCR Training (OpenPecha / BDRC Line Manifests)

## Purpose

This training path fine-tunes a `VisionEncoderDecoderModel` (e.g. `microsoft/trocr-base-stage1`) for Tibetan OCR on **line image + text** pairs.

It is intended for:

- OCR model training on OpenPecha/BDRC-style line manifests
- CER-based validation during training
- producing a reusable OCR model (`model/`, `tokenizer/`, `image_processor/`) for offline inference or future UI integration

## Input Format

The trainer accepts JSONL manifests (one line per sample) and supports multiple field aliases.

Common manifest fields:

- image path: `line_path`, `src__image`, `image`, `image_path`, ...
- text: `text`, `src__label`, `label`, `transcription`, ...

Relative image paths are resolved relative to the manifest location.

## CLI Entry Point

Use either:

- `python cli.py train-donut-ocr ...` (recommended)
- or `python scripts/train_donut_ocr.py ...`

## Example (OpenPecha OCR Lines)

```bash
python cli.py train-donut-ocr \
  --train_manifest /home/ubuntu/data/PechaBridge/datasets/openpecha_ocr_lines/train/meta/lines.jsonl \
  --val_manifest /home/ubuntu/data/PechaBridge/datasets/openpecha_ocr_lines/eval/meta/lines.jsonl \
  --output_dir /home/ubuntu/data/PechaBridge/models/donut_openpecha_bdrc \
  --model_name_or_path microsoft/trocr-base-stage1 \
  --tokenizer_path openpecha/BoSentencePiece \
  --image_preprocess_pipeline bdrc \
  --per_device_train_batch_size 64 \
  --num_train_epochs 200 \
  --eval_steps 3000 \
  --save_steps 3000 \
  --save_total_limit 20
```

## Image Preprocessing Pipelines

`--image_preprocess_pipeline` supports:

- `none`: HF image processor only
- `pb`: PechaBridge preprocessing
- `bdrc`: BDRC-like preprocessing (binarization/normalization helpers)

Use `bdrc` when training on historical Tibetan line scans and you want behavior closer to BDRC OCR pipelines.

## Metrics

### `loss`

- Token-level training loss (cross-entropy style)
- useful as optimization signal
- lower is better

### `eval_loss`

- Validation loss on the eval manifest
- helps detect overfitting

### `eval_cer`

- Character Error Rate (Levenshtein distance / reference length)
- primary OCR quality metric in this trainer
- lower is better (`0.0` perfect)

## Checkpoints

The trainer saves **step-based checkpoints**:

- `checkpoint-<step>`

Additionally, it creates human-readable checkpoint aliases (symlinks) after saves:

- `checkpoint-epoch-<N>-cer-<X>` -> `checkpoint-<step>`

This requires that an evaluation happened before the save (so CER is known).

## Output Artifacts

Inside `--output_dir`:

- `checkpoint-*` (HF checkpoints)
- `checkpoint-epoch-*-cer-*` (aliases/symlinks)
- `model/` (final model)
- `tokenizer/`
- `image_processor/`
- `train_summary.json`
- `device_report_pretrain.json` (debug report written before training starts)

## Operational Notes

- The trainer is robust to `transformers` 5.x API differences (`evaluation_strategy`/`eval_strategy`, `tokenizer`/`processing_class`).
- TroCR sinusoidal positional embeddings are patched so helper tensors are placed on the correct device (important for some `transformers` + `torch` combinations).
- TQDM shows live postfix metrics (`loss`, `lr`, `grad_norm`, `epoch`, and eval metrics after evaluation).

## Workbench Status

- Workbench currently supports the **DONUT OCR workflow runner** and monitoring.
- Standalone trained DONUT model inference from `model/` in the UI is not yet exposed as a dedicated tab.
