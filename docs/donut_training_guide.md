# DONUT/TroCR OCR Training Guide

Updated: 2026-04-29

This guide documents the current PechaBridge workflow for training the OCR model behind `python cli.py train-donut-ocr`.

It covers:

1. environment setup,
2. tokenizer download,
3. downloading and merging the OpenPecha OCR line datasets,
4. optional manifest preparation,
5. creating a safe tiny dataset,
6. why a tiny run is worth doing first,
7. square vs non-square training,
8. recommended tiny and full training commands,
9. how to select checkpoints and read training logs.

The examples assume a local clone of the repo. Replace paths as needed.

## 1. Why start with a tiny run

A tiny run is not just a convenience shortcut. In this project it is the safest way to de-risk a full OCR training run.

Why it matters:

1. It validates that manifests, image paths, tokenizer files, and labels are all wired correctly.
2. It exposes common failure modes early, before spending hours on a full run.
3. It gives you a warm checkpoint that is already generating non-empty, non-collapsed outputs.
4. It lets you compare preprocessing choices such as `gray` vs `rgb` quickly.
5. It helps you decide whether a square input (`image_size=384`) or a non-square letterboxed input is more promising for your data.

Typical failure modes that the tiny run catches:

1. empty predictions,
2. immediate stop-token behavior,
3. repetitive decoder loops,
4. very short generic outputs that ignore the image,
5. broken relative image paths after hand-copying manifests.

The recommended pattern is:

1. run a tiny experiment,
2. confirm the run is healthy,
3. choose the last or best tiny checkpoint,
4. start the full training run from that checkpoint.

## 2. Environment setup

From the repo root:

```bash
export PB_ROOT=/path/to/PechaBridge
cd "$PB_ROOT"
```

If you want a fresh Python 3.12 virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Convenient path variables used throughout the guide:

```bash
export TOKENIZER=$PB_ROOT/ext/BoSentencePiece
export DATASET_ROOT=$PB_ROOT/datasets/openpecha_ocr_lines
export TINY_DIR=$PB_ROOT/datasets/openpecha_ocr_lines_tiny_abs
export MODEL_DIR=$PB_ROOT/models
```

## 3. Download the tokenizer

`train-donut-ocr` expects a local BoSentencePiece tokenizer directory. The training code intentionally does not fall back to an arbitrary remote tokenizer.

Download it once:

```bash
python cli.py download-bosentencepiece-tokenizer \
  --dest "$TOKENIZER"
```

After that, the training commands in this guide use:

```bash
--tokenizer_path "$TOKENIZER"
```

## 4. Download and merge the OCR line datasets

PechaBridge provides a single command that downloads and merges the OpenPecha OCR line datasets into one canonical dataset layout:

```bash
python cli.py download-openpecha-ocr-lines \
  --output-dir "$DATASET_ROOT"
```

This produces a dataset root with canonical split folders such as:

```text
$DATASET_ROOT/
  train/
    meta/lines.jsonl
  test/
    meta/lines.jsonl
  eval/
    meta/lines.jsonl
```

Important detail:

1. the merged `lines.jsonl` files contain relative `line_path` values,
2. those paths are valid as long as the manifest stays in the expected dataset layout,
3. if you copy the manifest into another folder, the relative image paths may stop resolving.

That is why the tiny-dataset step below writes an explicit absolute `image` field.

## 5. Optional: prepare portable DONUT manifests

You do not have to run this step. `train-donut-ocr` can already train directly from the merged `train/meta/lines.jsonl` and `eval/meta/lines.jsonl` files.

Still, `prepare-donut-ocr-dataset` is useful when you want:

1. absolute `image` paths,
2. normalized OCR text,
3. dedicated `train_manifest.jsonl` and `val_manifest.jsonl` files,
4. a tokenizer corpus file for inspection,
5. more portable manifests that can be moved independently from the raw dataset tree.

Run it like this:

```bash
python cli.py prepare-donut-ocr-dataset \
  --dataset_dir "$DATASET_ROOT" \
  --output_dir "$DATASET_ROOT/donut_manifests" \
  --splits train,val \
  --text_field text
```

This writes manifests such as:

```text
$DATASET_ROOT/donut_manifests/
  train_manifest.jsonl
  val_manifest.jsonl
  tokenizer_corpus.jsonl
  prepare_summary.json
```

Use this step when you want cleaner, more portable inputs. Skip it when you are happy to train directly from the raw merged line manifests.

## 6. Build a safe tiny dataset

The safest tiny-dataset approach is:

1. sample from the merged raw manifests,
2. keep the original metadata,
3. add an absolute `image` field to every sampled row.

Create a deterministic tiny dataset:

```bash
mkdir -p "$TINY_DIR"

python - <<'PY'
import json
import random
from pathlib import Path

seed = 42
train_n = 512
val_n = 128

dataset_root = Path("/path/to/PechaBridge/datasets/openpecha_ocr_lines")
train_src = dataset_root / "train" / "meta" / "lines.jsonl"
val_src = dataset_root / "eval" / "meta" / "lines.jsonl"

out_dir = Path("/path/to/PechaBridge/datasets/openpecha_ocr_lines_tiny_abs")
train_dst = out_dir / "train_manifest.jsonl"
val_dst = out_dir / "val_manifest.jsonl"

def load_rows(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows

def sample_rows(rows, n, seed):
    rng = random.Random(seed)
    if len(rows) <= n:
        idx = list(range(len(rows)))
    else:
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        idx = sorted(idx[:n])
    return [rows[i] for i in idx]

def absolutize_image(row):
    row = dict(row)
    rel = str(row.get("line_path") or row.get("src__image") or row.get("image") or "").strip()
    if rel:
        row["image"] = str((dataset_root / rel).resolve())
    return row

train_rows = [absolutize_image(r) for r in sample_rows(load_rows(train_src), train_n, seed)]
val_rows = [absolutize_image(r) for r in sample_rows(load_rows(val_src), val_n, seed + 1)]

out_dir.mkdir(parents=True, exist_ok=True)
train_dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train_rows), encoding="utf-8")
val_dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in val_rows), encoding="utf-8")

print(train_dst)
print(val_dst)
PY
```

Replace `/path/to/PechaBridge` inside the snippet with your local repo path. For example, if your clone lives in `/home/nico/Code/PechaBridge`, use that path in both `dataset_root` and `out_dir`.

Why this form is preferred:

1. it keeps tiny manifests self-contained,
2. it avoids broken relative paths,
3. the training dataset will prefer the explicit `image` field,
4. you can move the tiny manifests without moving the original dataset tree.

## 7. Preprocessing pipelines

The OCR trainer supports several preprocessing modes:

1. `none`
2. `pb`
3. `gray`
4. `bdrc`
5. `rgb`

The most relevant ones for line OCR training are:

### `gray`

Recommended baseline for many experiments.

Behavior:

1. converts the crop to grayscale using the minimum RGB channel (`min_rgb`),
2. does not apply hard binarization,
3. converts the result back to RGB before the image processor sees it.

This often works well when color is not essential but you still want a gentle, stable cleanup.

### `rgb`

Useful when color matters, for example for red/black Tibetan prints. It preserves color information rather than reducing the line to grayscale first.

### `bdrc`

A more traditional OCR-style preprocessing path with grayscale plus stronger document cleanup and binarization behavior.

## 8. Train-time augmentations

For `gray`, `rgb`, and `bdrc`, train-time augmentations are enabled on the training split when `albumentations` is available.

That means:

1. the tiny run uses them,
2. the full run uses the same augmentation logic,
3. validation does not use them.

For the `gray` pipeline, the active training augmentations include:

1. `GaussianBlur`
2. `ElasticTransform`
3. `CoarseDropout`
4. `RandomBrightnessContrast`
5. `GaussNoise`
6. a resolution-invariance branch (`Downscale` or `Blur`)
7. mild `ShiftScaleRotate`

This is important when comparing runs: full-dataset training is not "less augmented" than the tiny run unless you explicitly change the pipeline or your environment is missing `albumentations`.

## 9. Square vs non-square inputs

This is one of the most important modeling choices in this training setup.

### Square input: `--image_size 384`

If you use a processor like `ViTImageProcessorFast` and do not enable letterboxing, the input is square-resized to `384 x 384`.

That means:

1. the entire line crop is passed to the model as one image,
2. long line images are compressed to fit the square input,
3. aspect ratio is not preserved in the simple square-resize path.

This can still work surprisingly well if train and eval use the same representation, but it does distort long line geometry.

### Non-square input: `--enable_letterboxing`

If you enable letterboxing and set a fixed target height and width, the crop is fitted into a non-square canvas.

Example:

```bash
--enable_letterboxing \
--target_height 256 \
--target_width 1024
```

Benefits:

1. preserves line geometry much better,
2. keeps the line wide rather than squeezing it into a square,
3. is usually a better representation for OCR line crops.

Trade-off:

1. more patches,
2. more memory,
3. lower batch sizes on the same GPU.

Practical recommendation:

1. start with `256 x 1024`,
2. keep dimensions divisible by 16,
3. move to `256 x 1536` or `320 x 2560` only if memory and throughput allow it.

## 10. Recommended tiny run

The recommended first experiment is a non-square `gray` run with letterboxing.

If you have enough GPU memory, this is a solid starting point:

```bash
CUDA_VISIBLE_DEVICES=0 python cli.py train-donut-ocr \
  --train_manifest "$TINY_DIR/train_manifest.jsonl" \
  --val_manifest "$TINY_DIR/val_manifest.jsonl" \
  --output_dir /dev/shm/donut_tiny_gray_lb_256x1024 \
  --model_name_or_path microsoft/trocr-base-stage1 \
  --tokenizer_path "$TOKENIZER" \
  --image_preprocess_pipeline gray \
  --enable_letterboxing \
  --target_height 256 \
  --target_width 1024 \
  --max_target_length 160 \
  --generation_max_length 160 \
  --generation_min_new_tokens 0 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --num_train_epochs 30 \
  --warmup_steps 40 \
  --eval_steps 20 \
  --save_steps 20 \
  --save_total_limit 10 \
  --num_workers 8 \
  --seed 42 \
  --val_eval_max_samples 128
```

If that is too large for your GPU, reduce:

```bash
--per_device_train_batch_size 8
--per_device_eval_batch_size 8
--gradient_accumulation_steps 2
```

If you want to compare color preservation, rerun the same recipe with:

```bash
--image_preprocess_pipeline rgb
```

and a different `--output_dir`.

## 11. When to use a square tiny run instead

Use a square run only when one of these is true:

1. you want to reproduce an older baseline,
2. you are intentionally reusing an existing square `image_processor`,
3. you want the cheapest possible quick smoke test.

Typical square baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python cli.py train-donut-ocr \
  --train_manifest "$TINY_DIR/train_manifest.jsonl" \
  --val_manifest "$TINY_DIR/val_manifest.jsonl" \
  --output_dir /dev/shm/donut_tiny_gray_384 \
  --model_name_or_path microsoft/trocr-base-stage1 \
  --tokenizer_path "$TOKENIZER" \
  --image_preprocess_pipeline gray \
  --image_size 384 \
  --max_target_length 160 \
  --generation_max_length 160 \
  --generation_min_new_tokens 0 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --num_train_epochs 30 \
  --warmup_steps 40 \
  --eval_steps 20 \
  --save_steps 20 \
  --save_total_limit 10 \
  --num_workers 8 \
  --seed 42 \
  --val_eval_max_samples 128
```

Remember: without `--enable_letterboxing`, long lines are still being pushed into a square representation.

## 12. How to read tiny-run health signals

The tiny run is healthy when most of the following are true:

1. `eval_cer` trends downward over time,
2. `eval_cer_empty_pred_count` stays at `0`,
3. `eval_pred_repetitive_ratio` stays near `0`,
4. predicted length starts moving toward reference length,
5. token diversity grows over time,
6. predictions are not dominated by only a handful of special tokens.

Red flags:

1. empty predictions for many samples,
2. special-token loops,
3. repetitive gibberish,
4. very short predictions that never grow,
5. CER stuck near 100 percent for a long time.

The `eval_token_forensics` log block is especially useful. It lets you compare:

1. predicted vs reference token diversity,
2. special-token ratios,
3. average non-pad lengths,
4. the most common predicted tokens.

## 13. Last checkpoint vs best model

This repo writes several training artifacts into the OCR output directory:

1. `checkpoint-<step>` directories,
2. `checkpoint-epoch-<N>-cer-<X>` symlink aliases,
3. `model/`,
4. `tokenizer/`,
5. `image_processor/`,
6. `train_summary.json`.

Important:

1. `checkpoint-<step>` is the actual Hugging Face checkpoint,
2. `model/` is the final saved model,
3. with evaluation enabled, the trainer loads the best model at the end before writing `model/`,
4. therefore `model/` is not automatically the same thing as the last step checkpoint.

If you want the last real checkpoint from the tiny run, resolve it explicitly:

```bash
export TINY_RUN=/dev/shm/donut_tiny_gray_lb_256x1024
export LAST_TINY_CKPT=$(find "$TINY_RUN" -maxdepth 1 -type d -name 'checkpoint-[0-9]*' | sort -V | tail -n 1)
echo "$LAST_TINY_CKPT"
```

If your tiny run used `--save_steps 20` and finished at global step `270`, the last real checkpoint is usually `checkpoint-260`, not `checkpoint-270`.

## 14. Start the full run from the tiny checkpoint

When moving from the tiny dataset to the full dataset:

1. start a new run,
2. use a new `--output_dir`,
3. point `--model_name_or_path` at the tiny checkpoint,
4. reuse the tiny run's `image_processor`,
5. do not use `--resume_from_checkpoint` for this transition.

Example full run that writes into the current repo folder rather than `/dev/shm`:

```bash
export TRAIN_MANIFEST=$DATASET_ROOT/train/meta/lines.jsonl
export VAL_MANIFEST=$DATASET_ROOT/eval/meta/lines.jsonl

export TINY_RUN=/dev/shm/donut_tiny_gray_lb_256x1024
export LAST_TINY_CKPT=$(find "$TINY_RUN" -maxdepth 1 -type d -name 'checkpoint-[0-9]*' | sort -V | tail -n 1)

CUDA_VISIBLE_DEVICES=0 python cli.py train-donut-ocr \
  --train_manifest "$TRAIN_MANIFEST" \
  --val_manifest "$VAL_MANIFEST" \
  --output_dir "$PB_ROOT/donut_full_gray_lb_256x1024" \
  --model_name_or_path "$LAST_TINY_CKPT" \
  --image_processor_path "$TINY_RUN/image_processor" \
  --tokenizer_path "$TOKENIZER" \
  --image_preprocess_pipeline gray \
  --enable_letterboxing \
  --target_height 256 \
  --target_width 1024 \
  --max_target_length 160 \
  --generation_max_length 160 \
  --generation_min_new_tokens 0 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --learning_rate 3e-5 \
  --weight_decay 0.01 \
  --num_train_epochs 80 \
  --warmup_steps 1000 \
  --eval_steps 1000 \
  --save_steps 1000 \
  --save_total_limit 5 \
  --num_workers 8 \
  --seed 42 \
  --val_eval_max_samples 300 \
  --report_to trackio \
  --run_name gray_lb_256x1024_full_lastckpt
```

If you run out of memory, lower the per-device batch sizes and compensate with gradient accumulation.

## 15. When `--resume_from_checkpoint` is the right tool

Use `--resume_from_checkpoint` only when you want to continue the same run.

Examples:

1. the same output directory,
2. the same dataset split setup,
3. the same experiment interrupted halfway through.

Do not use it when changing from tiny manifests to full manifests. That is a new run initialized from a checkpoint, not a resume.

## 16. Common mistakes

### Broken tiny manifests

Problem:

You copy `lines.jsonl` into a new folder and the relative `line_path` values no longer resolve.

Fix:

Write an absolute `image` field as shown in the tiny-dataset recipe.

### Tokenizer path is not local

Problem:

You pass `openpecha/BoSentencePiece` directly and the training code refuses to fall back.

Fix:

Download the tokenizer first and pass a local directory via `--tokenizer_path`.

### Assuming letterboxing is enabled by default

Problem:

You expect a wide OCR line representation but did not pass `--enable_letterboxing`.

Fix:

Always set `--enable_letterboxing` explicitly when you want a non-square input.

### Confusing `model/` with the last checkpoint

Problem:

You think the final `model/` directory is the last saved training step.

Fix:

Use the explicit `find ... checkpoint-*` command when you need the last real checkpoint.

## 17. Practical recommendations

1. Always start with a tiny run for each preprocessing family you care about.
2. Prefer a non-square letterboxed run for OCR lines unless you are intentionally reproducing a square baseline.
3. Keep the tiny run cheap and frequent enough to inspect logs closely.
4. Promote only healthy tiny checkpoints into full training.
5. Compare `gray` and `rgb` when color is likely to matter.
6. Reuse the tiny run's `image_processor` when starting the full run.
7. Keep training and evaluation geometry consistent between tiny and full runs.

## 18. Minimal checklist

Before starting the full run, confirm:

1. dependencies are installed,
2. BoSentencePiece exists locally,
3. the merged dataset is downloaded,
4. tiny manifests resolve images correctly,
5. the tiny run has non-empty predictions,
6. the tiny run shows no repetition collapse,
7. you know whether you want the last tiny checkpoint or the best tiny model.

If all seven are true, you are in a good place to launch the full OCR training run.
