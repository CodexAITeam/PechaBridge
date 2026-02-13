# Pseudo-Labeling + Label Studio (Quick Guide)

This guide describes the streamlined workflow for layout pseudo-labeling with three classes:
- `0`: `tibetan_number_word` (left)
- `1`: `tibetan_text` (center)
- `2`: `chinese_number_word` (right)

For regular day-to-day usage, prefer the Workbench-first flow: [README.md](README.md)

## Option A: Recommended (single command)

```bash
python run_pseudo_label_workflow.py \
  --input-dir data/my_inference_data \
  --work-dir datasets/workflow \
  --split train \
  --parser paddleocr_vl \
  --recursive \
  --allow-relabel \
  --keep-single-per-class \
  --image-ext .jpg
```

Optionally launch Label Studio directly:

```bash
python run_pseudo_label_workflow.py \
  --input-dir data/my_inference_data \
  --work-dir datasets/workflow \
  --split train \
  --parser paddleocr_vl \
  --recursive \
  --allow-relabel \
  --keep-single-per-class \
  --image-ext .jpg \
  --start-label-studio
```

The workflow executes:
1. `pseudo_label_from_vlm.py`
2. `layout_rule_filter.py`
3. `label-studio-converter import yolo`
4. optionally `label-studio`

## Option B: Manual (3 steps)

### 1) Generate pseudo-labels from VLM output

```bash
python pseudo_label_from_vlm.py \
  --input-dir data/my_inference_data \
  --output-dir datasets/pseudo-vlm \
  --split train \
  --parser paddleocr_vl \
  --recursive \
  --copy-images \
  --save-raw-json \
  --min-confidence 0.2
```

Alternative parsers: `qwen25vl`, `granite_docling`, `deepseek_ocr`, `mineru25`.

### 2) Apply layout rules

```bash
python layout_rule_filter.py \
  --input-split-dir datasets/pseudo-vlm/train \
  --output-split-dir datasets/pseudo-vlm-filtered/train \
  --allow-relabel \
  --keep-single-per-class \
  --copy-images
```

### 3) Create Label Studio tasks

```bash
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$(pwd)/datasets/pseudo-vlm-filtered

label-studio-converter import yolo \
  -i datasets/pseudo-vlm-filtered/train \
  -o ls-tasks-pseudo.json \
  --image-ext ".jpg" \
  --image-root-url "/data/local-files/?d=train/images"
```

Then run:

```bash
label-studio
```

## Notes

- `--image-ext` must match the actual file extension (`.jpg` or `.png`).
- Treat SBB/VLM outputs as review/pseudo-label sources; use manually corrected labels as final training data.

## Additional Documentation

- Workbench-first usage: [README.md](README.md)
- CLI reference: [README_CLI.md](README_CLI.md)
