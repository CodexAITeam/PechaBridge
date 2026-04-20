# Batch OCR – Quickstart

Run OCR on a full folder of Pecha images from the command line.

---

## Prerequisites

- Donut OCR model at `models/ocr/best`
- One of the line segmentation options below
- Input images in a folder (e.g. `337138764X/`)

---

## Option A — CV Line Segmentation (default)

Uses a YOLO **layout** model to detect text regions, then splits them into
lines using classical horizontal projection profiles.

```bash
python cli.py batch-ocr \
    --ocr-model    models/ocr/best \
    --layout-model models/layout/your_layout.pt \
    --input-dir    337138764X
```

> `--layout-engine cv` is the default and can be omitted.

**Output folder:** `337138764X__best/`

---

## Option B — YOLO Line Segmentation

Uses a dedicated YOLO line segmentation model to detect line boxes directly.

```bash
python cli.py batch-ocr \
    --ocr-model      models/ocr/best \
    --layout-engine  yolo_line \
    --line-model     models/yolo_line_seg.pt \
    --input-dir      337138764X
```

**Output folder:** `337138764X__best__yolo_line/`

---

## Option C — BDRC Line Segmentation

Uses the BDRC ONNX line model. Auto-downloads the default model if
`--bdrc-line-model` is omitted.

```bash
python cli.py batch-ocr \
    --ocr-model      models/ocr/best \
    --layout-engine  bdrc_line \
    --input-dir      337138764X
```

**Output folder:** `337138764X__best__bdrc_line/`

---

## Merge results into one file

After batch OCR completes, merge all per-image `.txt` files into a single file:

```bash
python scripts/merge_txt_files.py 337138764X__best merged.txt
```

Replace `337138764X__best` with the actual output folder name from above.

---

## Common options

| Flag | Default | Description |
|---|---|---|
| `--device` | `auto` | `cuda:0`, `cpu`, or `auto` |
| `--max-len` | model default | Max token length for Donut decoding |
| `--line-preprocess` | `gray` | Preprocessing for `yolo_line`: `none`, `gray`, `bdrc`, `rgb` |

---

## Full example with GPU

```bash
python cli.py batch-ocr \
    --ocr-model      models/ocr/best \
    --layout-engine  yolo_line \
    --line-model     models/yolo_line_seg.pt \
    --input-dir      337138764X \
    --device         cuda:0
```
