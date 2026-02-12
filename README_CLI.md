# PechaBridge CLI Reference

This document contains the command-line workflow and script reference.
If you are a regular user, prefer the UI in `README.md`.

## Main Scripts

- `generate_training_data.py`
- `train_model.py`
- `inference_sbb.py`
- `ocr_on_detections.py`
- `pseudo_label_from_vlm.py`
- `layout_rule_filter.py`
- `run_pseudo_label_workflow.py`

## Install

```bash
pip install -r requirements.txt
```

Optional UI/VLM extras:

```bash
pip install -r requirements-ui.txt
pip install -r requirements-vlm.txt
```

## Example CLI Workflow

### 1) Generate synthetic dataset

```bash
python generate_training_data.py \
  --train_samples 100 \
  --val_samples 100 \
  --font_path_tibetan ext/Microsoft\ Himalaya.ttf \
  --font_path_chinese ext/simkai.ttf \
  --dataset_name tibetan-yolo
```

### 2) Train model

```bash
python train_model.py --dataset tibetan-yolo --epochs 100 --export
```

### 3) Inference on SBB

```bash
python inference_sbb.py --ppn 337138764X --model runs/detect/train/weights/best.pt
```

### 4) OCR / parser inference

List available parsers:

```bash
python ocr_on_detections.py --list-parsers
```

Legacy parser:

```bash
python ocr_on_detections.py --source image.jpg --parser legacy --model runs/detect/train/weights/best.pt --lang bod
```

MinerU2.5 parser:

```bash
python ocr_on_detections.py --source image.jpg --parser mineru25 --mineru-command mineru
```

Transformer parser examples:

```bash
python ocr_on_detections.py --source image.jpg --parser paddleocr_vl
python ocr_on_detections.py --source image.jpg --parser qwen25vl
python ocr_on_detections.py --source image.jpg --parser granite_docling
python ocr_on_detections.py --source image.jpg --parser deepseek_ocr
```

## Label Studio (CLI)

```bash
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$(pwd)/datasets/tibetan-yolo

label-studio-converter import yolo \
  -i datasets/tibetan-yolo/train \
  -o ls-tasks.json \
  --image-ext ".png" \
  --image-root-url "/data/local-files/?d=train/images"
```

Start Label Studio:

```bash
label-studio
```

## Additional Docs

- Pseudo-labeling and Label Studio import details: [README_PSEUDO_LABELING_LABEL_STUDIO.md](README_PSEUDO_LABELING_LABEL_STUDIO.md)
