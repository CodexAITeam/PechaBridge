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
- `cli.py` (unified diffusion + retrieval-encoder commands)

## Install

```bash
pip install -r requirements.txt
```

`requirements.txt` is the unified dependency file for CLI, UI, VLM, diffusion/LoRA, and retrieval encoder training.
Legacy files `requirements-ui.txt`, `requirements-vlm.txt`, and `requirements-lora.txt` remain as compatibility wrappers.

## Unified CLI (`cli.py`)

Use:

```bash
python cli.py -h
```

Available subcommands:

- `prepare-texture-lora-dataset`
- `train-texture-lora`
- `texture-augment`
- `train-image-encoder`
- `train-text-encoder`
- `prepare-donut-ocr-dataset`
- `train-donut-ocr`
- `run-donut-ocr-workflow`

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

Optional: apply LoRA-based texture augmentation directly during data generation:

```bash
python generate_training_data.py \
  --train_samples 100 \
  --val_samples 20 \
  --font_path_tibetan ext/Microsoft\ Himalaya.ttf \
  --font_path_chinese ext/simkai.ttf \
  --dataset_name tibetan-yolo \
  --lora_augment_path ./models/texture-lora-sdxl/texture_lora.safetensors \
  --lora_augment_splits train \
  --lora_augment_targets images
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

### 5) Donut-style OCR workflow (Label 1 only)

End-to-end (generate synthetic data + prepare manifests + train OCR model):

```bash
python cli.py run-donut-ocr-workflow \
  --dataset_name tibetan-donut-ocr-label1 \
  --dataset_output_dir ./datasets \
  --font_path_tibetan "ext/Microsoft Himalaya.ttf" \
  --font_path_chinese ext/simkai.ttf \
  --train_samples 2000 \
  --val_samples 200 \
  --target_newline_token "<NL>" \
  --model_output_dir ./models/donut-ocr-label1
```

Optional with LoRA augmentation during the generation step:

```bash
python cli.py run-donut-ocr-workflow \
  --dataset_name tibetan-donut-ocr-label1 \
  --dataset_output_dir ./datasets \
  --font_path_tibetan "ext/Microsoft Himalaya.ttf" \
  --font_path_chinese ext/simkai.ttf \
  --lora_augment_path ./models/texture-lora-sdxl/texture_lora.safetensors \
  --lora_augment_splits train \
  --lora_augment_targets images_and_ocr_crops \
  --model_output_dir ./models/donut-ocr-label1
```

Manual step-by-step:

```bash
# A) Synthetic data + OCR crops/targets (label 1 only for crops)
python generate_training_data.py \
  --dataset_name tibetan-donut-ocr-label1 \
  --output_dir ./datasets \
  --font_path_tibetan "ext/Microsoft Himalaya.ttf" \
  --font_path_chinese ext/simkai.ttf \
  --train_samples 2000 \
  --val_samples 200 \
  --save_rendered_text_targets \
  --save_ocr_crops \
  --ocr_crop_labels 1 \
  --target_newline_token "<NL>"

# B) Prepare JSONL manifests from ocr_targets/ocr_crops (label_id=1)
python cli.py prepare-donut-ocr-dataset \
  --dataset_dir ./datasets/tibetan-donut-ocr-label1 \
  --output_dir ./datasets/tibetan-donut-ocr-label1/donut_ocr_label1 \
  --label_id 1

# C) Train VisionEncoderDecoder OCR model
python cli.py train-donut-ocr \
  --train_manifest ./datasets/tibetan-donut-ocr-label1/donut_ocr_label1/train_manifest.jsonl \
  --val_manifest ./datasets/tibetan-donut-ocr-label1/donut_ocr_label1/val_manifest.jsonl \
  --output_dir ./models/donut-ocr-label1 \
  --model_name_or_path microsoft/trocr-base-stage1 \
  --train_tokenizer
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
- Diffusion + LoRA details: [docs/texture_augmentation.md](docs/texture_augmentation.md)
- Retrieval roadmap: [docs/tibetan_ngram_retrieval_plan.md](docs/tibetan_ngram_retrieval_plan.md)
