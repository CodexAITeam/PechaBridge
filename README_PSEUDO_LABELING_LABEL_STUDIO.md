# Pseudo-Labeling + Label Studio (Kurzfassung)

Diese Anleitung beschreibt nur den schlanken Ablauf für Layout-Pseudo-Labels mit drei Klassen:
- `0`: `tibetan_number_word` (links)
- `1`: `tibetan_text` (mitte)
- `2`: `chinese_number_word` (rechts)

Für den normalen Betrieb nutze primär die Workbench: [README.md](README.md)

## Option A: Empfohlen (ein Kommando)

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

Optional Label Studio direkt starten:

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

Das Skript führt aus:
1. `pseudo_label_from_vlm.py`
2. `layout_rule_filter.py`
3. `label-studio-converter import yolo`
4. optional `label-studio`

## Option B: Manuell (3 Schritte)

### 1) Pseudo-Labels aus VLM erzeugen

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

Alternative Parser: `qwen25vl`, `granite_docling`, `deepseek_ocr`, `mineru25`.

### 2) Layout-Regeln anwenden

```bash
python layout_rule_filter.py \
  --input-split-dir datasets/pseudo-vlm/train \
  --output-split-dir datasets/pseudo-vlm-filtered/train \
  --allow-relabel \
  --keep-single-per-class \
  --copy-images
```

### 3) Label Studio Tasks erzeugen

```bash
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$(pwd)/datasets/pseudo-vlm-filtered

label-studio-converter import yolo \
  -i datasets/pseudo-vlm-filtered/train \
  -o ls-tasks-pseudo.json \
  --image-ext ".jpg" \
  --image-root-url "/data/local-files/?d=train/images"
```

Dann:

```bash
label-studio
```

## Hinweise

- `--image-ext` muss zu den tatsächlichen Bilddateien passen (`.jpg` oder `.png`).
- Die SBB/VLM-Outputs sollten als Review-/Pseudo-Label-Quelle behandelt werden; finale Trainingslabels nach manueller Korrektur übernehmen.

## Weitere Doku

- Workbench-first Nutzung: [README.md](README.md)
- CLI-Sammelreferenz: [README_CLI.md](README_CLI.md)
