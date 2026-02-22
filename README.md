![PechaBridge Hero](assets/hero_pb.jpeg)

# PechaBridge Workbench

## Project Description

PechaBridge is a workflow for **Tibetan pecha document understanding** with a focus on layout detection, synthetic data generation, SBB data ingestion, OCR/VLM-assisted parsing, diffusion-based texture augmentation, and retrieval-ready patch pipelines.

The project combines:

- synthetic YOLO dataset generation for Tibetan/number classes,
- training and evaluation of detection models,
- large-scale processing of SBB page images,
- optional VLM backends for layout extraction,
- SDXL/SD2.1 + ControlNet + LoRA texture adaptation,
- line sub-patch dataset generation with Option-A neighborhood metadata,
- weak OCR labeling + robust MNN mining for cross-page positives,
- ViT/DINOv2 retrieval training (mp-InfoNCE) and FAISS-based evaluation/search.

The primary entrypoint for end-to-end usage is the **Workbench UI** (`ui_workbench.py`).

## Core Features

- **Synthetic multi-class dataset generation**: Creates YOLO-ready pages for Tibetan number words, Tibetan text blocks, and Chinese number words.
- **OCR-ready target export**: Optionally saves rendered OCR targets with deterministic line linearization and optional OCR crop export by label.
- **Detection training and inference**: Provides Ultralytics YOLO training, validation, and inference workflows for local data and SBB pages.
- **Pseudo-labeling and rule-based filtering**: Supports VLM-assisted layout extraction plus post-filtering before annotation review.
- **Donut-style OCR workflow (Label 1)**: Runs generation, manifest preparation, tokenizer handling, and Vision Transformer encoder + autoregressive decoder training.
- **Diffusion texture adaptation**: Includes SDXL/SD2.1 + ControlNet augmentation and optional LoRA integration for more realistic page textures.
- **Patch retrieval dataset generation**: Generates line sub-patches (`datasets/text_patches`) with parquet metadata for retrieval training.
- **Weak supervision for retrieval**: Supports weak OCR labels and robust MNN mining for cross-page positives.
- **Retrieval encoder training + eval**: Trains ViT/DINOv2 patch encoders with mp-InfoNCE and exports FAISS-ready embeddings plus cross-page evaluation.
- **Retrieval encoder preparation (legacy/base)**: Includes unpaired image/text encoder training as a base for broader Tibetan retrieval experiments.

## Example: SBB Layout Analysis

The figure below shows an example layout analysis result for a Tibetan pecha page from the Staatsbibliothek zu Berlin (SBB).
Detected layout regions are overlaid on the original page image and illustrate the intended detection quality for real-world scans.

![Example layout analysis on an SBB pecha page](assets/pecha_layout%20analysis.jpeg)

## Project Goals

1. Build a robust pipeline for Tibetan page layout analysis that works with limited labeled data.
2. Improve model quality through synthetic data and realistic texture transfer from real scans.
3. Support scalable ingestion and weak supervision on large historical collections (for example SBB PPNs).
4. Prepare retrieval-ready representations (image and text encoders) for future Tibetan n-gram search.
5. Keep all major workflows reproducible in both UI and CLI.

## Roadmap

1. Data Foundation:
Synthetic generation, SBB download pipeline, and dataset QA/export workflows.
2. Detection and Parsing:
YOLO training/inference plus optional VLM-assisted layout parsing and pseudo-labeling.
3. Realism and Domain Adaptation:
Diffusion + LoRA texture workflows to bridge synthetic-to-real domain gaps.
4. Retrieval Readiness:
Train unpaired image/text encoders and establish schemas/pipelines for retrieval indexing.
5. Retrieval System:
Dual-encoder alignment, ANN indexing, provenance-aware search results, and iterative evaluation.

## Install

```bash
pip install -r requirements.txt
```

`requirements.txt` is now the **unified** dependency file for:

- Workbench UI
- VLM backends
- Diffusion + LoRA workflows
- Retrieval encoder training

Legacy files `requirements-ui.txt`, `requirements-vlm.txt`, and `requirements-lora.txt` remain as compatibility wrappers.

## Documentation Guide

- CLI command reference and end-to-end examples: [README_CLI.md](README_CLI.md)
- Pseudo-labeling and Label Studio workflow: [README_PSEUDO_LABELING_LABEL_STUDIO.md](README_PSEUDO_LABELING_LABEL_STUDIO.md)
- Patch dataset generation (YOLO textbox -> lines -> sub-patches): [docs/dataset_generation.md](docs/dataset_generation.md)
- Robust MNN mining for cross-page positives: [docs/mnn_mining.md](docs/mnn_mining.md)
- Retrieval training with mp-InfoNCE (MNN/OCR weak positives): [docs/retrieval_mpnce_training.md](docs/retrieval_mpnce_training.md)
- Weak OCR labeling for patch datasets: [docs/weak_ocr.md](docs/weak_ocr.md)
- Diffusion + LoRA details: [docs/texture_augmentation.md](docs/texture_augmentation.md)
- Retrieval roadmap: [docs/tibetan_ngram_retrieval_plan.md](docs/tibetan_ngram_retrieval_plan.md)
- Chinese number corpus note: [data/corpora/Chinese Number Words/README.md](data/corpora/Chinese%20Number%20Words/README.md)

## Start the Workbench

```bash
python ui_workbench.py
```

Optional runtime flags via environment variables:

```bash
export UI_HOST=127.0.0.1   # use 0.0.0.0 for remote server binding
export UI_PORT=7860
export UI_SHARE=false      # set true only if you explicitly want a public Gradio link
python ui_workbench.py
```

### SSH Port Forwarding (server -> laptop)

If the Workbench runs on a remote host, keep `UI_SHARE=false` and use SSH forwarding:

```bash
ssh -L 7860:127.0.0.1:7860 <user>@<server>
```

Then open `http://127.0.0.1:7860` on your laptop.

## Recommended Workflow (UI + CLI)

1. `Synthetic Data` / `PPN Downloader`: build or ingest page images.
2. `Ultralytics Training` + `Model Inference`: train and validate layout detection (YOLO).
3. `Workbench Preview Tabs`: inspect detected text blocks, line segmentation, and hierarchy overlays on real pecha pages.
4. `Label Studio Export` / pseudo-label workflow: generate reviewable labels when needed.
5. `gen-patches` (CLI): build `datasets/text_patches` from page images using YOLO textbox detection + classical line segmentation.
6. `weak_ocr_label` (CLI module, optional): create weak OCR labels on patch crops.
7. `mine-mnn-pairs` (CLI): mine robust cross-page positives for retrieval training.
8. `train-text-hierarchy-vit` (CLI): train patch retrieval encoder with mp-InfoNCE using `MNN`, `OCR`, or `both`.
9. `eval-faiss-crosspage` and `faiss-text-hierarchy-search` (CLI): evaluate and inspect retrieval behavior.

## Unified CLI

The project includes a unified CLI entrypoint:

```bash
python cli.py -h
```

Key commands:

```bash
# Texture LoRA dataset prep
python cli.py prepare-texture-lora-dataset --input_dir ./sbb_images --output_dir ./datasets/texture-lora-dataset

# Train texture LoRA (SDXL or SD2.1 via --model_family)
python cli.py train-texture-lora --dataset_dir ./datasets/texture-lora-dataset --output_dir ./models/texture-lora-sdxl

# Texture augmentation inference
python cli.py texture-augment --input_dir ./datasets/tibetan-yolo-ui/train/images --output_dir ./datasets/tibetan-yolo-ui-textured

# Train image encoder (self-supervised)
python cli.py train-image-encoder --input_dir ./sbb_images --output_dir ./models/image-encoder

# Train text encoder (unsupervised, Unicode-normalized)
python cli.py train-text-encoder --input_dir ./data/corpora --output_dir ./models/text-encoder

# Generate patch retrieval dataset (YOLO textbox -> lines -> multi-scale patches)
python cli.py gen-patches \
  --model ./models/layoutModels/layout_model.pt \
  --input-dir ./sbb_images \
  --output-dir ./datasets/text_patches \
  --no-samples 20 \
  --debug-dump 5

# Mine cross-page MNN positives for retrieval training
python cli.py mine-mnn-pairs \
  --dataset ./datasets/text_patches \
  --meta ./datasets/text_patches/meta/patches.parquet \
  --out ./datasets/text_patches/meta/mnn_pairs.parquet \
  --config ./configs/mnn_mining.yaml \
  --num-workers 8

# Train patch retrieval encoder with mp-InfoNCE (MNN/OCR/both)
python cli.py train-text-hierarchy-vit \
  --dataset-dir ./datasets/text_patches \
  --output-dir ./models/text_hierarchy_vit_mpnce \
  --model-name-or-path facebook/dinov2-base \
  --train-mode patch_mpnce \
  --positive-sources both \
  --pairs-parquet ./datasets/text_patches/meta/mnn_pairs.parquet \
  --weak-ocr-parquet ./datasets/text_patches/meta/weak_ocr.parquet

# Cross-page FAISS evaluation from exported embeddings
python cli.py eval-faiss-crosspage \
  --embeddings-npy ./models/text_hierarchy_vit_mpnce/faiss_embeddings.npy \
  --embeddings-meta ./models/text_hierarchy_vit_mpnce/faiss_embeddings_meta.parquet \
  --mnn-pairs ./datasets/text_patches/meta/mnn_pairs.parquet \
  --output-dir ./models/text_hierarchy_vit_mpnce/eval_crosspage

# Full label-1 OCR workflow (generate -> prepare -> train)
python cli.py run-donut-ocr-workflow \
  --dataset_name tibetan-donut-ocr-label1 \
  --dataset_output_dir ./datasets \
  --font_path_tibetan "ext/Microsoft Himalaya.ttf" \
  --font_path_chinese ext/simkai.ttf \
  --model_output_dir ./models/donut-ocr-label1
```

Weak OCR label generation for patch datasets is currently exposed as a module CLI:

```bash
python -m pechabridge.cli.weak_ocr_label \
  --dataset ./datasets/text_patches \
  --meta ./datasets/text_patches/meta/patches.parquet \
  --out ./datasets/text_patches/meta/weak_ocr.parquet
```

## Label Studio Notes

For local file serving in Label Studio, set:

```bash
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/absolute/path/to/your/dataset/root
```

Then use the Workbench export actions.

## CLI Documentation

CLI usage is documented separately in:

- [README_CLI.md](README_CLI.md)
- [README_PSEUDO_LABELING_LABEL_STUDIO.md](README_PSEUDO_LABELING_LABEL_STUDIO.md)
- [docs/dataset_generation.md](docs/dataset_generation.md)
- [docs/mnn_mining.md](docs/mnn_mining.md)
- [docs/retrieval_mpnce_training.md](docs/retrieval_mpnce_training.md)
- [docs/weak_ocr.md](docs/weak_ocr.md)
- [docs/texture_augmentation.md](docs/texture_augmentation.md)
- [docs/tibetan_ngram_retrieval_plan.md](docs/tibetan_ngram_retrieval_plan.md)

## License

MIT, see [LICENSE](LICENSE).
