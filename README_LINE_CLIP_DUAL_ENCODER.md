# Line-CLIP Dual Vision-Text Encoder (DINOv2 + Text Encoder)

## Purpose

`train-text-hierarchy-vit --train-mode line_clip` trains a **CLIP-style dual encoder** on line-level OCR manifests:

- vision encoder: line image -> embedding
- text encoder: line text -> embedding
- objective: align paired line image/text embeddings for retrieval

This is the recommended path for:

- text-to-line retrieval on Tibetan OCR datasets
- line-to-text retrieval
- pretraining a retrieval model before domain adaptation on Stabi pechas

## Input Format

JSONL manifest with line image + text pairs (OpenPecha / BDRC style).

The loader supports common aliases, including:

- image: `line_path`, `src__image`, `image`, `image_path`, ...
- text: `text`, `src__label`, `label`, `transcription`, ...

## CLI Example (Single GPU)

```bash
CUDA_VISIBLE_DEVICES=1 python cli.py train-text-hierarchy-vit \
  --dataset-dir /home/ubuntu/data/PechaBridge/datasets/openpecha_ocr_lines \
  --output-dir /home/ubuntu/data/PechaBridge/models/line_clip_openpecha_bdrc_dinov2_byt5_1gpu_bs32_lr5e5 \
  --train-mode line_clip \
  --train-manifest /home/ubuntu/data/PechaBridge/datasets/openpecha_ocr_lines/train/meta/lines.jsonl \
  --val-manifest /home/ubuntu/data/PechaBridge/datasets/openpecha_ocr_lines/eval/meta/lines.jsonl \
  --model-name-or-path facebook/dinov2-base \
  --text-encoder-name-or-path google/byt5-small \
  --image-preprocess-pipeline bdrc \
  --batch-size 32 \
  --lr 5e-5 \
  --weight-decay 0.05 \
  --warmup-steps 1000 \
  --text-max-length 256 \
  --num-train-epochs 20 \
  --num-workers 6 \
  --mixed-precision bf16 \
  --checkpoint-every-steps 2400
```

## Multi-GPU (Accelerate)

For true multi-GPU training, use `accelerate launch` (not plain `python` with `CUDA_VISIBLE_DEVICES` only):

```bash
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num_processes 3 cli.py train-text-hierarchy-vit \
  --dataset-dir /home/ubuntu/data/PechaBridge/datasets/openpecha_ocr_lines \
  --output-dir /home/ubuntu/data/PechaBridge/models/line_clip_openpecha_bdrc_dinov2_byt5_3gpu \
  --train-mode line_clip \
  --train-manifest /home/ubuntu/data/PechaBridge/datasets/openpecha_ocr_lines/train/meta/lines.jsonl \
  --val-manifest /home/ubuntu/data/PechaBridge/datasets/openpecha_ocr_lines/eval/meta/lines.jsonl \
  --model-name-or-path facebook/dinov2-base \
  --text-encoder-name-or-path google/byt5-small \
  --image-preprocess-pipeline bdrc \
  --batch-size 64 \
  --lr 5e-5 \
  --num-train-epochs 20 \
  --num-workers 8 \
  --mixed-precision bf16 \
  --checkpoint-every-steps 2400
```

## Text Encoder Choice

### `google/byt5-small` (recommended baseline)

- robust to OCR noise (byte-level)
- no tokenizer OOV issues
- good baseline for Tibetan OCR text

Tradeoff:

- long byte sequences -> increase `--text-max-length` (typically `256` or `384`)

### `BoSentencePiece`

- useful as a tokenizer in OCR workflows
- **not** a text encoder model by itself
- cannot be used directly as `--text-encoder-name-or-path`

## Metrics (Training Loop)

### In-batch metrics (always available)

- `loss`: CLIP-style symmetric InfoNCE loss (lower is better)
- `i2t`: image->text in-batch top-1 accuracy
- `t2i`: text->image in-batch top-1 accuracy

These are good optimization indicators but not final retrieval quality metrics.

### Periodic validation retrieval (now in training loop)

If `--val-manifest` is provided, the trainer periodically computes retrieval metrics and shows them in TQDM postfix:

- `vi2t@1`: validation image->text Recall@1
- `vt2i@1`: validation text->image Recall@1

The interval currently follows:

- `--checkpoint-every-steps`

At each evaluation step, the trainer also logs:

- `val_i2t_r1/r5/r10`
- `val_t2i_r1/r5/r10`

## Checkpoints

`line_clip` now writes periodic checkpoints during training (not only final save), controlled by:

- `--checkpoint-every-steps`

Example artifacts at checkpoint step `2400`:

- `checkpoint_step_0002400_text_hierarchy_vit_backbone/`
- `checkpoint_step_0002400_text_hierarchy_projection_head.pt`
- `checkpoint_step_0002400_training_config.json`

## Final Output Artifacts

Inside `--output_dir` after training:

- `text_hierarchy_vit_backbone/` (image backbone + processor)
- `text_hierarchy_projection_head.pt` (image projection head)
- `text_hierarchy_clip_text_encoder/` (HF text encoder + tokenizer)
- `text_hierarchy_clip_text_projection_head.pt`
- `faiss_embeddings.npy`
- `faiss_embeddings_meta.parquet`
- `training_config.json`

## Workbench Compatibility

Currently usable in UI:

- image backbone (`text_hierarchy_vit_backbone/`)
- image projection head (`text_hierarchy_projection_head.pt`)

Not yet directly used in UI:

- text encoder (`text_hierarchy_clip_text_encoder/`)
- text projection head (`text_hierarchy_clip_text_projection_head.pt`)

That means the Workbench can use the line-CLIP model for image-side encoding previews/search flows, while full dual-encoder text-query usage is still primarily CLI-driven.
