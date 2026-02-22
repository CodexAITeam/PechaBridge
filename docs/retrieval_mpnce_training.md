# Retrieval Training (mp-InfoNCE + MNN Pairs)

This workflow trains a ViT/DINOv2 patch encoder with:
- multi-positive InfoNCE (MNN + optional overlap/multiscale positives)
- optional overlap smoothness regularization
- two-phase schedule (frozen backbone, then partial unfreeze)
- FAISS-ready embedding export

## 1) Mine cross-page MNN positives

```bash
python cli.py mine-mnn-pairs \
  --dataset /home/ubuntu/data/PechaBridge/datasets/text_patches \
  --meta /home/ubuntu/data/PechaBridge/datasets/text_patches/meta/patches.parquet \
  --out /home/ubuntu/data/PechaBridge/datasets/text_patches/meta/mnn_pairs.parquet \
  --config /home/ubuntu/data/PechaBridge/configs/mnn_mining.yaml \
  --debug_dump 20
```

## 2) Train retrieval encoder (auto patch mode)

```bash
python cli.py train-text-hierarchy-vit \
  --dataset-dir /home/ubuntu/data/PechaBridge/datasets/text_patches \
  --output-dir /home/ubuntu/data/PechaBridge/models/text_hierarchy_vit_mpnce \
  --model-name-or-path facebook/dinov2-base \
  --train-mode auto \
  --patch-meta-parquet /home/ubuntu/data/PechaBridge/datasets/text_patches/meta/patches.parquet \
  --pairs-parquet /home/ubuntu/data/PechaBridge/datasets/text_patches/meta/mnn_pairs.parquet \
  --batch-size 64 \
  --num-train-epochs 10 \
  --phase1-epochs 2 \
  --phase2-epochs 8 \
  --unfreeze-last-n-blocks 2 \
  --temperature 0.07 \
  --w-overlap 0.3 \
  --w-multiscale 0.2 \
  --lambda-smooth 0.05
```

Training exports:
- `faiss_embeddings.npy`
- `faiss_embeddings_meta.parquet`
- backbone + projection head checkpoints

## 3) Cross-page FAISS evaluation

```bash
python cli.py eval-faiss-crosspage \
  --embeddings-npy /home/ubuntu/data/PechaBridge/models/text_hierarchy_vit_mpnce/faiss_embeddings.npy \
  --embeddings-meta /home/ubuntu/data/PechaBridge/models/text_hierarchy_vit_mpnce/faiss_embeddings_meta.parquet \
  --mnn-pairs /home/ubuntu/data/PechaBridge/datasets/text_patches/meta/mnn_pairs.parquet \
  --output-dir /home/ubuntu/data/PechaBridge/models/text_hierarchy_vit_mpnce/eval_crosspage \
  --recall-ks 1,5,10 \
  --exclude-same-page
```

Output summary:
- `faiss_crosspage_eval_summary.json`

