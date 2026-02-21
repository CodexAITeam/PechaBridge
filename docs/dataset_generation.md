# Patch Dataset Generation

This pipeline generates line sub-patches for retrieval training with **Option A implicit neighborhood metadata**.

## What it does

1. Detects Tibetan textboxes per page using model `m` (YOLO).
2. Detects lines inside each textbox using algorithm `A` (classical vertical profile segmentation).
3. Normalizes each line to fixed height `Ht`, computes ink map `J`, horizontal profile `p(x)`, and boundary minima.
4. Builds dense and boundary-aligned windows per scale.
5. Filters by ink ratio and samples candidates per line/scale.
6. Saves patch PNGs and Parquet metadata.

## CLI

Via package entrypoint:

```bash
python -m pechabridge.cli.gen_patches --config configs/patch_gen.yaml
```

Via project root CLI:

```bash
python cli.py gen-patches --config configs/patch_gen.yaml
```

Override YAML keys from CLI, e.g.:

```bash
python cli.py gen-patches \
  --config configs/patch_gen.yaml \
  --model models/your_layout_model.pt \
  --input-dir sbb_images \
  --output-dir datasets/pecha_line_patches \
  --no-samples 200 \
  --debug-dump 12
```

## Output layout

```
out_dataset/
  patches/
    doc={doc_id}/page={page_id}/line={line_id}/scale={scale_w}/patch_{patch_id}.png
  meta/
    patches.parquet
  debug/
    ... overlays (if --debug-dump > 0)
```

## Option A neighborhood

For each group `(doc_id, page_id, line_id, scale_w)`:

- metadata rows are sorted by `x0_px`
- `k` is assigned as contiguous `0..n-1`

Neighborhood can be derived on-the-fly from `(doc_id, page_id, line_id, scale_w, k)` during training.

