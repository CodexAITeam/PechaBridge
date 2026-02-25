# MNN Mining for Cross-Page Patch Positives

This module mines robust cross-page positive patch pairs from real Pecha scans.

## Input

- Patch metadata parquet (typically `meta/patches.parquet`)
- Patch images on disk (`patches/.../patch_*.png`)
- DINOv2-compatible image encoder (optional projection head checkpoint)

## Algorithm

1. Load/clean metadata and apply filters (`ink_ratio_min`, optional `boundary_score_min`).
2. Embed patches per `scale_w` with L2-normalized vectors.
3. Build FAISS inner-product index per scale.
4. Retrieve filtered neighbors with exclusion rules (same page/line/nearby).
5. Keep only mutual-nearest-neighbor candidates.
6. Stage 1 (fast): build a per-source shortlist of mutual candidates (no stability/multiscale/signature checks yet).
7. Stage 2 (slow): verify only the shortlist with deterministic stability checks and optional multi-scale/signature checks.
8. Keep top pairs per source patch.
9. Save `mnn_pairs.parquet` and a JSON summary.

## CLI

```bash
python -m pechabridge.cli.mine_mnn_pairs \
  --dataset /path/to/out_dataset \
  --meta /path/to/out_dataset/meta/patches.parquet \
  --out /path/to/out_dataset/meta/mnn_pairs.parquet \
  --config /path/to/configs/mnn_mining.yaml \
  --num-workers 8 \
  --debug_dump 50
```

`--debug_dump N` writes random pair preview grids to:

`<dataset>/debug/mnn_pairs/`

## Output

Parquet columns:

- `src_patch_id`, `dst_patch_id`
- `src_doc_id`, `src_page_id`, `src_line_id`, `src_scale_w`
- `dst_doc_id`, `dst_page_id`, `dst_line_id`, `dst_scale_w`
- `sim`
- `rank_src_to_dst`, `rank_dst_to_src`
- `stability_count`, `stability_ratio`
- `multi_scale_ok`
- `notes`

Summary JSON:

- same path as output parquet with suffix `.summary.json`
- includes counts, sim/stability stats, and top doc/page match sources

## Performance Notes

- `performance.two_stage_verify: true` (default) is recommended for large scales.
- For a quick first pass, disable `stability.enabled` and `multiscale.enabled`.
- Reduce `mining.topK` / `mining.mutual_topK` for faster candidate generation.
- `--num-workers` controls source-loop mining threads (and also embedding DataLoader / FAISS threads).
