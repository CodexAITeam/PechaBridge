#!/usr/bin/env python3
"""Probe line_clip Workbench corpus retrieval on random samples across splits."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ui_workbench import (
    _cer_single_ui,
    _get_or_build_line_clip_debug_corpus_ui,
    scan_line_clip_dual_models_ui,
)

LOGGER = logging.getLogger("probe_line_clip_workbench_random_samples")

DEFAULT_MODELS_DIR = "/home/ubuntu/data/PechaBridge/models"
DEFAULT_DATASET_ROOT = "/home/ubuntu/data/PechaBridge/datasets/openpecha_ocr_lines"
_SPLIT_ORDER = ("train", "val", "eval", "test")


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="probe-line-clip-workbench-random-samples",
        description=(
            "Use the best line_clip model bundle (same sorting as UI Workbench), load/build the persistent corpus cache "
            "for selected OCR splits, and inspect retrieval quality on random in-corpus samples."
        ),
        add_help=add_help,
    )
    p.add_argument("--models-dir", type=str, default=DEFAULT_MODELS_DIR, help="Directory containing line_clip bundles/checkpoints")
    p.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT, help="OCR dataset root with split/meta/lines.jsonl")
    p.add_argument("--splits", type=str, default="train,val,test", help="Comma-separated splits to probe (default: train,val,test)")
    p.add_argument("--samples-per-split", type=int, default=25, help="Random samples per split")
    p.add_argument("--top-k", type=int, default=5, help="Top-k neighbors to include in per-sample details")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    p.add_argument("--device", type=str, default="auto", help="Device used only if cache must be built (auto/cuda/cuda:0/cpu/mps)")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size used only if cache must be built")
    p.add_argument("--text-max-length", type=int, default=256, help="Tokenizer max length used in cache key/build")
    p.add_argument("--no-l2-normalize", action="store_true", help="Use non-normalized embeddings setting (must match cache/build config)")
    p.add_argument("--examples", type=int, default=3, help="How many random sample details to keep per split in JSON report")
    p.add_argument("--json-out", type=str, default="", help="Optional path to write full JSON report")
    return p


def _parse_splits(raw: str) -> List[str]:
    req = [s.strip().lower() for s in str(raw or "").split(",") if s.strip()]
    req = [s for s in req if s in _SPLIT_ORDER]
    seen = set()
    out: List[str] = []
    for s in req:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _available_splits(dataset_root: Path) -> List[str]:
    out: List[str] = []
    for split in _SPLIT_ORDER:
        if (dataset_root / split / "meta" / "lines.jsonl").exists():
            out.append(split)
    return out


def _pick_best_line_clip_bundle(models_dir: str) -> Dict[str, Any]:
    _, image_backbone, image_head, text_encoder, text_head, bundle_state_json, msg = scan_line_clip_dual_models_ui(models_dir)
    LOGGER.info(msg)
    bundles: Dict[str, Dict[str, Any]] = {}
    try:
        parsed = json.loads(bundle_state_json or "{}")
        if isinstance(parsed, dict):
            bundles = parsed
    except Exception:
        bundles = {}
    best_root = ""
    best_rec: Dict[str, Any] = {}
    for root, rec in bundles.items():
        if not isinstance(rec, dict):
            continue
        if str(rec.get("image_backbone", "")) == str(image_backbone) and str(rec.get("text_encoder", "")) == str(text_encoder):
            best_root = str(root)
            best_rec = rec
            break
    if not all([image_backbone, image_head, text_encoder, text_head]):
        raise RuntimeError(f"No valid line_clip bundle found in {models_dir}")
    return {
        "bundle_root": best_root or str(Path(image_backbone).resolve().parent),
        "image_backbone": str(image_backbone),
        "image_head": str(image_head),
        "text_encoder": str(text_encoder),
        "text_head": str(text_head),
        "bundle_label": str(best_rec.get("display_label", best_root or "")),
        "config_path": str(best_rec.get("config_path", "")),
    }


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    kk = max(1, min(int(k), int(scores.shape[0])))
    if kk >= int(scores.shape[0]):
        return np.argsort(-scores)
    part = np.argpartition(-scores, kk - 1)[:kk]
    return part[np.argsort(-scores[part])]


def _rank_of_gt(scores: np.ndarray, gt_idx: int) -> int:
    gt = float(scores[int(gt_idx)])
    return int(np.sum(scores > gt)) + 1


def _dot(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim != 1 or b.ndim != 1 or a.shape[0] != b.shape[0]:
        return 0.0
    return float(np.dot(a.astype(np.float32, copy=False), b.astype(np.float32, copy=False)))


def _split_probe(
    corpus: Dict[str, Any],
    split: str,
    rng: np.random.Generator,
    samples_per_split: int,
    top_k: int,
    examples_to_keep: int,
) -> Dict[str, Any]:
    rows = list(corpus.get("rows") or [])
    img = np.asarray(corpus.get("image_embeddings"), dtype=np.float32)
    txt = np.asarray(corpus.get("text_embeddings"), dtype=np.float32)
    n = int(len(rows))
    if n <= 0 or img.ndim != 2 or txt.ndim != 2 or img.shape[0] != n or txt.shape[0] != n:
        raise RuntimeError(f"Invalid corpus for split={split}: rows={n}, image_shape={getattr(img, 'shape', None)}, text_shape={getattr(txt, 'shape', None)}")

    m = min(max(1, int(samples_per_split)), n)
    sample_idx = rng.choice(n, size=m, replace=False).astype(int).tolist()
    top_k = min(max(1, int(top_k)), n)

    t2i_r1 = t2i_r5 = 0
    i2t_r1 = i2t_r5 = 0
    t2i_ranks: List[int] = []
    i2t_ranks: List[int] = []
    per_sample_details: List[Dict[str, Any]] = []
    t0 = time.time()

    for idx in sample_idx:
        gt_row = rows[idx] if 0 <= idx < n else {}
        gt_text = str(gt_row.get("_text_resolved", "") or "")
        gt_image_path = str(gt_row.get("_image_path_resolved", "") or "")

        # Text -> Image
        sims_t2i = img @ txt[idx]
        rank_t2i = _rank_of_gt(sims_t2i, idx)
        top_t2i = _topk_indices(sims_t2i, top_k)
        t2i_ranks.append(rank_t2i)
        t2i_r1 += int(rank_t2i <= 1)
        t2i_r5 += int(rank_t2i <= 5)

        # Image -> Text
        sims_i2t = txt @ img[idx]
        rank_i2t = _rank_of_gt(sims_i2t, idx)
        top_i2t = _topk_indices(sims_i2t, top_k)
        i2t_ranks.append(rank_i2t)
        i2t_r1 += int(rank_i2t <= 1)
        i2t_r5 += int(rank_i2t <= 5)

        if len(per_sample_details) < max(0, int(examples_to_keep)):
            t2i_hits: List[Dict[str, Any]] = []
            for rank, j in enumerate(top_t2i.tolist(), start=1):
                row_j = rows[j]
                pred_text = str(row_j.get("_text_resolved", "") or "")
                t2i_hits.append(
                    {
                        "rank": rank,
                        "idx": int(j),
                        "score": round(float(sims_t2i[j]), 6),
                        "is_gt": bool(int(j) == int(idx)),
                        "doc_id": str(row_j.get("doc_id", "")),
                        "page_id": str(row_j.get("page_id", "")),
                        "line_id": row_j.get("line_id"),
                        "image_path": str(row_j.get("_image_path_resolved", "")),
                        "text_preview": pred_text[:180],
                        "cer_vs_gt_text": round(float(_cer_single_ui(pred_text, gt_text)), 6),
                        "sem_text_vs_gt_text": round(_dot(txt[j], txt[idx]), 6),
                        "sem_image_vs_gt_image": round(_dot(img[j], img[idx]), 6),
                    }
                )
            i2t_hits: List[Dict[str, Any]] = []
            for rank, j in enumerate(top_i2t.tolist(), start=1):
                row_j = rows[j]
                pred_text = str(row_j.get("_text_resolved", "") or "")
                i2t_hits.append(
                    {
                        "rank": rank,
                        "idx": int(j),
                        "score": round(float(sims_i2t[j]), 6),
                        "is_gt": bool(int(j) == int(idx)),
                        "doc_id": str(row_j.get("doc_id", "")),
                        "page_id": str(row_j.get("page_id", "")),
                        "line_id": row_j.get("line_id"),
                        "text_preview": pred_text[:180],
                        "cer_vs_gt_text": round(float(_cer_single_ui(pred_text, gt_text)), 6),
                        "sem_text_vs_gt_text": round(_dot(txt[j], txt[idx]), 6),
                        "sem_image_vs_gt_image": round(_dot(img[j], img[idx]), 6),
                    }
                )
            per_sample_details.append(
                {
                    "sample_idx": int(idx),
                    "doc_id": str(gt_row.get("doc_id", "")),
                    "page_id": str(gt_row.get("page_id", "")),
                    "line_id": gt_row.get("line_id"),
                    "ground_truth_image_path": gt_image_path,
                    "ground_truth_text_preview": gt_text[:220],
                    "t2i": {
                        "gt_rank": int(rank_t2i),
                        "gt_score": round(float(sims_t2i[idx]), 6),
                        "topk": t2i_hits,
                    },
                    "i2t": {
                        "gt_rank": int(rank_i2t),
                        "gt_score": round(float(sims_i2t[idx]), 6),
                        "topk": i2t_hits,
                    },
                }
            )

    elapsed_s = max(1e-9, float(time.time() - t0))

    def _stats(ranks: List[int], hits1: int, hits5: int) -> Dict[str, Any]:
        if not ranks:
            return {"n": 0, "r1": 0.0, "r5": 0.0, "mrr": 0.0, "mean_rank": None, "median_rank": None}
        arr = np.asarray(ranks, dtype=np.int64)
        return {
            "n": int(arr.size),
            "r1": round(float(hits1 / arr.size), 6),
            "r5": round(float(hits5 / arr.size), 6),
            "mrr": round(float(np.mean(1.0 / arr.astype(np.float64))), 6),
            "mean_rank": round(float(np.mean(arr.astype(np.float64))), 3),
            "median_rank": round(float(np.median(arr.astype(np.float64))), 3),
        }

    return {
        "split": str(split),
        "corpus_count": int(n),
        "cache_source": str(corpus.get("cache_source", "")),
        "disk_cache_dir": str(corpus.get("disk_cache_dir", "")),
        "device": str(corpus.get("device", "")),
        "device_msg": str(corpus.get("device_msg", "")),
        "timing": dict(corpus.get("timing") or {}),
        "perf": dict(corpus.get("perf") or {}),
        "sampled_n": int(m),
        "probe_elapsed_s": round(float(elapsed_s), 4),
        "probe_samples_per_s": round(float(m / elapsed_s), 2),
        "metrics": {
            "text_to_image": _stats(t2i_ranks, t2i_r1, t2i_r5),
            "image_to_text": _stats(i2t_ranks, i2t_r1, i2t_r5),
        },
        "examples": per_sample_details,
    }


def run(args: argparse.Namespace) -> int:
    dataset_root = Path(str(args.dataset_root or "").strip()).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    req_splits = _parse_splits(str(args.splits or ""))
    avail_splits = _available_splits(dataset_root)
    splits = [s for s in req_splits if s in avail_splits] if req_splits else avail_splits
    if not splits:
        raise RuntimeError(f"No requested splits available under {dataset_root}; available={avail_splits}")

    bundle = _pick_best_line_clip_bundle(str(args.models_dir))
    if bundle.get("bundle_label"):
        LOGGER.info("Selected best line_clip bundle: %s", bundle["bundle_label"])
    else:
        LOGGER.info("Selected best line_clip bundle: %s", bundle.get("bundle_root"))

    l2_normalize = not bool(getattr(args, "no_l2_normalize", False))
    top_k = max(1, int(getattr(args, "top_k", 5)))
    seed = int(getattr(args, "seed", 42))
    samples_per_split = max(1, int(getattr(args, "samples_per_split", 25)))
    examples = max(0, int(getattr(args, "examples", 3)))
    batch_size = max(1, int(getattr(args, "batch_size", 32)))
    text_max_length = max(8, int(getattr(args, "text_max_length", 256)))

    rng = np.random.default_rng(seed)
    results: List[Dict[str, Any]] = []
    t0_all = time.time()
    for split in splits:
        LOGGER.info("Loading/building corpus cache for split=%s ...", split)
        corpus = _get_or_build_line_clip_debug_corpus_ui(
            dataset_root=str(dataset_root),
            split_name=str(split),
            image_backbone_path=str(bundle["image_backbone"]),
            image_head_path=str(bundle["image_head"]),
            text_encoder_dir=str(bundle["text_encoder"]),
            text_head_path=str(bundle["text_head"]),
            device_pref=str(getattr(args, "device", "auto") or "auto"),
            batch_size=batch_size,
            text_max_length=text_max_length,
            l2_normalize=l2_normalize,
        )
        split_report = _split_probe(
            corpus=corpus,
            split=str(split),
            rng=rng,
            samples_per_split=samples_per_split,
            top_k=top_k,
            examples_to_keep=examples,
        )
        t2i = split_report["metrics"]["text_to_image"]
        i2t = split_report["metrics"]["image_to_text"]
        LOGGER.info(
            "split=%s sampled=%d corpus=%d cache=%s | t2i r1=%.4f r5=%.4f mrr=%.4f | i2t r1=%.4f r5=%.4f mrr=%.4f | probe=%.2fs",
            split_report["split"],
            int(split_report["sampled_n"]),
            int(split_report["corpus_count"]),
            split_report.get("cache_source", ""),
            float(t2i["r1"]),
            float(t2i["r5"]),
            float(t2i["mrr"]),
            float(i2t["r1"]),
            float(i2t["r5"]),
            float(i2t["mrr"]),
            float(split_report["probe_elapsed_s"]),
        )
        results.append(split_report)

    payload = {
        "ok": True,
        "models_dir": str(Path(str(args.models_dir)).expanduser().resolve()),
        "dataset_root": str(dataset_root),
        "splits": splits,
        "bundle": bundle,
        "settings": {
            "samples_per_split": samples_per_split,
            "top_k": top_k,
            "seed": seed,
            "device": str(getattr(args, "device", "auto") or "auto"),
            "batch_size": batch_size,
            "text_max_length": text_max_length,
            "l2_normalize": bool(l2_normalize),
            "examples": examples,
        },
        "results": results,
        "total_elapsed_s": round(float(time.time() - t0_all), 3),
    }
    out_path = str(getattr(args, "json_out", "") or "").strip()
    if out_path:
        p = Path(out_path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Wrote JSON report to %s", p)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = create_parser(add_help=True).parse_args()
    return int(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
