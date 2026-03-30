#!/usr/bin/env python3
"""Probe line_clip Workbench corpus retrieval on random samples across splits."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ui_workbench import (
    _cer_single_ui,
    _get_or_build_line_clip_debug_corpus_ui,
    _line_clip_debug_cache_key,
    _line_clip_debug_disk_cache_dir_ui,
    _load_line_clip_debug_corpus_from_disk_ui,
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
    p.add_argument(
        "--cross-split",
        type=str,
        default="",
        help="Comma-separated cross probes as query:index (or query->index), e.g. eval:test,val:test",
    )
    p.add_argument(
        "--include-in-split",
        action="store_true",
        help="When --cross-split is set, also run in-split probes from --splits (default: cross-split only if --splits left at default)",
    )
    p.add_argument("--samples-per-split", type=int, default=25, help="Random samples per split")
    p.add_argument("--top-k", type=int, default=5, help="Top-k neighbors to include in per-sample details")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    p.add_argument("--device", type=str, default="auto", help="Device used only if cache must be built (auto/cuda/cuda:0/cpu/mps)")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size used only if cache must be built")
    p.add_argument("--text-max-length", type=int, default=256, help="Tokenizer max length used in cache key/build")
    p.add_argument("--no-l2-normalize", action="store_true", help="Use non-normalized embeddings setting (must match cache/build config)")
    p.add_argument(
        "--progress-every-batches",
        type=int,
        default=50,
        help="If cache build is needed, log embedding progress every N batches (0 disables)",
    )
    p.add_argument("--examples", type=int, default=3, help="How many random sample details to keep per split in JSON report")
    p.add_argument("--json-out", type=str, default="", help="Optional path to write full JSON report")
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only compact per-split summary lines to stdout (full report still written with --json-out)",
    )
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


def _parse_cross_split_pairs(raw: str, available: Sequence[str]) -> List[Tuple[str, str]]:
    avail = set(str(s) for s in available)
    out: List[Tuple[str, str]] = []
    seen = set()
    for tok in [x.strip().lower() for x in str(raw or "").split(",") if x.strip()]:
        if "->" in tok:
            parts = tok.split("->", 1)
        elif ":" in tok:
            parts = tok.split(":", 1)
        else:
            continue
        q = str(parts[0]).strip()
        i = str(parts[1]).strip()
        if not q or not i:
            continue
        if q not in avail or i not in avail:
            continue
        key = (q, i)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _available_splits(dataset_root: Path) -> List[str]:
    out: List[str] = []
    for split in _SPLIT_ORDER:
        if (dataset_root / split / "meta" / "lines.jsonl").exists():
            out.append(split)
    return out


def _split_sort_key(split: str) -> int:
    try:
        return int(_SPLIT_ORDER.index(str(split)))
    except Exception:
        return 999


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


def _norm_text_for_match(text: str) -> str:
    return str(text or "").strip()


def _build_text_to_indices(rows: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for idx, row in enumerate(rows):
        k = _norm_text_for_match(str(row.get("_text_resolved", "") or ""))
        if not k:
            continue
        out.setdefault(k, []).append(int(idx))
    return out


def _rank_of_best_positive(scores: np.ndarray, positives: Sequence[int]) -> Optional[int]:
    pos = [int(i) for i in positives if 0 <= int(i) < int(scores.shape[0])]
    if not pos:
        return None
    pos_scores = scores[np.asarray(pos, dtype=np.int64)]
    best = float(np.max(pos_scores))
    return int(np.sum(scores > best)) + 1


def _stats_from_ranks(ranks: List[int], hits1: int, hits5: int) -> Dict[str, Any]:
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
            "text_to_image": _stats_from_ranks(t2i_ranks, t2i_r1, t2i_r5),
            "image_to_text": _stats_from_ranks(i2t_ranks, i2t_r1, i2t_r5),
        },
        "examples": per_sample_details,
    }


def _load_corpus_for_split(
    *,
    split: str,
    dataset_root: Path,
    bundle: Dict[str, Any],
    device: str,
    batch_size: int,
    text_max_length: int,
    l2_normalize: bool,
    progress_every_batches: int,
) -> Dict[str, Any]:
    key = _line_clip_debug_cache_key(
        dataset_root=str(dataset_root),
        split_name=str(split),
        image_backbone_path=str(bundle["image_backbone"]),
        image_head_path=str(bundle["image_head"]),
        text_encoder_dir=str(bundle["text_encoder"]),
        text_head_path=str(bundle["text_head"]),
        text_max_length=text_max_length,
        l2_normalize=l2_normalize,
    )
    cache_dir = _line_clip_debug_disk_cache_dir_ui(key)
    LOGGER.info("split=%s checking corpus cache at: %s", split, str(cache_dir))
    disk_cached = _load_line_clip_debug_corpus_from_disk_ui(key)
    if disk_cached is None:
        LOGGER.info("split=%s cache MISS -> building corpus banks: image+text", split)
    else:
        has_image = bool(disk_cached.get("image_embeddings") is not None)
        has_text = bool(disk_cached.get("text_embeddings") is not None)
        LOGGER.info(
            "split=%s cache HIT on disk (image=%s text=%s) at %s",
            split,
            has_image,
            has_text,
            str(disk_cached.get("disk_cache_dir", "")),
        )
        if not (has_image and has_text):
            LOGGER.info("split=%s cache is partial -> building missing bank(s) to satisfy probe", split)
    LOGGER.info("Loading/building corpus cache for split=%s ...", split)
    return _get_or_build_line_clip_debug_corpus_ui(
        dataset_root=str(dataset_root),
        split_name=str(split),
        image_backbone_path=str(bundle["image_backbone"]),
        image_head_path=str(bundle["image_head"]),
        text_encoder_dir=str(bundle["text_encoder"]),
        text_head_path=str(bundle["text_head"]),
        device_pref=str(device or "auto"),
        batch_size=int(batch_size),
        text_max_length=int(text_max_length),
        l2_normalize=bool(l2_normalize),
        required_banks="both",
        progress_every_batches=int(progress_every_batches),
    )


def _cross_split_probe(
    *,
    query_corpus: Dict[str, Any],
    index_corpus: Dict[str, Any],
    query_split: str,
    index_split: str,
    rng: np.random.Generator,
    samples_per_split: int,
    top_k: int,
    examples_to_keep: int,
) -> Dict[str, Any]:
    q_rows = list(query_corpus.get("rows") or [])
    i_rows = list(index_corpus.get("rows") or [])
    q_img = np.asarray(query_corpus.get("image_embeddings"), dtype=np.float32)
    q_txt = np.asarray(query_corpus.get("text_embeddings"), dtype=np.float32)
    i_img = np.asarray(index_corpus.get("image_embeddings"), dtype=np.float32)
    i_txt = np.asarray(index_corpus.get("text_embeddings"), dtype=np.float32)
    nq = int(len(q_rows))
    ni = int(len(i_rows))
    if nq <= 0 or ni <= 0:
        raise RuntimeError(f"Invalid cross corpus sizes for {query_split}->{index_split}: nq={nq} ni={ni}")
    if q_img.ndim != 2 or q_txt.ndim != 2 or i_img.ndim != 2 or i_txt.ndim != 2:
        raise RuntimeError("Invalid embedding rank (expected 2D matrices for query/index image/text)")
    if q_img.shape[0] != nq or q_txt.shape[0] != nq or i_img.shape[0] != ni or i_txt.shape[0] != ni:
        raise RuntimeError(
            f"Embedding/row mismatch for {query_split}->{index_split}: "
            f"q_img={q_img.shape} q_txt={q_txt.shape} i_img={i_img.shape} i_txt={i_txt.shape}"
        )

    m = min(max(1, int(samples_per_split)), nq)
    sample_idx = rng.choice(nq, size=m, replace=False).astype(int).tolist()
    top_k = min(max(1, int(top_k)), ni)
    idx_by_text = _build_text_to_indices(i_rows)

    t2i_ranks: List[int] = []
    i2t_ranks: List[int] = []
    t2i_r1 = t2i_r5 = 0
    i2t_r1 = i2t_r5 = 0
    t2i_eval_n = 0
    i2t_eval_n = 0
    examples: List[Dict[str, Any]] = []
    t0 = time.time()

    for qidx in sample_idx:
        qrow = q_rows[qidx]
        qtext = str(qrow.get("_text_resolved", "") or "")
        qtext_key = _norm_text_for_match(qtext)
        positives = idx_by_text.get(qtext_key, [])
        positives_set = set(int(x) for x in positives)

        sims_t2i = i_img @ q_txt[qidx]
        sims_i2t = i_txt @ q_img[qidx]
        top_t2i = _topk_indices(sims_t2i, top_k)
        top_i2t = _topk_indices(sims_i2t, top_k)

        rank_t2i = _rank_of_best_positive(sims_t2i, positives)
        rank_i2t = _rank_of_best_positive(sims_i2t, positives)
        if rank_t2i is not None:
            t2i_eval_n += 1
            t2i_ranks.append(int(rank_t2i))
            t2i_r1 += int(rank_t2i <= 1)
            t2i_r5 += int(rank_t2i <= 5)
        if rank_i2t is not None:
            i2t_eval_n += 1
            i2t_ranks.append(int(rank_i2t))
            i2t_r1 += int(rank_i2t <= 1)
            i2t_r5 += int(rank_i2t <= 5)

        if len(examples) < max(0, int(examples_to_keep)):
            t2i_hits: List[Dict[str, Any]] = []
            for rank, j in enumerate(top_t2i.tolist(), start=1):
                r = i_rows[int(j)]
                pred_text = str(r.get("_text_resolved", "") or "")
                t2i_hits.append(
                    {
                        "rank": int(rank),
                        "idx": int(j),
                        "score": round(float(sims_t2i[j]), 6),
                        "is_positive_text_match": bool(int(j) in positives_set),
                        "doc_id": str(r.get("doc_id", "")),
                        "page_id": str(r.get("page_id", "")),
                        "line_id": r.get("line_id"),
                        "image_path": str(r.get("_image_path_resolved", "")),
                        "text_preview": pred_text[:180],
                        "cer_vs_query_text": round(float(_cer_single_ui(pred_text, qtext)), 6),
                        "sem_text_vs_query_text": round(_dot(i_txt[j], q_txt[qidx]), 6),
                        "sem_image_vs_query_image": round(_dot(i_img[j], q_img[qidx]), 6),
                    }
                )
            i2t_hits: List[Dict[str, Any]] = []
            for rank, j in enumerate(top_i2t.tolist(), start=1):
                r = i_rows[int(j)]
                pred_text = str(r.get("_text_resolved", "") or "")
                i2t_hits.append(
                    {
                        "rank": int(rank),
                        "idx": int(j),
                        "score": round(float(sims_i2t[j]), 6),
                        "is_positive_text_match": bool(int(j) in positives_set),
                        "doc_id": str(r.get("doc_id", "")),
                        "page_id": str(r.get("page_id", "")),
                        "line_id": r.get("line_id"),
                        "text_preview": pred_text[:180],
                        "cer_vs_query_text": round(float(_cer_single_ui(pred_text, qtext)), 6),
                        "sem_text_vs_query_text": round(_dot(i_txt[j], q_txt[qidx]), 6),
                        "sem_image_vs_query_image": round(_dot(i_img[j], q_img[qidx]), 6),
                    }
                )
            examples.append(
                {
                    "query_idx": int(qidx),
                    "query_doc_id": str(qrow.get("doc_id", "")),
                    "query_page_id": str(qrow.get("page_id", "")),
                    "query_line_id": qrow.get("line_id"),
                    "query_image_path": str(qrow.get("_image_path_resolved", "")),
                    "query_text_preview": qtext[:220],
                    "positive_text_match_count_in_index": int(len(positives)),
                    "t2i": {"best_positive_rank": rank_t2i, "topk": t2i_hits},
                    "i2t": {"best_positive_rank": rank_i2t, "topk": i2t_hits},
                }
            )

    elapsed_s = max(1e-9, float(time.time() - t0))
    t2i_stats = _stats_from_ranks(t2i_ranks, t2i_r1, t2i_r5)
    i2t_stats = _stats_from_ranks(i2t_ranks, i2t_r1, i2t_r5)

    return {
        "kind": "cross_split",
        "name": f"{query_split}->{index_split}",
        "query_split": str(query_split),
        "index_split": str(index_split),
        "query_count": int(nq),
        "index_count": int(ni),
        "sampled_n": int(m),
        "evaluable_n": {
            "text_to_image": int(t2i_eval_n),
            "image_to_text": int(i2t_eval_n),
        },
        "probe_elapsed_s": round(float(elapsed_s), 4),
        "probe_samples_per_s": round(float(m / elapsed_s), 2),
        "query_cache": {
            "source": str(query_corpus.get("cache_source", "")),
            "disk_cache_dir": str(query_corpus.get("disk_cache_dir", "")),
        },
        "index_cache": {
            "source": str(index_corpus.get("cache_source", "")),
            "disk_cache_dir": str(index_corpus.get("disk_cache_dir", "")),
        },
        "metrics": {
            "text_to_image": t2i_stats,
            "image_to_text": i2t_stats,
        },
        "examples": examples,
    }


def run(args: argparse.Namespace) -> int:
    dataset_root = Path(str(args.dataset_root or "").strip()).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    avail_splits = _available_splits(dataset_root)
    raw_splits = str(args.splits or "").strip()
    req_splits = _parse_splits(raw_splits)
    default_splits_token = "train,val,test"
    cross_pairs = _parse_cross_split_pairs(str(getattr(args, "cross_split", "") or ""), avail_splits)
    in_splits = [s for s in req_splits if s in avail_splits] if req_splits else []
    if cross_pairs and (not bool(getattr(args, "include_in_split", False))) and (raw_splits in {"", default_splits_token}):
        LOGGER.info("cross-split mode enabled; skipping default in-split probes (use --include-in-split to run both)")
        in_splits = []
    if not in_splits and not cross_pairs:
        raise RuntimeError(
            f"No probes selected. available_splits={avail_splits}. "
            "Set --splits and/or --cross-split (e.g. --cross-split eval:test)."
        )

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
    progress_every_batches = max(0, int(getattr(args, "progress_every_batches", 0) or 0))
    device = str(getattr(args, "device", "auto") or "auto")

    rng = np.random.default_rng(seed)
    results: List[Dict[str, Any]] = []
    t0_all = time.time()
    needed_splits = set(in_splits)
    for q, i in cross_pairs:
        needed_splits.add(str(q))
        needed_splits.add(str(i))
    corpora: Dict[str, Dict[str, Any]] = {}
    for split in sorted(list(needed_splits), key=_split_sort_key):
        corpora[str(split)] = _load_corpus_for_split(
            split=str(split),
            dataset_root=dataset_root,
            bundle=bundle,
            device=device,
            batch_size=batch_size,
            text_max_length=text_max_length,
            l2_normalize=l2_normalize,
            progress_every_batches=progress_every_batches,
        )

    for split in in_splits:
        corpus = corpora[str(split)]
        split_report = _split_probe(
            corpus=corpus,
            split=str(split),
            rng=rng,
            samples_per_split=samples_per_split,
            top_k=top_k,
            examples_to_keep=examples,
        )
        split_report["kind"] = "in_split"
        split_report["name"] = f"{split}->{split}"
        split_report["query_split"] = str(split)
        split_report["index_split"] = str(split)
        t2i = split_report["metrics"]["text_to_image"]
        i2t = split_report["metrics"]["image_to_text"]
        LOGGER.info(
            "in-split=%s sampled=%d corpus=%d cache=%s | t2i r1=%.4f r5=%.4f mrr=%.4f | i2t r1=%.4f r5=%.4f mrr=%.4f | probe=%.2fs",
            split_report["name"],
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

    for qsplit, isplit in cross_pairs:
        cross_report = _cross_split_probe(
            query_corpus=corpora[str(qsplit)],
            index_corpus=corpora[str(isplit)],
            query_split=str(qsplit),
            index_split=str(isplit),
            rng=rng,
            samples_per_split=samples_per_split,
            top_k=top_k,
            examples_to_keep=examples,
        )
        t2i = cross_report["metrics"]["text_to_image"]
        i2t = cross_report["metrics"]["image_to_text"]
        LOGGER.info(
            "cross=%s sampled=%d eval_n(t2i/i2t)=%d/%d | t2i r1=%.4f r5=%.4f mrr=%.4f | i2t r1=%.4f r5=%.4f mrr=%.4f | probe=%.2fs",
            cross_report["name"],
            int(cross_report["sampled_n"]),
            int((cross_report.get("evaluable_n") or {}).get("text_to_image", 0)),
            int((cross_report.get("evaluable_n") or {}).get("image_to_text", 0)),
            float(t2i["r1"]),
            float(t2i["r5"]),
            float(t2i["mrr"]),
            float(i2t["r1"]),
            float(i2t["r5"]),
            float(i2t["mrr"]),
            float(cross_report["probe_elapsed_s"]),
        )
        results.append(cross_report)

    payload = {
        "ok": True,
        "models_dir": str(Path(str(args.models_dir)).expanduser().resolve()),
        "dataset_root": str(dataset_root),
        "in_splits": in_splits,
        "cross_split_pairs": [{"query_split": q, "index_split": i} for (q, i) in cross_pairs],
        "available_splits": avail_splits,
        "bundle": bundle,
        "settings": {
            "samples_per_split": samples_per_split,
            "top_k": top_k,
            "seed": seed,
            "device": device,
            "batch_size": batch_size,
            "text_max_length": text_max_length,
            "l2_normalize": bool(l2_normalize),
            "examples": examples,
            "summary_only": bool(getattr(args, "summary_only", False)),
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
    if bool(getattr(args, "summary_only", False)):
        for r in results:
            t2i = (r.get("metrics") or {}).get("text_to_image", {})
            i2t = (r.get("metrics") or {}).get("image_to_text", {})
            print(
                "probe={name} kind={kind} sampled={sampled} "
                "t2i@1={t2i_r1:.4f} t2i@5={t2i_r5:.4f} t2i_mrr={t2i_mrr:.4f} "
                "i2t@1={i2t_r1:.4f} i2t@5={i2t_r5:.4f} i2t_mrr={i2t_mrr:.4f}".format(
                    name=str(r.get("name", r.get("split", ""))),
                    kind=str(r.get("kind", "in_split")),
                    sampled=int(r.get("sampled_n", 0) or 0),
                    t2i_r1=float(t2i.get("r1", 0.0) or 0.0),
                    t2i_r5=float(t2i.get("r5", 0.0) or 0.0),
                    t2i_mrr=float(t2i.get("mrr", 0.0) or 0.0),
                    i2t_r1=float(i2t.get("r1", 0.0) or 0.0),
                    i2t_r5=float(i2t.get("r5", 0.0) or 0.0),
                    i2t_mrr=float(i2t.get("mrr", 0.0) or 0.0),
                )
            )
    else:
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
