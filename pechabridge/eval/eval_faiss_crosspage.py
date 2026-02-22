"""FAISS evaluation for cross-page patch retrieval with same-page exclusion."""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import numpy as np
import pandas as pd

from pechabridge.mining.faiss_utils import build_faiss_index, l2_normalize_rows, search_index

LOGGER = logging.getLogger("eval_faiss_crosspage")


@dataclass(frozen=True)
class EvalPatch:
    patch_id: int
    doc_id: str
    page_id: str
    line_id: int
    scale_w: int
    x0_norm: float
    x1_norm: float


def _norm_id(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _parse_ks(raw: str) -> List[int]:
    vals: List[int] = []
    for tok in (raw or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = int(tok)
        except Exception:
            continue
        if v > 0:
            vals.append(v)
    out = sorted(set(vals))
    if not out:
        out = [1, 5, 10]
    return out


def _one_d_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    a_lo, a_hi = float(min(a0, a1)), float(max(a0, a1))
    b_lo, b_hi = float(min(b0, b1)), float(max(b0, b1))
    inter = max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))
    union = max(1e-12, (a_hi - a_lo) + (b_hi - b_lo) - inter)
    return float(inter / union)


def filter_ranked_indices(
    *,
    query_index: int,
    ranked_indices: Sequence[int],
    rows: Sequence[EvalPatch],
    exclude_same_page: bool,
) -> List[int]:
    """Filter ranked FAISS ids by self/same-page exclusion, preserving order."""
    q = rows[int(query_index)]
    out: List[int] = []
    for ridx in ranked_indices:
        j = int(ridx)
        if j < 0 or j >= len(rows):
            continue
        if j == int(query_index):
            continue
        cand = rows[j]
        if bool(exclude_same_page):
            if cand.doc_id == q.doc_id and cand.page_id == q.page_id:
                continue
        out.append(j)
    return out


def _load_rows(meta_path: Path) -> List[EvalPatch]:
    df = pd.read_parquet(meta_path)
    if df is None or df.empty:
        return []
    out: List[EvalPatch] = []
    for row in df.to_dict(orient="records"):
        try:
            out.append(
                EvalPatch(
                    patch_id=int(row.get("patch_id", -1)),
                    doc_id=_norm_id(row.get("doc_id", "")),
                    page_id=_norm_id(row.get("page_id", "")),
                    line_id=int(row.get("line_id", -1)),
                    scale_w=int(row.get("scale_w", -1)),
                    x0_norm=float(row.get("x0_norm", 0.0)),
                    x1_norm=float(row.get("x1_norm", 0.0)),
                )
            )
        except Exception:
            continue
    return [r for r in out if r.patch_id >= 0 and r.line_id >= 0 and r.scale_w > 0]


def _build_pid_to_idx(rows: Sequence[EvalPatch]) -> Dict[int, int]:
    return {int(r.patch_id): int(i) for i, r in enumerate(rows)}


def _load_mnn_map(
    *,
    pairs_path: Optional[Path],
    pid_to_idx: Mapping[int, int],
    sim_min: float,
    stability_min: float,
    require_multi_scale_ok: bool,
) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = {}
    if pairs_path is None:
        return out
    p = Path(pairs_path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        return out
    df = pd.read_parquet(p)
    if df is None or df.empty:
        return out
    for row in df.to_dict(orient="records"):
        try:
            src = int(row.get("src_patch_id", -1))
            dst = int(row.get("dst_patch_id", -1))
            sim = float(row.get("sim", 0.0))
            stability = float(row.get("stability_ratio", 0.0))
            multi_ok = bool(row.get("multi_scale_ok", False))
        except Exception:
            continue
        if src < 0 or dst < 0 or src == dst:
            continue
        if src not in pid_to_idx or dst not in pid_to_idx:
            continue
        if sim < float(sim_min):
            continue
        if stability < float(stability_min):
            continue
        if bool(require_multi_scale_ok) and (not multi_ok):
            continue
        out.setdefault(src, set()).add(dst)
        out.setdefault(dst, set()).add(src)
    return out


def _build_overlap_map(rows: Sequence[EvalPatch], t_iou: float) -> Dict[int, Set[int]]:
    grouped: Dict[tuple[str, str, int, int], List[EvalPatch]] = {}
    for r in rows:
        key = (r.doc_id, r.page_id, int(r.line_id), int(r.scale_w))
        grouped.setdefault(key, []).append(r)

    out: Dict[int, Set[int]] = {}
    for _key, grp in grouped.items():
        if len(grp) <= 1:
            continue
        grp_sorted = sorted(grp, key=lambda x: (x.x0_norm, x.x1_norm, x.patch_id))
        n = len(grp_sorted)
        for i in range(n):
            a = grp_sorted[i]
            for j in range(i + 1, n):
                b = grp_sorted[j]
                iou = _one_d_iou(a.x0_norm, a.x1_norm, b.x0_norm, b.x1_norm)
                if iou >= float(t_iou):
                    out.setdefault(int(a.patch_id), set()).add(int(b.patch_id))
                    out.setdefault(int(b.patch_id), set()).add(int(a.patch_id))
    return out


def _hit_any(top_patch_ids: Sequence[int], positives: Set[int], k: int) -> bool:
    if not positives:
        return False
    kk = max(0, int(k))
    if kk <= 0:
        return False
    for pid in top_patch_ids[:kk]:
        if int(pid) in positives:
            return True
    return False


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate FAISS cross-page retrieval from exported patch embeddings.",
        add_help=add_help,
    )
    parser.add_argument("--embeddings_npy", "--embeddings-npy", dest="embeddings_npy", type=str, required=True,
                       help="Path to exported embeddings .npy")
    parser.add_argument("--embeddings_meta", "--embeddings-meta", dest="embeddings_meta", type=str, required=True,
                       help="Path to embeddings metadata parquet")
    parser.add_argument("--mnn_pairs", "--mnn-pairs", dest="mnn_pairs", type=str, default="",
                       help="Optional mnn_pairs.parquet for MNN Recall@K")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=str, required=True,
                       help="Directory to write evaluation summary")
    parser.add_argument("--recall_ks", "--recall-ks", dest="recall_ks", type=str, default="1,5,10",
                       help="Comma-separated K values (e.g. 1,5,10)")
    parser.add_argument("--max_queries", "--max-queries", dest="max_queries", type=int, default=0,
                       help="Randomly sample at most N queries (0 = all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--search_overfetch", "--search-overfetch", dest="search_overfetch", type=int, default=8,
                       help="Retrieve this factor of max(K) before filtering")
    parser.add_argument("--exclude_same_page", "--exclude-same-page", dest="exclude_same_page", action="store_true",
                       help="Exclude same-page items from cross-page metrics")
    parser.add_argument("--no_exclude_same_page", "--no-exclude-same-page", dest="exclude_same_page", action="store_false",
                       help="Disable same-page exclusion in cross-page metrics")
    parser.set_defaults(exclude_same_page=True)
    parser.add_argument("--overlap_iou", "--overlap-iou", dest="overlap_iou", type=float, default=0.6,
                       help="IoU threshold for overlap positives")
    parser.add_argument("--pair_min_sim", "--pair-min-sim", dest="pair_min_sim", type=float, default=0.0,
                       help="MNN filter: minimum sim")
    parser.add_argument("--pair_min_stability_ratio", "--pair-min-stability-ratio", dest="pair_min_stability_ratio",
                       type=float, default=0.0, help="MNN filter: minimum stability ratio")
    parser.add_argument("--pair_require_multi_scale_ok", "--pair-require-multi-scale-ok", dest="pair_require_multi_scale_ok",
                       action="store_true", help="MNN filter: require multi_scale_ok=true")
    parser.add_argument("--l2_normalize", "--l2-normalize", dest="l2_normalize", action="store_true",
                       help="L2-normalize embeddings before FAISS (default on)")
    parser.add_argument("--no_l2_normalize", "--no-l2-normalize", dest="l2_normalize", action="store_false",
                       help="Disable L2 normalization before FAISS")
    parser.set_defaults(l2_normalize=True)
    parser.add_argument("--faiss_factory", "--faiss-factory", dest="faiss_factory", type=str, default="Flat",
                       help='FAISS factory string (default "Flat")')
    parser.add_argument("--use_gpu", "--use-gpu", dest="use_gpu", action="store_true",
                       help="Use GPU FAISS when available")
    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    emb_path = Path(args.embeddings_npy).expanduser().resolve()
    meta_path = Path(args.embeddings_meta).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not emb_path.exists() or not emb_path.is_file():
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")
    if not meta_path.exists() or not meta_path.is_file():
        raise FileNotFoundError(f"Embeddings metadata parquet not found: {meta_path}")

    rows = _load_rows(meta_path)
    if not rows:
        raise RuntimeError(f"No valid rows in embeddings meta: {meta_path}")

    emb = np.load(str(emb_path))
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim != 2:
        raise RuntimeError(f"Embeddings must be 2D, got shape={emb.shape}")
    if emb.shape[0] != len(rows):
        raise RuntimeError(f"Row count mismatch: embeddings={emb.shape[0]} meta={len(rows)}")
    if bool(args.l2_normalize):
        emb = l2_normalize_rows(emb)

    pid_to_idx = _build_pid_to_idx(rows)
    mnn_pairs_path = Path(args.mnn_pairs).expanduser().resolve() if str(args.mnn_pairs or "").strip() else None
    mnn_map = _load_mnn_map(
        pairs_path=mnn_pairs_path,
        pid_to_idx=pid_to_idx,
        sim_min=float(args.pair_min_sim),
        stability_min=float(args.pair_min_stability_ratio),
        require_multi_scale_ok=bool(args.pair_require_multi_scale_ok),
    )
    overlap_map = _build_overlap_map(rows, t_iou=float(args.overlap_iou))

    ks = _parse_ks(str(args.recall_ks))
    k_max = max(ks)
    n = emb.shape[0]
    overfetch = max(1, int(args.search_overfetch))
    k_search = min(n, max(k_max * overfetch, k_max + 32))

    LOGGER.info(
        "FAISS eval setup: patches=%d dim=%d k_search=%d ks=%s mnn_queries=%d overlap_queries=%d",
        n,
        emb.shape[1],
        k_search,
        ks,
        len([1 for r in rows if int(r.patch_id) in mnn_map]),
        len([1 for r in rows if int(r.patch_id) in overlap_map]),
    )

    index = build_faiss_index(
        embeddings=emb,
        faiss_factory=str(args.faiss_factory),
        metric="ip",
        use_gpu=bool(args.use_gpu),
    )
    dists, ids = search_index(index=index, queries=emb, topk=int(k_search))

    q_idx = list(range(n))
    if int(args.max_queries) > 0 and int(args.max_queries) < len(q_idx):
        rng = random.Random(int(args.seed))
        q_idx = rng.sample(q_idx, int(args.max_queries))
    q_idx = sorted(int(i) for i in q_idx)

    mnn_hits = {k: 0 for k in ks}
    overlap_hits = {k: 0 for k in ks}
    cross_same_doc_hits = {k: 0 for k in ks}
    cross_diff_doc_hits = {k: 0 for k in ks}
    mnn_q = 0
    overlap_q = 0

    for qi in q_idx:
        row_q = rows[int(qi)]
        ranked = [int(v) for v in ids[int(qi)].tolist()]
        ranked_cross = filter_ranked_indices(
            query_index=int(qi),
            ranked_indices=ranked,
            rows=rows,
            exclude_same_page=bool(args.exclude_same_page),
        )
        ranked_open = filter_ranked_indices(
            query_index=int(qi),
            ranked_indices=ranked,
            rows=rows,
            exclude_same_page=False,
        )

        top_cross_pids = [int(rows[j].patch_id) for j in ranked_cross]
        top_open_pids = [int(rows[j].patch_id) for j in ranked_open]

        pos_mnn = set(mnn_map.get(int(row_q.patch_id), set()))
        if pos_mnn:
            mnn_q += 1
            for k in ks:
                if _hit_any(top_cross_pids, pos_mnn, k):
                    mnn_hits[k] += 1

        pos_ov = set(overlap_map.get(int(row_q.patch_id), set()))
        if pos_ov:
            overlap_q += 1
            for k in ks:
                if _hit_any(top_open_pids, pos_ov, k):
                    overlap_hits[k] += 1

        for k in ks:
            local = ranked_cross[: int(k)]
            has_same_doc_diff_page = False
            has_diff_doc = False
            for j in local:
                r = rows[int(j)]
                if r.doc_id == row_q.doc_id and r.page_id != row_q.page_id:
                    has_same_doc_diff_page = True
                if r.doc_id != row_q.doc_id:
                    has_diff_doc = True
            if has_same_doc_diff_page:
                cross_same_doc_hits[k] += 1
            if has_diff_doc:
                cross_diff_doc_hits[k] += 1

    nq = max(1, len(q_idx))
    summary: Dict[str, Any] = {
        "num_patches": int(n),
        "num_queries": int(len(q_idx)),
        "embedding_dim": int(emb.shape[1]),
        "ks": [int(k) for k in ks],
        "exclude_same_page": bool(args.exclude_same_page),
        "mnn_queries_with_positives": int(mnn_q),
        "mnn_recall_at_k": {str(k): float(mnn_hits[k] / max(1, mnn_q)) for k in ks},
        "overlap_queries_with_positives": int(overlap_q),
        "overlap_recall_at_k": {str(k): float(overlap_hits[k] / max(1, overlap_q)) for k in ks},
        "crosspage_same_doc_diff_page_hit_at_k": {str(k): float(cross_same_doc_hits[k] / nq) for k in ks},
        "crosspage_diff_doc_hit_at_k": {str(k): float(cross_diff_doc_hits[k] / nq) for k in ks},
        "embeddings_path": str(emb_path),
        "embeddings_meta_path": str(meta_path),
        "mnn_pairs_path": str(mnn_pairs_path) if mnn_pairs_path else "",
    }

    out_path = (out_dir / "faiss_crosspage_eval_summary.json").resolve()
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved cross-page FAISS summary: %s", out_path)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    parser = create_parser(add_help=True)
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

