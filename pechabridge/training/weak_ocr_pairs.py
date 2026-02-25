"""Build weak-positive adjacency maps from weak OCR parquet outputs."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


_WS_RE = re.compile(r"\s+")


def normalize_ocr_text(text: str) -> str:
    """Normalize OCR text for weak positive grouping (lightweight, language-agnostic)."""
    t = str(text or "").strip()
    if not t:
        return ""
    return _WS_RE.sub(" ", t)


def _clamp01(x: float) -> float:
    if not math.isfinite(float(x)):
        return 0.0
    return float(max(0.0, min(1.0, float(x))))


def _pair_weight(conf_a: float, conf_b: float, weight_scale: float) -> float:
    # Conservative: geometric mean confidence, clamped to [0,1], then scaled.
    c = math.sqrt(_clamp01(conf_a) * _clamp01(conf_b))
    return float(c * float(weight_scale))


def load_ocr_weak_map(
    *,
    weak_ocr_parquet: Optional[Path],
    pair_min_confidence: float = 0.2,
    min_chars: int = 2,
    max_group_size: int = 128,
    max_neighbors_per_anchor: int = 0,
    weight_scale: float = 1.0,
    require_no_error: bool = True,
    patch_id_allowlist: Optional[Set[int]] = None,
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Load weak OCR labels and build bidirectional positive map by exact normalized text.

    Each normalized OCR string defines a weak cluster. Edges are weighted by OCR confidence.
    Large clusters can be skipped via `max_group_size` to avoid very common/noisy strings.
    """
    if weak_ocr_parquet is None:
        return {}
    p = Path(weak_ocr_parquet).expanduser().resolve()
    if not p.exists() or not p.is_file():
        return {}

    df = pd.read_parquet(p)
    if df is None or df.empty:
        return {}

    required_cols = {"patch_id", "text", "confidence"}
    if not required_cols.issubset(set(df.columns)):
        return {}

    allow = patch_id_allowlist if patch_id_allowlist else None
    rows: List[Tuple[int, str, float]] = []
    minc = float(pair_min_confidence)
    min_chars_int = max(0, int(min_chars))

    for row in df.to_dict(orient="records"):
        try:
            patch_id = int(row.get("patch_id", -1))
            conf = float(row.get("confidence", 0.0))
        except Exception:
            continue
        if patch_id < 0:
            continue
        if allow is not None and patch_id not in allow:
            continue
        if _clamp01(conf) < minc:
            continue
        if bool(require_no_error):
            err = str(row.get("error_code", "") or "").strip()
            if err:
                continue
        text_norm = normalize_ocr_text(str(row.get("text", "") or ""))
        if not text_norm:
            continue
        n_chars = int(row.get("char_count", 0) or 0)
        if n_chars <= 0:
            n_chars = len(text_norm)
        if n_chars < min_chars_int:
            continue
        rows.append((patch_id, text_norm, _clamp01(conf)))

    if not rows:
        return {}

    # Stable dedupe by patch_id (keep highest-confidence row).
    by_patch: Dict[int, Tuple[str, float]] = {}
    for patch_id, txt, conf in rows:
        prev = by_patch.get(int(patch_id))
        if prev is None or float(conf) > float(prev[1]):
            by_patch[int(patch_id)] = (str(txt), float(conf))

    groups: Dict[str, List[Tuple[int, float]]] = {}
    for patch_id, (txt, conf) in by_patch.items():
        groups.setdefault(str(txt), []).append((int(patch_id), float(conf)))

    out: Dict[int, Dict[int, float]] = {}
    max_group = max(0, int(max_group_size))
    for txt in sorted(groups.keys()):
        grp = groups[txt]
        if len(grp) < 2:
            continue
        if max_group > 0 and len(grp) > max_group:
            continue
        grp_sorted = sorted(grp, key=lambda x: (x[0], -x[1]))
        n = len(grp_sorted)
        for i in range(n):
            pid_i, conf_i = grp_sorted[i]
            for j in range(i + 1, n):
                pid_j, conf_j = grp_sorted[j]
                if pid_i == pid_j:
                    continue
                w = _pair_weight(conf_i, conf_j, weight_scale=float(weight_scale))
                if w <= 0.0:
                    continue
                out.setdefault(int(pid_i), {})
                out.setdefault(int(pid_j), {})
                prev_ij = out[int(pid_i)].get(int(pid_j))
                if prev_ij is None or w > prev_ij:
                    out[int(pid_i)][int(pid_j)] = float(w)
                prev_ji = out[int(pid_j)].get(int(pid_i))
                if prev_ji is None or w > prev_ji:
                    out[int(pid_j)][int(pid_i)] = float(w)

    neighbors_cap = max(0, int(max_neighbors_per_anchor))
    final: Dict[int, List[Tuple[int, float]]] = {}
    for src in sorted(out.keys()):
        items = sorted(out[src].items(), key=lambda kv: kv[1], reverse=True)
        if neighbors_cap > 0:
            items = items[:neighbors_cap]
        final[int(src)] = [(int(dst), float(w)) for dst, w in items]
    return final


def merge_positive_maps(*maps: Mapping[int, Sequence[Tuple[int, float]]]) -> Dict[int, List[Tuple[int, float]]]:
    """
    Merge multiple weighted adjacency maps (max weight per edge).

    Used by pair-aware sampler so in-batch positives can come from MNN and/or OCR.
    """
    merged: Dict[int, Dict[int, float]] = {}
    for mp in maps:
        if not mp:
            continue
        for src, neigh in mp.items():
            s = int(src)
            if not neigh:
                continue
            bucket = merged.setdefault(s, {})
            for dst, w in neigh:
                d = int(dst)
                if s == d:
                    continue
                ww = float(w)
                prev = bucket.get(d)
                if prev is None or ww > prev:
                    bucket[d] = ww
    out: Dict[int, List[Tuple[int, float]]] = {}
    for src in sorted(merged.keys()):
        out[src] = [(int(dst), float(w)) for dst, w in sorted(merged[src].items(), key=lambda kv: kv[1], reverse=True)]
    return out


__all__ = ["normalize_ocr_text", "load_ocr_weak_map", "merge_positive_maps"]

