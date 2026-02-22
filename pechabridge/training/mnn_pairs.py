"""Utilities to load patch metadata and MNN-positive maps from parquet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class PatchMeta:
    """Patch metadata row used for pair-aware training."""

    patch_id: int
    doc_id: str
    page_id: str
    line_id: int
    scale_w: int
    k: int
    x0_norm: float
    x1_norm: float
    ink_ratio: float
    boundary_score: float
    image_path: Path

    @property
    def center(self) -> float:
        return float((self.x0_norm + self.x1_norm) * 0.5)


def _sanitize_id(value: Any) -> str:
    txt = str(value or "").strip()
    if not txt:
        return "unknown"
    out: List[str] = []
    for ch in txt:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "unknown"


def _resolve_patch_path(dataset_dir: Path, row: Mapping[str, Any]) -> Optional[Path]:
    for col in ("patch_image_path", "patch_path"):
        raw = str(row.get(col, "") or "").strip()
        if not raw:
            continue
        p = Path(raw).expanduser()
        resolved = p.resolve() if p.is_absolute() else (dataset_dir / p).resolve()
        if resolved.exists() and resolved.is_file():
            return resolved

    try:
        patch_id = int(row.get("patch_id", -1))
        line_id = int(row.get("line_id", -1))
        scale_w = int(row.get("scale_w", -1))
    except Exception:
        return None
    doc = _sanitize_id(row.get("doc_id", ""))
    page = _sanitize_id(row.get("page_id", ""))
    p = (
        dataset_dir
        / "patches"
        / f"doc={doc}"
        / f"page={page}"
        / f"line={line_id}"
        / f"scale={scale_w}"
        / f"patch_{patch_id}.png"
    ).resolve()
    if p.exists() and p.is_file():
        return p
    return None


def load_patch_metadata(
    *,
    dataset_dir: Path,
    patch_meta_parquet: Path,
    ink_ratio_min: float = 0.0,
    boundary_score_min: float = 0.0,
) -> List[PatchMeta]:
    """Load filtered patch metadata rows."""
    meta_path = Path(patch_meta_parquet).expanduser().resolve()
    if not meta_path.exists() or not meta_path.is_file():
        raise FileNotFoundError(f"Patch metadata parquet not found: {meta_path}")

    df = pd.read_parquet(meta_path)
    if df is None or df.empty:
        return []

    out: List[PatchMeta] = []
    for row in df.to_dict(orient="records"):
        try:
            patch_id = int(row.get("patch_id", -1))
            line_id = int(row.get("line_id", -1))
            scale_w = int(row.get("scale_w", -1))
            k = int(row.get("k", -1))
            x0_norm = float(row.get("x0_norm", 0.0))
            x1_norm = float(row.get("x1_norm", 0.0))
            ink_ratio = float(row.get("ink_ratio", 0.0))
            boundary_score = float(row.get("boundary_score", 0.0))
        except Exception:
            continue
        if patch_id < 0 or line_id < 0 or scale_w <= 0:
            continue
        if ink_ratio < float(ink_ratio_min):
            continue
        if boundary_score < float(boundary_score_min):
            continue
        path = _resolve_patch_path(dataset_dir, row)
        if path is None:
            continue
        out.append(
            PatchMeta(
                patch_id=patch_id,
                doc_id=str(row.get("doc_id", "") or ""),
                page_id=str(row.get("page_id", "") or ""),
                line_id=line_id,
                scale_w=scale_w,
                k=k,
                x0_norm=x0_norm,
                x1_norm=x1_norm,
                ink_ratio=ink_ratio,
                boundary_score=boundary_score,
                image_path=path,
            )
        )

    # Dedupe patch_id deterministically.
    by_id: Dict[int, PatchMeta] = {}
    for rec in sorted(out, key=lambda r: r.patch_id):
        by_id[int(rec.patch_id)] = rec
    return list(by_id.values())


def patch_id_to_index_map(records: Sequence[PatchMeta]) -> Dict[int, int]:
    """Build patch_id -> dataset index map."""
    return {int(rec.patch_id): int(i) for i, rec in enumerate(records)}


def _pair_weight(sim: float, stability_ratio: float, scale: float) -> float:
    s = min(max(float(sim), 0.0), 1.0)
    r = min(max(float(stability_ratio), 0.0), 1.0)
    return float(s * r * float(scale))


def load_mnn_map(
    *,
    pairs_parquet: Optional[Path],
    pair_min_sim: float = 0.0,
    pair_min_stability_ratio: float = 0.0,
    require_multi_scale_ok: bool = False,
    weight_scale: float = 1.0,
    max_neighbors_per_anchor: int = 0,
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Load MNN pairs and build a bidirectional weighted map.

    Returns:
      dict[src_patch_id] -> sorted list[(dst_patch_id, weight)] (desc by weight).
    """
    if pairs_parquet is None:
        return {}
    p = Path(pairs_parquet).expanduser().resolve()
    if not p.exists() or not p.is_file():
        return {}
    df = pd.read_parquet(p)
    if df is None or df.empty:
        return {}

    # Keep best weight per directed edge first.
    directed: Dict[Tuple[int, int], float] = {}
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
        if sim < float(pair_min_sim):
            continue
        if stability < float(pair_min_stability_ratio):
            continue
        if bool(require_multi_scale_ok) and (not multi_ok):
            continue
        w = _pair_weight(sim=sim, stability_ratio=stability, scale=float(weight_scale))
        if w <= 0.0:
            continue
        key = (src, dst)
        prev = directed.get(key)
        if prev is None or w > prev:
            directed[key] = w

    # Bidirectional merge.
    out: Dict[int, Dict[int, float]] = {}
    for (src, dst), w in directed.items():
        out.setdefault(int(src), {})
        out.setdefault(int(dst), {})
        old_sd = out[int(src)].get(int(dst))
        if old_sd is None or w > old_sd:
            out[int(src)][int(dst)] = float(w)
        old_ds = out[int(dst)].get(int(src))
        if old_ds is None or w > old_ds:
            out[int(dst)][int(src)] = float(w)

    neighbors_cap = max(0, int(max_neighbors_per_anchor))
    final: Dict[int, List[Tuple[int, float]]] = {}
    for src in sorted(out.keys()):
        items = sorted(out[src].items(), key=lambda kv: kv[1], reverse=True)
        if neighbors_cap > 0:
            items = items[:neighbors_cap]
        final[int(src)] = [(int(dst), float(w)) for dst, w in items]
    return final


def one_d_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    """1D IoU on normalized [x0,x1] segments."""
    ax0 = float(min(a0, a1))
    ax1 = float(max(a0, a1))
    bx0 = float(min(b0, b1))
    bx1 = float(max(b0, b1))
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    union = max(1e-12, (ax1 - ax0) + (bx1 - bx0) - inter)
    return float(inter / union)


__all__ = [
    "PatchMeta",
    "load_patch_metadata",
    "patch_id_to_index_map",
    "load_mnn_map",
    "one_d_iou",
]

