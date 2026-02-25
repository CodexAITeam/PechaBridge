"""I/O helpers for patch dataset generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from PIL import Image


def sanitize_id(value: str) -> str:
    """Sanitize id for stable filesystem paths."""
    txt = str(value or "").strip()
    if not txt:
        return "unknown"
    out = []
    for ch in txt:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "unknown"


def derive_doc_id(image_path: Path, input_root: Path) -> str:
    """Derive document id from image relative path."""
    try:
        rel = image_path.relative_to(input_root)
    except Exception:
        return "doc"
    if len(rel.parts) >= 2:
        return str(rel.parts[0])
    return "doc"


def derive_page_id(image_path: Path, input_root: Path) -> str:
    """Derive page id from image relative path (without suffix)."""
    try:
        rel = image_path.relative_to(input_root)
    except Exception:
        rel = image_path
    parts = list(rel.with_suffix("").parts)
    if len(parts) >= 2:
        parts = parts[1:]
    return "__".join(parts) if parts else image_path.stem


def make_patch_path(
    out_dataset_dir: Path,
    *,
    doc_id: str,
    page_id: str,
    line_id: int,
    scale_w: int,
    patch_id: int,
) -> Path:
    """Build patch image path under the requested hierarchy."""
    d = sanitize_id(doc_id)
    p = sanitize_id(page_id)
    return (
        out_dataset_dir
        / "patches"
        / f"doc={d}"
        / f"page={p}"
        / f"line={int(line_id)}"
        / f"scale={int(scale_w)}"
        / f"patch_{int(patch_id)}.png"
    )


def save_patch_png(rgb: np.ndarray, out_path: Path) -> None:
    """Persist an RGB patch to PNG."""
    arr = np.asarray(rgb)
    if arr.size == 0:
        raise ValueError("Cannot save empty patch.")
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = arr.astype(np.uint8, copy=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out_path)


def _typed_dataframe(records: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    rows = list(records)
    if not rows:
        return pd.DataFrame(
            {
                "patch_id": pd.Series([], dtype="int64"),
                "doc_id": pd.Series([], dtype="string"),
                "page_id": pd.Series([], dtype="string"),
                "line_id": pd.Series([], dtype="int32"),
                "scale_w": pd.Series([], dtype="int16"),
                "k": pd.Series([], dtype="int32"),
                "x0_norm": pd.Series([], dtype="float32"),
                "x1_norm": pd.Series([], dtype="float32"),
                "line_h_px": pd.Series([], dtype="int16"),
                "line_w_px": pd.Series([], dtype="int32"),
                "boundary_score": pd.Series([], dtype="float32"),
                "ink_ratio": pd.Series([], dtype="float32"),
                "source_img_path": pd.Series([], dtype="string"),
                "line_x0": pd.Series([], dtype="int32"),
                "line_y0": pd.Series([], dtype="int32"),
                "line_x1": pd.Series([], dtype="int32"),
                "line_y1": pd.Series([], dtype="int32"),
                "x0_px": pd.Series([], dtype="int32"),
                "x1_px": pd.Series([], dtype="int32"),
                "tag": pd.Series([], dtype="string"),
                "patch_path": pd.Series([], dtype="string"),
            }
        )

    df = pd.DataFrame(rows)
    for col, dtype in [
        ("patch_id", "int64"),
        ("line_id", "int32"),
        ("scale_w", "int16"),
        ("k", "int32"),
        ("x0_norm", "float32"),
        ("x1_norm", "float32"),
        ("line_h_px", "int16"),
        ("line_w_px", "int32"),
        ("boundary_score", "float32"),
        ("ink_ratio", "float32"),
        ("line_x0", "int32"),
        ("line_y0", "int32"),
        ("line_x1", "int32"),
        ("line_y1", "int32"),
        ("x0_px", "int32"),
        ("x1_px", "int32"),
    ]:
        if col in df.columns:
            df[col] = df[col].astype(dtype, copy=False)

    for col in ["doc_id", "page_id", "source_img_path", "tag", "patch_path"]:
        if col in df.columns:
            df[col] = df[col].astype("string", copy=False)
    return df


def write_metadata_parquet(records: Iterable[Dict[str, Any]], out_dataset_dir: Path) -> Path:
    """Write metadata to out_dataset/meta/patches.parquet."""
    out_meta = (out_dataset_dir / "meta" / "patches.parquet").resolve()
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    df = _typed_dataframe(records)
    try:
        df.to_parquet(out_meta, index=False)
    except Exception as exc:
        raise RuntimeError(
            "Failed to write parquet metadata. Install pyarrow or fastparquet."
        ) from exc
    return out_meta


def summarize_metadata(records: List[Dict[str, Any]]) -> Dict[str, int]:
    """Compact summary stats for logs."""
    if not records:
        return {"patches": 0, "docs": 0, "pages": 0, "lines": 0}
    docs = {str(r.get("doc_id", "")) for r in records}
    pages = {(str(r.get("doc_id", "")), str(r.get("page_id", ""))) for r in records}
    lines = {
        (str(r.get("doc_id", "")), str(r.get("page_id", "")), int(r.get("line_id", -1)))
        for r in records
    }
    return {
        "patches": int(len(records)),
        "docs": int(len(docs)),
        "pages": int(len(pages)),
        "lines": int(len(lines)),
    }

