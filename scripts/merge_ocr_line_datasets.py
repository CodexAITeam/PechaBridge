#!/usr/bin/env python3
"""Merge two OCR line datasets (TextHierarchy + split/meta/lines.jsonl) into one dataset root.

Expected input layout (both datasets):
  <dataset_root>/
    train|eval|test/
      TextHierarchy/...
      meta/lines.jsonl
    meta/...

The script rewrites `line_path` to point into the merged output and preserves source
metadata with additional fields:
  - merge_source_dataset_tag
  - merge_source_dataset_root
  - merge_source_line_path

Image files can be copied, hardlinked, or symlinked.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

LOGGER = logging.getLogger("merge_ocr_line_datasets")

_SPLITS = ("train", "eval", "test")

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _safe_jsonl_rows(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _safe_write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _try_write_parquet_from_jsonl(jsonl_path: Path, parquet_path: Path) -> None:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        LOGGER.warning("pandas not available; skipping parquet write for %s", parquet_path)
        return
    rows = list(_safe_jsonl_rows(jsonl_path))
    if not rows:
        return
    df = pd.DataFrame(rows)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        for col in df.columns:
            if str(col).startswith("src__"):
                df[col] = df[col].map(lambda v: None if pd.isna(v) else str(v)).astype("string", copy=False)
        df.to_parquet(parquet_path, index=False)


def _dataset_tag(root: Path, explicit: str) -> str:
    if explicit.strip():
        return explicit.strip()
    return root.name


def _resolve_line_image_path(dataset_root: Path, row: Mapping[str, Any]) -> Optional[Path]:
    val = str(row.get("line_path", "") or "").strip()
    if not val:
        return None
    p = Path(val)
    cand = (dataset_root / p).resolve()
    if cand.exists() and cand.is_file():
        return cand
    # Fallback: some manifests may store split-relative path only.
    if not p.is_absolute():
        cand2 = (dataset_root / str(row.get("split", "")) / p).resolve()
        if cand2.exists() and cand2.is_file():
            return cand2
    return None


def _strip_split_prefix(path_parts: Sequence[str], split_name: str) -> Sequence[str]:
    parts = list(path_parts)
    if parts and parts[0] == split_name:
        return parts[1:]
    return parts


def _copy_like(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "symlink":
        os.symlink(src, dst)
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except Exception:
            shutil.copy2(src, dst)
            return
    shutil.copy2(src, dst)


def _iter_split_roots(dataset_root: Path, split_filter: Optional[set[str]]) -> Iterable[Tuple[str, Path]]:
    for split in _SPLITS:
        if split_filter and split not in split_filter:
            continue
        split_dir = dataset_root / split
        if (split_dir / "meta" / "lines.jsonl").exists():
            yield split, split_dir


def _count_manifest_rows(path: Path) -> int:
    count = 0
    for _row in _safe_jsonl_rows(path):
        count += 1
    return count


def _count_dataset_rows(dataset_root: Path, split_filter: Optional[set[str]]) -> int:
    total = 0
    for _split, split_dir in _iter_split_roots(dataset_root, split_filter):
        total += _count_manifest_rows(split_dir / "meta" / "lines.jsonl")
    return total


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge two OCR line datasets into one TextHierarchy-style dataset.")
    p.add_argument("--dataset-a", required=True, help="First dataset root")
    p.add_argument("--dataset-b", required=True, help="Second dataset root")
    p.add_argument("--output-dir", required=True, help="Merged output dataset root")
    p.add_argument("--tag-a", default="", help="Source tag for dataset A (default: folder name)")
    p.add_argument("--tag-b", default="", help="Source tag for dataset B (default: folder name)")
    p.add_argument("--copy-mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    p.add_argument("--split", dest="splits", action="append", default=[], help="Optional split filter (repeatable)")
    p.add_argument("--skip-parquet", action="store_true")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    p.add_argument("--verbose", action="store_true")
    return p


def _merge_one_dataset(
    *,
    dataset_root: Path,
    source_tag: str,
    out_dir: Path,
    split_filter: Optional[set[str]],
    copy_mode: str,
    totals: Counter,
    per_split: Dict[str, Counter],
    source_columns_merged: Dict[str, Dict[str, Any]],
    split_mapping_merged: Dict[str, str],
    progress: Optional[Any] = None,
) -> None:
    for split, split_dir in _iter_split_roots(dataset_root, split_filter):
        manifest = split_dir / "meta" / "lines.jsonl"
        out_split_root = out_dir / split
        out_meta = out_split_root / "meta"
        out_text = out_split_root / "TextHierarchy"
        out_meta.mkdir(parents=True, exist_ok=True)
        out_text.mkdir(parents=True, exist_ok=True)
        out_jsonl = out_meta / "lines.jsonl"
        append_fp = out_jsonl.open("a", encoding="utf-8")
        try:
            for row_idx, row in enumerate(_safe_jsonl_rows(manifest)):
                totals["rows_seen"] += 1
                per_split[split]["rows_seen"] += 1
                if progress is not None:
                    progress.update(1)
                if not isinstance(row, Mapping):
                    continue
                src_img = _resolve_line_image_path(dataset_root, row)
                if src_img is None:
                    totals["rows_skipped_missing_image"] += 1
                    per_split[split]["rows_skipped_missing_image"] += 1
                    continue

                old_rel = str(row.get("line_path", "") or "")
                old_rel_parts = _strip_split_prefix(Path(old_rel).parts, split)
                if old_rel_parts and old_rel_parts[0] == "TextHierarchy":
                    old_rel_parts = old_rel_parts[1:]
                # Namespace each source to avoid collisions.
                merged_rel = Path(split) / "TextHierarchy" / f"source={source_tag}" / Path(*old_rel_parts)
                dst_img = (out_dir / merged_rel).resolve()
                _copy_like(src_img, dst_img, copy_mode)

                out_row = dict(row)
                out_row["split"] = split
                out_row["canonical_split"] = split
                out_row["line_path"] = str(merged_rel)
                out_row["merge_source_dataset_tag"] = source_tag
                out_row["merge_source_dataset_root"] = str(dataset_root)
                out_row["merge_source_line_path"] = old_rel

                append_fp.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                totals["rows_saved"] += 1
                per_split[split]["rows_saved"] += 1
        finally:
            append_fp.close()

    # Merge root-level metadata helpers if present (namespaced by source tag).
    src_meta = dataset_root / "meta"
    source_cols_path = src_meta / "source_columns.json"
    if source_cols_path.exists():
        try:
            payload = json.loads(source_cols_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                for k, v in payload.items():
                    source_columns_merged[f"{source_tag}::{k}"] = v if isinstance(v, dict) else {"value": v}
        except Exception:
            pass
    split_mapping_path = src_meta / "split_mapping.json"
    if split_mapping_path.exists():
        try:
            payload = json.loads(split_mapping_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                for k, v in payload.items():
                    split_mapping_merged[str(k)] = str(v)
        except Exception:
            pass


def run(args: argparse.Namespace) -> Dict[str, Any]:
    _configure_logging(bool(args.verbose))
    a_root = Path(args.dataset_a).expanduser().resolve()
    b_root = Path(args.dataset_b).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    split_filter = {str(s).strip() for s in (args.splits or []) if str(s).strip()} or None
    tag_a = _dataset_tag(a_root, str(args.tag_a or ""))
    tag_b = _dataset_tag(b_root, str(args.tag_b or ""))
    if tag_a == tag_b:
        raise ValueError(f"Source tags must differ (got both {tag_a!r}). Use --tag-a/--tag-b.")

    totals = Counter()
    per_split: Dict[str, Counter] = defaultdict(Counter)
    source_columns_merged: Dict[str, Dict[str, Any]] = {}
    split_mapping_merged: Dict[str, str] = {}

    progress = None
    if not bool(args.no_progress):
        if tqdm is None:
            LOGGER.warning("tqdm not available; continuing without progress bar.")
        else:
            total_rows = _count_dataset_rows(a_root, split_filter) + _count_dataset_rows(b_root, split_filter)
            progress = tqdm(total=total_rows, desc="Merging OCR lines", unit="rows", dynamic_ncols=True)

    # Fresh output manifests.
    for split in _SPLITS:
        split_jsonl = out_dir / split / "meta" / "lines.jsonl"
        if split_jsonl.exists():
            split_jsonl.unlink()
    root_jsonl = out_dir / "meta" / "lines.jsonl"
    root_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if root_jsonl.exists():
        root_jsonl.unlink()

    try:
        _merge_one_dataset(
            dataset_root=a_root,
            source_tag=tag_a,
            out_dir=out_dir,
            split_filter=split_filter,
            copy_mode=str(args.copy_mode),
            totals=totals,
            per_split=per_split,
            source_columns_merged=source_columns_merged,
            split_mapping_merged=split_mapping_merged,
            progress=progress,
        )
        _merge_one_dataset(
            dataset_root=b_root,
            source_tag=tag_b,
            out_dir=out_dir,
            split_filter=split_filter,
            copy_mode=str(args.copy_mode),
            totals=totals,
            per_split=per_split,
            source_columns_merged=source_columns_merged,
            split_mapping_merged=split_mapping_merged,
            progress=progress,
        )
    finally:
        if progress is not None:
            progress.close()

    # Build root meta/lines.jsonl by concatenating split files.
    with root_jsonl.open("w", encoding="utf-8") as out_fp:
        for split in _SPLITS:
            split_jsonl = out_dir / split / "meta" / "lines.jsonl"
            if not split_jsonl.exists():
                continue
            for line in split_jsonl.open("r", encoding="utf-8"):
                out_fp.write(line)

    if not bool(args.skip_parquet):
        _try_write_parquet_from_jsonl(root_jsonl, out_dir / "meta" / "lines.parquet")
        for split in _SPLITS:
            split_jsonl = out_dir / split / "meta" / "lines.jsonl"
            if split_jsonl.exists():
                _try_write_parquet_from_jsonl(split_jsonl, out_dir / split / "meta" / "lines.parquet")

    summary = {
        "output_dir": str(out_dir),
        "dataset_a": str(a_root),
        "dataset_b": str(b_root),
        "tag_a": tag_a,
        "tag_b": tag_b,
        "copy_mode": str(args.copy_mode),
        "split_filter": sorted(split_filter) if split_filter else [],
        "totals": dict(totals),
        "per_split": {k: dict(v) for k, v in per_split.items()},
    }
    _write_json(out_dir / "meta" / "summary.json", summary)
    _write_json(out_dir / "meta" / "source_columns.json", source_columns_merged)
    _write_json(out_dir / "meta" / "split_mapping.json", split_mapping_merged)
    LOGGER.info("Merged datasets into %s (rows_saved=%d)", out_dir, int(totals.get("rows_saved", 0)))
    return summary


def main() -> int:
    args = create_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
