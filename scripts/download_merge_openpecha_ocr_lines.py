#!/usr/bin/env python3
"""Download and merge OpenPecha OCR Hugging Face datasets into a line dataset.

Output format is TextHierarchy-like (line folders with `line.png`) plus merged metadata.
Known HF split names are normalized into `train`, `test`, `eval` output folders.

out_dir/
  train|test|eval/
    TextHierarchy/
      dataset=<dataset>/doc=<doc_id>/page=<page_id>/line_<id>/line.png
    meta/
      lines.jsonl
      lines.parquet
  meta/
    lines.jsonl        # all splits combined
    lines.parquet      # all splits combined
    summary.json
    source_columns.json
    split_mapping.json

The script preserves source columns under the `src__*` prefix. Nested/list values are
serialized to JSON strings so they can be merged across datasets and written to parquet.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image, ImageOps
from tqdm.auto import tqdm

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

LOGGER = logging.getLogger("download_merge_openpecha_ocr_lines")

DEFAULT_DATASETS: List[str] = [
    "openpecha/OCR-Lhasakanjur",
    "openpecha/OCR-Drutsa",
    "openpecha/OCR-Betsug",
    "openpecha/OCR-Google_Books",
    "openpecha/OCR-Dergetenjur",
    "openpecha/OCR-Norbuketaka",
    "openpecha/OCR_Uchan",
]

IMAGE_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "image",
    "img",
    "line_image",
    "line_img",
    "crop",
    "line",
)
TEXT_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "text",
    "ocr_text",
    "line_text",
    "transcription",
    "label",
    "gt",
    "ground_truth",
    "content",
)
DOC_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "doc_id",
    "document_id",
    "pecha_id",
    "work_id",
    "text_id",
    "volume_id",
    "volume",
)
PAGE_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "page_id",
    "page",
    "page_no",
    "page_num",
    "image_id",
    "img_id",
    "folio",
    "filename",
    "file_name",
    "image_path",
    "img_path",
)
LINE_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "line_id",
    "line_no",
    "line_number",
    "row_id",
    "row_no",
    "row_number",
    "idx",
    "index",
)

_TRAIN_SPLIT_ALIASES = {"train", "training"}
_TEST_SPLIT_ALIASES = {"test", "testing"}
_EVAL_SPLIT_ALIASES = {"eval", "evaluation", "val", "valid", "validation", "dev", "development"}


def _configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


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


def _dataset_short_name(dataset_id: str) -> str:
    return _sanitize_id(dataset_id.split("/")[-1] if "/" in dataset_id else dataset_id)


def _canonical_split_name(split_name: str) -> str:
    raw = str(split_name or "").strip()
    norm = raw.lower()
    if norm in _TRAIN_SPLIT_ALIASES:
        return "train"
    if norm in _TEST_SPLIT_ALIASES:
        return "test"
    if norm in _EVAL_SPLIT_ALIASES:
        return "eval"
    return _sanitize_id(raw or "train")


def _is_hf_image_feature(feature: Any) -> bool:
    # Avoid hard-coupling to a specific imported class path/version.
    return feature is not None and feature.__class__.__name__ == "Image"


def _detect_image_column(
    column_names: Sequence[str],
    features: Optional[Mapping[str, Any]],
    explicit: Optional[str] = None,
) -> Optional[str]:
    if explicit:
        return explicit if explicit in set(column_names) else None
    if features:
        for name, feat in features.items():
            if name in column_names and _is_hf_image_feature(feat):
                return name
    lower_map = {str(c).lower(): str(c) for c in column_names}
    for cand in IMAGE_COLUMN_CANDIDATES:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _detect_text_column(column_names: Sequence[str], explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit if explicit in set(column_names) else None
    lower_map = {str(c).lower(): str(c) for c in column_names}
    for cand in TEXT_COLUMN_CANDIDATES:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    # Fallback: any column containing "text"
    for col in column_names:
        if "text" in str(col).lower():
            return str(col)
    return None


def _pick_column(
    row: Mapping[str, Any],
    *,
    explicit: Optional[str],
    candidates: Sequence[str],
) -> Optional[str]:
    if explicit and explicit in row:
        return explicit
    lower_lookup = {str(k).lower(): str(k) for k in row.keys()}
    for cand in candidates:
        key = lower_lookup.get(cand.lower())
        if key is not None:
            return key
    return None


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _normalize_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, bytes):
        return {"_type": "bytes", "length": len(value)}
    if isinstance(value, Image.Image):
        return {"_type": "PIL.Image", "mode": str(value.mode), "size": [int(value.width), int(value.height)]}
    if np is not None and isinstance(value, np.ndarray):
        return {
            "_type": "ndarray",
            "shape": [int(v) for v in value.shape],
            "dtype": str(value.dtype),
        }
    if isinstance(value, Mapping):
        return {str(k): _normalize_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_jsonable(v) for v in value]
    return str(value)


def _serialize_source_value(value: Any) -> Any:
    """Best-effort scalar serialization for merged JSONL/parquet metadata."""
    normalized = _normalize_jsonable(value)
    if normalized is None or isinstance(normalized, (str, bool, int, float)):
        return normalized
    # Nested structures become JSON strings to avoid parquet schema conflicts.
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def _coerce_pil_image(value: Any) -> Optional[Image.Image]:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.copy()
    if isinstance(value, Mapping):
        path_val = value.get("path")
        bytes_val = value.get("bytes")
        if isinstance(path_val, str) and path_val.strip():
            p = Path(path_val)
            if p.exists():
                with Image.open(p) as im:
                    return im.copy()
        if isinstance(bytes_val, (bytes, bytearray)):
            with Image.open(io.BytesIO(bytes(bytes_val))) as im:
                return im.copy()
    if isinstance(value, (bytes, bytearray)):
        with Image.open(io.BytesIO(bytes(value))) as im:
            return im.copy()
    if np is not None and isinstance(value, np.ndarray):
        return Image.fromarray(value)
    return None


def _save_line_png(image_value: Any, out_png: Path) -> Tuple[bool, Dict[str, Any]]:
    pil = _coerce_pil_image(image_value)
    if pil is None:
        return False, {"error": "unsupported_image_value"}
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_png, format="PNG")
    return True, {
        "width_px": int(pil.width),
        "height_px": int(pil.height),
        "image_mode": str(pil.mode),
    }


def _parse_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        if isinstance(value, str):
            txt = value.strip()
            if not txt:
                return None
            return int(float(txt))
        return int(value)
    except Exception:
        return None


def _build_ids(
    row: Mapping[str, Any],
    *,
    dataset_short: str,
    split_name: str,
    row_idx: int,
    doc_col_override: Optional[str],
    page_col_override: Optional[str],
    line_col_override: Optional[str],
    line_counters: MutableMapping[Tuple[str, str, str, str], int],
) -> Tuple[str, str, int, Dict[str, Any]]:
    doc_col = _pick_column(row, explicit=doc_col_override, candidates=DOC_COLUMN_CANDIDATES)
    page_col = _pick_column(row, explicit=page_col_override, candidates=PAGE_COLUMN_CANDIDATES)
    line_col = _pick_column(row, explicit=line_col_override, candidates=LINE_COLUMN_CANDIDATES)

    raw_doc = row.get(doc_col) if doc_col else None
    raw_page = row.get(page_col) if page_col else None
    raw_line = row.get(line_col) if line_col else None

    if raw_doc is None or str(raw_doc).strip() == "":
        doc_id = _sanitize_id(dataset_short)
    else:
        doc_id = _sanitize_id(raw_doc)
    if raw_page is None or str(raw_page).strip() == "":
        page_id = _sanitize_id(f"{split_name}_{row_idx:09d}")
    else:
        page_id = _sanitize_id(raw_page)

    line_id = _parse_optional_int(raw_line)
    counter_key = (dataset_short, split_name, doc_id, page_id)
    if line_id is None:
        line_counters[counter_key] += 1
        line_id = int(line_counters[counter_key])
    else:
        line_counters[counter_key] = max(int(line_counters[counter_key]), int(line_id))

    meta = {
        "doc_source_column": doc_col or "",
        "page_source_column": page_col or "",
        "line_source_column": line_col or "",
        "doc_source_value": _serialize_source_value(raw_doc),
        "page_source_value": _serialize_source_value(raw_page),
        "line_source_value": _serialize_source_value(raw_line),
    }
    return doc_id, page_id, int(line_id), meta


def _iter_dataset_dict_items(obj: Any) -> Iterable[Tuple[str, Any]]:
    if hasattr(obj, "items"):
        # Covers IterableDatasetDict without importing version-specific class.
        try:
            for split_name, ds in obj.items():
                yield str(split_name), ds
            return
        except Exception:
            pass
    # Single dataset fallback
    yield "train", obj


def _load_dataset_bundle(
    dataset_id: str,
    *,
    cache_dir: Optional[str],
    token: Optional[str],
    revision: Optional[str],
    trust_remote_code: bool,
    streaming: bool,
    all_configs: bool,
) -> List[Tuple[Optional[str], Any]]:
    try:
        from datasets import get_dataset_config_names, load_dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "The `datasets` package is required for this command. Install with `pip install datasets`."
        ) from exc

    kwargs = {
        "cache_dir": cache_dir,
        "token": token,
        "revision": revision,
        "trust_remote_code": trust_remote_code,
        "streaming": streaming,
    }
    kwargs = {k: v for k, v in kwargs.items() if v not in {None, ""}}

    bundles: List[Tuple[Optional[str], Any]] = []
    if not all_configs:
        bundles.append((None, load_dataset(dataset_id, **kwargs)))
        return bundles

    config_names: List[str] = []
    try:
        config_names = [str(c) for c in get_dataset_config_names(dataset_id, **{k: kwargs[k] for k in ("cache_dir", "token", "revision") if k in kwargs})]
    except Exception as exc:
        LOGGER.warning("Could not list configs for %s, falling back to default config: %s", dataset_id, exc)

    if not config_names:
        bundles.append((None, load_dataset(dataset_id, **kwargs)))
        return bundles

    for cfg in config_names:
        bundles.append((cfg, load_dataset(dataset_id, name=cfg, **kwargs)))
    return bundles


def _safe_jsonl_write_line(fp, record: Mapping[str, Any]) -> None:
    fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_parquet_from_jsonl(jsonl_path: Path, parquet_path: Path) -> None:
    if not jsonl_path.exists() or jsonl_path.stat().st_size <= 0:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_parquet(parquet_path, index=False)
        return
    df = pd.read_json(jsonl_path, lines=True)
    for col in ("doc_id", "page_id", "line_path", "source_dataset", "source_split", "source_config", "text"):
        if col in df.columns:
            df[col] = df[col].astype("string", copy=False)
    if "line_id" in df.columns:
        df["line_id"] = pd.to_numeric(df["line_id"], errors="coerce").fillna(-1).astype("int32")
    if "row_idx" in df.columns:
        df["row_idx"] = pd.to_numeric(df["row_idx"], errors="coerce").fillna(-1).astype("int64")
    for col in ("width_px", "height_px"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype("int32")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        # Retry with all source columns coerced to string for schema stability.
        for col in [c for c in df.columns if c.startswith("src__")]:
            df[col] = df[col].map(lambda v: None if pd.isna(v) else str(v)).astype("string", copy=False)
        df.to_parquet(parquet_path, index=False)


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and merge OpenPecha OCR HF datasets into a TextHierarchy-style line dataset.",
        add_help=add_help,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output dataset directory (creates train/test/eval subfolders with TextHierarchy/ and meta/).",
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        default=[],
        help="HF dataset id (repeatable). If omitted, the 7 OpenPecha OCR datasets are used.",
    )
    parser.add_argument(
        "--split",
        dest="splits",
        action="append",
        default=[],
        help="Optional split filter (repeatable), e.g. --split train --split validation.",
    )
    parser.add_argument("--image-column", type=str, default="", help="Explicit image column name.")
    parser.add_argument("--text-column", type=str, default="", help="Explicit text column name.")
    parser.add_argument("--doc-column", type=str, default="", help="Explicit doc-id source column name.")
    parser.add_argument("--page-column", type=str, default="", help="Explicit page-id source column name.")
    parser.add_argument("--line-column", type=str, default="", help="Explicit line-id source column name.")
    parser.add_argument("--cache-dir", type=str, default="", help="Optional Hugging Face datasets cache dir.")
    parser.add_argument("--revision", type=str, default="", help="Optional dataset revision/commit.")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN", ""), help="HF token (or set HF_TOKEN).")
    parser.add_argument("--max-rows-per-split", type=int, default=0, help="Debug cap per split (0 = no cap).")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to datasets.")
    parser.add_argument("--streaming", action="store_true", help="Use HF streaming mode.")
    parser.add_argument("--all-configs", action="store_true", help="Try to load all dataset configs if present.")
    parser.add_argument("--skip-parquet", action="store_true", help="Only write lines.jsonl (skip parquet conversion).")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    _configure_logging(bool(args.verbose))
    out_dir = Path(args.output_dir).expanduser().resolve()
    root_meta_dir = out_dir / "meta"
    root_jsonl_path = root_meta_dir / "lines.jsonl"
    root_parquet_path = root_meta_dir / "lines.parquet"
    summary_path = root_meta_dir / "summary.json"
    source_cols_path = root_meta_dir / "source_columns.json"
    split_mapping_path = root_meta_dir / "split_mapping.json"

    root_meta_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_process = [d.strip() for d in (args.datasets or []) if str(d).strip()] or list(DEFAULT_DATASETS)
    split_filter = {s.strip() for s in (args.splits or []) if str(s).strip()}

    source_columns: Dict[str, Dict[str, Any]] = {}
    split_mapping: Dict[str, str] = {}
    line_counters: Dict[Tuple[str, str, str, str], int] = defaultdict(int)
    path_collision_counters: Dict[Tuple[str, str, str, str, int], int] = defaultdict(int)

    totals = Counter(
        {
            "datasets_requested": len(datasets_to_process),
            "datasets_loaded": 0,
            "configs_loaded": 0,
            "splits_processed": 0,
            "rows_seen": 0,
            "rows_saved": 0,
            "rows_skipped_no_image_column": 0,
            "rows_skipped_image_decode": 0,
            "splits_skipped_by_filter": 0,
            "canonical_splits_created": 0,
        }
    )
    counts_by_dataset: Dict[str, Counter] = defaultdict(Counter)
    counts_by_canonical_split: Dict[str, Counter] = defaultdict(Counter)
    split_outputs: Dict[str, Dict[str, Any]] = {}

    with contextlib.ExitStack() as stack:
        root_fp = stack.enter_context(root_jsonl_path.open("w", encoding="utf-8"))

        def _ensure_split_output(canonical_split: str) -> Dict[str, Any]:
            state = split_outputs.get(canonical_split)
            if state is not None:
                return state
            split_root = out_dir / canonical_split
            split_text_root = split_root / "TextHierarchy"
            split_meta_dir = split_root / "meta"
            split_text_root.mkdir(parents=True, exist_ok=True)
            split_meta_dir.mkdir(parents=True, exist_ok=True)
            split_jsonl_path = split_meta_dir / "lines.jsonl"
            split_parquet_path = split_meta_dir / "lines.parquet"
            split_fp = stack.enter_context(split_jsonl_path.open("w", encoding="utf-8"))
            state = {
                "root": split_root,
                "text_root": split_text_root,
                "meta_dir": split_meta_dir,
                "jsonl_path": split_jsonl_path,
                "parquet_path": split_parquet_path,
                "fp": split_fp,
            }
            split_outputs[canonical_split] = state
            totals["canonical_splits_created"] = int(len(split_outputs))
            return state

        for dataset_id in datasets_to_process:
            dataset_short = _dataset_short_name(dataset_id)
            LOGGER.info("Loading dataset: %s", dataset_id)

            try:
                bundles = _load_dataset_bundle(
                    dataset_id,
                    cache_dir=(args.cache_dir or None),
                    token=(args.token or None),
                    revision=(args.revision or None),
                    trust_remote_code=bool(args.trust_remote_code),
                    streaming=bool(args.streaming),
                    all_configs=bool(args.all_configs),
                )
            except Exception as exc:
                LOGGER.exception("Failed to load dataset %s: %s", dataset_id, exc)
                continue

            totals["datasets_loaded"] += 1

            for config_name, ds_bundle in bundles:
                totals["configs_loaded"] += 1
                config_label = str(config_name) if config_name is not None else ""
                LOGGER.info(
                    "Processing bundle: dataset=%s config=%s",
                    dataset_id,
                    config_label or "<default>",
                )

                for split_name, split_ds in _iter_dataset_dict_items(ds_bundle):
                    if split_filter and split_name not in split_filter:
                        totals["splits_skipped_by_filter"] += 1
                        continue
                    totals["splits_processed"] += 1
                    canonical_split = _canonical_split_name(split_name)
                    split_mapping[str(split_name)] = canonical_split

                    column_names = list(getattr(split_ds, "column_names", []) or [])
                    features = getattr(split_ds, "features", None)
                    image_col = _detect_image_column(column_names, features, explicit=(args.image_column or None))
                    text_col = _detect_text_column(column_names, explicit=(args.text_column or None))

                    source_key = f"{dataset_id}::{config_label or 'default'}::{split_name}"
                    source_columns[source_key] = {
                        "dataset_id": dataset_id,
                        "dataset_short": dataset_short,
                        "config": config_label,
                        "split": split_name,
                        "canonical_split": canonical_split,
                        "columns": column_names,
                        "image_column_detected": image_col or "",
                        "text_column_detected": text_col or "",
                    }

                    if not image_col:
                        LOGGER.warning(
                            "No image column detected for %s (columns=%s). Skipping split.",
                            source_key,
                            column_names,
                        )
                        # Split skipped entirely; count rows unknown in streaming mode.
                        totals["rows_skipped_no_image_column"] += int(getattr(split_ds, "num_rows", 0) or 0)
                        continue

                    progress = tqdm(
                        enumerate(split_ds),
                        total=getattr(split_ds, "num_rows", None),
                        desc=f"{dataset_short}:{split_name}->{canonical_split}",
                    )
                    for row_idx, row in progress:
                        if int(args.max_rows_per_split) > 0 and row_idx >= int(args.max_rows_per_split):
                            break
                        totals["rows_seen"] += 1
                        counts_by_dataset[dataset_id]["rows_seen"] += 1

                        if not isinstance(row, Mapping):
                            row = {"value": row}

                        image_value = row.get(image_col)
                        if image_value is None:
                            totals["rows_skipped_image_decode"] += 1
                            counts_by_dataset[dataset_id]["rows_skipped_image_decode"] += 1
                            continue

                        doc_id, page_id, line_id, id_meta = _build_ids(
                            row,
                            dataset_short=dataset_short,
                            split_name=split_name,
                            row_idx=int(row_idx),
                            doc_col_override=(args.doc_column or None),
                            page_col_override=(args.page_column or None),
                            line_col_override=(args.line_column or None),
                            line_counters=line_counters,
                        )

                        collision_key = (dataset_short, canonical_split, doc_id, page_id, int(line_id))
                        dup_idx = int(path_collision_counters[collision_key])
                        path_collision_counters[collision_key] += 1

                        split_state = _ensure_split_output(canonical_split)
                        line_dir_name = f"line_{int(line_id):06d}" if dup_idx == 0 else f"line_{int(line_id):06d}_dup{dup_idx:03d}"
                        line_dir = (
                            Path(split_state["text_root"])
                            / f"dataset={dataset_short}"
                            / f"doc={doc_id}"
                            / f"page={page_id}"
                            / line_dir_name
                        )
                        line_png = line_dir / "line.png"
                        ok, img_meta = _save_line_png(image_value, line_png)
                        if not ok:
                            totals["rows_skipped_image_decode"] += 1
                            counts_by_dataset[dataset_id]["rows_skipped_image_decode"] += 1
                            continue

                        line_rel_path = str(line_png.relative_to(out_dir))
                        source_record: Dict[str, Any] = {}
                        for key, value in row.items():
                            pref_key = f"src__{key}"
                            if key == image_col:
                                source_record[pref_key] = line_rel_path
                            else:
                                source_record[pref_key] = _serialize_source_value(value)

                        text_value = _safe_text(row.get(text_col, "")) if text_col else ""
                        rec: Dict[str, Any] = {
                            "split": canonical_split,
                            "canonical_split": canonical_split,
                            "source_dataset": dataset_id,
                            "source_dataset_short": dataset_short,
                            "source_config": config_label,
                            "source_split": split_name,
                            "row_idx": int(row_idx),
                            "doc_id": doc_id,
                            "page_id": page_id,
                            "line_id": int(line_id),
                            "line_path": line_rel_path,
                            "text": text_value,
                            "image_column": image_col,
                            "text_column": text_col or "",
                            "line_folder_dup_idx": int(dup_idx),
                            "width_px": int(img_meta.get("width_px", -1)),
                            "height_px": int(img_meta.get("height_px", -1)),
                            "image_mode": str(img_meta.get("image_mode", "")),
                            **id_meta,
                            **source_record,
                        }
                        _safe_jsonl_write_line(root_fp, rec)
                        _safe_jsonl_write_line(split_state["fp"], rec)

                        totals["rows_saved"] += 1
                        counts_by_dataset[dataset_id]["rows_saved"] += 1
                        counts_by_dataset[dataset_id][f"split::{split_name}"] += 1
                        counts_by_canonical_split[canonical_split]["rows_saved"] += 1
                        counts_by_canonical_split[canonical_split][f"source_split::{split_name}"] += 1

    if not bool(args.skip_parquet):
        try:
            LOGGER.info("Converting merged metadata to parquet: %s", root_parquet_path)
            _write_parquet_from_jsonl(root_jsonl_path, root_parquet_path)
        except Exception as exc:
            LOGGER.exception("Failed to write parquet metadata: %s", exc)
        for canonical_split, split_state in split_outputs.items():
            try:
                LOGGER.info(
                    "Converting split metadata to parquet: split=%s path=%s",
                    canonical_split,
                    split_state["parquet_path"],
                )
                _write_parquet_from_jsonl(Path(split_state["jsonl_path"]), Path(split_state["parquet_path"]))
            except Exception as exc:
                LOGGER.exception(
                    "Failed to write split parquet metadata for %s: %s",
                    canonical_split,
                    exc,
                )

    summary: Dict[str, Any] = {
        "output_dir": str(out_dir),
        "meta_jsonl": str(root_jsonl_path),
        "meta_parquet": str(root_parquet_path),
        "datasets": datasets_to_process,
        "totals": dict(totals),
        "per_dataset": {k: dict(v) for k, v in counts_by_dataset.items()},
        "per_canonical_split": {k: dict(v) for k, v in counts_by_canonical_split.items()},
        "split_outputs": {
            k: {
                "root": str(v["root"]),
                "text_root": str(v["text_root"]),
                "meta_jsonl": str(v["jsonl_path"]),
                "meta_parquet": str(v["parquet_path"]),
            }
            for k, v in split_outputs.items()
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    source_cols_path.write_text(json.dumps(source_columns, ensure_ascii=False, indent=2), encoding="utf-8")
    split_mapping_path.write_text(json.dumps(split_mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Done. Saved summary to %s", summary_path)
    return summary


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
