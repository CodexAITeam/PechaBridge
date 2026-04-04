#!/usr/bin/env python3
"""Download OpenPecha line annotations and build an Ultralytics segment dataset."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import yaml
from PIL import Image, ImageOps
from tqdm.auto import tqdm

from pechabridge.ocr.line_segmentation import (
    coerce_polygon_points,
    polygon_to_box,
    polygon_to_yolo_segment_line,
)

LOGGER = logging.getLogger("download_openpecha_line_segmentation")

DEFAULT_DATASET_ID = "openpecha/OCR-Tibetan_line_segmentation_coordinate_annotation"
DEFAULT_CLASS_NAME = "line"
KNOWN_EMPTY_VALUES = {"", "unknown", "none", "null", "nan"}


def _configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _sanitize_id(value: Any) -> str:
    raw = _safe_text(value).strip()
    if not raw:
        return "unknown"
    out: List[str] = []
    for ch in raw:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "unknown"


def _parse_image_size(value: Any) -> Optional[Tuple[int, int]]:
    raw = _safe_text(value).strip().lower()
    if not raw:
        return None
    for sep in ("x", "×", ",", " "):
        if sep in raw:
            parts = [p for p in raw.replace("×", "x").replace(",", "x").replace(" ", "x").split("x") if p]
            if len(parts) >= 2:
                try:
                    width = int(float(parts[0]))
                    height = int(float(parts[1]))
                except Exception:
                    return None
                if width > 0 and height > 0:
                    return width, height
    return None


def _canonicalize_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[float, float, float]:
    vals = [max(0.0, float(train_ratio)), max(0.0, float(val_ratio)), max(0.0, float(test_ratio))]
    total = sum(vals)
    if total <= 0.0:
        raise ValueError("At least one split ratio must be > 0.")
    return vals[0] / total, vals[1] / total, vals[2] / total


def _assign_canonical_split(
    split_key: str,
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> str:
    train_ratio, val_ratio, test_ratio = _canonicalize_ratios(train_ratio, val_ratio, test_ratio)
    digest = hashlib.sha1(f"{int(seed)}::{split_key}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / float(0xFFFFFFFF)
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    if test_ratio <= 0.0:
        return "val"
    return "test"


def _page_sort_key(page: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (
        _safe_text(page.get("bdrc_work_id")).strip(),
        _safe_text(page.get("source_image")).strip(),
        _safe_text(page.get("image_url")).strip(),
    )


def _line_sort_key(line: Mapping[str, Any]) -> Tuple[int, int, str]:
    box = line.get("bbox") or [0, 0, 0, 0]
    x1, y1, _, _ = [int(v) for v in box]
    return (y1, x1, _safe_text(line.get("line_id")))


def _make_output_stem(page: Mapping[str, Any]) -> str:
    work_id = _sanitize_id(page.get("bdrc_work_id") or "unknown")
    source_name = Path(_safe_text(page.get("source_image")) or "page").stem
    source_part = _sanitize_id(source_name or "page")
    uniq_src = _safe_text(page.get("image_url")) or _safe_text(page.get("source_image")) or source_part
    uniq = hashlib.sha1(uniq_src.encode("utf-8")).hexdigest()[:10]
    return f"{work_id}__{source_part}__{uniq}"


def _probe_existing_png(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            return {
                "width_px": int(im.width),
                "height_px": int(im.height),
                "image_mode": str(im.mode),
            }
    except Exception:
        return None


def _download_image_to_png(
    url: str,
    out_path: Path,
    *,
    timeout_s: float,
    target_size: Optional[Tuple[int, int]],
    resume: bool,
) -> Dict[str, Any]:
    if bool(resume):
        existing = _probe_existing_png(out_path)
        if existing is not None:
            return existing

    req = urllib.request.Request(
        _safe_text(url).strip(),
        headers={"User-Agent": "PechaBridge-LineSeg-Downloader/1.0 (+https://github.com/openpecha)"},
    )
    with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
        payload = resp.read()
    with Image.open(io.BytesIO(payload)) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode != "RGB":
            im = im.convert("RGB")
        if target_size is not None and tuple(im.size) != tuple(target_size):
            im = im.resize(tuple(int(v) for v in target_size), Image.Resampling.BICUBIC)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(out_path, format="PNG")
        return {
            "width_px": int(im.width),
            "height_px": int(im.height),
            "image_mode": str(im.mode),
        }


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for rec in records:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_parquet(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(records))
    df.to_parquet(path, index=False)


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download openpecha/OCR-Tibetan_line_segmentation_coordinate_annotation "
            "and convert it into an Ultralytics segmentation dataset."
        ),
        add_help=add_help,
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output dataset directory.")
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        default=[],
        help=f"HF dataset id (repeatable). Default: {DEFAULT_DATASET_ID}",
    )
    parser.add_argument("--cache-dir", type=str, default="", help="Optional Hugging Face datasets cache dir.")
    parser.add_argument("--revision", type=str, default="", help="Optional dataset revision.")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN", ""), help="HF token or HF_TOKEN env.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to datasets.")
    parser.add_argument("--image-url-column", type=str, default="image_url", help="Image URL column.")
    parser.add_argument("--image-size-column", type=str, default="image_size", help="Image size column.")
    parser.add_argument("--polygon-column", type=str, default="line_coordinates", help="Polygon coordinate column.")
    parser.add_argument("--line-id-column", type=str, default="line_id", help="Line id column.")
    parser.add_argument("--source-image-column", type=str, default="source_image", help="Source page image column.")
    parser.add_argument("--work-id-column", type=str, default="bdrc_work_id", help="Document/work id column.")
    parser.add_argument("--format-column", type=str, default="format", help="Optional source format column.")
    parser.add_argument("--method-column", type=str, default="method", help="Optional annotation method column.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Deterministic train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Deterministic val split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Deterministic test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic split assignment.")
    parser.add_argument("--max-rows", type=int, default=0, help="Debug cap on source rows (0 = all).")
    parser.add_argument("--max-pages", type=int, default=0, help="Debug cap on grouped pages (0 = all).")
    parser.add_argument("--url-timeout-seconds", type=float, default=20.0, help="HTTP timeout for image downloads.")
    parser.add_argument("--resume", action="store_true", help="Reuse already-downloaded PNGs when present.")
    parser.add_argument("--class-name", type=str, default=DEFAULT_CLASS_NAME, help="Ultralytics class name.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return parser


def _iter_dataset_items(bundle: Any) -> Iterable[Tuple[str, Any]]:
    if hasattr(bundle, "items"):
        try:
            for split_name, split_ds in bundle.items():
                yield str(split_name), split_ds
            return
        except Exception:
            pass
    yield "train", bundle


def run(args: argparse.Namespace) -> Dict[str, Any]:
    _configure_logging(bool(args.verbose))

    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "The `datasets` package is required for this command. Install with `pip install datasets`."
        ) from exc

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    images_root = output_dir / "images"
    labels_root = output_dir / "labels"
    meta_root = output_dir / "meta"
    meta_root.mkdir(parents=True, exist_ok=True)

    datasets_to_process = [d.strip() for d in (args.datasets or []) if str(d).strip()] or [DEFAULT_DATASET_ID]
    load_kwargs = {
        "cache_dir": args.cache_dir or None,
        "revision": args.revision or None,
        "token": args.token or None,
        "trust_remote_code": bool(args.trust_remote_code),
    }
    load_kwargs = {k: v for k, v in load_kwargs.items() if v not in {"", None}}

    totals = Counter(
        {
            "datasets_requested": len(datasets_to_process),
            "datasets_loaded": 0,
            "rows_seen": 0,
            "rows_skipped_missing_url": 0,
            "rows_skipped_missing_size": 0,
            "rows_skipped_invalid_polygon": 0,
            "rows_kept": 0,
            "pages_grouped": 0,
            "pages_saved": 0,
            "pages_skipped_download_error": 0,
            "pages_skipped_no_valid_labels": 0,
            "labels_written": 0,
        }
    )
    pages_by_uid: Dict[str, Dict[str, Any]] = {}

    for dataset_id in datasets_to_process:
        LOGGER.info("Loading dataset: %s", dataset_id)
        dataset_bundle = load_dataset(dataset_id, **load_kwargs)
        totals["datasets_loaded"] += 1

        for raw_split_name, split_ds in _iter_dataset_items(dataset_bundle):
            LOGGER.info("Collecting rows from split=%s", raw_split_name)
            progress = tqdm(
                enumerate(split_ds),
                total=getattr(split_ds, "num_rows", None),
                desc=f"collect:{dataset_id}:{raw_split_name}",
            )
            for row_idx, row in progress:
                if int(args.max_rows or 0) > 0 and row_idx >= int(args.max_rows):
                    break
                if not isinstance(row, Mapping):
                    continue

                totals["rows_seen"] += 1
                image_url = _safe_text(row.get(args.image_url_column)).strip()
                if not image_url:
                    totals["rows_skipped_missing_url"] += 1
                    continue

                image_size = _parse_image_size(row.get(args.image_size_column))
                if image_size is None:
                    totals["rows_skipped_missing_size"] += 1
                    continue
                width_px, height_px = image_size

                polygon = coerce_polygon_points(row.get(args.polygon_column), width=width_px, height=height_px)
                if polygon is None:
                    totals["rows_skipped_invalid_polygon"] += 1
                    continue

                source_image = _safe_text(row.get(args.source_image_column)).strip() or f"{raw_split_name}_{row_idx:09d}.png"
                uid = image_url or source_image
                page = pages_by_uid.get(uid)
                if page is None:
                    work_id = _safe_text(row.get(args.work_id_column)).strip()
                    split_group = work_id if work_id.lower() not in KNOWN_EMPTY_VALUES else source_image
                    page = {
                        "dataset_id": dataset_id,
                        "raw_splits": [str(raw_split_name)],
                        "source_image": source_image,
                        "image_url": image_url,
                        "image_size": f"{width_px}x{height_px}",
                        "width_px": int(width_px),
                        "height_px": int(height_px),
                        "bdrc_work_id": work_id or "unknown",
                        "source_format": _safe_text(row.get(args.format_column)).strip(),
                        "annotation_method": _safe_text(row.get(args.method_column)).strip(),
                        "split_group_key": str(split_group),
                        "lines": [],
                    }
                    pages_by_uid[uid] = page
                elif raw_split_name not in page["raw_splits"]:
                    page["raw_splits"].append(str(raw_split_name))

                bbox = polygon_to_box(polygon, width=width_px, height=height_px)
                page["lines"].append(
                    {
                        "line_id": _safe_text(row.get(args.line_id_column)).strip() or f"line_{len(page['lines']) + 1:06d}",
                        "bbox": [int(v) for v in (bbox or (0, 0, 0, 0))],
                        "polygon": [[int(x), int(y)] for x, y in polygon],
                        "raw_split": str(raw_split_name),
                    }
                )
                totals["rows_kept"] += 1

    grouped_pages = sorted(pages_by_uid.values(), key=_page_sort_key)
    if int(args.max_pages or 0) > 0:
        grouped_pages = grouped_pages[: int(args.max_pages)]
    totals["pages_grouped"] = len(grouped_pages)

    page_records: List[Dict[str, Any]] = []
    line_records: List[Dict[str, Any]] = []
    split_counts = Counter()
    line_split_counts = Counter()

    for page in tqdm(grouped_pages, desc="download+write"):
        canonical_split = _assign_canonical_split(
            page["split_group_key"],
            seed=int(args.seed),
            train_ratio=float(args.train_ratio),
            val_ratio=float(args.val_ratio),
            test_ratio=float(args.test_ratio),
        )
        output_stem = _make_output_stem(page)
        image_path = images_root / canonical_split / f"{output_stem}.png"
        label_path = labels_root / canonical_split / f"{output_stem}.txt"

        try:
            image_meta = _download_image_to_png(
                page["image_url"],
                image_path,
                timeout_s=float(args.url_timeout_seconds),
                target_size=(int(page["width_px"]), int(page["height_px"])),
                resume=bool(args.resume),
            )
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            totals["pages_skipped_download_error"] += 1
            LOGGER.warning("Skipping page after download error (%s): %s", type(exc).__name__, page["image_url"])
            continue
        except Exception as exc:
            totals["pages_skipped_download_error"] += 1
            LOGGER.warning("Skipping page after unexpected image error (%s): %s", type(exc).__name__, page["image_url"])
            continue

        label_lines: List[str] = []
        page_line_count = 0
        sorted_lines = sorted(page["lines"], key=_line_sort_key)
        for order_idx, line in enumerate(sorted_lines, start=1):
            label_line = polygon_to_yolo_segment_line(
                line.get("polygon"),
                width=int(page["width_px"]),
                height=int(page["height_px"]),
                class_id=0,
            )
            bbox = polygon_to_box(
                line.get("polygon"),
                width=int(page["width_px"]),
                height=int(page["height_px"]),
            )
            if label_line is None or bbox is None:
                continue
            label_lines.append(label_line)
            page_line_count += 1
            line_records.append(
                {
                    "dataset_id": page["dataset_id"],
                    "canonical_split": canonical_split,
                    "raw_splits": list(page["raw_splits"]),
                    "bdrc_work_id": str(page["bdrc_work_id"]),
                    "source_image": str(page["source_image"]),
                    "image_url": str(page["image_url"]),
                    "image_path": str(image_path.relative_to(output_dir)),
                    "label_path": str(label_path.relative_to(output_dir)),
                    "line_id": str(line["line_id"]),
                    "line_no": int(order_idx),
                    "bbox": [int(v) for v in bbox],
                    "polygon": [[int(x), int(y)] for x, y in line["polygon"]],
                    "annotation_method": str(page["annotation_method"]),
                    "source_format": str(page["source_format"]),
                    "width_px": int(page["width_px"]),
                    "height_px": int(page["height_px"]),
                }
            )

        if not label_lines:
            totals["pages_skipped_no_valid_labels"] += 1
            continue

        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

        page_records.append(
            {
                "dataset_id": page["dataset_id"],
                "canonical_split": canonical_split,
                "raw_splits": list(page["raw_splits"]),
                "split_group_key": str(page["split_group_key"]),
                "bdrc_work_id": str(page["bdrc_work_id"]),
                "source_image": str(page["source_image"]),
                "image_url": str(page["image_url"]),
                "image_size": str(page["image_size"]),
                "width_px": int(page["width_px"]),
                "height_px": int(page["height_px"]),
                "saved_width_px": int(image_meta["width_px"]),
                "saved_height_px": int(image_meta["height_px"]),
                "image_mode": str(image_meta["image_mode"]),
                "image_path": str(image_path.relative_to(output_dir)),
                "label_path": str(label_path.relative_to(output_dir)),
                "line_count": int(page_line_count),
                "annotation_method": str(page["annotation_method"]),
                "source_format": str(page["source_format"]),
            }
        )
        totals["pages_saved"] += 1
        totals["labels_written"] += 1
        split_counts[canonical_split] += 1
        line_split_counts[canonical_split] += int(page_line_count)

    _write_jsonl(meta_root / "pages.jsonl", page_records)
    _write_jsonl(meta_root / "lines.jsonl", line_records)
    _write_parquet(meta_root / "pages.parquet", page_records)
    _write_parquet(meta_root / "lines.parquet", line_records)

    yaml_cfg: Dict[str, Any] = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {0: str(args.class_name or DEFAULT_CLASS_NAME)},
    }
    if split_counts.get("test", 0) > 0:
        yaml_cfg["test"] = "images/test"
    with (output_dir / "data.yaml").open("w", encoding="utf-8") as fp:
        yaml.safe_dump(yaml_cfg, fp, sort_keys=False, allow_unicode=True)

    summary = {
        "output_dir": str(output_dir),
        "dataset_ids": datasets_to_process,
        "data_yaml": str((output_dir / "data.yaml").resolve()),
        "class_name": str(args.class_name or DEFAULT_CLASS_NAME),
        "totals": dict(totals),
        "pages_per_split": dict(split_counts),
        "lines_per_split": dict(line_split_counts),
        "split_ratios": {
            "train": float(args.train_ratio),
            "val": float(args.val_ratio),
            "test": float(args.test_ratio),
        },
        "split_seed": int(args.seed),
        "split_unit": "bdrc_work_id when present, otherwise source_image",
        "meta_files": {
            "pages_jsonl": str((meta_root / "pages.jsonl").resolve()),
            "pages_parquet": str((meta_root / "pages.parquet").resolve()),
            "lines_jsonl": str((meta_root / "lines.jsonl").resolve()),
            "lines_parquet": str((meta_root / "lines.parquet").resolve()),
        },
    }
    (meta_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved Ultralytics line segmentation dataset to %s", output_dir)
    return summary


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
