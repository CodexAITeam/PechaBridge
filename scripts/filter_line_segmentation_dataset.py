#!/usr/bin/env python3
"""Filter Ultralytics line-segmentation labels into a new dataset root."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

try:
    from scripts.expand_line_segmentation_dataset import (
        _configure_logging,
        _copy_images_tree,
        _copy_or_link,
        _copy_tree,
        _dataset_root_from_cfg,
        _format_yolo_segment_line,
        _iter_files,
        _parse_yolo_segment_line,
        _resolve_dataset_yaml,
        _resolve_split_dirs,
    )
except Exception:  # pragma: no cover - direct script fallback
    import sys

    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.expand_line_segmentation_dataset import (
        _configure_logging,
        _copy_images_tree,
        _copy_or_link,
        _copy_tree,
        _dataset_root_from_cfg,
        _format_yolo_segment_line,
        _iter_files,
        _parse_yolo_segment_line,
        _resolve_dataset_yaml,
        _resolve_split_dirs,
    )

LOGGER = logging.getLogger("filter_line_segmentation_dataset")


def _polygon_width_height_ratio(points: Sequence[Tuple[float, float]]) -> float:
    if not points:
        return 0.0
    xs = [float(x) for x, _ in points]
    ys = [float(y) for _, y in points]
    width = max(0.0, max(xs) - min(xs))
    height = max(1e-9, max(ys) - min(ys))
    return float(width) / float(height)


def _filter_label_file(
    src: Path,
    dst: Path,
    *,
    min_width_height_ratio: float,
) -> Dict[str, int]:
    kept_raw = 0
    kept_instances = 0
    filtered_instances = 0
    empty_lines = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    out_lines: List[str] = []
    with src.open("r", encoding="utf-8") as fp:
        for line in fp:
            raw = line.rstrip("\n")
            if not raw.strip():
                out_lines.append("")
                empty_lines += 1
                continue
            parsed = _parse_yolo_segment_line(raw)
            if parsed is None:
                out_lines.append(raw)
                kept_raw += 1
                continue
            class_id, points = parsed
            if _polygon_width_height_ratio(points) < max(0.0, float(min_width_height_ratio)):
                filtered_instances += 1
                continue
            out_lines.append(_format_yolo_segment_line(class_id, points))
            kept_instances += 1
    dst.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return {
        "kept_instances": int(kept_instances),
        "filtered_instances": int(filtered_instances),
        "kept_raw_lines": int(kept_raw),
        "empty_lines": int(empty_lines),
    }


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Take an existing Ultralytics line-segmentation dataset and write a new dataset "
            "with tall/narrow line polygons filtered out."
        ),
        add_help=add_help,
    )
    parser.add_argument("--dataset", type=str, required=True, help="Source dataset root or data.yaml path.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output dataset directory.")
    parser.add_argument(
        "--min-width-height-ratio",
        type=float,
        default=1.0,
        help="Drop polygons whose derived bbox has width/height below this value.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["hardlink", "symlink", "copy"],
        default="hardlink",
        help="How to materialize image files in the output dataset.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing output directory before writing.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    _configure_logging(bool(args.verbose))

    source_yaml = _resolve_dataset_yaml(args.dataset)
    with source_yaml.open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {source_yaml}")

    source_root = _dataset_root_from_cfg(cfg, yaml_path=source_yaml)
    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and bool(args.overwrite):
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    out_cfg: Dict[str, Any] = {
        "path": str(output_root),
        "names": cfg.get("names", {0: "line"}),
    }
    if "nc" in cfg:
        out_cfg["nc"] = cfg["nc"]

    split_summaries: Dict[str, Any] = {}
    total_images = 0
    total_labels = 0
    total_kept_instances = 0
    total_filtered_instances = 0

    for split_name in ("train", "val", "test"):
        image_dir, labels_dir = _resolve_split_dirs(cfg, yaml_path=source_yaml, split_name=split_name)
        if image_dir is None:
            continue
        if not image_dir.exists():
            raise FileNotFoundError(f"Image split path for '{split_name}' does not exist: {image_dir}")

        out_images_dir = output_root / "images" / split_name
        out_labels_dir = output_root / "labels" / split_name
        out_images_dir.mkdir(parents=True, exist_ok=True)
        out_labels_dir.mkdir(parents=True, exist_ok=True)

        image_count = _copy_images_tree(image_dir, out_images_dir, mode=str(args.copy_mode))
        label_count = 0
        kept_instances = 0
        filtered_instances = 0
        kept_raw_lines = 0
        empty_lines = 0

        if labels_dir is not None and labels_dir.exists():
            for src_label in _iter_files(labels_dir):
                rel = src_label.relative_to(labels_dir)
                dst_label = out_labels_dir / rel
                if src_label.suffix.lower() != ".txt":
                    _copy_or_link(src_label, dst_label, "copy")
                    continue
                stats = _filter_label_file(
                    src_label,
                    dst_label,
                    min_width_height_ratio=float(args.min_width_height_ratio),
                )
                label_count += 1
                kept_instances += int(stats["kept_instances"])
                filtered_instances += int(stats["filtered_instances"])
                kept_raw_lines += int(stats["kept_raw_lines"])
                empty_lines += int(stats["empty_lines"])

        out_cfg[split_name] = f"images/{split_name}"
        split_summaries[split_name] = {
            "source_images_dir": str(image_dir),
            "source_labels_dir": str(labels_dir) if labels_dir is not None else "",
            "image_count": int(image_count),
            "label_file_count": int(label_count),
            "kept_instances": int(kept_instances),
            "filtered_instances": int(filtered_instances),
            "kept_raw_lines": int(kept_raw_lines),
            "empty_lines": int(empty_lines),
        }
        total_images += int(image_count)
        total_labels += int(label_count)
        total_kept_instances += int(kept_instances)
        total_filtered_instances += int(filtered_instances)

    _copy_tree(source_root / "meta", output_root / "meta", mode="copy")

    yaml_path = output_root / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(out_cfg, fp, sort_keys=False, allow_unicode=True)

    summary = {
        "source_yaml": str(source_yaml),
        "source_root": str(source_root),
        "output_dir": str(output_root),
        "data_yaml": str(yaml_path),
        "transform": {
            "type": "width_height_ratio_filter",
            "min_width_height_ratio": float(args.min_width_height_ratio),
        },
        "copy_mode": str(args.copy_mode),
        "splits": split_summaries,
        "totals": {
            "image_count": int(total_images),
            "label_file_count": int(total_labels),
            "kept_instances": int(total_kept_instances),
            "filtered_instances": int(total_filtered_instances),
        },
    }
    meta_root = output_root / "meta"
    meta_root.mkdir(parents=True, exist_ok=True)
    (meta_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info(
        "Filtered line polygons into %s (min_width_height_ratio=%.3f kept=%d filtered=%d)",
        output_root,
        float(args.min_width_height_ratio),
        int(total_kept_instances),
        int(total_filtered_instances),
    )
    return summary


if __name__ == "__main__":  # pragma: no cover
    run(create_parser().parse_args())
