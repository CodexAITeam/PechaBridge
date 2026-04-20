#!/usr/bin/env python3
"""Expand Ultralytics line-segmentation polygons vertically into a new dataset root."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

LOGGER = logging.getLogger("expand_line_segmentation_dataset")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _resolve_dataset_yaml(dataset_arg: str) -> Path:
    raw = Path(str(dataset_arg)).expanduser()
    candidates = []
    if raw.suffix.lower() in {".yml", ".yaml"}:
        candidates.append(raw)
    candidates.append(raw / "data.yaml")
    candidates.append(raw / "data.yml")

    root = Path(__file__).resolve().parent.parent
    candidates.append(root / raw)
    candidates.append(root / raw / "data.yaml")
    candidates.append(root / raw / "data.yml")
    candidates.append(root / "datasets" / raw)
    candidates.append(root / "datasets" / raw / "data.yaml")
    candidates.append(root / "datasets" / raw / "data.yml")

    for cand in candidates:
        try:
            resolved = cand.resolve()
        except Exception:
            continue
        if resolved.is_file() and resolved.suffix.lower() in {".yml", ".yaml"}:
            return resolved
    raise FileNotFoundError(
        f"Could not find dataset YAML for --dataset={dataset_arg}. "
        "Expected a dataset folder with data.yaml or a direct path to a YAML file."
    )


def _dataset_root_from_cfg(cfg: Dict[str, Any], *, yaml_path: Path) -> Path:
    raw_root = cfg.get("path", "")
    if raw_root:
        root = Path(str(raw_root)).expanduser()
        if not root.is_absolute():
            root = (yaml_path.parent / root).resolve()
        return root.resolve()
    return yaml_path.parent.resolve()


def _derive_labels_dir_from_images_dir(images_dir: Path) -> Path:
    if images_dir.name == "images":
        return (images_dir.parent / "labels").resolve()
    if images_dir.parent.name == "images":
        return (images_dir.parent.parent / "labels" / images_dir.name).resolve()
    return (images_dir.parent / "labels").resolve()


def _resolve_split_dirs(
    cfg: Dict[str, Any],
    *,
    yaml_path: Path,
    split_name: str,
) -> Tuple[Optional[Path], Optional[Path]]:
    raw_root = _dataset_root_from_cfg(cfg, yaml_path=yaml_path)
    split_ref = str(cfg.get(split_name, "") or "").strip()
    if not split_ref:
        return None, None
    image_dir = Path(split_ref).expanduser()
    if not image_dir.is_absolute():
        image_dir = (raw_root / image_dir).resolve()
    else:
        image_dir = image_dir.resolve()
    return image_dir, _derive_labels_dir_from_images_dir(image_dir)


def _copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
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


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.is_file()])


def _copy_images_tree(src: Optional[Path], dst: Path, *, mode: str) -> int:
    if src is None or not src.exists():
        return 0
    count = 0
    for path in _iter_files(src):
        if path.suffix.lower() not in _IMAGE_EXTS:
            continue
        rel = path.relative_to(src)
        _copy_or_link(path, dst / rel, mode)
        count += 1
    return count


def _copy_tree(src: Optional[Path], dst: Path, *, mode: str = "copy") -> int:
    if src is None or not src.exists():
        return 0
    count = 0
    for path in _iter_files(src):
        rel = path.relative_to(src)
        _copy_or_link(path, dst / rel, mode)
        count += 1
    return count


def _parse_yolo_segment_line(raw_line: str) -> Optional[Tuple[int, List[Tuple[float, float]]]]:
    text = str(raw_line or "").strip()
    if not text:
        return None
    parts = text.split()
    if len(parts) < 7:
        return None
    try:
        class_id = int(float(parts[0]))
        coords = [float(v) for v in parts[1:]]
    except Exception:
        return None
    if len(coords) < 6 or (len(coords) % 2) != 0:
        return None
    points: List[Tuple[float, float]] = []
    for idx in range(0, len(coords), 2):
        x = max(0.0, min(1.0, float(coords[idx])))
        y = max(0.0, min(1.0, float(coords[idx + 1])))
        points.append((x, y))
    if len(points) < 3:
        return None
    return class_id, points


def _expand_polygon_vertically(
    points: Sequence[Tuple[float, float]],
    *,
    top_ratio: float,
    bottom_ratio: float,
) -> List[Tuple[float, float]]:
    if not points:
        return []
    ys = [float(y) for _, y in points]
    ymin = min(ys)
    ymax = max(ys)
    height = max(1e-9, ymax - ymin)
    new_ymin = max(0.0, ymin - (height * max(0.0, float(top_ratio))))
    new_ymax = min(1.0, ymax + (height * max(0.0, float(bottom_ratio))))
    new_height = max(1e-9, new_ymax - new_ymin)

    out: List[Tuple[float, float]] = []
    for x, y in points:
        rel = (float(y) - ymin) / height
        ny = new_ymin + (rel * new_height)
        out.append((max(0.0, min(1.0, float(x))), max(0.0, min(1.0, ny))))
    return out


def _format_yolo_segment_line(class_id: int, points: Sequence[Tuple[float, float]]) -> str:
    coords: List[str] = []
    for x, y in points:
        coords.append(f"{float(x):.6f}")
        coords.append(f"{float(y):.6f}")
    return f"{int(class_id)} " + " ".join(coords)


def _transform_label_file(
    src: Path,
    dst: Path,
    *,
    top_ratio: float,
    bottom_ratio: float,
) -> Dict[str, int]:
    kept_raw = 0
    transformed = 0
    empty = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    out_lines: List[str] = []
    with src.open("r", encoding="utf-8") as fp:
        for line in fp:
            raw = line.rstrip("\n")
            if not raw.strip():
                out_lines.append("")
                empty += 1
                continue
            parsed = _parse_yolo_segment_line(raw)
            if parsed is None:
                out_lines.append(raw)
                kept_raw += 1
                continue
            class_id, points = parsed
            expanded = _expand_polygon_vertically(
                points,
                top_ratio=float(top_ratio),
                bottom_ratio=float(bottom_ratio),
            )
            out_lines.append(_format_yolo_segment_line(class_id, expanded))
            transformed += 1
    dst.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return {
        "transformed_instances": int(transformed),
        "kept_raw_lines": int(kept_raw),
        "empty_lines": int(empty),
    }


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Take an existing Ultralytics line-segmentation dataset and write a new dataset "
            "with vertically expanded line polygons."
        ),
        add_help=add_help,
    )
    parser.add_argument("--dataset", type=str, required=True, help="Source dataset root or data.yaml path.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output dataset directory.")
    parser.add_argument("--top-ratio", type=float, default=0.20, help="Extra relative height to add above each line polygon.")
    parser.add_argument("--bottom-ratio", type=float, default=0.20, help="Extra relative height to add below each line polygon.")
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
    total_instances = 0

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
        transformed_instances = 0
        kept_raw_lines = 0
        empty_lines = 0
        if labels_dir is not None and labels_dir.exists():
            for src_label in _iter_files(labels_dir):
                rel = src_label.relative_to(labels_dir)
                dst_label = out_labels_dir / rel
                if src_label.suffix.lower() != ".txt":
                    _copy_or_link(src_label, dst_label, "copy")
                    continue
                stats = _transform_label_file(
                    src_label,
                    dst_label,
                    top_ratio=float(args.top_ratio),
                    bottom_ratio=float(args.bottom_ratio),
                )
                label_count += 1
                transformed_instances += int(stats["transformed_instances"])
                kept_raw_lines += int(stats["kept_raw_lines"])
                empty_lines += int(stats["empty_lines"])

        out_cfg[split_name] = f"images/{split_name}"
        split_summaries[split_name] = {
            "source_images_dir": str(image_dir),
            "source_labels_dir": str(labels_dir) if labels_dir is not None else "",
            "image_count": int(image_count),
            "label_file_count": int(label_count),
            "transformed_instances": int(transformed_instances),
            "kept_raw_lines": int(kept_raw_lines),
            "empty_lines": int(empty_lines),
        }
        total_images += int(image_count)
        total_labels += int(label_count)
        total_instances += int(transformed_instances)

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
            "type": "vertical_polygon_expansion",
            "top_ratio": float(args.top_ratio),
            "bottom_ratio": float(args.bottom_ratio),
        },
        "copy_mode": str(args.copy_mode),
        "splits": split_summaries,
        "totals": {
            "image_count": int(total_images),
            "label_file_count": int(total_labels),
            "transformed_instances": int(total_instances),
        },
    }
    meta_root = output_root / "meta"
    meta_root.mkdir(parents=True, exist_ok=True)
    (meta_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info(
        "Expanded line polygons into %s (top_ratio=%.3f bottom_ratio=%.3f transformed_instances=%d)",
        output_root,
        float(args.top_ratio),
        float(args.bottom_ratio),
        int(total_instances),
    )
    return summary


if __name__ == "__main__":  # pragma: no cover
    run(create_parser().parse_args())
