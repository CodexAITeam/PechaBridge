#!/usr/bin/env python3
"""
Filter and optionally relabel YOLO annotations with layout rules:
- class 0 (tibetan_number_word) should be on the left
- class 1 (tibetan_text) should be in the center
- class 2 (chinese_number_word) should be on the right

Expected input layout:
  <input_split_dir>/images
  <input_split_dir>/labels
  <input_split_dir>/classes.txt (optional)

Output layout:
  <output_split_dir>/images
  <output_split_dir>/labels
  <output_split_dir>/classes.txt
  <output_split_dir>/layout_rule_report.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_CLASSES = ["tibetan_number_word", "tibetan_text", "chinese_number_word"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply left/center/right layout rules to YOLO labels")
    parser.add_argument("--input-split-dir", type=str, required=True, help="Input split directory")
    parser.add_argument("--output-split-dir", type=str, required=True, help="Output split directory")
    parser.add_argument("--left-max", type=float, default=0.33, help="Max x-center for left zone")
    parser.add_argument("--right-min", type=float, default=0.66, help="Min x-center for right zone")
    parser.add_argument("--allow-relabel", action="store_true", help="Relabel class by zone when mismatched")
    parser.add_argument("--keep-single-per-class", action="store_true", help="Keep at most one box per class")
    parser.add_argument("--copy-images", action="store_true", help="Copy images to output split")
    parser.add_argument("--dry-run", action="store_true", help="Compute report only, do not write labels")
    return parser.parse_args()


def parse_line(line: str) -> Tuple[int, float, float, float, float]:
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid YOLO line: {line.strip()}")
    cls = int(parts[0])
    cx = float(parts[1])
    cy = float(parts[2])
    w = float(parts[3])
    h = float(parts[4])
    return cls, cx, cy, w, h


def read_label_file(path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    rows: List[Tuple[int, float, float, float, float]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        rows.append(parse_line(line))
    return rows


def write_label_file(path: Path, rows: List[Tuple[int, float, float, float, float]]) -> None:
    lines = [f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cls, cx, cy, w, h in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def zone_for_x(cx: float, left_max: float, right_min: float) -> str:
    if cx <= left_max:
        return "left"
    if cx >= right_min:
        return "right"
    return "center"


def class_for_zone(zone: str) -> int:
    if zone == "left":
        return 0
    if zone == "right":
        return 2
    return 1


def choose_single(rows: List[Tuple[int, float, float, float, float]], cls: int) -> List[Tuple[int, float, float, float, float]]:
    candidates = [r for r in rows if r[0] == cls]
    if len(candidates) <= 1:
        return candidates

    if cls == 0:
        # left number: prefer farthest left, then larger area
        candidates.sort(key=lambda r: (r[1], -(r[3] * r[4])))
    elif cls == 2:
        # right number: prefer farthest right, then larger area
        candidates.sort(key=lambda r: (-r[1], -(r[3] * r[4])))
    else:
        # text body: prefer largest area
        candidates.sort(key=lambda r: (-(r[3] * r[4]), abs(0.5 - r[1])))
    return [candidates[0]]


def load_classes(input_split: Path) -> List[str]:
    classes_path = input_split / "classes.txt"
    if not classes_path.exists():
        return DEFAULT_CLASSES
    lines = [l.strip() for l in classes_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return lines if lines else DEFAULT_CLASSES


def main() -> None:
    args = parse_args()
    input_split = Path(args.input_split_dir)
    output_split = Path(args.output_split_dir)

    in_images = input_split / "images"
    in_labels = input_split / "labels"
    out_images = output_split / "images"
    out_labels = output_split / "labels"

    if not in_labels.exists():
        raise FileNotFoundError(f"Input labels folder not found: {in_labels}")

    label_files = sorted(in_labels.glob("*.txt"))
    classes = load_classes(input_split)

    report: Dict = {
        "input_split_dir": str(input_split),
        "output_split_dir": str(output_split),
        "files_processed": 0,
        "boxes_in": 0,
        "boxes_out": 0,
        "relabels": 0,
        "drops": 0,
        "kept_by_class": {"0": 0, "1": 0, "2": 0},
        "errors": [],
    }

    if not args.dry_run:
        out_labels.mkdir(parents=True, exist_ok=True)
        if args.copy_images and in_images.exists():
            out_images.mkdir(parents=True, exist_ok=True)
        (output_split / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    for label_path in label_files:
        report["files_processed"] += 1
        try:
            rows = read_label_file(label_path)
            report["boxes_in"] += len(rows)

            filtered: List[Tuple[int, float, float, float, float]] = []
            for cls, cx, cy, w, h in rows:
                if w <= 0 or h <= 0:
                    report["drops"] += 1
                    continue

                zone = zone_for_x(cx, args.left_max, args.right_min)
                expected_cls = class_for_zone(zone)
                out_cls = cls

                if cls != expected_cls:
                    if args.allow_relabel:
                        out_cls = expected_cls
                        report["relabels"] += 1
                    else:
                        report["drops"] += 1
                        continue

                filtered.append((out_cls, cx, cy, w, h))

            if args.keep_single_per_class:
                reduced: List[Tuple[int, float, float, float, float]] = []
                reduced.extend(choose_single(filtered, 0))
                reduced.extend(choose_single(filtered, 1))
                reduced.extend(choose_single(filtered, 2))
                filtered = reduced

            report["boxes_out"] += len(filtered)
            for cls, _, _, _, _ in filtered:
                if cls in (0, 1, 2):
                    report["kept_by_class"][str(cls)] += 1

            if not args.dry_run:
                out_label_path = out_labels / label_path.name
                write_label_file(out_label_path, filtered)

                if args.copy_images and in_images.exists():
                    # Try common extensions for matching image stem.
                    stem = label_path.stem
                    matched = None
                    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                        candidate = in_images / f"{stem}{ext}"
                        if candidate.exists():
                            matched = candidate
                            break
                    if matched is not None:
                        shutil.copy2(matched, out_images / matched.name)

        except Exception as exc:
            report["errors"].append({"file": str(label_path), "error": f"{type(exc).__name__}: {exc}"})

    report_path = output_split / "layout_rule_report.json"
    if not args.dry_run:
        report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Done. Files processed: {report['files_processed']}")
    print(f"Boxes in/out: {report['boxes_in']} -> {report['boxes_out']}")
    print(f"Relabels: {report['relabels']}, Drops: {report['drops']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
