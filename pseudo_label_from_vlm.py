#!/usr/bin/env python3
"""
Generate pseudo labels (YOLO format) from transformer-based parser backends.

Supported parser backends:
- mineru25
- paddleocr_vl
- qwen25vl
- granite_docling
- deepseek_ocr
- qwen3_vl (layout-only)
- groundingdino (layout-only)
- florence2 (layout-only)

Output dataset layout:
  <output_dir>/<split>/images/*.jpg|png
  <output_dir>/<split>/labels/*.txt
  <output_dir>/<split>/classes.txt
  <output_dir>/<split>/pseudo_labels_report.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image

from tibetan_utils.parsers import create_parser, parser_availability


SUPPORTED_PARSERS = [
    "mineru25",
    "paddleocr_vl",
    "qwen25vl",
    "granite_docling",
    "deepseek_ocr",
    "qwen3_vl",
    "groundingdino",
    "florence2",
]
CLASS_NAMES = ["tibetan_number_word", "tibetan_text", "chinese_number_word"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pseudo YOLO labels from VLM parser backends")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory with input images")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for pseudo-labeled dataset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split name (e.g. train/val)")
    parser.add_argument("--parser", choices=SUPPORTED_PARSERS, default="paddleocr_vl", help="Parser backend")
    parser.add_argument("--recursive", action="store_true", help="Scan input directory recursively")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Drop detections below confidence")
    parser.add_argument("--copy-images", action="store_true", help="Copy input images into output split/images")
    parser.add_argument("--save-raw-json", action="store_true", help="Save parser JSON outputs per image")
    parser.add_argument("--max-images", type=int, default=0, help="Process at most N images (0 = all)")

    # MinerU options
    parser.add_argument("--mineru-command", type=str, default="mineru", help="MinerU CLI command")
    parser.add_argument("--mineru-timeout", type=int, default=300, help="MinerU CLI timeout seconds")

    # HF/VLM options
    parser.add_argument("--hf-model-id", type=str, default="", help="Override Hugging Face model ID")
    parser.add_argument("--vlm-prompt", type=str, default="", help="Custom prompt for VLM backends")
    parser.add_argument("--vlm-max-new-tokens", type=int, default=1024, help="VLM max generated tokens")
    parser.add_argument("--hf-device", type=str, default="auto", help="HF device mode")
    parser.add_argument("--hf-dtype", type=str, default="auto", help="HF dtype")
    return parser.parse_args()


def find_images(input_dir: Path, recursive: bool) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    else:
        files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def map_detection_class(det: Dict) -> Optional[int]:
    if "class" in det:
        try:
            cls = int(det["class"])
            if cls in (0, 1, 2):
                return cls
        except Exception:
            pass

    label = str(det.get("label", "")).lower()
    text = str(det.get("text", "")).lower()
    combined = f"{label} {text}"
    if "tibetan_number" in combined or "tib_no" in combined:
        return 0
    if "chinese_number" in combined or "chi_no" in combined:
        return 2
    if "tibetan" in combined or "text" in combined:
        return 1
    return None


def box_to_yolo(box: Dict, img_w: int, img_h: int) -> Optional[Tuple[float, float, float, float]]:
    try:
        x = float(box.get("x", 0.0))
        y = float(box.get("y", 0.0))
        w = float(box.get("width", 0.0))
        h = float(box.get("height", 0.0))
    except Exception:
        return None

    if w <= 0.0 or h <= 0.0:
        return None

    # normalized center-size
    if max(abs(x), abs(y), abs(w), abs(h)) <= 1.5:
        cx, cy, bw, bh = x, y, w, h
    # xyxy packed as x,y,width,height
    elif x >= 0 and y >= 0 and w > x and h > y and w <= img_w and h <= img_h:
        x1, y1, x2, y2 = x, y, w, h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        cx = ((x1 + x2) / 2.0) / img_w
        cy = ((y1 + y2) / 2.0) / img_h
    # absolute center-size in pixels
    else:
        cx = x / img_w
        cy = y / img_h
        bw = w / img_w
        bh = h / img_h

    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    bw = min(max(bw, 1e-6), 1.0)
    bh = min(max(bh, 1e-6), 1.0)
    return cx, cy, bw, bh


def build_backend(args: argparse.Namespace):
    if args.parser == "mineru25":
        return create_parser(
            "mineru25",
            mineru_command=args.mineru_command,
            timeout_sec=args.mineru_timeout,
        )

    kwargs = {
        "prompt": args.vlm_prompt if args.vlm_prompt else None,
        "max_new_tokens": args.vlm_max_new_tokens,
        "hf_device": args.hf_device,
        "hf_dtype": args.hf_dtype,
    }
    if args.hf_model_id:
        kwargs["model_id"] = args.hf_model_id
    return create_parser(args.parser, **kwargs)


def write_classes_file(split_dir: Path) -> None:
    classes_path = split_dir / "classes.txt"
    classes_path.write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")


def open_image_size(image_path: Path) -> Tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size


def iter_with_limit(items: List[Path], max_images: int) -> Iterable[Path]:
    if max_images <= 0:
        yield from items
        return
    for i, item in enumerate(items):
        if i >= max_images:
            break
        yield item


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {in_dir}")

    ok, reason = parser_availability(args.parser)
    if not ok:
        raise RuntimeError(f"Parser '{args.parser}' unavailable: {reason}")

    images = find_images(in_dir, recursive=args.recursive)
    if not images:
        raise RuntimeError(f"No images found in: {in_dir}")

    backend = build_backend(args)

    split_dir = Path(args.output_dir) / args.split
    images_out = split_dir / "images"
    labels_out = split_dir / "labels"
    raw_out = split_dir / "raw_json"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    if args.save_raw_json:
        raw_out.mkdir(parents=True, exist_ok=True)
    write_classes_file(split_dir)

    report = {
        "parser": args.parser,
        "input_dir": str(in_dir),
        "split_dir": str(split_dir),
        "processed_images": 0,
        "images_with_labels": 0,
        "total_boxes": 0,
        "class_counts": {"0": 0, "1": 0, "2": 0},
        "errors": [],
    }

    for image_path in iter_with_limit(images, args.max_images):
        report["processed_images"] += 1
        try:
            img_w, img_h = open_image_size(image_path)
            doc = backend.parse(str(image_path), output_dir=None, image_name=image_path.name)
            out = doc.to_dict()
            detections = out.get("detections", [])

            yolo_lines: List[str] = []
            for det in detections:
                conf = float(det.get("confidence", 1.0))
                if conf < args.min_confidence:
                    continue

                cls = map_detection_class(det)
                if cls is None:
                    continue

                yolo_box = box_to_yolo(det.get("box", {}), img_w=img_w, img_h=img_h)
                if yolo_box is None:
                    continue

                cx, cy, bw, bh = yolo_box
                yolo_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                report["total_boxes"] += 1
                report["class_counts"][str(cls)] += 1

            label_path = labels_out / f"{image_path.stem}.txt"
            label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
            if yolo_lines:
                report["images_with_labels"] += 1

            if args.copy_images:
                shutil.copy2(image_path, images_out / image_path.name)

            if args.save_raw_json:
                raw_path = raw_out / f"{image_path.stem}.json"
                raw_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

        except Exception as exc:
            report["errors"].append({"image": str(image_path), "error": f"{type(exc).__name__}: {exc}"})

    report_path = split_dir / "pseudo_labels_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Done. Processed images: {report['processed_images']}")
    print(f"Labeled images: {report['images_with_labels']}")
    print(f"Total boxes: {report['total_boxes']}")
    print(f"Output split dir: {split_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
