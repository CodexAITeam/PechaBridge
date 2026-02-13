#!/usr/bin/env python3
"""Detect outlier pages in SBB image folders using VLM backends."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tibetan_utils.parsers import create_parser, parser_availability

LOGGER = logging.getLogger("detect_sbb_vlm_outliers")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_NORMAL = "normal_page"
CLASS_UNCERTAIN = "uncertain"

OUTLIER_LABEL_SYNONYMS: Dict[str, str] = {
    "calibration_chart": "calibration_chart",
    "calibration_card": "calibration_chart",
    "color_calibration_chart": "calibration_chart",
    "color_target": "color_target",
    "colour_target": "color_target",
    "book_cover": "book_cover",
    "cover_page": "book_cover",
    "workspace_photo": "workspace_photo",
    "desk_photo": "workspace_photo",
    "hand_or_margin_only": "hand_or_margin_only",
    "hand_only": "hand_or_margin_only",
    "only_margin_hand": "hand_or_margin_only",
    "margin_only": "hand_or_margin_only",
    "blank_page": "blank_page",
    "blank_or_nearly_blank": "blank_page",
    "non_document_photo": "non_document_photo",
    "artifact": "non_document_photo",
    "uncertain": "uncertain",
    "normal_page": "normal_page",
}

OUTLIER_CLASSES = {
    "calibration_chart",
    "color_target",
    "book_cover",
    "workspace_photo",
    "hand_or_margin_only",
    "blank_page",
    "non_document_photo",
}


DEFAULT_PROMPT = (
    "You classify one scanned page image from a Tibetan pecha workflow. "
    "Return strict JSON with key 'detections' as a list of objects. "
    "Each object must contain: label, confidence, text, bbox=[x0,y0,x1,y1]. "
    "Allowed labels only: "
    "normal_page, calibration_chart, color_target, book_cover, workspace_photo, "
    "hand_or_margin_only, blank_page, non_document_photo, uncertain. "
    "Use `text` for a short reason in one sentence. "
    "If the full image is one class, use bbox over the whole image. "
    "If multiple outlier artifacts exist, return multiple detections."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect outlier images (calibration chart, cover, workspace photos, hand/margin-only, etc.) "
                    "in sbb_images folders using VLM backends."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./sbb_images",
        help="Input image folder (recursive scan).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/sbb_vlm_outliers",
        help="Output folder for JSONL/CSV reports.",
    )
    parser.add_argument(
        "--parser",
        type=str,
        default="qwen25vl",
        choices=[
            "paddleocr_vl",
            "qwen25vl",
            "granite_docling",
            "deepseek_ocr",
            "qwen3_vl",
            "florence2",
        ],
        help="VLM parser backend key.",
    )
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default="",
        help="Optional Hugging Face model id override.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Custom VLM prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Max generated tokens for VLM response.",
    )
    parser.add_argument(
        "--hf-device",
        type=str,
        default="auto",
        help="HF device mode (auto/cpu/cuda).",
    )
    parser.add_argument(
        "--hf-dtype",
        type=str,
        default="auto",
        help="HF dtype (auto/float16/bfloat16/float32).",
    )
    parser.add_argument(
        "--min-outlier-confidence",
        type=float,
        default=0.55,
        help="Minimum confidence for assigning an outlier label.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional image limit (0 = all).",
    )
    parser.add_argument(
        "--copy-outliers",
        action="store_true",
        help="Copy detected outlier images into output_dir/outlier_images.",
    )
    parser.add_argument(
        "--quarantine-dir",
        type=str,
        default="",
        help="Optional quarantine folder. If set, outlier images are copied there (preserving relative paths).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Log progress every N processed images.",
    )
    return parser.parse_args()


def _discover_images(input_dir: Path) -> List[Path]:
    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _normalize_label(label: str) -> str:
    raw = (label or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in OUTLIER_LABEL_SYNONYMS:
        return OUTLIER_LABEL_SYNONYMS[raw]

    for key, mapped in OUTLIER_LABEL_SYNONYMS.items():
        if key in raw:
            return mapped
    return CLASS_UNCERTAIN


def _classify_detections(detections: List[Dict], min_conf: float) -> Dict[str, object]:
    if not detections:
        return {
            "is_outlier": False,
            "predicted_label": CLASS_UNCERTAIN,
            "confidence": 0.0,
            "reason": "no detections returned by VLM",
            "outlier_labels": [],
        }

    parsed = []
    for det in detections:
        label = _normalize_label(str(det.get("label", "")))
        conf = det.get("confidence", 0.0)
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0
        reason = str(det.get("text", "")).strip()
        parsed.append({"label": label, "confidence": conf_f, "reason": reason})

    parsed.sort(key=lambda x: x["confidence"], reverse=True)
    best = parsed[0]
    outlier_hits = [x for x in parsed if x["label"] in OUTLIER_CLASSES and x["confidence"] >= min_conf]

    if outlier_hits:
        top = outlier_hits[0]
        return {
            "is_outlier": True,
            "predicted_label": top["label"],
            "confidence": top["confidence"],
            "reason": top["reason"] or best["reason"],
            "outlier_labels": [x["label"] for x in outlier_hits],
        }

    if best["label"] == CLASS_NORMAL:
        return {
            "is_outlier": False,
            "predicted_label": CLASS_NORMAL,
            "confidence": best["confidence"],
            "reason": best["reason"],
            "outlier_labels": [],
        }

    return {
        "is_outlier": False,
        "predicted_label": best["label"],
        "confidence": best["confidence"],
        "reason": best["reason"],
        "outlier_labels": [],
    }


def _build_parser(args: argparse.Namespace):
    ok, reason = parser_availability(args.parser)
    if not ok:
        raise RuntimeError(f"Parser `{args.parser}` unavailable: {reason}")

    kwargs = {
        "prompt": args.prompt,
        "max_new_tokens": int(args.max_new_tokens),
        "hf_device": args.hf_device,
        "hf_dtype": args.hf_dtype,
    }
    if args.hf_model_id.strip():
        kwargs["model_id"] = args.hf_model_id.strip()
    return create_parser(args.parser, **kwargs)


def _safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return path.name


def run(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    outlier_dir = output_dir / "outlier_images"
    quarantine_dir: Optional[Path] = None
    if (args.quarantine_dir or "").strip():
        quarantine_dir = Path(args.quarantine_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_outliers:
        outlier_dir.mkdir(parents=True, exist_ok=True)
    if quarantine_dir is not None:
        quarantine_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    image_paths = _discover_images(input_dir)
    if args.limit > 0:
        image_paths = image_paths[: int(args.limit)]
    if not image_paths:
        raise RuntimeError(f"No images found in {input_dir}")

    parser = _build_parser(args)
    LOGGER.info("Using parser: %s", args.parser)
    LOGGER.info("Found %d images", len(image_paths))

    all_records: List[Dict[str, object]] = []
    outlier_records: List[Dict[str, object]] = []
    errors = 0

    for idx, image_path in enumerate(tqdm(image_paths, desc="VLM outlier scan"), start=1):
        rel_path = _safe_relpath(image_path, input_dir)
        try:
            parsed = parser.parse(str(image_path), image_name=image_path.name)
            parsed_dict = parsed.to_dict()
            detections = parsed_dict.get("detections", [])
            cls = _classify_detections(detections, min_conf=float(args.min_outlier_confidence))

            record = {
                "image_path": str(image_path),
                "relative_path": rel_path,
                "parser": args.parser,
                "is_outlier": bool(cls["is_outlier"]),
                "predicted_label": cls["predicted_label"],
                "confidence": float(cls["confidence"]),
                "reason": str(cls["reason"] or ""),
                "outlier_labels": cls["outlier_labels"],
                "detections": detections,
                "metadata": parsed_dict.get("metadata", {}),
            }
            all_records.append(record)

            if record["is_outlier"]:
                outlier_records.append(record)
                if args.copy_outliers:
                    target = outlier_dir / Path(rel_path).name
                    shutil.copy2(image_path, target)
                if quarantine_dir is not None:
                    quarantine_target = quarantine_dir / Path(rel_path)
                    quarantine_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(image_path, quarantine_target)
        except Exception as exc:
            errors += 1
            record = {
                "image_path": str(image_path),
                "relative_path": rel_path,
                "parser": args.parser,
                "is_outlier": False,
                "predicted_label": "error",
                "confidence": 0.0,
                "reason": f"{type(exc).__name__}: {exc}",
                "outlier_labels": [],
                "detections": [],
                "metadata": {},
            }
            all_records.append(record)

        if args.log_every > 0 and idx % int(args.log_every) == 0:
            LOGGER.info("Processed %d/%d | outliers=%d | errors=%d", idx, len(image_paths), len(outlier_records), errors)

    all_jsonl = output_dir / "all_results.jsonl"
    outlier_jsonl = output_dir / "outliers.jsonl"
    outlier_csv = output_dir / "outliers.csv"
    summary_json = output_dir / "summary.json"

    with all_jsonl.open("w", encoding="utf-8") as f:
        for row in all_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with outlier_jsonl.open("w", encoding="utf-8") as f:
        for row in outlier_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with outlier_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "relative_path",
                "predicted_label",
                "confidence",
                "reason",
                "image_path",
            ],
        )
        writer.writeheader()
        for row in outlier_records:
            writer.writerow(
                {
                    "relative_path": row["relative_path"],
                    "predicted_label": row["predicted_label"],
                    "confidence": f"{float(row['confidence']):.4f}",
                    "reason": row["reason"],
                    "image_path": row["image_path"],
                }
            )

    label_counts: Dict[str, int] = {}
    for row in all_records:
        label = str(row.get("predicted_label", CLASS_UNCERTAIN))
        label_counts[label] = label_counts.get(label, 0) + 1

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "quarantine_dir": str(quarantine_dir) if quarantine_dir is not None else "",
        "parser": args.parser,
        "total_images": len(image_paths),
        "outlier_images": len(outlier_records),
        "error_images": errors,
        "label_counts": label_counts,
        "all_results_jsonl": str(all_jsonl),
        "outliers_jsonl": str(outlier_jsonl),
        "outliers_csv": str(outlier_csv),
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    LOGGER.info("Finished. Total=%d | Outliers=%d | Errors=%d", len(image_paths), len(outlier_records), errors)
    LOGGER.info("Wrote: %s", all_jsonl)
    LOGGER.info("Wrote: %s", outlier_jsonl)
    LOGGER.info("Wrote: %s", outlier_csv)
    LOGGER.info("Wrote: %s", summary_json)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
