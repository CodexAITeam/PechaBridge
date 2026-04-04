#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ui_workbench import (
    _apply_donut_ui_preprocess,
    _compute_line_projection_state,
    _load_donut_ocr_runtime_cached,
    _resolve_donut_runtime_dirs,
    _segment_lines_in_text_crop,
    run_tibetan_text_line_split_classical,
)
from pechabridge.ocr.line_segmentation import (
    DEFAULT_LINE_SEGMENTATION_CONF,
    DEFAULT_LINE_SEGMENTATION_IMGSZ,
    DEFAULT_LINE_SEGMENTATION_PREPROCESS,
    normalize_line_segmentation_preprocess_pipeline,
    predict_line_regions,
)
from pechabridge.ocr.bdrc_inference import (
    BDRCLinePrediction,
    find_bdrc_line_model_dirs,
    find_bdrc_ocr_model_dirs,
    predict_bdrc_line_regions,
    run_bdrc_ocr,
)
from pechabridge.ocr.bdrc_model_download import (
    choose_default_bdrc_ocr_model_dir,
    ensure_default_bdrc_line_assets,
    ensure_default_bdrc_ocr_models,
)
try:
    from pechabridge.ocr.preprocess_bdrc import (
        BDRCPreprocessConfig,
        preprocess_image_bdrc,
    )
except Exception:
    BDRCPreprocessConfig = None
    preprocess_image_bdrc = None

try:
    from pechabridge.ocr.preprocess_rgb import (
        RGBLinePreprocessConfig,
        preprocess_image_rgb_lines,
    )
except Exception:
    RGBLinePreprocessConfig = None
    preprocess_image_rgb_lines = None

try:
    import torch
except Exception:
    torch = None


ROOT = Path(__file__).resolve().parent
MODE_AUTO = "Fully Automatic OCR"
MODE_MANUAL = "Manual Mode"
LINE_SEG_CLASSICAL = "Classical CV"
LINE_SEG_YOLO = "Pretrained YOLO Model"
LINE_SEG_BDRC = "BDRC Line Model"
OCR_ENGINE_DONUT = "DONUT"
OCR_ENGINE_BDRC = "BDRC OCR"
_DONUT_ACTIVE_RUNTIME: Dict[str, Any] = {"checkpoint": "", "runtime": None}

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def _scan_folder_for_images(folder_path: str) -> Tuple[List[str], str]:
    """Scan *folder_path* for image files and return (sorted_paths, status_msg)."""
    p = Path(folder_path).expanduser()
    if not folder_path.strip():
        return [], "Please enter a folder path."
    if not p.exists():
        return [], f"Folder not found: {folder_path}"
    if not p.is_dir():
        return [], f"Not a directory: {folder_path}"
    found: List[str] = []
    for child in sorted(p.iterdir()):
        if child.is_file() and child.suffix.lower() in _IMAGE_EXTENSIONS:
            found.append(str(child.resolve()))
    if not found:
        return [], f"No images found in {folder_path}"
    return found, f"Found {len(found)} image(s) in {p.name}"


def _make_thumbnails(image_paths: List[str]) -> List[Tuple[str, str]]:
    """Return a list of (path, caption) tuples suitable for gr.Gallery."""
    return [(p, Path(p).name) for p in image_paths]


def _is_manual_mode(mode: str) -> bool:
    return str(mode or "").strip() == MODE_MANUAL


def _normalize_line_segmentation_mode(mode: str) -> str:
    raw = str(mode or "").strip()
    if raw == LINE_SEG_YOLO:
        return LINE_SEG_YOLO
    if raw == LINE_SEG_BDRC:
        return LINE_SEG_BDRC
    return LINE_SEG_CLASSICAL


def _normalize_ocr_engine(engine: str) -> str:
    raw = str(engine or "").strip()
    if raw == OCR_ENGINE_BDRC:
        return OCR_ENGINE_BDRC
    return OCR_ENGINE_DONUT


def _load_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:
        return ImageFont.load_default()


def _is_hf_model_dir(p: Path) -> bool:
    if not p.exists() or not p.is_dir():
        return False
    if not (p / "config.json").exists():
        return False
    return (
        (p / "pytorch_model.bin").exists()
        or (p / "model.safetensors").exists()
        or (p / "model.safetensors.index.json").exists()
    )


def _is_repro_checkpoint(p: Path) -> bool:
    if not _is_hf_model_dir(p):
        return False
    repro = p / "repro"
    return (
        repro.exists()
        and (repro / "generate_config.json").exists()
        and (repro / "image_preprocess.json").exists()
        and (repro / "tokenizer").exists()
        and (repro / "image_processor").exists()
    )


def _checkpoint_step(p: Path) -> int:
    name = p.name.strip().lower()
    if name.startswith("checkpoint-"):
        raw = name.replace("checkpoint-", "", 1)
        try:
            return int(raw)
        except Exception:
            return -1
    return -1


def _is_plain_checkpoint_with_runtime_assets(p: Path) -> bool:
    if not _is_hf_model_dir(p):
        return False
    parent = p.parent
    has_parent_assets = (parent / "tokenizer").exists() and (parent / "image_processor").exists()
    has_local_gen_cfg = (p / "generation_config.json").exists()
    return bool(has_parent_assets or has_local_gen_cfg)


def _pretty_model_label(full_path: str) -> str:
    """Return a short human-readable label for a model path.

    Shows the path relative to the project root's ``models/`` directory so the
    dropdown stays compact.  Falls back to the last two path components when the
    path is outside the models tree.
    """
    try:
        p = Path(full_path)
        models_root = (ROOT / "models").resolve()
        rel = p.relative_to(models_root)
        return str(rel)
    except ValueError:
        p = Path(full_path)
        parts = p.parts
        return str(Path(*parts[-2:])) if len(parts) >= 2 else p.name


def _resolve_best_symlink(candidate: Path) -> Optional[Path]:
    """Resolve a ``best`` or ``best.pt`` symlink/path to a usable checkpoint path.

    Handles three cases:
    - ``best/``  — directory (real or symlink) → use as-is after resolving
    - ``best.pt`` → symlink/file that resolves to a *directory* → use that directory
    - ``best.pt`` → regular ``.pt`` weight file → use its parent directory
    """
    if not candidate.exists():
        return None
    try:
        resolved = candidate.resolve()
    except Exception:
        return None
    if resolved.is_dir():
        return resolved
    if resolved.is_file():
        # Could be a symlink pointing to a .pt file; treat parent as checkpoint dir
        return resolved.parent
    return None


def _list_donut_checkpoints() -> Tuple[List[Tuple[str, str]], str]:
    """Return ([(label, full_path), ...] sorted best-first, status_message).

    Priority order:
    1. ``models/ocr/best`` or ``models/ocr/best.pt`` (symlink or real file/dir)
    2. All other repro checkpoints under ``models/ocr/`` (highest step first)
    3. All other plain checkpoints under ``models/ocr/``
    4. Same scan under ``models/`` as fallback
    """
    preferred = (ROOT / "models" / "ocr").resolve()
    fallback_root = (ROOT / "models").resolve()
    repro_candidates: List[Path] = []
    plain_candidates: List[Path] = []

    # ── Check for explicit best / best.pt shortcut (symlinks supported) ──
    best_path: Optional[Path] = None
    for _name in ("best", "best.pt"):
        candidate = preferred / _name
        resolved = _resolve_best_symlink(candidate)
        if resolved is not None:
            best_path = resolved
            break

    def _scan_dir(base: Path) -> None:
        if not base.exists():
            return
        for p in sorted(base.rglob("checkpoint-*")):
            if _is_repro_checkpoint(p):
                repro_candidates.append(p.resolve())
            elif _is_plain_checkpoint_with_runtime_assets(p):
                plain_candidates.append(p.resolve())
        if not repro_candidates and not plain_candidates:
            for p in sorted(base.rglob("*")):
                if p.is_dir() and _is_repro_checkpoint(p):
                    repro_candidates.append(p.resolve())
                elif p.is_dir() and _is_plain_checkpoint_with_runtime_assets(p):
                    plain_candidates.append(p.resolve())

    _scan_dir(preferred)
    if not repro_candidates and not plain_candidates:
        _scan_dir(fallback_root)

    # Deduplicate while preserving order
    seen: set = set()
    all_repro: List[Path] = []
    for p in repro_candidates:
        if str(p) not in seen:
            seen.add(str(p))
            all_repro.append(p)
    all_plain: List[Path] = []
    for p in plain_candidates:
        if str(p) not in seen:
            seen.add(str(p))
            all_plain.append(p)

    all_repro = sorted(all_repro, key=_checkpoint_step, reverse=True)
    all_plain = sorted(all_plain, key=_checkpoint_step, reverse=True)

    # repro checkpoints first, then plain; label shows short relative path
    ordered: List[Tuple[str, str]] = [
        (_pretty_model_label(str(p)) + " ✓", str(p)) for p in all_repro
    ] + [
        (_pretty_model_label(str(p)), str(p)) for p in all_plain
    ]

    # Prepend best entry at the front (remove duplicate if rglob already found it)
    if best_path is not None:
        best_str = str(best_path)
        is_repro = _is_repro_checkpoint(best_path)
        best_label = "ocr/best ✓" if is_repro else "ocr/best"
        ordered = [(lbl, pth) for lbl, pth in ordered if pth != best_str]
        ordered.insert(0, (best_label, best_str))

    if not ordered:
        return [], "No DONUT checkpoint found (expected under models/ocr/ or models/)."

    tag = "best" if best_path is not None else ("repro" if all_repro else "plain")
    msg = f"Found {len(ordered)} DONUT checkpoint(s). Auto-selected ({tag}): {ordered[0][1]}"
    return ordered, msg


def _find_donut_checkpoint() -> Tuple[str, str]:
    """Return (best_checkpoint_path, status_message) — kept for backward compat."""
    choices, msg = _list_donut_checkpoints()
    return (choices[0][1] if choices else ""), msg


def _list_layout_models() -> Tuple[List[Tuple[str, str]], str]:
    """Return ([(label, full_path), ...] sorted, status_message).

    ``models/layout/best.pt`` (real file or symlink) is always placed first.
    """
    preferred = (ROOT / "models" / "layout").resolve()
    fallback_root = (ROOT / "models").resolve()
    model_exts = {".pt", ".onnx", ".torchscript"}
    candidates: List[Path] = []

    # ── Explicit best.pt shortcut (symlinks supported) ────────────────────
    best_path: Optional[Path] = None
    _best_candidate = preferred / "best.pt"
    if _best_candidate.exists() or _best_candidate.is_symlink():
        try:
            _resolved = _best_candidate.resolve()
            if _resolved.is_file():
                best_path = _resolved
        except Exception:
            pass

    def _scan(base: Path) -> None:
        if not base.exists():
            return
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in model_exts:
                continue
            # Skip macOS resource-fork files (._*)
            if p.name.startswith("._"):
                continue
            candidates.append(p.resolve())

    _scan(preferred)
    if not candidates:
        _scan(fallback_root)

    if not candidates and best_path is None:
        return [], "No layout analysis model found (expected under models/layout/)."

    ordered: List[Tuple[str, str]] = [
        (_pretty_model_label(str(p)), str(p)) for p in candidates
    ]

    # Ensure best.pt is at the front (remove duplicate if rglob already found it)
    if best_path is not None:
        best_str = str(best_path)
        ordered = [(lbl, pth) for lbl, pth in ordered if pth != best_str]
        ordered.insert(0, ("layout/best.pt", best_str))

    if not ordered:
        return [], "No layout analysis model found (expected under models/layout/)."

    msg = f"Found {len(ordered)} layout model(s). Auto-selected: {ordered[0][1]}"
    return ordered, msg


def _find_layout_model() -> Tuple[str, str]:
    """Return (best_model_path, status_message) — kept for backward compat."""
    choices, msg = _list_layout_models()
    return (choices[0][1] if choices else ""), msg


def _list_line_segmentation_models() -> Tuple[List[Tuple[str, str]], str]:
    """Return ([(label, full_path), ...] sorted, status_message)."""
    preferred = (ROOT / "models" / "line_segmentation").resolve()
    fallback_root = (ROOT / "models").resolve()
    model_exts = {".pt", ".onnx", ".torchscript"}
    candidates: List[Path] = []

    def _scan(base: Path) -> None:
        if not base.exists():
            return
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in model_exts:
                continue
            if p.name.startswith("._"):
                continue
            candidates.append(p.resolve())

    _scan(preferred)
    if not candidates:
        _scan(fallback_root)

    if not candidates:
        return [], "No line segmentation model found (expected under models/line_segmentation/)."

    ordered: List[Tuple[str, str]] = [
        (_pretty_model_label(str(p)), str(p)) for p in candidates
    ]
    msg = f"Found {len(ordered)} line segmentation model(s). Auto-selected: {ordered[0][1]}"
    return ordered, msg


def _list_bdrc_line_models() -> Tuple[List[Tuple[str, str]], str]:
    preferred = (ROOT / "models" / "bdrc").resolve()
    fallback_root = (ROOT / "models").resolve()
    found = find_bdrc_line_model_dirs(preferred)
    if not found:
        found = find_bdrc_line_model_dirs(fallback_root)
    if not found:
        return [], "No BDRC line model found (expected under models/bdrc/ or models/)."
    ordered = [(_pretty_model_label(str(p)), str(p)) for p in found]
    msg = f"Found {len(ordered)} BDRC line model(s). Auto-selected: {ordered[0][1]}"
    return ordered, msg


def _list_bdrc_ocr_models() -> Tuple[List[Tuple[str, str]], str]:
    preferred = (ROOT / "models" / "bdrc").resolve()
    fallback_root = (ROOT / "models").resolve()
    found = find_bdrc_ocr_model_dirs(preferred)
    if not found:
        found = find_bdrc_ocr_model_dirs(fallback_root)
    if not found:
        return [], "No BDRC OCR model found (expected under models/bdrc/ or models/)."
    ordered = [(_pretty_model_label(str(p)), str(p)) for p in found]
    msg = f"Found {len(ordered)} BDRC OCR model(s). Auto-selected: {ordered[0][1]}"
    return ordered, msg


def _ensure_default_bdrc_line_model_path(current_path: str) -> Tuple[str, Optional[str]]:
    raw = str(current_path or "").strip()
    if raw and Path(raw).exists():
        return raw, None
    result = ensure_default_bdrc_line_assets(ROOT / "models" / "bdrc")
    note = (
        f"Auto-downloaded default BDRC line assets ({', '.join(result.downloaded_items)}) to {result.root}"
        if result.downloaded_items
        else f"Using existing default BDRC line assets from {result.root}"
    )
    return str(result.line_dir), note


def _ensure_default_bdrc_ocr_model_path(current_path: str) -> Tuple[str, Optional[str]]:
    raw = str(current_path or "").strip()
    if raw and Path(raw).exists():
        return raw, None
    result = ensure_default_bdrc_ocr_models(ROOT / "models" / "bdrc")
    chosen = choose_default_bdrc_ocr_model_dir(result.root)
    note = (
        f"Auto-downloaded default BDRC OCR models to {result.root}; selected {chosen.name}"
        if result.downloaded
        else f"Using existing default BDRC OCR model {chosen.name} from {result.root}"
    )
    return str(chosen), note


def _normalize_box(box: List[int], w: int, h: int) -> Optional[List[int]]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(v) for v in box]
    except Exception:
        return None
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _sort_lines(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda r: (int(r["line_box"][1]), int(r["line_box"][0]), int(r["line_id"])))


def _rows_from_line_predictions(
    predictions: List[Any],
    *,
    image_w: int,
    image_h: int,
    offset_x: int = 0,
    offset_y: int = 0,
    source: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, pred in enumerate(predictions, start=1):
        box = _normalize_box(
            [
                int(offset_x) + int(pred.box[0]),
                int(offset_y) + int(pred.box[1]),
                int(offset_x) + int(pred.box[2]),
                int(offset_y) + int(pred.box[3]),
            ],
            image_w,
            image_h,
        )
        if box is None:
            continue
        polygon = [
            [int(offset_x) + int(x), int(offset_y) + int(y)]
            for x, y in list(getattr(pred, "polygon", []) or [])
        ]
        rows.append(
            {
                "line_id": idx,
                "line_box": box,
                "line_polygon": polygon,
                "line_confidence": float(getattr(pred, "confidence", 0.0)),
                "line_label": str(getattr(pred, "label", "line")),
                "line_class": int(getattr(pred, "class_id", 0)),
                "source": source,
            }
        )
    return _sort_lines(rows)


def _rows_from_bdrc_line_predictions(
    predictions: List[BDRCLinePrediction],
    *,
    image_w: int,
    image_h: int,
    offset_x: int = 0,
    offset_y: int = 0,
    source: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, pred in enumerate(predictions, start=1):
        box = _normalize_box(
            [
                int(offset_x) + int(pred.box[0]),
                int(offset_y) + int(pred.box[1]),
                int(offset_x) + int(pred.box[2]),
                int(offset_y) + int(pred.box[3]),
            ],
            image_w,
            image_h,
        )
        if box is None:
            continue
        polygon = [
            [int(offset_x) + int(x), int(offset_y) + int(y)]
            for x, y in list(getattr(pred, "polygon", []) or [])
        ]
        row: Dict[str, Any] = {
            "line_id": idx,
            "line_box": box,
            "line_polygon": polygon,
            "line_confidence": float(getattr(pred, "confidence", 1.0)),
            "line_label": str(getattr(pred, "label", "line")),
            "line_class": int(getattr(pred, "class_id", 0)),
            "page_angle": float(getattr(pred, "page_angle", 0.0)),
            "source": source,
        }
        crop_image = getattr(pred, "crop_image", None)
        if isinstance(crop_image, np.ndarray) and crop_image.size > 0:
            row["ocr_crop"] = np.asarray(crop_image).astype(np.uint8, copy=False)
        rows.append(row)
    return _sort_lines(rows)


def _extract_row_crop(src: np.ndarray, row: Dict[str, Any]) -> np.ndarray:
    crop = row.get("ocr_crop")
    if isinstance(crop, np.ndarray) and crop.size > 0:
        return np.asarray(crop).astype(np.uint8, copy=False)
    h, w = src.shape[:2]
    box = _normalize_box(row.get("line_box") or [], w, h)
    if box is None:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    x1, y1, x2, y2 = box
    return np.asarray(src[y1:y2, x1:x2]).astype(np.uint8, copy=False)


def _segment_full_image_lines(
    image: np.ndarray,
    *,
    line_segmentation_mode: str,
    line_segmentation_preprocess: str,
    layout_model: str,
    line_model: str,
    bdrc_line_model: str,
    device: str,
) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    src = np.asarray(image).astype(np.uint8, copy=False)
    h, w = src.shape[:2]
    mode = _normalize_line_segmentation_mode(line_segmentation_mode)

    if mode == LINE_SEG_YOLO:
        if not (line_model or "").strip():
            return [], "Line segmentation model is missing.", {"ok": False, "backend": "yolo_line_model", "reason": "missing_model"}
        preprocess_mode = normalize_line_segmentation_preprocess_pipeline(line_segmentation_preprocess)
        try:
            predictions = predict_line_regions(
                src,
                model_path=line_model,
                conf=DEFAULT_LINE_SEGMENTATION_CONF,
                imgsz=DEFAULT_LINE_SEGMENTATION_IMGSZ,
                preprocess_pipeline=preprocess_mode,
                device=device,
            )
        except Exception as exc:
            return [], f"Line segmentation inference failed: {type(exc).__name__}: {exc}", {
                "ok": False,
                "backend": "yolo_line_model",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        rows = _rows_from_line_predictions(
            predictions,
            image_w=w,
            image_h=h,
            source="full_auto_yolo_line_model",
        )
        status = f"Detected {len(rows)} line(s) with the pretrained line segmentation model."
        debug = {
            "ok": True,
            "backend": "yolo_line_model",
            "line_segmentation_mode": mode,
            "line_segmentation_model": str(line_model or ""),
            "line_segmentation_preprocess": preprocess_mode,
            "line_count": len(rows),
            "predict_conf": float(DEFAULT_LINE_SEGMENTATION_CONF),
            "predict_imgsz": int(DEFAULT_LINE_SEGMENTATION_IMGSZ),
            "predictions": [
                {
                    "line_box": list(rec.get("line_box") or []),
                    "line_polygon": rec.get("line_polygon") or [],
                    "confidence": float(rec.get("line_confidence", 0.0)),
                    "label": str(rec.get("line_label", "line")),
                    "class": int(rec.get("line_class", 0)),
                }
                for rec in rows[:256]
            ],
        }
        return rows, status, debug

    if mode == LINE_SEG_BDRC:
        if not (bdrc_line_model or "").strip():
            return [], "BDRC line model is missing.", {"ok": False, "backend": "bdrc_line_model", "reason": "missing_model"}
        try:
            predictions, bdrc_debug = predict_bdrc_line_regions(
                src,
                model_path=bdrc_line_model,
                device=device,
            )
        except Exception as exc:
            return [], f"BDRC line inference failed: {type(exc).__name__}: {exc}", {
                "ok": False,
                "backend": "bdrc_line_model",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        rows = _rows_from_bdrc_line_predictions(
            predictions,
            image_w=w,
            image_h=h,
            source="full_auto_bdrc_line_model",
        )
        status = f"Detected {len(rows)} line(s) with the BDRC line model."
        debug = dict(bdrc_debug)
        debug.update(
            {
                "ok": True,
                "line_segmentation_mode": mode,
                "line_segmentation_preprocess": "bdrc_internal",
                "line_segmentation_model": str(bdrc_line_model or ""),
            }
        )
        return rows, status, debug

    split_out = run_tibetan_text_line_split_classical(
        image=src,
        model_path=layout_model,
        conf=0.25,
        imgsz=1024,
        device=device if device != "auto" else "",
        min_line_height=10,
        projection_smooth=9,
        projection_threshold_rel=0.20,
        merge_gap_px=5,
        draw_parent_boxes=True,
        detect_red_text=False,
        red_min_redness=26,
        red_min_saturation=35,
        red_column_fill_rel=0.07,
        red_merge_gap_px=14,
        red_min_width_px=18,
        draw_red_boxes=False,
    )
    _, split_status, split_json, _, _, _, _, _, click_state = split_out
    line_records_raw = []
    if isinstance(click_state, dict):
        line_records_raw = list(click_state.get("line_boxes") or [])
    rows: List[Dict[str, Any]] = []
    for rec in line_records_raw:
        box = _normalize_box(rec.get("line_box") or [], w, h)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        rows.append(
            {
                "line_id": int(rec.get("line_id", len(rows) + 1)),
                "line_box": box,
                "line_polygon": [[x1, y1], [x2 - 1, y1], [x2 - 1, y2 - 1], [x1, y2 - 1]],
                "line_confidence": 1.0,
                "line_label": "classical_line",
                "line_class": 0,
                "source": "full_auto_classical",
            }
        )
    debug: Dict[str, Any]
    try:
        debug = json.loads(split_json) if (split_json or "").strip().startswith("{") else {"raw": split_json}
    except Exception:
        debug = {"raw": split_json}
    debug = {
        "ok": True,
        "backend": "classical_cv",
        "line_segmentation_mode": mode,
        "line_segmentation_preprocess": "none",
        "layout_model": str(layout_model or ""),
        "split_status": split_status,
        "split_debug": debug,
    }
    return _sort_lines(rows), split_status, debug


def _render_overlay(image: np.ndarray, lines: List[Dict[str, Any]], roi: Optional[List[int]] = None) -> np.ndarray:
    panel = Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(panel)
    font = _load_font()

    def _draw_strong_rect(box: List[int], color: Tuple[int, int, int], inner_width: int = 4, outer_width: int = 8) -> None:
        x1, y1, x2, y2 = [int(v) for v in box]
        # Outer dark stroke for contrast on bright backgrounds.
        draw.rectangle((x1, y1, x2, y2), outline=(0, 0, 0), width=max(2, int(outer_width)))
        # Inner bright stroke.
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, int(inner_width)))

    for i, rec in enumerate(_sort_lines(lines), start=1):
        x1, y1, x2, y2 = [int(v) for v in rec["line_box"]]
        _draw_strong_rect([x1, y1, x2, y2], color=(0, 255, 255), inner_width=4, outer_width=8)
        tag = f"line {i}"
        tx1 = x1 + 2
        ty1 = max(0, y1 - 20)
        tx2 = x1 + 18 + 8 * len(tag)
        ty2 = max(16, y1 - 2)
        draw.rectangle((tx1, ty1, tx2, ty2), fill=(0, 0, 0))
        draw.rectangle((tx1, ty1, tx2, ty2), outline=(0, 255, 255), width=2)
        draw.text((tx1 + 4, ty1 + 2), tag, fill=(255, 255, 255), font=font)
    if roi and len(roi) == 4:
        rx1, ry1, rx2, ry2 = [int(v) for v in roi]
        _draw_strong_rect([rx1, ry1, rx2, ry2], color=(255, 64, 255), inner_width=4, outer_width=8)
    return np.asarray(panel).astype(np.uint8, copy=False)


def _line_text(rows: List[Dict[str, Any]]) -> str:
    lines = _sort_lines(rows)
    return "\n".join([str(r.get("text", "") or "") for r in lines])


def _to_rgb_uint8(path: str) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.uint8)
    return arr


def _resolve_device_for_runtime(pref: str) -> str:
    p = str(pref or "auto").strip().lower()
    if p in {"", "auto"}:
        if torch is not None and bool(getattr(torch.cuda, "is_available", lambda: False)()):
            return "cuda:0"
        return "cpu"
    if p.startswith("cuda"):
        if torch is None or not bool(getattr(torch.cuda, "is_available", lambda: False)()):
            return "cpu"
    return p


def _ensure_donut_runtime_loaded(
    donut_checkpoint: str,
    device_pref: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    if torch is None:
        return None, "PyTorch not available."
    model_dir, tokenizer_dir, image_processor_dir, err = _resolve_donut_runtime_dirs(donut_checkpoint)
    if err:
        return None, str(err)

    checkpoint_key = str(Path(donut_checkpoint).expanduser().resolve())
    target_device = _resolve_device_for_runtime(device_pref)
    active_ckpt = str(_DONUT_ACTIVE_RUNTIME.get("checkpoint") or "")
    active_runtime = _DONUT_ACTIVE_RUNTIME.get("runtime")

    if active_runtime is not None and active_ckpt == checkpoint_key:
        cur_device = str(active_runtime.get("device") or "")
        if cur_device != target_device:
            try:
                active_runtime["model"].to(target_device)
                active_runtime["device"] = target_device
                active_runtime["device_msg"] = f"Moved DONUT runtime to {target_device}"
                return active_runtime, str(active_runtime["device_msg"])
            except Exception:
                # Fall back to loading a fresh runtime below.
                pass
        return active_runtime, f"DONUT runtime ready on {cur_device or target_device}"

    try:
        runtime = _load_donut_ocr_runtime_cached(
            str(model_dir),
            str(tokenizer_dir),
            str(image_processor_dir),
            target_device,
        )
    except Exception as exc:
        return None, f"Failed loading DONUT runtime: {type(exc).__name__}: {exc}"

    _DONUT_ACTIVE_RUNTIME["checkpoint"] = checkpoint_key
    _DONUT_ACTIVE_RUNTIME["runtime"] = runtime
    return runtime, f"DONUT runtime loaded on {runtime.get('device', target_device)}"


def _strip_special_token_strings_local(text: str, tokenizer: Any) -> str:
    out = str(text or "")
    try:
        toks = sorted(
            [str(t) for t in getattr(tokenizer, "all_special_tokens", []) if isinstance(t, str) and t],
            key=len,
            reverse=True,
        )
    except Exception:
        toks = []
    for tok in toks:
        out = out.replace(tok, "")
    return out


def _normalize_preprocess_preset(value: str) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"bdrc_no_bin", "grey"}:
        mode = "gray"
    if mode in {"rgb_lines", "rgb_line"}:
        mode = "rgb"
    if mode not in {"bdrc", "gray", "rgb"}:
        mode = "bdrc"
    return mode


def _effective_preprocess_overrides(
    preprocess_preset: str,
    bdrc_preprocess_overrides: Optional[Dict[str, Any]] = None,
    rgb_preprocess_overrides: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    mode = _normalize_preprocess_preset(preprocess_preset)
    if mode == "rgb":
        if isinstance(rgb_preprocess_overrides, dict) and len(rgb_preprocess_overrides) > 0:
            return dict(rgb_preprocess_overrides)
        return None
    if mode in {"bdrc", "gray"}:
        out: Dict[str, Any] = {}
        if isinstance(bdrc_preprocess_overrides, dict) and len(bdrc_preprocess_overrides) > 0:
            out.update(dict(bdrc_preprocess_overrides))
        if mode == "gray":
            out["binarize"] = False
            # Keep gray preset aligned with training-side defaults.
            out["gray_mode"] = "min_rgb"
        return out or None
    return None


def _apply_workbench_preprocess(
    image: Image.Image,
    preprocess_preset: str,
    bdrc_preprocess_overrides: Optional[Dict[str, Any]] = None,
    rgb_preprocess_overrides: Optional[Dict[str, Any]] = None,
) -> Image.Image:
    rgb = image.convert("RGB")
    mode = _normalize_preprocess_preset(preprocess_preset)
    effective_overrides = _effective_preprocess_overrides(
        preprocess_preset=mode,
        bdrc_preprocess_overrides=bdrc_preprocess_overrides,
        rgb_preprocess_overrides=rgb_preprocess_overrides,
    )

    if mode in {"bdrc", "gray"} and preprocess_image_bdrc is not None and BDRCPreprocessConfig is not None:
        try:
            if isinstance(effective_overrides, dict) and hasattr(BDRCPreprocessConfig, "from_dict"):
                cfg = BDRCPreprocessConfig.from_dict(effective_overrides)
            else:
                cfg = BDRCPreprocessConfig.vit_defaults()
            return preprocess_image_bdrc(image=rgb, config=cfg).convert("RGB")
        except Exception:
            return rgb

    if mode == "rgb" and preprocess_image_rgb_lines is not None and RGBLinePreprocessConfig is not None:
        try:
            if isinstance(effective_overrides, dict) and hasattr(RGBLinePreprocessConfig, "from_dict"):
                rgb_cfg = RGBLinePreprocessConfig.from_dict(effective_overrides)
            else:
                rgb_cfg = RGBLinePreprocessConfig.vit_defaults()
            return preprocess_image_rgb_lines(image=rgb, config=rgb_cfg).convert("RGB")
        except Exception:
            return rgb

    try:
        return _apply_donut_ui_preprocess(
            rgb,
            mode,
            bdrc_config_override=(
                effective_overrides
                if mode == "bdrc" and isinstance(effective_overrides, dict)
                else None
            ),
        ).convert("RGB")
    except Exception:
        return rgb


def _donut_preprocess_preview(
    crop: np.ndarray,
    preprocess_preset: str = "bdrc",
    bdrc_preprocess_overrides: Optional[Dict[str, Any]] = None,
    rgb_preprocess_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    pil = Image.fromarray(np.asarray(crop).astype(np.uint8)).convert("RGB")
    pre = np.asarray(pil).astype(np.uint8, copy=False)
    effective_preproc = _normalize_preprocess_preset(preprocess_preset)
    try:
        proc = _apply_workbench_preprocess(
            image=pil,
            preprocess_preset=effective_preproc,
            bdrc_preprocess_overrides=bdrc_preprocess_overrides,
            rgb_preprocess_overrides=rgb_preprocess_overrides,
        )
        post = np.asarray(proc).astype(np.uint8, copy=False)
    except Exception:
        post = pre
    return pre, post, effective_preproc


def _run_donut_on_crop_fallback(
    crop: np.ndarray,
    donut_checkpoint: str,
    device: str,
    max_len: int,
    preprocess_preset: str = "bdrc",
    bdrc_preprocess_overrides: Optional[Dict[str, Any]] = None,
    rgb_preprocess_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    if torch is None:
        raise RuntimeError("PyTorch not available.")
    runtime, rt_msg = _ensure_donut_runtime_loaded(donut_checkpoint, device)
    if runtime is None:
        raise RuntimeError(rt_msg)
    effective_preproc = _normalize_preprocess_preset(preprocess_preset)
    effective_overrides = _effective_preprocess_overrides(
        preprocess_preset=effective_preproc,
        bdrc_preprocess_overrides=bdrc_preprocess_overrides,
        rgb_preprocess_overrides=rgb_preprocess_overrides,
    )
    effective_max_len = max(8, int(max_len))

    pil = Image.fromarray(np.asarray(crop).astype(np.uint8)).convert("RGB")
    proc_pil = _apply_workbench_preprocess(
        image=pil,
        preprocess_preset=effective_preproc,
        bdrc_preprocess_overrides=bdrc_preprocess_overrides,
        rgb_preprocess_overrides=rgb_preprocess_overrides,
    )
    image_processor = runtime["image_processor"]
    tokenizer = runtime["tokenizer"]
    model = runtime["model"]
    dev = runtime["device"]

    pixel_values = image_processor(images=proc_pil, return_tensors="pt").pixel_values.to(dev)
    with torch.no_grad():
        generated = model.generate(pixel_values=pixel_values, max_length=int(effective_max_len), num_beams=1)
    text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0] if len(generated) else ""
    text = _strip_special_token_strings_local(text, tokenizer)
    return str(text or ""), {
        "ok": True,
        "fallback_used": False,
        "runtime_status": rt_msg,
        "image_preprocess_pipeline_requested": _normalize_preprocess_preset(preprocess_preset),
        "image_preprocess_pipeline_effective": effective_preproc,
        "preprocess_overrides_applied": bool(isinstance(effective_overrides, dict) and len(effective_overrides) > 0),
        "preprocess_overrides": (dict(effective_overrides) if isinstance(effective_overrides, dict) else None),
        "generation_max_length_effective": int(effective_max_len),
        "device": str(dev),
    }


def _run_donut_on_crop(
    crop: np.ndarray,
    donut_checkpoint: str,
    device: str,
    max_len: int,
    preprocess_preset: str = "bdrc",
    bdrc_preprocess_overrides: Optional[Dict[str, Any]] = None,
    rgb_preprocess_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any], np.ndarray, np.ndarray]:
    pre_img, post_img, effective_preproc = _donut_preprocess_preview(
        crop,
        preprocess_preset=preprocess_preset,
        bdrc_preprocess_overrides=bdrc_preprocess_overrides,
        rgb_preprocess_overrides=rgb_preprocess_overrides,
    )
    try:
        text_fb, dbg_fb = _run_donut_on_crop_fallback(
            crop,
            donut_checkpoint,
            device,
            max_len,
            preprocess_preset=preprocess_preset,
            bdrc_preprocess_overrides=bdrc_preprocess_overrides,
            rgb_preprocess_overrides=rgb_preprocess_overrides,
        )
        dbg_fb.setdefault("effective_preprocess_preview", effective_preproc)
        return text_fb, dbg_fb, pre_img, post_img
    except Exception as exc:
        return "", {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "image_preprocess_pipeline_effective": effective_preproc,
            "effective_preprocess_preview": effective_preproc,
        }, pre_img, post_img


def _run_bdrc_ocr_on_crop(
    crop: np.ndarray,
    bdrc_ocr_model: str,
    device: str,
    target_encoding: str = "unicode",
) -> Tuple[str, Dict[str, Any], np.ndarray, np.ndarray]:
    raw = np.asarray(crop).astype(np.uint8, copy=False)
    pre_img = raw
    try:
        text, dbg, preview_gray = run_bdrc_ocr(
            raw,
            model_path=bdrc_ocr_model,
            device=device,
            target_encoding=target_encoding,
        )
        post_img = np.stack([preview_gray] * 3, axis=-1) if preview_gray.ndim == 2 else np.asarray(preview_gray)
        dbg = dict(dbg)
        dbg.setdefault("effective_preprocess_preview", "bdrc_ocr_internal")
        return text, dbg, pre_img, np.asarray(post_img).astype(np.uint8, copy=False)
    except Exception as exc:
        return "", {
            "ok": False,
            "backend": "bdrc_ocr",
            "error": f"{type(exc).__name__}: {exc}",
            "image_preprocess_pipeline_effective": "bdrc_ocr_internal",
            "effective_preprocess_preview": "bdrc_ocr_internal",
        }, pre_img, pre_img


def _run_ocr_on_crop(
    crop: np.ndarray,
    *,
    ocr_engine: str,
    donut_checkpoint: str,
    bdrc_ocr_model: str,
    device: str,
    max_len: int,
    preprocess_preset: str = "bdrc",
    bdrc_preprocess_overrides: Optional[Dict[str, Any]] = None,
    rgb_preprocess_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any], np.ndarray, np.ndarray]:
    engine = _normalize_ocr_engine(ocr_engine)
    if engine == OCR_ENGINE_BDRC:
        return _run_bdrc_ocr_on_crop(
            crop,
            bdrc_ocr_model=bdrc_ocr_model,
            device=device,
            target_encoding="unicode",
        )
    return _run_donut_on_crop(
        crop,
        donut_checkpoint,
        device,
        max_len,
        preprocess_preset=preprocess_preset,
        bdrc_preprocess_overrides=bdrc_preprocess_overrides,
        rgb_preprocess_overrides=rgb_preprocess_overrides,
    )


@dataclass
class OCRRuntime:
    donut_path: str
    layout_path: str


def _base_state() -> Dict[str, Any]:
    return {
        "image_path": "",
        "image_name": "",
        "image": None,
        "line_rows": [],
        "manual_anchor": None,
        "last_mode": "",
        "last_debug_text": "",
    }


def _bdrc_ui_defaults() -> Dict[str, Any]:
    base: Dict[str, Any] = {}
    try:
        if BDRCPreprocessConfig is not None:
            base = dict(BDRCPreprocessConfig.vit_defaults().to_dict())
    except Exception:
        base = {}
    return {
        "gray_mode": str(base.get("gray_mode", "luma")),
        "normalize_background": bool(base.get("normalize_background", False)),
        "background_blur_ksize": int(base.get("background_blur_ksize", 0)),
        "background_strength": float(base.get("background_strength", 1.0)),
        "upscale_factor": float(base.get("upscale_factor", 1.0)),
        "upscale_interpolation": str(base.get("upscale_interpolation", "lanczos")),
        "binarize": bool(base.get("binarize", True)),
        "threshold_method": str(base.get("threshold_method", "adaptive")),
        "threshold_block_size": int(base.get("threshold_block_size", 51)),
        "threshold_c": int(base.get("threshold_c", 13)),
        "fixed_threshold": int(base.get("fixed_threshold", 120)),
        "morph_close": bool(base.get("morph_close", False)),
        "morph_close_kernel": int(base.get("morph_close_kernel", 2)),
        "remove_small_components": bool(base.get("remove_small_components", False)),
        "min_component_area": int(base.get("min_component_area", 12)),
    }


def _build_bdrc_preprocess_overrides_ui(
    gray_mode: str,
    normalize_background: bool,
    background_blur_ksize: int,
    background_strength: float,
    upscale_factor: float,
    upscale_interpolation: str,
    binarize: bool,
    threshold_method: str,
    threshold_block_size: int,
    threshold_c: int,
    fixed_threshold: int,
    morph_close: bool,
    morph_close_kernel: int,
    remove_small_components: bool,
    min_component_area: int,
) -> Dict[str, Any]:
    gm = str(gray_mode or "luma").strip().lower()
    if gm not in {"luma", "min_rgb", "max_rgb", "r", "g", "b"}:
        gm = "luma"

    interp = str(upscale_interpolation or "lanczos").strip().lower()
    if interp not in {"nearest", "linear", "cubic", "lanczos"}:
        interp = "lanczos"

    tmethod = str(threshold_method or "adaptive").strip().lower()
    if tmethod not in {"adaptive", "otsu", "fixed"}:
        tmethod = "adaptive"

    block = int(max(3, int(threshold_block_size)))
    if block % 2 == 0:
        block += 1

    blur_k = int(max(0, int(background_blur_ksize)))
    if blur_k > 0 and blur_k % 2 == 0:
        blur_k += 1

    close_k = int(max(0, int(morph_close_kernel)))
    if close_k > 0 and close_k % 2 == 0:
        close_k += 1

    return {
        "gray_mode": gm,
        "normalize_background": bool(normalize_background),
        "background_blur_ksize": blur_k,
        "background_strength": float(max(0.0, min(3.0, float(background_strength)))),
        "upscale_factor": float(max(1.0, float(upscale_factor))),
        "upscale_interpolation": interp,
        "binarize": bool(binarize),
        "threshold_method": tmethod,
        "adaptive_threshold": bool(tmethod == "adaptive"),
        "threshold_block_size": int(block),
        "threshold_c": int(threshold_c),
        "fixed_threshold": int(max(0, min(255, int(fixed_threshold)))),
        "morph_close": bool(morph_close),
        "morph_close_kernel": int(close_k),
        "remove_small_components": bool(remove_small_components),
        "min_component_area": int(max(0, int(min_component_area))),
    }


def _rgb_ui_defaults() -> Dict[str, Any]:
    base: Dict[str, Any] = {}
    try:
        if RGBLinePreprocessConfig is not None:
            base = dict(RGBLinePreprocessConfig.vit_defaults().to_dict())
    except Exception:
        base = {}
    return {
        "preserve_color": bool(base.get("preserve_color", True)),
        "normalize_background": bool(base.get("normalize_background", True)),
        "background_method": str(base.get("background_method", "shade_correct")),
        "background_blur_ksize": int(base.get("background_blur_ksize", 0)),
        "background_strength": float(base.get("background_strength", 0.35)),
        "contrast": float(base.get("contrast", 1.0)),
        "denoise": bool(base.get("denoise", False)),
        "morph_close": bool(base.get("morph_close", False)),
        "morph_close_kernel": int(base.get("morph_close_kernel", 3)),
        "remove_small_components": bool(base.get("remove_small_components", False)),
        "min_component_area": int(base.get("min_component_area", 12)),
        "upscale_factor": float(base.get("upscale_factor", 1.0)),
        "upscale_interpolation": str(base.get("upscale_interpolation", "lanczos")),
        "ink_normalization": bool(base.get("ink_normalization", True)),
        "ink_strength": float(base.get("ink_strength", 0.2)),
    }


def _build_rgb_preprocess_overrides_ui(
    preserve_color: bool,
    normalize_background: bool,
    background_method: str,
    background_blur_ksize: int,
    background_strength: float,
    contrast: float,
    denoise: bool,
    morph_close: bool,
    morph_close_kernel: int,
    remove_small_components: bool,
    min_component_area: int,
    upscale_factor: float,
    upscale_interpolation: str,
    ink_normalization: bool,
    ink_strength: float,
) -> Dict[str, Any]:
    bg_method = str(background_method or "shade_correct").strip().lower()
    if bg_method not in {"none", "shade_correct", "rolling_ball_like", "top_hat"}:
        bg_method = "shade_correct"

    interp = str(upscale_interpolation or "lanczos").strip().lower()
    if interp not in {"nearest", "linear", "cubic", "lanczos"}:
        interp = "lanczos"

    blur_k = int(max(0, int(background_blur_ksize)))
    if blur_k > 0 and blur_k % 2 == 0:
        blur_k += 1

    close_k = int(max(0, int(morph_close_kernel)))
    if close_k > 0 and close_k % 2 == 0:
        close_k += 1

    return {
        "preserve_color": bool(preserve_color),
        "normalize_background": bool(normalize_background),
        "background_method": bg_method,
        "background_blur_ksize": int(blur_k),
        "background_strength": float(max(0.0, min(1.0, float(background_strength)))),
        "contrast": float(max(0.5, min(2.5, float(contrast)))),
        "denoise": bool(denoise),
        "morph_close": bool(morph_close),
        "morph_close_kernel": int(close_k),
        "remove_small_components": bool(remove_small_components),
        "min_component_area": int(max(0, int(min_component_area))),
        "upscale_factor": float(max(1.0, float(upscale_factor))),
        "upscale_interpolation": interp,
        "ink_normalization": bool(ink_normalization),
        "ink_strength": float(max(0.0, min(1.0, float(ink_strength)))),
        # Keep deterministic preprocessing behavior.
        "to_grayscale_prob": 0.0,
    }


def _compose_preprocess_ui_settings(
    preprocess_preset: str,
    *,
    gray_mode: str,
    normalize_background: bool,
    background_blur_ksize: int,
    background_strength: float,
    upscale_factor: float,
    upscale_interpolation: str,
    binarize: bool,
    threshold_method: str,
    threshold_block_size: int,
    threshold_c: int,
    fixed_threshold: int,
    morph_close: bool,
    morph_close_kernel: int,
    remove_small_components: bool,
    min_component_area: int,
    rgb_preserve_color: bool,
    rgb_normalize_background: bool,
    rgb_background_method: str,
    rgb_background_blur_ksize: int,
    rgb_background_strength: float,
    rgb_contrast: float,
    rgb_denoise: bool,
    rgb_morph_close: bool,
    rgb_morph_close_kernel: int,
    rgb_remove_small_components: bool,
    rgb_min_component_area: int,
    rgb_upscale_factor: float,
    rgb_upscale_interpolation: str,
    rgb_ink_normalization: bool,
    rgb_ink_strength: float,
) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    preset = _normalize_preprocess_preset(preprocess_preset)
    bdrc_overrides = _build_bdrc_preprocess_overrides_ui(
        gray_mode=gray_mode,
        normalize_background=normalize_background,
        background_blur_ksize=int(background_blur_ksize),
        background_strength=float(background_strength),
        upscale_factor=float(upscale_factor),
        upscale_interpolation=upscale_interpolation,
        binarize=binarize,
        threshold_method=threshold_method,
        threshold_block_size=int(threshold_block_size),
        threshold_c=int(threshold_c),
        fixed_threshold=int(fixed_threshold),
        morph_close=morph_close,
        morph_close_kernel=int(morph_close_kernel),
        remove_small_components=remove_small_components,
        min_component_area=int(min_component_area),
    )
    rgb_overrides = _build_rgb_preprocess_overrides_ui(
        preserve_color=rgb_preserve_color,
        normalize_background=rgb_normalize_background,
        background_method=rgb_background_method,
        background_blur_ksize=int(rgb_background_blur_ksize),
        background_strength=float(rgb_background_strength),
        contrast=float(rgb_contrast),
        denoise=rgb_denoise,
        morph_close=rgb_morph_close,
        morph_close_kernel=int(rgb_morph_close_kernel),
        remove_small_components=rgb_remove_small_components,
        min_component_area=int(rgb_min_component_area),
        upscale_factor=float(rgb_upscale_factor),
        upscale_interpolation=rgb_upscale_interpolation,
        ink_normalization=rgb_ink_normalization,
        ink_strength=float(rgb_ink_strength),
    )

    if preset == "rgb":
        return preset, None, rgb_overrides
    if preset == "gray":
        gray_overrides = dict(bdrc_overrides)
        gray_overrides["binarize"] = False
        gray_overrides["gray_mode"] = "min_rgb"
        return preset, gray_overrides, None
    return preset, bdrc_overrides, None


def _run_full_auto(
    state: Dict[str, Any],
    ocr_engine: str,
    donut_checkpoint: str,
    bdrc_ocr_model: str,
    layout_model: str,
    line_segmentation_mode: str,
    line_segmentation_preprocess: str,
    line_model: str,
    bdrc_line_model: str,
    device: str,
    max_len: int,
    preprocess_preset: str = "bdrc",
    bdrc_preprocess_overrides: Optional[Dict[str, Any]] = None,
    rgb_preprocess_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, str, str, Dict[str, Any], str, Optional[np.ndarray], Optional[np.ndarray]]:
    image = state.get("image")
    if image is None:
        return None, "", "Please upload an image first.", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")
    engine = _normalize_ocr_engine(ocr_engine)
    auto_notes: List[str] = []
    if engine == OCR_ENGINE_BDRC and not bdrc_ocr_model.strip():
        try:
            bdrc_ocr_model, note = _ensure_default_bdrc_ocr_model_path(bdrc_ocr_model)
            if note:
                auto_notes.append(note)
        except Exception as exc:
            return image, "", f"BDRC OCR model is missing and auto-download failed: {type(exc).__name__}: {exc}", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")
    mode = _normalize_line_segmentation_mode(line_segmentation_mode)
    if mode == LINE_SEG_BDRC and not (bdrc_line_model or "").strip():
        try:
            bdrc_line_model, note = _ensure_default_bdrc_line_model_path(bdrc_line_model)
            if note:
                auto_notes.append(note)
        except Exception as exc:
            return image, "", f"BDRC line model is missing and auto-download failed: {type(exc).__name__}: {exc}", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")
    if engine == OCR_ENGINE_DONUT and not donut_checkpoint.strip():
        return image, "", "DONUT checkpoint is missing.", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")
    if engine == OCR_ENGINE_BDRC and not bdrc_ocr_model.strip():
        return image, "", "BDRC OCR model is missing.", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")
    if mode == LINE_SEG_CLASSICAL and not layout_model.strip():
        return image, "", "Layout model is missing.", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")
    if mode == LINE_SEG_YOLO and not (line_model or "").strip():
        return image, "", "Line segmentation model is missing.", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")
    if mode == LINE_SEG_BDRC and not (bdrc_line_model or "").strip():
        return image, "", "BDRC line model is missing.", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")

    src = np.asarray(image).astype(np.uint8, copy=False)
    h, w = src.shape[:2]
    line_records_raw, split_status, split_debug = _segment_full_image_lines(
        src,
        line_segmentation_mode=mode,
        line_segmentation_preprocess=line_segmentation_preprocess,
        layout_model=layout_model,
        line_model=line_model,
        bdrc_line_model=bdrc_line_model,
        device=device if device != "auto" else "",
    )
    rows: List[Dict[str, Any]] = []
    last_pre: Optional[np.ndarray] = None
    last_post: Optional[np.ndarray] = None
    for rec in line_records_raw:
        box = _normalize_box(rec.get("line_box") or [], w, h)
        if box is None:
            continue
        crop = _extract_row_crop(src, rec)
        text, dbg, pre_img, post_img = _run_ocr_on_crop(
            crop,
            ocr_engine=engine,
            donut_checkpoint=donut_checkpoint,
            bdrc_ocr_model=bdrc_ocr_model,
            device=device,
            max_len=max_len,
            preprocess_preset=preprocess_preset,
            bdrc_preprocess_overrides=bdrc_preprocess_overrides,
            rgb_preprocess_overrides=rgb_preprocess_overrides,
        )
        last_pre, last_post = pre_img, post_img
        rows.append(
            {
                "line_id": int(rec.get("line_id", len(rows) + 1)),
                "line_box": box,
                "line_polygon": rec.get("line_polygon") or None,
                "line_confidence": float(rec.get("line_confidence", 0.0)),
                "line_label": str(rec.get("line_label", "line")),
                "line_class": int(rec.get("line_class", 0)),
                "text": text,
                "ocr_debug": dbg,
                "ocr_engine": engine,
                "source": str(rec.get("source") or "full_auto"),
            }
        )

    rows = _sort_lines(rows)
    state["line_rows"] = rows
    state["manual_anchor"] = None
    state["last_mode"] = "full_auto"
    state["last_donut_pre"] = last_pre
    state["last_donut_post"] = last_post
    state["last_debug_text"] = str(rows[-1].get("text", "") if rows else "")
    transcript = _line_text(rows)
    debug = {
        "ok": True,
        "mode": "full_auto",
        "split_status": split_status,
        "line_count": len(rows),
        "ocr_engine": engine,
        "ocr_model": (str(donut_checkpoint or "") if engine == OCR_ENGINE_DONUT else str(bdrc_ocr_model or "")),
        "line_segmentation_mode": mode,
        "line_segmentation_preprocess": (
            normalize_line_segmentation_preprocess_pipeline(line_segmentation_preprocess)
            if mode == LINE_SEG_YOLO
            else ("bdrc_internal" if mode == LINE_SEG_BDRC else "none")
        ),
        "line_segmentation_model": (
            str(line_model or "")
            if mode == LINE_SEG_YOLO
            else (str(bdrc_line_model or "") if mode == LINE_SEG_BDRC else "")
        ),
        "image_preprocess_pipeline": _normalize_preprocess_preset(preprocess_preset),
        "preprocess_overrides": _effective_preprocess_overrides(
            preprocess_preset=preprocess_preset,
            bdrc_preprocess_overrides=bdrc_preprocess_overrides,
            rgb_preprocess_overrides=rgb_preprocess_overrides,
        ),
        "bdrc_preprocess_overrides": (
            dict(bdrc_preprocess_overrides)
            if isinstance(bdrc_preprocess_overrides, dict)
            else None
        ),
        "rgb_preprocess_overrides": (
            dict(rgb_preprocess_overrides)
            if isinstance(rgb_preprocess_overrides, dict)
            else None
        ),
        "split_json": split_debug,
        "auto_download_notes": auto_notes or None,
    }
    strong_overlay = _render_overlay(src, rows)
    status_msg = f"{split_status} OCR executed on {len(rows)} line(s)."
    if auto_notes:
        status_msg += " " + " ".join(auto_notes)
    return (
        strong_overlay,
        transcript,
        status_msg,
        state,
        json.dumps(debug, ensure_ascii=False, indent=2),
        state.get("last_donut_pre"),
        state.get("last_donut_post"),
    )


def _load_repro_ui_settings(ckpt_path: str) -> Tuple[str, str, Dict[str, Any], Dict[str, Any]]:
    """Load preprocessing settings from a repro bundle and return UI-ready values.

    Returns (preset, status_msg, bdrc_dict, rgb_dict).
    The dicts contain all keys expected by the Advanced View sliders.
    """
    repro_pipeline = _read_repro_preprocess_pipeline(ckpt_path)
    if repro_pipeline is None:
        # No repro bundle — return current defaults
        bdrc = _bdrc_ui_defaults()
        rgb = _rgb_ui_defaults()
        preset = "bdrc"
        msg = f"No repro bundle found for {Path(ckpt_path).name}. Defaults loaded."
        return preset, msg, bdrc, rgb

    preset = _normalize_preprocess_preset(repro_pipeline)

    # Reconstruct vit_defaults for the detected pipeline
    bdrc = _bdrc_ui_defaults()
    rgb = _rgb_ui_defaults()
    if preset in {"bdrc", "gray"} and BDRCPreprocessConfig is not None:
        try:
            base = dict(BDRCPreprocessConfig.vit_defaults().to_dict())
            if preset == "gray":
                base["binarize"] = False
                base["gray_mode"] = "min_rgb"
            # Map to UI keys
            bdrc = {
                "gray_mode": str(base.get("gray_mode", "luma")),
                "normalize_background": bool(base.get("normalize_background", False)),
                "background_blur_ksize": int(base.get("background_blur_ksize", 0)),
                "background_strength": float(base.get("background_strength", 1.0)),
                "upscale_factor": float(base.get("upscale_factor", 1.0)),
                "upscale_interpolation": str(base.get("upscale_interpolation", "lanczos")),
                "binarize": bool(base.get("binarize", True)),
                "threshold_method": str(base.get("threshold_method", "adaptive")),
                "threshold_block_size": int(base.get("threshold_block_size", 51)),
                "threshold_c": int(base.get("threshold_c", 13)),
                "fixed_threshold": int(base.get("fixed_threshold", 120)),
                "morph_close": bool(base.get("morph_close", False)),
                "morph_close_kernel": int(base.get("morph_close_kernel", 2)),
                "remove_small_components": bool(base.get("remove_small_components", False)),
                "min_component_area": int(base.get("min_component_area", 12)),
            }
            cfg_summary = ", ".join(f"{k}={v}" for k, v in sorted(bdrc.items()))
        except Exception:
            cfg_summary = "(could not read config)"
    elif preset == "rgb" and RGBLinePreprocessConfig is not None:
        try:
            base = dict(RGBLinePreprocessConfig.vit_defaults().to_dict())
            rgb = {
                "preserve_color": bool(base.get("preserve_color", True)),
                "normalize_background": bool(base.get("normalize_background", True)),
                "background_method": str(base.get("background_method", "shade_correct")),
                "background_blur_ksize": int(base.get("background_blur_ksize", 0)),
                "background_strength": float(base.get("background_strength", 0.35)),
                "contrast": float(base.get("contrast", 1.0)),
                "denoise": bool(base.get("denoise", False)),
                "morph_close": bool(base.get("morph_close", False)),
                "morph_close_kernel": int(base.get("morph_close_kernel", 3)),
                "remove_small_components": bool(base.get("remove_small_components", False)),
                "min_component_area": int(base.get("min_component_area", 12)),
                "upscale_factor": float(base.get("upscale_factor", 1.0)),
                "upscale_interpolation": str(base.get("upscale_interpolation", "lanczos")),
                "ink_normalization": bool(base.get("ink_normalization", True)),
                "ink_strength": float(base.get("ink_strength", 0.2)),
            }
            cfg_summary = ", ".join(f"{k}={v}" for k, v in sorted(rgb.items()))
        except Exception:
            cfg_summary = "(could not read config)"
    else:
        cfg_summary = "defaults"

    p = Path(ckpt_path)
    model_label = f"{p.parent.name}/{p.name}"
    msg = f"Loaded repro settings from {model_label}: pipeline={preset} | {cfg_summary}"
    return preset, msg, bdrc, rgb


def _resolve_effective_preprocess_config_dict(
    preset: str,
    pipeline_source: str,
    bdrc_overrides: Optional[Dict[str, Any]],
    rgb_overrides: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return the full config dict that will actually be applied during inference.

    For repro-bundle checkpoints the vit_defaults() are used (no UI overrides).
    For UI-fallback checkpoints the current UI override dict is used.
    """
    mode = _normalize_preprocess_preset(preset)
    if pipeline_source == "repro":
        # Reconstruct vit_defaults to show exact training-time parameters
        if mode in {"bdrc", "gray"} and BDRCPreprocessConfig is not None:
            try:
                base = dict(BDRCPreprocessConfig.vit_defaults().to_dict())
                if mode == "gray":
                    base["binarize"] = False
                    base["gray_mode"] = "min_rgb"
                return base
            except Exception:
                pass
        if mode == "rgb" and RGBLinePreprocessConfig is not None:
            try:
                return dict(RGBLinePreprocessConfig.vit_defaults().to_dict())
            except Exception:
                pass
        return {}
    else:
        # UI-configured overrides
        effective = _effective_preprocess_overrides(
            preprocess_preset=preset,
            bdrc_preprocess_overrides=bdrc_overrides,
            rgb_preprocess_overrides=rgb_overrides,
        )
        return dict(effective) if isinstance(effective, dict) else {}


def _read_repro_preprocess_pipeline(ckpt_path: str) -> Optional[str]:
    """Read the image_preprocess pipeline name saved in a repro checkpoint bundle.

    Returns the pipeline string (e.g. 'bdrc', 'gray', 'rgb') or None if not available.
    """
    try:
        repro = Path(ckpt_path) / "repro" / "image_preprocess.json"
        if repro.exists():
            data = json.loads(repro.read_text(encoding="utf-8"))
            pipeline = str(data.get("pipeline", "") or "").strip().lower()
            if pipeline and pipeline != "none":
                return pipeline
    except Exception:
        pass
    return None


def _run_comparison(
    state: Dict[str, Any],
    ocr_engine: str,
    layout_model: str,
    line_segmentation_mode: str,
    line_segmentation_preprocess: str,
    line_model: str,
    bdrc_line_model: str,
    device: str,
    max_len: int,
    preprocess_preset: str = "bdrc",
    bdrc_preprocess_overrides: Optional[Dict[str, Any]] = None,
    rgb_preprocess_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, str, str, Dict[str, Any], str, Optional[np.ndarray], Optional[np.ndarray]]:
    """Run all discovered DONUT checkpoints sequentially and write a comparison report.

    For each checkpoint the preprocessing pipeline is auto-detected from the
    repro bundle (``repro/image_preprocess.json``).  When no repro bundle is
    present the globally selected UI preset is used as a fallback.
    """
    image = state.get("image")
    if image is None:
        return None, "", "Please upload an image first.", state, "{}", None, None
    if _normalize_ocr_engine(ocr_engine) != OCR_ENGINE_DONUT:
        src = np.asarray(state.get("image")).astype(np.uint8, copy=False)
        return src, "", "Comparison mode currently supports DONUT checkpoints only.", state, "{}", None, None
    auto_notes: List[str] = []
    mode = _normalize_line_segmentation_mode(line_segmentation_mode)
    if mode == LINE_SEG_CLASSICAL and not layout_model.strip():
        return np.asarray(image).astype(np.uint8, copy=False), "", "Layout model is missing.", state, "{}", None, None
    if mode == LINE_SEG_YOLO and not (line_model or "").strip():
        return np.asarray(image).astype(np.uint8, copy=False), "", "Line segmentation model is missing.", state, "{}", None, None
    if mode == LINE_SEG_BDRC and not (bdrc_line_model or "").strip():
        try:
            bdrc_line_model, note = _ensure_default_bdrc_line_model_path(bdrc_line_model)
            if note:
                auto_notes.append(note)
        except Exception as exc:
            return np.asarray(image).astype(np.uint8, copy=False), "", f"BDRC line model is missing and auto-download failed: {type(exc).__name__}: {exc}", state, "{}", None, None

    checkpoints, scan_msg = _list_donut_checkpoints()
    if not checkpoints:
        return np.asarray(image).astype(np.uint8, copy=False), "", f"No DONUT checkpoints found. {scan_msg}", state, "{}", None, None

    src = np.asarray(image).astype(np.uint8, copy=False)
    line_records_raw, split_status, split_debug = _segment_full_image_lines(
        src,
        line_segmentation_mode=mode,
        line_segmentation_preprocess=line_segmentation_preprocess,
        layout_model=layout_model,
        line_model=line_model,
        bdrc_line_model=bdrc_line_model,
        device=device if device != "auto" else "",
    )

    h, w = src.shape[:2]
    image_name = str(state.get("image_name") or "image")

    report_lines: List[str] = [f"Image: {image_name}", ""]
    last_overlay = _render_overlay(src, line_records_raw)
    ckpt_debug: List[Dict[str, Any]] = []

    for label, ckpt_path in checkpoints:
        p = Path(ckpt_path)
        model_name = f"{p.parent.name}/{p.name}"

        # Auto-detect preprocessing pipeline from repro bundle; fall back to UI preset.
        repro_pipeline = _read_repro_preprocess_pipeline(ckpt_path)
        if repro_pipeline is not None:
            effective_preset = repro_pipeline
            # Use vit_defaults for the repro pipeline — no UI overrides applied,
            # matching the exact training-time configuration.
            effective_bdrc_overrides: Optional[Dict[str, Any]] = None
            effective_rgb_overrides: Optional[Dict[str, Any]] = None
            pipeline_source = "repro"
        else:
            effective_preset = preprocess_preset
            effective_bdrc_overrides = bdrc_preprocess_overrides
            effective_rgb_overrides = rgb_preprocess_overrides
            pipeline_source = "ui_fallback"

        rows: List[Dict[str, Any]] = []
        for rec in line_records_raw:
            box = _normalize_box(rec.get("line_box") or [], w, h)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            crop = src[y1:y2, x1:x2]
            try:
                text, _dbg, _pre, _post = _run_donut_on_crop(
                    crop,
                    ckpt_path,
                    device,
                    max_len,
                    preprocess_preset=effective_preset,
                    bdrc_preprocess_overrides=effective_bdrc_overrides,
                    rgb_preprocess_overrides=effective_rgb_overrides,
                )
            except Exception as exc:
                text = f"[ERROR: {exc}]"
            rows.append({
                "line_id": int(rec.get("line_id", len(rows) + 1)),
                "line_box": box,
                "text": text,
                "source": "comparison",
            })
        rows = _sort_lines(rows)
        transcript = _line_text(rows)
        report_lines.append(f"Model: {model_name}")
        pipeline_note = "from repro bundle" if pipeline_source == "repro" else "from UI selection (no repro bundle)"
        cfg_dict = _resolve_effective_preprocess_config_dict(
            preset=effective_preset,
            pipeline_source=pipeline_source,
            bdrc_overrides=effective_bdrc_overrides,
            rgb_overrides=effective_rgb_overrides,
        )
        cfg_str = ", ".join(f"{k}={v}" for k, v in sorted(cfg_dict.items())) if cfg_dict else "defaults"
        report_lines.append(f"Preprocessing: {effective_preset} ({pipeline_note})")
        report_lines.append(f"Preprocessing config: {cfg_str}")
        report_lines.append(f"Text:\n{transcript}")
        report_lines.append("")
        ckpt_debug.append({
            "checkpoint": ckpt_path,
            "model_name": model_name,
            "preprocess_pipeline": effective_preset,
            "pipeline_source": pipeline_source,
            "line_count": len(rows),
        })
        # Unload model from cache to free memory before loading next
        _DONUT_ACTIVE_RUNTIME["checkpoint"] = ""
        _DONUT_ACTIVE_RUNTIME["runtime"] = None

    full_report = "\n".join(report_lines).strip()
    state["line_rows"] = []
    state["last_mode"] = "comparison"
    state["last_debug_text"] = ""
    debug = {
        "ok": True,
        "mode": "comparison",
        "checkpoints_compared": len(checkpoints),
        "split_status": split_status,
        "ocr_engine": OCR_ENGINE_DONUT,
        "line_segmentation_mode": mode,
        "line_segmentation_preprocess": (
            normalize_line_segmentation_preprocess_pipeline(line_segmentation_preprocess)
            if mode == LINE_SEG_YOLO
            else ("bdrc_internal" if mode == LINE_SEG_BDRC else "none")
        ),
        "line_segmentation_model": (
            str(line_model or "")
            if mode == LINE_SEG_YOLO
            else (str(bdrc_line_model or "") if mode == LINE_SEG_BDRC else "")
        ),
        "line_count": len(line_records_raw),
        "split_json": split_debug,
        "checkpoints": ckpt_debug,
        "auto_download_notes": auto_notes or None,
    }
    status_msg = f"Comparison done: {len(checkpoints)} model(s), {len(line_records_raw)} line(s)."
    if auto_notes:
        status_msg += " " + " ".join(auto_notes)
    return (
        last_overlay,
        full_report,
        status_msg,
        state,
        json.dumps(debug, ensure_ascii=False, indent=2),
        None,
        None,
    )


def _find_line_hit(rows: List[Dict[str, Any]], x: int, y: int) -> Optional[Dict[str, Any]]:
    hits: List[Tuple[int, Dict[str, Any]]] = []
    for rec in rows:
        box = rec.get("line_box") or []
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        if x1 <= x <= x2 and y1 <= y <= y2:
            area = max(1, (x2 - x1) * (y2 - y1))
            hits.append((area, rec))
    if not hits:
        return None
    hits.sort(key=lambda t: t[0])
    return hits[0][1]


def _roi_is_single_line(roi_w: int, roi_h: int) -> bool:
    if roi_h <= 0 or roi_w <= 0:
        return False
    ratio = float(roi_w) / float(max(1, roi_h))
    return ratio >= 4.0 or roi_h <= 80


def _add_or_update_line(rows: List[Dict[str, Any]], row: Dict[str, Any]) -> List[Dict[str, Any]]:
    box = row.get("line_box") or [0, 0, 0, 0]
    for i, rec in enumerate(rows):
        if list(rec.get("line_box") or []) == list(box):
            rows[i] = row
            return _sort_lines(rows)
    rows.append(row)
    return _sort_lines(rows)


def _manual_click(
    state: Dict[str, Any],
    ocr_engine: str,
    donut_checkpoint: str,
    bdrc_ocr_model: str,
    layout_model: str,
    line_segmentation_mode: str,
    line_segmentation_preprocess: str,
    line_model: str,
    bdrc_line_model: str,
    device: str,
    max_len: int,
    preprocess_preset: str,
    bdrc_preprocess_overrides: Optional[Dict[str, Any]],
    rgb_preprocess_overrides: Optional[Dict[str, Any]],
    evt: gr.SelectData,
) -> Tuple[np.ndarray, str, str, Dict[str, Any], str, Optional[np.ndarray], Optional[np.ndarray]]:
    image = state.get("image")
    if image is None:
        return None, "", "Please upload an image first.", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")
    engine = _normalize_ocr_engine(ocr_engine)
    if engine == OCR_ENGINE_BDRC and not bdrc_ocr_model.strip():
        try:
            bdrc_ocr_model, _ = _ensure_default_bdrc_ocr_model_path(bdrc_ocr_model)
        except Exception as exc:
            return image, "", f"BDRC OCR model is missing and auto-download failed: {type(exc).__name__}: {exc}", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")
    mode = _normalize_line_segmentation_mode(line_segmentation_mode)
    if mode == LINE_SEG_BDRC and not (bdrc_line_model or "").strip():
        try:
            bdrc_line_model, _ = _ensure_default_bdrc_line_model_path(bdrc_line_model)
        except Exception as exc:
            return image, "", f"BDRC line model is missing and auto-download failed: {type(exc).__name__}: {exc}", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")
    if engine == OCR_ENGINE_DONUT and not donut_checkpoint.strip():
        return image, "", "DONUT checkpoint is missing.", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")
    if engine == OCR_ENGINE_BDRC and not bdrc_ocr_model.strip():
        return image, "", "BDRC OCR model is missing.", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")

    idx = getattr(evt, "index", None)
    if not isinstance(idx, (tuple, list)) or len(idx) < 2:
        return np.asarray(image), _line_text(state.get("line_rows") or []), "Click position not available.", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")
    try:
        click_x, click_y = int(idx[0]), int(idx[1])
    except Exception:
        return np.asarray(image), _line_text(state.get("line_rows") or []), "Invalid click position.", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")

    src = np.asarray(image).astype(np.uint8, copy=False)
    h, w = src.shape[:2]
    click_x = max(0, min(w - 1, click_x))
    click_y = max(0, min(h - 1, click_y))
    rows = list(state.get("line_rows") or [])

    hit = _find_line_hit(rows, click_x, click_y)
    if hit is not None:
        crop = _extract_row_crop(src, hit)
        text, dbg, pre_img, post_img = _run_ocr_on_crop(
            crop,
            ocr_engine=engine,
            donut_checkpoint=donut_checkpoint,
            bdrc_ocr_model=bdrc_ocr_model,
            device=device,
            max_len=max_len,
            preprocess_preset=preprocess_preset,
            bdrc_preprocess_overrides=bdrc_preprocess_overrides,
            rgb_preprocess_overrides=rgb_preprocess_overrides,
        )
        new_row = dict(hit)
        new_row["text"] = text
        new_row["ocr_debug"] = dbg
        new_row["source"] = "manual_line_click"
        rows = _add_or_update_line(rows, new_row)
        state["line_rows"] = rows
        state["last_donut_pre"] = pre_img
        state["last_donut_post"] = post_img
        state["last_debug_text"] = str(text or "")
        overlay = _render_overlay(src, rows)
        return overlay, _line_text(rows), f"Re-transcribed line at click ({click_x},{click_y}).", state, json.dumps(
            {"ok": True, "mode": "manual", "action": "clicked_existing_line", "line_box": new_row["line_box"]},
            ensure_ascii=False,
            indent=2,
        ), state.get("last_donut_pre"), state.get("last_donut_post")

    anchor = state.get("manual_anchor")
    if not isinstance(anchor, (list, tuple)) or len(anchor) != 2:
        state["manual_anchor"] = [int(click_x), int(click_y)]
        overlay = _render_overlay(src, rows)
        return overlay, _line_text(rows), f"Start point set at ({click_x},{click_y}). Click a second point to define ROI.", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")

    ax, ay = int(anchor[0]), int(anchor[1])
    state["manual_anchor"] = None
    x1, x2 = sorted([ax, click_x])
    y1, y2 = sorted([ay, click_y])
    if (x2 - x1) < 3 or (y2 - y1) < 3:
        overlay = _render_overlay(src, rows)
        return overlay, _line_text(rows), "ROI is too small. Please select a larger area.", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")
    roi = _normalize_box([x1, y1, x2, y2], w, h)
    return _manual_process_roi(
        state,
        ocr_engine,
        donut_checkpoint,
        bdrc_ocr_model,
        line_segmentation_mode,
        line_segmentation_preprocess,
        line_model,
        bdrc_line_model,
        device,
        max_len,
        roi,
        preprocess_preset=preprocess_preset,
        bdrc_preprocess_overrides=bdrc_preprocess_overrides,
        rgb_preprocess_overrides=rgb_preprocess_overrides,
    )


def _manual_process_roi(
    state: Dict[str, Any],
    ocr_engine: str,
    donut_checkpoint: str,
    bdrc_ocr_model: str,
    line_segmentation_mode: str,
    line_segmentation_preprocess: str,
    line_model: str,
    bdrc_line_model: str,
    device: str,
    max_len: int,
    roi: Optional[List[int]],
    preprocess_preset: str = "bdrc",
    bdrc_preprocess_overrides: Optional[Dict[str, Any]] = None,
    rgb_preprocess_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, str, str, Dict[str, Any], str, Optional[np.ndarray], Optional[np.ndarray]]:
    image = state.get("image")
    if image is None:
        return None, "", "Please upload an image first.", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")
    engine = _normalize_ocr_engine(ocr_engine)
    auto_notes: List[str] = []
    if engine == OCR_ENGINE_BDRC and not bdrc_ocr_model.strip():
        try:
            bdrc_ocr_model, note = _ensure_default_bdrc_ocr_model_path(bdrc_ocr_model)
            if note:
                auto_notes.append(note)
        except Exception as exc:
            return None, "", f"BDRC OCR model is missing and auto-download failed: {type(exc).__name__}: {exc}", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")
    mode = _normalize_line_segmentation_mode(line_segmentation_mode)
    if mode == LINE_SEG_BDRC and not (bdrc_line_model or "").strip():
        try:
            bdrc_line_model, note = _ensure_default_bdrc_line_model_path(bdrc_line_model)
            if note:
                auto_notes.append(note)
        except Exception as exc:
            return None, "", f"BDRC line model is missing and auto-download failed: {type(exc).__name__}: {exc}", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")
    src = np.asarray(image).astype(np.uint8, copy=False)
    h, w = src.shape[:2]
    rows = list(state.get("line_rows") or [])

    roi = _normalize_box(roi or [], w, h)
    if roi is None:
        overlay = _render_overlay(src, rows)
        return overlay, _line_text(rows), "ROI is invalid.", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")
    rx1, ry1, rx2, ry2 = roi
    roi_crop = src[ry1:ry2, rx1:rx2]
    roi_w = int(rx2 - rx1)
    roi_h = int(ry2 - ry1)

    created_rows: List[Dict[str, Any]] = []
    last_pre: Optional[np.ndarray] = None
    last_post: Optional[np.ndarray] = None
    if _roi_is_single_line(roi_w, roi_h):
        text, dbg, pre_img, post_img = _run_ocr_on_crop(
            roi_crop,
            ocr_engine=engine,
            donut_checkpoint=donut_checkpoint,
            bdrc_ocr_model=bdrc_ocr_model,
            device=device,
            max_len=max_len,
            preprocess_preset=preprocess_preset,
            bdrc_preprocess_overrides=bdrc_preprocess_overrides,
            rgb_preprocess_overrides=rgb_preprocess_overrides,
        )
        last_pre, last_post = pre_img, post_img
        created_rows.append(
            {
                "line_id": len(rows) + 1,
                "line_box": [rx1, ry1, rx2, ry2],
                "text": text,
                "ocr_debug": dbg,
                "source": "manual_roi_single_line",
            }
        )
        action = "roi_direct_line"
    else:
        line_rows_local: List[Dict[str, Any]] = []
        if mode == LINE_SEG_YOLO and (line_model or "").strip():
            preprocess_mode = normalize_line_segmentation_preprocess_pipeline(line_segmentation_preprocess)
            try:
                predictions = predict_line_regions(
                    roi_crop,
                    model_path=line_model,
                    conf=DEFAULT_LINE_SEGMENTATION_CONF,
                    imgsz=DEFAULT_LINE_SEGMENTATION_IMGSZ,
                    preprocess_pipeline=preprocess_mode,
                    device=device,
                )
                line_rows_local = _rows_from_line_predictions(
                    predictions,
                    image_w=roi_w,
                    image_h=roi_h,
                    source="manual_roi_yolo_line_model",
                )
            except Exception:
                line_rows_local = []
        elif mode == LINE_SEG_BDRC and (bdrc_line_model or "").strip():
            try:
                predictions, _dbg = predict_bdrc_line_regions(
                    roi_crop,
                    model_path=bdrc_line_model,
                    device=device,
                )
                line_rows_local = _rows_from_bdrc_line_predictions(
                    predictions,
                    image_w=roi_w,
                    image_h=roi_h,
                    source="manual_roi_bdrc_line_model",
                )
            except Exception:
                line_rows_local = []
        elif mode == LINE_SEG_CLASSICAL:
            line_state = _compute_line_projection_state(
                crop_rgb=roi_crop,
                projection_smooth=9,
                projection_threshold_rel=0.20,
            )
            line_boxes_local = _segment_lines_in_text_crop(
                crop_rgb=roi_crop,
                min_line_height=10,
                projection_smooth=9,
                projection_threshold_rel=0.20,
                merge_gap_px=5,
                projection_state=line_state,
            )
            for bx1, by1, bx2, by2 in line_boxes_local:
                line_rows_local.append(
                    {
                        "line_box": [int(bx1), int(by1), int(bx2), int(by2)],
                        "line_polygon": [
                            [int(bx1), int(by1)],
                            [int(bx2) - 1, int(by1)],
                            [int(bx2) - 1, int(by2) - 1],
                            [int(bx1), int(by2) - 1],
                        ],
                        "source": "manual_roi_line_split",
                    }
                )

        for local_rec in line_rows_local:
            local_box = local_rec.get("line_box") or []
            if len(local_box) != 4:
                continue
            bx1, by1, bx2, by2 = [int(v) for v in local_box]
            gx1, gy1, gx2, gy2 = rx1 + int(bx1), ry1 + int(by1), rx1 + int(bx2), ry1 + int(by2)
            gbox = _normalize_box([gx1, gy1, gx2, gy2], w, h)
            if gbox is None:
                continue
            local_copy = dict(local_rec)
            local_copy["line_box"] = gbox
            lcrop = _extract_row_crop(src, local_copy)
            text, dbg, pre_img, post_img = _run_ocr_on_crop(
                lcrop,
                ocr_engine=engine,
                donut_checkpoint=donut_checkpoint,
                bdrc_ocr_model=bdrc_ocr_model,
                device=device,
                max_len=max_len,
                preprocess_preset=preprocess_preset,
                bdrc_preprocess_overrides=bdrc_preprocess_overrides,
                rgb_preprocess_overrides=rgb_preprocess_overrides,
            )
            last_pre, last_post = pre_img, post_img
            created_rows.append(
                {
                    "line_id": len(rows) + len(created_rows) + 1,
                    "line_box": gbox,
                    "line_polygon": [
                        [int(rx1) + int(x), int(ry1) + int(y)]
                        for x, y in list(local_rec.get("line_polygon") or [])
                    ] or None,
                    "line_confidence": float(local_rec.get("line_confidence", 0.0)),
                    "line_label": str(local_rec.get("line_label", "line")),
                    "line_class": int(local_rec.get("line_class", 0)),
                    "text": text,
                    "ocr_debug": dbg,
                    "ocr_engine": engine,
                    "ocr_crop": local_rec.get("ocr_crop"),
                    "source": str(local_rec.get("source") or "manual_roi_line_split"),
                }
            )
        if not created_rows:
            text, dbg, pre_img, post_img = _run_ocr_on_crop(
                roi_crop,
                ocr_engine=engine,
                donut_checkpoint=donut_checkpoint,
                bdrc_ocr_model=bdrc_ocr_model,
                device=device,
                max_len=max_len,
                preprocess_preset=preprocess_preset,
                bdrc_preprocess_overrides=bdrc_preprocess_overrides,
                rgb_preprocess_overrides=rgb_preprocess_overrides,
            )
            last_pre, last_post = pre_img, post_img
            created_rows.append(
                {
                    "line_id": len(rows) + 1,
                    "line_box": [rx1, ry1, rx2, ry2],
                    "text": text,
                    "ocr_debug": dbg,
                    "source": "manual_roi_fallback_direct",
                }
            )
        action = (
            "roi_line_split_yolo_model"
            if mode == LINE_SEG_YOLO
            else ("roi_line_split_bdrc_model" if mode == LINE_SEG_BDRC else "roi_line_split")
        )

    for rec in created_rows:
        rows = _add_or_update_line(rows, rec)

    state["line_rows"] = rows
    state["last_mode"] = "manual"
    state["last_donut_pre"] = last_pre
    state["last_donut_post"] = last_post
    state["last_debug_text"] = str(created_rows[-1].get("text", "") if created_rows else "")
    overlay = _render_overlay(src, rows, roi=roi)
    debug = {
        "ok": True,
        "mode": "manual",
        "action": action,
        "roi": roi,
        "new_rows": created_rows,
        "total_rows": len(rows),
        "ocr_engine": engine,
        "line_segmentation_mode": mode,
        "line_segmentation_preprocess": (
            normalize_line_segmentation_preprocess_pipeline(line_segmentation_preprocess)
            if mode == LINE_SEG_YOLO
            else ("bdrc_internal" if mode == LINE_SEG_BDRC else "none")
        ),
        "line_segmentation_model": (
            str(line_model or "")
            if mode == LINE_SEG_YOLO
            else (str(bdrc_line_model or "") if mode == LINE_SEG_BDRC else "")
        ),
        "image_preprocess_pipeline": _normalize_preprocess_preset(preprocess_preset),
        "preprocess_overrides": _effective_preprocess_overrides(
            preprocess_preset=preprocess_preset,
            bdrc_preprocess_overrides=bdrc_preprocess_overrides,
            rgb_preprocess_overrides=rgb_preprocess_overrides,
        ),
        "bdrc_preprocess_overrides": (
            dict(bdrc_preprocess_overrides)
            if isinstance(bdrc_preprocess_overrides, dict)
            else None
        ),
        "rgb_preprocess_overrides": (
            dict(rgb_preprocess_overrides)
            if isinstance(rgb_preprocess_overrides, dict)
            else None
        ),
        "auto_download_notes": auto_notes or None,
    }
    status_msg = f"Manual ROI processed. New/updated lines: {len(created_rows)}."
    if auto_notes:
        status_msg += " " + " ".join(auto_notes)
    return (
        overlay,
        _line_text(rows),
        status_msg,
        state,
        json.dumps(debug, ensure_ascii=False, indent=2),
        state.get("last_donut_pre"),
        state.get("last_donut_post"),
    )


def _manual_full_image_roi(
    mode_s: str,
    state_s: Dict[str, Any],
    ocr_engine_s: str,
    donut_s: str,
    bdrc_ocr_model_s: str,
    line_segmentation_mode_s: str,
    line_segmentation_preprocess_s: str,
    line_model_s: str,
    bdrc_line_model_s: str,
    device_s: str,
    max_len_s: int,
    preprocess_preset: str = "bdrc",
    bdrc_preprocess_overrides: Optional[Dict[str, Any]] = None,
    rgb_preprocess_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, str, str, Dict[str, Any], str, Optional[np.ndarray], Optional[np.ndarray]]:
    if not _is_manual_mode(mode_s):
        img = state_s.get("image")
        overlay = np.asarray(img).astype(np.uint8, copy=False) if img is not None else None
        return overlay, _line_text(state_s.get("line_rows") or []), "This button is only active in Manual Mode.", state_s, "{}", state_s.get("last_donut_pre"), state_s.get("last_donut_post")
    img = state_s.get("image")
    if img is None:
        return None, "", "Please upload an image first.", state_s, "{}", state_s.get("last_donut_pre"), state_s.get("last_donut_post")
    src = np.asarray(img).astype(np.uint8, copy=False)
    h, w = src.shape[:2]
    return _manual_process_roi(
        state_s,
        ocr_engine_s,
        donut_s,
        bdrc_ocr_model_s,
        line_segmentation_mode_s,
        line_segmentation_preprocess_s,
        line_model_s,
        bdrc_line_model_s,
        device_s,
        int(max_len_s),
        [0, 0, w, h],
        preprocess_preset=preprocess_preset,
        bdrc_preprocess_overrides=bdrc_preprocess_overrides,
        rgb_preprocess_overrides=rgb_preprocess_overrides,
    )


def _on_upload(img_array: Any, state: Dict[str, Any]) -> Tuple[np.ndarray, str, Dict[str, Any], str]:
    """Accept a numpy array from gr.Image upload."""
    if img_array is None:
        return None, "", _base_state(), "No image loaded."
    try:
        img = np.asarray(img_array).astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
    except Exception as exc:
        return None, "", _base_state(), f"Failed to load image: {exc}"
    new_state = _base_state()
    new_state["image_path"] = ""
    new_state["image_name"] = "uploaded_image.png"
    new_state["image"] = img
    return img, "", new_state, "Image loaded."


def _save_results(
    state: Dict[str, Any],
    transcript_text: str,
    ocr_engine: str,
    donut_checkpoint: str,
    bdrc_ocr_model: str,
    layout_model: str,
    line_segmentation_mode: str,
    line_segmentation_preprocess: str,
    line_model: str,
    bdrc_line_model: str,
) -> str:
    image = state.get("image")
    if image is None:
        return "Nothing to save: no image loaded."
    name = str(state.get("image_name") or "image.png")
    stem = Path(name).stem
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (ROOT / "output" / "ocr" / f"{stem}_{ts}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    src = np.asarray(image).astype(np.uint8, copy=False)
    rows = _sort_lines(list(state.get("line_rows") or []))
    overlay = _render_overlay(src, rows)
    Image.fromarray(src).save(out_dir / "source.png")
    Image.fromarray(overlay).save(out_dir / "overlay.png")
    (out_dir / "transcript.txt").write_text(str(transcript_text or ""), encoding="utf-8")

    saved_rows: List[Dict[str, Any]] = []
    h, w = src.shape[:2]
    for i, rec in enumerate(rows, start=1):
        box = _normalize_box(rec.get("line_box") or [], w, h)
        if box is None:
            continue
        crop = _extract_row_crop(src, rec)
        crop_name = f"line_{i:03d}.png"
        Image.fromarray(crop).save(out_dir / crop_name)
        obj = dict(rec)
        obj.pop("ocr_crop", None)
        obj["line_no"] = i
        obj["line_box"] = box
        obj["line_crop_file"] = crop_name
        saved_rows.append(obj)

    normalized_engine = _normalize_ocr_engine(ocr_engine)
    normalized_mode = _normalize_line_segmentation_mode(line_segmentation_mode)
    payload = {
        "saved_at": datetime.now().isoformat(),
        "image_name": name,
        "image_path": str(state.get("image_path") or ""),
        "ocr_engine": normalized_engine,
        "ocr_model": (
            str(donut_checkpoint or "")
            if normalized_engine == OCR_ENGINE_DONUT
            else str(bdrc_ocr_model or "")
        ),
        "donut_checkpoint": str(donut_checkpoint or ""),
        "bdrc_ocr_model": str(bdrc_ocr_model or ""),
        "layout_model": str(layout_model or ""),
        "line_segmentation_mode": normalized_mode,
        "line_segmentation_preprocess": (
            normalize_line_segmentation_preprocess_pipeline(line_segmentation_preprocess)
            if normalized_mode == LINE_SEG_YOLO
            else ("bdrc_internal" if normalized_mode == LINE_SEG_BDRC else "none")
        ),
        "line_segmentation_model": (
            str(line_model or "")
            if normalized_mode == LINE_SEG_YOLO
            else (str(bdrc_line_model or "") if normalized_mode == LINE_SEG_BDRC else "")
        ),
        "yolo_line_segmentation_model": str(line_model or ""),
        "bdrc_line_model": str(bdrc_line_model or ""),
        "line_count": len(saved_rows),
        "lines": saved_rows,
    }
    (out_dir / "line_boxes_ocr.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"Saved to: {out_dir}"


def build_ui() -> gr.Blocks:
    # ── Compute startup defaults ─────────────────────────────────────────
    # Discover the best checkpoint first so we can load its repro pack
    # immediately and use those values as the initial slider values.
    _startup_donut_choices, _ = _list_donut_checkpoints()
    _startup_donut_path = _startup_donut_choices[0][1] if _startup_donut_choices else ""
    _startup_preset, _, _startup_bdrc, _startup_rgb = _load_repro_ui_settings(_startup_donut_path)

    bdrc_defaults = _startup_bdrc
    rgb_defaults = _startup_rgb
    ui_css = """
/* ── Folder Browser ─────────────────────────────────────────────────── */
#folder_gallery {
  min-height: 160px;
}
#folder_gallery .thumbnail-item {
  cursor: pointer;
  border-radius: 6px;
  overflow: hidden;
  transition: box-shadow 0.15s ease;
}
#folder_gallery .thumbnail-item:hover {
  box-shadow: 0 0 0 3px #6366f1;
}
#folder_gallery .thumbnail-item.selected {
  box-shadow: 0 0 0 4px #6366f1;
}
#folder_path_input {
  font-family: monospace;
}
/* ── Main image panel ───────────────────────────────────────────────── */
#ocr_image_panel {
  width: 100% !important;
  border: 1px dashed #cbd5e1;
  border-radius: 8px;
  overflow: auto;
}
#ocr_image_panel .image-container {
  overflow: auto !important;
  width: 100% !important;
}
#ocr_image_panel img {
  max-width: none !important;
  max-height: none !important;
  display: block;
  transform-origin: top left;
}
#img_zoom_row {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 4px;
}
#img_zoom_minus, #img_zoom_plus, #img_zoom_reset {
  min-width: 36px !important;
  width: 36px !important;
  padding: 0 !important;
}
#transcript_panel {
  min-height: 200px;
  resize: vertical;
  overflow: auto;
  border: 1px dashed #cbd5e1;
  border-radius: 8px;
  padding: 4px;
}
#transcript_box textarea {
  font-size: 18px !important;
  line-height: 1.45 !important;
  min-height: 200px !important;
  resize: both !important;
  overflow: auto !important;
}
#font_btn_plus, #font_btn_minus {
  min-width: 36px !important;
  width: 36px !important;
  padding: 0 !important;
}
#font_ctrl_col {
  max-width: 24px;
  min-width: 24px;
}
#debug_font_ctrl_col {
  max-width: 24px;
  min-width: 24px;
}
#debug_text_box textarea {
  font-size: 20px !important;
  line-height: 1.45 !important;
}
#donut_input_before,
#donut_input_after {
  margin-top: 15px !important;
  margin-bottom: 15px !important;
  padding-top: 15px !important;
}
#donut_input_before .image-container,
#donut_input_after .image-container {
  padding-top: 15px !important;
}
"""

    _image_zoom_js = """
function() {
  function initImageZoom() {
    const panel = document.querySelector('#ocr_image_panel');
    if (!panel || panel._zoomInit) return;
    panel._zoomInit = true;

    let scale = 1.0;
    let originX = 0, originY = 0;   // transform-origin in px relative to img
    let panX = 0, panY = 0;          // translate offset
    let isPanning = false;
    let panStartX = 0, panStartY = 0;
    let panOriginX = 0, panOriginY = 0;

    function getImg() { return panel.querySelector('img'); }

    function applyTransform() {
      const img = getImg();
      if (!img) return;
      img.style.transformOrigin = `${originX}px ${originY}px`;
      img.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
      img.style.maxWidth = 'none';
      img.style.maxHeight = 'none';
      img.style.display = 'block';
    }

    // Wheel zoom centered on cursor
    panel.addEventListener('wheel', function(e) {
      const img = getImg();
      if (!img) return;
      e.preventDefault();
      const rect = img.getBoundingClientRect();
      // cursor position relative to the image element (before transform)
      const mouseX = (e.clientX - rect.left) / scale;
      const mouseY = (e.clientY - rect.top)  / scale;
      const delta = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      const newScale = Math.min(16, Math.max(0.1, scale * delta));
      // Adjust pan so the point under the cursor stays fixed
      panX = e.clientX - rect.left - mouseX * newScale + panX - (e.clientX - rect.left - mouseX * scale);
      panY = e.clientY - rect.top  - mouseY * newScale + panY - (e.clientY - rect.top  - mouseY * scale);
      originX = mouseX;
      originY = mouseY;
      scale = newScale;
      applyTransform();
    }, { passive: false });

    // Middle-click or Alt+left-click drag to pan
    panel.addEventListener('mousedown', function(e) {
      if (e.button === 1 || (e.button === 0 && e.altKey)) {
        e.preventDefault();
        isPanning = true;
        panStartX = e.clientX;
        panStartY = e.clientY;
        panOriginX = panX;
        panOriginY = panY;
        panel.style.cursor = 'grabbing';
      }
    });
    window.addEventListener('mousemove', function(e) {
      if (!isPanning) return;
      panX = panOriginX + (e.clientX - panStartX);
      panY = panOriginY + (e.clientY - panStartY);
      applyTransform();
    });
    window.addEventListener('mouseup', function(e) {
      if (isPanning) {
        isPanning = false;
        panel.style.cursor = '';
      }
    });

    // Expose reset/zoom for the buttons
    panel._zoomReset = function() { scale=1; panX=0; panY=0; originX=0; originY=0; applyTransform(); };
    panel._zoomIn    = function() { scale=Math.min(16, Math.round((scale+0.25)*100)/100); applyTransform(); };
    panel._zoomOut   = function() { scale=Math.max(0.1, Math.round((scale-0.25)*100)/100); applyTransform(); };
  }

  // Re-init whenever Gradio re-renders the image
  const observer = new MutationObserver(initImageZoom);
  observer.observe(document.body, { childList: true, subtree: true });
  initImageZoom();
}
"""

    with gr.Blocks(title="OCR Workbench", css=ui_css, js=_image_zoom_js) as demo:
        donut_ckpt_choices, donut_msg = _list_donut_checkpoints()
        layout_model_choices, layout_msg = _list_layout_models()
        line_model_choices, line_model_msg = _list_line_segmentation_models()
        bdrc_line_model_choices, bdrc_line_model_msg = _list_bdrc_line_models()
        bdrc_ocr_model_choices, bdrc_ocr_model_msg = _list_bdrc_ocr_models()
        _default_donut = donut_ckpt_choices[0][1] if donut_ckpt_choices else ""
        _default_layout = layout_model_choices[0][1] if layout_model_choices else ""
        _default_line_model = line_model_choices[0][1] if line_model_choices else ""
        _default_bdrc_line_model = bdrc_line_model_choices[0][1] if bdrc_line_model_choices else ""
        _default_bdrc_ocr_model = bdrc_ocr_model_choices[0][1] if bdrc_ocr_model_choices else ""
        _donut_label = donut_ckpt_choices[0][0] if donut_ckpt_choices else "—"
        _layout_label = layout_model_choices[0][0] if layout_model_choices else "—"
        _line_label = line_model_choices[0][0] if line_model_choices else "—"
        _bdrc_line_label = bdrc_line_model_choices[0][0] if bdrc_line_model_choices else "—"
        _bdrc_ocr_label = bdrc_ocr_model_choices[0][0] if bdrc_ocr_model_choices else "—"

        # ── Top bar: title + model status + advanced toggle ─────────────────
        with gr.Row(equal_height=True):
            with gr.Column(scale=6):
                gr.Markdown("## OCR Workbench")
            with gr.Column(scale=4):
                # ── Compact model info (shown in simple mode) ───────────────
                with gr.Row() as model_info_row:
                    gr.Markdown(
                        f"🤖 DONUT `{_donut_label}`  &nbsp;|&nbsp;  🪵 BDRC OCR `{_bdrc_ocr_label}`"
                        f"  &nbsp;|&nbsp;  📐 Layout `{_layout_label}`  &nbsp;|&nbsp;  "
                        f"📏 YOLO `{_line_label}` / BDRC `{_bdrc_line_label}`",
                        elem_id="model_info_md",
                    )
            with gr.Column(scale=1, min_width=140):
                advanced_view = gr.Checkbox(label="Advanced View", value=False)

        gr.Markdown(
            "Automatic mode: choose classical CV, a pretrained YOLO line model, or the BDRC line model, "
            "then run DONUT or BDRC OCR. Manual mode: click an existing line or define an ROI with two clicks."
        )

        # ── Folder Browser ──────────────────────────────────────────────────
        with gr.Accordion("📁 Folder Browser", open=True):
            gr.Markdown(
                "Enter a folder path to browse images. "
                "Click a thumbnail to load it into the workbench below and automatically "
                "run the selected line segmentation backend + OCR engine."
            )
            with gr.Row():
                folder_path_input = gr.Textbox(
                    label="Image Folder",
                    placeholder="/path/to/your/images",
                    elem_id="folder_path_input",
                    scale=8,
                )
                folder_scan_btn = gr.Button("🔍 Scan", variant="primary", scale=1, min_width=80)
            folder_status = gr.Textbox(label="Status", interactive=False)
            # Hidden state: list of full image paths in the scanned folder
            folder_image_paths = gr.State([])
            folder_gallery = gr.Gallery(
                label="Images",
                elem_id="folder_gallery",
                columns=6,
                rows=3,
                object_fit="contain",
                height="auto",
                allow_preview=False,
            )

        # ── Advanced model selection (hidden in simple mode) ────────────────
        with gr.Row(visible=False) as advanced_model_row:
            with gr.Column(scale=10):
                # choices = [(label, full_path), ...]; value = full_path string
                donut_path = gr.Dropdown(
                    label="DONUT Checkpoint",
                    choices=donut_ckpt_choices,
                    value=_default_donut,
                    allow_custom_value=True,
                )
            with gr.Column(scale=1, min_width=110):
                donut_refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
                load_repro_btn = gr.Button("📋 Load repro settings", variant="secondary")
            with gr.Column(scale=10):
                # choices = [(label, full_path), ...]; value = full_path string
                layout_path = gr.Dropdown(
                    label="Layout Analysis Model",
                    choices=layout_model_choices,
                    value=_default_layout,
                    allow_custom_value=True,
                )
            with gr.Column(scale=1, min_width=110):
                layout_refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
        with gr.Row(visible=False) as advanced_bdrc_model_row:
            with gr.Column(scale=10):
                bdrc_ocr_model_path = gr.Dropdown(
                    label="BDRC OCR Model",
                    choices=bdrc_ocr_model_choices,
                    value=_default_bdrc_ocr_model,
                    allow_custom_value=True,
                )
            with gr.Column(scale=1, min_width=110):
                bdrc_ocr_refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
            with gr.Column(scale=10):
                bdrc_line_model_path = gr.Dropdown(
                    label="BDRC Line Model",
                    choices=bdrc_line_model_choices,
                    value=_default_bdrc_line_model,
                    allow_custom_value=True,
                )
            with gr.Column(scale=1, min_width=110):
                bdrc_line_model_refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
        with gr.Row():
            ocr_engine = gr.Radio(
                choices=[OCR_ENGINE_DONUT, OCR_ENGINE_BDRC],
                value=OCR_ENGINE_DONUT,
                label="OCR Engine",
            )
        with gr.Row():
            line_segmentation_mode = gr.Radio(
                choices=[LINE_SEG_CLASSICAL, LINE_SEG_YOLO, LINE_SEG_BDRC],
                value=LINE_SEG_CLASSICAL,
                label="Line Segmentation",
            )
        with gr.Row():
            with gr.Column(scale=10):
                line_model_path = gr.Dropdown(
                    label="Line Segmentation Model",
                    choices=line_model_choices,
                    value=_default_line_model,
                    allow_custom_value=True,
                )
            with gr.Column(scale=1, min_width=110):
                line_model_refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
        with gr.Row(visible=False) as advanced_scan_row:
            donut_info = gr.Textbox(label="DONUT Auto-Scan", value=donut_msg, interactive=False)
            layout_info = gr.Textbox(label="Layout Auto-Scan", value=layout_msg, interactive=False)
            line_model_info = gr.Textbox(label="Line Model Auto-Scan", value=line_model_msg, interactive=False)
            bdrc_ocr_info = gr.Textbox(label="BDRC OCR Auto-Scan", value=bdrc_ocr_model_msg, interactive=False)
            bdrc_line_model_info = gr.Textbox(label="BDRC Line Auto-Scan", value=bdrc_line_model_msg, interactive=False)

        with gr.Row():
            mode = gr.Radio(choices=[MODE_AUTO, MODE_MANUAL], value=MODE_AUTO, label="Mode")
        with gr.Row(visible=False) as advanced_runtime_row:
            device = gr.Dropdown(choices=["auto", "cuda:0", "cpu"], value="auto", label="Inference Device")
            max_len = gr.Number(label="OCR generation_max_length (DONUT only)", value=160, precision=0)
        with gr.Row(visible=False) as advanced_preprocess_select_row:
            preprocess_preset = gr.Dropdown(
                choices=[("bdrc", "bdrc"), ("gray", "gray"), ("RGB", "rgb")],
                value=_startup_preset,
                label="Preprocess Preset",
            )
            line_model_preprocess = gr.Dropdown(
                choices=[("none", "none"), ("bdrc", "bdrc"), ("gray", "gray"), ("RGB", "rgb")],
                value=DEFAULT_LINE_SEGMENTATION_PREPROCESS,
                label="Line Model Preprocess (YOLO only)",
            )
        with gr.Row(visible=False) as advanced_bdrc_row:
            with gr.Accordion("BDRC / Gray Preprocess (OCR Input)", open=False):
                gr.Markdown(
                    "1) Background normalize  2) Thresholding  3) Morph close + tiny component filter  "
                    "4) Upscale before binarization  5) Gray channel mode\n\n"
                    "Gray preset uses this config with `binarize=false`."
                )
                with gr.Row():
                    bdrc_gray_mode = gr.Dropdown(
                        choices=["luma", "min_rgb", "max_rgb", "r", "g", "b"],
                        value=str(bdrc_defaults["gray_mode"]),
                        label="5) gray_mode",
                    )
                    bdrc_upscale_factor = gr.Slider(
                        minimum=1.0,
                        maximum=3.0,
                        step=0.1,
                        value=float(bdrc_defaults["upscale_factor"]),
                        label="4) upscale_factor",
                    )
                    bdrc_upscale_interp = gr.Dropdown(
                        choices=["lanczos", "cubic", "linear", "nearest"],
                        value=str(bdrc_defaults["upscale_interpolation"]),
                        label="4) upscale_interpolation",
                    )
                with gr.Row():
                    bdrc_normalize_bg = gr.Checkbox(
                        value=bool(bdrc_defaults["normalize_background"]),
                        label="1) normalize_background",
                    )
                    bdrc_bg_blur_ksize = gr.Number(
                        value=int(bdrc_defaults["background_blur_ksize"]),
                        precision=0,
                        label="1) background_blur_ksize (0=auto)",
                    )
                    bdrc_bg_strength = gr.Slider(
                        minimum=0.0,
                        maximum=3.0,
                        step=0.05,
                        value=float(bdrc_defaults["background_strength"]),
                        label="1) background_strength",
                    )
                with gr.Row():
                    bdrc_binarize = gr.Checkbox(
                        value=bool(bdrc_defaults["binarize"]),
                        label="2) binarize",
                    )
                    bdrc_threshold_method = gr.Dropdown(
                        choices=["adaptive", "otsu", "fixed"],
                        value=str(bdrc_defaults["threshold_method"]),
                        label="2) threshold_method",
                    )
                    bdrc_fixed_threshold = gr.Slider(
                        minimum=0,
                        maximum=255,
                        step=1,
                        value=int(bdrc_defaults["fixed_threshold"]),
                        label="2) fixed_threshold",
                    )
                with gr.Row():
                    bdrc_threshold_block = gr.Number(
                        value=int(bdrc_defaults["threshold_block_size"]),
                        precision=0,
                        label="2) threshold_block_size (odd)",
                    )
                    bdrc_threshold_c = gr.Number(
                        value=int(bdrc_defaults["threshold_c"]),
                        precision=0,
                        label="2) threshold_c",
                    )
                with gr.Row():
                    bdrc_morph_close = gr.Checkbox(
                        value=bool(bdrc_defaults["morph_close"]),
                        label="3) morph_close",
                    )
                    bdrc_morph_close_kernel = gr.Number(
                        value=int(bdrc_defaults["morph_close_kernel"]),
                        precision=0,
                        label="3) morph_close_kernel (odd)",
                    )
                    bdrc_remove_small_components = gr.Checkbox(
                        value=bool(bdrc_defaults["remove_small_components"]),
                        label="3) remove_small_components",
                    )
                    bdrc_min_component_area = gr.Number(
                        value=int(bdrc_defaults["min_component_area"]),
                        precision=0,
                        label="3) min_component_area",
                    )
        with gr.Row(visible=False) as advanced_rgb_row:
            with gr.Accordion("RGB Preprocess (OCR Input)", open=False):
                gr.Markdown(
                    "RGB pipeline keeps color cues and exposes background correction, denoising, "
                    "ink normalization, morphology and upscale options."
                )
                with gr.Row():
                    rgb_preserve_color = gr.Checkbox(
                        value=bool(rgb_defaults["preserve_color"]),
                        label="preserve_color",
                    )
                    rgb_normalize_bg = gr.Checkbox(
                        value=bool(rgb_defaults["normalize_background"]),
                        label="normalize_background",
                    )
                    rgb_background_method = gr.Dropdown(
                        choices=["shade_correct", "rolling_ball_like", "top_hat", "none"],
                        value=str(rgb_defaults["background_method"]),
                        label="background_method",
                    )
                with gr.Row():
                    rgb_bg_blur_ksize = gr.Number(
                        value=int(rgb_defaults["background_blur_ksize"]),
                        precision=0,
                        label="background_blur_ksize (0=auto)",
                    )
                    rgb_bg_strength = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=float(rgb_defaults["background_strength"]),
                        label="background_strength",
                    )
                    rgb_contrast = gr.Slider(
                        minimum=0.5,
                        maximum=2.5,
                        step=0.05,
                        value=float(rgb_defaults["contrast"]),
                        label="contrast",
                    )
                with gr.Row():
                    rgb_denoise = gr.Checkbox(
                        value=bool(rgb_defaults["denoise"]),
                        label="denoise",
                    )
                    rgb_morph_close = gr.Checkbox(
                        value=bool(rgb_defaults["morph_close"]),
                        label="morph_close",
                    )
                    rgb_morph_close_kernel = gr.Number(
                        value=int(rgb_defaults["morph_close_kernel"]),
                        precision=0,
                        label="morph_close_kernel (odd)",
                    )
                with gr.Row():
                    rgb_remove_small_components = gr.Checkbox(
                        value=bool(rgb_defaults["remove_small_components"]),
                        label="remove_small_components",
                    )
                    rgb_min_component_area = gr.Number(
                        value=int(rgb_defaults["min_component_area"]),
                        precision=0,
                        label="min_component_area",
                    )
                    rgb_ink_normalization = gr.Checkbox(
                        value=bool(rgb_defaults["ink_normalization"]),
                        label="ink_normalization",
                    )
                    rgb_ink_strength = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=float(rgb_defaults["ink_strength"]),
                        label="ink_strength",
                    )
                with gr.Row():
                    rgb_upscale_factor = gr.Slider(
                        minimum=1.0,
                        maximum=3.0,
                        step=0.1,
                        value=float(rgb_defaults["upscale_factor"]),
                        label="upscale_factor",
                    )
                    rgb_upscale_interp = gr.Dropdown(
                        choices=["lanczos", "cubic", "linear", "nearest"],
                        value=str(rgb_defaults["upscale_interpolation"]),
                        label="upscale_interpolation",
                    )

        status = gr.Textbox(label="Status", interactive=False)
        with gr.Accordion("Debug JSON", open=False, visible=False) as debug_json_accordion:
            debug_json = gr.Code(language="json")
        with gr.Column():
            donut_input_before = gr.Image(
                label="OCR Input (Before Preprocess)",
                type="numpy",
                visible=False,
                elem_id="donut_input_before",
            )
            donut_input_after = gr.Image(
                label="OCR Input (After Preprocess)",
                type="numpy",
                visible=False,
                elem_id="donut_input_after",
            )
        with gr.Row(visible=False) as advanced_debug_text_row:
            debug_text = gr.Textbox(label="Debug Transcription", lines=4, elem_id="debug_text_box", interactive=True)
            with gr.Column(elem_id="debug_font_ctrl_col", scale=1, min_width=24):
                debug_font_plus_btn = gr.Button("+")
                debug_font_minus_btn = gr.Button("−")

        state = gr.State(_base_state())
        with gr.Row():
            run_btn = gr.Button("Run OCR", variant="primary")
            compare_btn = gr.Button("⚖ Compare All Models", variant="secondary")
            full_roi_btn = gr.Button("Process Full Image as ROI", visible=False)
            save_btn = gr.Button("Save", variant="secondary")

        # ── Image row (full width) ──────────────────────────────────────────
        with gr.Row(elem_id="img_zoom_row"):
            img_zoom_out_btn = gr.Button("🔍−", elem_id="img_zoom_minus", scale=0, min_width=36)
            img_zoom_reset_btn = gr.Button("⊙", elem_id="img_zoom_reset", scale=0, min_width=36)
            img_zoom_in_btn = gr.Button("🔍+", elem_id="img_zoom_plus", scale=0, min_width=36)
        with gr.Row():
            image_view = gr.Image(
                label="Drop image here or click to upload",
                type="numpy",
                interactive=True,
                elem_id="ocr_image_panel",
            )

        # ── Transcript row (full width) ─────────────────────────────────────
        with gr.Row():
            with gr.Column(elem_id="transcript_panel"):
                with gr.Row():
                    transcript = gr.Textbox(label="", lines=14, elem_id="transcript_box", show_label=False, scale=36)
                    with gr.Column(elem_id="font_ctrl_col", scale=1, min_width=24):
                        font_plus_btn = gr.Button("+", elem_id="font_btn_plus")
                        font_minus_btn = gr.Button("−", elem_id="font_btn_minus")
        save_status = gr.Textbox(label="Save Status", interactive=False, visible=False)

        _upload_evt = image_view.upload(
            fn=_on_upload,
            inputs=[image_view, state],
            outputs=[image_view, transcript, state, status],
        )
        _upload_evt.then(
            fn=lambda st: str((st or {}).get("last_debug_text", "") if isinstance(st, dict) else ""),
            inputs=[state],
            outputs=[debug_text],
        )

        def _run(
            mode_s: str,
            state_s: Dict[str, Any],
            ocr_engine_s: str,
            donut_s: str,
            bdrc_ocr_model_s: str,
            layout_s: str,
            line_segmentation_mode_s: str,
            line_segmentation_preprocess_s: str,
            line_model_s: str,
            bdrc_line_model_s: str,
            device_s: str,
            max_len_s: int,
            preprocess_preset_s: str,
            gray_mode_s: str,
            normalize_background_s: bool,
            background_blur_ksize_s: int,
            background_strength_s: float,
            upscale_factor_s: float,
            upscale_interpolation_s: str,
            binarize_s: bool,
            threshold_method_s: str,
            threshold_block_size_s: int,
            threshold_c_s: int,
            fixed_threshold_s: int,
            morph_close_s: bool,
            morph_close_kernel_s: int,
            remove_small_components_s: bool,
            min_component_area_s: int,
            rgb_preserve_color_s: bool,
            rgb_normalize_bg_s: bool,
            rgb_background_method_s: str,
            rgb_bg_blur_ksize_s: int,
            rgb_bg_strength_s: float,
            rgb_contrast_s: float,
            rgb_denoise_s: bool,
            rgb_morph_close_s: bool,
            rgb_morph_close_kernel_s: int,
            rgb_remove_small_components_s: bool,
            rgb_min_component_area_s: int,
            rgb_upscale_factor_s: float,
            rgb_upscale_interp_s: str,
            rgb_ink_normalization_s: bool,
            rgb_ink_strength_s: float,
        ):
            preproc_mode, bdrc_overrides, rgb_overrides = _compose_preprocess_ui_settings(
                preprocess_preset=preprocess_preset_s,
                gray_mode=gray_mode_s,
                normalize_background=normalize_background_s,
                background_blur_ksize=int(background_blur_ksize_s),
                background_strength=float(background_strength_s),
                upscale_factor=float(upscale_factor_s),
                upscale_interpolation=upscale_interpolation_s,
                binarize=binarize_s,
                threshold_method=threshold_method_s,
                threshold_block_size=int(threshold_block_size_s),
                threshold_c=int(threshold_c_s),
                fixed_threshold=int(fixed_threshold_s),
                morph_close=morph_close_s,
                morph_close_kernel=int(morph_close_kernel_s),
                remove_small_components=remove_small_components_s,
                min_component_area=int(min_component_area_s),
                rgb_preserve_color=rgb_preserve_color_s,
                rgb_normalize_background=rgb_normalize_bg_s,
                rgb_background_method=rgb_background_method_s,
                rgb_background_blur_ksize=int(rgb_bg_blur_ksize_s),
                rgb_background_strength=float(rgb_bg_strength_s),
                rgb_contrast=float(rgb_contrast_s),
                rgb_denoise=rgb_denoise_s,
                rgb_morph_close=rgb_morph_close_s,
                rgb_morph_close_kernel=int(rgb_morph_close_kernel_s),
                rgb_remove_small_components=rgb_remove_small_components_s,
                rgb_min_component_area=int(rgb_min_component_area_s),
                rgb_upscale_factor=float(rgb_upscale_factor_s),
                rgb_upscale_interpolation=rgb_upscale_interp_s,
                rgb_ink_normalization=rgb_ink_normalization_s,
                rgb_ink_strength=float(rgb_ink_strength_s),
            )
            if not _is_manual_mode(mode_s):
                return _run_full_auto(
                    state_s,
                    ocr_engine_s,
                    donut_s,
                    bdrc_ocr_model_s,
                    layout_s,
                    line_segmentation_mode_s,
                    line_segmentation_preprocess_s,
                    line_model_s,
                    bdrc_line_model_s,
                    device_s,
                    int(max_len_s),
                    preprocess_preset=preproc_mode,
                    bdrc_preprocess_overrides=bdrc_overrides,
                    rgb_preprocess_overrides=rgb_overrides,
                )
            img = state_s.get("image")
            overlay = np.asarray(img).astype(np.uint8, copy=False) if img is not None else None
            debug = {
                "ok": True,
                "mode": "manual_waiting",
                "ocr_engine": _normalize_ocr_engine(ocr_engine_s),
                "ocr_model": (
                    str(donut_s or "")
                    if _normalize_ocr_engine(ocr_engine_s) == OCR_ENGINE_DONUT
                    else str(bdrc_ocr_model_s or "")
                ),
                "line_segmentation_mode": _normalize_line_segmentation_mode(line_segmentation_mode_s),
                "line_segmentation_preprocess": (
                    normalize_line_segmentation_preprocess_pipeline(line_segmentation_preprocess_s)
                    if _normalize_line_segmentation_mode(line_segmentation_mode_s) == LINE_SEG_YOLO
                    else ("bdrc_internal" if _normalize_line_segmentation_mode(line_segmentation_mode_s) == LINE_SEG_BDRC else "none")
                ),
                "line_segmentation_model": (
                    str(line_model_s or "")
                    if _normalize_line_segmentation_mode(line_segmentation_mode_s) == LINE_SEG_YOLO
                    else (str(bdrc_line_model_s or "") if _normalize_line_segmentation_mode(line_segmentation_mode_s) == LINE_SEG_BDRC else "")
                ),
                "image_preprocess_pipeline": preproc_mode,
                "preprocess_overrides": _effective_preprocess_overrides(
                    preprocess_preset=preproc_mode,
                    bdrc_preprocess_overrides=bdrc_overrides,
                    rgb_preprocess_overrides=rgb_overrides,
                ),
                "bdrc_preprocess_overrides": bdrc_overrides,
                "rgb_preprocess_overrides": rgb_overrides,
            }
            return (
                overlay,
                _line_text(state_s.get("line_rows") or []),
                "Manual Mode: click an existing line or define ROI with two clicks.",
                state_s,
                json.dumps(debug, ensure_ascii=False, indent=2),
                state_s.get("last_donut_pre"),
                state_s.get("last_donut_post"),
            )

        _run_evt = run_btn.click(
            fn=_run,
            inputs=[
                mode,
                state,
                ocr_engine,
                donut_path,
                bdrc_ocr_model_path,
                layout_path,
                line_segmentation_mode,
                line_model_preprocess,
                line_model_path,
                bdrc_line_model_path,
                device,
                max_len,
                preprocess_preset,
                bdrc_gray_mode,
                bdrc_normalize_bg,
                bdrc_bg_blur_ksize,
                bdrc_bg_strength,
                bdrc_upscale_factor,
                bdrc_upscale_interp,
                bdrc_binarize,
                bdrc_threshold_method,
                bdrc_threshold_block,
                bdrc_threshold_c,
                bdrc_fixed_threshold,
                bdrc_morph_close,
                bdrc_morph_close_kernel,
                bdrc_remove_small_components,
                bdrc_min_component_area,
                rgb_preserve_color,
                rgb_normalize_bg,
                rgb_background_method,
                rgb_bg_blur_ksize,
                rgb_bg_strength,
                rgb_contrast,
                rgb_denoise,
                rgb_morph_close,
                rgb_morph_close_kernel,
                rgb_remove_small_components,
                rgb_min_component_area,
                rgb_upscale_factor,
                rgb_upscale_interp,
                rgb_ink_normalization,
                rgb_ink_strength,
            ],
            outputs=[image_view, transcript, status, state, debug_json, donut_input_before, donut_input_after],
        )
        _run_evt.then(
            fn=lambda st: str((st or {}).get("last_debug_text", "") if isinstance(st, dict) else ""),
            inputs=[state],
            outputs=[debug_text],
        )

        def _on_select(
            mode_s: str,
            state_s: Dict[str, Any],
            ocr_engine_s: str,
            donut_s: str,
            bdrc_ocr_model_s: str,
            layout_s: str,
            line_segmentation_mode_s: str,
            line_segmentation_preprocess_s: str,
            line_model_s: str,
            bdrc_line_model_s: str,
            device_s: str,
            max_len_s: int,
            preprocess_preset_s: str,
            gray_mode_s: str,
            normalize_background_s: bool,
            background_blur_ksize_s: int,
            background_strength_s: float,
            upscale_factor_s: float,
            upscale_interpolation_s: str,
            binarize_s: bool,
            threshold_method_s: str,
            threshold_block_size_s: int,
            threshold_c_s: int,
            fixed_threshold_s: int,
            morph_close_s: bool,
            morph_close_kernel_s: int,
            remove_small_components_s: bool,
            min_component_area_s: int,
            rgb_preserve_color_s: bool,
            rgb_normalize_bg_s: bool,
            rgb_background_method_s: str,
            rgb_bg_blur_ksize_s: int,
            rgb_bg_strength_s: float,
            rgb_contrast_s: float,
            rgb_denoise_s: bool,
            rgb_morph_close_s: bool,
            rgb_morph_close_kernel_s: int,
            rgb_remove_small_components_s: bool,
            rgb_min_component_area_s: int,
            rgb_upscale_factor_s: float,
            rgb_upscale_interp_s: str,
            rgb_ink_normalization_s: bool,
            rgb_ink_strength_s: float,
            evt: gr.SelectData,
        ):
            preproc_mode, bdrc_overrides, rgb_overrides = _compose_preprocess_ui_settings(
                preprocess_preset=preprocess_preset_s,
                gray_mode=gray_mode_s,
                normalize_background=normalize_background_s,
                background_blur_ksize=int(background_blur_ksize_s),
                background_strength=float(background_strength_s),
                upscale_factor=float(upscale_factor_s),
                upscale_interpolation=upscale_interpolation_s,
                binarize=binarize_s,
                threshold_method=threshold_method_s,
                threshold_block_size=int(threshold_block_size_s),
                threshold_c=int(threshold_c_s),
                fixed_threshold=int(fixed_threshold_s),
                morph_close=morph_close_s,
                morph_close_kernel=int(morph_close_kernel_s),
                remove_small_components=remove_small_components_s,
                min_component_area=int(min_component_area_s),
                rgb_preserve_color=rgb_preserve_color_s,
                rgb_normalize_background=rgb_normalize_bg_s,
                rgb_background_method=rgb_background_method_s,
                rgb_background_blur_ksize=int(rgb_bg_blur_ksize_s),
                rgb_background_strength=float(rgb_bg_strength_s),
                rgb_contrast=float(rgb_contrast_s),
                rgb_denoise=rgb_denoise_s,
                rgb_morph_close=rgb_morph_close_s,
                rgb_morph_close_kernel=int(rgb_morph_close_kernel_s),
                rgb_remove_small_components=rgb_remove_small_components_s,
                rgb_min_component_area=int(rgb_min_component_area_s),
                rgb_upscale_factor=float(rgb_upscale_factor_s),
                rgb_upscale_interpolation=rgb_upscale_interp_s,
                rgb_ink_normalization=rgb_ink_normalization_s,
                rgb_ink_strength=float(rgb_ink_strength_s),
            )
            if not _is_manual_mode(mode_s):
                image = state_s.get("image")
                if image is None:
                    return (
                        None,
                        _line_text(state_s.get("line_rows") or []),
                        "Please upload an image first.",
                        state_s,
                        "{}",
                        state_s.get("last_donut_pre"),
                        state_s.get("last_donut_post"),
                    )
                idx = getattr(evt, "index", None)
                if not isinstance(idx, (tuple, list)) or len(idx) < 2:
                    overlay = _render_overlay(np.asarray(image).astype(np.uint8, copy=False), state_s.get("line_rows") or [])
                    return (
                        overlay,
                        _line_text(state_s.get("line_rows") or []),
                        "Click position not available.",
                        state_s,
                        "{}",
                        state_s.get("last_donut_pre"),
                        state_s.get("last_donut_post"),
                    )
                try:
                    click_x, click_y = int(idx[0]), int(idx[1])
                except Exception:
                    overlay = _render_overlay(np.asarray(image).astype(np.uint8, copy=False), state_s.get("line_rows") or [])
                    return (
                        overlay,
                        _line_text(state_s.get("line_rows") or []),
                        "Invalid click position.",
                        state_s,
                        "{}",
                        state_s.get("last_donut_pre"),
                        state_s.get("last_donut_post"),
                    )

                src = np.asarray(image).astype(np.uint8, copy=False)
                h, w = src.shape[:2]
                click_x = max(0, min(w - 1, click_x))
                click_y = max(0, min(h - 1, click_y))
                rows = list(state_s.get("line_rows") or [])
                hit = _find_line_hit(rows, click_x, click_y)
                if hit is None:
                    overlay = _render_overlay(src, rows)
                    return (
                        overlay,
                        _line_text(rows),
                        "No detected line at this click position.",
                        state_s,
                        "{}",
                        state_s.get("last_donut_pre"),
                        state_s.get("last_donut_post"),
                    )

                engine = _normalize_ocr_engine(ocr_engine_s)
                auto_notes: List[str] = []
                if engine == OCR_ENGINE_DONUT and not (donut_s or "").strip():
                    overlay = _render_overlay(src, rows)
                    return (
                        overlay,
                        _line_text(rows),
                        "DONUT checkpoint is missing.",
                        state_s,
                        "{}",
                        state_s.get("last_donut_pre"),
                        state_s.get("last_donut_post"),
                    )
                if engine == OCR_ENGINE_BDRC and not (bdrc_ocr_model_s or "").strip():
                    try:
                        bdrc_ocr_model_s, note = _ensure_default_bdrc_ocr_model_path(bdrc_ocr_model_s)
                        if note:
                            auto_notes.append(note)
                    except Exception as exc:
                        overlay = _render_overlay(src, rows)
                        return (
                            overlay,
                            _line_text(rows),
                            f"BDRC OCR model is missing and auto-download failed: {type(exc).__name__}: {exc}",
                            state_s,
                            "{}",
                            state_s.get("last_donut_pre"),
                            state_s.get("last_donut_post"),
                        )

                box = [int(v) for v in hit.get("line_box") or [0, 0, 0, 0]]
                x1, y1, x2, y2 = box
                crop = _extract_row_crop(src, hit)
                text, dbg, pre_img, post_img = _run_ocr_on_crop(
                    crop,
                    ocr_engine=engine,
                    donut_checkpoint=donut_s,
                    bdrc_ocr_model=bdrc_ocr_model_s,
                    device=device_s,
                    max_len=int(max_len_s),
                    preprocess_preset=preproc_mode,
                    bdrc_preprocess_overrides=bdrc_overrides,
                    rgb_preprocess_overrides=rgb_overrides,
                )
                new_row = dict(hit)
                new_row["text"] = text
                new_row["ocr_debug"] = dbg
                new_row["source"] = "overlay_click_debug"
                rows = _add_or_update_line(rows, new_row)
                state_s["line_rows"] = rows
                state_s["last_donut_pre"] = pre_img
                state_s["last_donut_post"] = post_img
                state_s["last_debug_text"] = str(text or "")
                overlay = _render_overlay(src, rows)
                debug = {
                    "ok": True,
                    "mode": str(mode_s),
                    "action": "clicked_detected_line_for_debug_preview",
                    "ocr_engine": engine,
                    "ocr_model": (str(donut_s or "") if engine == OCR_ENGINE_DONUT else str(bdrc_ocr_model_s or "")),
                    "line_box": box,
                    "auto_download_notes": auto_notes or None,
                    "line_segmentation_mode": _normalize_line_segmentation_mode(line_segmentation_mode_s),
                    "line_segmentation_preprocess": (
                        normalize_line_segmentation_preprocess_pipeline(line_segmentation_preprocess_s)
                        if _normalize_line_segmentation_mode(line_segmentation_mode_s) == LINE_SEG_YOLO
                        else ("bdrc_internal" if _normalize_line_segmentation_mode(line_segmentation_mode_s) == LINE_SEG_BDRC else "none")
                    ),
                    "line_segmentation_model": (
                        str(line_model_s or "")
                        if _normalize_line_segmentation_mode(line_segmentation_mode_s) == LINE_SEG_YOLO
                        else (str(bdrc_line_model_s or "") if _normalize_line_segmentation_mode(line_segmentation_mode_s) == LINE_SEG_BDRC else "")
                    ),
                    "image_preprocess_pipeline": preproc_mode,
                    "preprocess_overrides": _effective_preprocess_overrides(
                        preprocess_preset=preproc_mode,
                        bdrc_preprocess_overrides=bdrc_overrides,
                        rgb_preprocess_overrides=rgb_overrides,
                    ),
                    "bdrc_preprocess_overrides": bdrc_overrides,
                    "rgb_preprocess_overrides": rgb_overrides,
                }
                return (
                    overlay,
                    _line_text(rows),
                    (
                        f"Updated debug preview from clicked line at ({click_x},{click_y})."
                        + ((" " + " ".join(auto_notes)) if auto_notes else "")
                    ),
                    state_s,
                    json.dumps(debug, ensure_ascii=False, indent=2),
                    pre_img,
                    post_img,
                )
            return _manual_click(
                state_s,
                ocr_engine_s,
                donut_s,
                bdrc_ocr_model_s,
                layout_s,
                line_segmentation_mode_s,
                line_segmentation_preprocess_s,
                line_model_s,
                bdrc_line_model_s,
                device_s,
                int(max_len_s),
                preproc_mode,
                bdrc_overrides,
                rgb_overrides,
                evt,
            )

        _select_evt = image_view.select(
            fn=_on_select,
            inputs=[
                mode,
                state,
                ocr_engine,
                donut_path,
                bdrc_ocr_model_path,
                layout_path,
                line_segmentation_mode,
                line_model_preprocess,
                line_model_path,
                bdrc_line_model_path,
                device,
                max_len,
                preprocess_preset,
                bdrc_gray_mode,
                bdrc_normalize_bg,
                bdrc_bg_blur_ksize,
                bdrc_bg_strength,
                bdrc_upscale_factor,
                bdrc_upscale_interp,
                bdrc_binarize,
                bdrc_threshold_method,
                bdrc_threshold_block,
                bdrc_threshold_c,
                bdrc_fixed_threshold,
                bdrc_morph_close,
                bdrc_morph_close_kernel,
                bdrc_remove_small_components,
                bdrc_min_component_area,
                rgb_preserve_color,
                rgb_normalize_bg,
                rgb_background_method,
                rgb_bg_blur_ksize,
                rgb_bg_strength,
                rgb_contrast,
                rgb_denoise,
                rgb_morph_close,
                rgb_morph_close_kernel,
                rgb_remove_small_components,
                rgb_min_component_area,
                rgb_upscale_factor,
                rgb_upscale_interp,
                rgb_ink_normalization,
                rgb_ink_strength,
            ],
            outputs=[image_view, transcript, status, state, debug_json, donut_input_before, donut_input_after],
        )
        _select_evt.then(
            fn=lambda st: str((st or {}).get("last_debug_text", "") if isinstance(st, dict) else ""),
            inputs=[state],
            outputs=[debug_text],
        )

        def _manual_full_roi_with_preprocess(
            mode_s: str,
            state_s: Dict[str, Any],
            ocr_engine_s: str,
            donut_s: str,
            bdrc_ocr_model_s: str,
            line_segmentation_mode_s: str,
            line_segmentation_preprocess_s: str,
            line_model_s: str,
            bdrc_line_model_s: str,
            device_s: str,
            max_len_s: int,
            preprocess_preset_s: str,
            gray_mode_s: str,
            normalize_background_s: bool,
            background_blur_ksize_s: int,
            background_strength_s: float,
            upscale_factor_s: float,
            upscale_interpolation_s: str,
            binarize_s: bool,
            threshold_method_s: str,
            threshold_block_size_s: int,
            threshold_c_s: int,
            fixed_threshold_s: int,
            morph_close_s: bool,
            morph_close_kernel_s: int,
            remove_small_components_s: bool,
            min_component_area_s: int,
            rgb_preserve_color_s: bool,
            rgb_normalize_bg_s: bool,
            rgb_background_method_s: str,
            rgb_bg_blur_ksize_s: int,
            rgb_bg_strength_s: float,
            rgb_contrast_s: float,
            rgb_denoise_s: bool,
            rgb_morph_close_s: bool,
            rgb_morph_close_kernel_s: int,
            rgb_remove_small_components_s: bool,
            rgb_min_component_area_s: int,
            rgb_upscale_factor_s: float,
            rgb_upscale_interp_s: str,
            rgb_ink_normalization_s: bool,
            rgb_ink_strength_s: float,
        ):
            preproc_mode, bdrc_overrides, rgb_overrides = _compose_preprocess_ui_settings(
                preprocess_preset=preprocess_preset_s,
                gray_mode=gray_mode_s,
                normalize_background=normalize_background_s,
                background_blur_ksize=int(background_blur_ksize_s),
                background_strength=float(background_strength_s),
                upscale_factor=float(upscale_factor_s),
                upscale_interpolation=upscale_interpolation_s,
                binarize=binarize_s,
                threshold_method=threshold_method_s,
                threshold_block_size=int(threshold_block_size_s),
                threshold_c=int(threshold_c_s),
                fixed_threshold=int(fixed_threshold_s),
                morph_close=morph_close_s,
                morph_close_kernel=int(morph_close_kernel_s),
                remove_small_components=remove_small_components_s,
                min_component_area=int(min_component_area_s),
                rgb_preserve_color=rgb_preserve_color_s,
                rgb_normalize_background=rgb_normalize_bg_s,
                rgb_background_method=rgb_background_method_s,
                rgb_background_blur_ksize=int(rgb_bg_blur_ksize_s),
                rgb_background_strength=float(rgb_bg_strength_s),
                rgb_contrast=float(rgb_contrast_s),
                rgb_denoise=rgb_denoise_s,
                rgb_morph_close=rgb_morph_close_s,
                rgb_morph_close_kernel=int(rgb_morph_close_kernel_s),
                rgb_remove_small_components=rgb_remove_small_components_s,
                rgb_min_component_area=int(rgb_min_component_area_s),
                rgb_upscale_factor=float(rgb_upscale_factor_s),
                rgb_upscale_interpolation=rgb_upscale_interp_s,
                rgb_ink_normalization=rgb_ink_normalization_s,
                rgb_ink_strength=float(rgb_ink_strength_s),
            )
            return _manual_full_image_roi(
                mode_s,
                state_s,
                ocr_engine_s,
                donut_s,
                bdrc_ocr_model_s,
                line_segmentation_mode_s,
                line_segmentation_preprocess_s,
                line_model_s,
                bdrc_line_model_s,
                device_s,
                int(max_len_s),
                preprocess_preset=preproc_mode,
                bdrc_preprocess_overrides=bdrc_overrides,
                rgb_preprocess_overrides=rgb_overrides,
            )

        _fullroi_evt = full_roi_btn.click(
            fn=_manual_full_roi_with_preprocess,
            inputs=[
                mode,
                state,
                ocr_engine,
                donut_path,
                bdrc_ocr_model_path,
                line_segmentation_mode,
                line_model_preprocess,
                line_model_path,
                bdrc_line_model_path,
                device,
                max_len,
                preprocess_preset,
                bdrc_gray_mode,
                bdrc_normalize_bg,
                bdrc_bg_blur_ksize,
                bdrc_bg_strength,
                bdrc_upscale_factor,
                bdrc_upscale_interp,
                bdrc_binarize,
                bdrc_threshold_method,
                bdrc_threshold_block,
                bdrc_threshold_c,
                bdrc_fixed_threshold,
                bdrc_morph_close,
                bdrc_morph_close_kernel,
                bdrc_remove_small_components,
                bdrc_min_component_area,
                rgb_preserve_color,
                rgb_normalize_bg,
                rgb_background_method,
                rgb_bg_blur_ksize,
                rgb_bg_strength,
                rgb_contrast,
                rgb_denoise,
                rgb_morph_close,
                rgb_morph_close_kernel,
                rgb_remove_small_components,
                rgb_min_component_area,
                rgb_upscale_factor,
                rgb_upscale_interp,
                rgb_ink_normalization,
                rgb_ink_strength,
            ],
            outputs=[image_view, transcript, status, state, debug_json, donut_input_before, donut_input_after],
        )
        _fullroi_evt.then(
            fn=lambda st: str((st or {}).get("last_debug_text", "") if isinstance(st, dict) else ""),
            inputs=[state],
            outputs=[debug_text],
        )

        def _compare(
            state_s: Dict[str, Any],
            ocr_engine_s: str,
            layout_s: str,
            line_segmentation_mode_s: str,
            line_segmentation_preprocess_s: str,
            line_model_s: str,
            bdrc_line_model_s: str,
            device_s: str,
            max_len_s: int,
            preprocess_preset_s: str,
            gray_mode_s: str,
            normalize_background_s: bool,
            background_blur_ksize_s: int,
            background_strength_s: float,
            upscale_factor_s: float,
            upscale_interpolation_s: str,
            binarize_s: bool,
            threshold_method_s: str,
            threshold_block_size_s: int,
            threshold_c_s: int,
            fixed_threshold_s: int,
            morph_close_s: bool,
            morph_close_kernel_s: int,
            remove_small_components_s: bool,
            min_component_area_s: int,
            rgb_preserve_color_s: bool,
            rgb_normalize_bg_s: bool,
            rgb_background_method_s: str,
            rgb_bg_blur_ksize_s: int,
            rgb_bg_strength_s: float,
            rgb_contrast_s: float,
            rgb_denoise_s: bool,
            rgb_morph_close_s: bool,
            rgb_morph_close_kernel_s: int,
            rgb_remove_small_components_s: bool,
            rgb_min_component_area_s: int,
            rgb_upscale_factor_s: float,
            rgb_upscale_interp_s: str,
            rgb_ink_normalization_s: bool,
            rgb_ink_strength_s: float,
        ):
            preproc_mode, bdrc_overrides, rgb_overrides = _compose_preprocess_ui_settings(
                preprocess_preset=preprocess_preset_s,
                gray_mode=gray_mode_s,
                normalize_background=normalize_background_s,
                background_blur_ksize=int(background_blur_ksize_s),
                background_strength=float(background_strength_s),
                upscale_factor=float(upscale_factor_s),
                upscale_interpolation=upscale_interpolation_s,
                binarize=binarize_s,
                threshold_method=threshold_method_s,
                threshold_block_size=int(threshold_block_size_s),
                threshold_c=int(threshold_c_s),
                fixed_threshold=int(fixed_threshold_s),
                morph_close=morph_close_s,
                morph_close_kernel=int(morph_close_kernel_s),
                remove_small_components=remove_small_components_s,
                min_component_area=int(min_component_area_s),
                rgb_preserve_color=rgb_preserve_color_s,
                rgb_normalize_background=rgb_normalize_bg_s,
                rgb_background_method=rgb_background_method_s,
                rgb_background_blur_ksize=int(rgb_bg_blur_ksize_s),
                rgb_background_strength=float(rgb_bg_strength_s),
                rgb_contrast=float(rgb_contrast_s),
                rgb_denoise=rgb_denoise_s,
                rgb_morph_close=rgb_morph_close_s,
                rgb_morph_close_kernel=int(rgb_morph_close_kernel_s),
                rgb_remove_small_components=rgb_remove_small_components_s,
                rgb_min_component_area=int(rgb_min_component_area_s),
                rgb_upscale_factor=float(rgb_upscale_factor_s),
                rgb_upscale_interpolation=rgb_upscale_interp_s,
                rgb_ink_normalization=rgb_ink_normalization_s,
                rgb_ink_strength=float(rgb_ink_strength_s),
            )
            return _run_comparison(
                state_s,
                ocr_engine_s,
                layout_s,
                line_segmentation_mode_s,
                line_segmentation_preprocess_s,
                line_model_s,
                bdrc_line_model_s,
                device_s,
                int(max_len_s),
                preprocess_preset=preproc_mode,
                bdrc_preprocess_overrides=bdrc_overrides,
                rgb_preprocess_overrides=rgb_overrides,
            )

        compare_btn.click(
            fn=_compare,
            inputs=[
                state,
                ocr_engine,
                layout_path,
                line_segmentation_mode,
                line_model_preprocess,
                line_model_path,
                bdrc_line_model_path,
                device,
                max_len,
                preprocess_preset,
                bdrc_gray_mode,
                bdrc_normalize_bg,
                bdrc_bg_blur_ksize,
                bdrc_bg_strength,
                bdrc_upscale_factor,
                bdrc_upscale_interp,
                bdrc_binarize,
                bdrc_threshold_method,
                bdrc_threshold_block,
                bdrc_threshold_c,
                bdrc_fixed_threshold,
                bdrc_morph_close,
                bdrc_morph_close_kernel,
                bdrc_remove_small_components,
                bdrc_min_component_area,
                rgb_preserve_color,
                rgb_normalize_bg,
                rgb_background_method,
                rgb_bg_blur_ksize,
                rgb_bg_strength,
                rgb_contrast,
                rgb_denoise,
                rgb_morph_close,
                rgb_morph_close_kernel,
                rgb_remove_small_components,
                rgb_min_component_area,
                rgb_upscale_factor,
                rgb_upscale_interp,
                rgb_ink_normalization,
                rgb_ink_strength,
            ],
            outputs=[image_view, transcript, status, state, debug_json, donut_input_before, donut_input_after],
        )

        save_btn.click(
            fn=_save_results,
            inputs=[
                state,
                transcript,
                ocr_engine,
                donut_path,
                bdrc_ocr_model_path,
                layout_path,
                line_segmentation_mode,
                line_model_preprocess,
                line_model_path,
                bdrc_line_model_path,
            ],
            outputs=[save_status],
        )

        font_plus_btn.click(
            fn=None,
            js="""
() => {
  const ta = document.querySelector('#transcript_box textarea');
  if (!ta) return;
  const cur = parseFloat(window.getComputedStyle(ta).fontSize) || 18;
  const next = Math.min(42, cur + 1);
  ta.style.setProperty('font-size', `${next}px`, 'important');
  ta.style.setProperty('line-height', '1.45', 'important');
}
""",
        )
        font_minus_btn.click(
            fn=None,
            js="""
() => {
  const ta = document.querySelector('#transcript_box textarea');
  if (!ta) return;
  const cur = parseFloat(window.getComputedStyle(ta).fontSize) || 18;
  const next = Math.max(10, cur - 1);
  ta.style.setProperty('font-size', `${next}px`, 'important');
  ta.style.setProperty('line-height', '1.45', 'important');
}
""",
        )

        img_zoom_in_btn.click(fn=None, js="() => { const p = document.querySelector('#ocr_image_panel'); if (p && p._zoomIn) p._zoomIn(); }")
        img_zoom_out_btn.click(fn=None, js="() => { const p = document.querySelector('#ocr_image_panel'); if (p && p._zoomOut) p._zoomOut(); }")
        img_zoom_reset_btn.click(fn=None, js="() => { const p = document.querySelector('#ocr_image_panel'); if (p && p._zoomReset) p._zoomReset(); }")

        debug_font_plus_btn.click(
            fn=None,
            js="""
() => {
  const ta = document.querySelector('#debug_text_box textarea');
  if (!ta) return;
  const cur = parseFloat(window.getComputedStyle(ta).fontSize) || 20;
  const next = Math.min(48, cur + 1);
  ta.style.setProperty('font-size', `${next}px`, 'important');
  ta.style.setProperty('line-height', '1.45', 'important');
}
""",
        )
        debug_font_minus_btn.click(
            fn=None,
            js="""
() => {
  const ta = document.querySelector('#debug_text_box textarea');
  if (!ta) return;
  const cur = parseFloat(window.getComputedStyle(ta).fontSize) || 20;
  const next = Math.max(10, cur - 1);
  ta.style.setProperty('font-size', `${next}px`, 'important');
  ta.style.setProperty('line-height', '1.45', 'important');
}
""",
        )

        def _toggle_advanced(show: bool, preprocess_preset_s: str):
            visible = bool(show)
            preproc_mode = _normalize_preprocess_preset(preprocess_preset_s)
            show_bdrc = visible and preproc_mode in {"bdrc", "gray"}
            show_rgb = visible and preproc_mode == "rgb"
            return (
                gr.update(visible=not visible),  # model_info_row  (simple summary)
                gr.update(visible=visible),       # advanced_model_row
                gr.update(visible=visible),       # advanced_bdrc_model_row
                gr.update(visible=visible),       # advanced_scan_row
                gr.update(visible=visible),       # advanced_runtime_row
                gr.update(visible=visible),       # advanced_preprocess_select_row
                gr.update(visible=show_bdrc),     # advanced_bdrc_row
                gr.update(visible=show_rgb),      # advanced_rgb_row
                gr.update(visible=visible),       # debug_json_accordion
                gr.update(visible=visible),       # donut_input_before
                gr.update(visible=visible),       # donut_input_after
                gr.update(visible=visible),       # save_status
                gr.update(visible=visible),       # advanced_debug_text_row
            )

        advanced_view.change(
            fn=_toggle_advanced,
            inputs=[advanced_view, preprocess_preset],
            outputs=[
                model_info_row,
                advanced_model_row,
                advanced_bdrc_model_row,
                advanced_scan_row,
                advanced_runtime_row,
                advanced_preprocess_select_row,
                advanced_bdrc_row,
                advanced_rgb_row,
                debug_json_accordion,
                donut_input_before,
                donut_input_after,
                save_status,
                advanced_debug_text_row,
            ],
        )

        def _on_preprocess_preset_change(show_advanced: bool, preprocess_preset_s: str):
            visible = bool(show_advanced)
            preproc_mode = _normalize_preprocess_preset(preprocess_preset_s)
            return (
                gr.update(visible=visible and preproc_mode in {"bdrc", "gray"}),
                gr.update(visible=visible and preproc_mode == "rgb"),
            )

        preprocess_preset.change(
            fn=_on_preprocess_preset_change,
            inputs=[advanced_view, preprocess_preset],
            outputs=[advanced_bdrc_row, advanced_rgb_row],
        )

        mode.change(
            fn=lambda m: gr.update(visible=_is_manual_mode(m)),
            inputs=[mode],
            outputs=[full_roi_btn],
        )

        def _refresh_donut():
            choices, msg = _list_donut_checkpoints()
            best = choices[0][1] if choices else ""
            return gr.update(choices=choices, value=best), msg

        def _refresh_layout():
            choices, msg = _list_layout_models()
            best = choices[0][1] if choices else ""
            return gr.update(choices=choices, value=best), msg

        def _refresh_line_model():
            choices, msg = _list_line_segmentation_models()
            best = choices[0][1] if choices else ""
            return gr.update(choices=choices, value=best), msg

        def _refresh_bdrc_ocr_model():
            choices, msg = _list_bdrc_ocr_models()
            best = choices[0][1] if choices else ""
            return gr.update(choices=choices, value=best), msg

        def _refresh_bdrc_line_model():
            choices, msg = _list_bdrc_line_models()
            best = choices[0][1] if choices else ""
            return gr.update(choices=choices, value=best), msg

        def _maybe_auto_download_bdrc_ocr_models(engine_s: str, current_path: str):
            if _normalize_ocr_engine(engine_s) != OCR_ENGINE_BDRC:
                return gr.update(), gr.update()
            try:
                chosen, note = _ensure_default_bdrc_ocr_model_path(current_path)
                choices, msg = _list_bdrc_ocr_models()
                status = msg if not note else f"{msg} | {note}"
                return gr.update(choices=choices, value=chosen), status
            except Exception as exc:
                return gr.update(), f"BDRC OCR auto-download failed: {type(exc).__name__}: {exc}"

        def _maybe_auto_download_bdrc_line_models(mode_s: str, current_path: str):
            if _normalize_line_segmentation_mode(mode_s) != LINE_SEG_BDRC:
                return gr.update(), gr.update()
            try:
                chosen, note = _ensure_default_bdrc_line_model_path(current_path)
                choices, msg = _list_bdrc_line_models()
                status = msg if not note else f"{msg} | {note}"
                return gr.update(choices=choices, value=chosen), status
            except Exception as exc:
                return gr.update(), f"BDRC line auto-download failed: {type(exc).__name__}: {exc}"

        donut_refresh_btn.click(
            fn=_refresh_donut,
            inputs=[],
            outputs=[donut_path, donut_info],
        )

        layout_refresh_btn.click(
            fn=_refresh_layout,
            inputs=[],
            outputs=[layout_path, layout_info],
        )

        line_model_refresh_btn.click(
            fn=_refresh_line_model,
            inputs=[],
            outputs=[line_model_path, line_model_info],
        )

        bdrc_ocr_refresh_btn.click(
            fn=_refresh_bdrc_ocr_model,
            inputs=[],
            outputs=[bdrc_ocr_model_path, bdrc_ocr_info],
        )

        bdrc_line_model_refresh_btn.click(
            fn=_refresh_bdrc_line_model,
            inputs=[],
            outputs=[bdrc_line_model_path, bdrc_line_model_info],
        )

        ocr_engine.change(
            fn=_maybe_auto_download_bdrc_ocr_models,
            inputs=[ocr_engine, bdrc_ocr_model_path],
            outputs=[bdrc_ocr_model_path, bdrc_ocr_info],
        )

        line_segmentation_mode.change(
            fn=_maybe_auto_download_bdrc_line_models,
            inputs=[line_segmentation_mode, bdrc_line_model_path],
            outputs=[bdrc_line_model_path, bdrc_line_model_info],
        )

        # ── Auto-load repro settings whenever the DONUT checkpoint changes ──
        def _auto_load_repro(ckpt_path: str):
            """Silently apply repro preprocess settings when checkpoint changes."""
            preset, _msg, bdrc, rgb = _load_repro_ui_settings(ckpt_path or "")
            return (
                gr.update(value=preset),
                gr.update(value=bdrc["gray_mode"]),
                gr.update(value=bdrc["normalize_background"]),
                gr.update(value=bdrc["background_blur_ksize"]),
                gr.update(value=bdrc["background_strength"]),
                gr.update(value=bdrc["upscale_factor"]),
                gr.update(value=bdrc["upscale_interpolation"]),
                gr.update(value=bdrc["binarize"]),
                gr.update(value=bdrc["threshold_method"]),
                gr.update(value=bdrc["threshold_block_size"]),
                gr.update(value=bdrc["threshold_c"]),
                gr.update(value=bdrc["fixed_threshold"]),
                gr.update(value=bdrc["morph_close"]),
                gr.update(value=bdrc["morph_close_kernel"]),
                gr.update(value=bdrc["remove_small_components"]),
                gr.update(value=bdrc["min_component_area"]),
                gr.update(value=rgb["preserve_color"]),
                gr.update(value=rgb["normalize_background"]),
                gr.update(value=rgb["background_method"]),
                gr.update(value=rgb["background_blur_ksize"]),
                gr.update(value=rgb["background_strength"]),
                gr.update(value=rgb["contrast"]),
                gr.update(value=rgb["denoise"]),
                gr.update(value=rgb["morph_close"]),
                gr.update(value=rgb["morph_close_kernel"]),
                gr.update(value=rgb["remove_small_components"]),
                gr.update(value=rgb["min_component_area"]),
                gr.update(value=rgb["upscale_factor"]),
                gr.update(value=rgb["upscale_interpolation"]),
                gr.update(value=rgb["ink_normalization"]),
                gr.update(value=rgb["ink_strength"]),
            )

        donut_path.change(
            fn=_auto_load_repro,
            inputs=[donut_path],
            outputs=[
                preprocess_preset,
                bdrc_gray_mode, bdrc_normalize_bg, bdrc_bg_blur_ksize,
                bdrc_bg_strength, bdrc_upscale_factor, bdrc_upscale_interp,
                bdrc_binarize, bdrc_threshold_method, bdrc_threshold_block,
                bdrc_threshold_c, bdrc_fixed_threshold, bdrc_morph_close,
                bdrc_morph_close_kernel, bdrc_remove_small_components, bdrc_min_component_area,
                rgb_preserve_color, rgb_normalize_bg, rgb_background_method,
                rgb_bg_blur_ksize, rgb_bg_strength, rgb_contrast,
                rgb_denoise, rgb_morph_close, rgb_morph_close_kernel,
                rgb_remove_small_components, rgb_min_component_area, rgb_upscale_factor,
                rgb_upscale_interp, rgb_ink_normalization, rgb_ink_strength,
            ],
        )

        def _on_load_repro(
            ckpt_path: str,
            cur_preset: str,
            cur_gray_mode: str, cur_normalize_bg: bool, cur_bg_blur: int,
            cur_bg_strength: float, cur_upscale: float, cur_upscale_interp: str,
            cur_binarize: bool, cur_threshold_method: str, cur_threshold_block: int,
            cur_threshold_c: int, cur_fixed_threshold: int, cur_morph_close: bool,
            cur_morph_kernel: int, cur_remove_small: bool, cur_min_area: int,
            cur_rgb_preserve: bool, cur_rgb_norm_bg: bool, cur_rgb_bg_method: str,
            cur_rgb_bg_blur: int, cur_rgb_bg_strength: float, cur_rgb_contrast: float,
            cur_rgb_denoise: bool, cur_rgb_morph: bool, cur_rgb_morph_kernel: int,
            cur_rgb_remove_small: bool, cur_rgb_min_area: int, cur_rgb_upscale: float,
            cur_rgb_upscale_interp: str, cur_rgb_ink_norm: bool, cur_rgb_ink_strength: float,
        ):
            preset, msg, bdrc, rgb = _load_repro_ui_settings(ckpt_path or "")

            # Build current state dicts for diffing
            cur_bdrc = {
                "gray_mode": cur_gray_mode, "normalize_background": cur_normalize_bg,
                "background_blur_ksize": cur_bg_blur, "background_strength": cur_bg_strength,
                "upscale_factor": cur_upscale, "upscale_interpolation": cur_upscale_interp,
                "binarize": cur_binarize, "threshold_method": cur_threshold_method,
                "threshold_block_size": cur_threshold_block, "threshold_c": cur_threshold_c,
                "fixed_threshold": cur_fixed_threshold, "morph_close": cur_morph_close,
                "morph_close_kernel": cur_morph_kernel, "remove_small_components": cur_remove_small,
                "min_component_area": cur_min_area,
            }
            cur_rgb = {
                "preserve_color": cur_rgb_preserve, "normalize_background": cur_rgb_norm_bg,
                "background_method": cur_rgb_bg_method, "background_blur_ksize": cur_rgb_bg_blur,
                "background_strength": cur_rgb_bg_strength, "contrast": cur_rgb_contrast,
                "denoise": cur_rgb_denoise, "morph_close": cur_rgb_morph,
                "morph_close_kernel": cur_rgb_morph_kernel, "remove_small_components": cur_rgb_remove_small,
                "min_component_area": cur_rgb_min_area, "upscale_factor": cur_rgb_upscale,
                "upscale_interpolation": cur_rgb_upscale_interp, "ink_normalization": cur_rgb_ink_norm,
                "ink_strength": cur_rgb_ink_strength,
            }

            # Compute diff
            changes = []
            if str(cur_preset) != str(preset):
                changes.append(f"preset: {cur_preset} → {preset}")
            for k, new_v in bdrc.items():
                old_v = cur_bdrc.get(k)
                if str(old_v) != str(new_v):
                    changes.append(f"{k}: {old_v} → {new_v}")
            for k, new_v in rgb.items():
                old_v = cur_rgb.get(k)
                if str(old_v) != str(new_v):
                    changes.append(f"{k}: {old_v} → {new_v}")

            if changes:
                diff_str = " | ".join(changes)
                status_msg = f"{msg} | Changed: {diff_str}"
            else:
                status_msg = f"{msg} | No changes (already matching)."

            return (
                gr.update(value=preset),
                status_msg,
                # BDRC sliders
                gr.update(value=bdrc["gray_mode"]),
                gr.update(value=bdrc["normalize_background"]),
                gr.update(value=bdrc["background_blur_ksize"]),
                gr.update(value=bdrc["background_strength"]),
                gr.update(value=bdrc["upscale_factor"]),
                gr.update(value=bdrc["upscale_interpolation"]),
                gr.update(value=bdrc["binarize"]),
                gr.update(value=bdrc["threshold_method"]),
                gr.update(value=bdrc["threshold_block_size"]),
                gr.update(value=bdrc["threshold_c"]),
                gr.update(value=bdrc["fixed_threshold"]),
                gr.update(value=bdrc["morph_close"]),
                gr.update(value=bdrc["morph_close_kernel"]),
                gr.update(value=bdrc["remove_small_components"]),
                gr.update(value=bdrc["min_component_area"]),
                # RGB sliders
                gr.update(value=rgb["preserve_color"]),
                gr.update(value=rgb["normalize_background"]),
                gr.update(value=rgb["background_method"]),
                gr.update(value=rgb["background_blur_ksize"]),
                gr.update(value=rgb["background_strength"]),
                gr.update(value=rgb["contrast"]),
                gr.update(value=rgb["denoise"]),
                gr.update(value=rgb["morph_close"]),
                gr.update(value=rgb["morph_close_kernel"]),
                gr.update(value=rgb["remove_small_components"]),
                gr.update(value=rgb["min_component_area"]),
                gr.update(value=rgb["upscale_factor"]),
                gr.update(value=rgb["upscale_interpolation"]),
                gr.update(value=rgb["ink_normalization"]),
                gr.update(value=rgb["ink_strength"]),
            )

        load_repro_btn.click(
            fn=_on_load_repro,
            inputs=[
                donut_path,
                preprocess_preset,
                bdrc_gray_mode, bdrc_normalize_bg, bdrc_bg_blur_ksize,
                bdrc_bg_strength, bdrc_upscale_factor, bdrc_upscale_interp,
                bdrc_binarize, bdrc_threshold_method, bdrc_threshold_block,
                bdrc_threshold_c, bdrc_fixed_threshold, bdrc_morph_close,
                bdrc_morph_close_kernel, bdrc_remove_small_components, bdrc_min_component_area,
                rgb_preserve_color, rgb_normalize_bg, rgb_background_method,
                rgb_bg_blur_ksize, rgb_bg_strength, rgb_contrast,
                rgb_denoise, rgb_morph_close, rgb_morph_close_kernel,
                rgb_remove_small_components, rgb_min_component_area, rgb_upscale_factor,
                rgb_upscale_interp, rgb_ink_normalization, rgb_ink_strength,
            ],
            outputs=[
                preprocess_preset,
                status,
                bdrc_gray_mode,
                bdrc_normalize_bg,
                bdrc_bg_blur_ksize,
                bdrc_bg_strength,
                bdrc_upscale_factor,
                bdrc_upscale_interp,
                bdrc_binarize,
                bdrc_threshold_method,
                bdrc_threshold_block,
                bdrc_threshold_c,
                bdrc_fixed_threshold,
                bdrc_morph_close,
                bdrc_morph_close_kernel,
                bdrc_remove_small_components,
                bdrc_min_component_area,
                rgb_preserve_color,
                rgb_normalize_bg,
                rgb_background_method,
                rgb_bg_blur_ksize,
                rgb_bg_strength,
                rgb_contrast,
                rgb_denoise,
                rgb_morph_close,
                rgb_morph_close_kernel,
                rgb_remove_small_components,
                rgb_min_component_area,
                rgb_upscale_factor,
                rgb_upscale_interp,
                rgb_ink_normalization,
                rgb_ink_strength,
            ],
        )

        # ── Folder Browser events ───────────────────────────────────────────
        def _on_folder_scan(folder: str):
            paths, msg = _scan_folder_for_images(folder)
            thumbs = _make_thumbnails(paths)
            return thumbs, paths, msg

        folder_scan_btn.click(
            fn=_on_folder_scan,
            inputs=[folder_path_input],
            outputs=[folder_gallery, folder_image_paths, folder_status],
        )
        # Also trigger scan when user presses Enter in the text box
        folder_path_input.submit(
            fn=_on_folder_scan,
            inputs=[folder_path_input],
            outputs=[folder_gallery, folder_image_paths, folder_status],
        )

        def _on_gallery_select(
            paths: List[str],
            ocr_engine_s: str,
            donut_s: str,
            bdrc_ocr_model_s: str,
            layout_s: str,
            line_segmentation_mode_s: str,
            line_segmentation_preprocess_s: str,
            line_model_s: str,
            bdrc_line_model_s: str,
            device_s: str,
            max_len_s: int,
            preprocess_preset_s: str,
            gray_mode_s: str,
            normalize_background_s: bool,
            background_blur_ksize_s: int,
            background_strength_s: float,
            upscale_factor_s: float,
            upscale_interpolation_s: str,
            binarize_s: bool,
            threshold_method_s: str,
            threshold_block_size_s: int,
            threshold_c_s: int,
            fixed_threshold_s: int,
            morph_close_s: bool,
            morph_close_kernel_s: int,
            remove_small_components_s: bool,
            min_component_area_s: int,
            rgb_preserve_color_s: bool,
            rgb_normalize_bg_s: bool,
            rgb_background_method_s: str,
            rgb_bg_blur_ksize_s: int,
            rgb_bg_strength_s: float,
            rgb_contrast_s: float,
            rgb_denoise_s: bool,
            rgb_morph_close_s: bool,
            rgb_morph_close_kernel_s: int,
            rgb_remove_small_components_s: bool,
            rgb_min_component_area_s: int,
            rgb_upscale_factor_s: float,
            rgb_upscale_interp_s: str,
            rgb_ink_normalization_s: bool,
            rgb_ink_strength_s: float,
            evt: gr.SelectData,
        ):
            # Resolve selected image path from gallery index
            idx = getattr(evt, "index", None)
            if idx is None or not isinstance(paths, list) or idx >= len(paths):
                return (
                    None, "", "Could not determine selected image.", _base_state(),
                    "{}", None, None,
                )
            img_path = paths[int(idx)]
            try:
                arr = _to_rgb_uint8(img_path)
            except Exception as exc:
                return (
                    None, "", f"Failed to load image: {exc}", _base_state(),
                    "{}", None, None,
                )

            new_state = _base_state()
            new_state["image_path"] = img_path
            new_state["image_name"] = Path(img_path).name
            new_state["image"] = arr

            # Build preprocess settings
            preproc_mode, bdrc_overrides, rgb_overrides = _compose_preprocess_ui_settings(
                preprocess_preset=preprocess_preset_s,
                gray_mode=gray_mode_s,
                normalize_background=normalize_background_s,
                background_blur_ksize=int(background_blur_ksize_s),
                background_strength=float(background_strength_s),
                upscale_factor=float(upscale_factor_s),
                upscale_interpolation=upscale_interpolation_s,
                binarize=binarize_s,
                threshold_method=threshold_method_s,
                threshold_block_size=int(threshold_block_size_s),
                threshold_c=int(threshold_c_s),
                fixed_threshold=int(fixed_threshold_s),
                morph_close=morph_close_s,
                morph_close_kernel=int(morph_close_kernel_s),
                remove_small_components=remove_small_components_s,
                min_component_area=int(min_component_area_s),
                rgb_preserve_color=rgb_preserve_color_s,
                rgb_normalize_background=rgb_normalize_bg_s,
                rgb_background_method=rgb_background_method_s,
                rgb_background_blur_ksize=int(rgb_bg_blur_ksize_s),
                rgb_background_strength=float(rgb_bg_strength_s),
                rgb_contrast=float(rgb_contrast_s),
                rgb_denoise=rgb_denoise_s,
                rgb_morph_close=rgb_morph_close_s,
                rgb_morph_close_kernel=int(rgb_morph_close_kernel_s),
                rgb_remove_small_components=rgb_remove_small_components_s,
                rgb_min_component_area=int(rgb_min_component_area_s),
                rgb_upscale_factor=float(rgb_upscale_factor_s),
                rgb_upscale_interpolation=rgb_upscale_interp_s,
                rgb_ink_normalization=rgb_ink_normalization_s,
                rgb_ink_strength=float(rgb_ink_strength_s),
            )

            # Run full-auto OCR (line segmentation + DONUT)
            overlay, transcript_text, ocr_status, updated_state, debug_j, pre_img, post_img = _run_full_auto(
                new_state,
                ocr_engine_s,
                donut_s,
                bdrc_ocr_model_s,
                layout_s,
                line_segmentation_mode_s,
                line_segmentation_preprocess_s,
                line_model_s,
                bdrc_line_model_s,
                device_s,
                int(max_len_s),
                preprocess_preset=preproc_mode,
                bdrc_preprocess_overrides=bdrc_overrides,
                rgb_preprocess_overrides=rgb_overrides,
            )
            return overlay, transcript_text, ocr_status, updated_state, debug_j, pre_img, post_img

        _gallery_select_evt = folder_gallery.select(
            fn=_on_gallery_select,
            inputs=[
                folder_image_paths,
                ocr_engine,
                donut_path,
                bdrc_ocr_model_path,
                layout_path,
                line_segmentation_mode,
                line_model_preprocess,
                line_model_path,
                bdrc_line_model_path,
                device,
                max_len,
                preprocess_preset,
                bdrc_gray_mode,
                bdrc_normalize_bg,
                bdrc_bg_blur_ksize,
                bdrc_bg_strength,
                bdrc_upscale_factor,
                bdrc_upscale_interp,
                bdrc_binarize,
                bdrc_threshold_method,
                bdrc_threshold_block,
                bdrc_threshold_c,
                bdrc_fixed_threshold,
                bdrc_morph_close,
                bdrc_morph_close_kernel,
                bdrc_remove_small_components,
                bdrc_min_component_area,
                rgb_preserve_color,
                rgb_normalize_bg,
                rgb_background_method,
                rgb_bg_blur_ksize,
                rgb_bg_strength,
                rgb_contrast,
                rgb_denoise,
                rgb_morph_close,
                rgb_morph_close_kernel,
                rgb_remove_small_components,
                rgb_min_component_area,
                rgb_upscale_factor,
                rgb_upscale_interp,
                rgb_ink_normalization,
                rgb_ink_strength,
            ],
            outputs=[image_view, transcript, status, state, debug_json, donut_input_before, donut_input_after],
        )
        _gallery_select_evt.then(
            fn=lambda st: str((st or {}).get("last_debug_text", "") if isinstance(st, dict) else ""),
            inputs=[state],
            outputs=[debug_text],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    host = os.environ.get("UI_HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("UI_PORT", "7865"))
    except ValueError:
        port = 7865
    share = os.environ.get("UI_SHARE", "").strip().lower() in {"1", "true", "yes", "on"}
    app.launch(server_name=host, server_port=port, share=share)
