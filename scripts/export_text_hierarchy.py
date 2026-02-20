#!/usr/bin/env python3
"""Batch-export text hierarchy crops from YOLO detections."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tibetan_utils.model_utils import ModelManager

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("opencv-python is required for export_text_hierarchy.py") from exc

try:
    from scipy.signal import find_peaks
except Exception:
    find_peaks = None


LOGGER = logging.getLogger("export_text_hierarchy")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_LAYOUT_CLASS_NAMES: Dict[int, str] = {
    0: "tibetan_number_word",
    1: "tibetan_text",
    2: "chinese_number_word",
}


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _list_images(input_dir: Path) -> List[Path]:
    return sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _normalize_layout_label(label: str) -> str:
    return "".join(ch for ch in (label or "").strip().lower() if ch.isalnum())


def _is_tibetan_text_detection(class_id: int, label: str) -> bool:
    norm = _normalize_layout_label(label)
    if norm in {"tibetantext", "tibetantextbox", "tibetanlines", "tibetanline"}:
        return True
    return norm in {"", f"class{class_id}"} and class_id == 1


def _is_tibetan_number_detection(class_id: int, label: str) -> bool:
    norm = _normalize_layout_label(label)
    if norm in {"tibetannumberword", "tibetannumber", "tibetannumbers"}:
        return True
    return norm in {"", f"class{class_id}"} and class_id == 0


def _is_chinese_number_detection(class_id: int, label: str) -> bool:
    norm = _normalize_layout_label(label)
    if norm in {"chinesenumberword", "chinesenumber", "chinesenumbers"}:
        return True
    return norm in {"", f"class{class_id}"} and class_id == 2


def _image_key(path: Path, base_dir: Path) -> str:
    rel = path.relative_to(base_dir)
    return "__".join(rel.with_suffix("").parts)


def _safe_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _compute_line_projection_state(
    crop_rgb: np.ndarray,
    projection_smooth: int,
    projection_threshold_rel: float,
) -> Dict[str, Any]:
    if crop_rgb is None or crop_rgb.size == 0:
        return {"ok": False, "reason": "empty_crop"}

    h, w = crop_rgb.shape[:2]
    if h < 4 or w < 4:
        return {"ok": False, "reason": "crop_too_small"}

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, open_kernel, iterations=1)

    kernel_w = max(3, int(round(w * 0.04)))
    kernel_w = min(kernel_w, max(1, w))
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    joined = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, line_kernel, iterations=1)

    projection = (joined > 0).sum(axis=1).astype(np.float32)
    if projection.size == 0 or float(np.max(projection)) <= 0.0:
        return {"ok": False, "reason": "empty_projection"}

    smooth = max(1, int(projection_smooth))
    if smooth % 2 == 0:
        smooth += 1
    if smooth > 1:
        conv = np.ones((smooth,), dtype=np.float32) / float(smooth)
        projection = np.convolve(projection, conv, mode="same")

    threshold_rel = min(max(float(projection_threshold_rel), 0.01), 0.95)
    threshold = max(1.0, float(np.max(projection)) * threshold_rel)
    mask = projection >= threshold

    return {
        "ok": True,
        "bw": bw,
        "projection": projection,
        "threshold": float(threshold),
        "mask": mask,
        "height": int(h),
        "width": int(w),
    }


def _line_runs_from_threshold_mask(mask: np.ndarray, min_line_height: int, merge_gap_px: int) -> List[Tuple[int, int]]:
    if mask is None or mask.size == 0:
        return []

    h = int(mask.shape[0])
    min_h = max(3, int(min_line_height))
    runs: List[Tuple[int, int]] = []
    y = 0
    while y < h:
        if not bool(mask[y]):
            y += 1
            continue
        y1 = y
        while y < h and bool(mask[y]):
            y += 1
        y2 = y
        if y2 - y1 >= min_h:
            runs.append((y1, y2))

    if not runs:
        return []

    merged: List[Tuple[int, int]] = []
    merge_gap = max(0, int(merge_gap_px))
    for y1, y2 in runs:
        if not merged:
            merged.append((y1, y2))
            continue
        py1, py2 = merged[-1]
        if y1 - py2 <= merge_gap:
            merged[-1] = (py1, y2)
        else:
            merged.append((y1, y2))
    return merged


def _segment_lines_in_text_crop(
    crop_rgb: np.ndarray,
    min_line_height: int,
    projection_smooth: int,
    projection_threshold_rel: float,
    merge_gap_px: int,
) -> List[Tuple[int, int, int, int]]:
    state = _compute_line_projection_state(
        crop_rgb=crop_rgb,
        projection_smooth=int(projection_smooth),
        projection_threshold_rel=float(projection_threshold_rel),
    )
    if not bool(state.get("ok")):
        return []

    bw = np.asarray(state["bw"])
    projection = np.asarray(state["projection"], dtype=np.float32)
    threshold = float(state["threshold"])
    mask = np.asarray(state["mask"], dtype=bool)
    h = int(state["height"])
    w = int(state["width"])

    min_h = max(3, int(min_line_height))
    line_runs: List[Tuple[int, int]] = []

    if find_peaks is not None and projection.size >= 5:
        maxv = float(np.max(projection))
        distance = max(min_h, int(round(h * 0.06)))
        prominence = max(2.0, maxv * max(0.06, float(projection_threshold_rel) * 0.55))
        peaks, _ = find_peaks(projection, distance=distance, prominence=prominence)
        if peaks.size > 0:
            keep = projection[peaks] >= max(1.0, threshold * 1.05)
            if bool(np.any(keep)):
                peaks = peaks[keep]
        if peaks.size > 0:
            peaks = np.sort(peaks.astype(np.int32))
            boundaries: List[int] = [0]
            for i in range(len(peaks) - 1):
                lo = int(peaks[i])
                hi = int(peaks[i + 1])
                if hi - lo < 2:
                    continue
                valley = lo + int(np.argmin(projection[lo : hi + 1]))
                valley = max(boundaries[-1] + 1, min(h - 1, valley))
                boundaries.append(valley)
            boundaries.append(h)
            for i in range(len(boundaries) - 1):
                y1 = int(boundaries[i])
                y2 = int(boundaries[i + 1])
                if y2 <= y1:
                    continue
                if y2 - y1 < min_h:
                    if line_runs:
                        py1, py2 = line_runs[-1]
                        line_runs[-1] = (py1, y2)
                    continue
                line_runs.append((y1, y2))

    if not line_runs:
        line_runs = _line_runs_from_threshold_mask(mask=mask, min_line_height=min_h, merge_gap_px=int(merge_gap_px))
    if not line_runs:
        return []

    out: List[Tuple[int, int, int, int]] = []
    for y1, y2 in line_runs:
        sub = bw[y1:y2, :]
        if sub.size == 0:
            continue
        foreground_pixels = int((sub > 0).sum())
        min_foreground = max(24, int((y2 - y1) * w * 0.002))
        if foreground_pixels < min_foreground:
            continue
        cols = np.where((sub > 0).sum(axis=0) > 0)[0]
        if cols.size == 0:
            continue
        x1 = max(0, int(cols[0]) - 2)
        x2 = min(w, int(cols[-1]) + 3)
        if x2 - x1 < 4:
            continue
        rows = np.where((sub[:, x1:x2] > 0).sum(axis=1) > 0)[0]
        if rows.size > 0:
            y1 = y1 + int(rows[0])
            y2 = y1 + int(rows[-1] - rows[0] + 1)
        pad = 1
        bx1 = max(0, x1 - pad)
        by1 = max(0, y1 - pad)
        bx2 = min(w, x2 + pad)
        by2 = min(h, y2 + pad)
        if bx2 > bx1 and by2 > by1:
            out.append((bx1, by1, bx2, by2))
    return out


def _compute_horizontal_projection_state(
    crop_rgb: np.ndarray,
    smooth_cols: int,
    threshold_rel: float,
) -> Dict[str, Any]:
    if crop_rgb is None or crop_rgb.size == 0:
        return {"ok": False, "reason": "empty_crop"}
    h, w = crop_rgb.shape[:2]
    if h < 3 or w < 3:
        return {"ok": False, "reason": "crop_too_small"}

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    projection = (bw > 0).sum(axis=0).astype(np.float32)
    if projection.size == 0 or float(np.max(projection)) <= 0.0:
        return {"ok": False, "reason": "empty_projection"}

    smooth = max(1, int(smooth_cols))
    if smooth % 2 == 0:
        smooth += 1
    if smooth > 1:
        conv = np.ones((smooth,), dtype=np.float32) / float(smooth)
        projection = np.convolve(projection, conv, mode="same")

    thr_rel = min(max(float(threshold_rel), 0.01), 0.95)
    threshold = max(1.0, float(np.max(projection)) * thr_rel)
    threshold = min(threshold, max(1.0, float(np.percentile(projection, 82.0))))
    mask = projection >= threshold
    return {
        "ok": True,
        "bw": bw,
        "projection": projection,
        "threshold": float(threshold),
        "mask": mask,
        "height": int(h),
        "width": int(w),
    }


def _segment_horizontal_runs_in_line_crop(
    line_crop_rgb: np.ndarray,
    smooth_cols: int,
    threshold_rel: float,
    min_width_px: int,
    merge_gap_px: int,
    horizontal_state: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Tuple[int, int, int, int]], Dict[str, Any]]:
    state = horizontal_state
    if state is None:
        state = _compute_horizontal_projection_state(
            crop_rgb=line_crop_rgb,
            smooth_cols=int(smooth_cols),
            threshold_rel=float(threshold_rel),
        )
    if not bool(state.get("ok")):
        return [], state

    projection = np.asarray(state.get("projection"), dtype=np.float32)
    threshold = float(state.get("threshold", 1.0))
    h = int(state.get("height", 0))
    w = int(state.get("width", 0))
    if projection.size == 0 or h <= 0 or w <= 0:
        state["horizontal_peaks"] = []
        state["horizontal_runs"] = []
        state["horizontal_method"] = "empty"
        return [], state

    min_w = max(3, int(min_width_px))
    merge_gap = max(0, int(merge_gap_px))
    peaks_used: List[int] = []
    runs: List[Tuple[int, int]] = []

    if find_peaks is not None and projection.size >= 5:
        maxv = float(np.max(projection))
        distance = max(3, int(round(w * 0.03)))
        prominence = max(1.0, maxv * max(0.05, float(threshold_rel) * 0.45))
        peaks, _ = find_peaks(projection, distance=distance, prominence=prominence)
        if peaks.size > 0:
            keep = projection[peaks] >= max(1.0, threshold * 0.65)
            if bool(np.any(keep)):
                peaks = peaks[keep]
        if peaks.size > 0:
            peaks = np.sort(peaks.astype(np.int32))
            peaks_used = [int(p) for p in peaks.tolist()]
            boundaries: List[int] = [0]
            for i in range(1, len(peaks)):
                lo = int(peaks[i - 1])
                hi = int(peaks[i])
                if hi - lo < 2:
                    continue
                left_valley = lo + int(np.argmin(projection[lo : hi + 1]))
                boundaries.append(int(max(1, min(w - 1, left_valley))))
            boundaries.append(w)
            boundaries = sorted(set(boundaries))
            if boundaries[0] != 0:
                boundaries = [0] + boundaries
            if boundaries[-1] != w:
                boundaries = boundaries + [w]
            for i in range(len(boundaries) - 1):
                x1 = int(boundaries[i])
                x2 = int(boundaries[i + 1])
                if x2 > x1:
                    runs.append((x1, x2))

    if not runs:
        runs = [(0, w)]

    if runs and len(runs) > 1:
        i = 0
        while i < len(runs):
            x1, x2 = runs[i]
            if (x2 - x1) >= min_w:
                i += 1
                continue
            if i < len(runs) - 1:
                _, nx2 = runs[i + 1]
                runs[i + 1] = (x1, nx2)
                del runs[i]
            else:
                px1, _ = runs[i - 1]
                runs[i - 1] = (px1, x2)
                del runs[i]
                i = max(0, i - 1)
            if len(runs) <= 1:
                break

    if runs:
        renorm: List[Tuple[int, int]] = []
        cursor = 0
        for i, (_, raw_x2) in enumerate(runs):
            if i == len(runs) - 1:
                x2 = w
            else:
                x2 = max(cursor + 1, min(w - 1, int(raw_x2)))
            renorm.append((cursor, x2))
            cursor = x2
        if renorm:
            lx1, _ = renorm[-1]
            renorm[-1] = (lx1, w)
        runs = [(x1, x2) for x1, x2 in renorm if x2 > x1]

    state["horizontal_peaks"] = peaks_used
    state["horizontal_runs"] = [[int(x1), int(x2)] for x1, x2 in runs]
    state["horizontal_method"] = ("scipy_find_peaks" if peaks_used else "threshold_mask_fallback")

    boxes = [(int(x1), 0, int(x2), int(h)) for x1, x2 in runs if x2 > x1]
    return boxes, state


def _left_valleys_for_peaks(projection: np.ndarray, peaks: List[int], width: int) -> List[int]:
    w = max(1, int(width))
    if projection is None or projection.size == 0 or w <= 0:
        return [0]
    sorted_peaks = sorted([int(p) for p in peaks if 0 <= int(p) < w])
    valleys: List[int] = [0]
    for i in range(1, len(sorted_peaks)):
        lo = int(sorted_peaks[i - 1])
        hi = int(sorted_peaks[i])
        if hi - lo < 2:
            continue
        valley = lo + int(np.argmin(projection[lo : hi + 1]))
        valleys.append(int(max(0, min(w - 1, valley))))
    return sorted(set(valleys))


def _build_wordbox_hierarchy_from_peaks(
    projection: np.ndarray,
    peaks: List[int],
    width: int,
    height: int,
    levels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    w = max(1, int(width))
    h = max(1, int(height))
    lvl = [2, 4, 8] if not levels else [max(1, int(v)) for v in levels]

    valley_starts = _left_valleys_for_peaks(projection=projection, peaks=peaks, width=w)
    valley_pool: List[int] = list(valley_starts)
    if find_peaks is not None and projection is not None and projection.size >= 5:
        minima, _ = find_peaks(-np.asarray(projection, dtype=np.float32), distance=max(2, int(round(w * 0.02))))
        valley_pool.extend(int(v) for v in minima.tolist())
    valley_pool.append(0)
    valley_pool = sorted(set(int(v) for v in valley_pool if 0 <= int(v) < w))

    levels_out: List[Dict[str, Any]] = []
    for n in lvl:
        requested_count = max(1, int(n))
        box_count = max(1, min(requested_count, w))
        desired_len = float(w) / float(requested_count)
        interior_valleys = sorted(int(v) for v in valley_pool if 0 < int(v) < w)

        boundaries: List[int] = [0]
        boundary_sources: List[str] = []
        for i in range(1, box_count):
            remaining_after = box_count - i
            min_x = boundaries[-1] + 1
            max_x = max(min_x, w - remaining_after)
            target = int(round(i * desired_len))
            target = max(min_x, min(max_x, target))
            candidates = [v for v in interior_valleys if min_x <= v <= max_x]
            if candidates:
                chosen = min(candidates, key=lambda v: (abs(int(v) - target), int(v)))
                source = "valley"
            else:
                chosen = target
                source = "fallback"
            chosen = int(max(min_x, min(max_x, chosen)))
            boundaries.append(chosen)
            boundary_sources.append(source)
        boundaries.append(w)

        boxes: List[Tuple[int, int, int, int]] = []
        for i in range(len(boundaries) - 1):
            x1 = max(0, min(w - 1, int(boundaries[i])))
            x2 = max(0, min(w, int(boundaries[i + 1])))
            if x2 > x1:
                boxes.append((x1, 0, x2, h))

        levels_out.append(
            {
                "count": int(box_count),
                "target_count": int(requested_count),
                "box_length_px": int(round(desired_len)),
                "box_lengths_px": [int(b[2] - b[0]) for b in boxes],
                "boundaries": [int(v) for v in boundaries],
                "boundary_sources": boundary_sources,
                "starts": [int(b[0]) for b in boxes],
                "boxes": [[int(a), int(b), int(c), int(d)] for (a, b, c, d) in boxes],
            }
        )

    return {
        "valley_starts": [int(v) for v in valley_starts],
        "valley_pool": [int(v) for v in valley_pool],
        "levels": levels_out,
    }


def _parse_hierarchy_levels(levels_text: str) -> List[int]:
    values: List[int] = []
    for part in (levels_text or "").split(","):
        raw = part.strip()
        if not raw:
            continue
        try:
            values.append(max(1, int(raw)))
        except ValueError:
            continue
    if not values:
        values = [2, 4, 8]
    return values


def _save_rgb_crop(rgb: np.ndarray, out_path: Path) -> bool:
    if rgb is None or rgb.size == 0:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8)).save(out_path)
    return True


def run(args) -> Dict[str, int]:
    _configure_logging()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    hierarchy_root = output_dir / "TextHierarchy"
    number_root = output_dir / "NumberCrops"
    hierarchy_root.mkdir(parents=True, exist_ok=True)
    number_root.mkdir(parents=True, exist_ok=True)

    image_paths = _list_images(input_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {input_dir}")

    model = ModelManager.load_model(str(model_path))
    predict_kwargs: Dict[str, Any] = {"conf": float(args.conf), "imgsz": int(args.imgsz)}
    if (args.device or "").strip():
        predict_kwargs["device"] = (args.device or "").strip()

    hierarchy_levels = _parse_hierarchy_levels(args.hierarchy_levels)
    summary = {
        "images_total": len(image_paths),
        "images_processed": 0,
        "detections_total": 0,
        "tibetan_text_blocks": 0,
        "lines_total": 0,
        "hierarchy_word_crops_saved": 0,
        "tibetan_number_crops_saved": 0,
        "chinese_number_crops_saved": 0,
    }

    for image_path in tqdm(image_paths, desc="text-hierarchy export"):
        with Image.open(image_path) as im:
            image_rgb = np.array(im.convert("RGB"))

        h, w = image_rgb.shape[:2]
        image_key = _image_key(image_path, input_dir)

        results = model.predict(source=image_rgb, **predict_kwargs)
        detections: List[Dict[str, Any]] = []
        for res in results:
            if not hasattr(res, "boxes") or res.boxes is None:
                continue
            boxes = res.boxes
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else []
            clss = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else []
            confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else []
            names = getattr(res, "names", None) or getattr(model, "names", None) or DEFAULT_LAYOUT_CLASS_NAMES
            for i in range(len(xyxy)):
                cls = int(clss[i]) if i < len(clss) else 0
                if isinstance(names, dict):
                    label = str(names.get(cls, f"class_{cls}"))
                elif isinstance(names, list) and 0 <= cls < len(names):
                    label = str(names[cls])
                else:
                    label = f"class_{cls}"
                box = _safe_box(*[int(v) for v in xyxy[i]], w=w, h=h)
                if box is None:
                    continue
                conf = float(confs[i]) if i < len(confs) else 0.0
                detections.append({"class": cls, "label": label, "box": box, "confidence": conf})

        summary["detections_total"] += len(detections)
        summary["images_processed"] += 1

        for det_idx, det in enumerate(detections, start=1):
            cls = int(det["class"])
            label = str(det["label"])
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            crop = image_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            if _is_tibetan_text_detection(cls, label):
                summary["tibetan_text_blocks"] += 1
                line_boxes = _segment_lines_in_text_crop(
                    crop_rgb=crop,
                    min_line_height=int(args.min_line_height),
                    projection_smooth=int(args.line_projection_smooth),
                    projection_threshold_rel=float(args.line_projection_threshold_rel),
                    merge_gap_px=int(args.line_merge_gap_px),
                )
                if not line_boxes:
                    ch, cw = crop.shape[:2]
                    line_boxes = [(0, 0, cw, ch)]

                text_block_dir = hierarchy_root / image_key / f"text_block_{det_idx:03d}"
                text_block_dir.mkdir(parents=True, exist_ok=True)
                meta_block: Dict[str, Any] = {
                    "image": str(image_path),
                    "image_key": image_key,
                    "det_index": int(det_idx),
                    "class": cls,
                    "label": label,
                    "confidence": float(det["confidence"]),
                    "text_box": [x1, y1, x2, y2],
                    "line_count": len(line_boxes),
                    "lines": [],
                }

                for line_idx, (lx1, ly1, lx2, ly2) in enumerate(line_boxes, start=1):
                    lx1 = max(0, min(crop.shape[1] - 1, int(lx1)))
                    ly1 = max(0, min(crop.shape[0] - 1, int(ly1)))
                    lx2 = max(0, min(crop.shape[1], int(lx2)))
                    ly2 = max(0, min(crop.shape[0], int(ly2)))
                    if lx2 <= lx1 or ly2 <= ly1:
                        continue

                    line_crop = crop[ly1:ly2, lx1:lx2]
                    if line_crop.size == 0:
                        continue
                    summary["lines_total"] += 1

                    line_dir = text_block_dir / f"line_{line_idx:03d}"
                    line_dir.mkdir(parents=True, exist_ok=True)
                    _save_rgb_crop(line_crop, line_dir / "line.png")

                    hstate = _compute_horizontal_projection_state(
                        crop_rgb=line_crop,
                        smooth_cols=int(args.horizontal_profile_smooth_cols),
                        threshold_rel=float(args.horizontal_profile_threshold_rel),
                    )
                    _, hstate = _segment_horizontal_runs_in_line_crop(
                        line_crop_rgb=line_crop,
                        smooth_cols=int(args.horizontal_profile_smooth_cols),
                        threshold_rel=float(args.horizontal_profile_threshold_rel),
                        min_width_px=int(args.horizontal_seg_min_width_px),
                        merge_gap_px=int(args.horizontal_seg_merge_gap_px),
                        horizontal_state=hstate,
                    )

                    projection = np.asarray(hstate.get("projection"), dtype=np.float32) if bool(hstate.get("ok")) else np.array([], dtype=np.float32)
                    peaks = [int(p) for p in (hstate.get("horizontal_peaks") or [])]
                    hierarchy = _build_wordbox_hierarchy_from_peaks(
                        projection=projection,
                        peaks=peaks,
                        width=int(hstate.get("width", line_crop.shape[1])),
                        height=int(hstate.get("height", line_crop.shape[0])),
                        levels=hierarchy_levels,
                    )

                    line_meta: Dict[str, Any] = {
                        "line_index": int(line_idx),
                        "line_box_local": [int(lx1), int(ly1), int(lx2), int(ly2)],
                        "line_box_global": [int(x1 + lx1), int(y1 + ly1), int(x1 + lx2), int(y1 + ly2)],
                        "horizontal_peaks": peaks,
                        "horizontal_method": str(hstate.get("horizontal_method", "unknown")),
                        "hierarchy": hierarchy,
                    }
                    meta_block["lines"].append(line_meta)

                    for level in (hierarchy.get("levels") or []):
                        target_count = int(level.get("target_count", level.get("count", 1)))
                        level_dir = line_dir / f"level_{target_count}"
                        level_dir.mkdir(parents=True, exist_ok=True)
                        for word_idx, box in enumerate(level.get("boxes") or [], start=1):
                            if not isinstance(box, (list, tuple)) or len(box) != 4:
                                continue
                            wx1, wy1, wx2, wy2 = [int(v) for v in box]
                            wx1 = max(0, min(line_crop.shape[1] - 1, wx1))
                            wy1 = max(0, min(line_crop.shape[0] - 1, wy1))
                            wx2 = max(0, min(line_crop.shape[1], wx2))
                            wy2 = max(0, min(line_crop.shape[0], wy2))
                            if wx2 <= wx1 or wy2 <= wy1:
                                continue
                            word_crop = line_crop[wy1:wy2, wx1:wx2]
                            if _save_rgb_crop(word_crop, level_dir / f"word_{word_idx:03d}.png"):
                                summary["hierarchy_word_crops_saved"] += 1

                (text_block_dir / "meta.json").write_text(
                    json.dumps(meta_block, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            elif _is_tibetan_number_detection(cls, label):
                out_path = number_root / "tibetan_number_word" / image_key / f"det_{det_idx:03d}.png"
                if _save_rgb_crop(crop, out_path):
                    summary["tibetan_number_crops_saved"] += 1
            elif _is_chinese_number_detection(cls, label):
                out_path = number_root / "chinese_number_word" / image_key / f"det_{det_idx:03d}.png"
                if _save_rgb_crop(crop, out_path):
                    summary["chinese_number_crops_saved"] += 1

    LOGGER.info("Done. Output: %s", output_dir)
    LOGGER.info(
        "images=%d detections=%d text_blocks=%d lines=%d hierarchy_crops=%d tib_numbers=%d chi_numbers=%d",
        summary["images_processed"],
        summary["detections_total"],
        summary["tibetan_text_blocks"],
        summary["lines_total"],
        summary["hierarchy_word_crops_saved"],
        summary["tibetan_number_crops_saved"],
        summary["chinese_number_crops_saved"],
    )
    return summary

