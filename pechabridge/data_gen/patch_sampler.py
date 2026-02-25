"""Patch sampling pipeline for line-level retrieval dataset generation."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

from .ink_profile import compute_ink_map, detect_profile_minima, horizontal_ink_profile, smooth_profile
from .io import (
    derive_doc_id,
    derive_page_id,
    make_patch_path,
    save_patch_png,
    summarize_metadata,
    write_metadata_parquet,
)

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    from scipy.signal import find_peaks
except Exception:  # pragma: no cover
    find_peaks = None


LOGGER = logging.getLogger("patch_sampler")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_LAYOUT_CLASS_NAMES: Dict[int, str] = {
    0: "tibetan_number_word",
    1: "tibetan_text",
    2: "chinese_number_word",
}


@dataclass(frozen=True)
class Box:
    """Detected text box on page."""

    xyxy: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    label: str


@dataclass(frozen=True)
class Line:
    """Detected line box on page."""

    xyxy: Tuple[int, int, int, int]
    confidence: float


class TextboxDetector(Protocol):
    """Model `m` interface."""

    def detect_textboxes(self, image_rgb: np.ndarray) -> List[Box]:
        """Return textbox detections."""


class LineDetector(Protocol):
    """Algorithm `A` interface."""

    def detect_lines(self, image_rgb: np.ndarray, textbox: Box) -> List[Line]:
        """Return line detections inside one textbox."""


@dataclass
class CandidateWindow:
    """Intermediate patch candidate on one normalized line."""

    x0_px: int
    x1_px: int
    tag: str
    source: str
    boundary_score: float
    ink_ratio: float


@dataclass
class PatchGenConfig:
    """Config for patch dataset generation."""

    model_path: str
    input_dir: str
    output_dir: str
    conf: float = 0.25
    imgsz: int = 1024
    device: str = ""
    seed: int = 42
    no_samples: int = 0
    target_height: int = 112
    widths: Sequence[int] = (256, 384, 512)
    overlap: float = 0.5
    jitter_enabled: bool = True
    jitter_frac: float = 0.04
    rmin: float = 0.01
    rmax: float = 1.0
    sigma_profile: float = 2.0
    min_dist_frac: float = 0.25
    min_dist_px: int = 0
    prominence: float = 0.08
    sigma_frac: float = 0.15
    n_per_line_per_scale: int = 12
    p_aligned: float = 0.6
    line_min_height: int = 10
    line_projection_smooth: int = 9
    line_projection_threshold_rel: float = 0.20
    line_merge_gap_px: int = 5
    use_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    binarize_ink: bool = False
    binarize_mode: str = "otsu"
    fixed_threshold: float = 0.5
    debug_dump: int = 0


def _load_overlay_font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _list_images(input_dir: Path) -> List[Path]:
    return sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _safe_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _normalize_layout_label(label: str) -> str:
    return "".join(ch for ch in str(label or "").strip().lower() if ch.isalnum())


def _is_tibetan_text_detection(class_id: int, label: str) -> bool:
    norm = _normalize_layout_label(label)
    if norm in {"tibetantext", "tibetantextbox", "tibetanlines", "tibetanline"}:
        return True
    return norm in {"", f"class{class_id}"} and int(class_id) == 1


def _resize_to_target_height(line_rgb: np.ndarray, target_height: int) -> np.ndarray:
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("opencv-python is required for line resize.")
    if line_rgb is None or line_rgb.size == 0:
        raise ValueError("Empty line crop.")
    h, w = line_rgb.shape[:2]
    th = max(8, int(target_height))
    scale = float(th) / float(max(1, h))
    tw = max(2, int(round(float(w) * scale)))
    resized = cv2.resize(line_rgb.astype(np.uint8), (tw, th), interpolation=cv2.INTER_CUBIC)
    return resized


def _compute_line_projection_state(
    crop_rgb: np.ndarray,
    projection_smooth: int,
    projection_threshold_rel: float,
) -> Dict[str, object]:
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("opencv-python is required for line segmentation.")
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
    gap = max(0, int(merge_gap_px))
    for y1, y2 in runs:
        if not merged:
            merged.append((y1, y2))
            continue
        py1, py2 = merged[-1]
        if y1 - py2 <= gap:
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
        projection_smooth=projection_smooth,
        projection_threshold_rel=projection_threshold_rel,
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
        line_runs = _line_runs_from_threshold_mask(mask=mask, min_line_height=min_h, merge_gap_px=merge_gap_px)
    if not line_runs:
        return []

    out: List[Tuple[int, int, int, int]] = []
    for y1, y2 in line_runs:
        sub = bw[y1:y2, :]
        if sub.size == 0:
            continue
        fg = int((sub > 0).sum())
        min_fg = max(24, int((y2 - y1) * w * 0.002))
        if fg < min_fg:
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


class YoloTextboxDetector:
    """Textbox detector wrapper using an Ultralytics model."""

    def __init__(self, model_path: str, conf: float, imgsz: int, device: str):
        from tibetan_utils.model_utils import ModelManager

        self.model = ModelManager.load_model(str(model_path))
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.device = str(device or "").strip()

    def detect_textboxes(self, image_rgb: np.ndarray) -> List[Box]:
        h, w = image_rgb.shape[:2]
        kwargs: Dict[str, object] = {"conf": self.conf, "imgsz": self.imgsz}
        if self.device:
            kwargs["device"] = self.device
        results = self.model.predict(source=image_rgb, **kwargs)
        out: List[Box] = []
        for res in results:
            if not hasattr(res, "boxes") or res.boxes is None:
                continue
            boxes = res.boxes
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else []
            clss = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else []
            confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else []
            names = getattr(res, "names", None) or getattr(self.model, "names", None) or DEFAULT_LAYOUT_CLASS_NAMES
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
                if not _is_tibetan_text_detection(class_id=cls, label=label):
                    continue
                out.append(Box(xyxy=box, confidence=conf, class_id=cls, label=label))
        return out


class ClassicalLineDetectorA:
    """Classical line detector `A` based on vertical projection segmentation."""

    def __init__(
        self,
        *,
        min_line_height: int,
        projection_smooth: int,
        projection_threshold_rel: float,
        merge_gap_px: int,
    ):
        self.min_line_height = int(min_line_height)
        self.projection_smooth = int(projection_smooth)
        self.projection_threshold_rel = float(projection_threshold_rel)
        self.merge_gap_px = int(merge_gap_px)

    def detect_lines(self, image_rgb: np.ndarray, textbox: Box) -> List[Line]:
        h, w = image_rgb.shape[:2]
        x1, y1, x2, y2 = textbox.xyxy
        box = _safe_box(x1, y1, x2, y2, w=w, h=h)
        if box is None:
            return []
        x1, y1, x2, y2 = box
        crop = image_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        local = _segment_lines_in_text_crop(
            crop_rgb=crop,
            min_line_height=self.min_line_height,
            projection_smooth=self.projection_smooth,
            projection_threshold_rel=self.projection_threshold_rel,
            merge_gap_px=self.merge_gap_px,
        )
        if not local:
            return [Line(xyxy=(x1, y1, x2, y2), confidence=float(textbox.confidence))]
        out: List[Line] = []
        for lx1, ly1, lx2, ly2 in local:
            g = _safe_box(x1 + lx1, y1 + ly1, x1 + lx2, y1 + ly2, w=w, h=h)
            if g is None:
                continue
            out.append(Line(xyxy=g, confidence=float(textbox.confidence)))
        return out


def _dense_windows(line_width: int, scale_w: int, overlap: float) -> List[Tuple[int, int, str]]:
    w = int(max(1, line_width))
    sw = int(max(2, scale_w))
    ov = float(min(0.95, max(0.0, overlap)))
    if w <= sw:
        return [(0, w, "dense_short")]

    step = max(1, int(round(sw * (1.0 - ov))))
    starts: List[int] = []
    x = 0
    while x + sw <= w:
        starts.append(int(x))
        x += step
    last = w - sw
    if not starts or starts[-1] != last:
        starts.append(int(last))

    out: List[Tuple[int, int, str]] = []
    for i, s in enumerate(starts):
        e = s + sw
        tag = "dense_end" if i == len(starts) - 1 and s != 0 else "dense"
        out.append((int(s), int(e), tag))
    return out


def _aligned_windows(
    line_width: int,
    scale_w: int,
    boundaries: np.ndarray,
    *,
    jitter_enabled: bool,
    jitter_frac: float,
    rng: np.random.Generator,
) -> List[Tuple[int, int, str]]:
    w = int(max(1, line_width))
    sw = int(max(2, scale_w))
    if boundaries.size == 0:
        return []
    out: List[Tuple[int, int, str]] = []
    for g in boundaries.astype(np.int32).tolist():
        for side in ("left", "right"):
            if side == "left":
                x0 = int(g)
                x1 = int(g + sw)
                tag = "aligned_left"
            else:
                x0 = int(g - sw)
                x1 = int(g)
                tag = "aligned_right"
            if jitter_enabled:
                jlim = float(max(0.0, jitter_frac)) * float(sw)
                shift = int(round(float(rng.uniform(-jlim, jlim))))
                x0 += shift
                x1 += shift
            if x0 < 0 or x1 > w:
                continue
            if x1 - x0 < 2:
                continue
            out.append((int(x0), int(x1), tag))
    return out


def _boundary_score(x0: int, x1: int, boundaries: np.ndarray, scale_w: int, sigma_frac: float) -> float:
    if boundaries.size == 0:
        return 0.0
    d_left = float(np.min(np.abs(boundaries.astype(np.float32) - float(x0))))
    d_right = float(np.min(np.abs(boundaries.astype(np.float32) - float(x1))))
    sigma_b = max(1.0, float(sigma_frac) * float(max(2, scale_w)))
    s2 = 2.0 * sigma_b * sigma_b
    return float(math.exp(-(d_left * d_left) / s2) + math.exp(-(d_right * d_right) / s2))


def _annotate_candidates(
    windows: Iterable[Tuple[int, int, str]],
    *,
    source: str,
    ink_map: np.ndarray,
    boundaries: np.ndarray,
    scale_w: int,
    sigma_frac: float,
    rmin: float,
    rmax: float,
) -> List[CandidateWindow]:
    h, w = ink_map.shape[:2]
    out: List[CandidateWindow] = []
    lower = float(max(0.0, rmin))
    upper = float(rmax)
    for x0, x1, tag in windows:
        x0 = max(0, min(w - 1, int(x0)))
        x1 = max(x0 + 1, min(w, int(x1)))
        patch = ink_map[:, x0:x1]
        if patch.size == 0:
            continue
        ink_ratio = float(np.mean(patch))
        if ink_ratio < lower:
            continue
        if upper > 0 and ink_ratio > upper:
            continue
        bscore = _boundary_score(x0, x1, boundaries=boundaries, scale_w=scale_w, sigma_frac=sigma_frac)
        out.append(
            CandidateWindow(
                x0_px=int(x0),
                x1_px=int(x1),
                tag=str(tag),
                source=str(source),
                boundary_score=float(bscore),
                ink_ratio=float(ink_ratio),
            )
        )
    return out


def _dedupe_candidates(cands: Sequence[CandidateWindow]) -> List[CandidateWindow]:
    by_span: Dict[Tuple[int, int], CandidateWindow] = {}
    for c in cands:
        key = (int(c.x0_px), int(c.x1_px))
        prev = by_span.get(key)
        if prev is None:
            by_span[key] = c
            continue
        if float(c.boundary_score) > float(prev.boundary_score):
            by_span[key] = c
    return sorted(by_span.values(), key=lambda z: (int(z.x0_px), int(z.x1_px), z.tag))


def _weighted_pick_index(weights: np.ndarray, rng: np.random.Generator) -> int:
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0:
        return -1
    total = float(np.sum(w))
    if total <= 0.0:
        return int(rng.integers(0, w.size))
    probs = w / total
    idx = int(rng.choice(np.arange(w.size), p=probs))
    return idx


def _sample_for_scale(
    dense: Sequence[CandidateWindow],
    aligned: Sequence[CandidateWindow],
    *,
    n: int,
    p_aligned: float,
    rng: np.random.Generator,
) -> List[CandidateWindow]:
    dense_pool = list(dense)
    aligned_pool = list(aligned)
    n_target = int(max(0, n))
    if n_target <= 0:
        return sorted(list({(c.x0_px, c.x1_px): c for c in list(dense_pool) + list(aligned_pool)}.values()), key=lambda c: c.x0_px)

    selected: List[CandidateWindow] = []
    selected_spans: set[Tuple[int, int]] = set()
    p = float(min(1.0, max(0.0, p_aligned)))
    while len(selected) < n_target and (dense_pool or aligned_pool):
        use_aligned = bool(rng.uniform() < p)
        if use_aligned and not aligned_pool:
            use_aligned = False
        if (not use_aligned) and not dense_pool and aligned_pool:
            use_aligned = True
        pool = aligned_pool if use_aligned else dense_pool
        if not pool:
            break

        if use_aligned:
            weights = np.asarray([max(1e-6, float(c.boundary_score)) for c in pool], dtype=np.float64)
            idx = _weighted_pick_index(weights, rng)
        else:
            idx = int(rng.integers(0, len(pool)))
        if idx < 0 or idx >= len(pool):
            break
        cand = pool.pop(idx)
        span = (int(cand.x0_px), int(cand.x1_px))
        dense_pool = [c for c in dense_pool if (c.x0_px, c.x1_px) != span]
        aligned_pool = [c for c in aligned_pool if (c.x0_px, c.x1_px) != span]
        if span in selected_spans:
            continue
        selected_spans.add(span)
        selected.append(cand)

    if len(selected) < n_target:
        remaining = [c for c in list(dense_pool) + list(aligned_pool) if (c.x0_px, c.x1_px) not in selected_spans]
        rng.shuffle(remaining)
        for cand in remaining:
            selected.append(cand)
            if len(selected) >= n_target:
                break
    return selected


def assign_option_a_k(records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """
    Assign contiguous Option-A neighborhood index k per (doc,page,line,scale), sorted by x0.
    """
    grouped: Dict[Tuple[str, str, int, int], List[Dict[str, object]]] = {}
    for rec in records:
        key = (
            str(rec.get("doc_id", "")),
            str(rec.get("page_id", "")),
            int(rec.get("line_id", -1)),
            int(rec.get("scale_w", -1)),
        )
        grouped.setdefault(key, []).append(dict(rec))

    out: List[Dict[str, object]] = []
    for key in sorted(grouped.keys()):
        rows = grouped[key]
        rows_sorted = sorted(
            rows,
            key=lambda r: (
                int(r.get("x0_px", 0)),
                int(r.get("x1_px", 0)),
                int(r.get("patch_id", 0)),
            ),
        )
        for idx, row in enumerate(rows_sorted):
            row["k"] = int(idx)
            out.append(row)
    out.sort(key=lambda r: int(r.get("patch_id", 0)))
    return out


def _draw_page_debug(
    image_rgb: np.ndarray,
    text_boxes: Sequence[Box],
    lines_by_box: Sequence[Sequence[Line]],
    out_path: Path,
) -> None:
    panel = Image.fromarray(np.asarray(image_rgb).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(panel)
    font = _load_overlay_font(size=13)
    for bi, box in enumerate(text_boxes, start=1):
        x1, y1, x2, y2 = box.xyxy
        draw.rectangle((x1, y1, x2, y2), outline=(60, 220, 255), width=2)
        draw.text((x1 + 2, max(0, y1 - 14)), f"tb{bi}", fill=(60, 220, 255), font=font)
        lines = lines_by_box[bi - 1] if bi - 1 < len(lines_by_box) else []
        for li, line in enumerate(lines, start=1):
            lx1, ly1, lx2, ly2 = line.xyxy
            draw.rectangle((lx1, ly1, lx2, ly2), outline=(255, 180, 20), width=2)
            draw.text((lx1 + 2, max(0, ly1 - 12)), f"l{li}", fill=(255, 180, 20), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def _draw_line_debug(
    line_resized_rgb: np.ndarray,
    boundaries: np.ndarray,
    dense_cands: Sequence[CandidateWindow],
    aligned_cands: Sequence[CandidateWindow],
    selected_cands: Sequence[CandidateWindow],
    out_path: Path,
) -> None:
    panel = Image.fromarray(np.asarray(line_resized_rgb).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(panel)
    font = _load_overlay_font(size=13)
    w, h = panel.size

    draw.rectangle((1, 1, min(w - 2, 370), 18), fill=(0, 0, 0))
    draw.text((4, 4), "green=minima orange=dense blue=aligned red=selected", fill=(245, 245, 245), font=font)
    for g in boundaries.astype(np.int32).tolist():
        gx = int(max(0, min(w - 1, g)))
        draw.line((gx, 0, gx, h - 1), fill=(30, 210, 60), width=1)

    for c in dense_cands:
        draw.rectangle((c.x0_px, 0, c.x1_px - 1, h - 1), outline=(245, 170, 50), width=1)
    for c in aligned_cands:
        draw.rectangle((c.x0_px, 0, c.x1_px - 1, h - 1), outline=(70, 150, 245), width=1)
    for c in selected_cands:
        draw.rectangle((c.x0_px, 0, c.x1_px - 1, h - 1), outline=(235, 70, 80), width=3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def generate_patch_dataset(config: PatchGenConfig) -> Dict[str, object]:
    """
    End-to-end generation:
    1) detect textboxes with m
    2) detect lines with A
    3) compute minima boundaries from horizontal ink profile
    4) sample multi-scale patches and save images + parquet metadata
    """
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("opencv-python is required.")

    input_dir = Path(config.input_dir).expanduser().resolve()
    output_dir = Path(config.output_dir).expanduser().resolve()
    model_path = Path(config.model_path).expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not model_path.exists() or not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    image_paths = _list_images(input_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {input_dir}")

    rng = np.random.default_rng(int(config.seed))
    found_images = len(image_paths)
    if 0 < int(config.no_samples) < len(image_paths):
        ids = np.arange(len(image_paths))
        rng.shuffle(ids)
        keep = sorted(int(i) for i in ids[: int(config.no_samples)])
        image_paths = [image_paths[i] for i in keep]
        LOGGER.info("Sampling pages: using %d of %d images", len(image_paths), found_images)

    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = (output_dir / "debug").resolve()
    debug_n = max(0, int(config.debug_dump))
    debug_indices: set[int] = set()
    if debug_n > 0:
        ids = np.arange(len(image_paths))
        rng.shuffle(ids)
        debug_indices = set(int(v) for v in ids[: min(debug_n, len(ids))].tolist())
        debug_dir.mkdir(parents=True, exist_ok=True)

    detector_m: TextboxDetector = YoloTextboxDetector(
        model_path=str(model_path),
        conf=float(config.conf),
        imgsz=int(config.imgsz),
        device=str(config.device or "").strip(),
    )
    detector_a: LineDetector = ClassicalLineDetectorA(
        min_line_height=int(config.line_min_height),
        projection_smooth=int(config.line_projection_smooth),
        projection_threshold_rel=float(config.line_projection_threshold_rel),
        merge_gap_px=int(config.line_merge_gap_px),
    )

    widths = sorted(set(int(max(2, w)) for w in config.widths))
    patch_id = 0
    all_records: List[Dict[str, object]] = []
    page_count = 0
    textbox_count = 0
    line_count = 0

    for page_idx, img_path in enumerate(tqdm(image_paths, desc="gen-patches")):
        with Image.open(img_path) as im:
            image_rgb = np.asarray(im.convert("RGB"))
        page_h, page_w = image_rgb.shape[:2]

        doc_id = derive_doc_id(img_path, input_dir)
        page_id = derive_page_id(img_path, input_dir)
        text_boxes = detector_m.detect_textboxes(image_rgb)
        textbox_count += len(text_boxes)
        page_count += 1

        lines_by_box: List[List[Line]] = []
        page_line_id = 0
        page_line_debug: List[Tuple[np.ndarray, np.ndarray, List[CandidateWindow], List[CandidateWindow], List[CandidateWindow], int]] = []

        for box in text_boxes:
            lines = detector_a.detect_lines(image_rgb, box)
            if not lines:
                lines = [Line(xyxy=box.xyxy, confidence=float(box.confidence))]
            lines_by_box.append(lines)

            for line in lines:
                lx1, ly1, lx2, ly2 = _safe_box(*line.xyxy, w=page_w, h=page_h) or (0, 0, 0, 0)
                if lx2 <= lx1 or ly2 <= ly1:
                    continue
                line_crop = image_rgb[ly1:ly2, lx1:lx2]
                if line_crop.size == 0:
                    continue

                page_line_id += 1
                line_count += 1
                line_resized = _resize_to_target_height(line_crop, target_height=int(config.target_height))
                line_h, line_w = line_resized.shape[:2]

                ink_map = compute_ink_map(
                    line_resized,
                    use_clahe=bool(config.use_clahe),
                    clahe_clip_limit=float(config.clahe_clip_limit),
                    clahe_tile_grid_size=int(config.clahe_tile_grid_size),
                    binarize=bool(config.binarize_ink),
                    binarize_mode=str(config.binarize_mode),
                    fixed_threshold=float(config.fixed_threshold),
                )
                profile_raw = horizontal_ink_profile(ink_map)
                profile_smooth = smooth_profile(profile_raw, sigma=float(config.sigma_profile))
                if int(config.min_dist_px) > 0:
                    min_dist = int(config.min_dist_px)
                else:
                    min_dist = max(1, int(round(float(config.min_dist_frac) * float(config.target_height))))
                boundaries = detect_profile_minima(
                    profile_smooth,
                    min_dist_px=min_dist,
                    prominence=float(config.prominence),
                )

                dense_dbg: List[CandidateWindow] = []
                aligned_dbg: List[CandidateWindow] = []
                selected_dbg: List[CandidateWindow] = []

                for scale_w in widths:
                    dense_w = _dense_windows(line_width=line_w, scale_w=scale_w, overlap=float(config.overlap))
                    aligned_w = _aligned_windows(
                        line_width=line_w,
                        scale_w=scale_w,
                        boundaries=boundaries,
                        jitter_enabled=bool(config.jitter_enabled),
                        jitter_frac=float(config.jitter_frac),
                        rng=rng,
                    )
                    dense = _dedupe_candidates(
                        _annotate_candidates(
                            dense_w,
                            source="dense",
                            ink_map=ink_map,
                            boundaries=boundaries,
                            scale_w=scale_w,
                            sigma_frac=float(config.sigma_frac),
                            rmin=float(config.rmin),
                            rmax=float(config.rmax),
                        )
                    )
                    aligned = _dedupe_candidates(
                        _annotate_candidates(
                            aligned_w,
                            source="aligned",
                            ink_map=ink_map,
                            boundaries=boundaries,
                            scale_w=scale_w,
                            sigma_frac=float(config.sigma_frac),
                            rmin=float(config.rmin),
                            rmax=float(config.rmax),
                        )
                    )
                    if not dense and not aligned:
                        continue

                    selected = _sample_for_scale(
                        dense=dense,
                        aligned=aligned,
                        n=int(config.n_per_line_per_scale),
                        p_aligned=float(config.p_aligned),
                        rng=rng,
                    )
                    selected = _dedupe_candidates(selected)
                    selected = sorted(selected, key=lambda c: (int(c.x0_px), int(c.x1_px), c.tag))
                    for cand in selected:
                        patch_id += 1
                        x0 = int(max(0, min(line_w - 1, cand.x0_px)))
                        x1 = int(max(x0 + 1, min(line_w, cand.x1_px)))
                        patch_rgb = line_resized[:, x0:x1]
                        out_patch = make_patch_path(
                            output_dir,
                            doc_id=doc_id,
                            page_id=page_id,
                            line_id=page_line_id,
                            scale_w=scale_w,
                            patch_id=patch_id,
                        )
                        save_patch_png(patch_rgb, out_patch)
                        all_records.append(
                            {
                                "patch_id": int(patch_id),
                                "doc_id": str(doc_id),
                                "page_id": str(page_id),
                                "line_id": int(page_line_id),
                                "scale_w": int(scale_w),
                                "k": -1,  # assigned later via Option A indexing
                                "x0_norm": float(x0 / max(1, line_w)),
                                "x1_norm": float(x1 / max(1, line_w)),
                                "line_h_px": int(line_h),
                                "line_w_px": int(line_w),
                                "boundary_score": float(cand.boundary_score),
                                "ink_ratio": float(cand.ink_ratio),
                                "source_img_path": str(img_path.resolve()),
                                "line_x0": int(lx1),
                                "line_y0": int(ly1),
                                "line_x1": int(lx2),
                                "line_y1": int(ly2),
                                "x0_px": int(x0),
                                "x1_px": int(x1),
                                "tag": str(cand.tag if cand.tag else cand.source),
                                "patch_path": str(out_patch.resolve()),
                            }
                        )

                    if page_idx in debug_indices:
                        dense_dbg.extend(dense[:50])
                        aligned_dbg.extend(aligned[:50])
                        selected_dbg.extend(selected[:50])

                if page_idx in debug_indices:
                    page_line_debug.append(
                        (
                            line_resized,
                            boundaries,
                            dense_dbg,
                            aligned_dbg,
                            selected_dbg,
                            page_line_id,
                        )
                    )

        if page_idx in debug_indices:
            doc_page = f"doc={doc_id}_page={page_id}"
            _draw_page_debug(
                image_rgb=image_rgb,
                text_boxes=text_boxes,
                lines_by_box=lines_by_box,
                out_path=debug_dir / f"{doc_page}_overlay.png",
            )
            if page_line_debug:
                ridx = int(rng.integers(0, len(page_line_debug)))
                line_img, boundaries, dense_dbg, aligned_dbg, selected_dbg, lid = page_line_debug[ridx]
                _draw_line_debug(
                    line_resized_rgb=line_img,
                    boundaries=boundaries,
                    dense_cands=dense_dbg,
                    aligned_cands=aligned_dbg,
                    selected_cands=selected_dbg,
                    out_path=debug_dir / f"{doc_page}_line={lid}_sampling.png",
                )

    all_records = assign_option_a_k(all_records)
    parquet_path = write_metadata_parquet(all_records, out_dataset_dir=output_dir)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "metadata_path": str(parquet_path),
        "pages_found": int(found_images),
        "pages_processed": int(page_count),
        "textboxes_detected": int(textbox_count),
        "lines_detected": int(line_count),
        "debug_dump": int(debug_n),
    }
    summary.update(summarize_metadata(all_records))
    return summary
