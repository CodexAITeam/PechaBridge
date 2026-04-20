"""Shared helpers for YOLO-based Tibetan line segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    from .preprocess_bdrc import BDRCPreprocessConfig, preprocess_image_bdrc
except Exception:  # pragma: no cover - optional at import time
    BDRCPreprocessConfig = None
    preprocess_image_bdrc = None

try:
    from .preprocess_rgb import RGBLinePreprocessConfig, preprocess_image_rgb_lines
except Exception:  # pragma: no cover - optional at import time
    RGBLinePreprocessConfig = None
    preprocess_image_rgb_lines = None

DEFAULT_LINE_SEGMENTATION_IMGSZ = 1280
DEFAULT_LINE_SEGMENTATION_CONF = 0.25
DEFAULT_LINE_SEGMENTATION_PREPROCESS = "gray"


@dataclass(frozen=True)
class LinePrediction:
    """One predicted text line."""

    box: Tuple[int, int, int, int]
    polygon: Tuple[Tuple[int, int], ...]
    confidence: float
    class_id: int
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": [int(v) for v in self.box],
            "polygon": [[int(x), int(y)] for x, y in self.polygon],
            "confidence": float(self.confidence),
            "class_id": int(self.class_id),
            "label": str(self.label),
        }


def _coerce_rgb_u8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.size == 0:
        raise ValueError("image is empty")
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[2] not in {3, 4}:
        raise ValueError(f"expected HxWx3/4 image array, got shape={arr.shape}")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def normalize_line_segmentation_preprocess_pipeline(pipeline: str) -> str:
    mode = str(pipeline or "none").strip().lower()
    if mode == "bdrc_no_bin":
        mode = "gray"
    if mode not in {"none", "bdrc", "gray", "rgb"}:
        return "none"
    return mode


def apply_line_segmentation_preprocess(
    image: Any,
    *,
    pipeline: str = DEFAULT_LINE_SEGMENTATION_PREPROCESS,
) -> np.ndarray:
    """Apply the same named preprocessing modes used by DONUT OCR training."""

    mode = normalize_line_segmentation_preprocess_pipeline(pipeline)
    if isinstance(image, Image.Image):
        pil = image.convert("RGB")
    else:
        pil = Image.fromarray(_coerce_rgb_u8(np.asarray(image)), mode="RGB")

    if mode == "none":
        out = pil
    elif mode == "bdrc":
        if preprocess_image_bdrc is None or BDRCPreprocessConfig is None:
            raise RuntimeError("BDRC preprocessing is unavailable in this environment.")
        out = preprocess_image_bdrc(image=pil, config=BDRCPreprocessConfig.vit_defaults()).convert("RGB")
    elif mode == "gray":
        if preprocess_image_bdrc is None or BDRCPreprocessConfig is None:
            raise RuntimeError("Gray preprocessing is unavailable because BDRC preprocessing could not be imported.")
        cfg = BDRCPreprocessConfig.vit_defaults()
        cfg = BDRCPreprocessConfig.from_dict({**cfg.to_dict(), "binarize": False, "gray_mode": "min_rgb"})
        out = preprocess_image_bdrc(image=pil, config=cfg).convert("RGB")
    elif mode == "rgb":
        if preprocess_image_rgb_lines is None or RGBLinePreprocessConfig is None:
            raise RuntimeError("RGB line preprocessing is unavailable in this environment.")
        out = preprocess_image_rgb_lines(image=pil, config=RGBLinePreprocessConfig.vit_defaults()).convert("RGB")
    else:  # pragma: no cover - guarded by normalization
        out = pil
    return np.asarray(out, dtype=np.uint8)


def _clip_box(raw_box: Sequence[float], *, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    if len(raw_box) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in raw_box[:4]]
    except Exception:
        return None
    x1 = max(0, min(int(width) - 1, x1))
    y1 = max(0, min(int(height) - 1, y1))
    x2 = max(0, min(int(width), x2))
    y2 = max(0, min(int(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def coerce_polygon_points(
    points: Any,
    *,
    width: int,
    height: int,
) -> Optional[List[Tuple[int, int]]]:
    """Convert arbitrary point arrays into a clipped polygon."""

    if points is None:
        return None
    try:
        arr = np.asarray(points, dtype=np.float32)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
        return None

    w = max(1, int(width))
    h = max(1, int(height))
    out: List[Tuple[int, int]] = []
    for raw_x, raw_y in arr[:, :2].tolist():
        if not np.isfinite(raw_x) or not np.isfinite(raw_y):
            continue
        x = max(0, min(w - 1, int(round(float(raw_x)))))
        y = max(0, min(h - 1, int(round(float(raw_y)))))
        pt = (x, y)
        if out and out[-1] == pt:
            continue
        out.append(pt)

    if len(out) >= 2 and out[0] == out[-1]:
        out.pop()
    if len(out) < 3 or len(set(out)) < 3:
        return None
    return out


def polygon_to_box(
    points: Any,
    *,
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    polygon = coerce_polygon_points(points, width=width, height=height)
    if not polygon:
        return None
    xs = [int(x) for x, _ in polygon]
    ys = [int(y) for _, y in polygon]
    x1 = max(0, min(xs))
    y1 = max(0, min(ys))
    x2 = min(int(width), max(xs) + 1)
    y2 = min(int(height), max(ys) + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def polygon_to_yolo_segment_line(
    points: Any,
    *,
    width: int,
    height: int,
    class_id: int = 0,
) -> Optional[str]:
    polygon = coerce_polygon_points(points, width=width, height=height)
    if not polygon:
        return None
    if int(width) <= 0 or int(height) <= 0:
        return None

    coords: List[str] = []
    for x, y in polygon:
        coords.append(f"{float(x) / float(width):.6f}")
        coords.append(f"{float(y) / float(height):.6f}")
    if len(coords) < 6:
        return None
    return f"{int(class_id)} " + " ".join(coords)


def sort_line_predictions(predictions: Iterable[LinePrediction]) -> List[LinePrediction]:
    return sorted(
        list(predictions),
        key=lambda pred: (
            (int(pred.box[1]) + int(pred.box[3])) / 2.0,
            int(pred.box[1]),
            int(pred.box[0]),
            (int(pred.box[0]) + int(pred.box[2])) / 2.0,
        ),
    )


@lru_cache(maxsize=4)
def _load_ultralytics_model(model_path: str):
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "Ultralytics is required for model-based line segmentation. "
            "Install it with `pip install ultralytics`."
        ) from exc
    return YOLO(model_path)


def predict_line_regions(
    image: np.ndarray,
    *,
    model_path: str,
    conf: float = DEFAULT_LINE_SEGMENTATION_CONF,
    imgsz: int = DEFAULT_LINE_SEGMENTATION_IMGSZ,
    preprocess_pipeline: str = "none",
    device: str = "",
    max_det: int = 512,
) -> List[LinePrediction]:
    """Run a YOLO segmentation/detection model and return line predictions."""

    rgb_raw = _coerce_rgb_u8(image)
    mode = normalize_line_segmentation_preprocess_pipeline(preprocess_pipeline)
    rgb = apply_line_segmentation_preprocess(rgb_raw, pipeline=mode) if mode != "none" else rgb_raw
    height, width = rgb.shape[:2]

    model_file = Path(model_path).expanduser().resolve()
    if not model_file.exists():
        raise FileNotFoundError(f"Line segmentation model not found: {model_file}")

    model = _load_ultralytics_model(str(model_file))
    predict_kwargs: dict[str, Any] = {
        "source": rgb,
        "conf": float(conf),
        "imgsz": int(imgsz),
        "verbose": False,
        "max_det": max(1, int(max_det)),
    }
    device_name = str(device or "").strip()
    if device_name and device_name.lower() != "auto":
        predict_kwargs["device"] = device_name

    results = model.predict(**predict_kwargs)
    predictions: List[LinePrediction] = []
    for res in results:
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            continue

        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else np.empty((0, 4), dtype=np.float32)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else np.zeros((len(xyxy),), dtype=np.float32)
        clss = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else np.zeros((len(xyxy),), dtype=np.float32)
        names = getattr(res, "names", None) or getattr(model, "names", None) or {}
        mask_xy = list(getattr(getattr(res, "masks", None), "xy", []) or [])

        for idx in range(len(xyxy)):
            box = _clip_box(xyxy[idx].tolist(), width=width, height=height)
            polygon = coerce_polygon_points(mask_xy[idx], width=width, height=height) if idx < len(mask_xy) else None
            if box is None and polygon is not None:
                box = polygon_to_box(polygon, width=width, height=height)
            if box is None:
                continue

            if polygon is None:
                x1, y1, x2, y2 = box
                polygon = [(x1, y1), (x2 - 1, y1), (x2 - 1, y2 - 1), (x1, y2 - 1)]

            cls_id = int(clss[idx]) if idx < len(clss) else 0
            if isinstance(names, dict):
                label = str(names.get(cls_id, f"class_{cls_id}"))
            elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
                label = str(names[cls_id])
            else:
                label = f"class_{cls_id}"

            predictions.append(
                LinePrediction(
                    box=box,
                    polygon=tuple((int(x), int(y)) for x, y in polygon),
                    confidence=float(confs[idx]) if idx < len(confs) else 0.0,
                    class_id=cls_id,
                    label=label,
                )
            )

    return sort_line_predictions(predictions)


__all__ = [
    "DEFAULT_LINE_SEGMENTATION_CONF",
    "DEFAULT_LINE_SEGMENTATION_IMGSZ",
    "DEFAULT_LINE_SEGMENTATION_PREPROCESS",
    "LinePrediction",
    "apply_line_segmentation_preprocess",
    "coerce_polygon_points",
    "normalize_line_segmentation_preprocess_pipeline",
    "polygon_to_box",
    "polygon_to_yolo_segment_line",
    "predict_line_regions",
    "sort_line_predictions",
]
