"""Helpers for running BDRC-style ONNX line detection and OCR models.

This module mirrors the core runtime ideas of the BDRC Tibetan OCR desktop app:
- line/layout ONNX inference with internal BDRC preprocessing
- contour-based postprocessing for page line extraction
- OCR ONNX inference with BDRC-like line normalization/padding

The implementation is intentionally self-contained so the same helpers can be
used from UI workbenches and CLI flows without pulling in the original app.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover - optional at import time
    cv2 = None

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional at import time
    ort = None

try:
    import pyewts
except Exception:  # pragma: no cover - optional at import time
    pyewts = None

from .preprocess_bdrc import (
    BDRCPreprocessConfig,
    bdrc_image_to_normalized_tensor,
    preprocess_image_bdrc,
    resize_to_height,
)


DEFAULT_BDRC_LINE_THRESHOLD = 0.9
DEFAULT_BDRC_LAYOUT_THRESHOLD = 0.8
DEFAULT_BDRC_LINE_K_FACTOR = 2.5
DEFAULT_BDRC_LINE_BBOX_TOLERANCE = 3.0


@dataclass(frozen=True)
class BDRCLineModelConfig:
    model_dir: str
    model_file: str
    patch_size: int
    model_kind: str
    classes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BDRCOCRModelConfig:
    model_dir: str
    model_file: str
    architecture: str
    input_width: int
    input_height: int
    input_layer: str
    output_layer: str
    squeeze_channel: bool
    swap_hw: bool
    encoder: str
    charset: Tuple[str, ...]
    add_blank: bool
    version: str = ""


@dataclass
class BDRCLinePrediction:
    box: Tuple[int, int, int, int]
    polygon: Tuple[Tuple[int, int], ...]
    confidence: float
    class_id: int
    label: str
    crop_image: Optional[np.ndarray] = None
    page_angle: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "box": [int(v) for v in self.box],
            "polygon": [[int(x), int(y)] for x, y in self.polygon],
            "confidence": float(self.confidence),
            "class_id": int(self.class_id),
            "label": str(self.label),
            "page_angle": float(self.page_angle),
        }


def _require_cv2() -> None:
    if cv2 is None:  # pragma: no cover - dependency guard
        raise RuntimeError("opencv-python is required for BDRC inference helpers.")


def _require_onnxruntime() -> None:
    if ort is None:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "onnxruntime is required for BDRC inference helpers. "
            "Install it with `pip install onnxruntime` or `onnxruntime-gpu`."
        )


def _coerce_rgb_u8(image: Any) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
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


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_bdrc_line_model_dir(path: str | Path) -> bool:
    p = Path(path).expanduser()
    if not p.exists():
        return False
    if p.is_file() and p.name == "config.json":
        cfg_path = p
    elif p.is_file() and p.suffix.lower() == ".onnx":
        cfg_path = p.parent / "config.json"
    else:
        cfg_path = p / "config.json"
    if not cfg_path.exists():
        return False
    try:
        payload = _read_json(cfg_path)
    except Exception:
        return False
    onnx_name = str(payload.get("onnx-model", "") or "").strip()
    if not onnx_name:
        return False
    return (cfg_path.parent / onnx_name).exists()


def is_bdrc_ocr_model_dir(path: str | Path) -> bool:
    p = Path(path).expanduser()
    if not p.exists():
        return False
    if p.is_file() and p.name == "model_config.json":
        cfg_path = p
    elif p.is_file() and p.suffix.lower() == ".onnx":
        cfg_path = p.parent / "model_config.json"
    else:
        cfg_path = p / "model_config.json"
    if not cfg_path.exists():
        return False
    try:
        payload = _read_json(cfg_path)
    except Exception:
        return False
    onnx_name = str(payload.get("onnx-model", "") or "").strip()
    if not onnx_name:
        return False
    return (cfg_path.parent / onnx_name).exists()


def find_bdrc_line_model_dirs(base: str | Path) -> List[Path]:
    root = Path(base).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return []
    out: List[Path] = []
    seen: set[str] = set()
    for cfg_path in sorted(root.rglob("config.json")):
        model_dir = cfg_path.parent.resolve()
        key = str(model_dir)
        if key in seen:
            continue
        if is_bdrc_line_model_dir(model_dir):
            seen.add(key)
            out.append(model_dir)
    return out


def find_bdrc_ocr_model_dirs(base: str | Path) -> List[Path]:
    root = Path(base).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return []
    out: List[Path] = []
    seen: set[str] = set()
    for cfg_path in sorted(root.rglob("model_config.json")):
        model_dir = cfg_path.parent.resolve()
        key = str(model_dir)
        if key in seen:
            continue
        if is_bdrc_ocr_model_dir(model_dir):
            seen.add(key)
            out.append(model_dir)
    return out


def resolve_bdrc_line_model_config(path: str | Path) -> BDRCLineModelConfig:
    p = Path(path).expanduser().resolve()
    if p.is_file() and p.name == "config.json":
        cfg_path = p
    elif p.is_file() and p.suffix.lower() == ".onnx":
        cfg_path = p.parent / "config.json"
    else:
        cfg_path = p / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"BDRC line model config not found: {cfg_path}")
    payload = _read_json(cfg_path)
    onnx_name = str(payload.get("onnx-model", "") or "").strip()
    if not onnx_name:
        raise ValueError(f"BDRC line model config is missing 'onnx-model': {cfg_path}")
    model_file = (cfg_path.parent / onnx_name).resolve()
    if not model_file.exists():
        raise FileNotFoundError(f"BDRC line model weights not found: {model_file}")
    classes = tuple(str(x) for x in (payload.get("classes") or []))
    model_kind = "layout" if classes else "line"
    return BDRCLineModelConfig(
        model_dir=str(cfg_path.parent.resolve()),
        model_file=str(model_file),
        patch_size=max(1, int(payload.get("patch_size", 512) or 512)),
        model_kind=model_kind,
        classes=classes,
    )


def resolve_bdrc_ocr_model_config(path: str | Path) -> BDRCOCRModelConfig:
    p = Path(path).expanduser().resolve()
    if p.is_file() and p.name == "model_config.json":
        cfg_path = p
    elif p.is_file() and p.suffix.lower() == ".onnx":
        cfg_path = p.parent / "model_config.json"
    else:
        cfg_path = p / "model_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"BDRC OCR model config not found: {cfg_path}")
    payload = _read_json(cfg_path)
    onnx_name = str(payload.get("onnx-model", "") or "").strip()
    if not onnx_name:
        raise ValueError(f"BDRC OCR model config is missing 'onnx-model': {cfg_path}")
    model_file = (cfg_path.parent / onnx_name).resolve()
    if not model_file.exists():
        raise FileNotFoundError(f"BDRC OCR model weights not found: {model_file}")
    charset = tuple(str(x) for x in (payload.get("charset") or []))
    return BDRCOCRModelConfig(
        model_dir=str(cfg_path.parent.resolve()),
        model_file=str(model_file),
        architecture=str(payload.get("architecture", "") or ""),
        input_width=max(1, int(payload.get("input_width", 2000) or 2000)),
        input_height=max(1, int(payload.get("input_height", 80) or 80)),
        input_layer=str(payload.get("input_layer", "input") or "input"),
        output_layer=str(payload.get("output_layer", "output") or "output"),
        squeeze_channel=bool(payload.get("squeeze_channel_dim", payload.get("squeeze_channel", "no")) == "yes"),
        swap_hw=bool(payload.get("swap_hw", "no") == "yes"),
        encoder=str(payload.get("encoder", "") or "").strip().lower(),
        charset=charset,
        add_blank=bool(payload.get("add_blank", "no") == "yes"),
        version=str(payload.get("version", "") or ""),
    )


def _device_provider_key(device: str) -> str:
    raw = str(device or "auto").strip().lower()
    if raw.startswith("cuda"):
        return "cuda"
    return "cpu"


def _ort_providers(device: str) -> List[Any]:
    _require_onnxruntime()
    available = set(str(x) for x in ort.get_available_providers())
    key = _device_provider_key(device)
    providers: List[Any] = []
    if key == "cuda" and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


@lru_cache(maxsize=8)
def _load_ort_session(model_file: str, provider_key: str):
    _require_onnxruntime()
    if provider_key == "cuda":
        providers = _ort_providers("cuda:0")
    else:
        providers = _ort_providers("cpu")
    return ort.InferenceSession(str(model_file), providers=providers)


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    denom = np.sum(exp, axis=axis, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return exp / denom


def _resize_to_width(image: np.ndarray, target_width: int) -> Tuple[np.ndarray, float]:
    _require_cv2()
    scale_ratio = float(target_width) / float(image.shape[1])
    resized = cv2.resize(
        image,
        (int(target_width), max(1, int(round(float(image.shape[0]) * scale_ratio)))),
        interpolation=cv2.INTER_LINEAR,
    )
    return resized, scale_ratio


def _pad_image(image: np.ndarray, pad_x: int, pad_y: int, pad_value: int = 255) -> np.ndarray:
    return np.pad(
        image,
        pad_width=((0, int(pad_y)), (0, int(pad_x)), (0, 0)),
        mode="constant",
        constant_values=int(pad_value),
    )


def _get_paddings(image: np.ndarray, patch_size: int) -> Tuple[int, int]:
    max_x = int(math.ceil(float(image.shape[1]) / float(patch_size))) * int(patch_size)
    max_y = int(math.ceil(float(image.shape[0]) / float(patch_size))) * int(patch_size)
    return max_x - int(image.shape[1]), max_y - int(image.shape[0])


def _bdrc_preprocess_detection_image(
    image: np.ndarray,
    *,
    patch_size: int,
    clamp_width: int = 4096,
    clamp_height: int = 2048,
) -> Tuple[np.ndarray, int, int]:
    _require_cv2()
    arr = _coerce_rgb_u8(image)
    if arr.shape[1] > arr.shape[0] and arr.shape[1] > clamp_width:
        arr, _ = _resize_to_width(arr, clamp_width)
    elif arr.shape[0] > arr.shape[1] and arr.shape[0] > clamp_height:
        arr, _ = resize_to_height(arr, clamp_height)
    elif arr.shape[0] < patch_size:
        arr, _ = resize_to_height(arr, patch_size)
    pad_x, pad_y = _get_paddings(arr, patch_size=patch_size)
    padded = _pad_image(arr, pad_x=pad_x, pad_y=pad_y, pad_value=255)
    return padded, int(pad_x), int(pad_y)


def _tile_image(padded_img: np.ndarray, *, patch_size: int) -> Tuple[List[np.ndarray], int]:
    x_steps = int(padded_img.shape[1] / int(patch_size))
    y_steps = int(padded_img.shape[0] / int(patch_size))
    y_splits = np.split(padded_img, y_steps, axis=0)
    patches = [np.split(x, x_steps, axis=1) for x in y_splits]
    return [x for xs in patches for x in xs], y_steps


def _stitch_predictions(prediction: np.ndarray, *, y_steps: int) -> np.ndarray:
    pred_y_split = np.split(prediction, int(y_steps), axis=0)
    x_slices = [np.hstack(x) for x in pred_y_split]
    return np.vstack(x_slices)


def _binarize_rgb(image: np.ndarray) -> np.ndarray:
    _require_cv2()
    line_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    bw = cv2.adaptiveThreshold(
        line_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51,
        13,
    )
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)


def _crop_prediction_to_image(
    image: np.ndarray,
    prediction: np.ndarray,
    *,
    pad_x: int,
    pad_y: int,
) -> np.ndarray:
    _require_cv2()
    x_lim = prediction.shape[1] - int(pad_x)
    y_lim = prediction.shape[0] - int(pad_y)
    cropped = prediction[:y_lim, :x_lim]
    return cv2.resize(cropped, dsize=(int(image.shape[1]), int(image.shape[0])))


def _get_input_output_names(session: Any) -> Tuple[str, str]:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return input_name, output_name


def _predict_bdrc_detection_mask(
    image: np.ndarray,
    *,
    cfg: BDRCLineModelConfig,
    device: str,
    class_threshold: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    _require_cv2()
    padded_img, pad_x, pad_y = _bdrc_preprocess_detection_image(image, patch_size=int(cfg.patch_size))
    tiles, y_steps = _tile_image(padded_img, patch_size=int(cfg.patch_size))
    tiles = np.asarray([_binarize_rgb(x).astype(np.float32) / 255.0 for x in tiles], dtype=np.float32)
    batch = np.transpose(tiles, axes=[0, 3, 1, 2]).astype(np.float32)

    session = _load_ort_session(str(cfg.model_file), _device_provider_key(device))
    input_name, output_name = _get_input_output_names(session)
    prediction = session.run([output_name], {input_name: batch})[0]

    debug = {
        "model_kind": cfg.model_kind,
        "patch_size": int(cfg.patch_size),
        "tiles": int(len(tiles)),
        "providers": list(session.get_providers()),
    }

    if cfg.model_kind == "layout":
        threshold = float(class_threshold if class_threshold is not None else DEFAULT_BDRC_LAYOUT_THRESHOLD)
        prediction = np.transpose(prediction, axes=[0, 2, 3, 1])
        prediction = _softmax(prediction, axis=-1)
        prediction = np.where(prediction > threshold, 1.0, 0.0)
        merged = _stitch_predictions(prediction, y_steps=y_steps)
        merged = _crop_prediction_to_image(_coerce_rgb_u8(image), merged, pad_x=pad_x, pad_y=pad_y)
        merged = (merged.astype(np.uint8) * 255)
        debug["class_threshold"] = threshold
        return merged, debug

    threshold = float(class_threshold if class_threshold is not None else DEFAULT_BDRC_LINE_THRESHOLD)
    prediction = np.squeeze(prediction, axis=1)
    prediction = 1.0 / (1.0 + np.exp(-prediction))
    prediction = np.where(prediction > threshold, 1.0, 0.0)
    merged = _stitch_predictions(prediction, y_steps=y_steps)
    merged = _crop_prediction_to_image(_coerce_rgb_u8(image), merged, pad_x=pad_x, pad_y=pad_y)
    merged = (merged.astype(np.uint8) * 255)
    debug["class_threshold"] = threshold
    return merged, debug


def _rotate_from_angle(image: np.ndarray, angle: float) -> np.ndarray:
    _require_cv2()
    rows, cols = image.shape[:2]
    rot_matrix = cv2.getRotationMatrix2D((cols / 2.0, rows / 2.0), float(angle), 1.0)
    return cv2.warpAffine(image, rot_matrix, (cols, rows), borderValue=(0, 0, 0))


def _inverse_rotate_points(points: np.ndarray, *, angle: float, width: int, height: int) -> np.ndarray:
    _require_cv2()
    if points.size == 0:
        return points
    matrix = cv2.getRotationMatrix2D((float(width) / 2.0, float(height) / 2.0), float(angle), 1.0)
    inv = cv2.invertAffineTransform(matrix)
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.transform(pts, inv).reshape(-1, 2)


def _clip_polygon(points: np.ndarray, *, width: int, height: int) -> Tuple[Tuple[int, int], ...]:
    out: List[Tuple[int, int]] = []
    for raw_x, raw_y in np.asarray(points, dtype=np.float32).reshape(-1, 2).tolist():
        if not np.isfinite(raw_x) or not np.isfinite(raw_y):
            continue
        x = max(0, min(int(width) - 1, int(round(float(raw_x)))))
        y = max(0, min(int(height) - 1, int(round(float(raw_y)))))
        pt = (x, y)
        if out and out[-1] == pt:
            continue
        out.append(pt)
    if len(out) >= 2 and out[0] == out[-1]:
        out.pop()
    if len(out) < 3:
        return tuple()
    return tuple(out)


def _contour_to_bbox(contour: np.ndarray, *, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    xs = [int(x) for x, _ in contour]
    ys = [int(y) for _, y in contour]
    if not xs or not ys:
        return None
    x1 = max(0, min(xs))
    y1 = max(0, min(ys))
    x2 = min(int(width), max(xs) + 1)
    y2 = min(int(height), max(ys) + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _get_rotation_angle_from_lines(line_mask: np.ndarray, *, max_angle: float = 5.0) -> float:
    _require_cv2()
    contours, _ = cv2.findContours(line_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask_threshold = float(line_mask.shape[0] * line_mask.shape[1]) * 0.001
    contours = [x for x in contours if cv2.contourArea(x) > mask_threshold]
    if not contours:
        return 0.0
    angles = [cv2.minAreaRect(x)[2] for x in contours]
    low_angles = [x for x in angles if abs(x) != 0.0 and x < max_angle]
    high_angles = [x for x in angles if abs(x) != 90.0 and x > (90 - max_angle)]
    if len(low_angles) > len(high_angles) and len(low_angles) > 0:
        return float(np.mean(low_angles))
    if len(high_angles) > 0:
        return float(-(90 - np.mean(high_angles)))
    return 0.0


def _build_raw_line_data(image: np.ndarray, line_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], float]:
    _require_cv2()
    if line_mask.ndim == 3:
        line_mask = cv2.cvtColor(line_mask, cv2.COLOR_BGR2GRAY)
    angle = _get_rotation_angle_from_lines(line_mask)
    rot_mask = _rotate_from_angle(line_mask, angle)
    rot_img = _rotate_from_angle(image, angle)
    contours, _ = cv2.findContours(rot_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = [x for x in contours if cv2.contourArea(x) > 10]
    return rot_img, rot_mask, contours, float(angle)


def _filter_line_contours(image: np.ndarray, line_contours: Iterable[np.ndarray], *, threshold: float = 0.01) -> List[np.ndarray]:
    _require_cv2()
    filtered: List[np.ndarray] = []
    for cnt in line_contours:
        _, _, w, h = cv2.boundingRect(cnt)
        if w > image.shape[1] * float(threshold) and h > 10:
            filtered.append(cnt)
    return filtered


def _mask_n_crop(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    _require_cv2()
    img = image.astype(np.uint8)
    mask_u8 = mask.astype(np.uint8)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    image_masked = cv2.bitwise_and(img, img, mask=mask_u8)
    image_masked = np.delete(image_masked, np.where(~image_masked.any(axis=1))[0], axis=0)
    image_masked = np.delete(image_masked, np.where(~image_masked.any(axis=0))[0], axis=1)
    if image_masked.size == 0:
        return np.zeros((max(1, int(mask.shape[0])), max(1, int(mask.shape[0]) * 2), 3), dtype=np.uint8)
    return image_masked


def _extract_line(image: np.ndarray, mask: np.ndarray, bbox_h: int, *, k_factor: float) -> np.ndarray:
    _require_cv2()
    k_size = max(1, int(round(float(bbox_h) * float(k_factor))))
    morph_rect = cv2.getStructuringElement(
        shape=cv2.MORPH_RECT,
        ksize=(int(k_size), max(1, int(round(float(k_size) * float(k_factor))))),
    )
    dilated_mask = cv2.dilate(mask, kernel=morph_rect, iterations=1)
    return _mask_n_crop(image, dilated_mask)


def _get_line_image(image: np.ndarray, mask: np.ndarray, bbox_h: int, *, bbox_tolerance: float, k_factor: float) -> Tuple[np.ndarray, float]:
    try:
        tmp_k = float(k_factor)
        line_img = _extract_line(image, mask, bbox_h, k_factor=tmp_k)
        attempts = 0
        while line_img.shape[0] > float(bbox_h) * float(bbox_tolerance) and attempts < 10:
            tmp_k -= 0.1
            if tmp_k <= 0.1:
                break
            line_img = _extract_line(image, mask, bbox_h, k_factor=tmp_k)
            attempts += 1
        return line_img, tmp_k
    except Exception:
        fallback = np.zeros((max(1, int(bbox_h)), max(1, int(bbox_h) * 2), 3), dtype=np.uint8)
        return fallback, float(k_factor)


def _extract_line_images(image: np.ndarray, contours: Sequence[np.ndarray], *, default_k: float, bbox_tolerance: float) -> List[np.ndarray]:
    _require_cv2()
    current_k = float(default_k)
    out: List[np.ndarray] = []
    for contour in contours:
        _, _, _, h = cv2.boundingRect(contour)
        tmp_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.drawContours(tmp_mask, [contour], -1, color=255, thickness=-1)
        line_img, adapted_k = _get_line_image(
            image,
            tmp_mask,
            h,
            bbox_tolerance=float(bbox_tolerance),
            k_factor=current_k,
        )
        out.append(line_img)
        current_k = float(adapted_k)
    return out


def _get_line_threshold(line_prediction: np.ndarray, *, slice_width: int = 20) -> float:
    _require_cv2()
    if line_prediction.ndim == 3:
        line_prediction = cv2.cvtColor(line_prediction, cv2.COLOR_BGR2GRAY)
    x, y, w, h = cv2.boundingRect(line_prediction)
    x_steps = (w // int(slice_width)) // 2
    bbox_numbers: List[Tuple[int, Sequence[np.ndarray]]] = []
    for step in range(1, x_steps + 1):
        x_offset = x_steps * step
        x_start = x + x_offset
        x_end = x_start + int(slice_width)
        _slice = line_prediction[y : y + h, x_start:x_end]
        contours, _ = cv2.findContours(_slice, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        bbox_numbers.append((len(contours), contours))
    sorted_list = sorted(bbox_numbers, key=lambda item: item[0], reverse=True)
    if not sorted_list:
        return 0.0
    n_contours, contours = sorted_list[0]
    if n_contours == 0:
        return 0.0
    y_points: List[int] = []
    for contour in contours:
        _, cy, _, ch = cv2.boundingRect(contour)
        y_points.append(int(cy + (ch // 2)))
    if not y_points:
        return 0.0
    return float(np.median(y_points) // n_contours)


def _sort_bbox_centers(bbox_centers: List[Tuple[int, int]], *, line_threshold: float) -> List[List[Tuple[int, int]]]:
    if not bbox_centers:
        return []
    sorted_bbox_centers: List[List[Tuple[int, int]]] = []
    tmp_line: List[Tuple[int, int]] = []
    for center in bbox_centers:
        if tmp_line:
            ys = [y for _, y in tmp_line]
            mean_y = float(np.mean(ys)) if ys else float(center[1])
            y_diff = abs(mean_y - float(center[1]))
            if y_diff > float(line_threshold):
                tmp_line.sort(key=lambda item: item[0])
                sorted_bbox_centers.append(tmp_line.copy())
                tmp_line = [center]
            else:
                tmp_line.append(center)
        else:
            tmp_line.append(center)
    if tmp_line:
        sorted_bbox_centers.append(tmp_line)
    for line in sorted_bbox_centers:
        line.sort(key=lambda item: item[0])
    return list(reversed(sorted_bbox_centers))


def _group_line_chunks(sorted_bbox_centers: List[List[Tuple[int, int]]], contours: Sequence[np.ndarray], centers: Sequence[Tuple[int, int]]) -> List[np.ndarray]:
    _require_cv2()
    contour_by_center = {
        (int(center[0]), int(center[1])): contour
        for center, contour in zip(centers, contours)
    }
    grouped: List[np.ndarray] = []
    for bbox_group in sorted_bbox_centers:
        if len(bbox_group) <= 1:
            contour = contour_by_center.get((int(bbox_group[0][0]), int(bbox_group[0][1]))) if bbox_group else None
            if contour is not None:
                grouped.append(contour)
            continue
        contour_stack = [contour_by_center[(int(cx), int(cy))] for cx, cy in bbox_group if (int(cx), int(cy)) in contour_by_center]
        if not contour_stack:
            continue
        stacked = np.vstack(contour_stack)
        grouped.append(cv2.convexHull(stacked))
    return grouped


def _sort_and_group_contours(line_mask: np.ndarray, contours: Sequence[np.ndarray], *, group_lines: bool) -> Tuple[List[np.ndarray], float]:
    _require_cv2()
    centers: List[Tuple[int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        centers.append((int(x + (w // 2)), int(y + (h // 2))))
    line_threshold = _get_line_threshold(line_mask)
    sorted_centers = _sort_bbox_centers(centers, line_threshold=line_threshold)
    if group_lines:
        grouped = _group_line_chunks(sorted_centers, contours, centers)
        return grouped, line_threshold
    flat_centers = [center for group in sorted_centers for center in group]
    contour_by_center = {(int(cx), int(cy)): contour for (cx, cy), contour in zip(centers, contours)}
    ordered = [contour_by_center[(int(cx), int(cy))] for cx, cy in flat_centers if (int(cx), int(cy)) in contour_by_center]
    return ordered, line_threshold


def predict_bdrc_line_regions(
    image: Any,
    *,
    model_path: str,
    device: str = "auto",
    class_threshold: Optional[float] = None,
    group_lines: bool = True,
    k_factor: float = DEFAULT_BDRC_LINE_K_FACTOR,
    bbox_tolerance: float = DEFAULT_BDRC_LINE_BBOX_TOLERANCE,
) -> Tuple[List[BDRCLinePrediction], Dict[str, Any]]:
    """Run a BDRC line/layout ONNX model plus the BDRC contour postprocessing chain."""

    _require_cv2()
    src = _coerce_rgb_u8(image)
    h, w = src.shape[:2]
    cfg = resolve_bdrc_line_model_config(model_path)
    prediction_mask, det_debug = _predict_bdrc_detection_mask(
        src,
        cfg=cfg,
        device=device,
        class_threshold=class_threshold,
    )
    if cfg.model_kind == "layout":
        try:
            line_idx = list(cfg.classes).index("line")
        except ValueError:
            line_idx = 2
        if prediction_mask.ndim != 3 or prediction_mask.shape[2] <= line_idx:
            raise RuntimeError("BDRC layout model did not return a usable line channel.")
        line_mask = prediction_mask[:, :, line_idx]
    else:
        line_mask = prediction_mask

    rot_img, rot_mask, line_contours, page_angle = _build_raw_line_data(src, line_mask)
    filtered = _filter_line_contours(rot_mask, line_contours)
    grouped, line_threshold = _sort_and_group_contours(rot_mask, filtered, group_lines=bool(group_lines))
    line_images = _extract_line_images(rot_img, grouped, default_k=float(k_factor), bbox_tolerance=float(bbox_tolerance))

    predictions: List[BDRCLinePrediction] = []
    for idx, contour in enumerate(grouped):
        pts = contour.reshape(-1, 2)
        orig_pts = _inverse_rotate_points(pts, angle=page_angle, width=w, height=h)
        polygon = _clip_polygon(orig_pts, width=w, height=h)
        if not polygon:
            continue
        box = _contour_to_bbox(np.asarray(polygon, dtype=np.int32), width=w, height=h)
        if box is None:
            continue
        crop = line_images[idx] if idx < len(line_images) else None
        if crop is not None:
            crop = _coerce_rgb_u8(crop)
        predictions.append(
            BDRCLinePrediction(
                box=box,
                polygon=polygon,
                confidence=1.0,
                class_id=0,
                label="line",
                crop_image=crop,
                page_angle=float(page_angle),
            )
        )

    predictions.sort(key=lambda pred: ((pred.box[1] + pred.box[3]) / 2.0, pred.box[1], pred.box[0]))
    debug = {
        "ok": True,
        "backend": "bdrc_line_model",
        "model_dir": cfg.model_dir,
        "model_file": cfg.model_file,
        "model_kind": cfg.model_kind,
        "classes": list(cfg.classes),
        "device": _device_provider_key(device),
        "page_angle": float(page_angle),
        "line_threshold": float(line_threshold),
        "group_lines": bool(group_lines),
        "k_factor": float(k_factor),
        "bbox_tolerance": float(bbox_tolerance),
        "raw_contours": int(len(line_contours)),
        "filtered_contours": int(len(filtered)),
        "grouped_contours": int(len(grouped)),
        "line_count": int(len(predictions)),
        "detection": det_debug,
        "predictions": [pred.to_dict() for pred in predictions[:256]],
    }
    return predictions, debug


def _bdrc_ocr_preprocess_config(cfg: BDRCOCRModelConfig) -> BDRCPreprocessConfig:
    base = BDRCPreprocessConfig.ocr_line_defaults()
    payload = dict(base.to_dict())
    payload["target_width"] = int(cfg.input_width)
    payload["target_height"] = int(cfg.input_height)
    payload["padding"] = "black"
    return BDRCPreprocessConfig.from_dict(payload)


def _pre_pad_line(image: np.ndarray) -> np.ndarray:
    arr = _coerce_rgb_u8(image)
    h = int(arr.shape[0])
    patch = np.ones((h, h, 3), dtype=np.uint8) * 255
    return np.hstack([patch, arr, patch])


def _ctc_greedy_decode(logits: np.ndarray, *, charset: Sequence[str], add_blank: bool) -> str:
    arr = np.asarray(logits)
    vocab_size = len(charset) + (1 if add_blank else 0)
    if arr.ndim == 1:
        arr = np.expand_dims(arr, axis=0)
    if arr.ndim == 2 and arr.shape[0] == vocab_size and arr.shape[1] != vocab_size:
        arr = np.transpose(arr, axes=[1, 0])
    elif arr.ndim > 2:
        arr = np.squeeze(arr)
        if arr.ndim == 2 and arr.shape[0] == vocab_size and arr.shape[1] != vocab_size:
            arr = np.transpose(arr, axes=[1, 0])
    if arr.ndim != 2:
        raise ValueError(f"Unexpected OCR logits shape: {arr.shape}")

    token_ids = np.argmax(arr, axis=-1).tolist()
    blank_id = 0 if add_blank else None
    prev: Optional[int] = None
    out: List[str] = []
    for raw_idx in token_ids:
        idx = int(raw_idx)
        if blank_id is not None and idx == blank_id:
            prev = idx
            continue
        if prev is not None and idx == prev:
            continue
        char_idx = idx - 1 if add_blank else idx
        if 0 <= char_idx < len(charset):
            out.append(str(charset[char_idx]))
        prev = idx
    return "".join(out)


def _maybe_convert_text(text: str, *, encoder: str, target_encoding: str) -> str:
    enc = str(encoder or "").strip().lower()
    tgt = str(target_encoding or "raw").strip().lower()
    if not text:
        return ""
    if pyewts is None:
        return text
    try:
        converter = pyewts.pyewts()
    except Exception:
        return text
    if enc == "wylie" and tgt in {"unicode", "bo"}:
        try:
            return str(converter.toUnicode(text))
        except Exception:
            return text
    if enc in {"stack", "unicode"} and tgt == "wylie":
        try:
            return str(converter.toWylie(text))
        except Exception:
            return text
    return text


def run_bdrc_ocr(
    image: Any,
    *,
    model_path: str,
    device: str = "auto",
    target_encoding: str = "raw",
) -> Tuple[str, Dict[str, Any], np.ndarray]:
    """Run a BDRC OCR ONNX model on one line image."""

    _require_onnxruntime()
    cfg = resolve_bdrc_ocr_model_config(model_path)
    rgb = _pre_pad_line(_coerce_rgb_u8(image))
    proc_cfg = _bdrc_ocr_preprocess_config(cfg)
    proc_pil = preprocess_image_bdrc(Image.fromarray(rgb, mode="RGB"), config=proc_cfg)
    proc_gray = np.asarray(proc_pil, dtype=np.uint8)
    tensor = bdrc_image_to_normalized_tensor(Image.fromarray(rgb, mode="RGB"), config=proc_cfg)
    tensor = tensor.reshape((1, int(cfg.input_height), int(cfg.input_width)))
    if bool(cfg.swap_hw):
        tensor = np.transpose(tensor, axes=[0, 2, 1])
    if not bool(cfg.squeeze_channel):
        tensor = np.expand_dims(tensor, axis=1)
    tensor = tensor.astype(np.float32)

    session = _load_ort_session(str(cfg.model_file), _device_provider_key(device))
    input_name = str(cfg.input_layer or _get_input_output_names(session)[0])
    output_name = str(cfg.output_layer or _get_input_output_names(session)[1])
    logits = session.run([output_name], {input_name: tensor})[0]
    logits = np.squeeze(logits)
    text_raw = _ctc_greedy_decode(logits, charset=cfg.charset, add_blank=bool(cfg.add_blank))
    text = _maybe_convert_text(text_raw, encoder=cfg.encoder, target_encoding=target_encoding)

    debug = {
        "ok": True,
        "backend": "bdrc_ocr",
        "model_dir": cfg.model_dir,
        "model_file": cfg.model_file,
        "device": _device_provider_key(device),
        "providers": list(session.get_providers()),
        "input_width": int(cfg.input_width),
        "input_height": int(cfg.input_height),
        "input_layer": str(input_name),
        "output_layer": str(output_name),
        "encoder": str(cfg.encoder),
        "target_encoding": str(target_encoding or "raw"),
        "charset_size": int(len(cfg.charset)),
        "add_blank": bool(cfg.add_blank),
        "swap_hw": bool(cfg.swap_hw),
        "squeeze_channel": bool(cfg.squeeze_channel),
        "output_length_chars": int(len(text)),
    }
    return str(text or "").strip(), debug, proc_gray


__all__ = [
    "BDRCLineModelConfig",
    "BDRCOCRModelConfig",
    "BDRCLinePrediction",
    "DEFAULT_BDRC_LINE_BBOX_TOLERANCE",
    "DEFAULT_BDRC_LINE_K_FACTOR",
    "DEFAULT_BDRC_LAYOUT_THRESHOLD",
    "DEFAULT_BDRC_LINE_THRESHOLD",
    "find_bdrc_line_model_dirs",
    "find_bdrc_ocr_model_dirs",
    "is_bdrc_line_model_dir",
    "is_bdrc_ocr_model_dir",
    "predict_bdrc_line_regions",
    "resolve_bdrc_line_model_config",
    "resolve_bdrc_ocr_model_config",
    "run_bdrc_ocr",
]
