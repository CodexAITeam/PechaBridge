"""Deterministic OCR pre-processing for patch images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageFilter, ImageOps

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    txt = str(value).strip().lower()
    if txt in {"1", "true", "yes", "y", "on"}:
        return True
    if txt in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _normalize_invert(value: Any) -> str:
    txt = str(value if value is not None else "auto").strip().lower()
    if txt in {"true", "false", "auto"}:
        return txt
    return "auto"


def _normalize_binarize(value: Any) -> str:
    txt = str(value if value is not None else "otsu").strip().lower()
    if txt in {"none", "otsu", "adaptive"}:
        return txt
    return "otsu"


@dataclass(frozen=True)
class PreprocessConfig:
    grayscale: bool = True
    invert: str = "auto"  # auto|true|false
    pad_px: int = 8
    resize_height: int = 192
    denoise: bool = False
    binarize: str = "otsu"  # none|otsu|adaptive
    clahe: bool = False

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "PreprocessConfig":
        p = dict(payload or {})
        return PreprocessConfig(
            grayscale=_as_bool(p.get("grayscale"), True),
            invert=_normalize_invert(p.get("invert")),
            pad_px=max(0, int(p.get("pad_px", 8))),
            resize_height=max(1, int(p.get("resize_height", 192))),
            denoise=_as_bool(p.get("denoise"), False),
            binarize=_normalize_binarize(p.get("binarize")),
            clahe=_as_bool(p.get("clahe"), False),
        )

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "grayscale": bool(self.grayscale),
            "invert": str(self.invert),
            "pad_px": int(self.pad_px),
            "resize_height": int(self.resize_height),
            "denoise": bool(self.denoise),
            "binarize": str(self.binarize),
            "clahe": bool(self.clahe),
        }


def _should_invert(gray_u8: np.ndarray, mode: str) -> bool:
    md = str(mode or "auto").strip().lower()
    if md == "true":
        return True
    if md == "false":
        return False
    dark_fraction = float(np.mean(gray_u8 < 128))
    return bool(dark_fraction > 0.5)


def _apply_clahe(gray_u8: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray_u8)
    return np.asarray(ImageOps.autocontrast(Image.fromarray(gray_u8), cutoff=0), dtype=np.uint8)


def _otsu_threshold(gray_u8: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        _thr, bw = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bw.astype(np.uint8)

    hist = np.bincount(gray_u8.reshape(-1), minlength=256).astype(np.float64)
    total = float(gray_u8.size)
    sum_all = float(np.sum(np.arange(256, dtype=np.float64) * hist))
    sum_bg = 0.0
    weight_bg = 0.0
    var_max = -1.0
    best_thr = 127

    for thr in range(256):
        weight_bg += hist[thr]
        if weight_bg <= 0.0:
            continue
        weight_fg = total - weight_bg
        if weight_fg <= 0.0:
            break
        sum_bg += float(thr) * hist[thr]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > var_max:
            var_max = between
            best_thr = thr
    return np.where(gray_u8 > best_thr, 255, 0).astype(np.uint8)


def _adaptive_threshold(gray_u8: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        return cv2.adaptiveThreshold(
            gray_u8,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            8,
        ).astype(np.uint8)
    return _otsu_threshold(gray_u8)


def _resize_to_height(gray_u8: np.ndarray, target_height: int) -> np.ndarray:
    th = max(1, int(target_height))
    h, w = gray_u8.shape[:2]
    if h == th:
        return gray_u8
    tw = max(1, int(round(float(w) * (float(th) / float(max(1, h))))))
    if cv2 is not None:
        return cv2.resize(gray_u8, (tw, th), interpolation=cv2.INTER_CUBIC).astype(np.uint8)
    resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
    return np.asarray(Image.fromarray(gray_u8).resize((tw, th), resample=resample), dtype=np.uint8)


def preprocess_patch_image(image: Image.Image, config: PreprocessConfig) -> Image.Image:
    """Apply deterministic OCR pre-processing and return an `L` image."""
    if image is None:
        raise ValueError("image is required")

    if bool(config.grayscale):
        gray_u8 = np.asarray(image.convert("L"), dtype=np.uint8)
    else:
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
        gray_u8 = np.mean(arr.astype(np.float32), axis=2).astype(np.uint8)

    if _should_invert(gray_u8, str(config.invert)):
        gray_u8 = (255 - gray_u8).astype(np.uint8)

    if bool(config.clahe):
        gray_u8 = _apply_clahe(gray_u8)

    if bool(config.denoise):
        if cv2 is not None:
            gray_u8 = cv2.medianBlur(gray_u8, 3).astype(np.uint8)
        else:
            gray_u8 = np.asarray(Image.fromarray(gray_u8).filter(ImageFilter.MedianFilter(size=3)), dtype=np.uint8)

    pad = max(0, int(config.pad_px))
    if pad > 0:
        gray_u8 = np.pad(gray_u8, ((pad, pad), (pad, pad)), mode="constant", constant_values=255)

    gray_u8 = _resize_to_height(gray_u8, int(config.resize_height))

    mode = _normalize_binarize(config.binarize)
    if mode == "otsu":
        gray_u8 = _otsu_threshold(gray_u8)
    elif mode == "adaptive":
        gray_u8 = _adaptive_threshold(gray_u8)

    return Image.fromarray(gray_u8.astype(np.uint8), mode="L")


__all__ = ["PreprocessConfig", "preprocess_patch_image"]
