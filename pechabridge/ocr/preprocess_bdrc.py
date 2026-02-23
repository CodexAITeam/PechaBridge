"""BDRC-style OCR image preprocessing utilities (adapted for reuse in PechaBridge).

This module mirrors the core image preprocessing logic used in BDRC's OCR training
pipeline:
  - optional adaptive/fixed threshold binarization
  - grayscale conversion
  - aspect-preserving resize + padding for OCR line models
  - optional normalization to [-1, 1]

The functions are intentionally usable from multiple training pipelines
(e.g. DONUT, CLIP-style dual encoders, OCR weak-labeling).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

import numpy as np
from PIL import Image

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


@dataclass(frozen=True)
class BDRCPreprocessConfig:
    """Configuration for BDRC-like deterministic preprocessing."""

    binarize: bool = True
    adaptive_threshold: bool = True
    threshold_block_size: int = 51
    threshold_c: int = 13
    fixed_threshold: int = 120
    pad_px: int = 0
    resize_height: int = 0  # 0 disables fixed-height resize
    target_width: int = 0  # only used by pad_ocr_line helpers
    target_height: int = 0  # only used by pad_ocr_line helpers
    padding: str = "black"  # black|white

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "BDRCPreprocessConfig":
        p = dict(payload or {})
        block = int(p.get("threshold_block_size", 51))
        if block % 2 == 0:
            block += 1
        block = max(3, block)
        return BDRCPreprocessConfig(
            binarize=_as_bool(p.get("binarize"), True),
            adaptive_threshold=_as_bool(p.get("adaptive_threshold"), True),
            threshold_block_size=block,
            threshold_c=int(p.get("threshold_c", 13)),
            fixed_threshold=int(p.get("fixed_threshold", 120)),
            pad_px=max(0, int(p.get("pad_px", 0))),
            resize_height=max(0, int(p.get("resize_height", 0))),
            target_width=max(0, int(p.get("target_width", 0))),
            target_height=max(0, int(p.get("target_height", 0))),
            padding=str(p.get("padding", "black") or "black").strip().lower(),
        )

    @staticmethod
    def vit_defaults() -> "BDRCPreprocessConfig":
        """Defaults for ViT/DINO-style encoders (binarize only, no forced OCR size)."""
        return BDRCPreprocessConfig(
            binarize=True,
            adaptive_threshold=True,
            threshold_block_size=51,
            threshold_c=13,
            fixed_threshold=120,
            pad_px=0,
            resize_height=0,
            target_width=0,
            target_height=0,
            padding="white",
        )

    @staticmethod
    def ocr_line_defaults() -> "BDRCPreprocessConfig":
        """Defaults mirroring classic OCR line preprocessing (BDRC-like)."""
        return BDRCPreprocessConfig(
            binarize=True,
            adaptive_threshold=True,
            threshold_block_size=51,
            threshold_c=13,
            fixed_threshold=120,
            pad_px=0,
            resize_height=0,
            target_width=2000,
            target_height=80,
            padding="black",
        )

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "binarize": bool(self.binarize),
            "adaptive_threshold": bool(self.adaptive_threshold),
            "threshold_block_size": int(self.threshold_block_size),
            "threshold_c": int(self.threshold_c),
            "fixed_threshold": int(self.fixed_threshold),
            "pad_px": int(self.pad_px),
            "resize_height": int(self.resize_height),
            "target_width": int(self.target_width),
            "target_height": int(self.target_height),
            "padding": str(self.padding),
        }


def _to_gray_u8(image: Image.Image) -> np.ndarray:
    if image is None:
        raise ValueError("image is required")
    return np.asarray(image.convert("L"), dtype=np.uint8)


def _binarize_gray_u8(
    gray_u8: np.ndarray,
    *,
    adaptive_threshold: bool = True,
    block_size: int = 51,
    c: int = 13,
    fixed_threshold: int = 120,
) -> np.ndarray:
    if gray_u8.ndim != 2:
        raise ValueError("Expected grayscale image [H,W].")
    if cv2 is None:
        thr = int(max(0, min(255, fixed_threshold)))
        return np.where(gray_u8 > thr, 255, 0).astype(np.uint8)
    if bool(adaptive_threshold):
        bs = int(max(3, block_size))
        if bs % 2 == 0:
            bs += 1
        return cv2.adaptiveThreshold(
            gray_u8,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            bs,
            int(c),
        ).astype(np.uint8)
    _thr, bw = cv2.threshold(gray_u8, int(max(0, min(255, fixed_threshold))), 255, cv2.THRESH_BINARY)
    return bw.astype(np.uint8)


def resize_to_height(gray_u8: np.ndarray, target_height: int) -> Tuple[np.ndarray, float]:
    h, w = gray_u8.shape[:2]
    th = max(1, int(target_height))
    if h <= 0 or w <= 0:
        return np.zeros((th, 1), dtype=np.uint8), 1.0
    ratio = float(th) / float(h)
    tw = max(1, int(round(float(w) * ratio)))
    if cv2 is not None:
        out = cv2.resize(gray_u8, (tw, th), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    else:
        resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
        out = np.asarray(Image.fromarray(gray_u8).resize((tw, th), resample=resample), dtype=np.uint8)
    return out, ratio


def _pad_center(gray_u8: np.ndarray, target_width: int, target_height: int, padding: str = "black") -> np.ndarray:
    th = max(1, int(target_height))
    tw = max(1, int(target_width))
    h, w = gray_u8.shape[:2]
    fill = 255 if str(padding or "").strip().lower() == "white" else 0
    canvas = np.full((th, tw), fill_value=fill, dtype=np.uint8)
    y0 = max(0, (th - h) // 2)
    x0 = max(0, (tw - w) // 2)
    y1 = min(th, y0 + h)
    x1 = min(tw, x0 + w)
    src_y1 = min(h, y1 - y0)
    src_x1 = min(w, x1 - x0)
    canvas[y0:y1, x0:x1] = gray_u8[:src_y1, :src_x1]
    return canvas


def pad_ocr_line(
    gray_u8: np.ndarray,
    *,
    target_width: int = 2000,
    target_height: int = 80,
    padding: str = "black",
) -> np.ndarray:
    """Aspect-preserving resize + pad to a fixed OCR line canvas."""
    h, w = gray_u8.shape[:2]
    tw = max(1, int(target_width))
    th = max(1, int(target_height))
    if h <= 0 or w <= 0:
        return np.zeros((th, tw), dtype=np.uint8)
    width_ratio = float(tw) / float(w)
    height_ratio = float(th) / float(h)
    if width_ratio < height_ratio:
        # Fit width
        new_h = max(1, int(round(float(h) * width_ratio)))
        if cv2 is not None:
            resized = cv2.resize(gray_u8, (tw, new_h), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        else:
            resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
            resized = np.asarray(Image.fromarray(gray_u8).resize((tw, new_h), resample=resample), dtype=np.uint8)
    else:
        # Fit height
        resized, _ = resize_to_height(gray_u8, th)
    out = _pad_center(resized, target_width=tw, target_height=th, padding=padding)
    if cv2 is not None:
        return cv2.resize(out, (tw, th), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    return out


def preprocess_image_bdrc(image: Image.Image, config: BDRCPreprocessConfig | None = None) -> Image.Image:
    """BDRC-style deterministic preprocessing returning a grayscale PIL image."""
    cfg = config or BDRCPreprocessConfig.vit_defaults()
    gray_u8 = _to_gray_u8(image)
    if bool(cfg.binarize):
        gray_u8 = _binarize_gray_u8(
            gray_u8,
            adaptive_threshold=bool(cfg.adaptive_threshold),
            block_size=int(cfg.threshold_block_size),
            c=int(cfg.threshold_c),
            fixed_threshold=int(cfg.fixed_threshold),
        )
    pad_px = max(0, int(cfg.pad_px))
    if pad_px > 0:
        fill = 255 if str(cfg.padding).lower() == "white" else 0
        gray_u8 = np.pad(gray_u8, ((pad_px, pad_px), (pad_px, pad_px)), mode="constant", constant_values=fill)
    if int(cfg.resize_height) > 0:
        gray_u8, _ = resize_to_height(gray_u8, int(cfg.resize_height))
    if int(cfg.target_width) > 0 and int(cfg.target_height) > 0:
        gray_u8 = pad_ocr_line(
            gray_u8,
            target_width=int(cfg.target_width),
            target_height=int(cfg.target_height),
            padding=str(cfg.padding),
        )
    return Image.fromarray(gray_u8.astype(np.uint8), mode="L")


def bdrc_image_to_normalized_tensor(
    image: Image.Image,
    config: BDRCPreprocessConfig | None = None,
) -> np.ndarray:
    """Convenience helper for OCR-style models: preprocess + normalize to [-1,1]."""
    out = preprocess_image_bdrc(image=image, config=config)
    arr = np.asarray(out, dtype=np.float32)
    return ((arr / 127.5) - 1.0).astype(np.float32)


__all__ = [
    "BDRCPreprocessConfig",
    "preprocess_image_bdrc",
    "resize_to_height",
    "pad_ocr_line",
    "bdrc_image_to_normalized_tensor",
]

