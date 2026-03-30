"""BDRC-style OCR image preprocessing utilities (adapted for reuse in PechaBridge).

This module mirrors the core image preprocessing logic used in BDRC's OCR training
pipeline and extends it with optional switches useful for historical Tibetan scans:
  - optional grayscale conversion variants (color-channel aware)
  - optional background normalization
  - optional pre-binarization upscaling
  - optional adaptive/otsu/fixed threshold binarization
  - optional morphology + tiny component filtering
  - aspect-preserving resize + padding for OCR line models
  - optional normalization to [-1, 1]

The functions are intentionally usable from multiple training pipelines
(e.g. DONUT, CLIP-style dual encoders, OCR weak-labeling).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

import numpy as np
from PIL import Image, ImageFilter

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


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (float, int)):
        return float(value)
    try:
        return float(str(value).strip())
    except Exception:
        return float(default)


def _as_choice(value: Any, *, allowed: set[str], default: str) -> str:
    txt = str(value or "").strip().lower()
    return txt if txt in allowed else str(default)


@dataclass(frozen=True)
class BDRCPreprocessConfig:
    """Configuration for BDRC-like deterministic preprocessing."""

    gray_mode: str = "luma"  # luma|min_rgb|max_rgb|r|g|b
    normalize_background: bool = False
    background_blur_ksize: int = 0  # 0 => auto
    background_strength: float = 1.0
    upscale_factor: float = 1.0  # >1 applies pre-binarization upscaling
    upscale_interpolation: str = "lanczos"  # nearest|linear|cubic|lanczos
    binarize: bool = True
    threshold_method: str = "adaptive"  # adaptive|otsu|fixed
    adaptive_threshold: bool = True
    threshold_block_size: int = 51
    threshold_c: int = 13
    fixed_threshold: int = 120
    morph_close: bool = False
    morph_close_kernel: int = 2
    remove_small_components: bool = False
    min_component_area: int = 12
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
        close_kernel = max(0, int(p.get("morph_close_kernel", 2)))
        if close_kernel > 0 and close_kernel % 2 == 0:
            close_kernel += 1
        blur_ksize = max(0, int(p.get("background_blur_ksize", 0)))
        if blur_ksize > 0 and blur_ksize % 2 == 0:
            blur_ksize += 1
        method = _as_choice(
            p.get("threshold_method"),
            allowed={"adaptive", "otsu", "fixed"},
            default=("adaptive" if _as_bool(p.get("adaptive_threshold"), True) else "fixed"),
        )
        return BDRCPreprocessConfig(
            gray_mode=_as_choice(
                p.get("gray_mode"),
                allowed={"luma", "min_rgb", "max_rgb", "r", "g", "b"},
                default="luma",
            ),
            normalize_background=_as_bool(p.get("normalize_background"), False),
            background_blur_ksize=blur_ksize,
            background_strength=max(0.0, min(3.0, _as_float(p.get("background_strength"), 1.0))),
            upscale_factor=max(1.0, _as_float(p.get("upscale_factor"), 1.0)),
            upscale_interpolation=_as_choice(
                p.get("upscale_interpolation"),
                allowed={"nearest", "linear", "cubic", "lanczos"},
                default="lanczos",
            ),
            binarize=_as_bool(p.get("binarize"), True),
            threshold_method=method,
            adaptive_threshold=_as_bool(p.get("adaptive_threshold"), True),
            threshold_block_size=block,
            threshold_c=int(p.get("threshold_c", 13)),
            fixed_threshold=int(p.get("fixed_threshold", 120)),
            morph_close=_as_bool(p.get("morph_close"), False),
            morph_close_kernel=close_kernel,
            remove_small_components=_as_bool(p.get("remove_small_components"), False),
            min_component_area=max(0, int(p.get("min_component_area", 12))),
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
            gray_mode="luma",
            normalize_background=False,
            background_blur_ksize=0,
            background_strength=1.0,
            upscale_factor=1.0,
            upscale_interpolation="lanczos",
            binarize=True,
            threshold_method="adaptive",
            adaptive_threshold=True,
            threshold_block_size=51,
            threshold_c=13,
            fixed_threshold=120,
            morph_close=False,
            morph_close_kernel=2,
            remove_small_components=False,
            min_component_area=12,
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
            gray_mode="luma",
            normalize_background=False,
            background_blur_ksize=0,
            background_strength=1.0,
            upscale_factor=1.0,
            upscale_interpolation="lanczos",
            binarize=True,
            threshold_method="adaptive",
            adaptive_threshold=True,
            threshold_block_size=51,
            threshold_c=13,
            fixed_threshold=120,
            morph_close=False,
            morph_close_kernel=2,
            remove_small_components=False,
            min_component_area=12,
            pad_px=0,
            resize_height=0,
            target_width=2000,
            target_height=80,
            padding="black",
        )

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "gray_mode": str(self.gray_mode),
            "normalize_background": bool(self.normalize_background),
            "background_blur_ksize": int(self.background_blur_ksize),
            "background_strength": float(self.background_strength),
            "upscale_factor": float(self.upscale_factor),
            "upscale_interpolation": str(self.upscale_interpolation),
            "binarize": bool(self.binarize),
            "threshold_method": str(self.threshold_method),
            "adaptive_threshold": bool(self.adaptive_threshold),
            "threshold_block_size": int(self.threshold_block_size),
            "threshold_c": int(self.threshold_c),
            "fixed_threshold": int(self.fixed_threshold),
            "morph_close": bool(self.morph_close),
            "morph_close_kernel": int(self.morph_close_kernel),
            "remove_small_components": bool(self.remove_small_components),
            "min_component_area": int(self.min_component_area),
            "pad_px": int(self.pad_px),
            "resize_height": int(self.resize_height),
            "target_width": int(self.target_width),
            "target_height": int(self.target_height),
            "padding": str(self.padding),
        }


def _to_gray_u8(image: Image.Image, gray_mode: str = "luma") -> np.ndarray:
    if image is None:
        raise ValueError("image is required")
    mode = _as_choice(
        gray_mode,
        allowed={"luma", "min_rgb", "max_rgb", "r", "g", "b"},
        default="luma",
    )
    if mode == "luma":
        return np.asarray(image.convert("L"), dtype=np.uint8)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    r = rgb[:, :, 0].astype(np.uint16)
    g = rgb[:, :, 1].astype(np.uint16)
    b = rgb[:, :, 2].astype(np.uint16)
    if mode == "min_rgb":
        return np.minimum(np.minimum(r, g), b).astype(np.uint8)
    if mode == "max_rgb":
        return np.maximum(np.maximum(r, g), b).astype(np.uint8)
    if mode == "r":
        return r.astype(np.uint8)
    if mode == "g":
        return g.astype(np.uint8)
    return b.astype(np.uint8)


def _binarize_gray_u8(
    gray_u8: np.ndarray,
    *,
    threshold_method: str = "adaptive",
    adaptive_threshold: bool = True,
    block_size: int = 51,
    c: int = 13,
    fixed_threshold: int = 120,
) -> np.ndarray:
    if gray_u8.ndim != 2:
        raise ValueError("Expected grayscale image [H,W].")
    method = _as_choice(
        threshold_method,
        allowed={"adaptive", "otsu", "fixed"},
        default=("adaptive" if bool(adaptive_threshold) else "fixed"),
    )
    if cv2 is None:
        thr = int(max(0, min(255, fixed_threshold)))
        return np.where(gray_u8 > thr, 255, 0).astype(np.uint8)
    if method == "adaptive":
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
    if method == "otsu":
        _thr, bw = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bw.astype(np.uint8)
    _thr, bw = cv2.threshold(gray_u8, int(max(0, min(255, fixed_threshold))), 255, cv2.THRESH_BINARY)
    return bw.astype(np.uint8)


def _resize_gray_u8(gray_u8: np.ndarray, target_w: int, target_h: int, interpolation: str) -> np.ndarray:
    tw = max(1, int(target_w))
    th = max(1, int(target_h))
    mode = _as_choice(
        interpolation,
        allowed={"nearest", "linear", "cubic", "lanczos"},
        default="linear",
    )
    if cv2 is not None:
        cv_interp = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
        }[mode]
        return cv2.resize(gray_u8, (tw, th), interpolation=cv_interp).astype(np.uint8)
    if hasattr(Image, "Resampling"):
        pil_interp = {
            "nearest": Image.Resampling.NEAREST,
            "linear": Image.Resampling.BILINEAR,
            "cubic": Image.Resampling.BICUBIC,
            "lanczos": Image.Resampling.LANCZOS,
        }[mode]
    else:
        pil_interp = {
            "nearest": Image.NEAREST,
            "linear": Image.BILINEAR,
            "cubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }[mode]
    return np.asarray(Image.fromarray(gray_u8).resize((tw, th), resample=pil_interp), dtype=np.uint8)


def _upscale_gray_u8(gray_u8: np.ndarray, factor: float, interpolation: str = "lanczos") -> np.ndarray:
    fac = max(1.0, float(factor))
    if fac <= 1.0 + 1e-6:
        return gray_u8
    h, w = gray_u8.shape[:2]
    tw = max(1, int(round(float(w) * fac)))
    th = max(1, int(round(float(h) * fac)))
    if tw == w and th == h:
        return gray_u8
    return _resize_gray_u8(gray_u8, tw, th, interpolation=interpolation)


def _normalize_background_u8(gray_u8: np.ndarray, blur_ksize: int = 0, strength: float = 1.0) -> np.ndarray:
    if gray_u8.ndim != 2:
        raise ValueError("Expected grayscale image [H,W].")
    h, w = gray_u8.shape[:2]
    if h <= 0 or w <= 0:
        return gray_u8

    k = int(max(0, blur_ksize))
    if k <= 0:
        auto = max(15, int(round(float(min(h, w)) * 0.08)))
        if auto % 2 == 0:
            auto += 1
        k = auto
    if k % 2 == 0:
        k += 1
    if k < 3:
        return gray_u8

    if cv2 is not None:
        bg = cv2.GaussianBlur(gray_u8, (k, k), sigmaX=0).astype(np.float32)
    else:
        radius = max(1.0, float(k) / 6.0)
        bg = np.asarray(Image.fromarray(gray_u8).filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32)

    src_f = gray_u8.astype(np.float32)
    bg = np.maximum(bg, 1.0)
    normalized = (src_f * 255.0) / bg
    alpha = max(0.0, min(3.0, float(strength)))
    if abs(alpha - 1.0) > 1e-6:
        normalized = (src_f * (1.0 - alpha)) + (normalized * alpha)
    return np.clip(normalized, 0.0, 255.0).astype(np.uint8)


def _morph_close_binary_u8(binary_u8: np.ndarray, kernel_size: int = 0) -> np.ndarray:
    ks = int(max(0, kernel_size))
    if ks <= 0 or cv2 is None:
        return binary_u8
    if ks % 2 == 0:
        ks += 1
    kernel = np.ones((ks, ks), dtype=np.uint8)
    return cv2.morphologyEx(binary_u8, cv2.MORPH_CLOSE, kernel, iterations=1).astype(np.uint8)


def _remove_small_components_binary_u8(binary_u8: np.ndarray, min_area: int = 0) -> np.ndarray:
    ma = int(max(0, min_area))
    if ma <= 0 or cv2 is None:
        return binary_u8
    inv = (255 - binary_u8).astype(np.uint8)
    fg = np.where(inv > 0, 1, 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    keep = np.zeros_like(fg, dtype=np.uint8)
    for idx in range(1, int(n_labels)):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area >= ma:
            keep[labels == idx] = 1
    out_inv = (keep * 255).astype(np.uint8)
    return (255 - out_inv).astype(np.uint8)


def resize_to_height(gray_u8: np.ndarray, target_height: int) -> Tuple[np.ndarray, float]:
    h, w = gray_u8.shape[:2]
    th = max(1, int(target_height))
    if h <= 0 or w <= 0:
        return np.zeros((th, 1), dtype=np.uint8), 1.0
    ratio = float(th) / float(h)
    tw = max(1, int(round(float(w) * ratio)))
    out = _resize_gray_u8(gray_u8, tw, th, interpolation="linear")
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
    gray_u8 = _to_gray_u8(image, gray_mode=str(cfg.gray_mode))
    if bool(cfg.normalize_background):
        gray_u8 = _normalize_background_u8(
            gray_u8,
            blur_ksize=int(cfg.background_blur_ksize),
            strength=float(cfg.background_strength),
        )
    if float(cfg.upscale_factor) > 1.0:
        gray_u8 = _upscale_gray_u8(
            gray_u8,
            factor=float(cfg.upscale_factor),
            interpolation=str(cfg.upscale_interpolation),
        )
    if bool(cfg.binarize):
        gray_u8 = _binarize_gray_u8(
            gray_u8,
            threshold_method=str(cfg.threshold_method),
            adaptive_threshold=bool(cfg.adaptive_threshold),
            block_size=int(cfg.threshold_block_size),
            c=int(cfg.threshold_c),
            fixed_threshold=int(cfg.fixed_threshold),
        )
        if bool(cfg.morph_close):
            gray_u8 = _morph_close_binary_u8(gray_u8, kernel_size=int(cfg.morph_close_kernel))
        if bool(cfg.remove_small_components):
            gray_u8 = _remove_small_components_binary_u8(gray_u8, min_area=int(cfg.min_component_area))
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
