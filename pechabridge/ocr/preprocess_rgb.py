"""Deterministic RGB line-scan preprocessing for Donut OCR training.

This pipeline is designed for line crops with potentially mixed ink colors
(e.g. black + red) where hard binarization can remove useful color cues.
The defaults are intentionally mild and RGB-preserving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

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
class RGBLinePreprocessConfig:
    """Configuration for deterministic RGB line preprocessing."""

    preserve_color: bool = True
    normalize_background: bool = True
    background_method: str = "shade_correct"  # none|shade_correct|rolling_ball_like|top_hat
    background_blur_ksize: int = 0
    background_strength: float = 0.35
    contrast: float = 1.0  # 1.0 means no contrast change
    denoise: bool = False
    morph_close: bool = False
    morph_close_kernel: int = 3
    remove_small_components: bool = False
    min_component_area: int = 12
    upscale_factor: float = 1.0
    upscale_interpolation: str = "lanczos"  # nearest|linear|cubic|lanczos
    ink_normalization: bool = True
    ink_strength: float = 0.2
    to_grayscale_prob: float = 0.0  # reserved for augment pipelines; ignored here

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "RGBLinePreprocessConfig":
        p = dict(payload or {})
        close_kernel = max(0, int(p.get("morph_close_kernel", 3)))
        if close_kernel > 0 and close_kernel % 2 == 0:
            close_kernel += 1
        blur_ksize = max(0, int(p.get("background_blur_ksize", 0)))
        if blur_ksize > 0 and blur_ksize % 2 == 0:
            blur_ksize += 1
        return RGBLinePreprocessConfig(
            preserve_color=_as_bool(p.get("preserve_color"), True),
            normalize_background=_as_bool(p.get("normalize_background"), True),
            background_method=_as_choice(
                p.get("background_method"),
                allowed={"none", "shade_correct", "rolling_ball_like", "top_hat"},
                default="shade_correct",
            ),
            background_blur_ksize=blur_ksize,
            background_strength=max(0.0, min(1.0, _as_float(p.get("background_strength"), 0.35))),
            contrast=max(0.5, min(2.5, _as_float(p.get("contrast"), 1.0))),
            denoise=_as_bool(p.get("denoise"), False),
            morph_close=_as_bool(p.get("morph_close"), False),
            morph_close_kernel=close_kernel,
            remove_small_components=_as_bool(p.get("remove_small_components"), False),
            min_component_area=max(0, int(p.get("min_component_area", 12))),
            upscale_factor=max(1.0, _as_float(p.get("upscale_factor"), 1.0)),
            upscale_interpolation=_as_choice(
                p.get("upscale_interpolation"),
                allowed={"nearest", "linear", "cubic", "lanczos"},
                default="lanczos",
            ),
            ink_normalization=_as_bool(p.get("ink_normalization"), True),
            ink_strength=max(0.0, min(1.0, _as_float(p.get("ink_strength"), 0.2))),
            # Deterministic preprocessing: this remains disabled here.
            to_grayscale_prob=0.0,
        )

    @staticmethod
    def vit_defaults() -> "RGBLinePreprocessConfig":
        """Defaults for RGB line scans before HF image processor normalization."""
        return RGBLinePreprocessConfig(
            preserve_color=True,
            normalize_background=True,
            background_method="shade_correct",
            background_blur_ksize=0,
            background_strength=0.35,
            contrast=1.0,
            denoise=False,
            morph_close=False,
            morph_close_kernel=3,
            remove_small_components=False,
            min_component_area=12,
            upscale_factor=1.0,
            upscale_interpolation="lanczos",
            ink_normalization=True,
            ink_strength=0.2,
            to_grayscale_prob=0.0,
        )

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "preserve_color": bool(self.preserve_color),
            "normalize_background": bool(self.normalize_background),
            "background_method": str(self.background_method),
            "background_blur_ksize": int(self.background_blur_ksize),
            "background_strength": float(self.background_strength),
            "contrast": float(self.contrast),
            "denoise": bool(self.denoise),
            "morph_close": bool(self.morph_close),
            "morph_close_kernel": int(self.morph_close_kernel),
            "remove_small_components": bool(self.remove_small_components),
            "min_component_area": int(self.min_component_area),
            "upscale_factor": float(self.upscale_factor),
            "upscale_interpolation": str(self.upscale_interpolation),
            "ink_normalization": bool(self.ink_normalization),
            "ink_strength": float(self.ink_strength),
            "to_grayscale_prob": 0.0,
        }


def _to_rgb_u8(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    elif arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] >= 3:
            arr = arr[:, :, :3]
        else:
            raise ValueError("Unsupported image shape for ndarray input.")
    else:
        raise ValueError("Expected PIL image or ndarray with shape [H,W] or [H,W,C].")

    if arr.dtype == np.uint8:
        return arr.copy()

    if np.issubdtype(arr.dtype, np.floating):
        arr_f = arr.astype(np.float32)
        max_val = float(np.nanmax(arr_f)) if arr_f.size > 0 else 0.0
        if max_val <= 1.0:
            arr_f = arr_f * 255.0
        return np.clip(arr_f, 0.0, 255.0).astype(np.uint8)
    return np.clip(arr.astype(np.float32), 0.0, 255.0).astype(np.uint8)


def _resize_rgb_u8(rgb_u8: np.ndarray, target_w: int, target_h: int, interpolation: str) -> np.ndarray:
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
        return cv2.resize(rgb_u8, (tw, th), interpolation=cv_interp).astype(np.uint8)

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
    return np.asarray(Image.fromarray(rgb_u8, mode="RGB").resize((tw, th), resample=pil_interp), dtype=np.uint8)


def _upscale_rgb_u8(rgb_u8: np.ndarray, factor: float, interpolation: str = "lanczos") -> np.ndarray:
    fac = max(1.0, float(factor))
    if fac <= 1.0 + 1e-6:
        return rgb_u8
    h, w = rgb_u8.shape[:2]
    tw = max(1, int(round(float(w) * fac)))
    th = max(1, int(round(float(h) * fac)))
    if tw == w and th == h:
        return rgb_u8
    return _resize_rgb_u8(rgb_u8, tw, th, interpolation=interpolation)


def _luma_u8(rgb_u8: np.ndarray) -> np.ndarray:
    r = rgb_u8[:, :, 0].astype(np.float32)
    g = rgb_u8[:, :, 1].astype(np.float32)
    b = rgb_u8[:, :, 2].astype(np.float32)
    luma = (0.299 * r) + (0.587 * g) + (0.114 * b)
    return np.clip(luma, 0.0, 255.0).astype(np.uint8)


def _auto_odd_kernel(h: int, w: int, *, ratio: float, minimum: int) -> int:
    k = max(int(minimum), int(round(float(min(h, w)) * float(ratio))))
    if k % 2 == 0:
        k += 1
    return max(3, k)


def _shade_correct_rgb_u8(rgb_u8: np.ndarray, *, blur_ksize: int = 0, strength: float = 0.35) -> np.ndarray:
    alpha = max(0.0, min(1.0, float(strength)))
    if alpha <= 1e-6:
        return rgb_u8
    h, w = rgb_u8.shape[:2]
    if h <= 0 or w <= 0:
        return rgb_u8

    k = int(max(0, blur_ksize))
    if k <= 0:
        k = _auto_odd_kernel(h, w, ratio=0.08, minimum=15)
    if k % 2 == 0:
        k += 1
    if k < 3:
        return rgb_u8

    if cv2 is not None:
        bg = cv2.GaussianBlur(rgb_u8, (k, k), sigmaX=0).astype(np.float32)
    else:
        radius = max(1.0, float(k) / 6.0)
        bg = np.asarray(Image.fromarray(rgb_u8, mode="RGB").filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32)

    src_f = rgb_u8.astype(np.float32)
    bg = np.maximum(bg, 1.0)
    corrected = (src_f * 255.0) / bg
    out = (src_f * (1.0 - alpha)) + (corrected * alpha)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _denoise_rgb_u8(rgb_u8: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        return cv2.fastNlMeansDenoisingColored(rgb_u8, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21)
    return np.asarray(Image.fromarray(rgb_u8, mode="RGB").filter(ImageFilter.MedianFilter(size=3)), dtype=np.uint8)


def _ink_normalize_rgb_u8(rgb_u8: np.ndarray, *, strength: float = 0.2) -> np.ndarray:
    alpha = max(0.0, min(1.0, float(strength)))
    if alpha <= 1e-6:
        return rgb_u8
    gray = _luma_u8(rgb_u8).astype(np.float32)
    h, w = gray.shape[:2]
    if h <= 0 or w <= 0:
        return rgb_u8
    k = _auto_odd_kernel(h, w, ratio=0.06, minimum=11)
    if cv2 is not None:
        bg = cv2.GaussianBlur(gray, (k, k), sigmaX=0).astype(np.float32)
    else:
        radius = max(1.0, float(k) / 6.0)
        bg = np.asarray(
            Image.fromarray(gray.astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(radius=radius)),
            dtype=np.float32,
        )

    bg = np.maximum(bg, 1.0)
    ink = np.clip((bg - gray) / bg, 0.0, 1.0)
    scale = 1.0 - (alpha * ink)
    out = rgb_u8.astype(np.float32) * scale[:, :, None]
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _apply_contrast_rgb_u8(rgb_u8: np.ndarray, *, contrast: float = 1.0) -> np.ndarray:
    fac = max(0.5, min(2.5, float(contrast)))
    if abs(fac - 1.0) <= 1e-6:
        return rgb_u8
    if cv2 is not None:
        lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
        l = lab[:, :, 0].astype(np.float32)
        l = ((l - 128.0) * fac) + 128.0
        lab[:, :, 0] = np.clip(l, 0.0, 255.0).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    pil = Image.fromarray(rgb_u8, mode="RGB")
    return np.asarray(ImageEnhance.Contrast(pil).enhance(fac), dtype=np.uint8)


def _otsu_foreground_mask_u8(gray_u8: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        _thr, bw_inv = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return bw_inv.astype(np.uint8)

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
    return np.where(gray_u8 <= best_thr, 255, 0).astype(np.uint8)


def _morph_close_mask_u8(mask_u8: np.ndarray, kernel_size: int = 0) -> np.ndarray:
    ks = int(max(0, kernel_size))
    if ks <= 1 or cv2 is None:
        return mask_u8
    if ks % 2 == 0:
        ks += 1
    kernel = np.ones((ks, ks), dtype=np.uint8)
    return cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1).astype(np.uint8)


def _remove_small_components_mask_u8(mask_u8: np.ndarray, min_area: int = 0) -> np.ndarray:
    ma = int(max(0, min_area))
    if ma <= 0 or cv2 is None:
        return mask_u8
    fg = np.where(mask_u8 > 0, 1, 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    keep = np.zeros_like(fg, dtype=np.uint8)
    for idx in range(1, int(n_labels)):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area >= ma:
            keep[labels == idx] = 1
    return (keep * 255).astype(np.uint8)


def _apply_foreground_mask(rgb_u8: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    fg = mask_u8 > 0
    if bool(np.all(fg)):
        return rgb_u8
    if cv2 is not None:
        bg = cv2.GaussianBlur(rgb_u8, (3, 3), sigmaX=0).astype(np.uint8)
    else:
        bg = np.asarray(Image.fromarray(rgb_u8, mode="RGB").filter(ImageFilter.GaussianBlur(radius=1.0)), dtype=np.uint8)
    out = bg
    out[fg] = rgb_u8[fg]
    return out


def preprocess_image_rgb_lines(
    image: Image.Image | np.ndarray,
    config: RGBLinePreprocessConfig | None = None,
) -> Image.Image:
    """Deterministic RGB preprocessing for line-scan OCR crops.

    Returns a PIL RGB image suitable for downstream HF `AutoImageProcessor`.
    """
    cfg = config or RGBLinePreprocessConfig.vit_defaults()
    rgb_u8 = _to_rgb_u8(image)

    if float(cfg.upscale_factor) > 1.0:
        rgb_u8 = _upscale_rgb_u8(
            rgb_u8,
            factor=float(cfg.upscale_factor),
            interpolation=str(cfg.upscale_interpolation),
        )

    bg_method = str(cfg.background_method or "shade_correct").strip().lower()
    if bool(cfg.normalize_background) and bg_method != "none":
        # Current implementation supports shade-correction for all non-none modes.
        rgb_u8 = _shade_correct_rgb_u8(
            rgb_u8,
            blur_ksize=int(cfg.background_blur_ksize),
            strength=float(cfg.background_strength),
        )

    if bool(cfg.denoise):
        rgb_u8 = _denoise_rgb_u8(rgb_u8)

    if bool(cfg.ink_normalization):
        rgb_u8 = _ink_normalize_rgb_u8(rgb_u8, strength=float(cfg.ink_strength))

    rgb_u8 = _apply_contrast_rgb_u8(rgb_u8, contrast=float(cfg.contrast))

    if bool(cfg.morph_close) or bool(cfg.remove_small_components):
        fg_mask = _otsu_foreground_mask_u8(_luma_u8(rgb_u8))
        if bool(cfg.morph_close):
            fg_mask = _morph_close_mask_u8(fg_mask, kernel_size=int(cfg.morph_close_kernel))
        if bool(cfg.remove_small_components):
            fg_mask = _remove_small_components_mask_u8(fg_mask, min_area=int(cfg.min_component_area))
        rgb_u8 = _apply_foreground_mask(rgb_u8, fg_mask)

    if not bool(cfg.preserve_color):
        gray = _luma_u8(rgb_u8)
        rgb_u8 = np.stack([gray, gray, gray], axis=2)

    # to_grayscale_prob is intentionally ignored for deterministic preprocessing.
    return Image.fromarray(rgb_u8.astype(np.uint8), mode="RGB")


__all__ = [
    "RGBLinePreprocessConfig",
    "preprocess_image_rgb_lines",
]
