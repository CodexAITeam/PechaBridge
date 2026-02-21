"""Ink-map and horizontal profile helpers for patch generation."""

from __future__ import annotations

from typing import List

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    from scipy.ndimage import gaussian_filter1d
except Exception:  # pragma: no cover
    gaussian_filter1d = None

try:
    from scipy.signal import find_peaks
except Exception:  # pragma: no cover
    find_peaks = None


def _require_cv2() -> None:
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("opencv-python is required for ink profile computation.")


def rgb_to_gray_float(image_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image to float grayscale in [0, 1]."""
    _require_cv2()
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Empty image.")
    gray_u8 = cv2.cvtColor(np.asarray(image_rgb).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return gray_u8.astype(np.float32) / 255.0


def compute_ink_map(
    image_rgb: np.ndarray,
    *,
    use_clahe: bool = False,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: int = 8,
    binarize: bool = False,
    binarize_mode: str = "otsu",
    fixed_threshold: float = 0.5,
) -> np.ndarray:
    """Compute ink intensity map J in [0,1], where darker pixels have higher values."""
    _require_cv2()
    gray = rgb_to_gray_float(image_rgb)
    gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)

    if use_clahe:
        grid = max(1, int(clahe_tile_grid_size))
        clahe = cv2.createCLAHE(clipLimit=float(max(0.1, clahe_clip_limit)), tileGridSize=(grid, grid))
        gray_u8 = clahe.apply(gray_u8)

    if binarize:
        mode = (binarize_mode or "otsu").strip().lower()
        if mode == "adaptive":
            bw = cv2.adaptiveThreshold(
                gray_u8,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                8,
            )
        elif mode == "fixed":
            thr = int(max(0, min(255, round(float(fixed_threshold) * 255.0))))
            _, bw = cv2.threshold(gray_u8, thr, 255, cv2.THRESH_BINARY)
        else:
            _, bw = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        gray_u8 = bw

    gray_norm = gray_u8.astype(np.float32) / 255.0
    ink = 1.0 - gray_norm
    return np.clip(ink, 0.0, 1.0).astype(np.float32)


def horizontal_ink_profile(ink_map: np.ndarray) -> np.ndarray:
    """Return horizontal ink profile p(x) = sum_y J(y, x)."""
    if ink_map is None or ink_map.size == 0:
        return np.array([], dtype=np.float32)
    return np.asarray(ink_map, dtype=np.float32).sum(axis=0).astype(np.float32)


def smooth_profile(profile: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth 1D profile with Gaussian (or box fallback)."""
    p = np.asarray(profile, dtype=np.float32)
    if p.size == 0:
        return p
    s = float(max(0.0, sigma))
    if s <= 1e-6:
        return p
    if gaussian_filter1d is not None:
        return gaussian_filter1d(p, sigma=s, mode="nearest").astype(np.float32)
    radius = max(1, int(round(2.0 * s)))
    kernel = np.ones((2 * radius + 1,), dtype=np.float32) / float(2 * radius + 1)
    return np.convolve(p, kernel, mode="same").astype(np.float32)


def _fallback_local_minima(profile: np.ndarray, min_dist_px: int, prominence_abs: float) -> List[int]:
    out: List[int] = []
    p = np.asarray(profile, dtype=np.float32)
    if p.size < 3:
        return out
    d = max(1, int(min_dist_px))
    last = -10**9
    for i in range(1, int(p.size) - 1):
        if not (p[i] <= p[i - 1] and p[i] <= p[i + 1]):
            continue
        local_max = max(float(p[max(0, i - d)]), float(p[min(int(p.size) - 1, i + d)]))
        prom = local_max - float(p[i])
        if prom < float(prominence_abs):
            continue
        if i - last < d:
            continue
        out.append(i)
        last = i
    return out


def detect_profile_minima(
    profile: np.ndarray,
    *,
    min_dist_px: int,
    prominence: float,
) -> np.ndarray:
    """
    Detect minima in a 1D profile.

    If `prominence` is in (0,1], it is interpreted as relative to max(profile).
    Otherwise it is treated as an absolute value in profile units.
    """
    p = np.asarray(profile, dtype=np.float32)
    if p.size == 0:
        return np.array([], dtype=np.int32)

    maxv = float(np.max(p)) if p.size > 0 else 0.0
    if 0.0 < float(prominence) <= 1.0:
        prom_abs = float(prominence) * max(1.0, maxv)
    else:
        prom_abs = float(max(0.0, prominence))

    dist = max(1, int(min_dist_px))
    if find_peaks is not None:
        peaks, _ = find_peaks(-p, distance=dist, prominence=prom_abs)
        if peaks.size > 0:
            return np.sort(peaks.astype(np.int32))
    fb = _fallback_local_minima(p, min_dist_px=dist, prominence_abs=prom_abs)
    return np.asarray(sorted(set(int(v) for v in fb)), dtype=np.int32)

