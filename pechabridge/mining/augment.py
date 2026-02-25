"""Deterministic test-time augmentations for MNN stability checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class JitterConfig:
    """Light geometric/photometric jitter configuration."""

    translate_px: int = 3
    scale_range: Tuple[float, float] = (0.98, 1.02)
    brightness: float = 0.05
    contrast: float = 0.05
    blur_sigma: float = 0.3


def _stable_seed(base_seed: int, patch_id: int, trial_idx: int) -> int:
    """Stable seed mixer for deterministic per-patch/per-trial randomness."""
    mask = (1 << 64) - 1
    a = int(base_seed) & 0xFFFFFFFF
    b = int(patch_id) & 0xFFFFFFFF
    c = int(trial_idx) & 0xFFFFFFFF
    x = (a ^ (b * 0x9E3779B97F4A7C15) ^ (c * 0xD1B54A32D192ED03)) & mask
    x ^= (x >> 30)
    x = (x * 0xBF58476D1CE4E5B9) & mask
    x ^= (x >> 27)
    x = (x * 0x94D049BB133111EB) & mask
    x ^= (x >> 31)
    return int(x & 0xFFFFFFFF)


def _rescale_on_canvas(image: Image.Image, scale: float, tx: int, ty: int) -> Image.Image:
    """Scale around center and paste on white canvas with translation."""
    rgb = image.convert("RGB")
    w, h = rgb.size
    if w <= 0 or h <= 0:
        return rgb

    s = float(max(0.5, min(2.0, scale)))
    nw = max(1, int(round(float(w) * s)))
    nh = max(1, int(round(float(h) * s)))
    resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
    scaled = rgb.resize((nw, nh), resample=resample)

    # Center alignment + integer translation jitter.
    ox = int(round((w - nw) / 2.0)) + int(tx)
    oy = int(round((h - nh) / 2.0)) + int(ty)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    canvas.paste(scaled, (ox, oy))
    return canvas


def apply_test_time_augmentation(
    image: Image.Image,
    *,
    patch_id: int,
    trial_idx: int,
    base_seed: int,
    jitter: JitterConfig,
) -> Image.Image:
    """Apply one deterministic jittered view used for retrieval stability checks."""
    rng = np.random.default_rng(_stable_seed(base_seed=base_seed, patch_id=patch_id, trial_idx=trial_idx))
    out = image.convert("RGB")

    tx_lim = max(0, int(jitter.translate_px))
    tx = int(rng.integers(-tx_lim, tx_lim + 1)) if tx_lim > 0 else 0
    ty = int(rng.integers(-tx_lim, tx_lim + 1)) if tx_lim > 0 else 0

    s0 = float(jitter.scale_range[0]) if jitter.scale_range else 1.0
    s1 = float(jitter.scale_range[1]) if jitter.scale_range else 1.0
    lo = min(s0, s1)
    hi = max(s0, s1)
    scale = float(rng.uniform(lo, hi)) if hi > lo else float(lo)
    out = _rescale_on_canvas(out, scale=scale, tx=tx, ty=ty)

    br = float(max(0.0, jitter.brightness))
    if br > 0.0:
        bfac = 1.0 + float(rng.uniform(-br, br))
        out = ImageEnhance.Brightness(out).enhance(bfac)

    ct = float(max(0.0, jitter.contrast))
    if ct > 0.0:
        cfac = 1.0 + float(rng.uniform(-ct, ct))
        out = ImageEnhance.Contrast(out).enhance(cfac)

    bs = float(max(0.0, jitter.blur_sigma))
    if bs > 0.0:
        radius = float(rng.uniform(0.0, bs))
        if radius > 1e-6:
            out = out.filter(ImageFilter.GaussianBlur(radius=radius))

    return out


def generate_augmented_views(
    image: Image.Image,
    *,
    patch_id: int,
    base_seed: int,
    n_trials: int,
    jitter: JitterConfig,
) -> List[Image.Image]:
    """Generate deterministic augmented views for trials [0..n_trials-1]."""
    out: List[Image.Image] = []
    n = max(0, int(n_trials))
    for t in range(n):
        out.append(
            apply_test_time_augmentation(
                image=image,
                patch_id=int(patch_id),
                trial_idx=int(t),
                base_seed=int(base_seed),
                jitter=jitter,
            )
        )
    return out


__all__ = [
    "JitterConfig",
    "apply_test_time_augmentation",
    "generate_augmented_views",
]
