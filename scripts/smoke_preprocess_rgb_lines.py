#!/usr/bin/env python3
"""Minimal smoke/debug script for deterministic RGB line preprocessing."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
import sys

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pechabridge.ocr.preprocess_rgb import RGBLinePreprocessConfig, preprocess_image_rgb_lines


def _to_png_bytes(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _synthetic_rgb_line() -> Image.Image:
    """Build a deterministic synthetic line with uneven paper + black/red strokes."""
    h, w = 96, 640
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    bg = 235.0 - (25.0 * y) + (12.0 * x)
    rgb = np.stack([bg + 7.0, bg + 2.0, bg - 8.0], axis=2)
    rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)

    # Black ink band
    rgb[28:35, 80:560, :] = np.array([20, 20, 20], dtype=np.uint8)
    # Red ink band
    rgb[58:66, 120:560, :] = np.array([185, 35, 30], dtype=np.uint8)
    # A few tiny noise dots in the paper background.
    rgb[10, 50, :] = np.array([110, 90, 70], dtype=np.uint8)
    rgb[18, 240, :] = np.array([100, 75, 65], dtype=np.uint8)
    rgb[84, 500, :] = np.array([95, 70, 60], dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test RGB line preprocessing determinism")
    p.add_argument("--input", type=str, default="", help="Optional input image path")
    p.add_argument("--output_dir", type=str, default="", help="Optional output directory for debug images")
    p.add_argument("--preserve_color", action="store_true", help="Force preserve_color=True")
    p.add_argument("--no_preserve_color", dest="preserve_color", action="store_false", help="Force preserve_color=False")
    p.set_defaults(preserve_color=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if str(args.input).strip():
        with Image.open(str(args.input)) as img:
            src = img.convert("RGB")
    else:
        src = _synthetic_rgb_line()

    cfg = RGBLinePreprocessConfig.vit_defaults()
    if args.preserve_color is not None:
        cfg = RGBLinePreprocessConfig.from_dict({**cfg.to_dict(), "preserve_color": bool(args.preserve_color)})

    out1 = preprocess_image_rgb_lines(src, config=cfg)
    out2 = preprocess_image_rgb_lines(src, config=cfg)

    if _to_png_bytes(out1) != _to_png_bytes(out2):
        raise RuntimeError("Non-deterministic output detected for RGB preprocessing.")
    if out1.mode != "RGB":
        raise RuntimeError(f"Unexpected output mode: {out1.mode}")

    print("RGB preprocess smoke test passed.")
    print(f"- mode: {out1.mode}")
    print(f"- size: {out1.size[0]}x{out1.size[1]}")
    print(f"- config: {cfg.to_dict()}")

    out_dir_raw = str(args.output_dir or "").strip()
    if out_dir_raw:
        out_dir = Path(out_dir_raw).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        src_path = out_dir / "rgb_preprocess_input.png"
        out_path = out_dir / "rgb_preprocess_output.png"
        src.save(src_path)
        out1.save(out_path)
        print(f"- wrote: {src_path}")
        print(f"- wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
