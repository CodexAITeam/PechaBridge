#!/usr/bin/env python3
"""Prepare texture-focused square crops for SDXL LoRA training."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tibetan_utils.arg_utils import create_prepare_texture_lora_dataset_parser

LOGGER = logging.getLogger("prepare_texture_lora_dataset")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "opencv-python is required for prepare_texture_lora_dataset.py"
    ) from exc


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _list_images(input_dir: Path) -> List[Path]:
    return sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _upscale_if_needed(image: np.ndarray, crop_size: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = max(crop_size / max(w, 1), crop_size / max(h, 1), 1.0)
    if scale > 1.0:
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return image, scale


def _compute_edge_mask(image_bgr: np.ndarray, canny_low: int, canny_high: int) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return (edges > 0).astype(np.uint8)


def _grid_positions(length: int, window: int, step: int) -> List[int]:
    if length <= window:
        return [0]
    positions = list(range(0, length - window + 1, step))
    last = length - window
    if not positions or positions[-1] != last:
        positions.append(last)
    return positions


def _build_candidates(mask: np.ndarray, crop_size: int) -> List[Tuple[int, int, float]]:
    h, w = mask.shape
    if h < crop_size or w < crop_size:
        return []

    step = max(crop_size // 8, 16)
    xs = _grid_positions(w, crop_size, step)
    ys = _grid_positions(h, crop_size, step)

    integral = np.pad(mask.astype(np.float32), ((1, 0), (1, 0)), mode="constant")
    integral = integral.cumsum(axis=0).cumsum(axis=1)
    area = float(crop_size * crop_size)

    candidates: List[Tuple[int, int, float]] = []
    for y in ys:
        y2 = y + crop_size
        for x in xs:
            x2 = x + crop_size
            edge_sum = (
                integral[y2, x2]
                - integral[y, x2]
                - integral[y2, x]
                + integral[y, x]
            )
            candidates.append((x, y, float(edge_sum / area)))
    return candidates


def _sample_pool(
    pool: Sequence[Tuple[int, int, float]], count: int, rng: np.random.Generator
) -> List[Tuple[int, int, float]]:
    if count <= 0 or not pool:
        return []
    replace = len(pool) < count
    indices = rng.choice(len(pool), size=count, replace=replace)
    return [pool[int(i)] for i in indices]


def _sample_crops(
    candidates: Sequence[Tuple[int, int, float]],
    num_crops: int,
    min_edge_density: float,
    rng: np.random.Generator,
) -> List[Tuple[int, int, float, str]]:
    if not candidates or num_crops <= 0:
        return []

    num_text = max(1, int(round(num_crops * 0.8))) if num_crops > 1 else 1
    num_background = max(0, num_crops - num_text)

    text_pool = [c for c in candidates if c[2] >= min_edge_density]
    background_pool = [c for c in candidates if c[2] < min_edge_density * 0.75]

    if not text_pool:
        cutoff = max(1, len(candidates) // 3)
        text_pool = sorted(candidates, key=lambda c: c[2], reverse=True)[:cutoff]
    if not background_pool:
        cutoff = max(1, len(candidates) // 3)
        background_pool = sorted(candidates, key=lambda c: c[2])[:cutoff]

    selected = []
    selected.extend((*c, "text") for c in _sample_pool(text_pool, num_text, rng))
    selected.extend((*c, "background") for c in _sample_pool(background_pool, num_background, rng))

    if len(selected) < num_crops:
        selected.extend((*c, "text") for c in _sample_pool(candidates, num_crops - len(selected), rng))

    return selected[:num_crops]


def run(args) -> dict:
    _configure_logging()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    images_dir = output_dir / "images"
    metadata_path = output_dir / "metadata.jsonl"

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if args.crop_size <= 0:
        raise ValueError("crop_size must be > 0")
    if args.num_crops_per_page <= 0:
        raise ValueError("num_crops_per_page must be > 0")

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    page_paths = _list_images(input_dir)
    if not page_paths:
        raise RuntimeError(f"No image files found in {input_dir}")

    LOGGER.info("Preparing texture LoRA dataset from %d pages", len(page_paths))

    total_crops = 0
    page_count = 0
    with metadata_path.open("w", encoding="utf-8") as meta_f:
        for page_idx, page_path in enumerate(tqdm(page_paths, desc="pages")):
            page_rel = page_path.relative_to(input_dir)
            image = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
            if image is None:
                LOGGER.warning("Skipping unreadable image: %s", page_path)
                continue

            source_h, source_w = image.shape[:2]
            image, scale_factor = _upscale_if_needed(image, args.crop_size)
            proc_h, proc_w = image.shape[:2]

            edge_mask = _compute_edge_mask(image, args.canny_low, args.canny_high)
            candidates = _build_candidates(edge_mask, args.crop_size)
            if not candidates:
                LOGGER.warning("Skipping page (smaller than crop after processing): %s", page_path)
                continue

            samples = _sample_crops(
                candidates=candidates,
                num_crops=args.num_crops_per_page,
                min_edge_density=args.min_edge_density,
                rng=rng,
            )

            for crop_idx, (x1, y1, edge_density, crop_type) in enumerate(samples):
                x2 = x1 + args.crop_size
                y2 = y1 + args.crop_size
                crop = image[y1:y2, x1:x2]

                crop_name = f"{page_path.stem}_{page_idx:05d}_{crop_idx:03d}.png"
                crop_path = images_dir / crop_name
                if not cv2.imwrite(str(crop_path), crop):
                    LOGGER.warning("Failed writing crop: %s", crop_path)
                    continue

                record = {
                    "filename": str(Path("images") / crop_name),
                    "source_page": str(page_rel),
                    "crop_bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "edge_density": float(edge_density),
                    "crop_type": crop_type,
                    "source_size": [int(source_w), int(source_h)],
                    "processed_size": [int(proc_w), int(proc_h)],
                    "scale_factor": float(scale_factor),
                }
                meta_f.write(json.dumps(record, ensure_ascii=True) + "\n")
                total_crops += 1

            page_count += 1

    LOGGER.info("Saved %d crops from %d pages to %s", total_crops, page_count, output_dir)
    LOGGER.info("Metadata written to %s", metadata_path)
    result = {
        "pages_processed": page_count,
        "crops_written": total_crops,
        "output_dir": str(output_dir),
        "metadata_path": str(metadata_path),
    }
    if str(getattr(args, "lora_augment_path", "")).strip():
        from scripts.texture_augment import run as run_texture_augment

        LOGGER.info("Applying optional LoRA augmentation on prepared crops: %s", images_dir)
        aug_args = argparse.Namespace(
            model_family=args.lora_augment_model_family,
            input_dir=str(images_dir),
            output_dir=str(images_dir),  # in-place augmentation keeps metadata stable
            strength=float(args.lora_augment_strength),
            steps=int(args.lora_augment_steps),
            guidance_scale=float(args.lora_augment_guidance_scale),
            seed=args.lora_augment_seed,
            controlnet_scale=float(args.lora_augment_controlnet_scale),
            lora_path=str(args.lora_augment_path),
            lora_scale=float(args.lora_augment_scale),
            prompt=str(args.lora_augment_prompt),
            base_model_id=str(args.lora_augment_base_model_id),
            controlnet_model_id=str(args.lora_augment_controlnet_model_id),
            canny_low=int(args.lora_augment_canny_low),
            canny_high=int(args.lora_augment_canny_high),
        )
        aug_report = run_texture_augment(aug_args)
        result["lora_augmentation"] = aug_report
    return result


def main(argv: list[str] | None = None) -> int:
    parser = create_prepare_texture_lora_dataset_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
