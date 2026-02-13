#!/usr/bin/env python3
"""Texture augmentation with SDXL/SD2.1 + ControlNet Canny (+ optional LoRA)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tibetan_utils.arg_utils import create_texture_augment_parser

LOGGER = logging.getLogger("texture_augment")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SDXL_DEFAULT = "stabilityai/stable-diffusion-xl-base-1.0"
SD21_DEFAULT = "stabilityai/stable-diffusion-2-1-base"
SDXL_CTRL_DEFAULT = "diffusers/controlnet-canny-sdxl-1.0"
SD21_CTRL_DEFAULT = "thibaud/controlnet-sd21-canny-diffusers"

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("opencv-python is required for texture_augment.py") from exc


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _resolve_model_ids(args) -> Tuple[str, str]:
    base_id = (args.base_model_id or "").strip()
    ctrl_id = (args.controlnet_model_id or "").strip()

    if args.model_family == "sd21":
        if not base_id or base_id == SDXL_DEFAULT:
            base_id = SD21_DEFAULT
        if not ctrl_id or ctrl_id == SDXL_CTRL_DEFAULT:
            ctrl_id = SD21_CTRL_DEFAULT
        return base_id, ctrl_id

    if not base_id:
        base_id = SDXL_DEFAULT
    if not ctrl_id:
        ctrl_id = SDXL_CTRL_DEFAULT
    return base_id, ctrl_id


def _list_images(input_dir: Path) -> List[Path]:
    return sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _device_and_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def _compute_canny_condition(image: Image.Image, canny_low: int, canny_high: int) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    edge_rgb = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(edge_rgb)


def _load_pipeline(args, base_model_id: str, controlnet_model_id: str, device: torch.device, dtype: torch.dtype):
    LOGGER.info("Loading ControlNet model: %s", controlnet_model_id)
    controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=dtype, use_safetensors=True)

    LOGGER.info("Loading base model (%s): %s", args.model_family, base_model_id)
    if args.model_family == "sd21":
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            base_model_id,
            controlnet=controlnet,
            torch_dtype=dtype,
            use_safetensors=True,
        )
    else:
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            base_model_id,
            controlnet=controlnet,
            torch_dtype=dtype,
            use_safetensors=True,
        )

    if args.lora_path:
        lora_path = Path(args.lora_path).expanduser().resolve()
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA path not found: {lora_path}")

        if lora_path.is_file():
            LOGGER.info("Loading LoRA file: %s", lora_path)
            pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)
        else:
            LOGGER.info("Loading LoRA directory: %s", lora_path)
            pipe.load_lora_weights(str(lora_path))

    pipe.to(device)

    if device.type == "cuda":
        pipe.enable_vae_tiling()

    return pipe


def run(args) -> dict:
    _configure_logging()

    if args.model_family not in {"sdxl", "sd21"}:
        raise ValueError("model_family must be one of: sdxl, sd21")

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if int(args.steps) <= 0:
        raise ValueError("steps must be > 0")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _list_images(input_dir)
    if not image_paths:
        raise RuntimeError(f"No input images found in {input_dir}")

    strength = min(max(float(args.strength), 0.0), 0.25)
    if strength != args.strength:
        LOGGER.warning("Clamped strength from %s to %s for geometry preservation", args.strength, strength)

    base_model_id, controlnet_model_id = _resolve_model_ids(args)
    device, dtype = _device_and_dtype()
    LOGGER.info("Using device=%s dtype=%s family=%s", device, dtype, args.model_family)

    pipe = _load_pipeline(
        args=args,
        base_model_id=base_model_id,
        controlnet_model_id=controlnet_model_id,
        device=device,
        dtype=dtype,
    )

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    generated = 0
    cross_attention_kwargs = {"scale": args.lora_scale} if args.lora_path else None

    for idx, input_path in enumerate(tqdm(image_paths, desc="augment")):
        rel_path = input_path.relative_to(input_dir)
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(input_path) as image:
            image = image.convert("RGB")
            control_image = _compute_canny_condition(image=image, canny_low=args.canny_low, canny_high=args.canny_high)

            generator = None
            if args.seed is not None:
                generator_device = device.type if device.type in {"cpu", "cuda"} else "cpu"
                generator = torch.Generator(device=generator_device)
                generator.manual_seed(int(args.seed) + idx)

            call_kwargs = {
                "prompt": args.prompt,
                "image": image,
                "control_image": control_image,
                "strength": strength,
                "num_inference_steps": int(args.steps),
                "guidance_scale": float(args.guidance_scale),
                "controlnet_conditioning_scale": float(args.controlnet_scale),
                "generator": generator,
            }
            if cross_attention_kwargs:
                call_kwargs["cross_attention_kwargs"] = cross_attention_kwargs

            if device.type == "cuda":
                with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                    result = pipe(**call_kwargs).images[0]
            else:
                with torch.inference_mode():
                    result = pipe(**call_kwargs).images[0]

            result.save(out_path)
            generated += 1

    LOGGER.info("Saved %d augmented images to %s", generated, output_dir)
    return {
        "images_processed": generated,
        "output_dir": str(output_dir),
        "model_family": args.model_family,
        "base_model_id": base_model_id,
        "controlnet_model_id": controlnet_model_id,
    }


def main(argv: list[str] | None = None) -> int:
    parser = create_texture_augment_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
