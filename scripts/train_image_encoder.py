#!/usr/bin/env python3
"""Train a self-supervised image encoder (SimCLR-style) for Tibetan page retrieval."""

from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel, get_scheduler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tibetan_utils.arg_utils import create_train_image_encoder_parser

LOGGER = logging.getLogger("train_image_encoder")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _configure_logging(is_main_process: bool) -> None:
    level = logging.INFO if is_main_process else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _discover_images(input_dir: Path) -> List[Path]:
    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


class TwoViewTransform:
    def __init__(self, resolution: int, mean: List[float], std: List[float]):
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=(resolution, resolution),
                    scale=(0.70, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.20, hue=0.02)],
                    p=0.7,
                ),
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomHorizontalFlip(p=0.1),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.8))], p=0.25),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform(image), self.transform(image)


class ImagePairDataset(Dataset):
    def __init__(self, image_paths: List[Path], transform: TwoViewTransform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            view_one, view_two = self.transform(rgb)
        return {"view_one": view_one, "view_two": view_two, "path": str(path)}


def _collate_pairs(batch):
    return {
        "view_one": torch.stack([row["view_one"] for row in batch], dim=0),
        "view_two": torch.stack([row["view_two"] for row in batch], dim=0),
        "path": [row["path"] for row in batch],
    }


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        hidden_dim = max(out_dim * 2, in_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    reps = torch.cat([z1, z2], dim=0)  # [2B, D]
    logits = torch.matmul(reps, reps.T).float() / max(temperature, 1e-6)
    diag_mask = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(diag_mask, -1e9)

    batch_size = z1.shape[0]
    labels = torch.arange(2 * batch_size, device=logits.device)
    labels = (labels + batch_size) % (2 * batch_size)
    return F.cross_entropy(logits, labels)


def _pooled_image_embedding(backbone: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    outputs = backbone(pixel_values=pixel_values, return_dict=True)
    pooler = getattr(outputs, "pooler_output", None)
    if pooler is not None:
        return pooler

    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        raise RuntimeError("Backbone output does not contain last_hidden_state or pooler_output")
    if hidden.ndim == 3:
        return hidden[:, 0]
    return hidden


@dataclass
class TrainingArtifacts:
    backbone_dir: Path
    projection_head_path: Path
    config_path: Path


def _save_artifacts(
    accelerator: Accelerator,
    output_dir: Path,
    backbone: nn.Module,
    projection_head: ProjectionHead,
    args,
    *,
    prefix: str = "",
    global_step: int | None = None,
) -> TrainingArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{prefix}_" if prefix else ""
    backbone_dir = output_dir / f"{suffix}image_encoder_backbone"
    projection_head_path = output_dir / f"{suffix}image_projection_head.pt"
    config_path = output_dir / f"{suffix}training_config.json"

    unwrapped_backbone = accelerator.unwrap_model(backbone)
    unwrapped_head = accelerator.unwrap_model(projection_head)
    unwrapped_backbone.save_pretrained(str(backbone_dir))

    torch.save(
        {
            "state_dict": unwrapped_head.state_dict(),
            "input_dim": int(unwrapped_head.net[0].in_features),
            "output_dim": int(unwrapped_head.net[-1].out_features),
            "model_name_or_path": args.model_name_or_path,
            "global_step": int(global_step or 0),
        },
        projection_head_path,
    )

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "task": "image_encoder_self_supervised",
                "model_name_or_path": args.model_name_or_path,
                "input_dir": str(Path(args.input_dir).expanduser().resolve()),
                "resolution": int(args.resolution),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "num_train_epochs": int(args.num_train_epochs),
                "max_train_steps": int(args.max_train_steps),
                "warmup_steps": int(args.warmup_steps),
                "projection_dim": int(args.projection_dim),
                "temperature": float(args.temperature),
                "mixed_precision": args.mixed_precision,
                "gradient_checkpointing": bool(args.gradient_checkpointing),
                "freeze_backbone": bool(args.freeze_backbone),
                "seed": int(args.seed),
                "global_step": int(global_step or 0),
                "backbone_dir": str(backbone_dir),
                "projection_head_path": str(projection_head_path),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return TrainingArtifacts(
        backbone_dir=backbone_dir,
        projection_head_path=projection_head_path,
        config_path=config_path,
    )


def run(args) -> dict:
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    _configure_logging(accelerator.is_main_process)
    set_seed(int(args.seed))

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_paths = _discover_images(input_dir)
    if not image_paths:
        raise RuntimeError(f"No training images found in {input_dir}")

    if accelerator.is_main_process:
        LOGGER.info("Found %d images for training", len(image_paths))

    image_processor = AutoImageProcessor.from_pretrained(args.model_name_or_path)
    image_mean = list(getattr(image_processor, "image_mean", [0.5, 0.5, 0.5]))
    image_std = list(getattr(image_processor, "image_std", [0.5, 0.5, 0.5]))
    if len(image_mean) == 1:
        image_mean = image_mean * 3
    if len(image_std) == 1:
        image_std = image_std * 3

    pair_transform = TwoViewTransform(
        resolution=int(args.resolution),
        mean=image_mean[:3],
        std=image_std[:3],
    )
    dataset = ImagePairDataset(image_paths=image_paths, transform=pair_transform)
    dataloader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate_pairs,
    )
    if len(dataloader) == 0:
        raise RuntimeError("DataLoader has zero batches. Reduce batch_size or add more images.")

    backbone = AutoModel.from_pretrained(args.model_name_or_path)
    if args.gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()

    hidden_size = getattr(backbone.config, "hidden_size", None)
    if hidden_size is None:
        raise RuntimeError("Backbone config has no hidden_size; choose a compatible vision model.")
    projection_head = ProjectionHead(in_dim=int(hidden_size), out_dim=int(args.projection_dim))

    if args.freeze_backbone:
        backbone.requires_grad_(False)
        backbone.eval()
        trainable_params = list(projection_head.parameters())
    else:
        trainable_params = list(backbone.parameters()) + list(projection_head.parameters())

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    num_update_steps_per_epoch = len(dataloader)
    max_train_steps = int(args.max_train_steps)
    if max_train_steps <= 0:
        max_train_steps = int(args.num_train_epochs) * num_update_steps_per_epoch
    num_train_epochs = int(math.ceil(max_train_steps / num_update_steps_per_epoch))

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_steps),
        num_training_steps=max_train_steps,
    )

    backbone, projection_head, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        backbone, projection_head, optimizer, dataloader, lr_scheduler
    )

    progress_bar = tqdm(
        total=max_train_steps,
        disable=not accelerator.is_local_main_process,
        desc="train-image-encoder",
    )

    global_step = 0
    cumulative_loss = 0.0
    for epoch in range(num_train_epochs):
        if not args.freeze_backbone:
            backbone.train()
        projection_head.train()

        for batch in dataloader:
            if global_step >= max_train_steps:
                break

            view_one = batch["view_one"].to(accelerator.device, non_blocking=True)
            view_two = batch["view_two"].to(accelerator.device, non_blocking=True)

            if args.freeze_backbone:
                with torch.no_grad():
                    emb_one = _pooled_image_embedding(backbone, view_one)
                    emb_two = _pooled_image_embedding(backbone, view_two)
            else:
                emb_one = _pooled_image_embedding(backbone, view_one)
                emb_two = _pooled_image_embedding(backbone, view_two)

            proj_one = projection_head(emb_one)
            proj_two = projection_head(emb_two)
            loss = _nt_xent_loss(proj_one, proj_two, float(args.temperature))

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            cumulative_loss += float(loss.detach().item())
            progress_bar.update(1)
            progress_bar.set_postfix(
                loss=f"{loss.detach().item():.4f}",
                lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                epoch=epoch + 1,
            )

            if (
                int(args.checkpoint_every_steps) > 0
                and global_step % int(args.checkpoint_every_steps) == 0
                and accelerator.is_main_process
            ):
                ckpt_prefix = args.checkpoint_name.strip() or "checkpoint"
                if args.checkpoint_overwrite:
                    prefix = ckpt_prefix
                else:
                    prefix = f"{ckpt_prefix}_step_{global_step:07d}"
                artifacts = _save_artifacts(
                    accelerator=accelerator,
                    output_dir=output_dir,
                    backbone=backbone,
                    projection_head=projection_head,
                    args=args,
                    prefix=prefix,
                    global_step=global_step,
                )
                LOGGER.info(
                    "Saved checkpoint at step %d -> %s",
                    global_step,
                    artifacts.backbone_dir,
                )

        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()
    final_artifacts = None
    if accelerator.is_main_process:
        final_artifacts = _save_artifacts(
            accelerator=accelerator,
            output_dir=output_dir,
            backbone=backbone,
            projection_head=projection_head,
            args=args,
            prefix="",
            global_step=global_step,
        )
        avg_loss = cumulative_loss / max(1, global_step)
        LOGGER.info("Training finished after %d steps. avg_loss=%.6f", global_step, avg_loss)
        LOGGER.info("Saved backbone: %s", final_artifacts.backbone_dir)
        LOGGER.info("Saved projection head: %s", final_artifacts.projection_head_path)
        LOGGER.info("Saved config: %s", final_artifacts.config_path)

    return {
        "global_step": global_step,
        "output_dir": str(output_dir),
        "backbone_dir": str(final_artifacts.backbone_dir) if final_artifacts else "",
        "projection_head_path": str(final_artifacts.projection_head_path) if final_artifacts else "",
        "config_path": str(final_artifacts.config_path) if final_artifacts else "",
    }


def main() -> None:
    parser = create_train_image_encoder_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
