#!/usr/bin/env python3
"""Train a ViT-based retrieval encoder on TextHierarchy or patch-parquet datasets."""

from __future__ import annotations

import json
import logging
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
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

from tibetan_utils.arg_utils import create_train_text_hierarchy_vit_parser

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

LOGGER = logging.getLogger("train_text_hierarchy_vit")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _configure_logging(is_main_process: bool) -> None:
    level = logging.INFO if is_main_process else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _parse_width_buckets(raw: str, patch_multiple: int, max_width: int) -> List[int]:
    vals: List[int] = []
    for tok in (raw or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = int(tok)
        except Exception:
            continue
        if v <= 0:
            continue
        vals.append(v)
    if not vals:
        vals = [int(max_width)]

    pm = max(1, int(patch_multiple))
    clipped: List[int] = []
    for v in vals:
        vv = max(pm, min(int(max_width), int(v)))
        vv = int(math.ceil(vv / pm) * pm)
        vv = min(vv, int(max_width))
        clipped.append(vv)
    out = sorted(set(clipped))
    if not out:
        out = [int(max_width)]
    return out


def _list_word_crops(line_dir: Path) -> List[Path]:
    return sorted(
        p
        for p in line_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and p.name.startswith("word_")
    )


@dataclass
class HierarchyGroup:
    key: str
    assets: List[Path]


def _sanitize_id(value: Any) -> str:
    txt = str(value or "").strip()
    if not txt:
        return "unknown"
    out: List[str] = []
    for ch in txt:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "unknown"


def _resolve_patch_asset_path(dataset_dir: Path, row: Dict[str, Any]) -> Path:
    raw_patch_path = str(row.get("patch_path", "") or "").strip()
    if raw_patch_path:
        try:
            raw = Path(raw_patch_path).expanduser()
            if raw.is_absolute():
                if raw.exists():
                    return raw.resolve()
            else:
                candidate = (dataset_dir / raw).resolve()
                if candidate.exists():
                    return candidate
        except Exception:
            pass

    doc_id = _sanitize_id(row.get("doc_id", ""))
    page_id = _sanitize_id(row.get("page_id", ""))
    line_id = int(row.get("line_id", -1))
    scale_w = int(row.get("scale_w", -1))
    patch_id = int(row.get("patch_id", -1))
    return (
        dataset_dir
        / "patches"
        / f"doc={doc_id}"
        / f"page={page_id}"
        / f"line={line_id}"
        / f"scale={scale_w}"
        / f"patch_{patch_id}.png"
    ).resolve()


def _discover_patch_parquet_groups(
    dataset_dir: Path,
    include_line_images: bool,
    include_word_crops: bool,
    min_assets_per_group: int,
) -> List[HierarchyGroup]:
    meta_path = dataset_dir / "meta" / "patches.parquet"
    if not meta_path.exists() or not meta_path.is_file():
        return []
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required to read patch parquet metadata.")

    use_patch_assets = bool(include_word_crops or include_line_images)
    if not use_patch_assets:
        LOGGER.warning(
            "Patch parquet dataset found at %s, but both include flags are disabled.",
            meta_path,
        )
        return []

    cols = [
        "patch_id",
        "doc_id",
        "page_id",
        "line_id",
        "scale_w",
        "patch_path",
    ]
    df = pd.read_parquet(meta_path, columns=cols)
    if df.empty:
        return []

    grouped: Dict[str, List[Path]] = {}
    for row in df.to_dict(orient="records"):
        try:
            line_id = int(row.get("line_id", -1))
            scale_w = int(row.get("scale_w", -1))
        except Exception:
            continue
        if line_id < 0 or scale_w <= 0:
            continue
        path = _resolve_patch_asset_path(dataset_dir, row)
        if not path.exists() or not path.is_file():
            continue
        doc_id = str(row.get("doc_id", "") or "")
        page_id = str(row.get("page_id", "") or "")
        key = (
            f"patches/doc={_sanitize_id(doc_id)}"
            f"/page={_sanitize_id(page_id)}"
            f"/line={line_id}"
            f"/scale={scale_w}"
        )
        grouped.setdefault(key, []).append(path)

    groups: List[HierarchyGroup] = []
    required = max(1, int(min_assets_per_group))
    for key in sorted(grouped.keys()):
        uniq = sorted({str(p.resolve()): p.resolve() for p in grouped[key]}.values(), key=lambda p: str(p))
        if len(uniq) >= required:
            groups.append(HierarchyGroup(key=key, assets=uniq))
    return groups


def _discover_hierarchy_groups(
    dataset_dir: Path,
    include_line_images: bool,
    include_word_crops: bool,
    include_number_crops: bool,
    min_assets_per_group: int,
) -> List[HierarchyGroup]:
    patch_groups = _discover_patch_parquet_groups(
        dataset_dir=dataset_dir,
        include_line_images=include_line_images,
        include_word_crops=include_word_crops,
        min_assets_per_group=min_assets_per_group,
    )
    if patch_groups:
        groups: List[HierarchyGroup] = list(patch_groups)
        if include_number_crops:
            number_root = dataset_dir / "NumberCrops"
            if number_root.exists() and number_root.is_dir():
                for img_path in sorted(
                    p for p in number_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                ):
                    try:
                        key = str(img_path.relative_to(dataset_dir))
                    except Exception:
                        key = str(img_path)
                    groups.append(HierarchyGroup(key=f"number::{key}", assets=[img_path]))
        LOGGER.info(
            "Using patch parquet dataset: groups=%d source=%s",
            len(groups),
            str((dataset_dir / "meta" / "patches.parquet").resolve()),
        )
        return groups

    groups: List[HierarchyGroup] = []
    text_root = dataset_dir / "TextHierarchy"
    if text_root.exists() and text_root.is_dir():
        for line_dir in sorted(text_root.rglob("line_*")):
            if not line_dir.is_dir():
                continue
            assets: List[Path] = []
            if include_line_images:
                p = line_dir / "line.png"
                if p.exists() and p.is_file():
                    assets.append(p)
            if include_word_crops:
                assets.extend(_list_word_crops(line_dir))
            assets = [p for p in assets if p.suffix.lower() in IMAGE_EXTENSIONS]
            if len(assets) >= int(min_assets_per_group):
                try:
                    key = str(line_dir.relative_to(dataset_dir))
                except Exception:
                    key = str(line_dir)
                groups.append(HierarchyGroup(key=key, assets=assets))

    if include_number_crops:
        number_root = dataset_dir / "NumberCrops"
        if number_root.exists() and number_root.is_dir():
            for img_path in sorted(
                p for p in number_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ):
                try:
                    key = str(img_path.relative_to(dataset_dir))
                except Exception:
                    key = str(img_path)
                groups.append(HierarchyGroup(key=f"number::{key}", assets=[img_path]))

    return groups


def _normalize_for_vit(
    image: Image.Image,
    target_height: int,
    width_buckets: Sequence[int],
    patch_multiple: int,
    max_width: int,
) -> Image.Image:
    rgb = image.convert("RGB")
    orig_w, orig_h = rgb.size
    if orig_w <= 0 or orig_h <= 0:
        return Image.new("RGB", (int(max_width), int(target_height)), (255, 255, 255))

    t_h = max(8, int(target_height))
    pm = max(1, int(patch_multiple))
    max_w = max(pm, int(max_width))

    # Keep aspect ratio. Primary target is fixed height.
    scale_h = float(t_h) / float(orig_h)
    scaled_w = max(1, int(round(orig_w * scale_h)))
    scaled_h = int(t_h)

    # If too wide, downscale isotropically to max_w, then pad vertically.
    if scaled_w > max_w:
        shrink = float(max_w) / float(scaled_w)
        scaled_w = int(max_w)
        scaled_h = max(1, int(round(float(t_h) * shrink)))

    resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
    resized = rgb.resize((scaled_w, scaled_h), resample=resample)

    bucket_candidates = sorted(int(v) for v in width_buckets if int(v) > 0)
    if not bucket_candidates:
        bucket_candidates = [max_w]
    target_w = next((b for b in bucket_candidates if b >= scaled_w), bucket_candidates[-1])
    target_w = max(scaled_w, min(max_w, int(target_w)))
    target_w = int(math.ceil(target_w / pm) * pm)
    target_w = min(max_w, max(scaled_w, target_w))

    canvas = Image.new("RGB", (target_w, t_h), (255, 255, 255))
    canvas.paste(resized, (0, 0))
    return canvas


class HierarchyTwoViewTransform:
    def __init__(
        self,
        target_height: int,
        width_buckets: Sequence[int],
        patch_multiple: int,
        max_width: int,
        mean: List[float],
        std: List[float],
    ):
        self.target_height = int(target_height)
        self.width_buckets = list(width_buckets)
        self.patch_multiple = int(patch_multiple)
        self.max_width = int(max_width)
        self.tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=mean, std=std)
        self.augment = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.08, hue=0.01)],
                    p=0.45,
                ),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.4))], p=0.15),
                transforms.RandomAffine(
                    degrees=1.2,
                    translate=(0.02, 0.04),
                    scale=(0.97, 1.03),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    fill=(255, 255, 255),
                ),
            ]
        )

    def _encode(self, image: Image.Image) -> torch.Tensor:
        normalized = _normalize_for_vit(
            image=image,
            target_height=self.target_height,
            width_buckets=self.width_buckets,
            patch_multiple=self.patch_multiple,
            max_width=self.max_width,
        )
        aug = self.augment(normalized)
        ten = self.tensor(aug)
        return self.normalize(ten)

    def __call__(self, image_a: Image.Image, image_b: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._encode(image_a), self._encode(image_b)


class HierarchyPairDataset(Dataset):
    def __init__(self, groups: List[HierarchyGroup], transform: HierarchyTwoViewTransform):
        self.groups = groups
        self.transform = transform

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        group = self.groups[idx]
        if not group.assets:
            raise RuntimeError(f"Group has no assets: {group.key}")

        path_a = random.choice(group.assets)
        if len(group.assets) > 1:
            path_b = random.choice(group.assets)
            for _ in range(4):
                if path_b != path_a:
                    break
                path_b = random.choice(group.assets)
        else:
            path_b = path_a

        with Image.open(path_a) as img_a:
            image_a = img_a.convert("RGB")
        with Image.open(path_b) as img_b:
            image_b = img_b.convert("RGB")

        view_one, view_two = self.transform(image_a, image_b)
        return {
            "view_one": view_one,
            "view_two": view_two,
            "group_key": group.key,
            "path_a": str(path_a),
            "path_b": str(path_b),
        }


def _pad_right_to_width(tensor: torch.Tensor, target_width: int) -> torch.Tensor:
    width = int(tensor.shape[-1])
    if width >= target_width:
        return tensor[:, :, :target_width]
    pad = target_width - width
    return F.pad(tensor, (0, pad, 0, 0), mode="replicate")


def _collate_pairs(batch: List[Dict[str, Any]], patch_multiple: int) -> Dict[str, Any]:
    if not batch:
        raise RuntimeError("Empty batch")
    pm = max(1, int(patch_multiple))
    widths = [int(row["view_one"].shape[-1]) for row in batch] + [int(row["view_two"].shape[-1]) for row in batch]
    max_w = max(widths)
    max_w = int(math.ceil(max_w / pm) * pm)
    view_one = torch.stack([_pad_right_to_width(row["view_one"], max_w) for row in batch], dim=0)
    view_two = torch.stack([_pad_right_to_width(row["view_two"], max_w) for row in batch], dim=0)
    return {
        "view_one": view_one,
        "view_two": view_two,
        "group_key": [row["group_key"] for row in batch],
        "path_a": [row["path_a"] for row in batch],
        "path_b": [row["path_b"] for row in batch],
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
    logits = torch.matmul(reps, reps.T).float() / max(float(temperature), 1e-6)
    diag_mask = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(diag_mask, -1e9)
    batch_size = z1.shape[0]
    labels = torch.arange(2 * batch_size, device=logits.device)
    labels = (labels + batch_size) % (2 * batch_size)
    return F.cross_entropy(logits, labels)


def _pooled_image_embedding(backbone: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    kwargs = {"pixel_values": pixel_values, "return_dict": True}
    try:
        outputs = backbone(interpolate_pos_encoding=True, **kwargs)
    except TypeError:
        outputs = backbone(**kwargs)

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
    image_processor: Any,
    args,
    *,
    num_groups: int,
    num_assets: int,
    width_buckets: Sequence[int],
    steps_per_epoch: int,
    effective_max_train_steps: int,
    effective_num_train_epochs: int,
    prefix: str = "",
    global_step: int = 0,
) -> TrainingArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{prefix}_" if prefix else ""
    backbone_dir = output_dir / f"{suffix}text_hierarchy_vit_backbone"
    projection_head_path = output_dir / f"{suffix}text_hierarchy_projection_head.pt"
    config_path = output_dir / f"{suffix}training_config.json"

    unwrapped_backbone = accelerator.unwrap_model(backbone)
    unwrapped_head = accelerator.unwrap_model(projection_head)
    unwrapped_backbone.save_pretrained(str(backbone_dir))
    if image_processor is not None and hasattr(image_processor, "save_pretrained"):
        try:
            image_processor.save_pretrained(str(backbone_dir))
        except Exception as exc:
            LOGGER.warning("Could not save image processor to %s: %s", backbone_dir, exc)

    if image_processor is not None:
        image_mean = list(getattr(image_processor, "image_mean", [0.5, 0.5, 0.5]))
        image_std = list(getattr(image_processor, "image_std", [0.5, 0.5, 0.5]))
    else:
        image_mean = [0.5, 0.5, 0.5]
        image_std = [0.5, 0.5, 0.5]
    if len(image_mean) == 1:
        image_mean = image_mean * 3
    if len(image_std) == 1:
        image_std = image_std * 3

    torch.save(
        {
            "state_dict": unwrapped_head.state_dict(),
            "input_dim": int(unwrapped_head.net[0].in_features),
            "output_dim": int(unwrapped_head.net[-1].out_features),
            "model_name_or_path": args.model_name_or_path,
            "global_step": int(global_step),
        },
        projection_head_path,
    )

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "task": "text_hierarchy_vit_retrieval",
                "dataset_dir": str(Path(args.dataset_dir).expanduser().resolve()),
                "model_name_or_path": args.model_name_or_path,
                "groups": int(num_groups),
                "assets": int(num_assets),
                "include_line_images": bool(args.include_line_images),
                "include_word_crops": bool(args.include_word_crops),
                "include_number_crops": bool(args.include_number_crops),
                "target_height": int(args.target_height),
                "max_width": int(args.max_width),
                "patch_multiple": int(args.patch_multiple),
                "width_buckets": [int(v) for v in width_buckets],
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "requested_num_train_epochs": int(args.num_train_epochs),
                "requested_max_train_steps": int(args.max_train_steps),
                "effective_num_train_epochs": int(effective_num_train_epochs),
                "effective_max_train_steps": int(effective_max_train_steps),
                "steps_per_epoch": int(steps_per_epoch),
                "warmup_steps": int(args.warmup_steps),
                "projection_dim": int(args.projection_dim),
                "temperature": float(args.temperature),
                "mixed_precision": args.mixed_precision,
                "gradient_checkpointing": bool(args.gradient_checkpointing),
                "freeze_backbone": bool(args.freeze_backbone),
                "image_mean": [float(v) for v in image_mean[:3]],
                "image_std": [float(v) for v in image_std[:3]],
                "global_step": int(global_step),
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


def run(args) -> Dict[str, Any]:
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    _configure_logging(accelerator.is_main_process)
    set_seed(int(args.seed))

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    width_buckets = _parse_width_buckets(
        raw=str(args.width_buckets),
        patch_multiple=int(args.patch_multiple),
        max_width=int(args.max_width),
    )
    groups = _discover_hierarchy_groups(
        dataset_dir=dataset_dir,
        include_line_images=bool(args.include_line_images),
        include_word_crops=bool(args.include_word_crops),
        include_number_crops=bool(args.include_number_crops),
        min_assets_per_group=max(1, int(args.min_assets_per_group)),
    )
    if not groups:
        raise RuntimeError(f"No hierarchy groups found in {dataset_dir}.")

    total_assets = sum(len(g.assets) for g in groups)
    if accelerator.is_main_process:
        LOGGER.info("Found %d groups / %d assets for training", len(groups), total_assets)

    image_processor = None
    image_stats_source = "default(0.5)"
    try:
        image_processor = AutoImageProcessor.from_pretrained(args.model_name_or_path)
        image_stats_source = f"AutoImageProcessor({args.model_name_or_path})"
    except Exception as exc:
        LOGGER.warning(
            "Could not load image processor from %s: %s. Falling back to default mean/std.",
            args.model_name_or_path,
            exc,
        )

    image_mean = list(getattr(image_processor, "image_mean", [0.5, 0.5, 0.5]))
    image_std = list(getattr(image_processor, "image_std", [0.5, 0.5, 0.5]))
    if len(image_mean) == 1:
        image_mean = image_mean * 3
    if len(image_std) == 1:
        image_std = image_std * 3
    if accelerator.is_main_process:
        LOGGER.info("Image normalization stats source: %s (mean=%s std=%s)", image_stats_source, image_mean[:3], image_std[:3])

    transform = HierarchyTwoViewTransform(
        target_height=int(args.target_height),
        width_buckets=width_buckets,
        patch_multiple=int(args.patch_multiple),
        max_width=int(args.max_width),
        mean=image_mean[:3],
        std=image_std[:3],
    )
    dataset = HierarchyPairDataset(groups=groups, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda batch: _collate_pairs(batch, patch_multiple=int(args.patch_multiple)),
    )
    if len(dataloader) == 0:
        raise RuntimeError("DataLoader has zero batches. Reduce batch_size or add more groups.")

    backbone = AutoModel.from_pretrained(args.model_name_or_path)
    if bool(args.gradient_checkpointing) and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()

    hidden_size = getattr(backbone.config, "hidden_size", None)
    if hidden_size is None:
        raise RuntimeError("Backbone config has no hidden_size. Use a ViT-like model with hidden_size.")
    projection_head = ProjectionHead(in_dim=int(hidden_size), out_dim=int(args.projection_dim))

    if bool(args.freeze_backbone):
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
    requested_num_train_epochs = int(args.num_train_epochs)
    requested_max_train_steps = int(args.max_train_steps)
    if max_train_steps <= 0:
        max_train_steps = requested_num_train_epochs * num_update_steps_per_epoch
    num_train_epochs = int(math.ceil(max_train_steps / num_update_steps_per_epoch))

    if accelerator.is_main_process:
        LOGGER.info(
            "Training plan: steps_per_epoch=%d, requested_epochs=%d, requested_max_steps=%d, effective_epochs=%d, effective_max_steps=%d",
            num_update_steps_per_epoch,
            requested_num_train_epochs,
            requested_max_train_steps,
            num_train_epochs,
            max_train_steps,
        )
        if requested_max_train_steps > 0 and requested_max_train_steps <= num_update_steps_per_epoch:
            LOGGER.warning(
                "max_train_steps=%d is <= steps_per_epoch=%d -> this runs at most one epoch.",
                requested_max_train_steps,
                num_update_steps_per_epoch,
            )

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
        desc="train-text-hierarchy-vit",
    )

    global_step = 0
    cumulative_loss = 0.0
    for epoch in range(num_train_epochs):
        if not bool(args.freeze_backbone):
            backbone.train()
        projection_head.train()

        for batch in dataloader:
            if global_step >= max_train_steps:
                break

            view_one = batch["view_one"].to(accelerator.device, non_blocking=True)
            view_two = batch["view_two"].to(accelerator.device, non_blocking=True)

            if bool(args.freeze_backbone):
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
                prefix = f"checkpoint_step_{global_step:07d}"
                artifacts = _save_artifacts(
                    accelerator=accelerator,
                    output_dir=output_dir,
                    backbone=backbone,
                    projection_head=projection_head,
                    image_processor=image_processor,
                    args=args,
                    num_groups=len(groups),
                    num_assets=total_assets,
                    width_buckets=width_buckets,
                    steps_per_epoch=num_update_steps_per_epoch,
                    effective_max_train_steps=max_train_steps,
                    effective_num_train_epochs=num_train_epochs,
                    prefix=prefix,
                    global_step=global_step,
                )
                LOGGER.info("Saved checkpoint at step %d -> %s", global_step, artifacts.backbone_dir)

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
            image_processor=image_processor,
            args=args,
            num_groups=len(groups),
            num_assets=total_assets,
            width_buckets=width_buckets,
            steps_per_epoch=num_update_steps_per_epoch,
            effective_max_train_steps=max_train_steps,
            effective_num_train_epochs=num_train_epochs,
            prefix="",
            global_step=global_step,
        )
        avg_loss = cumulative_loss / max(1, global_step)
        LOGGER.info("Training finished after %d steps. avg_loss=%.6f", global_step, avg_loss)
        LOGGER.info("Saved backbone: %s", final_artifacts.backbone_dir)
        LOGGER.info("Saved projection head: %s", final_artifacts.projection_head_path)
        LOGGER.info("Saved config: %s", final_artifacts.config_path)

    return {
        "global_step": int(global_step),
        "groups": int(len(groups)),
        "assets": int(total_assets),
        "output_dir": str(output_dir),
        "backbone_dir": str(final_artifacts.backbone_dir) if final_artifacts else "",
        "projection_head_path": str(final_artifacts.projection_head_path) if final_artifacts else "",
        "config_path": str(final_artifacts.config_path) if final_artifacts else "",
    }


def build_parser():
    return create_train_text_hierarchy_vit_parser(add_help=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
