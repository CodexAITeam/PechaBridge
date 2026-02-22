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
from pechabridge.training.losses import MpNCEConfig, multi_positive_infonce
from pechabridge.training.mnn_pairs import PatchMeta, load_mnn_map, load_patch_metadata, patch_id_to_index_map
from pechabridge.training.samplers.pair_batch_sampler import PairBatchSampler, SamplerConfig
from pechabridge.training.weak_ocr_pairs import load_ocr_weak_map, merge_positive_maps

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


class PatchTwoViewDataset(Dataset):
    """Dataset for patch-level mpNCE training with two augmented views per patch."""

    def __init__(self, records: Sequence[PatchMeta], transform: HierarchyTwoViewTransform):
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[int(idx)]
        with Image.open(rec.image_path) as im:
            rgb = im.convert("RGB")
        view_one, view_two = self.transform(rgb, rgb)
        return {
            "view_one": view_one,
            "view_two": view_two,
            "patch_id": int(rec.patch_id),
            "doc_id": str(rec.doc_id),
            "page_id": str(rec.page_id),
            "line_id": int(rec.line_id),
            "scale_w": int(rec.scale_w),
            "k": int(rec.k),
            "x0_norm": float(rec.x0_norm),
            "x1_norm": float(rec.x1_norm),
            "path": str(rec.image_path),
        }


def _patch_meta_from_batch_row(row: Dict[str, Any]) -> PatchMeta:
    return PatchMeta(
        patch_id=int(row["patch_id"]),
        doc_id=str(row["doc_id"]),
        page_id=str(row["page_id"]),
        line_id=int(row["line_id"]),
        scale_w=int(row["scale_w"]),
        k=int(row["k"]),
        x0_norm=float(row["x0_norm"]),
        x1_norm=float(row["x1_norm"]),
        ink_ratio=0.0,
        boundary_score=0.0,
        image_path=Path(str(row.get("path", ""))),
    )


def _collate_patch_pairs(batch: List[Dict[str, Any]], patch_multiple: int) -> Dict[str, Any]:
    if not batch:
        raise RuntimeError("Empty batch")
    pm = max(1, int(patch_multiple))
    widths = [int(row["view_one"].shape[-1]) for row in batch] + [int(row["view_two"].shape[-1]) for row in batch]
    max_w = int(math.ceil(max(widths) / pm) * pm)
    view_one = torch.stack([_pad_right_to_width(row["view_one"], max_w) for row in batch], dim=0)
    view_two = torch.stack([_pad_right_to_width(row["view_two"], max_w) for row in batch], dim=0)
    meta_rows = [
        {
            "patch_id": int(row["patch_id"]),
            "doc_id": str(row["doc_id"]),
            "page_id": str(row["page_id"]),
            "line_id": int(row["line_id"]),
            "scale_w": int(row["scale_w"]),
            "k": int(row["k"]),
            "x0_norm": float(row["x0_norm"]),
            "x1_norm": float(row["x1_norm"]),
            "path": str(row["path"]),
        }
        for row in batch
    ]
    return {
        "view_one": view_one,
        "view_two": view_two,
        "meta_rows": meta_rows,
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


def _freeze_backbone(backbone: nn.Module) -> None:
    for p in backbone.parameters():
        p.requires_grad = False


def _find_backbone_blocks(backbone: nn.Module) -> List[nn.Module]:
    # ViT / DINOv2 patterns.
    if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        try:
            return list(backbone.encoder.layer)
        except Exception:
            pass
    if hasattr(backbone, "vit") and hasattr(backbone.vit, "encoder") and hasattr(backbone.vit.encoder, "layer"):
        try:
            return list(backbone.vit.encoder.layer)
        except Exception:
            pass
    if hasattr(backbone, "blocks"):
        try:
            return list(backbone.blocks)
        except Exception:
            pass
    return []


def _unfreeze_last_n_blocks(backbone: nn.Module, n_blocks: int) -> int:
    blocks = _find_backbone_blocks(backbone)
    n = max(0, int(n_blocks))
    if n <= 0:
        return 0
    if not blocks:
        # Fallback: unfreeze full backbone if block structure is unknown.
        for p in backbone.parameters():
            p.requires_grad = True
        return sum(1 for p in backbone.parameters() if p.requires_grad)

    for blk in blocks[-n:]:
        for p in blk.parameters():
            p.requires_grad = True
    # Keep final norms trainable if present.
    for attr in ("layernorm", "post_layernorm", "ln_f", "norm"):
        mod = getattr(backbone, attr, None)
        if mod is None:
            continue
        for p in mod.parameters():
            p.requires_grad = True
    return sum(1 for p in backbone.parameters() if p.requires_grad)


class _PatchEvalDataset(Dataset):
    def __init__(
        self,
        records: Sequence[PatchMeta],
        *,
        target_height: int,
        width_buckets: Sequence[int],
        patch_multiple: int,
        max_width: int,
        mean: Sequence[float],
        std: Sequence[float],
    ):
        self.records = list(records)
        self.target_height = int(target_height)
        self.width_buckets = [int(v) for v in width_buckets]
        self.patch_multiple = int(patch_multiple)
        self.max_width = int(max_width)
        self.mean = torch.tensor([float(v) for v in mean[:3]], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([max(1e-6, float(v)) for v in std[:3]], dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[int(idx)]
        with Image.open(rec.image_path) as im:
            rgb = im.convert("RGB")
        normalized = _normalize_for_vit(
            image=rgb,
            target_height=self.target_height,
            width_buckets=self.width_buckets,
            patch_multiple=self.patch_multiple,
            max_width=self.max_width,
        )
        arr = np.asarray(normalized).astype(np.float32) / 255.0
        ten = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()
        ten = (ten - self.mean) / self.std
        return {"index": int(idx), "pixel_values": ten}


def _collate_eval_patches(batch: List[Dict[str, Any]], patch_multiple: int) -> Dict[str, Any]:
    if not batch:
        raise RuntimeError("Empty eval batch")
    pm = max(1, int(patch_multiple))
    widths = [int(row["pixel_values"].shape[-1]) for row in batch]
    max_w = int(math.ceil(max(widths) / pm) * pm)
    pixels = torch.stack([_pad_right_to_width(row["pixel_values"], max_w) for row in batch], dim=0)
    idx = torch.tensor([int(row["index"]) for row in batch], dtype=torch.long)
    return {"index": idx, "pixel_values": pixels}


def _export_faiss_embeddings(
    *,
    accelerator: Accelerator,
    backbone: nn.Module,
    projection_head: nn.Module,
    records: Sequence[PatchMeta],
    output_dir: Path,
    target_height: int,
    width_buckets: Sequence[int],
    patch_multiple: int,
    max_width: int,
    image_mean: Sequence[float],
    image_std: Sequence[float],
    batch_size: int,
    num_workers: int,
) -> Tuple[Path, Path]:
    ds = _PatchEvalDataset(
        records=records,
        target_height=target_height,
        width_buckets=width_buckets,
        patch_multiple=patch_multiple,
        max_width=max_width,
        mean=image_mean,
        std=image_std,
    )
    dl = DataLoader(
        ds,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=True,
        drop_last=False,
        collate_fn=lambda b: _collate_eval_patches(b, patch_multiple=int(patch_multiple)),
    )

    embeddings: Dict[int, np.ndarray] = {}
    ub = accelerator.unwrap_model(backbone).to(accelerator.device)
    uh = accelerator.unwrap_model(projection_head).to(accelerator.device)
    ub.eval()
    uh.eval()
    with torch.no_grad():
        for batch in tqdm(dl, desc="export-faiss-embeddings", disable=not accelerator.is_local_main_process):
            ids = batch["index"].tolist()
            pix = batch["pixel_values"].to(accelerator.device, non_blocking=True)
            emb = _pooled_image_embedding(ub, pix)
            emb = uh(emb)
            emb = F.normalize(emb, dim=-1)
            arr = emb.detach().cpu().numpy().astype(np.float32, copy=False)
            for i, ridx in enumerate(ids):
                embeddings[int(ridx)] = arr[i]
    if len(embeddings) != len(records):
        missing = [i for i in range(len(records)) if i not in embeddings]
        raise RuntimeError(f"Embedding export missing {len(missing)} records.")

    mat = np.stack([embeddings[i] for i in range(len(records))], axis=0).astype(np.float32, copy=False)
    emb_path = (output_dir / "faiss_embeddings.npy").resolve()
    np.save(emb_path, mat)

    meta_rows = []
    for rec in records:
        meta_rows.append(
            {
                "patch_id": int(rec.patch_id),
                "doc_id": str(rec.doc_id),
                "page_id": str(rec.page_id),
                "line_id": int(rec.line_id),
                "scale_w": int(rec.scale_w),
                "k": int(rec.k),
                "x0_norm": float(rec.x0_norm),
                "x1_norm": float(rec.x1_norm),
                "patch_image_path": str(rec.image_path),
            }
        )
    meta_df = pd.DataFrame(meta_rows)
    meta_path = (output_dir / "faiss_embeddings_meta.parquet").resolve()
    meta_df.to_parquet(meta_path, index=False)
    return emb_path, meta_path


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
    extra_config: Optional[Dict[str, Any]] = None,
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

    config_payload: Dict[str, Any] = {
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
    }
    if isinstance(extra_config, dict):
        config_payload.update(extra_config)

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, ensure_ascii=False, indent=2)

    return TrainingArtifacts(
        backbone_dir=backbone_dir,
        projection_head_path=projection_head_path,
        config_path=config_path,
    )


def _run_patch_mpnce_training(
    *,
    args,
    accelerator: Accelerator,
    dataset_dir: Path,
    output_dir: Path,
    width_buckets: Sequence[int],
    image_mean: Sequence[float],
    image_std: Sequence[float],
    image_processor: Any,
) -> Dict[str, Any]:
    meta_path = Path(getattr(args, "patch_meta_parquet", "") or "").expanduser().resolve()
    if not meta_path.exists() or not meta_path.is_file():
        meta_path = (dataset_dir / "meta" / "patches.parquet").resolve()
    if not meta_path.exists() or not meta_path.is_file():
        raise FileNotFoundError(f"Patch metadata parquet not found: {meta_path}")

    pairs_path_raw = str(getattr(args, "pairs_parquet", "") or "").strip()
    pairs_path = Path(pairs_path_raw).expanduser().resolve() if pairs_path_raw else (dataset_dir / "meta" / "mnn_pairs.parquet").resolve()
    if not pairs_path.exists() or not pairs_path.is_file():
        pairs_path = None
    weak_ocr_path_raw = str(getattr(args, "weak_ocr_parquet", "") or "").strip()
    weak_ocr_path = (
        Path(weak_ocr_path_raw).expanduser().resolve()
        if weak_ocr_path_raw
        else (dataset_dir / "meta" / "weak_ocr.parquet").resolve()
    )
    if not weak_ocr_path.exists() or not weak_ocr_path.is_file():
        # Fallback naming variants users may have used.
        alt_candidates = [
            (dataset_dir / "meta" / "weak_ocr_labels.parquet").resolve(),
            (dataset_dir / "meta" / "ocr_weak_labels.parquet").resolve(),
        ]
        found_alt = next((p for p in alt_candidates if p.exists() and p.is_file()), None)
        weak_ocr_path = found_alt if found_alt is not None else None

    records = load_patch_metadata(
        dataset_dir=dataset_dir,
        patch_meta_parquet=meta_path,
        ink_ratio_min=float(getattr(args, "ink_ratio_min", 0.0)),
        boundary_score_min=float(getattr(args, "boundary_score_min", 0.0)),
    )
    if not records:
        raise RuntimeError(f"No patch records found after filtering in {meta_path}")
    pid_to_index = patch_id_to_index_map(records)
    positive_sources = str(getattr(args, "positive_sources", "mnn") or "mnn").strip().lower()
    if positive_sources not in {"mnn", "ocr", "both"}:
        positive_sources = "mnn"
    use_mnn = positive_sources in {"mnn", "both"}
    use_ocr = positive_sources in {"ocr", "both"}

    mnn_map_all = load_mnn_map(
        pairs_parquet=pairs_path if use_mnn else None,
        pair_min_sim=float(getattr(args, "pair_min_sim", 0.25)),
        pair_min_stability_ratio=float(getattr(args, "pair_min_stability_ratio", 0.5)),
        require_multi_scale_ok=bool(getattr(args, "pair_require_multi_scale_ok", False)),
        weight_scale=float(getattr(args, "w_mnn_scale", 1.0)),
        max_neighbors_per_anchor=int(getattr(args, "max_neighbors_per_anchor", 0)),
    ) if use_mnn else {}
    ocr_map_all = load_ocr_weak_map(
        weak_ocr_parquet=weak_ocr_path if use_ocr else None,
        pair_min_confidence=float(getattr(args, "ocr_min_confidence", 0.2)),
        min_chars=int(getattr(args, "ocr_min_chars", 2)),
        max_group_size=int(getattr(args, "ocr_max_group_size", 128)),
        max_neighbors_per_anchor=int(getattr(args, "ocr_max_neighbors_per_anchor", 0)),
        weight_scale=float(getattr(args, "w_ocr_scale", 1.0)),
        require_no_error=bool(getattr(args, "ocr_require_no_error", True)),
        patch_id_allowlist=set(pid_to_index.keys()),
    ) if use_ocr else {}

    mnn_map = mnn_map_all if use_mnn else {}
    ocr_map = ocr_map_all if use_ocr else {}
    sampler_positive_map = merge_positive_maps(mnn_map, ocr_map)

    anchors_with_mnn = sum(1 for rec in records if int(rec.patch_id) in mnn_map and len(mnn_map[int(rec.patch_id)]) > 0)
    anchors_with_ocr = sum(1 for rec in records if int(rec.patch_id) in ocr_map and len(ocr_map[int(rec.patch_id)]) > 0)
    anchors_with_pairs = sum(
        1 for rec in records if int(rec.patch_id) in sampler_positive_map and len(sampler_positive_map[int(rec.patch_id)]) > 0
    )
    coverage = float(anchors_with_pairs) / float(max(1, len(records)))
    if accelerator.is_main_process:
        LOGGER.info(
            "Patch mode: patches=%d positive_sources=%s mnn_pairs_file=%s weak_ocr_file=%s anchors_any=%d coverage=%.4f anchors_mnn=%d anchors_ocr=%d",
            len(records),
            str(pairs_path) if pairs_path else "none",
            str(weak_ocr_path) if weak_ocr_path else "none",
            anchors_with_pairs,
            coverage,
            anchors_with_mnn,
            anchors_with_ocr,
        )
    if bool(getattr(args, "require_pairs", False)) and anchors_with_pairs <= 0:
        raise RuntimeError("require_pairs=True but no anchors with selected positives (MNN/OCR) after filtering.")
    if use_ocr and weak_ocr_path is None:
        raise FileNotFoundError(
            "positive_sources includes OCR but no weak OCR parquet was found. "
            "Use --weak_ocr_parquet or generate dataset/meta/weak_ocr.parquet."
        )

    transform = HierarchyTwoViewTransform(
        target_height=int(args.target_height),
        width_buckets=width_buckets,
        patch_multiple=int(args.patch_multiple),
        max_width=int(args.max_width),
        mean=list(image_mean[:3]),
        std=list(image_std[:3]),
    )
    dataset = PatchTwoViewDataset(records=records, transform=transform)
    pair_sampler = PairBatchSampler(
        records=records,
        patch_id_to_index=pid_to_index,
        mnn_map=sampler_positive_map,
        config=SamplerConfig(
            batch_size=int(args.batch_size),
            p_pair=float(getattr(args, "p_pair", 0.6)),
            hard_negative_ratio=float(getattr(args, "hard_negative_ratio", 0.2)),
            drop_last=True,
            seed=int(getattr(args, "pair_sampling_seed", args.seed)),
        ),
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=pair_sampler,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=True,
        collate_fn=lambda b: _collate_patch_pairs(b, patch_multiple=int(args.patch_multiple)),
    )
    if len(dataloader) == 0:
        raise RuntimeError("Pair-aware DataLoader has zero batches.")

    backbone = AutoModel.from_pretrained(args.model_name_or_path)
    if bool(args.gradient_checkpointing) and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
    hidden_size = getattr(backbone.config, "hidden_size", None)
    if hidden_size is None:
        raise RuntimeError("Backbone config has no hidden_size.")
    projection_head = ProjectionHead(in_dim=int(hidden_size), out_dim=int(args.projection_dim))

    # Two-phase schedule.
    phase1_epochs = max(0, int(getattr(args, "phase1_epochs", 0)))
    phase2_epochs = max(0, int(getattr(args, "phase2_epochs", 0)))
    if phase1_epochs <= 0 and phase2_epochs <= 0:
        total_epochs = max(1, int(args.num_train_epochs))
        phase1_epochs = max(1, int(math.ceil(total_epochs * 0.25)))
        phase2_epochs = max(0, int(total_epochs - phase1_epochs))
    total_sched_epochs = phase1_epochs + phase2_epochs

    freeze_all = bool(args.freeze_backbone) or phase1_epochs > 0
    if freeze_all:
        _freeze_backbone(backbone)
    if bool(args.freeze_backbone):
        trainable_backbone = 0
    else:
        if phase1_epochs <= 0:
            trainable_backbone = _unfreeze_last_n_blocks(backbone, int(getattr(args, "unfreeze_last_n_blocks", 2)))
        else:
            trainable_backbone = 0

    if accelerator.is_main_process:
        LOGGER.info(
            "Phase schedule: phase1_epochs=%d phase2_epochs=%d trainable_backbone_params_now=%d",
            phase1_epochs,
            phase2_epochs,
            trainable_backbone,
        )

    base_lr = float(args.lr)
    wd = float(args.weight_decay)
    optimizer = torch.optim.AdamW(
        [
            {"params": list(backbone.parameters()), "lr": 0.0 if freeze_all else base_lr},
            {"params": list(projection_head.parameters()), "lr": base_lr},
        ],
        lr=base_lr,
        weight_decay=wd,
    )

    steps_per_epoch = len(dataloader)
    req_max_steps = int(args.max_train_steps)
    if req_max_steps > 0:
        max_train_steps = int(req_max_steps)
        num_train_epochs = int(math.ceil(max_train_steps / max(1, steps_per_epoch)))
    else:
        num_train_epochs = int(max(1, total_sched_epochs))
        max_train_steps = int(num_train_epochs * steps_per_epoch)

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_steps),
        num_training_steps=max_train_steps,
    )

    backbone, projection_head, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        backbone, projection_head, optimizer, dataloader, lr_scheduler
    )

    mp_cfg = MpNCEConfig(
        tau=float(args.temperature),
        w_ocr=float(getattr(args, "w_ocr", 0.5)),
        w_overlap=float(getattr(args, "w_overlap", 0.3)),
        w_multiscale=float(getattr(args, "w_multiscale", 0.2)),
        t_iou=float(getattr(args, "t_iou", 0.6)),
        eps_center=float(getattr(args, "eps_center", 0.06)),
        min_positives_per_anchor=max(1, int(getattr(args, "min_positives_per_anchor", 1))),
        allow_self_fallback=bool(getattr(args, "allow_self_fallback", True)),
        use_mnn=bool(use_mnn),
        use_ocr=bool(use_ocr),
        exclude_same_page_in_denominator=bool(getattr(args, "exclude_same_page_in_denominator", False)),
        lambda_smooth=float(getattr(args, "lambda_smooth", 0.05)),
    )

    progress_bar = tqdm(total=max_train_steps, disable=not accelerator.is_local_main_process, desc="train-text-hierarchy-vit-mp")
    global_step = 0
    cumulative_loss = 0.0
    skipped_batches = 0
    fallback_count_total = 0.0
    mnn_pos_total = 0.0
    ocr_pos_total = 0.0
    overlap_pos_total = 0.0
    multiscale_pos_total = 0.0
    valid_anchor_total = 0.0
    avg_pos_accum = 0.0
    avg_pos_count = 0
    epoch_phase2_switched = False
    phase1_steps = int(max(0, phase1_epochs) * steps_per_epoch)

    for epoch in range(num_train_epochs):
        pair_sampler.set_epoch(epoch)
        if not bool(args.freeze_backbone):
            backbone.train()
        projection_head.train()

        # Phase transition.
        if (not epoch_phase2_switched) and (not bool(args.freeze_backbone)) and phase2_epochs > 0 and global_step >= phase1_steps:
            unwrapped_backbone = accelerator.unwrap_model(backbone)
            _freeze_backbone(unwrapped_backbone)
            trainable_now = _unfreeze_last_n_blocks(unwrapped_backbone, int(getattr(args, "unfreeze_last_n_blocks", 2)))
            scale_lr = float(getattr(args, "phase2_lr_scale", 0.1))
            for g in optimizer.param_groups:
                g["lr"] = float(g["lr"]) * scale_lr
            if hasattr(lr_scheduler, "base_lrs"):
                try:
                    lr_scheduler.base_lrs = [float(v) * scale_lr for v in lr_scheduler.base_lrs]  # type: ignore[attr-defined]
                except Exception:
                    pass
            epoch_phase2_switched = True
            if accelerator.is_main_process:
                LOGGER.info("Switched to phase2: unfreeze_last_n_blocks=%s trainable_params=%d", str(getattr(args, "unfreeze_last_n_blocks", 2)), trainable_now)

        for batch in dataloader:
            if global_step >= max_train_steps:
                break
            view_one = batch["view_one"].to(accelerator.device, non_blocking=True)
            view_two = batch["view_two"].to(accelerator.device, non_blocking=True)
            meta_rows = batch["meta_rows"]

            if bool(args.freeze_backbone) or (phase1_epochs > 0 and global_step < phase1_steps):
                with torch.no_grad():
                    emb_one = _pooled_image_embedding(backbone, view_one)
                    emb_two = _pooled_image_embedding(backbone, view_two)
            else:
                emb_one = _pooled_image_embedding(backbone, view_one)
                emb_two = _pooled_image_embedding(backbone, view_two)
            proj_one = projection_head(emb_one)
            proj_two = projection_head(emb_two)
            z = torch.cat([proj_one, proj_two], dim=0)

            # Duplicate metadata for the two views.
            metas_a = [_patch_meta_from_batch_row(r) for r in meta_rows]
            patch_ids_a = [int(r["patch_id"]) for r in meta_rows]
            metas = metas_a + metas_a
            patch_ids = patch_ids_a + patch_ids_a

            loss, stats = multi_positive_infonce(
                z=z,
                metas=metas,
                patch_ids=patch_ids,
                mnn_map=mnn_map,
                cfg=mp_cfg,
                ocr_map=ocr_map,
            )

            valid_anchors = int(round(float(stats.get("valid_anchors", 0.0))))
            min_valid = max(1, int(getattr(args, "min_valid_anchors_per_batch", 1)))
            if valid_anchors < min_valid:
                optimizer.zero_grad(set_to_none=True)
                skipped_batches += 1
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss="skip", valid=valid_anchors, skip=skipped_batches)
                continue

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            cumulative_loss += float(loss.detach().item())
            valid_anchor_total += float(stats.get("valid_anchors", 0.0))
            fallback_count_total += float(stats.get("fallback_pos", 0.0))
            mnn_pos_total += float(stats.get("mnn_pos", 0.0))
            ocr_pos_total += float(stats.get("ocr_pos", 0.0))
            overlap_pos_total += float(stats.get("overlap_pos", 0.0))
            multiscale_pos_total += float(stats.get("multiscale_pos", 0.0))
            avg_pos_accum += float(stats.get("avg_positives", 0.0))
            avg_pos_count += 1
            progress_bar.update(1)
            progress_bar.set_postfix(
                loss=f"{loss.detach().item():.4f}",
                valid=f"{valid_anchors}",
                mnn=f"{int(stats.get('mnn_pos', 0.0))}",
                ocr=f"{int(stats.get('ocr_pos', 0.0))}",
            )

            if (
                int(args.checkpoint_every_steps) > 0
                and global_step % int(args.checkpoint_every_steps) == 0
                and accelerator.is_main_process
            ):
                prefix = f"checkpoint_step_{global_step:07d}"
                _save_artifacts(
                    accelerator=accelerator,
                    output_dir=output_dir,
                    backbone=backbone,
                    projection_head=projection_head,
                    image_processor=image_processor,
                    args=args,
                    num_groups=len(records),
                    num_assets=len(records),
                    width_buckets=width_buckets,
                    steps_per_epoch=steps_per_epoch,
                    effective_max_train_steps=max_train_steps,
                    effective_num_train_epochs=num_train_epochs,
                    prefix=prefix,
                    global_step=global_step,
                    extra_config={
                        "task": "text_hierarchy_vit_mpnce_retrieval",
                        "patch_meta_parquet": str(meta_path),
                        "pairs_parquet": str(pairs_path) if pairs_path else "",
                        "weak_ocr_parquet": str(weak_ocr_path) if weak_ocr_path else "",
                        "positive_sources": positive_sources,
                    },
                )
        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()
    final_artifacts: Optional[TrainingArtifacts] = None
    emb_path = ""
    emb_meta_path = ""
    if accelerator.is_main_process:
        extra_cfg = {
            "task": "text_hierarchy_vit_mpnce_retrieval",
            "positive_sources": positive_sources,
            "patch_meta_parquet": str(meta_path),
            "pairs_parquet": str(pairs_path) if pairs_path else "",
            "weak_ocr_parquet": str(weak_ocr_path) if weak_ocr_path else "",
            "pair_min_sim": float(getattr(args, "pair_min_sim", 0.25)),
            "pair_min_stability_ratio": float(getattr(args, "pair_min_stability_ratio", 0.5)),
            "pair_require_multi_scale_ok": bool(getattr(args, "pair_require_multi_scale_ok", False)),
            "require_pairs": bool(getattr(args, "require_pairs", False)),
            "allow_self_fallback": bool(getattr(args, "allow_self_fallback", True)),
            "w_mnn_scale": float(getattr(args, "w_mnn_scale", 1.0)),
            "w_ocr_scale": float(getattr(args, "w_ocr_scale", 1.0)),
            "w_ocr": float(getattr(args, "w_ocr", 0.5)),
            "ocr_min_confidence": float(getattr(args, "ocr_min_confidence", 0.2)),
            "ocr_min_chars": int(getattr(args, "ocr_min_chars", 2)),
            "ocr_max_group_size": int(getattr(args, "ocr_max_group_size", 128)),
            "ocr_max_neighbors_per_anchor": int(getattr(args, "ocr_max_neighbors_per_anchor", 0)),
            "w_overlap": float(getattr(args, "w_overlap", 0.3)),
            "w_multiscale": float(getattr(args, "w_multiscale", 0.2)),
            "t_iou": float(getattr(args, "t_iou", 0.6)),
            "eps_center": float(getattr(args, "eps_center", 0.06)),
            "lambda_smooth": float(getattr(args, "lambda_smooth", 0.05)),
            "exclude_same_page_in_denominator": bool(getattr(args, "exclude_same_page_in_denominator", False)),
            "phase1_epochs": int(phase1_epochs),
            "phase2_epochs": int(phase2_epochs),
            "unfreeze_last_n_blocks": int(getattr(args, "unfreeze_last_n_blocks", 2)),
            "phase2_lr_scale": float(getattr(args, "phase2_lr_scale", 0.1)),
            "sampler_p_pair": float(getattr(args, "p_pair", 0.6)),
            "sampler_hard_negative_ratio": float(getattr(args, "hard_negative_ratio", 0.2)),
            "anchors_with_mnn": int(anchors_with_mnn),
            "anchors_with_ocr": int(anchors_with_ocr),
            "anchors_with_pairs": int(anchors_with_pairs),
            "coverage_ratio": float(coverage),
            "stats_valid_anchors_total": float(valid_anchor_total),
            "stats_mnn_pos_total": float(mnn_pos_total),
            "stats_ocr_pos_total": float(ocr_pos_total),
            "stats_overlap_pos_total": float(overlap_pos_total),
            "stats_multiscale_pos_total": float(multiscale_pos_total),
            "stats_fallback_pos_total": float(fallback_count_total),
            "stats_avg_positives_per_anchor": float(avg_pos_accum / max(1, avg_pos_count)),
            "stats_skipped_batches": int(skipped_batches),
            "loss_type": "mp_infonce",
        }
        final_artifacts = _save_artifacts(
            accelerator=accelerator,
            output_dir=output_dir,
            backbone=backbone,
            projection_head=projection_head,
            image_processor=image_processor,
            args=args,
            num_groups=len(records),
            num_assets=len(records),
            width_buckets=width_buckets,
            steps_per_epoch=steps_per_epoch,
            effective_max_train_steps=max_train_steps,
            effective_num_train_epochs=num_train_epochs,
            prefix="",
            global_step=global_step,
            extra_config=extra_cfg,
        )
        # FAISS-ready embedding export.
        emb_p, emb_meta_p = _export_faiss_embeddings(
            accelerator=accelerator,
            backbone=backbone,
            projection_head=projection_head,
            records=records,
            output_dir=output_dir,
            target_height=int(args.target_height),
            width_buckets=width_buckets,
            patch_multiple=int(args.patch_multiple),
            max_width=int(args.max_width),
            image_mean=image_mean[:3],
            image_std=image_std[:3],
            batch_size=max(1, int(args.batch_size)),
            num_workers=max(0, int(args.num_workers)),
        )
        emb_path = str(emb_p)
        emb_meta_path = str(emb_meta_p)
        avg_loss = cumulative_loss / max(1, global_step - skipped_batches)
        LOGGER.info("mpNCE training finished: steps=%d skipped=%d avg_loss=%.6f", global_step, skipped_batches, avg_loss)
        LOGGER.info("Average positives/anchor (batch-mean): %.4f", float(avg_pos_accum / max(1, avg_pos_count)))
        LOGGER.info(
            "Positive counts by source: mnn=%.0f ocr=%.0f overlap=%.0f multiscale=%.0f fallback=%.0f",
            mnn_pos_total,
            ocr_pos_total,
            overlap_pos_total,
            multiscale_pos_total,
            fallback_count_total,
        )
        LOGGER.info("FAISS-ready embeddings: %s / %s", emb_path, emb_meta_path)

    return {
        "mode": "mpnce_patch",
        "global_step": int(global_step),
        "groups": int(len(records)),
        "assets": int(len(records)),
        "output_dir": str(output_dir),
        "backbone_dir": str(final_artifacts.backbone_dir) if final_artifacts else "",
        "projection_head_path": str(final_artifacts.projection_head_path) if final_artifacts else "",
        "config_path": str(final_artifacts.config_path) if final_artifacts else "",
        "faiss_embeddings_path": emb_path,
        "faiss_embeddings_meta_path": emb_meta_path,
        "anchors_with_pairs": int(anchors_with_pairs),
        "anchors_with_mnn": int(anchors_with_mnn),
        "anchors_with_ocr": int(anchors_with_ocr),
        "coverage_ratio": float(coverage),
        "skipped_batches": int(skipped_batches),
    }


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

    train_mode = str(getattr(args, "train_mode", "auto") or "auto").strip().lower()
    if train_mode not in {"auto", "legacy", "patch_mpnce"}:
        train_mode = "auto"
    patch_meta_arg = str(getattr(args, "patch_meta_parquet", "") or "").strip()
    patch_meta_candidate = (
        Path(patch_meta_arg).expanduser().resolve()
        if patch_meta_arg
        else (dataset_dir / "meta" / "patches.parquet").resolve()
    )
    patch_meta_exists = patch_meta_candidate.exists() and patch_meta_candidate.is_file()
    use_patch_mode = (train_mode == "patch_mpnce") or (train_mode == "auto" and patch_meta_exists)
    if accelerator.is_main_process:
        LOGGER.info(
            "Training mode resolved: requested=%s resolved=%s patch_meta=%s exists=%s",
            train_mode,
            "patch_mpnce" if use_patch_mode else "legacy",
            str(patch_meta_candidate),
            str(patch_meta_exists),
        )
    if use_patch_mode:
        return _run_patch_mpnce_training(
            args=args,
            accelerator=accelerator,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            width_buckets=width_buckets,
            image_mean=image_mean,
            image_std=image_std,
            image_processor=image_processor,
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
        "mode": "legacy",
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
