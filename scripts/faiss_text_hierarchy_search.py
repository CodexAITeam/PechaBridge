#!/usr/bin/env python3
"""FAISS-based similarity search for TextHierarchy embeddings."""

from __future__ import annotations

import json
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_text_hierarchy_vit import (  # noqa: E402
    HierarchyGroup,
    ProjectionHead,
    _discover_hierarchy_groups,
    _normalize_for_vit,
    _parse_width_buckets,
    _pooled_image_embedding,
)
from tibetan_utils.arg_utils import create_faiss_text_hierarchy_search_parser  # noqa: E402

LOGGER = logging.getLogger("faiss_text_hierarchy_search")
ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class AssetItem:
    index: int
    path: Path
    group_id: int
    group_key: str
    subset: str
    source_key: str
    ppn: str
    page: str
    page_num: int


class AssetDataset(Dataset):
    def __init__(
        self,
        items: Sequence[AssetItem],
        *,
        target_height: int,
        width_buckets: Sequence[int],
        patch_multiple: int,
        max_width: int,
        mean: Sequence[float],
        std: Sequence[float],
    ):
        self.items = list(items)
        self.target_height = int(target_height)
        self.width_buckets = [int(v) for v in width_buckets]
        self.patch_multiple = int(patch_multiple)
        self.max_width = int(max_width)
        self.mean = torch.tensor([float(v) for v in mean[:3]], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([max(1e-6, float(v)) for v in std[:3]], dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        with Image.open(item.path) as im:
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


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _import_faiss():
    try:
        import faiss  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "faiss is required for similarity search. Install `faiss-cpu` or `faiss-gpu`."
        ) from exc
    return faiss


def _resolve_device(raw: str) -> str:
    pref = (raw or "").strip().lower()
    if pref in {"", "auto"}:
        if torch.cuda.is_available():
            return "cuda"
        mps_ok = False
        try:
            mps_ok = bool(torch.backends.mps.is_available())  # type: ignore[attr-defined]
        except Exception:
            mps_ok = False
        return "mps" if mps_ok else "cpu"
    if pref.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("Requested `%s` but CUDA is unavailable; using cpu.", pref)
        return "cpu"
    if pref == "mps":
        try:
            if bool(torch.backends.mps.is_available()):  # type: ignore[attr-defined]
                return "mps"
        except Exception:
            pass
        LOGGER.warning("Requested `mps` but MPS is unavailable; using cpu.")
        return "cpu"
    return pref


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _load_training_config(config_path: str, backbone_dir: Path) -> Dict[str, Any]:
    candidates: List[Path] = []
    if (config_path or "").strip():
        candidates.append(Path(config_path).expanduser().resolve())
    candidates.append(backbone_dir.parent / "training_config.json")
    candidates.extend(sorted(backbone_dir.parent.glob("*training_config.json")))
    for p in candidates:
        if not p.exists() or not p.is_file():
            continue
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            payload["_config_path"] = str(p)
            return payload
    return {}


def _resolve_normalization(args, config: Dict[str, Any], meta_norm: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    source = meta_norm if isinstance(meta_norm, dict) and meta_norm else config
    target_height = _to_int(source.get("target_height", 64), 64)
    max_width = _to_int(source.get("max_width", 1024), 1024)
    patch_multiple = _to_int(source.get("patch_multiple", 16), 16)
    buckets_raw = source.get("width_buckets", [256, 384, 512, 768])

    if int(args.target_height) > 0:
        target_height = int(args.target_height)
    if int(args.max_width) > 0:
        max_width = int(args.max_width)
    if int(args.patch_multiple) > 0:
        patch_multiple = int(args.patch_multiple)

    if (args.width_buckets or "").strip():
        raw = str(args.width_buckets)
    else:
        if isinstance(buckets_raw, str):
            raw = buckets_raw
        elif isinstance(buckets_raw, (tuple, list)):
            raw = ",".join(str(v) for v in buckets_raw)
        else:
            raw = "256,384,512,768"

    width_buckets = _parse_width_buckets(
        raw=raw,
        patch_multiple=max(1, int(patch_multiple)),
        max_width=max(16, int(max_width)),
    )
    return {
        "target_height": int(max(8, target_height)),
        "max_width": int(max(16, max_width)),
        "patch_multiple": int(max(1, patch_multiple)),
        "width_buckets": [int(v) for v in width_buckets],
    }


def _pad_right_to_width(tensor: torch.Tensor, target_width: int) -> torch.Tensor:
    width = int(tensor.shape[-1])
    if width >= target_width:
        return tensor[:, :, :target_width]
    pad = target_width - width
    return F.pad(tensor, (0, pad, 0, 0), mode="replicate")


def _collate(batch: List[Dict[str, Any]], patch_multiple: int) -> Dict[str, Any]:
    if not batch:
        raise RuntimeError("Empty batch.")
    pm = max(1, int(patch_multiple))
    max_w = max(int(row["pixel_values"].shape[-1]) for row in batch)
    max_w = int(math.ceil(max_w / pm) * pm)
    pixels = torch.stack([_pad_right_to_width(row["pixel_values"], max_w) for row in batch], dim=0)
    indices = torch.tensor([int(row["index"]) for row in batch], dtype=torch.long)
    return {"index": indices, "pixel_values": pixels}


def _load_projection_head(projection_head_path: str, hidden_size: int) -> Optional[ProjectionHead]:
    if not (projection_head_path or "").strip():
        return None
    path = Path(projection_head_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        LOGGER.warning("Projection head not found: %s", path)
        return None
    payload = torch.load(str(path), map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload.get("state_dict", {})
        in_dim = _to_int(payload.get("input_dim", hidden_size), hidden_size)
        out_dim = _to_int(payload.get("output_dim", in_dim), in_dim)
    else:
        state_dict = payload
        in_dim = int(hidden_size)
        out_dim = int(hidden_size)
    head = ProjectionHead(in_dim=int(max(1, in_dim)), out_dim=int(max(1, out_dim)))
    head.load_state_dict(state_dict, strict=False)
    head.eval()
    return head


def _extract_source_key(rel_path: Path) -> str:
    parts = list(rel_path.parts)
    if not parts:
        return rel_path.stem
    if parts[0] == "TextHierarchy" and len(parts) >= 2:
        return str(parts[1])
    if parts[0] == "NumberCrops" and len(parts) >= 3:
        return str(parts[2])
    return str(parts[0])


def _parse_ppn_from_source_key(source_key: str) -> str:
    txt = str(source_key or "")
    m = re.search(r"(?i)ppn[_-]?([0-9x]+)", txt)
    if m:
        return m.group(1).upper()
    for tok in txt.split("__"):
        tok_norm = tok.strip()
        if not tok_norm:
            continue
        m2 = re.fullmatch(r"(?i)([0-9]{7,}x?)", tok_norm)
        if m2:
            return m2.group(1).upper()
    return ""


def _parse_page_from_source_key(source_key: str) -> Tuple[str, int]:
    txt = str(source_key or "")
    patterns = [
        r"(?i)ppn[_-]?[0-9x]+[_-]0*([0-9]{1,9})",
        r"(?i)(?:page|seite|folio|image|img|scan|sheet|s)[_-]?0*([0-9]{1,9})",
    ]
    for pat in patterns:
        m = re.search(pat, txt)
        if m:
            raw = str(m.group(1))
            try:
                num = int(raw)
            except Exception:
                continue
            return str(num), int(num)

    tokens = [tok for tok in txt.split("__") if tok]
    if tokens:
        last_tok = tokens[-1]
        groups = re.findall(r"([0-9]{1,9})", last_tok)
        if groups:
            raw = groups[-1]
            num = int(raw)
            return str(num), int(num)
    for tok in reversed(tokens):
        m2 = re.search(r"0*([0-9]{3,9})$", tok)
        if m2:
            num = int(m2.group(1))
            return str(num), int(num)
    return "", -1


def _build_asset_items(dataset_dir: Path, groups: Sequence[HierarchyGroup]) -> List[AssetItem]:
    items: List[AssetItem] = []
    for gid, group in enumerate(groups):
        for asset in group.assets:
            rel = asset.relative_to(dataset_dir)
            source_key = _extract_source_key(rel)
            ppn = _parse_ppn_from_source_key(source_key)
            page, page_num = _parse_page_from_source_key(source_key)
            subset = rel.parts[0] if rel.parts else ""
            idx = len(items)
            items.append(
                AssetItem(
                    index=int(idx),
                    path=asset,
                    group_id=int(gid),
                    group_key=str(group.key),
                    subset=str(subset),
                    source_key=str(source_key),
                    ppn=str(ppn),
                    page=str(page),
                    page_num=int(page_num),
                )
            )
    return items


def _embed_items(
    items: Sequence[AssetItem],
    dataloader: DataLoader,
    backbone: AutoModel,
    projection_head: Optional[ProjectionHead],
    device: str,
    l2_normalize: bool,
) -> np.ndarray:
    n = len(items)
    out: Dict[int, np.ndarray] = {}
    backbone = backbone.to(device)
    backbone.eval()
    if projection_head is not None:
        projection_head = projection_head.to(device)
        projection_head.eval()

    pbar = tqdm(total=len(dataloader), desc="embed-index-assets")
    with torch.no_grad():
        for batch in dataloader:
            idxs = batch["index"].tolist()
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            emb = _pooled_image_embedding(backbone, pixels)
            if projection_head is not None:
                emb = projection_head(emb)
            if l2_normalize:
                emb = F.normalize(emb, dim=-1)
            arr = emb.detach().cpu().float().numpy()
            for i, ridx in enumerate(idxs):
                out[int(ridx)] = arr[i]
            pbar.update(1)
    pbar.close()

    if len(out) != n:
        miss = [i for i in range(n) if i not in out]
        raise RuntimeError(f"Missing embeddings for {len(miss)} assets.")
    return np.stack([out[i] for i in range(n)], axis=0).astype(np.float32, copy=False)


def _default_meta_path_for_index(index_path: Path) -> Path:
    return Path(str(index_path) + ".meta.json")


def _save_index_with_metadata(
    faiss_mod,
    *,
    index,
    index_path: Path,
    metadata: Dict[str, Any],
) -> Path:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss_mod.write_index(index, str(index_path))
    meta_path = _default_meta_path_for_index(index_path)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta_path


def _load_index_with_metadata(
    faiss_mod,
    *,
    index_path: Path,
    meta_path_override: str,
) -> Tuple[Any, Dict[str, Any], Path]:
    if not index_path.exists() or not index_path.is_file():
        raise FileNotFoundError(f"Index not found: {index_path}")
    meta_path = Path(meta_path_override).expanduser().resolve() if (meta_path_override or "").strip() else _default_meta_path_for_index(index_path)
    if not meta_path.exists() or not meta_path.is_file():
        raise FileNotFoundError(f"Metadata sidecar not found: {meta_path}")
    index = faiss_mod.read_index(str(index_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise RuntimeError("Invalid metadata sidecar format.")
    return index, meta, meta_path


def _encode_query_image(
    query_image_path: Path,
    *,
    norm: Dict[str, Any],
    image_mean: Sequence[float],
    image_std: Sequence[float],
    backbone: AutoModel,
    projection_head: Optional[ProjectionHead],
    device: str,
    l2_normalize: bool,
) -> np.ndarray:
    if not query_image_path.exists() or not query_image_path.is_file():
        raise FileNotFoundError(f"Query image not found: {query_image_path}")
    with Image.open(query_image_path) as im:
        rgb = im.convert("RGB")
    normalized = _normalize_for_vit(
        image=rgb,
        target_height=int(norm["target_height"]),
        width_buckets=[int(v) for v in norm["width_buckets"]],
        patch_multiple=int(norm["patch_multiple"]),
        max_width=int(norm["max_width"]),
    )
    arr = np.asarray(normalized).astype(np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()

    mean = [float(v) for v in image_mean[:3]]
    std = [max(1e-6, float(v)) for v in image_std[:3]]
    ten = (ten - torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)) / torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    pixels = ten.unsqueeze(0).to(device)

    backbone = backbone.to(device)
    backbone.eval()
    if projection_head is not None:
        projection_head = projection_head.to(device)
        projection_head.eval()

    with torch.no_grad():
        emb = _pooled_image_embedding(backbone, pixels)
        if projection_head is not None:
            emb = projection_head(emb)
        if l2_normalize:
            emb = F.normalize(emb, dim=-1)
    return emb[0].detach().cpu().float().numpy()


def run(args) -> Dict[str, Any]:
    _configure_logging()
    faiss_mod = _import_faiss()

    query_image = Path(args.query_image).expanduser().resolve()
    backbone_dir = Path(args.backbone_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not backbone_dir.exists() or not backbone_dir.is_dir():
        raise FileNotFoundError(f"Backbone directory not found: {backbone_dir}")

    config = _load_training_config(args.config_path, backbone_dir)

    index = None
    metadata: Dict[str, Any] = {}
    items: List[AssetItem] = []
    index_path_used = ""
    meta_path_used = ""

    index_path_arg = (args.index_path or "").strip()
    dataset_dir_arg = (args.dataset_dir or "").strip()
    if index_path_arg:
        index_path = Path(index_path_arg).expanduser().resolve()
        index, metadata, meta_path = _load_index_with_metadata(
            faiss_mod,
            index_path=index_path,
            meta_path_override=args.meta_path,
        )
        index_path_used = str(index_path)
        meta_path_used = str(meta_path)
        for i, rec in enumerate(metadata.get("assets") or []):
            if not isinstance(rec, dict):
                continue
            path_str = str(rec.get("path", "") or "")
            if not path_str:
                continue
            items.append(
                AssetItem(
                    index=int(i),
                    path=Path(path_str),
                    group_id=_to_int(rec.get("group_id", -1), -1),
                    group_key=str(rec.get("group_key", "")),
                    subset=str(rec.get("subset", "")),
                    source_key=str(rec.get("source_key", "")),
                    ppn=str(rec.get("ppn", "")),
                    page=str(rec.get("page", "")),
                    page_num=_to_int(rec.get("page_num", -1), -1),
                )
            )
        if index.ntotal <= 0:
            raise RuntimeError("Loaded FAISS index is empty.")
        if len(items) < int(index.ntotal):
            raise RuntimeError(
                f"Metadata has fewer assets ({len(items)}) than index entries ({index.ntotal})."
            )
        LOGGER.info("Loaded FAISS DB: %s (ntotal=%d)", index_path, int(index.ntotal))
    else:
        if not dataset_dir_arg:
            raise ValueError("Provide either --index-path or --dataset-dir.")
        dataset_dir = Path(dataset_dir_arg).expanduser().resolve()
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        groups = _discover_hierarchy_groups(
            dataset_dir=dataset_dir,
            include_line_images=bool(args.include_line_images),
            include_word_crops=bool(args.include_word_crops),
            include_number_crops=bool(args.include_number_crops),
            min_assets_per_group=max(1, int(args.min_assets_per_group)),
        )
        if not groups:
            raise RuntimeError(f"No hierarchy assets found in {dataset_dir}")

        items = _build_asset_items(dataset_dir=dataset_dir, groups=groups)
        if not items:
            raise RuntimeError(f"No assets produced from {dataset_dir}")
        LOGGER.info("Building FAISS DB from dataset: groups=%d assets=%d", len(groups), len(items))

        image_processor = AutoImageProcessor.from_pretrained(str(backbone_dir))
        image_mean = list(getattr(image_processor, "image_mean", [0.5, 0.5, 0.5]))
        image_std = list(getattr(image_processor, "image_std", [0.5, 0.5, 0.5]))
        if len(image_mean) == 1:
            image_mean = image_mean * 3
        if len(image_std) == 1:
            image_std = image_std * 3

        norm = _resolve_normalization(args, config, None)
        dataset = AssetDataset(
            items=items,
            target_height=int(norm["target_height"]),
            width_buckets=[int(v) for v in norm["width_buckets"]],
            patch_multiple=int(norm["patch_multiple"]),
            max_width=int(norm["max_width"]),
            mean=image_mean[:3],
            std=image_std[:3],
        )
        dataloader = DataLoader(
            dataset,
            batch_size=max(1, int(args.batch_size)),
            shuffle=False,
            num_workers=max(0, int(args.num_workers)),
            pin_memory=True,
            drop_last=False,
            collate_fn=lambda b: _collate(b, patch_multiple=int(norm["patch_multiple"])),
        )
        if len(dataloader) == 0:
            raise RuntimeError("DataLoader has zero batches.")

        backbone = AutoModel.from_pretrained(str(backbone_dir))
        hidden_size = _to_int(getattr(backbone.config, "hidden_size", 0), 0)
        if hidden_size <= 0:
            raise RuntimeError("Backbone config has no hidden_size.")
        projection_path_effective = (args.projection_head_path or "").strip()
        if not projection_path_effective:
            projection_path_effective = str(config.get("projection_head_path", "") or "").strip()
        projection_head = _load_projection_head(projection_path_effective, hidden_size=hidden_size)
        device = _resolve_device(args.device)

        l2_norm = bool(args.l2_normalize_embeddings)
        metric = str(args.metric).strip().lower()
        if metric not in {"cosine", "l2"}:
            raise ValueError("--metric must be one of: cosine, l2")
        if metric == "cosine" and not l2_norm:
            LOGGER.warning("metric=cosine with --no-l2-normalize-embeddings can degrade retrieval quality.")

        embeddings = _embed_items(
            items=items,
            dataloader=dataloader,
            backbone=backbone,
            projection_head=projection_head,
            device=device,
            l2_normalize=l2_norm,
        )
        dim = int(embeddings.shape[1])
        if metric == "cosine":
            index = faiss_mod.IndexFlatIP(dim)
        else:
            index = faiss_mod.IndexFlatL2(dim)
        index.add(embeddings.astype(np.float32, copy=False))

        metadata = {
            "type": "text_hierarchy_faiss_db",
            "version": 1,
            "metric": metric,
            "l2_normalize_embeddings": bool(l2_norm),
            "embedding_dim": int(dim),
            "dataset_dir": str(dataset_dir),
            "backbone_dir": str(backbone_dir),
            "projection_head_path": (
                str(Path(projection_path_effective).expanduser().resolve())
                if projection_path_effective
                else ""
            ),
            "config_path": str(config.get("_config_path", "")),
            "normalization": norm,
            "assets": [
                {
                    "id": int(item.index),
                    "path": str(item.path),
                    "group_id": int(item.group_id),
                    "group_key": str(item.group_key),
                    "subset": str(item.subset),
                    "source_key": str(item.source_key),
                    "ppn": str(item.ppn),
                    "page": str(item.page),
                    "page_num": int(item.page_num),
                }
                for item in items
            ],
        }

        if (args.save_index_path or "").strip():
            index_path = Path(args.save_index_path).expanduser().resolve()
        else:
            index_path = (output_dir / "text_hierarchy.faiss").resolve()
        meta_path = _save_index_with_metadata(
            faiss_mod,
            index=index,
            index_path=index_path,
            metadata=metadata,
        )
        index_path_used = str(index_path)
        meta_path_used = str(meta_path)
        LOGGER.info("Saved FAISS DB: %s", index_path)
        LOGGER.info("Saved metadata: %s", meta_path)

    if index is None:
        raise RuntimeError("FAISS index not initialized.")

    # Query model load (same backbone/projection for consistent embedding space).
    image_processor = AutoImageProcessor.from_pretrained(str(backbone_dir))
    image_mean = list(getattr(image_processor, "image_mean", [0.5, 0.5, 0.5]))
    image_std = list(getattr(image_processor, "image_std", [0.5, 0.5, 0.5]))
    if len(image_mean) == 1:
        image_mean = image_mean * 3
    if len(image_std) == 1:
        image_std = image_std * 3

    meta_norm = metadata.get("normalization") if isinstance(metadata, dict) else None
    norm = _resolve_normalization(args, config, meta_norm if isinstance(meta_norm, dict) else None)
    backbone = AutoModel.from_pretrained(str(backbone_dir))
    hidden_size = _to_int(getattr(backbone.config, "hidden_size", 0), 0)
    if hidden_size <= 0:
        raise RuntimeError("Backbone config has no hidden_size.")
    projection_head = _load_projection_head(args.projection_head_path, hidden_size=hidden_size)
    if projection_head is None and isinstance(metadata, dict):
        proj_from_meta = str(metadata.get("projection_head_path", "") or "")
        if proj_from_meta:
            projection_head = _load_projection_head(proj_from_meta, hidden_size=hidden_size)
    device = _resolve_device(args.device)

    # Metric + normalization from metadata if available.
    metric = str(metadata.get("metric", args.metric)).strip().lower()
    l2_norm = bool(metadata.get("l2_normalize_embeddings", bool(args.l2_normalize_embeddings)))
    qvec = _encode_query_image(
        query_image_path=query_image,
        norm=norm,
        image_mean=image_mean[:3],
        image_std=image_std[:3],
        backbone=backbone,
        projection_head=projection_head,
        device=device,
        l2_normalize=l2_norm,
    ).astype(np.float32, copy=False)

    top_k = max(1, int(args.top_k))
    top_k_eff = min(top_k, int(index.ntotal))
    if top_k_eff <= 0:
        raise RuntimeError("FAISS index is empty.")

    D, I = index.search(qvec.reshape(1, -1), top_k_eff)
    scores = D[0].tolist()
    idxs = I[0].tolist()

    results: List[Dict[str, Any]] = []
    for rank, (score, idx) in enumerate(zip(scores, idxs), start=1):
        ridx = int(idx)
        if ridx < 0 or ridx >= len(items):
            continue
        item = items[ridx]
        results.append(
            {
                "rank": int(rank),
                "index_id": int(ridx),
                "score": float(score),
                "distance": (float(score) if metric == "l2" else None),
                "similarity": (float(score) if metric != "l2" else None),
                "path": str(item.path),
                "group_id": int(item.group_id),
                "group_key": str(item.group_key),
                "subset": str(item.subset),
                "source_key": str(item.source_key),
                "ppn": str(item.ppn),
                "page": str(item.page),
                "page_num": int(item.page_num),
            }
        )

    query_source_key = _extract_source_key(query_image.relative_to(query_image.parent) if query_image.exists() else query_image)
    query_ppn = _parse_ppn_from_source_key(query_source_key)
    query_page, query_page_num = _parse_page_from_source_key(query_source_key)

    report = {
        "task": "faiss_text_hierarchy_similarity_search",
        "query_image": str(query_image),
        "query_ppn": str(query_ppn),
        "query_page": str(query_page),
        "query_page_num": int(query_page_num),
        "top_k": int(top_k),
        "returned": int(len(results)),
        "metric": str(metric),
        "l2_normalize_embeddings": bool(l2_norm),
        "backbone_dir": str(backbone_dir),
        "projection_head_path": (
            str(Path(args.projection_head_path).expanduser().resolve())
            if (args.projection_head_path or "").strip()
            else str(metadata.get("projection_head_path", ""))
        ),
        "index_path": str(index_path_used),
        "meta_path": str(meta_path_used),
        "normalization": norm,
        "results": results,
    }

    report_path = (output_dir / "faiss_search_results.json").resolve()
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info("Query: %s", query_image)
    LOGGER.info("Top-%d results: %d", top_k, len(results))
    LOGGER.info("Saved search report: %s", report_path)
    if results:
        best = results[0]
        LOGGER.info(
            "Best match: score=%.6f ppn=%s page=%s path=%s",
            float(best["score"]),
            str(best.get("ppn", "")),
            str(best.get("page", "")),
            str(best.get("path", "")),
        )

    return {
        "report_path": str(report_path),
        "index_path": str(index_path_used),
        "meta_path": str(meta_path_used),
        "results": int(len(results)),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = create_faiss_text_hierarchy_search_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
