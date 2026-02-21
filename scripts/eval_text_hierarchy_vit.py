#!/usr/bin/env python3
"""Evaluate a trained TextHierarchy ViT retrieval encoder."""

from __future__ import annotations

import csv
import json
import logging
import math
import random
import statistics
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
from tibetan_utils.arg_utils import create_eval_text_hierarchy_vit_parser  # noqa: E402


LOGGER = logging.getLogger("eval_text_hierarchy_vit")
ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class AssetRecord:
    index: int
    group_key: str
    group_id: int
    path: Path


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _resolve_device(raw: str) -> str:
    pref = (raw or "").strip().lower()
    if pref and pref != "auto":
        if pref.startswith("cuda") and not torch.cuda.is_available():
            LOGGER.warning("Requested %s but CUDA is unavailable; using cpu", raw)
            return "cpu"
        if pref == "mps":
            mps_ok = False
            try:
                mps_ok = bool(torch.backends.mps.is_available())  # type: ignore[attr-defined]
            except Exception:
                mps_ok = False
            if not mps_ok:
                LOGGER.warning("Requested mps but MPS is unavailable; using cpu")
                return "cpu"
        return pref
    if torch.cuda.is_available():
        return "cuda"
    mps_ok = False
    try:
        mps_ok = bool(torch.backends.mps.is_available())  # type: ignore[attr-defined]
    except Exception:
        mps_ok = False
    if mps_ok:
        return "mps"
    return "cpu"


def _load_eval_config(config_path: Optional[str], backbone_dir: Path) -> Dict[str, Any]:
    candidates: List[Path] = []
    if config_path:
        candidates.append(Path(config_path).expanduser().resolve())
    candidates.append(backbone_dir.parent / "training_config.json")
    candidates.extend(sorted(backbone_dir.parent.glob("*training_config.json")))

    seen: set[str] = set()
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if not p.exists() or not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            data["_config_path"] = str(p)
            return data
    return {}


def _normalize_stat_values(raw: Any, default: Sequence[float]) -> List[float]:
    if isinstance(raw, (list, tuple)) and raw:
        out: List[float] = []
        for val in raw[:3]:
            try:
                out.append(float(val))
            except Exception:
                continue
        if len(out) == 1:
            out = out * 3
        if len(out) >= 3:
            return out[:3]
    return [float(v) for v in default[:3]]


def _load_image_stats(backbone_dir: Path, config: Dict[str, Any]) -> Tuple[List[float], List[float], str]:
    default = [0.5, 0.5, 0.5]
    model_name = str(config.get("model_name_or_path", "") or "").strip()
    candidates = [str(backbone_dir)]
    if model_name:
        candidates.append(model_name)

    seen: set[str] = set()
    for source in candidates:
        if source in seen:
            continue
        seen.add(source)
        try:
            image_processor = AutoImageProcessor.from_pretrained(source)
            mean = _normalize_stat_values(getattr(image_processor, "image_mean", None), default)
            std = _normalize_stat_values(getattr(image_processor, "image_std", None), default)
            return mean, std, f"AutoImageProcessor({source})"
        except Exception as exc:
            LOGGER.warning("Could not load image processor from %s: %s", source, exc)

    if "image_mean" in config or "image_std" in config:
        mean = _normalize_stat_values(config.get("image_mean"), default)
        std = _normalize_stat_values(config.get("image_std"), default)
        return mean, std, "training_config"

    return list(default), list(default), "default(0.5)"


def _resolve_normalization_params(args, config: Dict[str, Any]) -> Dict[str, Any]:
    cfg_target = int(config.get("target_height", 64))
    cfg_maxw = int(config.get("max_width", 1024))
    cfg_patch = int(config.get("patch_multiple", 16))
    cfg_buckets = config.get("width_buckets", [256, 384, 512, 768])

    target_height = int(args.target_height) if int(args.target_height) > 0 else cfg_target
    max_width = int(args.max_width) if int(args.max_width) > 0 else cfg_maxw
    patch_multiple = int(args.patch_multiple) if int(args.patch_multiple) > 0 else cfg_patch

    if (args.width_buckets or "").strip():
        bucket_raw = str(args.width_buckets)
    else:
        if isinstance(cfg_buckets, str):
            bucket_raw = cfg_buckets
        elif isinstance(cfg_buckets, (list, tuple)):
            bucket_raw = ",".join(str(v) for v in cfg_buckets)
        else:
            bucket_raw = "256,384,512,768"

    width_buckets = _parse_width_buckets(
        raw=bucket_raw,
        patch_multiple=int(patch_multiple),
        max_width=int(max_width),
    )
    return {
        "target_height": int(max(8, target_height)),
        "max_width": int(max(16, max_width)),
        "patch_multiple": int(max(1, patch_multiple)),
        "width_buckets": [int(v) for v in width_buckets],
    }


def _parse_recall_ks(raw: str) -> List[int]:
    vals: List[int] = []
    for tok in (raw or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = int(tok)
        except Exception:
            continue
        if v > 0:
            vals.append(v)
    out = sorted(set(vals))
    if not out:
        out = [1, 5, 10]
    return out


def _build_asset_records(groups: Sequence[HierarchyGroup]) -> Tuple[List[AssetRecord], Dict[int, str], Dict[int, List[int]]]:
    records: List[AssetRecord] = []
    gid_to_key: Dict[int, str] = {}
    gid_to_indices: Dict[int, List[int]] = {}
    for gid, group in enumerate(groups):
        gid_to_key[gid] = group.key
        gid_to_indices[gid] = []
        for asset in group.assets:
            idx = len(records)
            rec = AssetRecord(index=idx, group_key=group.key, group_id=gid, path=asset)
            records.append(rec)
            gid_to_indices[gid].append(idx)
    return records, gid_to_key, gid_to_indices


class EvalAssetDataset(Dataset):
    def __init__(
        self,
        records: Sequence[AssetRecord],
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
        rec = self.records[idx]
        with Image.open(rec.path) as im:
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


def _pad_right_to_width(tensor: torch.Tensor, target_width: int) -> torch.Tensor:
    width = int(tensor.shape[-1])
    if width >= target_width:
        return tensor[:, :, :target_width]
    pad = target_width - width
    return F.pad(tensor, (0, pad, 0, 0), mode="replicate")


def _collate_eval(batch: List[Dict[str, Any]], patch_multiple: int) -> Dict[str, Any]:
    if not batch:
        raise RuntimeError("Empty eval batch")
    pm = max(1, int(patch_multiple))
    widths = [int(row["pixel_values"].shape[-1]) for row in batch]
    max_w = max(widths)
    max_w = int(math.ceil(max_w / pm) * pm)
    pixels = torch.stack([_pad_right_to_width(row["pixel_values"], max_w) for row in batch], dim=0)
    indices = torch.tensor([int(row["index"]) for row in batch], dtype=torch.long)
    return {"index": indices, "pixel_values": pixels}


def _load_projection_head(
    projection_path: str,
    hidden_size: int,
) -> Optional[ProjectionHead]:
    if not (projection_path or "").strip():
        return None
    path = Path(projection_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        LOGGER.warning("Projection head not found: %s", path)
        return None

    payload = torch.load(str(path), map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload.get("state_dict", {})
        in_dim = int(payload.get("input_dim", hidden_size))
        out_dim = int(payload.get("output_dim", in_dim))
    else:
        state_dict = payload
        in_dim = int(hidden_size)
        out_dim = int(hidden_size)

    if in_dim <= 0:
        in_dim = int(hidden_size)
    if out_dim <= 0:
        out_dim = int(in_dim)

    head = ProjectionHead(in_dim=int(in_dim), out_dim=int(out_dim))
    head.load_state_dict(state_dict, strict=False)
    head.eval()
    return head


def _embed_assets(
    records: Sequence[AssetRecord],
    dataloader: DataLoader,
    backbone: AutoModel,
    projection_head: Optional[ProjectionHead],
    device: str,
    l2_normalize_embeddings: bool,
) -> np.ndarray:
    n = len(records)
    embeddings: Dict[int, np.ndarray] = {}
    backbone = backbone.to(device)
    backbone.eval()
    if projection_head is not None:
        projection_head = projection_head.to(device)
        projection_head.eval()

    pbar = tqdm(total=len(dataloader), desc="embed-assets")
    with torch.no_grad():
        for batch in dataloader:
            idxs = batch["index"].tolist()
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            emb = _pooled_image_embedding(backbone, pixels)
            if projection_head is not None:
                emb = projection_head(emb)
            if l2_normalize_embeddings:
                emb = F.normalize(emb, dim=-1)
            emb_cpu = emb.detach().cpu().float().numpy()
            for i, row in enumerate(idxs):
                embeddings[int(row)] = emb_cpu[i]
            pbar.update(1)
    pbar.close()

    if len(embeddings) != n:
        missing = [i for i in range(n) if i not in embeddings]
        raise RuntimeError(f"Missing embeddings for {len(missing)} assets.")
    ordered = np.stack([embeddings[i] for i in range(n)], axis=0).astype(np.float32, copy=False)
    return ordered


def _evaluate_retrieval(
    records: Sequence[AssetRecord],
    embeddings: np.ndarray,
    group_to_indices: Dict[int, List[int]],
    recall_ks: Sequence[int],
    min_positives_per_query: int,
    max_queries: int,
    seed: int,
) -> Dict[str, Any]:
    if embeddings.ndim != 2 or embeddings.shape[0] != len(records):
        raise RuntimeError("Embedding shape mismatch.")

    n = len(records)
    if n < 2:
        raise RuntimeError("Need at least 2 assets for retrieval evaluation.")

    max_k = int(max(recall_ks))
    bank = torch.from_numpy(embeddings).float()
    if torch.any(torch.isnan(bank)):
        raise RuntimeError("NaN embeddings detected.")

    # Use cosine retrieval for robust ranking.
    bank = F.normalize(bank, dim=-1)
    group_ids = [int(r.group_id) for r in records]

    candidates: List[int] = []
    for i, gid in enumerate(group_ids):
        positives = len(group_to_indices.get(int(gid), [])) - 1
        if positives >= int(min_positives_per_query):
            candidates.append(i)
    if not candidates:
        raise RuntimeError("No evaluable queries with available positives.")

    query_indices = list(candidates)
    if int(max_queries) > 0 and len(query_indices) > int(max_queries):
        rng = random.Random(int(seed))
        query_indices = sorted(rng.sample(query_indices, int(max_queries)))

    recall_hits: Dict[int, int] = {int(k): 0 for k in recall_ks}
    ranks: List[int] = []
    reciprocal_ranks: List[float] = []
    per_query_rows: List[Dict[str, Any]] = []

    pbar = tqdm(total=len(query_indices), desc="eval-retrieval")
    for qidx in query_indices:
        q_gid = int(group_ids[qidx])
        pos_indices = [i for i in group_to_indices.get(q_gid, []) if int(i) != int(qidx)]
        if not pos_indices:
            pbar.update(1)
            continue

        scores = torch.mv(bank, bank[qidx])
        scores[qidx] = -1e9

        pos_tensor = torch.tensor(pos_indices, dtype=torch.long)
        pos_scores = scores[pos_tensor]
        best_pos_score = torch.max(pos_scores)
        rank = int((scores > best_pos_score).sum().item()) + 1
        ranks.append(rank)
        reciprocal_ranks.append(1.0 / float(rank))

        topk_idx = torch.topk(scores, k=min(max_k, scores.numel()), largest=True).indices.tolist()
        for k in recall_ks:
            kk = int(k)
            hits = [j for j in topk_idx[:kk] if int(group_ids[j]) == q_gid and int(j) != int(qidx)]
            if hits:
                recall_hits[int(kk)] += 1

        top1 = int(topk_idx[0]) if topk_idx else -1
        per_query_rows.append(
            {
                "query_index": int(qidx),
                "query_group_id": int(q_gid),
                "query_group_key": records[qidx].group_key,
                "query_path": str(records[qidx].path),
                "positive_count": int(len(pos_indices)),
                "best_positive_rank": int(rank),
                "top1_index": int(top1),
                "top1_path": (str(records[top1].path) if top1 >= 0 else ""),
                "top1_is_positive": bool(top1 in pos_indices),
            }
        )
        pbar.update(1)
    pbar.close()

    if not ranks:
        raise RuntimeError("No ranks computed. Check query/positive filtering.")

    qn = len(ranks)
    recall_at = {str(k): float(recall_hits[int(k)]) / float(qn) for k in recall_ks}
    metrics = {
        "queries_evaluated": int(qn),
        "recall_at": recall_at,
        "mrr": float(sum(reciprocal_ranks) / float(qn)),
        "mean_rank": float(sum(ranks) / float(qn)),
        "median_rank": float(statistics.median(ranks)),
        "max_rank": int(max(ranks)),
        "min_rank": int(min(ranks)),
    }
    return {
        "metrics": metrics,
        "per_query_rows": per_query_rows,
        "query_candidates_total": int(len(candidates)),
        "query_indices_used": [int(i) for i in query_indices],
    }


def run(args) -> Dict[str, Any]:
    _configure_logging()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    backbone_dir = Path(args.backbone_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not backbone_dir.exists() or not backbone_dir.is_dir():
        raise FileNotFoundError(f"Backbone directory not found: {backbone_dir}")

    config = _load_eval_config(args.config_path, backbone_dir)
    norm = _resolve_normalization_params(args, config)
    recall_ks = _parse_recall_ks(args.recall_ks)

    projection_path = (args.projection_head_path or "").strip()
    if not projection_path:
        config_proj = str(config.get("projection_head_path", "") or "").strip()
        if config_proj:
            proj_candidate = Path(config_proj).expanduser().resolve()
            if proj_candidate.exists():
                projection_path = str(proj_candidate)

    groups = _discover_hierarchy_groups(
        dataset_dir=dataset_dir,
        include_line_images=bool(args.include_line_images),
        include_word_crops=bool(args.include_word_crops),
        include_number_crops=bool(args.include_number_crops),
        min_assets_per_group=max(1, int(args.min_assets_per_group)),
    )
    if not groups:
        raise RuntimeError(f"No groups found under {dataset_dir}")

    records, group_id_to_key, group_to_indices = _build_asset_records(groups)
    if not records:
        raise RuntimeError("No assets found for evaluation.")

    LOGGER.info(
        "Evaluation assets: groups=%d, assets=%d, include_line_images=%s, include_word_crops=%s, include_number_crops=%s",
        len(groups),
        len(records),
        bool(args.include_line_images),
        bool(args.include_word_crops),
        bool(args.include_number_crops),
    )
    LOGGER.info(
        "Normalization: target_height=%d, max_width=%d, patch_multiple=%d, width_buckets=%s",
        norm["target_height"],
        norm["max_width"],
        norm["patch_multiple"],
        norm["width_buckets"],
    )

    image_mean, image_std, image_stats_source = _load_image_stats(backbone_dir=backbone_dir, config=config)
    LOGGER.info("Image normalization stats source: %s (mean=%s std=%s)", image_stats_source, image_mean, image_std)

    dataset = EvalAssetDataset(
        records=records,
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
        collate_fn=lambda b: _collate_eval(b, patch_multiple=int(norm["patch_multiple"])),
    )
    if len(dataloader) == 0:
        raise RuntimeError("DataLoader has zero batches.")

    backbone = AutoModel.from_pretrained(str(backbone_dir))
    hidden_size = int(getattr(backbone.config, "hidden_size", 0))
    if hidden_size <= 0:
        raise RuntimeError("Backbone config has no hidden_size.")
    projection_head = _load_projection_head(projection_path, hidden_size=hidden_size)

    device = _resolve_device(args.device)
    LOGGER.info("Embedding device: %s", device)
    embeddings = _embed_assets(
        records=records,
        dataloader=dataloader,
        backbone=backbone,
        projection_head=projection_head,
        device=device,
        l2_normalize_embeddings=bool(args.l2_normalize_embeddings),
    )

    eval_result = _evaluate_retrieval(
        records=records,
        embeddings=embeddings,
        group_to_indices=group_to_indices,
        recall_ks=recall_ks,
        min_positives_per_query=max(1, int(args.min_positives_per_query)),
        max_queries=max(0, int(args.max_queries)),
        seed=int(args.seed),
    )

    report = {
        "task": "text_hierarchy_vit_retrieval_eval",
        "dataset_dir": str(dataset_dir),
        "backbone_dir": str(backbone_dir),
        "projection_head_path": str(projection_path) if projection_path else "",
        "config_path": str(config.get("_config_path", "")),
        "assets_total": int(len(records)),
        "groups_total": int(len(groups)),
        "group_keys": {str(k): str(v) for k, v in group_id_to_key.items()},
        "query_candidates_total": int(eval_result["query_candidates_total"]),
        "normalization": norm,
        "recall_ks": [int(v) for v in recall_ks],
        "l2_normalize_embeddings": bool(args.l2_normalize_embeddings),
        "min_positives_per_query": int(args.min_positives_per_query),
        "metrics": eval_result["metrics"],
        "embedding_dim": int(embeddings.shape[1]),
        "args": {
            "include_line_images": bool(args.include_line_images),
            "include_word_crops": bool(args.include_word_crops),
            "include_number_crops": bool(args.include_number_crops),
            "min_assets_per_group": int(args.min_assets_per_group),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "device": str(args.device),
            "max_queries": int(args.max_queries),
            "seed": int(args.seed),
        },
    }

    report_path = output_dir / "eval_text_hierarchy_vit_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    per_query_rows = eval_result["per_query_rows"]
    csv_path = output_dir / "eval_text_hierarchy_vit_per_query.csv"
    if bool(args.write_per_query_csv):
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "query_index",
                    "query_group_id",
                    "query_group_key",
                    "query_path",
                    "positive_count",
                    "best_positive_rank",
                    "top1_index",
                    "top1_path",
                    "top1_is_positive",
                ],
            )
            writer.writeheader()
            for row in per_query_rows:
                writer.writerow(row)

    metrics = report["metrics"]
    LOGGER.info("Queries evaluated: %d", int(metrics["queries_evaluated"]))
    LOGGER.info("Recall@K: %s", metrics["recall_at"])
    LOGGER.info("MRR: %.6f | mean_rank: %.2f | median_rank: %.2f", float(metrics["mrr"]), float(metrics["mean_rank"]), float(metrics["median_rank"]))
    LOGGER.info("Saved report: %s", report_path)
    if bool(args.write_per_query_csv):
        LOGGER.info("Saved per-query CSV: %s", csv_path)

    return {
        "report_path": str(report_path),
        "per_query_csv_path": (str(csv_path) if bool(args.write_per_query_csv) else ""),
        "metrics": metrics,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = create_eval_text_hierarchy_vit_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
