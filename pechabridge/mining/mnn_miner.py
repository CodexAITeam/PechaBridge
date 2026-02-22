"""Robust MNN mining for cross-page Pecha patch positives."""

from __future__ import annotations

import json
import logging
import math
import threading
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel

from .augment import JitterConfig, generate_augmented_views
from .faiss_utils import build_faiss_index, l2_normalize_rows, search_index, set_faiss_num_threads

LOGGER = logging.getLogger("mnn_miner")


@dataclass(frozen=True)
class PatchRecord:
    """One patch row from metadata."""

    patch_id: int
    doc_id: str
    page_id: str
    line_id: int
    scale_w: int
    k: int
    x0_norm: float
    x1_norm: float
    line_w_px: int
    line_h_px: int
    boundary_score: float
    ink_ratio: float
    image_path: Path

    @property
    def center(self) -> float:
        return float((self.x0_norm + self.x1_norm) * 0.5)


@dataclass
class ModelConfig:
    backbone: str = "facebook/dinov2-base"
    proj_dim: int = 256
    checkpoint: str = ""
    batch_size_embed: int = 128
    device: str = "cuda"


@dataclass
class IndexConfig:
    faiss_factory: str = "Flat"
    use_gpu: bool = False
    normalize: bool = True


@dataclass
class MiningConfig:
    topK: int = 200
    mutual_topK: int = 200
    max_pairs_per_src: int = 3
    sim_min: float = 0.25
    exclude_same_page: bool = True
    exclude_same_doc: bool = False
    exclude_same_line: bool = True
    exclude_nearby_lines: int = 1
    exclude_nearby_k: int = 0


@dataclass
class StabilityConfig:
    enabled: bool = True
    n_trials: int = 8
    require_ratio: float = 0.5
    jitter: JitterConfig = field(default_factory=JitterConfig)
    deterministic_seed: int = 123


@dataclass
class MultiScaleConfig:
    enabled: bool = True
    scales: List[int] = field(default_factory=lambda: [256, 384, 512])
    require_two_scales: bool = True
    center_eps: float = 0.06


@dataclass
class SignatureConfig:
    enabled: bool = False
    method: str = "hproj"
    corr_min: float = 0.3


@dataclass
class FilteringConfig:
    ink_ratio_min: float = 0.02
    boundary_score_min: float = 0.0
    signature: SignatureConfig = field(default_factory=SignatureConfig)


@dataclass
class PerformanceConfig:
    shard_by_doc: bool = True
    max_patches: Optional[int] = None
    num_workers: int = 8


@dataclass
class MNNConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    mining: MiningConfig = field(default_factory=MiningConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    multiscale: MultiScaleConfig = field(default_factory=MultiScaleConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "MNNConfig":
        p = dict(payload or {})
        model_p = dict(p.get("model") or {})
        index_p = dict(p.get("index") or {})
        mining_p = dict(p.get("mining") or {})
        stability_p = dict(p.get("stability") or {})
        jitter_p = dict(stability_p.get("jitter") or {})
        multiscale_p = dict(p.get("multiscale") or {})
        filtering_p = dict(p.get("filtering") or {})
        sig_p = dict(filtering_p.get("signature") or {})
        perf_p = dict(p.get("performance") or {})

        return MNNConfig(
            model=ModelConfig(
                backbone=str(model_p.get("backbone", "facebook/dinov2-base")),
                proj_dim=int(model_p.get("proj_dim", 256)),
                checkpoint=str(model_p.get("checkpoint", "")),
                batch_size_embed=int(model_p.get("batch_size_embed", 128)),
                device=str(model_p.get("device", "cuda")),
            ),
            index=IndexConfig(
                faiss_factory=str(index_p.get("faiss_factory", "Flat")),
                use_gpu=bool(index_p.get("use_gpu", False)),
                normalize=bool(index_p.get("normalize", True)),
            ),
            mining=MiningConfig(
                topK=int(mining_p.get("topK", 200)),
                mutual_topK=int(mining_p.get("mutual_topK", 200)),
                max_pairs_per_src=int(mining_p.get("max_pairs_per_src", 3)),
                sim_min=float(mining_p.get("sim_min", 0.25)),
                exclude_same_page=bool(mining_p.get("exclude_same_page", True)),
                exclude_same_doc=bool(mining_p.get("exclude_same_doc", False)),
                exclude_same_line=bool(mining_p.get("exclude_same_line", True)),
                exclude_nearby_lines=int(mining_p.get("exclude_nearby_lines", 1)),
                exclude_nearby_k=int(mining_p.get("exclude_nearby_k", 0)),
            ),
            stability=StabilityConfig(
                enabled=bool(stability_p.get("enabled", True)),
                n_trials=int(stability_p.get("n_trials", 8)),
                require_ratio=float(stability_p.get("require_ratio", 0.5)),
                jitter=JitterConfig(
                    translate_px=int(jitter_p.get("translate_px", 3)),
                    scale_range=(
                        float((jitter_p.get("scale_range") or [0.98, 1.02])[0]),
                        float((jitter_p.get("scale_range") or [0.98, 1.02])[1]),
                    ),
                    brightness=float(jitter_p.get("brightness", 0.05)),
                    contrast=float(jitter_p.get("contrast", 0.05)),
                    blur_sigma=float(jitter_p.get("blur_sigma", 0.3)),
                ),
                deterministic_seed=int(stability_p.get("deterministic_seed", 123)),
            ),
            multiscale=MultiScaleConfig(
                enabled=bool(multiscale_p.get("enabled", True)),
                scales=[int(v) for v in (multiscale_p.get("scales") or [256, 384, 512])],
                require_two_scales=bool(multiscale_p.get("require_two_scales", True)),
                center_eps=float(multiscale_p.get("center_eps", 0.06)),
            ),
            filtering=FilteringConfig(
                ink_ratio_min=float(filtering_p.get("ink_ratio_min", 0.02)),
                boundary_score_min=float(filtering_p.get("boundary_score_min", 0.0)),
                signature=SignatureConfig(
                    enabled=bool(sig_p.get("enabled", False)),
                    method=str(sig_p.get("method", "hproj")),
                    corr_min=float(sig_p.get("corr_min", 0.3)),
                ),
            ),
            performance=PerformanceConfig(
                shard_by_doc=bool(perf_p.get("shard_by_doc", True)),
                max_patches=(int(perf_p.get("max_patches")) if perf_p.get("max_patches") is not None else None),
                num_workers=int(perf_p.get("num_workers", 8)),
            ),
        )


@dataclass
class ScaleState:
    scale_w: int
    records: List[PatchRecord]
    embeddings: np.ndarray
    index: Any
    index_patch_ids: np.ndarray
    patch_id_to_index: Dict[int, int]


@dataclass
class PairCandidate:
    src: PatchRecord
    dst: PatchRecord
    sim: float
    rank_src_to_dst: int
    rank_dst_to_src: int
    stability_count: int
    stability_ratio: float
    multi_scale_ok: bool
    notes: str


class ProjectionHead(nn.Module):
    """Projection head compatible with training checkpoints."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        hidden_dim = max(int(out_dim) * 2, int(in_dim) // 2)
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchPathDataset(Dataset):
    """Dataset of patch image paths used for batched embedding."""

    def __init__(self, records: Sequence[PatchRecord]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[int, str]:
        rec = self.records[int(idx)]
        return int(idx), str(rec.image_path)


def _normalize_id(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _sanitize_id(value: Any) -> str:
    txt = _normalize_id(value)
    if not txt:
        return "unknown"
    out: List[str] = []
    for ch in txt:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "unknown"


def _resolve_patch_path(dataset_dir: Path, row: Mapping[str, Any]) -> Optional[Path]:
    candidates = [str(row.get("patch_image_path", "") or "").strip(), str(row.get("patch_path", "") or "").strip()]
    for raw in candidates:
        if not raw:
            continue
        p = Path(raw).expanduser()
        resolved = p.resolve() if p.is_absolute() else (dataset_dir / p).resolve()
        if resolved.exists() and resolved.is_file():
            return resolved

    try:
        patch_id = int(row.get("patch_id", -1))
        line_id = int(row.get("line_id", -1))
        scale_w = int(row.get("scale_w", -1))
        doc = _sanitize_id(row.get("doc_id", ""))
        page = _sanitize_id(row.get("page_id", ""))
    except Exception:
        return None
    if patch_id < 0 or line_id < 0 or scale_w <= 0:
        return None
    constructed = (
        dataset_dir
        / "patches"
        / f"doc={doc}"
        / f"page={page}"
        / f"line={line_id}"
        / f"scale={scale_w}"
        / f"patch_{patch_id}.png"
    ).resolve()
    if constructed.exists() and constructed.is_file():
        return constructed
    return None


def is_excluded_pair(src: PatchRecord, dst: PatchRecord, cfg: MiningConfig) -> bool:
    """Pure exclusion rule predicate."""
    if int(src.patch_id) == int(dst.patch_id):
        return True

    same_doc = src.doc_id == dst.doc_id
    same_page = same_doc and src.page_id == dst.page_id
    same_line = same_page and int(src.line_id) == int(dst.line_id)

    if bool(cfg.exclude_same_page) and same_page:
        return True
    if bool(cfg.exclude_same_doc) and same_doc:
        return True
    if bool(cfg.exclude_same_line) and same_line:
        return True

    if int(cfg.exclude_nearby_lines) > 0 and same_page:
        if abs(int(src.line_id) - int(dst.line_id)) <= int(cfg.exclude_nearby_lines):
            return True

    if int(cfg.exclude_nearby_k) > 0 and same_line:
        if abs(int(src.k) - int(dst.k)) <= int(cfg.exclude_nearby_k):
            return True
    return False


def find_mutual_ranks(
    src_patch_id: int,
    dst_patch_id: int,
    src_to_candidates: Sequence[Tuple[int, float, int]],
    dst_to_candidates: Sequence[Tuple[int, float, int]],
    mutual_topk: int,
) -> Tuple[bool, int, int]:
    """
    Check mutual nearest-neighbor condition with filtered lists.

    Returns (is_mutual, rank_src_to_dst, rank_dst_to_src). Ranks are 1-based.
    """
    max_rank = max(1, int(mutual_topk))
    r_sd = -1
    r_ds = -1
    for pid, _sim, rank in src_to_candidates:
        if int(rank) > max_rank:
            break
        if int(pid) == int(dst_patch_id):
            r_sd = int(rank)
            break
    if r_sd < 0:
        return False, -1, -1
    for pid, _sim, rank in dst_to_candidates:
        if int(rank) > max_rank:
            break
        if int(pid) == int(src_patch_id):
            r_ds = int(rank)
            break
    if r_ds < 0:
        return False, -1, -1
    return True, int(r_sd), int(r_ds)


def select_center_match(records: Sequence[PatchRecord], center: float, center_eps: float) -> Optional[PatchRecord]:
    """Select record with nearest center if distance <= eps."""
    if not records:
        return None
    c = float(center)
    eps = float(max(0.0, center_eps))
    best: Optional[PatchRecord] = None
    best_dist = 1e9
    for rec in records:
        d = abs(float(rec.center) - c)
        if d < best_dist:
            best = rec
            best_dist = d
    if best is None:
        return None
    if best_dist > eps:
        return None
    return best


def _signature_hproj(image: Image.Image, bins: int = 64) -> np.ndarray:
    gray = np.asarray(image.convert("L")).astype(np.float32) / 255.0
    ink = 1.0 - gray
    prof = np.sum(ink, axis=0).astype(np.float32)
    if prof.size <= 1:
        return np.zeros((bins,), dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, prof.size, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, int(bins), dtype=np.float32)
    vec = np.interp(x_new, x_old, prof).astype(np.float32)
    vec = vec - float(np.mean(vec))
    std = float(np.std(vec))
    if std > 1e-6:
        vec = vec / std
    return vec


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float32).reshape(-1)
    y = np.asarray(b, dtype=np.float32).reshape(-1)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return -1.0
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return -1.0
    return float(np.dot(x, y) / denom)


def _resolve_device(pref: str) -> str:
    p = str(pref or "").strip().lower()
    if not p or p == "auto":
        if torch.cuda.is_available():
            return "cuda"
        try:
            if bool(torch.backends.mps.is_available()):  # type: ignore[attr-defined]
                return "mps"
        except Exception:
            pass
        return "cpu"
    if p.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    if p == "mps":
        try:
            if bool(torch.backends.mps.is_available()):  # type: ignore[attr-defined]
                return "mps"
        except Exception:
            pass
        return "cpu"
    return p


def _pooled_embedding(backbone: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
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
        raise RuntimeError("Backbone outputs have neither pooler_output nor last_hidden_state.")
    if hidden.ndim == 3:
        return hidden[:, 0]
    return hidden


class MNNMiner:
    """Cross-page robust mutual-nearest-neighbor miner."""

    def __init__(self, config: MNNConfig):
        self.cfg = config
        self.device = _resolve_device(config.model.device)
        faiss_threads = max(0, int(getattr(config.performance, "num_workers", 0)))
        faiss_thread_applied = set_faiss_num_threads(faiss_threads if faiss_threads > 0 else None)
        if faiss_thread_applied:
            LOGGER.info("Configured FAISS/OpenMP threads: %d", faiss_threads)
        else:
            LOGGER.info("FAISS/OpenMP thread config unchanged (requested=%s)", str(faiss_threads))
        self.image_processor = AutoImageProcessor.from_pretrained(str(config.model.backbone))
        self.backbone = AutoModel.from_pretrained(str(config.model.backbone)).to(self.device).eval()
        self.projection_head: Optional[ProjectionHead] = None
        self._load_projection_head_if_available()

        self._records_by_id: Dict[int, PatchRecord] = {}
        self._line_scale_index: Dict[Tuple[str, str, int, int], List[PatchRecord]] = {}
        self._scale_states: Dict[int, ScaleState] = {}
        self._retrieval_cache: "OrderedDict[Tuple[int, int, int], List[Tuple[int, float, int]]]" = OrderedDict()
        self._retrieval_cache_size = 200000
        self._signature_cache: Dict[int, np.ndarray] = {}
        self._retrieval_cache_lock = threading.Lock()
        self._signature_cache_lock = threading.Lock()
        self._embed_model_lock = threading.Lock()

    def _load_projection_head_if_available(self) -> None:
        ckpt = str(self.cfg.model.checkpoint or "").strip()
        if not ckpt:
            self.projection_head = None
            return
        path = Path(ckpt).expanduser().resolve()
        if not path.exists() or not path.is_file():
            LOGGER.warning("Projection checkpoint not found: %s", path)
            self.projection_head = None
            return
        payload = torch.load(str(path), map_location="cpu")
        if not isinstance(payload, dict):
            LOGGER.warning("Projection checkpoint format unsupported: %s", path)
            self.projection_head = None
            return
        if "state_dict" in payload:
            state_dict = payload.get("state_dict", {})
            in_dim = int(payload.get("input_dim", 0))
            out_dim = int(payload.get("output_dim", 0))
        else:
            state_dict = payload
            in_dim = int(payload.get("input_dim", 0))
            out_dim = int(payload.get("output_dim", 0))
        if in_dim <= 0:
            in_dim = int(getattr(self.backbone.config, "hidden_size", 0) or 0)
        if out_dim <= 0:
            out_dim = int(self.cfg.model.proj_dim)
        if in_dim <= 0 or out_dim <= 0:
            LOGGER.warning("Projection dims invalid in checkpoint: %s", path)
            self.projection_head = None
            return
        head = ProjectionHead(in_dim=in_dim, out_dim=out_dim)
        head.load_state_dict(state_dict, strict=False)
        self.projection_head = head.to(self.device).eval()

    def _collate_embed_batch(self, batch: List[Tuple[int, str]]) -> Dict[str, Any]:
        idxs: List[int] = []
        imgs: List[Image.Image] = []
        for idx, p in batch:
            idxs.append(int(idx))
            with Image.open(p) as im:
                imgs.append(im.convert("RGB"))
        enc = self.image_processor(images=imgs, return_tensors="pt")
        return {"index": torch.tensor(idxs, dtype=torch.long), "pixel_values": enc["pixel_values"]}

    def _embed_records(self, records: Sequence[PatchRecord]) -> np.ndarray:
        ds = PatchPathDataset(records)
        loader = DataLoader(
            ds,
            batch_size=max(1, int(self.cfg.model.batch_size_embed)),
            shuffle=False,
            num_workers=max(0, int(self.cfg.performance.num_workers)),
            pin_memory=True,
            drop_last=False,
            collate_fn=self._collate_embed_batch,
        )
        out: Dict[int, np.ndarray] = {}
        with torch.no_grad():
            for batch in tqdm(loader, desc="embed-patches", leave=False):
                bidx = batch["index"].tolist()
                pix = batch["pixel_values"].to(self.device, non_blocking=True)
                emb = _pooled_embedding(self.backbone, pix)
                if self.projection_head is not None:
                    emb = self.projection_head(emb)
                emb = emb.detach().float()
                if bool(self.cfg.index.normalize):
                    emb = F.normalize(emb, dim=-1)
                arr = emb.cpu().numpy().astype(np.float32, copy=False)
                for i, ridx in enumerate(bidx):
                    out[int(ridx)] = arr[i]
        if len(out) != len(records):
            missing = [i for i in range(len(records)) if i not in out]
            raise RuntimeError(f"Missing embeddings for {len(missing)} records.")
        mat = np.stack([out[i] for i in range(len(records))], axis=0).astype(np.float32, copy=False)
        if bool(self.cfg.index.normalize):
            mat = l2_normalize_rows(mat)
        return mat

    def _embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, 1), dtype=np.float32)
        # Keep threaded mining safe when stability checks call model forward.
        with self._embed_model_lock:
            enc = self.image_processor(images=[im.convert("RGB") for im in images], return_tensors="pt")
            pix = enc["pixel_values"].to(self.device)
            with torch.no_grad():
                emb = _pooled_embedding(self.backbone, pix)
                if self.projection_head is not None:
                    emb = self.projection_head(emb)
                emb = emb.detach().float()
                if bool(self.cfg.index.normalize):
                    emb = F.normalize(emb, dim=-1)
        arr = emb.cpu().numpy().astype(np.float32, copy=False)
        if bool(self.cfg.index.normalize):
            arr = l2_normalize_rows(arr)
        return arr

    def _cache_get(self, key: Tuple[int, int, int]) -> Optional[List[Tuple[int, float, int]]]:
        with self._retrieval_cache_lock:
            val = self._retrieval_cache.get(key)
            if val is None:
                return None
            self._retrieval_cache.move_to_end(key, last=True)
            return val

    def _cache_put(self, key: Tuple[int, int, int], value: List[Tuple[int, float, int]]) -> None:
        with self._retrieval_cache_lock:
            self._retrieval_cache[key] = value
            self._retrieval_cache.move_to_end(key, last=True)
            while len(self._retrieval_cache) > self._retrieval_cache_size:
                self._retrieval_cache.popitem(last=False)

    def _filtered_from_search(
        self,
        *,
        src: PatchRecord,
        scale_state: ScaleState,
        sims: np.ndarray,
        idxs: np.ndarray,
        topk: int,
    ) -> List[Tuple[int, float, int]]:
        out: List[Tuple[int, float, int]] = []
        for raw_rank, (sim, idx) in enumerate(zip(sims.tolist(), idxs.tolist()), start=1):
            ii = int(idx)
            if ii < 0 or ii >= len(scale_state.records):
                continue
            dst_pid = int(scale_state.index_patch_ids[ii])
            dst = self._records_by_id.get(dst_pid)
            if dst is None:
                continue
            if float(sim) < float(self.cfg.mining.sim_min):
                continue
            if is_excluded_pair(src, dst, self.cfg.mining):
                continue
            out.append((dst_pid, float(sim), len(out) + 1))
            if len(out) >= int(topk):
                break
        return out

    def _retrieve_filtered(self, scale_w: int, src_patch_id: int, topk: int) -> List[Tuple[int, float, int]]:
        key = (int(scale_w), int(src_patch_id), int(topk))
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        st = self._scale_states[int(scale_w)]
        src_idx = st.patch_id_to_index.get(int(src_patch_id))
        if src_idx is None:
            return []
        src = st.records[int(src_idx)]

        total = len(st.records)
        if total <= 1:
            return []
        need = max(1, int(topk))
        k_search = min(total, max(need * 4, need + 64))
        out: List[Tuple[int, float, int]] = []
        while True:
            d, i = search_index(st.index, st.embeddings[src_idx : src_idx + 1], topk=k_search)
            out = self._filtered_from_search(src=src, scale_state=st, sims=d[0], idxs=i[0], topk=need)
            if len(out) >= need or k_search >= total:
                break
            k_search = min(total, max(k_search + 64, k_search * 2))
        self._cache_put(key, out)
        return out

    def _retrieve_filtered_query_vec(
        self,
        *,
        scale_w: int,
        src: PatchRecord,
        query_vec: np.ndarray,
        topk: int,
    ) -> List[Tuple[int, float, int]]:
        st = self._scale_states[int(scale_w)]
        total = len(st.records)
        if total <= 1:
            return []
        q = np.ascontiguousarray(np.asarray(query_vec, dtype=np.float32).reshape(1, -1))
        need = max(1, int(topk))
        k_search = min(total, max(need * 4, need + 64))
        out: List[Tuple[int, float, int]] = []
        while True:
            d, i = search_index(st.index, q, topk=k_search)
            out = self._filtered_from_search(src=src, scale_state=st, sims=d[0], idxs=i[0], topk=need)
            if len(out) >= need or k_search >= total:
                break
            k_search = min(total, max(k_search + 64, k_search * 2))
        return out

    def _get_signature(self, rec: PatchRecord) -> np.ndarray:
        key = int(rec.patch_id)
        with self._signature_cache_lock:
            sig = self._signature_cache.get(key)
            if sig is not None:
                return sig
        with Image.open(rec.image_path) as im:
            sig = _signature_hproj(im.convert("RGB"))
        with self._signature_cache_lock:
            self._signature_cache[key] = sig
        return sig

    def _stability_for_pair(self, src: PatchRecord, dst: PatchRecord) -> Tuple[int, float]:
        cfg = self.cfg.stability
        if not bool(cfg.enabled):
            return 0, 1.0
        n_trials = max(1, int(cfg.n_trials))
        with Image.open(src.image_path) as im:
            views = generate_augmented_views(
                im.convert("RGB"),
                patch_id=int(src.patch_id),
                base_seed=int(cfg.deterministic_seed),
                n_trials=n_trials,
                jitter=cfg.jitter,
            )
        emb = self._embed_images(views)
        hits = 0
        for t in range(emb.shape[0]):
            cand = self._retrieve_filtered_query_vec(
                scale_w=int(src.scale_w),
                src=src,
                query_vec=emb[t],
                topk=int(self.cfg.mining.topK),
            )
            ids = {int(pid) for pid, _s, _r in cand}
            if int(dst.patch_id) in ids:
                hits += 1
        ratio = float(hits) / float(max(1, n_trials))
        return int(hits), float(ratio)

    def _corresponding_patch(self, rec: PatchRecord, target_scale: int) -> Optional[PatchRecord]:
        key = (str(rec.doc_id), str(rec.page_id), int(rec.line_id), int(target_scale))
        pool = self._line_scale_index.get(key, [])
        return select_center_match(pool, center=float(rec.center), center_eps=float(self.cfg.multiscale.center_eps))

    def _multiscale_ok(self, src: PatchRecord, dst: PatchRecord) -> bool:
        cfg = self.cfg.multiscale
        if not bool(cfg.enabled):
            return True

        scales = [int(s) for s in (cfg.scales or sorted(self._scale_states.keys()))]
        checks = 0
        for sw in scales:
            if sw == int(src.scale_w):
                continue
            if sw not in self._scale_states:
                continue
            src2 = self._corresponding_patch(src, target_scale=sw)
            dst2 = self._corresponding_patch(dst, target_scale=sw)
            if src2 is None or dst2 is None:
                continue
            list_s = self._retrieve_filtered(sw, int(src2.patch_id), int(self.cfg.mining.topK))
            list_d = self._retrieve_filtered(sw, int(dst2.patch_id), int(self.cfg.mining.mutual_topK))
            ok, _r1, _r2 = find_mutual_ranks(
                src_patch_id=int(src2.patch_id),
                dst_patch_id=int(dst2.patch_id),
                src_to_candidates=list_s,
                dst_to_candidates=list_d,
                mutual_topk=int(self.cfg.mining.mutual_topK),
            )
            if ok:
                checks += 1
                if not bool(cfg.require_two_scales):
                    return True
        if bool(cfg.require_two_scales):
            return checks >= 1
        return checks > 0

    def _mine_one_source(self, *, scale_w: int, src: PatchRecord) -> List[PairCandidate]:
        src_list = self._retrieve_filtered(
            scale_w=int(scale_w),
            src_patch_id=int(src.patch_id),
            topk=int(self.cfg.mining.topK),
        )
        if not src_list:
            return []

        cand_src: List[PairCandidate] = []
        used_dst: set[int] = set()
        for dst_id, sim, rank_sd in src_list:
            if int(dst_id) in used_dst:
                continue
            dst = self._records_by_id.get(int(dst_id))
            if dst is None:
                continue
            dst_list = self._retrieve_filtered(
                scale_w=int(scale_w),
                src_patch_id=int(dst.patch_id),
                topk=int(self.cfg.mining.mutual_topK),
            )
            ok_mut, _r_sd, rank_ds = find_mutual_ranks(
                src_patch_id=int(src.patch_id),
                dst_patch_id=int(dst.patch_id),
                src_to_candidates=src_list,
                dst_to_candidates=dst_list,
                mutual_topk=int(self.cfg.mining.mutual_topK),
            )
            if not ok_mut:
                continue

            notes: List[str] = ["mutual"]
            st_count = 0
            st_ratio = 1.0
            if bool(self.cfg.stability.enabled):
                st_count, st_ratio = self._stability_for_pair(src, dst)
                notes.append(f"stability={st_count}/{self.cfg.stability.n_trials}")
                if st_ratio < float(self.cfg.stability.require_ratio):
                    continue

            multi_ok = self._multiscale_ok(src, dst) if bool(self.cfg.multiscale.enabled) else True
            notes.append(f"multiscale={int(bool(multi_ok))}")
            if bool(self.cfg.multiscale.enabled) and bool(self.cfg.multiscale.require_two_scales) and (not multi_ok):
                continue

            if bool(self.cfg.filtering.signature.enabled):
                sig_a = self._get_signature(src)
                sig_b = self._get_signature(dst)
                c = _corr(sig_a, sig_b)
                notes.append(f"sigcorr={c:.3f}")
                if c < float(self.cfg.filtering.signature.corr_min):
                    continue

            cand_src.append(
                PairCandidate(
                    src=src,
                    dst=dst,
                    sim=float(sim),
                    rank_src_to_dst=int(rank_sd),
                    rank_dst_to_src=int(rank_ds),
                    stability_count=int(st_count),
                    stability_ratio=float(st_ratio),
                    multi_scale_ok=bool(multi_ok),
                    notes=";".join(notes),
                )
            )
            used_dst.add(int(dst.patch_id))

        cand_src.sort(
            key=lambda x: (
                float(x.stability_ratio),
                float(x.sim),
                1 if x.multi_scale_ok else 0,
            ),
            reverse=True,
        )
        return cand_src[: max(1, int(self.cfg.mining.max_pairs_per_src))]

    def _mine_scale(self, scale_w: int) -> List[PairCandidate]:
        st = self._scale_states[int(scale_w)]
        out: List[PairCandidate] = []

        src_records = list(st.records)
        if bool(self.cfg.performance.shard_by_doc):
            src_records.sort(key=lambda r: (r.doc_id, r.page_id, r.line_id, r.patch_id))
        else:
            src_records.sort(key=lambda r: r.patch_id)

        max_workers = max(1, int(getattr(self.cfg.performance, "num_workers", 1)))
        if max_workers <= 1:
            pbar = tqdm(total=len(src_records), desc=f"mine-mnn-scale-{scale_w}", leave=False)
            for src in src_records:
                out.extend(self._mine_one_source(scale_w=int(scale_w), src=src))
                pbar.update(1)
            pbar.close()
            return out

        workers = min(max_workers, max(1, len(src_records)))
        max_inflight = max(int(workers) * 4, int(workers))
        LOGGER.info(
            "Mining scale=%d with parallel source loop: workers=%d inflight=%d sources=%d",
            int(scale_w),
            int(workers),
            int(max_inflight),
            len(src_records),
        )

        by_src_id: Dict[int, List[PairCandidate]] = {}
        pending: Dict[Future[List[PairCandidate]], PatchRecord] = {}
        src_iter = iter(src_records)
        pbar = tqdm(total=len(src_records), desc=f"mine-mnn-scale-{scale_w}", leave=False)
        with ThreadPoolExecutor(max_workers=int(workers), thread_name_prefix=f"mnn{int(scale_w)}") as ex:
            while len(pending) < max_inflight:
                try:
                    src = next(src_iter)
                except StopIteration:
                    break
                pending[ex.submit(self._mine_one_source, scale_w=int(scale_w), src=src)] = src

            while pending:
                done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    src = pending.pop(fut)
                    try:
                        by_src_id[int(src.patch_id)] = fut.result()
                    except Exception:
                        LOGGER.exception("Mining failed for scale=%d src_patch_id=%d", int(scale_w), int(src.patch_id))
                        pbar.close()
                        raise
                    pbar.update(1)

                    try:
                        nxt = next(src_iter)
                    except StopIteration:
                        continue
                    pending[ex.submit(self._mine_one_source, scale_w=int(scale_w), src=nxt)] = nxt
        pbar.close()

        # Restore stable source order independent of task completion order.
        for src in src_records:
            out.extend(by_src_id.get(int(src.patch_id), []))
        return out

    def _build_debug_grid(
        self,
        pair: PairCandidate,
        out_path: Path,
        topk_preview: int = 5,
    ) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cells: List[Tuple[Image.Image, str]] = []

        with Image.open(pair.src.image_path) as im:
            cells.append((im.convert("RGB"), f"SRC {pair.src.patch_id}"))
        with Image.open(pair.dst.image_path) as im:
            cells.append((im.convert("RGB"), f"DST {pair.dst.patch_id} sim={pair.sim:.3f}"))

        src_list = self._retrieve_filtered(
            scale_w=int(pair.src.scale_w),
            src_patch_id=int(pair.src.patch_id),
            topk=max(int(topk_preview), int(self.cfg.mining.topK)),
        )[: max(1, int(topk_preview))]
        for pid, sim, rank in src_list:
            rec = self._records_by_id.get(int(pid))
            if rec is None:
                continue
            with Image.open(rec.image_path) as im:
                cells.append((im.convert("RGB"), f"R{rank} {pid} {sim:.3f}"))

        thumb_h = 112
        margin = 6
        rendered: List[Image.Image] = []
        labels: List[str] = []
        for img, lbl in cells:
            w, h = img.size
            tw = max(32, int(round(float(w) * (float(thumb_h) / float(max(1, h))))))
            resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
            rend = img.resize((tw, thumb_h), resample=resample)
            rendered.append(rend)
            labels.append(lbl)

        total_w = sum(im.size[0] for im in rendered) + margin * (len(rendered) + 1)
        total_h = thumb_h + 30
        canvas = Image.new("RGB", (total_w, total_h), (245, 245, 245))
        draw = ImageDraw.Draw(canvas)

        x = margin
        for im, lbl in zip(rendered, labels):
            canvas.paste(im, (x, 4))
            draw.text((x, thumb_h + 8), lbl, fill=(20, 20, 20))
            x += im.size[0] + margin
        canvas.save(out_path)

    def _debug_dump_pairs(self, pairs: Sequence[PairCandidate], dataset_dir: Path, n: int) -> None:
        count = max(0, int(n))
        if count <= 0 or not pairs:
            return
        rng = np.random.default_rng(int(self.cfg.stability.deterministic_seed))
        ids = np.arange(len(pairs))
        rng.shuffle(ids)
        take = ids[: min(count, len(pairs))]
        out_dir = (dataset_dir / "debug" / "mnn_pairs").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, pi in enumerate(take.tolist(), start=1):
            p = pairs[int(pi)]
            fname = f"pair_{idx:04d}_src={p.src.patch_id}_dst={p.dst.patch_id}.png"
            self._build_debug_grid(p, out_path=out_dir / fname)

    def _load_records(self, dataset_dir: Path, meta_path: Path) -> List[PatchRecord]:
        if not meta_path.exists() or not meta_path.is_file():
            raise FileNotFoundError(f"Metadata parquet not found: {meta_path}")
        df = pd.read_parquet(meta_path)
        if df is None or df.empty:
            raise RuntimeError(f"No rows in metadata: {meta_path}")

        out: List[PatchRecord] = []
        dropped_missing = 0
        for row in df.to_dict(orient="records"):
            try:
                patch_id = int(row.get("patch_id"))
                doc_id = _normalize_id(row.get("doc_id"))
                page_id = _normalize_id(row.get("page_id"))
                line_id = int(row.get("line_id", -1))
                scale_w = int(row.get("scale_w", -1))
                k = int(row.get("k", -1))
                x0 = float(row.get("x0_norm", 0.0))
                x1 = float(row.get("x1_norm", 0.0))
                lw = int(row.get("line_w_px", 0))
                lh = int(row.get("line_h_px", 0))
                bs = float(row.get("boundary_score", 0.0))
                ir = float(row.get("ink_ratio", 0.0))
            except Exception:
                continue
            if scale_w <= 0 or line_id < 0:
                continue
            if ir < float(self.cfg.filtering.ink_ratio_min):
                continue
            if bs < float(self.cfg.filtering.boundary_score_min):
                continue
            p = _resolve_patch_path(dataset_dir, row)
            if p is None:
                dropped_missing += 1
                continue
            out.append(
                PatchRecord(
                    patch_id=patch_id,
                    doc_id=doc_id,
                    page_id=page_id,
                    line_id=line_id,
                    scale_w=scale_w,
                    k=k,
                    x0_norm=x0,
                    x1_norm=x1,
                    line_w_px=lw,
                    line_h_px=lh,
                    boundary_score=bs,
                    ink_ratio=ir,
                    image_path=p,
                )
            )
        if dropped_missing > 0:
            LOGGER.warning("Dropped %d patches due to missing image path.", dropped_missing)

        # Stable dedupe by patch_id.
        by_id: Dict[int, PatchRecord] = {}
        for rec in sorted(out, key=lambda r: r.patch_id):
            by_id[int(rec.patch_id)] = rec
        records = list(by_id.values())

        max_patches = self.cfg.performance.max_patches
        if max_patches is not None and int(max_patches) > 0 and len(records) > int(max_patches):
            rng = np.random.default_rng(int(self.cfg.stability.deterministic_seed))
            idx = np.arange(len(records))
            rng.shuffle(idx)
            keep = sorted(int(i) for i in idx[: int(max_patches)].tolist())
            records = [records[i] for i in keep]
            LOGGER.info("max_patches=%d -> sampled %d records.", int(max_patches), len(records))
        return records

    def run(
        self,
        *,
        dataset_dir: Path,
        meta_path: Path,
        out_pairs_path: Path,
        debug_dump: int = 0,
    ) -> Dict[str, Any]:
        dataset_dir = Path(dataset_dir).expanduser().resolve()
        meta_path = Path(meta_path).expanduser().resolve()
        out_pairs_path = Path(out_pairs_path).expanduser().resolve()
        out_pairs_path.parent.mkdir(parents=True, exist_ok=True)

        records = self._load_records(dataset_dir=dataset_dir, meta_path=meta_path)
        if not records:
            raise RuntimeError("No valid patch records after filtering.")
        self._records_by_id = {int(r.patch_id): r for r in records}

        # Build line+scale lookup for cross-scale center matching.
        lsi: Dict[Tuple[str, str, int, int], List[PatchRecord]] = defaultdict(list)
        for rec in records:
            lsi[(rec.doc_id, rec.page_id, int(rec.line_id), int(rec.scale_w))].append(rec)
        for k in list(lsi.keys()):
            lsi[k] = sorted(lsi[k], key=lambda r: (r.center, r.patch_id))
        self._line_scale_index = dict(lsi)

        # Embed and index per scale.
        scales_present = sorted({int(r.scale_w) for r in records})
        if self.cfg.multiscale.scales:
            wanted_scales = sorted({int(v) for v in self.cfg.multiscale.scales})
        else:
            wanted_scales = scales_present
        scales = [s for s in scales_present if s in set(wanted_scales)] or scales_present

        for sw in scales:
            subset = [r for r in records if int(r.scale_w) == int(sw)]
            if len(subset) < 2:
                continue
            LOGGER.info("Embedding scale=%d patches=%d", sw, len(subset))
            emb = self._embed_records(subset)
            idx = build_faiss_index(
                emb,
                faiss_factory=str(self.cfg.index.faiss_factory),
                use_gpu=bool(self.cfg.index.use_gpu),
            )
            patch_ids = np.asarray([int(r.patch_id) for r in subset], dtype=np.int64)
            patch_to_idx = {int(pid): i for i, pid in enumerate(patch_ids.tolist())}
            self._scale_states[int(sw)] = ScaleState(
                scale_w=int(sw),
                records=subset,
                embeddings=emb,
                index=idx,
                index_patch_ids=patch_ids,
                patch_id_to_index=patch_to_idx,
            )
        if not self._scale_states:
            raise RuntimeError("No scale states built. Check filters / metadata.")

        # Mine pairs.
        all_pairs: List[PairCandidate] = []
        for sw in sorted(self._scale_states.keys()):
            mined = self._mine_scale(scale_w=int(sw))
            all_pairs.extend(mined)
            LOGGER.info("Scale=%d mined_pairs=%d", sw, len(mined))

        # Keep top N per src across all scales (defensive).
        grouped: Dict[int, List[PairCandidate]] = defaultdict(list)
        for p in all_pairs:
            grouped[int(p.src.patch_id)].append(p)
        final_pairs: List[PairCandidate] = []
        for src_id in sorted(grouped.keys()):
            rows = grouped[src_id]
            rows.sort(
                key=lambda x: (
                    float(x.stability_ratio),
                    float(x.sim),
                    1 if x.multi_scale_ok else 0,
                ),
                reverse=True,
            )
            final_pairs.extend(rows[: max(1, int(self.cfg.mining.max_pairs_per_src))])

        out_rows: List[Dict[str, Any]] = []
        for p in final_pairs:
            out_rows.append(
                {
                    "src_patch_id": int(p.src.patch_id),
                    "dst_patch_id": int(p.dst.patch_id),
                    "src_doc_id": str(p.src.doc_id),
                    "src_page_id": str(p.src.page_id),
                    "src_line_id": int(p.src.line_id),
                    "src_scale_w": int(p.src.scale_w),
                    "dst_doc_id": str(p.dst.doc_id),
                    "dst_page_id": str(p.dst.page_id),
                    "dst_line_id": int(p.dst.line_id),
                    "dst_scale_w": int(p.dst.scale_w),
                    "sim": float(p.sim),
                    "rank_src_to_dst": int(p.rank_src_to_dst),
                    "rank_dst_to_src": int(p.rank_dst_to_src),
                    "stability_count": int(p.stability_count),
                    "stability_ratio": float(p.stability_ratio),
                    "multi_scale_ok": bool(p.multi_scale_ok),
                    "notes": str(p.notes),
                }
            )
        out_df = pd.DataFrame(out_rows)
        if not out_df.empty:
            out_df = out_df.astype(
                {
                    "src_patch_id": "int64",
                    "dst_patch_id": "int64",
                    "src_doc_id": "string",
                    "src_page_id": "string",
                    "src_line_id": "int32",
                    "src_scale_w": "int16",
                    "dst_doc_id": "string",
                    "dst_page_id": "string",
                    "dst_line_id": "int32",
                    "dst_scale_w": "int16",
                    "sim": "float32",
                    "rank_src_to_dst": "int16",
                    "rank_dst_to_src": "int16",
                    "stability_count": "int16",
                    "stability_ratio": "float32",
                    "multi_scale_ok": "bool",
                    "notes": "string",
                },
                copy=False,
            )
        out_df.to_parquet(out_pairs_path, index=False)

        # Summary
        sims = out_df["sim"].tolist() if "sim" in out_df else []
        st_ratios = out_df["stability_ratio"].tolist() if "stability_ratio" in out_df else []
        by_scale = Counter(int(v) for v in out_df["src_scale_w"].tolist()) if "src_scale_w" in out_df else Counter()
        by_doc = Counter(str(v) for v in out_df["src_doc_id"].tolist()) if "src_doc_id" in out_df else Counter()
        by_page = Counter(
            f"{d}::{p}" for d, p in zip(out_df.get("src_doc_id", []), out_df.get("src_page_id", []))
        ) if not out_df.empty else Counter()
        summary = {
            "total_patches": int(len(records)),
            "total_pairs": int(len(out_df)),
            "scales_present": [int(s) for s in sorted(self._scale_states.keys())],
            "sim": {
                "min": float(np.min(sims)) if sims else None,
                "max": float(np.max(sims)) if sims else None,
                "mean": float(np.mean(sims)) if sims else None,
            },
            "stability_ratio": {
                "min": float(np.min(st_ratios)) if st_ratios else None,
                "max": float(np.max(st_ratios)) if st_ratios else None,
                "mean": float(np.mean(st_ratios)) if st_ratios else None,
            },
            "counts_by_scale": {str(k): int(v) for k, v in sorted(by_scale.items())},
            "top_docs": by_doc.most_common(10),
            "top_pages": by_page.most_common(10),
            "out_pairs_path": str(out_pairs_path),
        }
        summary_path = out_pairs_path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        self._debug_dump_pairs(final_pairs, dataset_dir=dataset_dir, n=int(debug_dump))
        summary["summary_path"] = str(summary_path)
        summary["debug_dump"] = int(max(0, debug_dump))
        return summary


__all__ = [
    "PatchRecord",
    "MNNConfig",
    "MNNMiner",
    "is_excluded_pair",
    "find_mutual_ranks",
    "select_center_match",
]
