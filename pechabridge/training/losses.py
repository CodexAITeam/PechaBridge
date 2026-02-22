"""Multi-positive InfoNCE loss utilities for MNN-driven training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

from pechabridge.training.mnn_pairs import PatchMeta, one_d_iou


@dataclass(frozen=True)
class MpNCEConfig:
    tau: float = 0.07
    w_overlap: float = 0.3
    w_multiscale: float = 0.2
    t_iou: float = 0.6
    eps_center: float = 0.06
    min_positives_per_anchor: int = 1
    allow_self_fallback: bool = True
    exclude_same_page_in_denominator: bool = False
    lambda_smooth: float = 0.05


def _same_page(a: PatchMeta, b: PatchMeta) -> bool:
    return str(a.doc_id) == str(b.doc_id) and str(a.page_id) == str(b.page_id)


def _is_overlap_pos(a: PatchMeta, b: PatchMeta, t_iou: float) -> bool:
    if str(a.doc_id) != str(b.doc_id) or str(a.page_id) != str(b.page_id):
        return False
    if int(a.line_id) != int(b.line_id):
        return False
    if int(a.scale_w) != int(b.scale_w):
        return False
    iou = one_d_iou(float(a.x0_norm), float(a.x1_norm), float(b.x0_norm), float(b.x1_norm))
    return float(iou) >= float(t_iou)


def _is_multiscale_pos(a: PatchMeta, b: PatchMeta, eps_center: float) -> bool:
    if str(a.doc_id) != str(b.doc_id) or str(a.page_id) != str(b.page_id):
        return False
    if int(a.line_id) != int(b.line_id):
        return False
    if int(a.scale_w) == int(b.scale_w):
        return False
    return abs(float(a.center) - float(b.center)) < float(eps_center)


def multi_positive_infonce(
    z: torch.Tensor,
    metas: Sequence[PatchMeta],
    patch_ids: Sequence[int],
    mnn_map: Mapping[int, Sequence[Tuple[int, float]]],
    cfg: MpNCEConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute weighted multi-positive InfoNCE + optional overlap smoothness.

    Args:
      z: [M, D] embeddings (already projected)
      metas: len M patch metadata (aligned with z rows)
      patch_ids: len M patch ids
      mnn_map: src_pid -> list[(dst_pid, weight)]
    """
    if z.ndim != 2:
        raise ValueError(f"Expected z [M,D], got shape={tuple(z.shape)}")
    if len(metas) != int(z.shape[0]) or len(patch_ids) != int(z.shape[0]):
        raise ValueError("metas/patch_ids length mismatch with z.")

    M = int(z.shape[0])
    if M < 2:
        return z.sum() * 0.0, {
            "valid_anchors": 0.0,
            "avg_positives": 0.0,
            "mnn_pos": 0.0,
            "overlap_pos": 0.0,
            "multiscale_pos": 0.0,
            "fallback_pos": 0.0,
            "smooth_pairs": 0.0,
        }

    z = F.normalize(z, dim=-1)
    tau = max(float(cfg.tau), 1e-6)
    logits = torch.matmul(z, z.T).float() / float(tau)

    losses: List[torch.Tensor] = []
    valid_anchors = 0
    total_pos = 0
    count_mnn = 0
    count_overlap = 0
    count_multiscale = 0
    count_fallback = 0
    smooth_pairs: List[Tuple[int, int]] = []

    # Fast lookup for in-batch indices by patch_id.
    pid_to_indices: Dict[int, List[int]] = {}
    for i, pid in enumerate(patch_ids):
        pid_to_indices.setdefault(int(pid), []).append(int(i))

    neg_inf = torch.tensor(float("-inf"), device=z.device, dtype=logits.dtype)

    for i in range(M):
        pid_i = int(patch_ids[i])
        meta_i = metas[i]
        pos_idx: List[int] = []
        pos_w: List[float] = []
        pos_kind: List[str] = []

        # MNN positives.
        neigh = mnn_map.get(pid_i, [])
        if neigh:
            neigh_map = {int(dst): float(w) for dst, w in neigh}
            for j in range(M):
                if i == j:
                    continue
                w = neigh_map.get(int(patch_ids[j]))
                if w is None:
                    continue
                pos_idx.append(j)
                pos_w.append(max(1e-8, float(w)))
                pos_kind.append("mnn")

        # Overlap and multiscale positives.
        for j in range(M):
            if i == j:
                continue
            if j in pos_idx:
                continue
            meta_j = metas[j]
            if _is_overlap_pos(meta_i, meta_j, cfg.t_iou):
                pos_idx.append(j)
                pos_w.append(max(1e-8, float(cfg.w_overlap)))
                pos_kind.append("overlap")
                if i < j:
                    smooth_pairs.append((i, j))
                continue
            if _is_multiscale_pos(meta_i, meta_j, cfg.eps_center):
                pos_idx.append(j)
                pos_w.append(max(1e-8, float(cfg.w_multiscale)))
                pos_kind.append("multiscale")

        # Self fallback only when no other positives.
        if (not pos_idx) and bool(cfg.allow_self_fallback):
            for j in pid_to_indices.get(pid_i, []):
                if j == i:
                    continue
                pos_idx.append(int(j))
                pos_w.append(1.0)
                pos_kind.append("fallback")

        if len(pos_idx) < max(1, int(cfg.min_positives_per_anchor)):
            continue

        # Denominator candidates.
        den_mask = torch.ones((M,), dtype=torch.bool, device=z.device)
        den_mask[i] = False
        if bool(cfg.exclude_same_page_in_denominator):
            for j in range(M):
                if j == i:
                    continue
                if j in pos_idx:
                    continue
                if _same_page(meta_i, metas[j]):
                    den_mask[j] = False
        den_idx = torch.where(den_mask)[0]
        if den_idx.numel() <= 0:
            continue

        row = logits[i]
        row_den = row.clone()
        row_den[~den_mask] = neg_inf
        log_den = torch.logsumexp(row_den, dim=0)

        pidx = torch.tensor(pos_idx, device=z.device, dtype=torch.long)
        pw = torch.tensor(pos_w, device=z.device, dtype=logits.dtype).clamp(min=1e-8)
        log_num = torch.logsumexp(row[pidx] + torch.log(pw), dim=0)

        li = -(log_num - log_den)
        losses.append(li)
        valid_anchors += 1
        total_pos += len(pos_idx)
        count_mnn += sum(1 for k in pos_kind if k == "mnn")
        count_overlap += sum(1 for k in pos_kind if k == "overlap")
        count_multiscale += sum(1 for k in pos_kind if k == "multiscale")
        count_fallback += sum(1 for k in pos_kind if k == "fallback")

    if not losses:
        return z.sum() * 0.0, {
            "valid_anchors": 0.0,
            "avg_positives": 0.0,
            "mnn_pos": 0.0,
            "overlap_pos": 0.0,
            "multiscale_pos": 0.0,
            "fallback_pos": 0.0,
            "smooth_pairs": float(len(smooth_pairs)),
        }

    loss_main = torch.stack(losses).mean()

    loss_smooth = z.sum() * 0.0
    if float(cfg.lambda_smooth) > 0.0 and smooth_pairs:
        dist_terms: List[torch.Tensor] = []
        for i, j in smooth_pairs:
            dist_terms.append(torch.sum((z[i] - z[j]) ** 2))
        if dist_terms:
            loss_smooth = torch.stack(dist_terms).mean()
            loss_main = loss_main + float(cfg.lambda_smooth) * loss_smooth

    stats = {
        "valid_anchors": float(valid_anchors),
        "avg_positives": float(total_pos / max(1, valid_anchors)),
        "mnn_pos": float(count_mnn),
        "overlap_pos": float(count_overlap),
        "multiscale_pos": float(count_multiscale),
        "fallback_pos": float(count_fallback),
        "smooth_pairs": float(len(smooth_pairs)),
    }
    return loss_main, stats


__all__ = ["MpNCEConfig", "multi_positive_infonce"]

