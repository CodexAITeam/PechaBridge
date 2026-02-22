"""Pair-aware batch sampler that increases in-batch MNN positive coverage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import Sampler

from pechabridge.training.mnn_pairs import PatchMeta


@dataclass(frozen=True)
class SamplerConfig:
    batch_size: int = 64
    p_pair: float = 0.6
    hard_negative_ratio: float = 0.2
    drop_last: bool = True
    seed: int = 42


class PairBatchSampler(Sampler[List[int]]):
    """
    Sampler that:
    - picks anchor indices
    - with probability p_pair includes an MNN-linked partner
    - injects same-page different-line hard negatives
    """

    def __init__(
        self,
        *,
        records: Sequence[PatchMeta],
        patch_id_to_index: Dict[int, int],
        mnn_map: Dict[int, List[Tuple[int, float]]],
        config: SamplerConfig,
    ):
        self.records = list(records)
        self.patch_id_to_index = dict(patch_id_to_index)
        self.mnn_map = dict(mnn_map)
        self.cfg = config
        self.batch_size = max(1, int(config.batch_size))
        self.drop_last = bool(config.drop_last)
        self.seed = int(config.seed)
        self.epoch = 0

        self.indices = np.arange(len(self.records), dtype=np.int64)
        self._by_page: Dict[Tuple[str, str], List[int]] = {}
        for idx, rec in enumerate(self.records):
            key = (str(rec.doc_id), str(rec.page_id))
            self._by_page.setdefault(key, []).append(int(idx))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        n = len(self.records)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def _pick_mnn_partner(self, anchor_idx: int, used: set[int], rng: np.random.Generator) -> Optional[int]:
        rec = self.records[int(anchor_idx)]
        neigh = self.mnn_map.get(int(rec.patch_id), [])
        if not neigh:
            return None
        # Sample weighted by edge weight.
        pids = []
        ws = []
        for pid, w in neigh:
            j = self.patch_id_to_index.get(int(pid))
            if j is None:
                continue
            if int(j) in used:
                continue
            pids.append(int(j))
            ws.append(max(1e-9, float(w)))
        if not pids:
            return None
        probs = np.asarray(ws, dtype=np.float64)
        probs = probs / float(np.sum(probs))
        pick = int(rng.choice(np.asarray(pids, dtype=np.int64), p=probs))
        return pick

    def _pick_hard_negative(self, anchor_idx: int, used: set[int], rng: np.random.Generator) -> Optional[int]:
        rec = self.records[int(anchor_idx)]
        page_key = (str(rec.doc_id), str(rec.page_id))
        pool = self._by_page.get(page_key, [])
        if not pool:
            return None
        candidates = [j for j in pool if j not in used and int(self.records[j].line_id) != int(rec.line_id)]
        if not candidates:
            return None
        return int(rng.choice(np.asarray(candidates, dtype=np.int64)))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        perm = self.indices.copy()
        rng.shuffle(perm)
        ptr = 0
        n = len(perm)

        while ptr < n:
            batch: List[int] = []
            used: set[int] = set()

            # Anchor + optional MNN pair.
            while len(batch) < self.batch_size and ptr < n:
                a = int(perm[ptr])
                ptr += 1
                if a in used:
                    continue
                batch.append(a)
                used.add(a)

                if len(batch) >= self.batch_size:
                    break
                if float(rng.uniform()) <= float(self.cfg.p_pair):
                    partner = self._pick_mnn_partner(anchor_idx=a, used=used, rng=rng)
                    if partner is not None and partner not in used:
                        batch.append(int(partner))
                        used.add(int(partner))

            # Inject hard negatives (same page, different line).
            target_hn = int(round(float(self.cfg.hard_negative_ratio) * float(self.batch_size)))
            hn_added = 0
            tries = 0
            while len(batch) < self.batch_size and hn_added < target_hn and tries < self.batch_size * 8:
                tries += 1
                if not batch:
                    break
                a = int(batch[int(rng.integers(0, len(batch)))])
                j = self._pick_hard_negative(anchor_idx=a, used=used, rng=rng)
                if j is None or j in used:
                    continue
                batch.append(int(j))
                used.add(int(j))
                hn_added += 1

            # Fill with random samples.
            while len(batch) < self.batch_size and ptr < n:
                j = int(perm[ptr])
                ptr += 1
                if j in used:
                    continue
                batch.append(j)
                used.add(j)

            if len(batch) < self.batch_size and self.drop_last:
                break
            if batch:
                yield batch

