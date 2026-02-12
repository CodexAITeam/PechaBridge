"""
SentenceTransformer-based text embedding for TibetanOCR verifier.

- Uses a pretrained SentenceTransformer (default: billingsmoore/minilm-bo)
- Returns L2-normalized embeddings as torch.FloatTensor
- Batched encoding for speed, with optional device placement
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise ImportError(
        "sentence-transformers is required. Install with: "
        "pip install -U sentence-transformers"
    ) from e


@dataclass
class STConfig:
    model_name: str = "billingsmoore/minilm-bo"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 64
    normalize: bool = True


class STTextEmbeddingProvider(nn.Module):
    """
    Frozen SentenceTransformer-based text encoder.

    .encode(texts: List[str]) -> torch.FloatTensor [B, D]
    """
    def __init__(self, cfg: Optional[STConfig] = None):
        super().__init__()
        self.cfg = cfg or STConfig()
        # SentenceTransformer handles its own device; keep torch params empty
        self._model = SentenceTransformer(self.cfg.model_name, device=self.cfg.device)
        # Run a quick dummy forward to discover output dim (e.g., 384)
        with torch.inference_mode():
            _probe = self._model.encode(["probe"], convert_to_numpy=True, normalize_embeddings=False)
        self.out_dim = int(_probe.shape[-1])

        # Freeze flag for consistency with nn.Module APIs (no trainable params)
        self.requires_grad_(False)

    @torch.no_grad()
    def encode(self, texts: List[str], device: Optional[str] = None) -> torch.Tensor:
        """
        Returns a torch tensor [B, D] on CPU by default; pass device="cuda" to move.
        """
        if not texts:
            return torch.empty(0, self.out_dim)

        embs: np.ndarray = self._model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we'll normalize below for consistency
            show_progress_bar=False,
        )
        t = torch.from_numpy(embs).float()  # CPU tensor
        if self.cfg.normalize:
            t = torch.nn.functional.normalize(t, dim=-1)

        if device is not None:
            t = t.to(device)
        return t
