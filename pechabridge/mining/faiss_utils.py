"""FAISS helpers for cosine/IP index build/load/search."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np


def import_faiss():
    """Lazy import faiss with actionable error message."""
    try:
        import faiss  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("faiss is required. Install `faiss-cpu` or `faiss-gpu`.") from exc
    return faiss


def set_faiss_num_threads(num_threads: Optional[int]) -> bool:
    """
    Configure FAISS/OpenMP threads if supported.

    Returns True when a thread setting was applied.
    """
    if num_threads is None:
        return False
    n = int(num_threads)
    if n <= 0:
        return False
    faiss = import_faiss()
    setter = getattr(faiss, "omp_set_num_threads", None)
    if setter is None:
        return False
    try:
        setter(int(n))
        return True
    except Exception:
        return False


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape={arr.shape}")
    norms = np.linalg.norm(arr, ord=2, axis=1, keepdims=True)
    norms = np.maximum(norms, float(eps))
    return arr / norms


def build_faiss_index(
    embeddings: np.ndarray,
    *,
    faiss_factory: str = "Flat",
    use_gpu: bool = False,
) -> Any:
    """
    Build IP-based FAISS index.

    Assumes embeddings are already L2-normalized if cosine similarity is desired.
    """
    faiss = import_faiss()
    x = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty 2D float32 array.")
    dim = int(x.shape[1])
    factory = str(faiss_factory or "Flat").strip()

    if factory.lower() in {"flat", "indexflatip"}:
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.index_factory(dim, factory, faiss.METRIC_INNER_PRODUCT)
        if not index.is_trained:
            index.train(x)

    index.add(x)

    if bool(use_gpu):
        try:  # pragma: no cover
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception:
            # Fall back to CPU index.
            pass
    return index


def search_index(index: Any, queries: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    """Batch FAISS search wrapper."""
    q = np.ascontiguousarray(np.asarray(queries, dtype=np.float32))
    if q.ndim != 2:
        raise ValueError(f"queries must be 2D array, got shape={q.shape}")
    k = max(1, int(topk))
    d, i = index.search(q, k)
    return d, i


def save_index(index: Any, index_path: Path) -> None:
    """Persist CPU FAISS index to disk."""
    faiss = import_faiss()
    p = Path(index_path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    idx = index
    try:  # pragma: no cover
        if hasattr(faiss, "index_gpu_to_cpu"):
            idx = faiss.index_gpu_to_cpu(index)
    except Exception:
        idx = index
    faiss.write_index(idx, str(p))


def load_index(index_path: Path, *, use_gpu: bool = False) -> Any:
    """Load FAISS index from disk with optional GPU transfer."""
    faiss = import_faiss()
    p = Path(index_path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"FAISS index not found: {p}")
    index = faiss.read_index(str(p))
    if bool(use_gpu):
        try:  # pragma: no cover
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception:
            pass
    return index


__all__ = [
    "import_faiss",
    "set_faiss_num_threads",
    "l2_normalize_rows",
    "build_faiss_index",
    "search_index",
    "save_index",
    "load_index",
]
