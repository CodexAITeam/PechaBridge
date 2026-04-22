"""Compatibility imports for LangChain/Qdrant variants."""

from __future__ import annotations

try:
    from langchain.embeddings.base import Embeddings
except ImportError:  # pragma: no cover - fallback for newer LangChain layouts
    from langchain_core.embeddings import Embeddings

try:
    from langchain.schema import Document
except ImportError:  # pragma: no cover - fallback for newer LangChain layouts
    from langchain_core.documents import Document

try:
    from langchain.vectorstores import Qdrant as LangChainQdrant
except ImportError:  # pragma: no cover - fallback for community split packages
    from langchain_community.vectorstores import Qdrant as LangChainQdrant

__all__ = ["Document", "Embeddings", "LangChainQdrant"]
