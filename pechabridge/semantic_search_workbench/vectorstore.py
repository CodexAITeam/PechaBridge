"""Qdrant-backed vector index for transcript lines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from .compat import Document, LangChainQdrant
from .config import QdrantConfig, SearchConfig
from .documents import TranscriptLine
from .embeddings import TibetanClipEmbeddings


@dataclass(frozen=True)
class SearchMatch:
    document: Document
    score: float


class QdrantSemanticIndex:
    """Manages collection creation, indexing, and similarity search."""

    def __init__(
        self,
        config: QdrantConfig,
        search_config: SearchConfig,
        embeddings: TibetanClipEmbeddings,
    ) -> None:
        self.config = config
        self.search_config = search_config
        self.embeddings = embeddings
        self.config.storage_path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.config.storage_path))
        self._vectorstore: LangChainQdrant | None = None

    def count(self) -> int:
        if not self.collection_exists():
            return 0
        return int(self.client.count(collection_name=self.config.collection_name, exact=True).count)

    def collection_exists(self) -> bool:
        try:
            return bool(self.client.collection_exists(self.config.collection_name))
        except AttributeError:  # pragma: no cover - older qdrant-client fallback
            try:
                self.client.get_collection(self.config.collection_name)
            except Exception:
                return False
            return True

    def ensure_collection(self, recreate: bool = False) -> None:
        if recreate and self.collection_exists():
            self.client.delete_collection(self.config.collection_name)

        if not self.collection_exists():
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.embeddings.embedding_size,
                    distance=self._distance_from_config(),
                ),
            )
        self._vectorstore = LangChainQdrant(
            client=self.client,
            collection_name=self.config.collection_name,
            embeddings=self.embeddings,
        )

    def rebuild(self, records: list[TranscriptLine]) -> int:
        self.ensure_collection(recreate=True)
        if not records:
            return 0

        for batch in self._batched(records, self.embeddings.config.batch_size):
            self._vectorstore.add_texts(
                texts=[record.text for record in batch],
                metadatas=[record.metadata for record in batch],
                ids=[record.point_id for record in batch],
                batch_size=self.embeddings.config.batch_size,
            )
        return len(records)

    def similarity_search(self, query: str, top_k: int | None = None) -> list[SearchMatch]:
        self.ensure_collection(recreate=False)
        limit = self.search_config.top_k if top_k is None else max(1, int(top_k))
        query_vector = self.embeddings.embed_query(query)
        if hasattr(self._vectorstore, "similarity_search_with_score_by_vector"):
            raw_results = self._vectorstore.similarity_search_with_score_by_vector(
                embedding=query_vector,
                k=limit,
            )
        else:  # pragma: no cover - fallback for older LangChain variants
            raw_results = self._vectorstore.similarity_search_with_score(
                query=query,
                k=limit,
            )
        matches = [
            SearchMatch(document=document, score=score)
            for document, score in raw_results
            if self.search_config.score_threshold is None or score >= self.search_config.score_threshold
        ]
        return matches

    def _distance_from_config(self) -> Distance:
        normalized = self.config.distance.strip().lower()
        mapping = {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "euclid": Distance.EUCLID,
            "manhattan": Distance.MANHATTAN,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported Qdrant distance strategy: {self.config.distance}")
        return mapping[normalized]

    @staticmethod
    def _batched(values: list[TranscriptLine], batch_size: int) -> Iterable[list[TranscriptLine]]:
        for start in range(0, len(values), batch_size):
            yield values[start : start + batch_size]
