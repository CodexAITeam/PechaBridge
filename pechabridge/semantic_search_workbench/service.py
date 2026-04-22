"""Service layer for indexing and querying transcript lines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import SemanticSearchConfig
from .documents import CorpusStats, TranscriptCorpusLoader
from .embeddings import TibetanClipEmbeddings
from .translation import TranslationProxy
from .vectorstore import QdrantSemanticIndex


@dataclass(frozen=True)
class IndexSummary:
    indexed_records: int
    pecha_count: int
    file_count: int
    rebuilt: bool
    collection_name: str


@dataclass(frozen=True)
class SearchHit:
    rank: int
    score: float
    source_label: str
    matched_line: str
    context: str
    metadata: dict[str, Any]
    back_translation: str | None


@dataclass(frozen=True)
class SearchResponse:
    original_query: str
    translated_query: str
    results: list[SearchHit]


class SemanticSearchWorkbenchService:
    """Coordinates ingestion, indexing, translation, and search."""

    def __init__(self, config: SemanticSearchConfig) -> None:
        self.config = config
        self.corpus_loader = TranscriptCorpusLoader(config)
        self.embeddings = TibetanClipEmbeddings(config.embedding)
        self.translation = TranslationProxy(config.translation)
        self.index = QdrantSemanticIndex(config.qdrant, config.search, self.embeddings)
        self._last_corpus_stats = CorpusStats(pecha_count=0, file_count=0, line_count=0)

    def initialize_index(self, force_rebuild: bool = False) -> IndexSummary:
        should_rebuild = (
            force_rebuild
            or self.config.qdrant.force_recreate_collection
            or self.config.search.rebuild_index_on_start
            or self.index.count() == 0
        )

        if should_rebuild:
            records, stats = self.corpus_loader.load()
            self._last_corpus_stats = stats
            indexed_records = self.index.rebuild(records)
            return IndexSummary(
                indexed_records=indexed_records,
                pecha_count=stats.pecha_count,
                file_count=stats.file_count,
                rebuilt=True,
                collection_name=self.config.qdrant.collection_name,
            )

        if self._last_corpus_stats.line_count == 0:
            _, stats = self.corpus_loader.load()
            self._last_corpus_stats = stats

        return IndexSummary(
            indexed_records=self.index.count(),
            pecha_count=self._last_corpus_stats.pecha_count,
            file_count=self._last_corpus_stats.file_count,
            rebuilt=False,
            collection_name=self.config.qdrant.collection_name,
        )

    def search(
        self,
        query: str,
        include_back_translation: bool | None = None,
    ) -> SearchResponse:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Please enter a search query.")

        translated_query = self.translation.translate_query_to_tibetan(normalized_query)
        matches = self.index.similarity_search(translated_query)
        should_back_translate = (
            self.config.search.include_back_translation_default
            if include_back_translation is None
            else include_back_translation
        )

        results: list[SearchHit] = []
        for rank, match in enumerate(matches, start=1):
            metadata = dict(match.document.metadata)
            source_label = self.corpus_loader.build_source_label(metadata)
            context = self.corpus_loader.get_context_window(
                relative_source_file=metadata["source_file"],
                center_line_index=int(metadata["line_index"]),
            )
            back_translation = None
            if should_back_translate and context:
                back_translation = self.translation.back_translate_to_english(context)
            results.append(
                SearchHit(
                    rank=rank,
                    score=float(match.score),
                    source_label=source_label,
                    matched_line=match.document.page_content,
                    context=context,
                    metadata=metadata,
                    back_translation=back_translation,
                )
            )

        return SearchResponse(
            original_query=normalized_query,
            translated_query=translated_query,
            results=results,
        )
