"""Service layer for indexing and querying transcript lines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
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
    top_k: int
    context_lines: int
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

    def initialize_index(
        self,
        force_rebuild: bool = False,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> IndexSummary:
        self._emit_progress(progress_callback, 0.05, "Inspecting local transcript index")
        should_rebuild = (
            force_rebuild
            or self.config.qdrant.force_recreate_collection
            or self.config.search.rebuild_index_on_start
            or self.index.count() == 0
        )

        if should_rebuild:
            self._emit_progress(progress_callback, 0.20, "Loading transcript files and metadata")
            records, stats = self.corpus_loader.load()
            self._last_corpus_stats = stats
            self._emit_progress(progress_callback, 0.60, "Writing embeddings into the Qdrant collection")
            indexed_records = self.index.rebuild(records)
            self._emit_progress(progress_callback, 1.00, "Index rebuild finished")
            return IndexSummary(
                indexed_records=indexed_records,
                pecha_count=stats.pecha_count,
                file_count=stats.file_count,
                rebuilt=True,
                collection_name=self.config.qdrant.collection_name,
            )

        if self._last_corpus_stats.line_count == 0:
            self._emit_progress(progress_callback, 0.50, "Refreshing corpus statistics")
            _, stats = self.corpus_loader.load()
            self._last_corpus_stats = stats

        self._emit_progress(progress_callback, 1.00, "Existing index loaded")
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
        top_k: int | None = None,
        context_lines: int | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> SearchResponse:
        self._emit_progress(progress_callback, 0.05, "Validating search query")
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Please enter a search query.")

        requested_top_k = self.config.search.top_k if top_k is None else max(1, int(top_k))
        requested_context_lines = (
            self.config.chunking.context_lines
            if context_lines is None
            else max(0, int(context_lines))
        )

        self._emit_progress(progress_callback, 0.20, "Translating the query into Classical Tibetan")
        translated_query = self.translation.translate_query_to_tibetan(normalized_query)
        self._emit_progress(progress_callback, 0.45, "Searching the Qdrant vector index")
        matches = self.index.similarity_search(translated_query, top_k=requested_top_k)
        should_back_translate = (
            self.config.search.include_back_translation_default
            if include_back_translation is None
            else include_back_translation
        )

        self._emit_progress(progress_callback, 0.65, "Collecting context windows around matching lines")
        results: list[SearchHit] = []
        total_matches = max(1, len(matches))
        for rank, match in enumerate(matches, start=1):
            metadata = dict(match.document.metadata)
            source_label = self.corpus_loader.build_source_label(metadata)
            context = self.corpus_loader.get_context_window(
                relative_source_file=metadata["source_file"],
                center_line_index=int(metadata["line_index"]),
                context_lines=requested_context_lines,
            )
            back_translation = None
            if should_back_translate and context:
                progress_start = 0.78
                progress_span = 0.17
                progress_value = progress_start + (rank - 1) / total_matches * progress_span
                self._emit_progress(
                    progress_callback,
                    progress_value,
                    f"Back-translating context {rank}/{total_matches}",
                )
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

        self._emit_progress(progress_callback, 1.00, "Search results ready")
        return SearchResponse(
            original_query=normalized_query,
            translated_query=translated_query,
            top_k=requested_top_k,
            context_lines=requested_context_lines,
            results=results,
        )

    @staticmethod
    def _emit_progress(
        progress_callback: Callable[[float, str], None] | None,
        value: float,
        description: str,
    ) -> None:
        if progress_callback is not None:
            progress_callback(value, description)
