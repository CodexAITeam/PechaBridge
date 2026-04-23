"""Service layer for indexing and querying transcript lines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Any
from urllib.parse import quote

try:
    import pyewts
except ImportError:  # pragma: no cover - optional runtime dependency
    pyewts = None

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
    collection_metadata: dict[str, Any]
    back_translation: str | None
    scan_url: str | None
    scan_filename: str | None
    scan_index: int | None
    source_file_url: str | None
    metadata_file_url: str | None


@dataclass(frozen=True)
class SearchResponse:
    original_query: str
    translated_query: str
    query_mode: str
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
        self._wylie_converter = self._build_wylie_converter()

    def initialize_index(
        self,
        force_rebuild: bool = False,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> IndexSummary:
        self._emit_progress(progress_callback, 0.05, "Checking the local Qdrant collection")
        should_rebuild = (
            force_rebuild
            or self.config.qdrant.force_recreate_collection
            or self.config.search.rebuild_index_on_start
            or self.index.count() == 0
        )

        if should_rebuild:
            self._emit_progress(progress_callback, 0.20, "Loading transcript files and metadata references")
            records, stats = self.corpus_loader.load()
            self._last_corpus_stats = stats
            self._emit_progress(progress_callback, 0.60, "Embedding transcript lines and refreshing the Qdrant collection")
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
            self._emit_progress(progress_callback, 0.50, "Refreshing corpus statistics for the status panel")
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
        query_mode: str = "DE / EN",
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

        translated_query = self._prepare_query(
            query=normalized_query,
            query_mode=query_mode,
            progress_callback=progress_callback,
        )
        self._emit_progress(progress_callback, 0.45, "Searching the local Qdrant vector index")
        matches = self.index.similarity_search(translated_query, top_k=requested_top_k)
        should_back_translate = (
            self.config.search.include_back_translation_default
            if include_back_translation is None
            else include_back_translation
        )

        self._emit_progress(progress_callback, 0.65, "Collecting context windows and resolving page scans")
        results: list[SearchHit] = []
        total_matches = max(1, len(matches))
        for rank, match in enumerate(matches, start=1):
            metadata = dict(match.document.metadata)
            collection_metadata = self.corpus_loader.resolve_collection_metadata(metadata)
            source_label = self.corpus_loader.build_source_label(metadata)
            context = self.corpus_loader.get_context_window(
                relative_source_file=metadata["source_file"],
                center_line_index=int(metadata["line_index"]),
                context_lines=requested_context_lines,
            )
            page_scan = self.corpus_loader.resolve_page_scan(metadata)
            source_file_url = self._build_gradio_file_url(
                self.corpus_loader.resolve_source_path(str(metadata["source_file"]))
            )
            metadata_path = self.corpus_loader.resolve_metadata_path(metadata)
            metadata_file_url = (
                self._build_gradio_file_url(metadata_path)
                if metadata_path is not None
                else None
            )
            back_translation = None
            if should_back_translate and context:
                progress_start = 0.78
                progress_span = 0.17
                progress_value = progress_start + (rank - 1) / total_matches * progress_span
                self._emit_progress(
                    progress_callback,
                    progress_value,
                    f"Preparing English support translation {rank}/{total_matches}",
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
                    collection_metadata=collection_metadata,
                    back_translation=back_translation,
                    scan_url=(str(page_scan.get("source_url")) if page_scan and page_scan.get("source_url") else None),
                    scan_filename=(
                        str(page_scan.get("filename"))
                        if page_scan and page_scan.get("filename")
                        else None
                    ),
                    scan_index=(
                        int(page_scan["index"])
                        if page_scan and isinstance(page_scan.get("index"), int)
                        else None
                    ),
                    source_file_url=source_file_url,
                    metadata_file_url=metadata_file_url,
                )
            )

        self._emit_progress(progress_callback, 1.00, "Search results ready")
        return SearchResponse(
            original_query=normalized_query,
            translated_query=translated_query,
            query_mode=query_mode,
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

    def _prepare_query(
        self,
        query: str,
        query_mode: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> str:
        normalized_mode = str(query_mode or "").strip().lower()

        if normalized_mode == "wylie (ewts)":
            self._emit_progress(progress_callback, 0.20, "Converting Wylie (EWTS) input into Tibetan Unicode")
            if self._wylie_converter is None:
                raise RuntimeError(
                    "Wylie input requires the `pyewts` package, but it is not available in the current environment."
                )
            try:
                converted_query = str(self._wylie_converter.toUnicode(query)).strip()
            except Exception as exc:  # pragma: no cover - depends on runtime library behavior
                raise RuntimeError(f"Could not convert the Wylie query to Tibetan Unicode: {exc}") from exc
            if not converted_query:
                raise ValueError("The Wylie query could not be converted into Tibetan Unicode.")
            return converted_query

        if normalized_mode in {"tibetan unicode", "tibetan"}:
            self._emit_progress(progress_callback, 0.20, "Using Tibetan Unicode input directly for retrieval")
            return query

        if normalized_mode in {
            "de / en",
            "german / english (translate first)",
            "natural language / tibetan",
        }:
            self._emit_progress(progress_callback, 0.20, "Translating the DE / EN query into Classical Tibetan")
            return self.translation.translate_query_to_tibetan(query)

        self._emit_progress(progress_callback, 0.20, "Translating the query into Classical Tibetan")
        return self.translation.translate_query_to_tibetan(query)

    @staticmethod
    def _build_wylie_converter() -> Any | None:
        if pyewts is None:
            return None
        try:
            return pyewts.pyewts()
        except Exception:  # pragma: no cover - depends on runtime library behavior
            return None

    @staticmethod
    def _build_gradio_file_url(path: Path) -> str | None:
        resolved_path = path.resolve()
        if not resolved_path.exists():
            return None
        return f"/file={quote(resolved_path.as_posix(), safe='/:')}"
