"""Transcript ingestion and context-window handling."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import SemanticSearchConfig

_NUMERIC_TOKEN_PATTERN = re.compile(r"(\d+)")
_PAGE_IMAGE_NUMBER_PATTERN = re.compile(r".*-([0-9]+)(?:_|$)")


@dataclass(frozen=True)
class TranscriptLine:
    point_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CorpusStats:
    pecha_count: int
    file_count: int
    line_count: int


class TranscriptCorpusLoader:
    """Loads pecha transcripts and emits one vector entry per transcript line."""

    def __init__(self, config: SemanticSearchConfig) -> None:
        self.config = config
        self._page_pattern = re.compile(config.corpus.page_number_pattern)
        self._metadata_cache: dict[str, dict[str, Any]] = {}

    def load(self) -> tuple[list[TranscriptLine], CorpusStats]:
        transcripts_root = self.config.corpus.transcripts_root
        if not transcripts_root.exists():
            raise FileNotFoundError(
                f"Transcript directory does not exist: {transcripts_root}"
            )

        records: list[TranscriptLine] = []
        pecha_count = 0
        file_count = 0

        for pecha_dir in sorted(path for path in transcripts_root.iterdir() if path.is_dir()):
            pecha_count += 1
            collection_metadata = self._load_collection_metadata(pecha_dir)
            pecha_relative_path = pecha_dir.relative_to(transcripts_root).as_posix()
            metadata_relative_path = (
                Path(pecha_relative_path) / self.config.corpus.metadata_filename
            ).as_posix()
            transcript_files = sorted(pecha_dir.glob(self.config.corpus.file_glob))
            for transcript_file in transcript_files:
                if transcript_file.name == self.config.corpus.metadata_filename:
                    continue
                file_count += 1
                raw_lines = self._split_raw_lines(
                    transcript_file.read_text(encoding=self.config.corpus.text_encoding)
                )
                page_number = self._extract_page_number(transcript_file)
                relative_path = transcript_file.relative_to(transcripts_root)

                for line_index, line_text in self._iter_indexable_lines(raw_lines):
                    record_id = hashlib.sha1(
                        f"{relative_path.as_posix()}:{line_index}".encode("utf-8")
                    ).hexdigest()
                    metadata = {
                        "pecha_title": pecha_dir.name,
                        "pecha_path": pecha_relative_path,
                        "metadata_file": metadata_relative_path,
                        "page_number": page_number,
                        "source_file": relative_path.as_posix(),
                        "source_filename": transcript_file.name,
                        "line_index": line_index,
                        "line_number": line_index + 1,
                    }
                    metadata.update(self._flatten_collection_metadata(collection_metadata))
                    records.append(TranscriptLine(point_id=record_id, text=line_text, metadata=metadata))

        stats = CorpusStats(
            pecha_count=pecha_count,
            file_count=file_count,
            line_count=len(records),
        )
        return records, stats

    def resolve_source_path(self, relative_source_file: str) -> Path:
        return (self.config.corpus.transcripts_root / relative_source_file).resolve()

    def get_context_window(
        self,
        relative_source_file: str,
        center_line_index: int,
        context_lines: int | None = None,
    ) -> str:
        source_path = self.resolve_source_path(relative_source_file)
        raw_lines = self._split_raw_lines(
            source_path.read_text(encoding=self.config.corpus.text_encoding)
        )
        if not raw_lines:
            return ""

        window_radius = (
            self.config.chunking.context_lines
            if context_lines is None
            else max(0, int(context_lines))
        )
        start_index = max(0, center_line_index - window_radius)
        end_index = min(len(raw_lines), center_line_index + window_radius + 1)
        context_rows: list[str] = []
        for line_number, line_text in enumerate(raw_lines[start_index:end_index], start=start_index + 1):
            if not self.config.chunking.keep_empty_lines and not line_text:
                continue
            marker = ">" if line_number - 1 == center_line_index else " "
            context_rows.append(f"{marker} {line_number:04d}: {line_text}")
        return "\n".join(context_rows)

    def build_source_label(self, metadata: dict[str, Any]) -> str:
        return (
            f"{metadata['pecha_title']} | page {metadata['page_number']} | "
            f"line {metadata['line_number']} | {metadata['source_file']}"
        )

    def resolve_collection_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        # Backward compatibility for older Qdrant payloads that inlined the full metadata.json.
        embedded_metadata = metadata.get("collection_metadata")
        if isinstance(embedded_metadata, dict):
            return copy.deepcopy(embedded_metadata)

        metadata_path = self.resolve_metadata_path(metadata)
        if metadata_path is None:
            return {}
        return self._load_collection_metadata_from_path(metadata_path)

    def resolve_page_scan(self, metadata: dict[str, Any]) -> dict[str, Any] | None:
        collection_metadata = self.resolve_collection_metadata(metadata)
        pages = collection_metadata.get("pages")
        if not isinstance(pages, list) or not pages:
            return None

        source_filename = str(metadata.get("source_filename", ""))
        page_number = metadata.get("page_number")
        source_tokens = set(self._extract_numeric_tokens(source_filename))
        page_number_int = self._coerce_int(page_number)

        for page in pages:
            if not isinstance(page, dict):
                continue
            page_index = self._coerce_int(page.get("index"))
            page_filename = str(page.get("filename", ""))
            page_filename_number = self._extract_page_number_from_scan_filename(page_filename)
            if page_number_int is not None and page_index is not None and page_index + 1 == page_number_int:
                return copy.deepcopy(page)
            if page_number_int is not None and page_index is not None and page_index == page_number_int:
                return copy.deepcopy(page)
            if page_number_int is not None and page_filename_number is not None and page_filename_number == page_number_int:
                return copy.deepcopy(page)
            page_tokens = set(self._extract_numeric_tokens(page_filename))
            if page_number_int is None and source_tokens and page_tokens and source_tokens.intersection(page_tokens):
                return copy.deepcopy(page)

        for page in pages:
            if not isinstance(page, dict):
                continue
            if page.get("source_url"):
                return copy.deepcopy(page)
        return None

    def resolve_metadata_path(self, metadata: dict[str, Any]) -> Path | None:
        metadata_file = metadata.get("metadata_file")
        if isinstance(metadata_file, str) and metadata_file:
            return (self.config.corpus.transcripts_root / metadata_file).resolve()

        pecha_path = metadata.get("pecha_path")
        if isinstance(pecha_path, str) and pecha_path:
            return (
                self.config.corpus.transcripts_root
                / pecha_path
                / self.config.corpus.metadata_filename
            ).resolve()

        source_file = metadata.get("source_file")
        if isinstance(source_file, str) and source_file:
            return (
                self.resolve_source_path(source_file).parent / self.config.corpus.metadata_filename
            ).resolve()
        return None

    def _load_collection_metadata(self, pecha_dir: Path) -> dict[str, Any]:
        metadata_path = pecha_dir / self.config.corpus.metadata_filename
        if not metadata_path.exists():
            return {}
        return self._load_collection_metadata_from_path(metadata_path)

    def _extract_page_number(self, transcript_file: Path) -> int | str:
        match = self._page_pattern.search(transcript_file.stem)
        if not match:
            return transcript_file.stem
        page_token = match.group(1) if match.groups() else match.group(0)
        return int(page_token) if page_token.isdigit() else page_token

    def _split_raw_lines(self, content: str) -> list[str]:
        raw_lines = content.split(self.config.chunking.split_separator)
        if self.config.chunking.strip_lines:
            return [line.strip() for line in raw_lines]
        return raw_lines

    def _iter_indexable_lines(self, raw_lines: list[str]) -> list[tuple[int, str]]:
        indexable_lines: list[tuple[int, str]] = []
        for index, line_text in enumerate(raw_lines):
            if not self.config.chunking.keep_empty_lines and not line_text:
                continue
            if len(line_text) < self.config.chunking.min_line_length:
                continue
            indexable_lines.append((index, line_text))
        return indexable_lines

    def _flatten_collection_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        flattened: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                flattened[f"collection_{key}"] = value
        return flattened

    def _load_collection_metadata_from_path(self, metadata_path: Path) -> dict[str, Any]:
        resolved_path = metadata_path.resolve()
        cache_key = resolved_path.as_posix()
        cached_metadata = self._metadata_cache.get(cache_key)
        if cached_metadata is not None:
            return copy.deepcopy(cached_metadata)

        if not resolved_path.exists():
            return {}

        loaded_metadata = json.loads(
            resolved_path.read_text(encoding=self.config.corpus.text_encoding)
        )
        if not isinstance(loaded_metadata, dict):
            return {}

        self._metadata_cache[cache_key] = loaded_metadata
        return copy.deepcopy(loaded_metadata)

    @staticmethod
    def _extract_numeric_tokens(value: str) -> list[int]:
        return [
            int(token)
            for token in _NUMERIC_TOKEN_PATTERN.findall(value)
        ]

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return None

    @staticmethod
    def _extract_page_number_from_scan_filename(value: str) -> int | None:
        match = _PAGE_IMAGE_NUMBER_PATTERN.search(value)
        if not match:
            return None
        token = match.group(1)
        return int(token) if token.isdigit() else None
