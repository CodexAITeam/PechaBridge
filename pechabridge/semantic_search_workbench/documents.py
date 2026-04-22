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
                        "page_number": page_number,
                        "source_file": relative_path.as_posix(),
                        "source_filename": transcript_file.name,
                        "line_index": line_index,
                        "line_number": line_index + 1,
                        "collection_metadata": collection_metadata,
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

    def _load_collection_metadata(self, pecha_dir: Path) -> dict[str, Any]:
        metadata_path = pecha_dir / self.config.corpus.metadata_filename
        if not metadata_path.exists():
            return {}
        return copy.deepcopy(
            json.loads(metadata_path.read_text(encoding=self.config.corpus.text_encoding))
        )

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
