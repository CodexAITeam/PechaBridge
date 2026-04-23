"""Configuration loading for the Semantic Search Workbench."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def _resolve_path(config_dir: Path, value: str | None) -> Path | None:
    if value in {None, ""}:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    return path


@dataclass(frozen=True)
class EnvironmentConfig:
    dotenv_path: Path | None
    required_env_vars: tuple[str, ...]


@dataclass(frozen=True)
class CorpusConfig:
    transcripts_root: Path
    file_glob: str
    metadata_filename: str
    text_encoding: str
    page_number_pattern: str


@dataclass(frozen=True)
class ChunkingConfig:
    split_separator: str
    strip_lines: bool
    keep_empty_lines: bool
    min_line_length: int
    context_lines: int


@dataclass(frozen=True)
class EmbeddingConfig:
    huggingface_model_id: str
    device: str
    batch_size: int
    max_length: int
    normalize_embeddings: bool
    trust_remote_code: bool
    cache_dir: Path | None
    document_prefix: str
    query_prefix: str


@dataclass(frozen=True)
class QdrantConfig:
    storage_path: Path
    collection_name: str
    distance: str
    force_recreate_collection: bool


@dataclass(frozen=True)
class TranslationConfig:
    enabled: bool
    provider: str
    model: str
    api_key_env: str
    base_url_env: str | None
    organization_env: str | None
    max_output_tokens: int
    temperature: float
    translate_query_prompt: str
    back_translate_prompt: str


@dataclass(frozen=True)
class SearchConfig:
    top_k: int
    score_threshold: float | None
    rebuild_index_on_start: bool
    include_back_translation_default: bool


@dataclass(frozen=True)
class UIConfig:
    title: str
    description: str
    server_name: str
    server_port: int
    share: bool
    show_api: bool
    concurrency_limit: int


@dataclass(frozen=True)
class APIConfig:
    enabled: bool
    host: str
    port: int
    docs_enabled: bool
    log_level: str


@dataclass(frozen=True)
class SemanticSearchConfig:
    config_path: Path
    environment: EnvironmentConfig
    corpus: CorpusConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    qdrant: QdrantConfig
    translation: TranslationConfig
    search: SearchConfig
    ui: UIConfig
    api: APIConfig

    @classmethod
    def from_file(cls, config_path: str | os.PathLike[str]) -> "SemanticSearchConfig":
        config_path = Path(config_path).expanduser().resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        config_dir = config_path.parent

        environment_raw = raw["environment"]
        environment = EnvironmentConfig(
            dotenv_path=_resolve_path(config_dir, environment_raw.get("dotenv_path")),
            required_env_vars=tuple(environment_raw.get("required_env_vars", [])),
        )
        if environment.dotenv_path is not None:
            load_dotenv(environment.dotenv_path, override=False)

        missing_vars = [name for name in environment.required_env_vars if not os.getenv(name)]
        if missing_vars:
            missing = ", ".join(sorted(missing_vars))
            raise RuntimeError(
                f"Missing required environment variables for Semantic Search Workbench: {missing}"
            )

        corpus_raw = raw["corpus"]
        chunking_raw = raw["chunking"]
        embedding_raw = raw["embedding"]
        qdrant_raw = raw["qdrant"]
        translation_raw = raw["translation"]
        search_raw = raw["search"]
        ui_raw = raw["ui"]
        api_raw = raw.get("api", {})

        return cls(
            config_path=config_path,
            environment=environment,
            corpus=CorpusConfig(
                transcripts_root=_resolve_path(config_dir, corpus_raw["transcripts_root"]),
                file_glob=corpus_raw["file_glob"],
                metadata_filename=corpus_raw["metadata_filename"],
                text_encoding=corpus_raw["text_encoding"],
                page_number_pattern=corpus_raw["page_number_pattern"],
            ),
            chunking=ChunkingConfig(
                split_separator=chunking_raw["split_separator"],
                strip_lines=bool(chunking_raw["strip_lines"]),
                keep_empty_lines=bool(chunking_raw["keep_empty_lines"]),
                min_line_length=int(chunking_raw["min_line_length"]),
                context_lines=int(chunking_raw["context_lines"]),
            ),
            embedding=EmbeddingConfig(
                huggingface_model_id=embedding_raw["huggingface_model_id"],
                device=embedding_raw["device"],
                batch_size=int(embedding_raw["batch_size"]),
                max_length=int(embedding_raw["max_length"]),
                normalize_embeddings=bool(embedding_raw["normalize_embeddings"]),
                trust_remote_code=bool(embedding_raw["trust_remote_code"]),
                cache_dir=_resolve_path(config_dir, embedding_raw.get("cache_dir")),
                document_prefix=embedding_raw["document_prefix"],
                query_prefix=embedding_raw["query_prefix"],
            ),
            qdrant=QdrantConfig(
                storage_path=_resolve_path(config_dir, qdrant_raw["storage_path"]),
                collection_name=qdrant_raw["collection_name"],
                distance=qdrant_raw["distance"],
                force_recreate_collection=bool(qdrant_raw["force_recreate_collection"]),
            ),
            translation=TranslationConfig(
                enabled=bool(translation_raw["enabled"]),
                provider=translation_raw["provider"],
                model=translation_raw["model"],
                api_key_env=translation_raw["api_key_env"],
                base_url_env=translation_raw.get("base_url_env"),
                organization_env=translation_raw.get("organization_env"),
                max_output_tokens=int(translation_raw["max_output_tokens"]),
                temperature=float(translation_raw["temperature"]),
                translate_query_prompt=translation_raw["translate_query_prompt"],
                back_translate_prompt=translation_raw["back_translate_prompt"],
            ),
            search=SearchConfig(
                top_k=int(search_raw["top_k"]),
                score_threshold=(
                    float(search_raw["score_threshold"])
                    if search_raw.get("score_threshold") is not None
                    else None
                ),
                rebuild_index_on_start=bool(search_raw["rebuild_index_on_start"]),
                include_back_translation_default=bool(search_raw["include_back_translation_default"]),
            ),
            ui=UIConfig(
                title=ui_raw["title"],
                description=ui_raw["description"],
                server_name=ui_raw["server_name"],
                server_port=int(ui_raw["server_port"]),
                share=bool(ui_raw["share"]),
                show_api=bool(ui_raw["show_api"]),
                concurrency_limit=int(ui_raw["concurrency_limit"]),
            ),
            api=APIConfig(
                enabled=bool(api_raw.get("enabled", False)),
                host=str(api_raw.get("host", ui_raw["server_name"])),
                port=int(api_raw.get("port", int(ui_raw["server_port"]) + 1)),
                docs_enabled=bool(api_raw.get("docs_enabled", True)),
                log_level=str(api_raw.get("log_level", "info")),
            ),
        )

    def as_runtime_summary(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "transcripts_root": str(self.corpus.transcripts_root),
            "qdrant_storage_path": str(self.qdrant.storage_path),
            "collection_name": self.qdrant.collection_name,
            "embedding_model": self.embedding.huggingface_model_id,
            "translation_model": self.translation.model,
            "server_name": self.ui.server_name,
            "server_port": self.ui.server_port,
            "api_enabled": self.api.enabled,
            "api_host": self.api.host,
            "api_port": self.api.port,
        }
