"""FastAPI microservice layer for the Semantic Search Workbench."""

from __future__ import annotations

from dataclasses import asdict
from threading import Thread
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import SemanticSearchConfig
from .service import IndexSummary, SearchHit, SearchResponse, SemanticSearchWorkbenchService
from .ui_workspace import build_citation, build_source_title, describe_score


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    query_mode: str = "DE / EN"
    include_back_translation: bool | None = None
    top_k: int | None = Field(default=None, ge=1)
    context_lines: int | None = Field(default=None, ge=0)


class ReindexRequest(BaseModel):
    force_rebuild: bool = True


def create_api_app(
    service: SemanticSearchWorkbenchService,
    initial_index_summary: IndexSummary | None = None,
    docs_enabled: bool = True,
) -> FastAPI:
    """Create the FastAPI app that exposes the workbench service boundaries."""

    app = FastAPI(
        title="Semantic Search Workbench API",
        description="Microservice endpoints for indexing and querying Tibetan transcript lines.",
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )
    index_summary = {"value": initial_index_summary}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/config")
    def runtime_config() -> dict[str, Any]:
        return service.config.as_runtime_summary()

    @app.get("/index")
    def index_status() -> dict[str, Any]:
        summary = index_summary["value"]
        if summary is None:
            summary = service.initialize_index(force_rebuild=False)
            index_summary["value"] = summary
        return serialize_index_summary(summary)

    @app.post("/index/rebuild")
    def rebuild_index(request: ReindexRequest) -> dict[str, Any]:
        summary = service.initialize_index(force_rebuild=request.force_rebuild)
        index_summary["value"] = summary
        return serialize_index_summary(summary)

    @app.post("/search")
    def search(request: SearchRequest) -> dict[str, Any]:
        response = service.search(
            request.query,
            query_mode=request.query_mode,
            include_back_translation=request.include_back_translation,
            top_k=request.top_k,
            context_lines=request.context_lines,
        )
        return serialize_search_response(response)

    return app


def create_api_app_from_config(config: SemanticSearchConfig) -> FastAPI:
    service = SemanticSearchWorkbenchService(config)
    summary = service.initialize_index(force_rebuild=False)
    return create_api_app(
        service=service,
        initial_index_summary=summary,
        docs_enabled=config.api.docs_enabled,
    )


def run_api_server(
    service: SemanticSearchWorkbenchService,
    config: SemanticSearchConfig,
    initial_index_summary: IndexSummary | None = None,
) -> None:
    import uvicorn

    app = create_api_app(
        service=service,
        initial_index_summary=initial_index_summary,
        docs_enabled=config.api.docs_enabled,
    )
    server_config = uvicorn.Config(
        app,
        host=config.api.host,
        port=config.api.port,
        log_level=config.api.log_level,
    )
    uvicorn.Server(server_config).run()


def start_api_server_thread(
    service: SemanticSearchWorkbenchService,
    config: SemanticSearchConfig,
    initial_index_summary: IndexSummary | None = None,
) -> Thread:
    thread = Thread(
        target=run_api_server,
        kwargs={
            "service": service,
            "config": config,
            "initial_index_summary": initial_index_summary,
        },
        daemon=True,
        name="semantic-search-workbench-api",
    )
    thread.start()
    return thread


def serialize_index_summary(summary: IndexSummary) -> dict[str, Any]:
    return asdict(summary)


def serialize_search_response(response: SearchResponse) -> dict[str, Any]:
    return {
        "original_query": response.original_query,
        "tibetan_retrieval_query": response.translated_query,
        "query_mode": response.query_mode,
        "top_k": response.top_k,
        "context_lines": response.context_lines,
        "results": [serialize_search_hit(hit) for hit in response.results],
    }


def serialize_search_hit(hit: SearchHit) -> dict[str, Any]:
    relevance_label, relevance_tone, relevance_hint = describe_score(hit.score)
    return {
        "rank": hit.rank,
        "score": hit.score,
        "relevance_label": relevance_label,
        "relevance_tone": relevance_tone,
        "relevance_hint": relevance_hint,
        "title": build_source_title(hit),
        "citation": build_citation(hit),
        "source_label": hit.source_label,
        "matched_line": hit.matched_line,
        "context": hit.context,
        "metadata": hit.metadata,
        "collection_metadata": hit.collection_metadata,
        "back_translation": hit.back_translation,
        "scan_url": hit.scan_url,
        "scan_filename": hit.scan_filename,
        "scan_index": hit.scan_index,
        "source_file_url": hit.source_file_url,
        "metadata_file_url": hit.metadata_file_url,
    }
