"""CLI entrypoint for the Semantic Search Workbench."""

from __future__ import annotations

import argparse
import logging
from typing import Sequence

from .config import SemanticSearchConfig

LOGGER = logging.getLogger("pechabridge.semantic_search_workbench")


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        add_help=add_help,
        description="Launch the Semantic Search Workbench for historical Tibetan transcripts.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to semantic-search-config.yaml",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force a full rebuild of the Qdrant collection before launching Gradio.",
    )
    parser.add_argument(
        "--reindex-only",
        action="store_true",
        help="Rebuild the index and exit without launching the Gradio workbench.",
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Start only the FastAPI workbench microservice and do not launch Gradio.",
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Disable the FastAPI microservice for this run even if api.enabled is true in the config.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    from .service import SemanticSearchWorkbenchService
    from .ui import build_workbench, launch_workbench

    config = SemanticSearchConfig.from_file(args.config)
    service = SemanticSearchWorkbenchService(config)
    summary = service.initialize_index(force_rebuild=bool(args.reindex or args.reindex_only))

    LOGGER.info("Semantic Search Workbench config: %s", config.as_runtime_summary())
    LOGGER.info(
        "Qdrant collection '%s' ready with %s records.",
        summary.collection_name,
        summary.indexed_records,
    )

    if args.reindex_only:
        return 0

    api_enabled = bool(config.api.enabled and not args.no_api)
    if args.api_only:
        from .api import run_api_server

        LOGGER.info(
            "Starting Semantic Search Workbench API on %s:%s.",
            config.api.host,
            config.api.port,
        )
        run_api_server(service=service, config=config, initial_index_summary=summary)
        return 0

    if api_enabled:
        from .api import start_api_server_thread

        start_api_server_thread(service=service, config=config, initial_index_summary=summary)
        LOGGER.info(
            "Semantic Search Workbench API available on http://%s:%s.",
            config.api.host,
            config.api.port,
        )

    app = build_workbench(service=service, initial_index_summary=summary)
    launch_workbench(app=app, config=config)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = create_parser()
    args = parser.parse_args(argv)
    return run(args)
