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
