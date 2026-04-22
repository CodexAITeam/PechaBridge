"""Gradio UI for the Semantic Search Workbench."""

from __future__ import annotations

import json

import gradio as gr

from .config import SemanticSearchConfig
from .service import IndexSummary, SearchResponse, SemanticSearchWorkbenchService


def build_workbench(
    service: SemanticSearchWorkbenchService,
    initial_index_summary: IndexSummary,
) -> gr.Blocks:
    config = service.config

    def on_search(query: str, include_back_translation: bool) -> tuple[str, str]:
        try:
            response = service.search(query, include_back_translation=include_back_translation)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc
        return response.translated_query, _render_search_results(response)

    def on_reindex() -> str:
        try:
            summary = service.initialize_index(force_rebuild=True)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc
        return _render_index_summary(summary)

    with gr.Blocks(title=config.ui.title) as app:
        gr.Markdown(f"# {config.ui.title}\n\n{config.ui.description}")

        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Search Query (DE / EN / Tibetan)",
                    lines=3,
                    placeholder="Enter a German or English concept to translate into Classical Tibetan and search across the indexed transcripts.",
                )
                include_back_translation = gr.Checkbox(
                    label="Include English back-translation of each context window",
                    value=config.search.include_back_translation_default,
                )
                with gr.Row():
                    search_button = gr.Button("Run Semantic Search", variant="primary")
                    reindex_button = gr.Button("Rebuild Index")

            with gr.Column(scale=2):
                index_status = gr.Markdown(_render_index_summary(initial_index_summary))

        translated_query = gr.Textbox(
            label="Translated Classical Tibetan Query",
            interactive=False,
        )
        results_markdown = gr.Markdown("No search executed yet.")

        search_button.click(
            fn=on_search,
            inputs=[query_input, include_back_translation],
            outputs=[translated_query, results_markdown],
        )
        query_input.submit(
            fn=on_search,
            inputs=[query_input, include_back_translation],
            outputs=[translated_query, results_markdown],
        )
        reindex_button.click(
            fn=on_reindex,
            outputs=[index_status],
        )

    return app


def launch_workbench(app: gr.Blocks, config: SemanticSearchConfig) -> None:
    queued_app = app.queue(default_concurrency_limit=config.ui.concurrency_limit)
    queued_app.launch(
        server_name=config.ui.server_name,
        server_port=config.ui.server_port,
        share=config.ui.share,
        show_api=config.ui.show_api,
    )


def _render_index_summary(summary: IndexSummary) -> str:
    action = "rebuilt" if summary.rebuilt else "loaded"
    return (
        "## Index Status\n\n"
        f"- Collection: `{summary.collection_name}`\n"
        f"- Records: `{summary.indexed_records}`\n"
        f"- Pechas: `{summary.pecha_count}`\n"
        f"- Source files: `{summary.file_count}`\n"
        f"- Last action: `{action}`"
    )


def _render_search_results(response: SearchResponse) -> str:
    if not response.results:
        return "No matching passages were found for the translated Tibetan query."

    rendered_hits: list[str] = []
    for hit in response.results:
        block = [
            f"## Hit {hit.rank}",
            f"**Source:** `{hit.source_label}`",
            f"**Score:** `{hit.score:.4f}`",
            "",
            "**Matched line:**",
            "",
            "```text",
            hit.matched_line,
            "```",
            "",
            "**Context window:**",
            "",
            "```text",
            hit.context,
            "```",
        ]
        if hit.back_translation:
            block.extend(
                [
                    "",
                    "**English back-translation:**",
                    "",
                    "```text",
                    hit.back_translation,
                    "```",
                ]
            )
        collection_metadata = hit.metadata.get("collection_metadata") or {}
        if collection_metadata:
            block.extend(
                [
                    "",
                    "**Collection metadata:**",
                    "",
                    "```json",
                    json.dumps(collection_metadata, ensure_ascii=False, indent=2),
                    "```",
                ]
            )
        rendered_hits.append("\n".join(block))
    return "\n\n".join(rendered_hits)
