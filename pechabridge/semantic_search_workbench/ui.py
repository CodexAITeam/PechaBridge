"""Gradio UI for the Semantic Search Workbench."""

from __future__ import annotations

import html
import json

import gradio as gr

from .config import SemanticSearchConfig
from .service import IndexSummary, SearchHit, SearchResponse, SemanticSearchWorkbenchService


WORKBENCH_CSS = """
:root {
  --ssw-bg: #f5f0e8;
  --ssw-paper: #fcf8f1;
  --ssw-paper-strong: #fffdf8;
  --ssw-ink: #241a14;
  --ssw-muted: #67564c;
  --ssw-line: rgba(86, 57, 38, 0.12);
  --ssw-accent: #8b2e1e;
  --ssw-accent-soft: #efe0d1;
  --ssw-success: #295f4e;
  --ssw-warning: #9a6a12;
  --ssw-shadow: 0 18px 40px rgba(76, 50, 33, 0.08);
}

.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(139, 46, 30, 0.10), transparent 28%),
    linear-gradient(180deg, #f8f3eb 0%, #f3ede4 48%, #efe8de 100%);
}

.ssw-shell {
  padding: 10px 0 18px;
}

.ssw-hero,
.ssw-panel,
.ssw-result-card,
.ssw-empty-state {
  background: linear-gradient(180deg, rgba(255, 253, 248, 0.98), rgba(252, 248, 241, 0.98));
  border: 1px solid var(--ssw-line);
  border-radius: 22px;
  box-shadow: var(--ssw-shadow);
}

.ssw-hero {
  padding: 28px 30px;
  margin-bottom: 18px;
}

.ssw-eyebrow {
  color: var(--ssw-accent);
  font: 700 12px/1.2 "Avenir Next", "Trebuchet MS", sans-serif;
  letter-spacing: 0.18em;
  text-transform: uppercase;
}

.ssw-title {
  color: var(--ssw-ink);
  font: 700 34px/1.08 "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
  margin: 12px 0 10px;
}

.ssw-description {
  color: var(--ssw-muted);
  font: 400 16px/1.65 "Avenir Next", "Segoe UI", sans-serif;
  max-width: 72ch;
  margin: 0;
}

.ssw-panel {
  padding: 18px 20px;
}

.ssw-panel-title {
  margin: 0 0 8px;
  color: var(--ssw-ink);
  font: 700 18px/1.25 "Iowan Old Style", "Palatino Linotype", serif;
}

.ssw-panel-copy {
  color: var(--ssw-muted);
  font: 400 14px/1.55 "Avenir Next", "Segoe UI", sans-serif;
  margin: 0;
}

.ssw-status-card {
  border-radius: 18px;
  padding: 16px 18px;
  border: 1px solid var(--ssw-line);
  background: rgba(255, 251, 245, 0.92);
}

.ssw-status-card[data-tone="success"] {
  border-color: rgba(41, 95, 78, 0.22);
  background: rgba(233, 244, 239, 0.95);
}

.ssw-status-card[data-tone="warning"] {
  border-color: rgba(154, 106, 18, 0.22);
  background: rgba(249, 239, 216, 0.95);
}

.ssw-status-card[data-tone="ready"] {
  border-color: rgba(139, 46, 30, 0.15);
  background: rgba(248, 239, 232, 0.96);
}

.ssw-status-label {
  color: var(--ssw-accent);
  font: 700 11px/1.2 "Avenir Next", "Trebuchet MS", sans-serif;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  margin-bottom: 7px;
}

.ssw-status-title {
  color: var(--ssw-ink);
  font: 700 19px/1.2 "Iowan Old Style", "Palatino Linotype", serif;
  margin: 0 0 6px;
}

.ssw-status-copy,
.ssw-status-meta {
  color: var(--ssw-muted);
  font: 400 14px/1.55 "Avenir Next", "Segoe UI", sans-serif;
  margin: 0;
}

.ssw-status-meta {
  margin-top: 8px;
}

.ssw-results-head {
  display: flex;
  justify-content: space-between;
  gap: 14px;
  align-items: end;
  margin-bottom: 14px;
}

.ssw-results-title {
  margin: 0;
  color: var(--ssw-ink);
  font: 700 24px/1.2 "Iowan Old Style", "Palatino Linotype", serif;
}

.ssw-results-subtitle {
  margin: 4px 0 0;
  color: var(--ssw-muted);
  font: 400 14px/1.55 "Avenir Next", "Segoe UI", sans-serif;
}

.ssw-chip-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.ssw-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border-radius: 999px;
  border: 1px solid rgba(139, 46, 30, 0.12);
  padding: 6px 10px;
  background: var(--ssw-accent-soft);
  color: var(--ssw-accent);
  font: 700 12px/1 "Avenir Next", "Trebuchet MS", sans-serif;
}

.ssw-hit-stack {
  display: grid;
  gap: 16px;
}

.ssw-result-card {
  padding: 18px 20px 16px;
}

.ssw-card-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: start;
  margin-bottom: 14px;
}

.ssw-card-kicker {
  color: var(--ssw-accent);
  font: 700 11px/1.2 "Avenir Next", "Trebuchet MS", sans-serif;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  margin-bottom: 6px;
}

.ssw-card-source {
  color: var(--ssw-ink);
  font: 700 18px/1.35 "Iowan Old Style", "Palatino Linotype", serif;
  margin: 0;
}

.ssw-card-meta {
  color: var(--ssw-muted);
  font: 400 13px/1.5 "Avenir Next", "Segoe UI", sans-serif;
  margin-top: 4px;
}

.ssw-score-pill {
  white-space: nowrap;
  border-radius: 999px;
  padding: 8px 11px;
  background: rgba(36, 26, 20, 0.05);
  color: var(--ssw-ink);
  font: 700 12px/1 "Avenir Next", "Trebuchet MS", sans-serif;
}

.ssw-section-label {
  color: var(--ssw-accent);
  font: 700 11px/1.2 "Avenir Next", "Trebuchet MS", sans-serif;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  margin-bottom: 8px;
}

.ssw-match-line {
  border-radius: 16px;
  padding: 14px 16px;
  margin-bottom: 14px;
  border: 1px solid rgba(139, 46, 30, 0.12);
  background: linear-gradient(180deg, rgba(250, 242, 233, 0.95), rgba(247, 236, 225, 0.95));
  color: var(--ssw-ink);
  font: 500 15px/1.7 "Avenir Next", "Segoe UI", sans-serif;
}

.ssw-context-box {
  border-radius: 16px;
  border: 1px solid var(--ssw-line);
  background: rgba(255, 252, 247, 0.95);
  overflow: hidden;
}

.ssw-context-line {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 12px;
  padding: 10px 14px;
  border-top: 1px solid rgba(86, 57, 38, 0.06);
  color: var(--ssw-muted);
  font: 400 13px/1.6 "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
}

.ssw-context-line:first-child {
  border-top: 0;
}

.ssw-context-line.is-hit {
  background: rgba(139, 46, 30, 0.08);
  color: var(--ssw-ink);
}

.ssw-context-marker {
  color: var(--ssw-accent);
  font-weight: 700;
}

.ssw-context-text {
  white-space: pre-wrap;
}

.ssw-details {
  margin-top: 12px;
  border-radius: 14px;
  border: 1px solid rgba(86, 57, 38, 0.10);
  background: rgba(255, 253, 249, 0.92);
  overflow: hidden;
}

.ssw-details summary {
  cursor: pointer;
  list-style: none;
  padding: 12px 14px;
  color: var(--ssw-ink);
  font: 700 13px/1.3 "Avenir Next", "Trebuchet MS", sans-serif;
}

.ssw-details summary::-webkit-details-marker {
  display: none;
}

.ssw-details-body {
  border-top: 1px solid rgba(86, 57, 38, 0.08);
  padding: 0 14px 14px;
}

.ssw-pre {
  margin: 0;
  white-space: pre-wrap;
  color: var(--ssw-muted);
  font: 400 13px/1.65 "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
}

.ssw-empty-state {
  padding: 22px 24px;
}

.ssw-empty-title {
  margin: 0 0 8px;
  color: var(--ssw-ink);
  font: 700 22px/1.2 "Iowan Old Style", "Palatino Linotype", serif;
}

.ssw-empty-copy,
.ssw-empty-list {
  color: var(--ssw-muted);
  font: 400 14px/1.65 "Avenir Next", "Segoe UI", sans-serif;
  margin: 0;
}

.ssw-empty-list {
  margin-top: 10px;
  padding-left: 18px;
}

@media (max-width: 900px) {
  .ssw-results-head,
  .ssw-card-head {
    display: block;
  }

  .ssw-score-pill {
    display: inline-flex;
    margin-top: 10px;
  }
}
"""


def build_workbench(
    service: SemanticSearchWorkbenchService,
    initial_index_summary: IndexSummary,
) -> gr.Blocks:
    config = service.config

    def on_search(
        query: str,
        include_back_translation: bool,
        top_k: int,
        context_lines: int,
        progress: gr.Progress = gr.Progress(track_tqdm=False),
    ) -> tuple[str, str, str]:
        try:
            response = service.search(
                query,
                include_back_translation=include_back_translation,
                top_k=top_k,
                context_lines=context_lines,
                progress_callback=lambda value, desc: progress(value, desc=desc),
            )
        except Exception as exc:
            raise gr.Error(str(exc)) from exc
        return (
            response.translated_query,
            _render_search_results(response),
            _render_activity_status(
                tone="success" if response.results else "warning",
                label="Search Activity",
                title="Search complete" if response.results else "No direct matches found",
                copy=(
                    f"Returned {len(response.results)} hit(s) using top_k={response.top_k} "
                    f"and context window radius={response.context_lines}."
                ),
                meta=(
                    "The translated Tibetan query is shown below so the retrieval step remains auditable."
                    if response.results
                    else "Try a broader concept, reduce specificity, or check whether the translated Tibetan query needs adjustment."
                ),
            ),
        )

    def on_reindex(progress: gr.Progress = gr.Progress(track_tqdm=False)) -> tuple[str, str]:
        try:
            summary = service.initialize_index(
                force_rebuild=True,
                progress_callback=lambda value, desc: progress(value, desc=desc),
            )
        except Exception as exc:
            raise gr.Error(str(exc)) from exc
        return (
            _render_index_summary(summary),
            _render_activity_status(
                tone="success",
                label="Index Activity",
                title="Index rebuild finished",
                copy=(
                    f"Collection `{summary.collection_name}` now contains {summary.indexed_records} indexed lines "
                    f"from {summary.file_count} source files."
                ),
                meta="The workbench is ready for the next query.",
            ),
        )

    with gr.Blocks(title=config.ui.title, css=WORKBENCH_CSS) as app:
        gr.HTML(_render_hero(config))

        with gr.Column(elem_classes=["ssw-shell"]):
            with gr.Row(equal_height=False):
                with gr.Column(scale=7, min_width=420):
                    gr.HTML(
                        _render_panel_intro(
                            title="Search Controls",
                            copy=(
                                "Start from German, English, or Tibetan. The workbench translates the query into "
                                "Classical Tibetan, runs dense vector search, and returns the matching line with local context."
                            ),
                        )
                    )
                    query_input = gr.Textbox(
                        label="Search Query (DE / EN / Tibetan)",
                        lines=3,
                        placeholder="Enter a concept, phrase, deity name, doctrinal topic, or formula to search across the indexed transcripts.",
                    )
                    include_back_translation = gr.Checkbox(
                        label="Include English back-translation of each context window",
                        value=config.search.include_back_translation_default,
                    )
                    with gr.Accordion("Search Settings", open=False):
                        with gr.Row():
                            top_k = gr.Slider(
                                label="Results to return",
                                minimum=1,
                                maximum=15,
                                step=1,
                                value=config.search.top_k,
                            )
                            context_lines = gr.Slider(
                                label="Context lines on each side",
                                minimum=0,
                                maximum=8,
                                step=1,
                                value=config.chunking.context_lines,
                            )
                    with gr.Row():
                        search_button = gr.Button("Run Semantic Search", variant="primary")
                        reindex_button = gr.Button("Rebuild Index")
                    gr.HTML(
                        _render_panel_intro(
                            title="Research Note",
                            copy=(
                                "The translated Tibetan query is surfaced explicitly, so the retrieval step stays "
                                "inspectable rather than hidden behind a single search box."
                            ),
                        )
                    )

                with gr.Column(scale=5, min_width=320):
                    gr.HTML(
                        _render_panel_intro(
                            title="Workbench Status",
                            copy="Monitor the local Qdrant collection and the most recent UI action from here.",
                        )
                    )
                    index_status = gr.HTML(_render_index_summary(initial_index_summary))
                    activity_status = gr.HTML(
                        _render_activity_status(
                            tone="ready",
                            label="Search Activity",
                            title="Ready to search",
                            copy=(
                                f"Local collection `{initial_index_summary.collection_name}` is available. "
                                "Enter a query to begin cross-lingual semantic retrieval."
                            ),
                            meta="No search has been executed in this session yet.",
                        )
                    )

            translated_query = gr.Textbox(
                label="Translated Classical Tibetan Query",
                interactive=False,
                placeholder="The workbench will display the Tibetan query that was actually embedded for retrieval.",
            )
            results_html = gr.HTML(_render_initial_results_state())

        search_button.click(
            fn=on_search,
            inputs=[query_input, include_back_translation, top_k, context_lines],
            outputs=[translated_query, results_html, activity_status],
        )
        query_input.submit(
            fn=on_search,
            inputs=[query_input, include_back_translation, top_k, context_lines],
            outputs=[translated_query, results_html, activity_status],
        )
        reindex_button.click(
            fn=on_reindex,
            outputs=[index_status, activity_status],
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


def _render_hero(config: SemanticSearchConfig) -> str:
    return (
        '<section class="ssw-hero">'
        '<div class="ssw-eyebrow">Semantic Search Workbench</div>'
        f'<h1 class="ssw-title">{html.escape(config.ui.title)}</h1>'
        f'<p class="ssw-description">{html.escape(config.ui.description)}</p>'
        "</section>"
    )


def _render_panel_intro(title: str, copy: str) -> str:
    return (
        '<section class="ssw-panel">'
        f'<h2 class="ssw-panel-title">{html.escape(title)}</h2>'
        f'<p class="ssw-panel-copy">{html.escape(copy)}</p>'
        "</section>"
    )


def _render_activity_status(
    tone: str,
    label: str,
    title: str,
    copy: str,
    meta: str | None = None,
) -> str:
    meta_html = (
        f'<p class="ssw-status-meta">{html.escape(meta)}</p>'
        if meta
        else ""
    )
    return (
        f'<section class="ssw-status-card" data-tone="{html.escape(tone)}">'
        f'<div class="ssw-status-label">{html.escape(label)}</div>'
        f'<h3 class="ssw-status-title">{html.escape(title)}</h3>'
        f'<p class="ssw-status-copy">{html.escape(copy)}</p>'
        f"{meta_html}"
        "</section>"
    )


def _render_index_summary(summary: IndexSummary) -> str:
    action = "rebuilt" if summary.rebuilt else "loaded"
    return _render_activity_status(
        tone="success" if summary.indexed_records else "warning",
        label="Index Status",
        title=f"Collection {summary.collection_name}",
        copy=(
            f"{summary.indexed_records} indexed lines across {summary.pecha_count} pecha folders "
            f"and {summary.file_count} source files."
        ),
        meta=f"Last action: {action}.",
    )


def _render_search_results(response: SearchResponse) -> str:
    if not response.results:
        return _render_empty_results_state(response)

    chips = "".join(
        [
            f'<span class="ssw-chip">Original query: {html.escape(response.original_query)}</span>',
            f'<span class="ssw-chip">Top K: {response.top_k}</span>',
            f'<span class="ssw-chip">Context radius: {response.context_lines}</span>',
        ]
    )
    cards = "".join(_render_hit_card(hit) for hit in response.results)
    return (
        '<section class="ssw-panel">'
        '<div class="ssw-results-head">'
        '<div>'
        '<h2 class="ssw-results-title">Search Results</h2>'
        '<p class="ssw-results-subtitle">'
        'Each hit shows the matched line, its local textual neighborhood, and optional scholarly support material.'
        "</p>"
        "</div>"
        f'<div class="ssw-chip-row">{chips}</div>'
        "</div>"
        f'<div class="ssw-hit-stack">{cards}</div>'
        "</section>"
    )


def _render_initial_results_state() -> str:
    return (
        '<section class="ssw-empty-state">'
        '<h2 class="ssw-empty-title">Ready for the first query</h2>'
        '<p class="ssw-empty-copy">'
        "The workbench will show the translated Tibetan query, ranked matches, context windows, and optional back-translations here."
        "</p>"
        '<ul class="ssw-empty-list">'
        "<li>Use a concept, phrase, topic, or proper name.</li>"
        "<li>Broader prompts often retrieve better than full natural-language questions.</li>"
        "<li>Use the search settings when you need more results or a wider local context.</li>"
        "</ul>"
        "</section>"
    )


def _render_empty_results_state(response: SearchResponse) -> str:
    return (
        '<section class="ssw-empty-state">'
        '<h2 class="ssw-empty-title">No matching passages found</h2>'
        '<p class="ssw-empty-copy">'
        f'The translated Tibetan query was <strong>{html.escape(response.translated_query)}</strong>. '
        "That makes the failure inspectable rather than opaque."
        "</p>"
        '<ul class="ssw-empty-list">'
        "<li>Try a broader or shorter concept phrase.</li>"
        "<li>Reduce specificity if the query contains several constraints.</li>"
        "<li>Expand Top K or increase the context radius to review near misses.</li>"
        "</ul>"
        "</section>"
    )


def _render_hit_card(hit: SearchHit) -> str:
    source_meta = (
        f"Page {html.escape(str(hit.metadata.get('page_number', '?')))} · "
        f"Line {html.escape(str(hit.metadata.get('line_number', '?')))}"
    )
    details_blocks: list[str] = []
    if hit.back_translation:
        details_blocks.append(
            _render_details_block(
                title="English back-translation",
                body=_render_preformatted(hit.back_translation),
            )
        )
    collection_metadata = hit.metadata.get("collection_metadata") or {}
    if collection_metadata:
        details_blocks.append(
            _render_details_block(
                title="Collection metadata",
                body=_render_preformatted(
                    json.dumps(collection_metadata, ensure_ascii=False, indent=2)
                ),
            )
        )

    return (
        '<article class="ssw-result-card">'
        '<div class="ssw-card-head">'
        '<div>'
        f'<div class="ssw-card-kicker">Hit {hit.rank}</div>'
        f'<h3 class="ssw-card-source">{html.escape(hit.source_label)}</h3>'
        f'<div class="ssw-card-meta">{source_meta}</div>'
        "</div>"
        f'<div class="ssw-score-pill">Score {hit.score:.4f}</div>'
        "</div>"
        '<div class="ssw-section-label">Matched Line</div>'
        f'<div class="ssw-match-line">{html.escape(hit.matched_line)}</div>'
        '<div class="ssw-section-label">Context Window</div>'
        f'{_render_context_box(hit.context)}'
        f'{"".join(details_blocks)}'
        "</article>"
    )


def _render_context_box(context: str) -> str:
    rows: list[str] = []
    for line in context.splitlines():
        if not line.strip():
            continue
        marker = line[:1]
        remainder = line[1:].lstrip() if marker in {">", " "} else line
        css_class = "ssw-context-line is-hit" if marker == ">" else "ssw-context-line"
        rows.append(
            f'<div class="{css_class}">'
            f'<span class="ssw-context-marker">{html.escape(marker if marker == ">" else "·")}</span>'
            f'<span class="ssw-context-text">{html.escape(remainder)}</span>'
            "</div>"
        )
    rows_html = "".join(rows) if rows else '<div class="ssw-context-line"><span class="ssw-context-marker">·</span><span class="ssw-context-text">No context available.</span></div>'
    return f'<div class="ssw-context-box">{rows_html}</div>'


def _render_details_block(title: str, body: str) -> str:
    return (
        '<details class="ssw-details">'
        f'<summary>{html.escape(title)}</summary>'
        f'<div class="ssw-details-body">{body}</div>'
        "</details>"
    )


def _render_preformatted(content: str) -> str:
    return f'<pre class="ssw-pre">{html.escape(content)}</pre>'
