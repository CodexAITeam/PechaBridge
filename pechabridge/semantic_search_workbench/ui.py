"""Gradio UI for the Semantic Search Workbench."""

from __future__ import annotations

from collections.abc import Iterator
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

.ssw-score-pill[data-tone="high"] {
  background: rgba(41, 95, 78, 0.14);
  color: var(--ssw-success);
}

.ssw-score-pill[data-tone="medium"] {
  background: rgba(154, 106, 18, 0.14);
  color: var(--ssw-warning);
}

.ssw-score-pill[data-tone="low"] {
  background: rgba(139, 46, 30, 0.12);
  color: var(--ssw-accent);
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

.ssw-source-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 10px;
  margin: 0 0 14px;
}

.ssw-source-item {
  border-radius: 14px;
  border: 1px solid rgba(86, 57, 38, 0.10);
  background: rgba(255, 252, 247, 0.96);
  padding: 11px 12px;
}

.ssw-source-item-label {
  color: var(--ssw-accent);
  font: 700 10px/1.2 "Avenir Next", "Trebuchet MS", sans-serif;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  margin-bottom: 6px;
}

.ssw-source-item-value {
  color: var(--ssw-ink);
  font: 600 13px/1.5 "Avenir Next", "Segoe UI", sans-serif;
  word-break: break-word;
}

.ssw-source-item-value code {
  font: inherit;
  color: inherit;
  background: rgba(36, 26, 20, 0.05);
  border-radius: 8px;
  padding: 2px 6px;
}

.ssw-link-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 0 0 14px;
}

.ssw-link-chip {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  border: 1px solid rgba(139, 46, 30, 0.14);
  padding: 7px 11px;
  background: rgba(255, 248, 240, 0.94);
  color: var(--ssw-accent);
  font: 700 12px/1 "Avenir Next", "Trebuchet MS", sans-serif;
  text-decoration: none;
}

.ssw-link-chip:hover {
  text-decoration: underline;
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

.ssw-scan-panel {
  margin-top: 14px;
}

.ssw-scan-missing {
  margin-top: 14px;
  border-radius: 16px;
  border: 1px dashed rgba(154, 106, 18, 0.28);
  background: rgba(249, 239, 216, 0.55);
  padding: 13px 14px;
  color: var(--ssw-muted);
  font: 400 13px/1.55 "Avenir Next", "Segoe UI", sans-serif;
}

.ssw-scan-frame {
  display: block;
  border-radius: 18px;
  overflow: hidden;
  border: 1px solid rgba(86, 57, 38, 0.12);
  background: rgba(255, 253, 248, 0.96);
  box-shadow: 0 12px 24px rgba(76, 50, 33, 0.08);
}

.ssw-scan-image {
  display: block;
  width: 100%;
  max-height: 210px;
  object-fit: contain;
  background: linear-gradient(180deg, rgba(245, 240, 232, 0.85), rgba(238, 231, 221, 0.90));
}

.ssw-scan-caption {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 10px;
  align-items: center;
  padding: 10px 12px 12px;
  color: var(--ssw-muted);
  font: 400 13px/1.55 "Avenir Next", "Segoe UI", sans-serif;
}

.ssw-scan-caption a {
  color: var(--ssw-accent);
  text-decoration: none;
  font-weight: 700;
}

.ssw-scan-caption a:hover {
  text-decoration: underline;
}

.ssw-pre {
  margin: 0;
  white-space: pre-wrap;
  color: var(--ssw-muted);
  font: 400 13px/1.65 "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
}

.ssw-loading-state {
  padding: 22px 24px;
}

.ssw-loading-list {
  margin: 10px 0 0;
  padding-left: 18px;
  color: var(--ssw-muted);
  font: 400 14px/1.65 "Avenir Next", "Segoe UI", sans-serif;
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

.ssw-workspace-card,
.ssw-compare-card {
  border-radius: 18px;
  border: 1px solid var(--ssw-line);
  background: rgba(255, 253, 248, 0.96);
  box-shadow: 0 12px 24px rgba(76, 50, 33, 0.06);
}

.ssw-workspace-card {
  padding: 18px 20px;
}

.ssw-focus-layout {
  display: grid;
  grid-template-columns: minmax(0, 1.15fr) minmax(280px, 0.85fr);
  gap: 18px;
  align-items: start;
}

.ssw-focus-title,
.ssw-compare-title {
  color: var(--ssw-ink);
  font: 700 20px/1.3 "Iowan Old Style", "Palatino Linotype", serif;
  margin: 0 0 6px;
}

.ssw-focus-card .ssw-scan-image {
  max-height: 420px;
}

.ssw-compare-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 14px;
}

.ssw-compare-card {
  padding: 15px 16px;
}

.ssw-compare-meta,
.ssw-export-copy {
  color: var(--ssw-muted);
  font: 400 13px/1.55 "Avenir Next", "Segoe UI", sans-serif;
  margin: 0 0 10px;
}

.ssw-compare-card .ssw-context-line {
  padding: 8px 10px;
  font-size: 12px;
}

@media (max-width: 900px) {
  .ssw-results-head,
  .ssw-card-head,
  .ssw-focus-layout {
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
        query_mode: str,
        include_back_translation: bool,
        top_k: int,
        context_lines: int,
        progress: gr.Progress = gr.Progress(track_tqdm=False),
    ) -> Iterator[
        tuple[
            str,
            str,
            str,
            SearchResponse | None,
            gr.Dropdown,
            gr.CheckboxGroup,
            gr.Dropdown,
            str,
            str,
            str,
            str,
        ]
    ]:
        normalized_query = query.strip()
        if not normalized_query:
            raise gr.Error("Please enter a search query.")

        yield (
            "",
            _render_loading_results_state(query_mode, top_k, context_lines),
            _render_activity_status(
                tone="ready",
                label="Search Activity",
                title="Search in progress",
                copy=(
                    f"Preparing a {query_mode} query against {config.qdrant.collection_name} "
                    f"with top_k={int(top_k)} and context radius={int(context_lines)}."
                ),
                meta=(
                    "The progress bar will walk through query conversion or translation, vector retrieval, "
                    "context assembly, and optional back-translation."
                ),
            ),
            None,
            gr.update(choices=["All pechas"], value="All pechas", interactive=False),
            gr.update(choices=[], value=[], interactive=False),
            gr.update(choices=[], value=None, interactive=False),
            _render_focus_empty_state(),
            _render_compare_empty_state(),
            "",
            "",
        )
        try:
            response = service.search(
                normalized_query,
                query_mode=query_mode,
                include_back_translation=include_back_translation,
                top_k=top_k,
                context_lines=context_lines,
                progress_callback=lambda value, desc: progress(value, desc=desc),
            )
        except Exception as exc:
            raise gr.Error(str(exc)) from exc
        yield (
            response.translated_query,
            _render_search_results(response),
            _render_search_activity_status(response),
            response,
            gr.update(
                choices=_build_pecha_filter_choices(response),
                value="All pechas",
                interactive=bool(response.results),
            ),
            gr.update(
                choices=_build_hit_choices(response),
                value=[],
                interactive=bool(response.results),
            ),
            gr.update(
                choices=_build_hit_choices(response),
                value=_first_hit_choice(response),
                interactive=bool(response.results),
            ),
            _render_focus_panel(_first_hit(response)),
            _render_compare_empty_state(response),
            "",
            "",
        )

    def on_reindex(
        progress: gr.Progress = gr.Progress(track_tqdm=False),
    ) -> Iterator[tuple[str, str]]:
        yield (
            _render_activity_status(
                tone="warning",
                label="Index Status",
                title="Rebuilding collection",
                copy=(
                    f"Reloading transcripts and metadata for {config.qdrant.collection_name} "
                    "before writing fresh embeddings."
                ),
                meta="Use this after transcript files or metadata.json entries have changed.",
            ),
            _render_activity_status(
                tone="ready",
                label="Index Activity",
                title="Index rebuild in progress",
                copy="Scanning pecha folders, resolving metadata, and refreshing the local Qdrant collection.",
                meta="The progress bar shows the current ingestion and embedding phase.",
            ),
        )
        try:
            summary = service.initialize_index(
                force_rebuild=True,
                progress_callback=lambda value, desc: progress(value, desc=desc),
            )
        except Exception as exc:
            raise gr.Error(str(exc)) from exc
        yield (
            _render_index_summary(summary),
            _render_activity_status(
                tone="success",
                label="Index Activity",
                title="Index rebuild finished",
                copy=(
                    f"Collection {summary.collection_name} now contains {summary.indexed_records} indexed lines "
                    f"from {summary.file_count} source files."
                ),
                meta="The workbench is ready for the next query.",
            ),
        )

    def on_filter_results(
        pecha_filter: str,
        response: SearchResponse | None,
    ) -> tuple[gr.CheckboxGroup, gr.Dropdown, str, str, str, str]:
        if response is None:
            return (
                gr.update(choices=[], value=[], interactive=False),
                gr.update(choices=[], value=None, interactive=False),
                _render_focus_empty_state(),
                _render_compare_empty_state(),
                "",
                "",
            )

        choices = _build_hit_choices(response, pecha_filter=pecha_filter)
        first_choice = choices[0] if choices else None
        focus_hit = _resolve_focus_hit(response, first_choice)
        return (
            gr.update(choices=choices, value=[], interactive=bool(choices)),
            gr.update(choices=choices, value=first_choice, interactive=bool(choices)),
            _render_focus_panel(focus_hit),
            _render_compare_empty_state(response, pecha_filter=pecha_filter),
            "",
            "",
        )

    def on_update_workspace(
        selected_hit_labels: list[str] | None,
        focused_hit_label: str | None,
        response: SearchResponse | None,
    ) -> tuple[str, str, str, str]:
        if response is None:
            return (_render_focus_empty_state(), _render_compare_empty_state(), "", "")

        selected_hits = _resolve_selected_hits(response, selected_hit_labels)
        focus_hit = _resolve_focus_hit(response, focused_hit_label)
        if focus_hit is None and selected_hits:
            focus_hit = selected_hits[0]
        if focus_hit is None:
            focus_hit = _first_hit(response)

        comparison_hits = selected_hits[:5]
        return (
            _render_focus_panel(focus_hit),
            _render_compare_panel(response, comparison_hits),
            _render_export_markdown(response, comparison_hits),
            _render_export_json(response, comparison_hits),
        )

    def on_clear_workspace(
        response: SearchResponse | None,
    ) -> tuple[gr.CheckboxGroup, gr.Dropdown, str, str, str, str]:
        first_choice = _first_hit_choice(response) if response is not None else None
        return (
            gr.update(value=[]),
            gr.update(value=first_choice),
            _render_focus_panel(_first_hit(response) if response is not None else None),
            _render_compare_empty_state(response),
            "",
            "",
        )

    with gr.Blocks(title=config.ui.title) as app:
        search_response_state = gr.State(None)
        gr.HTML(_render_hero(config))

        with gr.Column(elem_classes=["ssw-shell"]):
            with gr.Row(equal_height=False):
                with gr.Column(scale=7, min_width=420):
                    gr.HTML(
                        _render_panel_intro(
                            title="Search Controls",
                            copy=(
                                "Choose the input mode explicitly: `DE / EN` translates first, `Tibetan` uses the query directly, "
                                "and `Wylie (EWTS)` is converted locally before retrieval."
                            ),
                        )
                    )
                    query_input = gr.Textbox(
                        label="Search Query",
                        lines=3,
                        placeholder="Enter a concept, Tibetan string, or Wylie (EWTS) query.",
                    )
                    query_mode = gr.Radio(
                        label="Query Input Mode",
                        choices=[
                            "DE / EN",
                            "Tibetan",
                            "Wylie (EWTS)",
                        ],
                        value="DE / EN",
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
                                "The Tibetan query used for retrieval is surfaced explicitly, so the retrieval step stays "
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
                                f"Local collection {initial_index_summary.collection_name} is available. "
                                "Enter a query to begin cross-lingual semantic retrieval."
                            ),
                            meta="No search has been executed in this session yet.",
                        )
                    )

            translated_query = gr.Textbox(
                label="Tibetan Query Used For Retrieval",
                interactive=False,
                placeholder="The workbench will display the Tibetan query that was actually embedded for retrieval, whether translated or converted from Wylie.",
            )
            results_html = gr.HTML(_render_initial_results_state())

            gr.HTML(
                _render_panel_intro(
                    title="Research Workspace",
                    copy=(
                        "Phase 2 turns search hits into working material: filter by pecha, focus one hit with its scan, "
                        "select up to five hits for comparison, and export selected evidence for notes or citation drafts."
                    ),
                )
            )
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=320):
                    pecha_filter = gr.Dropdown(
                        label="Filter Results By Pecha",
                        choices=["All pechas"],
                        value="All pechas",
                        interactive=False,
                    )
                    hit_selector = gr.CheckboxGroup(
                        label="Pinned Hits For Comparison",
                        choices=[],
                        value=[],
                        interactive=False,
                    )
                    focus_selector = gr.Dropdown(
                        label="Focused Hit",
                        choices=[],
                        value=None,
                        interactive=False,
                    )
                    with gr.Row():
                        update_workspace_button = gr.Button("Update Workspace", variant="secondary")
                        clear_workspace_button = gr.Button("Clear Pins")

                with gr.Column(scale=8, min_width=420):
                    focus_html = gr.HTML(_render_focus_empty_state())

            compare_html = gr.HTML(_render_compare_empty_state())
            with gr.Accordion("Export Selected Hits", open=False):
                export_markdown = gr.Textbox(
                    label="Markdown Export",
                    lines=12,
                    interactive=False,
                    buttons=["copy"],
                    placeholder="Select hits and update the workspace to generate a Markdown export.",
                )
                export_json = gr.Textbox(
                    label="JSON Export",
                    lines=12,
                    interactive=False,
                    buttons=["copy"],
                    placeholder="Select hits and update the workspace to generate a JSON export.",
                )

        search_button.click(
            fn=on_search,
            inputs=[query_input, query_mode, include_back_translation, top_k, context_lines],
            outputs=[
                translated_query,
                results_html,
                activity_status,
                search_response_state,
                pecha_filter,
                hit_selector,
                focus_selector,
                focus_html,
                compare_html,
                export_markdown,
                export_json,
            ],
        )
        query_input.submit(
            fn=on_search,
            inputs=[query_input, query_mode, include_back_translation, top_k, context_lines],
            outputs=[
                translated_query,
                results_html,
                activity_status,
                search_response_state,
                pecha_filter,
                hit_selector,
                focus_selector,
                focus_html,
                compare_html,
                export_markdown,
                export_json,
            ],
        )
        reindex_button.click(
            fn=on_reindex,
            outputs=[index_status, activity_status],
        )
        pecha_filter.change(
            fn=on_filter_results,
            inputs=[pecha_filter, search_response_state],
            outputs=[
                hit_selector,
                focus_selector,
                focus_html,
                compare_html,
                export_markdown,
                export_json,
            ],
        )
        update_workspace_button.click(
            fn=on_update_workspace,
            inputs=[hit_selector, focus_selector, search_response_state],
            outputs=[focus_html, compare_html, export_markdown, export_json],
        )
        focus_selector.change(
            fn=on_update_workspace,
            inputs=[hit_selector, focus_selector, search_response_state],
            outputs=[focus_html, compare_html, export_markdown, export_json],
        )
        hit_selector.change(
            fn=on_update_workspace,
            inputs=[hit_selector, focus_selector, search_response_state],
            outputs=[focus_html, compare_html, export_markdown, export_json],
        )
        clear_workspace_button.click(
            fn=on_clear_workspace,
            inputs=[search_response_state],
            outputs=[
                hit_selector,
                focus_selector,
                focus_html,
                compare_html,
                export_markdown,
                export_json,
            ],
        )

    return app


def launch_workbench(app: gr.Blocks, config: SemanticSearchConfig) -> None:
    queued_app = app.queue(default_concurrency_limit=config.ui.concurrency_limit)
    queued_app.launch(
        server_name=config.ui.server_name,
        server_port=config.ui.server_port,
        share=config.ui.share,
        show_api=config.ui.show_api,
        allowed_paths=[str(config.corpus.transcripts_root.resolve())],
        css=WORKBENCH_CSS,
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


def _render_search_activity_status(response: SearchResponse) -> str:
    if not response.results:
        return _render_activity_status(
            tone="warning",
            label="Search Activity",
            title="No direct matches found",
            copy=f"No hits were returned using top_k={response.top_k} and context radius={response.context_lines}.",
            meta=(
                f"The Tibetan retrieval query is shown below. Mode used: {response.query_mode}. "
                "Try broadening the concept phrase or checking the query mode."
            ),
        )

    best_hit = response.results[0]
    best_label, _, best_hint = _describe_score(best_hit.score)
    return _render_activity_status(
        tone="success",
        label="Search Activity",
        title="Search complete",
        copy=(
            f"Returned {len(response.results)} hit(s) using top_k={response.top_k} "
            f"and context radius={response.context_lines}."
        ),
        meta=(
            f"Best hit: {best_label.lower()} ({best_hit.score:.4f}) for {best_hint}. "
            f"Input mode: {response.query_mode}. The Tibetan query used for retrieval is shown below."
        ),
    )


def _render_search_results(response: SearchResponse) -> str:
    if not response.results:
        return _render_empty_results_state(response)

    chips = "".join(
        [
            f'<span class="ssw-chip">Original query: {html.escape(response.original_query)}</span>',
            f'<span class="ssw-chip">Input mode: {html.escape(response.query_mode)}</span>',
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
        "The workbench will show the Tibetan retrieval query, ranked matches, compact source references, context windows, page scans, and optional back-translations here."
        "</p>"
        '<ul class="ssw-empty-list">'
        "<li>Use a concept, phrase, topic, or proper name.</li>"
        "<li>Switch the input mode to Wylie (EWTS) when entering transliterated Tibetan.</li>"
        "<li>Broader prompts often retrieve better than full natural-language questions.</li>"
        "<li>Use the search settings when you need more results or a wider local context.</li>"
        "</ul>"
        "</section>"
    )


def _render_loading_results_state(query_mode: str, top_k: int, context_lines: int) -> str:
    return (
        '<section class="ssw-empty-state ssw-loading-state">'
        '<h2 class="ssw-empty-title">Searching the corpus</h2>'
        '<p class="ssw-empty-copy">'
        f'The workbench is processing a <strong>{html.escape(query_mode)}</strong> query '
        f'with top_k={int(top_k)} and context radius={int(context_lines)}.'
        "</p>"
        '<ul class="ssw-loading-list">'
        "<li>Step 1: convert or translate the query into Tibetan retrieval text.</li>"
        "<li>Step 2: search the local Qdrant index for semantically related lines.</li>"
        "<li>Step 3: assemble context windows, source references, scans, and optional support translations.</li>"
        "</ul>"
        "</section>"
    )


def _render_empty_results_state(response: SearchResponse) -> str:
    return (
        '<section class="ssw-empty-state">'
        '<h2 class="ssw-empty-title">No matching passages found</h2>'
        '<p class="ssw-empty-copy">'
        f'The Tibetan retrieval query was <strong>{html.escape(response.translated_query)}</strong>. '
        f"That keeps the miss inspectable rather than opaque. Current input mode: <strong>{html.escape(response.query_mode)}</strong>."
        "</p>"
        '<ul class="ssw-empty-list">'
        "<li>Try a broader or shorter concept phrase.</li>"
        "<li>If you entered EWTS/Wylie, verify that the query mode is set correctly.</li>"
        "<li>Reduce specificity if the query contains several constraints.</li>"
        "<li>Expand Top K if you want to inspect weaker semantic neighbors.</li>"
        "<li>Increase the context radius when the match may sit just outside the current line window.</li>"
        "</ul>"
        "</section>"
    )


def _render_hit_card(hit: SearchHit) -> str:
    source_title = _build_source_title(hit)
    source_meta = (
        f"Page {html.escape(str(hit.metadata.get('page_number', '?')))} · "
        f"Line {html.escape(str(hit.metadata.get('line_number', '?')))} · "
        f"{html.escape(str(hit.metadata.get('source_file', '')))}"
    )
    score_label, score_tone, _ = _describe_score(hit.score)
    details_blocks: list[str] = []
    if hit.back_translation:
        details_blocks.append(
            _render_details_block(
                title="English back-translation",
                body=_render_preformatted(hit.back_translation),
            )
        )
    collection_metadata = hit.collection_metadata or {}
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
        f'<h3 class="ssw-card-source">{html.escape(source_title)}</h3>'
        f'<div class="ssw-card-meta">{source_meta}</div>'
        "</div>"
        f'<div class="ssw-score-pill" data-tone="{score_tone}">{html.escape(score_label)} · {hit.score:.4f}</div>'
        "</div>"
        f'{_render_source_reference_grid(hit)}'
        f'{_render_source_links(hit)}'
        '<div class="ssw-section-label">Matched Line</div>'
        f'<div class="ssw-match-line">{html.escape(hit.matched_line)}</div>'
        '<div class="ssw-section-label">Context Window</div>'
        f'{_render_context_box(hit.context)}'
        f'{_render_scan_block(hit)}'
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


def _render_source_reference_grid(hit: SearchHit) -> str:
    citation = _build_citation(hit)
    source_file = str(hit.metadata.get("source_file", ""))
    metadata_file = str(hit.metadata.get("metadata_file", ""))
    blocks = [
        _render_source_item("Citation", citation),
        _render_source_item("Source file", source_file),
    ]
    if metadata_file:
        blocks.append(_render_source_item("Metadata file", metadata_file))
    return f'<div class="ssw-source-grid">{"".join(blocks)}</div>'


def _render_source_item(label: str, value: str) -> str:
    safe_value = html.escape(value)
    return (
        '<div class="ssw-source-item">'
        f'<div class="ssw-source-item-label">{html.escape(label)}</div>'
        f'<div class="ssw-source-item-value"><code>{safe_value}</code></div>'
        "</div>"
    )


def _render_source_links(hit: SearchHit) -> str:
    links: list[str] = []
    if hit.source_file_url:
        source_file_url = html.escape(hit.source_file_url, quote=True)
        links.append(
            f'<a class="ssw-link-chip" href="{source_file_url}" target="_blank" rel="noopener noreferrer">Open source transcript</a>'
        )
    if hit.metadata_file_url:
        metadata_file_url = html.escape(hit.metadata_file_url, quote=True)
        links.append(
            f'<a class="ssw-link-chip" href="{metadata_file_url}" target="_blank" rel="noopener noreferrer">Open metadata.json</a>'
        )
    if hit.scan_url:
        scan_url = html.escape(hit.scan_url, quote=True)
        links.append(
            f'<a class="ssw-link-chip" href="{scan_url}" target="_blank" rel="noopener noreferrer">Open full image</a>'
        )
    if hit.collection_metadata:
        links.append('<span class="ssw-link-chip">Collection metadata below</span>')
    if not links:
        return ""
    return f'<div class="ssw-link-row">{"".join(links)}</div>'


def _render_scan_block(hit: SearchHit) -> str:
    if not hit.scan_url:
        return (
            '<section class="ssw-scan-missing">'
            '<div class="ssw-section-label">Associated Page Scan</div>'
            "No scan URL was resolved for this page. Check the page entry in metadata.json if this source should have an image."
            "</section>"
        )

    scan_meta_parts: list[str] = []
    if hit.scan_filename:
        scan_meta_parts.append(html.escape(hit.scan_filename))
    if hit.scan_index is not None:
        scan_meta_parts.append(f"metadata index {hit.scan_index}")
    scan_meta = " · ".join(scan_meta_parts)
    scan_meta_html = (
        f"{scan_meta}<br>"
        if scan_meta
        else ""
    )
    scan_url = html.escape(hit.scan_url, quote=True)
    alt_text = html.escape(f"Page scan for {hit.source_label}", quote=True)
    return (
        '<section class="ssw-scan-panel">'
        f'<div class="ssw-section-label">Associated Page Scan · Page {html.escape(str(hit.metadata.get("page_number", "?")))}</div>'
        f'<a class="ssw-scan-frame" href="{scan_url}" target="_blank" rel="noopener noreferrer">'
        f'<img class="ssw-scan-image" src="{scan_url}" alt="{alt_text}" loading="lazy" referrerpolicy="no-referrer">'
        "</a>"
        '<div class="ssw-scan-caption">'
        f'<span>{scan_meta_html}Compact preview for quick source checking.</span>'
        f'<a href="{scan_url}" target="_blank" rel="noopener noreferrer">Open full image</a>'
        "</div>"
        "</section>"
    )


def _build_source_title(hit: SearchHit) -> str:
    collection_metadata = hit.collection_metadata or {}
    candidate_keys = (
        "title",
        "display_title",
        "work_title",
        "catalog_title",
        "label",
        "name",
    )
    for key in candidate_keys:
        value = collection_metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

        flattened_value = hit.metadata.get(f"collection_{key}")
        if isinstance(flattened_value, str) and flattened_value.strip():
            return flattened_value.strip()

    pecha_title = hit.metadata.get("pecha_title")
    if isinstance(pecha_title, str) and pecha_title.strip():
        return pecha_title.strip()
    return hit.source_label


def _build_citation(hit: SearchHit) -> str:
    return (
        f"{_build_source_title(hit)}, "
        f"page {hit.metadata.get('page_number', '?')}, "
        f"line {hit.metadata.get('line_number', '?')}"
    )


def _describe_score(score: float) -> tuple[str, str, str]:
    if score >= 0.85:
        return ("High similarity", "high", "a close semantic match")
    if score >= 0.72:
        return ("Moderate similarity", "medium", "a useful nearby match")
    return ("Exploratory match", "low", "a looser semantic neighbor")


def _render_focus_empty_state() -> str:
    return (
        '<section class="ssw-empty-state">'
        '<h2 class="ssw-empty-title">No focused hit yet</h2>'
        '<p class="ssw-empty-copy">'
        "Run a search first. The workspace will then show one focused hit as a text-and-scan reading panel."
        "</p>"
        "</section>"
    )


def _render_focus_panel(hit: SearchHit | None) -> str:
    if hit is None:
        return _render_focus_empty_state()

    score_label, score_tone, score_hint = _describe_score(hit.score)
    return (
        '<section class="ssw-workspace-card ssw-focus-card">'
        '<div class="ssw-card-head">'
        '<div>'
        '<div class="ssw-card-kicker">Focused Reading</div>'
        f'<h3 class="ssw-focus-title">{html.escape(_build_source_title(hit))}</h3>'
        f'<p class="ssw-compare-meta">{html.escape(_build_citation(hit))}</p>'
        "</div>"
        f'<div class="ssw-score-pill" data-tone="{score_tone}">{html.escape(score_label)} · {hit.score:.4f}</div>'
        "</div>"
        '<div class="ssw-focus-layout">'
        "<div>"
        f'{_render_source_links(hit)}'
        '<div class="ssw-section-label">Matched Line</div>'
        f'<div class="ssw-match-line">{html.escape(hit.matched_line)}</div>'
        '<div class="ssw-section-label">Context Window</div>'
        f'{_render_context_box(hit.context)}'
        f'<p class="ssw-export-copy">Interpret as {html.escape(score_hint)}; verify wording against the scan before citing.</p>'
        "</div>"
        f'<div>{_render_scan_block(hit)}</div>'
        "</div>"
        "</section>"
    )


def _render_compare_empty_state(
    response: SearchResponse | None = None,
    pecha_filter: str | None = None,
) -> str:
    filter_copy = (
        f" Current filter: {html.escape(pecha_filter)}."
        if pecha_filter and pecha_filter != "All pechas"
        else ""
    )
    if response is not None and not response.results:
        return (
            '<section class="ssw-empty-state">'
            '<h2 class="ssw-empty-title">No hits available for comparison</h2>'
            '<p class="ssw-empty-copy">Run a broader search or adjust the query mode first.</p>'
            "</section>"
        )
    return (
        '<section class="ssw-empty-state">'
        '<h2 class="ssw-empty-title">Select hits to compare</h2>'
        '<p class="ssw-empty-copy">'
        "Pin up to five hits in the Research Workspace controls. The comparison view and exports will update from those selections."
        f"{filter_copy}"
        "</p>"
        "</section>"
    )


def _render_compare_panel(response: SearchResponse, hits: list[SearchHit]) -> str:
    if not hits:
        return _render_compare_empty_state(response)

    cards = "".join(_render_compare_card(hit) for hit in hits)
    return (
        '<section class="ssw-panel">'
        '<div class="ssw-results-head">'
        "<div>"
        '<h2 class="ssw-results-title">Pinned Hit Comparison</h2>'
        '<p class="ssw-results-subtitle">'
        f"Comparing {len(hits)} selected hit(s). Keep this small and deliberate for close philological reading."
        "</p>"
        "</div>"
        '<div class="ssw-chip-row">'
        f'<span class="ssw-chip">Original query: {html.escape(response.original_query)}</span>'
        f'<span class="ssw-chip">Retrieval mode: {html.escape(response.query_mode)}</span>'
        "</div>"
        "</div>"
        f'<div class="ssw-compare-grid">{cards}</div>'
        "</section>"
    )


def _render_compare_card(hit: SearchHit) -> str:
    score_label, score_tone, _ = _describe_score(hit.score)
    scan_link = (
        f'<a class="ssw-link-chip" href="{html.escape(hit.scan_url, quote=True)}" target="_blank" rel="noopener noreferrer">Scan</a>'
        if hit.scan_url
        else '<span class="ssw-link-chip">No scan</span>'
    )
    source_link = (
        f'<a class="ssw-link-chip" href="{html.escape(hit.source_file_url, quote=True)}" target="_blank" rel="noopener noreferrer">Transcript</a>'
        if hit.source_file_url
        else ""
    )
    return (
        '<article class="ssw-compare-card">'
        f'<div class="ssw-card-kicker">Hit {hit.rank}</div>'
        f'<h3 class="ssw-compare-title">{html.escape(_build_source_title(hit))}</h3>'
        f'<p class="ssw-compare-meta">{html.escape(_build_citation(hit))}</p>'
        f'<div class="ssw-score-pill" data-tone="{score_tone}">{html.escape(score_label)} · {hit.score:.4f}</div>'
        '<div class="ssw-section-label" style="margin-top: 12px;">Matched Line</div>'
        f'<div class="ssw-match-line">{html.escape(hit.matched_line)}</div>'
        '<div class="ssw-section-label">Context Window</div>'
        f'{_render_context_box(hit.context)}'
        f'<div class="ssw-link-row">{source_link}{scan_link}</div>'
        "</article>"
    )


def _build_pecha_filter_choices(response: SearchResponse) -> list[str]:
    pechas = sorted(
        {
            str(hit.metadata.get("pecha_title"))
            for hit in response.results
            if hit.metadata.get("pecha_title")
        }
    )
    return ["All pechas", *pechas]


def _build_hit_choices(
    response: SearchResponse,
    pecha_filter: str | None = None,
) -> list[str]:
    return [_hit_choice_label(hit) for hit in _filtered_hits(response, pecha_filter)]


def _filtered_hits(
    response: SearchResponse,
    pecha_filter: str | None = None,
) -> list[SearchHit]:
    if not pecha_filter or pecha_filter == "All pechas":
        return list(response.results)
    return [
        hit
        for hit in response.results
        if str(hit.metadata.get("pecha_title", "")) == pecha_filter
    ]


def _first_hit(response: SearchResponse | None) -> SearchHit | None:
    if response is None or not response.results:
        return None
    return response.results[0]


def _first_hit_choice(response: SearchResponse | None) -> str | None:
    first_hit = _first_hit(response)
    return _hit_choice_label(first_hit) if first_hit is not None else None


def _hit_choice_label(hit: SearchHit) -> str:
    page = hit.metadata.get("page_number", "?")
    line = hit.metadata.get("line_number", "?")
    return (
        f"Hit {hit.rank}: {_build_source_title(hit)} · "
        f"p. {page} · l. {line} · {hit.score:.4f}"
    )


def _resolve_selected_hits(
    response: SearchResponse,
    selected_hit_labels: list[str] | str | None,
) -> list[SearchHit]:
    if selected_hit_labels is None:
        return []
    labels = (
        [selected_hit_labels]
        if isinstance(selected_hit_labels, str)
        else list(selected_hit_labels)
    )
    ranks = {_rank_from_hit_choice(label) for label in labels}
    ranks.discard(None)
    return [hit for hit in response.results if hit.rank in ranks]


def _resolve_focus_hit(
    response: SearchResponse,
    focused_hit_label: str | None,
) -> SearchHit | None:
    rank = _rank_from_hit_choice(focused_hit_label)
    if rank is None:
        return None
    for hit in response.results:
        if hit.rank == rank:
            return hit
    return None


def _rank_from_hit_choice(label: str | None) -> int | None:
    if not label:
        return None
    prefix = str(label).split(":", 1)[0].strip()
    if not prefix.lower().startswith("hit "):
        return None
    rank_token = prefix[4:].strip()
    return int(rank_token) if rank_token.isdigit() else None


def _render_export_markdown(response: SearchResponse, hits: list[SearchHit]) -> str:
    if not hits:
        return ""

    sections = [
        "# Semantic Search Workbench Export",
        "",
        f"- Original query: {response.original_query}",
        f"- Retrieval query: {response.translated_query}",
        f"- Query mode: {response.query_mode}",
        f"- Selected hits: {len(hits)}",
        "",
    ]
    for hit in hits:
        score_label, _, _ = _describe_score(hit.score)
        sections.extend(
            [
                f"## Hit {hit.rank}: {_build_source_title(hit)}",
                "",
                f"- Citation: {_build_citation(hit)}",
                f"- Relevance: {score_label} ({hit.score:.4f})",
                f"- Source file: {hit.metadata.get('source_file', '')}",
                f"- Metadata file: {hit.metadata.get('metadata_file', '')}",
                f"- Scan URL: {hit.scan_url or ''}",
                "",
                "### Matched Line",
                "",
                hit.matched_line,
                "",
                "### Context",
                "",
                "```text",
                hit.context,
                "```",
                "",
            ]
        )
        if hit.back_translation:
            sections.extend(
                [
                    "### English Back-Translation",
                    "",
                    hit.back_translation,
                    "",
                ]
            )
    return "\n".join(sections).strip()


def _render_export_json(response: SearchResponse, hits: list[SearchHit]) -> str:
    if not hits:
        return ""
    payload = {
        "original_query": response.original_query,
        "tibetan_retrieval_query": response.translated_query,
        "query_mode": response.query_mode,
        "top_k": response.top_k,
        "context_lines": response.context_lines,
        "selected_hits": [_hit_to_export_payload(hit) for hit in hits],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _hit_to_export_payload(hit: SearchHit) -> dict[str, object]:
    score_label, _, _ = _describe_score(hit.score)
    return {
        "rank": hit.rank,
        "score": hit.score,
        "relevance_label": score_label,
        "title": _build_source_title(hit),
        "citation": _build_citation(hit),
        "page_number": hit.metadata.get("page_number"),
        "line_number": hit.metadata.get("line_number"),
        "source_file": hit.metadata.get("source_file"),
        "metadata_file": hit.metadata.get("metadata_file"),
        "matched_line": hit.matched_line,
        "context": hit.context,
        "back_translation": hit.back_translation,
        "scan_url": hit.scan_url,
        "scan_filename": hit.scan_filename,
        "source_file_url": hit.source_file_url,
        "metadata_file_url": hit.metadata_file_url,
    }
