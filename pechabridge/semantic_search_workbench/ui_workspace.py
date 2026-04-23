"""Research workspace rendering and selection helpers for the Gradio UI."""

from __future__ import annotations

import html
import json

from .service import SearchHit, SearchResponse


def render_focus_empty_state() -> str:
    return (
        '<section class="ssw-empty-state">'
        '<h2 class="ssw-empty-title">No focused hit yet</h2>'
        '<p class="ssw-empty-copy">'
        "Run a search first. The workspace will then show one focused hit as a text-and-scan reading panel."
        "</p>"
        "</section>"
    )


def render_focus_panel(hit: SearchHit | None) -> str:
    if hit is None:
        return render_focus_empty_state()

    score_label, score_tone, score_hint = describe_score(hit.score)
    return (
        '<section class="ssw-workspace-card ssw-focus-card">'
        '<div class="ssw-card-head">'
        "<div>"
        '<div class="ssw-card-kicker">Focused Reading</div>'
        f'<h3 class="ssw-focus-title">{html.escape(build_source_title(hit))}</h3>'
        f'<p class="ssw-compare-meta">{html.escape(build_citation(hit))}</p>'
        "</div>"
        f'<div class="ssw-score-pill" data-tone="{score_tone}">{html.escape(score_label)} · {hit.score:.4f}</div>'
        "</div>"
        '<div class="ssw-focus-layout">'
        "<div>"
        f"{render_source_links(hit)}"
        '<div class="ssw-section-label">Matched Line</div>'
        f'<div class="ssw-match-line">{html.escape(hit.matched_line)}</div>'
        '<div class="ssw-section-label">Context Window</div>'
        f"{render_context_box(hit.context)}"
        f'<p class="ssw-export-copy">Interpret as {html.escape(score_hint)}; verify wording against the scan before citing.</p>'
        "</div>"
        f"<div>{render_scan_block(hit)}</div>"
        "</div>"
        "</section>"
    )


def render_compare_empty_state(
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


def render_compare_panel(response: SearchResponse, hits: list[SearchHit]) -> str:
    if not hits:
        return render_compare_empty_state(response)

    cards = "".join(render_compare_card(hit) for hit in hits)
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


def render_compare_card(hit: SearchHit) -> str:
    score_label, score_tone, _ = describe_score(hit.score)
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
        f'<h3 class="ssw-compare-title">{html.escape(build_source_title(hit))}</h3>'
        f'<p class="ssw-compare-meta">{html.escape(build_citation(hit))}</p>'
        f'<div class="ssw-score-pill" data-tone="{score_tone}">{html.escape(score_label)} · {hit.score:.4f}</div>'
        '<div class="ssw-section-label" style="margin-top: 12px;">Matched Line</div>'
        f'<div class="ssw-match-line">{html.escape(hit.matched_line)}</div>'
        '<div class="ssw-section-label">Context Window</div>'
        f"{render_context_box(hit.context)}"
        f'<div class="ssw-link-row">{source_link}{scan_link}</div>'
        "</article>"
    )


def build_pecha_filter_choices(response: SearchResponse) -> list[str]:
    pechas = sorted(
        {
            str(hit.metadata.get("pecha_title"))
            for hit in response.results
            if hit.metadata.get("pecha_title")
        }
    )
    return ["All pechas", *pechas]


def build_hit_choices(
    response: SearchResponse,
    pecha_filter: str | None = None,
) -> list[str]:
    return [hit_choice_label(hit) for hit in filtered_hits(response, pecha_filter)]


def filtered_hits(
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


def first_hit(response: SearchResponse | None) -> SearchHit | None:
    if response is None or not response.results:
        return None
    return response.results[0]


def first_hit_choice(response: SearchResponse | None) -> str | None:
    hit = first_hit(response)
    return hit_choice_label(hit) if hit is not None else None


def hit_choice_label(hit: SearchHit) -> str:
    page = hit.metadata.get("page_number", "?")
    line = hit.metadata.get("line_number", "?")
    return (
        f"Hit {hit.rank}: {build_source_title(hit)} · "
        f"p. {page} · l. {line} · {hit.score:.4f}"
    )


def resolve_selected_hits(
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
    ranks = {rank_from_hit_choice(label) for label in labels}
    ranks.discard(None)
    return [hit for hit in response.results if hit.rank in ranks]


def resolve_focus_hit(
    response: SearchResponse,
    focused_hit_label: str | None,
) -> SearchHit | None:
    rank = rank_from_hit_choice(focused_hit_label)
    if rank is None:
        return None
    for hit in response.results:
        if hit.rank == rank:
            return hit
    return None


def rank_from_hit_choice(label: str | None) -> int | None:
    if not label:
        return None
    prefix = str(label).split(":", 1)[0].strip()
    if not prefix.lower().startswith("hit "):
        return None
    rank_token = prefix[4:].strip()
    return int(rank_token) if rank_token.isdigit() else None


def render_export_markdown(response: SearchResponse, hits: list[SearchHit]) -> str:
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
        score_label, _, _ = describe_score(hit.score)
        sections.extend(
            [
                f"## Hit {hit.rank}: {build_source_title(hit)}",
                "",
                f"- Citation: {build_citation(hit)}",
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


def render_export_json(response: SearchResponse, hits: list[SearchHit]) -> str:
    if not hits:
        return ""
    payload = {
        "original_query": response.original_query,
        "tibetan_retrieval_query": response.translated_query,
        "query_mode": response.query_mode,
        "top_k": response.top_k,
        "context_lines": response.context_lines,
        "selected_hits": [hit_to_export_payload(hit) for hit in hits],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def hit_to_export_payload(hit: SearchHit) -> dict[str, object]:
    score_label, _, _ = describe_score(hit.score)
    return {
        "rank": hit.rank,
        "score": hit.score,
        "relevance_label": score_label,
        "title": build_source_title(hit),
        "citation": build_citation(hit),
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


def build_source_title(hit: SearchHit) -> str:
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


def build_citation(hit: SearchHit) -> str:
    return (
        f"{build_source_title(hit)}, "
        f"page {hit.metadata.get('page_number', '?')}, "
        f"line {hit.metadata.get('line_number', '?')}"
    )


def describe_score(score: float) -> tuple[str, str, str]:
    if score >= 0.85:
        return ("High similarity", "high", "a close semantic match")
    if score >= 0.72:
        return ("Moderate similarity", "medium", "a useful nearby match")
    return ("Exploratory match", "low", "a looser semantic neighbor")


def render_context_box(context: str) -> str:
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
    rows_html = (
        "".join(rows)
        if rows
        else '<div class="ssw-context-line"><span class="ssw-context-marker">·</span><span class="ssw-context-text">No context available.</span></div>'
    )
    return f'<div class="ssw-context-box">{rows_html}</div>'


def render_source_links(hit: SearchHit) -> str:
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


def render_scan_block(hit: SearchHit) -> str:
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
    scan_meta_html = f"{scan_meta}<br>" if scan_meta else ""
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
