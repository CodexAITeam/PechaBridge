#!/usr/bin/env python3
"""Gradio workbench for reviewing extracted Donut OCR error samples."""

from __future__ import annotations

import argparse
import html
import json
import logging
from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gradio as gr
import pandas as pd

LOGGER = logging.getLogger("ocr_error_review_workbench")


TABLE_COLUMNS = [
    "idx",
    "filename",
    "cer",
    "split",
    "row_index",
    "pred_len",
    "ref_len",
    "source_dataset",
]


@dataclass
class WorkbenchConfig:
    dataset_dir: Path
    errors_jsonl: Path
    host: str
    port: int
    share: bool
    max_records: int
    newest: bool
    allowed_paths: List[str]


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a live Gradio workbench for inspecting Donut OCR high-CER extraction datasets.",
        add_help=add_help,
    )
    parser.add_argument(
        "--dataset_dir",
        "--dataset-dir",
        type=str,
        required=True,
        help="Extracted hard-sample dataset directory containing meta/errors.jsonl and images/.",
    )
    parser.add_argument(
        "--errors_jsonl",
        "--errors-jsonl",
        type=str,
        default="",
        help="Optional explicit errors JSONL path. Defaults to DATASET_DIR/meta/errors.jsonl.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Gradio server host. Use 0.0.0.0 for remote access.")
    parser.add_argument("--port", type=int, default=7862, help="Gradio server port.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share URL.")
    parser.add_argument("--max_records", "--max-records", type=int, default=1000, help="Maximum records to show per reload.")
    parser.add_argument("--oldest", action="store_true", help="Show oldest records instead of newest records.")
    return parser


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _resolve_config(args: argparse.Namespace) -> WorkbenchConfig:
    dataset_dir = Path(str(args.dataset_dir)).expanduser().resolve()
    raw_errors = str(getattr(args, "errors_jsonl", "") or "").strip()
    errors_jsonl = Path(raw_errors).expanduser().resolve() if raw_errors else dataset_dir / "meta" / "errors.jsonl"
    allowed = [str(dataset_dir.resolve())]
    try:
        allowed.append(str(errors_jsonl.parent.resolve()))
    except Exception:
        pass
    return WorkbenchConfig(
        dataset_dir=dataset_dir,
        errors_jsonl=errors_jsonl,
        host=str(getattr(args, "host", "127.0.0.1") or "127.0.0.1"),
        port=int(getattr(args, "port", 7862) or 7862),
        share=bool(getattr(args, "share", False)),
        max_records=max(1, int(getattr(args, "max_records", 1000) or 1000)),
        newest=not bool(getattr(args, "oldest", False)),
        allowed_paths=sorted(set(allowed)),
    )


def _safe_jsonl_records(path: Path, *, max_records: int, newest: bool) -> Tuple[List[Dict[str, Any]], int, int]:
    if not path.exists():
        return [], 0, 0
    records: deque[Dict[str, Any]] | List[Dict[str, Any]]
    records = deque(maxlen=max_records) if newest else []
    seen = 0
    skipped = 0
    with path.open("r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                # The extractor may be in the middle of writing the last line.
                skipped += 1
                continue
            if not isinstance(record, dict):
                skipped += 1
                continue
            seen += 1
            if newest:
                records.append(record)  # type: ignore[union-attr]
            elif len(records) < max_records:  # type: ignore[arg-type]
                records.append(record)  # type: ignore[union-attr]
    out = list(records)
    if newest:
        out = list(out)
    return out, seen, skipped


def _record_filename(record: Dict[str, Any]) -> str:
    for key in ("dataset_image", "copied_image", "image"):
        value = str(record.get(key, "") or "").strip()
        if value:
            return value
    row = record.get("row")
    if isinstance(row, dict):
        value = str(row.get("image", "") or row.get("image_path", "") or row.get("line_path", "") or "").strip()
        if value:
            return value
    return ""


def _resolve_image_path(record: Dict[str, Any], dataset_dir: Path) -> Optional[str]:
    dataset_image = str(record.get("dataset_image", "") or "").strip()
    if dataset_image:
        path = dataset_dir / dataset_image
        if path.exists():
            return str(path)
    for key in ("copied_image", "image"):
        raw = str(record.get(key, "") or "").strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = dataset_dir / path
        if path.exists():
            return str(path)
    return None


def _source_dataset(record: Dict[str, Any]) -> str:
    for key in ("source_dataset", "source_dataset_short", "source_split"):
        value = str(record.get(key, "") or "").strip()
        if value:
            return value
    row = record.get("row")
    if isinstance(row, dict):
        for key in ("source_dataset", "source_dataset_short", "source_split", "dataset", "source"):
            value = str(row.get(key, "") or "").strip()
            if value:
                return value
    return ""


def _filter_records(
    records: Sequence[Dict[str, Any]],
    *,
    min_cer: float,
    search: str,
) -> List[Dict[str, Any]]:
    needle = str(search or "").strip().lower()
    out: List[Dict[str, Any]] = []
    for record in records:
        try:
            cer = float(record.get("cer", 0.0) or 0.0)
        except Exception:
            cer = 0.0
        if cer < float(min_cer):
            continue
        if needle:
            haystack = "\n".join(
                [
                    _record_filename(record),
                    str(record.get("pred", "") or ""),
                    str(record.get("ref", "") or ""),
                    str(record.get("text_raw", "") or ""),
                    _source_dataset(record),
                ]
            ).lower()
            if needle not in haystack:
                continue
        out.append(record)
    out.sort(key=lambda r: float(r.get("cer", 0.0) or 0.0), reverse=True)
    return out


def _table_for_records(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        rows.append(
            {
                "idx": idx,
                "filename": _record_filename(record),
                "cer": round(float(record.get("cer", 0.0) or 0.0), 6),
                "split": str(record.get("dataset_split", "") or ""),
                "row_index": int(record.get("row_index", -1) or -1),
                "pred_len": int(record.get("pred_len", 0) or 0),
                "ref_len": int(record.get("ref_len", 0) or 0),
                "source_dataset": _source_dataset(record),
            }
        )
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def _status_markdown(
    *,
    config: WorkbenchConfig,
    seen: int,
    skipped: int,
    loaded: int,
    filtered: int,
) -> str:
    return (
        f"**Live source:** `{config.errors_jsonl}`\n\n"
        f"**Seen in file:** `{seen}` records | **Loaded window:** `{loaded}` | "
        f"**After filters:** `{filtered}` | **Skipped partial/bad lines:** `{skipped}`\n\n"
        "The extractor writes this JSONL incrementally, so click **Reload** while the dataset is still growing."
    )


def _html_diff(pred: str, ref: str) -> str:
    matcher = SequenceMatcher(a=str(ref or ""), b=str(pred or ""))
    chunks: List[str] = [
        "<div style='font-family: ui-monospace, SFMono-Regular, Menlo, monospace; line-height: 1.8; white-space: pre-wrap;'>"
    ]
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        ref_chunk = html.escape(ref[i1:i2])
        pred_chunk = html.escape(pred[j1:j2])
        if tag == "equal":
            chunks.append(ref_chunk)
        elif tag == "delete":
            chunks.append(f"<span style='background:#ffd7d7;text-decoration:line-through'>{ref_chunk}</span>")
        elif tag == "insert":
            chunks.append(f"<span style='background:#d7ffd9'>{pred_chunk}</span>")
        else:
            chunks.append(f"<span style='background:#ffe8a3;text-decoration:line-through'>{ref_chunk}</span>")
            chunks.append(f"<span style='background:#d7e8ff'>{pred_chunk}</span>")
    chunks.append("</div>")
    return "".join(chunks)


def _sample_view(records: Sequence[Dict[str, Any]], index: int, dataset_dir: Path) -> Tuple[Optional[str], str, str, str, str, str]:
    if not records:
        return None, "No sample selected.", "", "", "", ""
    idx = max(0, min(int(index), len(records) - 1))
    record = records[idx]
    pred = str(record.get("pred", "") or "")
    ref = str(record.get("ref", "") or "")
    raw = str(record.get("text_raw", "") or "")
    image_path = _resolve_image_path(record, dataset_dir)
    meta_rows = [
        ("Index", idx),
        ("CER", f"{float(record.get('cer', 0.0) or 0.0):.6f}"),
        ("Filename", _record_filename(record)),
        ("Dataset split", str(record.get("dataset_split", "") or "")),
        ("Source dataset", _source_dataset(record)),
        ("Source row index", str(record.get("row_index", "") or "")),
        ("Edit distance", str(record.get("edit_distance", "") or "")),
        ("Prediction length", str(record.get("pred_len", "") or "")),
        ("Reference length", str(record.get("ref_len", "") or "")),
        ("Image path", image_path or "not found"),
    ]
    metadata = "\n".join(f"**{key}:** `{html.escape(str(value))}`" for key, value in meta_rows)
    return image_path, metadata, pred, ref, raw, _html_diff(pred, ref)


def build_workbench(config: WorkbenchConfig) -> gr.Blocks:
    css = """
    .ocr-hero {padding: 18px 22px; border-radius: 18px; background: linear-gradient(135deg,#f4efe4,#e5f0ec); border: 1px solid #d8cfbd;}
    .ocr-hero h1 {margin: 0 0 6px 0; font-size: 28px;}
    .ocr-hero p {margin: 0; color: #4c4639;}
    textarea {font-family: ui-monospace, SFMono-Regular, Menlo, monospace !important;}
    """

    with gr.Blocks(title="Donut OCR Error Review", css=css) as app:
        gr.HTML(
            "<section class='ocr-hero'><h1>Donut OCR Error Review</h1>"
            "<p>Inspect high-CER samples while the extractor is still writing the dataset.</p></section>"
        )
        records_state = gr.State([])
        selected_index = gr.State(0)

        with gr.Row():
            min_cer = gr.Number(value=0.10, label="Minimum CER", precision=4)
            search = gr.Textbox(value="", label="Search pred/ref/filename/source", placeholder="optional")
            max_records = gr.Number(value=config.max_records, label="Max records to load", precision=0)
            newest = gr.Checkbox(value=config.newest, label="Newest records")
            reload_btn = gr.Button("Reload", variant="primary")

        status = gr.Markdown()
        table = gr.Dataframe(headers=TABLE_COLUMNS, datatype=["number", "str", "number", "str", "number", "number", "number", "str"], interactive=False, wrap=True)

        with gr.Row():
            prev_btn = gr.Button("Previous")
            index_box = gr.Number(value=0, label="Selected idx", precision=0)
            next_btn = gr.Button("Next")

        with gr.Row():
            image = gr.Image(label="Line image", type="filepath", height=220)
            metadata = gr.Markdown(label="Metadata")

        with gr.Row():
            pred_text = gr.Textbox(label="Prediction", lines=7, show_copy_button=True)
            ref_text = gr.Textbox(label="Ground truth / CER reference", lines=7, show_copy_button=True)
        raw_text = gr.Textbox(label="Raw manifest text", lines=4, show_copy_button=True)
        diff_html = gr.HTML(label="Diff")

        def reload_data(min_cer_value: float, search_value: str, max_records_value: int, newest_value: bool):
            cfg = WorkbenchConfig(
                dataset_dir=config.dataset_dir,
                errors_jsonl=config.errors_jsonl,
                host=config.host,
                port=config.port,
                share=config.share,
                max_records=max(1, int(max_records_value or config.max_records)),
                newest=bool(newest_value),
                allowed_paths=config.allowed_paths,
            )
            raw_records, seen, skipped = _safe_jsonl_records(
                cfg.errors_jsonl,
                max_records=cfg.max_records,
                newest=cfg.newest,
            )
            filtered = _filter_records(raw_records, min_cer=float(min_cer_value or 0.0), search=str(search_value or ""))
            df = _table_for_records(filtered)
            status_md = _status_markdown(
                config=cfg,
                seen=seen,
                skipped=skipped,
                loaded=len(raw_records),
                filtered=len(filtered),
            )
            sample = _sample_view(filtered, 0, cfg.dataset_dir)
            return df, status_md, filtered, 0, 0, *sample

        def select_row(records: Sequence[Dict[str, Any]], evt: gr.SelectData):
            row_idx = 0
            try:
                if isinstance(evt.index, tuple):
                    row_idx = int(evt.index[0])
                else:
                    row_idx = int(evt.index)
            except Exception:
                row_idx = 0
            sample = _sample_view(records, row_idx, config.dataset_dir)
            return row_idx, row_idx, *sample

        def move_selection(records: Sequence[Dict[str, Any]], current: int, delta: int):
            if not records:
                sample = _sample_view(records, 0, config.dataset_dir)
                return 0, 0, *sample
            idx = max(0, min(int(current or 0) + int(delta), len(records) - 1))
            sample = _sample_view(records, idx, config.dataset_dir)
            return idx, idx, *sample

        def jump_selection(records: Sequence[Dict[str, Any]], current: int):
            sample = _sample_view(records, int(current or 0), config.dataset_dir)
            idx = max(0, min(int(current or 0), max(0, len(records) - 1)))
            return idx, *sample

        reload_outputs = [table, status, records_state, selected_index, index_box, image, metadata, pred_text, ref_text, raw_text, diff_html]
        sample_outputs = [selected_index, index_box, image, metadata, pred_text, ref_text, raw_text, diff_html]
        jump_outputs = [selected_index, image, metadata, pred_text, ref_text, raw_text, diff_html]

        reload_btn.click(
            reload_data,
            inputs=[min_cer, search, max_records, newest],
            outputs=reload_outputs,
        )
        table.select(select_row, inputs=[records_state], outputs=sample_outputs)
        prev_btn.click(lambda records, current: move_selection(records, current, -1), inputs=[records_state, selected_index], outputs=sample_outputs)
        next_btn.click(lambda records, current: move_selection(records, current, 1), inputs=[records_state, selected_index], outputs=sample_outputs)
        index_box.submit(jump_selection, inputs=[records_state, index_box], outputs=jump_outputs)
        app.load(reload_data, inputs=[min_cer, search, max_records, newest], outputs=reload_outputs)

    return app


def run(args: argparse.Namespace) -> int:
    _configure_logging()
    config = _resolve_config(args)
    LOGGER.info("Launching OCR error review workbench for %s", config.errors_jsonl)
    app = build_workbench(config)
    app.queue(default_concurrency_limit=2).launch(
        server_name=config.host,
        server_port=config.port,
        share=config.share,
        allowed_paths=config.allowed_paths,
        show_api=False,
    )
    return 0


def main() -> int:
    parser = create_parser()
    return int(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
