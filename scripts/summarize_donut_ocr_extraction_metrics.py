#!/usr/bin/env python3
"""Summarize Donut OCR extraction CER metrics by checkpoint and source dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan run_donut_ocr_error_extraction_pair.sh output folders and summarize "
            "CER metrics per checkpoint, split, and source_dataset."
        ),
        add_help=add_help,
    )
    parser.add_argument(
        "--root_dir",
        "--root-dir",
        required=True,
        help="Folder containing donut_error_extract_* output directories.",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        default="",
        help="Directory for summary outputs. Defaults to <root_dir>/donut_ocr_extraction_metric_summary.",
    )
    parser.add_argument(
        "--jsonl_name",
        "--jsonl-name",
        default="errors.jsonl",
        help="JSONL filename to read below meta/ in each extraction output directory.",
    )
    parser.add_argument(
        "--include_empty",
        "--include-empty",
        action="store_true",
        help="Include discovered outputs whose JSONL has no usable CER records in the report.",
    )
    parser.add_argument(
        "--top_n",
        "--top-n",
        type=int,
        default=0,
        help="Optional max rows to include in the Markdown table (0 = all rows).",
    )
    return parser


@dataclass(frozen=True)
class MetricKey:
    checkpoint: str
    split: str
    source_dataset: str
    output_dir: str


def _as_float(value: object) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _first_str(*values: object) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _source_dataset(record: Dict[str, object]) -> str:
    row = record.get("row")
    row = row if isinstance(row, dict) else {}
    return _first_str(
        record.get("source_dataset"),
        row.get("source_dataset"),
        record.get("dataset"),
        row.get("dataset"),
        record.get("source"),
        row.get("source"),
        "UNKNOWN",
    )


def _checkpoint_name(record: Dict[str, object], output_dir: Path) -> str:
    checkpoint = _first_str(record.get("checkpoint"))
    if checkpoint:
        return Path(checkpoint).name
    name = output_dir.name
    marker = "_checkpoint-"
    if marker in name:
        suffix = name.split(marker, 1)[1]
        return "checkpoint-" + suffix.split("_", 1)[0]
    return "UNKNOWN"


def _split_name(record: Dict[str, object], output_dir: Path) -> str:
    manifest = _first_str(record.get("manifest"))
    manifest_name = Path(manifest).name.lower() if manifest else ""
    output_name = output_dir.name.lower()
    haystacks = (output_name, manifest_name, manifest.lower())
    for haystack in haystacks:
        if "_train_" in haystack or "/train/" in haystack or "train_manifest" in haystack:
            return "train"
        if "_val_" in haystack or "/val/" in haystack or "/eval/" in haystack or "val_manifest" in haystack or "eval_manifest" in haystack:
            return "val"
    return _first_str(record.get("split"), "unknown")


def _iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open(encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                yield row


def _summary_threshold(output_root: Path) -> Optional[float]:
    summary_path = output_root / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    threshold = _as_float(summary.get("cer_threshold"))
    if threshold is not None:
        return threshold
    quality = summary.get("quality_metrics")
    if isinstance(quality, dict):
        return _as_float(quality.get("cer_threshold"))
    return None


def _stats(values: List[float]) -> Dict[str, object]:
    values = [float(v) for v in values if math.isfinite(float(v))]
    values.sort()
    if not values:
        return {
            "n": 0,
            "cer_min": "",
            "cer_max": "",
            "cer_mean": "",
            "cer_median": "",
            "cer_std": "",
        }
    return {
        "n": len(values),
        "cer_min": min(values),
        "cer_max": max(values),
        "cer_mean": sum(values) / len(values),
        "cer_median": statistics.median(values),
        "cer_std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def _fmt(value: object, digits: int = 5) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _write_csv(path: Path, rows: List[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_report(path: Path, rows: List[Dict[str, object]], *, top_n: int, scanned: List[str], warnings: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shown_rows = rows if top_n <= 0 else rows[:top_n]
    lines = [
        "# Donut OCR Extraction Metrics Summary",
        "",
        f"- Scanned extraction files: {len(scanned)}",
        f"- Summary rows: {len(rows)}",
        "- CER values come from the scanned JSONL records. For full comparability, run extraction with `CER_THRESHOLD=-1`.",
        "",
    ]
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")
    lines.extend(
        [
            "## Metrics By Checkpoint / Split / Source Dataset",
            "",
            "| checkpoint | split | source_dataset | n | cer_min | cer_mean | cer_max | cer_median | cer_std |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in shown_rows:
        lines.append(
            "| {checkpoint} | {split} | {source_dataset} | {n} | {cer_min} | {cer_mean} | {cer_max} | {cer_median} | {cer_std} |".format(
                checkpoint=row.get("checkpoint", ""),
                split=row.get("split", ""),
                source_dataset=row.get("source_dataset", ""),
                n=row.get("n", ""),
                cer_min=_fmt(row.get("cer_min")),
                cer_mean=_fmt(row.get("cer_mean")),
                cer_max=_fmt(row.get("cer_max")),
                cer_median=_fmt(row.get("cer_median")),
                cer_std=_fmt(row.get("cer_std")),
            )
        )
    if top_n > 0 and len(rows) > top_n:
        lines.append("")
        lines.append(f"Showing top {top_n} rows of {len(rows)}. See CSV for the full table.")
    lines.append("")
    lines.append("## Scanned Files")
    lines.append("")
    for item in scanned:
        lines.append(f"- `{item}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, object]:
    root_dir = Path(str(args.root_dir)).expanduser().resolve()
    output_dir = Path(str(args.output_dir)).expanduser().resolve() if str(args.output_dir or "").strip() else root_dir / "donut_ocr_extraction_metric_summary"
    jsonl_name = str(args.jsonl_name or "errors.jsonl")
    jsonl_paths = sorted(root_dir.glob(f"**/meta/{jsonl_name}"))

    grouped: Dict[MetricKey, List[float]] = defaultdict(list)
    scanned: List[str] = []
    warnings: List[str] = []
    empty_outputs: List[str] = []

    for jsonl_path in jsonl_paths:
        output_root = jsonl_path.parent.parent
        scanned.append(str(jsonl_path))
        usable = 0
        threshold = _summary_threshold(output_root)
        for record in _iter_jsonl(jsonl_path):
            cer = _as_float(record.get("cer"))
            if cer is None:
                continue
            key = MetricKey(
                checkpoint=_checkpoint_name(record, output_root),
                split=_split_name(record, output_root),
                source_dataset=_source_dataset(record),
                output_dir=str(output_root),
            )
            grouped[key].append(cer)
            usable += 1
        if usable == 0:
            empty_outputs.append(str(jsonl_path))
        if threshold is not None and threshold >= 0.0:
            warnings.append(
                f"{jsonl_path} appears threshold-filtered (summary cer_threshold={threshold}); "
                "mean/min/max are over saved records only."
            )

    rows: List[Dict[str, object]] = []
    for key, values in grouped.items():
        row = {
            "checkpoint": key.checkpoint,
            "split": key.split,
            "source_dataset": key.source_dataset,
            "output_dir": key.output_dir,
        }
        row.update(_stats(values))
        rows.append(row)

    rows.sort(key=lambda r: (str(r.get("checkpoint")), str(r.get("split")), str(r.get("source_dataset"))))

    if bool(getattr(args, "include_empty", False)):
        for path in empty_outputs:
            rows.append(
                {
                    "checkpoint": "UNKNOWN",
                    "split": "unknown",
                    "source_dataset": "NO_USABLE_CER_RECORDS",
                    "output_dir": str(Path(path).parent.parent),
                    **_stats([]),
                }
            )
    elif empty_outputs:
        warnings.append(f"Skipped {len(empty_outputs)} JSONL files with no usable CER records.")

    fields = [
        "checkpoint",
        "split",
        "source_dataset",
        "n",
        "cer_min",
        "cer_mean",
        "cer_max",
        "cer_median",
        "cer_std",
        "output_dir",
    ]
    csv_path = output_dir / "source_dataset_cer_summary.csv"
    report_path = output_dir / "source_dataset_cer_summary.md"
    json_path = output_dir / "source_dataset_cer_summary.json"

    _write_csv(csv_path, rows, fields)
    _write_report(report_path, rows, top_n=int(getattr(args, "top_n", 0) or 0), scanned=scanned, warnings=warnings)
    json_path.write_text(
        json.dumps(
            {
                "root_dir": str(root_dir),
                "n_scanned_files": len(scanned),
                "n_summary_rows": len(rows),
                "warnings": warnings,
                "rows": rows,
                "outputs": {
                    "csv": str(csv_path),
                    "report": str(report_path),
                    "json": str(json_path),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote report: {report_path}")
    print(f"Wrote JSON: {json_path}")
    if warnings:
        print(f"Warnings: {len(warnings)}")
    return {"csv": str(csv_path), "report": str(report_path), "json": str(json_path), "rows": rows}


def main() -> int:
    run(create_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
