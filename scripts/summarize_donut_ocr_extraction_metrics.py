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
    parser.add_argument(
        "--cer_thresholds",
        "--cer-thresholds",
        default="0.01,0.05,0.10,0.20,0.50",
        help="Comma-separated CER thresholds for tail-count columns (default: 0.01,0.05,0.10,0.20,0.50).",
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


def _percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    q = min(1.0, max(0.0, float(q)))
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def _threshold_key(threshold: float) -> str:
    return f"{float(threshold):g}".replace(".", "p")


def _parse_thresholds(raw: str) -> List[float]:
    values: List[float] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        value = _as_float(part)
        if value is not None:
            values.append(float(value))
    return sorted(set(values))


def _stats(values: List[float], *, thresholds: List[float]) -> Dict[str, object]:
    values = [float(v) for v in values if math.isfinite(float(v))]
    values.sort()
    if not values:
        empty = {
            "n": 0,
            "cer_min": "",
            "cer_max": "",
            "cer_mean": "",
            "cer_median": "",
            "cer_p75": "",
            "cer_p90": "",
            "cer_p95": "",
            "cer_p99": "",
            "cer_std": "",
        }
        for threshold in thresholds:
            key = _threshold_key(threshold)
            empty[f"cer_gt_{key}_n"] = ""
            empty[f"cer_gt_{key}_rate"] = ""
        return empty
    stats = {
        "n": len(values),
        "cer_min": min(values),
        "cer_max": max(values),
        "cer_mean": sum(values) / len(values),
        "cer_median": statistics.median(values),
        "cer_p75": _percentile(values, 0.75),
        "cer_p90": _percentile(values, 0.90),
        "cer_p95": _percentile(values, 0.95),
        "cer_p99": _percentile(values, 0.99),
        "cer_std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }
    for threshold in thresholds:
        key = _threshold_key(threshold)
        count = sum(1 for value in values if value > threshold)
        stats[f"cer_gt_{key}_n"] = int(count)
        stats[f"cer_gt_{key}_rate"] = float(count / len(values))
    return {
        **stats,
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


def _write_report(
    path: Path,
    rows: List[Dict[str, object]],
    *,
    top_n: int,
    scanned: List[str],
    warnings: List[str],
    thresholds: List[float],
) -> None:
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
            "| checkpoint | split | source_dataset | n | mean | median | p90 | p95 | p99 | max | >10% | >20% |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in shown_rows:
        gt_10 = row.get("cer_gt_0p1_rate", "")
        gt_20 = row.get("cer_gt_0p2_rate", "")
        lines.append(
            "| {checkpoint} | {split} | {source_dataset} | {n} | {cer_mean} | {cer_median} | {cer_p90} | {cer_p95} | {cer_p99} | {cer_max} | {gt_10} | {gt_20} |".format(
                checkpoint=row.get("checkpoint", ""),
                split=row.get("split", ""),
                source_dataset=row.get("source_dataset", ""),
                n=row.get("n", ""),
                cer_mean=_fmt(row.get("cer_mean")),
                cer_median=_fmt(row.get("cer_median")),
                cer_p90=_fmt(row.get("cer_p90")),
                cer_p95=_fmt(row.get("cer_p95")),
                cer_p99=_fmt(row.get("cer_p99")),
                cer_max=_fmt(row.get("cer_max")),
                gt_10=_fmt(gt_10),
                gt_20=_fmt(gt_20),
            )
        )
    if top_n > 0 and len(rows) > top_n:
        lines.append("")
        lines.append(f"Showing top {top_n} rows of {len(rows)}. See CSV for the full table.")
    lines.append("")
    if thresholds:
        lines.append("## Tail Count Columns")
        lines.append("")
        for threshold in thresholds:
            key = _threshold_key(threshold)
            lines.append(f"- `cer_gt_{key}_n` / `cer_gt_{key}_rate`: samples with CER > {threshold:g}")
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
    thresholds = _parse_thresholds(str(getattr(args, "cer_thresholds", "") or ""))

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
        row.update(_stats(values, thresholds=thresholds))
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
                    **_stats([], thresholds=thresholds),
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
        "cer_median",
        "cer_p75",
        "cer_p90",
        "cer_p95",
        "cer_p99",
        "cer_max",
        "cer_std",
        *[
            field
            for threshold in thresholds
            for key in [_threshold_key(threshold)]
            for field in (f"cer_gt_{key}_n", f"cer_gt_{key}_rate")
        ],
        "output_dir",
    ]
    csv_path = output_dir / "source_dataset_cer_summary.csv"
    report_path = output_dir / "source_dataset_cer_summary.md"
    json_path = output_dir / "source_dataset_cer_summary.json"

    _write_csv(csv_path, rows, fields)
    _write_report(
        report_path,
        rows,
        top_n=int(getattr(args, "top_n", 0) or 0),
        scanned=scanned,
        warnings=warnings,
        thresholds=thresholds,
    )
    json_path.write_text(
        json.dumps(
            {
                "root_dir": str(root_dir),
                "n_scanned_files": len(scanned),
                "n_summary_rows": len(rows),
                "warnings": warnings,
                "cer_threshold_columns": thresholds,
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
