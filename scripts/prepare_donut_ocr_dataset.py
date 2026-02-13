#!/usr/bin/env python3
"""Prepare label-filtered OCR manifests for Donut-style OCR training."""

from __future__ import annotations

import json
import logging
import random
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tibetan_utils.arg_utils import create_prepare_donut_ocr_dataset_parser

LOGGER = logging.getLogger("prepare_donut_ocr_dataset")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _parse_splits(spec: str) -> List[str]:
    splits = [s.strip() for s in str(spec).split(",") if s.strip()]
    if not splits:
        return ["train", "val"]
    # Keep order but drop duplicates.
    seen = set()
    out: List[str] = []
    for split in splits:
        if split not in seen:
            seen.add(split)
            out.append(split)
    return out


def _normalize_newline_representation(text: str, token_mode: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if token_mode == "<NL>":
        return normalized.replace("\n", "<NL>")
    if token_mode == "\\n":
        return normalized.replace("<NL>", "\n")
    return normalized


def _normalize_text(
    text: str,
    *,
    normalization: str,
    output_newline_token: str,
    min_chars: int,
    max_chars: int,
) -> str:
    value = str(text).replace("\ufeff", "")
    value = _normalize_newline_representation(value, output_newline_token)
    if normalization != "none":
        value = unicodedata.normalize(normalization, value)
    value = value.strip()
    if max_chars > 0 and len(value) > max_chars:
        value = value[:max_chars]
    if len(value) < max(0, min_chars):
        return ""
    return value


def _iter_records_for_split(split_dir: Path) -> Iterable[Tuple[Path, int, Dict[str, object]]]:
    targets_dir = split_dir / "ocr_targets"
    if not targets_dir.exists():
        return
    for target_file in sorted(targets_dir.glob("*.json")):
        try:
            payload = json.loads(target_file.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.warning("Skipping unreadable target file %s: %s", target_file, exc)
            continue
        records = payload.get("records", [])
        if not isinstance(records, list):
            continue
        for idx, rec in enumerate(records):
            if isinstance(rec, dict):
                yield target_file, idx, rec


def _wrap_target_text(
    text: str,
    *,
    wrap_task_tokens: bool,
    task_start_token: str,
    task_end_token: str,
    include_class_token: bool,
    class_token: str,
) -> str:
    out = text
    if include_class_token and class_token:
        out = f"{class_token}{out}"
    if wrap_task_tokens:
        out = f"{task_start_token}{out}{task_end_token}"
    return out


def _collect_split_samples(
    *,
    dataset_dir: Path,
    split: str,
    label_id: int,
    text_field: str,
    normalization: str,
    output_newline_token: str,
    min_chars: int,
    max_chars: int,
    wrap_task_tokens: bool,
    task_start_token: str,
    task_end_token: str,
    include_class_token: bool,
    class_token: str,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    split_dir = dataset_dir / split
    stats = {
        "targets_seen": 0,
        "records_seen": 0,
        "label_filtered": 0,
        "missing_crop": 0,
        "empty_text": 0,
        "kept": 0,
    }
    samples: List[Dict[str, object]] = []

    for target_file, idx, rec in _iter_records_for_split(split_dir):
        stats["records_seen"] += 1
        class_id = rec.get("class_id")
        if class_id is None or int(class_id) != int(label_id):
            stats["label_filtered"] += 1
            continue

        crop_rel_path = rec.get("crop_rel_path")
        if not isinstance(crop_rel_path, str) or not crop_rel_path.strip():
            stats["missing_crop"] += 1
            continue
        image_path = (split_dir / crop_rel_path).resolve()
        if not image_path.exists():
            stats["missing_crop"] += 1
            continue

        raw_text = rec.get(text_field, "")
        if not isinstance(raw_text, str):
            raw_text = str(raw_text)
        normalized_text = _normalize_text(
            raw_text,
            normalization=normalization,
            output_newline_token=output_newline_token,
            min_chars=min_chars,
            max_chars=max_chars,
        )
        if not normalized_text:
            stats["empty_text"] += 1
            continue

        target_text = _wrap_target_text(
            normalized_text,
            wrap_task_tokens=wrap_task_tokens,
            task_start_token=task_start_token,
            task_end_token=task_end_token,
            include_class_token=include_class_token,
            class_token=class_token,
        )
        sample = {
            "id": f"{target_file.stem}_{idx:03d}",
            "split": split,
            "class_id": int(class_id),
            "image": str(image_path),
            "text": target_text,
            "text_raw": normalized_text,
            "source_target_file": str(target_file.resolve()),
        }
        samples.append(sample)
        stats["kept"] += 1
        stats["targets_seen"] += 1

    return samples, stats


def _dedupe_samples(samples: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    seen = set()
    for sample in samples:
        key = (sample.get("image"), sample.get("text"))
        if key in seen:
            continue
        seen.add(key)
        out.append(sample)
    return out


def _cap_samples(samples: List[Dict[str, object]], cap: int, seed: int) -> List[Dict[str, object]]:
    if cap <= 0 or len(samples) <= cap:
        return samples
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    keep = set(indices[:cap])
    return [sample for i, sample in enumerate(samples) if i in keep]


def _write_jsonl(path: Path, records: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(args) -> Dict[str, object]:
    _configure_logging()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    splits = _parse_splits(args.splits)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Preparing OCR manifests from %s", dataset_dir)
    LOGGER.info("Using splits=%s label_id=%d", splits, int(args.label_id))

    split_stats: Dict[str, Dict[str, int]] = {}
    manifest_paths: Dict[str, str] = {}
    train_text_records: List[Dict[str, str]] = []

    for split_idx, split in enumerate(splits):
        samples, stats = _collect_split_samples(
            dataset_dir=dataset_dir,
            split=split,
            label_id=int(args.label_id),
            text_field=args.text_field,
            normalization=args.normalization,
            output_newline_token=args.output_newline_token,
            min_chars=int(args.min_chars),
            max_chars=int(args.max_chars),
            wrap_task_tokens=bool(args.wrap_task_tokens),
            task_start_token=args.task_start_token,
            task_end_token=args.task_end_token,
            include_class_token=bool(args.include_class_token),
            class_token=args.class_token,
        )
        if args.dedupe:
            before = len(samples)
            samples = _dedupe_samples(samples)
            stats["deduped"] = max(0, before - len(samples))
        else:
            stats["deduped"] = 0

        if int(args.max_samples_per_split) > 0:
            before = len(samples)
            samples = _cap_samples(samples, int(args.max_samples_per_split), int(args.seed) + split_idx)
            stats["capped"] = max(0, before - len(samples))
        else:
            stats["capped"] = 0

        manifest_path = output_dir / f"{split}_manifest.jsonl"
        _write_jsonl(manifest_path, samples)
        manifest_paths[split] = str(manifest_path)
        split_stats[split] = stats
        LOGGER.info("Split %s: kept %d samples -> %s", split, len(samples), manifest_path)

        if split == "train":
            train_text_records = [{"text": str(row["text"])} for row in samples]

    tokenizer_corpus_path = output_dir / "tokenizer_corpus.jsonl"
    _write_jsonl(tokenizer_corpus_path, train_text_records)

    summary = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "label_id": int(args.label_id),
        "splits": splits,
        "manifests": manifest_paths,
        "tokenizer_corpus": str(tokenizer_corpus_path),
        "split_stats": split_stats,
    }
    summary_path = output_dir / "prepare_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Wrote summary to %s", summary_path)
    return summary


def main(argv: List[str] | None = None) -> int:
    parser = create_prepare_donut_ocr_dataset_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

