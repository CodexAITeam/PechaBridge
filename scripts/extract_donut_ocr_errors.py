#!/usr/bin/env python3
"""Extract OCR samples whose generated text exceeds a CER threshold."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import sys
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoImageProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.train_donut_ocr as train_mod

LOGGER = logging.getLogger("extract_donut_ocr_errors")

IMAGE_KEYS: Tuple[str, ...] = (
    "image",
    "image_path",
    "line_path",
    "line_image",
    "line_image_path",
    "image_rel_path",
    "line_image_rel_path",
    "src__image",
    "path",
)
TEXT_KEYS: Tuple[str, ...] = (
    "text",
    "transcription",
    "label",
    "ocr_text",
    "line_text",
    "normalized_text",
    "content",
    "src__label",
    "src__text",
)
SOURCE_DATASET_KEYS: Tuple[str, ...] = (
    "source_dataset",
    "source_dataset_short",
    "source_split",
    "dataset",
    "source",
    "hf_dataset",
    "dataset_name",
    "src__dataset",
    "src__source_dataset",
)
DEFAULT_EXCLUDED_SOURCE_DATASETS: Tuple[str, ...] = (
    "google books",
    "google_books",
    "google-books",
    "googlebooks",
)


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Donut/TrOCR OCR checkpoint on a train/eval manifest and export "
            "samples whose per-sample CER is above a threshold."
        ),
        add_help=add_help,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint/model directory to evaluate.")
    parser.add_argument("--manifest", type=str, default="", help="JSONL manifest to evaluate.")
    parser.add_argument(
        "--dataset_dir",
        "--dataset-dir",
        dest="dataset_dir",
        type=str,
        default="",
        help="Dataset root used when --manifest is omitted.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="eval",
        choices=["train", "eval", "val", "validation", "test"],
        help="Dataset split to use with --dataset_dir.",
    )
    parser.add_argument(
        "--cer_threshold",
        "--cer-threshold",
        "--threshold",
        dest="cer_threshold",
        type=float,
        default=0.10,
        help="Export samples with sample CER strictly greater than this threshold.",
    )
    parser.add_argument("--output_jsonl", "--output-jsonl", dest="output_jsonl", type=str, default="", help="Error JSONL output path.")
    parser.add_argument("--summary_json", "--summary-json", dest="summary_json", type=str, default="", help="Summary JSON output path.")
    parser.add_argument("--summary_report", "--summary-report", dest="summary_report", type=str, default="", help="Markdown summary report output path.")
    parser.add_argument(
        "--output_dataset_dir",
        "--output-dataset-dir",
        dest="output_dataset_dir",
        type=str,
        default="",
        help="Optional dataset root for high-CER samples. Writes images/, train_manifest.jsonl, and optional val_manifest.jsonl.",
    )
    parser.add_argument(
        "--val_fraction",
        "--val-fraction",
        dest="val_fraction",
        type=float,
        default=0.0,
        help="Fraction of extracted high-CER samples to place in val_manifest.jsonl. 0 writes only train_manifest.jsonl.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic train/val split of extracted errors.")
    parser.add_argument("--dataset_train_manifest", "--dataset-train-manifest", dest="dataset_train_manifest", type=str, default="train_manifest.jsonl", help="Filename for extracted training manifest under --output_dataset_dir.")
    parser.add_argument("--dataset_val_manifest", "--dataset-val-manifest", dest="dataset_val_manifest", type=str, default="val_manifest.jsonl", help="Filename for extracted validation manifest under --output_dataset_dir.")
    parser.add_argument("--dataset_cer_table", "--dataset-cer-table", dest="dataset_cer_table", type=str, default="cer_table.csv", help="Filename for the extracted filename/CER CSV table under --output_dataset_dir.")
    parser.add_argument("--dataset_summary_report", "--dataset-summary-report", dest="dataset_summary_report", type=str, default="summary_report.md", help="Filename for the Markdown quality/performance report under --output_dataset_dir.")
    parser.add_argument(
        "--dataset_image_mode",
        "--dataset-image-mode",
        dest="dataset_image_mode",
        choices=["copy", "reference", "symlink"],
        default="copy",
        help=(
            "How extracted dataset manifests refer to images: copy images into the dataset, "
            "reference original absolute paths, or create symlinks under images/. Default: copy."
        ),
    )
    parser.add_argument("--output_table_csv", "--output-table-csv", dest="output_table_csv", type=str, default="", help="Optional explicit filename/CER CSV table output path.")
    parser.add_argument(
        "--source_datasets",
        "--source-datasets",
        dest="source_datasets",
        type=str,
        default="",
        help="Optional comma-separated allow-list of source_dataset values to evaluate.",
    )
    parser.add_argument(
        "--exclude_source_datasets",
        "--exclude-source-datasets",
        dest="exclude_source_datasets",
        type=str,
        default=",".join(DEFAULT_EXCLUDED_SOURCE_DATASETS),
        help="Comma-separated source_dataset values to skip. Defaults to Google Books variants.",
    )
    parser.add_argument(
        "--include_google_books",
        "--include-google-books",
        dest="include_google_books",
        action="store_true",
        help="Disable the default Google Books source_dataset exclusion.",
    )
    parser.add_argument(
        "--output_all_jsonl",
        "--output-all-jsonl",
        dest="output_all_jsonl",
        type=str,
        default="",
        help="Optional JSONL with every evaluated prediction, not only high-CER rows.",
    )
    parser.add_argument(
        "--copy_images_dir",
        "--copy-images-dir",
        dest="copy_images_dir",
        type=str,
        default="",
        help="Optional directory to copy high-CER sample images into.",
    )
    parser.add_argument("--tokenizer_path", "--tokenizer-path", dest="tokenizer_path", type=str, default="", help="Tokenizer path. Defaults to checkpoint repro/ or run tokenizer.")
    parser.add_argument("--image_processor_path", "--image-processor-path", dest="image_processor_path", type=str, default="", help="Image processor path. Defaults to checkpoint repro/ or run image_processor.")
    parser.add_argument("--image_preprocess_pipeline", "--image-preprocess-pipeline", dest="image_preprocess_pipeline", type=str, default="", choices=["", "none", "pb", "bdrc", "gray", "rgb"], help="Image preprocessing pipeline. Empty tries repro config, else none.")
    parser.add_argument("--image_size", "--image-size", dest="image_size", type=int, default=0, help="Optional square image processor resize override. 0 keeps processor config.")
    parser.add_argument("--enable_letterboxing", "--enable-letterboxing", dest="enable_letterboxing", action="store_true", help="Apply final letterboxing in preprocessing.")
    parser.add_argument("--enable_fixed_resize", "--enable-fixed-resize", dest="enable_fixed_resize", action="store_true", help="Force fixed HxW image processor resize without padding.")
    parser.add_argument("--target_height", "--target-height", dest="target_height", type=int, default=256, help="Target height for letterbox/fixed resize.")
    parser.add_argument("--target_width", "--target-width", dest="target_width", type=int, default=1024, help="Target width for letterbox/fixed resize.")
    parser.add_argument("--max_target_length", "--max-target-length", dest="max_target_length", type=int, default=160, help="Target label token length used for CER reference decoding.")
    parser.add_argument("--generation_max_length", "--generation-max-length", dest="generation_max_length", type=int, default=0, help="Generation max length. 0 keeps checkpoint generation config or falls back to 160.")
    parser.add_argument("--generation_min_new_tokens", "--generation-min-new-tokens", dest="generation_min_new_tokens", type=int, default=0, help="Generation min new tokens. 0 disables.")
    parser.add_argument("--decoder_start_token", "--decoder-start-token", dest="decoder_start_token", type=str, default="<s_ocr>", help="Decoder task start token.")
    parser.add_argument("--extra_special_tokens", "--extra-special-tokens", dest="extra_special_tokens", type=str, default="<NL>,<s_ocr>,</s_ocr>,<s_cls1>", help="Comma-separated extra tokenizer special tokens.")
    parser.add_argument("--metric_newline_token", "--metric-newline-token", dest="metric_newline_token", type=str, choices=["<NL>", "\\n"], default="<NL>", help="Newline token normalization used for CER.")
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Evaluation device.")
    parser.add_argument("--log_stage_timings", "--log-stage-timings", dest="log_stage_timings", action="store_true", help="Log per-batch stage timings for load/preprocess, generate, decode+CER, and write/copy.")
    parser.add_argument("--stage_timing_every_n", "--stage-timing-every-n", dest="stage_timing_every_n", type=int, default=1, help="Log stage timings every N measured batches when --log_stage_timings is enabled.")
    parser.add_argument("--stage_timing_warmup_batches", "--stage-timing-warmup-batches", dest="stage_timing_warmup_batches", type=int, default=0, help="Skip the first N batches from stage timing aggregates.")
    parser.add_argument("--max_samples", "--max-samples", dest="max_samples", type=int, default=0, help="Evaluate at most N manifest rows after --start_index. 0 = all.")
    parser.add_argument("--start_index", "--start-index", dest="start_index", type=int, default=0, help="Skip manifest rows before this 0-based row index.")
    parser.add_argument("--limit_errors", "--limit-errors", dest="limit_errors", type=int, default=0, help="Stop after writing N high-CER errors. 0 = no limit.")
    parser.add_argument("--skip_image_file_check", "--skip-image-file-check", dest="skip_image_file_check", action="store_true", help="Skip image existence checks while scanning manifest.")
    parser.add_argument("--no_row", "--no-row", dest="include_row", action="store_false", help="Do not embed the full original manifest row in outputs.")
    parser.set_defaults(include_row=True)
    return parser


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _resolve_manifest_path(args: argparse.Namespace) -> Path:
    raw_manifest = str(getattr(args, "manifest", "") or "").strip()
    if raw_manifest:
        path = Path(raw_manifest).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        return path

    dataset_dir_raw = str(getattr(args, "dataset_dir", "") or "").strip()
    if not dataset_dir_raw:
        raise ValueError("Provide either --manifest or --dataset_dir with --split.")
    dataset_dir = Path(dataset_dir_raw).expanduser().resolve()
    split = str(getattr(args, "split", "eval") or "eval").strip()
    candidates = [split]
    if split in {"val", "validation"}:
        candidates.extend(["eval", "validation", "val"])
    elif split == "eval":
        candidates.extend(["val", "validation"])
    seen: set[str] = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        path = dataset_dir / name / "meta" / "lines.jsonl"
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Could not find split manifest under {dataset_dir} for split={split!r}")


def _read_manifest_slice(path: Path, *, start_index: int, max_samples: int) -> List[Tuple[int, Dict[str, object]]]:
    rows: List[Tuple[int, Dict[str, object]]] = []
    start = max(0, int(start_index))
    max_n = max(0, int(max_samples))
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < start:
                continue
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{idx + 1}: {exc}") from exc
            if not isinstance(row, dict):
                continue
            rows.append((int(idx), row))
            if max_n > 0 and len(rows) >= max_n:
                break
    return rows


def _first_nonempty(row: Dict[str, object], keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if value is not None and not isinstance(value, (dict, list, tuple)):
            text = str(value)
            if text.strip():
                return text
    return ""


def _normalize_source_dataset(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[\s_\-./:]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_source_dataset_filter(raw: str) -> set[str]:
    return {
        norm
        for norm in (_normalize_source_dataset(part) for part in str(raw or "").split(","))
        if norm
    }


def _source_dataset_for_row(row: Dict[str, object]) -> str:
    direct = _first_nonempty(row, SOURCE_DATASET_KEYS)
    if direct:
        return direct
    for nested_key in ("row", "metadata", "meta", "source_metadata", "src"):
        nested = row.get(nested_key)
        if isinstance(nested, dict):
            value = _first_nonempty(nested, SOURCE_DATASET_KEYS)
            if value:
                return value
    return ""


def _source_dataset_allowed(
    row: Dict[str, object],
    *,
    include_sources: set[str],
    exclude_sources: set[str],
) -> Tuple[bool, str, str]:
    source = _source_dataset_for_row(row)
    norm = _normalize_source_dataset(source)
    if include_sources and norm not in include_sources:
        return False, source, "not_included"
    if exclude_sources and norm in exclude_sources:
        return False, source, "excluded"
    return True, source, ""


def _checkpoint_step(path: Path) -> Optional[int]:
    match = re.search(r"checkpoint-(\d+)", str(path))
    return int(match.group(1)) if match else None


def _default_output_jsonl(checkpoint: Path, manifest: Path, threshold: float) -> Path:
    step = _checkpoint_step(checkpoint)
    step_part = f"step{step}" if step is not None else "model"
    threshold_part = f"{float(threshold):.4f}".replace(".", "p")
    return checkpoint / f"cer_errors_{manifest.stem}_{step_part}_gt_{threshold_part}.jsonl"


def _dataset_split_for_row(row_index: int, *, val_fraction: float, seed: int) -> str:
    frac = max(0.0, min(1.0, float(val_fraction)))
    if frac <= 0.0:
        return "train"
    digest = sha256(f"{int(seed)}:{int(row_index)}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False) / float(2**64 - 1)
    return "val" if value < frac else "train"


def _dataset_image_rel_path(src: Path, *, row_index: int, cer: float) -> str:
    suffix = src.suffix if src.suffix else ".png"
    name = f"row_{int(row_index):09d}_cer_{float(cer):.4f}".replace(".", "p") + suffix
    return str(Path("images") / name)


def _resolve_tokenizer_path(args: argparse.Namespace, checkpoint: Path) -> str:
    explicit = str(getattr(args, "tokenizer_path", "") or "").strip()
    if explicit:
        return explicit
    candidates = [
        checkpoint / "repro" / "tokenizer",
        checkpoint.parent / "tokenizer",
        checkpoint,
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError("Could not infer tokenizer path. Provide --tokenizer_path.")


def _resolve_image_processor_path(args: argparse.Namespace, checkpoint: Path) -> str:
    explicit = str(getattr(args, "image_processor_path", "") or "").strip()
    if explicit:
        return explicit
    candidates = [
        checkpoint / "repro" / "image_processor",
        checkpoint.parent / "image_processor",
        checkpoint,
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError("Could not infer image processor path. Provide --image_processor_path.")


def _infer_preprocess_pipeline(args: argparse.Namespace, checkpoint: Path) -> str:
    explicit = str(getattr(args, "image_preprocess_pipeline", "") or "").strip().lower()
    if explicit:
        return explicit
    repro_contract = checkpoint / "repro" / "image_preprocess.json"
    if repro_contract.exists():
        try:
            data = json.loads(repro_contract.read_text(encoding="utf-8"))
            pipeline = str(data.get("pipeline", "") or "").strip().lower()
            if pipeline:
                return pipeline
        except Exception as exc:
            LOGGER.warning("Could not read repro image preprocess contract %s: %s", repro_contract, exc)
    return "none"


def _processor_hw(image_processor) -> Tuple[Optional[int], Optional[int]]:
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict):
        h = size.get("height")
        w = size.get("width")
        try:
            return int(h), int(w)
        except Exception:
            return None, None
    if isinstance(size, int):
        return int(size), int(size)
    return None, None


def _configure_image_processor(args: argparse.Namespace, image_processor) -> None:
    image_size = int(getattr(args, "image_size", 0) or 0)
    if image_size > 0:
        train_mod._configure_image_processor(image_processor, image_size)
    if bool(getattr(args, "enable_letterboxing", False)):
        try:
            image_processor.size = {
                "height": int(getattr(args, "target_height")),
                "width": int(getattr(args, "target_width")),
            }
        except Exception as exc:
            LOGGER.warning("Could not set letterbox image processor size: %s", exc)
    if bool(getattr(args, "enable_fixed_resize", False)):
        train_mod._configure_image_processor_fixed_resize(
            image_processor,
            int(getattr(args, "target_height")),
            int(getattr(args, "target_width")),
        )


def _configure_model_geometry(model, image_processor, args: argparse.Namespace, checkpoint: Path) -> None:
    target_h, target_w = _processor_hw(image_processor)
    if bool(getattr(args, "enable_letterboxing", False)) or bool(getattr(args, "enable_fixed_resize", False)):
        target_h = int(getattr(args, "target_height"))
        target_w = int(getattr(args, "target_width"))
    if target_h is None or target_w is None:
        return
    try:
        model.config.encoder.image_size = [int(target_h), int(target_w)]
    except Exception:
        pass
    try:
        if hasattr(model, "encoder") and hasattr(model.encoder, "config"):
            model.encoder.config.image_size = [int(target_h), int(target_w)]
    except Exception:
        pass
    try:
        train_mod._set_encoder_patch_image_size(model, int(target_h), int(target_w))
    except Exception as exc:
        LOGGER.warning("Could not set encoder patch image size: %s", exc)
    try:
        train_mod._materialize_encoder_position_embeddings(
            model,
            target_h=int(target_h),
            target_w=int(target_w),
            checkpoint_source=str(checkpoint),
        )
    except Exception as exc:
        LOGGER.warning("Could not materialize encoder position embeddings: %s", exc)


def _build_tokenizer(args: argparse.Namespace, checkpoint: Path):
    tokenizer_path = _resolve_tokenizer_path(args, checkpoint)
    tokenizer = train_mod._load_tokenizer_robust(tokenizer_path)
    extra = train_mod._parse_csv_tokens(str(getattr(args, "extra_special_tokens", "") or ""))
    if extra:
        tokenizer.add_special_tokens({"additional_special_tokens": extra})
    train_mod._ensure_pad_token(tokenizer)
    LOGGER.info("Using tokenizer: %s len=%d", tokenizer_path, int(len(tokenizer)))
    return tokenizer


def _configure_generation(model, tokenizer, args: argparse.Namespace) -> None:
    decoder_start_token = str(getattr(args, "decoder_start_token", "<s_ocr>") or "<s_ocr>")
    decoder_start_candidates = train_mod._special_ids_for_token_literal(tokenizer, decoder_start_token)
    decoder_start_id = train_mod._resolve_single_token_id_no_mutation(tokenizer, decoder_start_token)
    checkpoint_decoder_start_id = getattr(getattr(model, "config", None), "decoder_start_token_id", None)
    try:
        checkpoint_decoder_start_id_int = int(checkpoint_decoder_start_id)
    except Exception:
        checkpoint_decoder_start_id_int = None
    if checkpoint_decoder_start_id_int is not None and checkpoint_decoder_start_id_int in decoder_start_candidates:
        decoder_start_id = checkpoint_decoder_start_id_int
    if decoder_start_id is None:
        raise ValueError(f"decoder_start_token not found in tokenizer: {decoder_start_token!r}")
    task_end_token = train_mod._paired_end_token_for_start(decoder_start_token)
    task_end_candidates = train_mod._special_ids_for_token_literal(tokenizer, task_end_token) if task_end_token else []
    task_end_id = train_mod._resolve_single_token_id_no_mutation(tokenizer, task_end_token) if task_end_token else None
    checkpoint_eos_id = getattr(getattr(model, "config", None), "eos_token_id", None)
    try:
        checkpoint_eos_id_int = int(checkpoint_eos_id)
    except Exception:
        checkpoint_eos_id_int = None
    if checkpoint_eos_id_int is not None and checkpoint_eos_id_int in task_end_candidates:
        task_end_id = checkpoint_eos_id_int
    effective_eos = int(task_end_id) if task_end_id is not None else (
        int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None
    )
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer pad_token_id is None.")

    model.config.decoder_start_token_id = int(decoder_start_id)
    model.config.pad_token_id = int(tokenizer.pad_token_id)
    if effective_eos is not None:
        model.config.eos_token_id = int(effective_eos)
    model.generation_config.decoder_start_token_id = int(decoder_start_id)
    model.generation_config.pad_token_id = int(tokenizer.pad_token_id)
    if effective_eos is not None:
        model.generation_config.eos_token_id = int(effective_eos)
    for config_obj in (
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "decoder", None),
        getattr(getattr(model, "decoder", None), "config", None),
        getattr(model, "generation_config", None),
    ):
        try:
            setattr(config_obj, "use_cache", True)
        except Exception:
            pass
    gen_max = int(getattr(args, "generation_max_length", 0) or 0)
    if gen_max <= 0:
        gen_max = int(getattr(model.generation_config, "max_length", 0) or 0) or 160
    model.generation_config.max_length = int(gen_max)
    model.generation_config.num_beams = 1
    model.generation_config.do_sample = False
    try:
        model.generation_config.no_repeat_ngram_size = 3
    except Exception:
        pass
    try:
        model.generation_config.repetition_penalty = 1.15
    except Exception:
        pass
    suppress_start_ids = train_mod._special_ids_for_token_literal(tokenizer, decoder_start_token)
    if not suppress_start_ids and int(decoder_start_id) in set(
        int(x) for x in getattr(tokenizer, "all_special_ids", []) if x is not None
    ):
        suppress_start_ids = [int(decoder_start_id)]
    try:
        model.generation_config.suppress_tokens = [int(x) for x in suppress_start_ids]
    except Exception:
        pass
    try:
        model.generation_config.bad_words_ids = [[int(x)] for x in suppress_start_ids]
    except Exception:
        pass
    min_new = max(0, int(getattr(args, "generation_min_new_tokens", 0) or 0))
    try:
        model.generation_config.min_new_tokens = int(min_new) if min_new > 0 else None
    except Exception:
        pass
    LOGGER.info(
        "Generation config: start_id=%s eos_id=%s pad_id=%s max_length=%d min_new=%d use_cache=%s",
        str(decoder_start_id),
        str(effective_eos),
        str(tokenizer.pad_token_id),
        int(gen_max),
        int(min_new),
        str(getattr(model.generation_config, "use_cache", None)),
    )
    try:
        train_mod._log_token_audit(
            tokenizer,
            decoder_start_token=decoder_start_token,
            task_end_token=task_end_token,
        )
    except Exception as exc:
        LOGGER.warning("Tokenizer audit failed: %s", exc)


def _load_repro_generate_config(checkpoint: Path) -> Dict[str, object]:
    path = Path(checkpoint) / "repro" / "generate_config.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Could not read repro generate config %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _apply_repro_generate_config(model, checkpoint: Path) -> Dict[str, object]:
    """Apply checkpoint repro generate_config fields for UI/train parity."""
    cfg = _load_repro_generate_config(checkpoint)
    if not cfg:
        return {}
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is None:
        return cfg
    applied: Dict[str, object] = {}
    for key in (
        "decoder_start_token_id",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "max_length",
        "max_new_tokens",
        "min_new_tokens",
        "num_beams",
        "do_sample",
        "temperature",
        "top_p",
        "repetition_penalty",
        "no_repeat_ngram_size",
        "length_penalty",
        "early_stopping",
        "bad_words_ids",
        "forced_bos_token_id",
        "forced_eos_token_id",
        "use_cache",
    ):
        value = cfg.get(key)
        if value is None:
            continue
        try:
            setattr(gen_cfg, key, value)
            if key in {"decoder_start_token_id", "eos_token_id", "pad_token_id"}:
                setattr(model.config, key, value)
            applied[key] = value
        except Exception as exc:
            LOGGER.warning("Could not apply repro generation field %s=%r: %s", key, value, exc)
    if applied:
        LOGGER.info(
            "Applied repro generate_config fields for extraction parity: %s",
            json.dumps(applied, ensure_ascii=False, default=str),
        )
    return cfg


def _build_generate_kwargs(model, pixel_values: torch.Tensor, args: argparse.Namespace) -> Dict[str, object]:
    gen_cfg = getattr(model, "generation_config", None)
    cfg_keys = (
        "decoder_start_token_id",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "max_length",
        "max_new_tokens",
        "min_new_tokens",
        "num_beams",
        "do_sample",
        "temperature",
        "top_p",
        "repetition_penalty",
        "no_repeat_ngram_size",
        "length_penalty",
        "early_stopping",
        "bad_words_ids",
        "forced_bos_token_id",
        "forced_eos_token_id",
        "use_cache",
    )
    kwargs: Dict[str, object] = {"pixel_values": pixel_values}
    for key in cfg_keys:
        value = getattr(gen_cfg, key, None) if gen_cfg is not None else None
        if value is not None:
            kwargs[key] = value
    gen_max = int(getattr(args, "generation_max_length", 0) or 0)
    if gen_max > 0:
        kwargs["max_length"] = int(gen_max)
    elif "max_length" not in kwargs and "max_new_tokens" not in kwargs:
        kwargs["max_length"] = 160
    if "num_beams" not in kwargs:
        kwargs["num_beams"] = 1
    if "do_sample" not in kwargs:
        kwargs["do_sample"] = False
    min_new_tokens = max(0, int(getattr(args, "generation_min_new_tokens", 0) or 0))
    if min_new_tokens > 0:
        kwargs["min_new_tokens"] = int(min_new_tokens)
    elif int(getattr(args, "generation_min_new_tokens", 0) or 0) == 0:
        kwargs.pop("min_new_tokens", None)
    if _should_force_interpolate_pos_encoding(model, pixel_values):
        kwargs["interpolate_pos_encoding"] = True
    return {k: v for k, v in kwargs.items() if v is not None}


def _decoder_vocab_size(model) -> Optional[int]:
    try:
        emb = model.decoder.get_input_embeddings()
        return int(emb.num_embeddings)
    except Exception:
        return None


def _ensure_vocab_size(model, tokenizer) -> None:
    decoder_vocab = _decoder_vocab_size(model)
    tok_len = int(len(tokenizer))
    if decoder_vocab is not None and int(decoder_vocab) != tok_len:
        LOGGER.warning("Resizing decoder token embeddings: model=%d tokenizer=%d", int(decoder_vocab), tok_len)
        model.decoder.resize_token_embeddings(tok_len)
        try:
            model.config.vocab_size = model.decoder.config.vocab_size
        except Exception:
            pass


def _patch_size(model) -> Tuple[int, int]:
    patch_size = getattr(getattr(getattr(getattr(model, "encoder", None), "embeddings", None), "patch_embeddings", None), "patch_size", None)
    try:
        if isinstance(patch_size, int):
            return max(1, int(patch_size)), max(1, int(patch_size))
        if patch_size is not None:
            return max(1, int(patch_size[0])), max(1, int(patch_size[1]))
    except Exception:
        pass
    return 16, 16


def _should_force_interpolate_pos_encoding(model, pixel_values: torch.Tensor) -> bool:
    try:
        if not train_mod._encoder_supports_interpolate_pos_encoding(model):
            return False
        if not train_mod._encoder_pos_embeddings_are_square_grid(model):
            return False
        current_count = train_mod._encoder_position_embedding_count(model)
        ph, pw = _patch_size(model)
        target_count = int(max(1, int(pixel_values.shape[-2]) // ph) * max(1, int(pixel_values.shape[-1]) // pw) + 1)
        if current_count is not None and int(current_count) == int(target_count):
            return False
        return True
    except Exception:
        return False


@dataclass
class ExtractionSample:
    row_index: int
    row: Dict[str, object]
    image_path: Path
    text: str


class ExtractionDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Tuple[int, Dict[str, object]]],
        *,
        manifest_path: Path,
        image_processor,
        tokenizer,
        image_preprocess_pipeline: str,
        max_target_length: int,
        decoder_start_token: str,
        skip_image_file_check: bool,
        enable_letterboxing: bool,
        target_height: int,
        target_width: int,
        include_source_datasets: set[str],
        exclude_source_datasets: set[str],
    ):
        self.manifest_path = manifest_path
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.image_preprocess_pipeline = str(image_preprocess_pipeline or "none")
        self.max_target_length = int(max_target_length)
        self.decoder_start_token = str(decoder_start_token or "")
        self.task_end_token = train_mod._paired_end_token_for_start(self.decoder_start_token)
        self.target_end_token_id = (
            train_mod._resolve_single_token_id_no_mutation(tokenizer, self.task_end_token)
            if self.task_end_token
            else None
        )
        self.enable_letterboxing = bool(enable_letterboxing)
        self.target_height = max(1, int(target_height))
        self.target_width = max(1, int(target_width))
        self.samples: List[ExtractionSample] = []
        self.stats = {
            "rows_total": 0,
            "kept": 0,
            "source_not_included": 0,
            "source_excluded": 0,
            "missing_image_field": 0,
            "missing_text_field": 0,
            "image_not_found": 0,
            "source_counts": {},
            "kept_source_counts": {},
        }
        for row_index, row in rows:
            self.stats["rows_total"] += 1
            allowed, source_dataset, source_filter_reason = _source_dataset_allowed(
                row,
                include_sources=include_source_datasets,
                exclude_sources=exclude_source_datasets,
            )
            source_key = str(source_dataset or "UNKNOWN")
            source_counts = self.stats["source_counts"]
            if isinstance(source_counts, dict):
                source_counts[source_key] = int(source_counts.get(source_key, 0)) + 1
            if not allowed:
                if source_filter_reason == "not_included":
                    self.stats["source_not_included"] += 1
                else:
                    self.stats["source_excluded"] += 1
                continue
            image_raw = _first_nonempty(row, IMAGE_KEYS)
            text_raw = _first_nonempty(row, TEXT_KEYS)
            if not image_raw:
                self.stats["missing_image_field"] += 1
                continue
            if not text_raw:
                self.stats["missing_text_field"] += 1
                continue
            image_path = train_mod._resolve_manifest_image_path(
                image_raw,
                self.manifest_path,
                skip_file_check=bool(skip_image_file_check),
            )
            if image_path is None:
                self.stats["image_not_found"] += 1
                continue
            self.samples.append(
                ExtractionSample(
                    row_index=int(row_index),
                    row=dict(row),
                    image_path=Path(image_path),
                    text=str(text_raw),
                )
            )
            self.stats["kept"] += 1
            kept_source_counts = self.stats["kept_source_counts"]
            if isinstance(kept_source_counts, dict):
                kept_source_counts[source_key] = int(kept_source_counts.get(source_key, 0)) + 1
        LOGGER.info("Loaded extraction dataset: %s", json.dumps(self.stats, ensure_ascii=False))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        with Image.open(sample.image_path) as image:
            rgb = image.convert("RGB")
        rgb = train_mod._apply_image_preprocess_pipeline(
            rgb,
            self.image_preprocess_pipeline,
            enable_letterboxing=self.enable_letterboxing,
            target_width=self.target_width,
            target_height=self.target_height,
        )
        if hasattr(self.image_processor, "preprocess"):
            proc = self.image_processor.preprocess(images=rgb, return_tensors="pt")
        else:
            proc = self.image_processor(images=rgb, return_tensors="pt")
        target_text = train_mod._format_ocr_target_text(
            sample.text,
            start_token=self.decoder_start_token,
            end_token=self.task_end_token,
        )
        labels = train_mod._encode_target_ids_with_terminal_preservation(
            self.tokenizer,
            target_text,
            max_target_length=self.max_target_length,
            terminal_token_id=self.target_end_token_id,
        )
        return {
            "pixel_values": proc.pixel_values.squeeze(0),
            "labels": labels,
            "meta": {
                "dataset_index": int(idx),
                "row_index": int(sample.row_index),
                "image": str(sample.image_path),
                "text_raw": sample.text,
                "row": sample.row,
            },
        }


def _pad_pixel_values(values: Sequence[torch.Tensor]) -> torch.Tensor:
    if not values:
        return torch.empty(0)
    shapes = [tuple(int(d) for d in v.shape) for v in values]
    if all(shape == shapes[0] for shape in shapes):
        return torch.stack(list(values), dim=0)
    channels = max(int(v.shape[0]) for v in values)
    height = max(int(v.shape[-2]) for v in values)
    width = max(int(v.shape[-1]) for v in values)
    dtype = values[0].dtype
    out = values[0].new_zeros((len(values), channels, height, width), dtype=dtype)
    for i, value in enumerate(values):
        c, h, w = int(value.shape[0]), int(value.shape[-2]), int(value.shape[-1])
        out[i, :c, :h, :w] = value
    return out


def _collate(features: Sequence[Dict[str, object]], *, pad_token_id: int) -> Dict[str, object]:
    pixel_values = _pad_pixel_values([f["pixel_values"] for f in features])  # type: ignore[index]
    labels_list = [list(f["labels"]) for f in features]  # type: ignore[index]
    max_len = max((len(x) for x in labels_list), default=0)
    labels = torch.full((len(labels_list), max_len), int(pad_token_id), dtype=torch.long)
    for i, ids in enumerate(labels_list):
        if ids:
            labels[i, : len(ids)] = torch.tensor([int(x) for x in ids], dtype=torch.long)
    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "meta": [f["meta"] for f in features],
    }


def _copy_error_image(src: Path, dst_dir: Path, *, row_index: int, cer: float) -> str:
    dst_dir.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix or ".png"
    name = f"row_{int(row_index):09d}_cer_{float(cer):.4f}".replace(".", "p") + suffix
    dst = dst_dir / name
    try:
        shutil.copy2(src, dst)
        return str(dst)
    except Exception as exc:
        LOGGER.warning("Could not copy image %s -> %s: %s", src, dst, exc)
        return ""


def _write_dataset_manifest_row(
    fp,
    *,
    dataset_dir: Path,
    source_image: Path,
    source_row: Optional[Dict[str, object]],
    record: Dict[str, object],
    split_name: str,
    image_mode: str,
) -> str:
    mode = str(image_mode or "copy").strip().lower()
    if mode == "reference":
        image_ref = str(source_image)
    else:
        image_ref = _dataset_image_rel_path(source_image, row_index=int(record["row_index"]), cer=float(record["cer"]))
        dst_image = dataset_dir / image_ref
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        if not dst_image.exists():
            if mode == "symlink":
                try:
                    dst_image.symlink_to(source_image)
                except Exception as exc:
                    LOGGER.warning("Could not symlink image %s -> %s: %s", dst_image, source_image, exc)
                    shutil.copy2(source_image, dst_image)
                    mode = "copy"
            else:
                shutil.copy2(source_image, dst_image)

    manifest_row: Dict[str, object] = dict(source_row or {})
    manifest_row["image"] = image_ref
    manifest_row["text"] = str(record.get("text_raw", "") or manifest_row.get("text", "") or "")
    manifest_row["source_image"] = str(source_image)
    manifest_row["source_manifest"] = str(record.get("manifest", ""))
    manifest_row["source_row_index"] = int(record.get("row_index", -1))
    manifest_row["error_split"] = str(split_name)
    manifest_row["error_image_mode"] = str(mode)
    manifest_row["error_cer"] = float(record.get("cer", 0.0) or 0.0)
    manifest_row["error_edit_distance"] = int(record.get("edit_distance", 0) or 0)
    manifest_row["error_ref_len"] = int(record.get("ref_len", 0) or 0)
    manifest_row["error_pred_len"] = int(record.get("pred_len", 0) or 0)
    manifest_row["error_pred"] = str(record.get("pred", "") or "")
    manifest_row["error_ref"] = str(record.get("ref", "") or "")
    manifest_row["error_checkpoint"] = str(record.get("checkpoint", "") or "")
    fp.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")
    return image_ref


def _safe_float_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"min": None, "max": None, "mean": None, "std": None}
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def _add_stage_timing(
    totals: Dict[str, float],
    maximums: Dict[str, float],
    name: str,
    seconds: float,
) -> None:
    value = max(0.0, float(seconds))
    totals[name] = float(totals.get(name, 0.0) + value)
    maximums[name] = max(float(maximums.get(name, 0.0)), value)


def _stage_timing_summary(
    totals: Dict[str, float],
    maximums: Dict[str, float],
    measured_batches: int,
    *,
    enabled: bool,
    every_n: int,
    warmup_batches: int,
) -> Dict[str, object]:
    count = max(0, int(measured_batches))
    total_batch = float(totals.get("batch_total", 0.0))
    stage_names = [
        "load_preprocess_wait",
        "to_device",
        "generate",
        "decode_cer",
        "write_copy",
        "batch_total",
    ]
    means = {
        name: (float(totals.get(name, 0.0)) / max(1, count) if count > 0 else 0.0)
        for name in stage_names
    }
    shares = {
        name: (float(totals.get(name, 0.0)) / max(1e-9, total_batch) if total_batch > 0 else 0.0)
        for name in stage_names
        if name != "batch_total"
    }
    return {
        "enabled": bool(enabled),
        "logged_every_n": int(every_n),
        "warmup_batches": int(warmup_batches),
        "batches_measured": int(count),
        "totals_seconds": {name: float(totals.get(name, 0.0)) for name in stage_names},
        "mean_seconds_per_batch": means,
        "max_seconds_per_batch": {name: float(maximums.get(name, 0.0)) for name in stage_names},
        "share_of_measured_batch_total": shares,
    }


def _safe_percentile_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    keys = ("p50", "p75", "p90", "p95", "p99")
    if not values:
        return {key: None for key in keys}
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def _cer_bucket_counts(values: Sequence[float]) -> Dict[str, int]:
    buckets = {
        "eq_0": 0,
        "gt_0_lte_0p01": 0,
        "gt_0p01_lte_0p05": 0,
        "gt_0p05_lte_0p10": 0,
        "gt_0p10_lte_0p20": 0,
        "gt_0p20_lte_0p50": 0,
        "gt_0p50_lte_1p00": 0,
        "gt_1p00": 0,
    }
    for value in values:
        v = float(value)
        if v <= 0.0:
            buckets["eq_0"] += 1
        elif v <= 0.01:
            buckets["gt_0_lte_0p01"] += 1
        elif v <= 0.05:
            buckets["gt_0p01_lte_0p05"] += 1
        elif v <= 0.10:
            buckets["gt_0p05_lte_0p10"] += 1
        elif v <= 0.20:
            buckets["gt_0p10_lte_0p20"] += 1
        elif v <= 0.50:
            buckets["gt_0p20_lte_0p50"] += 1
        elif v <= 1.00:
            buckets["gt_0p50_lte_1p00"] += 1
        else:
            buckets["gt_1p00"] += 1
    return buckets


def _fmt_report_value(value: object, *, digits: int = 4, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if percent:
            return f"{value * 100.0:.2f}%"
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(rows: Sequence[Tuple[str, object]], *, value_digits: int = 4) -> str:
    lines = ["| Metric | Value |", "| --- | --- |"]
    for key, value in rows:
        lines.append(f"| {key} | {_fmt_report_value(value, digits=value_digits)} |")
    return "\n".join(lines)


def _write_summary_report(summary: Dict[str, object], path: Path) -> None:
    quality = summary.get("quality_metrics")
    quality = quality if isinstance(quality, dict) else {}
    performance = summary.get("performance_metrics")
    performance = performance if isinstance(performance, dict) else {}
    length_metrics = summary.get("length_metrics")
    length_metrics = length_metrics if isinstance(length_metrics, dict) else {}
    cer_distribution = summary.get("cer_distribution")
    cer_distribution = cer_distribution if isinstance(cer_distribution, dict) else {}
    outputs = summary.get("outputs")
    outputs = outputs if isinstance(outputs, dict) else {}

    bucket_labels = {
        "eq_0": "CER = 0",
        "gt_0_lte_0p01": "0 < CER <= 1%",
        "gt_0p01_lte_0p05": "1% < CER <= 5%",
        "gt_0p05_lte_0p10": "5% < CER <= 10%",
        "gt_0p10_lte_0p20": "10% < CER <= 20%",
        "gt_0p20_lte_0p50": "20% < CER <= 50%",
        "gt_0p50_lte_1p00": "50% < CER <= 100%",
        "gt_1p00": "CER > 100%",
    }
    valid_n = max(1, int(quality.get("valid_n", 0) or 0))
    bucket_lines = ["| Bucket | Count | Ratio |", "| --- | ---: | ---: |"]
    bucket_counts = cer_distribution.get("buckets")
    bucket_counts = bucket_counts if isinstance(bucket_counts, dict) else {}
    for key, label in bucket_labels.items():
        count = int(bucket_counts.get(key, 0) or 0)
        bucket_lines.append(f"| {label} | {count} | {count / valid_n:.2%} |")

    stage_timing = performance.get("stage_timing")
    stage_timing = stage_timing if isinstance(stage_timing, dict) else {}
    stage_means = stage_timing.get("mean_seconds_per_batch")
    stage_means = stage_means if isinstance(stage_means, dict) else {}
    stage_shares = stage_timing.get("share_of_measured_batch_total")
    stage_shares = stage_shares if isinstance(stage_shares, dict) else {}
    stage_section: List[str] = []
    if bool(stage_timing.get("enabled", False)):
        stage_section = [
            "",
            "## Stage Timing",
            _markdown_table(
                [
                    ("Measured batches", stage_timing.get("batches_measured")),
                    ("Mean load/preprocess wait seconds", stage_means.get("load_preprocess_wait")),
                    ("Mean to-device seconds", stage_means.get("to_device")),
                    ("Mean generate seconds", stage_means.get("generate")),
                    ("Mean decode+CER seconds", stage_means.get("decode_cer")),
                    ("Mean write/copy seconds", stage_means.get("write_copy")),
                    ("Generate share", f"{float(stage_shares.get('generate', 0.0) or 0.0):.2%}"),
                    ("Write/copy share", f"{float(stage_shares.get('write_copy', 0.0) or 0.0):.2%}"),
                ]
            ),
        ]

    sections = [
        "# Donut OCR Error Extraction Report",
        "",
        "## Run",
        _markdown_table(
            [
                ("Checkpoint", summary.get("checkpoint", "")),
                ("Manifest", summary.get("manifest", "")),
                ("CER threshold", quality.get("cer_threshold")),
                ("Device", summary.get("device", "")),
                ("Image pipeline", summary.get("image_preprocess_pipeline", "")),
                ("Image processor size", summary.get("image_processor_size", "")),
            ]
        ),
        "",
        "## Quality",
        _markdown_table(
            [
                (
                    "Global CER",
                    f"{float(quality.get('global_cer', 0.0) or 0.0):.4f} ({float(quality.get('global_cer_percent', 0.0) or 0.0):.2f}%)",
                ),
                ("Sample CER mean", quality.get("sample_cer_mean")),
                ("Sample CER median", quality.get("sample_cer_p50")),
                ("Sample CER p95", quality.get("sample_cer_p95")),
                ("Sample CER max", quality.get("sample_cer_max")),
                ("Manifest rows selected", quality.get("manifest_rows_selected")),
                ("Usable samples after scan", quality.get("usable_samples")),
                ("Total evaluated samples", quality.get("total_seen")),
                ("Valid samples", quality.get("valid_n")),
                ("High-CER samples", quality.get("error_n")),
                ("High-CER ratio", f"{float(quality.get('error_ratio_valid', 0.0) or 0.0):.2%}"),
                ("Empty predictions", quality.get("empty_pred_count")),
                ("Empty prediction ratio", f"{float(quality.get('empty_pred_ratio', 0.0) or 0.0):.2%}"),
                ("Empty references", quality.get("empty_ref_count")),
            ]
        ),
        "",
        "## Lengths",
        _markdown_table(
            [
                ("Prediction length mean", length_metrics.get("pred_len_mean")),
                ("Prediction length p95", length_metrics.get("pred_len_p95")),
                ("Reference length mean", length_metrics.get("ref_len_mean")),
                ("Reference length p95", length_metrics.get("ref_len_p95")),
                ("Prediction/reference length ratio mean", length_metrics.get("pred_ref_len_ratio_mean")),
            ]
        ),
        "",
        "## CER Distribution",
        "\n".join(bucket_lines),
        "",
        "## Performance",
        _markdown_table(
            [
                ("Total wall time seconds", performance.get("total_wall_time_seconds")),
                ("Evaluation time seconds", performance.get("eval_time_seconds")),
                ("Samples per second", performance.get("samples_per_second")),
                ("Batches per second", performance.get("batches_per_second")),
                ("Batch size", performance.get("batch_size")),
                ("Num workers", performance.get("num_workers")),
                ("Batches", performance.get("batches")),
                ("CUDA peak memory GB", performance.get("cuda_peak_memory_gb")),
            ]
        ),
        *stage_section,
        "",
        "## Outputs",
        _markdown_table(
            [
                ("Errors JSONL", outputs.get("errors_jsonl", "")),
                ("All predictions JSONL", outputs.get("all_jsonl", "")),
                ("CER table CSV", outputs.get("cer_table_csv", "")),
                ("Dataset dir", outputs.get("dataset_dir", "")),
                ("Train manifest", outputs.get("train_manifest", "")),
                ("Val manifest", outputs.get("val_manifest", "")),
                ("Summary JSON", outputs.get("summary_json", "")),
                ("Summary report", str(path)),
            ]
        ),
        "",
    ]
    path.write_text("\n".join(sections), encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, object]:
    _configure_logging()
    run_start_time = time.perf_counter()
    if bool(getattr(args, "enable_letterboxing", False)) and bool(getattr(args, "enable_fixed_resize", False)):
        raise ValueError("--enable_letterboxing and --enable_fixed_resize are mutually exclusive.")
    checkpoint = Path(str(args.checkpoint)).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    manifest = _resolve_manifest_path(args)
    rows = _read_manifest_slice(
        manifest,
        start_index=int(getattr(args, "start_index", 0) or 0),
        max_samples=int(getattr(args, "max_samples", 0) or 0),
    )
    if not rows:
        raise RuntimeError(f"No manifest rows selected from {manifest}")

    threshold = float(getattr(args, "cer_threshold", 0.10))
    output_dataset_dir = Path(str(args.output_dataset_dir)).expanduser().resolve() if str(getattr(args, "output_dataset_dir", "") or "").strip() else None
    if output_dataset_dir is not None:
        output_dataset_dir.mkdir(parents=True, exist_ok=True)
    if str(args.output_jsonl or "").strip():
        output_jsonl = Path(str(args.output_jsonl)).expanduser().resolve()
    elif output_dataset_dir is not None:
        output_jsonl = output_dataset_dir / "meta" / "errors.jsonl"
    else:
        output_jsonl = _default_output_jsonl(checkpoint, manifest, threshold)
    if str(args.summary_json or "").strip():
        summary_json = Path(str(args.summary_json)).expanduser().resolve()
    elif output_dataset_dir is not None:
        summary_json = output_dataset_dir / "summary.json"
    else:
        summary_json = output_jsonl.with_suffix(".summary.json")
    if str(getattr(args, "summary_report", "") or "").strip():
        summary_report = Path(str(args.summary_report)).expanduser().resolve()
    elif output_dataset_dir is not None:
        summary_report = output_dataset_dir / str(getattr(args, "dataset_summary_report", "summary_report.md") or "summary_report.md")
    else:
        summary_report = output_jsonl.with_suffix(".summary_report.md")
    output_all_jsonl = Path(str(args.output_all_jsonl)).expanduser().resolve() if str(getattr(args, "output_all_jsonl", "") or "").strip() else None
    if str(getattr(args, "output_table_csv", "") or "").strip():
        output_table_csv = Path(str(args.output_table_csv)).expanduser().resolve()
    elif output_dataset_dir is not None:
        output_table_csv = output_dataset_dir / str(getattr(args, "dataset_cer_table", "cer_table.csv") or "cer_table.csv")
    else:
        output_table_csv = None
    copy_images_dir = Path(str(args.copy_images_dir)).expanduser().resolve() if str(getattr(args, "copy_images_dir", "") or "").strip() else None
    dataset_train_manifest = None
    dataset_val_manifest = None
    dataset_train_fp = None
    dataset_val_fp = None
    table_fp = None
    table_writer = None
    dataset_counts = {"train": 0, "val": 0}
    include_source_datasets = _parse_source_dataset_filter(str(getattr(args, "source_datasets", "") or ""))
    exclude_source_raw = "" if bool(getattr(args, "include_google_books", False)) else str(getattr(args, "exclude_source_datasets", "") or "")
    exclude_source_datasets = _parse_source_dataset_filter(exclude_source_raw)
    LOGGER.info(
        "Source dataset filter: include=%s exclude=%s",
        sorted(include_source_datasets),
        sorted(exclude_source_datasets),
    )

    tokenizer = _build_tokenizer(args, checkpoint)
    image_processor_path = _resolve_image_processor_path(args, checkpoint)
    image_processor = AutoImageProcessor.from_pretrained(image_processor_path)
    _configure_image_processor(args, image_processor)
    image_preprocess_pipeline = _infer_preprocess_pipeline(args, checkpoint)
    LOGGER.info("Using image_processor=%s preprocess=%s size=%s", image_processor_path, image_preprocess_pipeline, getattr(image_processor, "size", None))

    dataset = ExtractionDataset(
        rows,
        manifest_path=manifest,
        image_processor=image_processor,
        tokenizer=tokenizer,
        image_preprocess_pipeline=image_preprocess_pipeline,
        max_target_length=int(getattr(args, "max_target_length", 160) or 160),
        decoder_start_token=str(getattr(args, "decoder_start_token", "<s_ocr>") or "<s_ocr>"),
        skip_image_file_check=bool(getattr(args, "skip_image_file_check", False)),
        enable_letterboxing=bool(getattr(args, "enable_letterboxing", False)),
        target_height=int(getattr(args, "target_height", 256) or 256),
        target_width=int(getattr(args, "target_width", 1024) or 1024),
        include_source_datasets=include_source_datasets,
        exclude_source_datasets=exclude_source_datasets,
    )
    if len(dataset) <= 0:
        raise RuntimeError("No usable samples after manifest scan.")

    model = train_mod._load_ved_model_robust(str(checkpoint))
    _ensure_vocab_size(model, tokenizer)
    _configure_model_geometry(model, image_processor, args, checkpoint)
    _configure_generation(model, tokenizer, args)
    _apply_repro_generate_config(model, checkpoint)

    device = torch.device(str(getattr(args, "device", "cpu") or "cpu"))
    model.to(device)
    train_mod._prepare_model_for_generate_runtime(model)
    model.eval()

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_report.parent.mkdir(parents=True, exist_ok=True)
    if output_all_jsonl is not None:
        output_all_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if output_table_csv is not None:
        output_table_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_dataset_dir is not None:
        dataset_train_manifest = output_dataset_dir / str(getattr(args, "dataset_train_manifest", "train_manifest.jsonl") or "train_manifest.jsonl")
        dataset_val_manifest = output_dataset_dir / str(getattr(args, "dataset_val_manifest", "val_manifest.jsonl") or "val_manifest.jsonl")
        dataset_train_manifest.parent.mkdir(parents=True, exist_ok=True)
        dataset_val_manifest.parent.mkdir(parents=True, exist_ok=True)

    pad_id = int(tokenizer.pad_token_id)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(getattr(args, "batch_size", 16) or 16)),
        shuffle=False,
        num_workers=max(0, int(getattr(args, "num_workers", 0) or 0)),
        collate_fn=lambda features: _collate(features, pad_token_id=pad_id),
        pin_memory=str(device).startswith("cuda"),
    )

    total_seen = 0
    valid_n = 0
    error_n = 0
    empty_pred_n = 0
    empty_ref_n = 0
    edit_sum = 0
    ref_len_sum = 0
    cer_values: List[float] = []
    error_cer_values: List[float] = []
    pred_len_values: List[float] = []
    ref_len_values: List[float] = []
    pred_ref_len_ratios: List[float] = []
    batch_count = 0
    limit_errors = max(0, int(getattr(args, "limit_errors", 0) or 0))
    newline_token = str(getattr(args, "metric_newline_token", "<NL>") or "<NL>")
    logged_generate_kwargs = False
    log_stage_timings = bool(getattr(args, "log_stage_timings", False))
    stage_timing_every_n = max(1, int(getattr(args, "stage_timing_every_n", 1) or 1))
    stage_timing_warmup_batches = max(0, int(getattr(args, "stage_timing_warmup_batches", 0) or 0))
    stage_timing_totals: Dict[str, float] = {}
    stage_timing_max: Dict[str, float] = {}
    stage_timing_measured_batches = 0
    if device.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass

    eval_start_time = time.perf_counter()
    stage_prev_batch_done_t = eval_start_time
    with output_jsonl.open("w", encoding="utf-8") as error_fp:
        if output_dataset_dir is not None and dataset_train_manifest is not None:
            dataset_train_fp = dataset_train_manifest.open("w", encoding="utf-8")
            dataset_val_fp = dataset_val_manifest.open("w", encoding="utf-8") if dataset_val_manifest is not None and float(getattr(args, "val_fraction", 0.0) or 0.0) > 0.0 else None
        all_fp_ctx = output_all_jsonl.open("w", encoding="utf-8") if output_all_jsonl is not None else None
        if output_table_csv is not None:
            table_fp = output_table_csv.open("w", encoding="utf-8", newline="")
            table_writer = csv.writer(table_fp)
            table_writer.writerow(["filename", "cer"])
        try:
            with torch.no_grad():
                for batch in tqdm(loader, desc="extract-cer-errors"):
                    batch_ready_t = time.perf_counter()
                    load_preprocess_wait_s = batch_ready_t - stage_prev_batch_done_t
                    batch_stage_start_t = batch_ready_t
                    batch_count += 1
                    to_device_start_t = time.perf_counter()
                    pixel_values = batch["pixel_values"].to(device)
                    labels = batch["labels"].detach().cpu().numpy()
                    if log_stage_timings and device.type == "cuda":
                        try:
                            torch.cuda.synchronize(device)
                        except Exception:
                            pass
                    to_device_s = time.perf_counter() - to_device_start_t
                    gen_kwargs = _build_generate_kwargs(model, pixel_values, args)
                    if not logged_generate_kwargs:
                        loggable = {k: v for k, v in gen_kwargs.items() if k != "pixel_values"}
                        LOGGER.info("Effective generate kwargs: %s", json.dumps(loggable, ensure_ascii=False, default=str))
                        logged_generate_kwargs = True
                    if log_stage_timings and device.type == "cuda":
                        try:
                            torch.cuda.synchronize(device)
                        except Exception:
                            pass
                    generate_start_t = time.perf_counter()
                    gen_ids = model.generate(**gen_kwargs)
                    if log_stage_timings and device.type == "cuda":
                        try:
                            torch.cuda.synchronize(device)
                        except Exception:
                            pass
                    generate_s = time.perf_counter() - generate_start_t
                    decode_cer_start_t = time.perf_counter()
                    write_copy_s = 0.0
                    pred_norms = train_mod._decode_for_metric(
                        tokenizer,
                        gen_ids.detach().cpu().numpy(),
                        newline_token=newline_token,
                    )
                    ref_norms = train_mod._decode_for_metric(
                        tokenizer,
                        labels,
                        newline_token=newline_token,
                    )
                    for pred, ref, meta in zip(pred_norms, ref_norms, batch["meta"]):
                        total_seen += 1
                        pred = str(pred or "")
                        ref = str(ref or "")
                        if not ref:
                            empty_ref_n += 1
                            sample_cer: Optional[float] = None
                            edit_distance = None
                            ref_len = 0
                        else:
                            valid_n += 1
                            edit_distance = int(train_mod._levenshtein(pred, ref))
                            ref_len = max(1, len(ref))
                            edit_sum += int(edit_distance)
                            ref_len_sum += int(ref_len)
                            sample_cer = float(edit_distance / ref_len)
                            cer_values.append(float(sample_cer))
                            ref_len_values.append(float(len(ref)))
                        if not pred:
                            empty_pred_n += 1
                        pred_len_values.append(float(len(pred)))
                        if ref:
                            pred_ref_len_ratios.append(float(len(pred) / max(1, len(ref))))
                        row_index = int(meta.get("row_index", -1))
                        image_path = str(meta.get("image", ""))
                        record: Dict[str, object] = {
                            "row_index": row_index,
                            "dataset_index": int(meta.get("dataset_index", -1)),
                            "cer": sample_cer,
                            "edit_distance": edit_distance,
                            "ref_len": int(ref_len),
                            "pred_len": int(len(pred)),
                            "pred": pred,
                            "ref": ref,
                            "text_raw": str(meta.get("text_raw", "")),
                            "image": image_path,
                            "checkpoint": str(checkpoint),
                            "manifest": str(manifest),
                        }
                        row = meta.get("row")
                        if bool(getattr(args, "include_row", True)) and isinstance(row, dict):
                            record["row"] = row
                            for key in ("source_dataset", "source_dataset_short", "source_split", "doc_id", "page_id", "line_id", "line_path"):
                                if key in row:
                                    record[key] = row.get(key)
                        if sample_cer is not None and float(sample_cer) > threshold:
                            error_n += 1
                            error_cer_values.append(float(sample_cer))
                            if copy_images_dir is not None and image_path:
                                write_start_t = time.perf_counter()
                                copied = _copy_error_image(Path(image_path), copy_images_dir, row_index=row_index, cer=float(sample_cer))
                                write_copy_s += time.perf_counter() - write_start_t
                                if copied:
                                    record["copied_image"] = copied
                            if output_dataset_dir is not None and dataset_train_fp is not None and image_path:
                                err_split = _dataset_split_for_row(
                                    row_index,
                                    val_fraction=float(getattr(args, "val_fraction", 0.0) or 0.0),
                                    seed=int(getattr(args, "seed", 42) or 42),
                                )
                                target_fp = dataset_val_fp if err_split == "val" and dataset_val_fp is not None else dataset_train_fp
                                source_row = row if isinstance(row, dict) else None
                                write_start_t = time.perf_counter()
                                image_rel = _write_dataset_manifest_row(
                                    target_fp,
                                    dataset_dir=output_dataset_dir,
                                    source_image=Path(image_path),
                                    source_row=source_row,
                                    record=record,
                                    split_name=err_split,
                                    image_mode=str(getattr(args, "dataset_image_mode", "copy") or "copy"),
                                )
                                write_copy_s += time.perf_counter() - write_start_t
                                record["dataset_image"] = image_rel
                                record["dataset_image_mode"] = str(getattr(args, "dataset_image_mode", "copy") or "copy")
                                record["dataset_split"] = err_split
                                dataset_counts[err_split] = int(dataset_counts.get(err_split, 0)) + 1
                            if table_writer is not None:
                                write_start_t = time.perf_counter()
                                table_filename = str(record.get("dataset_image") or record.get("copied_image") or Path(image_path).name)
                                table_writer.writerow([table_filename, f"{float(sample_cer):.8f}"])
                                write_copy_s += time.perf_counter() - write_start_t
                            write_start_t = time.perf_counter()
                            error_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                            write_copy_s += time.perf_counter() - write_start_t
                        if all_fp_ctx is not None:
                            write_start_t = time.perf_counter()
                            all_fp_ctx.write(json.dumps(record, ensure_ascii=False) + "\n")
                            write_copy_s += time.perf_counter() - write_start_t
                        if limit_errors > 0 and error_n >= limit_errors:
                            break
                    decode_cer_s = max(0.0, time.perf_counter() - decode_cer_start_t - write_copy_s)
                    batch_total_s = time.perf_counter() - batch_stage_start_t
                    if log_stage_timings and batch_count > stage_timing_warmup_batches:
                        stage_timing_measured_batches += 1
                        for stage_name, stage_seconds in (
                            ("load_preprocess_wait", load_preprocess_wait_s),
                            ("to_device", to_device_s),
                            ("generate", generate_s),
                            ("decode_cer", decode_cer_s),
                            ("write_copy", write_copy_s),
                            ("batch_total", batch_total_s),
                        ):
                            _add_stage_timing(stage_timing_totals, stage_timing_max, stage_name, stage_seconds)
                        if stage_timing_measured_batches % stage_timing_every_n == 0:
                            LOGGER.info(
                                "stage_timing batch=%d measured=%d samples=%d load_preprocess_wait=%.4fs to_device=%.4fs generate=%.4fs decode_cer=%.4fs write_copy=%.4fs total=%.4fs",
                                int(batch_count),
                                int(stage_timing_measured_batches),
                                int(len(batch["meta"])),
                                float(load_preprocess_wait_s),
                                float(to_device_s),
                                float(generate_s),
                                float(decode_cer_s),
                                float(write_copy_s),
                                float(batch_total_s),
                            )
                    stage_prev_batch_done_t = time.perf_counter()
                    if limit_errors > 0 and error_n >= limit_errors:
                        break
        finally:
            if all_fp_ctx is not None:
                all_fp_ctx.close()
            if table_fp is not None:
                table_fp.close()
            if dataset_train_fp is not None:
                dataset_train_fp.close()
            if dataset_val_fp is not None:
                dataset_val_fp.close()

    if device.type == "cuda":
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
    eval_elapsed_seconds = max(0.0, time.perf_counter() - eval_start_time)
    total_elapsed_seconds = max(0.0, time.perf_counter() - run_start_time)
    cuda_peak_memory_gb: Optional[float] = None
    if device.type == "cuda":
        try:
            cuda_peak_memory_gb = float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 3))
        except Exception:
            cuda_peak_memory_gb = None

    global_cer = float(edit_sum / max(1, ref_len_sum))
    sample_cer_stats = _safe_float_stats(cer_values)
    sample_cer_percentiles = _safe_percentile_stats(cer_values)
    error_cer_stats = _safe_float_stats(error_cer_values)
    pred_len_stats = _safe_float_stats(pred_len_values)
    ref_len_stats = _safe_float_stats(ref_len_values)
    pred_ref_len_ratio_stats = _safe_float_stats(pred_ref_len_ratios)
    pred_len_percentiles = _safe_percentile_stats(pred_len_values)
    ref_len_percentiles = _safe_percentile_stats(ref_len_values)
    cer_buckets = _cer_bucket_counts(cer_values)
    quality_metrics: Dict[str, object] = {
        "cer_threshold": float(threshold),
        "global_cer": float(global_cer),
        "global_cer_percent": float(global_cer * 100.0),
        "manifest_rows_selected": int(len(rows)),
        "usable_samples": int(len(dataset)),
        "valid_n": int(valid_n),
        "total_seen": int(total_seen),
        "error_n": int(error_n),
        "error_ratio_valid": float(error_n / max(1, valid_n)),
        "empty_ref_count": int(empty_ref_n),
        "empty_pred_count": int(empty_pred_n),
        "empty_pred_ratio": float(empty_pred_n / max(1, total_seen)),
        "sample_cer_min": sample_cer_stats.get("min"),
        "sample_cer_max": sample_cer_stats.get("max"),
        "sample_cer_mean": sample_cer_stats.get("mean"),
        "sample_cer_std": sample_cer_stats.get("std"),
        "sample_cer_p50": sample_cer_percentiles.get("p50"),
        "sample_cer_p75": sample_cer_percentiles.get("p75"),
        "sample_cer_p90": sample_cer_percentiles.get("p90"),
        "sample_cer_p95": sample_cer_percentiles.get("p95"),
        "sample_cer_p99": sample_cer_percentiles.get("p99"),
        "error_cer_stats": error_cer_stats,
    }
    length_metrics: Dict[str, object] = {
        "pred_len_stats": pred_len_stats,
        "ref_len_stats": ref_len_stats,
        "pred_ref_len_ratio_stats": pred_ref_len_ratio_stats,
        "pred_len_mean": pred_len_stats.get("mean"),
        "pred_len_p95": pred_len_percentiles.get("p95"),
        "ref_len_mean": ref_len_stats.get("mean"),
        "ref_len_p95": ref_len_percentiles.get("p95"),
        "pred_ref_len_ratio_mean": pred_ref_len_ratio_stats.get("mean"),
    }
    effective_generation_max_length = int(
        getattr(args, "generation_max_length", 0)
        or getattr(getattr(model, "generation_config", None), "max_length", 0)
        or 160
    )
    stage_timing_metrics = _stage_timing_summary(
        stage_timing_totals,
        stage_timing_max,
        stage_timing_measured_batches,
        enabled=log_stage_timings,
        every_n=stage_timing_every_n,
        warmup_batches=stage_timing_warmup_batches,
    )
    performance_metrics: Dict[str, object] = {
        "total_wall_time_seconds": float(total_elapsed_seconds),
        "eval_time_seconds": float(eval_elapsed_seconds),
        "samples_per_second": float(total_seen / max(1e-9, eval_elapsed_seconds)),
        "valid_samples_per_second": float(valid_n / max(1e-9, eval_elapsed_seconds)),
        "batches_per_second": float(batch_count / max(1e-9, eval_elapsed_seconds)),
        "batches": int(batch_count),
        "batch_size": int(max(1, int(getattr(args, "batch_size", 16) or 16))),
        "num_workers": int(max(0, int(getattr(args, "num_workers", 0) or 0))),
        "generation_max_length": int(effective_generation_max_length),
        "cuda_peak_memory_gb": cuda_peak_memory_gb,
        "stage_timing": stage_timing_metrics,
    }
    outputs: Dict[str, object] = {
        "errors_jsonl": str(output_jsonl),
        "all_jsonl": str(output_all_jsonl) if output_all_jsonl is not None else "",
        "cer_table_csv": str(output_table_csv) if output_table_csv is not None else "",
        "dataset_dir": str(output_dataset_dir) if output_dataset_dir is not None else "",
        "train_manifest": str(dataset_train_manifest) if dataset_train_manifest is not None else "",
        "val_manifest": str(dataset_val_manifest) if dataset_val_manifest is not None and dataset_val_fp is not None else "",
        "summary_json": str(summary_json),
        "summary_report": str(summary_report),
    }
    summary: Dict[str, object] = {
        "checkpoint": str(checkpoint),
        "manifest": str(manifest),
        "output_jsonl": str(output_jsonl),
        "summary_json": str(summary_json),
        "summary_report": str(summary_report),
        "output_dataset_dir": str(output_dataset_dir) if output_dataset_dir is not None else "",
        "dataset_train_manifest": str(dataset_train_manifest) if dataset_train_manifest is not None else "",
        "dataset_val_manifest": str(dataset_val_manifest) if dataset_val_manifest is not None and dataset_val_fp is not None else "",
        "cer_table_csv": str(output_table_csv) if output_table_csv is not None else "",
        "dataset_counts": dataset_counts,
        "dataset_image_mode": str(getattr(args, "dataset_image_mode", "copy") or "copy"),
        "val_fraction": float(getattr(args, "val_fraction", 0.0) or 0.0),
        "output_all_jsonl": str(output_all_jsonl) if output_all_jsonl is not None else "",
        "copy_images_dir": str(copy_images_dir) if copy_images_dir is not None else "",
        "source_dataset_filter": {
            "include": sorted(include_source_datasets),
            "exclude": sorted(exclude_source_datasets),
            "include_google_books": bool(getattr(args, "include_google_books", False)),
        },
        "cer_threshold": float(threshold),
        "global_cer": float(global_cer),
        "global_cer_percent": float(global_cer * 100.0),
        "total_seen": int(total_seen),
        "valid_n": int(valid_n),
        "error_n": int(error_n),
        "error_ratio_valid": float(error_n / max(1, valid_n)),
        "empty_ref_count": int(empty_ref_n),
        "empty_pred_count": int(empty_pred_n),
        "sample_cer_stats": sample_cer_stats,
        "sample_cer_percentiles": sample_cer_percentiles,
        "error_cer_stats": error_cer_stats,
        "length_metrics": length_metrics,
        "cer_distribution": {
            "buckets": cer_buckets,
            "percentiles": sample_cer_percentiles,
        },
        "quality_metrics": quality_metrics,
        "performance_metrics": performance_metrics,
        "outputs": outputs,
        "dataset_scan_stats": dataset.stats,
        "device": str(device),
        "image_preprocess_pipeline": image_preprocess_pipeline,
        "image_processor_path": image_processor_path,
        "image_processor_size": getattr(image_processor, "size", None),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_report(summary, summary_report)
    LOGGER.info(
        "Extraction done: global_cer=%.4f (%.2f%%) valid=%d errors=%d threshold=%.4f samples_per_second=%.3f output=%s",
        float(global_cer),
        float(global_cer * 100.0),
        int(valid_n),
        int(error_n),
        float(threshold),
        float(performance_metrics["samples_per_second"]),
        output_jsonl,
    )
    return summary


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
