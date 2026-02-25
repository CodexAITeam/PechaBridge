#!/usr/bin/env python3
"""Warm persistent line_clip corpus caches used by the line_clip Workbench tab."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from ui_workbench import (
    _get_or_build_line_clip_debug_corpus_ui,
    scan_line_clip_dual_models_ui,
)

LOGGER = logging.getLogger("warm_line_clip_workbench_cache")

DEFAULT_MODELS_DIR = "/home/ubuntu/data/PechaBridge/models"
DEFAULT_DATASET_ROOT = "/home/ubuntu/data/PechaBridge/datasets/openpecha_ocr_lines"
_SPLIT_ORDER = ("train", "val", "eval", "test")


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="warm-line-clip-workbench-cache",
        description=(
            "Build/persist line_clip Workbench corpus embeddings (image/text) for all available OCR splits "
            "using the best detected line_clip model bundle (sorted by validation retrieval performance)."
        ),
        add_help=add_help,
    )
    p.add_argument("--models-dir", type=str, default=DEFAULT_MODELS_DIR, help="Directory containing line_clip model bundles/checkpoints")
    p.add_argument(
        "--dataset-root",
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help="OCR dataset root containing split folders with meta/lines.jsonl",
    )
    p.add_argument(
        "--splits",
        type=str,
        default="",
        help="Comma-separated splits to warm (default: auto-detect from dataset root, order train,val,eval,test)",
    )
    p.add_argument("--device", type=str, default="auto", help="Encoding device (auto/cuda/cuda:0/cpu/mps)")
    p.add_argument("--batch-size", type=int, default=32, help="Encoding batch size for image/text embedding generation")
    p.add_argument(
        "--image-batch-size",
        type=int,
        default=0,
        help="Image embedding batch size override (0 uses --batch-size)",
    )
    p.add_argument(
        "--text-batch-size",
        type=int,
        default=0,
        help="Text embedding batch size override (0 uses --batch-size)",
    )
    p.add_argument(
        "--image-preproc-workers",
        type=int,
        default=0,
        help="Optional CPU worker threads for image decode/preprocess per batch (0/1 disables)",
    )
    p.add_argument("--text-max-length", type=int, default=256, help="Tokenizer max length for text embeddings")
    p.add_argument("--no-l2-normalize", action="store_true", help="Disable L2 normalization before storing embeddings")
    p.add_argument(
        "--only",
        type=str,
        default="both",
        choices=["both", "image", "text"],
        help="Warm only image bank, only text bank, or both (default: both)",
    )
    p.add_argument(
        "--progress-every-batches",
        type=int,
        default=50,
        help="Log progress during embedding generation every N batches (0 disables progress logs)",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Exit immediately if one split fails (default: continue and report all results)",
    )
    p.set_defaults(tqdm=None)
    tqdm_group = p.add_mutually_exclusive_group()
    tqdm_group.add_argument("--tqdm", dest="tqdm", action="store_true", help="Enable tqdm progress bars for CLI runs")
    tqdm_group.add_argument("--no-tqdm", dest="tqdm", action="store_false", help="Disable tqdm progress bars")
    return p


def _parse_requested_splits(raw: str) -> List[str]:
    out: List[str] = []
    for tok in str(raw or "").split(","):
        s = tok.strip().lower()
        if not s:
            continue
        if s not in out:
            out.append(s)
    return out


def _detect_splits(dataset_root: Path) -> List[str]:
    splits: List[str] = []
    for split in _SPLIT_ORDER:
        manifest = dataset_root / split / "meta" / "lines.jsonl"
        if manifest.exists() and manifest.is_file():
            splits.append(split)
    return splits


def _pick_best_line_clip_bundle(models_dir: str) -> Dict[str, Any]:
    _, image_backbone, image_head, text_encoder, text_head, bundle_state_json, msg = scan_line_clip_dual_models_ui(models_dir)
    LOGGER.info(msg)
    bundles: Dict[str, Dict[str, Any]] = {}
    try:
        parsed = json.loads(bundle_state_json or "{}")
        if isinstance(parsed, dict):
            bundles = parsed
    except Exception:
        bundles = {}
    best_root = ""
    best_rec: Dict[str, Any] = {}
    for root, rec in bundles.items():
        if not isinstance(rec, dict):
            continue
        if str(rec.get("image_backbone", "")) == str(image_backbone) and str(rec.get("text_encoder", "")) == str(text_encoder):
            best_root = str(root)
            best_rec = rec
            break
    if not all([image_backbone, image_head, text_encoder, text_head]):
        raise RuntimeError(f"No valid line_clip bundle found in {models_dir}")
    return {
        "bundle_root": best_root or str(Path(image_backbone).resolve().parent),
        "image_backbone": str(image_backbone),
        "image_head": str(image_head),
        "text_encoder": str(text_encoder),
        "text_head": str(text_head),
        "bundle_label": str(best_rec.get("display_label", best_root or "")),
        "config_path": str(best_rec.get("config_path", "")),
    }


def run(args: argparse.Namespace) -> int:
    models_dir = str(args.models_dir or "").strip()
    dataset_root = Path(str(args.dataset_root or "").strip()).expanduser().resolve()
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    requested_splits = _parse_requested_splits(getattr(args, "splits", ""))
    available_splits = _detect_splits(dataset_root)
    if requested_splits:
        splits = [s for s in requested_splits if s in available_splits]
        missing = [s for s in requested_splits if s not in available_splits]
        if missing:
            LOGGER.warning("Requested splits not found (skipping): %s", ", ".join(missing))
    else:
        splits = available_splits
    if not splits:
        raise RuntimeError(f"No OCR manifests found under {dataset_root} (looked for */meta/lines.jsonl)")

    bundle = _pick_best_line_clip_bundle(models_dir)
    LOGGER.info("Using best line_clip bundle: %s", bundle.get("bundle_root"))
    if bundle.get("bundle_label"):
        LOGGER.info("Bundle label: %s", bundle.get("bundle_label"))

    t0_all = time.time()
    results: List[Dict[str, Any]] = []
    l2_normalize = not bool(getattr(args, "no_l2_normalize", False))
    batch_size = max(1, int(getattr(args, "batch_size", 32)))
    image_batch_size = max(0, int(getattr(args, "image_batch_size", 0) or 0))
    text_batch_size = max(0, int(getattr(args, "text_batch_size", 0) or 0))
    image_preproc_workers = max(0, int(getattr(args, "image_preproc_workers", 0) or 0))
    text_max_length = max(8, int(getattr(args, "text_max_length", 256)))
    only = str(getattr(args, "only", "both") or "both").strip().lower()
    if only not in {"both", "image", "text"}:
        only = "both"
    progress_every_batches = max(0, int(getattr(args, "progress_every_batches", 0) or 0))
    tqdm_arg = getattr(args, "tqdm", None)
    use_tqdm = bool(tqdm_arg) if tqdm_arg is not None else bool(tqdm is not None and sys.stderr.isatty())
    split_iter = splits
    split_pbar = None
    if use_tqdm and tqdm is not None:
        split_pbar = tqdm(splits, desc="warm-line-clip-cache", unit="split")
        split_iter = split_pbar

    for split in split_iter:
        t0 = time.time()
        rec: Dict[str, Any] = {"split": split}
        try:
            corpus = _get_or_build_line_clip_debug_corpus_ui(
                dataset_root=str(dataset_root),
                split_name=str(split),
                image_backbone_path=str(bundle["image_backbone"]),
                image_head_path=str(bundle["image_head"]),
                text_encoder_dir=str(bundle["text_encoder"]),
                text_head_path=str(bundle["text_head"]),
                device_pref=str(getattr(args, "device", "auto") or "auto"),
                batch_size=batch_size,
                text_max_length=text_max_length,
                l2_normalize=l2_normalize,
                required_banks=only,
                progress_every_batches=progress_every_batches,
                image_batch_size=(image_batch_size or None),
                text_batch_size=(text_batch_size or None),
                image_preproc_workers=image_preproc_workers,
                tqdm_progress=use_tqdm,
            )
            rec.update(
                {
                    "ok": True,
                    "count": int(corpus.get("count", 0)),
                    "cache_source": str(corpus.get("cache_source", "")),
                    "requested_banks": str(corpus.get("requested_banks", only)),
                    "available_banks": list(corpus.get("available_banks") or []),
                    "built_banks": list(corpus.get("built_banks") or []),
                    "disk_cache_dir": str(corpus.get("disk_cache_dir", "")),
                    "device": str(corpus.get("device", "")),
                    "device_msg": str(corpus.get("device_msg", "")),
                    "timing": dict(corpus.get("timing") or {}),
                    "perf": dict(corpus.get("perf") or {}),
                    "elapsed_s": round(float(time.time() - t0), 3),
                }
            )
        except Exception as exc:
            rec.update(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_s": round(float(time.time() - t0), 3),
                }
            )
            if bool(getattr(args, "fail_fast", False)):
                results.append(rec)
                if split_pbar is not None:
                    split_pbar.close()
                print(json.dumps(
                    {
                        "ok": False,
                        "models_dir": models_dir,
                        "dataset_root": str(dataset_root),
                        "bundle": bundle,
                        "results": results,
                        "total_elapsed_s": round(float(time.time() - t0_all), 3),
                    },
                    ensure_ascii=False,
                    indent=2,
                ))
                return 1
        LOGGER.info(
            "split=%s ok=%s count=%s cache=%s req=%s avail=%s built=%s hit=%s elapsed=%.1fs rows/s=%.1f img_s=%s txt_s=%s save_s=%s",
            rec.get("split"),
            rec.get("ok"),
            rec.get("count", "na"),
            rec.get("cache_source", "na"),
            rec.get("requested_banks", "na"),
            ",".join([str(x) for x in (rec.get("available_banks") or [])]) or "-",
            ",".join([str(x) for x in (rec.get("built_banks") or [])]) or "-",
            str((rec.get("timing") or {}).get("cache_hit_tier", "na")),
            float(rec.get("elapsed_s", 0.0)),
            float((rec.get("perf") or {}).get("rows_per_s_total", 0.0) or 0.0),
            str((rec.get("timing") or {}).get("image_encode_s", "na")),
            str((rec.get("timing") or {}).get("text_encode_s", "na")),
            str((rec.get("timing") or {}).get("disk_save_s", "na")),
        )
        if split_pbar is not None:
            try:
                split_pbar.set_postfix(
                    split=str(split),
                    ok=str(bool(rec.get("ok"))),
                    rows=int(rec.get("count", 0) or 0),
                    rps=float((rec.get("perf") or {}).get("rows_per_s_total", 0.0) or 0.0),
                )
            except Exception:
                pass
        results.append(rec)

    if split_pbar is not None:
        split_pbar.close()

    payload = {
        "ok": all(bool(r.get("ok")) for r in results),
        "models_dir": str(Path(models_dir).expanduser().resolve()),
        "dataset_root": str(dataset_root),
        "requested_splits": requested_splits,
        "available_splits": available_splits,
        "warmed_splits": splits,
        "bundle": bundle,
        "settings": {
            "device": str(getattr(args, "device", "auto") or "auto"),
            "batch_size": int(batch_size),
            "image_batch_size": int(image_batch_size or 0),
            "text_batch_size": int(text_batch_size or 0),
            "image_preproc_workers": int(image_preproc_workers),
            "text_max_length": int(text_max_length),
            "l2_normalize": bool(l2_normalize),
            "only": str(only),
            "progress_every_batches": int(progress_every_batches),
            "tqdm": bool(use_tqdm),
        },
        "results": results,
        "total_elapsed_s": round(float(time.time() - t0_all), 3),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if bool(payload["ok"]) else 1


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = create_parser(add_help=True)
    args = parser.parse_args()
    return int(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
