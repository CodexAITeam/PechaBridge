"""CLI entrypoint for weak OCR labeling."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml

from pechabridge.ocr.weak_labeler import WeakOCRConfig, run_weak_ocr_labeler

LOGGER = logging.getLogger("weak_ocr_label")


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate weak OCR labels for Pecha patch datasets.",
        add_help=add_help,
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset root directory.")
    parser.add_argument("--meta", type=str, required=True, help="Path to patches.parquet metadata.")
    parser.add_argument("--out", type=str, required=True, help="Output parquet path for weak OCR labels.")
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["tesseract", "vlm"],
        help="OCR backend override (default: value from YAML config).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/weak_ocr.yaml",
        help="YAML config path (default: configs/weak_ocr.yaml).",
    )
    parser.add_argument("--num_workers", type=int, default=None, help="Override worker count.")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard id in [0, num_shards).")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards.")
    parser.add_argument("--resume", action="store_true", help="Skip patch_ids already present in output parquet.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute patch_ids in this shard even if present.")
    parser.add_argument(
        "--debug_dump",
        "--debug-dump",
        dest="debug_dump",
        type=int,
        default=0,
        help="Dump N OCR examples to <dataset>/debug/weak_ocr.",
    )
    return parser


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a dict: {path}")
    return dict(payload)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    dataset = Path(args.dataset).expanduser().resolve()
    meta = Path(args.meta).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    cfg_path = Path(args.config).expanduser().resolve()

    if not dataset.exists() or not dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset}")
    if not meta.exists() or not meta.is_file():
        raise FileNotFoundError(f"Metadata parquet not found: {meta}")

    cfg_payload = _load_yaml(cfg_path)
    cfg = WeakOCRConfig.from_dict(
        cfg_payload,
        backend_override=args.backend,
        num_workers_override=args.num_workers,
    )

    LOGGER.info(
        "Weak OCR start: dataset=%s meta=%s out=%s backend=%s shard=%d/%d",
        dataset,
        meta,
        out,
        cfg.backend_name,
        int(args.shard_id),
        int(args.num_shards),
    )
    summary = run_weak_ocr_labeler(
        dataset_dir=dataset,
        meta_path=meta,
        out_path=out,
        config=cfg,
        shard_id=int(args.shard_id),
        num_shards=int(args.num_shards),
        resume=bool(args.resume),
        overwrite=bool(args.overwrite),
        debug_dump=int(args.debug_dump),
    )
    LOGGER.info("Weak OCR done: %s", json.dumps(summary, ensure_ascii=False))
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    parser = create_parser(add_help=True)
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
