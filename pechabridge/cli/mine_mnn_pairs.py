"""CLI entrypoint for robust MNN positive-pair mining."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml

from pechabridge.mining.mnn_miner import MNNConfig, MNNMiner

LOGGER = logging.getLogger("mine_mnn_pairs")


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mine robust cross-page MNN patch positive pairs.",
        add_help=add_help,
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset root directory.")
    parser.add_argument("--meta", type=str, required=True, help="Path to patches.parquet metadata.")
    parser.add_argument("--out", type=str, required=True, help="Output parquet path for mined pairs.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mnn_mining.yaml",
        help="YAML config path (default: configs/mnn_mining.yaml).",
    )
    parser.add_argument(
        "--debug_dump",
        "--debug-dump",
        dest="debug_dump",
        type=int,
        default=0,
        help="Dump N random pair grids under dataset/debug/mnn_pairs.",
    )
    parser.add_argument(
        "--num_workers",
        "--num-workers",
        "--num-worker",
        dest="num_workers",
        type=int,
        default=None,
        help="Override config.performance.num_workers (used for source-loop mining threads, embedding DataLoader, and FAISS/OpenMP threads).",
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
    out.parent.mkdir(parents=True, exist_ok=True)

    cfg_payload = _load_yaml(cfg_path)
    cfg = MNNConfig.from_dict(cfg_payload)
    if getattr(args, "num_workers", None) is not None:
        cfg.performance.num_workers = int(args.num_workers)

    LOGGER.info(
        "MNN mining start: dataset=%s meta=%s out=%s model=%s num_workers=%s",
        dataset,
        meta,
        out,
        cfg.model.backbone,
        str(cfg.performance.num_workers),
    )
    miner = MNNMiner(cfg)
    summary = miner.run(
        dataset_dir=dataset,
        meta_path=meta,
        out_pairs_path=out,
        debug_dump=int(args.debug_dump),
    )
    LOGGER.info("MNN mining done: %s", json.dumps(summary, ensure_ascii=False))
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    parser = create_parser(add_help=True)
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
