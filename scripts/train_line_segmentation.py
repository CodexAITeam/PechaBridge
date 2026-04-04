#!/usr/bin/env python3
"""Train a YOLO line segmentation model on an Ultralytics segment dataset."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

LOGGER = logging.getLogger("train_line_segmentation")


def _configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _resolve_dataset_yaml(dataset_arg: str) -> Path:
    raw = Path(str(dataset_arg)).expanduser()
    candidates = []
    if raw.suffix.lower() in {".yml", ".yaml"}:
        candidates.append(raw)
    candidates.append(raw / "data.yaml")
    candidates.append(raw / "data.yml")

    root = Path(__file__).resolve().parent.parent
    candidates.append(root / raw)
    candidates.append(root / raw / "data.yaml")
    candidates.append(root / raw / "data.yml")
    candidates.append(root / "datasets" / raw)
    candidates.append(root / "datasets" / raw / "data.yaml")
    candidates.append(root / "datasets" / raw / "data.yml")

    for cand in candidates:
        try:
            resolved = cand.resolve()
        except Exception:
            continue
        if resolved.is_file() and resolved.suffix.lower() in {".yml", ".yaml"}:
            return resolved
    raise FileNotFoundError(
        f"Could not find dataset YAML for --dataset={dataset_arg}. "
        "Expected a dataset folder with data.yaml or a direct path to a YAML file."
    )


def _normalize_dataset_yaml_for_ultralytics(yaml_path: Path) -> Path:
    with yaml_path.open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {yaml_path}")

    raw_root = cfg.get("path", "")
    if raw_root:
        root = Path(str(raw_root)).expanduser()
        if not root.is_absolute():
            root = (yaml_path.parent / root).resolve()
    else:
        root = yaml_path.parent.resolve()

    cfg["path"] = str(root)
    for key in ("train", "val", "test"):
        if key not in cfg or cfg[key] in {"", None}:
            continue
        split = Path(str(cfg[key])).expanduser()
        if not split.is_absolute():
            split = (root / split).resolve()
        cfg[key] = str(split)

    fd, tmp_name = tempfile.mkstemp(prefix="pechabridge_line_seg_", suffix=".yaml")
    os.close(fd)
    tmp_path = Path(tmp_name)
    with tmp_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(cfg, fp, sort_keys=False, allow_unicode=True)
    return tmp_path


def _read_dataset_summary(dataset_yaml: Path) -> Dict[str, Any]:
    with dataset_yaml.open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp) or {}
    if not isinstance(cfg, dict):
        return {}

    raw_root = cfg.get("path", "")
    if raw_root:
        root = Path(str(raw_root)).expanduser()
        if not root.is_absolute():
            root = (dataset_yaml.parent / root).resolve()
    else:
        root = dataset_yaml.parent.resolve()

    summary_path = root / "meta" / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a YOLO segmentation model for Tibetan line segmentation.",
        add_help=add_help,
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset YAML or dataset folder containing data.yaml.")
    parser.add_argument("--model", type=str, default="yolo11n-seg.pt", help="Segmentation checkpoint to start from.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--workers", type=int, default=8, help="Data loader workers.")
    parser.add_argument("--device", type=str, default="", help="Training device, e.g. cpu, cuda:0.")
    parser.add_argument("--project", type=str, default="runs/segment", help="Ultralytics project directory.")
    parser.add_argument("--name", type=str, default="tibetan_line_segmentation", help="Run name.")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=42, help="Training seed.")
    parser.add_argument("--cache", type=str, default="", help="Ultralytics cache mode, e.g. ram or disk.")
    parser.add_argument("--resume", action="store_true", help="Resume the latest interrupted run for this project/name.")
    parser.add_argument("--export", action="store_true", help="Export the best model as TorchScript after training.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    _configure_logging(bool(getattr(args, "verbose", False)))

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "Ultralytics is required for line segmentation training. "
            "Install it with `pip install ultralytics`."
        ) from exc

    dataset_yaml = _resolve_dataset_yaml(args.dataset)
    normalized_yaml = _normalize_dataset_yaml_for_ultralytics(dataset_yaml)
    dataset_summary = _read_dataset_summary(dataset_yaml)
    LOGGER.info("Using dataset YAML: %s", dataset_yaml)
    if normalized_yaml != dataset_yaml:
        LOGGER.info("Using normalized dataset YAML with absolute paths: %s", normalized_yaml)
    preprocess_pipeline = str(dataset_summary.get("image_preprocess_pipeline", "") or "").strip()
    if preprocess_pipeline:
        LOGGER.info("Dataset image preprocessing pipeline: %s", preprocess_pipeline)

    model_name = str(args.model or "").strip()
    if model_name and "-seg" not in model_name.lower() and not Path(model_name).expanduser().exists():
        LOGGER.warning(
            "Model %r does not look like a segmentation checkpoint. "
            "Recommended starters are `yolo11n-seg.pt`, `yolo11s-seg.pt`, etc.",
            model_name,
        )

    train_kwargs: Dict[str, Any] = {
        "data": str(normalized_yaml),
        "epochs": int(args.epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "workers": int(args.workers),
        "project": str(args.project),
        "name": str(args.name),
        "patience": int(args.patience),
        "seed": int(args.seed),
        "plots": True,
        "save_period": 10,
        "resume": bool(args.resume),
    }
    if str(args.device or "").strip():
        train_kwargs["device"] = str(args.device).strip()
    if str(args.cache or "").strip():
        train_kwargs["cache"] = str(args.cache).strip()

    model = YOLO(model_name)
    LOGGER.info(
        "Starting line segmentation training: model=%s epochs=%d imgsz=%d batch=%d",
        model_name,
        int(args.epochs),
        int(args.imgsz),
        int(args.batch),
    )
    results = model.train(**train_kwargs)

    run_dir = Path(args.project).expanduser().resolve() / str(args.name)
    best_model = run_dir / "weights" / "best.pt"
    export_path: Optional[str] = None
    if bool(args.export) and best_model.exists():
        LOGGER.info("Exporting best checkpoint to TorchScript: %s", best_model)
        export_path = str(YOLO(str(best_model)).export(format="torchscript"))

    summary = {
        "dataset_yaml": str(dataset_yaml),
        "normalized_dataset_yaml": str(normalized_yaml),
        "project": str(Path(args.project).expanduser().resolve()),
        "run_dir": str(run_dir),
        "best_model": str(best_model),
        "export_path": export_path or "",
        "results_dir": str(getattr(results, "save_dir", run_dir)),
        "model": model_name,
        "image_preprocess_pipeline": preprocess_pipeline,
        "epochs": int(args.epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
    }
    summary_path = run_dir / "line_segmentation_training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Training finished. Best model: %s", best_model)
    LOGGER.info("Saved summary to %s", summary_path)
    return summary


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
