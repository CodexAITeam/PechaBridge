#!/usr/bin/env python3
"""Train a YOLO line segmentation model on an Ultralytics segment dataset."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import yaml
from PIL import Image

from pechabridge.ocr.line_segmentation import (
    DEFAULT_LINE_SEGMENTATION_PREPROCESS,
    apply_line_segmentation_preprocess,
    normalize_line_segmentation_preprocess_pipeline,
)

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


def _dataset_root_from_cfg(cfg: Dict[str, Any], *, yaml_path: Path) -> Path:
    raw_root = cfg.get("path", "")
    if raw_root:
        root = Path(str(raw_root)).expanduser()
        if not root.is_absolute():
            root = (yaml_path.parent / root).resolve()
        return root.resolve()
    return yaml_path.parent.resolve()


def _derive_labels_dir_from_images_dir(images_dir: Path) -> Path:
    if images_dir.name == "images":
        return (images_dir.parent / "labels").resolve()
    if images_dir.parent.name == "images":
        return (images_dir.parent.parent / "labels" / images_dir.name).resolve()
    return (images_dir.parent / "labels").resolve()


def _resolve_split_dirs(
    cfg: Dict[str, Any],
    *,
    yaml_path: Path,
    split_name: str,
) -> Tuple[Optional[Path], Optional[Path]]:
    raw_root = _dataset_root_from_cfg(cfg, yaml_path=yaml_path)
    split_ref = str(cfg.get(split_name, "") or "").strip()
    if not split_ref:
        return None, None
    image_dir = Path(split_ref).expanduser()
    if not image_dir.is_absolute():
        image_dir = (raw_root / image_dir).resolve()
    else:
        image_dir = image_dir.resolve()
    return image_dir, _derive_labels_dir_from_images_dir(image_dir)


def _iter_image_files(images_dir: Path) -> Sequence[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def _copy_tree_if_exists(src: Optional[Path], dst: Path) -> int:
    if src is None or not src.exists():
        return 0
    count = 0
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        count += 1
    return count


def _build_preprocessed_training_dataset(
    source_yaml: Path,
    *,
    output_dir: Path,
    pipeline: str,
) -> Tuple[Path, Dict[str, Any]]:
    mode = normalize_line_segmentation_preprocess_pipeline(pipeline)
    if mode == "none":
        raise ValueError("_build_preprocessed_training_dataset requires a non-'none' pipeline.")

    with source_yaml.open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {source_yaml}")

    source_root = _dataset_root_from_cfg(cfg, yaml_path=source_yaml)
    output_root = output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    out_cfg: Dict[str, Any] = {
        "path": str(output_root),
        "names": cfg.get("names", {0: "line"}),
    }
    if "nc" in cfg:
        out_cfg["nc"] = cfg["nc"]

    summary: Dict[str, Any] = {
        "source_yaml": str(source_yaml),
        "source_root": str(source_root),
        "output_dir": str(output_root),
        "image_preprocess_pipeline": mode,
        "splits": {},
    }

    for split_name in ("train", "val", "test"):
        images_dir, labels_dir = _resolve_split_dirs(cfg, yaml_path=source_yaml, split_name=split_name)
        if images_dir is None:
            continue
        if not images_dir.exists():
            raise FileNotFoundError(f"Image split path for '{split_name}' does not exist: {images_dir}")

        out_images_dir = output_root / "images" / split_name
        out_labels_dir = output_root / "labels" / split_name
        out_images_dir.mkdir(parents=True, exist_ok=True)
        out_labels_dir.mkdir(parents=True, exist_ok=True)

        image_count = 0
        for image_path in _iter_image_files(images_dir):
            rel = image_path.relative_to(images_dir)
            target = out_images_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(image_path) as im:
                rgb = im.convert("RGB")
                processed = Image.fromarray(apply_line_segmentation_preprocess(rgb, pipeline=mode), mode="RGB")
                processed.save(target)
            image_count += 1

        label_file_count = _copy_tree_if_exists(labels_dir, out_labels_dir)
        out_cfg[split_name] = f"images/{split_name}"
        summary["splits"][split_name] = {
            "source_images_dir": str(images_dir),
            "source_labels_dir": str(labels_dir) if labels_dir is not None else "",
            "image_count": int(image_count),
            "label_file_count": int(label_file_count),
        }

    yaml_path = output_root / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(out_cfg, fp, sort_keys=False, allow_unicode=True)
    return yaml_path, summary


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
    parser.add_argument(
        "--image-preprocess-pipeline",
        type=str,
        default=DEFAULT_LINE_SEGMENTATION_PREPROCESS,
        choices=["none", "bdrc", "gray", "rgb"],
        help="Preprocess dataset images for this training run. Default: gray.",
    )
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
    dataset_summary = _read_dataset_summary(dataset_yaml)
    source_preprocess_pipeline = normalize_line_segmentation_preprocess_pipeline(
        str(
            dataset_summary.get("download_image_preprocess_pipeline")
            or dataset_summary.get("image_preprocess_pipeline")
            or "none"
        )
    )
    requested_preprocess_pipeline = normalize_line_segmentation_preprocess_pipeline(
        str(getattr(args, "image_preprocess_pipeline", DEFAULT_LINE_SEGMENTATION_PREPROCESS))
    )
    LOGGER.info("Using source dataset YAML: %s", dataset_yaml)
    LOGGER.info("Source dataset image preprocessing pipeline: %s", source_preprocess_pipeline)
    LOGGER.info("Requested training image preprocessing pipeline: %s", requested_preprocess_pipeline)

    preprocessing_summary: Dict[str, Any] = {}
    temporary_dataset_dir: Optional[tempfile.TemporaryDirectory[str]] = None
    effective_dataset_yaml = dataset_yaml
    effective_preprocess_pipeline = requested_preprocess_pipeline
    if requested_preprocess_pipeline == "none":
        LOGGER.info("Training directly on the downloaded dataset images without extra preprocessing.")
    elif source_preprocess_pipeline == requested_preprocess_pipeline and source_preprocess_pipeline != "none":
        LOGGER.info(
            "Source dataset is already preprocessed with %s. Reusing it directly to avoid double preprocessing.",
            source_preprocess_pipeline,
        )
        effective_preprocess_pipeline = source_preprocess_pipeline
    else:
        if source_preprocess_pipeline != "none":
            LOGGER.warning(
                "Source dataset already reports preprocessing=%s and training requested=%s. "
                "This will stack preprocessing. Re-download raw data or use --image-preprocess-pipeline none if that is not intended.",
                source_preprocess_pipeline,
                requested_preprocess_pipeline,
            )
        temporary_dataset_dir = tempfile.TemporaryDirectory(prefix="pechabridge_line_seg_train_")
        effective_dataset_yaml, preprocessing_summary = _build_preprocessed_training_dataset(
            dataset_yaml,
            output_dir=Path(temporary_dataset_dir.name),
            pipeline=requested_preprocess_pipeline,
        )
        LOGGER.info(
            "Built temporary preprocessed training dataset: %s",
            preprocessing_summary.get("output_dir", effective_dataset_yaml.parent),
        )

    normalized_yaml = _normalize_dataset_yaml_for_ultralytics(effective_dataset_yaml)
    if normalized_yaml != effective_dataset_yaml:
        LOGGER.info("Using normalized dataset YAML with absolute paths: %s", normalized_yaml)

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

    try:
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
            "source_dataset_yaml": str(dataset_yaml),
            "effective_dataset_yaml": str(effective_dataset_yaml),
            "normalized_dataset_yaml": str(normalized_yaml),
            "temporary_preprocessed_dataset_dir": (
                str(preprocessing_summary.get("output_dir", ""))
                if preprocessing_summary
                else ""
            ),
            "project": str(Path(args.project).expanduser().resolve()),
            "run_dir": str(run_dir),
            "best_model": str(best_model),
            "export_path": export_path or "",
            "results_dir": str(getattr(results, "save_dir", run_dir)),
            "model": model_name,
            "source_image_preprocess_pipeline": source_preprocess_pipeline,
            "requested_training_image_preprocess_pipeline": requested_preprocess_pipeline,
            "effective_training_image_preprocess_pipeline": effective_preprocess_pipeline,
            "preprocessing_summary": preprocessing_summary,
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
    finally:
        if temporary_dataset_dir is not None:
            temporary_dataset_dir.cleanup()


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
