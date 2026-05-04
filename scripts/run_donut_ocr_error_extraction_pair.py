#!/usr/bin/env python3
"""Build or run matching Donut OCR error extraction jobs for train and val."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional


def _env_or_default(name: str, default: str = "") -> str:
    return str(os.environ.get(name, "") or default)


def _slug(value: str) -> str:
    out = []
    for ch in str(value or "").strip():
        if ch.isalnum():
            out.append(ch)
        elif ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("._-")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "unknown"


def _infer_dataset_name(manifest: Path) -> str:
    parts = list(manifest.expanduser().resolve().parts)
    for split_name in ("train", "eval", "val", "validation", "test"):
        if split_name in parts:
            idx = parts.index(split_name)
            if idx > 0:
                return _slug(parts[idx - 1])
    if manifest.name in {"train_manifest.jsonl", "val_manifest.jsonl", "eval_manifest.jsonl"}:
        return _slug(manifest.parent.name)
    return _slug(manifest.stem)


def _threshold_slug(value: float) -> str:
    if float(value) < 0:
        return "all"
    prefix = "gt"
    raw = f"{abs(float(value)):.4f}".rstrip("0").rstrip(".")
    return f"{prefix}{raw.replace('.', 'p')}"


def _quote_cmd(parts: Iterable[str], *, cuda_visible_devices: str) -> str:
    env_prefix = ""
    if str(cuda_visible_devices).strip():
        env_prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(str(cuda_visible_devices).strip())} "
    return env_prefix + " ".join(shlex.quote(str(p)) for p in parts)


def _build_command(
    *,
    split: str,
    manifest: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> List[str]:
    cmd = [
        sys.executable,
        "cli.py",
        "extract-donut-ocr-errors",
        "--checkpoint",
        str(args.checkpoint),
        "--manifest",
        str(manifest),
        "--output_dataset_dir",
        str(output_dir),
        "--cer_threshold",
        str(args.cer_threshold),
        "--tokenizer_path",
        str(args.tokenizer_path),
        "--image_processor_path",
        str(args.image_processor_path),
        "--image_preprocess_pipeline",
        str(args.image_preprocess_pipeline),
        "--target_height",
        str(args.target_height),
        "--target_width",
        str(args.target_width),
        "--max_target_length",
        str(args.max_target_length),
        "--generation_max_length",
        str(args.generation_max_length),
        "--batch_size",
        str(args.batch_size),
        "--device",
        str(args.device),
        "--num_workers",
        str(args.num_workers),
    ]
    if bool(args.enable_fixed_resize):
        cmd.append("--enable_fixed_resize")
    if bool(args.enable_letterboxing):
        cmd.append("--enable_letterboxing")
    if bool(args.include_google_books):
        cmd.append("--include_google_books")
    if str(args.source_datasets).strip():
        cmd.extend(["--source_datasets", str(args.source_datasets)])
    if str(args.exclude_source_datasets).strip():
        cmd.extend(["--exclude_source_datasets", str(args.exclude_source_datasets)])
    if int(args.max_samples) > 0:
        cmd.extend(["--max_samples", str(args.max_samples)])
    if int(args.limit_errors) > 0:
        cmd.extend(["--limit_errors", str(args.limit_errors)])
    if split == "val" and float(args.val_fraction) <= 0:
        # The source manifest is already validation; avoid creating another nested val split.
        cmd.extend(["--val_fraction", "0"])
    elif float(args.val_fraction) > 0:
        cmd.extend(["--val_fraction", str(args.val_fraction)])
    return cmd


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=_env_or_default("CKPT"), help="Checkpoint path. Defaults to $CKPT.")
    parser.add_argument("--run_dir", default=_env_or_default("RUN"), help="Run dir containing tokenizer/image_processor. Defaults to $RUN.")
    parser.add_argument("--tokenizer_path", default="", help="Tokenizer path. Defaults to $RUN/tokenizer.")
    parser.add_argument("--image_processor_path", default="", help="Image processor path. Defaults to $RUN/image_processor.")
    parser.add_argument("--train_manifest", default=_env_or_default("TRAIN_MANIFEST"), help="Train manifest. Defaults to $TRAIN_MANIFEST.")
    parser.add_argument("--val_manifest", default=_env_or_default("VAL_MANIFEST"), help="Val/eval manifest. Defaults to $VAL_MANIFEST.")
    parser.add_argument("--dataset_name", default="", help="Dataset name for output dirs. Defaults to inferred manifest dataset.")
    parser.add_argument("--output_root", default=_env_or_default("PB_ROOT", os.getcwd()) + "/datasets", help="Root for generated output dirs.")
    parser.add_argument("--cuda_visible_devices", default=_env_or_default("CUDA_VISIBLE_DEVICES", "0"), help="CUDA_VISIBLE_DEVICES for launched commands.")
    parser.add_argument("--cer_threshold", type=float, default=-1.0)
    parser.add_argument("--include_google_books", action="store_true")
    parser.add_argument("--source_datasets", default="")
    parser.add_argument("--exclude_source_datasets", default="")
    parser.add_argument("--image_preprocess_pipeline", default="gray")
    parser.add_argument("--enable_fixed_resize", action="store_true", default=True)
    parser.add_argument("--enable_letterboxing", action="store_true")
    parser.add_argument("--target_height", type=int, default=256)
    parser.add_argument("--target_width", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=160)
    parser.add_argument("--generation_max_length", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--limit_errors", type=int, default=0)
    parser.add_argument("--val_fraction", type=float, default=0.0)
    parser.add_argument("--run", action="store_true", help="Actually run commands. Default only prints them.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if not str(args.checkpoint).strip():
        raise SystemExit("Missing --checkpoint or $CKPT.")
    run_dir = Path(str(args.run_dir)).expanduser() if str(args.run_dir).strip() else None
    if not str(args.tokenizer_path).strip():
        if run_dir is None:
            raise SystemExit("Missing --tokenizer_path or $RUN.")
        args.tokenizer_path = str(run_dir / "tokenizer")
    if not str(args.image_processor_path).strip():
        if run_dir is None:
            raise SystemExit("Missing --image_processor_path or $RUN.")
        args.image_processor_path = str(run_dir / "image_processor")

    manifests = {
        "train": Path(str(args.train_manifest)).expanduser() if str(args.train_manifest).strip() else None,
        "val": Path(str(args.val_manifest)).expanduser() if str(args.val_manifest).strip() else None,
    }
    missing = [name for name, path in manifests.items() if path is None]
    if missing:
        raise SystemExit(f"Missing manifest(s): {', '.join(missing)}. Provide --train_manifest/--val_manifest or env vars.")

    checkpoint_slug = _slug(Path(str(args.checkpoint)).expanduser().name)
    dataset_name = _slug(args.dataset_name or _infer_dataset_name(manifests["train"]))  # type: ignore[arg-type]
    threshold_slug = _threshold_slug(float(args.cer_threshold))
    output_root = Path(str(args.output_root)).expanduser().resolve()

    commands: List[List[str]] = []
    for split, manifest in manifests.items():
        assert manifest is not None
        out_dir = output_root / f"donut_error_extract_{dataset_name}_{split}_{checkpoint_slug}_{threshold_slug}"
        commands.append(_build_command(split=split, manifest=manifest, output_dir=out_dir, args=args))

    env = os.environ.copy()
    if str(args.cuda_visible_devices).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices).strip()

    for cmd in commands:
        print(_quote_cmd(cmd, cuda_visible_devices=str(args.cuda_visible_devices)))
        if args.run:
            subprocess.run(cmd, cwd=Path.cwd(), env=env, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
