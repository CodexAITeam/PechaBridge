#!/usr/bin/env python3
"""
End-to-end workflow:
1) pseudo labels from VLM
2) layout rule filtering
3) YOLO -> Label Studio task conversion
4) optional Label Studio start
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pseudo-labeling workflow up to Label Studio start")
    parser.add_argument("--input-dir", required=True, help="Input image directory")
    parser.add_argument("--work-dir", required=True, help="Base work directory for generated artifacts")
    parser.add_argument("--split", default="train", help="Split name (default: train)")
    parser.add_argument(
        "--parser",
        default="paddleocr_vl",
        choices=[
            "mineru25",
            "paddleocr_vl",
            "qwen25vl",
            "granite_docling",
            "deepseek_ocr",
            "qwen3_vl",
            "groundingdino",
            "florence2",
        ],
        help="Pseudo-label backend",
    )
    parser.add_argument("--recursive", action="store_true", help="Scan input recursively")
    parser.add_argument("--min-confidence", type=float, default=0.2, help="Min confidence threshold")
    parser.add_argument("--max-images", type=int, default=0, help="Max images (0 = all)")

    # VLM/MinerU options
    parser.add_argument("--hf-model-id", default="", help="HF model id override")
    parser.add_argument("--vlm-prompt", default="", help="Custom VLM prompt")
    parser.add_argument("--vlm-max-new-tokens", type=int, default=1024)
    parser.add_argument("--hf-device", default="auto")
    parser.add_argument("--hf-dtype", default="auto")
    parser.add_argument("--mineru-command", default="mineru")
    parser.add_argument("--mineru-timeout", type=int, default=300)

    # Layout filter options
    parser.add_argument("--left-max", type=float, default=0.33)
    parser.add_argument("--right-min", type=float, default=0.66)
    parser.add_argument("--allow-relabel", action="store_true")
    parser.add_argument("--keep-single-per-class", action="store_true")

    # Label Studio conversion options
    parser.add_argument("--image-ext", default=".jpg", help="Image extension for converter (.jpg/.png)")
    parser.add_argument("--tasks-json", default="ls-tasks-pseudo.json", help="Output tasks json filename")

    # Label Studio runtime
    parser.add_argument("--start-label-studio", action="store_true", help="Start Label Studio at end")
    return parser.parse_args()


def run_cmd(cmd: list[str], env: dict | None = None) -> None:
    pretty = " ".join(shlex.quote(c) for c in cmd)
    print(f"\n$ {pretty}")
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parent
    work_dir = Path(args.work_dir).resolve()
    pseudo_dir = work_dir / "pseudo-vlm"
    filtered_dir = work_dir / "pseudo-vlm-filtered"
    pseudo_split = pseudo_dir / args.split
    filtered_split = filtered_dir / args.split
    tasks_path = work_dir / args.tasks_json

    work_dir.mkdir(parents=True, exist_ok=True)

    # 1) Pseudo label generation
    pseudo_cmd = [
        sys.executable,
        str(root / "pseudo_label_from_vlm.py"),
        "--input-dir",
        str(Path(args.input_dir).resolve()),
        "--output-dir",
        str(pseudo_dir),
        "--split",
        args.split,
        "--parser",
        args.parser,
        "--copy-images",
        "--save-raw-json",
        "--min-confidence",
        str(args.min_confidence),
    ]
    if args.recursive:
        pseudo_cmd.append("--recursive")
    if args.max_images > 0:
        pseudo_cmd.extend(["--max-images", str(args.max_images)])

    pseudo_cmd.extend(["--vlm-max-new-tokens", str(args.vlm_max_new_tokens)])
    pseudo_cmd.extend(["--hf-device", args.hf_device, "--hf-dtype", args.hf_dtype])
    pseudo_cmd.extend(["--mineru-command", args.mineru_command, "--mineru-timeout", str(args.mineru_timeout)])
    if args.hf_model_id:
        pseudo_cmd.extend(["--hf-model-id", args.hf_model_id])
    if args.vlm_prompt:
        pseudo_cmd.extend(["--vlm-prompt", args.vlm_prompt])

    run_cmd(pseudo_cmd)

    # 2) Layout rule filtering
    filter_cmd = [
        sys.executable,
        str(root / "layout_rule_filter.py"),
        "--input-split-dir",
        str(pseudo_split),
        "--output-split-dir",
        str(filtered_split),
        "--left-max",
        str(args.left_max),
        "--right-min",
        str(args.right_min),
        "--copy-images",
    ]
    if args.allow_relabel:
        filter_cmd.append("--allow-relabel")
    if args.keep_single_per_class:
        filter_cmd.append("--keep-single-per-class")
    run_cmd(filter_cmd)

    # 3) Convert YOLO to Label Studio tasks
    convert_cmd = [
        "label-studio-converter",
        "import",
        "yolo",
        "-i",
        str(filtered_split),
        "-o",
        str(tasks_path),
        "--image-ext",
        args.image_ext,
        "--image-root-url",
        f"/data/local-files/?d={args.split}/images",
    ]
    run_cmd(convert_cmd)

    # 4) Prepare Label Studio env + optional launch
    ls_root = str(filtered_dir)
    print("\nLabel Studio environment:")
    print(f"export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true")
    print(f"export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={ls_root}")
    print(f"tasks json: {tasks_path}")
    print(f"filtered split: {filtered_split}")

    if args.start_label_studio:
        env = os.environ.copy()
        env["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
        env["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = ls_root
        run_cmd(["label-studio"], env=env)


if __name__ == "__main__":
    main()
