#!/usr/bin/env python3
"""Run full label-1 OCR workflow: synthetic generation -> manifest prep -> Donut-style training."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_donut_ocr_dataset import run as run_prepare_donut_ocr_dataset
from tibetan_utils.arg_utils import create_run_donut_ocr_workflow_parser

LOGGER = logging.getLogger("run_donut_ocr_workflow")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _run_cmd(cmd: List[str], cwd: Path) -> None:
    LOGGER.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run(args) -> Dict[str, object]:
    _configure_logging()
    dataset_output_dir = Path(args.dataset_output_dir).expanduser().resolve()
    dataset_dir = dataset_output_dir / args.dataset_name
    prepared_output_dir = (
        Path(args.prepared_output_dir).expanduser().resolve()
        if args.prepared_output_dir
        else (dataset_dir / "donut_ocr_label1")
    )
    model_output_dir = Path(args.model_output_dir).expanduser().resolve()

    summary: Dict[str, object] = {
        "dataset_dir": str(dataset_dir),
        "prepared_output_dir": str(prepared_output_dir),
        "model_output_dir": str(model_output_dir),
    }

    if not args.skip_generation:
        generate_cmd = [
            sys.executable,
            str(REPO_ROOT / "generate_training_data.py"),
            "--train_samples",
            str(int(args.train_samples)),
            "--val_samples",
            str(int(args.val_samples)),
            "--dataset_name",
            str(args.dataset_name),
            "--output_dir",
            str(dataset_output_dir),
            "--font_path_tibetan",
            str(args.font_path_tibetan),
            "--font_path_chinese",
            str(args.font_path_chinese),
            "--augmentation",
            str(args.augmentation),
            "--save_rendered_text_targets",
            "--save_ocr_crops",
            "--ocr_crop_labels",
            "1",
            "--target_newline_token",
            str(args.target_newline_token),
        ]
        if str(args.lora_augment_path).strip():
            generate_cmd.extend(
                [
                    "--lora_augment_path",
                    str(args.lora_augment_path),
                    "--lora_augment_model_family",
                    str(args.lora_augment_model_family),
                    "--lora_augment_base_model_id",
                    str(args.lora_augment_base_model_id),
                    "--lora_augment_controlnet_model_id",
                    str(args.lora_augment_controlnet_model_id),
                    "--lora_augment_prompt",
                    str(args.lora_augment_prompt),
                    "--lora_augment_scale",
                    str(float(args.lora_augment_scale)),
                    "--lora_augment_strength",
                    str(float(args.lora_augment_strength)),
                    "--lora_augment_steps",
                    str(int(args.lora_augment_steps)),
                    "--lora_augment_guidance_scale",
                    str(float(args.lora_augment_guidance_scale)),
                    "--lora_augment_controlnet_scale",
                    str(float(args.lora_augment_controlnet_scale)),
                    "--lora_augment_splits",
                    str(args.lora_augment_splits),
                    "--lora_augment_targets",
                    str(args.lora_augment_targets),
                    "--lora_augment_canny_low",
                    str(int(args.lora_augment_canny_low)),
                    "--lora_augment_canny_high",
                    str(int(args.lora_augment_canny_high)),
                ]
            )
            if args.lora_augment_seed is not None:
                generate_cmd.extend(["--lora_augment_seed", str(int(args.lora_augment_seed))])
        _run_cmd(generate_cmd, cwd=REPO_ROOT)
    else:
        LOGGER.info("Skipping synthetic generation step")

    train_manifest = prepared_output_dir / "train_manifest.jsonl"
    val_manifest = prepared_output_dir / "val_manifest.jsonl"
    if not args.skip_prepare:
        prepare_args = argparse.Namespace(
            dataset_dir=str(dataset_dir),
            output_dir=str(prepared_output_dir),
            label_id=1,
            splits="train,val",
            text_field="target_text",
            normalization="NFC",
            output_newline_token="keep",
            min_chars=1,
            max_chars=0,
            max_samples_per_split=0,
            seed=int(args.seed),
            wrap_task_tokens=True,
            no_wrap_task_tokens=False,
            task_start_token="<s_ocr>",
            task_end_token="</s_ocr>",
            include_class_token=True,
            no_include_class_token=False,
            class_token="<s_cls1>",
            dedupe=True,
            no_dedupe=False,
        )
        prepare_summary = run_prepare_donut_ocr_dataset(prepare_args)
        summary["prepare_summary"] = prepare_summary
    else:
        LOGGER.info("Skipping manifest preparation step")

    if not train_manifest.exists():
        raise FileNotFoundError(f"Missing train manifest: {train_manifest}")
    if not val_manifest.exists():
        LOGGER.warning("Validation manifest not found, training will run without eval: %s", val_manifest)

    if not args.skip_train:
        from scripts.train_donut_ocr import run as run_train_donut_ocr

        train_args = argparse.Namespace(
            train_manifest=str(train_manifest),
            val_manifest=str(val_manifest) if val_manifest.exists() else "",
            output_dir=str(model_output_dir),
            model_name_or_path=str(args.model_name_or_path),
            image_processor_path="",
            tokenizer_path=str(args.tokenizer_path),
            train_tokenizer=bool(args.train_tokenizer),
            tokenizer_vocab_size=int(args.tokenizer_vocab_size),
            tokenizer_output_dir="",
            extra_special_tokens="<NL>,<s_ocr>,</s_ocr>,<s_cls1>",
            decoder_start_token="<s_ocr>",
            image_size=int(args.image_size),
            image_preprocess_pipeline=str(getattr(args, "image_preprocess_pipeline", "none") or "none"),
            max_target_length=int(args.max_target_length),
            generation_max_length=int(args.max_target_length),
            per_device_train_batch_size=int(args.per_device_train_batch_size),
            per_device_eval_batch_size=int(args.per_device_eval_batch_size),
            gradient_accumulation_steps=1,
            learning_rate=float(args.learning_rate),
            weight_decay=0.01,
            num_train_epochs=float(args.num_train_epochs),
            warmup_steps=200,
            logging_steps=20,
            eval_steps=200,
            save_steps=200,
            save_total_limit=3,
            num_workers=4,
            seed=int(args.seed),
            fp16=False,
            bf16=False,
            metric_newline_token="<NL>" if str(args.target_newline_token) == "<NL>" else "\\n",
            resume_from_checkpoint="",
        )
        train_summary = run_train_donut_ocr(train_args)
        summary["train_summary"] = train_summary
    else:
        LOGGER.info("Skipping OCR training step")

    summary_path = model_output_dir / "workflow_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Workflow summary written to %s", summary_path)
    return summary


def main(argv: List[str] | None = None) -> int:
    parser = create_run_donut_ocr_workflow_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
