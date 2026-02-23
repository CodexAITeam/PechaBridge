#!/usr/bin/env python3
"""Train a Donut-style OCR model (VisionEncoderDecoder) on OCR crop manifests."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel,
    set_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tibetan_utils.arg_utils import create_train_donut_ocr_parser
from pechabridge.ocr.preprocess import PreprocessConfig as PBPreprocessConfig, preprocess_patch_image
from pechabridge.ocr.preprocess_bdrc import BDRCPreprocessConfig, preprocess_image_bdrc

LOGGER = logging.getLogger("train_donut_ocr")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _parse_csv_tokens(spec: str) -> List[str]:
    return [tok.strip() for tok in str(spec).split(",") if tok.strip()]


def _read_manifest(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{idx}: {exc}") from exc
            if not isinstance(row, dict):
                continue
            rows.append(row)
    return rows


def _resolve_val_manifest_path(train_manifest: Path, val_manifest_arg: str) -> Optional[Path]:
    raw = str(val_manifest_arg or "").strip()
    if raw:
        p = Path(raw).expanduser().resolve()
        if p.exists():
            return p
        # Quality-of-life fallback for prepared OpenPecha manifests using eval_manifest.jsonl.
        if p.name == "val_manifest.jsonl":
            alt = p.with_name("eval_manifest.jsonl")
            if alt.exists():
                LOGGER.info("val_manifest not found, using eval manifest instead: %s", alt)
                return alt
        return p

    candidates = [
        train_manifest.parent / "val_manifest.jsonl",
        train_manifest.parent / "eval_manifest.jsonl",
    ]
    for cand in candidates:
        if cand.exists():
            if cand.name != "val_manifest.jsonl":
                LOGGER.info("Using validation manifest fallback: %s", cand)
            return cand.resolve()
    return None


def _iter_texts(rows: Sequence[Dict[str, object]]) -> Iterable[str]:
    for row in rows:
        text = row.get("text", "")
        if isinstance(text, str) and text.strip():
            yield text


def _ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token_id is not None:
        return
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return
    tokenizer.add_special_tokens({"pad_token": "<pad>"})


def _load_or_build_tokenizer(args, train_rows: Sequence[Dict[str, object]]):
    base_path = args.tokenizer_path or args.model_name_or_path
    LOGGER.info("Loading tokenizer for OCR training from: %s", base_path)
    tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=True)
    special_tokens = _parse_csv_tokens(args.extra_special_tokens)

    if args.train_tokenizer:
        corpus_iter = _iter_texts(train_rows)
        LOGGER.info(
            "Training tokenizer from iterator (vocab_size=%d) using base=%s",
            int(args.tokenizer_vocab_size),
            base_path,
        )
        tokenizer = tokenizer.train_new_from_iterator(
            corpus_iter,
            vocab_size=int(args.tokenizer_vocab_size),
            new_special_tokens=special_tokens,
        )

    if special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    _ensure_pad_token(tokenizer)
    return tokenizer


def _configure_image_processor(image_processor, image_size: int):
    if image_size <= 0:
        return
    try:
        size = getattr(image_processor, "size", None)
        if isinstance(size, dict):
            if "height" in size and "width" in size:
                image_processor.size = {"height": int(image_size), "width": int(image_size)}
            elif "shortest_edge" in size:
                image_processor.size = {"shortest_edge": int(image_size)}
        elif isinstance(size, int):
            image_processor.size = int(image_size)
    except Exception as exc:
        LOGGER.warning("Could not update image processor size: %s", exc)


def _image_preprocess_pipeline_name(args) -> str:
    mode = str(getattr(args, "image_preprocess_pipeline", "none") or "none").strip().lower()
    if mode not in {"none", "pb", "bdrc"}:
        return "none"
    return mode


def _apply_image_preprocess_pipeline(image: Image.Image, pipeline: str) -> Image.Image:
    mode = str(pipeline or "none").strip().lower()
    if mode == "pb":
        return preprocess_patch_image(image=image, config=PBPreprocessConfig()).convert("RGB")
    if mode == "bdrc":
        return preprocess_image_bdrc(image=image, config=BDRCPreprocessConfig.vit_defaults()).convert("RGB")
    return image.convert("RGB")


@dataclass
class OCRSample:
    image_path: Path
    text: str


class OCRManifestDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, object]],
        *,
        image_processor,
        tokenizer,
        max_target_length: int,
        image_preprocess_pipeline: str = "none",
    ):
        self.samples: List[OCRSample] = []
        for row in rows:
            image_raw = row.get("image")
            text_raw = row.get("text")
            if not isinstance(image_raw, str) or not image_raw.strip():
                continue
            if not isinstance(text_raw, str) or not text_raw.strip():
                continue
            image_path = Path(image_raw).expanduser().resolve()
            if not image_path.exists():
                continue
            self.samples.append(OCRSample(image_path=image_path, text=text_raw))
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_target_length = int(max_target_length)
        self.image_preprocess_pipeline = str(image_preprocess_pipeline or "none")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        with Image.open(sample.image_path) as img:
            rgb = img.convert("RGB")
        rgb = _apply_image_preprocess_pipeline(rgb, self.image_preprocess_pipeline)
        pixel_values = self.image_processor(images=rgb, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.tokenizer(
            sample.text,
            truncation=True,
            max_length=self.max_target_length,
            add_special_tokens=True,
        )["input_ids"]
        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


class OCRDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: Sequence[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([item["pixel_values"] for item in features])
        label_inputs = [item["labels"] for item in features]
        padded = self.tokenizer.pad(
            {"input_ids": label_inputs},
            padding=True,
            return_tensors="pt",
        )["input_ids"]
        padded[padded == self.tokenizer.pad_token_id] = -100
        return {
            "pixel_values": pixel_values,
            "labels": padded,
        }


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]


def _normalize_for_metric(text: str, newline_token: str) -> str:
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    if newline_token == "<NL>":
        out = out.replace("<NL>", "\n")
    else:
        out = out.replace("<NL>", "\n")
    return out.strip()


def _char_error_rate(preds: Sequence[str], refs: Sequence[str], newline_token: str) -> float:
    total_dist = 0
    total_chars = 0
    for pred, ref in zip(preds, refs):
        pred_n = _normalize_for_metric(pred, newline_token)
        ref_n = _normalize_for_metric(ref, newline_token)
        total_dist += _levenshtein(pred_n, ref_n)
        total_chars += max(1, len(ref_n))
    return float(total_dist / max(1, total_chars))


def run(args) -> Dict[str, object]:
    _configure_logging()
    set_seed(int(args.seed))

    train_manifest = Path(args.train_manifest).expanduser().resolve()
    val_manifest = _resolve_val_manifest_path(train_manifest, str(getattr(args, "val_manifest", "") or ""))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _read_manifest(train_manifest)
    if not train_rows:
        raise RuntimeError(f"No training samples found in {train_manifest}")
    val_rows = _read_manifest(val_manifest) if val_manifest and val_manifest.exists() else []
    LOGGER.info("Loaded %d train rows and %d val rows", len(train_rows), len(val_rows))

    tokenizer = _load_or_build_tokenizer(args, train_rows)
    tokenizer_save_dir = Path(args.tokenizer_output_dir).expanduser().resolve() if args.tokenizer_output_dir else (output_dir / "tokenizer")
    tokenizer_save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(tokenizer_save_dir))
    LOGGER.info("Tokenizer saved to %s", tokenizer_save_dir)

    image_processor_source = args.image_processor_path or args.model_name_or_path
    image_processor = AutoImageProcessor.from_pretrained(image_processor_source)
    _configure_image_processor(image_processor, int(args.image_size))
    image_preproc_mode = _image_preprocess_pipeline_name(args)
    LOGGER.info("Donut OCR image preprocessing pipeline: %s", image_preproc_mode)
    image_processor_dir = output_dir / "image_processor"
    image_processor.save_pretrained(str(image_processor_dir))

    model = VisionEncoderDecoderModel.from_pretrained(args.model_name_or_path)
    model.decoder.resize_token_embeddings(len(tokenizer))

    decoder_start_id = tokenizer.convert_tokens_to_ids(args.decoder_start_token)
    unk_id = tokenizer.unk_token_id
    if (
        decoder_start_id is None
        or decoder_start_id < 0
        or (
            unk_id is not None
            and int(decoder_start_id) == int(unk_id)
            and str(args.decoder_start_token) != str(tokenizer.unk_token)
        )
    ):
        raise ValueError(f"decoder_start_token not found in tokenizer: {args.decoder_start_token}")
    model.config.decoder_start_token_id = int(decoder_start_id)
    model.config.pad_token_id = int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = int(tokenizer.eos_token_id)
    model.config.vocab_size = model.decoder.config.vocab_size

    model.generation_config.decoder_start_token_id = int(decoder_start_id)
    model.generation_config.pad_token_id = int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        model.generation_config.eos_token_id = int(tokenizer.eos_token_id)
    model.generation_config.max_length = int(args.generation_max_length)

    train_dataset = OCRManifestDataset(
        train_rows,
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_target_length=int(args.max_target_length),
        image_preprocess_pipeline=image_preproc_mode,
    )
    val_dataset = OCRManifestDataset(
        val_rows,
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_target_length=int(args.max_target_length),
        image_preprocess_pipeline=image_preproc_mode,
    ) if val_rows else None

    LOGGER.info("Train dataset size: %d", len(train_dataset))
    if val_dataset is not None:
        LOGGER.info("Val dataset size: %d", len(val_dataset))

    collator = OCRDataCollator(tokenizer)

    def _compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
        pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        ref_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
        cer = _char_error_rate(pred_texts, ref_texts, args.metric_newline_token)
        return {"cer": cer}

    has_eval = val_dataset is not None and len(val_dataset) > 0
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        per_device_eval_batch_size=int(args.per_device_eval_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        num_train_epochs=float(args.num_train_epochs),
        warmup_steps=int(args.warmup_steps),
        logging_steps=int(args.logging_steps),
        eval_steps=int(args.eval_steps),
        save_steps=int(args.save_steps),
        save_total_limit=int(args.save_total_limit),
        dataloader_num_workers=int(args.num_workers),
        predict_with_generate=bool(has_eval),
        generation_max_length=int(args.generation_max_length),
        fp16=bool(args.fp16),
        bf16=bool(args.bf16),
        remove_unused_columns=False,
        report_to=[],
        disable_tqdm=False,
        evaluation_strategy="steps" if has_eval else "no",
        save_strategy="steps",
        load_best_model_at_end=bool(has_eval),
        metric_for_best_model="cer" if has_eval else None,
        greater_is_better=False if has_eval else None,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if has_eval else None,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics if has_eval else None,
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    image_processor.save_pretrained(str(output_dir / "image_processor"))

    metrics: Dict[str, object] = dict(train_result.metrics or {})
    if has_eval:
        eval_metrics = trainer.evaluate()
        metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})

    summary = {
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest) if val_manifest else "",
        "output_dir": str(output_dir),
        "tokenizer_dir": str(output_dir / "tokenizer"),
        "image_processor_dir": str(output_dir / "image_processor"),
        "model_dir": str(output_dir / "model"),
        "train_samples": int(len(train_dataset)),
        "val_samples": int(len(val_dataset)) if val_dataset is not None else 0,
        "metrics": metrics,
    }
    summary_path = output_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Wrote training summary to %s", summary_path)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = create_train_donut_ocr_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
