#!/usr/bin/env python3
"""Compare multiple Donut/TrOCR OCR checkpoints on the same validation subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoImageProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.train_donut_ocr as train_mod


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare OCR checkpoints on a fixed val subset")
    p.add_argument("--val_manifest", type=str, required=True, help="Validation JSONL manifest")
    p.add_argument("--checkpoints", type=str, required=True, help="Comma-separated checkpoint/model dirs")
    p.add_argument("--tokenizer_path", type=str, default="/home/ubuntu/data/PechaBridge/ext/BoSentencePiece")
    p.add_argument("--image_processor_path", type=str, default="microsoft/trocr-base-stage1")
    p.add_argument("--image_preprocess_pipeline", type=str, default="bdrc", choices=["none", "pb", "bdrc"])
    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--max_target_length", type=int, default=160)
    p.add_argument("--generation_max_length", type=int, default=160)
    p.add_argument("--generation_min_new_tokens", type=int, default=16)
    p.add_argument("--decoder_start_token", type=str, default="<s_ocr>")
    p.add_argument("--extra_special_tokens", type=str, default="<NL>,<s_ocr>,</s_ocr>,<s_cls1>")
    p.add_argument("--metric_newline_token", type=str, default="<NL>", choices=["<NL>", "\\n"])
    p.add_argument("--num_samples", type=int, default=50, help="Fixed number of validation samples")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--show_examples", type=int, default=5, help="How many best/worst examples to print per checkpoint")
    return p.parse_args()


def _build_tokenizer(args: argparse.Namespace):
    tok = train_mod._load_tokenizer_robust(args.tokenizer_path)
    extra = train_mod._parse_csv_tokens(args.extra_special_tokens)
    if extra:
        tok.add_special_tokens({"additional_special_tokens": extra})
    train_mod._ensure_pad_token(tok)
    return tok


def _build_dataset(args: argparse.Namespace, tokenizer):
    rows = train_mod._read_manifest(Path(args.val_manifest).expanduser().resolve())
    image_processor = AutoImageProcessor.from_pretrained(args.image_processor_path)
    train_mod._configure_image_processor(image_processor, int(args.image_size))
    end_tok = train_mod._paired_end_token_for_start(args.decoder_start_token)
    ds = train_mod.OCRManifestDataset(
        rows,
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_target_length=int(args.max_target_length),
        image_preprocess_pipeline=args.image_preprocess_pipeline,
        target_start_token=str(args.decoder_start_token),
        target_end_token=str(end_tok),
        manifest_path=Path(args.val_manifest).expanduser().resolve(),
    )
    return ds, image_processor, end_tok


def _sample_subset(ds, n: int, seed: int):
    n = min(max(1, int(n)), len(ds))
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(len(ds), size=n, replace=False)).tolist()
    return Subset(ds, idx), idx


def _configure_model_for_eval(model, tokenizer, decoder_start_token: str, generation_max_length: int, generation_min_new_tokens: int):
    decoder_start_id = int(tokenizer.convert_tokens_to_ids(decoder_start_token))
    task_end_token = train_mod._paired_end_token_for_start(decoder_start_token)
    task_end_id = None
    if task_end_token:
        cid = tokenizer.convert_tokens_to_ids(task_end_token)
        if cid is not None and int(cid) >= 0 and (tokenizer.unk_token_id is None or int(cid) != int(tokenizer.unk_token_id)):
            task_end_id = int(cid)
    effective_eos = int(task_end_id) if task_end_id is not None else (int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None)
    model.config.decoder_start_token_id = decoder_start_id
    model.config.pad_token_id = int(tokenizer.pad_token_id)
    if effective_eos is not None:
        model.config.eos_token_id = int(effective_eos)
    model.generation_config.decoder_start_token_id = decoder_start_id
    model.generation_config.pad_token_id = int(tokenizer.pad_token_id)
    if effective_eos is not None:
        model.generation_config.eos_token_id = int(effective_eos)
    model.generation_config.max_length = int(generation_max_length)
    model.generation_config.num_beams = 1
    try:
        model.generation_config.min_new_tokens = max(0, int(generation_min_new_tokens))
    except Exception:
        pass


def _run_checkpoint(
    ckpt_path: str,
    *,
    subset,
    tokenizer,
    collator,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    model = train_mod._load_ved_model_robust(str(ckpt_path))
    _configure_model_for_eval(
        model,
        tokenizer,
        decoder_start_token=str(args.decoder_start_token),
        generation_max_length=int(args.generation_max_length),
        generation_min_new_tokens=int(args.generation_min_new_tokens),
    )
    model.to(device)
    model.eval()

    dl = DataLoader(subset, batch_size=int(args.batch_size), shuffle=False, collate_fn=collator)
    preds_all: List[str] = []
    refs_all: List[str] = []
    sample_rows: List[Dict[str, object]] = []
    cursor = 0
    with torch.no_grad():
        for batch in dl:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].detach().cpu().numpy()
            gen_ids = model.generate(
                pixel_values=pixel_values,
                decoder_start_token_id=model.config.decoder_start_token_id,
                eos_token_id=model.config.eos_token_id,
                pad_token_id=model.config.pad_token_id,
                max_length=int(args.generation_max_length),
                num_beams=1,
                do_sample=False,
                min_new_tokens=max(0, int(args.generation_min_new_tokens)),
            )
            pred_norms = train_mod._decode_for_metric(tokenizer, gen_ids.detach().cpu().numpy(), newline_token=args.metric_newline_token)
            labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
            ref_norms = train_mod._decode_for_metric(tokenizer, labels, newline_token=args.metric_newline_token)
            for p, r in zip(pred_norms, ref_norms):
                p = str(p or "")
                r = str(r or "")
                preds_all.append(p)
                refs_all.append(r)
                sample_rows.append(
                    {
                        "i": int(cursor),
                        "pred": p,
                        "ref": r,
                        "pred_len": len(p),
                        "ref_len": len(r),
                        "empty_pred": int(not p),
                        "empty_ref": int(not r),
                    }
                )
                cursor += 1

    valid_pairs = [(p, r) for p, r in zip(preds_all, refs_all) if r]
    empty_ref_count = sum(1 for r in refs_all if not r)
    empty_pred_count = sum(1 for p, r in zip(preds_all, refs_all) if (not p) and r)
    cer = train_mod._char_error_rate([p for p, _ in valid_pairs], [r for _, r in valid_pairs], args.metric_newline_token) if valid_pairs else 1.0
    for row in sample_rows:
        if row["ref"]:
            row["cer"] = train_mod._sample_cer(str(row["pred"]), str(row["ref"]))
        else:
            row["cer"] = None
    worst = [r for r in sample_rows if r["cer"] is not None]
    worst.sort(key=lambda x: float(x["cer"]), reverse=True)
    best = [r for r in worst if float(r["cer"]) <= 1.0]
    best.sort(key=lambda x: float(x["cer"]))
    return {
        "checkpoint": str(ckpt_path),
        "cer": float(cer),
        "cer_percent": float(cer) * 100.0,
        "valid_n": int(len(valid_pairs)),
        "empty_ref_count": int(empty_ref_count),
        "empty_pred_count": int(empty_pred_count),
        "worst_examples": worst[: max(0, int(args.show_examples))],
        "best_examples": best[: max(0, int(args.show_examples))],
    }


def main() -> int:
    args = _parse_args()
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    ckpts = [s.strip() for s in str(args.checkpoints).split(",") if s.strip()]
    if not ckpts:
        raise SystemExit("No checkpoints provided")

    tokenizer = _build_tokenizer(args)
    ds, _, _ = _build_dataset(args, tokenizer)
    subset, sampled_idx = _sample_subset(ds, int(args.num_samples), int(args.seed))
    collator = train_mod.OCRDataCollator(tokenizer)

    print(f"Using {len(subset)} validation samples (seed={args.seed})")
    print(f"Sample indices head: {sampled_idx[:10]}")
    print(f"Device: {device}")
    print("")

    results: List[Dict[str, object]] = []
    for ckpt in ckpts:
        print(f"=== {ckpt} ===")
        res = _run_checkpoint(ckpt, subset=subset, tokenizer=tokenizer, collator=collator, args=args, device=device)
        results.append(res)
        print(
            json.dumps(
                {
                    "checkpoint": res["checkpoint"],
                    "cer": round(float(res["cer"]), 4),
                    "cer_percent": round(float(res["cer_percent"]), 2),
                    "valid_n": res["valid_n"],
                    "empty_ref_count": res["empty_ref_count"],
                    "empty_pred_count": res["empty_pred_count"],
                },
                ensure_ascii=False,
            )
        )
        for tag in ("worst_examples", "best_examples"):
            rows = res.get(tag) or []
            if not rows:
                continue
            print(f"{tag}:")
            for ex in rows:
                print(
                    f"  [i={ex['i']}] cer={ex['cer']:.4f} ref_len={ex['ref_len']} pred_len={ex['pred_len']} "
                    f"REF={ex['ref'][:120]!r} PRD={ex['pred'][:120]!r}"
                )
        print("")

    print("=== summary (sorted by CER) ===")
    for r in sorted(results, key=lambda x: float(x["cer"])):
        print(f"{float(r['cer_percent']):6.2f}% | {r['checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
