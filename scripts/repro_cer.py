#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pechabridge.ocr.repro_pack import (  # noqa: E402
    FrozenValSubsetDataset,
    frozen_val_collate,
    load_checkpoint_bundle,
)
import scripts.train_donut_ocr as train_mod  # noqa: E402


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce CER offline from a training checkpoint repro bundle.")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint-<step> directory")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--tolerance", type=float, default=1e-6, help="Allowed absolute CER difference vs saved repro CER")
    p.add_argument("--run_twice", action="store_true", help="Determinism smoke test: run twice and assert identical preds/CER")
    p.add_argument("--show_examples", type=int, default=0)
    return p.parse_args(argv)


def _build_generate_kwargs(generate_cfg: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "decoder_start_token_id",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "max_length",
        "max_new_tokens",
        "min_new_tokens",
        "num_beams",
        "do_sample",
        "temperature",
        "top_p",
        "repetition_penalty",
        "no_repeat_ngram_size",
        "length_penalty",
        "early_stopping",
        "bad_words_ids",
        "forced_bos_token_id",
        "forced_eos_token_id",
        "use_cache",
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        v = generate_cfg.get(k)
        if v is not None:
            out[k] = v
    return out


def _run_once(
    *,
    model,
    tokenizer,
    image_processor,
    records,
    image_preprocess_pipeline: str,
    batch_size: int,
    device: torch.device,
    generate_cfg: Dict[str, Any],
    newline_token: str,
) -> Tuple[float, List[Dict[str, Any]], Dict[str, Any]]:
    ds = FrozenValSubsetDataset(
        records,
        image_processor=image_processor,
        image_preprocess_fn=train_mod._apply_image_preprocess_pipeline,
        image_preprocess_pipeline=image_preprocess_pipeline,
    )
    loader = DataLoader(ds, batch_size=max(1, int(batch_size)), shuffle=False, collate_fn=frozen_val_collate, num_workers=0)
    model = model.to(device)
    model.eval()
    gkwargs = _build_generate_kwargs(generate_cfg)

    pred_rows: List[Dict[str, Any]] = []
    edit_sum = 0
    ref_len_sum = 0
    empty_ref = 0
    empty_pred = 0
    with torch.no_grad():
        for batch in loader:
            px = batch["pixel_values"].to(device)
            gen_ids = model.generate(pixel_values=px, **gkwargs)
            pred_raw_list = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            for meta, pred_raw in zip(batch["meta"], pred_raw_list):
                pred_raw = str(pred_raw or "")
                pred_norm = train_mod._normalize_for_metric(pred_raw, newline_token)
                ref_norm = str(meta.get("gt_text_metric_norm", "") or "")
                if not ref_norm:
                    empty_ref += 1
                if not pred_norm:
                    empty_pred += 1
                if ref_norm:
                    edit_sum += int(train_mod._levenshtein(pred_norm, ref_norm))
                    ref_len_sum += max(1, len(ref_norm))
                pred_rows.append(
                    {
                        "idx": int(meta["idx"]),
                        "id": str(meta["id"]),
                        "gt_norm": ref_norm,
                        "pred_norm": pred_norm,
                        "gt_raw": str(meta.get("gt_text_raw", "")),
                        "gt_metric_raw": str(meta.get("gt_text_metric_raw", "")),
                        "pred_raw": pred_raw,
                    }
                )
    pred_rows = sorted(pred_rows, key=lambda r: int(r["idx"]))
    cer = float(edit_sum / max(1, ref_len_sum))
    metrics = {
        "cer": cer,
        "cer_percent": cer * 100.0,
        "valid_n": int(sum(1 for r in pred_rows if r["gt_norm"])),
        "empty_ref_count": int(empty_ref),
        "empty_pred_count": int(empty_pred),
        "edit_sum": int(edit_sum),
        "ref_len_sum": int(ref_len_sum),
    }
    return cer, pred_rows, metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    ckpt_dir = Path(args.ckpt).expanduser().resolve()
    repro_dir = ckpt_dir / "repro"
    model, tokenizer, image_processor, generate_cfg, norm_cfg, records, saved_cer_record = load_checkpoint_bundle(
        ckpt_dir,
        tokenizer_loader=train_mod._load_tokenizer_robust,
    )
    image_preprocess_contract = json.loads((repro_dir / "image_preprocess.json").read_text(encoding="utf-8"))
    image_preprocess_pipeline = str(image_preprocess_contract.get("pipeline", "none"))
    newline_token = str(norm_cfg.get("metric_newline_token", "<NL>") or "<NL>")
    device = torch.device(args.device)

    cer1, preds1, metrics1 = _run_once(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        records=records,
        image_preprocess_pipeline=image_preprocess_pipeline,
        batch_size=args.batch_size,
        device=device,
        generate_cfg=generate_cfg,
        newline_token=newline_token,
    )
    print(json.dumps({"checkpoint": str(ckpt_dir), **metrics1}, ensure_ascii=False))

    if args.show_examples > 0:
        for row in preds1[: int(args.show_examples)]:
            print(
                f"[i={row['idx']}] gt_len={len(row['gt_norm'])} pred_len={len(row['pred_norm'])} "
                f"GT={row['gt_norm'][:180]!r} PRD={row['pred_norm'][:180]!r}"
            )

    expected = saved_cer_record.get("repro_cer")
    if expected is None:
        raise RuntimeError(
            f"No saved repro CER found in {repro_dir}. Expected a file like cer_step_XXXXXX.json with key 'repro_cer'."
        )
    diff = abs(float(cer1) - float(expected))
    print(
        json.dumps(
            {
                "expected_repro_cer": float(expected),
                "offline_repro_cer": float(cer1),
                "abs_diff": float(diff),
                "tolerance": float(args.tolerance),
            },
            ensure_ascii=False,
        )
    )
    if diff > float(args.tolerance):
        raise AssertionError(
            f"Offline CER drift exceeds tolerance: offline={cer1:.10f} saved={float(expected):.10f} diff={diff:.10f}"
        )

    if args.run_twice:
        cer2, preds2, metrics2 = _run_once(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            records=records,
            image_preprocess_pipeline=image_preprocess_pipeline,
            batch_size=args.batch_size,
            device=device,
            generate_cfg=generate_cfg,
            newline_token=newline_token,
        )
        pred_sig_1 = [(r["id"], r["pred_raw"], r["pred_norm"]) for r in preds1]
        pred_sig_2 = [(r["id"], r["pred_raw"], r["pred_norm"]) for r in preds2]
        if pred_sig_1 != pred_sig_2 or abs(cer1 - cer2) > float(args.tolerance):
            raise AssertionError(
                f"Determinism smoke test failed: cer1={cer1:.10f}, cer2={cer2:.10f}, "
                f"preds_equal={pred_sig_1 == pred_sig_2}"
            )
        print(json.dumps({"determinism_check": "ok", **metrics2}, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

