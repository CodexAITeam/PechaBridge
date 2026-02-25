#!/usr/bin/env python3
"""Train a Donut-style OCR model (VisionEncoderDecoder) on OCR crop manifests."""

from __future__ import annotations

import json
import inspect
import logging
import os
import re
import shutil
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, Subset
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel,
    set_seed,
)
from transformers.trainer_callback import ProgressCallback, TrainerCallback

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tibetan_utils.arg_utils import create_train_donut_ocr_parser
from pechabridge.ocr.preprocess import PreprocessConfig as PBPreprocessConfig, preprocess_patch_image
from pechabridge.ocr.preprocess_bdrc import BDRCPreprocessConfig, preprocess_image_bdrc
from pechabridge.ocr.sentencepiece_tokenizer_adapter import (
    find_sentencepiece_model_path as sp_find_model_path,
    load_sentencepiece_tokenizer as load_sentencepiece_tokenizer_adapter,
)

LOGGER = logging.getLogger("train_donut_ocr")


_ZERO_WIDTH_CHARS = ("\u200b", "\u200c", "\u200d", "\ufeff")


def _paired_end_token_for_start(start_token: str) -> str:
    s = str(start_token or "").strip()
    if s.startswith("<s_") and s.endswith(">"):
        return "</" + s[1:]
    return ""


def _format_ocr_target_text(raw_text: str, *, start_token: str, end_token: str) -> str:
    text = str(raw_text or "")
    st = str(start_token or "").strip()
    et = str(end_token or "").strip()
    if not st or not et:
        return text
    # Avoid double-wrapping if the manifest text already contains task tokens.
    if text.startswith(st) and et in text:
        return text
    return f"{st}{text}{et}"


def _strip_special_token_strings(text: str, tokenizer) -> str:
    out = str(text or "")
    try:
        toks = sorted(
            [str(t) for t in getattr(tokenizer, "all_special_tokens", []) if isinstance(t, str) and t],
            key=len,
            reverse=True,
        )
    except Exception:
        toks = []
    for tok in toks:
        out = out.replace(tok, "")
    return out


def _configure_logging() -> None:
    rank_raw = os.environ.get("RANK", "")
    local_rank_raw = os.environ.get("LOCAL_RANK", "")
    try:
        rank = int(rank_raw) if str(rank_raw).strip() != "" else 0
    except Exception:
        rank = 0
    try:
        local_rank = int(local_rank_raw) if str(local_rank_raw).strip() != "" else rank
    except Exception:
        local_rank = rank
    is_primary = (rank == 0)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if not is_primary:
        # Suppress duplicate logs from non-zero ranks in torchrun/DDP.
        logging.disable(logging.CRITICAL)
        return
    LOGGER.info("Logging enabled on primary process only (RANK=%d LOCAL_RANK=%d)", rank, local_rank)


class DonutProgressCallback(ProgressCallback):
    """Progress bar callback that also shows compact train/eval metrics in TQDM postfix."""

    _POSTFIX_KEYS = (
        "loss",
        "eval_loss",
        "eval_cer",
        "learning_rate",
        "grad_norm",
        "epoch",
    )

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        logs = logs or {}
        if state.is_world_process_zero and self.training_bar is not None:
            postfix: Dict[str, object] = {}
            for key in self._POSTFIX_KEYS:
                if key not in logs:
                    continue
                val = logs[key]
                if isinstance(val, float):
                    if key == "learning_rate":
                        postfix["lr"] = f"{val:.2e}"
                    elif key in {"loss", "eval_loss", "eval_cer", "grad_norm"}:
                        postfix[key] = f"{val:.4f}"
                    elif key == "epoch":
                        postfix[key] = f"{val:.2f}"
                    else:
                        postfix[key] = f"{val:.4g}"
                else:
                    postfix[key] = val
            if postfix:
                try:
                    self.training_bar.set_postfix(postfix, refresh=False)
                except Exception:
                    pass
        super().on_log(args, state, control, logs=logs, **kwargs)


class DonutCheckpointAliasCallback(TrainerCallback):
    """Create human-readable checkpoint aliases containing epoch and CER.

    The actual HF checkpoints keep their default `checkpoint-<step>` names for
    resume compatibility. We add symlink aliases like:
      checkpoint-epoch-25-cer-0.1234 -> checkpoint-1192425
    """

    def __init__(self):
        self._last_eval_metrics: Dict[str, float] = {}

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # type: ignore[override]
        if isinstance(metrics, dict):
            try:
                self._last_eval_metrics = {
                    str(k): float(v) for k, v in metrics.items() if isinstance(v, (float, int))
                }
            except Exception:
                self._last_eval_metrics = {}
        return super().on_evaluate(args, state, control, **kwargs)

    def on_save(self, args, state, control, **kwargs):  # type: ignore[override]
        if not getattr(state, "is_world_process_zero", True):
            return control
        step = int(getattr(state, "global_step", 0) or 0)
        if step <= 0:
            return control
        output_dir = Path(str(getattr(args, "output_dir", ""))).expanduser().resolve()
        ckpt_dir = output_dir / f"checkpoint-{step}"
        if not ckpt_dir.exists():
            return control
        epoch_val = getattr(state, "epoch", None)
        epoch_num = int(round(float(epoch_val))) if epoch_val is not None else None
        cer = self._last_eval_metrics.get("eval_cer")
        cer_str = f"{float(cer):.4f}" if cer is not None else "na"
        epoch_str = str(epoch_num) if epoch_num is not None else "na"
        alias_name = f"checkpoint-epoch-{epoch_str}-cer-{cer_str}"
        alias_path = output_dir / alias_name
        try:
            if alias_path.exists() or alias_path.is_symlink():
                if alias_path.is_symlink() or alias_path.is_file():
                    alias_path.unlink()
                elif alias_path.is_dir():
                    # Don't remove real directories; skip to avoid destructive behavior.
                    LOGGER.warning("Checkpoint alias path exists as directory, skipping: %s", alias_path)
                    return control
            os.symlink(ckpt_dir.name, alias_path)  # relative symlink within same output dir
            LOGGER.info("Created checkpoint alias: %s -> %s", alias_path.name, ckpt_dir.name)
        except Exception as exc:
            LOGGER.warning("Could not create checkpoint alias %s: %s", alias_name, exc)
        return control


class DonutDebugSeq2SeqTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer that logs one decoded training sample (GT vs argmax decode) each step."""

    def __init__(
        self,
        *args,
        tokenizer_for_debug=None,
        metric_newline_token: str = "<NL>",
        debug_train_decode_preview: bool = True,
        debug_train_decode_every_steps: int = 1,
        debug_train_trace: bool = False,
        debug_train_trace_every_steps: int = 1,
        debug_train_trace_topk: int = 5,
        debug_train_trace_max_positions: int = 8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._debug_tokenizer = tokenizer_for_debug
        self._debug_metric_newline_token = str(metric_newline_token or "<NL>")
        self._debug_train_decode_preview = bool(debug_train_decode_preview)
        self._debug_train_decode_every_steps = max(1, int(debug_train_decode_every_steps or 1))
        self._debug_train_trace = bool(debug_train_trace)
        self._debug_train_trace_every_steps = max(1, int(debug_train_trace_every_steps or 1))
        self._debug_train_trace_topk = max(1, int(debug_train_trace_topk or 5))
        self._debug_train_trace_max_positions = max(1, int(debug_train_trace_max_positions or 8))

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # type: ignore[override]
        out = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            loss, outputs = out
        else:
            loss, outputs = out, None
        try:
            self._log_decoded_training_sample(inputs, outputs)
        except Exception as exc:
            if self.is_world_process_zero():
                LOGGER.warning("Could not log decoded training sample preview: %s", exc)
        return (loss, outputs) if return_outputs else loss

    def _log_decoded_training_sample(self, inputs, outputs) -> None:
        if not self.is_world_process_zero():
            return
        if self._debug_tokenizer is None:
            return
        if not getattr(model := getattr(self, "model", None), "training", False):
            return
        if outputs is None:
            return
        logits = getattr(outputs, "logits", None)
        labels = inputs.get("labels") if isinstance(inputs, dict) else None
        if logits is None or labels is None:
            return
        if not torch.is_tensor(logits) or not torch.is_tensor(labels):
            return
        if logits.ndim != 3 or labels.ndim != 2 or logits.shape[0] <= 0 or labels.shape[0] <= 0:
            return

        pred_ids = torch.argmax(logits[0].detach(), dim=-1).to("cpu")
        label_ids = labels[0].detach().to("cpu")
        pad_id = getattr(self._debug_tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = 0
        label_ids = label_ids.clone()
        label_ids[label_ids == -100] = int(pad_id)

        pred_text = self._debug_tokenizer.decode(pred_ids.tolist(), skip_special_tokens=True)
        gt_text = self._debug_tokenizer.decode(label_ids.tolist(), skip_special_tokens=True)
        pred_norm = _normalize_for_metric(str(pred_text or ""), self._debug_metric_newline_token)
        gt_norm = _normalize_for_metric(str(gt_text or ""), self._debug_metric_newline_token)
        step = int(getattr(self.state, "global_step", 0) or 0)
        if (not self._debug_train_decode_preview) or (step % self._debug_train_decode_every_steps != 0):
            if self._debug_train_trace and (step % self._debug_train_trace_every_steps == 0):
                self._log_verbose_training_trace(inputs=inputs, outputs=outputs, step=step, logits=logits, labels=labels)
            return
        pred_ids_head = pred_ids.tolist()[:32]
        label_ids_head = label_ids.tolist()[:32]
        LOGGER.info("train_decode step=%d | GT: %r", step, gt_norm[:500])
        LOGGER.info("train_decode step=%d | PRD: %r", step, pred_norm[:500])
        LOGGER.info(
            "train_decode step=%d | GT_len=%d PRD_len=%d label_ids_head=%s pred_ids_head=%s",
            step,
            len(gt_norm),
            len(pred_norm),
            label_ids_head,
            pred_ids_head,
        )
        if self._debug_train_trace and (step % self._debug_train_trace_every_steps == 0):
            self._log_verbose_training_trace(inputs=inputs, outputs=outputs, step=step, logits=logits, labels=labels)

    def _log_verbose_training_trace(self, *, inputs, outputs, step: int, logits: torch.Tensor, labels: torch.Tensor) -> None:
        tok = self._debug_tokenizer
        if tok is None:
            return
        try:
            pixel_values = inputs.get("pixel_values") if isinstance(inputs, dict) else None
            if torch.is_tensor(pixel_values) and pixel_values.ndim >= 4 and pixel_values.shape[0] > 0:
                pv0 = pixel_values[0].detach()
                pv0f = pv0.float()
                LOGGER.info(
                    "train_trace step=%d | pixel_values batch_shape=%s dtype=%s device=%s sample0_shape=%s min=%.6f max=%.6f mean=%.6f std=%.6f",
                    step,
                    tuple(int(x) for x in pixel_values.shape),
                    str(pixel_values.dtype),
                    str(pixel_values.device),
                    tuple(int(x) for x in pv0.shape),
                    float(pv0f.min().item()),
                    float(pv0f.max().item()),
                    float(pv0f.mean().item()),
                    float(pv0f.std(unbiased=False).item()),
                )
            LOGGER.info(
                "train_trace step=%d | labels batch_shape=%s dtype=%s device=%s",
                step,
                tuple(int(x) for x in labels.shape),
                str(labels.dtype),
                str(labels.device),
            )
            enc = getattr(outputs, "encoder_last_hidden_state", None)
            if torch.is_tensor(enc):
                enc0 = enc[0].detach().float()
                LOGGER.info(
                    "train_trace step=%d | encoder_last_hidden_state shape=%s dtype=%s device=%s sample0[min=%.6f max=%.6f mean=%.6f std=%.6f]",
                    step,
                    tuple(int(x) for x in enc.shape),
                    str(enc.dtype),
                    str(enc.device),
                    float(enc0.min().item()),
                    float(enc0.max().item()),
                    float(enc0.mean().item()),
                    float(enc0.std(unbiased=False).item()),
                )
            lg0 = logits[0].detach().float().cpu()
            LOGGER.info(
                "train_trace step=%d | logits shape=%s dtype=%s device=%s sample0[min=%.6f max=%.6f mean=%.6f std=%.6f]",
                step,
                tuple(int(x) for x in logits.shape),
                str(logits.dtype),
                str(logits.device),
                float(lg0.min().item()),
                float(lg0.max().item()),
                float(lg0.mean().item()),
                float(lg0.std(unbiased=False).item()),
            )

            label0 = labels[0].detach().cpu().clone()
            pad_id = getattr(tok, "pad_token_id", None)
            if pad_id is None:
                pad_id = 0
            label0[label0 == -100] = int(pad_id)
            pred0 = torch.argmax(logits[0].detach(), dim=-1).cpu()

            label0_ids = [int(x) for x in label0.tolist()]
            pred0_ids = [int(x) for x in pred0.tolist()]
            label0_tokens = tok.convert_ids_to_tokens(label0_ids[:32])
            pred0_tokens = tok.convert_ids_to_tokens(pred0_ids[:32])
            LOGGER.info("train_trace step=%d | label0_ids_head=%s", step, label0_ids[:32])
            LOGGER.info("train_trace step=%d | label0_toks_head=%s", step, label0_tokens)
            LOGGER.info("train_trace step=%d | pred0_ids_head=%s", step, pred0_ids[:32])
            LOGGER.info("train_trace step=%d | pred0_toks_head=%s", step, pred0_tokens)

            pos_cap = min(self._debug_train_trace_max_positions, int(lg0.shape[0]))
            topk = min(self._debug_train_trace_topk, int(lg0.shape[-1]))
            probs0 = torch.softmax(lg0[:pos_cap], dim=-1)
            top_probs, top_ids = torch.topk(probs0, k=topk, dim=-1)
            for pos in range(pos_cap):
                tgt_id = int(label0_ids[pos]) if pos < len(label0_ids) else None
                pred_id = int(pred0_ids[pos]) if pos < len(pred0_ids) else None
                tgt_tok = tok.convert_ids_to_tokens([tgt_id])[0] if tgt_id is not None else None
                pred_tok = tok.convert_ids_to_tokens([pred_id])[0] if pred_id is not None else None
                cands = []
                for j in range(topk):
                    cid = int(top_ids[pos, j].item())
                    cp = float(top_probs[pos, j].item())
                    ctok = tok.convert_ids_to_tokens([cid])[0]
                    cands.append({"id": cid, "tok": ctok, "p": round(cp, 6)})
                LOGGER.info(
                    "train_trace step=%d | pos=%d tgt=(%s,%r) pred=(%s,%r) topk=%s",
                    step,
                    pos,
                    str(tgt_id),
                    tgt_tok,
                    str(pred_id),
                    pred_tok,
                    cands,
                )

            special_ids = {
                "bos": getattr(tok, "bos_token_id", None),
                "eos": getattr(tok, "eos_token_id", None),
                "pad": getattr(tok, "pad_token_id", None),
                "unk": getattr(tok, "unk_token_id", None),
                "decoder_start(model)": getattr(getattr(self.model, "config", None), "decoder_start_token_id", None),
            }
            LOGGER.info("train_trace step=%d | special_token_ids=%s", step, special_ids)
        except Exception as exc:
            LOGGER.warning("Verbose train trace failed at step=%d: %s", step, exc)


def _named_meta_tensors(module: nn.Module) -> List[str]:
    names: List[str] = []
    for name, param in module.named_parameters():
        if getattr(param, "device", None) is not None and param.device.type == "meta":
            names.append(name)
    for name, buf in module.named_buffers():
        if getattr(buf, "device", None) is not None and buf.device.type == "meta":
            names.append(name)
    return sorted(set(names))


def _materialize_meta_buffers_inplace(module: nn.Module, *, device: str = "cpu") -> List[str]:
    """Replace meta-device buffers with zero-initialized real tensors.

    This is primarily needed for some HF/TroCR positional embedding helper
    buffers (e.g. `_float_tensor`) that can remain on `meta` in newer
    transformers/torch combinations.
    """
    fixed: List[str] = []

    def _walk(mod: nn.Module, prefix: str) -> None:
        for buf_name, buf in list(mod._buffers.items()):
            if buf is None or getattr(buf, "device", None) is None or buf.device.type != "meta":
                continue
            new_buf = torch.zeros(tuple(buf.shape), dtype=buf.dtype, device=device)
            mod._buffers[buf_name] = new_buf
            fixed.append(f"{prefix}{buf_name}")
        for child_name, child in mod.named_children():
            _walk(child, f"{prefix}{child_name}.")

    _walk(module, "")
    return fixed


def _summarize_module_devices(module: nn.Module) -> Dict[str, object]:
    param_counts: Counter[str] = Counter()
    param_numel: Counter[str] = Counter()
    buffer_counts: Counter[str] = Counter()
    buffer_numel: Counter[str] = Counter()
    meta_params: List[str] = []
    meta_buffers: List[str] = []

    for name, param in module.named_parameters():
        dev = str(param.device)
        param_counts[dev] += 1
        param_numel[dev] += int(param.numel())
        if param.device.type == "meta":
            meta_params.append(name)
    for name, buf in module.named_buffers():
        dev = str(buf.device)
        buffer_counts[dev] += 1
        buffer_numel[dev] += int(buf.numel())
        if buf.device.type == "meta":
            meta_buffers.append(name)

    return {
        "param_counts_by_device": dict(param_counts),
        "param_numel_by_device": dict(param_numel),
        "buffer_counts_by_device": dict(buffer_counts),
        "buffer_numel_by_device": dict(buffer_numel),
        "meta_params": meta_params,
        "meta_buffers": meta_buffers,
    }


def _find_plain_tensor_attrs(module: nn.Module, *, max_items: int = 512) -> List[Dict[str, object]]:
    """Find tensor attributes on modules that are neither params nor buffers."""
    found: List[Dict[str, object]] = []

    def _walk(mod: nn.Module, prefix: str) -> None:
        if len(found) >= max_items:
            return
        registered_params = set(mod._parameters.keys())
        registered_buffers = set(mod._buffers.keys())
        registered_children = set(mod._modules.keys())
        for name, value in vars(mod).items():
            if name in {"_parameters", "_buffers", "_modules"}:
                continue
            if name in registered_params or name in registered_buffers or name in registered_children:
                continue
            if torch.is_tensor(value):
                t = value
                found.append(
                    {
                        "path": f"{prefix}{name}",
                        "device": str(t.device),
                        "shape": tuple(int(d) for d in t.shape),
                        "dtype": str(t.dtype),
                        "module_type": type(mod).__name__,
                    }
                )
                if len(found) >= max_items:
                    return
        for child_name, child in mod.named_children():
            _walk(child, f"{prefix}{child_name}.")

    _walk(module, "")
    return found


def _materialize_meta_tensor_attrs_inplace(module: nn.Module, *, device: str = "cpu") -> List[str]:
    """Materialize meta-device *plain tensor attributes* on modules.

    Important for transformers versions where helper tensors (e.g. TrOCR sinusoidal
    positional embeddings `weights`) are not registered as buffers.
    """
    fixed: List[str] = []

    def _walk(mod: nn.Module, prefix: str) -> None:
        registered_params = set(mod._parameters.keys())
        registered_buffers = set(mod._buffers.keys())
        registered_children = set(mod._modules.keys())
        for name, value in list(vars(mod).items()):
            if name in {"_parameters", "_buffers", "_modules"}:
                continue
            if name in registered_params or name in registered_buffers or name in registered_children:
                continue
            if not torch.is_tensor(value) or value.device.type != "meta":
                continue
            replacement: torch.Tensor
            if type(mod).__name__ == "TrOCRSinusoidalPositionalEmbedding" and name == "weights":
                # Recompute deterministic sinusoidal table instead of zero-filling.
                try:
                    num_pos = int(value.shape[0]) if value.ndim >= 1 else int(getattr(mod, "padding_idx", 0) or 0) + 2048
                    emb_dim = int(getattr(mod, "embedding_dim"))
                    pad_idx = getattr(mod, "padding_idx", None)
                    replacement = mod.get_embedding(num_pos, emb_dim, pad_idx).to(device)
                except Exception:
                    replacement = torch.zeros(tuple(value.shape), dtype=value.dtype, device=device)
            else:
                replacement = torch.zeros(tuple(value.shape), dtype=value.dtype, device=device)
            setattr(mod, name, replacement)
            fixed.append(f"{prefix}{name}")
        for child_name, child in mod.named_children():
            _walk(child, f"{prefix}{child_name}.")

    _walk(module, "")
    return fixed


def _register_trocr_sinusoidal_weights_as_buffer(module: nn.Module) -> List[str]:
    """Convert TrOCR sinusoidal `weights` helper tensors into registered buffers.

    `TrOCRSinusoidalPositionalEmbedding.weights` is a plain tensor attribute in
    transformers 5.x, so `model.to(device)` and `DataParallel` do not move it.
    Registering it as a non-persistent buffer fixes device placement and keeps
    the original module logic intact (future assignments stay in `_buffers`).
    """
    converted: List[str] = []

    for prefix, mod in module.named_modules():
        if type(mod).__name__ != "TrOCRSinusoidalPositionalEmbedding":
            continue
        if not hasattr(mod, "weights"):
            continue
        weights = getattr(mod, "weights")
        if not torch.is_tensor(weights):
            continue
        # Already a registered buffer in some versions/custom forks.
        if "weights" in getattr(mod, "_buffers", {}):
            continue
        try:
            # Remove plain attribute first; register_buffer rejects existing attrs.
            delattr(mod, "weights")
        except Exception:
            LOGGER.exception("Failed to delete plain weights attr on %s", prefix or "<root>")
            continue
        try:
            mod.register_buffer("weights", weights, persistent=False)
            converted.append(f"{prefix}.weights" if prefix else "weights")
        except Exception:
            LOGGER.exception("Failed to register weights buffer on %s", prefix or "<root>")
            # Best effort restore plain attr to avoid leaving module broken.
            try:
                setattr(mod, "weights", weights)
            except Exception:
                pass
    return converted


def _scan_object_tensors(obj: object, *, max_depth: int = 2, max_items: int = 256) -> List[Dict[str, object]]:
    """Find torch tensors nested in common python containers / object __dict__.

    This is for diagnostics only (processor/tokenizer usually have no tensors).
    """
    found: List[Dict[str, object]] = []
    seen: set[int] = set()

    def _walk(x: object, path: str, depth: int) -> None:
        if len(found) >= max_items:
            return
        oid = id(x)
        if oid in seen:
            return
        seen.add(oid)
        if torch.is_tensor(x):
            t = x
            found.append(
                {
                    "path": path,
                    "device": str(t.device),
                    "shape": tuple(int(d) for d in t.shape),
                    "dtype": str(t.dtype),
                }
            )
            return
        if depth >= max_depth:
            return
        if isinstance(x, dict):
            for k, v in list(x.items())[:max_items]:
                _walk(v, f"{path}.{k}" if path else str(k), depth + 1)
            return
        if isinstance(x, (list, tuple)):
            for i, v in enumerate(list(x)[:max_items]):
                _walk(v, f"{path}[{i}]", depth + 1)
            return
        if hasattr(x, "__dict__"):
            try:
                items = list(vars(x).items())[:max_items]
            except Exception:
                return
            for k, v in items:
                _walk(v, f"{path}.{k}" if path else str(k), depth + 1)

    _walk(obj, "", 0)
    return found


def _describe_batch_like(data: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in data.items():
        if torch.is_tensor(value):
            t = value
            out[key] = {
                "kind": "tensor",
                "device": str(t.device),
                "shape": tuple(int(d) for d in t.shape),
                "dtype": str(t.dtype),
            }
        elif isinstance(value, list):
            out[key] = {
                "kind": "list",
                "len": len(value),
                "elem_type": type(value[0]).__name__ if value else "n/a",
            }
        else:
            out[key] = {"kind": type(value).__name__}
    return out


def _log_pretrain_device_report(
    *,
    train_dataset: "OCRManifestDataset",
    collator: "OCRDataCollator",
    image_processor: object,
    tokenizer: object,
    model: nn.Module,
    output_dir: Path,
) -> None:
    report: Dict[str, object] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "torch_cuda_current_device": (int(torch.cuda.current_device()) if torch.cuda.is_available() else None),
        "train_dataset_len": int(len(train_dataset)),
    }

    sample_report: Dict[str, object] = {}
    batch_report: Dict[str, object] = {}
    try:
        sample0 = train_dataset[0]
        sample_report = _describe_batch_like(sample0)
        sample_feats = [sample0]
        if len(train_dataset) > 1:
            sample_feats.append(train_dataset[1])
        collated = collator(sample_feats)
        batch_report = _describe_batch_like(collated)
    except Exception as exc:
        sample_report = {"error": f"{type(exc).__name__}: {exc}"}
        batch_report = {"error": f"{type(exc).__name__}: {exc}"}
        LOGGER.exception("Failed to build sample/collated batch for device report")

    proc_tensors = _scan_object_tensors(image_processor)
    tok_tensors = _scan_object_tensors(tokenizer)
    model_dev = _summarize_module_devices(model)
    model_plain_tensors = _find_plain_tensor_attrs(model)
    model_plain_meta = [x for x in model_plain_tensors if x.get("device") == "meta"]

    report["train_dataset_sample0"] = sample_report
    report["collated_batch_probe"] = batch_report
    report["image_processor_tensors"] = proc_tensors
    report["tokenizer_tensors"] = tok_tensors
    report["model_device_summary"] = {
        "param_counts_by_device": model_dev["param_counts_by_device"],
        "param_numel_by_device": model_dev["param_numel_by_device"],
        "buffer_counts_by_device": model_dev["buffer_counts_by_device"],
        "buffer_numel_by_device": model_dev["buffer_numel_by_device"],
        "meta_params_count": len(model_dev["meta_params"]),
        "meta_buffers_count": len(model_dev["meta_buffers"]),
    }
    report["model_meta_params"] = model_dev["meta_params"]
    report["model_meta_buffers"] = model_dev["meta_buffers"]
    report["model_plain_tensor_attrs"] = model_plain_tensors
    report["model_plain_meta_tensor_attrs"] = model_plain_meta

    report_path = output_dir / "device_report_pretrain.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info("Pre-train device report written to %s", report_path)
    LOGGER.info("Device report dataset sample0: %s", json.dumps(sample_report, ensure_ascii=False))
    LOGGER.info("Device report collated batch probe: %s", json.dumps(batch_report, ensure_ascii=False))
    if proc_tensors:
        LOGGER.info("Image processor tensor devices: %s", json.dumps(proc_tensors, ensure_ascii=False))
    else:
        LOGGER.info("Image processor tensor devices: none detected (config-only object)")
    if tok_tensors:
        LOGGER.info("Tokenizer tensor devices: %s", json.dumps(tok_tensors, ensure_ascii=False))
    else:
        LOGGER.info("Tokenizer tensor devices: none detected (config-only object)")
    LOGGER.info(
        "Model device summary: params=%s buffers=%s meta_params=%d meta_buffers=%d",
        json.dumps(model_dev["param_counts_by_device"], ensure_ascii=False),
        json.dumps(model_dev["buffer_counts_by_device"], ensure_ascii=False),
        len(model_dev["meta_params"]),
        len(model_dev["meta_buffers"]),
    )
    if model_dev["meta_params"]:
        LOGGER.warning("Model meta params: %s", ", ".join(model_dev["meta_params"]))
    if model_dev["meta_buffers"]:
        LOGGER.warning("Model meta buffers: %s", ", ".join(model_dev["meta_buffers"]))
    if model_plain_meta:
        LOGGER.warning("Model plain meta tensor attrs: %s", json.dumps(model_plain_meta, ensure_ascii=False))


def _drop_encoder_pooler_if_meta(model: VisionEncoderDecoderModel) -> None:
    encoder = getattr(model, "encoder", None)
    if encoder is None or not hasattr(encoder, "pooler"):
        return
    meta_names = [name for name in _named_meta_tensors(model) if name.startswith("encoder.pooler.")]
    if not meta_names:
        return
    LOGGER.warning(
        "Dropping encoder.pooler because meta tensors remain after load: %s",
        ", ".join(meta_names),
    )
    try:
        encoder.pooler = None
    except Exception:
        LOGGER.exception("Failed to drop encoder.pooler")
        return
    enc_cfg = getattr(encoder, "config", None)
    if enc_cfg is not None and hasattr(enc_cfg, "add_pooling_layer"):
        try:
            enc_cfg.add_pooling_layer = False
        except Exception:
            pass


def _load_ved_model_robust(model_name_or_path: str) -> VisionEncoderDecoderModel:
    """Load VisionEncoderDecoderModel robustly across transformers versions.

    On newer transformers builds, meta/lazy initialization can leave a few tensors
    on the `meta` device (commonly an unused encoder pooler), which later crashes
    when Trainer moves the model to CUDA. We explicitly disable low-memory meta
    loading when supported and drop the unused encoder pooler as a fallback.
    """
    load_kwargs: Dict[str, object] = {}
    try:
        sig = inspect.signature(VisionEncoderDecoderModel.from_pretrained)
        if "low_cpu_mem_usage" in sig.parameters:
            load_kwargs["low_cpu_mem_usage"] = False
    except Exception:
        # Signature inspection can fail on some wrappers; just try the default call.
        pass

    try:
        model = VisionEncoderDecoderModel.from_pretrained(model_name_or_path, **load_kwargs)
    except TypeError:
        # Older/newer variants may not accept the kwarg despite signature quirks.
        model = VisionEncoderDecoderModel.from_pretrained(model_name_or_path)

    meta_before = _named_meta_tensors(model)
    if meta_before:
        LOGGER.warning(
            "Model contains meta tensors after load (%d): %s",
            len(meta_before),
            ", ".join(meta_before[:20]) + (" ..." if len(meta_before) > 20 else ""),
        )
        _drop_encoder_pooler_if_meta(model)
    plain_fixed = _materialize_meta_tensor_attrs_inplace(model, device="cpu")
    if plain_fixed:
        LOGGER.warning(
            "Materialized meta plain tensor attrs after load (%d): %s",
            len(plain_fixed),
            ", ".join(plain_fixed[:20]) + (" ..." if len(plain_fixed) > 20 else ""),
        )
    registered_plain = _register_trocr_sinusoidal_weights_as_buffer(model)
    if registered_plain:
        LOGGER.warning(
            "Registered TrOCR sinusoidal plain tensor attrs as buffers after load (%d): %s",
            len(registered_plain),
            ", ".join(registered_plain[:20]) + (" ..." if len(registered_plain) > 20 else ""),
        )
    meta_after = _named_meta_tensors(model)
    if meta_after:
        raise RuntimeError(
            "Model still contains meta-device tensors after loading: "
            + ", ".join(meta_after[:20])
            + (" ..." if len(meta_after) > 20 else "")
            + ". This usually indicates a transformers loading bug/version mismatch. "
            "Try a different checkpoint or transformers version, or patch model initialization."
        )
    return model


def _parse_csv_tokens(spec: str) -> List[str]:
    return [tok.strip() for tok in str(spec).split(",") if tok.strip()]


def _ensure_spiece_alias_if_needed(tokenizer_dir: Path) -> Optional[Path]:
    """Create `spiece.model` alias for Albert tokenizers if repo ships `sentencepiece.model`.

    Some repos (including OpenPecha BoSentencePiece) store the SentencePiece model
    as `sentencepiece.model`, while `AlbertTokenizer` often looks for
    `spiece.model`. Creating an alias avoids loading a degenerate tokenizer with
    only special tokens.
    """
    try:
        td = Path(tokenizer_dir).expanduser().resolve()
    except Exception:
        td = Path(tokenizer_dir)
    if not td.exists() or not td.is_dir():
        return None
    sentencepiece_model = td / "sentencepiece.model"
    spiece_model = td / "spiece.model"
    if not sentencepiece_model.exists() or spiece_model.exists():
        return spiece_model if spiece_model.exists() else None
    try:
        try:
            spiece_model.symlink_to(sentencepiece_model.name)
            LOGGER.info("Created symlink tokenizer alias: %s -> %s", spiece_model, sentencepiece_model.name)
        except Exception:
            shutil.copy2(sentencepiece_model, spiece_model)
            LOGGER.info("Copied tokenizer alias: %s <- %s", spiece_model, sentencepiece_model.name)
        return spiece_model
    except Exception as exc:
        LOGGER.warning("Could not create spiece.model alias in %s: %s", td, exc)
        return None


def _maybe_use_local_bosentencepiece(base_path: str) -> str:
    spec = str(base_path or "").strip()
    if spec != "openpecha/BoSentencePiece":
        return spec
    local = (REPO_ROOT / "ext" / "BoSentencePiece").resolve()
    if local.exists() and local.is_dir():
        LOGGER.info("Using local BoSentencePiece mirror: %s", local)
        _ensure_spiece_alias_if_needed(local)
        return str(local)
    return spec


def _patch_sentencepiece_compat() -> bool:
    """Add legacy SentencePiece camelCase methods expected by some HF tokenizers."""
    try:
        import sentencepiece as spm
    except Exception:
        return False
    cls = getattr(spm, "SentencePieceProcessor", None)
    if cls is None:
        return False
    mappings = {
        "Load": "load",
        "LoadFromSerializedProto": "load_from_serialized_proto",
        "EncodeAsPieces": "encode_as_pieces",
        "EncodeAsIds": "encode_as_ids",
        "SampleEncodeAsPieces": "sample_encode_as_pieces",
        "SampleEncodeAsIds": "sample_encode_as_ids",
        "NBestEncodeAsPieces": "nbest_encode_as_pieces",
        "NBestEncodeAsIds": "nbest_encode_as_ids",
        "DecodePieces": "decode_pieces",
        "DecodeIds": "decode_ids",
        "PieceToId": "piece_to_id",
        "IdToPiece": "id_to_piece",
        "GetPieceSize": "get_piece_size",
        "SetEncodeExtraOptions": "set_encode_extra_options",
        "SetDecodeExtraOptions": "set_decode_extra_options",
    }
    patched = False
    for legacy, modern in mappings.items():
        if hasattr(cls, legacy) or not hasattr(cls, modern):
            continue
        try:
            setattr(cls, legacy, getattr(cls, modern))
            patched = True
        except Exception:
            continue
    return patched


def _is_degenerate_sp_tokenizer(tok) -> bool:
    try:
        tok_len = int(len(tok))
    except Exception:
        return False
    if tok_len > 32:
        return False
    sp_model = getattr(tok, "sp_model", None)
    return sp_model is None


def _tok_cfg_value_to_str(value) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
    return None


def _try_pretrained_fast_from_tokenizer_json(local_dir: Path):
    try:
        from transformers import PreTrainedTokenizerFast
    except Exception:
        return None
    tok_json = local_dir / "tokenizer.json"
    if not tok_json.exists():
        return None

    kwargs: Dict[str, object] = {}
    cfg_path = local_dir / "tokenizer_config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(cfg, dict):
                for key in ("unk_token", "bos_token", "eos_token", "sep_token", "pad_token", "cls_token", "mask_token"):
                    sval = _tok_cfg_value_to_str(cfg.get(key))
                    if sval:
                        kwargs[key] = sval
                addl = cfg.get("additional_special_tokens")
                if isinstance(addl, list):
                    addl_vals = [s for s in (_tok_cfg_value_to_str(x) for x in addl) if s]
                    if addl_vals:
                        kwargs["additional_special_tokens"] = addl_vals
                mml = cfg.get("model_max_length")
                if isinstance(mml, (int, float)) and int(mml) > 0:
                    kwargs["model_max_length"] = int(mml)
        except Exception:
            pass

    try:
        tok_fast = PreTrainedTokenizerFast(tokenizer_file=str(tok_json), **kwargs)
    except Exception:
        return None
    try:
        if int(len(tok_fast)) > 32:
            LOGGER.warning(
                "Recovered BoSentencePiece via PreTrainedTokenizerFast(tokenizer.json) fallback (len=%d) from %s",
                int(len(tok_fast)),
                local_dir,
            )
            return tok_fast
    except Exception:
        return None
    return None


def _try_albert_fast_from_local_dir(local_dir: Path):
    """Try explicit fast tokenizer load from local files when AutoTokenizer falls back badly."""
    try:
        from transformers import AlbertTokenizerFast
    except Exception:
        return None
    tok_json = local_dir / "tokenizer.json"
    if not tok_json.exists():
        return None
    try:
        tok_fast = AlbertTokenizerFast.from_pretrained(str(local_dir))
    except Exception:
        return None
    try:
        if int(len(tok_fast)) > 32:
            LOGGER.warning(
                "Recovered BoSentencePiece via AlbertTokenizerFast fallback (len=%d) from %s",
                int(len(tok_fast)),
                local_dir,
            )
            return tok_fast
    except Exception:
        return None
    return None


def _load_tokenizer_robust(base_path: str):
    if _patch_sentencepiece_compat():
        LOGGER.info("Applied sentencepiece compatibility shim (camelCase aliases).")
    spec = _maybe_use_local_bosentencepiece(base_path)
    local_dir: Optional[Path] = None
    p = Path(str(spec)).expanduser()
    if p.exists() and p.is_dir():
        local_dir = p.resolve()
        _ensure_spiece_alias_if_needed(local_dir)
        if sp_find_model_path(local_dir) is not None:
            tok = load_sentencepiece_tokenizer_adapter(local_dir)
            LOGGER.info(
                "Loaded tokenizer directly via sentencepiece adapter from %s (len=%d)",
                local_dir,
                int(len(tok)),
            )
            return tok

    if str(spec).strip() == "openpecha/BoSentencePiece":
        raise RuntimeError(
            "BoSentencePiece must be downloaded locally for direct sentencepiece loading. "
            "Run `python cli.py download-bosentencepiece-tokenizer` (or "
            "`python scripts/download_bosentencepiece_tokenizer.py`) and then use "
            "`--tokenizer_path ./ext/BoSentencePiece`."
        )

    tokenizer = AutoTokenizer.from_pretrained(spec, use_fast=True)
    if local_dir is not None and _is_degenerate_sp_tokenizer(tokenizer):
        tok_fast = _try_albert_fast_from_local_dir(local_dir)
        if tok_fast is not None:
            return tok_fast
        tok_ptf = _try_pretrained_fast_from_tokenizer_json(local_dir)
        if tok_ptf is not None:
            return tok_ptf

    # Fail fast on the known degenerate ALBERT fallback (only special tokens).
    tok_len = int(len(tokenizer))
    if tok_len <= 32:
        sp_model = getattr(tokenizer, "sp_model", None)
        if sp_model is None:
            hint = ""
            if local_dir is not None:
                hint = (
                    f" Local tokenizer dir={local_dir}. If it contains `sentencepiece.model`, "
                    "ensure `spiece.model` exists (the loader will try to create it automatically)."
                )
            raise RuntimeError(
                "Tokenizer loaded with suspiciously tiny vocab "
                f"(len={tok_len}, class={tokenizer.__class__.__name__}). "
                "This usually means the SentencePiece model was not loaded and only ALBERT special tokens are present."
                + hint
                + " To fix BoSentencePiece locally, run from the repo root: "
                "`python cli.py download-bosentencepiece-tokenizer` "
                "(or `python scripts/download_bosentencepiece_tokenizer.py`). "
                "Then set `--tokenizer_path ./ext/BoSentencePiece`. "
                "Also verify your environment has a working `sentencepiece` install "
                "(e.g. `python -c \"import sentencepiece; print(sentencepiece.__version__)\"`)."
            )
    return tokenizer


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


def _first_nonempty_str(row: Dict[str, object], keys: Sequence[str]) -> str:
    for key in keys:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _resolve_manifest_image_path(image_value: str, manifest_path: Optional[Path]) -> Optional[Path]:
    raw = str(image_value or "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if p.is_absolute():
        rp = p.resolve()
        return rp if rp.exists() else None
    candidates: List[Path] = []
    if manifest_path is not None:
        mp = Path(manifest_path).expanduser().resolve()
        candidates.append((mp.parent / p).resolve())
        candidates.append((mp.parent.parent / p).resolve())
        candidates.append((mp.parent.parent.parent / p).resolve())
    candidates.append((Path.cwd() / p).resolve())
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand
    return None


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
    tokenizer = _load_tokenizer_robust(base_path)
    special_tokens = _parse_csv_tokens(args.extra_special_tokens)

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
        target_start_token: str = "",
        target_end_token: str = "",
        manifest_path: Optional[Path] = None,
    ):
        self.samples: List[OCRSample] = []
        self.stats: Dict[str, int] = {
            "rows_total": 0,
            "kept": 0,
            "missing_image_field": 0,
            "missing_text_field": 0,
            "image_not_found": 0,
        }
        self.example_row_keys: List[str] = []
        self.manifest_path = Path(manifest_path).expanduser().resolve() if manifest_path is not None else None
        for row in rows:
            self.stats["rows_total"] += 1
            if not self.example_row_keys and isinstance(row, dict):
                self.example_row_keys = sorted(str(k) for k in row.keys())
            image_raw = _first_nonempty_str(
                row,
                [
                    "image",
                    "image_path",
                    "line_path",
                    "line_image",
                    "line_image_path",
                    "image_rel_path",
                    "line_image_rel_path",
                    "src__image",
                    "path",
                ],
            )
            text_raw = _first_nonempty_str(
                row,
                [
                    "text",
                    "transcription",
                    "label",
                    "ocr_text",
                    "line_text",
                    "normalized_text",
                    "content",
                    "src__label",
                    "src__text",
                ],
            )
            if not image_raw:
                self.stats["missing_image_field"] += 1
                continue
            if not text_raw:
                self.stats["missing_text_field"] += 1
                continue
            image_path = _resolve_manifest_image_path(image_raw, self.manifest_path)
            if image_path is None:
                self.stats["image_not_found"] += 1
                continue
            self.samples.append(OCRSample(image_path=image_path, text=text_raw))
            self.stats["kept"] += 1
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_target_length = int(max_target_length)
        self.image_preprocess_pipeline = str(image_preprocess_pipeline or "none")
        self.target_start_token = str(target_start_token or "")
        self.target_end_token = str(target_end_token or "")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        with Image.open(sample.image_path) as img:
            rgb = img.convert("RGB")
        rgb = _apply_image_preprocess_pipeline(rgb, self.image_preprocess_pipeline)
        pixel_values = self.image_processor(images=rgb, return_tensors="pt").pixel_values.squeeze(0)
        target_text = _format_ocr_target_text(
            sample.text,
            start_token=self.target_start_token,
            end_token=self.target_end_token,
        )
        labels = self.tokenizer(
            target_text,
            truncation=True,
            max_length=self.max_target_length,
            add_special_tokens=False,
        )["input_ids"]
        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


class OCRDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._sanity_logged = False

    def __call__(self, features: Sequence[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([item["pixel_values"] for item in features])
        label_inputs = [item["labels"] for item in features]
        padded = self.tokenizer.pad(
            {"input_ids": label_inputs},
            padding=True,
            return_tensors="pt",
        )["input_ids"]
        padded[padded == self.tokenizer.pad_token_id] = -100
        if not self._sanity_logged:
            self._sanity_logged = True
            try:
                total = int(padded.numel())
                masked = int((padded == -100).sum().item())
                non_ignored_per_row = (padded != -100).sum(dim=1).tolist()
                sample0 = padded[0].detach().cpu().clone()
                pad_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)
                sample0[sample0 == -100] = pad_id
                sample0_text = self.tokenizer.decode(sample0.tolist(), skip_special_tokens=False)
                LOGGER.info(
                    "label_sanity batch0 | masked_ratio=%.4f non_ignored[min=%d max=%d mean=%.1f] sample0_decoded_head=%r",
                    (masked / total) if total > 0 else 0.0,
                    int(min(non_ignored_per_row) if non_ignored_per_row else 0),
                    int(max(non_ignored_per_row) if non_ignored_per_row else 0),
                    float(sum(non_ignored_per_row) / max(1, len(non_ignored_per_row))),
                    str(sample0_text)[:200],
                )
            except Exception as exc:
                LOGGER.warning("Could not log label_sanity batch0: %s", exc)
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
    out = str(text or "")
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    out = out.replace("<NL>", "\n")
    for ch in _ZERO_WIDTH_CHARS:
        out = out.replace(ch, "")
    out = unicodedata.normalize("NFC", out)
    # Normalize horizontal whitespace while preserving line breaks.
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r" *\n *", "\n", out)
    _ = newline_token  # reserved for future metric variants
    return out.strip()


def _decode_for_metric(tokenizer, sequences, *, newline_token: str) -> List[str]:
    texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    out: List[str] = []
    for t in texts:
        cleaned = _strip_special_token_strings(str(t or ""), tokenizer)
        out.append(_normalize_for_metric(cleaned, newline_token))
    return out


def _sample_cer(pred: str, ref: str) -> float:
    denom = max(1, len(ref))
    return float(_levenshtein(pred, ref) / denom)


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
    val_eval_max_samples = max(0, int(getattr(args, "val_eval_max_samples", 0) or 0))
    if val_rows and val_eval_max_samples > 0 and len(val_rows) > val_eval_max_samples:
        original_val_rows = len(val_rows)
        rng = np.random.default_rng(int(args.seed))
        keep_idx = np.sort(rng.choice(len(val_rows), size=val_eval_max_samples, replace=False))
        val_rows = [val_rows[int(i)] for i in keep_idx.tolist()]
        LOGGER.info(
            "Validation rows subsampled for eval/CER: selected=%d of original=%d (seed=%d)",
            len(val_rows),
            original_val_rows,
            int(args.seed),
        )
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

    model = _load_ved_model_robust(args.model_name_or_path)
    model.decoder.resize_token_embeddings(len(tokenizer))
    fixed_meta_buffers = _materialize_meta_buffers_inplace(model, device="cpu")
    if fixed_meta_buffers:
        LOGGER.warning(
            "Materialized meta buffers after decoder resize (%d): %s",
            len(fixed_meta_buffers),
            ", ".join(fixed_meta_buffers[:20]) + (" ..." if len(fixed_meta_buffers) > 20 else ""),
        )
    fixed_plain_attrs = _materialize_meta_tensor_attrs_inplace(model, device="cpu")
    if fixed_plain_attrs:
        LOGGER.warning(
            "Materialized meta plain tensor attrs after decoder resize (%d): %s",
            len(fixed_plain_attrs),
            ", ".join(fixed_plain_attrs[:20]) + (" ..." if len(fixed_plain_attrs) > 20 else ""),
        )
    registered_plain_after_resize = _register_trocr_sinusoidal_weights_as_buffer(model)
    if registered_plain_after_resize:
        LOGGER.warning(
            "Registered TrOCR sinusoidal plain tensor attrs as buffers after decoder resize (%d): %s",
            len(registered_plain_after_resize),
            ", ".join(registered_plain_after_resize[:20]) + (" ..." if len(registered_plain_after_resize) > 20 else ""),
        )
    _drop_encoder_pooler_if_meta(model)
    meta_after_resize = _named_meta_tensors(model)
    if meta_after_resize:
        raise RuntimeError(
            "Model contains meta tensors after tokenizer resize: "
            + ", ".join(meta_after_resize[:20])
            + (" ..." if len(meta_after_resize) > 20 else "")
            + "."
        )
    plain_meta_after_resize = [x for x in _find_plain_tensor_attrs(model) if x.get("device") == "meta"]
    if plain_meta_after_resize:
        raise RuntimeError(
            "Model contains plain meta tensor attrs after tokenizer resize: "
            + json.dumps(plain_meta_after_resize[:20], ensure_ascii=False)
            + (" ..." if len(plain_meta_after_resize) > 20 else "")
        )

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
    task_end_token = _paired_end_token_for_start(str(args.decoder_start_token))
    task_end_id = None
    if task_end_token:
        try:
            _candidate = tokenizer.convert_tokens_to_ids(task_end_token)
            if _candidate is not None and int(_candidate) >= 0:
                if tokenizer.unk_token_id is None or int(_candidate) != int(tokenizer.unk_token_id):
                    task_end_id = int(_candidate)
        except Exception:
            task_end_id = None

    effective_eos_id = int(task_end_id) if task_end_id is not None else (
        int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None
    )

    model.config.decoder_start_token_id = int(decoder_start_id)
    model.config.pad_token_id = int(tokenizer.pad_token_id)
    if effective_eos_id is not None:
        model.config.eos_token_id = int(effective_eos_id)
    model.config.vocab_size = model.decoder.config.vocab_size

    model.generation_config.decoder_start_token_id = int(decoder_start_id)
    model.generation_config.pad_token_id = int(tokenizer.pad_token_id)
    if effective_eos_id is not None:
        model.generation_config.eos_token_id = int(effective_eos_id)
    model.generation_config.max_length = int(args.generation_max_length)
    model.generation_config.num_beams = 1
    gen_min_new_tokens = max(0, int(getattr(args, "generation_min_new_tokens", 0) or 0))
    if gen_min_new_tokens > 0:
        model.generation_config.min_new_tokens = int(gen_min_new_tokens)
    else:
        try:
            model.generation_config.min_new_tokens = 0
        except Exception:
            pass
    LOGGER.info(
        "Donut target formatting: start=%r end=%r eos_id=%s tokenizer_eos_id=%s gen_max_len=%d gen_min_new=%d beams=%d",
        str(args.decoder_start_token),
        task_end_token,
        str(effective_eos_id),
        str(getattr(tokenizer, "eos_token_id", None)),
        int(args.generation_max_length),
        int(gen_min_new_tokens),
        int(getattr(model.generation_config, "num_beams", 1) or 1),
    )

    train_dataset = OCRManifestDataset(
        train_rows,
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_target_length=int(args.max_target_length),
        image_preprocess_pipeline=image_preproc_mode,
        target_start_token=str(args.decoder_start_token),
        target_end_token=str(task_end_token),
        manifest_path=train_manifest,
    )
    val_dataset = OCRManifestDataset(
        val_rows,
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_target_length=int(args.max_target_length),
        image_preprocess_pipeline=image_preproc_mode,
        target_start_token=str(args.decoder_start_token),
        target_end_token=str(task_end_token),
        manifest_path=val_manifest,
    ) if val_rows else None

    LOGGER.info("Train dataset size: %d", len(train_dataset))
    if val_dataset is not None:
        LOGGER.info("Val dataset size: %d", len(val_dataset))
    LOGGER.info("Train manifest parsing stats: %s", json.dumps(train_dataset.stats, ensure_ascii=False))
    if getattr(train_dataset, "example_row_keys", None):
        LOGGER.info("Train manifest example keys: %s", ", ".join(train_dataset.example_row_keys))
    if val_dataset is not None:
        LOGGER.info("Val manifest parsing stats: %s", json.dumps(val_dataset.stats, ensure_ascii=False))
        if getattr(val_dataset, "example_row_keys", None):
            LOGGER.info("Val manifest example keys: %s", ", ".join(val_dataset.example_row_keys))
    if len(train_dataset) <= 0:
        raise RuntimeError(
            "Train dataset resolved to zero samples. Check manifest image/text field names and whether image paths are relative to the manifest directory."
        )

    collator = OCRDataCollator(tokenizer)
    zero_cer_debug_count = {"n": 0}
    high_cer_debug_count = {"n": 0}
    collapse_warn_count = {"n": 0}

    def _compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
        pred_texts = _decode_for_metric(tokenizer, predictions, newline_token=args.metric_newline_token)
        ref_texts = _decode_for_metric(tokenizer, labels, newline_token=args.metric_newline_token)
        pred_norms = [str(p or "") for p in pred_texts]
        ref_norms = [str(r or "") for r in ref_texts]
        valid_pairs: List[Tuple[str, str]] = []
        empty_ref_count = 0
        empty_pred_count = 0
        for p, r in zip(pred_norms, ref_norms):
            if not r:
                empty_ref_count += 1
                continue
            if not p:
                empty_pred_count += 1
            valid_pairs.append((p, r))
        if (
            valid_pairs
            and empty_pred_count >= len(valid_pairs)
            and collapse_warn_count["n"] < 10
        ):
            collapse_warn_count["n"] += 1
            LOGGER.error("=" * 120)
            LOGGER.error(
                "GENERATION COLLAPSE DETECTED: all predictions are empty after decoding on eval set "
                "(valid_n=%d, empty_pred=%d, empty_ref=%d). CER will read as 1.0 (100%%) but is not a useful quality signal.",
                int(len(valid_pairs)),
                int(empty_pred_count),
                int(empty_ref_count),
            )
            LOGGER.error(
                "Typical cause: model generates immediate task-end/EOS (e.g. <s_ocr></s_ocr>). "
                "Try earlier checkpoint, higher --generation_min_new_tokens, and inspect resume probe metrics."
            )
            for i, (pred, ref) in enumerate(valid_pairs[:3], start=1):
                LOGGER.error(
                    "collapse sample %d | REF_len=%d PRD_len=%d | REF=%r | PRD=%r",
                    i,
                    len(ref),
                    len(pred),
                    ref[:180],
                    pred[:180],
                )
            LOGGER.error("=" * 120)
        if valid_pairs:
            cer = _char_error_rate(
                [p for p, _ in valid_pairs],
                [r for _, r in valid_pairs],
                args.metric_newline_token,
            )
        else:
            cer = 1.0
            LOGGER.warning(
                "No valid non-empty references for CER in current eval batch/set (n=%d, empty_ref=%d). Returning CER=1.0 sentinel.",
                len(ref_texts),
                empty_ref_count,
            )
        if cer == 0.0 and zero_cer_debug_count["n"] < 3:
            zero_cer_debug_count["n"] += 1
            exact_matches = sum(int(p == r) for p, r in valid_pairs)
            LOGGER.warning(
                "eval_cer is exactly 0.0 on current eval set (n=%d, valid_n=%d, empty_ref=%d, empty_pred=%d, exact_matches=%d). "
                "This can be real on a tiny/easy sampled val subset, but inspect samples below.",
                len(ref_texts),
                len(valid_pairs),
                empty_ref_count,
                empty_pred_count,
                exact_matches,
            )
            for i, (pred, ref) in enumerate(valid_pairs[:5], start=1):
                LOGGER.warning("CER=0 sample %d | REF: %r", i, ref[:300])
                LOGGER.warning("CER=0 sample %d | PRD: %r", i, pred[:300])
        if cer > 1.0 and high_cer_debug_count["n"] < 3:
            high_cer_debug_count["n"] += 1
            sample_rows = []
            for p, r in valid_pairs[:]:
                sample_rows.append(
                    {
                        "ref_len": len(r),
                        "pred_len": len(p),
                        "sample_cer": round(_sample_cer(p, r), 4),
                        "ref": r[:180],
                        "pred": p[:180],
                    }
                )
            sample_rows = sorted(sample_rows, key=lambda x: (x["sample_cer"], x["pred_len"]), reverse=True)
            LOGGER.warning(
                "High eval_cer detected (%.4f ratio ~= %.1f%%) on current eval set; showing top sample outliers.",
                float(cer),
                float(cer) * 100.0,
            )
            for i, row in enumerate(sample_rows[:5], start=1):
                LOGGER.warning(
                    "high_cer sample %d | cer=%.4f ref_len=%d pred_len=%d | REF=%r | PRD=%r",
                    i,
                    float(row["sample_cer"]),
                    int(row["ref_len"]),
                    int(row["pred_len"]),
                    row["ref"],
                    row["pred"],
                )
        return {
            "cer": float(cer),
            "cer_percent": float(cer) * 100.0,
            "cer_valid_n": int(len(valid_pairs)),
            "cer_empty_ref_count": int(empty_ref_count),
            "cer_empty_pred_count": int(empty_pred_count),
        }

    has_eval = val_dataset is not None and len(val_dataset) > 0
    ta_kwargs = dict(
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
        save_strategy="steps",
        load_best_model_at_end=bool(has_eval),
        metric_for_best_model="cer" if has_eval else None,
        greater_is_better=False if has_eval else None,
    )
    ta_sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    if "evaluation_strategy" in ta_sig.parameters:
        ta_kwargs["evaluation_strategy"] = ("steps" if has_eval else "no")
    elif "eval_strategy" in ta_sig.parameters:
        ta_kwargs["eval_strategy"] = ("steps" if has_eval else "no")
    training_args = Seq2SeqTrainingArguments(**ta_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if has_eval else None,
        data_collator=collator,
        compute_metrics=_compute_metrics if has_eval else None,
    )
    trainer_sig = inspect.signature(Seq2SeqTrainer.__init__)
    if "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_sig.parameters:
        # Newer Transformers versions replaced `tokenizer=` with `processing_class=`.
        trainer_kwargs["processing_class"] = tokenizer
    debug_train_decode_preview = bool(getattr(args, "enable_train_decode_preview", False))
    debug_train_decode_every_steps = int(getattr(args, "debug_train_decode_every_steps", 100) or 100)
    debug_train_trace = bool(getattr(args, "debug_train_trace", False))
    debug_train_trace_every_steps = int(getattr(args, "debug_train_trace_every_steps", 1) or 1)
    debug_train_trace_topk = int(getattr(args, "debug_train_trace_topk", 5) or 5)
    debug_train_trace_max_positions = int(getattr(args, "debug_train_trace_max_positions", 8) or 8)
    if (not debug_train_decode_preview) or debug_train_decode_every_steps > 1:
        LOGGER.info(
            "DONUT train decode preview: enabled=%s every_steps=%d",
            bool(debug_train_decode_preview),
            int(debug_train_decode_every_steps),
        )
    if debug_train_trace:
        LOGGER.warning(
            "Advanced DONUT train trace enabled: every_steps=%d topk=%d max_positions=%d (high log volume / slower training).",
            debug_train_trace_every_steps,
            debug_train_trace_topk,
            debug_train_trace_max_positions,
        )

    trainer = DonutDebugSeq2SeqTrainer(
        **trainer_kwargs,
        tokenizer_for_debug=tokenizer,
        metric_newline_token=str(args.metric_newline_token),
        debug_train_decode_preview=debug_train_decode_preview,
        debug_train_decode_every_steps=debug_train_decode_every_steps,
        debug_train_trace=debug_train_trace,
        debug_train_trace_every_steps=debug_train_trace_every_steps,
        debug_train_trace_topk=debug_train_trace_topk,
        debug_train_trace_max_positions=debug_train_trace_max_positions,
    )
    try:
        trainer.remove_callback(ProgressCallback)
        trainer.add_callback(DonutProgressCallback())
        trainer.add_callback(DonutCheckpointAliasCallback())
        LOGGER.info("Enabled DONUT TQDM postfix metrics callback")
    except Exception as exc:
        LOGGER.warning("Could not replace default ProgressCallback: %s", exc)

    _log_pretrain_device_report(
        train_dataset=train_dataset,
        collator=collator,
        image_processor=image_processor,
        tokenizer=tokenizer,
        model=model,
        output_dir=output_dir,
    )

    resume_ckpt = str(getattr(args, "resume_from_checkpoint", "") or "").strip()
    resume_probe_n = max(0, int(getattr(args, "resume_probe_eval_samples", 5) or 0))
    if resume_ckpt and has_eval and resume_probe_n > 0:
        try:
            ckpt_path = Path(resume_ckpt).expanduser().resolve()
            if not ckpt_path.exists():
                LOGGER.warning("resume checkpoint probe skipped: checkpoint not found: %s", ckpt_path)
            elif not hasattr(trainer, "_load_from_checkpoint"):
                LOGGER.warning("resume checkpoint probe skipped: trainer has no _load_from_checkpoint() in this transformers version")
            else:
                LOGGER.info(
                    "Running resume checkpoint probe before training: checkpoint=%s n_val_samples=%d",
                    ckpt_path,
                    int(resume_probe_n),
                )
                trainer._load_from_checkpoint(str(ckpt_path))  # type: ignore[attr-defined]
                probe_count = min(int(resume_probe_n), len(val_dataset) if val_dataset is not None else 0)
                if probe_count > 0 and val_dataset is not None:
                    probe_ds = Subset(val_dataset, list(range(probe_count)))
                    probe_metrics = trainer.evaluate(eval_dataset=probe_ds, metric_key_prefix="resume_probe")
                    LOGGER.warning("Resume checkpoint probe metrics: %s", json.dumps(probe_metrics, ensure_ascii=False, default=str))
                    try:
                        valid_n = int(probe_metrics.get("resume_probe_cer_valid_n", 0) or 0)
                        empty_pred_n = int(probe_metrics.get("resume_probe_cer_empty_pred_count", 0) or 0)
                        if (
                            valid_n > 0
                            and empty_pred_n >= valid_n
                            and not bool(getattr(args, "allow_resume_probe_empty_pred", False))
                        ):
                            raise RuntimeError(
                                "Resume checkpoint probe detected generation collapse: "
                                f"empty predictions for all {empty_pred_n}/{valid_n} probe samples. "
                                "Choose an earlier checkpoint or override with --allow_resume_probe_empty_pred."
                            )
                    except RuntimeError:
                        raise
                    except Exception as exc:
                        LOGGER.warning("Could not validate resume checkpoint probe metrics: %s", exc)
                else:
                    LOGGER.warning("resume checkpoint probe skipped: no validation samples available")
        except Exception as exc:
            if isinstance(exc, RuntimeError) and "generation collapse" in str(exc):
                raise
            LOGGER.warning("Resume checkpoint probe failed (continuing with training): %s", exc)

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
