from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import inspect

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoImageProcessor, VisionEncoderDecoderModel


def _dist_is_initialized() -> bool:
    try:
        return bool(torch.distributed.is_available() and torch.distributed.is_initialized())
    except Exception:
        return False


def _dist_rank() -> int:
    if not _dist_is_initialized():
        return 0
    try:
        return int(torch.distributed.get_rank())
    except Exception:
        return 0


def _dist_world_size() -> int:
    if not _dist_is_initialized():
        return 1
    try:
        return int(torch.distributed.get_world_size())
    except Exception:
        return 1


def _dist_barrier() -> None:
    if not _dist_is_initialized():
        return
    try:
        torch.distributed.barrier()
    except Exception:
        pass


def _broadcast_obj(obj: Any, src: int = 0) -> Any:
    if not _dist_is_initialized():
        return obj
    bucket = [obj if _dist_rank() == src else None]
    try:
        torch.distributed.broadcast_object_list(bucket, src=src)
    except Exception:
        return obj
    return bucket[0]


def _all_gather_objects(local: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not _dist_is_initialized():
        return list(local)
    world = _dist_world_size()
    gathered: List[Optional[List[Dict[str, Any]]]] = [None for _ in range(world)]
    try:
        torch.distributed.all_gather_object(gathered, list(local))
    except Exception:
        return list(local)
    out: List[Dict[str, Any]] = []
    for part in gathered:
        if not part:
            continue
        out.extend(part)
    return out


def _all_reduce_sum_pair(edit_sum: int, ref_sum: int, device: torch.device) -> Tuple[int, int]:
    if not _dist_is_initialized():
        return int(edit_sum), int(ref_sum)
    t = torch.tensor([int(edit_sum), int(ref_sum)], dtype=torch.long, device=device)
    try:
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    except Exception:
        return int(edit_sum), int(ref_sum)
    return int(t[0].item()), int(t[1].item())


def _safe_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_jsonable(v) for v in obj]
    try:
        return str(obj)
    except Exception:
        return None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        except Exception:
            state["torch_cuda_rng_state_all"] = None
    else:
        state["torch_cuda_rng_state_all"] = None
    return state


@dataclass
class FrozenValRecord:
    idx: int
    sample_id: str
    image_path: str
    image_sha256: str
    gt_text_raw: str
    gt_text_metric_raw: str
    gt_text_metric_norm: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "idx": int(self.idx),
            "id": str(self.sample_id),
            "image_path": str(self.image_path),
            "image_sha256": str(self.image_sha256),
            "gt_text_raw": str(self.gt_text_raw),
            "gt_text_metric_raw": str(self.gt_text_metric_raw),
            "gt_text_metric_norm": str(self.gt_text_metric_norm),
        }

    @classmethod
    def from_json(cls, row: Dict[str, Any]) -> "FrozenValRecord":
        return cls(
            idx=int(row["idx"]),
            sample_id=str(row["id"]),
            image_path=str(row["image_path"]),
            image_sha256=str(row.get("image_sha256", "")),
            gt_text_raw=str(row.get("gt_text_raw", "")),
            gt_text_metric_raw=str(row.get("gt_text_metric_raw", "")),
            gt_text_metric_norm=str(row.get("gt_text_metric_norm", "")),
        )


class FrozenValSubsetDataset(Dataset):
    def __init__(
        self,
        records: Sequence[FrozenValRecord],
        *,
        image_processor,
        image_preprocess_fn: Callable[[Image.Image, str], Image.Image],
        image_preprocess_pipeline: str,
    ):
        self.records = list(records)
        self.image_processor = image_processor
        self.image_preprocess_fn = image_preprocess_fn
        self.image_preprocess_pipeline = str(image_preprocess_pipeline or "none")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        p = Path(rec.image_path)
        with Image.open(p) as img:
            rgb = img.convert("RGB")
        rgb = self.image_preprocess_fn(rgb, self.image_preprocess_pipeline)
        pixel_values = self.image_processor(images=rgb, return_tensors="pt").pixel_values.squeeze(0)
        return {
            "pixel_values": pixel_values,
            "meta": {
                "idx": int(rec.idx),
                "id": str(rec.sample_id),
                "image_path": str(rec.image_path),
                "image_sha256": str(rec.image_sha256),
                "gt_text_raw": str(rec.gt_text_raw),
                "gt_text_metric_raw": str(rec.gt_text_metric_raw),
                "gt_text_metric_norm": str(rec.gt_text_metric_norm),
            },
        }


def frozen_val_collate(features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "pixel_values": torch.stack([f["pixel_values"] for f in features]),
        "meta": [f["meta"] for f in features],
    }


class ReproPack:
    """Checkpoint repro bundle helper.

    This wraps the missing pieces not covered by HF Trainer checkpoints:
    exact eval subset IDs/data, generation contract, text-normalization contract,
    RNG/DDP metadata, and per-checkpoint prediction dumps.
    """

    def __init__(
        self,
        *,
        args,
        tokenizer,
        image_processor,
        image_preprocess_pipeline: str,
        val_dataset,
        decode_for_metric_fn: Callable[..., List[str]],
        normalize_for_metric_fn: Callable[[str, str], str],
        levenshtein_fn: Callable[[str, str], int],
        image_preprocess_fn: Callable[[Image.Image, str], Image.Image],
        format_target_text_fn: Callable[..., str],
        encode_target_ids_fn: Callable[..., List[int]],
        prepare_model_for_generate_fn: Optional[Callable[[Any], None]],
        logger,
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_preprocess_pipeline = str(image_preprocess_pipeline or "none")
        self.val_dataset = val_dataset
        self._decode_for_metric_fn = decode_for_metric_fn
        self._normalize_for_metric_fn = normalize_for_metric_fn
        self._levenshtein_fn = levenshtein_fn
        self._image_preprocess_fn = image_preprocess_fn
        self._format_target_text_fn = format_target_text_fn
        self._encode_target_ids_fn = encode_target_ids_fn
        self._prepare_model_for_generate_fn = prepare_model_for_generate_fn
        self._logger = logger
        self._frozen_records: List[FrozenValRecord] = []
        self._subset_frozen = False

    @staticmethod
    def capture_rng_state() -> Dict[str, Any]:
        return capture_rng_state()

    def _make_sample_id(self, idx: int, image_path: Path) -> str:
        raw = f"{idx}:{str(image_path.resolve())}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _encode_ref_for_metric(self, raw_text: str) -> Tuple[str, str]:
        start_tok = str(getattr(self.val_dataset, "target_start_token", "") or "")
        end_tok = str(getattr(self.val_dataset, "target_end_token", "") or "")
        target_text = self._format_target_text_fn(raw_text, start_token=start_tok, end_token=end_tok)
        max_target_length = int(getattr(self.val_dataset, "max_target_length", 0) or 0)
        end_tok_id = getattr(self.val_dataset, "target_end_token_id", None)
        if end_tok_id is None and end_tok:
            try:
                ids_tmp = self.tokenizer(end_tok, truncation=False, add_special_tokens=False).get("input_ids", [])
                if isinstance(ids_tmp, list) and len(ids_tmp) == 1:
                    end_tok_id = int(ids_tmp[0])
            except Exception:
                end_tok_id = None

        # Support both old and new helper signatures.
        try:
            sig = inspect.signature(self._encode_target_ids_fn)
            params = sig.parameters
        except Exception:
            params = {}
        if "terminal_token_id" in params:
            ids = self._encode_target_ids_fn(
                self.tokenizer,
                target_text,
                max_target_length=max_target_length,
                terminal_token_id=end_tok_id,
            )
        else:
            ids = self._encode_target_ids_fn(
                self.tokenizer,
                target_text,
                max_target_length=max_target_length,
                terminal_token=end_tok,
            )
        metric_raw = self.tokenizer.decode(ids, skip_special_tokens=True)
        metric_norm = self._normalize_for_metric_fn(str(metric_raw or ""), str(getattr(self.args, "metric_newline_token", "<NL>")))
        return str(metric_raw or ""), str(metric_norm or "")

    def freeze_val_subset(self) -> List[FrozenValRecord]:
        if self._subset_frozen:
            return list(self._frozen_records)
        records: List[FrozenValRecord] = []
        if self.val_dataset is None:
            self._frozen_records = []
            self._subset_frozen = True
            return []
        rank = _dist_rank()
        if rank == 0:
            samples = list(getattr(self.val_dataset, "samples", []) or [])
            for i, sample in enumerate(samples):
                img_path = Path(str(getattr(sample, "image_path"))).expanduser().resolve()
                gt_text_raw = str(getattr(sample, "text", "") or "")
                gt_metric_raw, gt_metric_norm = self._encode_ref_for_metric(gt_text_raw)
                try:
                    sha = _sha256_file(img_path)
                except Exception:
                    sha = ""
                records.append(
                    FrozenValRecord(
                        idx=int(i),
                        sample_id=self._make_sample_id(i, img_path),
                        image_path=str(img_path),
                        image_sha256=sha,
                        gt_text_raw=gt_text_raw,
                        gt_text_metric_raw=gt_metric_raw,
                        gt_text_metric_norm=gt_metric_norm,
                    )
                )
        if _dist_is_initialized():
            records_json = [r.to_json() for r in records] if rank == 0 else None
            records_json = _broadcast_obj(records_json, src=0)
            records = [FrozenValRecord.from_json(r) for r in (records_json or [])]
        self._frozen_records = records
        self._subset_frozen = True
        return list(self._frozen_records)

    def _build_frozen_dataset(self, records: Sequence[FrozenValRecord]) -> FrozenValSubsetDataset:
        return FrozenValSubsetDataset(
            records,
            image_processor=self.image_processor,
            image_preprocess_fn=self._image_preprocess_fn,
            image_preprocess_pipeline=self.image_preprocess_pipeline,
        )

    def _effective_generate_cfg(self, model) -> Dict[str, Any]:
        gc = getattr(model, "generation_config", None)
        cfg = {
            "decoder_start_token_id": getattr(gc, "decoder_start_token_id", getattr(getattr(model, "config", None), "decoder_start_token_id", None)),
            "bos_token_id": getattr(gc, "bos_token_id", getattr(getattr(model, "config", None), "bos_token_id", None)),
            "eos_token_id": getattr(gc, "eos_token_id", getattr(getattr(model, "config", None), "eos_token_id", None)),
            "pad_token_id": getattr(gc, "pad_token_id", getattr(getattr(model, "config", None), "pad_token_id", None)),
            "max_length": getattr(gc, "max_length", None),
            "max_new_tokens": getattr(gc, "max_new_tokens", None),
            "min_new_tokens": getattr(gc, "min_new_tokens", None),
            "num_beams": getattr(gc, "num_beams", None),
            "do_sample": getattr(gc, "do_sample", None),
            "temperature": getattr(gc, "temperature", None),
            "top_p": getattr(gc, "top_p", None),
            "repetition_penalty": getattr(gc, "repetition_penalty", None),
            "no_repeat_ngram_size": getattr(gc, "no_repeat_ngram_size", None),
            "length_penalty": getattr(gc, "length_penalty", None),
            "early_stopping": getattr(gc, "early_stopping", None),
            "bad_words_ids": getattr(gc, "bad_words_ids", None),
            "forced_bos_token_id": getattr(gc, "forced_bos_token_id", None),
            "forced_eos_token_id": getattr(gc, "forced_eos_token_id", None),
            "use_cache": getattr(gc, "use_cache", None),
        }
        return {k: _safe_jsonable(v) for k, v in cfg.items()}

    def _generate_kwargs(self, model) -> Dict[str, Any]:
        gc = self._effective_generate_cfg(model)
        kwargs: Dict[str, Any] = {}
        for k in [
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
            "use_cache",
        ]:
            v = gc.get(k)
            if v is not None:
                kwargs[k] = v
        if gc.get("bad_words_ids") is not None:
            kwargs["bad_words_ids"] = gc["bad_words_ids"]
        if gc.get("forced_bos_token_id") is not None:
            kwargs["forced_bos_token_id"] = gc["forced_bos_token_id"]
        if gc.get("forced_eos_token_id") is not None:
            kwargs["forced_eos_token_id"] = gc["forced_eos_token_id"]
        return kwargs

    def evaluate_fixed_subset(self, model, *, batch_size: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        gen_model = model
        if not hasattr(gen_model, "generate") and hasattr(gen_model, "module") and hasattr(gen_model.module, "generate"):
            gen_model = gen_model.module
        if self._prepare_model_for_generate_fn is not None:
            try:
                self._prepare_model_for_generate_fn(gen_model)
            except Exception as exc:
                self._logger.warning("ReproPack: prepare_model_for_generate failed before eval: %s", exc)
        records = self.freeze_val_subset()
        if not records:
            return {"cer": 1.0, "cer_percent": 100.0, "valid_n": 0, "empty_ref_count": 0, "empty_pred_count": 0}, []
        ds = self._build_frozen_dataset(records)
        sampler = None
        if _dist_is_initialized():
            sampler = DistributedSampler(ds, num_replicas=_dist_world_size(), rank=_dist_rank(), shuffle=False, drop_last=False)
        loader = DataLoader(
            ds,
            batch_size=max(1, int(batch_size)),
            shuffle=False if sampler is None else None,
            sampler=sampler,
            collate_fn=frozen_val_collate,
            num_workers=0,
            pin_memory=False,
        )

        was_training = bool(getattr(gen_model, "training", False))
        device = next(gen_model.parameters()).device
        gen_kwargs = self._generate_kwargs(gen_model)
        newline_token = str(getattr(self.args, "metric_newline_token", "<NL>") or "<NL>")
        local_rows: List[Dict[str, Any]] = []
        local_edit_sum = 0
        local_ref_sum = 0
        local_empty_ref = 0
        local_empty_pred = 0
        started = time.time()
        gen_model.eval()
        with torch.no_grad():
            for batch in loader:
                pixel_values = batch["pixel_values"].to(device)
                gen_ids = gen_model.generate(pixel_values=pixel_values, **gen_kwargs)
                pred_raw_list = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                metas = batch["meta"]
                for pred_raw, meta in zip(pred_raw_list, metas):
                    gt_raw = str(meta.get("gt_text_metric_raw", "") or "")
                    gt_norm = str(meta.get("gt_text_metric_norm", "") or "")
                    pred_raw = str(pred_raw or "")
                    pred_norm = self._normalize_for_metric_fn(pred_raw, newline_token)
                    ref_norm = gt_norm
                    if not ref_norm:
                        local_empty_ref += 1
                    if not pred_norm:
                        local_empty_pred += 1
                    if ref_norm:
                        dist = int(self._levenshtein_fn(pred_norm, ref_norm))
                        ref_len = max(1, len(ref_norm))
                        local_edit_sum += dist
                        local_ref_sum += ref_len
                        sample_cer = float(dist / ref_len)
                    else:
                        sample_cer = 1.0
                    local_rows.append(
                        {
                            "idx": int(meta["idx"]),
                            "id": str(meta["id"]),
                            "image_path": str(meta["image_path"]),
                            "image_sha256": str(meta.get("image_sha256", "")),
                            "gt_raw": str(meta.get("gt_text_raw", "")),
                            "gt_metric_raw": gt_raw,
                            "gt_norm": ref_norm,
                            "pred_raw": pred_raw,
                            "pred_norm": pred_norm,
                            "gt_len": int(len(ref_norm)),
                            "pred_len": int(len(pred_norm)),
                            "sample_cer": float(sample_cer),
                        }
                    )
        elapsed = float(time.time() - started)
        all_rows = _all_gather_objects(local_rows)
        all_rows = sorted(all_rows, key=lambda r: int(r.get("idx", 0)))
        edit_sum, ref_sum = _all_reduce_sum_pair(local_edit_sum, local_ref_sum, device=device)
        empty_ref_sum, _ = _all_reduce_sum_pair(local_empty_ref, 0, device=device)
        empty_pred_sum, _ = _all_reduce_sum_pair(local_empty_pred, 0, device=device)
        valid_n = sum(int(bool(str(r.get("gt_norm", "")))) for r in all_rows)
        cer = float(edit_sum / max(1, ref_sum))
        metrics = {
            "cer": float(cer),
            "cer_percent": float(cer) * 100.0,
            "valid_n": int(valid_n),
            "empty_ref_count": int(empty_ref_sum),
            "empty_pred_count": int(empty_pred_sum),
            "edit_sum": int(edit_sum),
            "ref_len_sum": int(ref_sum),
            "runtime": float(elapsed),
            "world_size": int(_dist_world_size()),
        }
        if was_training:
            gen_model.train()
        return metrics, all_rows

    def _normalization_contract(self) -> Dict[str, Any]:
        return {
            "unicode_normalization": "NFC",
            "newline_normalization": "\\r\\n and \\r -> \\n",
            "special_token_cleanup": "tokenizer.batch_decode(skip_special_tokens=True) plus explicit string removal of tokenizer.all_special_tokens",
            "zero_width_removed": ["\\u200b", "\\u200c", "\\u200d", "\\ufeff"],
            "whitespace_normalization": "collapse horizontal spaces/tabs, trim around newlines, strip",
            "metric_newline_token": str(getattr(self.args, "metric_newline_token", "<NL>") or "<NL>"),
        }

    def _normalization_py_source(self) -> str:
        return (
            "import re\n"
            "import unicodedata\n\n"
            "_ZERO_WIDTH = ('\\u200b','\\u200c','\\u200d','\\ufeff')\n\n"
            "def strip_special_token_strings(text, special_tokens):\n"
            "    out = str(text or '')\n"
            "    for tok in sorted([t for t in (special_tokens or []) if isinstance(t, str) and t], key=len, reverse=True):\n"
            "        out = out.replace(tok, '')\n"
            "    return out\n\n"
            "def normalize_for_metric(text, newline_token='<NL>'):\n"
            "    out = str(text or '')\n"
            "    out = out.replace('\\r\\n', '\\n').replace('\\r', '\\n')\n"
            "    out = out.replace('<NL>', '\\n')\n"
            "    for ch in _ZERO_WIDTH:\n"
            "        out = out.replace(ch, '')\n"
            "    out = unicodedata.normalize('NFC', out)\n"
            "    out = re.sub(r'[ \\t]+', ' ', out)\n"
            "    out = re.sub(r' *\\n *', '\\n', out)\n"
            "    _ = newline_token\n"
            "    return out.strip()\n"
        )

    def _image_preprocess_contract(self) -> Dict[str, Any]:
        ip = self.image_processor
        data: Dict[str, Any] = {
            "pipeline": str(self.image_preprocess_pipeline),
            "color_space": "RGB",
            "channel_order": "channels_first",
            "processor_class": str(ip.__class__.__name__),
        }
        try:
            if hasattr(ip, "to_dict"):
                data["image_processor_config"] = _safe_jsonable(ip.to_dict())
        except Exception:
            pass
        for attr in [
            "size",
            "do_resize",
            "do_rescale",
            "do_normalize",
            "resample",
            "image_mean",
            "image_std",
            "do_pad",
        ]:
            if hasattr(ip, attr):
                data[attr] = _safe_jsonable(getattr(ip, attr))
        data["image_size_arg"] = int(getattr(self.args, "image_size", 0) or 0)
        return data

    def _train_state_contract(self, trainer, state) -> Dict[str, Any]:
        scaler = getattr(trainer, "scaler", None)
        if scaler is None:
            try:
                scaler = getattr(getattr(trainer, "accelerator", None), "scaler", None)
            except Exception:
                scaler = None
        return {
            "global_step": int(getattr(state, "global_step", 0) or 0),
            "epoch": float(getattr(state, "epoch", 0.0) or 0.0),
            "gradient_accumulation_steps": int(getattr(trainer.args, "gradient_accumulation_steps", 1) or 1),
            "per_device_train_batch_size": int(getattr(trainer.args, "per_device_train_batch_size", 0) or 0),
            "per_device_eval_batch_size": int(getattr(trainer.args, "per_device_eval_batch_size", 0) or 0),
            "seed": int(getattr(self.args, "seed", 0) or 0),
            "data_seed": int(getattr(trainer.args, "data_seed", getattr(self.args, "seed", 0)) or 0),
            "fp16": bool(getattr(trainer.args, "fp16", False)),
            "bf16": bool(getattr(trainer.args, "bf16", False)),
            "torch_dtype_model_params": str(next(trainer.model.parameters()).dtype),
            "amp_scaler_present": bool(scaler is not None),
        }

    def _ddp_state_contract(self) -> Dict[str, Any]:
        return {
            "world_size": int(_dist_world_size()),
            "rank": int(_dist_rank()),
            "local_rank": int(os.environ.get("LOCAL_RANK", "0") or 0),
            "seed_base": int(getattr(self.args, "seed", 0) or 0),
            "dataloader_seed": int(getattr(self.args, "seed", 0) or 0),
            "sampler_epoch": None,
        }

    def save_checkpoint_bundle(
        self,
        ckpt_dir: Path,
        *,
        trainer,
        state,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
    ) -> Optional[Dict[str, Any]]:
        ckpt_dir = Path(ckpt_dir).expanduser().resolve()
        if not ckpt_dir.exists():
            return None
        records = self.freeze_val_subset()
        repro_dir = ckpt_dir / "repro"
        is_rank0 = (_dist_rank() == 0)
        if is_rank0:
            repro_dir.mkdir(parents=True, exist_ok=True)
        _dist_barrier()

        # Save tokenizer + image processor under repro/ to make the checkpoint self-contained.
        if is_rank0:
            try:
                self.tokenizer.save_pretrained(str(repro_dir / "tokenizer"))
            except Exception as exc:
                self._logger.warning("ReproPack: could not save tokenizer into %s: %s", repro_dir / "tokenizer", exc)
            try:
                self.image_processor.save_pretrained(str(repro_dir / "image_processor"))
            except Exception as exc:
                self._logger.warning("ReproPack: could not save image_processor into %s: %s", repro_dir / "image_processor", exc)

        generate_cfg = self._effective_generate_cfg(model)
        if is_rank0:
            _write_json(repro_dir / "generate_config.json", generate_cfg)
            _write_json(
                repro_dir / "metrics.json",
                {
                    "metric_name": "cer",
                    "definition": "sum(edit_distance(pred_norm, ref_norm)) / sum(max(1, len(ref_norm)))",
                    "aggregation": "global sums (DDP all_reduce)",
                    "empty_reference_policy": "references with empty normalized text are excluded from valid_n and counted in empty_ref_count",
                },
            )
            _write_json(repro_dir / "text_normalization.json", self._normalization_contract())
            (repro_dir / "text_normalization.py").write_text(self._normalization_py_source(), encoding="utf-8")
            _write_json(repro_dir / "image_preprocess.json", self._image_preprocess_contract())
            _write_json(repro_dir / "train_state.json", self._train_state_contract(trainer, state))
            _write_json(repro_dir / "ddp_state.json", self._ddp_state_contract())
            _write_jsonl(repro_dir / "val_subset.jsonl", [r.to_json() for r in records])

        rng_state = capture_rng_state()
        if is_rank0:
            torch.save(rng_state, repro_dir / "rng_state_rank0.pt")
            rng_summary = {
                "python_state_type": str(type(rng_state.get("python_random_state")).__name__),
                "numpy_state_algo": str((rng_state.get("numpy_random_state") or [None])[0]),
                "torch_cpu_rng_numel": int(len(rng_state.get("torch_cpu_rng_state", []))),
                "torch_cuda_rng_state_count": int(len(rng_state.get("torch_cuda_rng_state_all") or [])),
            }
            _write_json(repro_dir / "rng_state_rank0_summary.json", rng_summary)

        # Optional duplicate snapshots (HF already saves optimizer/scheduler in checkpoint root).
        if is_rank0:
            if optimizer is not None:
                try:
                    torch.save(optimizer.state_dict(), repro_dir / "optimizer_state_rank0.pt")
                except Exception:
                    pass
            if scheduler is not None:
                try:
                    torch.save(scheduler.state_dict(), repro_dir / "scheduler_state_rank0.pt")
                except Exception:
                    pass
            if scaler is not None:
                try:
                    torch.save(scaler.state_dict(), repro_dir / "scaler_state_rank0.pt")
                except Exception:
                    pass

        # DDP-safe CER and pred dump over the exact frozen subset.
        eval_metrics, pred_rows = self.evaluate_fixed_subset(model, batch_size=int(getattr(trainer.args, "per_device_eval_batch_size", 1) or 1))
        step = int(getattr(state, "global_step", 0) or 0)
        if is_rank0:
            cer_payload = {
                "step": int(step),
                "checkpoint_dir": str(ckpt_dir),
                **{f"repro_{k}": _safe_jsonable(v) for k, v in eval_metrics.items()},
            }
            _write_json(repro_dir / f"cer_step_{step:06d}.json", cer_payload)
            _write_jsonl(repro_dir / f"preds_step_{step:06d}.jsonl", pred_rows)
            self._logger.warning(
                "ReproPack checkpoint bundle saved: step=%d cer=%.6f (%.2f%%) valid_n=%d empty_pred=%d dir=%s",
                int(step),
                float(eval_metrics.get("cer", 1.0)),
                float(eval_metrics.get("cer_percent", 100.0)),
                int(eval_metrics.get("valid_n", 0)),
                int(eval_metrics.get("empty_pred_count", 0)),
                repro_dir,
            )
        _dist_barrier()
        return eval_metrics if is_rank0 else None


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Repro bundle missing required file: {label}: {path}")
    return path


def load_checkpoint_bundle(
    ckpt_dir: str | Path,
    *,
    tokenizer_loader: Optional[Callable[[str], Any]] = None,
    model_loader: Optional[Callable[[str], Any]] = None,
) -> Tuple[Any, Any, Any, Dict[str, Any], Dict[str, Any], List[FrozenValRecord], Dict[str, Any]]:
    ckpt = Path(ckpt_dir).expanduser().resolve()
    repro = ckpt / "repro"
    _require_file(ckpt / "config.json", "model config")
    _require_file(repro / "generate_config.json", "generate config")
    _require_file(repro / "text_normalization.json", "text normalization contract")
    _require_file(repro / "metrics.json", "metrics contract")
    _require_file(repro / "val_subset.jsonl", "val subset")
    if model_loader is not None:
        model = model_loader(str(ckpt))
    else:
        model = VisionEncoderDecoderModel.from_pretrained(str(ckpt))
    if tokenizer_loader is not None:
        tokenizer = tokenizer_loader(str(repro / "tokenizer"))
    else:
        from transformers import AutoTokenizer  # local import to avoid slow import on train path

        tokenizer = AutoTokenizer.from_pretrained(str(repro / "tokenizer"), use_fast=True)
    image_processor = AutoImageProcessor.from_pretrained(str(repro / "image_processor"))
    generate_cfg = json.loads((repro / "generate_config.json").read_text(encoding="utf-8"))
    norm_cfg = json.loads((repro / "text_normalization.json").read_text(encoding="utf-8"))
    rows = []
    with (repro / "val_subset.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(FrozenValRecord.from_json(json.loads(line)))
    cer_files = sorted(repro.glob("cer_step_*.json"))
    cer_record: Dict[str, Any] = {}
    if cer_files:
        cer_record = json.loads(cer_files[-1].read_text(encoding="utf-8"))
    return model, tokenizer, image_processor, generate_cfg, norm_cfg, rows, cer_record


def save_checkpoint_bundle(*args, **kwargs):
    # Convenience function matching the requested API; delegates to ReproPack instance method.
    rp = kwargs.pop("repro_pack", None)
    if rp is None:
        raise ValueError("save_checkpoint_bundle requires repro_pack=<ReproPack instance>")
    return rp.save_checkpoint_bundle(*args, **kwargs)
