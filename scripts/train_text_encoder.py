#!/usr/bin/env python3
"""Train an unsupervised Tibetan text encoder (SimCSE-style)."""

from __future__ import annotations

import csv
import json
import logging
import math
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_scheduler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tibetan_utils.arg_utils import create_train_text_encoder_parser

LOGGER = logging.getLogger("train_text_encoder")
TEXT_EXTENSIONS = {".txt", ".jsonl", ".csv", ".tsv"}
JSON_TEXT_KEYS = ("text", "content", "line", "sentence", "transcript")


def _configure_logging(is_main_process: bool) -> None:
    level = logging.INFO if is_main_process else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _normalize_text(value: str, normalization: str) -> str:
    text = value.strip()
    if not text:
        return ""
    if normalization != "none":
        text = unicodedata.normalize(normalization, text)
    return text


def _extract_text_from_json_obj(obj: Dict[str, object]) -> Optional[str]:
    for key in JSON_TEXT_KEYS:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _iter_lines_from_file(path: Path) -> Iterable[str]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                yield line
        return

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    text = _extract_text_from_json_obj(obj)
                    if text is not None:
                        yield text
        return

    delimiter = "\t" if suffix == ".tsv" else ","
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fields = [name for name in (reader.fieldnames or []) if name]
        text_field = None
        for candidate in JSON_TEXT_KEYS:
            if candidate in fields:
                text_field = candidate
                break

        if text_field is not None:
            for row in reader:
                value = row.get(text_field)
                if isinstance(value, str):
                    yield value
            return

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            yield line


def _discover_text_samples(
    input_dir: Path,
    normalization: str,
    min_chars: int,
    max_chars: int,
) -> List[str]:
    files = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in TEXT_EXTENSIONS
    )
    samples: List[str] = []
    for file_path in files:
        for raw in _iter_lines_from_file(file_path):
            cleaned = _normalize_text(raw, normalization=normalization)
            if not cleaned:
                continue
            char_count = len(cleaned)
            if char_count < min_chars:
                continue
            if max_chars > 0 and char_count > max_chars:
                cleaned = cleaned[:max_chars]
            samples.append(cleaned)
    return samples


class TextDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        return {"text": self.texts[idx]}


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def _encode_text_backbone(backbone: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if getattr(backbone.config, "is_encoder_decoder", False):
        encoder = backbone.get_encoder() if hasattr(backbone, "get_encoder") else getattr(backbone, "encoder", None)
        if encoder is None:
            raise RuntimeError("Could not access encoder module for encoder-decoder backbone.")
        outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
    else:
        outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    if last_hidden_state is None:
        raise RuntimeError("Backbone output does not contain last_hidden_state.")
    return _mean_pool(last_hidden_state, attention_mask)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        hidden_dim = max(out_dim * 2, in_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    reps = torch.cat([z1, z2], dim=0)
    logits = torch.matmul(reps, reps.T).float() / max(float(temperature), 1e-6)
    diag_mask = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(diag_mask, -1e9)

    batch_size = z1.shape[0]
    labels = torch.arange(2 * batch_size, device=logits.device)
    labels = (labels + batch_size) % (2 * batch_size)
    return F.cross_entropy(logits, labels)


@dataclass
class TrainingArtifacts:
    backbone_dir: Path
    tokenizer_dir: Path
    projection_head_path: Path
    config_path: Path


def _save_artifacts(
    accelerator: Accelerator,
    output_dir: Path,
    backbone: nn.Module,
    tokenizer,
    projection_head: ProjectionHead,
    args,
    *,
    num_samples: int,
    prefix: str = "",
    global_step: int | None = None,
) -> TrainingArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{prefix}_" if prefix else ""
    backbone_dir = output_dir / f"{suffix}text_encoder_backbone"
    tokenizer_dir = output_dir / f"{suffix}text_tokenizer"
    projection_head_path = output_dir / f"{suffix}text_projection_head.pt"
    config_path = output_dir / f"{suffix}training_config.json"

    unwrapped_backbone = accelerator.unwrap_model(backbone)
    unwrapped_head = accelerator.unwrap_model(projection_head)
    unwrapped_backbone.save_pretrained(str(backbone_dir))
    tokenizer.save_pretrained(str(tokenizer_dir))

    torch.save(
        {
            "state_dict": unwrapped_head.state_dict(),
            "input_dim": int(unwrapped_head.net[0].in_features),
            "output_dim": int(unwrapped_head.net[-1].out_features),
            "model_name_or_path": args.model_name_or_path,
            "global_step": int(global_step or 0),
        },
        projection_head_path,
    )

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "task": "text_encoder_unsupervised",
                "model_name_or_path": args.model_name_or_path,
                "input_dir": str(Path(args.input_dir).expanduser().resolve()),
                "normalization": args.normalization,
                "min_chars": int(args.min_chars),
                "max_chars": int(args.max_chars),
                "max_length": int(args.max_length),
                "num_samples": int(num_samples),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "num_train_epochs": int(args.num_train_epochs),
                "max_train_steps": int(args.max_train_steps),
                "warmup_steps": int(args.warmup_steps),
                "projection_dim": int(args.projection_dim),
                "temperature": float(args.temperature),
                "mixed_precision": args.mixed_precision,
                "gradient_checkpointing": bool(args.gradient_checkpointing),
                "freeze_backbone": bool(args.freeze_backbone),
                "seed": int(args.seed),
                "global_step": int(global_step or 0),
                "backbone_dir": str(backbone_dir),
                "tokenizer_dir": str(tokenizer_dir),
                "projection_head_path": str(projection_head_path),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return TrainingArtifacts(
        backbone_dir=backbone_dir,
        tokenizer_dir=tokenizer_dir,
        projection_head_path=projection_head_path,
        config_path=config_path,
    )


def run(args) -> dict:
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    _configure_logging(accelerator.is_main_process)
    set_seed(int(args.seed))

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    samples = _discover_text_samples(
        input_dir=input_dir,
        normalization=args.normalization,
        min_chars=int(args.min_chars),
        max_chars=int(args.max_chars),
    )
    if len(samples) < 2:
        raise RuntimeError(f"Not enough text samples found in {input_dir} (found {len(samples)}).")

    if accelerator.is_main_process:
        LOGGER.info("Loaded %d text samples", len(samples))

    dataset = TextDataset(samples)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    def collate_fn(batch):
        texts = [row["text"] for row in batch]
        tokenized = tokenizer(
            texts,
            max_length=int(args.max_length),
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        return tokenized

    dataloader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    if len(dataloader) == 0:
        raise RuntimeError("DataLoader has zero batches. Reduce batch_size or add more text samples.")

    backbone = AutoModel.from_pretrained(args.model_name_or_path)
    if args.gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()

    hidden_size = getattr(backbone.config, "hidden_size", None)
    if hidden_size is None and hasattr(backbone.config, "d_model"):
        hidden_size = getattr(backbone.config, "d_model")
    if hidden_size is None:
        raise RuntimeError("Backbone config has no hidden_size/d_model.")
    projection_head = ProjectionHead(in_dim=int(hidden_size), out_dim=int(args.projection_dim))

    if args.freeze_backbone:
        backbone.requires_grad_(False)
        trainable_params = list(projection_head.parameters())
    else:
        trainable_params = list(backbone.parameters()) + list(projection_head.parameters())

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    num_update_steps_per_epoch = len(dataloader)
    max_train_steps = int(args.max_train_steps)
    if max_train_steps <= 0:
        max_train_steps = int(args.num_train_epochs) * num_update_steps_per_epoch
    num_train_epochs = int(math.ceil(max_train_steps / num_update_steps_per_epoch))

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_steps),
        num_training_steps=max_train_steps,
    )

    backbone, projection_head, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        backbone, projection_head, optimizer, dataloader, lr_scheduler
    )

    progress_bar = tqdm(
        total=max_train_steps,
        disable=not accelerator.is_local_main_process,
        desc="train-text-encoder",
    )

    global_step = 0
    cumulative_loss = 0.0
    for epoch in range(num_train_epochs):
        backbone.train()
        projection_head.train()

        for batch in dataloader:
            if global_step >= max_train_steps:
                break

            input_ids = batch["input_ids"].to(accelerator.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(accelerator.device, non_blocking=True)

            if args.freeze_backbone:
                with torch.no_grad():
                    emb_one = _encode_text_backbone(backbone, input_ids=input_ids, attention_mask=attention_mask)
                    emb_two = _encode_text_backbone(backbone, input_ids=input_ids, attention_mask=attention_mask)
            else:
                emb_one = _encode_text_backbone(backbone, input_ids=input_ids, attention_mask=attention_mask)
                emb_two = _encode_text_backbone(backbone, input_ids=input_ids, attention_mask=attention_mask)

            proj_one = projection_head(emb_one)
            proj_two = projection_head(emb_two)
            loss = _nt_xent_loss(proj_one, proj_two, temperature=float(args.temperature))

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            cumulative_loss += float(loss.detach().item())
            progress_bar.update(1)
            progress_bar.set_postfix(
                loss=f"{loss.detach().item():.4f}",
                lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                epoch=epoch + 1,
            )

            if (
                int(args.checkpoint_every_steps) > 0
                and global_step % int(args.checkpoint_every_steps) == 0
                and accelerator.is_main_process
            ):
                ckpt_prefix = args.checkpoint_name.strip() or "checkpoint"
                if args.checkpoint_overwrite:
                    prefix = ckpt_prefix
                else:
                    prefix = f"{ckpt_prefix}_step_{global_step:07d}"
                artifacts = _save_artifacts(
                    accelerator=accelerator,
                    output_dir=output_dir,
                    backbone=backbone,
                    tokenizer=tokenizer,
                    projection_head=projection_head,
                    args=args,
                    num_samples=len(samples),
                    prefix=prefix,
                    global_step=global_step,
                )
                LOGGER.info("Saved checkpoint at step %d -> %s", global_step, artifacts.backbone_dir)

        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()
    final_artifacts = None
    if accelerator.is_main_process:
        final_artifacts = _save_artifacts(
            accelerator=accelerator,
            output_dir=output_dir,
            backbone=backbone,
            tokenizer=tokenizer,
            projection_head=projection_head,
            args=args,
            num_samples=len(samples),
            prefix="",
            global_step=global_step,
        )
        avg_loss = cumulative_loss / max(1, global_step)
        LOGGER.info("Training finished after %d steps. avg_loss=%.6f", global_step, avg_loss)
        LOGGER.info("Saved backbone: %s", final_artifacts.backbone_dir)
        LOGGER.info("Saved tokenizer: %s", final_artifacts.tokenizer_dir)
        LOGGER.info("Saved projection head: %s", final_artifacts.projection_head_path)
        LOGGER.info("Saved config: %s", final_artifacts.config_path)

    return {
        "global_step": global_step,
        "output_dir": str(output_dir),
        "backbone_dir": str(final_artifacts.backbone_dir) if final_artifacts else "",
        "tokenizer_dir": str(final_artifacts.tokenizer_dir) if final_artifacts else "",
        "projection_head_path": str(final_artifacts.projection_head_path) if final_artifacts else "",
        "config_path": str(final_artifacts.config_path) if final_artifacts else "",
    }


def main() -> None:
    parser = create_train_text_encoder_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
