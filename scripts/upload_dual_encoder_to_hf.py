#!/usr/bin/env python3
"""Upload a PechaBridge Dual Image-Text Encoder checkpoint to HuggingFace Hub.

The dual encoder consists of two sub-models saved side-by-side in a checkpoint
directory:

    <checkpoint_dir>/
        <prefix>_text_hierarchy_clip_text_encoder/   ← fine-tuned ByT5 text encoder
            config.json
            model.safetensors
            tokenizer_config.json
            added_tokens.json
        <prefix>_text_hierarchy_vit_backbone/        ← fine-tuned DINOv2 image encoder
            config.json
            model.safetensors
            preprocessor_config.json
        <prefix>_training_config.json                ← training hyper-parameters & metrics

The script uploads both sub-models into a single HF repo using the flat layout
expected by ``scripts/download_pechabridge_models.py`` (and ``cli.py
download-models --models encoder``):

    <repo>/
        text_encoder/
            config.json
            model.safetensors
            tokenizer_config.json
            added_tokens.json
        vit_backbone/
            config.json
            model.safetensors
            preprocessor_config.json
        training_config.json
        README.md

Usage
-----
    python scripts/upload_dual_encoder_to_hf.py \\
        --checkpoint models/encoders/best \\
        --repo-id your-username/PechaBridgeDualEncoder \\
        --token hf_xxxxxxxxxxxx

    # Private repo:
    python scripts/upload_dual_encoder_to_hf.py \\
        --checkpoint models/encoders/best \\
        --repo-id your-username/PechaBridgeDualEncoder \\
        --token hf_xxxxxxxxxxxx \\
        --private

    # Dry-run (show what would be uploaded without actually uploading):
    python scripts/upload_dual_encoder_to_hf.py \\
        --checkpoint models/encoders/best \\
        --repo-id your-username/PechaBridgeDualEncoder \\
        --token hf_xxxxxxxxxxxx \\
        --dry-run

    # Token from environment variable HF_TOKEN:
    export HF_TOKEN=hf_xxxxxxxxxxxx
    python scripts/upload_dual_encoder_to_hf.py \\
        --checkpoint models/encoders/best \\
        --repo-id your-username/PechaBridgeDualEncoder

Notes
-----
- The checkpoint directory is scanned for sub-directories whose names end with
  ``_text_hierarchy_clip_text_encoder`` and ``_text_hierarchy_vit_backbone``.
- A ``*_training_config.json`` file at the checkpoint root is also uploaded as
  ``training_config.json``.
- Large model files are handled via HF LFS automatically.
- Each file is uploaded individually with a tqdm byte-level progress bar.
- Token can also be set via the HF_TOKEN environment variable.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Sub-directory discovery
# ---------------------------------------------------------------------------

def _find_subdir(ckpt: Path, suffix: str) -> Optional[Path]:
    """Return the first sub-directory whose name ends with *suffix*."""
    candidates = sorted(
        p for p in ckpt.iterdir()
        if p.is_dir() and p.name.endswith(suffix) and not p.name.startswith(".")
    )
    return candidates[0] if candidates else None


def _find_training_config(ckpt: Path) -> Optional[Path]:
    """Return the first *_training_config.json at the checkpoint root."""
    candidates = sorted(
        p for p in ckpt.glob("*_training_config.json")
        if not p.name.startswith(".")
    )
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# File collection helpers
# ---------------------------------------------------------------------------

_TEXT_ENCODER_FILES = [
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "tokenizer_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "vocab.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
]

_VIT_BACKBONE_FILES = [
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "preprocessor_config.json",
    "image_processor_config.json",
]


def _collect_subdir_files(src: Path, known_names: List[str]) -> List[Tuple[Path, str]]:
    """Collect files from *src* that match *known_names* (exact) or *.model globs.

    Returns list of (local_path, filename) tuples.
    """
    found: List[Tuple[Path, str]] = []
    seen: set[str] = set()

    for name in known_names:
        p = src / name
        if p.exists() and p.is_file() and not p.name.startswith("."):
            found.append((p, p.name))
            seen.add(p.name)

    # Catch sharded weights
    for pattern in ("model-*-of-*.safetensors", "pytorch_model-*-of-*.bin"):
        for p in sorted(src.glob(pattern)):
            if p.name not in seen and not p.name.startswith("."):
                found.append((p, p.name))
                seen.add(p.name)

    # Catch SentencePiece / BoSentencePiece models
    for p in sorted(src.glob("*.model")):
        if p.name not in seen and not p.name.startswith("."):
            found.append((p, p.name))
            seen.add(p.name)

    return found


# ---------------------------------------------------------------------------
# README generator
# ---------------------------------------------------------------------------

def _parse_metrics(training_config: Optional[Path]) -> dict:
    """Read last_val_metrics from training_config.json if present."""
    if training_config is None or not training_config.exists():
        return {}
    try:
        data = json.loads(training_config.read_text(encoding="utf-8"))
        return data.get("last_val_metrics", {})
    except Exception:
        return {}


def _parse_training_config(training_config: Optional[Path]) -> dict:
    """Read the full training config dict (best-effort)."""
    if training_config is None or not training_config.exists():
        return {}
    try:
        return json.loads(training_config.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _build_readme(repo_id: str, ckpt: Path, training_config: Optional[Path]) -> str:
    cfg = _parse_training_config(training_config)
    metrics = cfg.get("last_val_metrics", {})

    model_name = repo_id.split("/")[-1]
    global_step = cfg.get("global_step", "?")
    backbone = cfg.get("model_name_or_path", "facebook/dinov2-base")
    text_enc = cfg.get("text_encoder_name_or_path", "google/byt5-small")
    projection_dim = cfg.get("projection_dim", 256)
    image_preprocess = cfg.get("image_preprocess_pipeline", "bdrc")
    target_height = cfg.get("target_height", 64)
    max_width = cfg.get("max_width", 1024)
    groups = cfg.get("groups", "?")
    batch_size = cfg.get("batch_size", "?")
    lr = cfg.get("lr", "?")

    i2t_r1 = metrics.get("val_i2t_r1", None)
    t2i_r1 = metrics.get("val_t2i_r1", None)
    i2t_r5 = metrics.get("val_i2t_r5", None)
    t2i_r5 = metrics.get("val_t2i_r5", None)

    def pct(v):
        return f"{v * 100:.2f}%" if v is not None else "—"

    lines = [
        f"# {model_name}",
        "",
        "Tibetan dual image-text encoder trained with a CLIP-style symmetric InfoNCE loss "
        "on Tibetan pecha line images paired with their OCR transcripts. "
        "The model enables cross-modal retrieval between line images and Tibetan text.",
        "",
        "## Architecture",
        "",
        f"| Component | Base model | Role |",
        f"|-----------|-----------|------|",
        f"| **Image encoder** (ViT backbone) | `{backbone}` | Encodes line-image crops → 256-d embedding |",
        f"| **Text encoder** (CLIP text encoder) | `{text_enc}` | Encodes Tibetan transcript lines → 256-d embedding |",
        "",
        f"- **Projection dimension**: {projection_dim}",
        f"- **Image preprocessing pipeline**: `{image_preprocess}` "
        f"(target height {target_height}px, max width {max_width}px, aspect-preserving)",
        f"- **Training step**: {global_step}",
        f"- **Training pairs**: {groups:,}" if isinstance(groups, int) else f"- **Training pairs**: {groups}",
        "",
        "## Retrieval performance (validation set)",
        "",
        "| Metric | Score |",
        "|--------|-------|",
        f"| Image→Text R@1 | {pct(i2t_r1)} |",
        f"| Text→Image R@1 | {pct(t2i_r1)} |",
        f"| Image→Text R@5 | {pct(i2t_r5)} |",
        f"| Text→Image R@5 | {pct(t2i_r5)} |",
        "",
        "## Recommended usage — PechaBridge CLI",
        "",
        "```bash",
        "# 1. Clone PechaBridge and install dependencies",
        "git clone https://github.com/CodexAITeam/PechaBridge.git && cd PechaBridge",
        "pip install -r requirements.txt",
        "",
        "# 2. Download this encoder (alongside OCR + line segmentation models)",
        "python cli.py download-models --models encoder",
        "",
        "# 3. Launch the Semantic Search Workbench",
        "#    Set huggingface_model_id in your workbench config to:",
        f"#      {repo_id}",
        "#    (pointing to the text_encoder/ sub-directory for text-only search)",
        "python cli.py semantic-search-workbench --config configs/semantic_search.yaml",
        "```",
        "",
        "## Repository layout",
        "",
        "```",
        f"{model_name}/",
        "  text_encoder/          ← fine-tuned ByT5 text encoder + tokenizer",
        "    config.json",
        "    model.safetensors",
        "    tokenizer_config.json",
        "    added_tokens.json",
        "  vit_backbone/          ← fine-tuned DINOv2 image encoder + preprocessor",
        "    config.json",
        "    model.safetensors",
        "    preprocessor_config.json",
        "  training_config.json   ← full training hyper-parameters & validation metrics",
        "  README.md",
        "```",
        "",
        "## Python usage",
        "",
        "### Text-only embedding (Semantic Search Workbench)",
        "",
        "The Semantic Search Workbench uses only the **text encoder** sub-directory "
        "via `AutoModel` + `AutoTokenizer`. Point `huggingface_model_id` in your "
        "workbench YAML config to `{repo_id}/text_encoder` (or the local path "
        "`models/encoders/PechaBridgeDualEncoder/text_encoder`).",
        "",
        "```python",
        "from transformers import AutoModel, AutoTokenizer",
        "import torch, torch.nn.functional as F",
        "",
        f'tokenizer = AutoTokenizer.from_pretrained("{repo_id}", subfolder="text_encoder")',
        f'text_model = AutoModel.from_pretrained("{repo_id}", subfolder="text_encoder").eval()',
        "",
        "texts = [\"བོད་ཀྱི་གནའ་རབས་ལོ་རྒྱུས།\"]",
        "enc = tokenizer(texts, return_tensors=\"pt\", padding=True, truncation=True, max_length=256)",
        "with torch.no_grad():",
        "    out = text_model(**enc)",
        "    # mean-pool over sequence length",
        "    mask = enc[\"attention_mask\"].unsqueeze(-1).float()",
        "    emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)",
        "    emb = F.normalize(emb, p=2, dim=1)",
        "print(emb.shape)  # (1, 256)",
        "```",
        "",
        "### Image embedding (cross-modal retrieval)",
        "",
        "```python",
        "from transformers import AutoModel, AutoImageProcessor",
        "from pechabridge.ocr.preprocess_bdrc import BDRCPreprocessConfig, preprocess_image_bdrc",
        "from PIL import Image",
        "import torch, torch.nn.functional as F",
        "",
        f'processor = AutoImageProcessor.from_pretrained("{repo_id}", subfolder="vit_backbone")',
        f'vit_model = AutoModel.from_pretrained("{repo_id}", subfolder="vit_backbone").eval()',
        "",
        "# Load and BDRC-preprocess a line-image crop",
        "image = Image.open(\"line_crop.png\").convert(\"RGB\")",
        f"cfg = BDRCPreprocessConfig(pipeline=\"{image_preprocess}\", target_height={target_height}, max_width={max_width})",
        "preprocessed = preprocess_image_bdrc(image, cfg)",
        "",
        "inputs = processor(images=preprocessed, return_tensors=\"pt\")",
        "with torch.no_grad():",
        "    out = vit_model(**inputs)",
        "    # CLS token",
        "    emb = F.normalize(out.last_hidden_state[:, 0, :], p=2, dim=1)",
        "print(emb.shape)  # (1, 768)  — project to 256-d with the projection head if needed",
        "```",
        "",
        "## Training details",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Batch size | {batch_size} |",
        f"| Learning rate | {lr} |",
        f"| Loss | CLIP symmetric InfoNCE |",
        f"| Mixed precision | {cfg.get('mixed_precision', 'bf16')} |",
        f"| Warmup steps | {cfg.get('warmup_steps', '?')} |",
        f"| Temperature | {cfg.get('temperature', 0.1)} |",
        "",
        "## Training framework",
        "",
        "- **Framework**: [PechaBridge](https://github.com/CodexAITeam/PechaBridge)",
        "- **Training data**: Tibetan pecha line images from OpenPecha and BDRC collections",
        "- **Image preprocessing**: BDRC-style adaptive binarisation, background normalisation, "
        "aspect-preserving resize and padding",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Progress-aware upload (identical to upload_donut_to_hf.py)
# ---------------------------------------------------------------------------

class _TqdmReader(io.RawIOBase):
    """Wraps a file object and updates a tqdm bar as bytes are read."""

    def __init__(self, fh: io.BufferedReader, bar):
        self._fh = fh
        self._bar = bar

    def read(self, n: int = -1) -> bytes:
        chunk = self._fh.read(n)
        if chunk:
            self._bar.update(len(chunk))
        return chunk

    def readinto(self, b) -> int:
        n = self._fh.readinto(b)
        if n:
            self._bar.update(n)
        return n

    def readable(self) -> bool:
        return True


def _upload_file_with_progress(
    api,
    local_path: Path,
    path_in_repo: str,
    repo_id: str,
    commit_message: str,
) -> None:
    """Upload a single file to HF with a tqdm byte-level progress bar."""
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    size = local_path.stat().st_size
    label = f"  {path_in_repo} ({size / 1024 / 1024:.1f} MB)"

    if has_tqdm:
        with tqdm(
            total=size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=label,
            ncols=90,
            leave=True,
        ) as bar:
            with open(local_path, "rb") as fh:
                wrapped = io.BufferedReader(_TqdmReader(fh, bar))  # type: ignore[arg-type]
                api.upload_file(
                    path_or_fileobj=wrapped,
                    path_in_repo=path_in_repo,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=commit_message,
                )
    else:
        print(f"  Uploading {label} …")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )


# ---------------------------------------------------------------------------
# Main upload logic
# ---------------------------------------------------------------------------

def upload_dual_encoder(
    checkpoint_path: str,
    repo_id: str,
    token: str,
    *,
    private: bool = False,
    dry_run: bool = False,
    commit_message: str = "",
) -> None:
    """Upload the dual encoder checkpoint to HuggingFace Hub.

    Parameters
    ----------
    checkpoint_path:
        Path to the checkpoint directory (e.g. ``models/encoders/best``).
        Must contain sub-directories ending with
        ``_text_hierarchy_clip_text_encoder`` and
        ``_text_hierarchy_vit_backbone``.
    repo_id:
        HuggingFace repo ID in the form ``username/model-name``.
    token:
        HuggingFace API token.
    private:
        Create the repo as private (default: public).
    dry_run:
        Show what would be uploaded without actually uploading.
    commit_message:
        Custom commit message (auto-generated if empty).
    """
    ckpt = Path(checkpoint_path).expanduser().resolve()
    if not ckpt.exists():
        print(f"ERROR: Checkpoint directory not found: {ckpt}", file=sys.stderr)
        sys.exit(1)
    if not ckpt.is_dir():
        print(f"ERROR: Expected a directory, got: {ckpt}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Dual Encoder → HuggingFace Upload")
    print(f"{'='*60}")
    print(f"  Checkpoint : {ckpt}")
    print(f"  Repo ID    : {repo_id}")
    print(f"  Private    : {private}")
    print(f"  Dry-run    : {dry_run}")
    print(f"{'='*60}\n")

    # --- Locate sub-directories ---
    text_enc_dir = _find_subdir(ckpt, "_text_hierarchy_clip_text_encoder")
    vit_dir = _find_subdir(ckpt, "_text_hierarchy_vit_backbone")
    training_config = _find_training_config(ckpt)

    if text_enc_dir is None:
        print(
            "ERROR: Could not find a sub-directory ending with "
            "'_text_hierarchy_clip_text_encoder' in:\n"
            f"  {ckpt}",
            file=sys.stderr,
        )
        sys.exit(1)

    if vit_dir is None:
        print(
            "ERROR: Could not find a sub-directory ending with "
            "'_text_hierarchy_vit_backbone' in:\n"
            f"  {ckpt}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  Text encoder dir : {text_enc_dir.name}")
    print(f"  ViT backbone dir : {vit_dir.name}")
    if training_config:
        print(f"  Training config  : {training_config.name}")
    else:
        print("  Training config  : (not found — skipping)")
    print()

    # --- Collect files ---
    text_enc_files = _collect_subdir_files(text_enc_dir, _TEXT_ENCODER_FILES)
    vit_files = _collect_subdir_files(vit_dir, _VIT_BACKBONE_FILES)

    if not text_enc_files:
        print(
            f"ERROR: No model files found in text encoder directory: {text_enc_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not vit_files:
        print(
            f"ERROR: No model files found in ViT backbone directory: {vit_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Build upload manifest ---
    # Layout in the HF repo:
    #   text_encoder/<file>
    #   vit_backbone/<file>
    #   training_config.json
    #   README.md
    manifest: List[Tuple[Path, str]] = []

    for local_path, fname in text_enc_files:
        manifest.append((local_path, f"text_encoder/{fname}"))

    for local_path, fname in vit_files:
        manifest.append((local_path, f"vit_backbone/{fname}"))

    if training_config is not None:
        manifest.append((training_config, "training_config.json"))

    # README (written to a temp file)
    readme_content = _build_readme(repo_id=repo_id, ckpt=ckpt, training_config=training_config)
    readme_tmp = Path(tempfile.mktemp(suffix=".md", prefix="pb_enc_readme_"))
    readme_tmp.write_text(readme_content, encoding="utf-8")
    manifest.append((readme_tmp, "README.md"))

    # --- Print manifest ---
    total_bytes = sum(p.stat().st_size for p, _ in manifest)
    print("Files to upload:")
    print(f"  {'Destination in repo':<50} {'Size':>10}")
    print(f"  {'-'*50} {'-'*10}")
    for local_path, dest in manifest:
        size_mb = local_path.stat().st_size / 1024 / 1024
        print(f"  {dest:<50} {size_mb:>9.1f}M")
    print(f"\n  Total: {total_bytes / 1024 / 1024:.1f} MB  ({len(manifest)} files)")

    if dry_run:
        print("\nDry-run mode — nothing uploaded.")
        readme_tmp.unlink(missing_ok=True)
        return

    # --- Import huggingface_hub ---
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print(
            "ERROR: huggingface_hub is not installed.\n"
            "Install it with:  pip install huggingface-hub",
            file=sys.stderr,
        )
        readme_tmp.unlink(missing_ok=True)
        sys.exit(1)

    api = HfApi(token=token)

    # --- Create repo ---
    print(f"\nCreating/verifying repo '{repo_id}' on HuggingFace …")
    try:
        create_repo(repo_id=repo_id, token=token, private=private, repo_type="model", exist_ok=True)
        print(f"  Repo ready: https://huggingface.co/{repo_id}\n")
    except Exception as exc:
        print(f"ERROR creating repo: {exc}", file=sys.stderr)
        readme_tmp.unlink(missing_ok=True)
        sys.exit(1)

    # --- Upload file by file with progress bar ---
    msg = commit_message or f"Upload dual encoder checkpoint {ckpt.name} via PechaBridge"
    print(f"Uploading {len(manifest)} file(s) — each file shows a byte-level progress bar:\n")

    try:
        for i, (local_path, path_in_repo) in enumerate(manifest, 1):
            print(f"[{i}/{len(manifest)}]")
            try:
                _upload_file_with_progress(
                    api=api,
                    local_path=local_path,
                    path_in_repo=path_in_repo,
                    repo_id=repo_id,
                    commit_message=msg,
                )
            except Exception as exc:
                print(f"\nERROR uploading {path_in_repo}: {exc}", file=sys.stderr)
                raise
    except Exception:
        readme_tmp.unlink(missing_ok=True)
        sys.exit(1)

    readme_tmp.unlink(missing_ok=True)

    print(f"\n{'='*60}")
    print(f"  ✓ Upload complete!")
    print(f"  Model URL: https://huggingface.co/{repo_id}")
    print(f"{'='*60}")
    print(f"\nTo download with PechaBridge:")
    print(f"  python cli.py download-models --models encoder")
    print(f"\nTo use the text encoder in the Semantic Search Workbench,")
    print(f"set in your workbench YAML config:")
    print(f"  embedding:")
    print(f"    huggingface_model_id: {repo_id}/text_encoder")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Upload a PechaBridge Dual Image-Text Encoder checkpoint to HuggingFace Hub.\n\n"
            "Uploads both the ByT5 text encoder and the DINOv2 ViT backbone into a single\n"
            "HF repo under text_encoder/ and vit_backbone/ sub-directories.\n"
            "Each file is uploaded with a tqdm progress bar."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload with HF token:
  python scripts/upload_dual_encoder_to_hf.py \\
      --checkpoint models/encoders/best \\
      --repo-id myuser/PechaBridgeDualEncoder \\
      --token hf_xxxxxxxxxxxx

  # Private repo:
  python scripts/upload_dual_encoder_to_hf.py \\
      --checkpoint models/encoders/best \\
      --repo-id myuser/PechaBridgeDualEncoder \\
      --token hf_xxxxxxxxxxxx \\
      --private

  # Dry-run (show what would be uploaded):
  python scripts/upload_dual_encoder_to_hf.py \\
      --checkpoint models/encoders/best \\
      --repo-id myuser/PechaBridgeDualEncoder \\
      --token hf_xxxxxxxxxxxx \\
      --dry-run

  # Token from environment variable HF_TOKEN:
  export HF_TOKEN=hf_xxxxxxxxxxxx
  python scripts/upload_dual_encoder_to_hf.py \\
      --checkpoint models/encoders/best \\
      --repo-id myuser/PechaBridgeDualEncoder
""",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        metavar="PATH",
        help=(
            "Path to the dual encoder checkpoint directory "
            "(e.g. models/encoders/best). "
            "Must contain sub-directories ending with "
            "'_text_hierarchy_clip_text_encoder' and '_text_hierarchy_vit_backbone'."
        ),
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        metavar="USER/MODEL",
        help="HuggingFace repo ID in the form 'username/model-name'.",
    )

    auth_group = parser.add_argument_group(
        "Authentication (one of --token or --username/--password required)"
    )
    auth_group.add_argument(
        "--token",
        default="",
        metavar="HF_TOKEN",
        help=(
            "HuggingFace API token (starts with 'hf_'). "
            "Can also be set via the HF_TOKEN environment variable."
        ),
    )
    auth_group.add_argument(
        "--username",
        default="",
        metavar="USERNAME",
        help="HuggingFace username (used together with --password as alternative to --token).",
    )
    auth_group.add_argument(
        "--password",
        default="",
        metavar="PASSWORD_OR_TOKEN",
        help=(
            "HuggingFace password or API token. "
            "HuggingFace recommends using API tokens (hf_...) instead of passwords."
        ),
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the HuggingFace repo as private (default: public).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading anything.",
    )
    parser.add_argument(
        "--commit-message",
        default="",
        metavar="MSG",
        help="Custom commit message for the HF upload (default: auto-generated).",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    # --- Resolve token ---
    token = str(args.token or "").strip()
    if not token:
        token = str(os.environ.get("HF_TOKEN", "") or "").strip()
    if not token and args.username and args.password:
        token = str(args.password or "").strip()

    if not token and not args.dry_run:
        print(
            "ERROR: No HuggingFace token provided.\n"
            "Use --token hf_xxx, set HF_TOKEN env var, or use --username + --password.",
            file=sys.stderr,
        )
        return 1

    upload_dual_encoder(
        checkpoint_path=args.checkpoint,
        repo_id=args.repo_id,
        token=token,
        private=args.private,
        dry_run=args.dry_run,
        commit_message=args.commit_message,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
