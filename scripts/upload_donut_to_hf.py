#!/usr/bin/env python3
"""Upload a DONUT OCR checkpoint to HuggingFace Hub.

Collects all required files from the checkpoint directory (model weights,
tokenizer, image processor, repro bundle) and pushes them to a HF repo.
Each file is uploaded individually with a tqdm byte-level progress bar.

Usage
-----
    python scripts/upload_donut_to_hf.py \\
        --checkpoint /path/to/checkpoint-5000 \\
        --repo-id your-username/your-model-name \\
        --token hf_xxxxxxxxxxxx

    # Or use --username + --password (HF login):
    python scripts/upload_donut_to_hf.py \\
        --checkpoint /path/to/checkpoint-5000 \\
        --repo-id your-username/your-model-name \\
        --username your-username \\
        --password hf_xxxxxxxxxxxx

    # Private repo:
    python scripts/upload_donut_to_hf.py \\
        --checkpoint /path/to/checkpoint-5000 \\
        --repo-id your-username/your-model-name \\
        --token hf_xxxxxxxxxxxx \\
        --private

    # Dry-run (show what would be uploaded without actually uploading):
    python scripts/upload_donut_to_hf.py \\
        --checkpoint /path/to/checkpoint-5000 \\
        --repo-id your-username/your-model-name \\
        --token hf_xxxxxxxxxxxx \\
        --dry-run

Notes
-----
- Large model files are handled via HF LFS automatically.
- Each file is uploaded individually with a tqdm progress bar (bytes).
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
# File collection helpers
# ---------------------------------------------------------------------------

def _collect_model_files(ckpt: Path) -> List[Path]:
    """Return all model-related files from the checkpoint root."""
    candidates = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ]
    found: List[Path] = []
    for name in candidates:
        p = ckpt / name
        if p.exists():
            found.append(p)
    # sharded weights
    for pattern in ("model-*-of-*.safetensors", "pytorch_model-*-of-*.bin"):
        found.extend(sorted(ckpt.glob(pattern)))
    return found


def _collect_tokenizer_files(src: Path) -> List[Path]:
    """Return all tokenizer files from a directory."""
    patterns = [
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "vocab.txt",
        "added_tokens.json",
        "*.model",          # SentencePiece / BoSentencePiece
        "spiece.model",
        "sentencepiece.bpe.model",
    ]
    found: List[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for p in sorted(src.glob(pattern)):
            if p.name not in seen and p.is_file():
                found.append(p)
                seen.add(p.name)
    return found


def _collect_image_processor_files(src: Path) -> List[Path]:
    """Return all image processor files from a directory."""
    found: List[Path] = []
    for name in ("preprocessor_config.json", "image_processor_config.json"):
        p = src / name
        if p.exists():
            found.append(p)
    return found


def _locate_tokenizer_dir(ckpt: Path) -> Optional[Path]:
    """Same search order as _load_donut_runtime() in batch_ocr.py."""
    repro = ckpt / "repro"
    if (repro / "tokenizer").exists():
        return repro / "tokenizer"
    if (ckpt.parent / "tokenizer").exists():
        return ckpt.parent / "tokenizer"
    if (ckpt / "tokenizer_config.json").exists():
        return ckpt
    return None


def _locate_image_processor_dir(ckpt: Path) -> Optional[Path]:
    """Same search order as _load_donut_runtime() in batch_ocr.py."""
    repro = ckpt / "repro"
    if (repro / "image_processor").exists():
        return repro / "image_processor"
    if (ckpt.parent / "image_processor").exists():
        return ckpt.parent / "image_processor"
    if (ckpt / "preprocessor_config.json").exists():
        return ckpt
    return None


def _locate_repro_files(ckpt: Path) -> List[Tuple[Path, str]]:
    """Return list of (src_path, dest_in_repo) for repro bundle files."""
    repro = ckpt / "repro"
    result: List[Tuple[Path, str]] = []
    if not repro.exists():
        return result
    critical = [
        "generate_config.json",
        "image_preprocess.json",
        "text_normalization.json",
        "text_normalization.py",
        "metrics.json",
    ]
    for name in critical:
        p = repro / name
        if p.exists():
            result.append((p, f"repro/{name}"))
    for p in sorted(repro.glob("cer_step_*.json")):
        result.append((p, f"repro/{p.name}"))
    return result


# ---------------------------------------------------------------------------
# README generator
# ---------------------------------------------------------------------------

def _build_readme(repo_id: str, ckpt: Path, preprocess_pipeline: str, has_repro: bool) -> str:
    step = ""
    if ckpt.name.startswith("checkpoint-"):
        step = ckpt.name.replace("checkpoint-", "step ")
    lines = [
        f"# {repo_id.split('/')[-1]}",
        "",
        "Tibetan OCR model (DONUT / VisionEncoderDecoder architecture) trained with PechaBridge.",
        "",
        "## Usage",
        "",
        "```python",
        "from transformers import AutoImageProcessor, VisionEncoderDecoderModel, AutoTokenizer",
        "from PIL import Image",
        "",
        f'model = VisionEncoderDecoderModel.from_pretrained("{repo_id}")',
        f'tokenizer = AutoTokenizer.from_pretrained("{repo_id}")',
        f'image_processor = AutoImageProcessor.from_pretrained("{repo_id}")',
        "",
        "image = Image.open('line_crop.png').convert('RGB')",
        "pixel_values = image_processor(images=image, return_tensors='pt').pixel_values",
        "generated_ids = model.generate(pixel_values)",
        "text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]",
        "```",
        "",
        "## Details",
        "",
        f"- **Checkpoint**: `{ckpt.name}`" + (f" ({step})" if step else ""),
        f"- **Image preprocessing pipeline**: `{preprocess_pipeline}`",
        f"- **Repro bundle included**: {'yes' if has_repro else 'no'}",
        "",
        "## PechaBridge",
        "",
        "Trained with [PechaBridge](https://github.com/OpenPecha/PechaBridge).",
        "",
        "```bash",
        "python cli.py batch-ocr \\",
        f'    --ocr-model {repo_id} \\',
        "    --input-dir /path/to/images \\",
        "    --engine donut",
        "```",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Progress-aware upload
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

def upload_donut_checkpoint(
    checkpoint_path: str,
    repo_id: str,
    token: str,
    *,
    private: bool = False,
    dry_run: bool = False,
    commit_message: str = "",
) -> None:
    ckpt = Path(checkpoint_path).expanduser().resolve()
    if not ckpt.exists():
        print(f"ERROR: Checkpoint directory not found: {ckpt}", file=sys.stderr)
        sys.exit(1)
    if not ckpt.is_dir():
        print(f"ERROR: Expected a directory, got: {ckpt}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  DONUT → HuggingFace Upload")
    print(f"{'='*60}")
    print(f"  Checkpoint : {ckpt}")
    print(f"  Repo ID    : {repo_id}")
    print(f"  Private    : {private}")
    print(f"  Dry-run    : {dry_run}")
    print(f"{'='*60}\n")

    # --- Collect files ---
    model_files = _collect_model_files(ckpt)
    if not model_files:
        print("ERROR: No model weights found (model.safetensors / pytorch_model.bin).", file=sys.stderr)
        sys.exit(1)

    tokenizer_dir = _locate_tokenizer_dir(ckpt)
    if tokenizer_dir is None:
        print("ERROR: Could not find tokenizer directory.", file=sys.stderr)
        sys.exit(1)
    tokenizer_files = _collect_tokenizer_files(tokenizer_dir)
    if not tokenizer_files:
        print(f"ERROR: No tokenizer files found in {tokenizer_dir}", file=sys.stderr)
        sys.exit(1)

    image_processor_dir = _locate_image_processor_dir(ckpt)
    if image_processor_dir is None:
        print("ERROR: Could not find image processor directory.", file=sys.stderr)
        sys.exit(1)
    image_processor_files = _collect_image_processor_files(image_processor_dir)
    if not image_processor_files:
        print(f"ERROR: No image processor files found in {image_processor_dir}", file=sys.stderr)
        sys.exit(1)

    repro_files = _locate_repro_files(ckpt)

    # Read preprocessing pipeline for README
    preprocess_pipeline = "bdrc"
    repro_preprocess_path = ckpt / "repro" / "image_preprocess.json"
    if repro_preprocess_path.exists():
        try:
            data = json.loads(repro_preprocess_path.read_text(encoding="utf-8"))
            preprocess_pipeline = str(data.get("pipeline", "bdrc") or "bdrc")
        except Exception:
            pass

    # Build upload manifest: list of (local_path, path_in_repo)
    # All files land at the repo root (HF hub style) except repro/
    manifest: List[Tuple[Path, str]] = []
    used_names: set[str] = set()

    for f in model_files:
        manifest.append((f, f.name))
        used_names.add(f.name)

    for f in tokenizer_files:
        if f.name not in used_names:
            manifest.append((f, f.name))
            used_names.add(f.name)

    for f in image_processor_files:
        if f.name not in used_names:
            manifest.append((f, f.name))
            used_names.add(f.name)

    for src, dest_rel in repro_files:
        manifest.append((src, dest_rel))

    # README (written to a temp file)
    readme_content = _build_readme(
        repo_id=repo_id,
        ckpt=ckpt,
        preprocess_pipeline=preprocess_pipeline,
        has_repro=bool(repro_files),
    )
    readme_tmp = Path(tempfile.mktemp(suffix=".md", prefix="pb_readme_"))
    readme_tmp.write_text(readme_content, encoding="utf-8")
    manifest.append((readme_tmp, "README.md"))

    # --- Print manifest ---
    total_bytes = sum(p.stat().st_size for p, _ in manifest)
    print("Files to upload:")
    print(f"  {'File':<45} {'Size':>10}")
    print(f"  {'-'*45} {'-'*10}")
    for local_path, dest in manifest:
        size_mb = local_path.stat().st_size / 1024 / 1024
        src_label = f"{dest}"
        print(f"  {src_label:<45} {size_mb:>9.1f}M")
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
    msg = commit_message or f"Upload DONUT checkpoint {ckpt.name} via PechaBridge"
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
    print(f"\nTo use it with PechaBridge:")
    print(f"  python cli.py batch-ocr \\")
    print(f"      --ocr-model {repo_id} \\")
    print(f"      --input-dir /path/to/images \\")
    print(f"      --engine donut")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Upload a DONUT OCR checkpoint to HuggingFace Hub.\n\n"
            "Automatically collects model weights, tokenizer, image processor,\n"
            "and repro bundle. Each file is uploaded with a tqdm progress bar."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload with HF token:
  python scripts/upload_donut_to_hf.py \\
      --checkpoint ./models/donut-ocr/checkpoint-5000 \\
      --repo-id myuser/tibetan-donut-ocr \\
      --token hf_xxxxxxxxxxxx

  # Private repo:
  python scripts/upload_donut_to_hf.py \\
      --checkpoint ./models/donut-ocr/checkpoint-5000 \\
      --repo-id myuser/tibetan-donut-ocr \\
      --token hf_xxxxxxxxxxxx \\
      --private

  # Dry-run (show what would be uploaded):
  python scripts/upload_donut_to_hf.py \\
      --checkpoint ./models/donut-ocr/checkpoint-5000 \\
      --repo-id myuser/tibetan-donut-ocr \\
      --token hf_xxxxxxxxxxxx \\
      --dry-run

  # Token from environment variable HF_TOKEN:
  export HF_TOKEN=hf_xxxxxxxxxxxx
  python scripts/upload_donut_to_hf.py \\
      --checkpoint ./models/donut-ocr/checkpoint-5000 \\
      --repo-id myuser/tibetan-donut-ocr
""",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        metavar="PATH",
        help=(
            "Path to the DONUT checkpoint directory (e.g. ./models/donut-ocr/checkpoint-5000). "
            "Must contain config.json and model weights. "
            "Tokenizer/image_processor are searched in repro/, parent dir, or checkpoint root."
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
        # Treat password as token (HF no longer supports plain password auth)
        token = str(args.password or "").strip()

    if not token and not args.dry_run:
        print(
            "ERROR: No HuggingFace token provided.\n"
            "Use --token hf_xxx, set HF_TOKEN env var, or use --username + --password.",
            file=sys.stderr,
        )
        return 1

    upload_donut_checkpoint(
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
