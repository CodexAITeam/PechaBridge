#!/usr/bin/env python3
"""Download PechaBridge OCR, Line Segmentation, and Dual Encoder models from HuggingFace Hub.

Downloads models into the exact directory structure expected by:
- ui_ocr_workbench.py  (auto-scans models/ocr/ and models/line_segmentation/)
- ui_workbench.py      (scans models/line_segmentation/)
- cli.py batch-ocr     (--ocr-model, --line-model)
- Semantic Search Workbench (huggingface_model_id in YAML config)

Target layout after download
-----------------------------
models/
  ocr/
    PechaBridgeOCR/          ← DONUT checkpoint (HF snapshot)
      config.json
      model.safetensors
      generation_config.json
      preprocessor_config.json
      tokenizer_config.json
      sentencepiece.bpe.model
      special_tokens_map.json
      repro/
        generate_config.json
        image_preprocess.json
        ...
  line_segmentation/
    PechaBridgeLineSegmentation.pt   ← YOLO .pt file
  encoders/
    PechaBridgeDualEncoder/          ← Dual image-text encoder (HF snapshot)
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

Usage
-----
    # Download all models (OCR + line segmentation + dual encoder):
    python scripts/download_pechabridge_models.py

    # Download only OCR model:
    python scripts/download_pechabridge_models.py --models ocr

    # Download only line segmentation model:
    python scripts/download_pechabridge_models.py --models line

    # Download only dual encoder:
    python scripts/download_pechabridge_models.py --models encoder

    # Download OCR + encoder (no line segmentation):
    python scripts/download_pechabridge_models.py --models ocr,encoder

    # Custom destination:
    python scripts/download_pechabridge_models.py --dest /path/to/models

    # Force re-download even if already present:
    python scripts/download_pechabridge_models.py --force

    # Use HF token (for private repos):
    python scripts/download_pechabridge_models.py --token hf_xxxxxxxxxxxx
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_ROOT = REPO_ROOT / "models"

OCR_HF_REPO = "TibetanCodexAITeam/PechaBridgeOCR"
LINE_SEG_HF_REPO = "TibetanCodexAITeam/PechaBridgeLineSegmentation"
DUAL_ENCODER_HF_REPO = "TibetanCodexAITeam/PechaBridgeDualEncoder"

# The .pt file name inside the line segmentation HF repo.
# snapshot_download fetches the whole repo; we then look for the .pt file.
LINE_SEG_PT_GLOB = "*.pt"

# Sub-directory name used locally for the dual encoder.
DUAL_ENCODER_LOCAL_NAME = "PechaBridgeDualEncoder"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_hf_hub():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print(
            "ERROR: huggingface_hub is not installed.\n"
            "Install it with:  pip install huggingface-hub",
            file=sys.stderr,
        )
        sys.exit(1)


def _snapshot(repo_id: str, dest: Path, *, token: Optional[str], force: bool) -> Path:
    """Download a full HF repo snapshot into dest/. Returns the snapshot path."""
    from huggingface_hub import snapshot_download

    # huggingface_hub caches by default; local_dir forces a copy to our target.
    kwargs = dict(
        repo_id=repo_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )
    if token:
        kwargs["token"] = token

    if dest.exists() and not force:
        # Check if it looks complete (has config.json for models)
        if (dest / "config.json").exists() or list(dest.glob("*.pt")):
            print(f"  Already present at {dest}  (use --force to re-download)")
            return dest

    print(f"  Downloading {repo_id} → {dest} …")
    snapshot_download(**kwargs)
    return dest


def _find_pt_file(directory: Path) -> Optional[Path]:
    """Find the first .pt file in a directory (recursively)."""
    candidates = sorted(directory.rglob("*.pt"))
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Download: DONUT OCR model
# ---------------------------------------------------------------------------

def download_ocr_model(
    models_root: Path,
    *,
    token: Optional[str],
    force: bool,
) -> Path:
    """Download the DONUT OCR model into models/ocr/PechaBridgeOCR/."""
    ocr_dir = models_root / "ocr" / "PechaBridgeOCR"
    ocr_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[OCR Model]  {OCR_HF_REPO}")
    _snapshot(OCR_HF_REPO, ocr_dir, token=token, force=force)

    # Verify the result looks usable
    if not (ocr_dir / "config.json").exists():
        print(
            f"WARNING: config.json not found in {ocr_dir}. "
            "The download may be incomplete.",
            file=sys.stderr,
        )
    else:
        print(f"  ✓ OCR model ready at: {ocr_dir}")

    # Print repro bundle status
    repro = ocr_dir / "repro"
    if (repro / "image_preprocess.json").exists():
        import json
        try:
            data = json.loads((repro / "image_preprocess.json").read_text(encoding="utf-8"))
            pipeline = data.get("pipeline", "?")
            print(f"  ✓ Repro bundle present  (preprocessing pipeline: {pipeline})")
        except Exception:
            print(f"  ✓ Repro bundle present")
    else:
        print(
            f"  ⚠ No repro/image_preprocess.json found — "
            "preprocessing pipeline will fall back to 'bdrc'."
        )

    return ocr_dir


# ---------------------------------------------------------------------------
# Download: Line Segmentation model
# ---------------------------------------------------------------------------

def download_line_segmentation_model(
    models_root: Path,
    *,
    token: Optional[str],
    force: bool,
) -> Path:
    """Download the YOLO line segmentation model into models/line_segmentation/."""
    line_dir = models_root / "line_segmentation"
    line_dir.mkdir(parents=True, exist_ok=True)

    # Target .pt path
    target_pt = line_dir / "PechaBridgeLineSegmentation.pt"

    print(f"\n[Line Segmentation Model]  {LINE_SEG_HF_REPO}")

    if target_pt.exists() and not force:
        print(f"  Already present at {target_pt}  (use --force to re-download)")
        return target_pt

    # Download into a temp staging dir, then copy the .pt file
    import tempfile
    with tempfile.TemporaryDirectory(prefix="pb-line-seg-dl-") as tmp_s:
        tmp = Path(tmp_s)
        _snapshot(LINE_SEG_HF_REPO, tmp, token=token, force=True)

        pt_file = _find_pt_file(tmp)
        if pt_file is None:
            print(
                f"ERROR: No .pt file found in downloaded repo {LINE_SEG_HF_REPO}.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"  Copying {pt_file.name} → {target_pt} …")
        shutil.copy2(pt_file, target_pt)

    print(f"  ✓ Line segmentation model ready at: {target_pt}")
    return target_pt


# ---------------------------------------------------------------------------
# Download: Dual Image-Text Encoder
# ---------------------------------------------------------------------------

def download_dual_encoder(
    models_root: Path,
    *,
    token: Optional[str],
    force: bool,
) -> Path:
    """Download the dual encoder into models/encoders/PechaBridgeDualEncoder/.

    The HF repo layout is:
        text_encoder/   ← ByT5 text encoder + tokenizer
        vit_backbone/   ← DINOv2 image encoder + preprocessor
        training_config.json

    After download the Semantic Search Workbench can be pointed at:
        models/encoders/PechaBridgeDualEncoder/text_encoder
    """
    encoder_dir = models_root / "encoders" / DUAL_ENCODER_LOCAL_NAME
    encoder_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Dual Encoder]  {DUAL_ENCODER_HF_REPO}")
    _snapshot(DUAL_ENCODER_HF_REPO, encoder_dir, token=token, force=force)

    # Verify the result looks usable
    text_enc_dir = encoder_dir / "text_encoder"
    vit_dir = encoder_dir / "vit_backbone"

    if not (text_enc_dir / "config.json").exists():
        print(
            f"WARNING: text_encoder/config.json not found in {encoder_dir}. "
            "The download may be incomplete.",
            file=sys.stderr,
        )
    else:
        print(f"  ✓ Text encoder ready at : {text_enc_dir}")

    if not (vit_dir / "config.json").exists():
        print(
            f"WARNING: vit_backbone/config.json not found in {encoder_dir}. "
            "The download may be incomplete.",
            file=sys.stderr,
        )
    else:
        print(f"  ✓ ViT backbone ready at : {vit_dir}")

    training_cfg = encoder_dir / "training_config.json"
    if training_cfg.exists():
        import json
        try:
            data = json.loads(training_cfg.read_text(encoding="utf-8"))
            metrics = data.get("last_val_metrics", {})
            i2t_r1 = metrics.get("val_i2t_r1")
            t2i_r1 = metrics.get("val_t2i_r1")
            if i2t_r1 is not None and t2i_r1 is not None:
                print(
                    f"  ✓ Training config present  "
                    f"(i2t R@1={i2t_r1*100:.2f}%  t2i R@1={t2i_r1*100:.2f}%)"
                )
            else:
                print(f"  ✓ Training config present")
        except Exception:
            print(f"  ✓ Training config present")
    else:
        print("  ⚠ No training_config.json found.")

    return encoder_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def download_models(
    models_root: Path,
    *,
    which: List[str],
    token: Optional[str],
    force: bool,
) -> dict:
    """Download the requested models. Returns dict with result paths."""
    _require_hf_hub()

    results = {}

    if "ocr" in which:
        results["ocr"] = download_ocr_model(models_root, token=token, force=force)

    if "line" in which:
        results["line"] = download_line_segmentation_model(models_root, token=token, force=force)

    if "encoder" in which:
        results["encoder"] = download_dual_encoder(models_root, token=token, force=force)

    return results


def _print_usage_hint(results: dict) -> None:
    print("\n" + "=" * 60)
    print("  Models downloaded. Usage examples:")
    print("=" * 60)

    ocr_path = results.get("ocr")
    line_path = results.get("line")
    encoder_path = results.get("encoder")

    if ocr_path and line_path:
        print("\n  # Batch OCR (DONUT + YOLO line segmentation):")
        print(f"  python cli.py batch-ocr \\")
        print(f"      --ocr-model     {ocr_path} \\")
        print(f"      --line-model    {line_path} \\")
        print(f"      --layout-engine yolo_line \\")
        print(f"      --ocr-engine    donut \\")
        print(f"      --input-dir     /path/to/images")

    if ocr_path:
        print(f"\n  # OCR model path for --ocr-model:")
        print(f"  {ocr_path}")

    if line_path:
        print(f"\n  # Line segmentation model path for --line-model:")
        print(f"  {line_path}")

    if encoder_path:
        text_enc_path = encoder_path / "text_encoder"
        print(f"\n  # Dual encoder downloaded to:")
        print(f"  {encoder_path}")
        print(f"\n  # Use the text encoder in the Semantic Search Workbench by setting")
        print(f"  # huggingface_model_id in your workbench YAML config to:")
        print(f"  #   {text_enc_path}")
        print(f"  # or the HF repo subfolder:")
        print(f"  #   {DUAL_ENCODER_HF_REPO}/text_encoder")

    print(f"\n  # The UI workbenches (ui_ocr_workbench.py, ui_workbench.py)")
    print(f"  # will auto-detect OCR and line segmentation models on startup.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        add_help=add_help,
        description=(
            "Download PechaBridge OCR, Line Segmentation, and Dual Encoder models from HuggingFace.\n\n"
            "Places models in the exact directory structure expected by the UI workbenches,\n"
            "CLI tools, and the Semantic Search Workbench."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models downloaded:
  OCR:              TibetanCodexAITeam/PechaBridgeOCR
                    → models/ocr/PechaBridgeOCR/
  Line Segmentation: TibetanCodexAITeam/PechaBridgeLineSegmentation
                    → models/line_segmentation/PechaBridgeLineSegmentation.pt
  Dual Encoder:     TibetanCodexAITeam/PechaBridgeDualEncoder
                    → models/encoders/PechaBridgeDualEncoder/
                       text_encoder/   (use this path in semantic_search.yaml)
                       vit_backbone/

Examples:
  # Download all models (default):
  python scripts/download_pechabridge_models.py

  # Download only OCR model:
  python scripts/download_pechabridge_models.py --models ocr

  # Download only line segmentation model:
  python scripts/download_pechabridge_models.py --models line

  # Download only dual encoder:
  python scripts/download_pechabridge_models.py --models encoder

  # Download OCR + encoder (no line segmentation):
  python scripts/download_pechabridge_models.py --models ocr,encoder

  # Custom destination root:
  python scripts/download_pechabridge_models.py --dest /data/models

  # Force re-download:
  python scripts/download_pechabridge_models.py --force

  # Private repo token:
  python scripts/download_pechabridge_models.py --token hf_xxxxxxxxxxxx
""",
    )

    parser.add_argument(
        "--models",
        default="all",
        metavar="SELECTION",
        help=(
            "Which models to download. "
            "Comma-separated list of: ocr, line, encoder. "
            "Use 'all' to download everything (default: all)."
        ),
    )
    parser.add_argument(
        "--dest",
        default=str(DEFAULT_MODELS_ROOT),
        metavar="PATH",
        help=(
            f"Root models directory (default: {DEFAULT_MODELS_ROOT}). "
            "OCR model goes into <dest>/ocr/PechaBridgeOCR/, "
            "line segmentation into <dest>/line_segmentation/."
        ),
    )
    parser.add_argument(
        "--token",
        default="",
        metavar="HF_TOKEN",
        help=(
            "HuggingFace API token for private repos. "
            "Can also be set via the HF_TOKEN environment variable."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if models are already present.",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    # Resolve token
    token = str(args.token or "").strip() or str(os.environ.get("HF_TOKEN", "") or "").strip() or None

    # Resolve which models to download
    raw = str(args.models or "all").strip().lower()
    if raw == "all":
        which = ["ocr", "line", "encoder"]
    else:
        which = [x.strip() for x in raw.split(",") if x.strip()]
        invalid = set(which) - {"ocr", "line", "encoder"}
        if invalid:
            print(
                f"ERROR: Unknown model selection: {sorted(invalid)}. "
                "Use 'ocr', 'line', 'encoder', or 'all'.",
                file=sys.stderr,
            )
            return 1

    models_root = Path(args.dest).expanduser().resolve()

    print(f"\n{'='*60}")
    print(f"  PechaBridge Model Download")
    print(f"{'='*60}")
    print(f"  Destination : {models_root}")
    print(f"  Models      : {', '.join(which)}")
    print(f"  Force       : {args.force}")
    print(f"{'='*60}")

    results = download_models(models_root, which=which, token=token, force=args.force)
    _print_usage_hint(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
