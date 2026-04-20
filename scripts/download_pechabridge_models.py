#!/usr/bin/env python3
"""Download PechaBridge OCR and Line Segmentation models from HuggingFace Hub.

Downloads models into the exact directory structure expected by:
- ui_ocr_workbench.py  (auto-scans models/ocr/ and models/line_segmentation/)
- ui_workbench.py      (scans models/line_segmentation/)
- cli.py batch-ocr     (--ocr-model, --line-model)

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

Usage
-----
    # Download both (default):
    python scripts/download_pechabridge_models.py

    # Download only OCR model:
    python scripts/download_pechabridge_models.py --models ocr

    # Download only line segmentation model:
    python scripts/download_pechabridge_models.py --models line

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

# The .pt file name inside the line segmentation HF repo.
# snapshot_download fetches the whole repo; we then look for the .pt file.
LINE_SEG_PT_GLOB = "*.pt"


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

    return results


def _print_usage_hint(results: dict) -> None:
    print("\n" + "=" * 60)
    print("  Models downloaded. Usage examples:")
    print("=" * 60)

    ocr_path = results.get("ocr")
    line_path = results.get("line")

    if ocr_path and line_path:
        print("\n  # Batch OCR (DONUT + YOLO line segmentation):")
        print(f"  python cli.py batch-ocr \\")
        print(f"      --ocr-model    {ocr_path} \\")
        print(f"      --line-model   {line_path} \\")
        print(f"      --layout-engine yolo_line \\")
        print(f"      --engine donut \\")
        print(f"      --input-dir /path/to/images")

    if ocr_path:
        print(f"\n  # OCR model path for --ocr-model:")
        print(f"  {ocr_path}")

    if line_path:
        print(f"\n  # Line segmentation model path for --line-model:")
        print(f"  {line_path}")

    print(f"\n  # The UI workbenches (ui_ocr_workbench.py, ui_workbench.py)")
    print(f"  # will auto-detect these models on startup.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download PechaBridge OCR and Line Segmentation models from HuggingFace.\n\n"
            "Places models in the exact directory structure expected by the UI workbenches\n"
            "and CLI tools (models/ocr/ and models/line_segmentation/)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models downloaded:
  OCR:              TibetanCodexAITeam/PechaBridgeOCR
                    → models/ocr/PechaBridgeOCR/
  Line Segmentation: TibetanCodexAITeam/PechaBridgeLineSegmentation
                    → models/line_segmentation/PechaBridgeLineSegmentation.pt

Examples:
  # Download both models (default):
  python scripts/download_pechabridge_models.py

  # Download only OCR model:
  python scripts/download_pechabridge_models.py --models ocr

  # Download only line segmentation model:
  python scripts/download_pechabridge_models.py --models line

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
            "Comma-separated list of: ocr, line. "
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
        which = ["ocr", "line"]
    else:
        which = [x.strip() for x in raw.split(",") if x.strip()]
        invalid = set(which) - {"ocr", "line"}
        if invalid:
            print(f"ERROR: Unknown model selection: {sorted(invalid)}. Use 'ocr', 'line', or 'all'.", file=sys.stderr)
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
