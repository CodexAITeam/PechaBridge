#!/usr/bin/env python3
"""Batch OCR CLI command for PechaBridge.

Iterates over all images in a Pecha folder, runs layout detection + DONUT OCR,
and writes per-image text files plus a repro.yaml for full reproducibility.

Usage example::

    python cli.py batch-ocr \\
        --ocr-model  models/ocr/my_run/checkpoint-5000 \\
        --layout-model models/layout/yolo_layout.pt \\
        --input-dir  /data/pechas/W1234 \\
        --device     cuda:0

Output is written to a subfolder of the input directory's parent::

    /data/pechas/W1234__checkpoint-5000/
        repro.yaml
        image_001.txt
        image_002.txt
        ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger("pechabridge.cli.batch_ocr")

# ---------------------------------------------------------------------------
# Image extensions we consider
# ---------------------------------------------------------------------------
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Repro-pack helpers (mirrors ui_ocr_workbench._read_repro_preprocess_pipeline)
# ---------------------------------------------------------------------------

def _read_repro_preprocess_pipeline(ckpt_path: str) -> Optional[str]:
    """Read the image_preprocess pipeline name from a repro checkpoint bundle.

    Returns the pipeline string (e.g. ``'bdrc'``, ``'gray'``, ``'rgb'``) or
    ``None`` if the checkpoint has no repro bundle.
    """
    try:
        repro = Path(ckpt_path) / "repro" / "image_preprocess.json"
        if repro.exists():
            data = json.loads(repro.read_text(encoding="utf-8"))
            pipeline = str(data.get("pipeline", "") or "").strip().lower()
            if pipeline and pipeline != "none":
                return pipeline
    except Exception:
        pass
    return None


def _normalize_preprocess_preset(value: str) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"bdrc_no_bin", "grey"}:
        mode = "gray"
    if mode in {"rgb_lines", "rgb_line"}:
        mode = "rgb"
    if mode not in {"bdrc", "gray", "rgb"}:
        mode = "bdrc"
    return mode


# ---------------------------------------------------------------------------
# Preprocessing helpers (mirrors ui_ocr_workbench._apply_workbench_preprocess)
# ---------------------------------------------------------------------------

def _apply_preprocess(image: Any, pipeline: str) -> Any:
    """Apply the preprocessing pipeline to a PIL image.

    Loads the appropriate config class and applies ``vit_defaults()``, which
    exactly matches the training-time configuration stored in the repro bundle.
    """
    from PIL import Image as PILImage

    rgb = image.convert("RGB")
    mode = _normalize_preprocess_preset(pipeline)

    if mode in {"bdrc", "gray"}:
        try:
            from pechabridge.ocr.preprocess_bdrc import (
                BDRCPreprocessConfig,
                preprocess_image_bdrc,
            )
            cfg = BDRCPreprocessConfig.vit_defaults()
            if mode == "gray":
                # Gray preset: same as bdrc but without binarization
                from dataclasses import replace
                cfg = replace(cfg, binarize=False, gray_mode="min_rgb")
            return preprocess_image_bdrc(image=rgb, config=cfg).convert("RGB")
        except Exception as exc:
            LOGGER.warning("BDRC preprocess failed (%s), returning raw RGB.", exc)
            return rgb

    if mode == "rgb":
        try:
            from pechabridge.ocr.preprocess_rgb import (
                RGBLinePreprocessConfig,
                preprocess_image_rgb_lines,
            )
            cfg = RGBLinePreprocessConfig.vit_defaults()
            return preprocess_image_rgb_lines(image=rgb, config=cfg).convert("RGB")
        except Exception as exc:
            LOGGER.warning("RGB preprocess failed (%s), returning raw RGB.", exc)
            return rgb

    return rgb


# ---------------------------------------------------------------------------
# DONUT runtime loader
# ---------------------------------------------------------------------------

def _load_donut_runtime(
    ckpt_path: str,
    device: str,
) -> Dict[str, Any]:
    """Load a DONUT VisionEncoderDecoder runtime from a checkpoint directory.

    Supports both repro-bundle checkpoints (tokenizer/image_processor under
    ``repro/``) and plain checkpoints (tokenizer/image_processor next to the
    checkpoint or in the parent directory).

    Uses ``_load_ved_model_robust`` from the training script to handle
    meta-device tensors that newer transformers versions can leave behind
    (e.g. unused encoder pooler), which would otherwise crash when the model
    is moved to CUDA.

    Returns a dict with keys: ``model``, ``tokenizer``, ``image_processor``,
    ``device``, ``generate_config``.
    """
    from transformers import AutoImageProcessor, AutoTokenizer

    # Reuse the training-side robust loader that handles meta-device tensors,
    # sinusoidal buffer registration, and shape-mismatch retries.
    from scripts.train_donut_ocr import (
        _load_tokenizer_robust,
        _load_ved_model_robust,
        _paired_end_token_for_start,
        _prepare_model_for_generate_runtime,
    )

    ckpt = Path(ckpt_path).expanduser().resolve()
    repro = ckpt / "repro"

    # --- Locate tokenizer and image_processor ---
    if (repro / "tokenizer").exists() and (repro / "image_processor").exists():
        tokenizer_dir = str(repro / "tokenizer")
        image_processor_dir = str(repro / "image_processor")
        LOGGER.info("Loading tokenizer + image_processor from repro bundle: %s", repro)
    elif (ckpt.parent / "tokenizer").exists() and (ckpt.parent / "image_processor").exists():
        tokenizer_dir = str(ckpt.parent / "tokenizer")
        image_processor_dir = str(ckpt.parent / "image_processor")
        LOGGER.info("Loading tokenizer + image_processor from parent dir: %s", ckpt.parent)
    else:
        # Fall back to loading directly from the checkpoint (HF hub style)
        tokenizer_dir = str(ckpt)
        image_processor_dir = str(ckpt)
        LOGGER.info("Loading tokenizer + image_processor from checkpoint dir: %s", ckpt)

    tokenizer = _load_tokenizer_robust(tokenizer_dir)
    image_processor = AutoImageProcessor.from_pretrained(image_processor_dir)

    # --- Load generation config ---
    generate_config: Dict[str, Any] = {}
    gen_cfg_path = repro / "generate_config.json"
    if gen_cfg_path.exists():
        try:
            generate_config = json.loads(gen_cfg_path.read_text(encoding="utf-8"))
            LOGGER.info("Loaded generate_config from repro bundle.")
        except Exception as exc:
            LOGGER.warning("Could not read generate_config.json: %s", exc)

    # --- Load model robustly (handles meta-device tensors) ---
    LOGGER.info("Loading DONUT model from %s …", ckpt)
    model = _load_ved_model_robust(str(ckpt))

    # --- Align token IDs between model config and tokenizer ---
    try:
        if tokenizer.pad_token_id is not None:
            model.config.pad_token_id = int(tokenizer.pad_token_id)
            model.generation_config.pad_token_id = int(tokenizer.pad_token_id)
        decoder_start_id = getattr(model.config, "decoder_start_token_id", None)
        if decoder_start_id is None or int(decoder_start_id) < 0:
            for cand in (
                getattr(tokenizer, "bos_token_id", None),
                getattr(tokenizer, "cls_token_id", None),
                getattr(tokenizer, "eos_token_id", None),
            ):
                if cand is not None and int(cand) >= 0:
                    decoder_start_id = int(cand)
                    break
        if decoder_start_id is not None and int(decoder_start_id) >= 0:
            model.config.decoder_start_token_id = int(decoder_start_id)
            model.generation_config.decoder_start_token_id = int(decoder_start_id)
        eos_id = getattr(model.config, "eos_token_id", None)
        if eos_id is None or int(eos_id) < 0:
            eos_candidate = None
            try:
                start_tok = tokenizer.convert_ids_to_tokens(int(decoder_start_id)) if decoder_start_id is not None else None
                end_tok = _paired_end_token_for_start(str(start_tok or ""))
                if end_tok:
                    end_id = tokenizer.convert_tokens_to_ids(end_tok)
                    unk_id = getattr(tokenizer, "unk_token_id", None)
                    if (
                        end_id is not None
                        and int(end_id) >= 0
                        and (unk_id is None or int(end_id) != int(unk_id))
                    ):
                        eos_candidate = int(end_id)
            except Exception:
                eos_candidate = None
            if eos_candidate is None and tokenizer.eos_token_id is not None:
                eos_candidate = int(tokenizer.eos_token_id)
            if eos_candidate is not None:
                model.config.eos_token_id = int(eos_candidate)
                model.generation_config.eos_token_id = int(eos_candidate)
    except Exception as exc:
        LOGGER.warning("Token ID alignment failed (non-fatal): %s", exc)

    # --- Sanitize any remaining meta/plain tensors before moving to device ---
    try:
        _prepare_model_for_generate_runtime(model)
    except Exception as exc:
        LOGGER.warning("_prepare_model_for_generate_runtime failed (non-fatal): %s", exc)

    model.eval()
    model.to(device)
    LOGGER.info("DONUT model loaded on %s.", device)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "image_processor": image_processor,
        "device": device,
        "generate_config": generate_config,
    }


# ---------------------------------------------------------------------------
# Single-crop OCR
# ---------------------------------------------------------------------------

def _ocr_crop(
    crop_rgb: Any,  # PIL Image
    runtime: Dict[str, Any],
    pipeline: str,
    max_len: int,
) -> str:
    """Preprocess a single line crop and run DONUT inference."""
    import torch

    proc_pil = _apply_preprocess(crop_rgb, pipeline)

    image_processor = runtime["image_processor"]
    tokenizer = runtime["tokenizer"]
    model = runtime["model"]
    dev = runtime["device"]
    gen_cfg = runtime["generate_config"]

    pixel_values = image_processor(images=proc_pil, return_tensors="pt").pixel_values.to(dev)

    # Build generate kwargs from repro generate_config, fall back to max_len
    gen_kwargs: Dict[str, Any] = {}
    for k in [
        "decoder_start_token_id", "bos_token_id", "eos_token_id", "pad_token_id",
        "max_length", "max_new_tokens", "min_new_tokens", "num_beams",
        "do_sample", "temperature", "top_p", "repetition_penalty",
        "no_repeat_ngram_size", "length_penalty", "early_stopping", "use_cache",
    ]:
        v = gen_cfg.get(k)
        if v is not None:
            gen_kwargs[k] = v

    # CLI --max-len overrides the repro generate_config max_length
    if max_len > 0:
        gen_kwargs["max_length"] = int(max_len)
    elif "max_length" not in gen_kwargs and "max_new_tokens" not in gen_kwargs:
        gen_kwargs["max_length"] = 160

    with torch.no_grad():
        generated = model.generate(pixel_values=pixel_values, **gen_kwargs)

    text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0] if len(generated) else ""

    # Strip any remaining special token strings
    try:
        special_toks = sorted(
            [str(t) for t in getattr(tokenizer, "all_special_tokens", []) if isinstance(t, str) and t],
            key=len,
            reverse=True,
        )
        for tok in special_toks:
            text = text.replace(tok, "")
    except Exception:
        pass

    return str(text or "").strip()


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------

def _detect_lines(
    image_np: Any,  # np.ndarray HxWx3 uint8
    layout_model_path: str,
    device: str,
) -> List[Dict[str, Any]]:
    """Run YOLO-based layout detection and return sorted line records."""
    from ui_workbench import run_tibetan_text_line_split_classical

    split_out = run_tibetan_text_line_split_classical(
        image=image_np,
        model_path=layout_model_path,
        conf=0.25,
        imgsz=1024,
        device=device if device not in {"", "auto"} else "",
        min_line_height=10,
        projection_smooth=9,
        projection_threshold_rel=0.20,
        merge_gap_px=5,
        draw_parent_boxes=True,
        detect_red_text=False,
        red_min_redness=26,
        red_min_saturation=35,
        red_column_fill_rel=0.07,
        red_merge_gap_px=14,
        red_min_width_px=18,
        draw_red_boxes=False,
    )
    # split_out = (overlay, split_status, split_json, ..., click_state)
    click_state = split_out[-1]
    if isinstance(click_state, dict):
        return list(click_state.get("line_boxes") or [])
    return []


def _normalize_box(
    box: List[int], w: int, h: int
) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(v) for v in box]
    except Exception:
        return None
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Per-image OCR
# ---------------------------------------------------------------------------

def _ocr_image(
    image_path: Path,
    runtime: Dict[str, Any],
    layout_model_path: str,
    pipeline: str,
    max_len: int,
    device: str,
) -> Tuple[str, int]:
    """Run full OCR on a single image.

    Returns ``(transcript_text, line_count)``.
    """
    import numpy as np
    from PIL import Image

    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.asarray(img_pil, dtype=np.uint8)
    h, w = img_np.shape[:2]

    line_records = _detect_lines(img_np, layout_model_path, device)
    LOGGER.info("  %s: %d line(s) detected.", image_path.name, len(line_records))

    lines: List[Tuple[int, int, str]] = []  # (y1, line_id, text)
    for rec in line_records:
        box = _normalize_box(rec.get("line_box") or [], w, h)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        crop_np = img_np[y1:y2, x1:x2]
        crop_pil = Image.fromarray(crop_np, mode="RGB")
        text = _ocr_crop(crop_pil, runtime, pipeline, max_len)
        line_id = int(rec.get("line_id", len(lines) + 1))
        lines.append((y1, line_id, text))

    # Sort top-to-bottom, then left-to-right by line_id as tiebreaker
    lines.sort(key=lambda t: (t[0], t[1]))
    transcript = "\n".join(t for _, _, t in lines)
    return transcript, len(lines)


# ---------------------------------------------------------------------------
# repro.yaml writer
# ---------------------------------------------------------------------------

def _write_repro_yaml(
    out_dir: Path,
    *,
    ocr_model: str,
    layout_model: str,
    input_dir: str,
    pipeline: str,
    pipeline_source: str,
    device: str,
    max_len: int,
    image_files: List[str],
    cli_argv: List[str],
) -> None:
    """Write a repro.yaml file documenting the full run configuration."""
    import shlex

    # Reconstruct the CLI command that was (or would be) used
    cli_cmd = "python cli.py batch-ocr " + " ".join(shlex.quote(a) for a in cli_argv)

    lines = [
        "# PechaBridge batch-ocr reproduction file",
        f"# Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "command: |",
        f"  {cli_cmd}",
        "",
        "ocr_model: " + _yaml_str(ocr_model),
        "layout_model: " + _yaml_str(layout_model),
        "input_dir: " + _yaml_str(input_dir),
        "output_dir: " + _yaml_str(str(out_dir)),
        "preprocess_pipeline: " + _yaml_str(pipeline),
        "pipeline_source: " + _yaml_str(pipeline_source),
        "device: " + _yaml_str(device),
        "max_len: " + str(max_len),
        "",
        "images:",
    ]
    for img in sorted(image_files):
        lines.append("  - " + _yaml_str(img))

    (out_dir / "repro.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _yaml_str(value: str) -> str:
    """Minimal YAML string quoting (single-quotes, escape internal single-quotes)."""
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    import numpy as np

    try:
        import torch
    except ImportError:
        LOGGER.error("PyTorch is required for batch-ocr. Install it first.")
        return 1

    ocr_model = str(args.ocr_model).strip()
    layout_model = str(args.layout_model).strip()
    input_dir = Path(args.input_dir).expanduser().resolve()
    device_pref = str(getattr(args, "device", "auto") or "auto").strip().lower()
    max_len = int(getattr(args, "max_len", 0) or 0)

    # --- Resolve device ---
    if device_pref in {"", "auto"}:
        device = "cuda:0" if (torch.cuda.is_available()) else "cpu"
    else:
        device = device_pref
    LOGGER.info("Using device: %s", device)

    # --- Validate paths ---
    if not input_dir.is_dir():
        LOGGER.error("Input directory does not exist: %s", input_dir)
        return 1
    if not Path(ocr_model).exists():
        LOGGER.error("OCR model path does not exist: %s", ocr_model)
        return 1
    if not Path(layout_model).exists():
        LOGGER.error("Layout model path does not exist: %s", layout_model)
        return 1

    # --- Detect preprocessing pipeline from repro bundle ---
    repro_pipeline = _read_repro_preprocess_pipeline(ocr_model)
    if repro_pipeline is not None:
        pipeline = _normalize_preprocess_preset(repro_pipeline)
        pipeline_source = "repro_bundle"
        LOGGER.info(
            "Preprocessing pipeline loaded from repro bundle: %s (from %s/repro/image_preprocess.json)",
            pipeline,
            Path(ocr_model).name,
        )
    else:
        pipeline = "bdrc"
        pipeline_source = "default_fallback"
        LOGGER.warning(
            "No repro bundle found at %s/repro/image_preprocess.json — "
            "falling back to default pipeline: %s",
            Path(ocr_model).name,
            pipeline,
        )

    # --- Collect images ---
    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )
    if not image_files:
        LOGGER.error("No image files found in %s", input_dir)
        return 1
    LOGGER.info("Found %d image(s) in %s", len(image_files), input_dir)

    # --- Build output directory: <parent>/<input_dir_name>__<model_name> ---
    model_name = Path(ocr_model).name
    out_dir_name = f"{input_dir.name}__{model_name}"
    out_dir = input_dir.parent / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Output directory: %s", out_dir)

    # --- Load DONUT runtime ---
    LOGGER.info("Loading OCR model from %s …", ocr_model)
    try:
        runtime = _load_donut_runtime(ocr_model, device)
    except Exception as exc:
        LOGGER.error("Failed to load OCR model: %s", exc)
        return 1

    # --- Write repro.yaml ---
    # Reconstruct the CLI args that were passed (best-effort)
    cli_argv_repro = [
        "--ocr-model", ocr_model,
        "--layout-model", layout_model,
        "--input-dir", str(input_dir),
        "--device", device,
    ]
    if max_len > 0:
        cli_argv_repro += ["--max-len", str(max_len)]

    _write_repro_yaml(
        out_dir,
        ocr_model=ocr_model,
        layout_model=layout_model,
        input_dir=str(input_dir),
        pipeline=pipeline,
        pipeline_source=pipeline_source,
        device=device,
        max_len=max_len,
        image_files=[p.name for p in image_files],
        cli_argv=cli_argv_repro,
    )
    LOGGER.info("repro.yaml written to %s", out_dir / "repro.yaml")

    # --- Process each image ---
    total_lines = 0
    errors = 0
    for i, img_path in enumerate(image_files, start=1):
        LOGGER.info("[%d/%d] Processing %s …", i, len(image_files), img_path.name)
        n_lines = 0
        try:
            transcript, n_lines = _ocr_image(
                img_path,
                runtime=runtime,
                layout_model_path=layout_model,
                pipeline=pipeline,
                max_len=max_len,
                device=device,
            )
            total_lines += n_lines
        except Exception as exc:
            LOGGER.error("  Error processing %s: %s", img_path.name, exc)
            transcript = f"[ERROR: {exc}]"
            errors += 1

        out_txt = out_dir / (img_path.stem + ".txt")
        out_txt.write_text(transcript, encoding="utf-8")
        LOGGER.info("  → %s (%d line(s))", out_txt.name, n_lines)

    LOGGER.info(
        "Done. %d image(s) processed, %d total line(s), %d error(s). Output: %s",
        len(image_files),
        total_lines,
        errors,
        out_dir,
    )
    return 0 if errors == 0 else 1


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="batch-ocr",
        description=(
            "Batch OCR a folder of Pecha images using a DONUT OCR model and a YOLO layout model.\n\n"
            "The preprocessing pipeline is automatically loaded from the OCR model's repro bundle "
            "(repro/image_preprocess.json). If no repro bundle is present, the 'bdrc' pipeline is "
            "used as a fallback.\n\n"
            "Output is written to a subfolder of the input directory's parent, named "
            "<input_dir_name>__<ocr_model_name>/. Each image produces a .txt file with the "
            "transcribed text (one line per detected text line). A repro.yaml file is also written "
            "documenting the full run configuration and the CLI command for reproduction."
        ),
        add_help=add_help,
    )
    parser.add_argument(
        "--ocr-model",
        "--ocr_model",
        dest="ocr_model",
        type=str,
        required=True,
        help=(
            "Path to the DONUT OCR checkpoint directory. "
            "Must contain config.json and model weights. "
            "If a repro/ subdirectory is present, the preprocessing pipeline and "
            "tokenizer/image_processor are loaded from it automatically."
        ),
    )
    parser.add_argument(
        "--layout-model",
        "--layout_model",
        dest="layout_model",
        type=str,
        required=True,
        help="Path to the YOLO layout model file (.pt / .onnx / .torchscript).",
    )
    parser.add_argument(
        "--input-dir",
        "--input_dir",
        dest="input_dir",
        type=str,
        required=True,
        help=(
            "Directory containing Pecha images to OCR. "
            "Supported formats: jpg, jpeg, png, tif, tiff, bmp, webp. "
            "Files are processed in sorted order."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "Inference device. 'auto' selects cuda:0 if available, otherwise cpu. "
            "Examples: 'cpu', 'cuda:0', 'cuda:1'. Default: auto."
        ),
    )
    parser.add_argument(
        "--max-len",
        "--max_len",
        dest="max_len",
        type=int,
        default=0,
        help=(
            "Override the DONUT generation max_length. "
            "0 means use the value from the repro bundle's generate_config.json, "
            "or fall back to 160 if not present. Default: 0 (use repro config)."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _parser = create_parser()
    _args = _parser.parse_args()
    sys.exit(run(_args))
