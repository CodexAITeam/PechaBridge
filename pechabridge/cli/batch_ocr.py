#!/usr/bin/env python3
"""Batch OCR CLI command for PechaBridge.

Iterates over all images in a Pecha folder, runs layout detection + OCR,
and writes per-image text files plus a repro.yaml for full reproducibility.

Supported engines:
  - ``donut``      (default) — DONUT VisionEncoderDecoder model
  - ``bdrc_ocr``   — BDRC-style ONNX OCR model
  - ``tesseract``  — pytesseract, useful for baseline comparison

Usage examples::

    # DONUT engine (default)
    python cli.py batch-ocr \\
        --ocr-model    models/ocr/my_run/checkpoint-5000 \\
        --layout-model models/layout/yolo_layout.pt \\
        --input-dir    /data/pechas/W1234 \\
        --device       cuda:0

    # Tesseract engine (no --ocr-model needed)
    python cli.py batch-ocr \\
        --engine       tesseract \\
        --layout-model models/layout/yolo_layout.pt \\
        --input-dir    /data/pechas/W1234 \\
        --tess-lang    bod

    # BDRC OCR + BDRC line model
    python cli.py batch-ocr \\
        --engine          bdrc_ocr \\
        --layout-engine   bdrc_line \\
        --bdrc-ocr-model  models/bdrc/ocr/Woodblock \\
        --bdrc-line-model models/bdrc/Models/Lines \\
        --input-dir       /data/pechas/W1234

Output is written to a subfolder of the input directory's parent::

    # DONUT:
    /data/pechas/W1234__checkpoint-5000/
        repro.yaml
        image_001.txt
        image_001_overlay.jpg   ← line-box overlay (saved by default)
        ...

    # BDRC OCR:
    /data/pechas/W1234__bdrc_ocr_Woodblock/
        repro.yaml
        image_001.txt
        image_001_overlay.jpg
        ...

    # Tesseract:
    /data/pechas/W1234__tesseract_bod/
        repro.yaml
        image_001.txt
        image_001_overlay.jpg
        ...

Pass ``--no-save-overlay`` to skip saving the overlay images.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("pechabridge.cli.batch_ocr")
REPO_ROOT = Path(__file__).resolve().parents[2]

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
    from transformers import AutoImageProcessor

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
# Tesseract runtime loader
# ---------------------------------------------------------------------------

def _load_tesseract_runtime(
    lang: str,
    psm: int,
    oem: int,
    tesseract_cmd: Optional[str],
) -> Dict[str, Any]:
    """Initialise a TesseractBackend and return a runtime dict.

    Returns a dict with keys: ``backend``, ``lang_used``.
    """
    from pechabridge.ocr.backends.tesseract_backend import TesseractBackend

    backend = TesseractBackend(
        tesseract_cmd=tesseract_cmd or None,
        lang=lang,
        oem=oem,
        psm=psm,
    )
    LOGGER.info(
        "Tesseract backend ready. lang_requested=%s lang_used=%s oem=%d psm=%d",
        lang,
        backend.lang_used,
        oem,
        psm,
    )
    return {
        "backend": backend,
        "lang_used": backend.lang_used,
    }


# ---------------------------------------------------------------------------
# BDRC OCR runtime loader
# ---------------------------------------------------------------------------

def _load_bdrc_ocr_runtime(
    model_path: str,
    device: str,
    target_encoding: str,
) -> Dict[str, Any]:
    from pechabridge.ocr.bdrc_inference import resolve_bdrc_ocr_model_config

    cfg = resolve_bdrc_ocr_model_config(model_path)
    LOGGER.info(
        "BDRC OCR backend ready. model=%s input=%dx%d encoder=%s",
        Path(cfg.model_dir).name,
        cfg.input_width,
        cfg.input_height,
        cfg.encoder,
    )
    return {
        "model_path": str(model_path),
        "resolved_model_dir": str(cfg.model_dir),
        "device": str(device),
        "target_encoding": str(target_encoding or "unicode"),
    }


def _ensure_default_bdrc_line_model_dir() -> Tuple[str, str]:
    from pechabridge.ocr.bdrc_model_download import ensure_default_bdrc_line_assets

    result = ensure_default_bdrc_line_assets(REPO_ROOT / "models" / "bdrc")
    chosen = str(result.line_dir)
    if result.downloaded_items:
        note = f"Auto-downloaded default BDRC line assets ({', '.join(result.downloaded_items)}) to {result.root}"
    else:
        note = f"Using existing default BDRC line assets from {result.root}"
    return chosen, note


def _ensure_default_bdrc_ocr_model_dir() -> Tuple[str, str]:
    from pechabridge.ocr.bdrc_model_download import (
        choose_default_bdrc_ocr_model_dir,
        ensure_default_bdrc_ocr_models,
    )

    result = ensure_default_bdrc_ocr_models(REPO_ROOT / "models" / "bdrc")
    chosen = str(choose_default_bdrc_ocr_model_dir(result.root))
    if result.downloaded:
        note = f"Auto-downloaded default BDRC OCR models to {result.root}; selected {Path(chosen).name}"
    else:
        note = f"Using existing default BDRC OCR model {Path(chosen).name} from {result.root}"
    return chosen, note


# ---------------------------------------------------------------------------
# Single-crop OCR — DONUT
# ---------------------------------------------------------------------------

def _ocr_crop_donut(
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
# Single-crop OCR — Tesseract
# ---------------------------------------------------------------------------

def _ocr_crop_tesseract(
    crop_rgb: Any,  # PIL Image
    runtime: Dict[str, Any],
) -> str:
    """Run Tesseract OCR on a single line crop via TesseractBackend."""
    backend = runtime["backend"]
    result = backend.ocr_image(crop_rgb, meta={})
    return str(result.text or "").strip()


def _ocr_crop_bdrc(
    crop_rgb: Any,  # PIL Image
    runtime: Dict[str, Any],
) -> str:
    """Run a BDRC OCR ONNX model on a single prepared line crop."""
    from pechabridge.ocr.bdrc_inference import run_bdrc_ocr

    text, _debug, _preview = run_bdrc_ocr(
        crop_rgb,
        model_path=str(runtime.get("model_path") or ""),
        device=str(runtime.get("device") or "auto"),
        target_encoding=str(runtime.get("target_encoding") or "unicode"),
    )
    return str(text or "").strip()


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------

def _detect_lines_classical(
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


def _detect_lines_yolo_model(
    image_np: Any,  # np.ndarray HxWx3 uint8
    line_model_path: str,
    device: str,
    line_preprocess: str,
) -> List[Dict[str, Any]]:
    from pechabridge.ocr.line_segmentation import (
        DEFAULT_LINE_SEGMENTATION_CONF,
        DEFAULT_LINE_SEGMENTATION_IMGSZ,
        normalize_line_segmentation_preprocess_pipeline,
        predict_line_regions,
    )

    src = np.asarray(image_np, dtype=np.uint8)
    h, w = src.shape[:2]
    preprocess_mode = normalize_line_segmentation_preprocess_pipeline(line_preprocess)
    predictions = predict_line_regions(
        src,
        model_path=line_model_path,
        conf=DEFAULT_LINE_SEGMENTATION_CONF,
        imgsz=DEFAULT_LINE_SEGMENTATION_IMGSZ,
        preprocess_pipeline=preprocess_mode,
        device=device,
    )
    records: List[Dict[str, Any]] = []
    for idx, pred in enumerate(predictions, start=1):
        x1, y1, x2, y2 = [int(v) for v in pred.box]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        records.append(
            {
                "line_box": [x1, y1, x2, y2],
                "line_id": idx,
                "line_polygon": [[int(px), int(py)] for px, py in pred.polygon],
                "line_confidence": float(pred.confidence),
                "line_label": str(pred.label),
                "line_class": int(pred.class_id),
            }
        )
    return records


def _detect_lines_bdrc(
    image_np: Any,  # np.ndarray HxWx3 uint8
    bdrc_line_model_path: str,
    device: str,
    k_factor: float,
    bbox_tolerance: float,
    class_threshold: float,
    merge_lines: bool,
    use_tps: bool,
    tps_threshold: float,
) -> List[Dict[str, Any]]:
    from pechabridge.ocr.bdrc_inference import predict_bdrc_line_regions

    predictions, _debug = predict_bdrc_line_regions(
        image_np,
        model_path=bdrc_line_model_path,
        device=device,
        class_threshold=class_threshold,
        group_lines=bool(merge_lines),
        k_factor=k_factor,
        bbox_tolerance=bbox_tolerance,
        use_tps=bool(use_tps),
        tps_threshold=float(tps_threshold),
    )
    records: List[Dict[str, Any]] = []
    for idx, pred in enumerate(predictions, start=1):
        rec: Dict[str, Any] = {
            "line_box": [int(v) for v in pred.box],
            "line_id": idx,
            "line_polygon": [[int(px), int(py)] for px, py in pred.polygon],
            "line_confidence": float(pred.confidence),
            "line_label": str(pred.label),
            "line_class": int(pred.class_id),
            "page_angle": float(pred.page_angle),
        }
        if pred.crop_image is not None:
            rec["ocr_crop"] = np.asarray(pred.crop_image, dtype=np.uint8)
        records.append(rec)
    return records


def _detect_lines_tesseract(
    image_pil: Any,  # PIL.Image.Image RGB
    tess_runtime: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Use Tesseract page segmentation to detect line bounding boxes.

    Calls ``pytesseract.image_to_data()`` with PSM 3 (auto page segmentation,
    no OSD) to get word-level bounding boxes, then aggregates them into
    line-level records compatible with the ``{"line_box": [x1,y1,x2,y2],
    "line_id": N}`` format expected by ``_ocr_image()``.

    Returns records sorted top-to-bottom.
    """
    try:
        import pytesseract
        from pytesseract import Output
    except ImportError:
        LOGGER.error("pytesseract is not installed. Run: pip install pytesseract")
        return []

    backend = tess_runtime.get("backend")
    lang = tess_runtime.get("lang_used", "bod")
    oem = tess_runtime.get("oem", 1)

    # PSM 3 = fully automatic page segmentation (no OSD) — best for full pages
    config = f"--oem {oem} --psm 3"
    if backend is not None and hasattr(backend, "tesseract_cmd"):
        pytesseract.pytesseract.tesseract_cmd = backend.tesseract_cmd

    try:
        data = pytesseract.image_to_data(
            image_pil,
            lang=lang,
            config=config,
            output_type=Output.DICT,
        )
    except Exception as exc:
        LOGGER.warning("Tesseract layout detection failed: %s", exc)
        return []

    # Aggregate word boxes into line boxes keyed by (block_num, par_num, line_num)
    line_map: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    n_words = len(data.get("level", []))
    for idx in range(n_words):
        level = int(data["level"][idx])
        if level != 5:  # level 5 = word
            continue
        conf_raw = data["conf"][idx]
        try:
            conf_val = float(conf_raw)
        except (TypeError, ValueError):
            conf_val = -1.0
        if conf_val < 0:
            continue  # skip non-word rows

        bx = int(data["left"][idx])
        by = int(data["top"][idx])
        bw = int(data["width"][idx])
        bh = int(data["height"][idx])
        if bw <= 0 or bh <= 0:
            continue

        key = (int(data["block_num"][idx]), int(data["par_num"][idx]), int(data["line_num"][idx]))
        if key not in line_map:
            line_map[key] = {"x1": bx, "y1": by, "x2": bx + bw, "y2": by + bh}
        else:
            rec = line_map[key]
            rec["x1"] = min(rec["x1"], bx)
            rec["y1"] = min(rec["y1"], by)
            rec["x2"] = max(rec["x2"], bx + bw)
            rec["y2"] = max(rec["y2"], by + bh)

    # Convert to sorted line records
    records: List[Dict[str, Any]] = []
    for line_id, (key, coords) in enumerate(
        sorted(line_map.items(), key=lambda kv: (kv[1]["y1"], kv[1]["x1"])),
        start=1,
    ):
        records.append({
            "line_box": [coords["x1"], coords["y1"], coords["x2"], coords["y2"]],
            "line_id": line_id,
        })

    return records


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
# Line-box overlay renderer (mirrors ui_ocr_workbench._render_overlay)
# ---------------------------------------------------------------------------

def _render_line_box_overlay(
    image_np: Any,  # np.ndarray HxWx3 uint8
    line_records: List[Dict[str, Any]],
) -> Any:  # PIL Image RGB
    """Render detected line bounding boxes onto the source image.

    Draws a cyan double-stroke rectangle and a numbered label for every
    detected line, exactly as the OCR Workbench UI does.  Returns a PIL
    ``Image`` in RGB mode so the caller can save it in any format.
    """
    from PIL import Image, ImageDraw, ImageFont

    panel = Image.fromarray(np.asarray(image_np).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(panel)

    # Load a small font; fall back to the built-in bitmap font if unavailable.
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    def _draw_strong_rect(
        box: List[int],
        color: Tuple[int, int, int],
        inner_width: int = 4,
        outer_width: int = 8,
    ) -> None:
        x1, y1, x2, y2 = [int(v) for v in box]
        draw.rectangle((x1, y1, x2, y2), outline=(0, 0, 0), width=max(2, int(outer_width)))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, int(inner_width)))

    # Sort records top-to-bottom before numbering (same order as transcript)
    sorted_records = sorted(
        line_records,
        key=lambda r: (
            int((r.get("line_box") or [0, 0, 0, 0])[1]),
            int((r.get("line_box") or [0, 0, 0, 0])[0]),
        ),
    )

    for i, rec in enumerate(sorted_records, start=1):
        box = rec.get("line_box")
        if not box or len(box) < 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        _draw_strong_rect([x1, y1, x2, y2], color=(0, 255, 255), inner_width=4, outer_width=8)
        tag = f"line {i}"
        tx1 = x1 + 2
        ty1 = max(0, y1 - 20)
        tx2 = x1 + 18 + 8 * len(tag)
        ty2 = max(16, y1 - 2)
        draw.rectangle((tx1, ty1, tx2, ty2), fill=(0, 0, 0))
        draw.rectangle((tx1, ty1, tx2, ty2), outline=(0, 255, 255), width=2)
        draw.text((tx1 + 4, ty1 + 2), tag, fill=(255, 255, 255), font=font)

    return panel


# ---------------------------------------------------------------------------
# Per-image OCR
# ---------------------------------------------------------------------------

def _ocr_image(
    image_path: Path,
    engine: str,
    runtime: Dict[str, Any],
    layout_model_path: str,
    line_model_path: str,
    bdrc_line_model_path: str,
    pipeline: str,
    max_len: int,
    device: str,
    layout_engine: str = "cv",
    line_preprocess: str = "gray",
    bdrc_line_k_factor: float = 2.5,
    bdrc_line_bbox_tolerance: float = 3.0,
    bdrc_line_class_threshold: float = 0.0,
    bdrc_line_merge_lines: bool = True,
    bdrc_line_use_tps: bool = True,
    bdrc_line_tps_threshold: float = 0.25,
) -> Tuple[str, int, Any, List[Dict[str, Any]]]:
    """Run full OCR on a single image.

    Returns ``(transcript_text, line_count, image_np, line_records)``.

    ``image_np`` is the original image as a ``np.ndarray`` (HxWx3 uint8) and
    ``line_records`` is the list of detected line dicts (each with a
    ``"line_box"`` key) so the caller can render an overlay image.

    ``layout_engine`` controls how line bounding boxes are detected:
    - ``"cv"``        (default) Classical CV segmentation via
                      ``run_tibetan_text_line_split_classical()``.
    - ``"yolo_line"`` YOLO segmentation/detection line model via
                      ``predict_line_regions()``.
    - ``"bdrc_line"`` BDRC ONNX line/layout model plus BDRC contour postprocessing.
    - ``"tesseract"`` Tesseract page segmentation (PSM 3) via
                      ``pytesseract.image_to_data()``.  The ``runtime``
                      dict must contain a ``"tess_layout_runtime"`` key
                      when the OCR engine is DONUT or BDRC OCR.
    """
    from PIL import Image

    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.asarray(img_pil, dtype=np.uint8)
    h, w = img_np.shape[:2]

    if layout_engine == "tesseract":
        # Use the dedicated Tesseract layout runtime (may differ from OCR runtime)
        tess_layout_rt = runtime.get("tess_layout_runtime") or runtime
        line_records = _detect_lines_tesseract(img_pil, tess_layout_rt)
    elif layout_engine == "yolo_line":
        line_records = _detect_lines_yolo_model(img_np, line_model_path, device, line_preprocess)
    elif layout_engine == "bdrc_line":
        line_records = _detect_lines_bdrc(
            img_np,
            bdrc_line_model_path,
            device,
            k_factor=bdrc_line_k_factor,
            bbox_tolerance=bdrc_line_bbox_tolerance,
            class_threshold=bdrc_line_class_threshold,
            merge_lines=bdrc_line_merge_lines,
            use_tps=bdrc_line_use_tps,
            tps_threshold=bdrc_line_tps_threshold,
        )
    else:
        line_records = _detect_lines_classical(img_np, layout_model_path, device)
    LOGGER.info(
        "  %s: %d line(s) detected (layout_engine=%s).",
        image_path.name, len(line_records), layout_engine,
    )

    lines: List[Tuple[int, int, str]] = []  # (y1, line_id, text)
    valid_records: List[Dict[str, Any]] = []  # records that passed box validation
    for rec in line_records:
        box = _normalize_box(rec.get("line_box") or [], w, h)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        # Store the normalised box back so the overlay uses clipped coordinates
        rec_with_norm = dict(rec)
        rec_with_norm["line_box"] = list(box)
        valid_records.append(rec_with_norm)

        crop_np = rec.get("ocr_crop")
        if not isinstance(crop_np, np.ndarray) or crop_np.size == 0:
            crop_np = img_np[y1:y2, x1:x2]
        crop_pil = Image.fromarray(crop_np, mode="RGB")

        if engine == "tesseract":
            text = _ocr_crop_tesseract(crop_pil, runtime)
        elif engine == "bdrc_ocr":
            text = _ocr_crop_bdrc(crop_pil, runtime)
        else:
            text = _ocr_crop_donut(crop_pil, runtime, pipeline, max_len)

        line_id = int(rec.get("line_id", len(lines) + 1))
        lines.append((y1, line_id, text))

    # Sort top-to-bottom, then left-to-right by line_id as tiebreaker
    lines.sort(key=lambda t: (t[0], t[1]))
    transcript = "\n".join(t for _, _, t in lines)
    return transcript, len(lines), img_np, valid_records


# ---------------------------------------------------------------------------
# repro.yaml writer
# ---------------------------------------------------------------------------

def _write_repro_yaml(
    out_dir: Path,
    *,
    engine: str,
    layout_engine: str,
    ocr_model: str,
    layout_model: str,
    input_dir: str,
    pipeline: str,
    pipeline_source: str,
    device: str,
    max_len: int,
    tess_lang: str,
    tess_psm: int,
    tess_oem: int,
    tess_lang_used: str,
    tess_layout_lang: str,
    tess_layout_psm: int,
    tess_layout_oem: int,
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
        "engine: " + _yaml_str(engine),
        "layout_engine: " + _yaml_str(layout_engine),
    ]

    if engine == "tesseract":
        lines += [
            "tesseract_lang_requested: " + _yaml_str(tess_lang),
            "tesseract_lang_used: " + _yaml_str(tess_lang_used),
            "tesseract_psm: " + str(tess_psm),
            "tesseract_oem: " + str(tess_oem),
        ]
    else:
        lines += [
            "ocr_model: " + _yaml_str(ocr_model),
            "preprocess_pipeline: " + _yaml_str(pipeline),
            "pipeline_source: " + _yaml_str(pipeline_source),
            "max_len: " + str(max_len),
        ]

    if layout_engine == "tesseract":
        lines += [
            "tesseract_layout_lang: " + _yaml_str(tess_layout_lang),
            "tesseract_layout_psm: " + str(tess_layout_psm),
            "tesseract_layout_oem: " + str(tess_layout_oem),
        ]
    else:
        lines += [
            "layout_model: " + _yaml_str(layout_model),
        ]

    lines += [
        "input_dir: " + _yaml_str(input_dir),
        "output_dir: " + _yaml_str(str(out_dir)),
        "device: " + _yaml_str(device),
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
    engine = str(getattr(args, "engine", "donut") or "donut").strip().lower()
    layout_engine = str(getattr(args, "layout_engine", "cv") or "cv").strip().lower()
    layout_model = str(getattr(args, "layout_model", "") or "").strip()
    line_model = str(getattr(args, "line_model", "") or "").strip()
    bdrc_line_model = str(getattr(args, "bdrc_line_model", "") or "").strip()
    input_dir = Path(args.input_dir).expanduser().resolve()
    device_pref = str(getattr(args, "device", "auto") or "auto").strip().lower()
    max_len = int(getattr(args, "max_len", 0) or 0)
    line_preprocess = str(getattr(args, "line_preprocess", "gray") or "gray").strip().lower()
    bdrc_line_k_factor = float(getattr(args, "bdrc_line_k_factor", 2.5) or 2.5)
    bdrc_line_bbox_tolerance = float(getattr(args, "bdrc_line_bbox_tolerance", 3.0) or 3.0)
    bdrc_line_class_threshold = float(getattr(args, "bdrc_line_class_threshold", 0.0) or 0.0)
    bdrc_line_merge_lines = bool(getattr(args, "bdrc_line_merge_lines", True))
    bdrc_line_use_tps = bool(getattr(args, "bdrc_line_use_tps", True))
    bdrc_line_tps_threshold = float(getattr(args, "bdrc_line_tps_threshold", 0.25) or 0.25)

    # Tesseract OCR-specific args
    tess_lang = str(getattr(args, "tess_lang", "bod") or "bod").strip()
    tess_psm = int(getattr(args, "tess_psm", 6) or 6)
    tess_oem = int(getattr(args, "tess_oem", 1) or 1)
    tess_cmd = str(getattr(args, "tess_cmd", "") or "").strip() or None

    # Tesseract layout-specific args (separate from OCR args)
    tess_layout_lang = str(getattr(args, "tess_layout_lang", "bod") or "bod").strip()
    tess_layout_psm = int(getattr(args, "tess_layout_psm", 3) or 3)
    tess_layout_oem = int(getattr(args, "tess_layout_oem", 1) or 1)

    # DONUT-specific args
    ocr_model = str(getattr(args, "ocr_model", "") or "").strip()
    bdrc_ocr_model = str(getattr(args, "bdrc_ocr_model", "") or "").strip()
    bdrc_ocr_target_encoding = str(getattr(args, "bdrc_ocr_target_encoding", "unicode") or "unicode").strip().lower()

    if engine == "bdrc-ocr":
        engine = "bdrc_ocr"
    if layout_engine in {"yolo", "classical", "yolo_classical"}:
        layout_engine = "cv"
    if layout_engine in {"yolo-line", "yolo_seg", "yolo-seg", "line_model"}:
        layout_engine = "yolo_line"
    if layout_engine in {"bdrc-line", "bdrc"}:
        layout_engine = "bdrc_line"

    # --- Validate engines ---
    if engine not in {"donut", "tesseract", "bdrc_ocr"}:
        LOGGER.error("Unknown engine '%s'. Choose 'donut', 'bdrc_ocr' or 'tesseract'.", engine)
        return 1
    if layout_engine not in {"cv", "yolo_line", "bdrc_line", "tesseract"}:
        LOGGER.error(
            "Unknown layout-engine '%s'. Choose 'cv', 'yolo_line', 'bdrc_line' or 'tesseract'.",
            layout_engine,
        )
        return 1

    # --- Resolve device (only needed for DONUT + YOLO layout model) ---
    if device_pref in {"", "auto"}:
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = device_pref
    LOGGER.info("Using device: %s", device)
    bdrc_auto_notes: List[str] = []

    # --- Validate paths ---
    if not input_dir.is_dir():
        LOGGER.error("Input directory does not exist: %s", input_dir)
        return 1
    if layout_engine == "cv":
        if not layout_model:
            LOGGER.error("--layout-model is required when --layout-engine yolo (default).")
            return 1
        if not Path(layout_model).exists():
            LOGGER.error("Layout model path does not exist: %s", layout_model)
            return 1
    if layout_engine == "yolo_line":
        if not line_model:
            LOGGER.error("--line-model is required when --layout-engine yolo_line.")
            return 1
        if not Path(line_model).exists():
            LOGGER.error("Line segmentation model path does not exist: %s", line_model)
            return 1
    if layout_engine == "bdrc_line":
        if not bdrc_line_model:
            try:
                bdrc_line_model, note = _ensure_default_bdrc_line_model_dir()
                bdrc_auto_notes.append(note)
                LOGGER.info("%s", note)
            except Exception as exc:
                LOGGER.error(
                    "No --bdrc-line-model was provided and auto-download failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
                return 1
        if not Path(bdrc_line_model).exists():
            LOGGER.error("BDRC line model path does not exist: %s", bdrc_line_model)
            return 1
    if engine == "donut":
        if not ocr_model:
            LOGGER.error("--ocr-model is required for engine 'donut'.")
            return 1
        if not Path(ocr_model).exists():
            LOGGER.error("OCR model path does not exist: %s", ocr_model)
            return 1
    if engine == "bdrc_ocr":
        if not bdrc_ocr_model:
            try:
                bdrc_ocr_model, note = _ensure_default_bdrc_ocr_model_dir()
                bdrc_auto_notes.append(note)
                LOGGER.info("%s", note)
            except Exception as exc:
                LOGGER.error(
                    "No --bdrc-ocr-model was provided and auto-download failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
                return 1
        if not Path(bdrc_ocr_model).exists():
            LOGGER.error("BDRC OCR model path does not exist: %s", bdrc_ocr_model)
            return 1

    # --- Detect preprocessing pipeline from repro bundle (DONUT only) ---
    pipeline = "bdrc"
    pipeline_source = "n/a"
    if engine == "donut":
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
    elif engine == "bdrc_ocr":
        pipeline = "bdrc_ocr_internal"
        pipeline_source = "bdrc_ocr_internal"

    # --- Collect images ---
    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )
    if not image_files:
        LOGGER.error("No image files found in %s", input_dir)
        return 1
    LOGGER.info("Found %d image(s) in %s", len(image_files), input_dir)

    # --- Build output directory ---
    # DONUT:     <parent>/<input_dir_name>__<model_name>[__tess_layout]
    # Tesseract: <parent>/<input_dir_name>__tesseract_<lang>[__tess_layout]
    if engine == "tesseract":
        model_slug = f"tesseract_{tess_lang}"
    elif engine == "bdrc_ocr":
        model_slug = f"bdrc_ocr_{Path(bdrc_ocr_model).name}"
    else:
        model_slug = Path(ocr_model).name
    if layout_engine == "tesseract":
        model_slug = f"{model_slug}__tess_layout"
    elif layout_engine == "yolo_line":
        model_slug = f"{model_slug}__yolo_line"
    elif layout_engine == "bdrc_line":
        model_slug = f"{model_slug}__bdrc_line"
    out_dir_name = f"{input_dir.name}__{model_slug}"
    out_dir = input_dir.parent / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Output directory: %s", out_dir)

    # --- Log resolved model configuration ---
    LOGGER.info("=" * 60)
    LOGGER.info("  batch-ocr configuration")
    LOGGER.info("  engine        : %s", engine)
    LOGGER.info("  layout_engine : %s", layout_engine)
    if layout_engine == "cv":
        LOGGER.info("  layout_model  : %s", layout_model)
    elif layout_engine == "yolo_line":
        LOGGER.info("  line_model    : %s", line_model)
    elif layout_engine == "bdrc_line":
        LOGGER.info("  bdrc_line_model : %s", bdrc_line_model)
    if engine == "donut":
        LOGGER.info("  ocr_model     : %s", ocr_model)
    elif engine == "bdrc_ocr":
        LOGGER.info("  bdrc_ocr_model: %s", bdrc_ocr_model)
    elif engine == "tesseract":
        LOGGER.info("  tess_lang     : %s", tess_lang)
    LOGGER.info("  device        : %s", device)
    LOGGER.info("  input_dir     : %s", input_dir)
    LOGGER.info("  output_dir    : %s", out_dir)
    LOGGER.info("=" * 60)

    # --- Load OCR runtime ---
    tess_lang_used = tess_lang
    if engine == "tesseract":
        LOGGER.info("Initialising Tesseract OCR backend (lang=%s, psm=%d, oem=%d) …", tess_lang, tess_psm, tess_oem)
        try:
            runtime = _load_tesseract_runtime(tess_lang, tess_psm, tess_oem, tess_cmd)
            tess_lang_used = runtime["lang_used"]
        except Exception as exc:
            LOGGER.error("Failed to initialise Tesseract backend: %s", exc)
            return 1
    elif engine == "donut":
        LOGGER.info("Loading DONUT model from %s …", ocr_model)
        try:
            import torch  # noqa: F401 — ensure torch is available for DONUT
        except ImportError:
            LOGGER.error("PyTorch is required for engine 'donut'. Install it first.")
            return 1
        try:
            runtime = _load_donut_runtime(ocr_model, device)
        except Exception as exc:
            LOGGER.error("Failed to load DONUT model: %s", exc)
            return 1
    else:
        LOGGER.info("Initialising BDRC OCR backend from %s …", bdrc_ocr_model)
        try:
            runtime = _load_bdrc_ocr_runtime(
                bdrc_ocr_model,
                device=device,
                target_encoding=bdrc_ocr_target_encoding,
            )
        except Exception as exc:
            LOGGER.error("Failed to load BDRC OCR backend: %s", exc)
            return 1

    # --- Load Tesseract layout runtime (if layout_engine == "tesseract") ---
    # When the OCR engine is DONUT but layout is Tesseract, we need a separate
    # Tesseract runtime just for page segmentation (PSM 3, no OCR output used).
    if layout_engine == "tesseract":
        LOGGER.info(
            "Initialising Tesseract layout backend (lang=%s, psm=%d, oem=%d) …",
            tess_layout_lang, tess_layout_psm, tess_layout_oem,
        )
        try:
            tess_layout_runtime = _load_tesseract_runtime(
                tess_layout_lang, tess_layout_psm, tess_layout_oem, tess_cmd
            )
        except Exception as exc:
            LOGGER.error("Failed to initialise Tesseract layout backend: %s", exc)
            return 1
        # Embed the layout runtime inside the main runtime dict so _ocr_image can find it
        runtime["tess_layout_runtime"] = tess_layout_runtime

    # --- Write repro.yaml ---
    cli_argv_repro = [
        "--engine", engine,
        "--layout-engine", layout_engine,
        "--input-dir", str(input_dir),
        "--device", device,
    ]
    if layout_engine == "cv":
        cli_argv_repro += ["--layout-model", layout_model]
    elif layout_engine == "yolo_line":
        cli_argv_repro += ["--line-model", line_model, "--line-preprocess", line_preprocess]
    elif layout_engine == "bdrc_line":
        cli_argv_repro += [
            "--bdrc-line-model", bdrc_line_model,
            "--bdrc-line-k-factor", str(bdrc_line_k_factor),
            "--bdrc-line-bbox-tolerance", str(bdrc_line_bbox_tolerance),
            "--bdrc-line-class-threshold", str(bdrc_line_class_threshold),
            "--bdrc-line-tps-threshold", str(bdrc_line_tps_threshold),
        ]
        cli_argv_repro += ["--bdrc-line-merge-lines"] if bdrc_line_merge_lines else ["--bdrc-line-no-merge-lines"]
        cli_argv_repro += ["--bdrc-line-use-tps"] if bdrc_line_use_tps else ["--bdrc-line-no-use-tps"]
    if engine == "tesseract":
        cli_argv_repro += [
            "--tess-lang", tess_lang,
            "--tess-psm", str(tess_psm),
            "--tess-oem", str(tess_oem),
        ]
        if tess_cmd:
            cli_argv_repro += ["--tess-cmd", tess_cmd]
    elif engine == "bdrc_ocr":
        cli_argv_repro += [
            "--bdrc-ocr-model", bdrc_ocr_model,
            "--bdrc-ocr-target-encoding", bdrc_ocr_target_encoding,
        ]
    else:
        cli_argv_repro += ["--ocr-model", ocr_model]
        if max_len > 0:
            cli_argv_repro += ["--max-len", str(max_len)]
    if layout_engine == "tesseract":
        cli_argv_repro += [
            "--tess-layout-lang", tess_layout_lang,
            "--tess-layout-psm", str(tess_layout_psm),
            "--tess-layout-oem", str(tess_layout_oem),
        ]

    _write_repro_yaml(
        out_dir,
        engine=engine,
        layout_engine=layout_engine,
        ocr_model=(bdrc_ocr_model if engine == "bdrc_ocr" else ocr_model),
        layout_model=layout_model,
        input_dir=str(input_dir),
        pipeline=pipeline,
        pipeline_source=pipeline_source,
        device=device,
        max_len=max_len,
        tess_lang=tess_lang,
        tess_psm=tess_psm,
        tess_oem=tess_oem,
        tess_lang_used=tess_lang_used,
        tess_layout_lang=tess_layout_lang,
        tess_layout_psm=tess_layout_psm,
        tess_layout_oem=tess_layout_oem,
        image_files=[p.name for p in image_files],
        cli_argv=cli_argv_repro,
    )
    LOGGER.info("repro.yaml written to %s", out_dir / "repro.yaml")

    save_overlay = bool(getattr(args, "save_overlay", True))

    # --- Process each image ---
    total_lines = 0
    errors = 0
    for i, img_path in enumerate(image_files, start=1):
        LOGGER.info("[%d/%d] Processing %s …", i, len(image_files), img_path.name)
        n_lines = 0
        img_np_result: Any = None
        line_records_result: List[Dict[str, Any]] = []
        try:
            transcript, n_lines, img_np_result, line_records_result = _ocr_image(
                img_path,
                engine=engine,
                runtime=runtime,
                layout_model_path=layout_model,
                line_model_path=line_model,
                bdrc_line_model_path=bdrc_line_model,
                pipeline=pipeline,
                max_len=max_len,
                device=device,
                layout_engine=layout_engine,
                line_preprocess=line_preprocess,
                bdrc_line_k_factor=bdrc_line_k_factor,
                bdrc_line_bbox_tolerance=bdrc_line_bbox_tolerance,
                bdrc_line_class_threshold=bdrc_line_class_threshold,
                bdrc_line_merge_lines=bdrc_line_merge_lines,
                bdrc_line_use_tps=bdrc_line_use_tps,
                bdrc_line_tps_threshold=bdrc_line_tps_threshold,
            )
            total_lines += n_lines
        except Exception as exc:
            LOGGER.error("  Error processing %s: %s", img_path.name, exc)
            transcript = f"[ERROR: {exc}]"
            errors += 1

        out_txt = out_dir / (img_path.stem + ".txt")
        out_txt.write_text(transcript, encoding="utf-8")
        LOGGER.info("  → %s (%d line(s))", out_txt.name, n_lines)

        # --- Save line-box overlay image ---
        if save_overlay and img_np_result is not None:
            try:
                overlay_pil = _render_line_box_overlay(img_np_result, line_records_result)
                out_overlay = out_dir / (img_path.stem + "_overlay.jpg")
                overlay_pil.save(str(out_overlay), format="JPEG", quality=92)
                LOGGER.info("  → %s (overlay)", out_overlay.name)
            except Exception as exc:
                LOGGER.warning("  Could not save overlay for %s: %s", img_path.name, exc)

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
            "Batch OCR a folder of Pecha images using a layout model + OCR engine.\n\n"
            "OCR engines (--engine):\n"
            "  donut      (default) DONUT VisionEncoderDecoder model. Preprocessing pipeline\n"
            "             is auto-detected from the repro bundle (repro/image_preprocess.json).\n"
            "  bdrc_ocr   BDRC-style ONNX OCR model with BDRC line normalization.\n"
            "  tesseract  pytesseract baseline. No --ocr-model needed.\n\n"
            "Layout engines (--layout-engine):\n"
            "  yolo       (default) Classical CV line segmentation via layout model + projections.\n"
            "             Requires --layout-model.\n"
            "  yolo_line  YOLO line segmentation model. Requires --line-model.\n"
            "  bdrc_line  BDRC ONNX line/layout model with BDRC contour postprocessing.\n"
            "             Auto-downloads the default BDRC line model when --bdrc-line-model is omitted.\n"
            "  tesseract  Tesseract page segmentation (PSM 3). No --layout-model needed.\n"
            "             Combine with --engine donut or --engine bdrc_ocr to use\n"
            "             Tesseract layout + OCR.\n\n"
            "Output is written to a subfolder of the input directory's parent:\n"
            "  DONUT+YOLO:      <input_dir_name>__<checkpoint_name>/\n"
            "  BDRC+BDRCLine:   <input_dir_name>__bdrc_ocr_<model_name>__bdrc_line/\n"
            "  DONUT+TessLayout:<input_dir_name>__<checkpoint_name>__tess_layout/\n"
            "  Tesseract:       <input_dir_name>__tesseract_<lang>/\n\n"
            "Each image produces a .txt file (one line per detected text line) and\n"
            "an *_overlay.jpg with the detected line boxes drawn on the source image\n"
            "(pass --no-save-overlay to skip the overlay images).\n"
            "A repro.yaml documents the full run config and CLI command."
        ),
        add_help=add_help,
    )

    # --- Engine selection ---
    parser.add_argument(
        "--engine",
        type=str,
        default="donut",
        choices=["donut", "bdrc_ocr", "tesseract"],
        help="OCR engine to use. 'donut' (default), 'bdrc_ocr' or 'tesseract'.",
    )
    parser.add_argument(
        "--layout-engine",
        "--layout_engine",
        dest="layout_engine",
        type=str,
        default="cv",
        choices=["cv", "yolo_line", "bdrc_line", "tesseract"],
        help=(
            "Layout detection engine used to find line bounding boxes before OCR. "
            "'cv' (default) uses the classical CV splitter driven by --layout-model. "
            "Alias: 'yolo'. "
            "'yolo_line' uses a YOLO line segmentation model specified by --line-model. "
            "'bdrc_line' uses a BDRC line/layout ONNX model specified by --bdrc-line-model. "
            "'tesseract' uses pytesseract page segmentation (PSM 3) — "
            "no --layout-model needed. "
            "Combining --layout-engine tesseract with --engine donut lets you use "
            "Tesseract's page segmentation to crop lines and then run DONUT on each crop."
        ),
    )

    # --- Shared args ---
    parser.add_argument(
        "--layout-model",
        "--layout_model",
        dest="layout_model",
        type=str,
        default="",
        help=(
            "Path to the YOLO layout model file (.pt / .onnx / .torchscript). "
            "Required when --layout-engine yolo (default classical splitter). "
            "Ignored for --layout-engine yolo_line, bdrc_line and tesseract."
        ),
    )
    parser.add_argument(
        "--line-model",
        "--line_model",
        dest="line_model",
        type=str,
        default="",
        help=(
            "Path to the trained YOLO line segmentation model (.pt / .onnx / .torchscript). "
            "Required when --layout-engine yolo_line."
        ),
    )
    parser.add_argument(
        "--line-preprocess",
        "--line_preprocess",
        dest="line_preprocess",
        type=str,
        default="gray",
        choices=["none", "bdrc", "gray", "rgb"],
        help="Preprocessing pipeline for --layout-engine yolo_line. Default: gray.",
    )
    parser.add_argument(
        "--bdrc-line-model",
        "--bdrc_line_model",
        dest="bdrc_line_model",
        type=str,
        default="",
        help=(
            "Path to a BDRC line/layout model directory (or config.json/onnx file). "
            "Optional when --layout-engine bdrc_line. If omitted, the default BDRC line model "
            "is downloaded into models/bdrc automatically."
        ),
    )
    parser.add_argument(
        "--bdrc-line-k-factor",
        "--bdrc_line_k_factor",
        dest="bdrc_line_k_factor",
        type=float,
        default=2.5,
        help="Morphological expansion factor for BDRC line crops. Default: 2.5.",
    )
    parser.add_argument(
        "--bdrc-line-bbox-tolerance",
        "--bdrc_line_bbox_tolerance",
        dest="bdrc_line_bbox_tolerance",
        type=float,
        default=3.0,
        help="Adaptive crop height tolerance for BDRC line extraction. Default: 3.0.",
    )
    parser.add_argument(
        "--bdrc-line-class-threshold",
        "--bdrc_line_class_threshold",
        dest="bdrc_line_class_threshold",
        type=float,
        default=0.0,
        help=(
            "Optional explicit class threshold for BDRC line/layout ONNX inference. "
            "0.0 means use backend defaults."
        ),
    )
    parser.add_argument(
        "--bdrc-line-merge-lines",
        "--bdrc_line_merge_lines",
        dest="bdrc_line_merge_lines",
        action="store_true",
        default=True,
        help="Merge fragmented BDRC contours into single lines before OCR. Default: enabled.",
    )
    parser.add_argument(
        "--bdrc-line-no-merge-lines",
        "--bdrc_line_no_merge_lines",
        dest="bdrc_line_merge_lines",
        action="store_false",
        help="Disable contour grouping for BDRC line detection.",
    )
    parser.add_argument(
        "--bdrc-line-use-tps",
        "--bdrc_line_use_tps",
        dest="bdrc_line_use_tps",
        action="store_true",
        default=True,
        help="Enable BDRC global TPS dewarping before final line extraction. Default: enabled.",
    )
    parser.add_argument(
        "--bdrc-line-no-use-tps",
        "--bdrc_line_no_use_tps",
        dest="bdrc_line_use_tps",
        action="store_false",
        help="Disable BDRC global TPS dewarping.",
    )
    parser.add_argument(
        "--bdrc-line-tps-threshold",
        "--bdrc_line_tps_threshold",
        dest="bdrc_line_tps_threshold",
        type=float,
        default=0.25,
        help="Minimum curved-line ratio required before BDRC TPS dewarping is applied. Default: 0.25.",
    )
    parser.add_argument(
        "--save-overlay",
        "--save_overlay",
        dest="save_overlay",
        action="store_true",
        default=True,
        help=(
            "Save a JPEG overlay image (*_overlay.jpg) next to each .txt file, "
            "with the detected line bounding boxes drawn on the source image. "
            "Enabled by default."
        ),
    )
    parser.add_argument(
        "--no-save-overlay",
        "--no_save_overlay",
        dest="save_overlay",
        action="store_false",
        help="Skip saving the line-box overlay images.",
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
            "Inference device for layout model and DONUT. "
            "'auto' selects cuda:0 if available, otherwise cpu. "
            "Examples: 'cpu', 'cuda:0', 'cuda:1'. Default: auto."
        ),
    )

    # --- DONUT-specific args ---
    donut_group = parser.add_argument_group("DONUT engine options")
    donut_group.add_argument(
        "--ocr-model",
        "--ocr_model",
        dest="ocr_model",
        type=str,
        default="",
        help=(
            "Path to the DONUT OCR checkpoint directory (required for --engine donut). "
            "Must contain config.json and model weights. "
            "If a repro/ subdirectory is present, the preprocessing pipeline and "
            "tokenizer/image_processor are loaded from it automatically."
        ),
    )
    donut_group.add_argument(
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

    bdrc_group = parser.add_argument_group("BDRC OCR engine options")
    bdrc_group.add_argument(
        "--bdrc-ocr-model",
        "--bdrc_ocr_model",
        dest="bdrc_ocr_model",
        type=str,
        default="",
        help=(
            "Path to a BDRC OCR model directory (or model_config.json/onnx file). "
            "Optional when --engine bdrc_ocr. If omitted, the default BDRC OCR bundle "
            "is downloaded into models/bdrc automatically."
        ),
    )
    bdrc_group.add_argument(
        "--bdrc-ocr-target-encoding",
        "--bdrc_ocr_target_encoding",
        dest="bdrc_ocr_target_encoding",
        type=str,
        default="unicode",
        choices=["raw", "unicode", "wylie"],
        help=(
            "Optional post-conversion for BDRC OCR output. "
            "'raw' keeps model output as-is. Default: unicode."
        ),
    )

    # --- Tesseract OCR engine options ---
    tess_group = parser.add_argument_group(
        "Tesseract OCR engine options",
        "Used when --engine tesseract (OCR) or --layout-engine tesseract (layout detection).",
    )
    tess_group.add_argument(
        "--tess-lang",
        "--tess_lang",
        dest="tess_lang",
        type=str,
        default="bod",
        help=(
            "Tesseract language code for OCR (--engine tesseract). "
            "Default: 'bod' (Tibetan). Falls back to 'eng' if 'bod' is not installed."
        ),
    )
    tess_group.add_argument(
        "--tess-psm",
        "--tess_psm",
        dest="tess_psm",
        type=int,
        default=6,
        help=(
            "Tesseract page segmentation mode (PSM) for OCR crops. "
            "6 = assume a single uniform block of text (default, good for line crops). "
            "7 = treat the image as a single text line."
        ),
    )
    tess_group.add_argument(
        "--tess-oem",
        "--tess_oem",
        dest="tess_oem",
        type=int,
        default=1,
        help=(
            "Tesseract OCR engine mode (OEM) for OCR. "
            "0 = Legacy, 1 = LSTM (default), 2 = Legacy+LSTM, 3 = Default."
        ),
    )
    tess_group.add_argument(
        "--tess-cmd",
        "--tess_cmd",
        dest="tess_cmd",
        type=str,
        default="",
        help=(
            "Path to the tesseract binary if not on PATH. "
            "Shared by both OCR and layout detection. "
            "Leave empty to use the system default."
        ),
    )

    # --- Tesseract layout engine options ---
    tess_layout_group = parser.add_argument_group(
        "Tesseract layout engine options",
        "Used when --layout-engine tesseract to control page segmentation for line detection.",
    )
    tess_layout_group.add_argument(
        "--tess-layout-lang",
        "--tess_layout_lang",
        dest="tess_layout_lang",
        type=str,
        default="bod",
        help=(
            "Tesseract language code for layout detection (--layout-engine tesseract). "
            "Default: 'bod' (Tibetan). Falls back to 'eng' if 'bod' is not installed."
        ),
    )
    tess_layout_group.add_argument(
        "--tess-layout-psm",
        "--tess_layout_psm",
        dest="tess_layout_psm",
        type=int,
        default=3,
        help=(
            "Tesseract PSM for layout/page segmentation. "
            "3 = fully automatic page segmentation, no OSD (default). "
            "1 = automatic page segmentation with OSD."
        ),
    )
    tess_layout_group.add_argument(
        "--tess-layout-oem",
        "--tess_layout_oem",
        dest="tess_layout_oem",
        type=int,
        default=1,
        help=(
            "Tesseract OEM for layout detection. "
            "0 = Legacy, 1 = LSTM (default), 2 = Legacy+LSTM, 3 = Default."
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
