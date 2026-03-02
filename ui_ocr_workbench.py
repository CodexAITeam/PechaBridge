#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ui_workbench import (
    _apply_donut_ui_preprocess,
    _compute_line_projection_state,
    _load_donut_ocr_runtime_cached,
    _load_donut_repro_bundle_cfg_ui,
    _resolve_donut_runtime_dirs,
    _segment_lines_in_text_crop,
    run_donut_ocr_inference_ui,
    run_tibetan_text_line_split_classical,
)
try:
    from pechabridge.ocr.preprocess_bdrc import BDRCPreprocessConfig
except Exception:
    BDRCPreprocessConfig = None

try:
    import torch
except Exception:
    torch = None


ROOT = Path(__file__).resolve().parent
MODE_AUTO = "Fully Automatic OCR"
MODE_MANUAL = "Manual Mode"
_DONUT_ACTIVE_RUNTIME: Dict[str, Any] = {"checkpoint": "", "runtime": None}


def _is_manual_mode(mode: str) -> bool:
    return str(mode or "").strip() == MODE_MANUAL


def _load_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:
        return ImageFont.load_default()


def _is_hf_model_dir(p: Path) -> bool:
    if not p.exists() or not p.is_dir():
        return False
    if not (p / "config.json").exists():
        return False
    return (
        (p / "pytorch_model.bin").exists()
        or (p / "model.safetensors").exists()
        or (p / "model.safetensors.index.json").exists()
    )


def _is_repro_checkpoint(p: Path) -> bool:
    if not _is_hf_model_dir(p):
        return False
    repro = p / "repro"
    return (
        repro.exists()
        and (repro / "generate_config.json").exists()
        and (repro / "image_preprocess.json").exists()
        and (repro / "tokenizer").exists()
        and (repro / "image_processor").exists()
    )


def _checkpoint_step(p: Path) -> int:
    name = p.name.strip().lower()
    if name.startswith("checkpoint-"):
        raw = name.replace("checkpoint-", "", 1)
        try:
            return int(raw)
        except Exception:
            return -1
    return -1


def _is_plain_checkpoint_with_runtime_assets(p: Path) -> bool:
    if not _is_hf_model_dir(p):
        return False
    parent = p.parent
    has_parent_assets = (parent / "tokenizer").exists() and (parent / "image_processor").exists()
    has_local_gen_cfg = (p / "generation_config.json").exists()
    return bool(has_parent_assets or has_local_gen_cfg)


def _find_donut_checkpoint() -> Tuple[str, str]:
    preferred = (ROOT / "models" / "ocr").resolve()
    fallback_root = (ROOT / "models").resolve()
    repro_candidates: List[Path] = []
    plain_candidates: List[Path] = []

    if preferred.exists():
        for p in sorted(preferred.rglob("checkpoint-*")):
            if _is_repro_checkpoint(p):
                repro_candidates.append(p.resolve())
            elif _is_plain_checkpoint_with_runtime_assets(p):
                plain_candidates.append(p.resolve())
        if not repro_candidates and not plain_candidates:
            for p in sorted(preferred.rglob("*")):
                if p.is_dir() and _is_repro_checkpoint(p):
                    repro_candidates.append(p.resolve())
                elif p.is_dir() and _is_plain_checkpoint_with_runtime_assets(p):
                    plain_candidates.append(p.resolve())

    if not repro_candidates and not plain_candidates and fallback_root.exists():
        for p in sorted(fallback_root.rglob("checkpoint-*")):
            if _is_repro_checkpoint(p):
                repro_candidates.append(p.resolve())
            elif _is_plain_checkpoint_with_runtime_assets(p):
                plain_candidates.append(p.resolve())
        if not repro_candidates and not plain_candidates:
            for p in sorted(fallback_root.rglob("*")):
                if p.is_dir() and _is_repro_checkpoint(p):
                    repro_candidates.append(p.resolve())
                elif p.is_dir() and _is_plain_checkpoint_with_runtime_assets(p):
                    plain_candidates.append(p.resolve())

    if repro_candidates:
        repro_candidates = sorted(repro_candidates, key=_checkpoint_step, reverse=True)
        return str(repro_candidates[0]), f"Auto-selected DONUT checkpoint (repro): {repro_candidates[0]}"

    if plain_candidates:
        plain_candidates = sorted(plain_candidates, key=_checkpoint_step, reverse=True)
        return (
            str(plain_candidates[0]),
            "Auto-selected DONUT checkpoint (without repro, using parent tokenizer/image_processor): "
            f"{plain_candidates[0]}",
        )

    return "", "No DONUT checkpoint found (expected under models/ocr/ or models/)."


def _find_layout_model() -> Tuple[str, str]:
    preferred = (ROOT / "models" / "layoutAnalysis").resolve()
    fallback_root = (ROOT / "models").resolve()
    model_exts = {".pt", ".onnx", ".torchscript"}
    candidates: List[Path] = []

    def _scan(base: Path) -> None:
        if not base.exists():
            return
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in model_exts:
                continue
            candidates.append(p.resolve())

    _scan(preferred)
    if not candidates:
        _scan(fallback_root)

    if not candidates:
        return "", "No layout analysis model found (expected under models/layoutAnalysis/)."
    return str(candidates[0]), f"Auto-selected layout model: {candidates[0]}"


def _normalize_box(box: List[int], w: int, h: int) -> Optional[List[int]]:
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
    return [x1, y1, x2, y2]


def _sort_lines(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda r: (int(r["line_box"][1]), int(r["line_box"][0]), int(r["line_id"])))


def _render_overlay(image: np.ndarray, lines: List[Dict[str, Any]], roi: Optional[List[int]] = None) -> np.ndarray:
    panel = Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(panel)
    font = _load_font()

    def _draw_strong_rect(box: List[int], color: Tuple[int, int, int], inner_width: int = 4, outer_width: int = 8) -> None:
        x1, y1, x2, y2 = [int(v) for v in box]
        # Outer dark stroke for contrast on bright backgrounds.
        draw.rectangle((x1, y1, x2, y2), outline=(0, 0, 0), width=max(2, int(outer_width)))
        # Inner bright stroke.
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, int(inner_width)))

    for i, rec in enumerate(_sort_lines(lines), start=1):
        x1, y1, x2, y2 = [int(v) for v in rec["line_box"]]
        _draw_strong_rect([x1, y1, x2, y2], color=(0, 255, 255), inner_width=4, outer_width=8)
        tag = f"line {i}"
        tx1 = x1 + 2
        ty1 = max(0, y1 - 20)
        tx2 = x1 + 18 + 8 * len(tag)
        ty2 = max(16, y1 - 2)
        draw.rectangle((tx1, ty1, tx2, ty2), fill=(0, 0, 0))
        draw.rectangle((tx1, ty1, tx2, ty2), outline=(0, 255, 255), width=2)
        draw.text((tx1 + 4, ty1 + 2), tag, fill=(255, 255, 255), font=font)
    if roi and len(roi) == 4:
        rx1, ry1, rx2, ry2 = [int(v) for v in roi]
        _draw_strong_rect([rx1, ry1, rx2, ry2], color=(255, 64, 255), inner_width=4, outer_width=8)
    return np.asarray(panel).astype(np.uint8, copy=False)


def _line_text(rows: List[Dict[str, Any]]) -> str:
    lines = _sort_lines(rows)
    return "\n".join([str(r.get("text", "") or "") for r in lines])


def _to_rgb_uint8(path: str) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.uint8)
    return arr


def _resolve_device_for_runtime(pref: str) -> str:
    p = str(pref or "auto").strip().lower()
    if p in {"", "auto"}:
        if torch is not None and bool(getattr(torch.cuda, "is_available", lambda: False)()):
            return "cuda:0"
        return "cpu"
    if p.startswith("cuda"):
        if torch is None or not bool(getattr(torch.cuda, "is_available", lambda: False)()):
            return "cpu"
    return p


def _ensure_donut_runtime_loaded(
    donut_checkpoint: str,
    device_pref: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    if torch is None:
        return None, "PyTorch not available."
    model_dir, tokenizer_dir, image_processor_dir, err = _resolve_donut_runtime_dirs(donut_checkpoint)
    if err:
        return None, str(err)

    checkpoint_key = str(Path(donut_checkpoint).expanduser().resolve())
    target_device = _resolve_device_for_runtime(device_pref)
    active_ckpt = str(_DONUT_ACTIVE_RUNTIME.get("checkpoint") or "")
    active_runtime = _DONUT_ACTIVE_RUNTIME.get("runtime")

    if active_runtime is not None and active_ckpt == checkpoint_key:
        cur_device = str(active_runtime.get("device") or "")
        if cur_device != target_device:
            try:
                active_runtime["model"].to(target_device)
                active_runtime["device"] = target_device
                active_runtime["device_msg"] = f"Moved DONUT runtime to {target_device}"
                return active_runtime, str(active_runtime["device_msg"])
            except Exception:
                # Fall back to loading a fresh runtime below.
                pass
        return active_runtime, f"DONUT runtime ready on {cur_device or target_device}"

    try:
        runtime = _load_donut_ocr_runtime_cached(
            str(model_dir),
            str(tokenizer_dir),
            str(image_processor_dir),
            target_device,
        )
    except Exception as exc:
        return None, f"Failed loading DONUT runtime: {type(exc).__name__}: {exc}"

    _DONUT_ACTIVE_RUNTIME["checkpoint"] = checkpoint_key
    _DONUT_ACTIVE_RUNTIME["runtime"] = runtime
    return runtime, f"DONUT runtime loaded on {runtime.get('device', target_device)}"


def _strip_special_token_strings_local(text: str, tokenizer: Any) -> str:
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


def _donut_preprocess_preview(
    crop: np.ndarray,
    donut_checkpoint: str,
    bdrc_preprocess_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    pil = Image.fromarray(np.asarray(crop).astype(np.uint8)).convert("RGB")
    pre = np.asarray(pil).astype(np.uint8, copy=False)
    effective_preproc = "bdrc"
    lock_preproc_to_bdrc = bool(
        isinstance(bdrc_preprocess_overrides, dict) and len(bdrc_preprocess_overrides) > 0
    )
    try:
        if not lock_preproc_to_bdrc:
            repro_cfg = _load_donut_repro_bundle_cfg_ui(donut_checkpoint)
            if bool(repro_cfg.get("has_repro")):
                ip_cfg = repro_cfg.get("image_preprocess") or {}
                if isinstance(ip_cfg, dict):
                    pipe = str(ip_cfg.get("pipeline", "") or "").strip().lower()
                    if pipe in {"none", "pb", "bdrc"}:
                        effective_preproc = pipe
    except Exception:
        pass
    try:
        proc = _apply_donut_ui_preprocess(
            pil,
            effective_preproc,
            bdrc_config_override=(
                bdrc_preprocess_overrides
                if (
                    effective_preproc == "bdrc"
                    and isinstance(bdrc_preprocess_overrides, dict)
                    and len(bdrc_preprocess_overrides) > 0
                )
                else None
            ),
        )
        post = np.asarray(proc).astype(np.uint8, copy=False)
    except Exception:
        post = pre
    return pre, post, effective_preproc


def _run_donut_on_crop_fallback(
    crop: np.ndarray,
    donut_checkpoint: str,
    device: str,
    max_len: int,
    bdrc_preprocess_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    if torch is None:
        raise RuntimeError("PyTorch not available.")
    runtime, rt_msg = _ensure_donut_runtime_loaded(donut_checkpoint, device)
    if runtime is None:
        raise RuntimeError(rt_msg)
    repro_cfg = _load_donut_repro_bundle_cfg_ui(donut_checkpoint)
    effective_preproc = "bdrc"
    lock_preproc_to_bdrc = bool(
        isinstance(bdrc_preprocess_overrides, dict) and len(bdrc_preprocess_overrides) > 0
    )
    effective_max_len = max(8, int(max_len))
    if bool(repro_cfg.get("has_repro")):
        ip_cfg = repro_cfg.get("image_preprocess") or {}
        if isinstance(ip_cfg, dict):
            pipe = str(ip_cfg.get("pipeline", "") or "").strip().lower()
            if pipe in {"none", "pb", "bdrc"} and not lock_preproc_to_bdrc:
                effective_preproc = pipe
        gcfg = repro_cfg.get("generate_config") or {}
        if isinstance(gcfg, dict) and gcfg.get("max_length") is not None:
            effective_max_len = max(8, int(gcfg["max_length"]))

    pil = Image.fromarray(np.asarray(crop).astype(np.uint8)).convert("RGB")
    proc_pil = _apply_donut_ui_preprocess(
        pil,
        effective_preproc,
        bdrc_config_override=(
            bdrc_preprocess_overrides
            if (
                effective_preproc == "bdrc"
                and isinstance(bdrc_preprocess_overrides, dict)
                and len(bdrc_preprocess_overrides) > 0
            )
            else None
        ),
    )
    image_processor = runtime["image_processor"]
    tokenizer = runtime["tokenizer"]
    model = runtime["model"]
    dev = runtime["device"]

    pixel_values = image_processor(images=proc_pil, return_tensors="pt").pixel_values.to(dev)
    with torch.no_grad():
        generated = model.generate(pixel_values=pixel_values, max_length=int(effective_max_len), num_beams=1)
    text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0] if len(generated) else ""
    text = _strip_special_token_strings_local(text, tokenizer)
    return str(text or ""), {
        "ok": True,
        "fallback_used": True,
        "fallback_reason": "ui_workbench _strip_special_token_strings NameError",
        "runtime_status": rt_msg,
        "image_preprocess_pipeline_effective": effective_preproc,
        "image_preprocess_locked_by_bdrc_overrides": bool(lock_preproc_to_bdrc),
        "bdrc_preprocess_overrides_applied": bool(
            effective_preproc == "bdrc"
            and isinstance(bdrc_preprocess_overrides, dict)
            and len(bdrc_preprocess_overrides) > 0
        ),
        "generation_max_length_effective": int(effective_max_len),
        "device": str(dev),
    }


@dataclass
class OCRRuntime:
    donut_path: str
    layout_path: str


def _base_state() -> Dict[str, Any]:
    return {
        "image_path": "",
        "image_name": "",
        "image": None,
        "line_rows": [],
        "manual_anchor": None,
        "last_mode": "",
        "last_debug_text": "",
    }


def _bdrc_ui_defaults() -> Dict[str, Any]:
    base: Dict[str, Any] = {}
    try:
        if BDRCPreprocessConfig is not None:
            base = dict(BDRCPreprocessConfig.vit_defaults().to_dict())
    except Exception:
        base = {}
    return {
        "gray_mode": str(base.get("gray_mode", "luma")),
        "normalize_background": bool(base.get("normalize_background", False)),
        "background_blur_ksize": int(base.get("background_blur_ksize", 0)),
        "background_strength": float(base.get("background_strength", 1.0)),
        "upscale_factor": float(base.get("upscale_factor", 1.0)),
        "upscale_interpolation": str(base.get("upscale_interpolation", "lanczos")),
        "binarize": bool(base.get("binarize", True)),
        "threshold_method": str(base.get("threshold_method", "adaptive")),
        "threshold_block_size": int(base.get("threshold_block_size", 51)),
        "threshold_c": int(base.get("threshold_c", 13)),
        "fixed_threshold": int(base.get("fixed_threshold", 120)),
        "morph_close": bool(base.get("morph_close", False)),
        "morph_close_kernel": int(base.get("morph_close_kernel", 2)),
        "remove_small_components": bool(base.get("remove_small_components", False)),
        "min_component_area": int(base.get("min_component_area", 12)),
    }


def _build_bdrc_preprocess_overrides_ui(
    gray_mode: str,
    normalize_background: bool,
    background_blur_ksize: int,
    background_strength: float,
    upscale_factor: float,
    upscale_interpolation: str,
    binarize: bool,
    threshold_method: str,
    threshold_block_size: int,
    threshold_c: int,
    fixed_threshold: int,
    morph_close: bool,
    morph_close_kernel: int,
    remove_small_components: bool,
    min_component_area: int,
) -> Dict[str, Any]:
    gm = str(gray_mode or "luma").strip().lower()
    if gm not in {"luma", "min_rgb", "max_rgb", "r", "g", "b"}:
        gm = "luma"

    interp = str(upscale_interpolation or "lanczos").strip().lower()
    if interp not in {"nearest", "linear", "cubic", "lanczos"}:
        interp = "lanczos"

    tmethod = str(threshold_method or "adaptive").strip().lower()
    if tmethod not in {"adaptive", "otsu", "fixed"}:
        tmethod = "adaptive"

    block = int(max(3, int(threshold_block_size)))
    if block % 2 == 0:
        block += 1

    blur_k = int(max(0, int(background_blur_ksize)))
    if blur_k > 0 and blur_k % 2 == 0:
        blur_k += 1

    close_k = int(max(0, int(morph_close_kernel)))
    if close_k > 0 and close_k % 2 == 0:
        close_k += 1

    return {
        "gray_mode": gm,
        "normalize_background": bool(normalize_background),
        "background_blur_ksize": blur_k,
        "background_strength": float(max(0.0, min(3.0, float(background_strength)))),
        "upscale_factor": float(max(1.0, float(upscale_factor))),
        "upscale_interpolation": interp,
        "binarize": bool(binarize),
        "threshold_method": tmethod,
        "adaptive_threshold": bool(tmethod == "adaptive"),
        "threshold_block_size": int(block),
        "threshold_c": int(threshold_c),
        "fixed_threshold": int(max(0, min(255, int(fixed_threshold)))),
        "morph_close": bool(morph_close),
        "morph_close_kernel": int(close_k),
        "remove_small_components": bool(remove_small_components),
        "min_component_area": int(max(0, int(min_component_area))),
    }


def _run_donut_on_crop(
    crop: np.ndarray,
    donut_checkpoint: str,
    device: str,
    max_len: int,
    bdrc_preprocess_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any], np.ndarray, np.ndarray]:
    pre_img, post_img, effective_preproc = _donut_preprocess_preview(
        crop,
        donut_checkpoint,
        bdrc_preprocess_overrides=bdrc_preprocess_overrides,
    )
    _, pred, debug_json = run_donut_ocr_inference_ui(
        image=crop,
        model_root_or_model_dir=donut_checkpoint,
        image_preprocess_pipeline="bdrc",
        device_preference=device,
        generation_max_length=int(max_len),
        num_beams=1,
        bdrc_preprocess_overrides=bdrc_preprocess_overrides,
    )
    try:
        dbg = json.loads(debug_json or "{}")
    except Exception:
        dbg = {"raw_debug": debug_json}
    dbg_obj = dbg if isinstance(dbg, dict) else {"raw_debug": dbg}
    dbg_obj.setdefault("effective_preprocess_preview", effective_preproc)
    if bool(dbg_obj.get("ok")):
        return str(pred or ""), dbg_obj, pre_img, post_img

    err_msg = str(dbg_obj.get("error", "") or "")
    if "_strip_special_token_strings" in err_msg:
        try:
            text_fb, dbg_fb = _run_donut_on_crop_fallback(
                crop,
                donut_checkpoint,
                device,
                max_len,
                bdrc_preprocess_overrides=bdrc_preprocess_overrides,
            )
            dbg_fb.setdefault("effective_preprocess_preview", effective_preproc)
            return text_fb, dbg_fb, pre_img, post_img
        except Exception as exc:
            return str(pred or ""), {
                "ok": False,
                "error": err_msg,
                "fallback_attempted": True,
                "fallback_error": f"{type(exc).__name__}: {exc}",
                "effective_preprocess_preview": effective_preproc,
            }, pre_img, post_img
    return str(pred or ""), dbg_obj, pre_img, post_img


def _run_full_auto(
    state: Dict[str, Any],
    donut_checkpoint: str,
    layout_model: str,
    device: str,
    max_len: int,
    bdrc_preprocess_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, str, str, Dict[str, Any], str, Optional[np.ndarray], Optional[np.ndarray]]:
    image = state.get("image")
    if image is None:
        return None, "", "Please upload an image first.", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")
    if not donut_checkpoint.strip():
        return image, "", "DONUT checkpoint is missing.", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")
    if not layout_model.strip():
        return image, "", "Layout model is missing.", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")

    split_out = run_tibetan_text_line_split_classical(
        image=np.asarray(image).astype(np.uint8, copy=False),
        model_path=layout_model,
        conf=0.25,
        imgsz=1024,
        device=device if device != "auto" else "",
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
    overlay, split_status, split_json, _, _, _, _, _, click_state = split_out
    line_records_raw = []
    if isinstance(click_state, dict):
        line_records_raw = click_state.get("line_boxes") or []

    src = np.asarray(image).astype(np.uint8, copy=False)
    h, w = src.shape[:2]
    rows: List[Dict[str, Any]] = []
    last_pre: Optional[np.ndarray] = None
    last_post: Optional[np.ndarray] = None
    for rec in line_records_raw:
        box = _normalize_box(rec.get("line_box") or [], w, h)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        crop = src[y1:y2, x1:x2]
        text, dbg, pre_img, post_img = _run_donut_on_crop(
            crop,
            donut_checkpoint,
            device,
            max_len,
            bdrc_preprocess_overrides=bdrc_preprocess_overrides,
        )
        last_pre, last_post = pre_img, post_img
        rows.append(
            {
                "line_id": int(rec.get("line_id", len(rows) + 1)),
                "line_box": box,
                "text": text,
                "ocr_debug": dbg,
                "source": "full_auto",
            }
        )

    rows = _sort_lines(rows)
    state["line_rows"] = rows
    state["manual_anchor"] = None
    state["last_mode"] = "full_auto"
    state["last_donut_pre"] = last_pre
    state["last_donut_post"] = last_post
    state["last_debug_text"] = str(rows[-1].get("text", "") if rows else "")
    transcript = _line_text(rows)
    debug = {
        "ok": True,
        "mode": "full_auto",
        "split_status": split_status,
        "line_count": len(rows),
        "bdrc_preprocess_overrides": (
            dict(bdrc_preprocess_overrides)
            if isinstance(bdrc_preprocess_overrides, dict)
            else None
        ),
        "split_json": json.loads(split_json) if (split_json or "").strip().startswith("{") else split_json,
    }
    strong_overlay = _render_overlay(src, rows)
    return (
        strong_overlay,
        transcript,
        f"{split_status} OCR executed on {len(rows)} line(s).",
        state,
        json.dumps(debug, ensure_ascii=False, indent=2),
        state.get("last_donut_pre"),
        state.get("last_donut_post"),
    )


def _find_line_hit(rows: List[Dict[str, Any]], x: int, y: int) -> Optional[Dict[str, Any]]:
    hits: List[Tuple[int, Dict[str, Any]]] = []
    for rec in rows:
        box = rec.get("line_box") or []
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        if x1 <= x <= x2 and y1 <= y <= y2:
            area = max(1, (x2 - x1) * (y2 - y1))
            hits.append((area, rec))
    if not hits:
        return None
    hits.sort(key=lambda t: t[0])
    return hits[0][1]


def _roi_is_single_line(roi_w: int, roi_h: int) -> bool:
    if roi_h <= 0 or roi_w <= 0:
        return False
    ratio = float(roi_w) / float(max(1, roi_h))
    return ratio >= 4.0 or roi_h <= 80


def _add_or_update_line(rows: List[Dict[str, Any]], row: Dict[str, Any]) -> List[Dict[str, Any]]:
    box = row.get("line_box") or [0, 0, 0, 0]
    for i, rec in enumerate(rows):
        if list(rec.get("line_box") or []) == list(box):
            rows[i] = row
            return _sort_lines(rows)
    rows.append(row)
    return _sort_lines(rows)


def _manual_click(
    state: Dict[str, Any],
    donut_checkpoint: str,
    layout_model: str,
    device: str,
    max_len: int,
    bdrc_preprocess_overrides: Optional[Dict[str, Any]],
    evt: gr.SelectData,
) -> Tuple[np.ndarray, str, str, Dict[str, Any], str, Optional[np.ndarray], Optional[np.ndarray]]:
    image = state.get("image")
    if image is None:
        return None, "", "Please upload an image first.", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")
    if not donut_checkpoint.strip():
        return image, "", "DONUT checkpoint is missing.", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")

    idx = getattr(evt, "index", None)
    if not isinstance(idx, (tuple, list)) or len(idx) < 2:
        return np.asarray(image), _line_text(state.get("line_rows") or []), "Click position not available.", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")
    try:
        click_x, click_y = int(idx[0]), int(idx[1])
    except Exception:
        return np.asarray(image), _line_text(state.get("line_rows") or []), "Invalid click position.", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")

    src = np.asarray(image).astype(np.uint8, copy=False)
    h, w = src.shape[:2]
    click_x = max(0, min(w - 1, click_x))
    click_y = max(0, min(h - 1, click_y))
    rows = list(state.get("line_rows") or [])

    hit = _find_line_hit(rows, click_x, click_y)
    if hit is not None:
        x1, y1, x2, y2 = [int(v) for v in hit["line_box"]]
        crop = src[y1:y2, x1:x2]
        text, dbg, pre_img, post_img = _run_donut_on_crop(
            crop,
            donut_checkpoint,
            device,
            max_len,
            bdrc_preprocess_overrides=bdrc_preprocess_overrides,
        )
        new_row = dict(hit)
        new_row["text"] = text
        new_row["ocr_debug"] = dbg
        new_row["source"] = "manual_line_click"
        rows = _add_or_update_line(rows, new_row)
        state["line_rows"] = rows
        state["last_donut_pre"] = pre_img
        state["last_donut_post"] = post_img
        state["last_debug_text"] = str(text or "")
        overlay = _render_overlay(src, rows)
        return overlay, _line_text(rows), f"Re-transcribed line at click ({click_x},{click_y}).", state, json.dumps(
            {"ok": True, "mode": "manual", "action": "clicked_existing_line", "line_box": new_row["line_box"]},
            ensure_ascii=False,
            indent=2,
        ), state.get("last_donut_pre"), state.get("last_donut_post")

    anchor = state.get("manual_anchor")
    if not isinstance(anchor, (list, tuple)) or len(anchor) != 2:
        state["manual_anchor"] = [int(click_x), int(click_y)]
        overlay = _render_overlay(src, rows)
        return overlay, _line_text(rows), f"Start point set at ({click_x},{click_y}). Click a second point to define ROI.", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")

    ax, ay = int(anchor[0]), int(anchor[1])
    state["manual_anchor"] = None
    x1, x2 = sorted([ax, click_x])
    y1, y2 = sorted([ay, click_y])
    if (x2 - x1) < 3 or (y2 - y1) < 3:
        overlay = _render_overlay(src, rows)
        return overlay, _line_text(rows), "ROI is too small. Please select a larger area.", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")
    roi = _normalize_box([x1, y1, x2, y2], w, h)
    return _manual_process_roi(
        state,
        donut_checkpoint,
        device,
        max_len,
        roi,
        bdrc_preprocess_overrides=bdrc_preprocess_overrides,
    )


def _manual_process_roi(
    state: Dict[str, Any],
    donut_checkpoint: str,
    device: str,
    max_len: int,
    roi: Optional[List[int]],
    bdrc_preprocess_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, str, str, Dict[str, Any], str, Optional[np.ndarray], Optional[np.ndarray]]:
    image = state.get("image")
    if image is None:
        return None, "", "Please upload an image first.", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")
    src = np.asarray(image).astype(np.uint8, copy=False)
    h, w = src.shape[:2]
    rows = list(state.get("line_rows") or [])

    roi = _normalize_box(roi or [], w, h)
    if roi is None:
        overlay = _render_overlay(src, rows)
        return overlay, _line_text(rows), "ROI is invalid.", state, "{}", state.get("last_donut_pre"), state.get("last_donut_post")
    rx1, ry1, rx2, ry2 = roi
    roi_crop = src[ry1:ry2, rx1:rx2]
    roi_w = int(rx2 - rx1)
    roi_h = int(ry2 - ry1)

    created_rows: List[Dict[str, Any]] = []
    last_pre: Optional[np.ndarray] = None
    last_post: Optional[np.ndarray] = None
    if _roi_is_single_line(roi_w, roi_h):
        text, dbg, pre_img, post_img = _run_donut_on_crop(
            roi_crop,
            donut_checkpoint,
            device,
            max_len,
            bdrc_preprocess_overrides=bdrc_preprocess_overrides,
        )
        last_pre, last_post = pre_img, post_img
        created_rows.append(
            {
                "line_id": len(rows) + 1,
                "line_box": [rx1, ry1, rx2, ry2],
                "text": text,
                "ocr_debug": dbg,
                "source": "manual_roi_single_line",
            }
        )
        action = "roi_direct_line"
    else:
        line_state = _compute_line_projection_state(
            crop_rgb=roi_crop,
            projection_smooth=9,
            projection_threshold_rel=0.20,
        )
        line_boxes_local = _segment_lines_in_text_crop(
            crop_rgb=roi_crop,
            min_line_height=10,
            projection_smooth=9,
            projection_threshold_rel=0.20,
            merge_gap_px=5,
            projection_state=line_state,
        )
        for bx1, by1, bx2, by2 in line_boxes_local:
            gx1, gy1, gx2, gy2 = rx1 + int(bx1), ry1 + int(by1), rx1 + int(bx2), ry1 + int(by2)
            gbox = _normalize_box([gx1, gy1, gx2, gy2], w, h)
            if gbox is None:
                continue
            lx1, ly1, lx2, ly2 = gbox
            lcrop = src[ly1:ly2, lx1:lx2]
            text, dbg, pre_img, post_img = _run_donut_on_crop(
                lcrop,
                donut_checkpoint,
                device,
                max_len,
                bdrc_preprocess_overrides=bdrc_preprocess_overrides,
            )
            last_pre, last_post = pre_img, post_img
            created_rows.append(
                {
                    "line_id": len(rows) + len(created_rows) + 1,
                    "line_box": gbox,
                    "text": text,
                    "ocr_debug": dbg,
                    "source": "manual_roi_line_split",
                }
            )
        if not created_rows:
            text, dbg, pre_img, post_img = _run_donut_on_crop(
                roi_crop,
                donut_checkpoint,
                device,
                max_len,
                bdrc_preprocess_overrides=bdrc_preprocess_overrides,
            )
            last_pre, last_post = pre_img, post_img
            created_rows.append(
                {
                    "line_id": len(rows) + 1,
                    "line_box": [rx1, ry1, rx2, ry2],
                    "text": text,
                    "ocr_debug": dbg,
                    "source": "manual_roi_fallback_direct",
                }
            )
        action = "roi_line_split"

    for rec in created_rows:
        rows = _add_or_update_line(rows, rec)

    state["line_rows"] = rows
    state["last_mode"] = "manual"
    state["last_donut_pre"] = last_pre
    state["last_donut_post"] = last_post
    state["last_debug_text"] = str(created_rows[-1].get("text", "") if created_rows else "")
    overlay = _render_overlay(src, rows, roi=roi)
    debug = {
        "ok": True,
        "mode": "manual",
        "action": action,
        "roi": roi,
        "new_rows": created_rows,
        "total_rows": len(rows),
        "bdrc_preprocess_overrides": (
            dict(bdrc_preprocess_overrides)
            if isinstance(bdrc_preprocess_overrides, dict)
            else None
        ),
    }
    return (
        overlay,
        _line_text(rows),
        f"Manual ROI processed. New/updated lines: {len(created_rows)}.",
        state,
        json.dumps(debug, ensure_ascii=False, indent=2),
        state.get("last_donut_pre"),
        state.get("last_donut_post"),
    )


def _manual_full_image_roi(
    mode_s: str,
    state_s: Dict[str, Any],
    donut_s: str,
    device_s: str,
    max_len_s: int,
    bdrc_preprocess_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, str, str, Dict[str, Any], str, Optional[np.ndarray], Optional[np.ndarray]]:
    if not _is_manual_mode(mode_s):
        img = state_s.get("image")
        overlay = np.asarray(img).astype(np.uint8, copy=False) if img is not None else None
        return overlay, _line_text(state_s.get("line_rows") or []), "This button is only active in Manual Mode.", state_s, "{}", state_s.get("last_donut_pre"), state_s.get("last_donut_post")
    img = state_s.get("image")
    if img is None:
        return None, "", "Please upload an image first.", state_s, "{}", state_s.get("last_donut_pre"), state_s.get("last_donut_post")
    src = np.asarray(img).astype(np.uint8, copy=False)
    h, w = src.shape[:2]
    return _manual_process_roi(
        state_s,
        donut_s,
        device_s,
        int(max_len_s),
        [0, 0, w, h],
        bdrc_preprocess_overrides=bdrc_preprocess_overrides,
    )


def _on_upload(file_obj: Any, state: Dict[str, Any]) -> Tuple[np.ndarray, str, Dict[str, Any], str]:
    if file_obj is None:
        return None, "", _base_state(), "No image loaded."
    path = getattr(file_obj, "name", "") or str(file_obj)
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return None, "", _base_state(), f"Datei nicht gefunden: {p}"
    img = _to_rgb_uint8(str(p))
    new_state = _base_state()
    new_state["image_path"] = str(p)
    new_state["image_name"] = p.name
    new_state["image"] = img
    return img, "", new_state, f"Image loaded: {p.name}"


def _save_results(
    state: Dict[str, Any],
    transcript_text: str,
    donut_checkpoint: str,
    layout_model: str,
) -> str:
    image = state.get("image")
    if image is None:
        return "Nothing to save: no image loaded."
    name = str(state.get("image_name") or "image.png")
    stem = Path(name).stem
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (ROOT / "output" / "ocr" / f"{stem}_{ts}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    src = np.asarray(image).astype(np.uint8, copy=False)
    rows = _sort_lines(list(state.get("line_rows") or []))
    overlay = _render_overlay(src, rows)
    Image.fromarray(src).save(out_dir / "source.png")
    Image.fromarray(overlay).save(out_dir / "overlay.png")
    (out_dir / "transcript.txt").write_text(str(transcript_text or ""), encoding="utf-8")

    saved_rows: List[Dict[str, Any]] = []
    h, w = src.shape[:2]
    for i, rec in enumerate(rows, start=1):
        box = _normalize_box(rec.get("line_box") or [], w, h)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        crop = src[y1:y2, x1:x2]
        crop_name = f"line_{i:03d}.png"
        Image.fromarray(crop).save(out_dir / crop_name)
        obj = dict(rec)
        obj["line_no"] = i
        obj["line_box"] = box
        obj["line_crop_file"] = crop_name
        saved_rows.append(obj)

    payload = {
        "saved_at": datetime.now().isoformat(),
        "image_name": name,
        "image_path": str(state.get("image_path") or ""),
        "donut_checkpoint": str(donut_checkpoint or ""),
        "layout_model": str(layout_model or ""),
        "line_count": len(saved_rows),
        "lines": saved_rows,
    }
    (out_dir / "line_boxes_ocr.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"Saved to: {out_dir}"


def build_ui() -> gr.Blocks:
    donut_ckpt, donut_msg = _find_donut_checkpoint()
    layout_model, layout_msg = _find_layout_model()
    bdrc_defaults = _bdrc_ui_defaults()
    ui_css = """
#ocr_image_panel {
  min-height: 300px;
  min-width: 320px;
  resize: both;
  overflow: auto;
  border: 1px dashed #cbd5e1;
  border-radius: 8px;
}
#ocr_image_panel .image-container,
#ocr_image_panel img {
  max-height: none !important;
}
#transcript_panel {
  min-height: 300px;
  min-width: 320px;
  resize: both;
  overflow: auto;
  border: 1px dashed #cbd5e1;
  border-radius: 8px;
  padding: 4px;
}
#transcript_box textarea {
  font-size: 18px !important;
  line-height: 1.45 !important;
  min-height: 300px !important;
  resize: both !important;
  overflow: auto !important;
}
#font_btn_plus, #font_btn_minus {
  min-width: 36px !important;
  width: 36px !important;
  padding: 0 !important;
}
#font_ctrl_col {
  max-width: 24px;
  min-width: 24px;
}
#debug_font_ctrl_col {
  max-width: 24px;
  min-width: 24px;
}
#debug_text_box textarea {
  font-size: 20px !important;
  line-height: 1.45 !important;
}
#donut_input_before,
#donut_input_after {
  margin-top: 10px !important;
  margin-bottom: 10px !important;
}
"""

    with gr.Blocks(title="OCR Workbench (DONUT)", css=ui_css) as demo:
        gr.Markdown("## OCR Workbench (DONUT + Layout)")
        gr.Markdown(
            "Automatic mode: layout -> line detection -> OCR. Manual mode: click an existing line or define an ROI with two clicks."
        )
        advanced_view = gr.Checkbox(label="Advanced View", value=False)

        with gr.Row():
            donut_path = gr.Textbox(label="DONUT Checkpoint", value=donut_ckpt)
            layout_path = gr.Textbox(label="Layout Analysis Model", value=layout_model)
        with gr.Row(visible=False) as advanced_scan_row:
            donut_info = gr.Textbox(label="DONUT Auto-Scan", value=donut_msg, interactive=False)
            layout_info = gr.Textbox(label="Layout Auto-Scan", value=layout_msg, interactive=False)

        with gr.Row():
            mode = gr.Radio(choices=[MODE_AUTO, MODE_MANUAL], value=MODE_AUTO, label="Mode")
        with gr.Row(visible=False) as advanced_runtime_row:
            device = gr.Dropdown(choices=["auto", "cuda:0", "cpu"], value="auto", label="Inference Device")
            max_len = gr.Number(label="DONUT generation_max_length", value=160, precision=0)
        with gr.Row(visible=False) as advanced_bdrc_row:
            with gr.Accordion("BDRC Preprocess 1-5 (DONUT Input)", open=False):
                gr.Markdown(
                    "1) Background normalize  2) Thresholding  3) Morph close + tiny component filter  "
                    "4) Upscale before binarization  5) Gray channel mode"
                )
                with gr.Row():
                    bdrc_gray_mode = gr.Dropdown(
                        choices=["luma", "min_rgb", "max_rgb", "r", "g", "b"],
                        value=str(bdrc_defaults["gray_mode"]),
                        label="5) gray_mode",
                    )
                    bdrc_upscale_factor = gr.Slider(
                        minimum=1.0,
                        maximum=3.0,
                        step=0.1,
                        value=float(bdrc_defaults["upscale_factor"]),
                        label="4) upscale_factor",
                    )
                    bdrc_upscale_interp = gr.Dropdown(
                        choices=["lanczos", "cubic", "linear", "nearest"],
                        value=str(bdrc_defaults["upscale_interpolation"]),
                        label="4) upscale_interpolation",
                    )
                with gr.Row():
                    bdrc_normalize_bg = gr.Checkbox(
                        value=bool(bdrc_defaults["normalize_background"]),
                        label="1) normalize_background",
                    )
                    bdrc_bg_blur_ksize = gr.Number(
                        value=int(bdrc_defaults["background_blur_ksize"]),
                        precision=0,
                        label="1) background_blur_ksize (0=auto)",
                    )
                    bdrc_bg_strength = gr.Slider(
                        minimum=0.0,
                        maximum=3.0,
                        step=0.05,
                        value=float(bdrc_defaults["background_strength"]),
                        label="1) background_strength",
                    )
                with gr.Row():
                    bdrc_binarize = gr.Checkbox(
                        value=bool(bdrc_defaults["binarize"]),
                        label="2) binarize",
                    )
                    bdrc_threshold_method = gr.Dropdown(
                        choices=["adaptive", "otsu", "fixed"],
                        value=str(bdrc_defaults["threshold_method"]),
                        label="2) threshold_method",
                    )
                    bdrc_fixed_threshold = gr.Slider(
                        minimum=0,
                        maximum=255,
                        step=1,
                        value=int(bdrc_defaults["fixed_threshold"]),
                        label="2) fixed_threshold",
                    )
                with gr.Row():
                    bdrc_threshold_block = gr.Number(
                        value=int(bdrc_defaults["threshold_block_size"]),
                        precision=0,
                        label="2) threshold_block_size (odd)",
                    )
                    bdrc_threshold_c = gr.Number(
                        value=int(bdrc_defaults["threshold_c"]),
                        precision=0,
                        label="2) threshold_c",
                    )
                with gr.Row():
                    bdrc_morph_close = gr.Checkbox(
                        value=bool(bdrc_defaults["morph_close"]),
                        label="3) morph_close",
                    )
                    bdrc_morph_close_kernel = gr.Number(
                        value=int(bdrc_defaults["morph_close_kernel"]),
                        precision=0,
                        label="3) morph_close_kernel (odd)",
                    )
                    bdrc_remove_small_components = gr.Checkbox(
                        value=bool(bdrc_defaults["remove_small_components"]),
                        label="3) remove_small_components",
                    )
                    bdrc_min_component_area = gr.Number(
                        value=int(bdrc_defaults["min_component_area"]),
                        precision=0,
                        label="3) min_component_area",
                    )

        status = gr.Textbox(label="Status", interactive=False)
        debug_json = gr.Code(label="Debug JSON", language="json", visible=False)
        with gr.Column():
            donut_input_before = gr.Image(
                label="DONUT Input (Before Preprocess)",
                type="numpy",
                visible=False,
                elem_id="donut_input_before",
            )
            donut_input_after = gr.Image(
                label="DONUT Input (After Preprocess)",
                type="numpy",
                visible=False,
                elem_id="donut_input_after",
            )
        with gr.Row(visible=False) as advanced_debug_text_row:
            debug_text = gr.Textbox(label="Debug Transcription", lines=4, elem_id="debug_text_box")
            with gr.Column(elem_id="debug_font_ctrl_col", scale=1, min_width=24):
                debug_font_plus_btn = gr.Button("+")
                debug_font_minus_btn = gr.Button("−")

        state = gr.State(_base_state())
        with gr.Row():
            image_file = gr.File(label="Upload Image", file_types=["image"])
            run_btn = gr.Button("Run OCR", variant="primary")
            full_roi_btn = gr.Button("Process Full Image as ROI", visible=False)
            save_btn = gr.Button("Save", variant="secondary")

        with gr.Row():
            image_view = gr.Image(
                label="Image / Overlay",
                type="numpy",
                interactive=True,
                height=300,
                elem_id="ocr_image_panel",
                show_label=False,
                scale=3,
            )
            with gr.Column(elem_id="transcript_panel", scale=7):
                with gr.Row():
                    transcript = gr.Textbox(label="", lines=28, elem_id="transcript_box", show_label=False, scale=36)
                    with gr.Column(elem_id="font_ctrl_col", scale=1, min_width=24):
                        font_plus_btn = gr.Button("+", elem_id="font_btn_plus")
                        font_minus_btn = gr.Button("−", elem_id="font_btn_minus")
        save_status = gr.Textbox(label="Save Status", interactive=False, visible=False)

        _upload_evt = image_file.change(
            fn=_on_upload,
            inputs=[image_file, state],
            outputs=[image_view, transcript, state, status],
        )
        _upload_evt.then(
            fn=lambda st: str((st or {}).get("last_debug_text", "") if isinstance(st, dict) else ""),
            inputs=[state],
            outputs=[debug_text],
        )

        def _run(
            mode_s: str,
            state_s: Dict[str, Any],
            donut_s: str,
            layout_s: str,
            device_s: str,
            max_len_s: int,
            gray_mode_s: str,
            normalize_background_s: bool,
            background_blur_ksize_s: int,
            background_strength_s: float,
            upscale_factor_s: float,
            upscale_interpolation_s: str,
            binarize_s: bool,
            threshold_method_s: str,
            threshold_block_size_s: int,
            threshold_c_s: int,
            fixed_threshold_s: int,
            morph_close_s: bool,
            morph_close_kernel_s: int,
            remove_small_components_s: bool,
            min_component_area_s: int,
        ):
            bdrc_overrides = _build_bdrc_preprocess_overrides_ui(
                gray_mode=gray_mode_s,
                normalize_background=normalize_background_s,
                background_blur_ksize=int(background_blur_ksize_s),
                background_strength=float(background_strength_s),
                upscale_factor=float(upscale_factor_s),
                upscale_interpolation=upscale_interpolation_s,
                binarize=binarize_s,
                threshold_method=threshold_method_s,
                threshold_block_size=int(threshold_block_size_s),
                threshold_c=int(threshold_c_s),
                fixed_threshold=int(fixed_threshold_s),
                morph_close=morph_close_s,
                morph_close_kernel=int(morph_close_kernel_s),
                remove_small_components=remove_small_components_s,
                min_component_area=int(min_component_area_s),
            )
            if not _is_manual_mode(mode_s):
                return _run_full_auto(
                    state_s,
                    donut_s,
                    layout_s,
                    device_s,
                    int(max_len_s),
                    bdrc_preprocess_overrides=bdrc_overrides,
                )
            img = state_s.get("image")
            overlay = np.asarray(img).astype(np.uint8, copy=False) if img is not None else None
            debug = {
                "ok": True,
                "mode": "manual_waiting",
                "bdrc_preprocess_overrides": bdrc_overrides,
            }
            return (
                overlay,
                _line_text(state_s.get("line_rows") or []),
                "Manual Mode: click an existing line or define ROI with two clicks.",
                state_s,
                json.dumps(debug, ensure_ascii=False, indent=2),
                state_s.get("last_donut_pre"),
                state_s.get("last_donut_post"),
            )

        _run_evt = run_btn.click(
            fn=_run,
            inputs=[
                mode,
                state,
                donut_path,
                layout_path,
                device,
                max_len,
                bdrc_gray_mode,
                bdrc_normalize_bg,
                bdrc_bg_blur_ksize,
                bdrc_bg_strength,
                bdrc_upscale_factor,
                bdrc_upscale_interp,
                bdrc_binarize,
                bdrc_threshold_method,
                bdrc_threshold_block,
                bdrc_threshold_c,
                bdrc_fixed_threshold,
                bdrc_morph_close,
                bdrc_morph_close_kernel,
                bdrc_remove_small_components,
                bdrc_min_component_area,
            ],
            outputs=[image_view, transcript, status, state, debug_json, donut_input_before, donut_input_after],
        )
        _run_evt.then(
            fn=lambda st: str((st or {}).get("last_debug_text", "") if isinstance(st, dict) else ""),
            inputs=[state],
            outputs=[debug_text],
        )

        def _on_select(
            mode_s: str,
            state_s: Dict[str, Any],
            donut_s: str,
            layout_s: str,
            device_s: str,
            max_len_s: int,
            gray_mode_s: str,
            normalize_background_s: bool,
            background_blur_ksize_s: int,
            background_strength_s: float,
            upscale_factor_s: float,
            upscale_interpolation_s: str,
            binarize_s: bool,
            threshold_method_s: str,
            threshold_block_size_s: int,
            threshold_c_s: int,
            fixed_threshold_s: int,
            morph_close_s: bool,
            morph_close_kernel_s: int,
            remove_small_components_s: bool,
            min_component_area_s: int,
            evt: gr.SelectData,
        ):
            bdrc_overrides = _build_bdrc_preprocess_overrides_ui(
                gray_mode=gray_mode_s,
                normalize_background=normalize_background_s,
                background_blur_ksize=int(background_blur_ksize_s),
                background_strength=float(background_strength_s),
                upscale_factor=float(upscale_factor_s),
                upscale_interpolation=upscale_interpolation_s,
                binarize=binarize_s,
                threshold_method=threshold_method_s,
                threshold_block_size=int(threshold_block_size_s),
                threshold_c=int(threshold_c_s),
                fixed_threshold=int(fixed_threshold_s),
                morph_close=morph_close_s,
                morph_close_kernel=int(morph_close_kernel_s),
                remove_small_components=remove_small_components_s,
                min_component_area=int(min_component_area_s),
            )
            if not _is_manual_mode(mode_s):
                image = state_s.get("image")
                if image is None:
                    return (
                        None,
                        _line_text(state_s.get("line_rows") or []),
                        "Please upload an image first.",
                        state_s,
                        "{}",
                        state_s.get("last_donut_pre"),
                        state_s.get("last_donut_post"),
                    )
                idx = getattr(evt, "index", None)
                if not isinstance(idx, (tuple, list)) or len(idx) < 2:
                    overlay = _render_overlay(np.asarray(image).astype(np.uint8, copy=False), state_s.get("line_rows") or [])
                    return (
                        overlay,
                        _line_text(state_s.get("line_rows") or []),
                        "Click position not available.",
                        state_s,
                        "{}",
                        state_s.get("last_donut_pre"),
                        state_s.get("last_donut_post"),
                    )
                try:
                    click_x, click_y = int(idx[0]), int(idx[1])
                except Exception:
                    overlay = _render_overlay(np.asarray(image).astype(np.uint8, copy=False), state_s.get("line_rows") or [])
                    return (
                        overlay,
                        _line_text(state_s.get("line_rows") or []),
                        "Invalid click position.",
                        state_s,
                        "{}",
                        state_s.get("last_donut_pre"),
                        state_s.get("last_donut_post"),
                    )

                src = np.asarray(image).astype(np.uint8, copy=False)
                h, w = src.shape[:2]
                click_x = max(0, min(w - 1, click_x))
                click_y = max(0, min(h - 1, click_y))
                rows = list(state_s.get("line_rows") or [])
                hit = _find_line_hit(rows, click_x, click_y)
                if hit is None:
                    overlay = _render_overlay(src, rows)
                    return (
                        overlay,
                        _line_text(rows),
                        "No detected line at this click position.",
                        state_s,
                        "{}",
                        state_s.get("last_donut_pre"),
                        state_s.get("last_donut_post"),
                    )

                if not (donut_s or "").strip():
                    overlay = _render_overlay(src, rows)
                    return (
                        overlay,
                        _line_text(rows),
                        "DONUT checkpoint is missing.",
                        state_s,
                        "{}",
                        state_s.get("last_donut_pre"),
                        state_s.get("last_donut_post"),
                    )

                x1, y1, x2, y2 = [int(v) for v in hit.get("line_box") or [0, 0, 0, 0]]
                crop = src[y1:y2, x1:x2]
                text, dbg, pre_img, post_img = _run_donut_on_crop(
                    crop,
                    donut_s,
                    device_s,
                    int(max_len_s),
                    bdrc_preprocess_overrides=bdrc_overrides,
                )
                new_row = dict(hit)
                new_row["text"] = text
                new_row["ocr_debug"] = dbg
                new_row["source"] = "overlay_click_debug"
                rows = _add_or_update_line(rows, new_row)
                state_s["line_rows"] = rows
                state_s["last_donut_pre"] = pre_img
                state_s["last_donut_post"] = post_img
                state_s["last_debug_text"] = str(text or "")
                overlay = _render_overlay(src, rows)
                debug = {
                    "ok": True,
                    "mode": str(mode_s),
                    "action": "clicked_detected_line_for_debug_preview",
                    "line_box": [x1, y1, x2, y2],
                    "bdrc_preprocess_overrides": bdrc_overrides,
                }
                return (
                    overlay,
                    _line_text(rows),
                    f"Updated debug preview from clicked line at ({click_x},{click_y}).",
                    state_s,
                    json.dumps(debug, ensure_ascii=False, indent=2),
                    pre_img,
                    post_img,
                )
            return _manual_click(
                state_s,
                donut_s,
                layout_s,
                device_s,
                int(max_len_s),
                bdrc_overrides,
                evt,
            )

        _select_evt = image_view.select(
            fn=_on_select,
            inputs=[
                mode,
                state,
                donut_path,
                layout_path,
                device,
                max_len,
                bdrc_gray_mode,
                bdrc_normalize_bg,
                bdrc_bg_blur_ksize,
                bdrc_bg_strength,
                bdrc_upscale_factor,
                bdrc_upscale_interp,
                bdrc_binarize,
                bdrc_threshold_method,
                bdrc_threshold_block,
                bdrc_threshold_c,
                bdrc_fixed_threshold,
                bdrc_morph_close,
                bdrc_morph_close_kernel,
                bdrc_remove_small_components,
                bdrc_min_component_area,
            ],
            outputs=[image_view, transcript, status, state, debug_json, donut_input_before, donut_input_after],
        )
        _select_evt.then(
            fn=lambda st: str((st or {}).get("last_debug_text", "") if isinstance(st, dict) else ""),
            inputs=[state],
            outputs=[debug_text],
        )

        def _manual_full_roi_with_bdrc(
            mode_s: str,
            state_s: Dict[str, Any],
            donut_s: str,
            device_s: str,
            max_len_s: int,
            gray_mode_s: str,
            normalize_background_s: bool,
            background_blur_ksize_s: int,
            background_strength_s: float,
            upscale_factor_s: float,
            upscale_interpolation_s: str,
            binarize_s: bool,
            threshold_method_s: str,
            threshold_block_size_s: int,
            threshold_c_s: int,
            fixed_threshold_s: int,
            morph_close_s: bool,
            morph_close_kernel_s: int,
            remove_small_components_s: bool,
            min_component_area_s: int,
        ):
            bdrc_overrides = _build_bdrc_preprocess_overrides_ui(
                gray_mode=gray_mode_s,
                normalize_background=normalize_background_s,
                background_blur_ksize=int(background_blur_ksize_s),
                background_strength=float(background_strength_s),
                upscale_factor=float(upscale_factor_s),
                upscale_interpolation=upscale_interpolation_s,
                binarize=binarize_s,
                threshold_method=threshold_method_s,
                threshold_block_size=int(threshold_block_size_s),
                threshold_c=int(threshold_c_s),
                fixed_threshold=int(fixed_threshold_s),
                morph_close=morph_close_s,
                morph_close_kernel=int(morph_close_kernel_s),
                remove_small_components=remove_small_components_s,
                min_component_area=int(min_component_area_s),
            )
            return _manual_full_image_roi(
                mode_s,
                state_s,
                donut_s,
                device_s,
                int(max_len_s),
                bdrc_preprocess_overrides=bdrc_overrides,
            )

        _fullroi_evt = full_roi_btn.click(
            fn=_manual_full_roi_with_bdrc,
            inputs=[
                mode,
                state,
                donut_path,
                device,
                max_len,
                bdrc_gray_mode,
                bdrc_normalize_bg,
                bdrc_bg_blur_ksize,
                bdrc_bg_strength,
                bdrc_upscale_factor,
                bdrc_upscale_interp,
                bdrc_binarize,
                bdrc_threshold_method,
                bdrc_threshold_block,
                bdrc_threshold_c,
                bdrc_fixed_threshold,
                bdrc_morph_close,
                bdrc_morph_close_kernel,
                bdrc_remove_small_components,
                bdrc_min_component_area,
            ],
            outputs=[image_view, transcript, status, state, debug_json, donut_input_before, donut_input_after],
        )
        _fullroi_evt.then(
            fn=lambda st: str((st or {}).get("last_debug_text", "") if isinstance(st, dict) else ""),
            inputs=[state],
            outputs=[debug_text],
        )

        save_btn.click(
            fn=_save_results,
            inputs=[state, transcript, donut_path, layout_path],
            outputs=[save_status],
        )

        font_plus_btn.click(
            fn=None,
            js="""
() => {
  const ta = document.querySelector('#transcript_box textarea');
  if (!ta) return;
  const cur = parseFloat(window.getComputedStyle(ta).fontSize) || 18;
  const next = Math.min(42, cur + 1);
  ta.style.setProperty('font-size', `${next}px`, 'important');
  ta.style.setProperty('line-height', '1.45', 'important');
}
""",
        )
        font_minus_btn.click(
            fn=None,
            js="""
() => {
  const ta = document.querySelector('#transcript_box textarea');
  if (!ta) return;
  const cur = parseFloat(window.getComputedStyle(ta).fontSize) || 18;
  const next = Math.max(10, cur - 1);
  ta.style.setProperty('font-size', `${next}px`, 'important');
  ta.style.setProperty('line-height', '1.45', 'important');
}
""",
        )

        debug_font_plus_btn.click(
            fn=None,
            js="""
() => {
  const ta = document.querySelector('#debug_text_box textarea');
  if (!ta) return;
  const cur = parseFloat(window.getComputedStyle(ta).fontSize) || 20;
  const next = Math.min(48, cur + 1);
  ta.style.setProperty('font-size', `${next}px`, 'important');
  ta.style.setProperty('line-height', '1.45', 'important');
}
""",
        )
        debug_font_minus_btn.click(
            fn=None,
            js="""
() => {
  const ta = document.querySelector('#debug_text_box textarea');
  if (!ta) return;
  const cur = parseFloat(window.getComputedStyle(ta).fontSize) || 20;
  const next = Math.max(10, cur - 1);
  ta.style.setProperty('font-size', `${next}px`, 'important');
  ta.style.setProperty('line-height', '1.45', 'important');
}
""",
        )

        def _toggle_advanced(show: bool):
            visible = bool(show)
            return (
                gr.update(visible=visible),
                gr.update(visible=visible),
                gr.update(visible=visible),
                gr.update(visible=visible),
                gr.update(visible=visible),
                gr.update(visible=visible),
                gr.update(visible=visible),
                gr.update(visible=visible),
            )

        advanced_view.change(
            fn=_toggle_advanced,
            inputs=[advanced_view],
            outputs=[advanced_scan_row, advanced_runtime_row, advanced_bdrc_row, debug_json, donut_input_before, donut_input_after, save_status, advanced_debug_text_row],
        )

        mode.change(
            fn=lambda m: gr.update(visible=_is_manual_mode(m)),
            inputs=[mode],
            outputs=[full_roi_btn],
        )

        def _on_device_change(device_s: str, donut_s: str):
            _rt, msg = _ensure_donut_runtime_loaded(donut_s, device_s)
            return msg

        device.change(
            fn=_on_device_change,
            inputs=[device, donut_path],
            outputs=[status],
        )

    return demo


if __name__ == "__main__":
    # Warm-load DONUT runtime on startup (prefer GPU).
    try:
        _ckpt, _ = _find_donut_checkpoint()
        if _ckpt:
            _ensure_donut_runtime_loaded(_ckpt, "cuda:0")
    except Exception:
        pass
    app = build_ui()
    host = os.environ.get("UI_HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("UI_PORT", "7865"))
    except ValueError:
        port = 7865
    share = os.environ.get("UI_SHARE", "").strip().lower() in {"1", "true", "yes", "on"}
    app.launch(server_name=host, server_port=port, share=share)
