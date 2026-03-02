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
    import torch
except Exception:
    torch = None


ROOT = Path(__file__).resolve().parent


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
        return str(repro_candidates[0]), f"Auto-gewählter DONUT-Checkpoint (repro): {repro_candidates[0]}"

    if plain_candidates:
        plain_candidates = sorted(plain_candidates, key=_checkpoint_step, reverse=True)
        return (
            str(plain_candidates[0]),
            "Auto-gewählter DONUT-Checkpoint (ohne repro, mit tokenizer/image_processor im Parent): "
            f"{plain_candidates[0]}",
        )

    return "", "Kein DONUT-Checkpoint gefunden (erwartet unter models/ocr/ oder models/)."


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
        return "", "Kein Layout-Analysemodell gefunden (erwartet unter models/layoutAnalysis/)."
    return str(candidates[0]), f"Auto-gewähltes Layout-Modell: {candidates[0]}"


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
    for i, rec in enumerate(_sort_lines(lines), start=1):
        x1, y1, x2, y2 = [int(v) for v in rec["line_box"]]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 176, 0), width=3)
        tag = f"line {i}"
        draw.rectangle((x1 + 2, max(0, y1 - 16), x1 + 12 + 7 * len(tag), max(12, y1 - 2)), fill=(0, 0, 0))
        draw.text((x1 + 4, max(0, y1 - 15)), tag, fill=(255, 176, 0), font=font)
    if roi and len(roi) == 4:
        rx1, ry1, rx2, ry2 = [int(v) for v in roi]
        draw.rectangle((rx1, ry1, rx2, ry2), outline=(70, 200, 255), width=3)
    return np.asarray(panel).astype(np.uint8, copy=False)


def _line_text(rows: List[Dict[str, Any]]) -> str:
    lines = _sort_lines(rows)
    return "\n".join([str(r.get("text", "") or "") for r in lines])


def _to_rgb_uint8(path: str) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.uint8)
    return arr


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


def _run_donut_on_crop_fallback(
    crop: np.ndarray,
    donut_checkpoint: str,
    device: str,
    max_len: int,
) -> Tuple[str, Dict[str, Any]]:
    if torch is None:
        raise RuntimeError("PyTorch nicht verfügbar.")
    model_dir, tokenizer_dir, image_processor_dir, err = _resolve_donut_runtime_dirs(donut_checkpoint)
    if err:
        raise RuntimeError(err)
    repro_cfg = _load_donut_repro_bundle_cfg_ui(donut_checkpoint)
    effective_preproc = "bdrc"
    effective_max_len = max(8, int(max_len))
    if bool(repro_cfg.get("has_repro")):
        ip_cfg = repro_cfg.get("image_preprocess") or {}
        if isinstance(ip_cfg, dict):
            pipe = str(ip_cfg.get("pipeline", "") or "").strip().lower()
            if pipe in {"none", "pb", "bdrc"}:
                effective_preproc = pipe
        gcfg = repro_cfg.get("generate_config") or {}
        if isinstance(gcfg, dict) and gcfg.get("max_length") is not None:
            effective_max_len = max(8, int(gcfg["max_length"]))

    runtime = _load_donut_ocr_runtime_cached(
        str(model_dir),
        str(tokenizer_dir),
        str(image_processor_dir),
        str(device or "auto"),
    )
    pil = Image.fromarray(np.asarray(crop).astype(np.uint8)).convert("RGB")
    proc_pil = _apply_donut_ui_preprocess(pil, effective_preproc)
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
        "image_preprocess_pipeline_effective": effective_preproc,
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
    }


def _run_donut_on_crop(
    crop: np.ndarray,
    donut_checkpoint: str,
    device: str,
    max_len: int,
) -> Tuple[str, Dict[str, Any]]:
    _, pred, debug_json = run_donut_ocr_inference_ui(
        image=crop,
        model_root_or_model_dir=donut_checkpoint,
        image_preprocess_pipeline="bdrc",
        device_preference=device,
        generation_max_length=int(max_len),
        num_beams=1,
    )
    try:
        dbg = json.loads(debug_json or "{}")
    except Exception:
        dbg = {"raw_debug": debug_json}
    dbg_obj = dbg if isinstance(dbg, dict) else {"raw_debug": dbg}
    if bool(dbg_obj.get("ok")):
        return str(pred or ""), dbg_obj

    err_msg = str(dbg_obj.get("error", "") or "")
    if "_strip_special_token_strings" in err_msg:
        try:
            return _run_donut_on_crop_fallback(crop, donut_checkpoint, device, max_len)
        except Exception as exc:
            return str(pred or ""), {
                "ok": False,
                "error": err_msg,
                "fallback_attempted": True,
                "fallback_error": f"{type(exc).__name__}: {exc}",
            }
    return str(pred or ""), dbg_obj


def _run_full_auto(
    state: Dict[str, Any],
    donut_checkpoint: str,
    layout_model: str,
    device: str,
    max_len: int,
) -> Tuple[np.ndarray, str, str, Dict[str, Any], str]:
    image = state.get("image")
    if image is None:
        return None, "", "Bitte zuerst ein Bild hochladen.", state, "{}"
    if not donut_checkpoint.strip():
        return image, "", "DONUT-Checkpoint fehlt.", state, "{}"
    if not layout_model.strip():
        return image, "", "Layout-Modell fehlt.", state, "{}"

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
    for rec in line_records_raw:
        box = _normalize_box(rec.get("line_box") or [], w, h)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        crop = src[y1:y2, x1:x2]
        text, dbg = _run_donut_on_crop(crop, donut_checkpoint, device, max_len)
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
    transcript = _line_text(rows)
    debug = {
        "ok": True,
        "mode": "full_auto",
        "split_status": split_status,
        "line_count": len(rows),
        "split_json": json.loads(split_json) if (split_json or "").strip().startswith("{") else split_json,
    }
    return overlay, transcript, f"{split_status} OCR auf {len(rows)} Zeilen.", state, json.dumps(debug, ensure_ascii=False, indent=2)


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
    evt: gr.SelectData,
) -> Tuple[np.ndarray, str, str, Dict[str, Any], str]:
    image = state.get("image")
    if image is None:
        return None, "", "Bitte zuerst ein Bild hochladen.", state, "{}"
    if not donut_checkpoint.strip():
        return image, "", "DONUT-Checkpoint fehlt.", state, "{}"

    idx = getattr(evt, "index", None)
    if not isinstance(idx, (tuple, list)) or len(idx) < 2:
        return np.asarray(image), _line_text(state.get("line_rows") or []), "Klickposition nicht verfügbar.", state, "{}"
    try:
        click_x, click_y = int(idx[0]), int(idx[1])
    except Exception:
        return np.asarray(image), _line_text(state.get("line_rows") or []), "Ungültige Klickposition.", state, "{}"

    src = np.asarray(image).astype(np.uint8, copy=False)
    h, w = src.shape[:2]
    click_x = max(0, min(w - 1, click_x))
    click_y = max(0, min(h - 1, click_y))
    rows = list(state.get("line_rows") or [])

    hit = _find_line_hit(rows, click_x, click_y)
    if hit is not None:
        x1, y1, x2, y2 = [int(v) for v in hit["line_box"]]
        crop = src[y1:y2, x1:x2]
        text, dbg = _run_donut_on_crop(crop, donut_checkpoint, device, max_len)
        new_row = dict(hit)
        new_row["text"] = text
        new_row["ocr_debug"] = dbg
        new_row["source"] = "manual_line_click"
        rows = _add_or_update_line(rows, new_row)
        state["line_rows"] = rows
        overlay = _render_overlay(src, rows)
        return overlay, _line_text(rows), f"Zeile bei Klick ({click_x},{click_y}) neu transkribiert.", state, json.dumps(
            {"ok": True, "mode": "manual", "action": "clicked_existing_line", "line_box": new_row["line_box"]},
            ensure_ascii=False,
            indent=2,
        )

    anchor = state.get("manual_anchor")
    if not isinstance(anchor, (list, tuple)) or len(anchor) != 2:
        state["manual_anchor"] = [int(click_x), int(click_y)]
        overlay = _render_overlay(src, rows)
        return overlay, _line_text(rows), f"Startpunkt gesetzt bei ({click_x},{click_y}). Zweiten Klick für ROI setzen.", state, "{}"

    ax, ay = int(anchor[0]), int(anchor[1])
    state["manual_anchor"] = None
    x1, x2 = sorted([ax, click_x])
    y1, y2 = sorted([ay, click_y])
    if (x2 - x1) < 3 or (y2 - y1) < 3:
        overlay = _render_overlay(src, rows)
        return overlay, _line_text(rows), "ROI zu klein. Bitte größeren Bereich wählen.", state, "{}"
    roi = _normalize_box([x1, y1, x2, y2], w, h)
    return _manual_process_roi(state, donut_checkpoint, device, max_len, roi)


def _manual_process_roi(
    state: Dict[str, Any],
    donut_checkpoint: str,
    device: str,
    max_len: int,
    roi: Optional[List[int]],
) -> Tuple[np.ndarray, str, str, Dict[str, Any], str]:
    image = state.get("image")
    if image is None:
        return None, "", "Bitte zuerst ein Bild hochladen.", state, "{}"
    src = np.asarray(image).astype(np.uint8, copy=False)
    h, w = src.shape[:2]
    rows = list(state.get("line_rows") or [])

    roi = _normalize_box(roi or [], w, h)
    if roi is None:
        overlay = _render_overlay(src, rows)
        return overlay, _line_text(rows), "ROI ungültig.", state, "{}"
    rx1, ry1, rx2, ry2 = roi
    roi_crop = src[ry1:ry2, rx1:rx2]
    roi_w = int(rx2 - rx1)
    roi_h = int(ry2 - ry1)

    created_rows: List[Dict[str, Any]] = []
    if _roi_is_single_line(roi_w, roi_h):
        text, dbg = _run_donut_on_crop(roi_crop, donut_checkpoint, device, max_len)
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
            text, dbg = _run_donut_on_crop(lcrop, donut_checkpoint, device, max_len)
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
            text, dbg = _run_donut_on_crop(roi_crop, donut_checkpoint, device, max_len)
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
    overlay = _render_overlay(src, rows, roi=roi)
    debug = {
        "ok": True,
        "mode": "manual",
        "action": action,
        "roi": roi,
        "new_rows": created_rows,
        "total_rows": len(rows),
    }
    return overlay, _line_text(rows), f"Manuelle Auswahl verarbeitet. Neue/aktualisierte Zeilen: {len(created_rows)}.", state, json.dumps(debug, ensure_ascii=False, indent=2)


def _manual_full_image_roi(
    mode_s: str,
    state_s: Dict[str, Any],
    donut_s: str,
    device_s: str,
    max_len_s: int,
) -> Tuple[np.ndarray, str, str, Dict[str, Any], str]:
    if not str(mode_s).startswith("manu"):
        img = state_s.get("image")
        overlay = np.asarray(img).astype(np.uint8, copy=False) if img is not None else None
        return overlay, _line_text(state_s.get("line_rows") or []), "Button nur im manuellen Modus aktiv.", state_s, "{}"
    img = state_s.get("image")
    if img is None:
        return None, "", "Bitte zuerst ein Bild hochladen.", state_s, "{}"
    src = np.asarray(img).astype(np.uint8, copy=False)
    h, w = src.shape[:2]
    return _manual_process_roi(state_s, donut_s, device_s, int(max_len_s), [0, 0, w, h])


def _on_upload(file_obj: Any, state: Dict[str, Any]) -> Tuple[np.ndarray, str, Dict[str, Any], str]:
    if file_obj is None:
        return None, "", _base_state(), "Kein Bild geladen."
    path = getattr(file_obj, "name", "") or str(file_obj)
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return None, "", _base_state(), f"Datei nicht gefunden: {p}"
    img = _to_rgb_uint8(str(p))
    new_state = _base_state()
    new_state["image_path"] = str(p)
    new_state["image_name"] = p.name
    new_state["image"] = img
    return img, "", new_state, f"Bild geladen: {p.name}"


def _save_results(
    state: Dict[str, Any],
    transcript_text: str,
    donut_checkpoint: str,
    layout_model: str,
) -> str:
    image = state.get("image")
    if image is None:
        return "Nichts zu speichern: kein Bild geladen."
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
    return f"Gespeichert unter: {out_dir}"


def build_ui() -> gr.Blocks:
    donut_ckpt, donut_msg = _find_donut_checkpoint()
    layout_model, layout_msg = _find_layout_model()
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
#transcript_box textarea {
  font-size: 18px !important;
  line-height: 1.45 !important;
}
#font_btn_plus, #font_btn_minus {
  min-width: 36px !important;
}
"""

    with gr.Blocks(title="OCR Workbench (DONUT)", css=ui_css) as demo:
        gr.Markdown("## OCR Workbench (DONUT + Layout)")
        gr.Markdown(
            "Vollautomatik: Layout -> Zeilen -> OCR. Manuell: Klick auf vorhandene Zeile oder ROI per 2 Klicks (Start/Ende)."
        )

        with gr.Row():
            donut_path = gr.Textbox(label="DONUT Checkpoint", value=donut_ckpt)
            layout_path = gr.Textbox(label="Layout Analysemodell", value=layout_model)
        with gr.Row():
            donut_info = gr.Textbox(label="DONUT Auto-Scan", value=donut_msg, interactive=False)
            layout_info = gr.Textbox(label="Layout Auto-Scan", value=layout_msg, interactive=False)

        with gr.Row():
            mode = gr.Radio(choices=["voll-automatische OCR", "manueller Modus"], value="voll-automatische OCR", label="Modus")
            device = gr.Dropdown(choices=["auto", "cuda:0", "cpu"], value="auto", label="Inferenzgerät")
            max_len = gr.Number(label="DONUT generation_max_length", value=512, precision=0)

        status = gr.Textbox(label="Status", interactive=False)
        debug_json = gr.Code(label="Debug JSON", language="json")

        state = gr.State(_base_state())
        font_size_state = gr.State(18)
        font_style_html = gr.HTML("<style>#transcript_box textarea{font-size:18px !important;line-height:1.45 !important;}</style>")
        with gr.Row():
            image_file = gr.File(label="Bild hochladen", file_types=["image"])
            run_btn = gr.Button("OCR Ausführen", variant="primary")
            full_roi_btn = gr.Button("Ganzes Bild als ROI (manuell)")
            save_btn = gr.Button("Save", variant="secondary")

        with gr.Row():
            image_view = gr.Image(
                label="Bild / Overlay",
                type="numpy",
                interactive=True,
                height=300,
                elem_id="ocr_image_panel",
            )
            with gr.Column():
                with gr.Row():
                    gr.Markdown("**Transkribierter Text (editierbar)**")
                    font_minus_btn = gr.Button("−", elem_id="font_btn_minus")
                    font_plus_btn = gr.Button("+", elem_id="font_btn_plus")
                transcript = gr.Textbox(label="", lines=28, elem_id="transcript_box")
        save_status = gr.Textbox(label="Save Status", interactive=False)

        image_file.change(
            fn=_on_upload,
            inputs=[image_file, state],
            outputs=[image_view, transcript, state, status],
        )

        def _run(mode_s: str, state_s: Dict[str, Any], donut_s: str, layout_s: str, device_s: str, max_len_s: int):
            if str(mode_s).startswith("voll"):
                return _run_full_auto(state_s, donut_s, layout_s, device_s, int(max_len_s))
            img = state_s.get("image")
            overlay = np.asarray(img).astype(np.uint8, copy=False) if img is not None else None
            return overlay, _line_text(state_s.get("line_rows") or []), "Manueller Modus: im Bild klicken (Zeile oder ROI mit 2 Klicks).", state_s, "{}"

        run_btn.click(
            fn=_run,
            inputs=[mode, state, donut_path, layout_path, device, max_len],
            outputs=[image_view, transcript, status, state, debug_json],
        )

        def _on_select(mode_s: str, state_s: Dict[str, Any], donut_s: str, layout_s: str, device_s: str, max_len_s: int, evt: gr.SelectData):
            if not str(mode_s).startswith("manu"):
                overlay = state_s.get("image")
                return overlay, _line_text(state_s.get("line_rows") or []), "Klicks sind im manuellen Modus aktiv.", state_s, "{}"
            return _manual_click(state_s, donut_s, layout_s, device_s, int(max_len_s), evt)

        image_view.select(
            fn=_on_select,
            inputs=[mode, state, donut_path, layout_path, device, max_len],
            outputs=[image_view, transcript, status, state, debug_json],
        )

        full_roi_btn.click(
            fn=_manual_full_image_roi,
            inputs=[mode, state, donut_path, device, max_len],
            outputs=[image_view, transcript, status, state, debug_json],
        )

        save_btn.click(
            fn=_save_results,
            inputs=[state, transcript, donut_path, layout_path],
            outputs=[save_status],
        )

        def _change_font(delta: int, current: int) -> Tuple[str, int]:
            size = int(current or 18) + int(delta)
            size = max(10, min(42, size))
            style = f"<style>#transcript_box textarea{{font-size:{size}px !important;line-height:1.45 !important;}}</style>"
            return style, size

        font_plus_btn.click(
            fn=lambda s: _change_font(1, s),
            inputs=[font_size_state],
            outputs=[font_style_html, font_size_state],
        )
        font_minus_btn.click(
            fn=lambda s: _change_font(-1, s),
            inputs=[font_size_state],
            outputs=[font_style_html, font_size_state],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    host = os.environ.get("UI_HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("UI_PORT", "7865"))
    except ValueError:
        port = 7865
    share = os.environ.get("UI_SHARE", "").strip().lower() in {"1", "true", "yes", "on"}
    app.launch(server_name=host, server_port=port, share=share)
