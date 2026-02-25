#!/usr/bin/env python3
"""
PechaBridge Workbench UI.

Provides:
- CLI options audit (reads --help output from project scripts)
- Synthetic data generation
- Dataset preview with YOLO label boxes
- Label Studio export and optional launch
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import zipfile
import csv
import hashlib
import base64
import html
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None

try:
    from scipy.signal import find_peaks
except Exception:
    find_peaks = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as torch_f
except Exception:
    torch = None
    nn = None
    torch_f = None

try:
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
except Exception:
    AutoImageProcessor = None
    AutoModel = None
    AutoTokenizer = None

try:
    from pechabridge.ocr.preprocess import PreprocessConfig as _PBPreprocessConfig, preprocess_patch_image as _pb_preprocess_patch_image
except Exception:
    _PBPreprocessConfig = None
    _pb_preprocess_patch_image = None

try:
    from pechabridge.ocr.preprocess_bdrc import BDRCPreprocessConfig as _BDRCPreprocessConfig, preprocess_image_bdrc as _preprocess_image_bdrc
except Exception:
    _BDRCPreprocessConfig = None
    _preprocess_image_bdrc = None


ROOT = Path(__file__).resolve().parent
DEFAULT_TEXTURE_PROMPT = (
    "scanned printed Tibetan pecha page, paper texture, ink bleed, aged grayscale scan, "
    "realistic Tibetan glyph stroke thickness, subtle hand-written-like ink edge variation"
)

CLI_SCRIPTS = [
    "generate_training_data.py",
    "train_model.py",
    "inference_sbb.py",
    "ocr_on_detections.py",
    "pseudo_label_from_vlm.py",
    "layout_rule_filter.py",
    "run_pseudo_label_workflow.py",
    "scripts/train_image_encoder.py",
    "scripts/train_text_encoder.py",
    "scripts/train_text_hierarchy_vit.py",
    "scripts/eval_text_hierarchy_vit.py",
    "scripts/faiss_text_hierarchy_search.py",
    "cli.py",
]

TRANSFORMER_PARSERS = [
    "paddleocr_vl",
    "qwen25vl",
    "granite_docling",
    "deepseek_ocr",
    "qwen3_vl",
    "groundingdino",
    "florence2",
    "mineru25",
]

VLM_VRAM_HINTS: Dict[str, str] = {
    "paddleocr_vl": "~10-16 GB",
    "qwen25vl": "~14-24 GB",
    "granite_docling": "~4-8 GB",
    "deepseek_ocr": "~10-20 GB",
    "qwen3_vl": "~16-32 GB",
    "groundingdino": "~6-10 GB",
    "florence2": "~8-16 GB",
    "mineru25": "CLI backend, GPU optional",
}

SBB_GRID_COLS = 5
SBB_GRID_ROWS = 9
SBB_GRID_PAGE_SIZE = SBB_GRID_COLS * SBB_GRID_ROWS
SBB_GRID_THUMB_SIZE = 160
SBB_THUMB_CACHE_MAX = 1200
_SBB_THUMB_CACHE: Dict[str, Tuple[float, np.ndarray]] = {}
_SBB_MEAN_CACHE: Dict[str, Tuple[float, float]] = {}
DEFAULT_LAYOUT_CLASS_NAMES: Dict[int, str] = {
    0: "tibetan_number_word",
    1: "tibetan_text",
    2: "chinese_number_word",
}
LABEL_FONT_SIZE = 12
TEXT_HIERARCHY_SUBSET_CHOICES = [
    "PatchDataset (debug)",
    "PatchDataset (all)",
    "PatchDataset (scale=256)",
    "PatchDataset (scale=384)",
    "PatchDataset (scale=512)",
    "Legacy TextHierarchy (line.png)",
    "Legacy TextHierarchy (word crops)",
    "Legacy NumberCrops (tibetan_number_word)",
    "Legacy NumberCrops (chinese_number_word)",
    "All Images",
]


def _load_overlay_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", LABEL_FONT_SIZE)
    except Exception:
        return ImageFont.load_default()


def _run_cmd(cmd: List[str], timeout: int = 3600) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        ok = proc.returncode == 0
        return ok, proc.stdout
    except subprocess.TimeoutExpired:
        return False, f"Command timed out after {timeout}s: {' '.join(cmd)}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def collect_cli_help() -> str:
    blocks: List[str] = ["# CLI Options Audit\n"]
    for script in CLI_SCRIPTS:
        cmd = [sys.executable, script, "-h"]
        ok, out = _run_cmd(cmd, timeout=120)
        blocks.append(f"## {script}")
        blocks.append(f"Command: `{shlex.join(cmd)}`")
        if ok:
            blocks.append("```text")
            blocks.append(out.strip() or "(no output)")
            blocks.append("```")
        else:
            blocks.append("```text")
            blocks.append(out.strip() or "(failed without output)")
            blocks.append("```")
    return "\n".join(blocks)


def _list_datasets(base_dir: str) -> List[str]:
    p = Path(base_dir).expanduser().resolve()
    if not p.exists():
        return []
    out = []
    for child in sorted([c for c in p.iterdir() if c.is_dir()]):
        if (child / "train").exists() or (child / "val").exists():
            out.append(str(child))
    return out


def _list_dataset_names(base_dir: str) -> List[str]:
    p = Path(base_dir).expanduser().resolve()
    if not p.exists():
        return []
    out = []
    for child in sorted([c for c in p.iterdir() if c.is_dir()]):
        if (child / "train").exists() or (child / "val").exists():
            out.append(child.name)
    # Also include YAML dataset configs created at base level
    for yml in sorted([c for c in p.iterdir() if c.is_file() and c.suffix.lower() in {".yaml", ".yml"}]):
        out.append(yml.name)
    return out


def _list_images(split_images_dir: Path) -> List[str]:
    if not split_images_dir.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([p.name for p in split_images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _read_yolo_labels(label_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not label_path.exists():
        return rows
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        try:
            cls = int(parts[0])
        except Exception:
            continue

        # YOLO detect format: class cx cy w h
        if len(parts) == 5:
            try:
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                rows.append(
                    {
                        "class": cls,
                        "type": "bbox",
                        "cx": cx,
                        "cy": cy,
                        "bw": bw,
                        "bh": bh,
                    }
                )
            except Exception:
                continue
            continue

        # YOLO segment format: class x1 y1 x2 y2 ... xn yn
        if len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
            coords: List[float] = []
            ok = True
            for tok in parts[1:]:
                try:
                    coords.append(float(tok))
                except Exception:
                    ok = False
                    break
            if not ok:
                continue
            points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            if len(points) >= 3:
                rows.append(
                    {
                        "class": cls,
                        "type": "polygon",
                        "points": points,
                    }
                )
            continue

    return rows


def _detection_to_yolo(box: Dict[str, Any], img_w: int, img_h: int) -> Optional[Tuple[float, float, float, float]]:
    try:
        x = float(box.get("x", 0.0))
        y = float(box.get("y", 0.0))
        w = float(box.get("width", 0.0))
        h = float(box.get("height", 0.0))
    except Exception:
        return None

    if w <= 0 or h <= 0:
        return None

    # Normalized center-size
    if max(abs(x), abs(y), abs(w), abs(h)) <= 1.5:
        cx, cy, bw, bh = x, y, w, h
    # Packed xyxy in x,y,width,height
    elif x >= 0 and y >= 0 and w > x and h > y and w <= img_w and h <= img_h:
        x1, y1, x2, y2 = x, y, w, h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        cx = ((x1 + x2) / 2.0) / img_w
        cy = ((y1 + y2) / 2.0) / img_h
    # Absolute center-size pixels
    else:
        cx = x / img_w
        cy = y / img_h
        bw = w / img_w
        bh = h / img_h

    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    bw = min(max(bw, 1e-6), 1.0)
    bh = min(max(bh, 1e-6), 1.0)
    return cx, cy, bw, bh


def _map_detection_class(det: Dict[str, Any]) -> int:
    if "class" in det:
        try:
            cls = int(det["class"])
            if cls in (0, 1, 2):
                return cls
        except Exception:
            pass
    label = str(det.get("label", "")).lower()
    text = str(det.get("text", "")).lower()
    combined = f"{label} {text}"
    if "tibetan_number" in combined or "tib_no" in combined:
        return 0
    if "chinese_number" in combined or "chi_no" in combined:
        return 2
    return 1


def _draw_yolo_boxes(image_path: Path, label_path: Path) -> Tuple[np.ndarray, str]:
    # File may still be in-flight during live generation; retry briefly.
    last_err: Optional[Exception] = None
    img = None
    for _ in range(3):
        try:
            with Image.open(image_path) as im:
                img = im.convert("RGB")
            break
        except (OSError, SyntaxError, ValueError) as exc:
            last_err = exc
            time.sleep(0.08)

    if img is None:
        raise OSError(f"Could not open image {image_path}: {last_err}")

    w, h = img.size
    draw = ImageDraw.Draw(img)
    font = _load_overlay_font()
    labels = _read_yolo_labels(label_path)

    class_colors = {
        0: (255, 140, 0),
        1: (0, 220, 255),
        2: (130, 255, 130),
    }
    class_names = dict(DEFAULT_LAYOUT_CLASS_NAMES)
    try:
        # label path pattern: <dataset>/<split>/labels/<file>.txt
        split_dir = label_path.parent.parent
        dataset_root = split_dir.parent
        classes_file = dataset_root / "classes.txt"
        if classes_file.exists():
            lines = [ln.strip() for ln in classes_file.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
            for i, name in enumerate(lines):
                class_names[i] = name
    except Exception:
        pass

    lines = []

    def _draw_tag(x: int, y: int, tag_text: str, color: Tuple[int, int, int]) -> None:
        tx = int(max(0, x))
        ty = int(max(0, y - 16))
        try:
            bbox = draw.textbbox((tx + 2, ty + 1), tag_text, font=font)
            tw = max(12, int(bbox[2] - bbox[0]) + 4)
            th = max(12, int(bbox[3] - bbox[1]) + 2)
        except Exception:
            tw = max(12, 9 * len(tag_text))
            th = 14
        draw.rectangle((tx, ty, tx + tw, ty + th), fill=(0, 0, 0))
        draw.text((tx + 2, ty + 1), tag_text, fill=color, font=font)

    for i, ann in enumerate(labels, start=1):
        cls = int(ann.get("class", -1))
        color = class_colors.get(cls, (255, 80, 80))
        cls_name = class_names.get(cls, f"class_{cls}")
        tag = f"{cls}:{cls_name}"

        if ann.get("type") == "bbox":
            cx = float(ann["cx"])
            cy = float(ann["cy"])
            bw = float(ann["bw"])
            bh = float(ann["bh"])
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            _draw_tag(x1 + 2, y1, tag, color)
            lines.append(f"{i}. class={cls} label={cls_name} bbox cx={cx:.4f} cy={cy:.4f} w={bw:.4f} h={bh:.4f}")
            continue

        points = ann.get("points") or []
        if not isinstance(points, list) or len(points) < 3:
            continue

        norm = True
        for x, y in points:
            if abs(float(x)) > 1.5 or abs(float(y)) > 1.5:
                norm = False
                break
        if norm:
            pts_px = [(int(float(x) * w), int(float(y) * h)) for x, y in points]
        else:
            pts_px = [(int(float(x)), int(float(y))) for x, y in points]

        draw.polygon(pts_px, outline=color)
        xs = [p[0] for p in pts_px]
        ys = [p[1] for p in pts_px]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        _draw_tag(x1 + 2, y1, tag, color)
        lines.append(f"{i}. class={cls} label={cls_name} polygon n={len(pts_px)} bbox=({x1},{y1},{x2},{y2})")

    summary = f"Found {len(labels)} boxes\n" + ("\n".join(lines[:25]) if lines else "")
    if len(lines) > 25:
        summary += f"\n... +{len(lines)-25} more"
    return np.array(img), summary


def _normalize_to_bbox_lines(
    label_src: Path,
    image_path: Path,
    class_id_map: Optional[Dict[int, int]] = None,
) -> List[str]:
    anns = _read_yolo_labels(label_src)
    if not anns:
        return []

    img_w = 0
    img_h = 0
    try:
        with Image.open(image_path) as im:
            img_w, img_h = im.size
    except Exception:
        img_w, img_h = 0, 0

    out_lines: List[str] = []
    for ann in anns:
        cls = int(ann.get("class", -1))
        if cls < 0:
            continue
        if class_id_map and cls in class_id_map:
            cls = int(class_id_map[cls])
        if ann.get("type") == "bbox":
            cx = float(ann["cx"])
            cy = float(ann["cy"])
            bw = float(ann["bw"])
            bh = float(ann["bh"])
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            bw = min(max(bw, 1e-6), 1.0)
            bh = min(max(bh, 1e-6), 1.0)
            out_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            continue

        points = ann.get("points") or []
        if not isinstance(points, list) or len(points) < 3:
            continue

        norm = True
        for x, y in points:
            if abs(float(x)) > 1.5 or abs(float(y)) > 1.5:
                norm = False
                break

        if norm:
            xs = [float(x) for x, _ in points]
            ys = [float(y) for _, y in points]
        else:
            if img_w <= 0 or img_h <= 0:
                continue
            xs = [float(x) / float(img_w) for x, _ in points]
            ys = [float(y) / float(img_h) for _, y in points]

        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        x1 = min(max(x1, 0.0), 1.0)
        y1 = min(max(y1, 0.0), 1.0)
        x2 = min(max(x2, 0.0), 1.0)
        y2 = min(max(y2, 0.0), 1.0)
        bw = max(1e-6, x2 - x1)
        bh = max(1e-6, y2 - y1)
        cx = x1 + (bw / 2.0)
        cy = y1 + (bh / 2.0)
        out_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return out_lines


def _canonical_class_id_from_name(name: str) -> Optional[int]:
    txt = (name or "").strip().lower().replace("_", " ")
    if not txt:
        return None
    if "chinese" in txt:
        return 2
    if "tibetan" in txt and "number" in txt:
        return 0
    if "tibetan" in txt and ("text" in txt or "body" in txt):
        return 1
    if txt in {"text body", "text", "body text"}:
        return 1
    return None


def _infer_ls_class_id_map(split_dir: Path) -> Tuple[Dict[int, int], str]:
    candidates = [split_dir, split_dir.parent]
    classes_by_id: Dict[int, str] = {}

    for base in candidates:
        notes = base / "notes.json"
        if notes.exists():
            try:
                payload = json.loads(notes.read_text(encoding="utf-8"))
                categories = payload.get("categories", []) if isinstance(payload, dict) else []
                for cat in categories:
                    if not isinstance(cat, dict):
                        continue
                    cid = cat.get("id")
                    cname = cat.get("name")
                    if isinstance(cid, int) and isinstance(cname, str):
                        classes_by_id[cid] = cname
            except Exception:
                pass
        if classes_by_id:
            break

    if not classes_by_id:
        for base in candidates:
            classes_file = base / "classes.txt"
            if not classes_file.exists():
                continue
            try:
                lines = [ln.strip() for ln in classes_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
            except Exception:
                lines = []
            for i, name in enumerate(lines):
                classes_by_id[i] = name
            if classes_by_id:
                break

    if not classes_by_id:
        return {}, "No notes.json/classes.txt found; using class ids as-is."

    class_id_map: Dict[int, int] = {}
    mapped = 0
    unknown = 0
    for src_id, src_name in sorted(classes_by_id.items()):
        target = _canonical_class_id_from_name(src_name)
        if target is None:
            unknown += 1
            continue
        class_id_map[int(src_id)] = int(target)
        mapped += 1

    msg = (
        f"Class remap from Label Studio metadata: mapped={mapped}, unknown={unknown}, "
        f"map={{{', '.join([f'{k}->{v}' for k, v in sorted(class_id_map.items())])}}}"
    )
    return class_id_map, msg


def _resolve_dataset_train_split(dataset_choice: str, datasets_base: str) -> Tuple[Optional[Path], str]:
    ds = (dataset_choice or "").strip()
    if not ds:
        return None, "Synthetic dataset is empty."

    base = Path(datasets_base).expanduser().resolve()
    raw = Path(ds).expanduser()
    candidates: List[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((Path.cwd() / raw).resolve())
        candidates.append((base / raw).resolve())

    if raw.suffix.lower() not in {".yaml", ".yml"}:
        candidates.append((base / f"{ds}.yaml").resolve())
        candidates.append((base / f"{ds}.yml").resolve())

    for cand in candidates:
        if cand.is_dir():
            if (cand / "train" / "images").exists() and (cand / "train" / "labels").exists():
                return (cand / "train").resolve(), f"Resolved synthetic split: {cand / 'train'}"
            if (cand / "images").exists() and (cand / "labels").exists():
                return cand.resolve(), f"Resolved synthetic split: {cand}"
            yml = cand / "data.yml"
            yaml_alt = cand / "data.yaml"
            for y in (yml, yaml_alt):
                if y.exists():
                    cand = y
                    break

        if cand.is_file() and cand.suffix.lower() in {".yaml", ".yml"}:
            try:
                cfg = yaml.safe_load(cand.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            root = Path(str(cfg.get("path", ""))).expanduser()
            if not str(root):
                root = cand.parent
            elif not root.is_absolute():
                root = (cand.parent / root).resolve()
            train_rel = str(cfg.get("train", "train/images")).strip() or "train/images"
            train_images = (root / train_rel).resolve() if not Path(train_rel).is_absolute() else Path(train_rel).resolve()
            if train_images.name == "images":
                split_dir = train_images.parent
            else:
                split_dir = (root / "train").resolve()
            if (split_dir / "images").exists() and (split_dir / "labels").exists():
                return split_dir, f"Resolved synthetic split from YAML: {split_dir}"

    return None, f"Could not resolve synthetic dataset: {ds}"


def _resolve_ls_export_split(ls_export_dir: str) -> Tuple[Optional[Path], str]:
    base = Path(ls_export_dir).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        return None, f"LS export directory not found: {base}"

    if (base / "images").exists() and (base / "labels").exists():
        return base, f"Resolved LS split: {base}"

    for split_name in ("train", "val", "test"):
        split_dir = base / split_name
        if (split_dir / "images").exists() and (split_dir / "labels").exists():
            return split_dir, f"Resolved LS split: {split_dir}"

    yml = base / "data.yml"
    yaml_alt = base / "data.yaml"
    yaml_path = yml if yml.exists() else (yaml_alt if yaml_alt.exists() else None)
    if yaml_path is not None:
        try:
            cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            root = Path(str(cfg.get("path", ""))).expanduser()
            if not str(root):
                root = yaml_path.parent
            elif not root.is_absolute():
                root = (yaml_path.parent / root).resolve()
            train_rel = str(cfg.get("train", "train/images")).strip() or "train/images"
            train_images = (root / train_rel).resolve() if not Path(train_rel).is_absolute() else Path(train_rel).resolve()
            split_dir = train_images.parent if train_images.name == "images" else (root / "train").resolve()
            if (split_dir / "images").exists() and (split_dir / "labels").exists():
                return split_dir, f"Resolved LS split from YAML: {split_dir}"
        except Exception:
            pass

    return None, f"Could not find images/labels under LS export dir: {base}"


def _resolve_uploaded_file_path(upload_obj: Any) -> Optional[Path]:
    if upload_obj is None:
        return None
    if isinstance(upload_obj, str):
        p = Path(upload_obj).expanduser().resolve()
        return p if p.exists() else None
    if isinstance(upload_obj, dict):
        for key in ("name", "path"):
            v = upload_obj.get(key)
            if isinstance(v, str) and v:
                p = Path(v).expanduser().resolve()
                if p.exists():
                    return p
    return None


def _resolve_ls_export_split_from_path_or_zip(ls_export_dir: str, ls_export_zip: Any) -> Tuple[Optional[Path], str]:
    zip_path = _resolve_uploaded_file_path(ls_export_zip)
    if zip_path is not None:
        if zip_path.suffix.lower() != ".zip":
            return None, f"Uploaded file is not a .zip: {zip_path}"
        extract_root = Path(tempfile.gettempdir()) / "pechabridge_ls_exports" / f"lszip_{int(time.time() * 1000)}"
        extract_root.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_root)
        except Exception as exc:
            return None, f"Failed to extract ZIP {zip_path}: {type(exc).__name__}: {exc}"

        # Try extraction root first, then direct child dirs (common: single top-level folder).
        split, msg = _resolve_ls_export_split(str(extract_root))
        if split is not None:
            return split, f"{msg} (from ZIP: {zip_path.name})"
        for child in sorted([p for p in extract_root.iterdir() if p.is_dir()]):
            split, msg = _resolve_ls_export_split(str(child))
            if split is not None:
                return split, f"{msg} (from ZIP: {zip_path.name})"
        return None, f"Could not resolve LS export structure after ZIP extraction: {extract_root}"

    return _resolve_ls_export_split(ls_export_dir)


def _inspect_label_format(split_dir: Path) -> Dict[str, int]:
    labels_dir = split_dir / "labels"
    stats = {"files": 0, "rows": 0, "bbox_rows": 0, "polygon_rows": 0, "invalid_rows": 0}
    if not labels_dir.exists():
        return stats

    label_files = sorted(labels_dir.glob("*.txt"))
    stats["files"] = len(label_files)
    for lf in label_files:
        for raw in lf.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            stats["rows"] += 1
            if len(parts) == 5:
                stats["bbox_rows"] += 1
            elif len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                stats["polygon_rows"] += 1
            else:
                stats["invalid_rows"] += 1
    return stats


def _label_format_summary(stats: Dict[str, int]) -> str:
    rows = int(stats.get("rows", 0))
    bbox_rows = int(stats.get("bbox_rows", 0))
    poly_rows = int(stats.get("polygon_rows", 0))
    invalid_rows = int(stats.get("invalid_rows", 0))
    files = int(stats.get("files", 0))

    if rows == 0:
        fmt = "empty"
    elif invalid_rows == rows:
        fmt = "invalid"
    elif poly_rows > 0 and bbox_rows == 0 and invalid_rows == 0:
        fmt = "polygon"
    elif bbox_rows > 0 and poly_rows == 0 and invalid_rows == 0:
        fmt = "bbox"
    else:
        fmt = "mixed"

    return (
        f"Label format check: {fmt} "
        f"(files={files}, rows={rows}, bbox={bbox_rows}, polygon={poly_rows}, invalid={invalid_rows})"
    )


def run_generate_synthetic(
    background_train: str,
    background_val: str,
    output_dir: str,
    dataset_name: str,
    corp_tib_num: str,
    corp_tib_text: str,
    corp_chi_num: str,
    train_samples: int,
    val_samples: int,
    font_tib: str,
    font_chi: str,
    image_width: int,
    image_height: int,
    augmentation: str,
    annotations_file_path: str,
    single_label: bool,
    debug: bool,
):
    cmd = [
        sys.executable,
        "generate_training_data.py",
        "--background_train",
        background_train,
        "--background_val",
        background_val,
        "--output_dir",
        output_dir,
        "--dataset_name",
        dataset_name,
        "--corpora_tibetan_numbers_path",
        corp_tib_num,
        "--corpora_tibetan_text_path",
        corp_tib_text,
        "--corpora_chinese_numbers_path",
        corp_chi_num,
        "--train_samples",
        str(int(train_samples)),
        "--val_samples",
        str(int(val_samples)),
        "--font_path_tibetan",
        font_tib,
        "--font_path_chinese",
        font_chi,
        "--image_width",
        str(int(image_width)),
        "--image_height",
        str(int(image_height)),
        "--augmentation",
        augmentation,
    ]
    if annotations_file_path.strip():
        cmd.extend(["--annotations_file_path", annotations_file_path.strip()])
    if single_label:
        cmd.append("--single_label")
    if debug:
        cmd.append("--debug")

    ok, out = _run_cmd(cmd, timeout=7200)
    dataset_path = str((Path(output_dir).expanduser().resolve() / dataset_name))
    status = "Success" if ok else "Failed"
    return f"{status}\nDataset path: {dataset_path}\n\n{out}", dataset_path


def _latest_generated_sample(dataset_dir: str) -> Tuple[Optional[np.ndarray], str]:
    dataset = Path(dataset_dir).expanduser().resolve()
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    candidates: List[Tuple[float, str, Path]] = []

    for split in ("train", "val"):
        split_images = dataset / split / "images"
        if not split_images.exists():
            continue
        for p in split_images.iterdir():
            if not p.is_file() or p.suffix.lower() not in exts:
                continue
            try:
                mt = p.stat().st_mtime
            except Exception:
                continue
            candidates.append((mt, split, p))

    if not candidates:
        return None, f"Waiting for generated images in {dataset} ..."

    # Try newest first; if the newest file is still being written, fall back.
    for _, split, image_path in sorted(candidates, key=lambda x: x[0], reverse=True):
        label_path = dataset / split / "labels" / f"{image_path.stem}.txt"
        try:
            rendered, summary = _draw_yolo_boxes(image_path, label_path)
            head = f"Latest sample: {split}/images/{image_path.name}"
            return rendered, f"{head}\n{summary}"
        except OSError:
            continue

    return None, f"Waiting for stable image write in {dataset} ..."


def run_generate_synthetic_live(
    background_train: str,
    background_val: str,
    output_dir: str,
    dataset_name: str,
    corp_tib_num: str,
    corp_tib_text: str,
    corp_chi_num: str,
    train_samples: int,
    val_samples: int,
    font_tib: str,
    font_chi: str,
    image_width: int,
    image_height: int,
    augmentation: str,
    annotations_file_path: str,
    single_label: bool,
    debug: bool,
):
    cmd = [
        sys.executable,
        "generate_training_data.py",
        "--background_train",
        background_train,
        "--background_val",
        background_val,
        "--output_dir",
        output_dir,
        "--dataset_name",
        dataset_name,
        "--corpora_tibetan_numbers_path",
        corp_tib_num,
        "--corpora_tibetan_text_path",
        corp_tib_text,
        "--corpora_chinese_numbers_path",
        corp_chi_num,
        "--train_samples",
        str(int(train_samples)),
        "--val_samples",
        str(int(val_samples)),
        "--font_path_tibetan",
        font_tib,
        "--font_path_chinese",
        font_chi,
        "--image_width",
        str(int(image_width)),
        "--image_height",
        str(int(image_height)),
        "--augmentation",
        augmentation,
    ]
    if annotations_file_path.strip():
        cmd.extend(["--annotations_file_path", annotations_file_path.strip()])
    if single_label:
        cmd.append("--single_label")
    if debug:
        cmd.append("--debug")

    dataset_path = str((Path(output_dir).expanduser().resolve() / dataset_name))
    log_lines: List[str] = []

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
    except Exception as exc:
        msg = f"Failed\nDataset path: {dataset_path}\n\n{type(exc).__name__}: {exc}"
        preview_img, preview_txt = _latest_generated_sample(dataset_path)
        yield msg, dataset_path, preview_img, preview_txt
        return

    if proc.stdout is not None:
        try:
            os.set_blocking(proc.stdout.fileno(), False)
        except Exception:
            pass

    # Immediate first paint so users see feedback right away.
    first_img, first_txt = _latest_generated_sample(dataset_path)
    yield f"Running ...\nDataset path: {dataset_path}\n", dataset_path, first_img, first_txt

    last_preview_ts = 0.0
    last_emit_log_count = 0
    partial = ""
    stream_failed = False
    stream_fail_msg = ""
    while True:
        got_output = False
        if (not stream_failed) and proc.stdout is not None:
            try:
                chunk = proc.stdout.read()
            except BlockingIOError:
                chunk = b""
            except Exception as exc:
                # Fallback: continue preview polling even if stdout streaming breaks.
                stream_failed = True
                stream_fail_msg = f"stdout stream disabled ({type(exc).__name__}: {exc})"
                chunk = b""
            if chunk:
                got_output = True
                if isinstance(chunk, bytes):
                    chunk_text = chunk.decode("utf-8", errors="replace")
                else:
                    chunk_text = str(chunk)
                partial += chunk_text
                parts = partial.splitlines(keepends=True)
                keep = ""
                for piece in parts:
                    if piece.endswith("\n"):
                        log_lines.append(piece.rstrip("\n"))
                    else:
                        keep = piece
                partial = keep

        now = time.time()
        should_emit = (now - last_preview_ts >= 0.25) or (len(log_lines) != last_emit_log_count)
        if should_emit:
            preview_img, preview_txt = _latest_generated_sample(dataset_path)
            tail = "\n".join(log_lines[-400:])
            if stream_failed and stream_fail_msg:
                if tail:
                    tail = f"{tail}\n[warning] {stream_fail_msg}"
                else:
                    tail = f"[warning] {stream_fail_msg}"
            running_msg = f"Running ...\nDataset path: {dataset_path}\n\n{tail}"
            yield running_msg, dataset_path, preview_img, preview_txt
            last_preview_ts = now
            last_emit_log_count = len(log_lines)

        if proc.poll() is not None:
            break

        if not got_output:
            time.sleep(0.15)

    if proc.stdout is not None:
        try:
            rest = proc.stdout.read()
        except Exception:
            rest = b""
        if rest:
            if isinstance(rest, bytes):
                partial += rest.decode("utf-8", errors="replace")
            else:
                partial += str(rest)
        if partial:
            log_lines.extend(partial.splitlines())

    ok = proc.returncode == 0
    status = "Success" if ok else "Failed"
    final_log = f"{status}\nDataset path: {dataset_path}\n\n" + "\n".join(log_lines[-1200:])
    preview_img, preview_txt = _latest_generated_sample(dataset_path)
    yield final_log, dataset_path, preview_img, preview_txt


def _list_images_recursive(root_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if not root_dir.exists():
        return []
    return sorted([p for p in root_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def _latest_output_mtime(folder: str) -> Optional[float]:
    out_dir = Path(folder).expanduser().resolve()
    if not out_dir.exists():
        return None
    mts: List[float] = []
    for image_path in _list_images_recursive(out_dir):
        try:
            mts.append(float(image_path.stat().st_mtime))
        except Exception:
            continue
    return max(mts) if mts else None


def _latest_image_from_folder(folder: str, min_mtime: Optional[float] = None) -> Tuple[Optional[np.ndarray], str]:
    out_dir = Path(folder).expanduser().resolve()
    if not out_dir.exists():
        return None, f"Output folder not found: {out_dir}"

    candidates: List[Tuple[float, Path]] = []
    for image_path in _list_images_recursive(out_dir):
        try:
            mt = float(image_path.stat().st_mtime)
            if min_mtime is not None and mt <= float(min_mtime):
                continue
            candidates.append((mt, image_path))
        except Exception:
            continue

    if not candidates:
        if min_mtime is not None:
            return None, f"No new images generated yet in {out_dir}"
        return None, f"No images found in {out_dir}"

    for _, image_path in sorted(candidates, key=lambda x: x[0], reverse=True):
        last_err: Optional[Exception] = None
        for _ in range(3):
            try:
                with Image.open(image_path) as im:
                    arr = np.array(im.convert("RGB"))
                rel = image_path.relative_to(out_dir)
                return arr, f"Latest output: {rel}"
            except (OSError, SyntaxError, ValueError) as exc:
                last_err = exc
                time.sleep(0.08)
        if last_err is not None:
            continue

    return None, f"Waiting for stable output image write in {out_dir} ..."


def preview_latest_texture_output(output_dir: str):
    return _latest_image_from_folder(output_dir)


def _save_debug_upload_image(upload_image: Optional[np.ndarray]) -> Tuple[Optional[Path], Optional[str]]:
    if upload_image is None:
        return None, "Please upload or paste an image for debug inference."

    try:
        arr = np.asarray(upload_image)
        if arr.size == 0:
            return None, "Uploaded debug image is empty."

        if arr.dtype.kind == "f":
            arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
            if float(np.max(arr)) <= 1.0:
                arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        if arr.ndim == 3 and arr.shape[2] > 4:
            arr = arr[:, :, :3]
        if arr.ndim not in {2, 3}:
            return None, f"Unsupported upload shape: {arr.shape}"

        tmp_root = Path(tempfile.gettempdir()) / "pechabridge_texture_debug_uploads"
        run_dir = tmp_root / f"run_{int(time.time() * 1000)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        image_path = run_dir / "debug_input.png"
        Image.fromarray(arr).convert("RGB").save(image_path)
        return run_dir, None
    except Exception as exc:
        return None, f"Failed to serialize uploaded image ({type(exc).__name__}: {exc})"


def run_texture_augment_upload_live(
    upload_image: Optional[np.ndarray],
    output_dir: str,
    model_family: str,
    strength: float,
    steps: int,
    guidance_scale: float,
    seed: Optional[int],
    controlnet_scale: float,
    disable_controlnet: bool,
    lora_path: str,
    lora_scale: float,
    prompt: str,
    base_model_id: str,
    controlnet_model_id: str,
    canny_low: int,
    canny_high: int,
):
    staged_input_dir, err = _save_debug_upload_image(upload_image)
    out_dir = Path(output_dir).expanduser().resolve()

    if err:
        preview_img, preview_txt = _latest_image_from_folder(str(out_dir))
        yield f"Failed: {err}", preview_img, preview_txt
        return

    first = True
    for msg, preview_img, preview_txt in run_texture_augment_live(
        input_dir=str(staged_input_dir),
        output_dir=output_dir,
        model_family=model_family,
        strength=strength,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        controlnet_scale=controlnet_scale,
        disable_controlnet=disable_controlnet,
        lora_path=lora_path,
        lora_scale=lora_scale,
        prompt=prompt,
        base_model_id=base_model_id,
        controlnet_model_id=controlnet_model_id,
        canny_low=canny_low,
        canny_high=canny_high,
    ):
        if first:
            msg = f"Debug upload staged in: {staged_input_dir}\n\n{msg}"
            first = False
        yield msg, preview_img, preview_txt


def run_texture_augment_live(
    input_dir: str,
    output_dir: str,
    model_family: str,
    strength: float,
    steps: int,
    guidance_scale: float,
    seed: Optional[int],
    controlnet_scale: float,
    disable_controlnet: bool,
    lora_path: str,
    lora_scale: float,
    prompt: str,
    base_model_id: str,
    controlnet_model_id: str,
    canny_low: int,
    canny_high: int,
):
    in_dir = Path(input_dir).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    script_path = ROOT / "scripts" / "texture_augment.py"

    if not script_path.exists():
        msg = f"Failed: script not found: {script_path}"
        preview_img, preview_txt = _latest_image_from_folder(str(out_dir))
        yield msg, preview_img, preview_txt
        return
    if not in_dir.exists() or not in_dir.is_dir():
        msg = f"Failed: input_dir does not exist: {in_dir}"
        preview_img, preview_txt = _latest_image_from_folder(str(out_dir))
        yield msg, preview_img, preview_txt
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    pre_run_latest_mtime = _latest_output_mtime(str(out_dir))

    seed_arg: Optional[int] = None
    try:
        if seed is not None:
            seed_int = int(float(seed))
            if seed_int >= 0:
                seed_arg = seed_int
    except Exception:
        seed_arg = None

    cmd = [
        sys.executable,
        "-u",
        str(script_path),
        "--model_family",
        (model_family or "sdxl").strip(),
        "--input_dir",
        str(in_dir),
        "--output_dir",
        str(out_dir),
        "--strength",
        str(float(strength)),
        "--steps",
        str(int(steps)),
        "--guidance_scale",
        str(float(guidance_scale)),
        "--controlnet_scale",
        str(float(controlnet_scale)),
        "--lora_scale",
        str(float(lora_scale)),
        "--prompt",
        (prompt or DEFAULT_TEXTURE_PROMPT).strip(),
        "--base_model_id",
        (base_model_id or "stabilityai/stable-diffusion-xl-base-1.0").strip(),
        "--controlnet_model_id",
        (controlnet_model_id or "diffusers/controlnet-canny-sdxl-1.0").strip(),
        "--canny_low",
        str(int(canny_low)),
        "--canny_high",
        str(int(canny_high)),
    ]
    if disable_controlnet:
        cmd.append("--disable_controlnet")
    if seed_arg is not None:
        cmd.extend(["--seed", str(seed_arg)])
    lora = (lora_path or "").strip()
    if lora:
        cmd.extend(["--lora_path", lora])

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
    except Exception as exc:
        msg = f"Failed\nOutput dir: {out_dir}\n\n{type(exc).__name__}: {exc}"
        preview_img, preview_txt = _latest_image_from_folder(str(out_dir))
        yield msg, preview_img, preview_txt
        return

    if proc.stdout is not None:
        try:
            os.set_blocking(proc.stdout.fileno(), False)
        except Exception:
            pass

    log_lines: List[str] = []
    partial = ""
    last_emit_ts = 0.0
    stream_failed = False
    stream_fail_msg = ""

    preview_img, preview_txt = _latest_image_from_folder(str(out_dir), min_mtime=pre_run_latest_mtime)
    yield (
        f"Running {str(model_family).upper()} texture augmentation ...\\n"
        f"Input dir: {in_dir}\n"
        f"Output dir: {out_dir}\n"
        f"Command: {shlex.join(cmd)}\n",
        preview_img,
        preview_txt,
    )

    while True:
        got_output = False
        if (not stream_failed) and proc.stdout is not None:
            try:
                chunk = proc.stdout.read()
            except BlockingIOError:
                chunk = b""
            except Exception as exc:
                stream_failed = True
                stream_fail_msg = f"stdout stream disabled ({type(exc).__name__}: {exc})"
                chunk = b""

            if chunk:
                got_output = True
                if isinstance(chunk, bytes):
                    chunk_text = chunk.decode("utf-8", errors="replace")
                else:
                    chunk_text = str(chunk)
                partial += chunk_text.replace("\r", "\n")
                parts = partial.splitlines(keepends=True)
                keep = ""
                for piece in parts:
                    if piece.endswith("\n"):
                        log_lines.append(piece.rstrip("\n"))
                    else:
                        keep = piece
                partial = keep

        now = time.time()
        if now - last_emit_ts >= 1.0:
            preview_img, preview_txt = _latest_image_from_folder(str(out_dir), min_mtime=pre_run_latest_mtime)
            tail = _tail_lines_newest_first(log_lines, 800)
            if stream_failed and stream_fail_msg:
                tail = f"{tail}\n[warning] {stream_fail_msg}" if tail else f"[warning] {stream_fail_msg}"
            running_msg = (
                f"Running {str(model_family).upper()} texture augmentation ...\\n"
                f"Input dir: {in_dir}\n"
                f"Output dir: {out_dir}\n\n{tail}"
            )
            yield running_msg, preview_img, preview_txt
            last_emit_ts = now

        if proc.poll() is not None:
            break

        if not got_output:
            time.sleep(0.15)

    if proc.stdout is not None:
        try:
            rest = proc.stdout.read()
        except Exception:
            rest = b""
        if rest:
            if isinstance(rest, bytes):
                partial += rest.decode("utf-8", errors="replace").replace("\r", "\n")
            else:
                partial += str(rest).replace("\r", "\n")
        if partial:
            log_lines.extend(partial.splitlines())

    ok = proc.returncode == 0
    status = "Success" if ok else "Failed"
    preview_img, preview_txt = _latest_image_from_folder(str(out_dir), min_mtime=pre_run_latest_mtime)
    if preview_img is None:
        preview_img, preview_txt = _latest_image_from_folder(str(out_dir))
    final_msg = (
        f"{status}\nInput dir: {in_dir}\nOutput dir: {out_dir}\n\n"
        + _tail_lines_newest_first(log_lines, 3000)
    )
    yield final_msg, preview_img, preview_txt


def run_prepare_texture_lora_dataset_live(
    input_dir: str,
    output_dir: str,
    crop_size: int,
    num_crops_per_page: int,
    min_edge_density: float,
    seed: Optional[int],
    canny_low: int,
    canny_high: int,
):
    in_dir = Path(input_dir).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    script_path = ROOT / "scripts" / "prepare_texture_lora_dataset.py"
    preview_root = out_dir / "images"

    if not script_path.exists():
        msg = f"Failed: script not found: {script_path}"
        preview_img, preview_txt = _latest_image_from_folder(str(preview_root))
        yield msg, preview_img, preview_txt, str(out_dir)
        return
    if not in_dir.exists() or not in_dir.is_dir():
        msg = f"Failed: input_dir does not exist: {in_dir}"
        preview_img, preview_txt = _latest_image_from_folder(str(preview_root))
        yield msg, preview_img, preview_txt, str(out_dir)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    seed_arg = 42
    try:
        if seed is not None:
            seed_arg = int(float(seed))
    except Exception:
        seed_arg = 42

    cmd = [
        sys.executable,
        "-u",
        str(script_path),
        "--input_dir",
        str(in_dir),
        "--output_dir",
        str(out_dir),
        "--crop_size",
        str(int(crop_size)),
        "--num_crops_per_page",
        str(int(num_crops_per_page)),
        "--min_edge_density",
        str(float(min_edge_density)),
        "--seed",
        str(seed_arg),
        "--canny_low",
        str(int(canny_low)),
        "--canny_high",
        str(int(canny_high)),
    ]

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
    except Exception as exc:
        msg = f"Failed\nOutput dir: {out_dir}\n\n{type(exc).__name__}: {exc}"
        preview_img, preview_txt = _latest_image_from_folder(str(preview_root))
        yield msg, preview_img, preview_txt, str(out_dir)
        return

    if proc.stdout is not None:
        try:
            os.set_blocking(proc.stdout.fileno(), False)
        except Exception:
            pass

    log_lines: List[str] = []
    partial = ""
    last_emit_ts = 0.0
    stream_failed = False
    stream_fail_msg = ""

    preview_img, preview_txt = _latest_image_from_folder(str(preview_root))
    yield (
        "Preparing texture LoRA dataset ...\n"
        f"Input dir: {in_dir}\n"
        f"Output dir: {out_dir}\n"
        f"Command: {shlex.join(cmd)}\n",
        preview_img,
        preview_txt,
        str(out_dir),
    )

    while True:
        got_output = False
        if (not stream_failed) and proc.stdout is not None:
            try:
                chunk = proc.stdout.read()
            except BlockingIOError:
                chunk = b""
            except Exception as exc:
                stream_failed = True
                stream_fail_msg = f"stdout stream disabled ({type(exc).__name__}: {exc})"
                chunk = b""

            if chunk:
                got_output = True
                if isinstance(chunk, bytes):
                    chunk_text = chunk.decode("utf-8", errors="replace")
                else:
                    chunk_text = str(chunk)
                partial += chunk_text.replace("\r", "\n")
                parts = partial.splitlines(keepends=True)
                keep = ""
                for piece in parts:
                    if piece.endswith("\n"):
                        log_lines.append(piece.rstrip("\n"))
                    else:
                        keep = piece
                partial = keep

        now = time.time()
        if now - last_emit_ts >= 1.0:
            preview_img, preview_txt = _latest_image_from_folder(str(preview_root))
            tail = _tail_lines_newest_first(log_lines, 800)
            if stream_failed and stream_fail_msg:
                tail = f"{tail}\n[warning] {stream_fail_msg}" if tail else f"[warning] {stream_fail_msg}"
            running_msg = (
                "Preparing texture LoRA dataset ...\n"
                f"Input dir: {in_dir}\n"
                f"Output dir: {out_dir}\n\n{tail}"
            )
            yield running_msg, preview_img, preview_txt, str(out_dir)
            last_emit_ts = now

        if proc.poll() is not None:
            break

        if not got_output:
            time.sleep(0.15)

    if proc.stdout is not None:
        try:
            rest = proc.stdout.read()
        except Exception:
            rest = b""
        if rest:
            if isinstance(rest, bytes):
                partial += rest.decode("utf-8", errors="replace").replace("\r", "\n")
            else:
                partial += str(rest).replace("\r", "\n")
        if partial:
            log_lines.extend(partial.splitlines())

    ok = proc.returncode == 0
    status = "Success" if ok else "Failed"
    preview_img, preview_txt = _latest_image_from_folder(str(preview_root))
    final_msg = (
        f"{status}\nInput dir: {in_dir}\nOutput dir: {out_dir}\n\n"
        + _tail_lines_newest_first(log_lines, 3000)
    )
    yield final_msg, preview_img, preview_txt, str(out_dir)


def run_train_texture_lora_live(
    dataset_dir: str,
    output_dir: str,
    model_family: str,
    resolution: int,
    batch_size: int,
    lr: float,
    max_train_steps: int,
    rank: int,
    lora_alpha: float,
    mixed_precision: str,
    gradient_checkpointing: bool,
    prompt: str,
    seed: Optional[int],
    base_model_id: str,
    train_text_encoder: bool,
    num_workers: int,
    lora_weights_name: str,
):
    dataset_path = Path(dataset_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    script_path = ROOT / "scripts" / "train_texture_lora_sdxl.py"

    lora_name = (lora_weights_name or "texture_lora.safetensors").strip()
    expected_lora_path = output_path / lora_name
    expected_cfg_path = output_path / "training_config.json"

    if not script_path.exists():
        msg = f"Failed: script not found: {script_path}"
        yield msg, str(expected_lora_path), str(expected_cfg_path), str(expected_lora_path)
        return
    if not dataset_path.exists() or not dataset_path.is_dir():
        msg = f"Failed: dataset_dir does not exist: {dataset_path}"
        yield msg, str(expected_lora_path), str(expected_cfg_path), str(expected_lora_path)
        return

    output_path.mkdir(parents=True, exist_ok=True)

    seed_arg = 42
    try:
        if seed is not None:
            seed_arg = int(float(seed))
    except Exception:
        seed_arg = 42

    cmd = [
        sys.executable,
        "-u",
        str(script_path),
        "--model_family",
        (model_family or "sdxl").strip(),
        "--dataset_dir",
        str(dataset_path),
        "--output_dir",
        str(output_path),
        "--resolution",
        str(int(resolution)),
        "--batch_size",
        str(int(batch_size)),
        "--lr",
        str(float(lr)),
        "--max_train_steps",
        str(int(max_train_steps)),
        "--rank",
        str(int(rank)),
        "--lora_alpha",
        str(float(lora_alpha)),
        "--mixed_precision",
        (mixed_precision or "no").strip(),
        "--prompt",
        (prompt or DEFAULT_TEXTURE_PROMPT).strip(),
        "--seed",
        str(seed_arg),
        "--base_model_id",
        (base_model_id or "stabilityai/stable-diffusion-xl-base-1.0").strip(),
        "--num_workers",
        str(int(num_workers)),
        "--lora_weights_name",
        lora_name,
    ]
    if gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    if train_text_encoder:
        cmd.append("--train_text_encoder")

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
    except Exception as exc:
        msg = f"Failed\nOutput dir: {output_path}\n\n{type(exc).__name__}: {exc}"
        yield msg, str(expected_lora_path), str(expected_cfg_path), str(expected_lora_path)
        return

    if proc.stdout is not None:
        try:
            os.set_blocking(proc.stdout.fileno(), False)
        except Exception:
            pass

    log_lines: List[str] = []
    partial = ""
    last_emit_ts = 0.0
    stream_failed = False
    stream_fail_msg = ""

    yield (
        "Training texture LoRA ...\n"
        f"Dataset dir: {dataset_path}\n"
        f"Output dir: {output_path}\n"
        f"Expected LoRA: {expected_lora_path}\n"
        f"Command: {shlex.join(cmd)}\n",
        str(expected_lora_path),
        str(expected_cfg_path),
        str(expected_lora_path),
    )

    while True:
        got_output = False
        if (not stream_failed) and proc.stdout is not None:
            try:
                chunk = proc.stdout.read()
            except BlockingIOError:
                chunk = b""
            except Exception as exc:
                stream_failed = True
                stream_fail_msg = f"stdout stream disabled ({type(exc).__name__}: {exc})"
                chunk = b""

            if chunk:
                got_output = True
                if isinstance(chunk, bytes):
                    chunk_text = chunk.decode("utf-8", errors="replace")
                else:
                    chunk_text = str(chunk)
                partial += chunk_text.replace("\r", "\n")
                parts = partial.splitlines(keepends=True)
                keep = ""
                for piece in parts:
                    if piece.endswith("\n"):
                        log_lines.append(piece.rstrip("\n"))
                    else:
                        keep = piece
                partial = keep

        now = time.time()
        if now - last_emit_ts >= 1.0:
            tail = _tail_lines_newest_first(log_lines, 800)
            if stream_failed and stream_fail_msg:
                tail = f"{tail}\n[warning] {stream_fail_msg}" if tail else f"[warning] {stream_fail_msg}"
            running_msg = (
                "Training texture LoRA ...\n"
                f"Dataset dir: {dataset_path}\n"
                f"Output dir: {output_path}\n"
                f"Expected LoRA: {expected_lora_path}\n\n{tail}"
            )
            yield running_msg, str(expected_lora_path), str(expected_cfg_path), str(expected_lora_path)
            last_emit_ts = now

        if proc.poll() is not None:
            break

        if not got_output:
            time.sleep(0.15)

    if proc.stdout is not None:
        try:
            rest = proc.stdout.read()
        except Exception:
            rest = b""
        if rest:
            if isinstance(rest, bytes):
                partial += rest.decode("utf-8", errors="replace").replace("\r", "\n")
            else:
                partial += str(rest).replace("\r", "\n")
        if partial:
            log_lines.extend(partial.splitlines())

    ok = proc.returncode == 0
    status = "Success" if ok else "Failed"
    final_msg = (
        f"{status}\nDataset dir: {dataset_path}\nOutput dir: {output_path}\n"
        f"LoRA path: {expected_lora_path}\n\n"
        + _tail_lines_newest_first(log_lines, 3000)
    )
    yield final_msg, str(expected_lora_path), str(expected_cfg_path), str(expected_lora_path)


def run_train_image_encoder_live(
    input_dir: str,
    output_dir: str,
    model_name_or_path: str,
    resolution: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_train_epochs: int,
    max_train_steps: int,
    warmup_steps: int,
    projection_dim: int,
    temperature: float,
    mixed_precision: str,
    gradient_checkpointing: bool,
    freeze_backbone: bool,
    num_workers: int,
    seed: Optional[int],
    checkpoint_every_steps: int,
):
    input_path = Path(input_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    script_path = ROOT / "scripts" / "train_image_encoder.py"

    expected_backbone = output_path / "image_encoder_backbone"
    expected_head = output_path / "image_projection_head.pt"
    expected_cfg = output_path / "training_config.json"

    if not script_path.exists():
        msg = f"Failed: script not found: {script_path}"
        yield msg, str(expected_backbone), str(expected_head), str(expected_cfg)
        return
    if not input_path.exists() or not input_path.is_dir():
        msg = f"Failed: input_dir does not exist: {input_path}"
        yield msg, str(expected_backbone), str(expected_head), str(expected_cfg)
        return

    output_path.mkdir(parents=True, exist_ok=True)

    seed_arg = 42
    try:
        if seed is not None:
            seed_arg = int(float(seed))
    except Exception:
        seed_arg = 42

    cmd = [
        sys.executable,
        "-u",
        str(script_path),
        "--input_dir",
        str(input_path),
        "--output_dir",
        str(output_path),
        "--model_name_or_path",
        (model_name_or_path or "facebook/dinov2-base").strip(),
        "--resolution",
        str(int(resolution)),
        "--batch_size",
        str(int(batch_size)),
        "--lr",
        str(float(lr)),
        "--weight_decay",
        str(float(weight_decay)),
        "--num_train_epochs",
        str(int(num_train_epochs)),
        "--max_train_steps",
        str(int(max_train_steps)),
        "--warmup_steps",
        str(int(warmup_steps)),
        "--projection_dim",
        str(int(projection_dim)),
        "--temperature",
        str(float(temperature)),
        "--mixed_precision",
        (mixed_precision or "fp16").strip(),
        "--num_workers",
        str(int(num_workers)),
        "--seed",
        str(seed_arg),
        "--checkpoint_every_steps",
        str(int(checkpoint_every_steps)),
    ]
    if gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    if freeze_backbone:
        cmd.append("--freeze_backbone")

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
    except Exception as exc:
        msg = f"Failed\nOutput dir: {output_path}\n\n{type(exc).__name__}: {exc}"
        yield msg, str(expected_backbone), str(expected_head), str(expected_cfg)
        return

    if proc.stdout is not None:
        try:
            os.set_blocking(proc.stdout.fileno(), False)
        except Exception:
            pass

    log_lines: List[str] = []
    partial = ""
    last_emit_ts = 0.0
    stream_failed = False
    stream_fail_msg = ""

    yield (
        "Training image encoder ...\n"
        f"Input dir: {input_path}\n"
        f"Output dir: {output_path}\n"
        f"Expected backbone: {expected_backbone}\n"
        f"Command: {shlex.join(cmd)}\n",
        str(expected_backbone),
        str(expected_head),
        str(expected_cfg),
    )

    while True:
        got_output = False
        if (not stream_failed) and proc.stdout is not None:
            try:
                chunk = proc.stdout.read()
            except BlockingIOError:
                chunk = b""
            except Exception as exc:
                stream_failed = True
                stream_fail_msg = f"stdout stream disabled ({type(exc).__name__}: {exc})"
                chunk = b""

            if chunk:
                got_output = True
                chunk_text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else str(chunk)
                partial += chunk_text.replace("\r", "\n")
                parts = partial.splitlines(keepends=True)
                keep = ""
                for piece in parts:
                    if piece.endswith("\n"):
                        log_lines.append(piece.rstrip("\n"))
                    else:
                        keep = piece
                partial = keep

        now = time.time()
        if now - last_emit_ts >= 1.0:
            tail = _tail_lines_newest_first(log_lines, 800)
            if stream_failed and stream_fail_msg:
                tail = f"{tail}\n[warning] {stream_fail_msg}" if tail else f"[warning] {stream_fail_msg}"
            running_msg = (
                "Training image encoder ...\n"
                f"Input dir: {input_path}\n"
                f"Output dir: {output_path}\n"
                f"Expected backbone: {expected_backbone}\n\n{tail}"
            )
            yield running_msg, str(expected_backbone), str(expected_head), str(expected_cfg)
            last_emit_ts = now

        if proc.poll() is not None:
            break

        if not got_output:
            time.sleep(0.15)

    if proc.stdout is not None:
        try:
            rest = proc.stdout.read()
        except Exception:
            rest = b""
        if rest:
            partial += rest.decode("utf-8", errors="replace").replace("\r", "\n") if isinstance(rest, bytes) else str(rest).replace("\r", "\n")
        if partial:
            log_lines.extend(partial.splitlines())

    ok = proc.returncode == 0
    status = "Success" if ok else "Failed"
    final_msg = (
        f"{status}\nInput dir: {input_path}\nOutput dir: {output_path}\n"
        f"Backbone path: {expected_backbone}\n\n"
        + _tail_lines_newest_first(log_lines, 3000)
    )
    yield final_msg, str(expected_backbone), str(expected_head), str(expected_cfg)


def run_train_text_encoder_live(
    input_dir: str,
    output_dir: str,
    model_name_or_path: str,
    normalization: str,
    min_chars: int,
    max_chars: int,
    max_length: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_train_epochs: int,
    max_train_steps: int,
    warmup_steps: int,
    projection_dim: int,
    temperature: float,
    mixed_precision: str,
    gradient_checkpointing: bool,
    freeze_backbone: bool,
    num_workers: int,
    seed: Optional[int],
    checkpoint_every_steps: int,
):
    input_path = Path(input_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    script_path = ROOT / "scripts" / "train_text_encoder.py"

    expected_backbone = output_path / "text_encoder_backbone"
    expected_tokenizer = output_path / "text_tokenizer"
    expected_head = output_path / "text_projection_head.pt"
    expected_cfg = output_path / "training_config.json"

    if not script_path.exists():
        msg = f"Failed: script not found: {script_path}"
        yield msg, str(expected_backbone), str(expected_tokenizer), str(expected_head), str(expected_cfg)
        return
    if not input_path.exists() or not input_path.is_dir():
        msg = f"Failed: input_dir does not exist: {input_path}"
        yield msg, str(expected_backbone), str(expected_tokenizer), str(expected_head), str(expected_cfg)
        return

    output_path.mkdir(parents=True, exist_ok=True)

    seed_arg = 42
    try:
        if seed is not None:
            seed_arg = int(float(seed))
    except Exception:
        seed_arg = 42

    cmd = [
        sys.executable,
        "-u",
        str(script_path),
        "--input_dir",
        str(input_path),
        "--output_dir",
        str(output_path),
        "--model_name_or_path",
        (model_name_or_path or "google/byt5-small").strip(),
        "--normalization",
        (normalization or "NFC").strip(),
        "--min_chars",
        str(int(min_chars)),
        "--max_chars",
        str(int(max_chars)),
        "--max_length",
        str(int(max_length)),
        "--batch_size",
        str(int(batch_size)),
        "--lr",
        str(float(lr)),
        "--weight_decay",
        str(float(weight_decay)),
        "--num_train_epochs",
        str(int(num_train_epochs)),
        "--max_train_steps",
        str(int(max_train_steps)),
        "--warmup_steps",
        str(int(warmup_steps)),
        "--projection_dim",
        str(int(projection_dim)),
        "--temperature",
        str(float(temperature)),
        "--mixed_precision",
        (mixed_precision or "fp16").strip(),
        "--num_workers",
        str(int(num_workers)),
        "--seed",
        str(seed_arg),
        "--checkpoint_every_steps",
        str(int(checkpoint_every_steps)),
    ]
    if gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    if freeze_backbone:
        cmd.append("--freeze_backbone")

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
    except Exception as exc:
        msg = f"Failed\nOutput dir: {output_path}\n\n{type(exc).__name__}: {exc}"
        yield msg, str(expected_backbone), str(expected_tokenizer), str(expected_head), str(expected_cfg)
        return

    if proc.stdout is not None:
        try:
            os.set_blocking(proc.stdout.fileno(), False)
        except Exception:
            pass

    log_lines: List[str] = []
    partial = ""
    last_emit_ts = 0.0
    stream_failed = False
    stream_fail_msg = ""

    yield (
        "Training text encoder ...\n"
        f"Input dir: {input_path}\n"
        f"Output dir: {output_path}\n"
        f"Expected backbone: {expected_backbone}\n"
        f"Command: {shlex.join(cmd)}\n",
        str(expected_backbone),
        str(expected_tokenizer),
        str(expected_head),
        str(expected_cfg),
    )

    while True:
        got_output = False
        if (not stream_failed) and proc.stdout is not None:
            try:
                chunk = proc.stdout.read()
            except BlockingIOError:
                chunk = b""
            except Exception as exc:
                stream_failed = True
                stream_fail_msg = f"stdout stream disabled ({type(exc).__name__}: {exc})"
                chunk = b""

            if chunk:
                got_output = True
                chunk_text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else str(chunk)
                partial += chunk_text.replace("\r", "\n")
                parts = partial.splitlines(keepends=True)
                keep = ""
                for piece in parts:
                    if piece.endswith("\n"):
                        log_lines.append(piece.rstrip("\n"))
                    else:
                        keep = piece
                partial = keep

        now = time.time()
        if now - last_emit_ts >= 1.0:
            tail = _tail_lines_newest_first(log_lines, 800)
            if stream_failed and stream_fail_msg:
                tail = f"{tail}\n[warning] {stream_fail_msg}" if tail else f"[warning] {stream_fail_msg}"
            running_msg = (
                "Training text encoder ...\n"
                f"Input dir: {input_path}\n"
                f"Output dir: {output_path}\n"
                f"Expected backbone: {expected_backbone}\n\n{tail}"
            )
            yield running_msg, str(expected_backbone), str(expected_tokenizer), str(expected_head), str(expected_cfg)
            last_emit_ts = now

        if proc.poll() is not None:
            break

        if not got_output:
            time.sleep(0.15)

    if proc.stdout is not None:
        try:
            rest = proc.stdout.read()
        except Exception:
            rest = b""
        if rest:
            partial += rest.decode("utf-8", errors="replace").replace("\r", "\n") if isinstance(rest, bytes) else str(rest).replace("\r", "\n")
        if partial:
            log_lines.extend(partial.splitlines())

    ok = proc.returncode == 0
    status = "Success" if ok else "Failed"
    final_msg = (
        f"{status}\nInput dir: {input_path}\nOutput dir: {output_path}\n"
        f"Backbone path: {expected_backbone}\n\n"
        + _tail_lines_newest_first(log_lines, 3000)
    )
    yield final_msg, str(expected_backbone), str(expected_tokenizer), str(expected_head), str(expected_cfg)


def _read_json_file_pretty(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return "{}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return "{}"
    try:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception:
        return "{}"


def run_gen_patches_live(
    config_path: str,
    model_path: str,
    input_dir: str,
    output_dir: str,
    conf: float,
    imgsz: int,
    device: str,
    no_samples: int,
    seed: int,
    target_height: int,
    widths: str,
    overlap: float,
    rmin: float,
    prominence: float,
    n_per_line_per_scale: int,
    p_aligned: float,
    debug_dump: int,
):
    cli_path = ROOT / "cli.py"
    cfg = (config_path or "").strip()
    model = Path(model_path).expanduser().resolve()
    in_dir = Path(input_dir).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    expected_meta = (out_dir / "meta" / "patches.parquet").resolve()

    if not cli_path.exists():
        msg = f"Failed: script not found: {cli_path}"
        yield msg, str(out_dir), str(expected_meta)
        return
    if not model.exists() or not model.is_file():
        msg = f"Failed: model path not found: {model}"
        yield msg, str(out_dir), str(expected_meta)
        return
    if not in_dir.exists() or not in_dir.is_dir():
        msg = f"Failed: input_dir does not exist: {in_dir}"
        yield msg, str(out_dir), str(expected_meta)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        str(cli_path),
        "gen-patches",
    ]
    if cfg:
        cmd.extend(["--config", str(Path(cfg).expanduser().resolve())])
    cmd.extend(
        [
            "--model",
            str(model),
            "--input-dir",
            str(in_dir),
            "--output-dir",
            str(out_dir),
            "--conf",
            str(float(conf)),
            "--imgsz",
            str(int(imgsz)),
            "--device",
            str((device or "").strip()),
            "--no-samples",
            str(int(no_samples)),
            "--seed",
            str(int(seed)),
            "--target-height",
            str(int(target_height)),
            "--widths",
            str((widths or "").strip()),
            "--overlap",
            str(float(overlap)),
            "--rmin",
            str(float(rmin)),
            "--prominence",
            str(float(prominence)),
            "--n-per-line-per-scale",
            str(int(n_per_line_per_scale)),
            "--p-aligned",
            str(float(p_aligned)),
            "--debug-dump",
            str(int(debug_dump)),
        ]
    )

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
    except Exception as exc:
        msg = f"Failed\nOutput dir: {out_dir}\n\n{type(exc).__name__}: {exc}"
        yield msg, str(out_dir), str(expected_meta)
        return

    if proc.stdout is not None:
        try:
            os.set_blocking(proc.stdout.fileno(), False)
        except Exception:
            pass

    log_lines: List[str] = []
    partial = ""
    last_emit_ts = 0.0
    stream_failed = False
    stream_fail_msg = ""

    yield (
        "Generating patch dataset ...\n"
        f"Model: {model}\n"
        f"Input dir: {in_dir}\n"
        f"Output dir: {out_dir}\n"
        f"Expected metadata: {expected_meta}\n"
        f"Command: {shlex.join(cmd)}\n",
        str(out_dir),
        str(expected_meta),
    )

    while True:
        got_output = False
        if (not stream_failed) and proc.stdout is not None:
            try:
                chunk = proc.stdout.read()
            except BlockingIOError:
                chunk = b""
            except Exception as exc:
                stream_failed = True
                stream_fail_msg = f"stdout stream disabled ({type(exc).__name__}: {exc})"
                chunk = b""

            if chunk:
                got_output = True
                chunk_text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else str(chunk)
                partial += chunk_text.replace("\r", "\n")
                parts = partial.splitlines(keepends=True)
                keep = ""
                for piece in parts:
                    if piece.endswith("\n"):
                        log_lines.append(piece.rstrip("\n"))
                    else:
                        keep = piece
                partial = keep

        now = time.time()
        if now - last_emit_ts >= 1.0:
            tail = _tail_lines_newest_first(log_lines, 800)
            if stream_failed and stream_fail_msg:
                tail = f"{tail}\n[warning] {stream_fail_msg}" if tail else f"[warning] {stream_fail_msg}"
            running_msg = (
                "Generating patch dataset ...\n"
                f"Model: {model}\n"
                f"Input dir: {in_dir}\n"
                f"Output dir: {out_dir}\n"
                f"Expected metadata: {expected_meta}\n\n{tail}"
            )
            yield running_msg, str(out_dir), str(expected_meta)
            last_emit_ts = now

        if proc.poll() is not None:
            break
        if not got_output:
            time.sleep(0.15)

    if proc.stdout is not None:
        try:
            rest = proc.stdout.read()
        except Exception:
            rest = b""
        if rest:
            partial += rest.decode("utf-8", errors="replace").replace("\r", "\n") if isinstance(rest, bytes) else str(rest).replace("\r", "\n")
        if partial:
            log_lines.extend(partial.splitlines())

    ok = proc.returncode == 0
    status = "Success" if ok else "Failed"
    final_msg = (
        f"{status}\nModel: {model}\nInput dir: {in_dir}\nOutput dir: {out_dir}\n"
        f"Metadata path: {expected_meta}\n\n"
        + _tail_lines_newest_first(log_lines, 3000)
    )
    yield final_msg, str(out_dir), str(expected_meta)


def run_eval_text_hierarchy_vit_live(
    dataset_dir: str,
    backbone_dir: str,
    projection_head_path: str,
    output_dir: str,
    config_path: str,
    include_line_images: bool,
    include_word_crops: bool,
    include_number_crops: bool,
    min_assets_per_group: int,
    min_positives_per_query: int,
    target_height: int,
    max_width: int,
    patch_multiple: int,
    width_buckets: str,
    batch_size: int,
    num_workers: int,
    device: str,
    l2_normalize_embeddings: bool,
    recall_ks: str,
    max_queries: int,
    seed: int,
    write_per_query_csv: bool,
):
    dataset_path = Path(dataset_dir).expanduser().resolve()
    backbone_path = Path(backbone_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    script_path = ROOT / "scripts" / "eval_text_hierarchy_vit.py"

    expected_report = output_path / "eval_text_hierarchy_vit_report.json"
    expected_csv = output_path / "eval_text_hierarchy_vit_per_query.csv"

    if not script_path.exists():
        msg = f"Failed: script not found: {script_path}"
        yield msg, str(expected_report), str(expected_csv), "{}"
        return
    if not dataset_path.exists() or not dataset_path.is_dir():
        msg = f"Failed: dataset_dir does not exist: {dataset_path}"
        yield msg, str(expected_report), str(expected_csv), "{}"
        return
    if not backbone_path.exists() or not backbone_path.is_dir():
        msg = f"Failed: backbone_dir does not exist: {backbone_path}"
        yield msg, str(expected_report), str(expected_csv), "{}"
        return

    output_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        str(script_path),
        "--dataset_dir",
        str(dataset_path),
        "--backbone_dir",
        str(backbone_path),
        "--output_dir",
        str(output_path),
        "--min_assets_per_group",
        str(int(min_assets_per_group)),
        "--min_positives_per_query",
        str(int(min_positives_per_query)),
        "--target_height",
        str(int(target_height)),
        "--max_width",
        str(int(max_width)),
        "--patch_multiple",
        str(int(patch_multiple)),
        "--batch_size",
        str(int(batch_size)),
        "--num_workers",
        str(int(num_workers)),
        "--device",
        (device or "auto").strip(),
        "--recall_ks",
        (recall_ks or "1,5,10").strip(),
        "--max_queries",
        str(int(max_queries)),
        "--seed",
        str(int(seed)),
    ]
    proj = (projection_head_path or "").strip()
    if proj:
        cmd.extend(["--projection_head_path", str(Path(proj).expanduser().resolve())])
    cfg = (config_path or "").strip()
    if cfg:
        cmd.extend(["--config_path", str(Path(cfg).expanduser().resolve())])
    wb = (width_buckets or "").strip()
    if wb:
        cmd.extend(["--width_buckets", wb])

    cmd.append("--include_line_images" if bool(include_line_images) else "--no_include_line_images")
    cmd.append("--include_word_crops" if bool(include_word_crops) else "--no_include_word_crops")
    if bool(include_number_crops):
        cmd.append("--include_number_crops")
    cmd.append("--l2_normalize_embeddings" if bool(l2_normalize_embeddings) else "--no_l2_normalize_embeddings")
    cmd.append("--write_per_query_csv" if bool(write_per_query_csv) else "--no_write_per_query_csv")

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
    except Exception as exc:
        msg = f"Failed\nOutput dir: {output_path}\n\n{type(exc).__name__}: {exc}"
        yield msg, str(expected_report), str(expected_csv), "{}"
        return

    if proc.stdout is not None:
        try:
            os.set_blocking(proc.stdout.fileno(), False)
        except Exception:
            pass

    log_lines: List[str] = []
    partial = ""
    last_emit_ts = 0.0
    stream_failed = False
    stream_fail_msg = ""

    yield (
        "Evaluating TextHierarchy ViT ...\n"
        f"Dataset dir: {dataset_path}\n"
        f"Backbone dir: {backbone_path}\n"
        f"Output dir: {output_path}\n"
        f"Expected report: {expected_report}\n"
        f"Command: {shlex.join(cmd)}\n",
        str(expected_report),
        str(expected_csv),
        "{}",
    )

    while True:
        got_output = False
        if (not stream_failed) and proc.stdout is not None:
            try:
                chunk = proc.stdout.read()
            except BlockingIOError:
                chunk = b""
            except Exception as exc:
                stream_failed = True
                stream_fail_msg = f"stdout stream disabled ({type(exc).__name__}: {exc})"
                chunk = b""
            if chunk:
                got_output = True
                chunk_text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else str(chunk)
                partial += chunk_text.replace("\r", "\n")
                parts = partial.splitlines(keepends=True)
                keep = ""
                for piece in parts:
                    if piece.endswith("\n"):
                        log_lines.append(piece.rstrip("\n"))
                    else:
                        keep = piece
                partial = keep

        now = time.time()
        if now - last_emit_ts >= 1.0:
            tail = _tail_lines_newest_first(log_lines, 800)
            if stream_failed and stream_fail_msg:
                tail = f"{tail}\n[warning] {stream_fail_msg}" if tail else f"[warning] {stream_fail_msg}"
            running_msg = (
                "Evaluating TextHierarchy ViT ...\n"
                f"Dataset dir: {dataset_path}\n"
                f"Backbone dir: {backbone_path}\n"
                f"Output dir: {output_path}\n"
                f"Expected report: {expected_report}\n\n{tail}"
            )
            yield running_msg, str(expected_report), str(expected_csv), _read_json_file_pretty(expected_report)
            last_emit_ts = now

        if proc.poll() is not None:
            break
        if not got_output:
            time.sleep(0.15)

    if proc.stdout is not None:
        try:
            rest = proc.stdout.read()
        except Exception:
            rest = b""
        if rest:
            partial += rest.decode("utf-8", errors="replace").replace("\r", "\n") if isinstance(rest, bytes) else str(rest).replace("\r", "\n")
        if partial:
            log_lines.extend(partial.splitlines())

    ok = proc.returncode == 0
    status = "Success" if ok else "Failed"
    final_msg = (
        f"{status}\nDataset dir: {dataset_path}\nBackbone dir: {backbone_path}\n"
        f"Output dir: {output_path}\nReport path: {expected_report}\n\n"
        + _tail_lines_newest_first(log_lines, 3000)
    )
    yield final_msg, str(expected_report), str(expected_csv), _read_json_file_pretty(expected_report)


def run_faiss_text_hierarchy_search_live(
    query_image: str,
    backbone_dir: str,
    projection_head_path: str,
    output_dir: str,
    index_path: str,
    meta_path: str,
    dataset_dir: str,
    save_index_path: str,
    metric: str,
    top_k: int,
    include_line_images: bool,
    include_word_crops: bool,
    include_number_crops: bool,
    min_assets_per_group: int,
    config_path: str,
    target_height: int,
    max_width: int,
    patch_multiple: int,
    width_buckets: str,
    batch_size: int,
    num_workers: int,
    device: str,
    l2_normalize_embeddings: bool,
):
    query_path = Path(query_image).expanduser().resolve()
    backbone_path = Path(backbone_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    script_path = ROOT / "scripts" / "faiss_text_hierarchy_search.py"

    idx_input = (index_path or "").strip()
    save_idx_input = (save_index_path or "").strip()
    if idx_input:
        expected_index = Path(idx_input).expanduser().resolve()
    elif save_idx_input:
        expected_index = Path(save_idx_input).expanduser().resolve()
    else:
        expected_index = (output_path / "text_hierarchy.faiss").resolve()
    expected_meta = (
        Path(meta_path).expanduser().resolve()
        if (meta_path or "").strip()
        else Path(str(expected_index) + ".meta.json")
    )
    expected_report = (output_path / "faiss_search_results.json").resolve()

    if not script_path.exists():
        msg = f"Failed: script not found: {script_path}"
        yield msg, str(expected_report), str(expected_index), str(expected_meta), "{}"
        return
    if not query_path.exists() or not query_path.is_file():
        msg = f"Failed: query_image does not exist: {query_path}"
        yield msg, str(expected_report), str(expected_index), str(expected_meta), "{}"
        return
    if not backbone_path.exists() or not backbone_path.is_dir():
        msg = f"Failed: backbone_dir does not exist: {backbone_path}"
        yield msg, str(expected_report), str(expected_index), str(expected_meta), "{}"
        return
    if not idx_input:
        dataset_path = Path(dataset_dir).expanduser().resolve()
        if not dataset_path.exists() or not dataset_path.is_dir():
            msg = f"Failed: dataset_dir does not exist (required when no index_path): {dataset_path}"
            yield msg, str(expected_report), str(expected_index), str(expected_meta), "{}"
            return

    output_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        str(script_path),
        "--query_image",
        str(query_path),
        "--backbone_dir",
        str(backbone_path),
        "--output_dir",
        str(output_path),
        "--metric",
        (metric or "cosine").strip().lower(),
        "--top_k",
        str(int(top_k)),
        "--min_assets_per_group",
        str(int(min_assets_per_group)),
        "--target_height",
        str(int(target_height)),
        "--max_width",
        str(int(max_width)),
        "--patch_multiple",
        str(int(patch_multiple)),
        "--batch_size",
        str(int(batch_size)),
        "--num_workers",
        str(int(num_workers)),
        "--device",
        (device or "auto").strip(),
    ]
    proj = (projection_head_path or "").strip()
    if proj:
        cmd.extend(["--projection_head_path", str(Path(proj).expanduser().resolve())])
    cfg = (config_path or "").strip()
    if cfg:
        cmd.extend(["--config_path", str(Path(cfg).expanduser().resolve())])
    wb = (width_buckets or "").strip()
    if wb:
        cmd.extend(["--width_buckets", wb])
    if idx_input:
        cmd.extend(["--index_path", str(Path(idx_input).expanduser().resolve())])
        if (meta_path or "").strip():
            cmd.extend(["--meta_path", str(Path(meta_path).expanduser().resolve())])
    else:
        cmd.extend(["--dataset_dir", str(Path(dataset_dir).expanduser().resolve())])
        if save_idx_input:
            cmd.extend(["--save_index_path", str(Path(save_idx_input).expanduser().resolve())])

    cmd.append("--include_line_images" if bool(include_line_images) else "--no_include_line_images")
    cmd.append("--include_word_crops" if bool(include_word_crops) else "--no_include_word_crops")
    if bool(include_number_crops):
        cmd.append("--include_number_crops")
    cmd.append("--l2_normalize_embeddings" if bool(l2_normalize_embeddings) else "--no_l2_normalize_embeddings")

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
    except Exception as exc:
        msg = f"Failed\nOutput dir: {output_path}\n\n{type(exc).__name__}: {exc}"
        yield msg, str(expected_report), str(expected_index), str(expected_meta), "{}"
        return

    if proc.stdout is not None:
        try:
            os.set_blocking(proc.stdout.fileno(), False)
        except Exception:
            pass

    log_lines: List[str] = []
    partial = ""
    last_emit_ts = 0.0
    stream_failed = False
    stream_fail_msg = ""

    yield (
        "Running FAISS similarity search ...\n"
        f"Query image: {query_path}\n"
        f"Backbone dir: {backbone_path}\n"
        f"Output dir: {output_path}\n"
        f"Expected report: {expected_report}\n"
        f"Expected index: {expected_index}\n"
        f"Command: {shlex.join(cmd)}\n",
        str(expected_report),
        str(expected_index),
        str(expected_meta),
        "{}",
    )

    while True:
        got_output = False
        if (not stream_failed) and proc.stdout is not None:
            try:
                chunk = proc.stdout.read()
            except BlockingIOError:
                chunk = b""
            except Exception as exc:
                stream_failed = True
                stream_fail_msg = f"stdout stream disabled ({type(exc).__name__}: {exc})"
                chunk = b""
            if chunk:
                got_output = True
                chunk_text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else str(chunk)
                partial += chunk_text.replace("\r", "\n")
                parts = partial.splitlines(keepends=True)
                keep = ""
                for piece in parts:
                    if piece.endswith("\n"):
                        log_lines.append(piece.rstrip("\n"))
                    else:
                        keep = piece
                partial = keep

        now = time.time()
        if now - last_emit_ts >= 1.0:
            tail = _tail_lines_newest_first(log_lines, 800)
            if stream_failed and stream_fail_msg:
                tail = f"{tail}\n[warning] {stream_fail_msg}" if tail else f"[warning] {stream_fail_msg}"
            running_msg = (
                "Running FAISS similarity search ...\n"
                f"Query image: {query_path}\n"
                f"Backbone dir: {backbone_path}\n"
                f"Output dir: {output_path}\n"
                f"Expected report: {expected_report}\n\n{tail}"
            )
            yield running_msg, str(expected_report), str(expected_index), str(expected_meta), _read_json_file_pretty(expected_report)
            last_emit_ts = now

        if proc.poll() is not None:
            break
        if not got_output:
            time.sleep(0.15)

    if proc.stdout is not None:
        try:
            rest = proc.stdout.read()
        except Exception:
            rest = b""
        if rest:
            partial += rest.decode("utf-8", errors="replace").replace("\r", "\n") if isinstance(rest, bytes) else str(rest).replace("\r", "\n")
        if partial:
            log_lines.extend(partial.splitlines())

    ok = proc.returncode == 0
    status = "Success" if ok else "Failed"
    final_msg = (
        f"{status}\nQuery image: {query_path}\nBackbone dir: {backbone_path}\n"
        f"Output dir: {output_path}\nReport path: {expected_report}\n"
        f"Index path: {expected_index}\nMetadata path: {expected_meta}\n\n"
        + _tail_lines_newest_first(log_lines, 3000)
    )
    yield final_msg, str(expected_report), str(expected_index), str(expected_meta), _read_json_file_pretty(expected_report)


def run_faiss_text_hierarchy_search_live_with_mode(
    index_mode: str,
    query_image: str,
    backbone_dir: str,
    projection_head_path: str,
    output_dir: str,
    index_path: str,
    meta_path: str,
    dataset_dir: str,
    save_index_path: str,
    metric: str,
    top_k: int,
    include_line_images: bool,
    include_word_crops: bool,
    include_number_crops: bool,
    min_assets_per_group: int,
    config_path: str,
    target_height: int,
    max_width: int,
    patch_multiple: int,
    width_buckets: str,
    batch_size: int,
    num_workers: int,
    device: str,
    l2_normalize_embeddings: bool,
):
    mode = (index_mode or "").strip().lower()
    use_existing = "existing" in mode
    eff_index = (index_path or "").strip() if use_existing else ""
    eff_meta = (meta_path or "").strip() if use_existing else ""
    eff_dataset = "" if use_existing else (dataset_dir or "").strip()
    yield from run_faiss_text_hierarchy_search_live(
        query_image=query_image,
        backbone_dir=backbone_dir,
        projection_head_path=projection_head_path,
        output_dir=output_dir,
        index_path=eff_index,
        meta_path=eff_meta,
        dataset_dir=eff_dataset,
        save_index_path=save_index_path,
        metric=metric,
        top_k=top_k,
        include_line_images=include_line_images,
        include_word_crops=include_word_crops,
        include_number_crops=include_number_crops,
        min_assets_per_group=min_assets_per_group,
        config_path=config_path,
        target_height=target_height,
        max_width=max_width,
        patch_multiple=patch_multiple,
        width_buckets=width_buckets,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        l2_normalize_embeddings=l2_normalize_embeddings,
    )


def scan_text_hierarchy_retrieval_artifacts(models_dir: str):
    base = Path(models_dir).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        empty = gr.update(choices=[], value=None)
        return empty, empty, empty, "", "", "", "", f"Models directory not found: {base}"

    backbones: List[str] = []
    heads: List[str] = []
    indices: List[str] = []
    for p in sorted(base.rglob("*")):
        if p.is_file():
            low = p.name.lower()
            sfx = p.suffix.lower()
            if sfx == ".faiss":
                indices.append(str(p.resolve()))
                continue
            if sfx == ".pt" and "projection_head" in low:
                heads.append(str(p.resolve()))
                continue
            continue

        if not p.is_dir():
            continue
        if not (p / "config.json").exists():
            continue
        has_weights = (p / "pytorch_model.bin").exists() or (p / "model.safetensors").exists() or (p / "model.safetensors.index.json").exists()
        if has_weights:
            backbones.append(str(p.resolve()))

    backbones = sorted(set(backbones))
    heads = sorted(set(heads))
    indices = sorted(set(indices))

    backbone_default = backbones[0] if backbones else ""
    head_default = _suggest_projection_head_for_backbone(backbone_default) if backbone_default else ""
    if not head_default and heads:
        head_default = heads[0]
    index_default = indices[0] if indices else ""
    meta_default = str(Path(index_default + ".meta.json")) if index_default else ""

    msg = (
        f"Found {len(backbones)} backbone(s), {len(heads)} projection head(s), "
        f"{len(indices)} FAISS index file(s) in {base}"
    )
    return (
        gr.update(choices=backbones, value=(backbone_default if backbone_default else None)),
        gr.update(choices=heads, value=(head_default if head_default else None)),
        gr.update(choices=indices, value=(index_default if index_default else None)),
        backbone_default,
        head_default,
        index_default,
        meta_default,
        msg,
    )


def scan_text_hierarchy_retrieval_artifacts_for_ui(models_dir: str):
    (
        backbone_update,
        head_update,
        index_update,
        backbone_default,
        head_default,
        index_default,
        meta_default,
        msg,
    ) = scan_text_hierarchy_retrieval_artifacts(models_dir)
    return (
        backbone_update,
        head_update,
        index_update,
        backbone_default,
        head_default,
        index_default,
        meta_default,
        msg,
        index_default,
        meta_default,
    )


def on_faiss_index_change_ui(index_path: str):
    idx_raw = (index_path or "").strip()
    if not idx_raw:
        return "", ""
    idx = Path(idx_raw).expanduser().resolve()
    return str(idx), str(Path(str(idx) + ".meta.json"))


def _file_to_path_text(file_path: Optional[str]):
    return (file_path or "").strip()


def run_donut_ocr_workflow_live(
    dataset_name: str,
    dataset_output_dir: str,
    train_samples: int,
    val_samples: int,
    font_path_tibetan: str,
    font_path_chinese: str,
    augmentation: str,
    target_newline_token: str,
    prepared_output_dir: str,
    model_output_dir: str,
    model_name_or_path: str,
    _unused_tokenizer_retrain_flag: bool,
    _unused_tokenizer_retrain_vocab_size: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    num_train_epochs: float,
    learning_rate: float,
    max_target_length: int,
    image_size: int,
    seed: int,
    skip_generation: bool,
    skip_prepare: bool,
    skip_train: bool,
    lora_augment_path: str,
    lora_augment_model_family: str,
    lora_augment_base_model_id: str,
    lora_augment_controlnet_model_id: str,
    lora_augment_prompt: str,
    lora_augment_scale: float,
    lora_augment_strength: float,
    lora_augment_steps: int,
    lora_augment_guidance_scale: float,
    lora_augment_controlnet_scale: float,
    lora_augment_seed: Optional[int],
    lora_augment_splits: str,
    lora_augment_targets: str,
    lora_augment_canny_low: int,
    lora_augment_canny_high: int,
):
    script_path = ROOT / "scripts" / "run_donut_ocr_workflow.py"
    dataset_name_clean = (dataset_name or "tibetan-donut-ocr-label1").strip()
    dataset_output_path = Path(dataset_output_dir).expanduser().resolve()
    dataset_dir = dataset_output_path / dataset_name_clean
    prepared_dir = (
        Path(prepared_output_dir).expanduser().resolve()
        if (prepared_output_dir or "").strip()
        else (dataset_dir / "donut_ocr_label1")
    )
    model_output_path = Path(model_output_dir).expanduser().resolve()
    summary_path = model_output_path / "workflow_summary.json"

    if not script_path.exists():
        msg = f"Failed: script not found: {script_path}"
        yield msg, str(dataset_dir), str(prepared_dir), str(model_output_path), str(summary_path)
        return

    if not skip_generation:
        tib_font = Path(font_path_tibetan).expanduser().resolve()
        chi_font = Path(font_path_chinese).expanduser().resolve()
        if not tib_font.exists():
            msg = f"Failed: Tibetan font not found: {tib_font}"
            yield msg, str(dataset_dir), str(prepared_dir), str(model_output_path), str(summary_path)
            return
        if not chi_font.exists():
            msg = f"Failed: Chinese font not found: {chi_font}"
            yield msg, str(dataset_dir), str(prepared_dir), str(model_output_path), str(summary_path)
            return

    model_output_path.mkdir(parents=True, exist_ok=True)

    seed_arg = 42
    try:
        seed_arg = int(float(seed))
    except Exception:
        seed_arg = 42

    cmd = [
        sys.executable,
        "-u",
        str(script_path),
        "--dataset_name",
        dataset_name_clean,
        "--dataset_output_dir",
        str(dataset_output_path),
        "--train_samples",
        str(int(train_samples)),
        "--val_samples",
        str(int(val_samples)),
        "--font_path_tibetan",
        str(Path(font_path_tibetan).expanduser()),
        "--font_path_chinese",
        str(Path(font_path_chinese).expanduser()),
        "--augmentation",
        (augmentation or "noise").strip(),
        "--target_newline_token",
        (target_newline_token or "<NL>").strip(),
        "--model_output_dir",
        str(model_output_path),
        "--model_name_or_path",
        (model_name_or_path or "microsoft/trocr-base-stage1").strip(),
        "--per_device_train_batch_size",
        str(int(per_device_train_batch_size)),
        "--per_device_eval_batch_size",
        str(int(per_device_eval_batch_size)),
        "--num_train_epochs",
        str(float(num_train_epochs)),
        "--learning_rate",
        str(float(learning_rate)),
        "--max_target_length",
        str(int(max_target_length)),
        "--image_size",
        str(int(image_size)),
        "--seed",
        str(seed_arg),
    ]

    prepared_output_clean = (prepared_output_dir or "").strip()
    if prepared_output_clean:
        cmd.extend(["--prepared_output_dir", str(Path(prepared_output_clean).expanduser())])
    if skip_generation:
        cmd.append("--skip_generation")
    if skip_prepare:
        cmd.append("--skip_prepare")
    if skip_train:
        cmd.append("--skip_train")

    lora_path = (lora_augment_path or "").strip()
    if lora_path:
        cmd.extend(
            [
                "--lora_augment_path",
                lora_path,
                "--lora_augment_model_family",
                (lora_augment_model_family or "sdxl").strip(),
                "--lora_augment_base_model_id",
                (lora_augment_base_model_id or "stabilityai/stable-diffusion-xl-base-1.0").strip(),
                "--lora_augment_controlnet_model_id",
                (lora_augment_controlnet_model_id or "diffusers/controlnet-canny-sdxl-1.0").strip(),
                "--lora_augment_prompt",
                (lora_augment_prompt or DEFAULT_TEXTURE_PROMPT).strip(),
                "--lora_augment_scale",
                str(float(lora_augment_scale)),
                "--lora_augment_strength",
                str(float(lora_augment_strength)),
                "--lora_augment_steps",
                str(int(lora_augment_steps)),
                "--lora_augment_guidance_scale",
                str(float(lora_augment_guidance_scale)),
                "--lora_augment_controlnet_scale",
                str(float(lora_augment_controlnet_scale)),
                "--lora_augment_splits",
                (lora_augment_splits or "train").strip(),
                "--lora_augment_targets",
                (lora_augment_targets or "images_and_ocr_crops").strip(),
                "--lora_augment_canny_low",
                str(int(lora_augment_canny_low)),
                "--lora_augment_canny_high",
                str(int(lora_augment_canny_high)),
            ]
        )
        try:
            if lora_augment_seed is not None:
                lora_seed_int = int(float(lora_augment_seed))
                if lora_seed_int >= 0:
                    cmd.extend(["--lora_augment_seed", str(lora_seed_int)])
        except Exception:
            pass

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
    except Exception as exc:
        msg = f"Failed\nModel output dir: {model_output_path}\n\n{type(exc).__name__}: {exc}"
        yield msg, str(dataset_dir), str(prepared_dir), str(model_output_path), str(summary_path)
        return

    if proc.stdout is not None:
        try:
            os.set_blocking(proc.stdout.fileno(), False)
        except Exception:
            pass

    log_lines: List[str] = []
    partial = ""
    last_emit_ts = 0.0
    stream_failed = False
    stream_fail_msg = ""

    yield (
        "Running Donut OCR workflow (label 1) ...\n"
        f"Dataset dir: {dataset_dir}\n"
        f"Prepared dir: {prepared_dir}\n"
        f"Model output dir: {model_output_path}\n"
        f"Summary path: {summary_path}\n"
        f"Command: {shlex.join(cmd)}\n",
        str(dataset_dir),
        str(prepared_dir),
        str(model_output_path),
        str(summary_path),
    )

    while True:
        got_output = False
        if (not stream_failed) and proc.stdout is not None:
            try:
                chunk = proc.stdout.read()
            except BlockingIOError:
                chunk = b""
            except Exception as exc:
                stream_failed = True
                stream_fail_msg = f"stdout stream disabled ({type(exc).__name__}: {exc})"
                chunk = b""

            if chunk:
                got_output = True
                if isinstance(chunk, bytes):
                    chunk_text = chunk.decode("utf-8", errors="replace")
                else:
                    chunk_text = str(chunk)
                partial += chunk_text.replace("\r", "\n")
                parts = partial.splitlines(keepends=True)
                keep = ""
                for piece in parts:
                    if piece.endswith("\n"):
                        log_lines.append(piece.rstrip("\n"))
                    else:
                        keep = piece
                partial = keep

        now = time.time()
        if now - last_emit_ts >= 1.0:
            tail = _tail_lines_newest_first(log_lines, 800)
            if stream_failed and stream_fail_msg:
                tail = f"{tail}\n[warning] {stream_fail_msg}" if tail else f"[warning] {stream_fail_msg}"
            running_msg = (
                "Running Donut OCR workflow (label 1) ...\n"
                f"Dataset dir: {dataset_dir}\n"
                f"Prepared dir: {prepared_dir}\n"
                f"Model output dir: {model_output_path}\n"
                f"Summary path: {summary_path}\n\n{tail}"
            )
            yield (
                running_msg,
                str(dataset_dir),
                str(prepared_dir),
                str(model_output_path),
                str(summary_path),
            )
            last_emit_ts = now

        if proc.poll() is not None:
            break

        if not got_output:
            time.sleep(0.15)

    if proc.stdout is not None:
        try:
            rest = proc.stdout.read()
        except Exception:
            rest = b""
        if rest:
            if isinstance(rest, bytes):
                partial += rest.decode("utf-8", errors="replace").replace("\r", "\n")
            else:
                partial += str(rest).replace("\r", "\n")
        if partial:
            log_lines.extend(partial.splitlines())

    ok = proc.returncode == 0
    status = "Success" if ok else "Failed"
    final_msg = (
        f"{status}\nDataset dir: {dataset_dir}\nPrepared dir: {prepared_dir}\n"
        f"Model output dir: {model_output_path}\nSummary path: {summary_path}\n\n"
        + _tail_lines_newest_first(log_lines, 3000)
    )
    yield final_msg, str(dataset_dir), str(prepared_dir), str(model_output_path), str(summary_path)


def _resolve_donut_runtime_dirs(model_root_or_model_dir: str) -> Tuple[Optional[Path], Optional[Path], Optional[Path], str]:
    raw = (model_root_or_model_dir or "").strip()
    if not raw:
        return None, None, None, "Empty model path."
    root = Path(raw).expanduser().resolve()
    if not root.exists():
        return None, None, None, f"Path not found: {root}"

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

    # Accept either output root (`.../donut_openpecha_bdrc`) or direct model dir (`.../model`).
    if _is_hf_model_dir(root):
        model_dir = root
        base_dir = root.parent
    else:
        model_dir = root / "model"
        base_dir = root

    tokenizer_dir = base_dir / "tokenizer"
    image_processor_dir = base_dir / "image_processor"

    if not _is_hf_model_dir(model_dir):
        return None, None, None, f"Could not find DONUT/TroCR model dir at {model_dir}"
    if not tokenizer_dir.exists():
        tokenizer_dir = model_dir
    if not image_processor_dir.exists():
        image_processor_dir = model_dir
    return model_dir, tokenizer_dir, image_processor_dir, ""


@lru_cache(maxsize=8)
def _load_donut_ocr_runtime_cached(model_dir_s: str, tokenizer_dir_s: str, image_processor_dir_s: str, device_pref: str):
    if torch is None:
        raise RuntimeError("PyTorch not available in UI environment.")
    if AutoImageProcessor is None or AutoTokenizer is None:
        raise RuntimeError("transformers not available in UI environment.")

    device, device_msg = _resolve_torch_device_for_encoding(device_pref)
    model_dir = Path(model_dir_s).expanduser().resolve()
    tokenizer_dir = Path(tokenizer_dir_s).expanduser().resolve()
    image_processor_dir = Path(image_processor_dir_s).expanduser().resolve()

    from scripts.train_donut_ocr import (
        _load_ved_model_robust,
        _paired_end_token_for_start,
        _strip_special_token_strings,
    )  # reuse training-side HF/meta/token fixes

    model = _load_ved_model_robust(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)
    image_processor = AutoImageProcessor.from_pretrained(str(image_processor_dir))
    # Align generation/model token IDs with the loaded tokenizer (same idea as training script).
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
    except Exception:
        pass
    model.eval()
    model.to(device)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "image_processor": image_processor,
        "device": device,
        "device_msg": device_msg,
        "model_dir": str(model_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "image_processor_dir": str(image_processor_dir),
    }


def _apply_donut_ui_preprocess(image: Image.Image, pipeline: str) -> Image.Image:
    rgb = image.convert("RGB")
    # Prefer the exact training-side preprocessing function to avoid UI/training mismatches.
    try:
        from scripts.train_donut_ocr import _apply_image_preprocess_pipeline as _apply_train_donut_preproc

        return _apply_train_donut_preproc(rgb, str(pipeline or "none")).convert("RGB")
    except Exception:
        pass

    mode = (pipeline or "none").strip().lower()
    if mode == "pb" and _pb_preprocess_patch_image is not None and _PBPreprocessConfig is not None:
        try:
            return _pb_preprocess_patch_image(image=rgb, config=_PBPreprocessConfig()).convert("RGB")
        except Exception:
            return rgb
    if mode == "bdrc" and _preprocess_image_bdrc is not None and _BDRCPreprocessConfig is not None:
        try:
            return _preprocess_image_bdrc(image=rgb, config=_BDRCPreprocessConfig.vit_defaults()).convert("RGB")
        except Exception:
            return rgb
    return rgb


def _apply_line_clip_ui_preprocess_from_runtime_cfg(image: Image.Image, runtime_cfg: Dict[str, Any]) -> Image.Image:
    rgb = image.convert("RGB")
    cfg_obj = runtime_cfg if isinstance(runtime_cfg, dict) else {}
    mode = str(cfg_obj.get("image_preprocess_pipeline", "none") or "none").strip().lower()
    if mode not in {"pb", "bdrc"}:
        return rgb

    if mode == "pb" and _pb_preprocess_patch_image is not None and _PBPreprocessConfig is not None:
        try:
            pb_payload = cfg_obj.get("image_preprocess_pb")
            if isinstance(pb_payload, dict) and hasattr(_PBPreprocessConfig, "from_dict"):
                pb_cfg = _PBPreprocessConfig.from_dict(pb_payload)
            else:
                pb_cfg = _PBPreprocessConfig()
            return _pb_preprocess_patch_image(image=rgb, config=pb_cfg).convert("RGB")
        except Exception:
            return rgb

    if mode == "bdrc" and _preprocess_image_bdrc is not None and _BDRCPreprocessConfig is not None:
        try:
            bdrc_payload = cfg_obj.get("image_preprocess_bdrc")
            if isinstance(bdrc_payload, dict) and hasattr(_BDRCPreprocessConfig, "from_dict"):
                bdrc_cfg = _BDRCPreprocessConfig.from_dict(bdrc_payload)
            else:
                bdrc_cfg = _BDRCPreprocessConfig.vit_defaults()
            return _preprocess_image_bdrc(image=rgb, config=bdrc_cfg).convert("RGB")
        except Exception:
            return rgb

    return rgb


def run_donut_ocr_inference_ui(
    image: Optional[np.ndarray],
    model_root_or_model_dir: str,
    image_preprocess_pipeline: str,
    device_preference: str,
    generation_max_length: int,
    num_beams: int,
):
    if image is None:
        return None, "", "{}"
    model_dir, tokenizer_dir, image_processor_dir, err = _resolve_donut_runtime_dirs(model_root_or_model_dir)
    if err:
        return image, f"Failed: {err}", json.dumps({"ok": False, "error": err}, ensure_ascii=False, indent=2)

    try:
        runtime = _load_donut_ocr_runtime_cached(
            str(model_dir),
            str(tokenizer_dir),
            str(image_processor_dir),
            str(device_preference or "auto"),
        )
    except Exception as exc:
        msg = f"Failed loading DONUT runtime: {type(exc).__name__}: {exc}"
        return image, msg, json.dumps({"ok": False, "error": msg}, ensure_ascii=False, indent=2)

    try:
        pil = Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
        proc_pil = _apply_donut_ui_preprocess(pil, image_preprocess_pipeline)
        image_processor = runtime["image_processor"]
        tokenizer = runtime["tokenizer"]
        model = runtime["model"]
        device = runtime["device"]

        pixel_values = image_processor(images=proc_pil, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        with torch.no_grad():
            generated = model.generate(
                pixel_values=pixel_values,
                max_length=max(8, int(generation_max_length)),
                num_beams=max(1, int(num_beams)),
            )
        text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0] if len(generated) else ""
        text = _strip_special_token_strings(str(text or ""), tokenizer)
        gen_ids: List[int] = []
        try:
            if len(generated):
                gen_ids = [int(x) for x in generated[0].detach().cpu().tolist()]
        except Exception:
            gen_ids = []

        debug = {
            "ok": True,
            "model_dir": runtime["model_dir"],
            "tokenizer_dir": runtime["tokenizer_dir"],
            "image_processor_dir": runtime["image_processor_dir"],
            "device": runtime["device"],
            "device_msg": runtime.get("device_msg", ""),
            "image_preprocess_pipeline": (image_preprocess_pipeline or "none"),
            "generation_max_length": int(generation_max_length),
            "num_beams": int(num_beams),
            "decoder_start_token_id": getattr(model.config, "decoder_start_token_id", None),
            "pad_token_id": getattr(model.config, "pad_token_id", None),
            "eos_token_id": getattr(model.config, "eos_token_id", None),
            "tokenizer_bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "tokenizer_pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "tokenizer_eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "generated_token_ids_head": gen_ids[:32],
            "output_length_chars": len(text),
        }
        return np.array(proc_pil), text, json.dumps(debug, ensure_ascii=False, indent=2)
    except Exception as exc:
        msg = f"Failed DONUT OCR inference: {type(exc).__name__}: {exc}"
        return image, msg, json.dumps({"ok": False, "error": msg}, ensure_ascii=False, indent=2)


def run_donut_ocr_inference_with_gt_ui(
    image: Optional[np.ndarray],
    model_root_or_model_dir: str,
    image_preprocess_pipeline: str,
    device_preference: str,
    generation_max_length: int,
    num_beams: int,
    ground_truth_text: str,
):
    preview, pred_text, debug_json = run_donut_ocr_inference_ui(
        image=image,
        model_root_or_model_dir=model_root_or_model_dir,
        image_preprocess_pipeline=image_preprocess_pipeline,
        device_preference=device_preference,
        generation_max_length=generation_max_length,
        num_beams=num_beams,
    )
    gt = str(ground_truth_text or "")
    cer_text = ""
    try:
        debug_obj = json.loads(debug_json or "{}")
    except Exception:
        debug_obj = {"raw_debug": debug_json}
    if isinstance(debug_obj, dict) and bool(debug_obj.get("ok")) and gt.strip():
        cer = _cer_single_ui(pred_text or "", gt)
        cer_text = f"{cer:.6f}"
        debug_obj["ground_truth_length_chars"] = len(gt)
        debug_obj["cer"] = cer
        debug_json = json.dumps(debug_obj, ensure_ascii=False, indent=2)
    return preview, pred_text, gt, cer_text, debug_json


def _image_np_to_data_uri_png_ui(image: Optional[np.ndarray], max_width_px: int = 320) -> str:
    if image is None:
        return ""
    try:
        arr = np.asarray(image)
    except Exception:
        return ""
    if arr.size == 0:
        return ""
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        return ""
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] >= 4:
        arr = arr[:, :, :3]
    arr = arr.astype(np.uint8, copy=False)
    pil = Image.fromarray(arr).convert("RGB")
    if int(max_width_px) > 0 and pil.width > int(max_width_px):
        scale = float(max_width_px) / float(max(1, pil.width))
        new_h = max(1, int(round(pil.height * scale)))
        resampling_ns = getattr(Image, "Resampling", Image)
        pil = pil.resize((int(max_width_px), int(new_h)), resample=resampling_ns.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _render_line_ocr_cards_html_ui(rows: List[Dict[str, Any]], note: str = "") -> str:
    safe_note = html.escape(str(note or ""))
    header = (
        f"<div style='margin:0 0 10px 0;color:#475569;font-size:12px;'>{safe_note}</div>"
        if safe_note
        else ""
    )
    if not rows:
        return (
            "<div style='font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;'>"
            f"{header}"
            "<div style='border:1px solid #e2e8f0;border-radius:12px;padding:14px;background:#f8fafc;color:#64748b;'>"
            "Noch keine Zeilen-Ergebnisse. Bild hochladen und Workflow starten."
            "</div>"
            "</div>"
        )

    cards: List[str] = []
    for rec in rows:
        line_no = int(rec.get("line_no", 0))
        line_id = int(rec.get("line_id", 0))
        line_box = rec.get("line_box") or [0, 0, 0, 0]
        line_box_txt = html.escape(str([int(v) for v in line_box[:4]]))
        text_val = str(rec.get("text", "") or "")
        text_html = (
            html.escape(text_val)
            if text_val.strip()
            else "<span style='color:#94a3b8;font-style:italic;'>(leer)</span>"
        )
        img_uri = str(rec.get("image_uri", "") or "")
        status_badge = ""
        if not bool(rec.get("ok", True)):
            err = html.escape(str(rec.get("error", "OCR fehlgeschlagen")))
            status_badge = (
                "<div style='margin-top:6px;color:#b91c1c;font-size:12px;'>"
                f"{err}</div>"
            )
        cards.append(
            "<div style='display:grid;grid-template-columns:minmax(110px,180px) 1fr;gap:12px;"
            "align-items:start;border:1px solid #e2e8f0;border-radius:14px;padding:10px;"
            "background:#ffffff;box-shadow:0 1px 2px rgba(15,23,42,0.05);margin-bottom:10px;'>"
            "<div>"
            f"<img src='{img_uri}' alt='line {line_no}' "
            "style='width:100%;height:auto;display:block;border-radius:10px;border:1px solid #e2e8f0;background:#f8fafc;' />"
            "</div>"
            "<div>"
            "<div style='display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:6px;'>"
            f"<span style='background:#0f172a;color:#fff;border-radius:999px;padding:2px 8px;font-size:11px;'>Zeile {line_no}</span>"
            f"<span style='background:#e2e8f0;color:#334155;border-radius:999px;padding:2px 8px;font-size:11px;'>line_id {line_id}</span>"
            f"<span style='color:#64748b;font-size:11px;'>bbox {line_box_txt}</span>"
            "</div>"
            "<div style='white-space:pre-wrap;line-height:1.35;font-size:14px;color:#0f172a;"
            "background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:8px;'>"
            f"{text_html}</div>"
            f"{status_badge}"
            "</div>"
            "</div>"
        )

    return (
        "<div style='font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;'>"
        f"{header}"
        + "".join(cards)
        + "</div>"
    )


def run_page_line_donut_ocr_preview_ui(
    input_mode: str,
    image: Optional[np.ndarray],
    direct_line_image: Optional[np.ndarray],
    line_model_path: str,
    line_conf: float,
    line_imgsz: int,
    line_device: str,
    line_min_height: int,
    line_projection_smooth: int,
    line_projection_threshold_rel: float,
    line_merge_gap_px: int,
    donut_model_dir: str,
    donut_preproc: str,
    donut_device: str,
    donut_max_len: int,
    donut_num_beams: int,
    max_lines: int,
):
    empty_html = _render_line_ocr_cards_html_ui(
        [],
        note="Minimaler Quick-OCR-Workflow: Seite -> Lines -> DONUT oder direktes Line-Image -> DONUT.",
    )
    mode = str(input_mode or "Page -> Lines -> DONUT").strip()
    requested_preproc = str(donut_preproc or "bdrc").strip().lower()
    # Quick tab is opinionated: use BDRC preprocessing, while the lower-level helper still prefers
    # the train-script preprocessing implementation when importable.
    effective_preproc = "bdrc"

    if mode == "Single Line -> DONUT":
        line_img = direct_line_image if direct_line_image is not None else image
        if line_img is None:
            debug = {
                "ok": False,
                "mode": mode,
                "error": "Please upload a line image (or provide an image in the fallback page slot).",
            }
            return None, "Please upload a line image.", "", empty_html, json.dumps(debug, ensure_ascii=False, indent=2)

        preview, pred_text, debug_json = run_donut_ocr_inference_ui(
            image=line_img,
            model_root_or_model_dir=donut_model_dir,
            image_preprocess_pipeline=effective_preproc,
            device_preference=donut_device,
            generation_max_length=donut_max_len,
            num_beams=donut_num_beams,
        )
        try:
            debug_obj = json.loads(debug_json or "{}")
        except Exception:
            debug_obj = {"raw_debug": debug_json}
        ok = bool(isinstance(debug_obj, dict) and debug_obj.get("ok"))
        line_arr = np.asarray(line_img).astype(np.uint8, copy=False)
        h = int(line_arr.shape[0]) if line_arr.ndim >= 2 else 0
        w = int(line_arr.shape[1]) if line_arr.ndim >= 2 else 0
        text_out = str(pred_text or "")
        err_text = "" if ok else str((debug_obj.get("error") if isinstance(debug_obj, dict) else "") or "DONUT OCR failed.")
        html_out = _render_line_ocr_cards_html_ui(
            [
                {
                    "line_no": 1,
                    "line_id": 1,
                    "line_box": [0, 0, w, h],
                    "text": text_out if ok else "",
                    "ok": ok,
                    "error": err_text,
                    "image_uri": _image_np_to_data_uri_png_ui(line_arr, max_width_px=420),
                }
            ],
            note="Direkter Zeilenmodus. Nur BDRC-Vorverarbeitung + DONUT OCR (UI nutzt Train-Script-Preprocessing-Helper, falls verfügbar).",
        )
        debug = {
            "ok": ok,
            "mode": mode,
            "requested_donut_preprocess": requested_preproc,
            "effective_donut_preprocess": effective_preproc,
            "line_image_size": {"width": w, "height": h},
            "ocr": debug_obj,
        }
        status = "Single line mode: BDRC preprocess + DONUT OCR."
        if not ok:
            status += f" Error: {err_text}"
        overlay_out = preview if preview is not None else line_arr
        return overlay_out, status, text_out, html_out, json.dumps(debug, ensure_ascii=False, indent=2)

    if image is None:
        debug = {"ok": False, "mode": mode, "error": "Please upload a page image."}
        return None, "Please upload a page image.", "", empty_html, json.dumps(debug, ensure_ascii=False, indent=2)

    split_out = run_tibetan_text_line_split_classical(
        image=image,
        model_path=line_model_path,
        conf=line_conf,
        imgsz=line_imgsz,
        device=line_device,
        min_line_height=line_min_height,
        projection_smooth=line_projection_smooth,
        projection_threshold_rel=line_projection_threshold_rel,
        merge_gap_px=line_merge_gap_px,
        draw_parent_boxes=True,
        detect_red_text=False,
        red_min_redness=26,
        red_min_saturation=35,
        red_column_fill_rel=0.07,
        red_merge_gap_px=14,
        red_min_width_px=18,
        draw_red_boxes=False,
    )
    (
        overlay,
        split_status,
        split_json,
        _line_profile,
        _selected_line_view,
        _hierarchy_view,
        _selected_line_profile,
        _click_status,
        click_state,
    ) = split_out

    line_records_raw = []
    source_image = None
    if isinstance(click_state, dict):
        line_records_raw = click_state.get("line_boxes") or []
        source_image = click_state.get("image")
    if source_image is None:
        source_image = image

    try:
        src = np.asarray(source_image).astype(np.uint8, copy=False)
    except Exception:
        src = np.asarray(image).astype(np.uint8, copy=False)

    if src.size == 0 or src.ndim < 2:
        debug = {
            "ok": False,
            "stage": "line_split",
            "split_status": split_status,
            "error": "Source image unavailable after line split.",
        }
        return overlay, split_status, "", empty_html, json.dumps(debug, ensure_ascii=False, indent=2)

    h, w = src.shape[:2]
    parsed_lines: List[Dict[str, Any]] = []
    for rec in line_records_raw:
        if not isinstance(rec, dict):
            continue
        box = rec.get("line_box") or []
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = [int(v) for v in box]
        except Exception:
            continue
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        parsed_lines.append(
            {
                "line_id": int(rec.get("line_id", len(parsed_lines) + 1)),
                "line_box": [x1, y1, x2, y2],
            }
        )

    parsed_lines = sorted(parsed_lines, key=lambda r: (int(r["line_box"][1]), int(r["line_box"][0]), int(r["line_id"])))
    total_detected_lines = len(parsed_lines)
    line_limit = max(0, int(max_lines))
    truncated = False
    if line_limit > 0 and len(parsed_lines) > line_limit:
        parsed_lines = parsed_lines[:line_limit]
        truncated = True

    if not parsed_lines:
        note = "Keine Zeilen erkannt. Prüfe YOLO-Modellpfad/Klassen oder das Bild."
        html_out = _render_line_ocr_cards_html_ui([], note=note)
        debug = {
            "ok": False,
            "stage": "line_split",
            "split_status": split_status,
            "total_detected_lines": int(total_detected_lines),
            "split_json": json.loads(split_json) if (split_json or "").strip().startswith("{") else split_json,
        }
        return overlay, split_status, "", html_out, json.dumps(debug, ensure_ascii=False, indent=2)

    html_rows: List[Dict[str, Any]] = []
    copy_lines: List[str] = []
    ocr_debug_rows: List[Dict[str, Any]] = []
    ocr_failures = 0

    for line_no, rec in enumerate(parsed_lines, start=1):
        x1, y1, x2, y2 = [int(v) for v in rec["line_box"]]
        crop = src[y1:y2, x1:x2]
        if crop.size == 0:
            ocr_failures += 1
            html_rows.append(
                {
                    "line_no": int(line_no),
                    "line_id": int(rec["line_id"]),
                    "line_box": [x1, y1, x2, y2],
                    "text": "",
                    "ok": False,
                    "error": "Leerer Crop",
                    "image_uri": _image_np_to_data_uri_png_ui(crop),
                }
            )
            copy_lines.append(f"[{line_no:02d}] ")
            ocr_debug_rows.append(
                {
                    "line_no": int(line_no),
                    "line_id": int(rec["line_id"]),
                    "line_box": [x1, y1, x2, y2],
                    "ok": False,
                    "error": "empty_crop",
                }
            )
            continue

        _, pred_text, debug_json = run_donut_ocr_inference_ui(
            image=crop,
            model_root_or_model_dir=donut_model_dir,
            image_preprocess_pipeline=effective_preproc,
            device_preference=donut_device,
            generation_max_length=donut_max_len,
            num_beams=donut_num_beams,
        )
        try:
            debug_obj = json.loads(debug_json or "{}")
        except Exception:
            debug_obj = {"raw_debug": debug_json}
        ok = bool(isinstance(debug_obj, dict) and debug_obj.get("ok"))
        if not ok:
            ocr_failures += 1
        text_out = str(pred_text or "")
        err_text = ""
        if not ok:
            err_text = str((debug_obj.get("error") if isinstance(debug_obj, dict) else "") or "DONUT OCR failed.")
        html_rows.append(
            {
                "line_no": int(line_no),
                "line_id": int(rec["line_id"]),
                "line_box": [x1, y1, x2, y2],
                "text": text_out if ok else "",
                "ok": ok,
                "error": err_text,
                "image_uri": _image_np_to_data_uri_png_ui(crop, max_width_px=360),
            }
        )
        copy_lines.append(f"[{line_no:02d}] {text_out if ok else ''}")
        ocr_debug_rows.append(
            {
                "line_no": int(line_no),
                "line_id": int(rec["line_id"]),
                "line_box": [x1, y1, x2, y2],
                "ok": ok,
                "text": (text_out if ok else ""),
                "error": err_text,
                "debug": debug_obj,
            }
        )

    note = f"Erkannte Zeilen: {total_detected_lines}. DONUT ausgeführt auf {len(parsed_lines)} Zeile(n)."
    if truncated:
        note += f" Anzeige/Verarbeitung auf max_lines={line_limit} begrenzt."
    if ocr_failures:
        note += f" OCR-Fehler: {ocr_failures}."
    html_out = _render_line_ocr_cards_html_ui(html_rows, note=note)

    split_json_obj: Any
    try:
        split_json_obj = json.loads(split_json or "{}")
    except Exception:
        split_json_obj = split_json
    debug = {
        "ok": ocr_failures == 0,
        "mode": mode,
        "requested_donut_preprocess": requested_preproc,
        "effective_donut_preprocess": effective_preproc,
        "split_status": split_status,
        "ocr_status": note,
        "total_detected_lines": int(total_detected_lines),
        "processed_lines": int(len(parsed_lines)),
        "ocr_failures": int(ocr_failures),
        "truncated": bool(truncated),
        "split_json": split_json_obj,
        "ocr_lines": ocr_debug_rows,
    }
    final_status = split_status + " " + note + " DONUT preprocess=bdrc (train-script helper preferred)."
    copy_text = "\n".join(copy_lines)
    return overlay, final_status.strip(), copy_text, html_out, json.dumps(debug, ensure_ascii=False, indent=2)


def scan_donut_ocr_models_ui(models_dir: str):
    base = Path(models_dir).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        return gr.update(choices=[], value=None), "", f"Directory not found: {base}"

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

    found_entries: List[str] = []

    for p in sorted(base.rglob("*")):
        if not p.is_dir():
            continue

        # Final train output root layout: <root>/model + tokenizer + image_processor
        if p.name == "model" and _is_hf_model_dir(p):
            parent = p.parent
            tok = parent / "tokenizer"
            imgp = parent / "image_processor"
            if tok.exists() and imgp.exists():
                found_entries.append(str(parent.resolve()))
            continue

        # In-training checkpoints: <root>/checkpoint-XXXX + tokenizer + image_processor in parent
        if p.name.startswith("checkpoint-") and _is_hf_model_dir(p):
            parent = p.parent
            tok = parent / "tokenizer"
            imgp = parent / "image_processor"
            if tok.exists() and imgp.exists():
                found_entries.append(str(p.resolve()))

    found_entries = sorted(set(found_entries))
    default = found_entries[0] if found_entries else ""
    msg = (
        f"Found {len(found_entries)} DONUT/TroCR model entries (final outputs and/or checkpoints) in {base}"
    )
    return gr.update(choices=found_entries, value=(default if default else None)), default, msg


def _read_jsonl_rows_ui(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists() or not path.is_file():
        return rows
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except Exception:
        return []
    return rows


def _resolve_ocr_manifest_image_path_ui(image_value: str, manifest_path: Path) -> Optional[Path]:
    raw = (image_value or "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p if p.exists() and p.is_file() else None
    candidates = [
        (manifest_path.parent / p).resolve(),
        (manifest_path.parent.parent / p).resolve(),
        (manifest_path.parent.parent.parent / p).resolve(),
        (ROOT / p).resolve(),
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def _extract_ocr_manifest_row_fields_ui(row: Dict[str, Any]) -> Tuple[str, str]:
    image_keys = ["line_path", "src__image", "image", "image_path", "line_image", "line_image_path", "path"]
    text_keys = ["text", "src__label", "label", "transcription", "ocr_text", "line_text"]
    img = ""
    txt = ""
    for k in image_keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            img = v.strip()
            break
    for k in text_keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            txt = v.strip()
            break
    return img, txt


def scan_donut_ocr_datasets_ui(datasets_dir: str):
    base = Path(datasets_dir).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        empty = gr.update(choices=[], value=None)
        return empty, gr.update(choices=[], value=None), "", f"Directory not found: {base}"

    dataset_roots: List[str] = []
    split_map: Dict[str, List[str]] = {}
    for p in sorted(base.rglob("meta/lines.jsonl")):
        split_dir = p.parent.parent  # <root>/<split>/meta/lines.jsonl
        root_dir = split_dir.parent
        split_name = split_dir.name
        if split_name not in {"train", "val", "eval", "test"}:
            continue
        key = str(root_dir.resolve())
        split_map.setdefault(key, [])
        split_map[key].append(split_name)
    for k in sorted(split_map.keys()):
        split_map[k] = sorted(set(split_map[k]), key=lambda s: {"train": 0, "val": 1, "eval": 2, "test": 3}.get(s, 9))
        dataset_roots.append(k)

    default_dataset = dataset_roots[0] if dataset_roots else ""
    default_splits = split_map.get(default_dataset, [])
    default_split = default_splits[0] if default_splits else None
    state_json = json.dumps(split_map, ensure_ascii=False)
    msg = f"Found {len(dataset_roots)} OCR dataset root(s) with manifests in {base}"
    return (
        gr.update(choices=dataset_roots, value=(default_dataset if default_dataset else None)),
        gr.update(choices=default_splits, value=default_split),
        state_json,
        msg,
    )


def on_donut_ocr_dataset_change_ui(dataset_root: str, split_map_json: str):
    root = (dataset_root or "").strip()
    try:
        split_map = json.loads(split_map_json or "{}")
    except Exception:
        split_map = {}
    splits = split_map.get(root, []) if isinstance(split_map, dict) else []
    default_split = splits[0] if splits else None
    return gr.update(choices=splits, value=default_split)


def _levenshtein_ui(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(
                curr[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            ))
        prev = curr
    return prev[-1]


def _cer_single_ui(pred: str, ref: str) -> float:
    denom = max(1, len(ref or ""))
    return float(_levenshtein_ui(pred or "", ref or "") / denom)


def load_random_donut_ocr_dataset_sample_ui(dataset_root: str, split_name: str, seed: float):
    root = Path((dataset_root or "").strip()).expanduser().resolve()
    split = (split_name or "").strip()
    if not root.exists():
        return None, "", "{}", ""
    manifest = root / split / "meta" / "lines.jsonl"
    rows = _read_jsonl_rows_ui(manifest)
    if not rows:
        msg = f"No rows found in {manifest}"
        return None, "", json.dumps({"ok": False, "error": msg}, ensure_ascii=False, indent=2), ""
    try:
        seed_i = int(float(seed))
    except Exception:
        seed_i = int(time.time())
    rng = np.random.default_rng(seed_i)
    order = rng.permutation(len(rows)).tolist()
    chosen_row = None
    chosen_img_path: Optional[Path] = None
    for idx in order:
        row = rows[int(idx)]
        img_raw, txt = _extract_ocr_manifest_row_fields_ui(row)
        img_path = _resolve_ocr_manifest_image_path_ui(img_raw, manifest)
        if img_path is None or not txt:
            continue
        chosen_row = row
        chosen_img_path = img_path
        break
    if chosen_row is None or chosen_img_path is None:
        msg = f"No usable image/text row found in {manifest}"
        return None, "", json.dumps({"ok": False, "error": msg}, ensure_ascii=False, indent=2), ""
    with Image.open(chosen_img_path) as im:
        rgb = im.convert("RGB")
    _, gt = _extract_ocr_manifest_row_fields_ui(chosen_row)
    debug = {
        "ok": True,
        "dataset_root": str(root),
        "split": split,
        "manifest": str(manifest),
        "image_path": str(chosen_img_path),
        "line_id": chosen_row.get("line_id"),
        "page_id": chosen_row.get("page_id"),
        "doc_id": chosen_row.get("doc_id"),
    }
    return np.array(rgb), str(gt), json.dumps(debug, ensure_ascii=False, indent=2), str(chosen_img_path)


def scan_line_clip_dual_models_ui(models_dir: str):
    base = Path(models_dir).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        empty = gr.update(choices=[], value=None)
        return empty, "", "", "", "", "{}", f"Directory not found: {base}"

    bundles: Dict[str, Dict[str, str]] = {}
    bundle_rows: List[Dict[str, Any]] = []

    def _parse_ckpt_sort_fields(name: str) -> Tuple[int, float, float]:
        low = str(name or "").lower()
        step = -1
        i2t1 = -1.0
        t2i1 = -1.0
        m_step = re.search(r"checkpoint_step_(\d+)", low)
        if m_step:
            try:
                step = int(m_step.group(1))
            except Exception:
                step = -1
        m_i2t1 = re.search(r"_i2t1-([0-9]+p[0-9]+|na)", low)
        if m_i2t1:
            tok = m_i2t1.group(1)
            if tok != "na":
                try:
                    i2t1 = float(tok.replace("p", "."))
                except Exception:
                    i2t1 = -1.0
        m_t2i1 = re.search(r"_t2i1-([0-9]+p[0-9]+|na)", low)
        if m_t2i1:
            tok = m_t2i1.group(1)
            if tok != "na":
                try:
                    t2i1 = float(tok.replace("p", "."))
                except Exception:
                    t2i1 = -1.0
        return step, i2t1, t2i1

    def _to_f(raw: Any, default: float = -1.0) -> float:
        try:
            val = float(raw)
            if math.isfinite(val):
                return float(val)
        except Exception:
            pass
        return float(default)

    def _fmt_metric(v: float) -> str:
        if not math.isfinite(float(v)) or float(v) < 0.0:
            return "na"
        return f"{float(v):.4f}"

    for cfg_path in sorted(base.rglob("*training_config.json")):
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if str(data.get("mode", "")).strip().lower() != "line_clip":
            continue
        root = cfg_path.parent.resolve()
        stem = cfg_path.stem
        prefix = ""
        if stem.endswith("_training_config"):
            prefix = stem[: -len("_training_config")]
        elif stem != "training_config":
            prefix = stem
        pfx = f"{prefix}_" if prefix else ""

        img_backbone = root / f"{pfx}text_hierarchy_vit_backbone"
        img_head = root / f"{pfx}text_hierarchy_projection_head.pt"
        txt_encoder = root / f"{pfx}text_hierarchy_clip_text_encoder"
        txt_head = root / f"{pfx}text_hierarchy_clip_text_projection_head.pt"
        if not (img_backbone.exists() and img_head.exists() and txt_encoder.exists() and txt_head.exists()):
            continue
        key = str(root if not prefix else (root / prefix))
        last_val = data.get("last_val_metrics") if isinstance(data.get("last_val_metrics"), dict) else {}
        cfg_i2t1 = _to_f(last_val.get("val_i2t_r1", -1.0))
        cfg_t2i1 = _to_f(last_val.get("val_t2i_r1", -1.0))
        cfg_i2t5 = _to_f(last_val.get("val_i2t_r5", -1.0))
        cfg_t2i5 = _to_f(last_val.get("val_t2i_r5", -1.0))
        cfg_n = int(_to_int(last_val.get("val_count", 0), 0))
        cfg_step = int(_to_int(data.get("global_step", -1), -1))
        name_step, name_i2t1, name_t2i1 = _parse_ckpt_sort_fields(Path(key).name)
        # Prefer explicit config metrics; fallback to checkpoint suffix parsing.
        sort_i2t1 = cfg_i2t1 if cfg_i2t1 >= 0.0 else name_i2t1
        sort_t2i1 = cfg_t2i1 if cfg_t2i1 >= 0.0 else name_t2i1
        sort_step = max(cfg_step, name_step)
        score_primary = ((sort_i2t1 if sort_i2t1 >= 0 else -1.0) + (sort_t2i1 if sort_t2i1 >= 0 else -1.0)) / 2.0

        bundle_name = Path(key).name
        metrics_label = (
            f"i2t@1={_fmt_metric(cfg_i2t1 if cfg_i2t1 >= 0 else name_i2t1)} "
            f"t2i@1={_fmt_metric(cfg_t2i1 if cfg_t2i1 >= 0 else name_t2i1)} "
            f"i2t@5={_fmt_metric(cfg_i2t5)} t2i@5={_fmt_metric(cfg_t2i5)}"
        )
        extra_label = f"step={sort_step if sort_step >= 0 else 'na'} n={cfg_n if cfg_n > 0 else 'na'}"
        dropdown_label = f"{bundle_name} | {metrics_label} | {extra_label}"
        bundles[key] = {
            "image_backbone": str(img_backbone.resolve()),
            "image_head": str(img_head.resolve()),
            "text_encoder": str(txt_encoder.resolve()),
            "text_head": str(txt_head.resolve()),
            "bundle_root": str(Path(key).resolve()),
            "config_path": str(cfg_path.resolve()),
            "display_label": dropdown_label,
            "sort_score_primary": f"{float(score_primary):.6f}",
            "sort_i2t1": f"{float(sort_i2t1):.6f}",
            "sort_t2i1": f"{float(sort_t2i1):.6f}",
            "sort_step": str(int(sort_step)),
            "val_count": str(int(cfg_n)),
        }
        bundle_rows.append(
            {
                "key": key,
                "label": dropdown_label,
                "score_primary": float(score_primary),
                "i2t1": float(sort_i2t1),
                "t2i1": float(sort_t2i1),
                "step": int(sort_step),
            }
        )

    # Prefer strongest validation performance first, then step.
    bundle_rows = sorted(
        bundle_rows,
        key=lambda p: (
            float(p.get("score_primary", -1.0)),
            float(p.get("i2t1", -1.0)),
            float(p.get("t2i1", -1.0)),
            int(p.get("step", -1)),
            str(p.get("key", "")),
        ),
        reverse=True,
    )
    found_roots = [str(r.get("key", "")) for r in bundle_rows if str(r.get("key", ""))]
    choice_tuples = [(str(r.get("label", r.get("key", ""))), str(r.get("key", ""))) for r in bundle_rows]
    default_root = found_roots[0] if found_roots else ""
    default_bundle = bundles.get(default_root, {}) if default_root else {}
    state_json = json.dumps(bundles, ensure_ascii=False)
    msg = f"Found {len(found_roots)} line_clip model bundle(s) in {base} (sorted by val retrieval performance, then step)"
    return (
        gr.update(choices=choice_tuples, value=(default_root if default_root else None)),
        default_bundle.get("image_backbone", ""),
        default_bundle.get("image_head", ""),
        default_bundle.get("text_encoder", ""),
        default_bundle.get("text_head", ""),
        state_json,
        msg,
    )


def on_line_clip_bundle_change_ui(bundle_root: str, bundle_state_json: str):
    root = str(bundle_root or "").strip()
    bundles: Dict[str, Dict[str, str]] = {}
    try:
        bundles = json.loads(str(bundle_state_json or "{}"))
    except Exception:
        bundles = {}
    rec = bundles.get(root, {}) if isinstance(bundles, dict) else {}
    return (
        str(rec.get("image_backbone", "") or ""),
        str(rec.get("image_head", "") or ""),
        str(rec.get("text_encoder", "") or ""),
        str(rec.get("text_head", "") or ""),
    )


def _load_ui_projection_head_from_file(projection_head_path: str, in_dim_hint: int) -> Optional[Any]:
    if torch is None or nn is None:
        return None
    ppath_raw = (projection_head_path or "").strip()
    if not ppath_raw:
        return None
    ppath = Path(ppath_raw).expanduser().resolve()
    if not ppath.exists() or not ppath.is_file():
        return None
    payload = torch.load(str(ppath), map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload.get("state_dict", {})
        in_dim = _to_int(payload.get("input_dim", 0), 0)
        out_dim = _to_int(payload.get("output_dim", 0), 0)
    else:
        state_dict = payload
        in_dim = 0
        out_dim = 0
    if in_dim <= 0:
        in_dim = int(in_dim_hint)
    if out_dim <= 0:
        out_dim = int(in_dim)
    if in_dim <= 0 or out_dim <= 0:
        return None
    head = _UIProjectionHead(in_dim=int(in_dim), out_dim=int(out_dim))
    head.load_state_dict(state_dict, strict=False)
    head.eval()
    return head


def _pooled_text_embedding_for_ui(text_encoder: Any, input_ids: Any, attention_mask: Any):
    kwargs: Dict[str, Any] = {"input_ids": input_ids, "return_dict": True}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    model_for_forward = text_encoder
    base_model = text_encoder
    seen_ids = set()
    while hasattr(base_model, "module") and id(base_model) not in seen_ids:
        seen_ids.add(id(base_model))
        try:
            base_model = getattr(base_model, "module")
        except Exception:
            break
    cfg = getattr(base_model, "config", None)
    if bool(getattr(cfg, "is_encoder_decoder", False)) and hasattr(base_model, "get_encoder"):
        try:
            model_for_forward = base_model.get_encoder()
        except Exception:
            model_for_forward = text_encoder
    outputs = model_for_forward(**kwargs)
    pooler = getattr(outputs, "pooler_output", None)
    if pooler is not None:
        return pooler
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        raise RuntimeError("Text encoder output has no pooler_output/last_hidden_state")
    if hidden.ndim != 3:
        return hidden
    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.to(hidden.device).float().unsqueeze(-1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (hidden * mask).sum(dim=1) / denom


@lru_cache(maxsize=4)
def _load_line_clip_dual_runtime_cached(
    image_backbone_path: str,
    image_head_path: str,
    text_encoder_dir: str,
    text_head_path: str,
    device_pref: str,
):
    if torch is None or AutoModel is None or AutoTokenizer is None or nn is None:
        raise RuntimeError("line_clip debug requires torch + transformers.")
    image_processor, image_backbone, image_head = _load_hierarchy_vit_encoder_bundle(image_backbone_path, image_head_path)
    txt_dir = Path((text_encoder_dir or "").strip()).expanduser().resolve()
    if not txt_dir.exists() or not txt_dir.is_dir():
        raise FileNotFoundError(f"text encoder dir not found: {txt_dir}")
    text_tokenizer = AutoTokenizer.from_pretrained(str(txt_dir), use_fast=True)
    text_encoder = AutoModel.from_pretrained(str(txt_dir))
    txt_hidden = _to_int(getattr(getattr(text_encoder, "config", None), "hidden_size", 0), 0)
    if txt_hidden <= 0:
        txt_hidden = _to_int(getattr(getattr(text_encoder, "config", None), "d_model", 0), 0)
    text_head = _load_ui_projection_head_from_file(text_head_path, in_dim_hint=txt_hidden)
    if text_head is None:
        raise RuntimeError(f"Could not load text projection head: {text_head_path}")

    resolved_device, device_note = _resolve_torch_device_for_encoding(device_pref)
    image_backbone = image_backbone.to(resolved_device).eval()
    text_encoder = text_encoder.to(resolved_device).eval()
    if image_head is not None:
        image_head = image_head.to(resolved_device).eval()
    text_head = text_head.to(resolved_device).eval()
    runtime_cfg = _load_hierarchy_encoder_runtime_config(image_backbone_path, image_head_path)
    return {
        "image_processor": image_processor,
        "image_backbone": image_backbone,
        "image_head": image_head,
        "text_tokenizer": text_tokenizer,
        "text_encoder": text_encoder,
        "text_head": text_head,
        "runtime_cfg": runtime_cfg,
        "device": resolved_device,
        "device_msg": device_note,
    }


def _collect_ocr_split_rows_ui(dataset_root: str, split_name: str) -> List[Dict[str, Any]]:
    root = Path((dataset_root or "").strip()).expanduser().resolve()
    split = str(split_name or "").strip()
    manifest = root / split / "meta" / "lines.jsonl"
    rows = _read_jsonl_rows_ui(manifest)
    usable: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        img_raw, txt = _extract_ocr_manifest_row_fields_ui(row)
        img_path = _resolve_ocr_manifest_image_path_ui(img_raw, manifest)
        if img_path is None:
            continue
        rec = dict(row)
        rec["_image_path_resolved"] = str(img_path)
        rec["_text_resolved"] = str(txt or "")
        rec["_sample_index"] = int(i)
        usable.append(rec)
    return usable


def _prepare_line_clip_image_tensor_ui(
    image_path: str,
    runtime_cfg: Dict[str, Any],
    width_buckets: List[int],
    target_height: int,
    patch_multiple: int,
    max_width: int,
):
    if torch is None:
        raise RuntimeError("PyTorch not available")
    cfg_obj = runtime_cfg if isinstance(runtime_cfg, dict) else {}
    with Image.open(str(image_path)) as im:
        rgb = im.convert("RGB")
    rgb = _apply_line_clip_ui_preprocess_from_runtime_cfg(rgb, cfg_obj)
    norm_img = _normalize_for_vit_encoding(
        image=rgb,
        target_height=int(target_height),
        width_buckets=width_buckets,
        patch_multiple=int(patch_multiple),
        max_width=int(max_width),
    )
    arr = np.asarray(norm_img).astype(np.float32, copy=False) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _encode_line_clip_image_crops_ui(
    runtime: Dict[str, Any],
    image_paths: List[str],
    batch_size: int,
    l2_normalize: bool,
    progress_label: str = "",
    progress_every_batches: int = 0,
    image_preproc_workers: int = 0,
    tqdm_progress: bool = False,
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch not available")
    image_backbone = runtime["image_backbone"]
    image_head = runtime["image_head"]
    image_processor = runtime["image_processor"]
    runtime_cfg = runtime.get("runtime_cfg") or {}
    runtime_cfg_dict = runtime_cfg if isinstance(runtime_cfg, dict) else {}
    device = runtime["device"]
    target_height = _to_int(runtime_cfg_dict.get("target_height", 64), 64)
    patch_multiple = _to_int(runtime_cfg_dict.get("patch_multiple", 16), 16)
    max_width_cfg = _to_int(runtime_cfg_dict.get("max_width", 1024), 1024)
    width_buckets = _parse_width_buckets_for_encoding(
        raw=runtime_cfg_dict.get("width_buckets", [256, 384, 512, 768]),
        patch_multiple=patch_multiple,
        max_width=max_width_cfg,
    )
    mean = list(getattr(image_processor, "image_mean", [0.5, 0.5, 0.5]))
    std = list(getattr(image_processor, "image_std", [0.5, 0.5, 0.5]))
    if len(mean) == 1:
        mean = mean * 3
    if len(std) == 1:
        std = std * 3
    mean_t = torch.tensor([float(v) for v in mean[:3]], dtype=torch.float32).view(1, 3, 1, 1).to(device)
    std_t = torch.tensor([max(1e-6, float(v)) for v in std[:3]], dtype=torch.float32).view(1, 3, 1, 1).to(device)

    bs = max(1, int(batch_size))
    total = int(len(image_paths))
    if total <= 0:
        return np.zeros((0, 1), dtype=np.float32)
    out_arr: Optional[np.ndarray] = None
    use_non_blocking = str(device).startswith("cuda")
    max_workers = max(0, int(image_preproc_workers or 0))
    executor: Optional[ThreadPoolExecutor] = None
    if max_workers > 1:
        executor = ThreadPoolExecutor(max_workers=max_workers)
    inference_ctx = getattr(torch, "inference_mode", torch.no_grad)
    pbar = None

    def _prep_path(p: str):
        return _prepare_line_clip_image_tensor_ui(
            image_path=p,
            runtime_cfg=runtime_cfg_dict,
            width_buckets=width_buckets,
            target_height=target_height,
            patch_multiple=patch_multiple,
            max_width=max_width_cfg,
        )

    try:
        with inference_ctx():
            total_batches = int(math.ceil(total / max(1, bs))) if total > 0 else 0
            if bool(tqdm_progress) and _tqdm is not None and total_batches > 0:
                pbar = _tqdm(
                    total=total_batches,
                    desc=str(progress_label or "line_clip image encode"),
                    unit="batch",
                    leave=False,
                )
            for batch_idx, start in enumerate(range(0, len(image_paths), bs), start=1):
                chunk = image_paths[start : start + bs]
                if executor is not None:
                    tensors = list(executor.map(_prep_path, chunk))
                else:
                    tensors = [_prep_path(p) for p in chunk]
                max_w = max(int(t.shape[-1]) for t in tensors)
                pm = max(1, int(patch_multiple))
                max_w = int(math.ceil(max_w / pm) * pm)
                padded = []
                for t in tensors:
                    if int(t.shape[-1]) < max_w:
                        t = torch_f.pad(t, (0, max_w - int(t.shape[-1]), 0, 0), mode="replicate")
                    elif int(t.shape[-1]) > max_w:
                        t = t[:, :, :max_w]
                    padded.append(t)
                px = torch.stack(padded, dim=0)
                px = px.to(device, non_blocking=use_non_blocking)
                px = (px - mean_t) / std_t
                emb = _pooled_image_embedding_for_ui(image_backbone, px)
                if image_head is not None:
                    emb = image_head(emb)
                if bool(l2_normalize) and torch_f is not None:
                    emb = torch_f.normalize(emb, dim=-1)
                emb_np = emb.detach().cpu().float().numpy()
                if out_arr is None:
                    if emb_np.ndim != 2:
                        raise RuntimeError("Image embedding output must be rank-2.")
                    out_arr = np.empty((total, int(emb_np.shape[1])), dtype=np.float32)
                out_arr[start : start + emb_np.shape[0]] = emb_np
                if pbar is not None:
                    pbar.update(1)
                if (not bool(tqdm_progress)) and int(progress_every_batches or 0) > 0 and (
                    batch_idx % int(progress_every_batches) == 0 or batch_idx == total_batches
                ):
                    label = str(progress_label or "line_clip image encode")
                    LOGGER.info(
                        "%s progress: batch %d/%d samples %d/%d",
                        label,
                        int(batch_idx),
                        int(total_batches),
                        int(min(start + len(chunk), total)),
                        int(total),
                    )
    finally:
        if pbar is not None:
            pbar.close()
        if executor is not None:
            executor.shutdown(wait=True)
    return out_arr if out_arr is not None else np.zeros((0, 1), dtype=np.float32)


def _encode_line_clip_texts_ui(
    runtime: Dict[str, Any],
    texts: List[str],
    batch_size: int,
    l2_normalize: bool,
    text_max_length: int,
    progress_label: str = "",
    progress_every_batches: int = 0,
    tqdm_progress: bool = False,
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch not available")
    tokenizer = runtime["text_tokenizer"]
    text_encoder = runtime["text_encoder"]
    text_head = runtime["text_head"]
    device = runtime["device"]
    bs = max(1, int(batch_size))
    total = int(len(texts))
    if total <= 0:
        return np.zeros((0, 1), dtype=np.float32)
    out_arr: Optional[np.ndarray] = None
    use_non_blocking = str(device).startswith("cuda")
    inference_ctx = getattr(torch, "inference_mode", torch.no_grad)
    pbar = None
    try:
        with inference_ctx():
            total_batches = int(math.ceil(total / max(1, bs))) if total > 0 else 0
            if bool(tqdm_progress) and _tqdm is not None and total_batches > 0:
                pbar = _tqdm(
                    total=total_batches,
                    desc=str(progress_label or "line_clip text encode"),
                    unit="batch",
                    leave=False,
                )
            for batch_idx, start in enumerate(range(0, len(texts), bs), start=1):
                chunk = [str(t or "") for t in texts[start : start + bs]]
                tok_kwargs: Dict[str, Any] = {
                    "padding": True,
                    "truncation": True,
                    "max_length": max(8, int(text_max_length)),
                    "return_tensors": "pt",
                }
                if use_non_blocking:
                    tok_kwargs["pad_to_multiple_of"] = 8
                tok = tokenizer(chunk, **tok_kwargs)
                input_ids = tok["input_ids"].to(device, non_blocking=use_non_blocking)
                attn = tok.get("attention_mask")
                if attn is not None:
                    attn = attn.to(device, non_blocking=use_non_blocking)
                txt_emb = _pooled_text_embedding_for_ui(text_encoder, input_ids, attn)
                txt_z = text_head(txt_emb)
                if bool(l2_normalize) and torch_f is not None:
                    txt_z = torch_f.normalize(txt_z, dim=-1)
                txt_np = txt_z.detach().cpu().float().numpy()
                if out_arr is None:
                    if txt_np.ndim != 2:
                        raise RuntimeError("Text embedding output must be rank-2.")
                    out_arr = np.empty((total, int(txt_np.shape[1])), dtype=np.float32)
                out_arr[start : start + txt_np.shape[0]] = txt_np
                if pbar is not None:
                    pbar.update(1)
                if (not bool(tqdm_progress)) and int(progress_every_batches or 0) > 0 and (
                    batch_idx % int(progress_every_batches) == 0 or batch_idx == total_batches
                ):
                    label = str(progress_label or "line_clip text encode")
                    LOGGER.info(
                        "%s progress: batch %d/%d samples %d/%d",
                        label,
                        int(batch_idx),
                        int(total_batches),
                        int(min(start + len(chunk), total)),
                        int(total),
                    )
    finally:
        if pbar is not None:
            pbar.close()
    return out_arr if out_arr is not None else np.zeros((0, 1), dtype=np.float32)


_LINE_CLIP_DEBUG_CACHE: Dict[str, Dict[str, Any]] = {}


def _line_clip_debug_disk_cache_root_ui() -> Path:
    return (Path(tempfile.gettempdir()) / "pechabridge_ui" / "line_clip_corpus_cache").resolve()


def _line_clip_debug_disk_cache_id_ui(key: str) -> str:
    raw = str(key or "")
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:24]


def _line_clip_debug_disk_cache_dir_ui(key: str) -> Path:
    return (_line_clip_debug_disk_cache_root_ui() / _line_clip_debug_disk_cache_id_ui(key)).resolve()


def _load_line_clip_debug_corpus_from_disk_ui(key: str) -> Optional[Dict[str, Any]]:
    cache_dir = _line_clip_debug_disk_cache_dir_ui(key)
    meta_path = (cache_dir / "meta.json").resolve()
    rows_path = (cache_dir / "rows.json").resolve()
    img_path = (cache_dir / "image_embeddings.npy").resolve()
    txt_path = (cache_dir / "text_embeddings.npy").resolve()
    if not (cache_dir.exists() and meta_path.exists() and rows_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            return None
        if str(meta.get("key", "") or "") != str(key):
            return None
        rows = json.loads(rows_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            return None
        image_emb = None
        text_emb = None
        has_image = bool(img_path.exists())
        has_text = bool(txt_path.exists())
        if not (has_image or has_text):
            return None
        if has_image:
            _img = np.load(str(img_path))
            if _img.ndim != 2 or int(_img.shape[0]) != len(rows):
                return None
            image_emb = np.asarray(_img, dtype=np.float32)
        if has_text:
            _txt = np.load(str(txt_path))
            if _txt.ndim != 2 or int(_txt.shape[0]) != len(rows):
                return None
            text_emb = np.asarray(_txt, dtype=np.float32)
        available_banks = []
        if image_emb is not None:
            available_banks.append("image")
        if text_emb is not None:
            available_banks.append("text")
        return {
            "rows": rows,
            "image_embeddings": image_emb,
            "text_embeddings": text_emb,
            "device": "",
            "device_msg": "Loaded corpus embeddings from disk cache.",
            "count": int(len(rows)),
            "cache_source": "disk",
            "disk_cache_dir": str(cache_dir),
            "available_banks": available_banks,
        }
    except Exception:
        return None


def _save_line_clip_debug_corpus_to_disk_ui(key: str, corpus: Dict[str, Any]) -> str:
    cache_dir = _line_clip_debug_disk_cache_dir_ui(key)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = list(corpus.get("rows") or [])
    image_raw = corpus.get("image_embeddings", None)
    text_raw = corpus.get("text_embeddings", None)
    image_emb = None if image_raw is None else np.asarray(image_raw, dtype=np.float32)
    text_emb = None if text_raw is None else np.asarray(text_raw, dtype=np.float32)
    rows_path = (cache_dir / "rows.json").resolve()
    img_path = (cache_dir / "image_embeddings.npy").resolve()
    txt_path = (cache_dir / "text_embeddings.npy").resolve()
    meta_path = (cache_dir / "meta.json").resolve()
    rows_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    if image_emb is not None:
        np.save(str(img_path), image_emb.astype(np.float32, copy=False))
    if text_emb is not None:
        np.save(str(txt_path), text_emb.astype(np.float32, copy=False))
    meta = {
        "key": str(key),
        "created_at_unix": float(time.time()),
        "rows_count": int(len(rows)),
        "has_image_embeddings": bool(image_emb is not None or img_path.exists()),
        "has_text_embeddings": bool(text_emb is not None or txt_path.exists()),
        "image_shape": ([int(v) for v in image_emb.shape] if image_emb is not None else None),
        "text_shape": ([int(v) for v in text_emb.shape] if text_emb is not None else None),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(cache_dir)


def _line_clip_debug_cache_key(
    dataset_root: str,
    split_name: str,
    image_backbone_path: str,
    image_head_path: str,
    text_encoder_dir: str,
    text_head_path: str,
    text_max_length: int,
    l2_normalize: bool,
) -> str:
    return "||".join([
        str(Path(dataset_root).expanduser().resolve()),
        str(split_name or ""),
        str(Path(image_backbone_path).expanduser().resolve()),
        str(Path(image_head_path).expanduser().resolve()) if (image_head_path or "").strip() else "",
        str(Path(text_encoder_dir).expanduser().resolve()),
        str(Path(text_head_path).expanduser().resolve()),
        str(int(text_max_length)),
        "1" if bool(l2_normalize) else "0",
    ])


def _get_or_build_line_clip_debug_corpus_ui(
    dataset_root: str,
    split_name: str,
    image_backbone_path: str,
    image_head_path: str,
    text_encoder_dir: str,
    text_head_path: str,
    device_pref: str,
    batch_size: int,
    text_max_length: int,
    l2_normalize: bool,
    required_banks: str = "both",
    progress_every_batches: int = 0,
    image_batch_size: Optional[int] = None,
    text_batch_size: Optional[int] = None,
    image_preproc_workers: int = 0,
    tqdm_progress: bool = False,
) -> Dict[str, Any]:
    required_banks_norm = str(required_banks or "both").strip().lower()
    if required_banks_norm not in {"both", "image", "text"}:
        required_banks_norm = "both"
    need_image = required_banks_norm in {"both", "image"}
    need_text = required_banks_norm in {"both", "text"}
    t_start = time.time()
    key = _line_clip_debug_cache_key(
        dataset_root=dataset_root,
        split_name=split_name,
        image_backbone_path=image_backbone_path,
        image_head_path=image_head_path,
        text_encoder_dir=text_encoder_dir,
        text_head_path=text_head_path,
        text_max_length=text_max_length,
        l2_normalize=l2_normalize,
    )
    cached = _LINE_CLIP_DEBUG_CACHE.get(key)
    if cached is not None:
        cached_has_img = bool(isinstance(cached, dict) and cached.get("image_embeddings") is not None)
        cached_has_txt = bool(isinstance(cached, dict) and cached.get("text_embeddings") is not None)
        if ((not need_image or cached_has_img) and (not need_text or cached_has_txt)):
            if isinstance(cached, dict):
                cached.setdefault("available_banks", [b for b, ok in (("image", cached_has_img), ("text", cached_has_txt)) if ok])
                cached["requested_banks"] = required_banks_norm
                cached["built_banks"] = list(cached.get("built_banks") or [])
                cached.setdefault("timing", {})
                cached["timing"]["cache_lookup_s"] = round(float(time.time() - t_start), 4)
                cached["timing"]["cache_hit_tier"] = "memory"
            return cached
        if isinstance(cached, dict):
            cached.setdefault("timing", {})
            cached["timing"]["cache_lookup_s"] = round(float(time.time() - t_start), 4)
    t_disk_load0 = time.time()
    disk_cached = _load_line_clip_debug_corpus_from_disk_ui(key)
    if disk_cached is not None:
        disk_has_img = bool(disk_cached.get("image_embeddings") is not None)
        disk_has_txt = bool(disk_cached.get("text_embeddings") is not None)
        resolved_device, device_note = _resolve_torch_device_for_encoding(device_pref)
        disk_cached["device"] = str(resolved_device)
        cache_msg = str(disk_cached.get("device_msg", "") or "")
        if device_note:
            cache_msg = (cache_msg + " " + str(device_note)).strip()
        disk_cached["device_msg"] = cache_msg
        disk_cached["timing"] = {
            "cache_hit_tier": "disk",
            "disk_load_s": round(float(time.time() - t_disk_load0), 4),
            "cache_lookup_s": round(float(time.time() - t_start), 4),
        }
        disk_cached["perf"] = {
            "rows_per_s_total": (
                round(float(disk_cached.get("count", 0)) / max(1e-9, float(time.time() - t_start)), 2)
                if int(disk_cached.get("count", 0)) > 0 else 0.0
            ),
        }
        disk_cached["requested_banks"] = required_banks_norm
        if ((not need_image or disk_has_img) and (not need_text or disk_has_txt)):
            _LINE_CLIP_DEBUG_CACHE[key] = disk_cached
            return disk_cached
        rows = list(disk_cached.get("rows") or [])
        rows_collect_s = 0.0
    else:
        t_rows0 = time.time()
        rows = _collect_ocr_split_rows_ui(dataset_root, split_name)
        rows_collect_s = float(time.time() - t_rows0)
    if not rows:
        raise RuntimeError("No usable OCR rows found for selected dataset/split.")

    existing_image_emb = None
    existing_text_emb = None
    if isinstance(cached, dict):
        if cached.get("image_embeddings") is not None:
            existing_image_emb = np.asarray(cached.get("image_embeddings"), dtype=np.float32)
        if cached.get("text_embeddings") is not None:
            existing_text_emb = np.asarray(cached.get("text_embeddings"), dtype=np.float32)
    if disk_cached is not None:
        if existing_image_emb is None and disk_cached.get("image_embeddings") is not None:
            existing_image_emb = np.asarray(disk_cached.get("image_embeddings"), dtype=np.float32)
        if existing_text_emb is None and disk_cached.get("text_embeddings") is not None:
            existing_text_emb = np.asarray(disk_cached.get("text_embeddings"), dtype=np.float32)
    built_banks: List[str] = []
    need_build_image = bool(need_image and existing_image_emb is None)
    need_build_text = bool(need_text and existing_text_emb is None)
    image_bs = max(1, int(image_batch_size if image_batch_size is not None else batch_size))
    text_bs = max(1, int(text_batch_size if text_batch_size is not None else batch_size))
    image_preproc_workers = max(0, int(image_preproc_workers or 0))

    t_runtime0 = time.time()
    runtime = _load_line_clip_dual_runtime_cached(
        image_backbone_path,
        image_head_path,
        text_encoder_dir,
        text_head_path,
        device_pref,
    )
    runtime_load_s = float(time.time() - t_runtime0)
    image_paths = [str(r.get("_image_path_resolved", "")) for r in rows]
    texts = [str(r.get("_text_resolved", "")) for r in rows]
    image_encode_s = 0.0
    text_encode_s = 0.0
    image_emb = existing_image_emb
    text_emb = existing_text_emb
    if need_build_image:
        t_img0 = time.time()
        image_emb = _encode_line_clip_image_crops_ui(
            runtime,
            image_paths=image_paths,
            batch_size=image_bs,
            l2_normalize=l2_normalize,
            progress_label=f"line_clip cache build [{split_name}] image",
            progress_every_batches=int(progress_every_batches or 0),
            image_preproc_workers=image_preproc_workers,
            tqdm_progress=bool(tqdm_progress),
        )
        image_encode_s = float(time.time() - t_img0)
        built_banks.append("image")
    if need_build_text:
        t_txt0 = time.time()
        text_emb = _encode_line_clip_texts_ui(
            runtime,
            texts=texts,
            batch_size=text_bs,
            l2_normalize=l2_normalize,
            text_max_length=text_max_length,
            progress_label=f"line_clip cache build [{split_name}] text",
            progress_every_batches=int(progress_every_batches or 0),
            tqdm_progress=bool(tqdm_progress),
        )
        text_encode_s = float(time.time() - t_txt0)
        built_banks.append("text")
    count_rows = int(len(rows))
    total_build_before_save_s = float(time.time() - t_start)
    image_shape = [int(v) for v in image_emb.shape] if image_emb is not None and getattr(image_emb, "ndim", 0) == 2 else []
    text_shape = [int(v) for v in text_emb.shape] if text_emb is not None and getattr(text_emb, "ndim", 0) == 2 else []
    image_bytes = int(getattr(image_emb, "nbytes", 0)) if image_emb is not None else 0
    text_bytes = int(getattr(text_emb, "nbytes", 0)) if text_emb is not None else 0
    available_banks = []
    if image_emb is not None:
        available_banks.append("image")
    if text_emb is not None:
        available_banks.append("text")
    corpus = {
        "rows": rows,
        "image_embeddings": (image_emb.astype(np.float32, copy=False) if image_emb is not None else None),
        "text_embeddings": (text_emb.astype(np.float32, copy=False) if text_emb is not None else None),
        "device": str(runtime.get("device", "")),
        "device_msg": str(runtime.get("device_msg", "")),
        "count": count_rows,
        "cache_source": "built",
        "requested_banks": required_banks_norm,
        "available_banks": available_banks,
        "built_banks": built_banks,
        "timing": {
            "cache_hit_tier": "miss_build",
            "rows_collect_s": round(rows_collect_s, 4),
            "runtime_load_s": round(runtime_load_s, 4),
            "image_encode_s": round(image_encode_s, 4),
            "text_encode_s": round(text_encode_s, 4),
            "build_before_save_s": round(total_build_before_save_s, 4),
        },
        "perf": {
            "image_shape": image_shape,
            "text_shape": text_shape,
            "image_bytes": image_bytes,
            "text_bytes": text_bytes,
            "rows_per_s_image_encode": round(count_rows / max(image_encode_s, 1e-9), 2),
            "rows_per_s_text_encode": round(count_rows / max(text_encode_s, 1e-9), 2),
            "rows_per_s_build_before_save": round(count_rows / max(total_build_before_save_s, 1e-9), 2),
            "image_batch_size": int(image_bs),
            "text_batch_size": int(text_bs),
            "image_preproc_workers": int(image_preproc_workers),
        },
    }
    t_save0 = time.time()
    try:
        disk_cache_dir = _save_line_clip_debug_corpus_to_disk_ui(key, corpus)
        corpus["disk_cache_dir"] = str(disk_cache_dir)
    except Exception as exc:
        corpus["disk_cache_write_error"] = f"{type(exc).__name__}: {exc}"
    disk_save_s = float(time.time() - t_save0)
    total_elapsed_s = float(time.time() - t_start)
    timing = corpus.get("timing") if isinstance(corpus.get("timing"), dict) else {}
    timing["disk_save_s"] = round(disk_save_s, 4)
    timing["total_elapsed_s"] = round(total_elapsed_s, 4)
    corpus["timing"] = timing
    perf = corpus.get("perf") if isinstance(corpus.get("perf"), dict) else {}
    perf["rows_per_s_total"] = round(count_rows / max(total_elapsed_s, 1e-9), 2)
    corpus["perf"] = perf
    _LINE_CLIP_DEBUG_CACHE[key] = corpus
    if len(_LINE_CLIP_DEBUG_CACHE) > 3:
        for k in list(_LINE_CLIP_DEBUG_CACHE.keys())[:-3]:
            _LINE_CLIP_DEBUG_CACHE.pop(k, None)
    return corpus


def load_random_line_clip_debug_sample_ui(dataset_root: str, split_name: str, seed: float):
    image_arr, gt_text, meta_json, img_path = load_random_donut_ocr_dataset_sample_ui(dataset_root, split_name, seed)
    sample_state = {}
    try:
        meta = json.loads(meta_json or "{}")
    except Exception:
        meta = {}
    if isinstance(meta, dict) and bool(meta.get("ok")):
        sample_state = {
            "dataset_root": str(meta.get("dataset_root", dataset_root)),
            "split": str(meta.get("split", split_name)),
            "image_path": str(meta.get("image_path", img_path)),
            "ground_truth": str(gt_text or ""),
            "doc_id": str(meta.get("doc_id", "")),
            "page_id": str(meta.get("page_id", "")),
            "line_id": _to_int(meta.get("line_id", -1), -1),
        }
    return image_arr, gt_text, gt_text, meta_json, img_path, sample_state


def _line_clip_debug_find_ground_truth_index(rows: List[Dict[str, Any]], sample_state: Dict[str, Any]) -> int:
    img_path = str(sample_state.get("image_path", "") or "")
    if img_path:
        for i, r in enumerate(rows):
            if str(r.get("_image_path_resolved", "") or "") == img_path:
                return int(i)
    gt = str(sample_state.get("ground_truth", "") or "")
    page_id = str(sample_state.get("page_id", "") or "")
    line_id = _to_int(sample_state.get("line_id", -1), -1)
    for i, r in enumerate(rows):
        if page_id and str(r.get("page_id", "") or "") != page_id:
            continue
        if line_id >= 0 and _to_int(r.get("line_id", -1), -1) != line_id:
            continue
        if gt and str(r.get("_text_resolved", "") or "") == gt:
            return int(i)
    return -1


def _load_image_as_rgb_np_ui(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")).astype(np.uint8, copy=False)


def _write_line_clip_debug_results_csv_ui(
    *,
    rows: List[Dict[str, Any]],
    order: np.ndarray,
    sims: np.ndarray,
    top_k: int,
    gt_idx: int,
    query_desc: Dict[str, Any],
    extra_metrics_by_index: Optional[Dict[int, Dict[str, Any]]] = None,
) -> str:
    out_dir = (Path(tempfile.gettempdir()) / "pechabridge_ui" / "line_clip_debug_exports").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    mode = str(query_desc.get("mode", "retrieval") or "retrieval")
    safe_mode = re.sub(r"[^a-zA-Z0-9_.-]+", "_", mode).strip("_") or "retrieval"
    out_path = (out_dir / f"{safe_mode}_{ts}.csv").resolve()
    k = max(1, min(int(top_k), int(len(order))))
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "score",
                "sample_index",
                "doc_id",
                "page_id",
                "line_id",
                "image_path",
                "text",
                "is_ground_truth",
                "query_mode",
                    "query_text",
                    "query_image_path",
                    "cer_vs_gt_text",
                    "sem_text_vs_gt_text",
                    "sem_image_vs_gt_image",
                ],
        )
        writer.writeheader()
        for rank, idx in enumerate(order[:k].tolist(), start=1):
            rec = rows[int(idx)]
            extra = (extra_metrics_by_index or {}).get(int(idx), {})
            writer.writerow(
                {
                    "rank": int(rank),
                    "score": float(sims[int(idx)]),
                    "sample_index": int(rec.get("_sample_index", idx)),
                    "doc_id": str(rec.get("doc_id", "") or ""),
                    "page_id": str(rec.get("page_id", "") or ""),
                    "line_id": _to_int(rec.get("line_id", -1), -1),
                    "image_path": str(rec.get("_image_path_resolved", "") or ""),
                    "text": str(rec.get("_text_resolved", "") or ""),
                    "is_ground_truth": bool(int(idx) == int(gt_idx) and gt_idx >= 0),
                    "query_mode": str(query_desc.get("mode", "") or ""),
                    "query_text": str(query_desc.get("query_text", "") or ""),
                    "query_image_path": str(query_desc.get("query_image_path", "") or ""),
                    "cer_vs_gt_text": (extra.get("cer_vs_gt_text") if "cer_vs_gt_text" in extra else None),
                    "sem_text_vs_gt_text": (extra.get("sem_text_vs_gt_text") if "sem_text_vs_gt_text" in extra else None),
                    "sem_image_vs_gt_image": (extra.get("sem_image_vs_gt_image") if "sem_image_vs_gt_image" in extra else None),
                }
            )
    return str(out_path)


def _cosine_sim_np_ui(a: np.ndarray, b: np.ndarray) -> float:
    va = np.asarray(a, dtype=np.float32).reshape(-1)
    vb = np.asarray(b, dtype=np.float32).reshape(-1)
    if va.size == 0 or vb.size == 0 or va.shape[0] != vb.shape[0]:
        return float("nan")
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na <= 0.0 or nb <= 0.0:
        return float("nan")
    return float(np.dot(va, vb) / (na * nb))


def run_line_clip_dataset_debug_retrieval_ui(
    sample_image: Optional[np.ndarray],
    sample_ground_truth: str,
    query_text: str,
    query_mode: str,
    dataset_root: str,
    split_name: str,
    image_backbone_path: str,
    image_head_path: str,
    text_encoder_dir: str,
    text_head_path: str,
    device_pref: str,
    batch_size: int,
    text_max_length: int,
    l2_normalize: bool,
    top_k: int,
    sample_state: Dict[str, Any],
):
    gt_placeholder = _line_profile_placeholder("Load a random OCR sample or upload one.")
    if not (dataset_root or "").strip() or not (split_name or "").strip():
        return gt_placeholder, [], "{}", "", "Please scan/select an OCR dataset + split.", sample_state
    if not (image_backbone_path or "").strip() or not (text_encoder_dir or "").strip() or not (text_head_path or "").strip():
        return gt_placeholder, [], "{}", "", "Please select a complete line_clip model (image backbone/head + text encoder/head).", sample_state
    qmode = str(query_mode or "Text -> Image").strip().lower()
    use_text_query = qmode.startswith("text")
    use_image_to_text = (not use_text_query) and ("text" in qmode)
    if use_text_query and not str(query_text or "").strip():
        return gt_placeholder, [], "{}", "", "Text query is empty.", sample_state
    if (not use_text_query) and sample_image is None:
        return gt_placeholder, [], "{}", "", "No sample image available for image query.", sample_state

    try:
        corpus = _get_or_build_line_clip_debug_corpus_ui(
            dataset_root=dataset_root,
            split_name=split_name,
            image_backbone_path=image_backbone_path,
            image_head_path=image_head_path,
            text_encoder_dir=text_encoder_dir,
            text_head_path=text_head_path,
            device_pref=device_pref,
            batch_size=max(1, int(batch_size)),
            text_max_length=max(8, int(text_max_length)),
            l2_normalize=bool(l2_normalize),
        )
        rows = corpus["rows"]
        img_emb = np.asarray(corpus["image_embeddings"], dtype=np.float32)
        txt_emb = np.asarray(corpus.get("text_embeddings", np.zeros((0, 1), dtype=np.float32)), dtype=np.float32)
        runtime = _load_line_clip_dual_runtime_cached(
            image_backbone_path,
            image_head_path,
            text_encoder_dir,
            text_head_path,
            device_pref,
        )
        if img_emb.ndim != 2 or img_emb.shape[0] != len(rows):
            raise RuntimeError("Cached image embeddings are invalid.")
        if txt_emb.ndim != 2 or txt_emb.shape[0] != len(rows):
            raise RuntimeError("Cached text embeddings are invalid.")

        if use_text_query:
            qtxt = str(query_text or "")
            q_vec = _encode_line_clip_texts_ui(
                runtime,
                texts=[qtxt],
                batch_size=1,
                l2_normalize=bool(l2_normalize),
                text_max_length=max(8, int(text_max_length)),
            )[0]
            query_desc = {"mode": "text_to_image", "query_text": qtxt}
            bank = img_emb
        else:
            q_img = np.asarray(sample_image).astype(np.uint8)
            tmp_path = _persist_faiss_ui_query_crop(q_img)
            if not tmp_path:
                raise RuntimeError("Could not persist temporary query image.")
            q_vec = _encode_line_clip_image_crops_ui(
                runtime,
                image_paths=[tmp_path],
                batch_size=1,
                l2_normalize=bool(l2_normalize),
            )[0]
            query_desc = {
                "mode": ("image_to_text" if use_image_to_text else "image_to_image"),
                "query_image_path": tmp_path,
            }
            bank = txt_emb if use_image_to_text else img_emb

        sims = np.matmul(bank, np.asarray(q_vec, dtype=np.float32))
        order = np.argsort(-sims)
        k = max(1, min(int(top_k), 50))
        top_idx = [int(i) for i in order[:k].tolist()]

        gt_idx = _line_clip_debug_find_ground_truth_index(rows, sample_state if isinstance(sample_state, dict) else {})
        gt_image_path = ""
        gt_image = gt_placeholder
        if gt_idx >= 0:
            gt_image_path = str(rows[gt_idx].get("_image_path_resolved", "") or "")
        elif isinstance(sample_state, dict):
            gt_image_path = str(sample_state.get("image_path", "") or "")
        if gt_image_path:
            try:
                gt_image = _load_image_as_rgb_np_ui(gt_image_path)
            except Exception:
                gt_image = gt_placeholder

        gallery_items: List[Tuple[np.ndarray, str]] = []
        rank_of_gt = -1
        topk_extra_metrics_by_index: Dict[int, Dict[str, Any]] = {}
        gt_text_ref = str(sample_ground_truth or "")
        gt_img_vec: Optional[np.ndarray] = None
        gt_txt_vec: Optional[np.ndarray] = None
        for rank, idx in enumerate(order.tolist(), start=1):
            if int(idx) == int(gt_idx):
                rank_of_gt = int(rank)
                break
        if gt_idx >= 0 and gt_idx < len(rows):
            try:
                gt_img_vec = np.asarray(img_emb[int(gt_idx)], dtype=np.float32)
            except Exception:
                gt_img_vec = None
            try:
                gt_txt_vec = np.asarray(txt_emb[int(gt_idx)], dtype=np.float32)
            except Exception:
                gt_txt_vec = None
        elif use_text_query and gt_text_ref.strip():
            try:
                gt_txt_vec = _encode_line_clip_texts_ui(
                    runtime,
                    texts=[gt_text_ref],
                    batch_size=1,
                    l2_normalize=bool(l2_normalize),
                    text_max_length=max(8, int(text_max_length)),
                )[0]
            except Exception:
                gt_txt_vec = None
        for rank, idx in enumerate(top_idx, start=1):
            rec = rows[idx]
            p = str(rec.get("_image_path_resolved", "") or "")
            try:
                img_np = _load_image_as_rgb_np_ui(p)
            except Exception:
                img_np = _line_profile_placeholder("Could not load result image.")
            txt = str(rec.get("_text_resolved", "") or "")
            is_gt = (idx == gt_idx and gt_idx >= 0)
            extra_lines: List[str] = []
            if use_text_query:
                extra: Dict[str, Any] = {}
                if gt_text_ref:
                    cer_val = _cer_single_ui(txt, gt_text_ref)
                    extra["cer_vs_gt_text"] = float(cer_val)
                    extra_lines.append(f"CERvsGT={cer_val:.4f}")
                if gt_txt_vec is not None and idx < len(txt_emb):
                    sem_txt = _cosine_sim_np_ui(np.asarray(txt_emb[idx], dtype=np.float32), gt_txt_vec)
                    if math.isfinite(sem_txt):
                        extra["sem_text_vs_gt_text"] = float(sem_txt)
                        extra_lines.append(f"txt~GTtxt={sem_txt:.4f}")
                if gt_img_vec is not None and idx < len(img_emb):
                    sem_img = _cosine_sim_np_ui(np.asarray(img_emb[idx], dtype=np.float32), gt_img_vec)
                    if math.isfinite(sem_img):
                        extra["sem_image_vs_gt_image"] = float(sem_img)
                        extra_lines.append(f"img~GTimg={sem_img:.4f}")
                if extra:
                    topk_extra_metrics_by_index[int(idx)] = extra
            caption = (
                f"#{rank} sim={float(sims[idx]):.4f}"
                + (" [GT]" if is_gt else "")
                + f"\npage={str(rec.get('page_id', ''))} line={_to_int(rec.get('line_id', -1), -1)}"
                + (f"\n{' | '.join(extra_lines)}" if extra_lines else "")
                + (f"\n{textwrap.shorten(txt, width=120, placeholder='...')}" if 'textwrap' in globals() else f"\n{txt[:120]}")
            )
            gallery_items.append((img_np, caption))

        export_csv_path = _write_line_clip_debug_results_csv_ui(
            rows=rows,
            order=order,
            sims=np.asarray(sims, dtype=np.float32),
            top_k=k,
            gt_idx=gt_idx,
            query_desc=query_desc,
            extra_metrics_by_index=topk_extra_metrics_by_index,
        )
        debug = {
            "ok": True,
            "query": query_desc,
            "dataset_root": str(Path(dataset_root).expanduser().resolve()),
            "split": str(split_name),
            "corpus_count": int(len(rows)),
            "top_k": int(k),
            "ground_truth_index": int(gt_idx),
            "ground_truth_rank_in_full_retrieval": int(rank_of_gt),
            "ground_truth_image_path": gt_image_path,
            "ground_truth_text_length": len(str(sample_ground_truth or "")),
            "device": str(corpus.get("device", "")),
            "device_msg": str(corpus.get("device_msg", "")),
            "cache_source": str(corpus.get("cache_source", "")),
            "disk_cache_dir": str(corpus.get("disk_cache_dir", "")),
            "results_csv_path": export_csv_path,
            "topk_text_to_image_metrics": (topk_extra_metrics_by_index if use_text_query else {}),
        }
        status = (
            f"line_clip debug retrieval ({'Text->Image' if use_text_query else ('Image->Text' if use_image_to_text else 'Image->Image')}) "
            f"on split={split_name}: corpus={len(rows)} topk={k} "
            f"| GT rank={rank_of_gt if rank_of_gt > 0 else 'not found'}"
            f" | CSV={export_csv_path}"
        )
        return gt_image, gallery_items, json.dumps(debug, ensure_ascii=False, indent=2), export_csv_path, status, sample_state
    except Exception as exc:
        msg = f"Line CLIP debug retrieval failed: {type(exc).__name__}: {exc}"
        return gt_placeholder, [], json.dumps({"ok": False, "error": msg}, ensure_ascii=False, indent=2), "", msg, sample_state


def refresh_image_list(dataset_dir: str, split: str):
    split_images = Path(dataset_dir).expanduser().resolve() / split / "images"
    images = _list_images(split_images)
    value = images[0] if images else None
    return gr.update(choices=images, value=value), f"{len(images)} image(s) found in {split_images}"


def preview_sample(dataset_dir: str, split: str, image_name: str):
    if not dataset_dir or not image_name:
        return None, "Select dataset/split/image first."
    image_path = Path(dataset_dir).expanduser().resolve() / split / "images" / image_name
    label_path = Path(dataset_dir).expanduser().resolve() / split / "labels" / f"{Path(image_name).stem}.txt"
    if not image_path.exists():
        return None, f"Image not found: {image_path}"
    rendered, summary = _draw_yolo_boxes(image_path, label_path)
    return rendered, summary


def preview_adjacent_sample(dataset_dir: str, split: str, current_image: str, step: int):
    split_images = Path(dataset_dir).expanduser().resolve() / split / "images"
    images = _list_images(split_images)
    if not images:
        return gr.update(choices=[], value=None), None, "No images found."

    if current_image in images:
        idx = images.index(current_image)
    else:
        idx = 0

    next_idx = (idx + int(step)) % len(images)
    next_image = images[next_idx]
    rendered, summary = preview_sample(dataset_dir, split, next_image)
    return gr.update(choices=images, value=next_image), rendered, summary


def _list_text_hierarchy_assets(root: Path, subset: str) -> List[Path]:
    subset_name = (subset or "").strip()
    if not root.exists():
        return []
    def _sorted_by_mtime_desc(paths: List[Path]) -> List[Path]:
        return sorted(
            paths,
            key=lambda p: float(p.stat().st_mtime) if p.exists() else 0.0,
            reverse=True,
        )
    if subset_name == "PatchDataset (all)":
        return sorted((root / "patches").rglob("patch_*.png"))
    if subset_name.startswith("PatchDataset (scale=") and subset_name.endswith(")"):
        wanted = subset_name[len("PatchDataset (scale=") : -1].strip()
        if wanted:
            return sorted((root / "patches").rglob(f"scale={wanted}/patch_*.png"))
    if subset_name == "PatchDataset (debug)":
        return _sorted_by_mtime_desc(list((root / "debug").rglob("*.png")))
    if subset_name == "Legacy TextHierarchy (line.png)":
        return sorted((root / "TextHierarchy").rglob("line_*/line.png"))
    if subset_name == "Legacy TextHierarchy (word crops)":
        return sorted((root / "TextHierarchy").rglob("line_*/level_*/word_*.png"))
    if subset_name == "Legacy NumberCrops (tibetan_number_word)":
        return sorted((root / "NumberCrops" / "tibetan_number_word").rglob("*"))
    if subset_name == "Legacy NumberCrops (chinese_number_word)":
        return sorted((root / "NumberCrops" / "chinese_number_word").rglob("*"))
    return _list_images_recursive(root)


def scan_text_hierarchy_assets(root_dir: str, subset: str):
    root = Path(root_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return gr.update(choices=[], value=None), f"Directory not found: {root}"

    assets = [p for p in _list_text_hierarchy_assets(root, subset) if p.is_file()]
    rels: List[str] = []
    for p in assets:
        try:
            rels.append(str(p.relative_to(root)))
        except Exception:
            rels.append(str(p))

    msg = f"Found {len(rels)} asset(s) in {root} for subset '{subset}'."
    return gr.update(choices=rels, value=(rels[0] if rels else None)), msg


def _sanitize_patch_id(value: Any) -> str:
    txt = str(value or "").strip()
    if not txt:
        return "unknown"
    out: List[str] = []
    for ch in txt:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "unknown"


def _resolve_patch_meta_path(root: Path) -> Optional[Path]:
    meta = root / "meta" / "patches.parquet"
    if meta.exists() and meta.is_file():
        return meta
    return None


def _resolve_patch_path_from_meta(root: Path, row: Dict[str, Any]) -> Optional[Path]:
    for path_col in ("patch_path", "patch_image_path"):
        raw = str(row.get(path_col, "") or "").strip()
        if raw:
            p = Path(raw).expanduser()
            if p.is_absolute():
                resolved_abs = p.resolve()
                if resolved_abs.exists():
                    return resolved_abs
            else:
                resolved_rel = (root / p).resolve()
                if resolved_rel.exists():
                    return resolved_rel
    try:
        patch_id = int(row.get("patch_id", -1))
        line_id = int(row.get("line_id", -1))
        scale_w = int(row.get("scale_w", -1))
        doc = _sanitize_patch_id(row.get("doc_id", ""))
        page = _sanitize_patch_id(row.get("page_id", ""))
    except Exception:
        return None
    if patch_id < 0 or line_id < 0 or scale_w <= 0:
        return None
    return (
        root
        / "patches"
        / f"doc={doc}"
        / f"page={page}"
        / f"line={line_id}"
        / f"scale={scale_w}"
        / f"patch_{patch_id}.png"
    ).resolve()


@lru_cache(maxsize=6)
def _load_patch_meta_lookup_cached(meta_path: str, root_dir: str, mtime_ns: int) -> Dict[str, Dict[str, Any]]:
    _ = mtime_ns
    if pd is None:
        return {}
    root = Path(root_dir).expanduser().resolve()
    mp = Path(meta_path).expanduser().resolve()
    if not mp.exists() or not mp.is_file():
        return {}
    try:
        df = pd.read_parquet(mp)
    except Exception:
        return {}
    if df is None or getattr(df, "empty", True):
        return {}

    keep_cols = [
        "patch_id",
        "doc_id",
        "page_id",
        "line_id",
        "scale_w",
        "k",
        "x0_norm",
        "x1_norm",
        "line_h_px",
        "line_w_px",
        "boundary_score",
        "ink_ratio",
        "source_img_path",
        "line_x0",
        "line_y0",
        "line_x1",
        "line_y1",
        "x0_px",
        "x1_px",
        "tag",
        "patch_path",
        "patch_image_path",
    ]
    cols = [c for c in keep_cols if c in df.columns]
    if cols:
        df = df[cols]

    lookup: Dict[str, Dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        resolved = _resolve_patch_path_from_meta(root, row)
        if resolved is None:
            continue
        rec: Dict[str, Any] = {}
        for k, v in row.items():
            if pd.isna(v):
                rec[k] = None
            elif isinstance(v, (np.generic,)):
                rec[k] = v.item()
            else:
                rec[k] = v
        rec["resolved_patch_path"] = str(resolved)
        lookup[str(resolved)] = rec
    return lookup


def _lookup_patch_meta_record(root: Path, asset_path: Path) -> Optional[Dict[str, Any]]:
    meta_path = _resolve_patch_meta_path(root)
    if meta_path is None:
        return None
    try:
        mtime_ns = int(meta_path.stat().st_mtime_ns)
    except Exception:
        mtime_ns = 0
    lookup = _load_patch_meta_lookup_cached(str(meta_path), str(root), int(mtime_ns))
    return lookup.get(str(asset_path.resolve()))


def _resolve_mnn_pairs_meta_path(root: Path) -> Optional[Path]:
    p = (root / "meta" / "mnn_pairs.parquet").resolve()
    if p.exists() and p.is_file():
        return p
    return None


@lru_cache(maxsize=4)
def _load_mnn_patch_match_index_cached(
    root_dir: str,
    patch_meta_path: str,
    patch_meta_mtime_ns: int,
    mnn_pairs_path: str,
    mnn_pairs_mtime_ns: int,
) -> Dict[str, Any]:
    _ = patch_meta_mtime_ns, mnn_pairs_mtime_ns
    if pd is None:
        return {"patches": {}, "neighbors": {}, "rows": [], "error": "pandas not installed"}

    root = Path(root_dir).expanduser().resolve()
    patch_meta = Path(patch_meta_path).expanduser().resolve()
    mnn_pairs = Path(mnn_pairs_path).expanduser().resolve()
    if not patch_meta.exists() or not patch_meta.is_file():
        return {"patches": {}, "neighbors": {}, "rows": [], "error": f"patches parquet not found: {patch_meta}"}
    if not mnn_pairs.exists() or not mnn_pairs.is_file():
        return {"patches": {}, "neighbors": {}, "rows": [], "error": f"mnn pairs parquet not found: {mnn_pairs}"}

    try:
        patch_df = pd.read_parquet(patch_meta)
    except Exception as exc:
        return {"patches": {}, "neighbors": {}, "rows": [], "error": f"Failed to read patches parquet: {type(exc).__name__}: {exc}"}
    try:
        pairs_df = pd.read_parquet(mnn_pairs)
    except Exception as exc:
        return {"patches": {}, "neighbors": {}, "rows": [], "error": f"Failed to read mnn pairs parquet: {type(exc).__name__}: {exc}"}

    if patch_df is None or getattr(patch_df, "empty", True):
        return {"patches": {}, "neighbors": {}, "rows": [], "error": "patches parquet is empty"}
    if pairs_df is None or getattr(pairs_df, "empty", True):
        return {"patches": {}, "neighbors": {}, "rows": [], "error": "mnn pairs parquet is empty"}

    patch_cols_keep = [
        "patch_id",
        "doc_id",
        "page_id",
        "line_id",
        "scale_w",
        "k",
        "x0_norm",
        "x1_norm",
        "ink_ratio",
        "boundary_score",
        "patch_path",
        "patch_image_path",
    ]
    pair_cols_keep = [
        "src_patch_id",
        "dst_patch_id",
        "sim",
        "stability_ratio",
        "multi_scale_ok",
        "rank_src_to_dst",
        "rank_dst_to_src",
        "src_doc_id",
        "src_page_id",
        "src_line_id",
        "src_scale_w",
        "dst_doc_id",
        "dst_page_id",
        "dst_line_id",
        "dst_scale_w",
    ]
    patch_cols = [c for c in patch_cols_keep if c in patch_df.columns]
    if patch_cols:
        patch_df = patch_df[patch_cols]
    pair_cols = [c for c in pair_cols_keep if c in pairs_df.columns]
    if pair_cols:
        pairs_df = pairs_df[pair_cols]

    patches: Dict[int, Dict[str, Any]] = {}
    for row in patch_df.to_dict(orient="records"):
        try:
            pid = int(row.get("patch_id", -1))
        except Exception:
            continue
        if pid < 0:
            continue
        resolved_path = _resolve_patch_path_from_meta(root, row)
        rec: Dict[str, Any] = {}
        for k, v in row.items():
            try:
                is_na = bool(pd.isna(v))
            except Exception:
                is_na = False
            if is_na:
                rec[k] = None
            elif isinstance(v, np.generic):
                rec[k] = v.item()
            else:
                rec[k] = v
        rec["patch_id"] = int(pid)
        rec["resolved_patch_path"] = str(resolved_path) if resolved_path is not None else ""
        # Prefer a record with a resolvable patch image path.
        prev = patches.get(pid)
        if prev is None or (not str(prev.get("resolved_patch_path", "")).strip() and rec["resolved_patch_path"]):
            patches[pid] = rec

    neighbors_map: Dict[int, Dict[int, Dict[str, Any]]] = {}

    def _safe_float(x: Any, default: float = 0.0) -> float:
        try:
            y = float(x)
            return y if math.isfinite(y) else float(default)
        except Exception:
            return float(default)

    def _safe_int_or_none(x: Any) -> Optional[int]:
        try:
            return int(x)
        except Exception:
            return None

    def _coerce_scalar(v: Any) -> Any:
        if isinstance(v, np.generic):
            return v.item()
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        return str(v)

    for row in pairs_df.to_dict(orient="records"):
        try:
            src = int(row.get("src_patch_id", -1))
            dst = int(row.get("dst_patch_id", -1))
        except Exception:
            continue
        if src < 0 or dst < 0 or src == dst:
            continue
        sim = _safe_float(row.get("sim", 0.0), 0.0)
        stability = _safe_float(row.get("stability_ratio", 0.0), 0.0)
        rank_sd = _safe_int_or_none(row.get("rank_src_to_dst"))
        rank_ds = _safe_int_or_none(row.get("rank_dst_to_src"))
        multi_ok = bool(row.get("multi_scale_ok", False))

        shared = {
            "sim": float(sim),
            "stability_ratio": float(stability),
            "multi_scale_ok": bool(multi_ok),
            "rank_src_to_dst": rank_sd,
            "rank_dst_to_src": rank_ds,
        }

        rec_fwd = dict(shared)
        rec_fwd.update(
            {
                "src_patch_id": int(src),
                "dst_patch_id": int(dst),
                "direction": "src_to_dst",
                "src_doc_id": _coerce_scalar(row.get("src_doc_id")),
                "src_page_id": _coerce_scalar(row.get("src_page_id")),
                "src_line_id": _safe_int_or_none(row.get("src_line_id")),
                "src_scale_w": _safe_int_or_none(row.get("src_scale_w")),
                "dst_doc_id": _coerce_scalar(row.get("dst_doc_id")),
                "dst_page_id": _coerce_scalar(row.get("dst_page_id")),
                "dst_line_id": _safe_int_or_none(row.get("dst_line_id")),
                "dst_scale_w": _safe_int_or_none(row.get("dst_scale_w")),
            }
        )
        rec_rev = dict(shared)
        rec_rev.update(
            {
                "src_patch_id": int(dst),
                "dst_patch_id": int(src),
                "direction": "dst_to_src",
                "src_doc_id": _coerce_scalar(row.get("dst_doc_id")),
                "src_page_id": _coerce_scalar(row.get("dst_page_id")),
                "src_line_id": _safe_int_or_none(row.get("dst_line_id")),
                "src_scale_w": _safe_int_or_none(row.get("dst_scale_w")),
                "dst_doc_id": _coerce_scalar(row.get("src_doc_id")),
                "dst_page_id": _coerce_scalar(row.get("src_page_id")),
                "dst_line_id": _safe_int_or_none(row.get("src_line_id")),
                "dst_scale_w": _safe_int_or_none(row.get("src_scale_w")),
                "rank_src_to_dst": rank_ds,
                "rank_dst_to_src": rank_sd,
            }
        )

        for a, b, rec in ((src, dst, rec_fwd), (dst, src, rec_rev)):
            bucket = neighbors_map.setdefault(int(a), {})
            prev = bucket.get(int(b))
            if prev is None:
                bucket[int(b)] = rec
                continue
            prev_score = (_safe_float(prev.get("sim", 0.0)), _safe_float(prev.get("stability_ratio", 0.0)))
            new_score = (_safe_float(rec.get("sim", 0.0)), _safe_float(rec.get("stability_ratio", 0.0)))
            if new_score > prev_score:
                bucket[int(b)] = rec

    neighbors: Dict[int, List[Dict[str, Any]]] = {}
    rows: List[Dict[str, Any]] = []
    for src_pid in sorted(neighbors_map.keys()):
        items = list(neighbors_map[src_pid].values())
        items_sorted = sorted(
            items,
            key=lambda r: (
                float(_safe_float(r.get("sim", 0.0))),
                float(_safe_float(r.get("stability_ratio", 0.0))),
                -int(_safe_int_or_none(r.get("rank_src_to_dst")) or 999999),
            ),
            reverse=True,
        )
        neighbors[src_pid] = items_sorted
        patch_rec = patches.get(int(src_pid), {})
        rows.append(
            {
                "patch_id": int(src_pid),
                "match_count": int(len(items_sorted)),
                "doc_id": patch_rec.get("doc_id", ""),
                "page_id": patch_rec.get("page_id", ""),
                "line_id": patch_rec.get("line_id", None),
                "scale_w": patch_rec.get("scale_w", None),
                "k": patch_rec.get("k", None),
                "ink_ratio": patch_rec.get("ink_ratio", None),
                "boundary_score": patch_rec.get("boundary_score", None),
                "patch_path": patch_rec.get("resolved_patch_path", ""),
            }
        )

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            int(r.get("match_count", 0)),
            float(_safe_float(r.get("ink_ratio", 0.0))),
            int(r.get("patch_id", -1)),
        ),
        reverse=True,
    )
    return {"patches": patches, "neighbors": neighbors, "rows": rows_sorted, "error": ""}


def _load_mnn_patch_match_index(root_dir: str, patch_meta_path: str, mnn_pairs_path: str) -> Dict[str, Any]:
    root = Path(root_dir).expanduser().resolve()
    patch_meta = Path(patch_meta_path).expanduser().resolve()
    mnn_pairs = Path(mnn_pairs_path).expanduser().resolve()
    pm_mtime = int(patch_meta.stat().st_mtime_ns) if patch_meta.exists() else 0
    mp_mtime = int(mnn_pairs.stat().st_mtime_ns) if mnn_pairs.exists() else 0
    return _load_mnn_patch_match_index_cached(str(root), str(patch_meta), pm_mtime, str(mnn_pairs), mp_mtime)


def scan_mnn_patch_matches_ui(root_dir: str, patch_meta_path: str, mnn_pairs_path: str, min_match_count: float, max_list_rows: float):
    root = Path(root_dir or "").expanduser().resolve() if str(root_dir or "").strip() else None
    if root is None or not root.exists() or not root.is_dir():
        return (
            gr.update(choices=[], value=None),
            [],
            f"Dataset root not found: {root}",
            "{}",
        )
    if pd is None:
        return gr.update(choices=[], value=None), [], "pandas is not installed.", "{}"

    patch_meta = Path(patch_meta_path).expanduser().resolve() if str(patch_meta_path or "").strip() else (_resolve_patch_meta_path(root) or Path(""))
    mnn_pairs = Path(mnn_pairs_path).expanduser().resolve() if str(mnn_pairs_path or "").strip() else (_resolve_mnn_pairs_meta_path(root) or Path(""))

    if not patch_meta.exists():
        return gr.update(choices=[], value=None), [], f"patches.parquet not found: {patch_meta}", "{}"
    if not mnn_pairs.exists():
        return gr.update(choices=[], value=None), [], f"mnn_pairs.parquet not found: {mnn_pairs}", "{}"

    payload = _load_mnn_patch_match_index(str(root), str(patch_meta), str(mnn_pairs))
    err = str(payload.get("error", "") or "").strip()
    if err:
        return gr.update(choices=[], value=None), [], err, "{}"

    rows_all = list(payload.get("rows", []))
    min_count = max(0, int(min_match_count or 0))
    rows_filtered = [r for r in rows_all if int(r.get("match_count", 0)) >= min_count]
    limit = max(1, int(max_list_rows or 200))
    rows_show = rows_filtered[:limit]

    choices: List[str] = []
    for r in rows_show:
        choices.append(
            f"patch_id={int(r.get('patch_id', -1))} | matches={int(r.get('match_count', 0))} | "
            f"doc={r.get('doc_id', '')} | page={r.get('page_id', '')} | line={r.get('line_id', '')} | "
            f"scale={r.get('scale_w', '')} | k={r.get('k', '')}"
        )

    df_rows = []
    for r in rows_show:
        df_rows.append(
            [
                int(r.get("patch_id", -1)),
                int(r.get("match_count", 0)),
                str(r.get("doc_id", "")),
                str(r.get("page_id", "")),
                ("" if r.get("line_id", None) is None else int(r.get("line_id"))),
                ("" if r.get("scale_w", None) is None else int(r.get("scale_w"))),
                ("" if r.get("k", None) is None else int(r.get("k"))),
                ("" if r.get("ink_ratio", None) is None else round(float(r.get("ink_ratio")), 4)),
                ("" if r.get("boundary_score", None) is None else round(float(r.get("boundary_score")), 4)),
            ]
        )

    summary = {
        "dataset_root": str(root),
        "patch_meta_path": str(patch_meta),
        "mnn_pairs_path": str(mnn_pairs),
        "rows_total_with_matches": int(len(rows_all)),
        "rows_after_min_match_count": int(len(rows_filtered)),
        "rows_shown": int(len(rows_show)),
        "min_match_count": int(min_count),
    }
    status = (
        f"Loaded MNN viewer index: {len(rows_all)} source patches with matches. "
        f"Showing {len(rows_show)} (min_match_count={min_count}, max_list_rows={limit})."
    )
    return gr.update(choices=choices, value=(choices[0] if choices else None)), df_rows, status, json.dumps(summary, ensure_ascii=False, indent=2)


def preview_mnn_patch_matches_ui(root_dir: str, patch_meta_path: str, mnn_pairs_path: str, patch_choice: str, max_matches: float):
    if not root_dir or not patch_choice:
        return [], "Select dataset root and patch first.", "{}"
    m = re.search(r"patch_id=(\d+)", str(patch_choice))
    if m is None:
        return [], f"Could not parse patch_id from selection: {patch_choice}", "{}"
    patch_id = int(m.group(1))

    root = Path(root_dir).expanduser().resolve()
    patch_meta = Path(patch_meta_path).expanduser().resolve() if str(patch_meta_path or "").strip() else (_resolve_patch_meta_path(root) or Path(""))
    mnn_pairs = Path(mnn_pairs_path).expanduser().resolve() if str(mnn_pairs_path or "").strip() else (_resolve_mnn_pairs_meta_path(root) or Path(""))
    payload = _load_mnn_patch_match_index(str(root), str(patch_meta), str(mnn_pairs))
    err = str(payload.get("error", "") or "").strip()
    if err:
        return [], err, "{}"

    patches = payload.get("patches", {}) or {}
    neighbors = payload.get("neighbors", {}) or {}
    src_rec = patches.get(int(patch_id))
    if src_rec is None:
        return [], f"Patch {patch_id} not found in patches.parquet.", "{}"

    src_path = Path(str(src_rec.get("resolved_patch_path", "") or "")).expanduser()
    if not src_path.exists() or not src_path.is_file():
        return [], f"Source patch image not found for patch_id={patch_id}.", "{}"

    try:
        with Image.open(src_path) as im:
            src_img = np.array(im.convert("RGB"))
    except Exception as exc:
        return [], f"Failed to load source patch image: {type(exc).__name__}: {exc}", "{}"

    gallery_items: List[Tuple[np.ndarray, str]] = []
    gallery_items.append((src_img, f"SRC patch_id={patch_id}"))

    neigh_list = list(neighbors.get(int(patch_id), []))
    limit = max(0, int(max_matches or 0))
    if limit > 0:
        neigh_list = neigh_list[:limit]

    shown_neighbors: List[Dict[str, Any]] = []
    missing_count = 0
    for rec in neigh_list:
        dst_pid = int(rec.get("dst_patch_id", -1))
        dst_meta = patches.get(dst_pid)
        if dst_meta is None:
            missing_count += 1
            continue
        dst_path = Path(str(dst_meta.get("resolved_patch_path", "") or "")).expanduser()
        if not dst_path.exists() or not dst_path.is_file():
            missing_count += 1
            continue
        try:
            with Image.open(dst_path) as im:
                dst_img = np.array(im.convert("RGB"))
        except Exception:
            missing_count += 1
            continue

        sim = rec.get("sim", None)
        stability = rec.get("stability_ratio", None)
        caption = (
            f"dst={dst_pid} | sim={float(sim):.3f} | stab={float(stability):.2f}"
            if sim is not None and stability is not None
            else f"dst={dst_pid}"
        )
        gallery_items.append((dst_img, caption))
        shown_neighbors.append(
            {
                "dst_patch_id": int(dst_pid),
                "sim": (None if sim is None else float(sim)),
                "stability_ratio": (None if stability is None else float(stability)),
                "multi_scale_ok": bool(rec.get("multi_scale_ok", False)),
                "rank_src_to_dst": rec.get("rank_src_to_dst"),
                "rank_dst_to_src": rec.get("rank_dst_to_src"),
                "dst_doc_id": dst_meta.get("doc_id", rec.get("dst_doc_id")),
                "dst_page_id": dst_meta.get("page_id", rec.get("dst_page_id")),
                "dst_line_id": dst_meta.get("line_id", rec.get("dst_line_id")),
                "dst_scale_w": dst_meta.get("scale_w", rec.get("dst_scale_w")),
                "dst_k": dst_meta.get("k", None),
                "dst_patch_path": str(dst_path),
            }
        )

    status = (
        f"patch_id={patch_id}: showing {len(shown_neighbors)} MNN matches"
        f"{f' (requested limit={limit})' if limit > 0 else ''}"
        f"{f', missing/unreadable={missing_count}' if missing_count > 0 else ''}."
    )
    meta_json = {
        "source_patch": {
            "patch_id": int(patch_id),
            "doc_id": src_rec.get("doc_id"),
            "page_id": src_rec.get("page_id"),
            "line_id": src_rec.get("line_id"),
            "scale_w": src_rec.get("scale_w"),
            "k": src_rec.get("k"),
            "patch_path": str(src_path),
        },
        "match_count_total": int(len(neighbors.get(int(patch_id), []))),
        "match_count_shown": int(len(shown_neighbors)),
        "matches": shown_neighbors,
    }
    return gallery_items, status, json.dumps(meta_json, ensure_ascii=False, indent=2)


def _extract_first_int_suffix(name: str, prefix: str) -> Optional[int]:
    if not name.startswith(prefix):
        return None
    tail = name[len(prefix) :]
    digits = "".join(ch for ch in tail if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except Exception:
        return None


def _find_text_hierarchy_meta_path(root: Path, asset_path: Path) -> Optional[Path]:
    current = asset_path.parent
    while True:
        candidate = current / "meta.json"
        if candidate.exists() and candidate.is_file():
            return candidate
        if current == root or current.parent == current:
            break
        current = current.parent
    return None


def preview_text_hierarchy_asset(root_dir: str, asset_rel_path: str):
    if not root_dir or not asset_rel_path:
        return None, "Select a root directory and an asset.", "{}"

    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        return None, f"Directory not found: {root}", "{}"

    asset_path = (root / asset_rel_path).expanduser().resolve()
    try:
        asset_path.relative_to(root)
    except Exception:
        return None, f"Asset is outside root: {asset_path}", "{}"
    if not asset_path.exists() or not asset_path.is_file():
        return None, f"Asset not found: {asset_path}", "{}"

    try:
        with Image.open(asset_path) as im:
            image_np = np.array(im.convert("RGB"))
    except Exception as exc:
        return None, f"Failed to load image: {type(exc).__name__}: {exc}", "{}"

    h, w = image_np.shape[:2]
    info_lines: List[str] = [
        f"asset: {asset_rel_path}",
        f"size: {w}x{h}",
    ]
    payload: Dict[str, Any] = {
        "asset": asset_rel_path,
        "size": {"width": int(w), "height": int(h)},
    }

    line_idx: Optional[int] = None
    level_target: Optional[int] = None
    word_idx: Optional[int] = None
    for part in asset_path.parts:
        if line_idx is None:
            line_idx = _extract_first_int_suffix(part, "line_")
        if level_target is None:
            level_target = _extract_first_int_suffix(part, "level_")
    if asset_path.stem.startswith("word_"):
        word_idx = _extract_first_int_suffix(asset_path.stem, "word_")

    if line_idx is not None:
        info_lines.append(f"line_index: {line_idx}")
        payload["line_index"] = int(line_idx)
    if level_target is not None:
        info_lines.append(f"level_target: {level_target}")
        payload["level_target"] = int(level_target)
    if word_idx is not None:
        info_lines.append(f"word_index: {word_idx}")
        payload["word_index"] = int(word_idx)

    patch_record = _lookup_patch_meta_record(root=root, asset_path=asset_path)
    if patch_record is not None:
        payload["patch_meta"] = patch_record
        info_lines.extend(
            [
                f"doc_id: {patch_record.get('doc_id', '')}",
                f"page_id: {patch_record.get('page_id', '')}",
                f"line_id: {patch_record.get('line_id', '')}",
                f"scale_w: {patch_record.get('scale_w', '')}",
                f"k: {patch_record.get('k', '')}",
                f"tag: {patch_record.get('tag', '')}",
                f"ink_ratio: {patch_record.get('ink_ratio', '')}",
                f"boundary_score: {patch_record.get('boundary_score', '')}",
            ]
        )
        payload["meta_source"] = "meta/patches.parquet"
    else:
        meta_path = _find_text_hierarchy_meta_path(root, asset_path)
        if meta_path is not None:
            try:
                meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
                try:
                    payload["meta_path"] = str(meta_path.relative_to(root))
                except Exception:
                    payload["meta_path"] = str(meta_path)
                payload["meta"] = meta_data
                lines = meta_data.get("lines") if isinstance(meta_data, dict) else None
                if isinstance(lines, list) and line_idx is not None:
                    line_record = next(
                        (x for x in lines if int(x.get("line_index", -1)) == int(line_idx) and isinstance(x, dict)),
                        None,
                    )
                    if line_record is not None:
                        payload["line_record"] = line_record
                        info_lines.append("line_record: found")
                        if level_target is not None:
                            levels = ((line_record.get("hierarchy") or {}).get("levels") or [])
                            level_record = next(
                                (
                                    lv
                                    for lv in levels
                                    if isinstance(lv, dict)
                                    and int(lv.get("target_count", lv.get("count", -1))) == int(level_target)
                                ),
                                None,
                            )
                            if level_record is not None:
                                payload["level_record"] = level_record
                                info_lines.append("level_record: found")
            except Exception as exc:
                info_lines.append(f"meta parse failed: {type(exc).__name__}")
                payload["meta_error"] = f"{type(exc).__name__}: {exc}"
        else:
            info_lines.append("metadata: not found (ok for loose image folders)")

    return image_np, "\n".join(info_lines), json.dumps(payload, ensure_ascii=False, indent=2)


def preview_adjacent_text_hierarchy_asset(root_dir: str, subset: str, current_asset: str, step: int):
    root = Path(root_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return gr.update(choices=[], value=None), None, f"Directory not found: {root}", "{}"

    assets = [p for p in _list_text_hierarchy_assets(root, subset) if p.is_file()]
    if not assets:
        return gr.update(choices=[], value=None), None, "No assets found.", "{}"

    rels: List[str] = []
    for p in assets:
        try:
            rels.append(str(p.relative_to(root)))
        except Exception:
            rels.append(str(p))

    if current_asset in rels:
        idx = rels.index(current_asset)
    else:
        idx = 0
    next_idx = (idx + int(step)) % len(rels)
    next_asset = rels[next_idx]
    img, info, payload = preview_text_hierarchy_asset(root_dir, next_asset)
    return gr.update(choices=rels, value=next_asset), img, info, payload


def export_to_label_studio(
    split_dir: str,
    image_ext: str,
    tasks_json: str,
    image_root_url: str,
):
    split_path = Path(split_dir).expanduser().resolve()
    out_path = Path(tasks_json).expanduser().resolve()
    cmd = [
        "label-studio-converter",
        "import",
        "yolo",
        "-i",
        str(split_path),
        "-o",
        str(out_path),
        "--image-ext",
        image_ext,
        "--image-root-url",
        image_root_url,
    ]
    ok, out = _run_cmd(cmd, timeout=1200)
    status = "Success" if ok else "Failed"
    env_hint = (
        "Set these env vars before starting Label Studio:\n"
        f"export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true\n"
        f"export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={split_path.parent.resolve()}\n"
    )
    return f"{status}\nTasks file: {out_path}\n\n{env_hint}\n{out}"


def start_label_studio(local_files_root: str):
    env = os.environ.copy()
    env["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
    env["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = str(Path(local_files_root).expanduser().resolve())
    try:
        proc = subprocess.Popen(
            ["label-studio"],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return f"Label Studio started (PID {proc.pid}). Local files root: {env['LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT']}"
    except FileNotFoundError:
        return "label-studio command not found. Install with: pip install label-studio"
    except Exception as exc:
        return f"Failed to start Label Studio: {type(exc).__name__}: {exc}"


def _format_vlm_parser_choices() -> List[str]:
    try:
        from tibetan_utils.parsers import list_parser_specs
        specs = {s.key: s for s in list_parser_specs()}
    except Exception:
        specs = {}

    labels: List[str] = []
    for key in TRANSFORMER_PARSERS:
        spec = specs.get(key)
        hint = VLM_VRAM_HINTS.get(key, "unknown")
        if spec is None:
            labels.append(f"{key} (VRAM: {hint})")
        else:
            labels.append(f"{key} - {spec.display_name} (VRAM: {hint})")
    return labels


def _extract_parser_key(parser_choice: str) -> str:
    return parser_choice.split(" - ", 1)[0].strip()


def _vlm_backend_status_markdown() -> str:
    lines = ["### VLM Backend Status"]
    try:
        from tibetan_utils.parsers import parser_availability
        for key in TRANSFORMER_PARSERS:
            ok, reason = parser_availability(key)
            status = "OK" if ok else "N/A"
            lines.append(f"- `{key}`: **{status}** ({reason})")
    except Exception as exc:
        lines.append(f"- status unavailable: {type(exc).__name__}: {exc}")
    return "\n".join(lines)


@lru_cache(maxsize=8)
def _build_vlm_backend(
    parser_key: str,
    hf_model_id: str,
    prompt: str,
    max_new_tokens: int,
    hf_device: str,
    hf_dtype: str,
    mineru_command: str,
    mineru_timeout: int,
):
    from tibetan_utils.parsers import create_parser

    if parser_key == "mineru25":
        return create_parser(
            "mineru25",
            mineru_command=mineru_command,
            timeout_sec=mineru_timeout,
        )

    kwargs: Dict[str, Any] = {
        "prompt": prompt or None,
        "max_new_tokens": max_new_tokens,
        "hf_device": hf_device,
        "hf_dtype": hf_dtype,
    }
    if hf_model_id:
        kwargs["model_id"] = hf_model_id
    return create_parser(parser_key, **kwargs)


def _to_xyxy(
    box: Dict[str, float],
    image_width: int,
    image_height: int
) -> Optional[Tuple[int, int, int, int]]:
    x = float(box.get("x", 0.0))
    y = float(box.get("y", 0.0))
    w = float(box.get("width", 0.0))
    h = float(box.get("height", 0.0))

    if w <= 0 or h <= 0:
        return None

    if max(abs(x), abs(y), abs(w), abs(h)) <= 1.5:
        x1 = int((x - w / 2.0) * image_width)
        y1 = int((y - h / 2.0) * image_height)
        x2 = int((x + w / 2.0) * image_width)
        y2 = int((y + h / 2.0) * image_height)
    elif x >= 0 and y >= 0 and w > x and h > y and w <= image_width and h <= image_height:
        x1 = int(x)
        y1 = int(y)
        x2 = int(w)
        y2 = int(h)
    else:
        x1 = int(x - w / 2.0)
        y1 = int(y - h / 2.0)
        x2 = int(x + w / 2.0)
        y2 = int(y + h / 2.0)

    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))
    x2 = max(0, min(image_width - 1, x2))
    y2 = max(0, min(image_height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _render_detected_regions(image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    base = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(base)
    font = _load_overlay_font()
    color_by_class = {0: (255, 120, 0), 1: (0, 200, 255), 2: (120, 255, 120)}

    width, height = base.size
    for det in detections:
        box = det.get("box") or {}
        xyxy = _to_xyxy(box, width, height)
        if xyxy is None:
            continue
        class_id = int(det.get("class", 1))
        color = color_by_class.get(class_id, (255, 80, 80))
        draw.rectangle(xyxy, outline=color, width=3)
        label = str(det.get("label") or f"class_{class_id}")
        conf = float(det.get("confidence", 0.0))
        tag = f"{label} ({conf:.2f})"
        tx, ty = xyxy[0], max(0, xyxy[1] - 16)
        try:
            bbox = draw.textbbox((tx + 2, ty + 1), tag, font=font)
            tw = max(12, int(bbox[2] - bbox[0]) + 4)
            th = max(12, int(bbox[3] - bbox[1]) + 2)
        except Exception:
            tw = max(12, 9 * len(tag))
            th = 14
        draw.rectangle((tx, ty, tx + tw, ty + th), fill=(0, 0, 0))
        draw.text((tx + 2, ty + 1), tag, fill=color, font=font)

    return np.array(base)


def _tail_lines_newest_first(lines: List[str], limit: int) -> str:
    if not lines:
        return ""
    return "\n".join(reversed(lines[-limit:]))


def run_vlm_layout_inference(
    image: np.ndarray,
    parser_choice: str,
    prompt: str,
    hf_model_id: str,
    max_new_tokens: int,
    hf_device: str,
    hf_dtype: str,
    mineru_command: str,
    mineru_timeout: int,
):
    if image is None:
        return None, "Please upload or paste an image.", "{}"

    parser_key = _extract_parser_key(parser_choice)
    try:
        from tibetan_utils.parsers import parser_availability
        ok, reason = parser_availability(parser_key)
        if not ok:
            return image, f"Backend `{parser_key}` unavailable: {reason}", "{}"
    except Exception as exc:
        return image, f"Availability check failed: {type(exc).__name__}: {exc}", "{}"

    try:
        backend = _build_vlm_backend(
            parser_key=parser_key,
            hf_model_id=(hf_model_id or "").strip(),
            prompt=(prompt or "").strip(),
            max_new_tokens=int(max_new_tokens),
            hf_device=(hf_device or "auto").strip(),
            hf_dtype=(hf_dtype or "auto").strip(),
            mineru_command=(mineru_command or "mineru").strip(),
            mineru_timeout=int(mineru_timeout),
        )
        doc = backend.parse(image, output_dir=None, image_name="ui_image.png")
        out = doc.to_dict()
        detections = out.get("detections", [])
        overlay = _render_detected_regions(image, detections)
        return overlay, f"{len(detections)} regions detected with `{parser_key}`.", json.dumps(out, ensure_ascii=False, indent=2)
    except Exception as exc:
        return image, f"Error: {type(exc).__name__}: {exc}", "{}"


def run_ultralytics_train(
    dataset: str,
    model: str,
    epochs: int,
    batch: int,
    imgsz: int,
    workers: int,
    device: str,
    project: str,
    name: str,
    patience: int,
    export: bool,
    wandb_enabled: bool,
    wandb_project: str,
    wandb_entity: str,
    wandb_tags: str,
    wandb_name: str,
):
    cmd = [
        sys.executable,
        "train_model.py",
        "--dataset",
        dataset,
        "--model",
        model,
        "--epochs",
        str(int(epochs)),
        "--batch",
        str(int(batch)),
        "--imgsz",
        str(int(imgsz)),
        "--workers",
        str(int(workers)),
        "--device",
        (device or "").strip(),
        "--project",
        project,
        "--name",
        name,
        "--patience",
        str(int(patience)),
    ]
    if export:
        cmd.append("--export")
    if wandb_enabled:
        cmd.append("--wandb")
        if wandb_project.strip():
            cmd.extend(["--wandb-project", wandb_project.strip()])
        if wandb_entity.strip():
            cmd.extend(["--wandb-entity", wandb_entity.strip()])
        if wandb_tags.strip():
            cmd.extend(["--wandb-tags", wandb_tags.strip()])
        if wandb_name.strip():
            cmd.extend(["--wandb-name", wandb_name.strip()])

    ok, out = _run_cmd(cmd, timeout=86400)
    status = "Success" if ok else "Failed"

    best_model = _find_latest_best_model(project=project, name=name)
    expected_best_model = _project_dir_abs(project) / name / "weights" / "best.pt"

    if ok and best_model is not None:
        copied_model, copy_msg = _archive_layout_best_model(
            best_model_path=best_model,
            dataset=dataset,
            model=model,
            name=name,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            patience=patience,
        )
        final_path = copied_model or best_model
        return (
            f"{status}\nBest model path: {best_model}\nArchived model path: {final_path}\n{copy_msg}\n\n{out}",
            str(final_path),
        )

    if ok and best_model is None:
        locate_msg = (
            f"Could not locate best.pt under project={_project_dir_abs(project)} name={name}. "
            "Check whether Ultralytics auto-incremented run naming."
        )
        return f"{status}\nExpected best model path: {expected_best_model}\n{locate_msg}\n\n{out}", str(expected_best_model)

    return f"{status}\nExpected best model path: {expected_best_model}\n\n{out}", str(expected_best_model)


def _build_ultralytics_train_cmd(
    dataset: str,
    model: str,
    epochs: int,
    batch: int,
    imgsz: int,
    workers: int,
    device: str,
    project: str,
    name: str,
    patience: int,
    export: bool,
    wandb_enabled: bool,
    wandb_project: str,
    wandb_entity: str,
    wandb_tags: str,
    wandb_name: str,
) -> List[str]:
    cmd = [
        sys.executable,
        "-u",
        "train_model.py",
        "--dataset",
        dataset,
        "--model",
        model,
        "--epochs",
        str(int(epochs)),
        "--batch",
        str(int(batch)),
        "--imgsz",
        str(int(imgsz)),
        "--workers",
        str(int(workers)),
        "--device",
        (device or "").strip(),
        "--project",
        project,
        "--name",
        name,
        "--patience",
        str(int(patience)),
    ]
    if export:
        cmd.append("--export")
    if wandb_enabled:
        cmd.append("--wandb")
        if wandb_project.strip():
            cmd.extend(["--wandb-project", wandb_project.strip()])
        if wandb_entity.strip():
            cmd.extend(["--wandb-entity", wandb_entity.strip()])
        if wandb_tags.strip():
            cmd.extend(["--wandb-tags", wandb_tags.strip()])
        if wandb_name.strip():
            cmd.extend(["--wandb-name", wandb_name.strip()])
    return cmd


def _sanitize_slug(value: str, default: str = "na", max_len: int = 48) -> str:
    txt = (value or "").strip()
    if not txt:
        return default
    txt = txt.replace(os.sep, "_")
    txt = re.sub(r"[^A-Za-z0-9._-]+", "-", txt)
    txt = re.sub(r"-{2,}", "-", txt).strip("-._")
    if not txt:
        return default
    return txt[:max_len]


def _project_dir_abs(project: str) -> Path:
    p = Path(project).expanduser()
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


def _find_latest_best_model(project: str, name: str) -> Optional[Path]:
    project_dir = _project_dir_abs(project)
    candidates: List[Path] = []

    exact = project_dir / name / "weights" / "best.pt"
    if exact.exists():
        candidates.append(exact)

    if project_dir.exists():
        for p in project_dir.glob(f"{name}*/weights/best.pt"):
            if p.is_file():
                candidates.append(p)

    if not candidates:
        return None

    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _extract_lr0_from_run_dir(run_dir: Optional[Path]) -> str:
    if run_dir is None:
        return "unknown"
    args_yaml = run_dir / "args.yaml"
    if not args_yaml.exists():
        return "unknown"
    try:
        for raw in args_yaml.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if line.startswith("lr0:"):
                val = line.split(":", 1)[1].strip()
                return _sanitize_slug(val, default="unknown", max_len=20)
    except Exception:
        return "unknown"
    return "unknown"


def _archive_layout_best_model(
    best_model_path: Path,
    dataset: str,
    model: str,
    name: str,
    epochs: int,
    batch: int,
    imgsz: int,
    patience: int,
) -> Tuple[Optional[Path], str]:
    if not best_model_path.exists():
        return None, f"best.pt not found at {best_model_path}"

    run_dir = best_model_path.parent.parent
    lr0 = _extract_lr0_from_run_dir(run_dir)
    dataset_tok = _sanitize_slug(Path(str(dataset)).stem if dataset else "", default="dataset")
    model_tok = _sanitize_slug(Path(str(model)).name, default="model")
    run_tok = _sanitize_slug(name, default="run")
    ts = time.strftime("%Y%m%d-%H%M%S")

    target_dir = (ROOT / "models" / "layoutModels").resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"layout_{dataset_tok}_{model_tok}_lr{lr0}_ep{int(epochs)}_bs{int(batch)}_"
        f"sz{int(imgsz)}_pat{int(patience)}_{run_tok}_{ts}.pt"
    )
    target = target_dir / filename
    shutil.copy2(best_model_path, target)
    return target, f"Copied best model to {target}"


def run_ultralytics_train_live(
    dataset: str,
    model: str,
    epochs: int,
    batch: int,
    imgsz: int,
    workers: int,
    device: str,
    project: str,
    name: str,
    patience: int,
    export: bool,
    wandb_enabled: bool,
    wandb_project: str,
    wandb_entity: str,
    wandb_tags: str,
    wandb_name: str,
):
    cmd = _build_ultralytics_train_cmd(
        dataset=dataset,
        model=model,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        workers=workers,
        device=device,
        project=project,
        name=name,
        patience=patience,
        export=export,
        wandb_enabled=wandb_enabled,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_tags=wandb_tags,
        wandb_name=wandb_name,
    )
    expected_best_model = _project_dir_abs(project) / name / "weights" / "best.pt"

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
    except Exception as exc:
        msg = f"Failed\nExpected best model path: {expected_best_model}\n\n{type(exc).__name__}: {exc}"
        yield msg, str(expected_best_model)
        return

    if proc.stdout is not None:
        try:
            os.set_blocking(proc.stdout.fileno(), False)
        except Exception:
            pass

    log_lines: List[str] = []
    partial = ""
    last_emit_ts = 0.0
    last_emit_count = 0
    stream_failed = False
    stream_fail_msg = ""

    yield f"Running ...\nExpected best model path: {expected_best_model}\n", str(expected_best_model)

    while True:
        got_output = False
        if (not stream_failed) and proc.stdout is not None:
            try:
                chunk = proc.stdout.read()
            except BlockingIOError:
                chunk = b""
            except Exception as exc:
                stream_failed = True
                stream_fail_msg = f"stdout stream disabled ({type(exc).__name__}: {exc})"
                chunk = b""

            if chunk:
                got_output = True
                if isinstance(chunk, bytes):
                    chunk_text = chunk.decode("utf-8", errors="replace")
                else:
                    chunk_text = str(chunk)
                chunk_text = chunk_text.replace("\r", "\n")
                partial += chunk_text
                parts = partial.splitlines(keepends=True)
                keep = ""
                for piece in parts:
                    if piece.endswith("\n"):
                        log_lines.append(piece.rstrip("\n"))
                    else:
                        keep = piece
                partial = keep

        now = time.time()
        # Emit in calmer batches to avoid aggressive UI re-renders/jumping.
        should_emit = (now - last_emit_ts >= 1.5)
        if should_emit:
            tail = _tail_lines_newest_first(log_lines, 800)
            if stream_failed and stream_fail_msg:
                tail = f"{tail}\n[warning] {stream_fail_msg}" if tail else f"[warning] {stream_fail_msg}"
            running_msg = f"Running ...\nExpected best model path: {expected_best_model}\n\n{tail}"
            yield running_msg, str(expected_best_model)
            last_emit_ts = now
            last_emit_count = len(log_lines)

        if proc.poll() is not None:
            break

        if not got_output:
            time.sleep(0.15)

    if proc.stdout is not None:
        try:
            rest = proc.stdout.read()
        except Exception:
            rest = b""
        if rest:
            if isinstance(rest, bytes):
                partial += rest.decode("utf-8", errors="replace").replace("\r", "\n")
            else:
                partial += str(rest).replace("\r", "\n")
        if partial:
            log_lines.extend(partial.splitlines())

    ok = proc.returncode == 0
    status = "Success" if ok else "Failed"
    best_model = _find_latest_best_model(project=project, name=name)
    if ok and best_model is not None:
        copied_model, copy_msg = _archive_layout_best_model(
            best_model_path=best_model,
            dataset=dataset,
            model=model,
            name=name,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            patience=patience,
        )
        final_path = copied_model or best_model
        final_msg = (
            f"{status}\nBest model path: {best_model}\nArchived model path: {final_path}\n{copy_msg}\n\n"
            + _tail_lines_newest_first(log_lines, 3000)
        )
        yield final_msg, str(final_path)
        return

    if ok and best_model is None:
        locate_msg = (
            f"Could not locate best.pt under project={_project_dir_abs(project)} name={name}. "
            "Check whether Ultralytics auto-incremented run naming."
        )
        final_msg = f"{status}\nExpected best model path: {expected_best_model}\n{locate_msg}\n\n" + _tail_lines_newest_first(log_lines, 3000)
        yield final_msg, str(expected_best_model)
        return

    final_msg = f"{status}\nExpected best model path: {expected_best_model}\n\n" + _tail_lines_newest_first(log_lines, 3000)
    yield final_msg, str(expected_best_model)


def _ultralytics_model_presets() -> List[str]:
    return [
        "yolo26n.pt",
        "yolo26s.pt",
        "yolo26m.pt",
        "yolo26l.pt",
        "yolo26x.pt",
    ]


def scan_ultralytics_models(models_dir: str):
    base = Path(models_dir).expanduser().resolve()
    presets = _ultralytics_model_presets()
    if not base.exists():
        return gr.update(choices=presets, value=presets[0]), f"Models directory not found: {base}. Using presets only."

    local_models = sorted([str(p.resolve()) for p in base.rglob("*.pt") if p.is_file()])
    choices = presets + [m for m in local_models if m not in presets]
    default = choices[0] if choices else None
    return gr.update(choices=choices, value=default), f"Found {len(local_models)} local .pt model(s) in {base} (+ presets)."


def resolve_train_model(model_choice: str, model_override: str) -> str:
    if model_override and model_override.strip():
        return model_override.strip()
    return (model_choice or "").strip()


def run_ultralytics_train_from_ui(
    dataset: str,
    model_choice: str,
    model_override: str,
    epochs: int,
    batch: int,
    imgsz: int,
    workers: int,
    device: str,
    project: str,
    name: str,
    patience: int,
    export: bool,
    wandb_enabled: bool,
    wandb_project: str,
    wandb_entity: str,
    wandb_tags: str,
    wandb_name: str,
):
    model = resolve_train_model(model_choice, model_override)
    yield from run_ultralytics_train_live(
        dataset=dataset,
        model=model,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        workers=workers,
        device=device,
        project=project,
        name=name,
        patience=patience,
        export=export,
        wandb_enabled=wandb_enabled,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_tags=wandb_tags,
        wandb_name=wandb_name,
    )


@lru_cache(maxsize=4)
def _load_yolo_model(model_path: str):
    from ultralytics import YOLO
    return YOLO(model_path)


def run_trained_model_inference(
    image: np.ndarray,
    model_path: str,
    conf: float,
    imgsz: int,
    device: str,
):
    if image is None:
        return None, "Please provide an image.", "[]"
    model_file = Path(model_path).expanduser().resolve()
    if not model_file.exists():
        return image, f"Model not found: {model_file}", "[]"

    try:
        model = _load_yolo_model(str(model_file))
        kwargs: Dict[str, Any] = {"conf": float(conf), "imgsz": int(imgsz)}
        if (device or "").strip():
            kwargs["device"] = (device or "").strip()

        results = model.predict(source=image, **kwargs)
        overlay = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        draw = ImageDraw.Draw(overlay)
        detections = []
        # Stable per-class colors for clearer overlays.
        base_class_colors = {
            0: (255, 140, 0),
            1: (0, 220, 255),
            2: (130, 255, 130),
            3: (255, 100, 180),
            4: (200, 180, 255),
            5: (255, 220, 120),
        }

        for res in results:
            if not hasattr(res, "boxes") or res.boxes is None:
                continue
            boxes = res.boxes
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else []
            confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else []
            clss = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else []
            names = getattr(res, "names", None) or getattr(model, "names", None) or {}
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
                c = float(confs[i]) if i < len(confs) else 0.0
                cls = int(clss[i]) if i < len(clss) else 0
                if isinstance(names, dict):
                    class_name = str(names.get(cls, f"class_{cls}"))
                elif isinstance(names, list) and 0 <= cls < len(names):
                    class_name = str(names[cls])
                else:
                    class_name = f"class_{cls}"
                detections.append({
                    "class": cls,
                    "label": class_name,
                    "confidence": c,
                    "box": [x1, y1, x2, y2],
                })
                if cls in base_class_colors:
                    color = base_class_colors[cls]
                else:
                    color = (
                        60 + ((cls * 67) % 180),
                        60 + ((cls * 97) % 180),
                        60 + ((cls * 131) % 180),
                    )
                draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
                tag = f"{class_name} ({cls}): {c:.2f}"
                tx, ty = x1 + 2, max(0, y1 - 16)
                draw.rectangle((tx, ty, tx + 9 * len(tag), ty + 14), fill=(0, 0, 0))
                draw.text((tx + 2, ty + 1), tag, fill=color)

        status = f"{len(detections)} detections"
        return np.array(overlay), status, json.dumps(detections, ensure_ascii=False, indent=2)
    except Exception as exc:
        return image, f"Inference failed: {type(exc).__name__}: {exc}", "[]"


def _normalize_layout_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (label or "").strip().lower())


def _is_tibetan_text_detection(class_id: int, label: str) -> bool:
    norm = _normalize_layout_label(label)
    if norm in {"tibetantext", "tibetantextbox", "tibetanlines", "tibetanline"}:
        return True
    if norm in {"", f"class{class_id}"} and class_id == 1:
        return True
    return False


def _line_profile_placeholder(message: str) -> np.ndarray:
    img = Image.new("RGB", (980, 320), (246, 246, 246))
    draw = ImageDraw.Draw(img)
    font = _load_overlay_font()
    draw.rectangle((1, 1, 978, 318), outline=(180, 180, 180), width=2)
    draw.text((24, 20), "Line Projection Profile", fill=(40, 40, 40), font=font)
    draw.text((24, 52), message, fill=(90, 90, 90), font=font)
    return np.array(img)


def _render_line_profile_plot(
    projection: np.ndarray,
    threshold: float,
    mask: np.ndarray,
    line_boxes_local: List[Tuple[int, int, int, int]],
    title: str,
    peaks: Optional[List[int]] = None,
) -> np.ndarray:
    if projection is None or projection.size == 0:
        return _line_profile_placeholder("No projection data available.")

    n = int(projection.size)
    plot_w, plot_h = 980, 320
    left, top, right, bottom = 56, 34, 960, 276
    inner_w = max(1, right - left)
    inner_h = max(1, bottom - top)

    img = Image.new("RGB", (plot_w, plot_h), (248, 248, 248))
    draw = ImageDraw.Draw(img)
    font = _load_overlay_font()

    draw.rectangle((left, top, right, bottom), outline=(140, 140, 140), width=2)

    def _x_from_row(row_idx: int) -> int:
        if n <= 1:
            return left
        t = float(max(0, min(n - 1, int(row_idx)))) / float(n - 1)
        return left + int(round(t * inner_w))

    if mask is not None and mask.size == n:
        i = 0
        while i < n:
            if not bool(mask[i]):
                i += 1
                continue
            j = i
            while j < n and bool(mask[j]):
                j += 1
            mx1 = _x_from_row(i)
            mx2 = _x_from_row(max(i, j - 1))
            if mx2 <= mx1:
                mx2 = mx1 + 1
            draw.rectangle((mx1, top + 1, mx2, bottom - 1), fill=(255, 247, 215))
            i = j

    for _, y1, _, y2 in line_boxes_local:
        lx1 = _x_from_row(int(y1))
        lx2 = _x_from_row(max(int(y1), int(y2) - 1))
        if lx2 <= lx1:
            lx2 = lx1 + 1
        draw.rectangle((lx1, top + 2, lx2, bottom - 2), outline=(255, 168, 0), width=2)

    for peak in (peaks or []):
        px = _x_from_row(int(peak))
        draw.line((px, top + 1, px, bottom - 1), fill=(46, 158, 78), width=1)
        draw.ellipse((px - 3, top + 3, px + 3, top + 9), fill=(46, 158, 78))

    vmax = max(1.0, float(np.max(projection)), float(threshold))
    threshold_y = top + int(round((1.0 - (min(vmax, max(0.0, float(threshold))) / vmax)) * inner_h))
    draw.line((left, threshold_y, right, threshold_y), fill=(230, 70, 70), width=2)

    pts: List[Tuple[int, int]] = []
    for i in range(n):
        px = _x_from_row(i)
        val = float(max(0.0, projection[i]))
        py = top + int(round((1.0 - min(1.0, val / vmax)) * inner_h))
        pts.append((px, py))
    if len(pts) == 1:
        x0, y0 = pts[0]
        draw.ellipse((x0 - 2, y0 - 2, x0 + 2, y0 + 2), fill=(35, 105, 225))
    else:
        draw.line(pts, fill=(35, 105, 225), width=2)

    draw.text((left, 8), title, fill=(30, 30, 30), font=font)
    draw.text((left, bottom + 8), f"rows: {n}", fill=(80, 80, 80), font=font)
    draw.text((left + 130, bottom + 8), f"max: {float(np.max(projection)):.1f}", fill=(80, 80, 80), font=font)
    draw.text((left + 275, bottom + 8), f"threshold: {float(threshold):.1f}", fill=(220, 60, 60), font=font)
    draw.text((left + 470, bottom + 8), "blue=profile, red=threshold, green=peaks, orange=final lines", fill=(90, 90, 90), font=font)

    return np.array(img)


def _compute_horizontal_projection_state(
    crop_rgb: np.ndarray,
    smooth_cols: int,
    threshold_rel: float,
) -> Dict[str, Any]:
    if cv2 is None:
        return {"ok": False, "reason": "opencv_missing"}
    if crop_rgb is None or crop_rgb.size == 0:
        return {"ok": False, "reason": "empty_crop"}

    h, w = crop_rgb.shape[:2]
    if h < 3 or w < 3:
        return {"ok": False, "reason": "crop_too_small"}

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Keep thin strokes while bridging tiny vertical breaks.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    projection = (bw > 0).sum(axis=0).astype(np.float32)
    if projection.size == 0 or float(np.max(projection)) <= 0.0:
        return {"ok": False, "reason": "empty_projection"}

    smooth = max(1, int(smooth_cols))
    if smooth % 2 == 0:
        smooth += 1
    if smooth > 1:
        conv = np.ones((smooth,), dtype=np.float32) / float(smooth)
        projection = np.convolve(projection, conv, mode="same")

    thr_rel = min(max(float(threshold_rel), 0.01), 0.95)
    threshold = max(1.0, float(np.max(projection)) * thr_rel)
    threshold = min(threshold, max(1.0, float(np.percentile(projection, 82.0))))
    mask = projection >= threshold

    return {
        "ok": True,
        "bw": bw,
        "projection": projection,
        "threshold": float(threshold),
        "mask": mask,
        "height": int(h),
        "width": int(w),
        "smooth_cols": int(smooth),
        "threshold_rel": float(thr_rel),
    }


def _render_horizontal_profile_plot(
    projection: np.ndarray,
    threshold: float,
    mask: np.ndarray,
    title: str,
    peaks: Optional[List[int]] = None,
    runs: Optional[List[List[int]]] = None,
) -> np.ndarray:
    if projection is None or projection.size == 0:
        return _line_profile_placeholder("No horizontal profile data available.")

    n = int(projection.size)
    plot_w, plot_h = 980, 320
    left, top, right, bottom = 56, 34, 960, 276
    inner_w = max(1, right - left)
    inner_h = max(1, bottom - top)

    img = Image.new("RGB", (plot_w, plot_h), (248, 248, 248))
    draw = ImageDraw.Draw(img)
    font = _load_overlay_font()
    draw.rectangle((left, top, right, bottom), outline=(140, 140, 140), width=2)

    def _x_from_col(col_idx: int) -> int:
        if n <= 1:
            return left
        t = float(max(0, min(n - 1, int(col_idx)))) / float(n - 1)
        return left + int(round(t * inner_w))

    if runs:
        for r in runs:
            if not isinstance(r, (tuple, list)) or len(r) != 2:
                continue
            rx1 = _x_from_col(int(r[0]))
            rx2 = _x_from_col(max(int(r[0]), int(r[1]) - 1))
            if rx2 <= rx1:
                rx2 = rx1 + 1
            draw.rectangle((rx1, top + 1, rx2, bottom - 1), fill=(255, 245, 220))

    if mask is not None and mask.size == n:
        i = 0
        while i < n:
            if not bool(mask[i]):
                i += 1
                continue
            j = i
            while j < n and bool(mask[j]):
                j += 1
            mx1 = _x_from_col(i)
            mx2 = _x_from_col(max(i, j - 1))
            if mx2 <= mx1:
                mx2 = mx1 + 1
            draw.rectangle((mx1, top + 1, mx2, bottom - 1), fill=(233, 248, 235))
            i = j

    for peak in (peaks or []):
        px = _x_from_col(int(peak))
        draw.line((px, top + 1, px, bottom - 1), fill=(46, 158, 78), width=1)
        draw.ellipse((px - 2, top + 3, px + 2, top + 7), fill=(46, 158, 78))

    vmax = max(1.0, float(np.max(projection)), float(threshold))
    threshold_y = top + int(round((1.0 - (min(vmax, max(0.0, float(threshold))) / vmax)) * inner_h))
    draw.line((left, threshold_y, right, threshold_y), fill=(230, 70, 70), width=2)

    pts: List[Tuple[int, int]] = []
    for i in range(n):
        px = _x_from_col(i)
        val = float(max(0.0, projection[i]))
        py = top + int(round((1.0 - min(1.0, val / vmax)) * inner_h))
        pts.append((px, py))
    if len(pts) == 1:
        x0, y0 = pts[0]
        draw.ellipse((x0 - 2, y0 - 2, x0 + 2, y0 + 2), fill=(35, 105, 225))
    else:
        draw.line(pts, fill=(35, 105, 225), width=2)

    draw.text((left, 8), title, fill=(30, 30, 30), font=font)
    draw.text((left, bottom + 8), f"cols: {n}", fill=(80, 80, 80), font=font)
    draw.text((left + 130, bottom + 8), f"max: {float(np.max(projection)):.1f}", fill=(80, 80, 80), font=font)
    draw.text((left + 275, bottom + 8), f"threshold: {float(threshold):.1f}", fill=(220, 60, 60), font=font)
    draw.text((left + 470, bottom + 8), "blue=profile, red=threshold, green=peaks, orange=peak-runs", fill=(90, 90, 90), font=font)
    return np.array(img)


def _render_clicked_line_overlay(
    line_crop_rgb: np.ndarray,
    boxes_local: List[Tuple[int, int, int, int]],
    peaks: Optional[List[int]] = None,
    peak_axis: str = "y",
) -> np.ndarray:
    if line_crop_rgb is None or line_crop_rgb.size == 0:
        return _line_profile_placeholder("No line crop available.")

    overlay = Image.fromarray(line_crop_rgb.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    font = _load_overlay_font()

    for i, (x1, y1, x2, y2) in enumerate(boxes_local, start=1):
        draw.rectangle((x1, y1, x2, y2), outline=(255, 170, 0), width=2)
        tag = f"peak-box {i}"
        tx, ty = x1 + 2, max(0, y1 - 14)
        draw.rectangle((tx, ty, tx + 8 * len(tag), ty + 12), fill=(0, 0, 0))
        draw.text((tx + 2, ty + 1), tag, fill=(255, 170, 0), font=font)

    if peak_axis == "x":
        for p in (peaks or []):
            px = int(max(0, min(overlay.width - 1, int(p))))
            draw.line((px, 0, px, overlay.height - 1), fill=(46, 158, 78), width=1)
    else:
        for p in (peaks or []):
            py = int(max(0, min(overlay.height - 1, int(p))))
            draw.line((0, py, overlay.width - 1, py), fill=(46, 158, 78), width=1)

    return np.array(overlay)


def _segment_horizontal_runs_in_line_crop(
    line_crop_rgb: np.ndarray,
    smooth_cols: int,
    threshold_rel: float,
    min_width_px: int,
    merge_gap_px: int,
    horizontal_state: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Tuple[int, int, int, int]], Dict[str, Any]]:
    state = horizontal_state
    if state is None:
        state = _compute_horizontal_projection_state(
            crop_rgb=line_crop_rgb,
            smooth_cols=int(smooth_cols),
            threshold_rel=float(threshold_rel),
        )
    if not bool(state.get("ok")):
        return [], state

    bw = np.asarray(state.get("bw"))
    projection = np.asarray(state.get("projection"), dtype=np.float32)
    threshold = float(state.get("threshold", 1.0))
    h = int(state.get("height", 0))
    w = int(state.get("width", 0))
    if projection.size == 0 or h <= 0 or w <= 0:
        state["horizontal_peaks"] = []
        state["horizontal_runs"] = []
        state["horizontal_method"] = "empty"
        return [], state

    min_w = max(3, int(min_width_px))
    merge_gap = max(0, int(merge_gap_px))
    peaks_used: List[int] = []
    runs: List[Tuple[int, int]] = []

    if find_peaks is not None and projection.size >= 5:
        maxv = float(np.max(projection))
        distance = max(3, int(round(w * 0.03)))
        prominence = max(1.0, maxv * max(0.05, float(threshold_rel) * 0.45))
        peaks, _ = find_peaks(projection, distance=distance, prominence=prominence)
        if peaks.size > 0:
            keep = projection[peaks] >= max(1.0, threshold * 0.65)
            if bool(np.any(keep)):
                peaks = peaks[keep]

        if peaks.size > 0:
            peaks = np.sort(peaks.astype(np.int32))
            peaks_used = [int(p) for p in peaks.tolist()]

            # Boundaries follow your rule:
            # start at line begin, then left valleys of subsequent peaks, then line end.
            boundaries: List[int] = [0]
            for i in range(1, len(peaks)):
                lo = int(peaks[i - 1])
                hi = int(peaks[i])
                if hi - lo < 2:
                    continue
                left_valley_of_next_peak = lo + int(np.argmin(projection[lo : hi + 1]))
                boundaries.append(left_valley_of_next_peak)
            boundaries.append(w)

            norm_boundaries: List[int] = [0]
            for b in boundaries[1:-1]:
                bx = int(max(norm_boundaries[-1] + 1, min(w - 1, int(b))))
                norm_boundaries.append(bx)
            norm_boundaries.append(w)

            for i in range(len(norm_boundaries) - 1):
                x1 = int(norm_boundaries[i])
                x2 = int(norm_boundaries[i + 1])
                if x2 > x1:
                    runs.append((x1, x2))

    if not runs:
        runs = [(0, w)]

    # Keep full coverage while avoiding tiny fragments:
    # tiny run is merged into a direct neighbor (prefer right).
    if runs and len(runs) > 1:
        i = 0
        while i < len(runs):
            x1, x2 = runs[i]
            if (x2 - x1) >= min_w:
                i += 1
                continue
            if i < len(runs) - 1:
                nx1, nx2 = runs[i + 1]
                if nx1 - x2 <= merge_gap:
                    runs[i + 1] = (x1, nx2)
                else:
                    runs[i + 1] = (x1, nx2)
                del runs[i]
            else:
                px1, px2 = runs[i - 1]
                if x1 - px2 <= merge_gap:
                    runs[i - 1] = (px1, x2)
                else:
                    runs[i - 1] = (px1, x2)
                del runs[i]
                i = max(0, i - 1)

    # Re-normalize to guarantee contiguous full-width coverage.
    if runs:
        renorm: List[Tuple[int, int]] = []
        cursor = 0
        for i, (_, x2_raw) in enumerate(runs):
            if i == len(runs) - 1:
                x2 = w
            else:
                x2 = max(cursor + 1, min(w - 1, int(x2_raw)))
            renorm.append((cursor, x2))
            cursor = x2
        if renorm:
            lx1, _ = renorm[-1]
            renorm[-1] = (lx1, w)
        runs = [(x1, x2) for x1, x2 in renorm if x2 > x1]

    state["horizontal_peaks"] = peaks_used
    state["horizontal_runs"] = [[int(x1), int(x2)] for x1, x2 in runs]
    state["horizontal_method"] = ("scipy_find_peaks" if peaks_used else "threshold_mask_fallback")

    boxes: List[Tuple[int, int, int, int]] = []
    for x1, x2 in runs:
        # Full-height boxes, contiguous from line start to line end.
        bx1 = max(0, min(w - 1, int(x1)))
        bx2 = max(0, min(w, int(x2)))
        if bx2 > bx1:
            boxes.append((bx1, 0, bx2, h))

    return boxes, state


def _left_valleys_for_peaks(projection: np.ndarray, peaks: List[int], width: int) -> List[int]:
    w = max(1, int(width))
    if projection is None or projection.size == 0 or w <= 0:
        return [0]

    sorted_peaks = sorted([int(p) for p in peaks if 0 <= int(p) < w])
    valleys: List[int] = [0]
    for i in range(1, len(sorted_peaks)):
        lo = int(sorted_peaks[i - 1])
        hi = int(sorted_peaks[i])
        if hi - lo < 2:
            continue
        valley = lo + int(np.argmin(projection[lo : hi + 1]))
        valleys.append(int(max(0, min(w - 1, valley))))
    return sorted(set(valleys))


def _build_wordbox_hierarchy_from_peaks(
    projection: np.ndarray,
    peaks: List[int],
    width: int,
    height: int,
    levels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    w = max(1, int(width))
    h = max(1, int(height))
    lvl = [2, 4, 8] if not levels else [max(1, int(v)) for v in levels]

    valley_starts = _left_valleys_for_peaks(projection=projection, peaks=peaks, width=w)
    valley_pool: List[int] = list(valley_starts)
    if find_peaks is not None and projection is not None and projection.size >= 5:
        minima, _ = find_peaks(-np.asarray(projection, dtype=np.float32), distance=max(2, int(round(w * 0.02))))
        valley_pool.extend(int(v) for v in minima.tolist())
    valley_pool.append(0)
    valley_pool = sorted(set(int(v) for v in valley_pool if 0 <= int(v) < w))

    levels_out: List[Dict[str, Any]] = []
    for n in lvl:
        requested_count = max(1, int(n))
        box_count = max(1, min(requested_count, w))
        desired_len = float(w) / float(requested_count)
        interior_valleys = sorted(int(v) for v in valley_pool if 0 < int(v) < w)

        boundaries: List[int] = [0]
        boundary_sources: List[str] = []
        for i in range(1, box_count):
            remaining_after = box_count - i
            min_x = boundaries[-1] + 1
            max_x = max(min_x, w - remaining_after)

            target = int(round(i * desired_len))
            target = max(min_x, min(max_x, target))

            candidates = [v for v in interior_valleys if min_x <= v <= max_x]
            if candidates:
                chosen = min(candidates, key=lambda v: (abs(int(v) - target), int(v)))
                source = "valley"
            else:
                chosen = target
                source = "fallback"

            chosen = int(max(min_x, min(max_x, chosen)))
            boundaries.append(chosen)
            boundary_sources.append(source)
        boundaries.append(w)

        boxes: List[Tuple[int, int, int, int]] = []
        for i in range(len(boundaries) - 1):
            x1 = int(boundaries[i])
            x2 = int(boundaries[i + 1])
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w, x2))
            if x2 > x1:
                boxes.append((x1, 0, x2, h))

        starts = [int(b[0]) for b in boxes]
        widths = [int(b[2] - b[0]) for b in boxes]
        levels_out.append(
            {
                "count": int(box_count),
                "target_count": int(requested_count),
                "box_length_px": int(round(desired_len)),
                "box_lengths_px": widths,
                "boundaries": [int(v) for v in boundaries],
                "boundary_sources": boundary_sources,
                "starts": [int(s) for s in starts],
                "boxes": [[int(a), int(b), int(c), int(d)] for (a, b, c, d) in boxes],
            }
        )

    return {
        "valley_starts": [int(v) for v in valley_starts],
        "valley_pool": [int(v) for v in valley_pool],
        "levels": levels_out,
    }


def _render_wordbox_hierarchy_view(
    line_crop_rgb: np.ndarray,
    hierarchy: Dict[str, Any],
    peaks: Optional[List[int]] = None,
) -> np.ndarray:
    if line_crop_rgb is None or line_crop_rgb.size == 0:
        return _line_profile_placeholder("No line crop available for hierarchy view.")

    base = line_crop_rgb.astype(np.uint8)
    h, w = base.shape[:2]
    font = _load_overlay_font()
    panels: List[np.ndarray] = []

    ref = Image.fromarray(base).convert("RGB")
    dref = ImageDraw.Draw(ref)
    dref.text((6, 6), "Reference (clicked line)", fill=(20, 20, 20), font=font)
    for p in (peaks or []):
        px = int(max(0, min(w - 1, int(p))))
        dref.line((px, 0, px, h - 1), fill=(46, 158, 78), width=1)
    panels.append(np.array(ref))

    palette = {
        2: (245, 160, 66),
        4: (80, 160, 255),
        8: (220, 95, 95),
    }
    for lvl in (hierarchy.get("levels") or []):
        n = int(lvl.get("count", 0))
        boxes = lvl.get("boxes") or []
        col = palette.get(n, (240, 180, 50))
        panel = Image.fromarray(base).convert("RGB")
        draw = ImageDraw.Draw(panel)
        tag = f"{n} wordboxes (len ~= 1/{n} line width)"
        draw.rectangle((2, 2, min(w - 2, 12 + 8 * len(tag)), 18), fill=(0, 0, 0))
        draw.text((6, 5), tag, fill=col, font=font)
        for i, box in enumerate(boxes, start=1):
            if not isinstance(box, (tuple, list)) or len(box) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            draw.rectangle((x1, y1, x2, y2), outline=col, width=2)
            label = f"{n}:{i}"
            tx, ty = x1 + 2, max(0, y1 + 2)
            draw.rectangle((tx, ty, tx + 7 * len(label), ty + 11), fill=(0, 0, 0))
            draw.text((tx + 1, ty + 1), label, fill=col, font=font)
        for p in (peaks or []):
            px = int(max(0, min(w - 1, int(p))))
            draw.line((px, 0, px, h - 1), fill=(46, 158, 78), width=1)
        panels.append(np.array(panel))

    sep = np.full((6, w, 3), 245, dtype=np.uint8)
    out = panels[0]
    for p in panels[1:]:
        out = np.vstack([out, sep, p])
    return out


def preview_clicked_line_horizontal_profile(
    click_state: Dict[str, Any],
    evt: gr.SelectData,
):
    profile_placeholder = _line_profile_placeholder("Click a line box in the overlay to inspect its horizontal profile.")
    view_placeholder = _line_profile_placeholder("Click a line box in the overlay to open the extracted line view.")
    hierarchy_placeholder = _line_profile_placeholder("Click a line box to generate the 2/4/8 wordbox hierarchy.")
    if not isinstance(click_state, dict):
        return view_placeholder, hierarchy_placeholder, profile_placeholder, "Run the line split first."

    image = click_state.get("image")
    line_boxes = click_state.get("line_boxes") or []
    if image is None or not line_boxes:
        return view_placeholder, hierarchy_placeholder, profile_placeholder, "No line boxes available. Run inference first."

    idx = getattr(evt, "index", None)
    if not isinstance(idx, (tuple, list)) or len(idx) < 2:
        return view_placeholder, hierarchy_placeholder, profile_placeholder, "Click position not available."
    try:
        click_x = int(idx[0])
        click_y = int(idx[1])
    except Exception:
        return view_placeholder, hierarchy_placeholder, profile_placeholder, "Invalid click position."

    def _hits_for(px: int, py: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for rec in line_boxes:
            box = rec.get("line_box") or []
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            if x1 <= px <= x2 and y1 <= py <= y2:
                area = max(1, (x2 - x1) * (y2 - y1))
                hit = dict(rec)
                hit["_area"] = float(area)
                out.append(hit)
        return out

    hits = _hits_for(click_x, click_y)
    if not hits:
        # Some frontend variants report (row, col) instead of (x, y).
        hits = _hits_for(click_y, click_x)

    if not hits:
        return view_placeholder, hierarchy_placeholder, profile_placeholder, f"No line box at click ({click_x}, {click_y})."

    chosen = sorted(hits, key=lambda r: float(r.get("_area", 1.0)))[0]
    x1, y1, x2, y2 = [int(v) for v in (chosen.get("line_box") or [0, 0, 0, 0])]
    if x2 <= x1 or y2 <= y1:
        return view_placeholder, hierarchy_placeholder, profile_placeholder, "Selected line box is invalid."

    src = np.asarray(image)
    h, w = src.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return view_placeholder, hierarchy_placeholder, profile_placeholder, "Selected line box is out of bounds."

    line_crop = src[y1:y2, x1:x2]
    hstate = _compute_horizontal_projection_state(
        crop_rgb=line_crop,
        smooth_cols=int(click_state.get("horizontal_profile_smooth_cols", 21)),
        threshold_rel=float(click_state.get("horizontal_profile_threshold_rel", 0.20)),
    )
    local_boxes, hstate = _segment_horizontal_runs_in_line_crop(
        line_crop_rgb=line_crop,
        smooth_cols=int(click_state.get("horizontal_profile_smooth_cols", 21)),
        threshold_rel=float(click_state.get("horizontal_profile_threshold_rel", 0.20)),
        min_width_px=int(click_state.get("horizontal_seg_min_width_px", 14)),
        merge_gap_px=int(click_state.get("horizontal_seg_merge_gap_px", 6)),
        horizontal_state=hstate,
    )
    selected_view = _render_clicked_line_overlay(
        line_crop_rgb=line_crop,
        boxes_local=local_boxes,
        peaks=[int(p) for p in (hstate.get("horizontal_peaks") or [])] if bool(hstate.get("ok")) else [],
        peak_axis="x",
    )

    if not bool(hstate.get("ok")):
        reason = str(hstate.get("reason", "unknown"))
        return selected_view, hierarchy_placeholder, profile_placeholder, f"Could not compute horizontal profile ({reason})."

    projection = np.asarray(hstate["projection"], dtype=np.float32)
    profile = _render_horizontal_profile_plot(
        projection=projection,
        threshold=float(hstate["threshold"]),
        mask=np.asarray(hstate["mask"], dtype=bool),
        title=f"Horizontal profile for line {int(chosen.get('line_id', -1))} [{x1}, {y1}, {x2}, {y2}]",
        peaks=[int(p) for p in (hstate.get("horizontal_peaks") or [])],
        runs=[list(r) for r in (hstate.get("horizontal_runs") or [])],
    )
    hierarchy = _build_wordbox_hierarchy_from_peaks(
        projection=projection,
        peaks=[int(p) for p in (hstate.get("horizontal_peaks") or [])],
        width=int(hstate.get("width", line_crop.shape[1])),
        height=int(hstate.get("height", line_crop.shape[0])),
        levels=[2, 4, 8],
    )
    hierarchy_view = _render_wordbox_hierarchy_view(
        line_crop_rgb=line_crop,
        hierarchy=hierarchy,
        peaks=[int(p) for p in (hstate.get("horizontal_peaks") or [])],
    )
    method = str(hstate.get("horizontal_method", "unknown"))
    level_map: Dict[int, int] = {}
    for rec in (hierarchy.get("levels") or []):
        n = int(rec.get("count", 0))
        level_map[n] = len(rec.get("boxes") or [])
    l2 = int(level_map.get(2, 0))
    l4 = int(level_map.get(4, 0))
    l8 = int(level_map.get(8, 0))
    status = (
        f"Selected line {int(chosen.get('line_id', -1))} at ({click_x}, {click_y}) "
        f"box=[{x1}, {y1}, {x2}, {y2}] peaks={len(hstate.get('horizontal_peaks') or [])} "
        f"boxes={len(local_boxes)} hierarchy(2/4/8)=({l2}/{l4}/{l8}) method={method}"
    )
    return selected_view, hierarchy_view, profile, status


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _parse_width_buckets_for_encoding(raw: Any, patch_multiple: int, max_width: int) -> List[int]:
    vals: List[int] = []
    if isinstance(raw, str):
        tokens = [tok.strip() for tok in raw.split(",")]
    elif isinstance(raw, (tuple, list)):
        tokens = [str(tok).strip() for tok in raw]
    else:
        tokens = []
    for tok in tokens:
        if not tok:
            continue
        try:
            v = int(tok)
        except Exception:
            continue
        if v > 0:
            vals.append(v)
    if not vals:
        vals = [int(max_width)]

    pm = max(1, int(patch_multiple))
    mw = max(pm, int(max_width))
    out: List[int] = []
    for v in vals:
        vv = max(pm, min(mw, int(v)))
        vv = int(math.ceil(float(vv) / float(pm)) * pm)
        vv = max(pm, min(mw, vv))
        out.append(vv)
    unique = sorted(set(out))
    return unique if unique else [mw]


def _normalize_for_vit_encoding(
    image: Image.Image,
    target_height: int,
    width_buckets: List[int],
    patch_multiple: int,
    max_width: int,
) -> Image.Image:
    rgb = image.convert("RGB")
    orig_w, orig_h = rgb.size
    if orig_w <= 0 or orig_h <= 0:
        return Image.new("RGB", (max(16, int(max_width)), max(8, int(target_height))), (255, 255, 255))

    t_h = max(8, int(target_height))
    pm = max(1, int(patch_multiple))
    mw = max(pm, int(max_width))

    scale_h = float(t_h) / float(orig_h)
    scaled_w = max(1, int(round(float(orig_w) * scale_h)))
    scaled_h = int(t_h)
    if scaled_w > mw:
        shrink = float(mw) / float(scaled_w)
        scaled_w = int(mw)
        scaled_h = max(1, int(round(float(t_h) * shrink)))

    resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
    resized = rgb.resize((scaled_w, scaled_h), resample=resample)

    buckets = sorted(int(v) for v in width_buckets if int(v) > 0)
    if not buckets:
        buckets = [mw]
    target_w = next((b for b in buckets if b >= scaled_w), buckets[-1])
    target_w = max(scaled_w, min(mw, int(target_w)))
    target_w = int(math.ceil(float(target_w) / float(pm)) * pm)
    target_w = min(mw, max(scaled_w, target_w))

    canvas = Image.new("RGB", (target_w, t_h), (255, 255, 255))
    canvas.paste(resized, (0, 0))
    return canvas


def _pooled_image_embedding_for_ui(backbone: Any, pixel_values: Any):
    kwargs = {"pixel_values": pixel_values, "return_dict": True}
    try:
        outputs = backbone(interpolate_pos_encoding=True, **kwargs)
    except TypeError:
        outputs = backbone(**kwargs)

    pooler = getattr(outputs, "pooler_output", None)
    if pooler is not None:
        return pooler
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        raise RuntimeError("Encoder output has no pooler_output/last_hidden_state")
    if hidden.ndim == 3:
        return hidden[:, 0]
    return hidden


if nn is not None:
    class _UIProjectionHead(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            hidden_dim = max(out_dim * 2, in_dim // 2, 32)
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

        def forward(self, x):
            return self.net(x)
else:
    class _UIProjectionHead:
        pass


def _resolve_torch_device_for_encoding(preferred: str) -> Tuple[str, str]:
    if torch is None:
        return "cpu", "PyTorch not available; using CPU placeholder."
    pref = (preferred or "").strip().lower()
    if pref in {"", "auto"}:
        if bool(getattr(torch.cuda, "is_available", lambda: False)()):
            return "cuda", ""
        mps_ok = False
        try:
            mps_ok = bool(torch.backends.mps.is_available())  # type: ignore[attr-defined]
        except Exception:
            mps_ok = False
        if mps_ok:
            return "mps", ""
        return "cpu", ""
    if pref.startswith("cuda") and not bool(getattr(torch.cuda, "is_available", lambda: False)()):
        return "cpu", f"Requested device `{preferred}` not available. Falling back to CPU."
    if pref == "mps":
        try:
            if bool(torch.backends.mps.is_available()):  # type: ignore[attr-defined]
                return "mps", ""
        except Exception:
            pass
        return "cpu", "Requested device `mps` not available. Falling back to CPU."
    return pref, ""


@lru_cache(maxsize=24)
def _load_hierarchy_encoder_runtime_config(backbone_path: str, projection_head_path: str) -> Dict[str, Any]:
    defaults = {
        "target_height": 64,
        "max_width": 1024,
        "patch_multiple": 16,
        "width_buckets": [256, 384, 512, 768],
        "image_preprocess_pipeline": "none",
        "image_preprocess_pb": None,
        "image_preprocess_bdrc": None,
        "source": "defaults",
    }
    bpath = Path(backbone_path).expanduser().resolve() if (backbone_path or "").strip() else None
    hpath = Path(projection_head_path).expanduser().resolve() if (projection_head_path or "").strip() else None
    candidates: List[Path] = []
    if bpath is not None:
        candidates.append(bpath.parent / "training_config.json")
        for p in sorted(bpath.parent.glob("*training_config.json")):
            candidates.append(p)
    if hpath is not None:
        candidates.append(hpath.parent / "training_config.json")
        for p in sorted(hpath.parent.glob("*training_config.json")):
            candidates.append(p)

    seen: set[str] = set()
    uniq: List[Path] = []
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)

    for cfg_path in uniq:
        if not cfg_path.exists() or not cfg_path.is_file():
            continue
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        out = dict(defaults)
        out["target_height"] = max(8, _to_int(data.get("target_height", out["target_height"]), out["target_height"]))
        out["max_width"] = max(16, _to_int(data.get("max_width", out["max_width"]), out["max_width"]))
        out["patch_multiple"] = max(1, _to_int(data.get("patch_multiple", out["patch_multiple"]), out["patch_multiple"]))
        raw_buckets = data.get("width_buckets", out["width_buckets"])
        out["width_buckets"] = _parse_width_buckets_for_encoding(
            raw=raw_buckets,
            patch_multiple=out["patch_multiple"],
            max_width=out["max_width"],
        )
        out["image_preprocess_pipeline"] = str(data.get("image_preprocess_pipeline", "none") or "none").strip().lower()
        out["image_preprocess_pb"] = data.get("image_preprocess_pb") if isinstance(data.get("image_preprocess_pb"), dict) else None
        out["image_preprocess_bdrc"] = (
            data.get("image_preprocess_bdrc") if isinstance(data.get("image_preprocess_bdrc"), dict) else None
        )
        out["source"] = str(cfg_path)
        return out
    return defaults


def _suggest_projection_head_for_backbone(backbone_path: str) -> str:
    if not (backbone_path or "").strip():
        return ""
    bpath = Path(backbone_path).expanduser().resolve()
    if not bpath.exists():
        return ""
    cfg = _load_hierarchy_encoder_runtime_config(str(bpath), "")
    cfg_source = str(cfg.get("source", "") or "")
    if cfg_source and cfg_source != "defaults":
        try:
            cfg_data = json.loads(Path(cfg_source).read_text(encoding="utf-8"))
            proj = str(cfg_data.get("projection_head_path", "") or "").strip()
            if proj:
                p = Path(proj).expanduser().resolve()
                if p.exists():
                    return str(p)
        except Exception:
            pass

    local = [
        p for p in sorted(bpath.parent.glob("*projection_head*.pt"))
        if "clip_text_projection_head" not in p.name.lower()
    ]
    if local:
        return str(local[0].resolve())
    return ""


def scan_models_for_line_and_encoder_ui(models_dir: str):
    base = Path(models_dir).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        empty = gr.update(choices=[], value=None)
        return empty, empty, empty, "", "", "", f"Directory not found: {base}"

    yolo_models: List[str] = []
    encoder_backbones: List[str] = []
    projection_heads: List[str] = []
    model_exts = {".pt", ".onnx", ".torchscript"}

    for p in sorted(base.rglob("*")):
        if p.is_file():
            sfx = p.suffix.lower()
            if sfx not in model_exts:
                continue
            lname = p.name.lower()
            if "lora" in lname:
                continue
            if "projection_head" in lname:
                # Keep only image-side projection heads in this dropdown.
                # `line_clip` training additionally saves a text projection head
                # (`...clip_text_projection_head.pt`) which is not compatible
                # with the image encoder UI flows.
                if "clip_text_projection_head" in lname:
                    continue
                projection_heads.append(str(p.resolve()))
                continue
            yolo_models.append(str(p.resolve()))
            continue

        if not p.is_dir():
            continue
        if not (p / "config.json").exists():
            continue
        has_weights = (
            (p / "pytorch_model.bin").exists()
            or (p / "model.safetensors").exists()
            or (p / "model.safetensors.index.json").exists()
        )
        if not has_weights:
            continue
        pname = p.name.lower()
        if any(k in pname for k in ("backbone", "vit", "dino", "swin", "beit")):
            encoder_backbones.append(str(p.resolve()))

    yolo_models = sorted(set(yolo_models))
    encoder_backbones = sorted(set(encoder_backbones))
    projection_heads = sorted(set(projection_heads))

    yolo_default = yolo_models[0] if yolo_models else ""
    enc_default = encoder_backbones[0] if encoder_backbones else ""
    head_default = _suggest_projection_head_for_backbone(enc_default) if enc_default else ""
    if not head_default and projection_heads:
        head_default = projection_heads[0]

    status = (
        f"Found {len(yolo_models)} YOLO model(s), "
        f"{len(encoder_backbones)} encoder backbone(s), "
        f"{len(projection_heads)} projection head(s) in {base}"
    )
    return (
        gr.update(choices=yolo_models, value=(yolo_default if yolo_default else None)),
        gr.update(choices=encoder_backbones, value=(enc_default if enc_default else None)),
        gr.update(choices=projection_heads, value=(head_default if head_default else None)),
        yolo_default,
        enc_default,
        head_default,
        status,
    )


def on_encoder_backbone_change_ui(backbone_path: str):
    value = (backbone_path or "").strip()
    suggested = _suggest_projection_head_for_backbone(value)
    return value, suggested


def _hierarchy_boxes_for_level(hierarchy: Dict[str, Any], level_count: int) -> List[List[int]]:
    lvl = max(1, int(level_count))
    levels = hierarchy.get("levels") or []
    for rec in levels:
        if int(rec.get("count", -1)) == lvl:
            boxes = rec.get("boxes") or []
            return [[int(v) for v in box] for box in boxes if isinstance(box, (tuple, list)) and len(box) == 4]
    if levels:
        boxes = levels[0].get("boxes") or []
        return [[int(v) for v in box] for box in boxes if isinstance(box, (tuple, list)) and len(box) == 4]
    return []


def _render_hierarchy_level_overlay(
    line_crop_rgb: np.ndarray,
    boxes: List[List[int]],
    level_count: int,
    selected_idx: int = -1,
) -> np.ndarray:
    if line_crop_rgb is None or line_crop_rgb.size == 0:
        return _line_profile_placeholder("No selected line available.")

    palette = {2: (245, 160, 66), 4: (80, 160, 255), 8: (220, 95, 95)}
    color = palette.get(int(level_count), (255, 180, 80))

    panel = Image.fromarray(line_crop_rgb.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(panel)
    font = _load_overlay_font()
    tag = f"Click block to encode (level={int(level_count)}, boxes={len(boxes)})"
    draw.rectangle((2, 2, min(panel.width - 2, 14 + 8 * len(tag)), 18), fill=(0, 0, 0))
    draw.text((6, 5), tag, fill=color, font=font)
    for i, box in enumerate(boxes, start=0):
        x1, y1, x2, y2 = [int(v) for v in box]
        stroke = 3 if i == int(selected_idx) else 2
        line_col = (40, 220, 120) if i == int(selected_idx) else color
        draw.rectangle((x1, y1, x2, y2), outline=line_col, width=stroke)
        label = f"{int(level_count)}:{i + 1}"
        tx, ty = x1 + 2, max(0, y1 + 2)
        draw.rectangle((tx, ty, tx + 7 * len(label), ty + 11), fill=(0, 0, 0))
        draw.text((tx + 1, ty + 1), label, fill=line_col, font=font)
    return np.array(panel)


def _find_clicked_box_index(boxes: List[List[int]], click_x: int, click_y: int) -> int:
    hits: List[Tuple[int, int]] = []
    for i, box in enumerate(boxes):
        if not isinstance(box, (tuple, list)) or len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        if x1 <= click_x <= x2 and y1 <= click_y <= y2:
            area = max(1, (x2 - x1) * (y2 - y1))
            hits.append((area, i))
    if not hits:
        return -1
    hits = sorted(hits, key=lambda x: x[0])
    return int(hits[0][1])


def _compute_clicked_line_hierarchy_bundle(click_state: Dict[str, Any], evt: gr.SelectData) -> Dict[str, Any]:
    profile_placeholder = _line_profile_placeholder("Click a line box in the overlay to inspect its horizontal profile.")
    view_placeholder = _line_profile_placeholder("Click a line box in the overlay to open the extracted line view.")
    if not isinstance(click_state, dict):
        return {"ok": False, "reason": "Run line split first.", "selected_view": view_placeholder, "profile": profile_placeholder}

    image = click_state.get("image")
    line_boxes = click_state.get("line_boxes") or []
    if image is None or not line_boxes:
        return {"ok": False, "reason": "No line boxes available.", "selected_view": view_placeholder, "profile": profile_placeholder}

    idx = getattr(evt, "index", None)
    if not isinstance(idx, (tuple, list)) or len(idx) < 2:
        return {"ok": False, "reason": "Click position not available.", "selected_view": view_placeholder, "profile": profile_placeholder}
    try:
        click_x = int(idx[0])
        click_y = int(idx[1])
    except Exception:
        return {"ok": False, "reason": "Invalid click position.", "selected_view": view_placeholder, "profile": profile_placeholder}

    def _hits_for(px: int, py: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for rec in line_boxes:
            box = rec.get("line_box") or []
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            if x1 <= px <= x2 and y1 <= py <= y2:
                area = max(1, (x2 - x1) * (y2 - y1))
                hit = dict(rec)
                hit["_area"] = float(area)
                out.append(hit)
        return out

    hits = _hits_for(click_x, click_y)
    if not hits:
        hits = _hits_for(click_y, click_x)
    if not hits:
        return {
            "ok": False,
            "reason": f"No line box at click ({click_x}, {click_y}).",
            "selected_view": view_placeholder,
            "profile": profile_placeholder,
        }

    chosen = sorted(hits, key=lambda r: float(r.get("_area", 1.0)))[0]
    x1, y1, x2, y2 = [int(v) for v in (chosen.get("line_box") or [0, 0, 0, 0])]
    if x2 <= x1 or y2 <= y1:
        return {"ok": False, "reason": "Selected line box is invalid.", "selected_view": view_placeholder, "profile": profile_placeholder}

    src = np.asarray(image)
    h, w = src.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return {"ok": False, "reason": "Selected line box is out of bounds.", "selected_view": view_placeholder, "profile": profile_placeholder}

    line_crop = src[y1:y2, x1:x2]
    hstate = _compute_horizontal_projection_state(
        crop_rgb=line_crop,
        smooth_cols=int(click_state.get("horizontal_profile_smooth_cols", 21)),
        threshold_rel=float(click_state.get("horizontal_profile_threshold_rel", 0.20)),
    )
    local_boxes, hstate = _segment_horizontal_runs_in_line_crop(
        line_crop_rgb=line_crop,
        smooth_cols=int(click_state.get("horizontal_profile_smooth_cols", 21)),
        threshold_rel=float(click_state.get("horizontal_profile_threshold_rel", 0.20)),
        min_width_px=int(click_state.get("horizontal_seg_min_width_px", 14)),
        merge_gap_px=int(click_state.get("horizontal_seg_merge_gap_px", 6)),
        horizontal_state=hstate,
    )
    selected_view = _render_clicked_line_overlay(
        line_crop_rgb=line_crop,
        boxes_local=local_boxes,
        peaks=[int(p) for p in (hstate.get("horizontal_peaks") or [])] if bool(hstate.get("ok")) else [],
        peak_axis="x",
    )
    if not bool(hstate.get("ok")):
        return {
            "ok": False,
            "reason": f"Could not compute horizontal profile ({str(hstate.get('reason', 'unknown'))}).",
            "selected_view": selected_view,
            "profile": profile_placeholder,
        }

    projection = np.asarray(hstate.get("projection"), dtype=np.float32)
    profile = _render_horizontal_profile_plot(
        projection=projection,
        threshold=float(hstate.get("threshold", 0.0)),
        mask=np.asarray(hstate.get("mask"), dtype=bool),
        title=f"Horizontal profile for line {int(chosen.get('line_id', -1))} [{x1}, {y1}, {x2}, {y2}]",
        peaks=[int(p) for p in (hstate.get("horizontal_peaks") or [])],
        runs=[list(r) for r in (hstate.get("horizontal_runs") or [])],
    )
    hierarchy = _build_wordbox_hierarchy_from_peaks(
        projection=projection,
        peaks=[int(p) for p in (hstate.get("horizontal_peaks") or [])],
        width=int(hstate.get("width", line_crop.shape[1])),
        height=int(hstate.get("height", line_crop.shape[0])),
        levels=[2, 4, 8],
    )
    return {
        "ok": True,
        "selected_view": selected_view,
        "profile": profile,
        "line_crop": line_crop,
        "line_id": int(chosen.get("line_id", -1)),
        "line_box": [x1, y1, x2, y2],
        "hierarchy": hierarchy,
        "horizontal_state": hstate,
        "local_boxes": local_boxes,
        "click_xy": [int(click_x), int(click_y)],
    }


def run_tibetan_text_line_split_for_embedding_ui(
    image: np.ndarray,
    model_path: str,
    conf: float,
    imgsz: int,
    device: str,
    min_line_height: int,
    projection_smooth: int,
    projection_threshold_rel: float,
    merge_gap_px: int,
):
    out = run_tibetan_text_line_split_classical(
        image=image,
        model_path=model_path,
        conf=conf,
        imgsz=imgsz,
        device=device,
        min_line_height=min_line_height,
        projection_smooth=projection_smooth,
        projection_threshold_rel=projection_threshold_rel,
        merge_gap_px=merge_gap_px,
        draw_parent_boxes=True,
        detect_red_text=False,
        red_min_redness=26,
        red_min_saturation=35,
        red_column_fill_rel=0.07,
        red_merge_gap_px=14,
        red_min_width_px=18,
        draw_red_boxes=False,
    )
    (
        overlay,
        status,
        out_json,
        line_profile,
        selected_line_view,
        _unused_hierarchy_view,
        selected_line_profile,
        click_status,
        click_state,
    ) = out
    word_overlay = _line_profile_placeholder("Click a line in the overlay, choose level 2/4/8, then click a block.")
    word_crop = _line_profile_placeholder("Selected text block crop appears here.")
    vector_text = ""
    encode_status = "Click a line box in the overlay."
    selected_state: Dict[str, Any] = {}
    return (
        overlay,
        status,
        out_json,
        line_profile,
        selected_line_view,
        selected_line_profile,
        click_status,
        click_state,
        word_overlay,
        word_crop,
        vector_text,
        encode_status,
        selected_state,
    )


def _persist_faiss_ui_query_crop(query_crop_rgb: np.ndarray) -> str:
    if query_crop_rgb is None:
        return ""
    arr = np.asarray(query_crop_rgb)
    if arr.size == 0:
        return ""
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = arr.astype(np.uint8, copy=False)
    out_dir = (Path(tempfile.gettempdir()) / "pechabridge_ui" / "faiss_query_crops").resolve()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = f"{int(time.time() * 1000)}_{int(time.time_ns() % 1000000)}"
        out_path = out_dir / f"query_{stamp}.png"
        Image.fromarray(arr).save(out_path)
        return str(out_path)
    except Exception:
        return ""


def _render_faiss_query_selection_overlay(
    line_crop_rgb: np.ndarray,
    selection_x: Optional[List[int]],
    anchor_x: Optional[int],
) -> np.ndarray:
    if line_crop_rgb is None or np.asarray(line_crop_rgb).size == 0:
        return _line_profile_placeholder("Click a line in the overlay first.")

    base = np.asarray(line_crop_rgb).astype(np.uint8, copy=False)
    panel = Image.fromarray(base).convert("RGB")
    draw = ImageDraw.Draw(panel)
    font = _load_overlay_font()
    w, h = panel.width, panel.height

    tag = "Click twice in this line to define query range (x-start, x-end)."
    draw.rectangle((2, 2, min(w - 2, 14 + 8 * len(tag)), 18), fill=(0, 0, 0))
    draw.text((6, 5), tag, fill=(245, 180, 72), font=font)

    if isinstance(selection_x, (list, tuple)) and len(selection_x) >= 2:
        sx1 = max(0, min(w - 1, int(selection_x[0])))
        sx2 = max(sx1 + 1, min(w, int(selection_x[1])))
        draw.rectangle((sx1, 0, sx2 - 1, h - 1), outline=(80, 200, 255), width=3)
        label = f"query [{sx1},{sx2})"
        draw.rectangle((sx1 + 2, 20, min(w - 2, sx1 + 12 + 8 * len(label)), 34), fill=(0, 0, 0))
        draw.text((sx1 + 4, 22), label, fill=(80, 200, 255), font=font)

    if anchor_x is not None:
        ax = max(0, min(w - 1, int(anchor_x)))
        draw.line((ax, 0, ax, h - 1), fill=(255, 90, 90), width=2)
        draw.rectangle((max(0, ax - 3), 0, min(w - 1, ax + 3), min(h - 1, 7)), fill=(255, 90, 90))

    return np.asarray(panel)


def _extract_click_x_for_line(evt: gr.SelectData, line_width: int) -> Optional[int]:
    idx = getattr(evt, "index", None)
    if not isinstance(idx, (tuple, list)) or len(idx) < 2:
        return None
    try:
        a = int(idx[0])
        b = int(idx[1])
    except Exception:
        return None
    if line_width <= 0:
        return None
    if 0 <= a < line_width:
        return int(a)
    if 0 <= b < line_width:
        return int(b)
    return int(max(0, min(line_width - 1, a)))


def run_tibetan_text_line_split_for_faiss_ui(
    image: np.ndarray,
    model_path: str,
    conf: float,
    imgsz: int,
    device: str,
    min_line_height: int,
    projection_smooth: int,
    projection_threshold_rel: float,
    merge_gap_px: int,
):
    out = run_tibetan_text_line_split_classical(
        image=image,
        model_path=model_path,
        conf=conf,
        imgsz=imgsz,
        device=device,
        min_line_height=min_line_height,
        projection_smooth=projection_smooth,
        projection_threshold_rel=projection_threshold_rel,
        merge_gap_px=merge_gap_px,
        draw_parent_boxes=True,
        detect_red_text=False,
        red_min_redness=26,
        red_min_saturation=35,
        red_column_fill_rel=0.07,
        red_merge_gap_px=14,
        red_min_width_px=18,
        draw_red_boxes=False,
    )
    (
        overlay,
        status,
        out_json,
        line_profile,
        selected_line_view,
        _unused_hierarchy_view,
        selected_line_profile,
        click_status,
        click_state,
    ) = out
    query_crop = _line_profile_placeholder("Select a line first, then choose a query range.")
    query_status = "Click a line box in the overlay, then click twice inside the selected line."
    query_state: Dict[str, Any] = {}
    return (
        overlay,
        status,
        out_json,
        line_profile,
        selected_line_view,
        selected_line_profile,
        click_status,
        click_state,
        query_crop,
        "",
        query_status,
        query_state,
    )


def prepare_clicked_line_for_faiss_query_ui(
    click_state: Dict[str, Any],
    evt: gr.SelectData,
):
    line_view_placeholder = _line_profile_placeholder("Click a line box in the overlay first.")
    line_profile_placeholder = _line_profile_placeholder("Selected line horizontal profile appears here.")
    query_crop_placeholder = _line_profile_placeholder("Selected FAISS query crop appears here.")

    bundle = _compute_clicked_line_hierarchy_bundle(click_state=click_state, evt=evt)
    if not bool(bundle.get("ok")):
        reason = str(bundle.get("reason", "Line selection failed."))
        return (
            line_view_placeholder,
            line_profile_placeholder,
            reason,
            query_crop_placeholder,
            "",
            reason,
            {},
        )

    line_crop = np.asarray(bundle.get("line_crop"))
    if line_crop.size == 0:
        reason = "Selected line crop is empty."
        return (
            line_view_placeholder,
            line_profile_placeholder,
            reason,
            query_crop_placeholder,
            "",
            reason,
            {},
        )

    h, w = line_crop.shape[:2]
    if h < 2 or w < 2:
        reason = "Selected line crop is too small."
        return (
            line_view_placeholder,
            line_profile_placeholder,
            reason,
            query_crop_placeholder,
            "",
            reason,
            {},
        )

    selection_x = [0, int(w)]
    query_crop = line_crop[:, int(selection_x[0]) : int(selection_x[1])]
    query_path = _persist_faiss_ui_query_crop(query_crop)
    overlay = _render_faiss_query_selection_overlay(line_crop_rgb=line_crop, selection_x=selection_x, anchor_x=None)
    line_profile = (
        np.asarray(bundle.get("profile"))
        if isinstance(bundle.get("profile"), np.ndarray)
        else line_profile_placeholder
    )
    line_status = (
        f"Selected line {int(bundle.get('line_id', -1))} "
        f"box={bundle.get('line_box') or [0, 0, 0, 0]} size={w}x{h}"
    )
    query_status = (
        f"Using full line as query ({w}px width). "
        "Click twice in Selected Line View to refine."
    )
    query_state = {
        "line_crop": line_crop.astype(np.uint8, copy=False),
        "line_id": int(bundle.get("line_id", -1)),
        "line_box": [int(v) for v in (bundle.get("line_box") or [0, 0, 0, 0])],
        "selection_x": [int(selection_x[0]), int(selection_x[1])],
        "anchor_x": None,
        "query_path": str(query_path),
    }
    return (
        overlay,
        line_profile,
        line_status,
        query_crop.astype(np.uint8, copy=False),
        str(query_path),
        query_status,
        query_state,
    )


def select_faiss_query_range_in_line_ui(
    query_state: Dict[str, Any],
    evt: gr.SelectData,
):
    query_crop_placeholder = _line_profile_placeholder("Select a line first.")
    line_view_placeholder = _line_profile_placeholder("Click a line box in the overlay first.")

    if not isinstance(query_state, dict) or query_state.get("line_crop") is None:
        return line_view_placeholder, query_crop_placeholder, "", "No selected line available.", {}

    line_crop = np.asarray(query_state.get("line_crop"))
    if line_crop.size == 0:
        return line_view_placeholder, query_crop_placeholder, "", "No selected line available.", {}

    h, w = line_crop.shape[:2]
    selection_x_raw = query_state.get("selection_x") or [0, int(w)]
    sx1 = max(0, min(w - 1, _to_int(selection_x_raw[0], 0)))
    sx2 = max(sx1 + 1, min(w, _to_int(selection_x_raw[1], w)))
    current_crop = line_crop[:, sx1:sx2]
    current_path = str(query_state.get("query_path", "") or "")
    if not current_path:
        current_path = _persist_faiss_ui_query_crop(current_crop)
        query_state["query_path"] = str(current_path)

    click_x = _extract_click_x_for_line(evt=evt, line_width=w)
    if click_x is None:
        overlay = _render_faiss_query_selection_overlay(
            line_crop_rgb=line_crop,
            selection_x=[sx1, sx2],
            anchor_x=query_state.get("anchor_x"),
        )
        return overlay, current_crop, current_path, "Click position not available.", query_state

    anchor = query_state.get("anchor_x")
    if anchor is None:
        query_state["anchor_x"] = int(click_x)
        overlay = _render_faiss_query_selection_overlay(
            line_crop_rgb=line_crop,
            selection_x=[sx1, sx2],
            anchor_x=int(click_x),
        )
        status = f"Start point set at x={int(click_x)}. Click end point now."
        return overlay, current_crop, current_path, status, query_state

    ax = max(0, min(w - 1, _to_int(anchor, 0)))
    nx1 = int(min(ax, int(click_x)))
    nx2 = int(max(ax, int(click_x)) + 1)
    if nx2 - nx1 < 2:
        nx2 = min(w, nx1 + 2)
    if nx2 <= nx1:
        nx1, nx2 = 0, int(w)

    query_state["selection_x"] = [int(nx1), int(nx2)]
    query_state["anchor_x"] = None
    query_crop = line_crop[:, int(nx1) : int(nx2)]
    query_path = _persist_faiss_ui_query_crop(query_crop)
    query_state["query_path"] = str(query_path)
    overlay = _render_faiss_query_selection_overlay(
        line_crop_rgb=line_crop,
        selection_x=[int(nx1), int(nx2)],
        anchor_x=None,
    )
    status = f"Query range selected: x=[{int(nx1)},{int(nx2)}) width={int(nx2 - nx1)} px."
    return overlay, query_crop.astype(np.uint8, copy=False), str(query_path), status, query_state


def reset_faiss_query_to_full_line_ui(query_state: Dict[str, Any]):
    query_crop_placeholder = _line_profile_placeholder("Select a line first.")
    line_view_placeholder = _line_profile_placeholder("Click a line box in the overlay first.")
    if not isinstance(query_state, dict) or query_state.get("line_crop") is None:
        return line_view_placeholder, query_crop_placeholder, "", "No selected line available.", {}

    line_crop = np.asarray(query_state.get("line_crop"))
    if line_crop.size == 0:
        return line_view_placeholder, query_crop_placeholder, "", "No selected line available.", {}

    h, w = line_crop.shape[:2]
    if h < 2 or w < 2:
        return line_view_placeholder, query_crop_placeholder, "", "Selected line is too small.", query_state

    query_state["selection_x"] = [0, int(w)]
    query_state["anchor_x"] = None
    query_crop = line_crop[:, : int(w)]
    query_path = _persist_faiss_ui_query_crop(query_crop)
    query_state["query_path"] = str(query_path)
    overlay = _render_faiss_query_selection_overlay(
        line_crop_rgb=line_crop,
        selection_x=[0, int(w)],
        anchor_x=None,
    )
    status = f"Reset query range to full line width ({int(w)} px)."
    return overlay, query_crop.astype(np.uint8, copy=False), str(query_path), status, query_state


def prepare_clicked_line_for_embedding_ui(
    click_state: Dict[str, Any],
    hierarchy_level: str,
    evt: gr.SelectData,
):
    word_overlay_placeholder = _line_profile_placeholder("Click a line in the overlay first.")
    word_crop_placeholder = _line_profile_placeholder("Selected text block crop appears here.")
    level_count = max(1, _to_int(hierarchy_level, 2))

    bundle = _compute_clicked_line_hierarchy_bundle(click_state=click_state, evt=evt)
    if not bool(bundle.get("ok")):
        reason = str(bundle.get("reason", "Line selection failed."))
        return (
            bundle.get("selected_view") if isinstance(bundle.get("selected_view"), np.ndarray) else _line_profile_placeholder(reason),
            bundle.get("profile") if isinstance(bundle.get("profile"), np.ndarray) else _line_profile_placeholder(reason),
            word_overlay_placeholder,
            reason,
            {},
            word_crop_placeholder,
            "",
            reason,
        )

    hierarchy = bundle.get("hierarchy") or {}
    boxes_level = _hierarchy_boxes_for_level(hierarchy=hierarchy, level_count=level_count)
    word_overlay = _render_hierarchy_level_overlay(
        line_crop_rgb=np.asarray(bundle["line_crop"]),
        boxes=boxes_level,
        level_count=level_count,
        selected_idx=-1,
    )
    selected_state = {
        "line_crop": np.asarray(bundle["line_crop"]),
        "line_id": int(bundle.get("line_id", -1)),
        "line_box": [int(v) for v in (bundle.get("line_box") or [0, 0, 0, 0])],
        "hierarchy": hierarchy,
        "active_level": int(level_count),
        "active_boxes": boxes_level,
        "selected_box_index": -1,
        "click_xy": bundle.get("click_xy") or None,
    }
    line_status = (
        f"Selected line {int(bundle.get('line_id', -1))} "
        f"box={selected_state['line_box']} | level={int(level_count)} | blocks={len(boxes_level)}"
    )
    return (
        np.asarray(bundle["selected_view"]),
        np.asarray(bundle["profile"]),
        word_overlay,
        line_status,
        selected_state,
        word_crop_placeholder,
        "",
        "Click a block in the hierarchy overlay to encode it.",
    )


def update_hierarchy_level_for_embedding_ui(
    selected_state: Dict[str, Any],
    hierarchy_level: str,
):
    level_count = max(1, _to_int(hierarchy_level, 2))
    if not isinstance(selected_state, dict) or selected_state.get("line_crop") is None:
        placeholder = _line_profile_placeholder("Click a line in the overlay first.")
        return placeholder, "No selected line available.", {}, _line_profile_placeholder("Selected text block crop appears here."), "", "Click a line first."

    line_crop = np.asarray(selected_state.get("line_crop"))
    hierarchy = selected_state.get("hierarchy") or {}
    boxes_level = _hierarchy_boxes_for_level(hierarchy=hierarchy, level_count=level_count)
    overlay = _render_hierarchy_level_overlay(
        line_crop_rgb=line_crop,
        boxes=boxes_level,
        level_count=level_count,
        selected_idx=-1,
    )
    selected_state["active_level"] = int(level_count)
    selected_state["active_boxes"] = boxes_level
    selected_state["selected_box_index"] = -1
    status = (
        f"Line {int(selected_state.get('line_id', -1))}: level={int(level_count)} blocks={len(boxes_level)}"
    )
    return overlay, status, selected_state, _line_profile_placeholder("Selected text block crop appears here."), "", "Click a block in the hierarchy overlay to encode it."


@lru_cache(maxsize=6)
def _load_hierarchy_vit_encoder_bundle(backbone_path: str, projection_head_path: str):
    if torch is None or AutoModel is None or AutoImageProcessor is None or nn is None:
        raise RuntimeError("Encoding requires torch + transformers. Please install requirements.")

    bpath = Path(backbone_path).expanduser().resolve()
    if not bpath.exists() or not bpath.is_dir():
        raise FileNotFoundError(f"Encoder backbone not found: {bpath}")

    image_processor = AutoImageProcessor.from_pretrained(str(bpath))
    backbone = AutoModel.from_pretrained(str(bpath))
    backbone.eval()

    projection_head = None
    ppath_raw = (projection_head_path or "").strip()
    if ppath_raw:
        ppath = Path(ppath_raw).expanduser().resolve()
        if ppath.exists() and ppath.is_file():
            payload = torch.load(str(ppath), map_location="cpu")
            if isinstance(payload, dict) and "state_dict" in payload:
                state_dict = payload.get("state_dict", {})
                in_dim = _to_int(payload.get("input_dim", 0), 0)
                out_dim = _to_int(payload.get("output_dim", 0), 0)
            else:
                state_dict = payload
                in_dim = 0
                out_dim = 0
            hidden_size = _to_int(getattr(backbone.config, "hidden_size", 0), 0)
            if in_dim <= 0:
                in_dim = hidden_size
            if out_dim <= 0:
                out_dim = in_dim
            if in_dim > 0 and out_dim > 0:
                head = _UIProjectionHead(in_dim=int(in_dim), out_dim=int(out_dim))
                head.load_state_dict(state_dict, strict=False)
                head.eval()
                projection_head = head

    return image_processor, backbone, projection_head


def encode_clicked_hierarchy_block_ui(
    selected_state: Dict[str, Any],
    encoder_backbone_path: str,
    projection_head_path: str,
    encoder_device: str,
    l2_normalize: bool,
    vector_decimals: int,
    evt: gr.SelectData,
):
    if not isinstance(selected_state, dict) or selected_state.get("line_crop") is None:
        placeholder = _line_profile_placeholder("Click a line and choose a hierarchy level first.")
        return placeholder, _line_profile_placeholder("Selected text block crop appears here."), "", "No selected line hierarchy available.", {}

    boxes = selected_state.get("active_boxes") or []
    level_count = _to_int(selected_state.get("active_level", 2), 2)
    line_crop = np.asarray(selected_state.get("line_crop"))
    if line_crop.size == 0 or not boxes:
        overlay = _render_hierarchy_level_overlay(line_crop, boxes, level_count, selected_idx=-1)
        return overlay, _line_profile_placeholder("No block available for encoding."), "", "No active hierarchy blocks to encode.", selected_state

    idx = getattr(evt, "index", None)
    if not isinstance(idx, (tuple, list)) or len(idx) < 2:
        overlay = _render_hierarchy_level_overlay(line_crop, boxes, level_count, selected_idx=-1)
        return overlay, _line_profile_placeholder("Click position not available."), "", "Click position not available.", selected_state
    try:
        click_x = int(idx[0])
        click_y = int(idx[1])
    except Exception:
        overlay = _render_hierarchy_level_overlay(line_crop, boxes, level_count, selected_idx=-1)
        return overlay, _line_profile_placeholder("Invalid click position."), "", "Invalid click position.", selected_state

    box_idx = _find_clicked_box_index(boxes, click_x, click_y)
    if box_idx < 0:
        box_idx = _find_clicked_box_index(boxes, click_y, click_x)
    if box_idx < 0:
        overlay = _render_hierarchy_level_overlay(line_crop, boxes, level_count, selected_idx=-1)
        return overlay, _line_profile_placeholder("No hierarchy block at clicked position."), "", "No hierarchy block at clicked position.", selected_state

    x1, y1, x2, y2 = [int(v) for v in boxes[box_idx]]
    h, w = line_crop.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        overlay = _render_hierarchy_level_overlay(line_crop, boxes, level_count, selected_idx=-1)
        return overlay, _line_profile_placeholder("Selected block is invalid."), "", "Selected block is invalid.", selected_state

    crop = line_crop[y1:y2, x1:x2]
    overlay = _render_hierarchy_level_overlay(line_crop, boxes, level_count, selected_idx=box_idx)
    selected_state["selected_box_index"] = int(box_idx)

    backbone_raw = (encoder_backbone_path or "").strip()
    if not backbone_raw:
        return (
            overlay,
            crop,
            "",
            "Please select an encoder backbone model.",
            selected_state,
        )

    try:
        runtime_cfg = _load_hierarchy_encoder_runtime_config(backbone_raw, projection_head_path or "")
        width_buckets = _parse_width_buckets_for_encoding(
            raw=runtime_cfg.get("width_buckets", "256,384,512,768"),
            patch_multiple=_to_int(runtime_cfg.get("patch_multiple", 16), 16),
            max_width=_to_int(runtime_cfg.get("max_width", 1024), 1024),
        )
        image_processor, backbone, projection_head = _load_hierarchy_vit_encoder_bundle(
            backbone_raw,
            (projection_head_path or "").strip(),
        )

        mean = list(getattr(image_processor, "image_mean", [0.5, 0.5, 0.5]))
        std = list(getattr(image_processor, "image_std", [0.5, 0.5, 0.5]))
        if len(mean) == 1:
            mean = mean * 3
        if len(std) == 1:
            std = std * 3
        mean = [float(v) for v in mean[:3]]
        std = [max(1e-6, float(v)) for v in std[:3]]

        norm_img = _normalize_for_vit_encoding(
            image=Image.fromarray(crop.astype(np.uint8)).convert("RGB"),
            target_height=_to_int(runtime_cfg.get("target_height", 64), 64),
            width_buckets=width_buckets,
            patch_multiple=_to_int(runtime_cfg.get("patch_multiple", 16), 16),
            max_width=_to_int(runtime_cfg.get("max_width", 1024), 1024),
        )
        arr = np.asarray(norm_img).astype(np.float32) / 255.0
        ten = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()
        mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        pixel_values = ((ten - mean_t) / std_t).unsqueeze(0)

        resolved_device, device_note = _resolve_torch_device_for_encoding(encoder_device)
        backbone = backbone.to(resolved_device)
        backbone.eval()
        if projection_head is not None:
            projection_head = projection_head.to(resolved_device)
            projection_head.eval()

        with torch.no_grad():
            emb = _pooled_image_embedding_for_ui(backbone, pixel_values.to(resolved_device))
            if projection_head is not None:
                emb = projection_head(emb)
            if bool(l2_normalize) and torch_f is not None:
                emb = torch_f.normalize(emb, dim=-1)
        vec = emb[0].detach().cpu().float().tolist()
        decimals = max(2, min(9, _to_int(vector_decimals, 6)))
        vec_round = [round(float(v), decimals) for v in vec]
        vec_norm = float(np.linalg.norm(np.asarray(vec, dtype=np.float32)))

        vector_payload = {
            "line_id": int(selected_state.get("line_id", -1)),
            "line_box": [int(v) for v in (selected_state.get("line_box") or [0, 0, 0, 0])],
            "hierarchy_level": int(level_count),
            "block_index_1based": int(box_idx + 1),
            "block_box_local": [int(x1), int(y1), int(x2), int(y2)],
            "crop_size": {"width": int(x2 - x1), "height": int(y2 - y1)},
            "encoder_backbone": str(Path(backbone_raw).expanduser().resolve()),
            "projection_head": str(Path(projection_head_path).expanduser().resolve()) if (projection_head_path or "").strip() else "",
            "device_used": resolved_device,
            "embedding_dim": int(len(vec_round)),
            "embedding_norm_l2": round(vec_norm, 6),
            "l2_normalized": bool(l2_normalize),
            "normalization": {
                "target_height": int(runtime_cfg.get("target_height", 64)),
                "max_width": int(runtime_cfg.get("max_width", 1024)),
                "patch_multiple": int(runtime_cfg.get("patch_multiple", 16)),
                "width_buckets": [int(v) for v in width_buckets],
                "config_source": str(runtime_cfg.get("source", "defaults")),
            },
            "vector": vec_round,
        }
        vector_text = json.dumps(vector_payload, ensure_ascii=False, indent=2)
        msg = (
            f"Encoded block {int(box_idx + 1)}/{len(boxes)} (level {int(level_count)}), "
            f"dim={len(vec_round)}, device={resolved_device}"
        )
        if device_note:
            msg = f"{msg}. {device_note}"
        return overlay, crop, vector_text, msg, selected_state
    except Exception as exc:
        return overlay, crop, "", f"Encoding failed: {type(exc).__name__}: {exc}", selected_state


def _compute_line_projection_state(
    crop_rgb: np.ndarray,
    projection_smooth: int,
    projection_threshold_rel: float,
) -> Dict[str, Any]:
    if cv2 is None:
        return {"ok": False, "reason": "opencv_missing"}
    if crop_rgb is None or crop_rgb.size == 0:
        return {"ok": False, "reason": "empty_crop"}

    h, w = crop_rgb.shape[:2]
    if h < 4 or w < 4:
        return {"ok": False, "reason": "crop_too_small"}

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, open_kernel, iterations=1)

    kernel_w = max(3, int(round(w * 0.04)))
    kernel_w = min(kernel_w, max(1, w))
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    joined = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, line_kernel, iterations=1)

    projection = (joined > 0).sum(axis=1).astype(np.float32)
    if projection.size == 0 or float(np.max(projection)) <= 0.0:
        return {"ok": False, "reason": "empty_projection"}

    smooth = max(1, int(projection_smooth))
    if smooth % 2 == 0:
        smooth += 1
    if smooth > 1:
        conv = np.ones((smooth,), dtype=np.float32) / float(smooth)
        projection = np.convolve(projection, conv, mode="same")

    threshold_rel = min(max(float(projection_threshold_rel), 0.01), 0.95)
    threshold = max(1.0, float(np.max(projection)) * threshold_rel)
    mask = projection >= threshold

    return {
        "ok": True,
        "bw": bw,
        "projection": projection,
        "threshold": float(threshold),
        "mask": mask,
        "height": int(h),
        "width": int(w),
        "projection_smooth": int(smooth),
        "projection_threshold_rel": float(threshold_rel),
    }


def _line_runs_from_threshold_mask(
    mask: np.ndarray,
    min_line_height: int,
    merge_gap_px: int,
) -> List[Tuple[int, int]]:
    if mask is None or mask.size == 0:
        return []

    h = int(mask.shape[0])
    min_h = max(3, int(min_line_height))
    runs: List[Tuple[int, int]] = []
    y = 0
    while y < h:
        if not bool(mask[y]):
            y += 1
            continue
        y1 = y
        while y < h and bool(mask[y]):
            y += 1
        y2 = y
        if y2 - y1 >= min_h:
            runs.append((y1, y2))

    if not runs:
        return []

    merged: List[Tuple[int, int]] = []
    merge_gap = max(0, int(merge_gap_px))
    for y1, y2 in runs:
        if not merged:
            merged.append((y1, y2))
            continue
        py1, py2 = merged[-1]
        if y1 - py2 <= merge_gap:
            merged[-1] = (py1, y2)
        else:
            merged.append((y1, y2))
    return merged


def _segment_lines_in_text_crop(
    crop_rgb: np.ndarray,
    min_line_height: int,
    projection_smooth: int,
    projection_threshold_rel: float,
    merge_gap_px: int,
    projection_state: Optional[Dict[str, Any]] = None,
) -> List[Tuple[int, int, int, int]]:
    state = projection_state
    if state is None:
        state = _compute_line_projection_state(
            crop_rgb=crop_rgb,
            projection_smooth=int(projection_smooth),
            projection_threshold_rel=float(projection_threshold_rel),
        )
    if not bool(state.get("ok")):
        return []

    bw = state["bw"]
    h = int(state["height"])
    w = int(state["width"])
    projection = np.asarray(state["projection"], dtype=np.float32)
    threshold = float(state["threshold"])
    mask = state["mask"]

    min_h = max(3, int(min_line_height))
    line_runs: List[Tuple[int, int]] = []
    peaks_used: List[int] = []

    if find_peaks is not None and projection.size >= 5:
        maxv = float(np.max(projection))
        distance = max(min_h, int(round(h * 0.06)))
        prominence_rel = min(0.30, max(0.06, float(projection_threshold_rel) * 0.55))
        prominence = max(2.0, maxv * prominence_rel)
        peaks, _ = find_peaks(
            projection,
            distance=distance,
            prominence=prominence,
        )
        if peaks.size > 0:
            strong = projection[peaks] >= max(1.0, threshold * 1.05)
            if bool(np.any(strong)):
                peaks = peaks[strong]

        if peaks.size > 0:
            peaks = np.sort(peaks.astype(np.int32))
            peaks_used = [int(p) for p in peaks.tolist()]

            boundaries: List[int] = [0]
            for i in range(len(peaks) - 1):
                lo = int(peaks[i])
                hi = int(peaks[i + 1])
                if hi - lo < 2:
                    continue
                valley = lo + int(np.argmin(projection[lo : hi + 1]))
                valley = max(boundaries[-1] + 1, min(h - 1, valley))
                boundaries.append(valley)
            boundaries.append(h)

            if len(boundaries) >= 2:
                for i in range(len(boundaries) - 1):
                    y1 = int(boundaries[i])
                    y2 = int(boundaries[i + 1])
                    if y2 <= y1:
                        continue
                    if y2 - y1 < min_h:
                        if line_runs:
                            py1, py2 = line_runs[-1]
                            line_runs[-1] = (py1, y2)
                        continue
                    line_runs.append((y1, y2))

    if not line_runs:
        line_runs = _line_runs_from_threshold_mask(
            mask=mask,
            min_line_height=min_h,
            merge_gap_px=int(merge_gap_px),
        )
        peaks_used = []

    if not line_runs:
        state["peaks"] = []
        state["line_runs"] = []
        return []

    state["peaks"] = peaks_used
    state["line_runs"] = [[int(y1), int(y2)] for y1, y2 in line_runs]

    candidates: List[Tuple[int, int, int, int]] = []
    for y1, y2 in line_runs:
        sub = bw[y1:y2, :]
        if sub.size == 0:
            continue
        foreground_pixels = int((sub > 0).sum())
        min_foreground = max(24, int((y2 - y1) * w * 0.002))
        if foreground_pixels < min_foreground:
            continue

        cols = np.where((sub > 0).sum(axis=0) > 0)[0]
        if cols.size == 0:
            continue
        x1 = max(0, int(cols[0]) - 2)
        x2 = min(w, int(cols[-1]) + 3)
        if x2 - x1 < 4:
            continue
        candidates.append((x1, y1, x2, y2))

    if not candidates:
        return []

    tightened: List[Tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in candidates:
        sub = bw[y1:y2, x1:x2]
        if sub.size == 0:
            continue
        rows = np.where((sub > 0).sum(axis=1) > 0)[0]
        if rows.size > 0:
            y1 = y1 + int(rows[0])
            y2 = y1 + int(rows[-1] - rows[0] + 1)
        pad = 1
        tx1 = max(0, x1 - pad)
        ty1 = max(0, y1 - pad)
        tx2 = min(w, x2 + pad)
        ty2 = min(h, y2 + pad)
        if tx2 > tx1 and ty2 > ty1:
            tightened.append((tx1, ty1, tx2, ty2))

    return tightened


def _segment_red_runs_in_line_crop(
    line_crop_rgb: np.ndarray,
    min_redness: int,
    min_saturation: int,
    min_column_fill_rel: float,
    merge_gap_px: int,
    min_width_px: int,
) -> List[Tuple[int, int, int, int]]:
    if cv2 is None:
        return []
    if line_crop_rgb is None or line_crop_rgb.size == 0:
        return []

    h, w = line_crop_rgb.shape[:2]
    if h < 3 or w < 3:
        return []

    rgb = line_crop_rgb.astype(np.uint8, copy=False)
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    redness = r - np.maximum(g, b)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    sat = hsv[:, :, 1].astype(np.int16)

    red_floor = max(0, int(min_redness))
    positive = redness[redness > 0]
    if positive.size >= 40:
        red_floor = max(red_floor, int(np.percentile(positive, 72)))

    sat_floor = max(0, int(min_saturation))
    hue_mask = (hue <= 24) | (hue >= 150)
    red_mask = (redness >= red_floor) & (sat >= sat_floor) & hue_mask
    if not bool(np.any(red_mask)):
        return []

    mask = (red_mask.astype(np.uint8) * 255)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    close_w = max(3, int(round(w * 0.02)))
    close_w = min(close_w, max(1, w))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    col_fill = (mask > 0).sum(axis=0).astype(np.float32)
    if col_fill.size == 0 or float(np.max(col_fill)) <= 0.0:
        return []

    smooth = max(1, int(round(w * 0.01)))
    if smooth % 2 == 0:
        smooth += 1
    if smooth > 1:
        conv = np.ones((smooth,), dtype=np.float32) / float(smooth)
        col_fill = np.convolve(col_fill, conv, mode="same")

    col_rel = min(max(float(min_column_fill_rel), 0.01), 0.95)
    col_threshold = max(1.0, float(h) * col_rel)
    active_cols = col_fill >= col_threshold
    if not bool(np.any(active_cols)):
        return []

    runs: List[Tuple[int, int]] = []
    x = 0
    while x < w:
        if not bool(active_cols[x]):
            x += 1
            continue
        x1 = x
        while x < w and bool(active_cols[x]):
            x += 1
        runs.append((x1, x))

    merged: List[Tuple[int, int]] = []
    merge_gap = max(0, int(merge_gap_px))
    for x1, x2 in runs:
        if not merged:
            merged.append((x1, x2))
            continue
        px1, px2 = merged[-1]
        if x1 - px2 <= merge_gap:
            merged[-1] = (px1, x2)
        else:
            merged.append((x1, x2))

    min_width = max(2, int(min_width_px))
    out_boxes: List[Tuple[int, int, int, int]] = []
    for x1, x2 in merged:
        if x2 - x1 < min_width:
            continue
        sub = mask[:, x1:x2]
        if sub.size == 0:
            continue
        rows = np.where((sub > 0).sum(axis=1) > 0)[0]
        cols = np.where((sub > 0).sum(axis=0) > 0)[0]
        if rows.size == 0 or cols.size == 0:
            continue

        rx1 = x1 + int(cols[0])
        rx2 = x1 + int(cols[-1]) + 1
        ry1 = int(rows[0])
        ry2 = int(rows[-1]) + 1
        if rx2 - rx1 < min_width or ry2 - ry1 < 2:
            continue

        pad = 1
        bx1 = max(0, rx1 - pad)
        by1 = max(0, ry1 - pad)
        bx2 = min(w, rx2 + pad)
        by2 = min(h, ry2 + pad)
        if bx2 > bx1 and by2 > by1:
            out_boxes.append((bx1, by1, bx2, by2))

    return out_boxes


def run_tibetan_text_line_split_classical(
    image: np.ndarray,
    model_path: str,
    conf: float,
    imgsz: int,
    device: str,
    min_line_height: int,
    projection_smooth: int,
    projection_threshold_rel: float,
    merge_gap_px: int,
    draw_parent_boxes: bool,
    detect_red_text: bool,
    red_min_redness: int,
    red_min_saturation: int,
    red_column_fill_rel: float,
    red_merge_gap_px: int,
    red_min_width_px: int,
    draw_red_boxes: bool,
):
    empty_profile = _line_profile_placeholder("Run inference to see the projection profile used for line splitting.")
    empty_selected_view = _line_profile_placeholder("Click a line box in the overlay to open the extracted line view.")
    empty_hierarchy_view = _line_profile_placeholder("Click a line box to generate the 2/4/8 wordbox hierarchy.")
    empty_selected_profile = _line_profile_placeholder("Click a line box in the overlay to inspect its horizontal profile.")
    empty_selected_status = "Click a line box in the overlay."
    empty_click_state = {
        "image": None,
        "line_boxes": [],
        "horizontal_profile_smooth_cols": 21,
        "horizontal_profile_threshold_rel": 0.20,
        "horizontal_seg_min_width_px": 14,
        "horizontal_seg_merge_gap_px": 6,
        "line_min_height": 10,
        "line_projection_smooth": 9,
        "line_projection_threshold_rel": 0.20,
        "line_merge_gap_px": 5,
    }
    if image is None:
        return None, "Please provide an image.", "{}", empty_profile, empty_selected_view, empty_hierarchy_view, empty_selected_profile, empty_selected_status, empty_click_state
    if cv2 is None:
        return image, "opencv-python is required for classical line segmentation.", "{}", empty_profile, empty_selected_view, empty_hierarchy_view, empty_selected_profile, empty_selected_status, empty_click_state

    model_file = Path(model_path).expanduser().resolve()
    if not model_file.exists():
        return image, f"Model not found: {model_file}", "{}", empty_profile, empty_selected_view, empty_hierarchy_view, empty_selected_profile, empty_selected_status, empty_click_state

    try:
        model = _load_yolo_model(str(model_file))
        kwargs: Dict[str, Any] = {"conf": float(conf), "imgsz": int(imgsz)}
        if (device or "").strip():
            kwargs["device"] = (device or "").strip()
        results = model.predict(source=image, **kwargs)
    except Exception as exc:
        return image, f"Inference failed: {type(exc).__name__}: {exc}", "{}", empty_profile, empty_selected_view, empty_hierarchy_view, empty_selected_profile, empty_selected_status, empty_click_state

    h, w = image.shape[:2]
    overlay = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    font = _load_overlay_font()

    line_color = (255, 170, 0)
    text_box_color = (0, 220, 255)
    red_box_color = (255, 70, 110)
    unknown_box_color = (190, 190, 190)

    detections: List[Dict[str, Any]] = []
    for res in results:
        if not hasattr(res, "boxes") or res.boxes is None:
            continue
        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else []
        confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else []
        clss = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else []
        names = getattr(res, "names", None) or getattr(model, "names", None) or {}

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            c = float(confs[i]) if i < len(confs) else 0.0
            cls = int(clss[i]) if i < len(clss) else 0
            if isinstance(names, dict):
                class_name = str(names.get(cls, f"class_{cls}"))
            elif isinstance(names, list) and 0 <= cls < len(names):
                class_name = str(names[cls])
            else:
                class_name = f"class_{cls}"

            detections.append(
                {
                    "class": cls,
                    "label": class_name,
                    "confidence": c,
                    "box": [x1, y1, x2, y2],
                }
            )

    tibetan_results: List[Dict[str, Any]] = []
    all_line_count = 0
    all_red_count = 0
    profile_image = _line_profile_placeholder("No tibetan_text box selected.")
    profile_selected_area = -1
    profile_meta: Dict[str, Any] = {}
    line_click_records: List[Dict[str, Any]] = []

    for det in detections:
        cls = int(det["class"])
        label = str(det["label"])
        x1, y1, x2, y2 = [int(v) for v in det["box"]]

        is_tibetan = _is_tibetan_text_detection(cls, label)
        if draw_parent_boxes:
            color = text_box_color if is_tibetan else unknown_box_color
            draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
            tag = f"{label} ({cls}) {float(det['confidence']):.2f}"
            tx, ty = x1 + 2, max(0, y1 - 16)
            draw.rectangle((tx, ty, tx + 9 * len(tag), ty + 14), fill=(0, 0, 0))
            draw.text((tx + 2, ty + 1), tag, fill=color, font=font)

        if not is_tibetan:
            continue

        crop = image[y1:y2, x1:x2]
        line_state = _compute_line_projection_state(
            crop_rgb=crop,
            projection_smooth=int(projection_smooth),
            projection_threshold_rel=float(projection_threshold_rel),
        )
        line_boxes_local = _segment_lines_in_text_crop(
            crop_rgb=crop,
            min_line_height=int(min_line_height),
            projection_smooth=int(projection_smooth),
            projection_threshold_rel=float(projection_threshold_rel),
            merge_gap_px=int(merge_gap_px),
            projection_state=line_state,
        )
        if bool(line_state.get("ok")):
            crop_area = max(1, int((x2 - x1) * (y2 - y1)))
            if crop_area > profile_selected_area:
                projection = np.asarray(line_state.get("projection"), dtype=np.float32)
                profile_image = _render_line_profile_plot(
                    projection=projection,
                    threshold=float(line_state.get("threshold", 0.0)),
                    mask=np.asarray(line_state.get("mask"), dtype=bool),
                    line_boxes_local=line_boxes_local,
                    title=f"Line profile for tibetan_text box [{x1}, {y1}, {x2}, {y2}]",
                    peaks=[int(p) for p in (line_state.get("peaks") or [])],
                )
                profile_selected_area = crop_area
                profile_meta = {
                    "source_text_box": [x1, y1, x2, y2],
                    "projection_length": int(projection.size),
                    "threshold": float(line_state.get("threshold", 0.0)),
                    "projection_smooth_rows": int(line_state.get("projection_smooth", 1)),
                    "projection_threshold_rel": float(line_state.get("projection_threshold_rel", 0.0)),
                    "line_count": len(line_boxes_local),
                    "peak_count": int(len(line_state.get("peaks") or [])),
                    "peaks": [int(p) for p in (line_state.get("peaks") or [])],
                    "line_runs": line_state.get("line_runs"),
                    "line_method": ("scipy_find_peaks" if line_state.get("peaks") else "threshold_mask_fallback"),
                }

        line_boxes_global: List[List[int]] = []
        line_details: List[Dict[str, Any]] = []
        red_boxes_for_text: List[List[int]] = []
        for lx1, ly1, lx2, ly2 in line_boxes_local:
            gx1 = max(0, min(w, x1 + int(lx1)))
            gy1 = max(0, min(h, y1 + int(ly1)))
            gx2 = max(0, min(w, x1 + int(lx2)))
            gy2 = max(0, min(h, y1 + int(ly2)))
            if gx2 <= gx1 or gy2 <= gy1:
                continue
            draw.rectangle((gx1, gy1, gx2, gy2), outline=line_color, width=3)
            all_line_count += 1
            line_tag = f"line {all_line_count}"
            ltx, lty = gx1 + 2, max(0, gy1 - 14)
            draw.rectangle((ltx, lty, ltx + 9 * len(line_tag), lty + 13), fill=(0, 0, 0))
            draw.text((ltx + 2, lty + 1), line_tag, fill=line_color, font=font)
            line_boxes_global.append([gx1, gy1, gx2, gy2])
            line_click_records.append(
                {
                    "line_id": int(all_line_count),
                    "line_box": [gx1, gy1, gx2, gy2],
                    "text_box": [x1, y1, x2, y2],
                    "class": cls,
                    "label": label,
                }
            )

            line_red_boxes_global: List[List[int]] = []
            if bool(detect_red_text):
                line_crop = crop[int(ly1) : int(ly2), int(lx1) : int(lx2)]
                red_boxes_local = _segment_red_runs_in_line_crop(
                    line_crop_rgb=line_crop,
                    min_redness=int(red_min_redness),
                    min_saturation=int(red_min_saturation),
                    min_column_fill_rel=float(red_column_fill_rel),
                    merge_gap_px=int(red_merge_gap_px),
                    min_width_px=int(red_min_width_px),
                )
                for rx1, ry1, rx2, ry2 in red_boxes_local:
                    rgx1 = max(0, min(w, x1 + int(lx1) + int(rx1)))
                    rgy1 = max(0, min(h, y1 + int(ly1) + int(ry1)))
                    rgx2 = max(0, min(w, x1 + int(lx1) + int(rx2)))
                    rgy2 = max(0, min(h, y1 + int(ly1) + int(ry2)))
                    if rgx2 <= rgx1 or rgy2 <= rgy1:
                        continue
                    line_red_boxes_global.append([rgx1, rgy1, rgx2, rgy2])
                    red_boxes_for_text.append([rgx1, rgy1, rgx2, rgy2])
                    all_red_count += 1
                    if draw_red_boxes:
                        draw.rectangle((rgx1, rgy1, rgx2, rgy2), outline=red_box_color, width=3)
                        red_tag = f"red {all_red_count}"
                        rtx, rty = rgx1 + 2, max(0, rgy1 - 14)
                        draw.rectangle((rtx, rty, rtx + 9 * len(red_tag), rty + 13), fill=(0, 0, 0))
                        draw.text((rtx + 2, rty + 1), red_tag, fill=red_box_color, font=font)

            line_details.append(
                {
                    "line_box": [gx1, gy1, gx2, gy2],
                    "red_boxes": line_red_boxes_global,
                }
            )

        tibetan_results.append(
            {
                "class": cls,
                "label": label,
                "confidence": float(det["confidence"]),
                "text_box": [x1, y1, x2, y2],
                "lines": line_boxes_global,
                "line_details": line_details,
                "red_boxes": red_boxes_for_text,
            }
        )

    labels_seen = sorted({str(d["label"]) for d in detections})
    status = (
        f"Detected {len(detections)} boxes. "
        f"Matched {len(tibetan_results)} tibetan_text box(es). "
        f"Extracted {all_line_count} line box(es)."
    )
    if bool(detect_red_text):
        status += f" Found {all_red_count} red segment box(es)."
    if not tibetan_results and labels_seen:
        status += f" Labels found: {', '.join(labels_seen[:8])}"
    if tibetan_results and profile_selected_area < 0:
        profile_image = _line_profile_placeholder(
            "tibetan_text boxes were detected, but projection profile could not be computed."
        )
    if not tibetan_results:
        profile_image = _line_profile_placeholder("No tibetan_text detections. No line profile available.")

    payload = {
        "model_path": str(model_file),
        "image_size": {"width": int(w), "height": int(h)},
        "detections_total": len(detections),
        "tibetan_text_boxes": tibetan_results,
        "line_boxes_total": all_line_count,
        "red_boxes_total": all_red_count,
        "red_detection_enabled": bool(detect_red_text),
        "line_profile_preview": (profile_meta if profile_meta else None),
    }
    click_state = {
        "image": image.astype(np.uint8, copy=False),
        "line_boxes": line_click_records,
        "horizontal_profile_smooth_cols": 21,
        "horizontal_profile_threshold_rel": 0.20,
        "horizontal_seg_min_width_px": 14,
        "horizontal_seg_merge_gap_px": 6,
        "line_min_height": int(min_line_height),
        "line_projection_smooth": int(projection_smooth),
        "line_projection_threshold_rel": float(projection_threshold_rel),
        "line_merge_gap_px": int(merge_gap_px),
    }
    return (
        np.array(overlay),
        status,
        json.dumps(payload, ensure_ascii=False, indent=2),
        profile_image,
        empty_selected_view,
        empty_hierarchy_view,
        empty_selected_profile,
        empty_selected_status,
        click_state,
    )


def download_ppn_images(
    ppn: str,
    output_dir: str,
    max_images: int,
    no_ssl_verify: bool,
    download_workers: int,
):
    if not ppn or not ppn.strip():
        return "Please provide a PPN.", gr.update(choices=[], value=None)

    try:
        from tibetan_utils.sbb_utils import get_images_from_sbb, download_image, get_sbb_metadata
    except Exception as exc:
        return f"Could not import SBB utilities: {type(exc).__name__}: {exc}", gr.update(choices=[], value=None)

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    verify_ssl = not no_ssl_verify

    try:
        urls = get_images_from_sbb(ppn.strip(), verify_ssl=verify_ssl)
        if not urls:
            return f"No images found for PPN {ppn}.", gr.update(choices=[], value=None)
        if int(max_images) > 0:
            urls = urls[: int(max_images)]

        downloaded: List[Path] = []
        workers = max(1, int(download_workers))
        if workers == 1 or len(urls) <= 1:
            for url in urls:
                saved = download_image(url, output_dir=str(out_dir), verify_ssl=verify_ssl)
                if saved:
                    downloaded.append(Path(saved))
        else:
            max_workers = min(workers, len(urls))
            ordered: List[Optional[Path]] = [None] * len(urls)

            def _dl(pair: Tuple[int, str]) -> Tuple[int, Optional[Path]]:
                idx, url = pair
                saved = download_image(url, output_dir=str(out_dir), verify_ssl=verify_ssl)
                return idx, (Path(saved) if saved else None)

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_dl, pair) for pair in enumerate(urls)]
                for fut in as_completed(futures):
                    idx, p = fut.result()
                    ordered[idx] = p

            downloaded = [p for p in ordered if p is not None]

        md = get_sbb_metadata(ppn.strip(), verify_ssl=verify_ssl)
        names = sorted([p.name for p in downloaded])
        summary = (
            f"Downloaded {len(downloaded)} image(s) for PPN {ppn} to {out_dir}\n"
            f"Workers: {workers}\n"
            f"Title: {md.get('title')}\nAuthor: {md.get('author')}\nPages: {md.get('pages')}\n"
        )
        return summary, gr.update(choices=names, value=(names[0] if names else None))
    except Exception as exc:
        return f"PPN download failed: {type(exc).__name__}: {exc}", gr.update(choices=[], value=None)


def preview_downloaded_image(output_dir: str, image_name: str):
    if not output_dir or not image_name:
        return None, "Select an image."
    p = Path(output_dir).expanduser().resolve() / image_name
    if not p.exists():
        return None, f"Not found: {p}"
    img = np.array(Image.open(p).convert("RGB"))
    return img, f"Loaded {p.name}"


def run_layout_on_ppn_list(
    ppn_list_text: str,
    analysis_mode: str,
    model_path: str,
    conf: float,
    imgsz: int,
    device: str,
    parser_choice: str,
    prompt: str,
    hf_model_id: str,
    max_new_tokens: int,
    hf_device: str,
    hf_dtype: str,
    mineru_command: str,
    mineru_timeout: int,
    output_dataset_root: str,
    max_images_per_ppn: int,
    no_ssl_verify: bool,
    save_overlays: bool,
):
    try:
        from tibetan_utils.sbb_utils import get_images_from_sbb, download_image
    except Exception as exc:
        return f"Failed to import SBB utilities: {type(exc).__name__}: {exc}", "", ""

    ppns = [p.strip() for p in (ppn_list_text or "").replace(",", "\n").splitlines() if p.strip()]
    if not ppns:
        return "Please provide at least one PPN.", "", ""

    use_yolo = str(analysis_mode).startswith("YOLO")
    parser_key = _extract_parser_key(parser_choice)

    backend = None
    yolo_model = None

    if use_yolo:
        model_file = Path(model_path).expanduser().resolve()
        if not model_file.exists():
            return f"YOLO model not found: {model_file}", "", ""
        try:
            yolo_model = _load_yolo_model(str(model_file))
        except Exception as exc:
            return f"Could not load YOLO model: {type(exc).__name__}: {exc}", "", ""
    else:
        try:
            from tibetan_utils.parsers import parser_availability
            ok, reason = parser_availability(parser_key)
            if not ok:
                return f"Backend `{parser_key}` unavailable: {reason}", "", ""
        except Exception as exc:
            return f"Availability check failed: {type(exc).__name__}: {exc}", "", ""

    out_root = Path(output_dataset_root).expanduser().resolve()
    split_dir = out_root / "test"
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    raw_dir = split_dir / "raw_json"
    overlays_dir = split_dir / "overlays"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if save_overlays:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    (split_dir / "classes.txt").write_text(
        "tibetan_number_word\ntibetan_text\nchinese_number_word\n",
        encoding="utf-8",
    )

    if not use_yolo:
        backend = _build_vlm_backend(
            parser_key=parser_key,
            hf_model_id=(hf_model_id or "").strip(),
            prompt=(prompt or "").strip(),
            max_new_tokens=int(max_new_tokens),
            hf_device=(hf_device or "auto").strip(),
            hf_dtype=(hf_dtype or "auto").strip(),
            mineru_command=(mineru_command or "mineru").strip(),
            mineru_timeout=int(mineru_timeout),
        )

    verify_ssl = not no_ssl_verify
    processed = 0
    failed = 0
    logs: List[str] = []

    for ppn in ppns:
        try:
            urls = get_images_from_sbb(ppn, verify_ssl=verify_ssl)
            if int(max_images_per_ppn) > 0:
                urls = urls[: int(max_images_per_ppn)]
            logs.append(f"PPN {ppn}: {len(urls)} image url(s)")
        except Exception as exc:
            failed += 1
            logs.append(f"PPN {ppn}: metadata failed ({type(exc).__name__}: {exc})")
            continue

        for idx, url in enumerate(urls, start=1):
            try:
                saved = download_image(url, output_dir=str(images_dir), verify_ssl=verify_ssl)
                if not saved:
                    failed += 1
                    logs.append(f"PPN {ppn} #{idx}: download failed")
                    continue
                src = Path(saved)
                ppn_name = f"PPN{ppn}_{src.name}"
                dst = images_dir / ppn_name
                if src.name != ppn_name:
                    src.rename(dst)
                else:
                    dst = src

                img = np.array(Image.open(dst).convert("RGB"))
                if use_yolo:
                    kwargs: Dict[str, Any] = {"conf": float(conf), "imgsz": int(imgsz)}
                    if (device or "").strip():
                        kwargs["device"] = (device or "").strip()
                    results = yolo_model.predict(source=img, **kwargs)
                    detections = []
                    h, w = img.shape[:2]
                    for res in results:
                        if not hasattr(res, "boxes") or res.boxes is None:
                            continue
                        boxes = res.boxes
                        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else []
                        confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else []
                        clss = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else []
                        for i in range(len(xyxy)):
                            x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
                            c = float(confs[i]) if i < len(confs) else 0.0
                            cls = int(clss[i]) if i < len(clss) else 1
                            detections.append({
                                "class": cls,
                                "label": f"class_{cls}",
                                "confidence": c,
                                "text": "",
                                "box": {
                                    "x": (x1 + x2) / 2.0,
                                    "y": (y1 + y2) / 2.0,
                                    "width": max(1.0, x2 - x1),
                                    "height": max(1.0, y2 - y1),
                                },
                            })
                    out = {
                        "image_name": dst.name,
                        "parser": "yolo_pretrained",
                        "detections": detections,
                        "metadata": {"model_path": str(model_file)},
                    }
                else:
                    doc = backend.parse(img, output_dir=None, image_name=dst.name)
                    out = doc.to_dict()
                    detections = out.get("detections", [])

                yolo_lines: List[str] = []
                h, w = img.shape[:2]
                for det in detections:
                    cls = _map_detection_class(det)
                    yolo_box = _detection_to_yolo(det.get("box", {}), img_w=w, img_h=h)
                    if yolo_box is None:
                        continue
                    cx, cy, bw, bh = yolo_box
                    yolo_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                (labels_dir / f"{dst.stem}.txt").write_text(
                    "\n".join(yolo_lines) + ("\n" if yolo_lines else ""),
                    encoding="utf-8",
                )
                (raw_dir / f"{dst.stem}.json").write_text(
                    json.dumps(out, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                if save_overlays:
                    overlay = _render_detected_regions(img, detections)
                    Image.fromarray(overlay).save(overlays_dir / f"{dst.stem}.png")

                processed += 1
            except Exception as exc:
                failed += 1
                logs.append(f"PPN {ppn} #{idx}: failed ({type(exc).__name__}: {exc})")

    summary = (
        f"Processed images: {processed}\n"
        f"Failed images: {failed}\n"
        f"Label backend: {'YOLO pretrained' if use_yolo else parser_key}\n"
        f"Output test split: {split_dir}\n"
        "Note: SBB data is stored as TEST-only (split=test), not train/val."
    )
    return summary, str(split_dir), "\n".join(logs[:200])


def prepare_combined_labelstudio_split(
    synthetic_split_dir: str,
    sbb_test_split_dir: str,
    combined_output_split_dir: str,
    sbb_train_ratio: float = 0.7,
    seed: int = 42,
):
    syn = Path(synthetic_split_dir).expanduser().resolve()
    sbb = Path(sbb_test_split_dir).expanduser().resolve()
    out = Path(combined_output_split_dir).expanduser().resolve()
    out_root = out.parent if out.name.lower() in {"train", "val", "test"} else out
    out_train_images = out_root / "train" / "images"
    out_train_labels = out_root / "train" / "labels"
    out_val_images = out_root / "val" / "images"
    out_val_labels = out_root / "val" / "labels"
    out_train_images.mkdir(parents=True, exist_ok=True)
    out_train_labels.mkdir(parents=True, exist_ok=True)
    out_val_images.mkdir(parents=True, exist_ok=True)
    out_val_labels.mkdir(parents=True, exist_ok=True)

    if not (syn / "images").exists() or not (syn / "labels").exists():
        return f"Synthetic split invalid: {syn}", ""
    if not (sbb / "images").exists() or not (sbb / "labels").exists():
        return f"SBB test split invalid: {sbb}", ""

    sbb_class_id_map, sbb_class_map_msg = _infer_ls_class_id_map(sbb)

    ratio = min(max(float(sbb_train_ratio), 0.0), 1.0)

    def _copy_split(src_split: Path, prefix: str, dst_images: Path, dst_labels: Path):
        cnt = 0
        converted = 0
        for img in sorted((src_split / "images").glob("*")):
            if not img.is_file():
                continue
            new_name = f"{prefix}_{img.name}"
            target_img = dst_images / new_name
            target_lbl = dst_labels / f"{Path(new_name).stem}.txt"
            lbl_src = src_split / "labels" / f"{img.stem}.txt"
            target_img.write_bytes(img.read_bytes())
            if lbl_src.exists():
                bbox_lines = _normalize_to_bbox_lines(lbl_src, img)
                target_lbl.write_text("\n".join(bbox_lines) + ("\n" if bbox_lines else ""), encoding="utf-8")
                converted += 1
            else:
                target_lbl.write_text("", encoding="utf-8")
            cnt += 1
        return cnt, converted

    syn_to_val = syn.name.lower() == "val"
    syn_dst_images = out_val_images if syn_to_val else out_train_images
    syn_dst_labels = out_val_labels if syn_to_val else out_train_labels
    syn_count, syn_converted = _copy_split(syn, "syn", syn_dst_images, syn_dst_labels)

    sbb_images = sorted([p for p in (sbb / "images").glob("*") if p.is_file()])
    rng = np.random.default_rng(int(seed))
    if sbb_images:
        perm = list(rng.permutation(len(sbb_images)))
        sbb_images = [sbb_images[i] for i in perm]
    split_idx = int(round(len(sbb_images) * ratio))
    split_idx = max(0, min(split_idx, len(sbb_images)))
    sbb_train = sbb_images[:split_idx]
    sbb_val = sbb_images[split_idx:]

    def _copy_sbb_subset(images: List[Path], dst_images: Path, dst_labels: Path):
        cnt = 0
        converted = 0
        for img in images:
            new_name = f"sbb_{img.name}"
            target_img = dst_images / new_name
            target_lbl = dst_labels / f"{Path(new_name).stem}.txt"
            lbl_src = sbb / "labels" / f"{img.stem}.txt"
            target_img.write_bytes(img.read_bytes())
            if lbl_src.exists():
                bbox_lines = _normalize_to_bbox_lines(lbl_src, img, class_id_map=sbb_class_id_map)
                target_lbl.write_text("\n".join(bbox_lines) + ("\n" if bbox_lines else ""), encoding="utf-8")
                converted += 1
            else:
                target_lbl.write_text("", encoding="utf-8")
            cnt += 1
        return cnt, converted

    sbb_train_count, sbb_train_converted = _copy_sbb_subset(sbb_train, out_train_images, out_train_labels)
    sbb_val_count, sbb_val_converted = _copy_sbb_subset(sbb_val, out_val_images, out_val_labels)
    sbb_count = sbb_train_count + sbb_val_count
    sbb_converted = sbb_train_converted + sbb_val_converted

    (out_root / "classes.txt").write_text(
        "tibetan_number_word\ntibetan_text\nchinese_number_word\n",
        encoding="utf-8",
    )
    yaml_path = out_root.parent / f"{out_root.name}.yaml"
    data_yaml = {
        "path": str(out_root),
        "train": "train/images",
        "val": "val/images",
        "test": "",
        "nc": 3,
        "names": {
            0: "tibetan_number_word",
            1: "tibetan_text",
            2: "chinese_number_word",
        },
    }
    yaml_path.write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    msg = (
        f"Combined dataset created at {out_root}\n"
        f"Synthetic images copied: {syn_count} -> {'val' if syn_to_val else 'train'}\n"
        f"SBB images split train/val: {sbb_train_count}/{sbb_val_count} (ratio={ratio:.2f}, seed={int(seed)})\n"
        f"SBB total copied: {sbb_count}\n"
        f"{sbb_class_map_msg}\n"
        f"Label files normalized to YOLO bbox format: syn={syn_converted}, sbb={sbb_converted}\n"
        f"Wrote dataset YAML: {yaml_path}\n"
        f"Train split: {out_root / 'train'}\n"
        f"Val split: {out_root / 'val'}"
    )
    return msg, str((out_root / "train").resolve())


def prepare_labelstudio_from_ls_export_ui(
    ls_export_dir: str,
    ls_export_zip: Any,
    synthetic_dataset: str,
    datasets_base_dir: str,
    combined_output_split_dir: str,
    current_split_dir: str,
    current_local_files_root: str,
    current_image_root_url: str,
    current_tasks_json: str,
    current_vlm_export_split_dir: str,
    current_vlm_export_image_root_url: str,
    current_vlm_export_tasks_json: str,
):
    syn_split, syn_msg = _resolve_dataset_train_split(synthetic_dataset, datasets_base_dir)
    if syn_split is None:
        return (
            f"Prepare failed: {syn_msg}",
            "",
            current_split_dir,
            current_local_files_root,
            current_image_root_url,
            current_tasks_json,
            current_vlm_export_split_dir,
            current_vlm_export_image_root_url,
            current_vlm_export_tasks_json,
        )

    ls_split, ls_msg = _resolve_ls_export_split_from_path_or_zip(ls_export_dir, ls_export_zip)
    if ls_split is None:
        return (
            f"Prepare failed: {ls_msg}",
            "",
            current_split_dir,
            current_local_files_root,
            current_image_root_url,
            current_tasks_json,
            current_vlm_export_split_dir,
            current_vlm_export_image_root_url,
            current_vlm_export_tasks_json,
        )

    ls_label_stats = _inspect_label_format(ls_split)
    ls_label_msg = _label_format_summary(ls_label_stats)

    msg, combined = prepare_combined_labelstudio_split(
        synthetic_split_dir=str(syn_split),
        sbb_test_split_dir=str(ls_split),
        combined_output_split_dir=combined_output_split_dir,
    )
    if not combined:
        return (
            f"{syn_msg}\n{ls_msg}\n{ls_label_msg}\n{msg}",
            "",
            current_split_dir,
            current_local_files_root,
            current_image_root_url,
            current_tasks_json,
            current_vlm_export_split_dir,
            current_vlm_export_image_root_url,
            current_vlm_export_tasks_json,
        )

    combined_path = Path(combined).expanduser().resolve()
    local_files_root = str(combined_path.parent)
    image_root_url = f"/data/local-files/?d={combined_path.name}/images"
    tasks_json = str((ROOT / "ls-tasks-combined-ui.json").resolve())
    msg2 = f"{syn_msg}\n{ls_msg}\n{ls_label_msg}\n\n{msg}\n\nLabel Studio fields updated in tab 8."

    return (
        msg2,
        str(combined_path),
        str(combined_path),
        local_files_root,
        image_root_url,
        tasks_json,
        str(combined_path),
        image_root_url,
        tasks_json,
    )


def prepare_combined_for_labelstudio_ui(
    synthetic_split_dir: str,
    sbb_test_split_dir: str,
    sbb_export_zip: Any,
    combined_output_split_dir: str,
    sbb_train_ratio: float,
    split_seed: int,
    current_split_dir: str,
    current_local_files_root: str,
    current_image_root_url: str,
    current_tasks_json: str,
    current_vlm_export_split_dir: str,
    current_vlm_export_image_root_url: str,
    current_vlm_export_tasks_json: str,
):
    resolved_sbb_split, sbb_msg = _resolve_ls_export_split_from_path_or_zip(sbb_test_split_dir, sbb_export_zip)
    if resolved_sbb_split is None:
        return (
            f"Combine failed: {sbb_msg}",
            "",
            current_split_dir,
            current_local_files_root,
            current_image_root_url,
            current_tasks_json,
            current_vlm_export_split_dir,
            current_vlm_export_image_root_url,
            current_vlm_export_tasks_json,
        )

    msg, combined = prepare_combined_labelstudio_split(
        synthetic_split_dir=synthetic_split_dir,
        sbb_test_split_dir=str(resolved_sbb_split),
        combined_output_split_dir=combined_output_split_dir,
        sbb_train_ratio=float(sbb_train_ratio),
        seed=int(split_seed),
    )
    if not combined:
        return (
            msg,
            "",
            current_split_dir,
            current_local_files_root,
            current_image_root_url,
            current_tasks_json,
            current_vlm_export_split_dir,
            current_vlm_export_image_root_url,
            current_vlm_export_tasks_json,
        )

    combined_path = Path(combined).expanduser().resolve()
    local_files_root = str(combined_path.parent)
    image_root_url = f"/data/local-files/?d={combined_path.name}/images"
    tasks_json = str((ROOT / "ls-tasks-combined-ui.json").resolve())
    msg2 = f"{sbb_msg}\n{msg}\n\nLabel Studio fields updated in tab 3."

    return (
        msg2,
        str(combined_path),
        str(combined_path),
        local_files_root,
        image_root_url,
        tasks_json,
        str(combined_path),
        image_root_url,
        tasks_json,
    )


def scan_pretrained_models(models_dir: str):
    base = Path(models_dir).expanduser().resolve()
    if not base.exists():
        return gr.update(choices=[], value=None), f"Models directory not found: {base}"
    exts = {".pt", ".torchscript", ".onnx"}
    models = sorted([str(p.resolve()) for p in base.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    return gr.update(choices=models, value=(models[0] if models else None)), f"Found {len(models)} model(s) in {base}"


def scan_yolo_split_dirs(base_dir: str):
    base = Path(base_dir).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        return gr.update(choices=[], value=None), f"Directory not found: {base}"

    found: List[str] = []
    if (base / "images").exists() and (base / "labels").exists():
        found.append(str(base.resolve()))
    # Typical split roots first.
    for split in ("train", "val", "test"):
        candidate = base / split
        if (candidate / "images").exists() and (candidate / "labels").exists():
            found.append(str(candidate.resolve()))
    # Recursive fallback.
    for p in sorted(base.rglob("*")):
        if not p.is_dir():
            continue
        if (p / "images").exists() and (p / "labels").exists():
            path_str = str(p.resolve())
            if path_str not in found:
                found.append(path_str)

    return gr.update(choices=found, value=(found[0] if found else None)), f"Found {len(found)} split dir(s) in {base}"


def scan_ultralytics_inference_models(models_dir: str):
    base = Path(models_dir).expanduser().resolve()
    if not base.exists():
        return gr.update(choices=[], value=None), f"Models directory not found: {base}"

    exts = {".pt", ".torchscript", ".onnx"}
    blocked_exts = {".safetensor", ".safetensors"}
    models: List[str] = []
    ignored_lora = 0

    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        sfx = p.suffix.lower()
        if sfx in blocked_exts:
            ignored_lora += 1
            continue
        if sfx not in exts:
            continue
        lname = p.name.lower()
        if "lora" in lname:
            ignored_lora += 1
            continue
        models.append(str(p.resolve()))

    return (
        gr.update(choices=models, value=(models[0] if models else None)),
        f"Found {len(models)} Ultralytics model(s) in {base}. Ignored LoRA-like files: {ignored_lora}.",
    )


def scan_lora_models(models_dir: str):
    base = Path(models_dir).expanduser().resolve()
    if not base.exists():
        return gr.update(choices=[], value=None), "", f"Models directory not found: {base}"

    exts = {".safetensors", ".safetensor"}
    models = sorted([str(p.resolve()) for p in base.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    selected = models[0] if models else ""
    msg = f"Found {len(models)} LoRA safetensor model(s) in {base}"
    return gr.update(choices=models, value=(selected if selected else None)), selected, msg


def _list_sbb_grid_images(sbb_dir: Path) -> List[Path]:
    if not sbb_dir.exists():
        return []
    images: List[Path] = []
    for p in _list_images_recursive(sbb_dir):
        try:
            rel_parts = [part.lower() for part in p.relative_to(sbb_dir).parts]
        except Exception:
            rel_parts = [part.lower() for part in p.parts]
        if "quarantine" in rel_parts:
            continue
        images.append(p)
    return sorted(images)


def _sbb_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return path.name


def _load_sbb_thumbnail(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        rgb.thumbnail((SBB_GRID_THUMB_SIZE, SBB_GRID_THUMB_SIZE), resample=resample)
        canvas = Image.new("RGB", (SBB_GRID_THUMB_SIZE, SBB_GRID_THUMB_SIZE), (245, 245, 245))
        ox = max(0, (SBB_GRID_THUMB_SIZE - rgb.width) // 2)
        oy = max(0, (SBB_GRID_THUMB_SIZE - rgb.height) // 2)
        canvas.paste(rgb, (ox, oy))
    return np.array(canvas)


def _get_sbb_thumbnail_cached(path: Path) -> np.ndarray:
    key = str(path.resolve())
    try:
        mtime = float(path.stat().st_mtime)
    except Exception:
        mtime = 0.0

    cached = _SBB_THUMB_CACHE.get(key)
    if cached is not None:
        cached_mtime, cached_img = cached
        if abs(cached_mtime - mtime) < 1e-6:
            return cached_img

    thumb = _load_sbb_thumbnail(path)
    _SBB_THUMB_CACHE[key] = (mtime, thumb)

    # Keep cache bounded to avoid memory blowup on very large collections.
    if len(_SBB_THUMB_CACHE) > SBB_THUMB_CACHE_MAX:
        for old_key in list(_SBB_THUMB_CACHE.keys())[: len(_SBB_THUMB_CACHE) - SBB_THUMB_CACHE_MAX]:
            _SBB_THUMB_CACHE.pop(old_key, None)
    return thumb


def _prefetch_sbb_page_thumbnails(paths: List[Path]) -> None:
    if not paths:
        return
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(_get_sbb_thumbnail_cached, p) for p in paths]
        for fut in futures:
            try:
                fut.result()
            except Exception:
                continue


def _get_sbb_mean_cached(path: Path) -> float:
    key = str(path.resolve())
    try:
        mtime = float(path.stat().st_mtime)
    except Exception:
        mtime = 0.0

    cached = _SBB_MEAN_CACHE.get(key)
    if cached is not None:
        cached_mtime, cached_mean = cached
        if abs(cached_mtime - mtime) < 1e-6:
            return float(cached_mean)

    thumb = _get_sbb_thumbnail_cached(path)
    mean_val = float(np.mean(thumb))
    _SBB_MEAN_CACHE[key] = (mtime, mean_val)
    return mean_val


def _sort_sbb_images(image_paths: List[Path], sort_mode: str) -> List[Path]:
    mode = (sort_mode or "name").strip().lower()
    if mode != "mean_color":
        return sorted(image_paths)

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_get_sbb_mean_cached, p): p for p in image_paths}
        pairs: List[Tuple[float, Path]] = []
        for fut, path in futures.items():
            try:
                pairs.append((float(fut.result()), path))
            except Exception:
                pairs.append((255.0, path))
    pairs.sort(key=lambda x: (x[0], str(x[1])))
    return [p for _, p in pairs]


def _sbb_grid_render_page(
    sbb_images_dir: str,
    page_index: int,
    note: str = "",
    sort_mode: str = "name",
) -> List[Any]:
    base = Path(sbb_images_dir).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        status = f"Folder not found: {base}"
        outputs: List[Any] = [status, 0, []]
        for _ in range(SBB_GRID_PAGE_SIZE):
            outputs.extend([
                gr.update(value=None, visible=False),
                gr.update(value=False, visible=False),
                gr.update(value="", visible=False),
            ])
        return outputs

    all_images = _sort_sbb_images(_list_sbb_grid_images(base), sort_mode=sort_mode)
    total = len(all_images)
    if total == 0:
        status = f"No images found in {base} (excluding quarantine)."
        outputs = [status, 0, []]
        for _ in range(SBB_GRID_PAGE_SIZE):
            outputs.extend([
                gr.update(value=None, visible=False),
                gr.update(value=False, visible=False),
                gr.update(value="", visible=False),
            ])
        return outputs

    total_pages = (total + SBB_GRID_PAGE_SIZE - 1) // SBB_GRID_PAGE_SIZE
    page = max(0, min(int(page_index), total_pages - 1))
    start = page * SBB_GRID_PAGE_SIZE
    end = min(total, start + SBB_GRID_PAGE_SIZE)
    page_paths = all_images[start:end]
    next_start = end
    next_end = min(total, next_start + SBB_GRID_PAGE_SIZE)
    next_page_paths = all_images[next_start:next_end]

    prefix = f"{note}\n" if note else ""
    mode_text = "mean_color (dark->bright)" if (sort_mode or "name").lower() == "mean_color" else "name"
    status = f"{prefix}Page {page + 1}/{total_pages} | showing {start + 1}-{end} of {total} | sort={mode_text}"
    outputs = [status, page, [str(p) for p in page_paths], sort_mode]

    for i in range(SBB_GRID_PAGE_SIZE):
        if i < len(page_paths):
            image_path = page_paths[i]
            rel = _sbb_rel(image_path, base)
            caption = f"{start + i + 1}/{total} - {rel}"
            try:
                thumb = _get_sbb_thumbnail_cached(image_path)
                img_update = gr.update(value=thumb, visible=True)
            except Exception as exc:
                img_update = gr.update(value=None, visible=False)
                caption = f"{caption} (preview error: {type(exc).__name__})"
            outputs.extend([
                img_update,
                gr.update(value=False, visible=True),
                gr.update(value=caption, visible=True),
            ])
        else:
            outputs.extend([
                gr.update(value=None, visible=False),
                gr.update(value=False, visible=False),
                gr.update(value="", visible=False),
            ])

    # Prefetch next page thumbnails so "Vor" displays instantly.
    _prefetch_sbb_page_thumbnails(next_page_paths)
    return outputs


def sbb_grid_refresh(sbb_images_dir: str, sort_mode: str):
    return _sbb_grid_render_page(
        sbb_images_dir=sbb_images_dir,
        page_index=0,
        note="Refreshed.",
        sort_mode=sort_mode,
    )


def sbb_grid_prev(sbb_images_dir: str, current_page: int, sort_mode: str):
    return _sbb_grid_render_page(
        sbb_images_dir=sbb_images_dir,
        page_index=int(current_page) - 1,
        sort_mode=sort_mode,
    )


def sbb_grid_next(sbb_images_dir: str, current_page: int, sort_mode: str):
    return _sbb_grid_render_page(
        sbb_images_dir=sbb_images_dir,
        page_index=int(current_page) + 1,
        sort_mode=sort_mode,
    )


def sbb_grid_first(sbb_images_dir: str, sort_mode: str):
    return _sbb_grid_render_page(
        sbb_images_dir=sbb_images_dir,
        page_index=0,
        sort_mode=sort_mode,
    )


def sbb_grid_last(sbb_images_dir: str, sort_mode: str):
    base = Path(sbb_images_dir).expanduser().resolve()
    all_images = _sort_sbb_images(_list_sbb_grid_images(base), sort_mode=sort_mode)
    total = len(all_images)
    if total <= 0:
        return _sbb_grid_render_page(
            sbb_images_dir=sbb_images_dir,
            page_index=0,
            sort_mode=sort_mode,
        )
    last_page = max(0, (total - 1) // SBB_GRID_PAGE_SIZE)
    return _sbb_grid_render_page(
        sbb_images_dir=sbb_images_dir,
        page_index=last_page,
        sort_mode=sort_mode,
    )


def sbb_grid_go_to_page(sbb_images_dir: str, page_number: int, sort_mode: str):
    try:
        page_int = int(float(page_number))
    except Exception:
        page_int = 1
    # UI input is 1-based; internal is 0-based.
    page_index = max(0, page_int - 1)
    return _sbb_grid_render_page(
        sbb_images_dir=sbb_images_dir,
        page_index=page_index,
        sort_mode=sort_mode,
    )


def sbb_grid_sort_by_mean(sbb_images_dir: str):
    return _sbb_grid_render_page(
        sbb_images_dir=sbb_images_dir,
        page_index=0,
        note="Sorted by mean color value (dark -> bright).",
        sort_mode="mean_color",
    )


def sbb_grid_quarantine(
    sbb_images_dir: str,
    current_page: int,
    current_paths: List[str],
    sort_mode: str,
    *checked_flags,
):
    base = Path(sbb_images_dir).expanduser().resolve()
    quarantine_root = base / "quarantine"
    moved = 0
    skipped = 0
    failed = 0

    selected_paths = list(current_paths or [])
    for path_str, checked in zip(selected_paths, checked_flags):
        if not checked:
            continue
        src = Path(path_str).expanduser().resolve()
        if not src.exists():
            skipped += 1
            continue
        try:
            rel = src.relative_to(base)
        except Exception:
            skipped += 1
            continue
        if rel.parts and rel.parts[0].lower() == "quarantine":
            skipped += 1
            continue
        dst = quarantine_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst = dst.with_name(f"{dst.stem}_{int(time.time())}{dst.suffix}")
        try:
            shutil.move(str(src), str(dst))
            moved += 1
        except Exception:
            failed += 1

    note = f"Quarantine moved={moved}, skipped={skipped}, failed={failed} -> {quarantine_root}"
    return _sbb_grid_render_page(
        sbb_images_dir=sbb_images_dir,
        page_index=int(current_page),
        note=note,
        sort_mode=sort_mode,
    )


def run_ppn_image_analysis(
    output_dir: str,
    image_name: str,
    analysis_mode: str,
    model_path: str,
    conf: float,
    imgsz: int,
    device: str,
    parser_choice: str,
    prompt: str,
    hf_model_id: str,
    max_new_tokens: int,
    hf_device: str,
    hf_dtype: str,
    mineru_command: str,
    mineru_timeout: int,
):
    if not output_dir or not image_name:
        return None, "Select a downloaded image first.", "{}"
    p = Path(output_dir).expanduser().resolve() / image_name
    if not p.exists():
        return None, f"Downloaded image not found: {p}", "{}"

    img = np.array(Image.open(p).convert("RGB"))
    if analysis_mode == "YOLO (models/)":
        overlay, status, det_json = run_trained_model_inference(
            image=img,
            model_path=model_path,
            conf=conf,
            imgsz=imgsz,
            device=device,
        )
        return overlay, status, det_json

    overlay, status, out_json = run_vlm_layout_inference(
        image=img,
        parser_choice=parser_choice,
        prompt=prompt,
        hf_model_id=hf_model_id,
        max_new_tokens=max_new_tokens,
        hf_device=hf_device,
        hf_dtype=hf_dtype,
        mineru_command=mineru_command,
        mineru_timeout=mineru_timeout,
    )
    return overlay, status, out_json


def build_ui() -> gr.Blocks:
    preferred_root = Path("/home/ubuntu/data/PechaBridge").resolve()
    workspace_root = preferred_root if preferred_root.exists() else ROOT

    default_dataset_base = str((workspace_root / "datasets").resolve())
    default_dataset = str((workspace_root / "datasets" / "tibetan-yolo").resolve())
    default_split_dir = str((workspace_root / "datasets" / "tibetan-yolo" / "train").resolve())
    default_donut_dataset_name = "tibetan-donut-ocr-label1"
    default_donut_dataset_output_dir = str((workspace_root / "datasets").resolve())
    default_donut_model_output_dir = str((workspace_root / "models" / "donut-ocr-label1").resolve())
    default_donut_font_tibetan = str((workspace_root / "ext" / "Microsoft Himalaya.ttf").resolve())
    default_donut_font_chinese = str((workspace_root / "ext" / "simkai.ttf").resolve())
    default_texture_input_dir = str((workspace_root / "datasets" / "tibetan-yolo-ui" / "train" / "images").resolve())
    default_texture_output_dir = str((workspace_root / "datasets" / "tibetan-yolo-ui-textured").resolve())
    default_texture_real_pages_dir = str((workspace_root / "sbb_images").resolve())
    default_sbb_grid_dir = str((workspace_root / "sbb_images").resolve())
    default_text_hierarchy_dir = str((workspace_root / "datasets" / "text_patches").resolve())
    default_texture_lora_dataset_dir = str((workspace_root / "datasets" / "texture-lora-dataset").resolve())
    default_texture_lora_output_dir = str((workspace_root / "models" / "texture-lora-sdxl").resolve())
    default_image_encoder_input_dir = str((workspace_root / "sbb_images").resolve())
    default_image_encoder_output_dir = str((workspace_root / "models" / "image-encoder").resolve())
    default_text_hierarchy_backbone_dir = str((workspace_root / "models" / "text_hierarchy_vit" / "text_hierarchy_vit_backbone").resolve())
    default_text_hierarchy_projection_head = str((workspace_root / "models" / "text_hierarchy_vit" / "text_hierarchy_projection_head.pt").resolve())
    default_text_hierarchy_eval_dir = str((workspace_root / "models" / "text_hierarchy_vit" / "eval").resolve())
    default_text_hierarchy_faiss_dir = str((workspace_root / "models" / "text_hierarchy_vit" / "faiss_search").resolve())
    default_text_hierarchy_faiss_index = str((workspace_root / "models" / "text_hierarchy_vit" / "faiss_search" / "text_patches.faiss").resolve())
    default_prompt = (
        "Extract page layout blocks and OCR text. "
        "Return strict JSON with key 'detections' containing a list of objects with: "
        "text, label, confidence, and bbox=[x0,y0,x1,y1]."
    )
    vlm_choices = _format_vlm_parser_choices()

    with gr.Blocks(title="PechaBridge Workbench") as demo:
        gr.Markdown("# PechaBridge Workbench")
        gr.Markdown(
            "Use this UI from data generation to model inference, plus VLM parsing and SBB downloads."
        )

        def _scan_dataset_dirs(base: str):
            choices = _list_datasets(base)
            return gr.update(choices=choices, value=(choices[0] if choices else None))

        def _scan_train_datasets(base: str):
            choices = _list_dataset_names(base)
            return gr.update(choices=choices, value=(choices[0] if choices else None))

        # 12b) Minimal end-to-end page -> lines -> DONUT OCR preview
        with gr.Tab("12b. Page -> Lines -> DONUT OCR"):
            gr.Markdown(
                "Minimal workflow: entweder Seite hochladen (Textbox-Erkennung + Line-Segmentierung + DONUT OCR) "
                "oder direkt ein einzelnes Line-Image hochladen (nur BDRC-Vorverarbeitung + DONUT OCR). "
                "Die Vorverarbeitung orientiert sich am DONUT-Train-Script (`bdrc`, Helper wird bevorzugt)."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    lineocr_input_mode = gr.Radio(
                        choices=["Page -> Lines -> DONUT", "Single Line -> DONUT"],
                        value="Page -> Lines -> DONUT",
                        label="Input Mode",
                    )
                    lineocr_image = gr.Image(type="numpy", label="Page Image (for page mode)", sources=["upload", "clipboard"])
                    lineocr_direct_line_image = gr.Image(
                        type="numpy",
                        label="Direct Line Image (for single-line mode)",
                        sources=["upload", "clipboard"],
                    )
                    with gr.Accordion("Model Selection", open=True):
                        lineocr_line_models_dir = gr.Textbox(
                            label="Scan line models dir",
                            value=str((workspace_root / "models").resolve()),
                        )
                        with gr.Row():
                            lineocr_line_scan_btn = gr.Button("Scan Line Models")
                            lineocr_line_scan_status = gr.Textbox(label="Line Scan Status", interactive=False)
                        lineocr_line_model_select = gr.Dropdown(
                            choices=[],
                            value=None,
                            label="Detected Ultralytics line/text model",
                            allow_custom_value=True,
                        )
                        lineocr_line_model = gr.Textbox(
                            label="line_model_path",
                            value=str((ROOT / "runs" / "detect" / "train" / "weights" / "best.pt").resolve()),
                        )

                        lineocr_donut_models_dir = gr.Textbox(
                            label="Scan DONUT models dir",
                            value=str((workspace_root / "models").resolve()),
                        )
                        with gr.Row():
                            lineocr_donut_scan_btn = gr.Button("Scan DONUT Models")
                            lineocr_donut_scan_status = gr.Textbox(label="DONUT Scan Status", interactive=False)
                        lineocr_donut_model_select = gr.Dropdown(
                            choices=[],
                            value=None,
                            label="Detected DONUT/TroCR model",
                            allow_custom_value=True,
                        )
                        lineocr_donut_model_dir = gr.Textbox(
                            label="donut_model_dir",
                            value=default_donut_model_output_dir,
                        )

                    with gr.Accordion("Advanced (optional)", open=False):
                        with gr.Row():
                            lineocr_line_conf = gr.Slider(0.01, 0.99, value=0.25, step=0.01, label="line_conf")
                            lineocr_line_imgsz = gr.Number(label="line_imgsz", value=1024, precision=0)
                        lineocr_line_device = gr.Textbox(label="line_device", value="")
                        with gr.Row():
                            lineocr_min_line_height = gr.Number(label="min_line_height_px", value=10, precision=0)
                            lineocr_merge_gap = gr.Number(label="line_merge_gap_px", value=5, precision=0)
                        with gr.Row():
                            lineocr_proj_smooth = gr.Number(label="projection_smooth_rows", value=9, precision=0)
                            lineocr_proj_thresh = gr.Slider(
                                0.05,
                                0.80,
                                value=0.20,
                                step=0.01,
                                label="projection_threshold_rel",
                            )
                        with gr.Row():
                            lineocr_donut_preproc = gr.Textbox(
                                label="donut_preprocess (fixed in Quick tab)",
                                value="bdrc",
                                interactive=False,
                            )
                            lineocr_donut_device = gr.Dropdown(
                                choices=["auto", "cuda:0", "cuda:1", "cuda:2", "cuda:3", "cpu"],
                                value="auto",
                                label="donut_device",
                            )
                        with gr.Row():
                            lineocr_donut_max_len = gr.Number(label="generation_max_length", value=512, precision=0)
                            lineocr_donut_beams = gr.Number(label="num_beams", value=1, precision=0)
                        lineocr_max_lines = gr.Number(label="max_lines (0=all)", value=0, precision=0)

                    lineocr_run_btn = gr.Button("Run Quick OCR (Page/Line)", variant="primary")

                with gr.Column(scale=1):
                    lineocr_overlay = gr.Image(type="numpy", label="Preview (Page Overlay or Preprocessed Line)")
                    lineocr_status = gr.Textbox(label="Status", lines=3, interactive=False)
                    lineocr_copy_text = gr.Textbox(label="Line Text (Copy/Paste)", lines=10)
                    lineocr_cards_html = gr.HTML(
                        value=_render_line_ocr_cards_html_ui(
                            [],
                            note="Noch nicht ausgeführt.",
                        )
                    )
                    with gr.Accordion("Debug JSON", open=False):
                        lineocr_debug_json = gr.Code(label="Workflow Debug JSON", language="json")

            lineocr_line_scan_btn.click(
                fn=scan_ultralytics_inference_models,
                inputs=[lineocr_line_models_dir],
                outputs=[lineocr_line_model_select, lineocr_line_scan_status],
            )
            lineocr_line_model_select.change(
                fn=lambda x: x or "",
                inputs=[lineocr_line_model_select],
                outputs=[lineocr_line_model],
            )
            lineocr_donut_scan_btn.click(
                fn=scan_donut_ocr_models_ui,
                inputs=[lineocr_donut_models_dir],
                outputs=[lineocr_donut_model_select, lineocr_donut_model_dir, lineocr_donut_scan_status],
            )
            lineocr_donut_model_select.change(
                fn=lambda x: x or "",
                inputs=[lineocr_donut_model_select],
                outputs=[lineocr_donut_model_dir],
            )
            lineocr_run_btn.click(
                fn=run_page_line_donut_ocr_preview_ui,
                inputs=[
                    lineocr_input_mode,
                    lineocr_image,
                    lineocr_direct_line_image,
                    lineocr_line_model,
                    lineocr_line_conf,
                    lineocr_line_imgsz,
                    lineocr_line_device,
                    lineocr_min_line_height,
                    lineocr_proj_smooth,
                    lineocr_proj_thresh,
                    lineocr_merge_gap,
                    lineocr_donut_model_dir,
                    lineocr_donut_preproc,
                    lineocr_donut_device,
                    lineocr_donut_max_len,
                    lineocr_donut_beams,
                    lineocr_max_lines,
                ],
                outputs=[
                    lineocr_overlay,
                    lineocr_status,
                    lineocr_copy_text,
                    lineocr_cards_html,
                    lineocr_debug_json,
                ],
            )

        # 1) Hello / workflow overview
        with gr.Tab("1. Hello"):
            gr.Markdown("## Workflow Overview")
            gr.Markdown(
                "Use the tabs left-to-right. A practical flow is: Synthetic Data -> Diffusion + LoRA (texture) -> "
                "Donut OCR Workflow -> Retrieval Encoders -> Batch VLM Layout (SBB) -> Dataset Preview -> Ultralytics Training -> Model Inference -> "
                "Tibetan Line Split (CV) -> Patch Dataset (gen-patches) -> Hierarchy Encode Preview -> VLM Layout (single image) -> Label Studio Export."
            )
            gr.Markdown("### Tabs")
            gr.Markdown(
                "1. Hello: Short guide and workflow.\n"
                "2. Synthetic Data: Generate synthetic training/validation data.\n"
                "3. Batch VLM Layout (SBB): Batch-annotate SBB PPNs, combine with synthetic data, export.\n"
                "4. Dataset Preview: Visual QA with bounding boxes.\n"
                "5. Ultralytics Training: Train YOLO models.\n"
                "6. Model Inference: Run inference with trained models.\n"
                "6b. Tibetan Line Split (CV): Detect `tibetan_text` boxes, split into lines, and optionally detect red text boxes per line.\n"
                "6c. Patch Dataset Debug View: Generate and browse `datasets/text_patches` (`patches/`, `meta/patches.parquet`, `debug/`).\n"
                "6d. Hierarchy Encode Preview: Detect lines, build 2/4/8 hierarchy, click a block, and output latent vector.\n"
                "7. VLM Layout: Run transformer-based layout parsing on a single image.\n"
                "8. Label Studio Export: Convert YOLO split folders to Label Studio tasks and launch Label Studio.\n"
                "9. PPN Downloader: Download and analyze SBB images.\n"
                "10. SBB Grid Review: Browse sbb_images in 9x9 pages and move selected pages to quarantine.\n"
                "11. Diffusion + LoRA: Prepare texture crops, train LoRA, and run SDXL/SD2.1 + ControlNet inference.\n"
                "12. Donut OCR Workflow: Run synthetic generation + manifest prep + Donut-style OCR training on label 1.\n"
                "13. Retrieval Encoders: Train + evaluate hierarchy encoders and run FAISS similarity search.\n"
                "14. CLI Audit: Show all CLI options from project scripts."
            )

        # 2) Data generation
        with gr.Tab("2. Synthetic Data"):
            gr.Markdown("Generate synthetic multi-class YOLO data using `generate_training_data.py`.")
            with gr.Row():
                with gr.Column():
                    background_train = gr.Textbox(label="background_train", value="./data/tibetan numbers/backgrounds/")
                    background_val = gr.Textbox(label="background_val", value="./data/tibetan numbers/backgrounds/")
                    output_dir = gr.Textbox(label="output_dir", value="./datasets")
                    dataset_name = gr.Textbox(label="dataset_name", value="tibetan-yolo-ui")
                    corp_tib_num = gr.Textbox(label="corpora_tibetan_numbers_path", value="./data/corpora/Tibetan Number Words/")
                    corp_tib_text = gr.Textbox(label="corpora_tibetan_text_path", value="./data/corpora/UVA Tibetan Spoken Corpus/")
                    corp_chi_num = gr.Textbox(label="corpora_chinese_numbers_path", value="./data/corpora/Chinese Number Words/")
                with gr.Column():
                    train_samples = gr.Number(label="train_samples", value=100, precision=0)
                    val_samples = gr.Number(label="val_samples", value=100, precision=0)
                    font_tib = gr.Textbox(label="font_path_tibetan", value="ext/Microsoft Himalaya.ttf")
                    font_chi = gr.Textbox(label="font_path_chinese", value="ext/simkai.ttf")
                    image_width = gr.Number(label="image_width", value=1024, precision=0)
                    image_height = gr.Number(label="image_height", value=361, precision=0)
                    augmentation = gr.Dropdown(label="augmentation", choices=["rotate", "noise", "none"], value="noise")
                    annotations_file_path = gr.Textbox(label="annotations_file_path", value="./data/tibetan numbers/annotations/tibetan_chinese_no/bg_PPN337138764X_00000005.txt")
                    single_label = gr.Checkbox(label="single_label", value=False)
                    debug = gr.Checkbox(label="debug", value=False)

            generate_btn = gr.Button("Generate Dataset", variant="primary")
            gen_log = gr.Textbox(label="Generation Log", lines=18)
            generated_dataset_path = gr.Textbox(label="Generated Dataset Path", interactive=False)
            gen_live_preview = gr.Image(label="Live Preview (Latest Generated + BBoxes)", type="numpy")
            gen_live_preview_status = gr.Textbox(label="Live Preview Status", lines=6, interactive=False)
            generate_btn.click(
                fn=run_generate_synthetic_live,
                inputs=[
                    background_train,
                    background_val,
                    output_dir,
                    dataset_name,
                    corp_tib_num,
                    corp_tib_text,
                    corp_chi_num,
                    train_samples,
                    val_samples,
                    font_tib,
                    font_chi,
                    image_width,
                    image_height,
                    augmentation,
                    annotations_file_path,
                    single_label,
                    debug,
                ],
                outputs=[gen_log, generated_dataset_path, gen_live_preview, gen_live_preview_status],
            )

        # 3) Batch VLM on SBB + combine/export
        with gr.Tab("3. Batch VLM Layout (SBB)"):
            gr.Markdown("### Batch VLM Layout on SBB PPNs (test-only)")
            gr.Markdown(
                "Annotate images from one or more PPNs with the selected VLM backend. "
                "Output is stored as a YOLO-like `test` split for review/export only."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    batch_analysis_mode = gr.Radio(
                        choices=["VLM", "YOLO (pretrained)"],
                        value="VLM",
                        label="Batch Labeling Backend",
                    )
                    with gr.Row():
                        batch_models_dir = gr.Textbox(label="models_dir (YOLO)", value=str((ROOT / "models").resolve()))
                        batch_scan_models_btn = gr.Button("Scan models/")
                    batch_model_select = gr.Dropdown(label="Pretrained Model (YOLO)", choices=[])
                    batch_model_scan_msg = gr.Textbox(label="Model Scan Status", interactive=False)
                    with gr.Row():
                        batch_yolo_conf = gr.Slider(0.01, 0.99, value=0.25, step=0.01, label="conf (YOLO)")
                        batch_yolo_imgsz = gr.Number(label="imgsz (YOLO)", value=1024, precision=0)
                    batch_yolo_device = gr.Textbox(label="device (YOLO)", value="cuda:0")

                    batch_vlm_parser = gr.Dropdown(
                        label="Parser Backend",
                        choices=vlm_choices,
                        value=(vlm_choices[0] if vlm_choices else None),
                    )
                    batch_vlm_prompt = gr.Textbox(label="Prompt", value=default_prompt, lines=5)
                    batch_vlm_hf_model = gr.Textbox(label="HF Model ID Override (optional)", value="")
                    with gr.Row():
                        batch_vlm_max_new_tokens = gr.Slider(64, 4096, value=1024, step=32, label="Max New Tokens")
                        batch_vlm_hf_device = gr.Dropdown(choices=["auto", "cpu", "cuda"], value="auto", label="HF Device")
                    batch_vlm_hf_dtype = gr.Dropdown(
                        choices=["auto", "float16", "bfloat16", "float32"],
                        value="auto",
                        label="HF DType",
                    )
                    with gr.Accordion("MinerU Options", open=False):
                        batch_vlm_mineru_command = gr.Textbox(label="MinerU Command", value="mineru")
                        batch_vlm_mineru_timeout = gr.Number(label="MinerU Timeout (s)", value=300, precision=0)
                with gr.Column(scale=1):
                    batch_vlm_ppn_list = gr.Textbox(
                        label="PPN List (comma or newline separated)",
                        value="337138764X",
                        lines=4,
                    )
                    batch_vlm_ppn_output_root = gr.Textbox(
                        label="Output dataset root",
                        value=str((ROOT / "datasets" / "sbb-vlm-layout").resolve()),
                    )
                    with gr.Row():
                        batch_vlm_ppn_max_images = gr.Number(label="max_images_per_ppn (0=all)", value=5, precision=0)
                        batch_vlm_ppn_no_ssl = gr.Checkbox(label="no_ssl_verify", value=False)
                    batch_vlm_ppn_save_overlays = gr.Checkbox(label="save_overlays", value=True)
                    batch_vlm_ppn_run_btn = gr.Button("Run Batch Labeling on PPN List", variant="primary")

            batch_vlm_ppn_summary = gr.Textbox(label="Batch Status", lines=8, interactive=False)
            batch_vlm_ppn_test_split = gr.Textbox(label="Generated SBB Test Split", interactive=False)
            batch_vlm_ppn_logs = gr.Textbox(label="Batch Logs", lines=10, interactive=False)

            batch_scan_models_btn.click(
                fn=scan_pretrained_models,
                inputs=[batch_models_dir],
                outputs=[batch_model_select, batch_model_scan_msg],
            )

            batch_vlm_ppn_run_btn.click(
                fn=run_layout_on_ppn_list,
                inputs=[
                    batch_vlm_ppn_list,
                    batch_analysis_mode,
                    batch_model_select,
                    batch_yolo_conf,
                    batch_yolo_imgsz,
                    batch_yolo_device,
                    batch_vlm_parser,
                    batch_vlm_prompt,
                    batch_vlm_hf_model,
                    batch_vlm_max_new_tokens,
                    batch_vlm_hf_device,
                    batch_vlm_hf_dtype,
                    batch_vlm_mineru_command,
                    batch_vlm_mineru_timeout,
                    batch_vlm_ppn_output_root,
                    batch_vlm_ppn_max_images,
                    batch_vlm_ppn_no_ssl,
                    batch_vlm_ppn_save_overlays,
                ],
                outputs=[batch_vlm_ppn_summary, batch_vlm_ppn_test_split, batch_vlm_ppn_logs],
            )

            gr.Markdown("### Combine Synthetic + SBB Test for Label Studio")
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        batch_vlm_combine_syn_split = gr.Textbox(
                            label="Synthetic split directory (train or val)",
                            value=str((ROOT / "datasets" / "tibetan-yolo" / "train").resolve()),
                        )
                        batch_vlm_combine_syn_scan_btn = gr.Button("Scan Synthetic", variant="secondary")
                    batch_vlm_combine_syn_select = gr.Dropdown(
                        label="Detected synthetic split dirs",
                        choices=[],
                        value=None,
                        allow_custom_value=True,
                    )
                    with gr.Row():
                        batch_vlm_combine_sbb_split = gr.Textbox(
                            label="SBB split directory",
                            value=str((ROOT / "datasets" / "sbb-vlm-layout" / "test").resolve()),
                        )
                        batch_vlm_combine_sbb_scan_btn = gr.Button("Scan SBB", variant="secondary")
                    batch_vlm_combine_sbb_zip = gr.File(
                        label="Label Studio export ZIP (optional)",
                        file_types=[".zip"],
                        type="filepath",
                    )
                    batch_vlm_combine_sbb_select = gr.Dropdown(
                        label="Detected SBB split dirs",
                        choices=[],
                        value=None,
                        allow_custom_value=True,
                    )
                    with gr.Row():
                        batch_vlm_sbb_train_ratio = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.7,
                            step=0.01,
                            label="SBB train ratio (val = 1-ratio)",
                        )
                        batch_vlm_split_seed = gr.Number(label="split_seed", value=42, precision=0)
                    batch_vlm_combine_out_split = gr.Textbox(
                        label="Combined output root (creates train/ and val/)",
                        value=str((ROOT / "datasets" / "combined-layout-labelstudio").resolve()),
                    )
                    batch_vlm_combine_btn = gr.Button("Prepare Combined Split + Fill Label Studio Tab", variant="secondary")
                with gr.Column(scale=1):
                    batch_vlm_combine_status = gr.Textbox(label="Combine Status", lines=8, interactive=False)
                    batch_vlm_combine_result = gr.Textbox(label="Combined Split Path", interactive=False)
                    batch_vlm_combine_scan_status = gr.Textbox(label="Split Scan Status", interactive=False)

            batch_vlm_ppn_test_split.change(
                fn=lambda x: x,
                inputs=[batch_vlm_ppn_test_split],
                outputs=[batch_vlm_combine_sbb_split],
            )
            batch_vlm_combine_syn_scan_btn.click(
                fn=scan_yolo_split_dirs,
                inputs=[batch_vlm_combine_syn_split],
                outputs=[batch_vlm_combine_syn_select, batch_vlm_combine_scan_status],
            )
            batch_vlm_combine_sbb_scan_btn.click(
                fn=scan_yolo_split_dirs,
                inputs=[batch_vlm_combine_sbb_split],
                outputs=[batch_vlm_combine_sbb_select, batch_vlm_combine_scan_status],
            )
            batch_vlm_combine_syn_select.change(
                fn=lambda x: x or "",
                inputs=[batch_vlm_combine_syn_select],
                outputs=[batch_vlm_combine_syn_split],
            )
            batch_vlm_combine_sbb_select.change(
                fn=lambda x: x or "",
                inputs=[batch_vlm_combine_sbb_select],
                outputs=[batch_vlm_combine_sbb_split],
            )

            gr.Markdown("### Export to Label Studio (from this tab)")
            with gr.Row():
                with gr.Column(scale=1):
                    batch_vlm_export_split_dir = gr.Textbox(
                        label="YOLO Split Directory",
                        value=str((ROOT / "datasets" / "combined-layout-labelstudio" / "train").resolve()),
                    )
                    batch_vlm_export_image_ext = gr.Dropdown(
                        label="image-ext",
                        choices=[".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"],
                        value=".png",
                    )
                with gr.Column(scale=1):
                    batch_vlm_export_tasks_json = gr.Textbox(
                        label="tasks json output",
                        value=str((ROOT / "ls-tasks-combined-ui.json").resolve()),
                    )
                    batch_vlm_export_image_root_url = gr.Textbox(
                        label="image-root-url",
                        value="/data/local-files/?d=train/images",
                    )
            batch_vlm_export_btn = gr.Button("Export to Label Studio", variant="secondary")
            batch_vlm_export_log = gr.Textbox(label="Batch VLM Export Log", lines=10, interactive=False)

            batch_vlm_combine_result.change(
                fn=lambda x: x,
                inputs=[batch_vlm_combine_result],
                outputs=[batch_vlm_export_split_dir],
            )
            batch_vlm_export_btn.click(
                fn=export_to_label_studio,
                inputs=[
                    batch_vlm_export_split_dir,
                    batch_vlm_export_image_ext,
                    batch_vlm_export_tasks_json,
                    batch_vlm_export_image_root_url,
                ],
                outputs=[batch_vlm_export_log],
            )

        # 4) Visual QA
        with gr.Tab("4. Dataset Preview"):
            gr.Markdown("Inspect generated dataset and render YOLO label boxes.")
            with gr.Row():
                dataset_base = gr.Textbox(label="Datasets Base Directory", value=default_dataset_base)
                scan_datasets_btn = gr.Button("Scan Datasets")
            dataset_select = gr.Dropdown(label="Dataset Directory", choices=[default_dataset], value=default_dataset)
            split_select = gr.Dropdown(label="Split", choices=["train", "val"], value="train")
            with gr.Row():
                refresh_images_btn = gr.Button("Refresh Image List")
                image_select = gr.Dropdown(label="Image", choices=[])
            preview_hint = gr.Textbox(label="Preview Status", interactive=False)
            preview_btn = gr.Button("Render Preview", variant="primary")
            with gr.Row():
                prev_img_btn = gr.Button("Zurueck")
                next_img_btn = gr.Button("Vor")
            preview_img = gr.Image(label="Image with Label Boxes", type="numpy")
            preview_txt = gr.Textbox(label="Label Summary", lines=12, interactive=False)

            scan_datasets_btn.click(fn=_scan_dataset_dirs, inputs=[dataset_base], outputs=[dataset_select])
            refresh_images_btn.click(
                fn=refresh_image_list,
                inputs=[dataset_select, split_select],
                outputs=[image_select, preview_hint],
            )
            preview_btn.click(
                fn=preview_sample,
                inputs=[dataset_select, split_select, image_select],
                outputs=[preview_img, preview_txt],
            )
            prev_img_btn.click(
                fn=lambda d, s, i: preview_adjacent_sample(d, s, i, -1),
                inputs=[dataset_select, split_select, image_select],
                outputs=[image_select, preview_img, preview_txt],
            )
            next_img_btn.click(
                fn=lambda d, s, i: preview_adjacent_sample(d, s, i, 1),
                inputs=[dataset_select, split_select, image_select],
                outputs=[image_select, preview_img, preview_txt],
            )

        # 5) Training
        with gr.Tab("5. Ultralytics Training"):
            gr.Markdown("Train a detection model via `train_model.py`.")
            train_dataset_choices = _list_dataset_names(default_dataset_base)
            default_train_dataset = train_dataset_choices[0] if train_dataset_choices else "tibetan-yolo"
            train_model_presets = _ultralytics_model_presets()

            with gr.Row():
                train_dataset_base = gr.Textbox(label="Datasets Base Directory", value=default_dataset_base)
                train_scan_btn = gr.Button("Scan Training Datasets")
            train_dataset = gr.Dropdown(
                label="dataset (name, folder path, or .yaml path)",
                choices=train_dataset_choices,
                value=default_train_dataset,
                allow_custom_value=True,
            )
            with gr.Row():
                train_models_dir = gr.Textbox(label="Models Directory", value=str((ROOT / "models").resolve()))
                train_scan_models_btn = gr.Button("Scan Models")
            train_model = gr.Dropdown(
                label="model",
                choices=train_model_presets,
                value=train_model_presets[0],
                allow_custom_value=True,
            )
            train_model_scan_msg = gr.Textbox(label="Model Scan Status", interactive=False)
            train_model_override = gr.Textbox(
                label="model_override (optional)",
                value="",
                placeholder="Optional explicit path or model id; overrides selected model",
            )

            with gr.Row():
                with gr.Column():
                    train_epochs = gr.Number(label="epochs", value=100, precision=0)
                    train_batch = gr.Number(label="batch", value=16, precision=0)
                    train_imgsz = gr.Number(label="imgsz", value=1024, precision=0)
                with gr.Column():
                    train_workers = gr.Number(label="workers", value=8, precision=0)
                    train_device = gr.Textbox(label="device", value="cuda:0")
                    train_project = gr.Textbox(label="project", value=str((workspace_root / "runs" / "detect").resolve()))
                    train_name = gr.Textbox(label="name", value="train-ui")
                    train_patience = gr.Number(label="patience", value=50, precision=0)
                    train_export = gr.Checkbox(label="export", value=True)

            with gr.Accordion("Weights & Biases", open=False):
                train_wandb = gr.Checkbox(label="wandb", value=False)
                train_wandb_project = gr.Textbox(label="wandb_project", value="PechaBridge")
                train_wandb_entity = gr.Textbox(label="wandb_entity", value="")
                train_wandb_tags = gr.Textbox(label="wandb_tags", value="")
                train_wandb_name = gr.Textbox(label="wandb_name", value="")

            train_run_btn = gr.Button("Start Training", variant="primary")
            train_log = gr.Textbox(label="Training Log", lines=18)
            train_best_model = gr.Textbox(label="Best Model Path", interactive=False)

            train_scan_btn.click(fn=_scan_train_datasets, inputs=[train_dataset_base], outputs=[train_dataset])
            train_scan_models_btn.click(
                fn=scan_ultralytics_models,
                inputs=[train_models_dir],
                outputs=[train_model, train_model_scan_msg],
            )
            train_run_btn.click(
                fn=run_ultralytics_train_from_ui,
                inputs=[
                    train_dataset,
                    train_model,
                    train_model_override,
                    train_epochs,
                    train_batch,
                    train_imgsz,
                    train_workers,
                    train_device,
                    train_project,
                    train_name,
                    train_patience,
                    train_export,
                    train_wandb,
                    train_wandb_project,
                    train_wandb_entity,
                    train_wandb_tags,
                    train_wandb_name,
                ],
                outputs=[train_log, train_best_model],
            )

        # 6) Inference with trained model
        with gr.Tab("6. Model Inference"):
            gr.Markdown("Run inference with a trained Ultralytics model and preview detections.")
            with gr.Row():
                with gr.Column(scale=1):
                    infer_models_dir = gr.Textbox(label="models_dir", value=str((workspace_root / "models").resolve()))
                    infer_scan_models_btn = gr.Button("Scan Models")
                    infer_model_select = gr.Dropdown(label="Detected Ultralytics Model", choices=[], allow_custom_value=True)
                    infer_model_scan_msg = gr.Textbox(label="Model Scan Status", interactive=False)
                    infer_model = gr.Textbox(label="model_path", value=str((ROOT / "runs" / "detect" / "train" / "weights" / "best.pt").resolve()))
                    infer_conf = gr.Slider(0.01, 0.99, value=0.25, step=0.01, label="conf")
                    infer_imgsz = gr.Number(label="imgsz", value=1024, precision=0)
                    infer_device = gr.Textbox(label="device", value="")
                    infer_btn = gr.Button("Run Inference", variant="primary")
                with gr.Column(scale=1):
                    infer_image_in = gr.Image(type="numpy", label="Input Image", sources=["upload", "clipboard"])
                    infer_image_out = gr.Image(type="numpy", label="Predictions Overlay")
                    infer_status = gr.Textbox(label="Status", interactive=False)
                    infer_json = gr.Code(label="Detections JSON", language="json")

            infer_btn.click(
                fn=run_trained_model_inference,
                inputs=[infer_image_in, infer_model, infer_conf, infer_imgsz, infer_device],
                outputs=[infer_image_out, infer_status, infer_json],
            )
            infer_scan_models_btn.click(
                fn=scan_ultralytics_inference_models,
                inputs=[infer_models_dir],
                outputs=[infer_model_select, infer_model_scan_msg],
            )
            infer_model_select.change(
                fn=lambda x: x or "",
                inputs=[infer_model_select],
                outputs=[infer_model],
            )

        # 6b) Classical CV line splitting inside tibetan_text boxes
        with gr.Tab("6b. Tibetan Line Split (CV)"):
            gr.Markdown(
                "Upload an image, run YOLO detection, then split only `tibetan_text` boxes into line boxes "
                "using classical image processing (thresholding + horizontal projection). "
                "Optionally, detect red annotations as separate boxes inside each detected line."
            )
            line_click_state = gr.State({})
            with gr.Row():
                with gr.Column(scale=1):
                    line_models_dir = gr.Textbox(label="models_dir", value=str((workspace_root / "models").resolve()))
                    line_scan_models_btn = gr.Button("Scan Models")
                    line_model_select = gr.Dropdown(label="Detected Ultralytics Model", choices=[], allow_custom_value=True)
                    line_model_scan_msg = gr.Textbox(label="Model Scan Status", interactive=False)
                    line_model = gr.Textbox(
                        label="model_path",
                        value=str((ROOT / "runs" / "detect" / "train" / "weights" / "best.pt").resolve()),
                    )
                    line_conf = gr.Slider(0.01, 0.99, value=0.25, step=0.01, label="conf")
                    line_imgsz = gr.Number(label="imgsz", value=1024, precision=0)
                    line_device = gr.Textbox(label="device", value="")

                    with gr.Accordion("Classical Segmentation Parameters", open=False):
                        line_min_height = gr.Number(label="min_line_height_px", value=10, precision=0)
                        line_projection_smooth = gr.Number(label="projection_smooth_rows", value=9, precision=0)
                        line_projection_thresh = gr.Slider(
                            minimum=0.05,
                            maximum=0.80,
                            value=0.20,
                            step=0.01,
                            label="projection_threshold_rel",
                        )
                        line_merge_gap = gr.Number(label="merge_gap_px", value=5, precision=0)
                        line_draw_parents = gr.Checkbox(label="draw_detected_text_boxes", value=True)

                    with gr.Accordion("Red Text Detection (per line)", open=False):
                        line_detect_red = gr.Checkbox(label="detect_red_text_in_lines", value=True)
                        line_draw_red_boxes = gr.Checkbox(label="draw_red_text_boxes", value=True)
                        line_red_min_redness = gr.Number(label="red_min_redness (R-max(G,B))", value=26, precision=0)
                        line_red_min_saturation = gr.Number(label="red_min_saturation", value=35, precision=0)
                        line_red_col_fill_rel = gr.Slider(
                            minimum=0.01,
                            maximum=0.50,
                            value=0.07,
                            step=0.01,
                            label="red_column_fill_rel",
                        )
                        line_red_merge_gap = gr.Number(label="red_merge_gap_px", value=14, precision=0)
                        line_red_min_width = gr.Number(label="red_min_width_px", value=18, precision=0)

                    line_run_btn = gr.Button("Split Tibetan Text into Lines", variant="primary")

                with gr.Column(scale=1):
                    line_image_in = gr.Image(type="numpy", label="Input Image", sources=["upload", "clipboard"])
                    line_image_out = gr.Image(type="numpy", label="Overlay (Text + Line + Red Boxes)", interactive=True)
                    line_profile_out = gr.Image(type="numpy", label="Line Projection Profile (used for segmentation)")
                    line_selected_view = gr.Image(type="numpy", label="Selected Line Block + Peak-Boxes (same algorithm)")
                    line_hierarchy_view = gr.Image(type="numpy", label="Selected Line Wordbox Hierarchy (2/4/8)")
                    line_selected_profile = gr.Image(type="numpy", label="Selected Line Horizontal Profile (click in overlay)")
                    line_click_status = gr.Textbox(label="Clicked Line", interactive=False)
                    line_status = gr.Textbox(label="Status", interactive=False)
                    line_json = gr.Code(label="Line Segmentation JSON", language="json")

            line_run_btn.click(
                fn=run_tibetan_text_line_split_classical,
                inputs=[
                    line_image_in,
                    line_model,
                    line_conf,
                    line_imgsz,
                    line_device,
                    line_min_height,
                    line_projection_smooth,
                    line_projection_thresh,
                    line_merge_gap,
                    line_draw_parents,
                    line_detect_red,
                    line_red_min_redness,
                    line_red_min_saturation,
                    line_red_col_fill_rel,
                    line_red_merge_gap,
                    line_red_min_width,
                    line_draw_red_boxes,
                ],
                outputs=[
                    line_image_out,
                    line_status,
                    line_json,
                    line_profile_out,
                    line_selected_view,
                    line_hierarchy_view,
                    line_selected_profile,
                    line_click_status,
                    line_click_state,
                ],
            )
            line_image_out.select(
                fn=preview_clicked_line_horizontal_profile,
                inputs=[line_click_state],
                outputs=[line_selected_view, line_hierarchy_view, line_selected_profile, line_click_status],
            )
            line_scan_models_btn.click(
                fn=scan_ultralytics_inference_models,
                inputs=[line_models_dir],
                outputs=[line_model_select, line_model_scan_msg],
            )
            line_model_select.change(
                fn=lambda x: x or "",
                inputs=[line_model_select],
                outputs=[line_model],
            )

        # 6c) Generate + preview patch dataset
        with gr.Tab("6c. Patch Dataset Debug View"):
            gr.Markdown(
                "Generate and inspect the new patch dataset from `cli.py gen-patches` "
                "(`datasets/text_patches`: line patches + `meta/patches.parquet` + `debug/` overlays)."
            )
            with gr.Accordion("Generate Patch Dataset", open=False):
                with gr.Row():
                    with gr.Column(scale=1):
                        th_gen_config = gr.Textbox(label="config (optional)", value=str((ROOT / "configs" / "patch_gen.yaml").resolve()))
                        th_gen_model = gr.Textbox(
                            label="model_path",
                            value=str((ROOT / "runs" / "detect" / "train" / "weights" / "best.pt").resolve()),
                        )
                        th_gen_input_dir = gr.Textbox(label="input_dir", value=str((workspace_root / "sbb_images").resolve()))
                        th_gen_output_dir = gr.Textbox(label="output_dir", value=default_text_hierarchy_dir)
                    with gr.Column(scale=1):
                        with gr.Row():
                            th_gen_conf = gr.Slider(0.01, 0.99, value=0.25, step=0.01, label="conf")
                            th_gen_imgsz = gr.Number(label="imgsz", value=1024, precision=0)
                        with gr.Row():
                            th_gen_device = gr.Textbox(label="device", value="")
                            th_gen_samples = gr.Number(label="no_samples (0=all)", value=0, precision=0)
                        with gr.Row():
                            th_gen_seed = gr.Number(label="seed", value=42, precision=0)
                            th_gen_target_h = gr.Number(label="target_height", value=112, precision=0)
                        with gr.Row():
                            th_gen_widths = gr.Textbox(label="widths", value="256,384,512")
                            th_gen_overlap = gr.Slider(0.0, 0.9, value=0.5, step=0.05, label="overlap")
                        with gr.Row():
                            th_gen_rmin = gr.Number(label="rmin", value=0.01, precision=3)
                            th_gen_prominence = gr.Number(label="prominence", value=0.08, precision=3)
                        with gr.Row():
                            th_gen_n = gr.Number(label="n_per_line_per_scale", value=12, precision=0)
                            th_gen_p_aligned = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="p_aligned")
                        with gr.Row():
                            th_gen_debug = gr.Number(label="debug_dump", value=0, precision=0)
                            th_gen_btn = gr.Button("Run gen-patches", variant="primary")
                th_gen_log = gr.Textbox(label="Generation Log", lines=10, interactive=False)
                th_meta_path = gr.Textbox(label="metadata_path", interactive=False)

            with gr.Row():
                th_root_dir = gr.Textbox(label="patch_dataset_root_dir", value=default_text_hierarchy_dir)
                th_subset = gr.Dropdown(
                    label="Subset",
                    choices=TEXT_HIERARCHY_SUBSET_CHOICES,
                    value="PatchDataset (debug)",
                )
                th_scan_btn = gr.Button("Scan Assets")
            with gr.Row():
                with gr.Column(scale=1):
                    th_asset_select = gr.Dropdown(label="Asset", choices=[], allow_custom_value=True)
                    with gr.Row():
                        th_prev_btn = gr.Button("Zurueck")
                        th_next_btn = gr.Button("Vor")
                    th_render_btn = gr.Button("Render Asset", variant="primary")
                    th_scan_status = gr.Textbox(label="Scan Status", interactive=False)
                    th_info = gr.Textbox(label="Asset Info", lines=8, interactive=False)
                with gr.Column(scale=1):
                    th_image = gr.Image(type="numpy", label="Asset Preview")
                    th_json = gr.Code(label="Asset Metadata JSON", language="json")

            th_gen_btn.click(
                fn=run_gen_patches_live,
                inputs=[
                    th_gen_config,
                    th_gen_model,
                    th_gen_input_dir,
                    th_gen_output_dir,
                    th_gen_conf,
                    th_gen_imgsz,
                    th_gen_device,
                    th_gen_samples,
                    th_gen_seed,
                    th_gen_target_h,
                    th_gen_widths,
                    th_gen_overlap,
                    th_gen_rmin,
                    th_gen_prominence,
                    th_gen_n,
                    th_gen_p_aligned,
                    th_gen_debug,
                ],
                outputs=[th_gen_log, th_root_dir, th_meta_path],
            )
            th_root_dir.change(
                fn=lambda x: x or "",
                inputs=[th_root_dir],
                outputs=[th_gen_output_dir],
            )
            th_scan_btn.click(
                fn=scan_text_hierarchy_assets,
                inputs=[th_root_dir, th_subset],
                outputs=[th_asset_select, th_scan_status],
            )
            th_subset.change(
                fn=scan_text_hierarchy_assets,
                inputs=[th_root_dir, th_subset],
                outputs=[th_asset_select, th_scan_status],
            )
            th_render_btn.click(
                fn=preview_text_hierarchy_asset,
                inputs=[th_root_dir, th_asset_select],
                outputs=[th_image, th_info, th_json],
            )
            th_asset_select.change(
                fn=preview_text_hierarchy_asset,
                inputs=[th_root_dir, th_asset_select],
                outputs=[th_image, th_info, th_json],
            )
            th_prev_btn.click(
                fn=lambda root, subset, cur: preview_adjacent_text_hierarchy_asset(root, subset, cur, -1),
                inputs=[th_root_dir, th_subset, th_asset_select],
                outputs=[th_asset_select, th_image, th_info, th_json],
            )
            th_next_btn.click(
                fn=lambda root, subset, cur: preview_adjacent_text_hierarchy_asset(root, subset, cur, 1),
                inputs=[th_root_dir, th_subset, th_asset_select],
                outputs=[th_asset_select, th_image, th_info, th_json],
            )

        # 6c2) Inspect MNN matches for individual patches in a patch dataset
        with gr.Tab("6c2. MNN Patch Match Viewer"):
            gr.Markdown(
                "Browse `meta/mnn_pairs.parquet` for a patch dataset. "
                "Scan patches with their MNN match counts, select a patch, then preview source + MNN matches side-by-side."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    mnn_view_root_dir = gr.Textbox(label="patch_dataset_root_dir", value=default_text_hierarchy_dir)
                    mnn_view_patch_meta = gr.Textbox(
                        label="patches_parquet (optional)",
                        value=str((Path(default_text_hierarchy_dir) / "meta" / "patches.parquet").resolve()),
                    )
                    mnn_view_pairs_meta = gr.Textbox(
                        label="mnn_pairs_parquet (optional)",
                        value=str((Path(default_text_hierarchy_dir) / "meta" / "mnn_pairs.parquet").resolve()),
                    )
                    with gr.Row():
                        mnn_view_min_matches = gr.Number(label="min_match_count", value=1, precision=0)
                        mnn_view_max_rows = gr.Number(label="max_list_rows", value=200, precision=0)
                    with gr.Row():
                        mnn_view_scan_btn = gr.Button("Scan MNN Patches", variant="primary")
                        mnn_view_preview_btn = gr.Button("Preview Selected Patch")
                    mnn_view_patch_select = gr.Dropdown(label="Patch (with MNN count)", choices=[], allow_custom_value=False)
                    mnn_view_status = gr.Textbox(label="Status", interactive=False)
                with gr.Column(scale=1):
                    mnn_view_list = gr.Dataframe(
                        headers=["patch_id", "match_count", "doc_id", "page_id", "line_id", "scale_w", "k", "ink_ratio", "boundary_score"],
                        label="Patches with MNN match counts",
                        interactive=False,
                        wrap=True,
                    )
            with gr.Row():
                mnn_view_max_matches = gr.Number(label="max_matches_to_render (0=all)", value=24, precision=0)
            mnn_view_gallery = gr.Gallery(
                label="",
                columns=4,
                height="auto",
                object_fit="contain",
            )
            mnn_view_json = gr.Code(label="Selected Patch + MNN Match Metadata JSON", language="json")

            mnn_view_scan_btn.click(
                fn=scan_mnn_patch_matches_ui,
                inputs=[mnn_view_root_dir, mnn_view_patch_meta, mnn_view_pairs_meta, mnn_view_min_matches, mnn_view_max_rows],
                outputs=[mnn_view_patch_select, mnn_view_list, mnn_view_status, mnn_view_json],
            )
            mnn_view_root_dir.change(
                fn=lambda x: (
                    str((Path(x or "").expanduser().resolve() / "meta" / "patches.parquet").resolve()) if str(x or "").strip() else "",
                    str((Path(x or "").expanduser().resolve() / "meta" / "mnn_pairs.parquet").resolve()) if str(x or "").strip() else "",
                ),
                inputs=[mnn_view_root_dir],
                outputs=[mnn_view_patch_meta, mnn_view_pairs_meta],
            )
            mnn_view_preview_btn.click(
                fn=preview_mnn_patch_matches_ui,
                inputs=[mnn_view_root_dir, mnn_view_patch_meta, mnn_view_pairs_meta, mnn_view_patch_select, mnn_view_max_matches],
                outputs=[mnn_view_gallery, mnn_view_status, mnn_view_json],
            )
            mnn_view_patch_select.change(
                fn=preview_mnn_patch_matches_ui,
                inputs=[mnn_view_root_dir, mnn_view_patch_meta, mnn_view_pairs_meta, mnn_view_patch_select, mnn_view_max_matches],
                outputs=[mnn_view_gallery, mnn_view_status, mnn_view_json],
            )

        # 6d) Interactive hierarchy + embedding preview on uploaded image
        with gr.Tab("6d. Hierarchy Encode Preview"):
            gr.Markdown(
                "Scan `models/` for passende Modelle, run line detection on an uploaded image, "
                "split lines into 2/4/8 hierarchy blocks, click one block, and output its latent vector."
            )
            he_line_click_state = gr.State({})
            he_selected_state = gr.State({})
            with gr.Row():
                with gr.Column(scale=1):
                    he_models_dir = gr.Textbox(label="models_dir", value=str((workspace_root / "models").resolve()))
                    he_scan_models_btn = gr.Button("Scan Models")
                    he_yolo_model_select = gr.Dropdown(
                        label="Detected YOLO Model (line detection)",
                        choices=[],
                        allow_custom_value=True,
                    )
                    he_encoder_backbone_select = gr.Dropdown(
                        label="Detected Encoder Backbone (ViT/DINO)",
                        choices=[],
                        allow_custom_value=True,
                    )
                    he_projection_head_select = gr.Dropdown(
                        label="Detected Projection Head (.pt)",
                        choices=[],
                        allow_custom_value=True,
                    )
                    he_scan_status = gr.Textbox(label="Model Scan Status", interactive=False)

                    he_line_model = gr.Textbox(label="line_model_path", value="")
                    he_encoder_backbone = gr.Textbox(label="encoder_backbone_path", value="")
                    he_projection_head = gr.Textbox(label="projection_head_path (optional)", value="")

                    with gr.Accordion("Line Segmentation Parameters", open=False):
                        he_conf = gr.Slider(0.01, 0.99, value=0.25, step=0.01, label="conf")
                        he_imgsz = gr.Number(label="imgsz", value=1024, precision=0)
                        he_device = gr.Textbox(label="line_detection_device", value="")
                        he_min_height = gr.Number(label="min_line_height_px", value=10, precision=0)
                        he_proj_smooth = gr.Number(label="projection_smooth_rows", value=9, precision=0)
                        he_proj_thresh = gr.Slider(0.05, 0.80, value=0.20, step=0.01, label="projection_threshold_rel")
                        he_merge_gap = gr.Number(label="merge_gap_px", value=5, precision=0)

                    with gr.Accordion("Hierarchy + Encoding Options", open=True):
                        he_hierarchy_level = gr.Dropdown(label="Hierarchy Level", choices=["2", "4", "8"], value="2")
                        he_encoder_device = gr.Textbox(label="encoder_device", value="auto")
                        he_l2_norm = gr.Checkbox(label="l2_normalize_embedding", value=True)
                        he_vector_decimals = gr.Number(label="vector_decimals", value=6, precision=0)

                    he_run_btn = gr.Button("Detect + Split + Prepare Hierarchy", variant="primary")

                with gr.Column(scale=1):
                    he_image_in = gr.Image(type="numpy", label="Input Image", sources=["upload", "clipboard"])
                    he_overlay = gr.Image(type="numpy", label="Overlay (Detected tibetan_text + Line Boxes)", interactive=True)
                    he_line_profile = gr.Image(type="numpy", label="Line Projection Profile")
                    he_selected_line_view = gr.Image(type="numpy", label="Selected Line (Peak-based Split)")
                    he_selected_line_profile = gr.Image(type="numpy", label="Selected Line Horizontal Profile")
                    he_word_overlay = gr.Image(type="numpy", label="Hierarchy Blocks (click to encode)", interactive=True)
                    he_word_crop = gr.Image(type="numpy", label="Selected Text Block Crop")
                    he_run_status = gr.Textbox(label="Run Status", interactive=False)
                    he_line_status = gr.Textbox(label="Line Selection Status", interactive=False)
                    he_encode_status = gr.Textbox(label="Encoding Status", interactive=False)
                    he_json = gr.Code(label="Line Segmentation JSON", language="json")
                    he_vector = gr.Code(label="Latent Vector JSON", language="json")

            he_scan_models_btn.click(
                fn=scan_models_for_line_and_encoder_ui,
                inputs=[he_models_dir],
                outputs=[
                    he_yolo_model_select,
                    he_encoder_backbone_select,
                    he_projection_head_select,
                    he_line_model,
                    he_encoder_backbone,
                    he_projection_head,
                    he_scan_status,
                ],
            )
            he_yolo_model_select.change(
                fn=lambda x: x or "",
                inputs=[he_yolo_model_select],
                outputs=[he_line_model],
            )
            he_encoder_backbone_select.change(
                fn=on_encoder_backbone_change_ui,
                inputs=[he_encoder_backbone_select],
                outputs=[he_encoder_backbone, he_projection_head],
            )
            he_projection_head_select.change(
                fn=lambda x: x or "",
                inputs=[he_projection_head_select],
                outputs=[he_projection_head],
            )

            he_run_btn.click(
                fn=run_tibetan_text_line_split_for_embedding_ui,
                inputs=[
                    he_image_in,
                    he_line_model,
                    he_conf,
                    he_imgsz,
                    he_device,
                    he_min_height,
                    he_proj_smooth,
                    he_proj_thresh,
                    he_merge_gap,
                ],
                outputs=[
                    he_overlay,
                    he_run_status,
                    he_json,
                    he_line_profile,
                    he_selected_line_view,
                    he_selected_line_profile,
                    he_line_status,
                    he_line_click_state,
                    he_word_overlay,
                    he_word_crop,
                    he_vector,
                    he_encode_status,
                    he_selected_state,
                ],
            )
            he_overlay.select(
                fn=prepare_clicked_line_for_embedding_ui,
                inputs=[he_line_click_state, he_hierarchy_level],
                outputs=[
                    he_selected_line_view,
                    he_selected_line_profile,
                    he_word_overlay,
                    he_line_status,
                    he_selected_state,
                    he_word_crop,
                    he_vector,
                    he_encode_status,
                ],
            )
            he_hierarchy_level.change(
                fn=update_hierarchy_level_for_embedding_ui,
                inputs=[he_selected_state, he_hierarchy_level],
                outputs=[
                    he_word_overlay,
                    he_line_status,
                    he_selected_state,
                    he_word_crop,
                    he_vector,
                    he_encode_status,
                ],
            )
            he_word_overlay.select(
                fn=encode_clicked_hierarchy_block_ui,
                inputs=[
                    he_selected_state,
                    he_encoder_backbone,
                    he_projection_head,
                    he_encoder_device,
                    he_l2_norm,
                    he_vector_decimals,
                ],
                outputs=[he_word_overlay, he_word_crop, he_vector, he_encode_status, he_selected_state],
            )

        # 7) VLM parsing
        with gr.Tab("7. VLM Layout"):
            gr.Markdown("Transformer-based layout parsing integrated from the VLM UI.")
            vlm_status = gr.Markdown(_vlm_backend_status_markdown())
            with gr.Row():
                with gr.Column(scale=1):
                    vlm_parser = gr.Dropdown(
                        label="Parser Backend",
                        choices=vlm_choices,
                        value=(vlm_choices[0] if vlm_choices else None),
                    )
                    vlm_prompt = gr.Textbox(label="Prompt", value=default_prompt, lines=5)
                    vlm_hf_model = gr.Textbox(label="HF Model ID Override (optional)", value="")
                    with gr.Row():
                        vlm_max_new_tokens = gr.Slider(64, 4096, value=1024, step=32, label="Max New Tokens")
                        vlm_hf_device = gr.Dropdown(choices=["auto", "cpu", "cuda"], value="auto", label="HF Device")
                    vlm_hf_dtype = gr.Dropdown(
                        choices=["auto", "float16", "bfloat16", "float32"],
                        value="auto",
                        label="HF DType",
                    )
                    with gr.Accordion("MinerU Options", open=False):
                        vlm_mineru_command = gr.Textbox(label="MinerU Command", value="mineru")
                        vlm_mineru_timeout = gr.Number(label="MinerU Timeout (s)", value=300, precision=0)
                    vlm_run_btn = gr.Button("Detect Layout", variant="primary")
                    vlm_refresh_btn = gr.Button("Refresh Backend Status")
                with gr.Column(scale=1):
                    vlm_image = gr.Image(type="numpy", label="Image (Upload or Clipboard Paste)", sources=["upload", "clipboard"])
                    vlm_overlay = gr.Image(type="numpy", label="Detected Regions (Overlay)")
                    vlm_status_text = gr.Textbox(label="Status", interactive=False)
                    vlm_json = gr.Code(label="JSON Output", language="json")

            vlm_run_btn.click(
                fn=run_vlm_layout_inference,
                inputs=[
                    vlm_image,
                    vlm_parser,
                    vlm_prompt,
                    vlm_hf_model,
                    vlm_max_new_tokens,
                    vlm_hf_device,
                    vlm_hf_dtype,
                    vlm_mineru_command,
                    vlm_mineru_timeout,
                ],
                outputs=[vlm_overlay, vlm_status_text, vlm_json],
            )
            vlm_refresh_btn.click(fn=_vlm_backend_status_markdown, inputs=[], outputs=[vlm_status])

        # 8) Export to Label Studio
        with gr.Tab("8. Label Studio Export"):
            gr.Markdown(
                "Simplified flow: pick your Label-Studio export folder + synthetic dataset, "
                "then auto-prepare a combined split (including polygon->bbox normalization)."
            )
            with gr.Row():
                ls_export_dir_input = gr.Textbox(
                    label="Label Studio export folder",
                    value=str((workspace_root / "datasets" / "ls-sbb").resolve()),
                )
                ls_export_zip_input = gr.File(
                    label="Label Studio export ZIP (optional)",
                    file_types=[".zip"],
                    type="filepath",
                )
                ls_syn_dataset_base = gr.Textbox(
                    label="Synthetic datasets base",
                    value=default_dataset_base,
                )
                ls_scan_syn_btn = gr.Button("Scan Synthetic Datasets")
            ls_syn_dataset = gr.Dropdown(
                label="Synthetic dataset (train split)",
                choices=train_dataset_choices,
                value=default_train_dataset,
                allow_custom_value=True,
            )
            ls_combined_out = gr.Textbox(
                label="Combined output split dir",
                value=str((workspace_root / "datasets" / "combined-layout-labelstudio" / "train").resolve()),
            )
            ls_prepare_btn = gr.Button("Prepare Combined Split from LS Export + Synthetic", variant="primary")
            ls_prepare_status = gr.Textbox(label="Prepare Status", lines=8, interactive=False)
            split_dir = gr.Textbox(label="YOLO Split Directory", value=default_split_dir)
            image_ext = gr.Dropdown(label="image-ext", choices=[".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"], value=".png")
            tasks_json = gr.Textbox(label="tasks json output", value=str((ROOT / "ls-tasks-ui.json").resolve()))
            image_root_url = gr.Textbox(label="image-root-url", value="/data/local-files/?d=train/images")
            export_btn = gr.Button("Export to Label Studio Tasks", variant="primary")
            export_log = gr.Textbox(label="Export Log", lines=14)

            local_files_root = gr.Textbox(
                label="Label Studio local files root",
                value=str((ROOT / "datasets" / "tibetan-yolo").resolve()),
            )
            start_ls_btn = gr.Button("Start Label Studio")
            start_ls_msg = gr.Textbox(label="Label Studio Status", interactive=False)

            export_btn.click(
                fn=export_to_label_studio,
                inputs=[split_dir, image_ext, tasks_json, image_root_url],
                outputs=[export_log],
            )
            start_ls_btn.click(
                fn=start_label_studio,
                inputs=[local_files_root],
                outputs=[start_ls_msg],
            )

            ls_scan_syn_btn.click(
                fn=_scan_train_datasets,
                inputs=[ls_syn_dataset_base],
                outputs=[ls_syn_dataset],
            )
            ls_prepare_btn.click(
                fn=prepare_labelstudio_from_ls_export_ui,
                inputs=[
                    ls_export_dir_input,
                    ls_export_zip_input,
                    ls_syn_dataset,
                    ls_syn_dataset_base,
                    ls_combined_out,
                    split_dir,
                    local_files_root,
                    image_root_url,
                    tasks_json,
                    batch_vlm_export_split_dir,
                    batch_vlm_export_image_root_url,
                    batch_vlm_export_tasks_json,
                ],
                outputs=[
                    ls_prepare_status,
                    batch_vlm_combine_result,
                    split_dir,
                    local_files_root,
                    image_root_url,
                    tasks_json,
                    batch_vlm_export_split_dir,
                    batch_vlm_export_image_root_url,
                    batch_vlm_export_tasks_json,
                ],
            )

            batch_vlm_combine_btn.click(
                fn=prepare_combined_for_labelstudio_ui,
                inputs=[
                    batch_vlm_combine_syn_split,
                    batch_vlm_combine_sbb_split,
                    batch_vlm_combine_sbb_zip,
                    batch_vlm_combine_out_split,
                    batch_vlm_sbb_train_ratio,
                    batch_vlm_split_seed,
                    split_dir,
                    local_files_root,
                    image_root_url,
                    tasks_json,
                    batch_vlm_export_split_dir,
                    batch_vlm_export_image_root_url,
                    batch_vlm_export_tasks_json,
                ],
                outputs=[
                    batch_vlm_combine_status,
                    batch_vlm_combine_result,
                    split_dir,
                    local_files_root,
                    image_root_url,
                    tasks_json,
                    batch_vlm_export_split_dir,
                    batch_vlm_export_image_root_url,
                    batch_vlm_export_tasks_json,
                ],
            )

        # 9) PPN download
        with gr.Tab("9. PPN Downloader"):
            gr.Markdown("Download Staatsbibliothek zu Berlin images by PPN (uses existing SBB utilities).")
            with gr.Row():
                with gr.Column(scale=1):
                    ppn_value = gr.Textbox(label="PPN", value="")
                    ppn_output_dir = gr.Textbox(label="output_dir", value=str((ROOT / "sbb_images").resolve()))
                    ppn_max_images = gr.Number(label="max_images (0=all)", value=0, precision=0)
                    ppn_download_workers = gr.Number(label="download_workers", value=8, precision=0)
                    ppn_no_ssl = gr.Checkbox(label="no_ssl_verify", value=False)
                    ppn_download_btn = gr.Button("Download Images", variant="primary")
                    ppn_log = gr.Textbox(label="Download Log", lines=10)
                with gr.Column(scale=1):
                    ppn_image_select = gr.Dropdown(label="Downloaded Image", choices=[])
                    ppn_preview_btn = gr.Button("Preview Selected Image")
                    ppn_preview_img = gr.Image(type="numpy", label="Preview")
                    ppn_preview_msg = gr.Textbox(label="Preview Status", interactive=False)

            gr.Markdown("### Analyze Downloaded Image (YOLO model from `models/` or VLM)")
            with gr.Row():
                with gr.Column(scale=1):
                    ppn_analysis_mode = gr.Radio(
                        choices=["YOLO (models/)", "VLM"],
                        value="YOLO (models/)",
                        label="Analysis Mode",
                    )
                    ppn_models_dir = gr.Textbox(label="models_dir", value=str((ROOT / "models").resolve()))
                    ppn_scan_models_btn = gr.Button("Scan models/")
                    ppn_model_select = gr.Dropdown(label="Pretrained Model", choices=[])
                    ppn_model_scan_msg = gr.Textbox(label="Model Scan Status", interactive=False)
                    ppn_conf = gr.Slider(0.01, 0.99, value=0.25, step=0.01, label="conf (YOLO)")
                    ppn_imgsz = gr.Number(label="imgsz (YOLO)", value=1024, precision=0)
                    ppn_device = gr.Textbox(label="device (YOLO)", value="")
                with gr.Column(scale=1):
                    ppn_vlm_parser = gr.Dropdown(
                        label="VLM Parser",
                        choices=vlm_choices,
                        value=(vlm_choices[0] if vlm_choices else None),
                    )
                    ppn_vlm_prompt = gr.Textbox(label="VLM Prompt", value=default_prompt, lines=4)
                    ppn_vlm_hf_model = gr.Textbox(label="HF Model ID Override", value="")
                    ppn_vlm_max_new_tokens = gr.Slider(64, 4096, value=1024, step=32, label="Max New Tokens")
                    with gr.Row():
                        ppn_vlm_hf_device = gr.Dropdown(choices=["auto", "cpu", "cuda"], value="auto", label="HF Device")
                        ppn_vlm_hf_dtype = gr.Dropdown(
                            choices=["auto", "float16", "bfloat16", "float32"],
                            value="auto",
                            label="HF DType",
                        )
                    ppn_mineru_command = gr.Textbox(label="MinerU Command", value="mineru")
                    ppn_mineru_timeout = gr.Number(label="MinerU Timeout (s)", value=300, precision=0)

            ppn_analyze_btn = gr.Button("Analyze Selected Downloaded Image", variant="primary")
            ppn_overlay = gr.Image(type="numpy", label="Analysis Overlay")
            ppn_analysis_status = gr.Textbox(label="Analysis Status", interactive=False)
            ppn_analysis_json = gr.Code(label="Analysis JSON", language="json")

            ppn_download_btn.click(
                fn=download_ppn_images,
                inputs=[ppn_value, ppn_output_dir, ppn_max_images, ppn_no_ssl, ppn_download_workers],
                outputs=[ppn_log, ppn_image_select],
            )
            ppn_preview_btn.click(
                fn=preview_downloaded_image,
                inputs=[ppn_output_dir, ppn_image_select],
                outputs=[ppn_preview_img, ppn_preview_msg],
            )
            ppn_scan_models_btn.click(
                fn=scan_pretrained_models,
                inputs=[ppn_models_dir],
                outputs=[ppn_model_select, ppn_model_scan_msg],
            )
            ppn_analyze_btn.click(
                fn=run_ppn_image_analysis,
                inputs=[
                    ppn_output_dir,
                    ppn_image_select,
                    ppn_analysis_mode,
                    ppn_model_select,
                    ppn_conf,
                    ppn_imgsz,
                    ppn_device,
                    ppn_vlm_parser,
                    ppn_vlm_prompt,
                    ppn_vlm_hf_model,
                    ppn_vlm_max_new_tokens,
                    ppn_vlm_hf_device,
                    ppn_vlm_hf_dtype,
                    ppn_mineru_command,
                    ppn_mineru_timeout,
                ],
                outputs=[ppn_overlay, ppn_analysis_status, ppn_analysis_json],
            )

        # 10) SBB grid review + quarantine
        with gr.Tab("10. SBB Grid Review"):
            gr.Markdown(
                "Browse images from `sbb_images` as lightweight 9x9 thumbnails. "
                "Select pages via checkboxes and move them to `sbb_images/quarantine`."
            )
            with gr.Row():
                sbb_grid_dir = gr.Textbox(label="sbb_images_dir", value=default_sbb_grid_dir)
                sbb_grid_refresh_btn = gr.Button("Refresh", variant="secondary")
                sbb_grid_sort_mean_btn = gr.Button("Sort by Mean Color")
                sbb_grid_first_btn = gr.Button("Anfang")
                sbb_grid_prev_btn = gr.Button("Zurueck", elem_id="sbb-grid-prev-btn")
                sbb_grid_next_btn = gr.Button("Vor", elem_id="sbb-grid-next-btn")
                sbb_grid_last_btn = gr.Button("Ende")
                sbb_grid_quarantine_btn = gr.Button("Quarantine Selected", variant="primary")
            with gr.Row():
                sbb_grid_page_input = gr.Number(label="Page (1-based)", value=1, precision=0)
                sbb_grid_go_btn = gr.Button("Go to Page")
            sbb_grid_status = gr.Textbox(label="Grid Status", interactive=False)
            sbb_grid_page_state = gr.State(0)
            sbb_grid_paths_state = gr.State([])
            sbb_grid_sort_state = gr.State("name")
            gr.HTML(
                """
<script>
(() => {
  if (window.__sbbGridHotkeysInstalled) return;
  window.__sbbGridHotkeysInstalled = true;
  document.addEventListener("keydown", (ev) => {
    const tag = (ev.target && ev.target.tagName ? ev.target.tagName.toLowerCase() : "");
    const isTyping = tag === "input" || tag === "textarea" || ev.target?.isContentEditable;
    if (isTyping) return;
    if (ev.key === "ArrowLeft") {
      const btn = document.getElementById("sbb-grid-prev-btn");
      if (btn) btn.click();
    } else if (ev.key === "ArrowRight") {
      const btn = document.getElementById("sbb-grid-next-btn");
      if (btn) btn.click();
    }
  });
})();
</script>
                """
            )

            sbb_grid_images: List[gr.Image] = []
            sbb_grid_checks: List[gr.Checkbox] = []
            sbb_grid_names: List[gr.Textbox] = []

            for _row_idx in range(SBB_GRID_ROWS):
                with gr.Row():
                    for _col_idx in range(SBB_GRID_COLS):
                        with gr.Column(min_width=110):
                            img = gr.Image(
                                type="numpy",
                                label=None,
                                show_label=False,
                                height=SBB_GRID_THUMB_SIZE,
                                width=SBB_GRID_THUMB_SIZE,
                                visible=False,
                            )
                            chk = gr.Checkbox(label="Q", value=False, visible=False)
                            name = gr.Textbox(show_label=False, value="", interactive=False, visible=False)
                            sbb_grid_images.append(img)
                            sbb_grid_checks.append(chk)
                            sbb_grid_names.append(name)

            sbb_grid_outputs: List[Any] = [sbb_grid_status, sbb_grid_page_state, sbb_grid_paths_state, sbb_grid_sort_state]
            for _img, _chk, _name in zip(sbb_grid_images, sbb_grid_checks, sbb_grid_names):
                sbb_grid_outputs.extend([_img, _chk, _name])

            sbb_grid_refresh_btn.click(
                fn=sbb_grid_refresh,
                inputs=[sbb_grid_dir, sbb_grid_sort_state],
                outputs=sbb_grid_outputs,
            )
            sbb_grid_sort_mean_btn.click(
                fn=sbb_grid_sort_by_mean,
                inputs=[sbb_grid_dir],
                outputs=sbb_grid_outputs,
            )
            sbb_grid_prev_btn.click(
                fn=sbb_grid_prev,
                inputs=[sbb_grid_dir, sbb_grid_page_state, sbb_grid_sort_state],
                outputs=sbb_grid_outputs,
            )
            sbb_grid_next_btn.click(
                fn=sbb_grid_next,
                inputs=[sbb_grid_dir, sbb_grid_page_state, sbb_grid_sort_state],
                outputs=sbb_grid_outputs,
            )
            sbb_grid_first_btn.click(
                fn=sbb_grid_first,
                inputs=[sbb_grid_dir, sbb_grid_sort_state],
                outputs=sbb_grid_outputs,
            )
            sbb_grid_last_btn.click(
                fn=sbb_grid_last,
                inputs=[sbb_grid_dir, sbb_grid_sort_state],
                outputs=sbb_grid_outputs,
            )
            sbb_grid_go_btn.click(
                fn=sbb_grid_go_to_page,
                inputs=[sbb_grid_dir, sbb_grid_page_input, sbb_grid_sort_state],
                outputs=sbb_grid_outputs,
            )
            sbb_grid_quarantine_btn.click(
                fn=sbb_grid_quarantine,
                inputs=[sbb_grid_dir, sbb_grid_page_state, sbb_grid_paths_state, sbb_grid_sort_state, *sbb_grid_checks],
                outputs=sbb_grid_outputs,
            )

        # 11) Diffusion + LoRA texture augmentation
        with gr.Tab("11. Diffusion + LoRA"):
            gr.Markdown(
                "End-to-end texture workflow: "
                "A) prepare LoRA crop dataset from real/SBB pages, "
                "B) train a texture LoRA, "
                "C) run SDXL or SD2.1 + ControlNet Canny inference with optional pre-trained LoRA."
            )
            gr.Markdown("### A) Prepare Texture LoRA Dataset (from SBB / real pecha pages)")
            with gr.Row():
                with gr.Column(scale=1):
                    prep_input_dir = gr.Textbox(
                        label="real_pages_input_dir (e.g. SBB images folder)",
                        value=default_texture_real_pages_dir,
                    )
                    prep_output_dir = gr.Textbox(
                        label="prepared_dataset_output_dir",
                        value=default_texture_lora_dataset_dir,
                    )
                    with gr.Row():
                        prep_crop_size = gr.Number(label="crop_size", value=1024, precision=0)
                        prep_num_crops_per_page = gr.Number(label="num_crops_per_page", value=12, precision=0)
                    with gr.Row():
                        prep_min_edge_density = gr.Slider(0.0, 0.20, value=0.025, step=0.001, label="min_edge_density")
                        prep_seed = gr.Number(label="seed", value=42, precision=0)
                    with gr.Row():
                        prep_canny_low = gr.Number(label="canny_low", value=100, precision=0)
                        prep_canny_high = gr.Number(label="canny_high", value=200, precision=0)
                    prep_run_btn = gr.Button("Prepare LoRA Dataset", variant="primary")
                    prep_log = gr.Textbox(label="Prepare Log", lines=12, interactive=False)
                with gr.Column(scale=1):
                    prep_preview = gr.Image(type="numpy", label="Latest Prepared Crop")
                    prep_preview_status = gr.Textbox(label="Prepare Preview Status", interactive=False)

            gr.Markdown("### B) Train Texture LoRA")
            with gr.Row():
                with gr.Column(scale=1):
                    train_texture_dataset_dir = gr.Textbox(label="dataset_dir", value=default_texture_lora_dataset_dir)
                    train_texture_output_dir = gr.Textbox(label="output_dir", value=default_texture_lora_output_dir)
                    train_texture_model_family = gr.Dropdown(
                        choices=["sdxl", "sd21"],
                        value="sdxl",
                        label="model_family",
                    )
                    with gr.Row():
                        train_texture_resolution = gr.Number(label="resolution", value=1024, precision=0)
                        train_texture_batch_size = gr.Number(label="batch_size", value=1, precision=0)
                    with gr.Row():
                        train_texture_lr = gr.Number(label="lr", value=1e-4, precision=6)
                        train_texture_steps = gr.Number(label="max_train_steps", value=1500, precision=0)
                    with gr.Row():
                        train_texture_rank = gr.Number(label="rank", value=16, precision=0)
                        train_texture_alpha = gr.Number(label="lora_alpha", value=16.0, precision=2)
                    with gr.Row():
                        train_texture_mixed_precision = gr.Dropdown(
                            choices=["no", "fp16", "bf16"],
                            value="no",
                            label="mixed_precision",
                        )
                        train_texture_workers = gr.Number(label="num_workers", value=4, precision=0)
                    with gr.Row():
                        train_texture_gc = gr.Checkbox(label="gradient_checkpointing", value=True)
                        train_texture_te = gr.Checkbox(label="train_text_encoder", value=False)
                    train_texture_prompt = gr.Textbox(label="prompt", value=DEFAULT_TEXTURE_PROMPT, lines=2)
                    with gr.Row():
                        train_texture_seed = gr.Number(label="seed", value=42, precision=0)
                        train_texture_lora_name = gr.Textbox(label="lora_weights_name", value="texture_lora.safetensors")
                    train_texture_base_model = gr.Textbox(
                        label="base_model_id",
                        value="stabilityai/stable-diffusion-xl-base-1.0",
                    )
                    train_texture_run_btn = gr.Button("Train Texture LoRA", variant="primary")
                with gr.Column(scale=1):
                    train_texture_log = gr.Textbox(label="Train LoRA Log", lines=16, interactive=False)
                    train_texture_lora_path = gr.Textbox(label="Trained LoRA Path", interactive=False)
                    train_texture_cfg_path = gr.Textbox(label="Training Config Path", interactive=False)

            gr.Markdown("### C) Texture Inference (use pre-trained or freshly trained LoRA)")
            with gr.Row():
                with gr.Column(scale=1):
                    diff_input_dir = gr.Textbox(label="input_dir (synthetic renders)", value=default_texture_input_dir)
                    diff_output_dir = gr.Textbox(label="output_dir", value=default_texture_output_dir)
                    diff_model_family = gr.Dropdown(
                        choices=["sdxl", "sd21"],
                        value="sdxl",
                        label="model_family",
                    )
                    diff_prompt = gr.Textbox(
                        label="prompt",
                        value=DEFAULT_TEXTURE_PROMPT,
                        lines=2,
                    )
                    diff_lora_path = gr.Textbox(
                        label="lora_path (optional, auto-filled after training)",
                        value="",
                        placeholder="Path to LoRA directory or *.safetensors",
                    )
                    with gr.Row():
                        diff_lora_models_dir = gr.Textbox(
                            label="lora_models_dir",
                            value=str((ROOT / "models").resolve()),
                        )
                        diff_scan_lora_btn = gr.Button("Scan LoRA Models")
                    diff_lora_select = gr.Dropdown(
                        label="Detected LoRA (.safetensor/.safetensors)",
                        choices=[],
                        value=None,
                        allow_custom_value=True,
                    )
                    diff_lora_scan_msg = gr.Textbox(label="LoRA Scan Status", interactive=False)
                    diff_debug_upload_image = gr.Image(
                        type="numpy",
                        label="debug_upload_image (optional, for single-image test)",
                        sources=["upload", "clipboard"],
                    )
                    with gr.Row():
                        diff_strength = gr.Slider(
                            minimum=0.0,
                            maximum=0.25,
                            value=0.2,
                            step=0.01,
                            label="strength (<= 0.25 for structure preservation)",
                        )
                        diff_steps = gr.Number(label="steps", value=28, precision=0)
                    with gr.Row():
                        diff_guidance_scale = gr.Slider(0.0, 4.0, value=1.0, step=0.1, label="guidance_scale")
                        diff_controlnet_scale = gr.Slider(0.5, 3.0, value=2.0, step=0.1, label="controlnet_scale")
                    diff_disable_controlnet = gr.Checkbox(
                        label="disable_controlnet (plain img2img)",
                        value=False,
                    )
                    with gr.Row():
                        diff_lora_scale = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="lora_scale")
                        diff_seed = gr.Number(
                            label="seed (-1 for random)",
                            value=123,
                            precision=0,
                        )
                    with gr.Accordion("Advanced Models / Canny", open=False):
                        diff_base_model_id = gr.Textbox(
                            label="base_model_id",
                            value="stabilityai/stable-diffusion-xl-base-1.0",
                        )
                        diff_controlnet_model_id = gr.Textbox(
                            label="controlnet_model_id",
                            value="diffusers/controlnet-canny-sdxl-1.0",
                        )
                        with gr.Row():
                            diff_canny_low = gr.Number(label="canny_low", value=100, precision=0)
                            diff_canny_high = gr.Number(label="canny_high", value=200, precision=0)

                    with gr.Row():
                        diff_run_btn = gr.Button("Run Texture Augmentation", variant="primary")
                        diff_run_upload_btn = gr.Button("Run Texture Augmentation (Uploaded Image)")
                        diff_preview_btn = gr.Button("Preview Latest Output")
                with gr.Column(scale=1):
                    diff_preview = gr.Image(type="numpy", label="Latest Augmented Output")
                    diff_preview_status = gr.Textbox(label="Preview Status", interactive=False)
                    diff_log = gr.Textbox(label="Texture Augmentation Log", lines=20, interactive=False)

            prep_run_btn.click(
                fn=run_prepare_texture_lora_dataset_live,
                inputs=[
                    prep_input_dir,
                    prep_output_dir,
                    prep_crop_size,
                    prep_num_crops_per_page,
                    prep_min_edge_density,
                    prep_seed,
                    prep_canny_low,
                    prep_canny_high,
                ],
                outputs=[prep_log, prep_preview, prep_preview_status, train_texture_dataset_dir],
            )

            train_texture_run_btn.click(
                fn=run_train_texture_lora_live,
                inputs=[
                    train_texture_dataset_dir,
                    train_texture_output_dir,
                    train_texture_model_family,
                    train_texture_resolution,
                    train_texture_batch_size,
                    train_texture_lr,
                    train_texture_steps,
                    train_texture_rank,
                    train_texture_alpha,
                    train_texture_mixed_precision,
                    train_texture_gc,
                    train_texture_prompt,
                    train_texture_seed,
                    train_texture_base_model,
                    train_texture_te,
                    train_texture_workers,
                    train_texture_lora_name,
                ],
                outputs=[train_texture_log, train_texture_lora_path, train_texture_cfg_path, diff_lora_path],
            )

            diff_run_btn.click(
                fn=run_texture_augment_live,
                inputs=[
                    diff_input_dir,
                    diff_output_dir,
                    diff_model_family,
                    diff_strength,
                    diff_steps,
                    diff_guidance_scale,
                    diff_seed,
                    diff_controlnet_scale,
                    diff_disable_controlnet,
                    diff_lora_path,
                    diff_lora_scale,
                    diff_prompt,
                    diff_base_model_id,
                    diff_controlnet_model_id,
                    diff_canny_low,
                    diff_canny_high,
                ],
                outputs=[diff_log, diff_preview, diff_preview_status],
            )
            diff_run_upload_btn.click(
                fn=run_texture_augment_upload_live,
                inputs=[
                    diff_debug_upload_image,
                    diff_output_dir,
                    diff_model_family,
                    diff_strength,
                    diff_steps,
                    diff_guidance_scale,
                    diff_seed,
                    diff_controlnet_scale,
                    diff_disable_controlnet,
                    diff_lora_path,
                    diff_lora_scale,
                    diff_prompt,
                    diff_base_model_id,
                    diff_controlnet_model_id,
                    diff_canny_low,
                    diff_canny_high,
                ],
                outputs=[diff_log, diff_preview, diff_preview_status],
            )
            diff_preview_btn.click(
                fn=preview_latest_texture_output,
                inputs=[diff_output_dir],
                outputs=[diff_preview, diff_preview_status],
            )
            diff_scan_lora_btn.click(
                fn=scan_lora_models,
                inputs=[diff_lora_models_dir],
                outputs=[diff_lora_select, diff_lora_path, diff_lora_scan_msg],
            )
            diff_lora_select.change(
                fn=lambda x: x or "",
                inputs=[diff_lora_select],
                outputs=[diff_lora_path],
            )

        # 12) Donut OCR inference / evaluation
        with gr.Tab("12. Donut OCR Inference"):
            gr.Markdown(
                "Test trained DONUT/TroCR OCR checkpoints or final model outputs on uploaded line images or "
                "random samples from OCR datasets. Displays OCR output, ground truth, and CER."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    donut_test_models_dir = gr.Textbox(
                        label="Scan models dir",
                        value=str((workspace_root / "models").resolve()),
                    )
                    with gr.Row():
                        donut_test_scan_btn = gr.Button("Scan DONUT Models")
                        donut_test_scan_msg = gr.Textbox(label="Scan Status", interactive=False)
                    donut_test_model_select = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Detected DONUT/TroCR model outputs / checkpoints",
                    )
                    donut_test_model_dir = gr.Textbox(
                        label="Trained model output dir (or checkpoint / direct model dir)",
                        value=default_donut_model_output_dir,
                        placeholder="/path/to/models/donut_openpecha_bdrc/checkpoint-4000",
                    )
                    with gr.Row():
                        donut_test_preproc = gr.Dropdown(
                            choices=["none", "pb", "bdrc"],
                            value="bdrc",
                            label="image_preprocess_pipeline",
                        )
                        donut_test_device = gr.Dropdown(
                            choices=["auto", "cuda:0", "cuda:1", "cuda:2", "cuda:3", "cpu"],
                            value="auto",
                            label="device",
                        )
                    with gr.Row():
                        donut_test_max_len = gr.Number(label="generation_max_length", value=512, precision=0)
                        donut_test_beams = gr.Number(label="num_beams", value=1, precision=0)
                    with gr.Accordion("Load Random OCR Sample From Dataset", open=False):
                        donut_test_datasets_dir = gr.Textbox(
                            label="Scan datasets dir",
                            value=str((workspace_root / "datasets").resolve()),
                        )
                        with gr.Row():
                            donut_test_dataset_scan_btn = gr.Button("Scan OCR Datasets")
                            donut_test_dataset_scan_msg = gr.Textbox(label="Dataset Scan Status", interactive=False)
                        donut_test_dataset_split_map_state = gr.State("{}")
                        with gr.Row():
                            donut_test_dataset_root = gr.Dropdown(
                                choices=[],
                                value=None,
                                label="Detected OCR dataset roots",
                                allow_custom_value=True,
                            )
                            donut_test_dataset_split = gr.Dropdown(
                                choices=[],
                                value=None,
                                label="Split",
                            )
                        with gr.Row():
                            donut_test_dataset_seed = gr.Number(label="random_seed", value=42, precision=0)
                            donut_test_dataset_sample_btn = gr.Button("Load Random Sample")
                        donut_test_dataset_sample_path = gr.Textbox(label="Selected sample image path", interactive=False)
                        donut_test_dataset_sample_meta = gr.Textbox(label="Sample Metadata JSON", lines=8, interactive=False)
                    donut_test_image = gr.Image(label="Line Image (upload / dataset sample)", type="numpy")
                    donut_test_btn = gr.Button("Run DONUT OCR Inference", variant="primary")
                with gr.Column(scale=1):
                    donut_test_preview = gr.Image(label="Preprocessed Image", type="numpy")
                    with gr.Row():
                        donut_test_text = gr.Textbox(label="OCR Output", lines=8)
                        donut_test_gt = gr.Textbox(label="Ground Truth", lines=8)
                    donut_test_cer = gr.Textbox(label="CER (vs Ground Truth)", lines=1)
                    donut_test_debug = gr.Textbox(label="Runtime / Debug JSON", lines=14)

            donut_test_btn.click(
                fn=run_donut_ocr_inference_with_gt_ui,
                inputs=[
                    donut_test_image,
                    donut_test_model_dir,
                    donut_test_preproc,
                    donut_test_device,
                    donut_test_max_len,
                    donut_test_beams,
                    donut_test_gt,
                ],
                outputs=[
                    donut_test_preview,
                    donut_test_text,
                    donut_test_gt,
                    donut_test_cer,
                    donut_test_debug,
                ],
            )
            donut_test_scan_btn.click(
                fn=scan_donut_ocr_models_ui,
                inputs=[donut_test_models_dir],
                outputs=[donut_test_model_select, donut_test_model_dir, donut_test_scan_msg],
            )
            donut_test_model_select.change(
                fn=lambda x: x or "",
                inputs=[donut_test_model_select],
                outputs=[donut_test_model_dir],
            )
            donut_test_dataset_scan_btn.click(
                fn=scan_donut_ocr_datasets_ui,
                inputs=[donut_test_datasets_dir],
                outputs=[
                    donut_test_dataset_root,
                    donut_test_dataset_split,
                    donut_test_dataset_split_map_state,
                    donut_test_dataset_scan_msg,
                ],
            )
            donut_test_dataset_root.change(
                fn=on_donut_ocr_dataset_change_ui,
                inputs=[donut_test_dataset_root, donut_test_dataset_split_map_state],
                outputs=[donut_test_dataset_split],
            )
            donut_test_dataset_sample_btn.click(
                fn=load_random_donut_ocr_dataset_sample_ui,
                inputs=[
                    donut_test_dataset_root,
                    donut_test_dataset_split,
                    donut_test_dataset_seed,
                ],
                outputs=[
                    donut_test_image,
                    donut_test_gt,
                    donut_test_dataset_sample_meta,
                    donut_test_dataset_sample_path,
                ],
            )

        # 13) Pretrained line_clip workbench
        with gr.Tab("13. line_clip Workbench"):
            gr.Markdown(
                "Experiment with a pretrained line_clip dual encoder only. "
                "Scan model bundles automatically (sorted by validation retrieval performance), "
                "load a sample line, and run Text->Image / Image->Image / Image->Text retrieval."
            )
            lineclip_dbg_sample_state = gr.State({})
            lineclip_dbg_bundle_state = gr.State("{}")
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        lineclip_dbg_models_dir = gr.Textbox(label="models_dir (line_clip scan)", value=str((workspace_root / "models").resolve()))
                        lineclip_dbg_scan_models_btn = gr.Button("Scan line_clip Models")
                    lineclip_dbg_model_scan_status = gr.Textbox(label="line_clip Model Scan Status", interactive=False)
                    lineclip_dbg_bundle_select = gr.Dropdown(
                        label="Detected line_clip bundles (auto-scanned, sorted by val retrieval performance)",
                        choices=[],
                        value=None,
                        allow_custom_value=True,
                    )
                    with gr.Row():
                        lineclip_dbg_datasets_dir = gr.Textbox(label="datasets_dir (OCR manifests)", value=str((workspace_root / "datasets").resolve()))
                        lineclip_dbg_scan_datasets_btn = gr.Button("Scan OCR Datasets")
                    lineclip_dbg_dataset_scan_status = gr.Textbox(label="OCR Dataset Scan Status", interactive=False)
                    lineclip_dbg_dataset_split_state = gr.State("{}")
                    with gr.Row():
                        lineclip_dbg_dataset_root = gr.Dropdown(
                            label="OCR dataset root",
                            choices=[],
                            value=None,
                            allow_custom_value=True,
                        )
                        lineclip_dbg_split = gr.Dropdown(label="Split", choices=[], value=None)
                    with gr.Row():
                        lineclip_dbg_seed = gr.Number(label="random_seed", value=42, precision=0)
                        lineclip_dbg_load_sample_btn = gr.Button("Load Random Sample")
                    lineclip_dbg_sample_path = gr.Textbox(label="Sample image path", interactive=False)
                    lineclip_dbg_sample_meta = gr.Textbox(label="Sample metadata JSON", lines=7, interactive=False)
                    lineclip_dbg_sample_image = gr.Image(label="Query/GT line image", type="numpy")
                    with gr.Accordion("Advanced: Detected Artifact Paths", open=False):
                        with gr.Row():
                            lineclip_dbg_img_backbone = gr.Textbox(label="image_backbone_dir")
                            lineclip_dbg_img_head = gr.Textbox(label="image_projection_head")
                        with gr.Row():
                            lineclip_dbg_txt_encoder = gr.Textbox(label="text_encoder_dir")
                            lineclip_dbg_txt_head = gr.Textbox(label="text_projection_head")
                with gr.Column(scale=1):
                    with gr.Row():
                        lineclip_dbg_query_mode = gr.Radio(
                            choices=["Text -> Image", "Image -> Image", "Image -> Text"],
                            value="Text -> Image",
                            label="Query mode",
                        )
                        lineclip_dbg_device = gr.Dropdown(
                            choices=["auto", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3", "cpu", "mps"],
                            value="auto",
                            label="Inference Device",
                        )
                    lineclip_dbg_gt_text = gr.Textbox(label="Ground-truth text", lines=4)
                    lineclip_dbg_query_text = gr.Textbox(label="Text query (used for Text -> Image)", lines=4)
                    lineclip_dbg_topk = gr.Number(label="top_k", value=5, precision=0)
                    lineclip_dbg_run_btn = gr.Button("Run line_clip Debug Retrieval", variant="primary")
                    lineclip_dbg_export_csv = gr.Textbox(label="Results CSV Export Path", interactive=False)
                    lineclip_dbg_status = gr.Textbox(label="Debug Retrieval Status", interactive=False)
                    lineclip_dbg_debug_json = gr.Code(label="Debug JSON", language="json")
                    with gr.Accordion("Advanced Retrieval Settings", open=False):
                        lineclip_dbg_l2 = gr.Checkbox(label="l2_normalize_embeddings", value=True)
                        with gr.Row():
                            lineclip_dbg_batch = gr.Number(label="embedding_batch_size", value=32, precision=0)
                            lineclip_dbg_text_max_len = gr.Number(label="text_max_length", value=256, precision=0)
            with gr.Row():
                with gr.Column(scale=1):
                    lineclip_dbg_gt_image = gr.Image(label="Ground Truth Line (reference)", type="numpy")
                with gr.Column(scale=2):
                    lineclip_dbg_results_gallery = gr.Gallery(
                        label="Top-K Retrieval Results (side-by-side vs ground truth)",
                        columns=5,
                        rows=1,
                        object_fit="contain",
                        height=280,
                    )

            lineclip_dbg_scan_models_btn.click(
                fn=scan_line_clip_dual_models_ui,
                inputs=[lineclip_dbg_models_dir],
                outputs=[
                    lineclip_dbg_bundle_select,
                    lineclip_dbg_img_backbone,
                    lineclip_dbg_img_head,
                    lineclip_dbg_txt_encoder,
                    lineclip_dbg_txt_head,
                    lineclip_dbg_bundle_state,
                    lineclip_dbg_model_scan_status,
                ],
            )
            lineclip_dbg_bundle_select.change(
                fn=on_line_clip_bundle_change_ui,
                inputs=[lineclip_dbg_bundle_select, lineclip_dbg_bundle_state],
                outputs=[
                    lineclip_dbg_img_backbone,
                    lineclip_dbg_img_head,
                    lineclip_dbg_txt_encoder,
                    lineclip_dbg_txt_head,
                ],
            )
            lineclip_dbg_scan_datasets_btn.click(
                fn=scan_donut_ocr_datasets_ui,
                inputs=[lineclip_dbg_datasets_dir],
                outputs=[
                    lineclip_dbg_dataset_root,
                    lineclip_dbg_split,
                    lineclip_dbg_dataset_split_state,
                    lineclip_dbg_dataset_scan_status,
                ],
            )
            lineclip_dbg_dataset_root.change(
                fn=on_donut_ocr_dataset_change_ui,
                inputs=[lineclip_dbg_dataset_root, lineclip_dbg_dataset_split_state],
                outputs=[lineclip_dbg_split],
            )
            lineclip_dbg_load_sample_btn.click(
                fn=load_random_line_clip_debug_sample_ui,
                inputs=[lineclip_dbg_dataset_root, lineclip_dbg_split, lineclip_dbg_seed],
                outputs=[
                    lineclip_dbg_sample_image,
                    lineclip_dbg_gt_text,
                    lineclip_dbg_query_text,
                    lineclip_dbg_sample_meta,
                    lineclip_dbg_sample_path,
                    lineclip_dbg_sample_state,
                ],
            )
            lineclip_dbg_run_btn.click(
                fn=run_line_clip_dataset_debug_retrieval_ui,
                inputs=[
                    lineclip_dbg_sample_image,
                    lineclip_dbg_gt_text,
                    lineclip_dbg_query_text,
                    lineclip_dbg_query_mode,
                    lineclip_dbg_dataset_root,
                    lineclip_dbg_split,
                    lineclip_dbg_img_backbone,
                    lineclip_dbg_img_head,
                    lineclip_dbg_txt_encoder,
                    lineclip_dbg_txt_head,
                    lineclip_dbg_device,
                    lineclip_dbg_batch,
                    lineclip_dbg_text_max_len,
                    lineclip_dbg_l2,
                    lineclip_dbg_topk,
                    lineclip_dbg_sample_state,
                ],
                outputs=[
                    lineclip_dbg_gt_image,
                    lineclip_dbg_results_gallery,
                    lineclip_dbg_debug_json,
                    lineclip_dbg_export_csv,
                    lineclip_dbg_status,
                    lineclip_dbg_sample_state,
                ],
            )

        # 14) Retrieval advanced tools (training / eval / FAISS)
        with gr.Tab("14. Retrieval Advanced"):
            gr.Markdown(
                "Advanced retrieval tools: training, evaluation, and FAISS search for Tibetan n-gram retrieval. "
                "Use the dedicated `line_clip Workbench` tab for pretrained line_clip experiments."
            )

            gr.Markdown("### Shared Artifact Scan (Backbone / Projection / FAISS)")
            with gr.Row():
                retr_models_dir = gr.Textbox(label="models_dir", value=str((workspace_root / "models").resolve()))
                retr_scan_btn = gr.Button("Scan Retrieval Artifacts")
            with gr.Row():
                retr_backbone_select = gr.Dropdown(label="Detected Backbone Dirs", choices=[], allow_custom_value=True)
                retr_head_select = gr.Dropdown(label="Detected Projection Heads", choices=[], allow_custom_value=True)
                retr_faiss_select = gr.Dropdown(label="Detected FAISS Indices", choices=[], allow_custom_value=True)
            retr_scan_status = gr.Textbox(label="Artifact Scan Status", interactive=False)
            with gr.Row():
                retr_backbone_path = gr.Textbox(label="shared_backbone_dir", value=default_text_hierarchy_backbone_dir)
                retr_head_path = gr.Textbox(label="shared_projection_head_path", value=default_text_hierarchy_projection_head)
            with gr.Row():
                retr_faiss_index_path = gr.Textbox(label="shared_faiss_index_path", value=default_text_hierarchy_faiss_index)
                retr_faiss_meta_path = gr.Textbox(
                    label="shared_faiss_meta_path",
                    value=f"{default_text_hierarchy_faiss_index}.meta.json",
                )

            gr.Markdown("### A) Train Image Encoder (SimCLR-style)")
            with gr.Row():
                with gr.Column(scale=1):
                    image_enc_input_dir = gr.Textbox(label="input_dir (images)", value=default_image_encoder_input_dir)
                    image_enc_output_dir = gr.Textbox(label="output_dir", value=default_image_encoder_output_dir)
                    image_enc_model = gr.Textbox(label="model_name_or_path", value="facebook/dinov2-base")
                    with gr.Row():
                        image_enc_resolution = gr.Number(label="resolution", value=448, precision=0)
                        image_enc_batch_size = gr.Number(label="batch_size", value=8, precision=0)
                    with gr.Row():
                        image_enc_lr = gr.Number(label="lr", value=1e-4, precision=6)
                        image_enc_weight_decay = gr.Number(label="weight_decay", value=0.01, precision=4)
                    with gr.Row():
                        image_enc_epochs = gr.Number(label="num_train_epochs", value=5, precision=0)
                        image_enc_max_steps = gr.Number(label="max_train_steps (0=use epochs)", value=0, precision=0)
                    with gr.Row():
                        image_enc_warmup = gr.Number(label="warmup_steps", value=200, precision=0)
                        image_enc_proj_dim = gr.Number(label="projection_dim", value=256, precision=0)
                    with gr.Row():
                        image_enc_temperature = gr.Slider(0.01, 1.0, value=0.1, step=0.01, label="temperature")
                        image_enc_mixed_precision = gr.Dropdown(
                            choices=["no", "fp16", "bf16"],
                            value="fp16",
                            label="mixed_precision",
                        )
                    with gr.Row():
                        image_enc_workers = gr.Number(label="num_workers", value=4, precision=0)
                        image_enc_seed = gr.Number(label="seed", value=42, precision=0)
                    with gr.Row():
                        image_enc_checkpoint_every = gr.Number(label="checkpoint_every_steps (0=off)", value=0, precision=0)
                        image_enc_gc = gr.Checkbox(label="gradient_checkpointing", value=False)
                    image_enc_freeze = gr.Checkbox(label="freeze_backbone", value=False)
                    image_enc_run_btn = gr.Button("Train Image Encoder", variant="primary")
                with gr.Column(scale=1):
                    image_enc_log = gr.Textbox(label="Image Encoder Log", lines=16, interactive=False)
                    image_enc_backbone_path = gr.Textbox(label="Backbone Path", interactive=False)
                    image_enc_head_path = gr.Textbox(label="Projection Head Path", interactive=False)
                    image_enc_cfg_path = gr.Textbox(label="Training Config Path", interactive=False)

            gr.Markdown("### B) Evaluate Patch/Hierarchy ViT (Recall@K / MRR)")
            with gr.Row():
                with gr.Column(scale=1):
                    eval_th_dataset_dir = gr.Textbox(label="dataset_dir (patch dataset root)", value=default_text_hierarchy_dir)
                    eval_th_output_dir = gr.Textbox(label="output_dir", value=default_text_hierarchy_eval_dir)
                    eval_th_config_path = gr.Textbox(label="config_path (optional)", value="")
                    with gr.Row():
                        eval_th_include_lines = gr.Checkbox(label="include_line_images", value=True)
                        eval_th_include_words = gr.Checkbox(label="include_word_crops", value=True)
                        eval_th_include_numbers = gr.Checkbox(label="include_number_crops", value=False)
                    with gr.Row():
                        eval_th_min_assets = gr.Number(label="min_assets_per_group", value=1, precision=0)
                        eval_th_min_pos = gr.Number(label="min_positives_per_query", value=1, precision=0)
                    with gr.Row():
                        eval_th_target_h = gr.Number(label="target_height (0=auto)", value=0, precision=0)
                        eval_th_max_w = gr.Number(label="max_width (0=auto)", value=0, precision=0)
                    with gr.Row():
                        eval_th_patch_mult = gr.Number(label="patch_multiple (0=auto)", value=0, precision=0)
                        eval_th_width_buckets = gr.Textbox(label="width_buckets (optional)", value="")
                    with gr.Row():
                        eval_th_batch = gr.Number(label="batch_size", value=32, precision=0)
                        eval_th_workers = gr.Number(label="num_workers", value=4, precision=0)
                    with gr.Row():
                        eval_th_device = gr.Dropdown(choices=["auto", "cpu", "cuda", "mps"], value="auto", label="device")
                        eval_th_l2 = gr.Checkbox(label="l2_normalize_embeddings", value=True)
                    with gr.Row():
                        eval_th_recall_ks = gr.Textbox(label="recall_ks", value="1,5,10")
                        eval_th_max_queries = gr.Number(label="max_queries (0=all)", value=0, precision=0)
                    with gr.Row():
                        eval_th_seed = gr.Number(label="seed", value=42, precision=0)
                        eval_th_write_csv = gr.Checkbox(label="write_per_query_csv", value=True)
                    eval_th_run_btn = gr.Button("Run Hierarchy Eval", variant="primary")
                with gr.Column(scale=1):
                    eval_th_log = gr.Textbox(label="Hierarchy Eval Log", lines=14, interactive=False)
                    eval_th_report_path = gr.Textbox(label="Eval Report Path", interactive=False)
                    eval_th_csv_path = gr.Textbox(label="Per-Query CSV Path", interactive=False)
                    eval_th_report_json = gr.Code(label="Eval Report JSON", language="json")

            gr.Markdown("### C) FAISS Similarity Search")
            faiss_line_click_state = gr.State({})
            faiss_query_state = gr.State({})
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion("Interactive Query Builder (Image -> Text/Line Detection -> Query Range)", open=True):
                        faiss_query_image_in = gr.Image(
                            type="numpy",
                            label="query source image",
                            sources=["upload", "clipboard"],
                        )
                        with gr.Row():
                            faiss_line_models_dir = gr.Textbox(label="line_models_dir", value=str((workspace_root / "models").resolve()))
                            faiss_line_scan_btn = gr.Button("Scan Line Models")
                        faiss_line_model_select = gr.Dropdown(
                            label="Detected Ultralytics Line Model",
                            choices=[],
                            allow_custom_value=True,
                        )
                        faiss_line_model_scan_status = gr.Textbox(label="Line Model Scan Status", interactive=False)
                        faiss_line_model = gr.Textbox(
                            label="line_model_path",
                            value=str((ROOT / "runs" / "detect" / "train" / "weights" / "best.pt").resolve()),
                        )
                        with gr.Row():
                            faiss_line_conf = gr.Slider(0.01, 0.99, value=0.25, step=0.01, label="conf")
                            faiss_line_imgsz = gr.Number(label="imgsz", value=1024, precision=0)
                        with gr.Row():
                            faiss_line_device = gr.Textbox(label="line_detection_device", value="")
                            faiss_line_min_height = gr.Number(label="min_line_height_px", value=10, precision=0)
                        with gr.Row():
                            faiss_line_proj_smooth = gr.Number(label="projection_smooth_rows", value=9, precision=0)
                            faiss_line_proj_thresh = gr.Slider(0.05, 0.80, value=0.20, step=0.01, label="projection_threshold_rel")
                        faiss_line_merge_gap = gr.Number(label="merge_gap_px", value=5, precision=0)
                        with gr.Row():
                            faiss_detect_btn = gr.Button("Detect Text + Split Lines")
                            faiss_query_reset_btn = gr.Button("Reset Query to Full Line")

                    with gr.Accordion("Query Image Path (optional fallback)", open=False):
                        faiss_query_file = gr.File(label="query_image file", type="filepath")
                        faiss_query_path = gr.Textbox(label="query_image path", value="")

                    faiss_mode = gr.Radio(
                        choices=["Use Existing FAISS Index", "Build FAISS Index From Dataset"],
                        value="Use Existing FAISS Index",
                        label="Index Mode",
                    )
                    faiss_output_dir = gr.Textbox(label="output_dir", value=default_text_hierarchy_faiss_dir)
                    faiss_index_path = gr.Textbox(label="index_path", value=default_text_hierarchy_faiss_index)
                    faiss_meta_path = gr.Textbox(label="meta_path (optional)", value=f"{default_text_hierarchy_faiss_index}.meta.json")
                    faiss_dataset_dir = gr.Textbox(label="dataset_dir (patch dataset, build mode)", value=default_text_hierarchy_dir)
                    faiss_save_index_path = gr.Textbox(label="save_index_path (for build mode)", value=default_text_hierarchy_faiss_index)
                    with gr.Row():
                        faiss_metric = gr.Dropdown(choices=["cosine", "l2"], value="cosine", label="metric")
                        faiss_top_k = gr.Number(label="top_k", value=10, precision=0)
                    with gr.Row():
                        faiss_include_lines = gr.Checkbox(label="include_line_images", value=True)
                        faiss_include_words = gr.Checkbox(label="include_word_crops", value=True)
                        faiss_include_numbers = gr.Checkbox(label="include_number_crops", value=False)
                    with gr.Row():
                        faiss_min_assets = gr.Number(label="min_assets_per_group", value=1, precision=0)
                        faiss_l2 = gr.Checkbox(label="l2_normalize_embeddings", value=True)
                    with gr.Row():
                        faiss_config_path = gr.Textbox(label="config_path (optional)", value="")
                        faiss_device = gr.Dropdown(choices=["auto", "cpu", "cuda", "mps"], value="auto", label="device")
                    with gr.Row():
                        faiss_target_h = gr.Number(label="target_height (0=auto)", value=0, precision=0)
                        faiss_max_w = gr.Number(label="max_width (0=auto)", value=0, precision=0)
                    with gr.Row():
                        faiss_patch_mult = gr.Number(label="patch_multiple (0=auto)", value=0, precision=0)
                        faiss_width_buckets = gr.Textbox(label="width_buckets (optional)", value="")
                    with gr.Row():
                        faiss_batch = gr.Number(label="batch_size", value=32, precision=0)
                        faiss_workers = gr.Number(label="num_workers", value=4, precision=0)
                    faiss_run_btn = gr.Button("Run FAISS Search", variant="primary")
                with gr.Column(scale=1):
                    faiss_overlay = gr.Image(type="numpy", label="Overlay (Detected tibetan_text + Line Boxes)", interactive=True)
                    faiss_line_profile = gr.Image(type="numpy", label="Line Projection Profile")
                    faiss_selected_line_view = gr.Image(type="numpy", label="Selected Line View (click twice for query range)", interactive=True)
                    faiss_selected_line_profile = gr.Image(type="numpy", label="Selected Line Horizontal Profile")
                    faiss_query_crop = gr.Image(type="numpy", label="Selected Query Crop")
                    faiss_detect_status = gr.Textbox(label="Detection Status", interactive=False)
                    faiss_line_select_status = gr.Textbox(label="Line Selection Status", interactive=False)
                    faiss_query_select_status = gr.Textbox(label="Query Selection Status", interactive=False)
                    faiss_detect_json = gr.Code(label="Line Detection JSON", language="json")
                    faiss_log = gr.Textbox(label="FAISS Search Log", lines=12, interactive=False)
                    faiss_report_path = gr.Textbox(label="Search Report Path", interactive=False)
                    faiss_index_out = gr.Textbox(label="Resolved Index Path", interactive=False)
                    faiss_meta_out = gr.Textbox(label="Resolved Metadata Path", interactive=False)
                    faiss_results_json = gr.Code(label="Search Results JSON", language="json")

            image_enc_run_btn.click(
                fn=run_train_image_encoder_live,
                inputs=[
                    image_enc_input_dir,
                    image_enc_output_dir,
                    image_enc_model,
                    image_enc_resolution,
                    image_enc_batch_size,
                    image_enc_lr,
                    image_enc_weight_decay,
                    image_enc_epochs,
                    image_enc_max_steps,
                    image_enc_warmup,
                    image_enc_proj_dim,
                    image_enc_temperature,
                    image_enc_mixed_precision,
                    image_enc_gc,
                    image_enc_freeze,
                    image_enc_workers,
                    image_enc_seed,
                    image_enc_checkpoint_every,
                ],
                outputs=[
                    image_enc_log,
                    image_enc_backbone_path,
                    image_enc_head_path,
                    image_enc_cfg_path,
                ],
            )

            retr_scan_btn.click(
                fn=scan_text_hierarchy_retrieval_artifacts_for_ui,
                inputs=[retr_models_dir],
                outputs=[
                    retr_backbone_select,
                    retr_head_select,
                    retr_faiss_select,
                    retr_backbone_path,
                    retr_head_path,
                    retr_faiss_index_path,
                    retr_faiss_meta_path,
                    retr_scan_status,
                    faiss_index_path,
                    faiss_meta_path,
                ],
            )
            retr_backbone_select.change(
                fn=on_encoder_backbone_change_ui,
                inputs=[retr_backbone_select],
                outputs=[retr_backbone_path, retr_head_path],
            )
            retr_head_select.change(
                fn=lambda x: x or "",
                inputs=[retr_head_select],
                outputs=[retr_head_path],
            )
            retr_faiss_select.change(
                fn=on_faiss_index_change_ui,
                inputs=[retr_faiss_select],
                outputs=[retr_faiss_index_path, retr_faiss_meta_path],
            )

            eval_th_run_btn.click(
                fn=run_eval_text_hierarchy_vit_live,
                inputs=[
                    eval_th_dataset_dir,
                    retr_backbone_path,
                    retr_head_path,
                    eval_th_output_dir,
                    eval_th_config_path,
                    eval_th_include_lines,
                    eval_th_include_words,
                    eval_th_include_numbers,
                    eval_th_min_assets,
                    eval_th_min_pos,
                    eval_th_target_h,
                    eval_th_max_w,
                    eval_th_patch_mult,
                    eval_th_width_buckets,
                    eval_th_batch,
                    eval_th_workers,
                    eval_th_device,
                    eval_th_l2,
                    eval_th_recall_ks,
                    eval_th_max_queries,
                    eval_th_seed,
                    eval_th_write_csv,
                ],
                outputs=[eval_th_log, eval_th_report_path, eval_th_csv_path, eval_th_report_json],
            )

            faiss_line_scan_btn.click(
                fn=scan_ultralytics_inference_models,
                inputs=[faiss_line_models_dir],
                outputs=[faiss_line_model_select, faiss_line_model_scan_status],
            )
            faiss_line_model_select.change(
                fn=lambda x: x or "",
                inputs=[faiss_line_model_select],
                outputs=[faiss_line_model],
            )
            faiss_detect_btn.click(
                fn=run_tibetan_text_line_split_for_faiss_ui,
                inputs=[
                    faiss_query_image_in,
                    faiss_line_model,
                    faiss_line_conf,
                    faiss_line_imgsz,
                    faiss_line_device,
                    faiss_line_min_height,
                    faiss_line_proj_smooth,
                    faiss_line_proj_thresh,
                    faiss_line_merge_gap,
                ],
                outputs=[
                    faiss_overlay,
                    faiss_detect_status,
                    faiss_detect_json,
                    faiss_line_profile,
                    faiss_selected_line_view,
                    faiss_selected_line_profile,
                    faiss_line_select_status,
                    faiss_line_click_state,
                    faiss_query_crop,
                    faiss_query_path,
                    faiss_query_select_status,
                    faiss_query_state,
                ],
            )
            faiss_overlay.select(
                fn=prepare_clicked_line_for_faiss_query_ui,
                inputs=[faiss_line_click_state],
                outputs=[
                    faiss_selected_line_view,
                    faiss_selected_line_profile,
                    faiss_line_select_status,
                    faiss_query_crop,
                    faiss_query_path,
                    faiss_query_select_status,
                    faiss_query_state,
                ],
            )
            faiss_selected_line_view.select(
                fn=select_faiss_query_range_in_line_ui,
                inputs=[faiss_query_state],
                outputs=[
                    faiss_selected_line_view,
                    faiss_query_crop,
                    faiss_query_path,
                    faiss_query_select_status,
                    faiss_query_state,
                ],
            )
            faiss_query_reset_btn.click(
                fn=reset_faiss_query_to_full_line_ui,
                inputs=[faiss_query_state],
                outputs=[
                    faiss_selected_line_view,
                    faiss_query_crop,
                    faiss_query_path,
                    faiss_query_select_status,
                    faiss_query_state,
                ],
            )

            faiss_query_file.change(
                fn=_file_to_path_text,
                inputs=[faiss_query_file],
                outputs=[faiss_query_path],
            )
            retr_faiss_index_path.change(
                fn=lambda x: x or "",
                inputs=[retr_faiss_index_path],
                outputs=[faiss_index_path],
            )
            retr_faiss_meta_path.change(
                fn=lambda x: x or "",
                inputs=[retr_faiss_meta_path],
                outputs=[faiss_meta_path],
            )

            faiss_run_btn.click(
                fn=run_faiss_text_hierarchy_search_live_with_mode,
                inputs=[
                    faiss_mode,
                    faiss_query_path,
                    retr_backbone_path,
                    retr_head_path,
                    faiss_output_dir,
                    faiss_index_path,
                    faiss_meta_path,
                    faiss_dataset_dir,
                    faiss_save_index_path,
                    faiss_metric,
                    faiss_top_k,
                    faiss_include_lines,
                    faiss_include_words,
                    faiss_include_numbers,
                    faiss_min_assets,
                    faiss_config_path,
                    faiss_target_h,
                    faiss_max_w,
                    faiss_patch_mult,
                    faiss_width_buckets,
                    faiss_batch,
                    faiss_workers,
                    faiss_device,
                    faiss_l2,
                ],
                outputs=[faiss_log, faiss_report_path, faiss_index_out, faiss_meta_out, faiss_results_json],
            )

        # 14) CLI reference
        with gr.Tab("14. CLI Audit"):
            audit_btn = gr.Button("Scan All CLI Options")
            audit_out = gr.Markdown()
            audit_btn.click(fn=collect_cli_help, inputs=[], outputs=[audit_out])

    return demo


if __name__ == "__main__":
    app = build_ui()
    host = os.environ.get("UI_HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("UI_PORT", "7860"))
    except ValueError:
        port = 7860
    share = os.environ.get("UI_SHARE", "").strip().lower() in {"1", "true", "yes", "on"}
    app.launch(server_name=host, server_port=port, share=share)
