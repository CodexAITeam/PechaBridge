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
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw


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


def _read_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    rows = []
    if not label_path.exists():
        return rows
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            rows.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
        except Exception:
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
    labels = _read_yolo_labels(label_path)

    class_colors = {
        0: (255, 140, 0),
        1: (0, 220, 255),
        2: (130, 255, 130),
    }
    lines = []
    for i, (cls, cx, cy, bw, bh) in enumerate(labels, start=1):
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        color = class_colors.get(cls, (255, 80, 80))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text((x1 + 2, max(0, y1 - 14)), f"{cls}", fill=color)
        lines.append(f"{i}. class={cls} cx={cx:.4f} cy={cy:.4f} w={bw:.4f} h={bh:.4f}")

    summary = f"Found {len(labels)} boxes\n" + ("\n".join(lines[:25]) if lines else "")
    if len(lines) > 25:
        summary += f"\n... +{len(lines)-25} more"
    return np.array(img), summary


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
        (mixed_precision or "fp16").strip(),
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
    train_tokenizer: bool,
    tokenizer_vocab_size: int,
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
        "--tokenizer_vocab_size",
        str(int(tokenizer_vocab_size)),
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
    if train_tokenizer:
        cmd.append("--train_tokenizer")
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
        draw.rectangle((tx, ty, tx + 9 * len(tag), ty + 14), fill=(0, 0, 0))
        draw.text((tx + 2, ty + 1), tag, fill=color)

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
    best_model = Path(project).expanduser().resolve() / name / "weights" / "best.pt"
    status = "Success" if ok else "Failed"
    return f"{status}\nBest model path: {best_model}\n\n{out}", str(best_model)


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
    best_model = Path(project).expanduser().resolve() / name / "weights" / "best.pt"

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
        msg = f"Failed\nBest model path: {best_model}\n\n{type(exc).__name__}: {exc}"
        yield msg, str(best_model)
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

    yield f"Running ...\nBest model path: {best_model}\n", str(best_model)

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
            running_msg = f"Running ...\nBest model path: {best_model}\n\n{tail}"
            yield running_msg, str(best_model)
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
    final_msg = f"{status}\nBest model path: {best_model}\n\n" + _tail_lines_newest_first(log_lines, 3000)
    yield final_msg, str(best_model)


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
):
    syn = Path(synthetic_split_dir).expanduser().resolve()
    sbb = Path(sbb_test_split_dir).expanduser().resolve()
    out = Path(combined_output_split_dir).expanduser().resolve()
    out_images = out / "images"
    out_labels = out / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    if not (syn / "images").exists() or not (syn / "labels").exists():
        return f"Synthetic split invalid: {syn}", ""
    if not (sbb / "images").exists() or not (sbb / "labels").exists():
        return f"SBB test split invalid: {sbb}", ""

    def _copy_split(src_split: Path, prefix: str):
        cnt = 0
        for img in sorted((src_split / "images").glob("*")):
            if not img.is_file():
                continue
            new_name = f"{prefix}_{img.name}"
            target_img = out_images / new_name
            target_lbl = out_labels / f"{Path(new_name).stem}.txt"
            lbl_src = src_split / "labels" / f"{img.stem}.txt"
            target_img.write_bytes(img.read_bytes())
            if lbl_src.exists():
                target_lbl.write_text(lbl_src.read_text(encoding="utf-8"), encoding="utf-8")
            else:
                target_lbl.write_text("", encoding="utf-8")
            cnt += 1
        return cnt

    syn_count = _copy_split(syn, "syn")
    sbb_count = _copy_split(sbb, "sbb")

    (out / "classes.txt").write_text(
        "tibetan_number_word\ntibetan_text\nchinese_number_word\n",
        encoding="utf-8",
    )

    msg = (
        f"Combined split created at {out}\n"
        f"Synthetic images: {syn_count}\n"
        f"SBB test images: {sbb_count}\n"
        "Use this split only for annotation/review; keep SBB out of train/val model training."
    )
    return msg, str(out)


def prepare_combined_for_labelstudio_ui(
    synthetic_split_dir: str,
    sbb_test_split_dir: str,
    combined_output_split_dir: str,
    current_split_dir: str,
    current_local_files_root: str,
    current_image_root_url: str,
    current_tasks_json: str,
    current_vlm_export_split_dir: str,
    current_vlm_export_image_root_url: str,
    current_vlm_export_tasks_json: str,
):
    msg, combined = prepare_combined_labelstudio_split(
        synthetic_split_dir=synthetic_split_dir,
        sbb_test_split_dir=sbb_test_split_dir,
        combined_output_split_dir=combined_output_split_dir,
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
    msg2 = f"{msg}\n\nLabel Studio fields updated in tab 3."

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
    default_dataset_base = str((ROOT / "datasets").resolve())
    default_dataset = str((ROOT / "datasets" / "tibetan-yolo").resolve())
    default_split_dir = str((ROOT / "datasets" / "tibetan-yolo" / "train").resolve())
    default_donut_dataset_name = "tibetan-donut-ocr-label1"
    default_donut_dataset_output_dir = str((ROOT / "datasets").resolve())
    default_donut_model_output_dir = str((ROOT / "models" / "donut-ocr-label1").resolve())
    default_donut_font_tibetan = str((ROOT / "ext" / "Microsoft Himalaya.ttf").resolve())
    default_donut_font_chinese = str((ROOT / "ext" / "simkai.ttf").resolve())
    default_texture_input_dir = str((ROOT / "datasets" / "tibetan-yolo-ui" / "train" / "images").resolve())
    default_texture_output_dir = str((ROOT / "datasets" / "tibetan-yolo-ui-textured").resolve())
    default_texture_real_pages_dir = str((ROOT / "sbb_images").resolve())
    default_sbb_grid_dir = str((ROOT / "sbb_images").resolve())
    default_texture_lora_dataset_dir = str((ROOT / "datasets" / "texture-lora-dataset").resolve())
    default_texture_lora_output_dir = str((ROOT / "models" / "texture-lora-sdxl").resolve())
    default_image_encoder_input_dir = str((ROOT / "sbb_images").resolve())
    default_image_encoder_output_dir = str((ROOT / "models" / "image-encoder").resolve())
    default_text_encoder_input_dir = str((ROOT / "data" / "corpora").resolve())
    default_text_encoder_output_dir = str((ROOT / "models" / "text-encoder").resolve())
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

        # 1) Hello / workflow overview
        with gr.Tab("1. Hello"):
            gr.Markdown("## Workflow Overview")
            gr.Markdown(
                "Use the tabs left-to-right. A practical flow is: Synthetic Data -> Diffusion + LoRA (texture) -> "
                "Donut OCR Workflow -> Retrieval Encoders -> Batch VLM Layout (SBB) -> Dataset Preview -> Ultralytics Training -> Model Inference -> "
                "VLM Layout (single image) -> Label Studio Export."
            )
            gr.Markdown("### Tabs")
            gr.Markdown(
                "1. Hello: Short guide and workflow.\n"
                "2. Synthetic Data: Generate synthetic training/validation data.\n"
                "3. Batch VLM Layout (SBB): Batch-annotate SBB PPNs, combine with synthetic data, export.\n"
                "4. Dataset Preview: Visual QA with bounding boxes.\n"
                "5. Ultralytics Training: Train YOLO models.\n"
                "6. Model Inference: Run inference with trained models.\n"
                "7. VLM Layout: Run transformer-based layout parsing on a single image.\n"
                "8. Label Studio Export: Convert YOLO split folders to Label Studio tasks and launch Label Studio.\n"
                "9. PPN Downloader: Download and analyze SBB images.\n"
                "10. SBB Grid Review: Browse sbb_images in 9x9 pages and move selected pages to quarantine.\n"
                "11. Diffusion + LoRA: Prepare texture crops, train LoRA, and run SDXL/SD2.1 + ControlNet inference.\n"
                "12. Donut OCR Workflow: Run synthetic generation + manifest prep + Donut-style OCR training on label 1.\n"
                "13. Retrieval Encoders: Train image and text encoders for future n-gram retrieval.\n"
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
                    batch_vlm_combine_syn_split = gr.Textbox(
                        label="Synthetic split directory (train or val)",
                        value=str((ROOT / "datasets" / "tibetan-yolo" / "train").resolve()),
                    )
                    batch_vlm_combine_sbb_split = gr.Textbox(
                        label="SBB test split directory",
                        value=str((ROOT / "datasets" / "sbb-vlm-layout" / "test").resolve()),
                    )
                    batch_vlm_combine_out_split = gr.Textbox(
                        label="Combined output split directory",
                        value=str((ROOT / "datasets" / "combined-layout-labelstudio" / "train").resolve()),
                    )
                    batch_vlm_combine_btn = gr.Button("Prepare Combined Split + Fill Label Studio Tab", variant="secondary")
                with gr.Column(scale=1):
                    batch_vlm_combine_status = gr.Textbox(label="Combine Status", lines=8, interactive=False)
                    batch_vlm_combine_result = gr.Textbox(label="Combined Split Path", interactive=False)

            batch_vlm_ppn_test_split.change(
                fn=lambda x: x,
                inputs=[batch_vlm_ppn_test_split],
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
                    train_project = gr.Textbox(label="project", value="runs/detect")
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
            gr.Markdown("Convert YOLO split directory to Label Studio tasks.")
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

            batch_vlm_combine_btn.click(
                fn=prepare_combined_for_labelstudio_ui,
                inputs=[
                    batch_vlm_combine_syn_split,
                    batch_vlm_combine_sbb_split,
                    batch_vlm_combine_out_split,
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
                sbb_grid_prev_btn = gr.Button("Zurueck", elem_id="sbb-grid-prev-btn")
                sbb_grid_next_btn = gr.Button("Vor", elem_id="sbb-grid-next-btn")
                sbb_grid_quarantine_btn = gr.Button("Quarantine Selected", variant="primary")
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
                            value="fp16",
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

        # 12) Donut OCR workflow
        with gr.Tab("12. Donut OCR Workflow"):
            gr.Markdown(
                "Run label-1 Donut-style OCR training end-to-end "
                "(synthetic generation -> OCR manifest prep -> Vision Transformer encoder + autoregressive decoder training)."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    donut_dataset_name = gr.Textbox(label="dataset_name", value=default_donut_dataset_name)
                    donut_dataset_output_dir = gr.Textbox(label="dataset_output_dir", value=default_donut_dataset_output_dir)
                    donut_prepared_output_dir = gr.Textbox(
                        label="prepared_output_dir (optional)",
                        value="",
                        placeholder="Leave empty for <dataset>/donut_ocr_label1",
                    )
                    donut_model_output_dir = gr.Textbox(label="model_output_dir", value=default_donut_model_output_dir)
                    donut_model_name = gr.Textbox(
                        label="model_name_or_path",
                        value="microsoft/trocr-base-stage1",
                    )
                    with gr.Row():
                        donut_train_samples = gr.Number(label="train_samples", value=2000, precision=0)
                        donut_val_samples = gr.Number(label="val_samples", value=200, precision=0)
                    with gr.Row():
                        donut_font_tibetan = gr.Textbox(label="font_path_tibetan", value=default_donut_font_tibetan)
                        donut_font_chinese = gr.Textbox(label="font_path_chinese", value=default_donut_font_chinese)
                    with gr.Row():
                        donut_augmentation = gr.Dropdown(
                            choices=["rotate", "noise", "none"],
                            value="noise",
                            label="augmentation",
                        )
                        donut_newline_token = gr.Dropdown(
                            choices=["<NL>", "\\n"],
                            value="<NL>",
                            label="target_newline_token",
                        )
                    with gr.Row():
                        donut_train_batch = gr.Number(label="per_device_train_batch_size", value=4, precision=0)
                        donut_eval_batch = gr.Number(label="per_device_eval_batch_size", value=4, precision=0)
                    with gr.Row():
                        donut_epochs = gr.Number(label="num_train_epochs", value=8.0)
                        donut_lr = gr.Number(label="learning_rate", value=5e-5)
                    with gr.Row():
                        donut_max_target_length = gr.Number(label="max_target_length", value=512, precision=0)
                        donut_image_size = gr.Number(label="image_size", value=384, precision=0)
                    with gr.Row():
                        donut_seed = gr.Number(label="seed", value=42, precision=0)
                        donut_tokenizer_vocab_size = gr.Number(label="tokenizer_vocab_size", value=16000, precision=0)
                    with gr.Row():
                        donut_train_tokenizer = gr.Checkbox(label="train_tokenizer", value=True)
                        donut_skip_generation = gr.Checkbox(label="skip_generation", value=False)
                        donut_skip_prepare = gr.Checkbox(label="skip_prepare", value=False)
                        donut_skip_train = gr.Checkbox(label="skip_train", value=False)

                    with gr.Accordion("Optional LoRA augmentation during generation", open=False):
                        donut_lora_path = gr.Textbox(
                            label="lora_augment_path",
                            value="",
                            placeholder="Optional path to LoRA .safetensors or folder",
                        )
                        with gr.Row():
                            donut_lora_family = gr.Dropdown(
                                choices=["sdxl", "sd21"],
                                value="sdxl",
                                label="lora_augment_model_family",
                            )
                            donut_lora_targets = gr.Dropdown(
                                choices=["images", "images_and_ocr_crops"],
                                value="images_and_ocr_crops",
                                label="lora_augment_targets",
                            )
                        donut_lora_prompt = gr.Textbox(
                            label="lora_augment_prompt",
                            value=DEFAULT_TEXTURE_PROMPT,
                            lines=2,
                        )
                        with gr.Row():
                            donut_lora_splits = gr.Textbox(label="lora_augment_splits", value="train")
                            donut_lora_seed = gr.Number(label="lora_augment_seed (-1=unset)", value=-1, precision=0)
                        with gr.Row():
                            donut_lora_scale = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="lora_augment_scale")
                            donut_lora_strength = gr.Slider(0.0, 0.25, value=0.2, step=0.01, label="lora_augment_strength")
                            donut_lora_steps = gr.Number(label="lora_augment_steps", value=28, precision=0)
                        with gr.Row():
                            donut_lora_guidance = gr.Slider(0.0, 4.0, value=1.0, step=0.1, label="lora_augment_guidance_scale")
                            donut_lora_controlnet = gr.Slider(0.5, 3.0, value=2.0, step=0.1, label="lora_augment_controlnet_scale")
                        with gr.Row():
                            donut_lora_canny_low = gr.Number(label="lora_augment_canny_low", value=100, precision=0)
                            donut_lora_canny_high = gr.Number(label="lora_augment_canny_high", value=200, precision=0)
                        donut_lora_base_model = gr.Textbox(
                            label="lora_augment_base_model_id",
                            value="stabilityai/stable-diffusion-xl-base-1.0",
                        )
                        donut_lora_controlnet_model = gr.Textbox(
                            label="lora_augment_controlnet_model_id",
                            value="diffusers/controlnet-canny-sdxl-1.0",
                        )

                    donut_run_btn = gr.Button("Run Donut OCR Workflow", variant="primary")
                with gr.Column(scale=1):
                    donut_log = gr.Textbox(label="Workflow Log", lines=24, interactive=False)
                    donut_dataset_dir = gr.Textbox(label="Dataset Dir", interactive=False)
                    donut_prepared_dir = gr.Textbox(label="Prepared Manifest Dir", interactive=False)
                    donut_model_dir = gr.Textbox(label="Model Output Dir", interactive=False)
                    donut_summary_path = gr.Textbox(label="Workflow Summary Path", interactive=False)

            donut_run_btn.click(
                fn=run_donut_ocr_workflow_live,
                inputs=[
                    donut_dataset_name,
                    donut_dataset_output_dir,
                    donut_train_samples,
                    donut_val_samples,
                    donut_font_tibetan,
                    donut_font_chinese,
                    donut_augmentation,
                    donut_newline_token,
                    donut_prepared_output_dir,
                    donut_model_output_dir,
                    donut_model_name,
                    donut_train_tokenizer,
                    donut_tokenizer_vocab_size,
                    donut_train_batch,
                    donut_eval_batch,
                    donut_epochs,
                    donut_lr,
                    donut_max_target_length,
                    donut_image_size,
                    donut_seed,
                    donut_skip_generation,
                    donut_skip_prepare,
                    donut_skip_train,
                    donut_lora_path,
                    donut_lora_family,
                    donut_lora_base_model,
                    donut_lora_controlnet_model,
                    donut_lora_prompt,
                    donut_lora_scale,
                    donut_lora_strength,
                    donut_lora_steps,
                    donut_lora_guidance,
                    donut_lora_controlnet,
                    donut_lora_seed,
                    donut_lora_splits,
                    donut_lora_targets,
                    donut_lora_canny_low,
                    donut_lora_canny_high,
                ],
                outputs=[
                    donut_log,
                    donut_dataset_dir,
                    donut_prepared_dir,
                    donut_model_dir,
                    donut_summary_path,
                ],
            )

        # 13) Retrieval encoder training
        with gr.Tab("13. Retrieval Encoders"):
            gr.Markdown(
                "Train unpaired encoders for future Tibetan n-gram retrieval: "
                "A) self-supervised image encoder on page images, "
                "B) unsupervised text encoder on Tibetan text files."
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

            gr.Markdown("### B) Train Text Encoder (SimCSE-style)")
            with gr.Row():
                with gr.Column(scale=1):
                    text_enc_input_dir = gr.Textbox(label="input_dir (text files)", value=default_text_encoder_input_dir)
                    text_enc_output_dir = gr.Textbox(label="output_dir", value=default_text_encoder_output_dir)
                    text_enc_model = gr.Textbox(label="model_name_or_path", value="google/byt5-small")
                    with gr.Row():
                        text_enc_normalization = gr.Dropdown(
                            choices=["NFC", "NFKC", "NFD", "NFKD", "none"],
                            value="NFC",
                            label="normalization",
                        )
                        text_enc_max_length = gr.Number(label="max_length", value=256, precision=0)
                    with gr.Row():
                        text_enc_min_chars = gr.Number(label="min_chars", value=2, precision=0)
                        text_enc_max_chars = gr.Number(label="max_chars", value=512, precision=0)
                    with gr.Row():
                        text_enc_batch_size = gr.Number(label="batch_size", value=16, precision=0)
                        text_enc_lr = gr.Number(label="lr", value=5e-5, precision=6)
                    with gr.Row():
                        text_enc_weight_decay = gr.Number(label="weight_decay", value=0.01, precision=4)
                        text_enc_epochs = gr.Number(label="num_train_epochs", value=5, precision=0)
                    with gr.Row():
                        text_enc_max_steps = gr.Number(label="max_train_steps (0=use epochs)", value=0, precision=0)
                        text_enc_warmup = gr.Number(label="warmup_steps", value=200, precision=0)
                    with gr.Row():
                        text_enc_proj_dim = gr.Number(label="projection_dim", value=256, precision=0)
                        text_enc_temperature = gr.Slider(0.01, 1.0, value=0.05, step=0.01, label="temperature")
                    with gr.Row():
                        text_enc_mixed_precision = gr.Dropdown(
                            choices=["no", "fp16", "bf16"],
                            value="fp16",
                            label="mixed_precision",
                        )
                        text_enc_workers = gr.Number(label="num_workers", value=4, precision=0)
                    with gr.Row():
                        text_enc_seed = gr.Number(label="seed", value=42, precision=0)
                        text_enc_checkpoint_every = gr.Number(label="checkpoint_every_steps (0=off)", value=0, precision=0)
                    with gr.Row():
                        text_enc_gc = gr.Checkbox(label="gradient_checkpointing", value=False)
                        text_enc_freeze = gr.Checkbox(label="freeze_backbone", value=False)
                    text_enc_run_btn = gr.Button("Train Text Encoder", variant="primary")
                with gr.Column(scale=1):
                    text_enc_log = gr.Textbox(label="Text Encoder Log", lines=16, interactive=False)
                    text_enc_backbone_path = gr.Textbox(label="Backbone Path", interactive=False)
                    text_enc_tokenizer_path = gr.Textbox(label="Tokenizer Path", interactive=False)
                    text_enc_head_path = gr.Textbox(label="Projection Head Path", interactive=False)
                    text_enc_cfg_path = gr.Textbox(label="Training Config Path", interactive=False)

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

            text_enc_run_btn.click(
                fn=run_train_text_encoder_live,
                inputs=[
                    text_enc_input_dir,
                    text_enc_output_dir,
                    text_enc_model,
                    text_enc_normalization,
                    text_enc_min_chars,
                    text_enc_max_chars,
                    text_enc_max_length,
                    text_enc_batch_size,
                    text_enc_lr,
                    text_enc_weight_decay,
                    text_enc_epochs,
                    text_enc_max_steps,
                    text_enc_warmup,
                    text_enc_proj_dim,
                    text_enc_temperature,
                    text_enc_mixed_precision,
                    text_enc_gc,
                    text_enc_freeze,
                    text_enc_workers,
                    text_enc_seed,
                    text_enc_checkpoint_every,
                ],
                outputs=[
                    text_enc_log,
                    text_enc_backbone_path,
                    text_enc_tokenizer_path,
                    text_enc_head_path,
                    text_enc_cfg_path,
                ],
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
