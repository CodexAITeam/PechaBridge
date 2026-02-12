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
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent

CLI_SCRIPTS = [
    "generate_training_data.py",
    "train_model.py",
    "inference_sbb.py",
    "ocr_on_detections.py",
    "pseudo_label_from_vlm.py",
    "layout_rule_filter.py",
    "run_pseudo_label_workflow.py",
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
        except OSError as exc:
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
        if spec is None:
            labels.append(key)
        else:
            labels.append(f"{key} - {spec.display_name}")
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

        for res in results:
            if not hasattr(res, "boxes") or res.boxes is None:
                continue
            boxes = res.boxes
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else []
            confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else []
            clss = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else []
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
                c = float(confs[i]) if i < len(confs) else 0.0
                cls = int(clss[i]) if i < len(clss) else 0
                detections.append({
                    "class": cls,
                    "confidence": c,
                    "box": [x1, y1, x2, y2],
                })
                color = (0, 220, 255)
                draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
                draw.text((x1 + 2, max(0, y1 - 14)), f"{cls}:{c:.2f}", fill=color)

        status = f"{len(detections)} detections"
        return np.array(overlay), status, json.dumps(detections, ensure_ascii=False, indent=2)
    except Exception as exc:
        return image, f"Inference failed: {type(exc).__name__}: {exc}", "[]"


def download_ppn_images(
    ppn: str,
    output_dir: str,
    max_images: int,
    no_ssl_verify: bool,
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
        for url in urls:
            saved = download_image(url, output_dir=str(out_dir), verify_ssl=verify_ssl)
            if saved:
                downloaded.append(Path(saved))

        md = get_sbb_metadata(ppn.strip(), verify_ssl=verify_ssl)
        names = sorted([p.name for p in downloaded])
        summary = (
            f"Downloaded {len(downloaded)} image(s) for PPN {ppn} to {out_dir}\n"
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


def run_vlm_on_ppn_list(
    ppn_list_text: str,
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

    parser_key = _extract_parser_key(parser_choice)
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
                "Use the tabs left-to-right. A practical flow is: Synthetic Data -> Batch VLM Layout (SBB) -> "
                "Dataset Preview -> Ultralytics Training -> Model Inference -> VLM Layout (single image) -> Label Studio Export."
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
                "10. CLI Audit: Show all CLI options from project scripts."
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
                    batch_vlm_ppn_run_btn = gr.Button("Run VLM on PPN List", variant="primary")

            batch_vlm_ppn_summary = gr.Textbox(label="Batch Status", lines=8, interactive=False)
            batch_vlm_ppn_test_split = gr.Textbox(label="Generated SBB Test Split", interactive=False)
            batch_vlm_ppn_logs = gr.Textbox(label="Batch Logs", lines=10, interactive=False)

            batch_vlm_ppn_run_btn.click(
                fn=run_vlm_on_ppn_list,
                inputs=[
                    batch_vlm_ppn_list,
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
                inputs=[ppn_value, ppn_output_dir, ppn_max_images, ppn_no_ssl],
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

        # 10) CLI reference
        with gr.Tab("10. CLI Audit"):
            audit_btn = gr.Button("Scan All CLI Options")
            audit_out = gr.Markdown()
            audit_btn.click(fn=collect_cli_help, inputs=[], outputs=[audit_out])

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch()
