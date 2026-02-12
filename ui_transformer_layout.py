#!/usr/bin/env python3
"""
Interactive UI for transformer-based layout/OCR parsing.

Features:
- Select parser backend
- Optional model-id override
- Editable prompt (with sensible default)
- Paste or upload image
- Render detected areas as bounding boxes over image
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from tibetan_utils.parsers import create_parser, list_parser_specs, parser_availability


DEFAULT_PROMPT = (
    "Extract page layout blocks and OCR text. "
    "Return strict JSON with key 'detections' containing a list of objects with: "
    "text, label, confidence, and bbox=[x0,y0,x1,y1]."
)

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


def _format_parser_choices() -> List[str]:
    labels: List[str] = []
    specs = {s.key: s for s in list_parser_specs()}
    for key in TRANSFORMER_PARSERS:
        spec = specs.get(key)
        if spec is None:
            continue
        labels.append(f"{key} - {spec.display_name}")
    return labels


def _extract_key(parser_choice: str) -> str:
    return parser_choice.split(" - ", 1)[0].strip()


@lru_cache(maxsize=8)
def _build_backend(
    parser_key: str,
    hf_model_id: str,
    prompt: str,
    max_new_tokens: int,
    hf_device: str,
    hf_dtype: str,
    mineru_command: str,
    mineru_timeout: int,
):
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

    # Case A: normalized center+size.
    if max(abs(x), abs(y), abs(w), abs(h)) <= 1.5:
        x1 = int((x - w / 2.0) * image_width)
        y1 = int((y - h / 2.0) * image_height)
        x2 = int((x + w / 2.0) * image_width)
        y2 = int((y + h / 2.0) * image_height)
    # Case B: xyxy packed into x,y,width,height (legacy interop quirk).
    elif x >= 0 and y >= 0 and w > x and h > y and w <= image_width and h <= image_height:
        x1 = int(x)
        y1 = int(y)
        x2 = int(w)
        y2 = int(h)
    # Case C: absolute center+size in pixels.
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


def _render_overlay(image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    base = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(base)

    color_by_class = {
        0: (255, 120, 0),    # orange
        1: (0, 200, 255),    # cyan
        2: (120, 255, 120),  # green
    }

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
        conf = det.get("confidence", 0.0)
        text = f"{label} ({conf:.2f})"
        tx, ty = xyxy[0], max(0, xyxy[1] - 16)
        draw.rectangle((tx, ty, tx + 9 * len(text), ty + 14), fill=(0, 0, 0))
        draw.text((tx + 2, ty + 1), text, fill=color)

    return np.array(base)


def _availability_markdown() -> str:
    lines = ["### Backend Status"]
    specs = {s.key: s for s in list_parser_specs()}
    for key in TRANSFORMER_PARSERS:
        if key not in specs:
            continue
        ok, reason = parser_availability(key)
        state = "OK" if ok else "N/A"
        lines.append(f"- `{key}`: **{state}** ({reason})")
    return "\n".join(lines)


def run_inference(
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
        return None, "Please upload an image or paste one from your clipboard.", "[]"

    parser_key = _extract_key(parser_choice)
    available, reason = parser_availability(parser_key)
    if not available:
        return image, f"Backend `{parser_key}` is not available: {reason}", "[]"

    try:
        backend = _build_backend(
            parser_key=parser_key,
            hf_model_id=(hf_model_id or "").strip(),
            prompt=(prompt or "").strip(),
            max_new_tokens=int(max_new_tokens),
            hf_device=(hf_device or "auto").strip(),
            hf_dtype=(hf_dtype or "auto").strip(),
            mineru_command=(mineru_command or "mineru").strip(),
            mineru_timeout=int(mineru_timeout),
        )
        doc = backend.parse(image, output_dir=None, image_name="pasted_image.png")
        result = doc.to_dict()
        detections = result.get("detections", [])
        overlay = _render_overlay(image, detections)
        status = f"{len(detections)} regions detected with `{parser_key}`."
        pretty_json = json.dumps(result, ensure_ascii=False, indent=2)
        return overlay, status, pretty_json
    except Exception as exc:
        return image, f"Error: {type(exc).__name__}: {exc}", "[]"


def build_demo() -> gr.Blocks:
    parser_choices = _format_parser_choices()
    with gr.Blocks(title="PechaBridge Transformer Layout UI") as demo:
        gr.Markdown("# PechaBridge Transformer Layout UI")
        gr.Markdown(
            "Select a transformer-based layout/OCR model, customize the prompt, and paste or upload an image. "
            "Detected regions are rendered as bounding boxes."
        )
        availability_md = gr.Markdown(_availability_markdown())

        with gr.Row():
            with gr.Column(scale=1):
                parser_choice = gr.Dropdown(
                    choices=parser_choices,
                    value=parser_choices[0] if parser_choices else None,
                    label="Parser Backend",
                )
                prompt = gr.Textbox(
                    label="Prompt",
                    value=DEFAULT_PROMPT,
                    lines=5,
                )
                hf_model_id = gr.Textbox(
                    label="HF Model ID Override (optional)",
                    value="",
                    placeholder="e.g. Qwen/Qwen2.5-VL-7B-Instruct",
                )
                with gr.Row():
                    max_new_tokens = gr.Slider(64, 4096, value=1024, step=32, label="Max New Tokens")
                    hf_device = gr.Dropdown(
                        choices=["auto", "cpu", "cuda"],
                        value="auto",
                        label="HF Device",
                    )
                hf_dtype = gr.Dropdown(
                    choices=["auto", "float16", "bfloat16", "float32"],
                    value="auto",
                    label="HF DType",
                )
                with gr.Accordion("MinerU Options", open=False):
                    mineru_command = gr.Textbox(label="MinerU Command", value="mineru")
                    mineru_timeout = gr.Number(label="MinerU Timeout (s)", value=300, precision=0)

                run_btn = gr.Button("Detect Layout", variant="primary")

            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="numpy",
                    label="Image (Upload or Clipboard Paste)",
                    sources=["upload", "clipboard"],
                )
                image_output = gr.Image(type="numpy", label="Detected Regions (Overlay)")
                status_out = gr.Textbox(label="Status", interactive=False)
                json_out = gr.Code(label="JSON Output", language="json")

        run_btn.click(
            fn=run_inference,
            inputs=[
                image_input,
                parser_choice,
                prompt,
                hf_model_id,
                max_new_tokens,
                hf_device,
                hf_dtype,
                mineru_command,
                mineru_timeout,
            ],
            outputs=[image_output, status_out, json_out],
        )

        refresh_btn = gr.Button("Refresh Backend Status")
        refresh_btn.click(fn=_availability_markdown, inputs=[], outputs=[availability_md])

    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch()
