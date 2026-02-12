"""
Registry for document parser backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Dict, List


@dataclass(frozen=True)
class ParserSpec:
    key: str
    display_name: str
    family: str
    stage: str
    supports_layout: bool
    supports_ocr: bool
    license: str
    notes: str


PARSER_SPECS: Dict[str, ParserSpec] = {
    "legacy": ParserSpec(
        key="legacy",
        display_name="YOLO + Tesseract (legacy)",
        family="Ultralytics + Tesseract",
        stage="stable",
        supports_layout=True,
        supports_ocr=True,
        license="AGPL-3.0 + Apache-2.0",
        notes="Current default pipeline in this repository."
    ),
    "mineru25": ParserSpec(
        key="mineru25",
        display_name="MinerU2.5",
        family="OpenDataLab MinerU",
        stage="preview",
        supports_layout=True,
        supports_ocr=True,
        license="AGPL-3.0",
        notes="Modern layout+OCR parser. Requires additional runtime dependencies."
    ),
    "paddleocr_vl": ParserSpec(
        key="paddleocr_vl",
        display_name="PaddleOCR-VL",
        family="PaddleOCR-VL",
        stage="preview",
        supports_layout=True,
        supports_ocr=True,
        license="Apache-2.0",
        notes="Transformer VLM backend via Hugging Face."
    ),
    "qwen25vl": ParserSpec(
        key="qwen25vl",
        display_name="Qwen2.5-VL",
        family="Qwen-VL",
        stage="preview",
        supports_layout=True,
        supports_ocr=True,
        license="Apache-2.0",
        notes="General VLM backend for layout-aware OCR extraction."
    ),
    "granite_docling": ParserSpec(
        key="granite_docling",
        display_name="Granite-Docling-258M",
        family="IBM Granite",
        stage="preview",
        supports_layout=True,
        supports_ocr=True,
        license="Apache-2.0",
        notes="Lightweight document parsing backend."
    ),
    "deepseek_ocr": ParserSpec(
        key="deepseek_ocr",
        display_name="DeepSeek OCR (VLM)",
        family="DeepSeek-VL",
        stage="preview",
        supports_layout=True,
        supports_ocr=True,
        license="Model-specific",
        notes="Transformer VLM backend using a DeepSeek-VL model."
    ),
    "qwen3_vl": ParserSpec(
        key="qwen3_vl",
        display_name="Qwen3-VL (layout-only)",
        family="Qwen-VL",
        stage="preview",
        supports_layout=True,
        supports_ocr=False,
        license="Apache-2.0",
        notes="Layout-only bounding box extraction (no OCR text expected)."
    ),
    "groundingdino": ParserSpec(
        key="groundingdino",
        display_name="GroundingDINO (layout-only)",
        family="GroundingDINO",
        stage="preview",
        supports_layout=True,
        supports_ocr=False,
        license="Apache-2.0",
        notes="Zero-shot grounding for region detection (no OCR text)."
    ),
    "florence2": ParserSpec(
        key="florence2",
        display_name="Florence-2 (layout-only)",
        family="Microsoft Florence",
        stage="preview",
        supports_layout=True,
        supports_ocr=False,
        license="MIT",
        notes="Layout-only bounding box extraction via vision-language prompting."
    ),
}


PARSER_CLASS_PATHS = {
    "legacy": ("tibetan_utils.parsers.legacy_yolo_tesseract", "LegacyYoloTesseractParser"),
    "mineru25": ("tibetan_utils.parsers.mineru25", "MinerU25Parser"),
    "paddleocr_vl": ("tibetan_utils.parsers.transformer_vlm", "PaddleOCRVLParser"),
    "qwen25vl": ("tibetan_utils.parsers.transformer_vlm", "Qwen25VLParser"),
    "granite_docling": ("tibetan_utils.parsers.transformer_vlm", "GraniteDoclingParser"),
    "deepseek_ocr": ("tibetan_utils.parsers.transformer_vlm", "DeepSeekOCRParser"),
    "qwen3_vl": ("tibetan_utils.parsers.transformer_vlm", "Qwen3VLLayoutParser"),
    "groundingdino": ("tibetan_utils.parsers.transformer_vlm", "GroundingDINOLayoutParser"),
    "florence2": ("tibetan_utils.parsers.transformer_vlm", "Florence2LayoutParser"),
}


def list_parser_specs() -> List[ParserSpec]:
    """Return parser specs sorted by key."""
    return [PARSER_SPECS[k] for k in sorted(PARSER_SPECS.keys())]


def create_parser(parser_key: str, **kwargs):
    """Factory for parser backends."""
    if parser_key not in PARSER_CLASS_PATHS:
        supported = ", ".join(sorted(PARSER_CLASS_PATHS.keys()))
        raise ValueError(f"Unknown parser '{parser_key}'. Supported: {supported}")
    parser_cls = _load_parser_class(parser_key)
    return parser_cls(**kwargs)


def parser_availability(parser_key: str) -> tuple[bool, str]:
    """Check if parser backend dependencies are available."""
    if parser_key not in PARSER_CLASS_PATHS:
        return False, "unknown parser key"
    try:
        parser_cls = _load_parser_class(parser_key)
    except Exception as exc:
        return False, f"import error: {exc}"
    return parser_cls.is_available()


def _load_parser_class(parser_key: str):
    mod_name, cls_name = PARSER_CLASS_PATHS[parser_key]
    module = import_module(mod_name)
    return getattr(module, cls_name)
