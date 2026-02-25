"""Stub OCR-VLM backend implementation for future replacement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from PIL import Image

from .base import OCRBackend, OCRResult


class VLMBackendStub(OCRBackend):
    """Loads VLM config but intentionally leaves OCR calls unimplemented."""

    backend_name = "vlm"

    def __init__(self, *, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = dict(config or {})
        self.runtime_config: Dict[str, Any] = self._load_json_config(self.config)
        self.lang_used = str(self.config.get("lang", "") or "").strip()

    @staticmethod
    def _load_json_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
        path_txt = str(cfg.get("json_config_path", "") or cfg.get("json_config", "") or "").strip()
        if path_txt:
            path = Path(path_txt).expanduser().resolve()
            if not path.exists() or not path.is_file():
                raise FileNotFoundError(f"VLM json_config_path does not exist: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"VLM json config must be a dict: {path}")
            return dict(payload)

        inline = cfg.get("json")
        if isinstance(inline, dict):
            return dict(inline)
        return {}

    def ocr_image(self, image: Image.Image, meta: Mapping[str, Any]) -> OCRResult:
        raise NotImplementedError(
            "VLMBackendStub is a placeholder. Implement API calls here to replace Tesseract later."
        )


__all__ = ["VLMBackendStub"]
