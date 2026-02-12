"""
Legacy parser adapter for YOLO detection + Tesseract OCR.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

from tibetan_utils.model_utils import ModelManager
from tibetan_utils.ocr_utils import process_image_with_ocr

from .base import DocumentParser, ParsedDetection, ParsedDocument


class LegacyYoloTesseractParser(DocumentParser):
    """Adapter for the existing detection+OCR pipeline."""

    parser_name = "legacy"

    def __init__(
        self,
        model_path: str,
        lang: str = "eng+deu",
        conf: float = 0.25,
        tesseract_config: str = "",
        save_crops: bool = False,
    ):
        if not model_path:
            raise ValueError("model_path is required for parser='legacy'")
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}")
        self.model = ModelManager.load_model(str(model_file))
        self.lang = lang
        self.conf = conf
        self.tesseract_config = tesseract_config
        self.save_crops = save_crops

    def parse(
        self,
        image: Union[str, np.ndarray],
        output_dir: Optional[str] = None,
        image_name: Optional[str] = None
    ) -> ParsedDocument:
        result = process_image_with_ocr(
            image,
            self.model,
            output_dir=output_dir,
            lang=self.lang,
            conf=self.conf,
            tesseract_config=self.tesseract_config,
            save_crops=self.save_crops
        )
        detections = [
            ParsedDetection(
                id=det["id"],
                box=det["box"],
                confidence=det["confidence"],
                class_id=det["class"],
                text=det["text"],
            )
            for det in result.get("detections", [])
        ]
        resolved_name = image_name
        if not resolved_name:
            if isinstance(image, str):
                resolved_name = os.path.basename(image)
            else:
                resolved_name = result.get("image_name", "image.jpg")
        return ParsedDocument(
            image_name=resolved_name,
            parser=self.parser_name,
            detections=detections,
            metadata={"backend": "ultralytics+yolo+tesseract"},
        )
