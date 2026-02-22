"""Base OCR backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional

from PIL import Image


@dataclass(frozen=True)
class OCRResult:
    """Normalized OCR result payload returned by backends."""

    text: str
    confidence: float
    tokens: Optional[List[str]] = None
    char_count: int = 0
    word_count: int = 0
    raw_json: Optional[str] = None
    lang_used: str = ""


class OCRBackend(ABC):
    """Backend contract used by the weak OCR labeler pipeline."""

    backend_name: str = "unknown"

    @property
    def name(self) -> str:
        return str(self.backend_name)

    @abstractmethod
    def ocr_image(self, image: Image.Image, meta: Mapping[str, Any]) -> OCRResult:
        """Run OCR on one image and return normalized results."""
        raise NotImplementedError

