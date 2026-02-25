"""OCR backend implementations."""

from .base import OCRBackend, OCRResult
from .tesseract_backend import TesseractBackend
from .vlm_backend_stub import VLMBackendStub

__all__ = ["OCRBackend", "OCRResult", "TesseractBackend", "VLMBackendStub"]
