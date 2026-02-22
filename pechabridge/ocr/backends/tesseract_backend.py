"""Tesseract OCR backend implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from PIL import Image

from .base import OCRBackend, OCRResult

try:
    import pytesseract
    from pytesseract import Output
except Exception:  # pragma: no cover
    pytesseract = None
    Output = None


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


class TesseractBackend(OCRBackend):
    """Weak OCR backend powered by pytesseract."""

    backend_name = "tesseract"

    def __init__(
        self,
        *,
        tesseract_cmd: Optional[str] = None,
        lang: str = "bod",
        oem: int = 1,
        psm: int = 6,
        extra_config: str = "",
    ) -> None:
        if pytesseract is None:  # pragma: no cover
            raise RuntimeError("pytesseract is required for TesseractBackend.")

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = str(Path(tesseract_cmd).expanduser())

        self.requested_lang = str(lang or "bod").strip() or "bod"
        self.oem = int(oem)
        self.psm = int(psm)
        self.extra_config = str(extra_config or "").strip()
        self.lang_used = self._resolve_lang(self.requested_lang)
        self.tesseract_cmd = str(getattr(pytesseract.pytesseract, "tesseract_cmd", ""))
        self._config_str = self._build_config()

    def _build_config(self) -> str:
        parts = [f"--oem {self.oem}", f"--psm {self.psm}"]
        if self.extra_config:
            parts.append(self.extra_config)
        return " ".join(parts).strip()

    def _resolve_lang(self, requested_lang: str) -> str:
        want = str(requested_lang or "").strip() or "bod"
        try:
            installed = set(str(v).strip() for v in (pytesseract.get_languages(config="") or []))
        except Exception:
            installed = set()

        if not installed:
            if want.lower() == "bod":
                return "eng"
            return want
        if want in installed:
            return want
        if "eng" in installed:
            return "eng"
        return sorted(installed)[0]

    @staticmethod
    def _parse_confidences(values: Sequence[Any]) -> List[float]:
        out: List[float] = []
        for raw in values:
            try:
                score = float(raw)
            except Exception:
                continue
            if score >= 0.0:
                out.append(float(score))
        return out

    def ocr_image(self, image: Image.Image, meta: Mapping[str, Any]) -> OCRResult:
        if pytesseract is None or Output is None:  # pragma: no cover
            raise RuntimeError("pytesseract is required for Tesseract OCR.")

        text = str(
            pytesseract.image_to_string(
                image,
                lang=self.lang_used,
                config=self._config_str,
            )
            or ""
        ).strip()
        data = pytesseract.image_to_data(
            image,
            lang=self.lang_used,
            config=self._config_str,
            output_type=Output.DICT,
        )

        raw_tokens = list((data or {}).get("text") or [])
        tokens = [str(tok).strip() for tok in raw_tokens if str(tok).strip()]

        valid_conf = self._parse_confidences((data or {}).get("conf") or [])
        if valid_conf:
            confidence = _clamp01(float(np.mean(valid_conf)) / 100.0)
            fallback_conf = None
        else:
            confidence = float("nan")
            nonspace_chars = sum(1 for ch in text if not ch.isspace())
            nonspace_frac = float(nonspace_chars) / float(max(1, len(text)))
            fallback_conf = _clamp01((float(nonspace_chars) / 50.0) * nonspace_frac)

        char_count = int(len(text))
        word_count = int(len(tokens)) if tokens else int(len([w for w in text.split() if w]))

        raw_payload: Dict[str, Any] = {
            "requested_lang": self.requested_lang,
            "lang_used": self.lang_used,
            "config": self._config_str,
            "valid_conf_count": int(len(valid_conf)),
            "mean_word_conf_0_100": (float(np.mean(valid_conf)) if valid_conf else None),
            "fallback_confidence": fallback_conf,
            "word_count_from_data": int(len(tokens)),
            "patch_id": int(meta.get("patch_id", -1)) if meta else -1,
        }
        raw_json = json.dumps(raw_payload, ensure_ascii=False)

        return OCRResult(
            text=text,
            confidence=float(confidence),
            tokens=tokens if tokens else None,
            char_count=char_count,
            word_count=word_count,
            raw_json=raw_json,
            lang_used=self.lang_used,
        )


__all__ = ["TesseractBackend"]
