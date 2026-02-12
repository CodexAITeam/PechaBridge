"""
Transformer VLM parser adapters for layout-aware OCR extraction.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .base import DocumentParser, ParsedDetection, ParsedDocument


class TransformersVLMParser(DocumentParser):
    """Generic parser using Hugging Face vision-language models."""

    parser_name = "transformer_vlm"

    def __init__(
        self,
        model_id: str,
        prompt: Optional[str] = None,
        max_new_tokens: int = 1024,
        hf_device: str = "auto",
        hf_dtype: str = "auto",
        layout_only: bool = False,
    ):
        if not model_id:
            raise ValueError("model_id is required for transformer parser backends")
        self.model_id = model_id
        self.layout_only = layout_only
        if prompt:
            self.prompt = prompt
        elif self.layout_only:
            self.prompt = (
                "Detect only page layout regions (no OCR). "
                "Return strict JSON with key 'detections' containing a list of objects with: "
                "label, confidence, and bbox=[x0,y0,x1,y1]. "
                "Use labels from: tibetan_number_word, tibetan_text, chinese_number_word."
            )
        else:
            self.prompt = (
                "Extract page layout blocks and OCR text. "
                "Return strict JSON with key 'detections' containing a list of objects with: "
                "text, label, confidence, and bbox=[x0,y0,x1,y1]."
            )
        self.max_new_tokens = max_new_tokens
        self.hf_device = hf_device
        self.hf_dtype = hf_dtype

        self._processor = None
        self._model = None
        self._pipeline = None

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            return True, "transformers + torch available"
        except Exception as exc:
            return False, f"missing dependency: {exc}"

    def parse(
        self,
        image: Union[str, np.ndarray],
        output_dir: Optional[str] = None,
        image_name: Optional[str] = None
    ) -> ParsedDocument:
        pil_img, resolved_name = self._load_image(image, image_name=image_name)

        raw_text = self._generate_raw_response(pil_img)
        normalized = self._parse_json_response(raw_text)
        detections = self._normalize_detections(normalized.get("detections", []))

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            raw_path = Path(output_dir) / f"{Path(resolved_name).stem}_{self.parser_name}_raw.txt"
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(raw_text)

        return ParsedDocument(
            image_name=resolved_name,
            parser=self.parser_name,
            detections=detections,
            metadata={
                "model_id": self.model_id,
                "backend": "transformers",
            },
        )

    def _load_image(
        self,
        image: Union[str, np.ndarray],
        image_name: Optional[str] = None
    ) -> Tuple[Image.Image, str]:
        if isinstance(image, str):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Input image not found: {image_path}")
            return Image.open(image_path).convert("RGB"), image_path.name

        resolved_name = image_name or "image.png"
        if image.ndim == 2:
            pil_img = Image.fromarray(image).convert("RGB")
        else:
            pil_img = Image.fromarray(image[:, :, :3]).convert("RGB")
        return pil_img, resolved_name

    def _generate_raw_response(self, image: Image.Image) -> str:
        errors: List[str] = []

        try:
            return self._generate_with_auto_model(image)
        except Exception as exc:
            errors.append(f"AutoModel path failed: {type(exc).__name__}: {exc}")

        try:
            return self._generate_with_pipeline(image)
        except Exception as exc:
            errors.append(f"pipeline path failed: {type(exc).__name__}: {exc}")

        raise RuntimeError(" | ".join(errors))

    def _generate_with_auto_model(self, image: Image.Image) -> str:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

        if self._model is None:
            dtype = self._resolve_dtype(torch)
            model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
            if dtype != "auto":
                model_kwargs["torch_dtype"] = dtype
            if self.hf_device == "auto":
                model_kwargs["device_map"] = "auto"
            self._model = AutoModelForImageTextToText.from_pretrained(self.model_id, **model_kwargs)
            self._model.eval()

        inputs = self._processor(images=image, text=self.prompt, return_tensors="pt")
        model_device = getattr(self._model, "device", None)
        if model_device is not None:
            for key, value in list(inputs.items()):
                if hasattr(value, "to"):
                    inputs[key] = value.to(model_device)

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        text = self._processor.batch_decode(output_ids, skip_special_tokens=True)
        return text[0] if text else ""

    def _generate_with_pipeline(self, image: Image.Image) -> str:
        from transformers import pipeline

        if self._pipeline is None:
            pipe_kwargs = {"model": self.model_id, "trust_remote_code": True}
            if self.hf_device == "auto":
                pipe_kwargs["device_map"] = "auto"
            self._pipeline = pipeline("image-text-to-text", **pipe_kwargs)

        result = self._pipeline(image, prompt=self.prompt, max_new_tokens=self.max_new_tokens)
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict):
                return str(
                    first.get("generated_text")
                    or first.get("text")
                    or first.get("output_text")
                    or first
                )
        return str(result)

    def _parse_json_response(self, raw_text: str) -> Dict[str, Any]:
        clean = raw_text.strip()
        if not clean:
            return {"detections": []}

        clean = self._strip_markdown_fence(clean)
        for candidate in (clean, self._extract_json_candidate(clean)):
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list):
                    return {"detections": parsed}
                if isinstance(parsed, dict):
                    if "detections" in parsed and isinstance(parsed["detections"], list):
                        return parsed
                    if "blocks" in parsed and isinstance(parsed["blocks"], list):
                        return {"detections": parsed["blocks"]}
                    return {"detections": [parsed]}
            except Exception:
                continue

        # Final fallback:
        # - OCR-capable parsers keep raw text as a synthetic detection.
        # - layout-only parsers return no detections on unparseable output.
        if self.layout_only:
            return {"detections": []}
        return {"detections": [{"text": clean, "label": "raw_text", "bbox": [0, 0, 1, 1], "confidence": 0.1}]}

    @staticmethod
    def _strip_markdown_fence(text: str) -> str:
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        return text.strip()

    @staticmethod
    def _extract_json_candidate(text: str) -> str:
        m_obj = re.search(r"\{.*\}", text, re.DOTALL)
        m_arr = re.search(r"\[.*\]", text, re.DOTALL)
        candidates = [m.group(0) for m in (m_obj, m_arr) if m]
        if not candidates:
            return ""
        return min(candidates, key=len)

    def _normalize_detections(self, raw_detections: List[Dict[str, Any]]) -> List[ParsedDetection]:
        out: List[ParsedDetection] = []
        for idx, det in enumerate(raw_detections):
            if not isinstance(det, dict):
                continue
            label = str(det.get("label") or det.get("type") or det.get("class_name") or "text")
            text = str(det.get("text") or det.get("content") or det.get("ocr") or "").strip()
            if (not text) and (not self.layout_only):
                continue
            if self.layout_only:
                text = ""
            conf = det.get("confidence", det.get("score", det.get("prob", 1.0)))
            try:
                conf_f = float(conf)
            except Exception:
                conf_f = 1.0
            box = self._extract_box(det)
            class_id = self._map_class_id(label)
            out.append(
                ParsedDetection(
                    id=idx,
                    box=box,
                    confidence=conf_f,
                    class_id=class_id,
                    text=text,
                    label=label,
                )
            )
        return out

    @staticmethod
    def _extract_box(det: Dict[str, Any]) -> Dict[str, float]:
        if all(k in det for k in ("x", "y", "width", "height")):
            return {
                "x": float(det["x"]),
                "y": float(det["y"]),
                "width": float(det["width"]),
                "height": float(det["height"]),
            }

        bbox = det.get("bbox") or det.get("box")
        if isinstance(bbox, list) and len(bbox) >= 4:
            x0, y0, x1, y1 = bbox[:4]
            return {
                "x": float((x0 + x1) / 2.0),
                "y": float((y0 + y1) / 2.0),
                "width": float(abs(x1 - x0)),
                "height": float(abs(y1 - y0)),
            }

        return {"x": 0.5, "y": 0.5, "width": 1.0, "height": 1.0}

    @staticmethod
    def _map_class_id(label: str) -> int:
        lowered = label.lower()
        if "tibetan_number" in lowered or "tib_no" in lowered:
            return 0
        if "tibetan" in lowered or "text" in lowered:
            return 1
        if "chinese_number" in lowered or "chi_no" in lowered:
            return 2
        return 1

    def _resolve_dtype(self, torch_module):
        if self.hf_dtype in ("auto", "", None):
            return "auto"
        if hasattr(torch_module, self.hf_dtype):
            return getattr(torch_module, self.hf_dtype)
        return "auto"


class PaddleOCRVLParser(TransformersVLMParser):
    parser_name = "paddleocr_vl"

    def __init__(self, **kwargs):
        model_id = kwargs.pop("model_id", "PaddlePaddle/PaddleOCR-VL")
        super().__init__(model_id=model_id, **kwargs)


class Qwen25VLParser(TransformersVLMParser):
    parser_name = "qwen25vl"

    def __init__(self, **kwargs):
        model_id = kwargs.pop("model_id", "Qwen/Qwen2.5-VL-7B-Instruct")
        super().__init__(model_id=model_id, **kwargs)


class GraniteDoclingParser(TransformersVLMParser):
    parser_name = "granite_docling"

    def __init__(self, **kwargs):
        model_id = kwargs.pop("model_id", "ibm-granite/granite-docling-258M")
        super().__init__(model_id=model_id, **kwargs)


class DeepSeekOCRParser(TransformersVLMParser):
    parser_name = "deepseek_ocr"

    def __init__(self, **kwargs):
        model_id = kwargs.pop("model_id", "deepseek-ai/deepseek-vl2-small")
        super().__init__(model_id=model_id, **kwargs)


class Qwen3VLLayoutParser(TransformersVLMParser):
    parser_name = "qwen3_vl"

    def __init__(self, **kwargs):
        model_id = kwargs.pop("model_id", "Qwen/Qwen3-VL-8B-Instruct")
        super().__init__(model_id=model_id, layout_only=True, **kwargs)


class Florence2LayoutParser(TransformersVLMParser):
    parser_name = "florence2"

    def __init__(self, **kwargs):
        model_id = kwargs.pop("model_id", "microsoft/Florence-2-large")
        super().__init__(model_id=model_id, layout_only=True, **kwargs)


class GroundingDINOLayoutParser(DocumentParser):
    """GroundingDINO layout-only detector backend (no OCR text extraction)."""

    parser_name = "groundingdino"

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        hf_device: str = "auto",
        hf_dtype: str = "auto",
        box_threshold: float = 0.20,
        text_threshold: float = 0.20,
        labels: Optional[List[str]] = None,
        **_: Any,
    ):
        self.model_id = model_id
        self.hf_device = hf_device
        self.hf_dtype = hf_dtype
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.labels = labels or ["tibetan_number_word", "tibetan_text", "chinese_number_word"]
        self._processor = None
        self._model = None

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            return True, "transformers + torch available"
        except Exception as exc:
            return False, f"missing dependency: {exc}"

    def parse(
        self,
        image: Union[str, np.ndarray],
        output_dir: Optional[str] = None,
        image_name: Optional[str] = None
    ) -> ParsedDocument:
        pil_img, resolved_name = self._load_image(image, image_name=image_name)
        detections = self._detect(pil_img)

        return ParsedDocument(
            image_name=resolved_name,
            parser=self.parser_name,
            detections=detections,
            metadata={
                "model_id": self.model_id,
                "backend": "groundingdino",
                "layout_only": True,
            },
        )

    def _load_image(
        self,
        image: Union[str, np.ndarray],
        image_name: Optional[str] = None
    ) -> Tuple[Image.Image, str]:
        if isinstance(image, str):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Input image not found: {image_path}")
            return Image.open(image_path).convert("RGB"), image_path.name

        resolved_name = image_name or "image.png"
        if image.ndim == 2:
            pil_img = Image.fromarray(image).convert("RGB")
        else:
            pil_img = Image.fromarray(image[:, :, :3]).convert("RGB")
        return pil_img, resolved_name

    def _detect(self, image: Image.Image) -> List[ParsedDetection]:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

        if self._model is None:
            kwargs: Dict[str, Any] = {"trust_remote_code": True}
            dtype = self._resolve_dtype(torch)
            if dtype != "auto":
                kwargs["torch_dtype"] = dtype
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id, **kwargs)
            self._model.eval()

        model_device = self._resolve_device(torch)
        if model_device is not None:
            self._model.to(model_device)

        prompt = ". ".join(self.labels)
        inputs = self._processor(images=image, text=prompt, return_tensors="pt")
        if model_device is not None:
            for key, value in list(inputs.items()):
                if hasattr(value, "to"):
                    inputs[key] = value.to(model_device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = [(image.height, image.width)]
        processed = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes,
        )
        if not processed:
            return []

        result = processed[0]
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        labels = result.get("labels", [])

        detections: List[ParsedDetection] = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = [float(v) for v in boxes[i].tolist()]
            label = str(labels[i]) if i < len(labels) else "tibetan_text"
            score = float(scores[i].item()) if i < len(scores) else 1.0
            class_id = self._map_class_id(label)
            detections.append(
                ParsedDetection(
                    id=i,
                    box={
                        "x": (x1 + x2) / 2.0,
                        "y": (y1 + y2) / 2.0,
                        "width": abs(x2 - x1),
                        "height": abs(y2 - y1),
                    },
                    confidence=score,
                    class_id=class_id,
                    text="",
                    label=label,
                )
            )
        return detections

    @staticmethod
    def _map_class_id(label: str) -> int:
        lowered = label.lower()
        if "tibetan_number" in lowered or "tib_no" in lowered or "left" in lowered:
            return 0
        if "chinese_number" in lowered or "chi_no" in lowered or "right" in lowered:
            return 2
        return 1

    def _resolve_dtype(self, torch_module):
        if self.hf_dtype in ("auto", "", None):
            return "auto"
        if hasattr(torch_module, self.hf_dtype):
            return getattr(torch_module, self.hf_dtype)
        return "auto"

    def _resolve_device(self, torch_module):
        if self.hf_device in ("", "auto", None):
            if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
                return torch_module.device("cuda")
            return torch_module.device("cpu")
        if self.hf_device == "cuda":
            if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
                return torch_module.device("cuda")
            return torch_module.device("cpu")
        return torch_module.device(self.hf_device)
