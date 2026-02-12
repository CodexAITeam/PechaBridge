"""
MinerU2.5 parser adapter.

This adapter is intentionally conservative: it integrates via MinerU CLI output
and normalizes extracted JSON into the project-wide ParsedDocument schema.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
from PIL import Image

from .base import DocumentParser, ParsedDetection, ParsedDocument


class MinerU25Parser(DocumentParser):
    """Adapter for MinerU2.5 (layout + OCR) via CLI integration."""

    parser_name = "mineru25"

    def __init__(self, mineru_command: str = "mineru", timeout_sec: int = 300):
        self.mineru_command = mineru_command
        self.timeout_sec = timeout_sec

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        has_cli = shutil.which("mineru") is not None
        if has_cli:
            return True, "MinerU CLI found in PATH"
        return False, "MinerU CLI not found. Install MinerU and ensure `mineru` is in PATH."

    def parse(
        self,
        image: Union[str, np.ndarray],
        output_dir: Optional[str] = None,
        image_name: Optional[str] = None
    ) -> ParsedDocument:
        with tempfile.TemporaryDirectory(prefix="mineru25_") as work_dir:
            work_path = Path(work_dir)
            input_image = self._prepare_input_image(image, work_path, image_name=image_name)
            mineru_output = work_path / "mineru_output"
            mineru_output.mkdir(parents=True, exist_ok=True)

            self._run_mineru(input_image, mineru_output)
            json_paths = list(mineru_output.rglob("*.json"))
            if not json_paths:
                raise RuntimeError(
                    "MinerU finished but no JSON output was found. "
                    "Check MinerU CLI version and output format options."
                )

            combined_detections: List[ParsedDetection] = []
            raw_files = []
            for json_path in json_paths:
                raw_files.append(str(json_path))
                data = self._load_json(json_path)
                combined_detections.extend(self._extract_detections(data, start_id=len(combined_detections)))

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                merged_output = {
                    "image_name": input_image.name,
                    "parser": self.parser_name,
                    "detections": [d.to_dict() for d in combined_detections],
                    "metadata": {"raw_files": raw_files}
                }
                out_json = Path(output_dir) / f"{input_image.stem}_mineru25.json"
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(merged_output, f, ensure_ascii=False, indent=2)

            return ParsedDocument(
                image_name=input_image.name,
                parser=self.parser_name,
                detections=combined_detections,
                metadata={"raw_files": raw_files},
            )

    def _prepare_input_image(
        self,
        image: Union[str, np.ndarray],
        work_path: Path,
        image_name: Optional[str] = None,
    ) -> Path:
        if isinstance(image, str):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Input image not found: {image_path}")
            return image_path

        resolved_name = image_name or "image.png"
        if not resolved_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            resolved_name = f"{resolved_name}.png"
        image_path = work_path / resolved_name
        pil_img = Image.fromarray(image)
        pil_img.save(image_path)
        return image_path

    def _run_mineru(self, input_image: Path, output_dir: Path) -> None:
        # Different MinerU versions use slightly different flags; try common variants.
        cmd_variants = [
            [self.mineru_command, "--input", str(input_image), "--output", str(output_dir), "--format", "json"],
            [self.mineru_command, "-i", str(input_image), "-o", str(output_dir), "--json"],
            [self.mineru_command, str(input_image), "-o", str(output_dir)],
        ]

        last_error = None
        for cmd in cmd_variants:
            try:
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.timeout_sec,
                    check=False,
                )
            except FileNotFoundError:
                raise RuntimeError(
                    "MinerU CLI executable not found. Install MinerU and add `mineru` to PATH."
                ) from None
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                continue

            if proc.returncode == 0:
                return

            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            last_error = f"exit={proc.returncode}, stderr={stderr}, stdout={stdout}"

        raise RuntimeError(
            "MinerU CLI invocation failed for known command variants. "
            f"Last error: {last_error}"
        )

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _extract_detections(self, obj: Any, start_id: int = 0) -> List[ParsedDetection]:
        detections: List[ParsedDetection] = []
        next_id = start_id

        for node in self._walk(obj):
            if not isinstance(node, dict):
                continue

            text = node.get("text") or node.get("content") or ""
            if not isinstance(text, str) or not text.strip():
                continue

            bbox = self._extract_bbox(node)
            conf = self._extract_confidence(node)
            label = node.get("type") or node.get("label") or "text"

            detections.append(
                ParsedDetection(
                    id=next_id,
                    box=bbox,
                    confidence=conf,
                    class_id=0,
                    text=text.strip(),
                    label=str(label),
                )
            )
            next_id += 1

        return detections

    @staticmethod
    def _walk(obj: Any) -> Iterable[Any]:
        if isinstance(obj, dict):
            yield obj
            for value in obj.values():
                yield from MinerU25Parser._walk(value)
            return
        if isinstance(obj, list):
            for item in obj:
                yield from MinerU25Parser._walk(item)

    @staticmethod
    def _extract_confidence(node: Dict[str, Any]) -> float:
        for key in ("confidence", "score", "prob"):
            val = node.get(key)
            if isinstance(val, (int, float)):
                return float(val)
        return 1.0

    @staticmethod
    def _extract_bbox(node: Dict[str, Any]) -> Dict[str, float]:
        # Prefer explicit keys.
        if all(k in node for k in ("x", "y", "width", "height")):
            return {
                "x": float(node["x"]),
                "y": float(node["y"]),
                "width": float(node["width"]),
                "height": float(node["height"]),
            }

        # Common variant: bbox=[x0,y0,x1,y1].
        bbox = node.get("bbox") or node.get("box")
        if isinstance(bbox, list) and len(bbox) >= 4:
            x0, y0, x1, y1 = bbox[:4]
            return {
                "x": float((x0 + x1) / 2.0),
                "y": float((y0 + y1) / 2.0),
                "width": float(abs(x1 - x0)),
                "height": float(abs(y1 - y0)),
            }

        # Fallback when no geometry exists in raw output.
        return {"x": 0.5, "y": 0.5, "width": 1.0, "height": 1.0}
