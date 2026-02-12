"""
Base interfaces and shared helpers for document parsers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class ParsedDetection:
    """Standardized detection block for OCR/layout output."""
    id: int
    box: Dict[str, float]
    confidence: float
    class_id: int
    text: str
    label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["class"] = data.pop("class_id")
        return data


@dataclass
class ParsedDocument:
    """Standardized document parse output."""
    image_name: str
    parser: str
    detections: List[ParsedDetection]
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_name": self.image_name,
            "parser": self.parser,
            "detections": [d.to_dict() for d in self.detections],
            "metadata": self.metadata or {}
        }


class DocumentParser(ABC):
    """Abstract parser interface."""

    parser_name: str = "unknown"

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        """
        Return availability and reason.
        """
        return True, "available"

    @abstractmethod
    def parse(
        self,
        image: Union[str, np.ndarray],
        output_dir: Optional[str] = None,
        image_name: Optional[str] = None
    ) -> ParsedDocument:
        """
        Parse a single image path or array into standardized output.
        """
        raise NotImplementedError
