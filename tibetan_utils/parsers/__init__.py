"""
Parser backends for OCR/layout extraction.
"""

from .base import DocumentParser, ParsedDetection, ParsedDocument
from .registry import create_parser, list_parser_specs, parser_availability

__all__ = [
    "DocumentParser",
    "ParsedDetection",
    "ParsedDocument",
    "create_parser",
    "list_parser_specs",
    "parser_availability",
]
