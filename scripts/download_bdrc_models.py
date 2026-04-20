#!/usr/bin/env python3
"""Download the default BDRC OCR app model assets into models/bdrc."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pechabridge.ocr.bdrc_model_download import create_parser, run


if __name__ == "__main__":
    raise SystemExit(run(create_parser().parse_args()))
