"""
Tibetan OCR utility package.

This package keeps top-level imports best-effort so submodules with optional
dependencies (e.g. cv2, torch) do not break unrelated commands.
"""

from importlib import import_module

_SUBMODULES = [
    "arg_utils",
    "image_utils",
    "io_utils",
    "model_utils",
    "ocr_utils",
    "retrieval_schema",
    "sbb_utils",
    "config",
    "parsers",
]

for _name in _SUBMODULES:
    try:
        _mod = import_module(f"{__name__}.{_name}")
        _public = getattr(_mod, "__all__", None)
        if _public is None:
            _public = [n for n in dir(_mod) if not n.startswith("_")]
        globals().update({k: getattr(_mod, k) for k in _public})
    except Exception:
        # Optional dependencies may be missing; explicit submodule imports
        # (e.g. `from tibetan_utils.arg_utils import ...`) still work.
        continue
