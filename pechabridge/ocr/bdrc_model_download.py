"""Download helpers for the default BDRC OCR app model assets.

The original BDRC desktop app distributes its OCR model bundle as a release ZIP
and keeps the line/layout ONNX files in the app repository as Git LFS assets.
This module mirrors that setup so PechaBridge can fetch the same defaults.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence
from urllib.request import Request, urlopen

from .bdrc_inference import find_bdrc_ocr_model_dirs

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BDRC_MODELS_ROOT = (REPO_ROOT / "models" / "bdrc").resolve()

BDRC_APP_RELEASE_TAG = "v0.3.0"
BDRC_OCR_BUNDLE_TAG = "v0.1"
BDRC_APP_REPO = "buda-base/tibetan-ocr-app"

DEFAULT_BDRC_LINE_URL = (
    f"https://media.githubusercontent.com/media/{BDRC_APP_REPO}/{BDRC_APP_RELEASE_TAG}/Models/Lines/PhotiLines.onnx"
)
DEFAULT_BDRC_LAYOUT_URL = (
    f"https://media.githubusercontent.com/media/{BDRC_APP_REPO}/{BDRC_APP_RELEASE_TAG}/Models/Layout/photi.onnx"
)
DEFAULT_BDRC_OCR_ZIP_URL = (
    f"https://github.com/{BDRC_APP_REPO}/releases/download/{BDRC_OCR_BUNDLE_TAG}/bdrc_ocr_models_1.0.zip"
)

DEFAULT_BDRC_LINE_SHA256 = "4862ced821dbac199608dbc0463fd984b7cb2ab202ecfdf3ca95e29d648ff171"
DEFAULT_BDRC_LAYOUT_SHA256 = "645da1c27058c11adfbe254ef9b364f64c95748d2c94d4f9265294e4e83b033e"
DEFAULT_BDRC_LINE_SIZE = 89724544
DEFAULT_BDRC_LAYOUT_SIZE = 90467862

DEFAULT_BDRC_LINE_CONFIG = {
    "onnx-model": "PhotiLines.onnx",
    "patch_size": 512,
}
DEFAULT_BDRC_LAYOUT_CONFIG = {
    "checkpoint": "photi.pth",
    "onnx-model": "photi.onnx",
    "patch_size": 512,
    "classes": ["background", "image", "line", "margin", "caption"],
}


@dataclass(frozen=True)
class EnsureBDRCLineAssetsResult:
    root: Path
    line_dir: Path
    layout_dir: Path
    downloaded_items: tuple[str, ...]


@dataclass(frozen=True)
class EnsureBDRCOCRModelsResult:
    root: Path
    model_dirs: tuple[Path, ...]
    downloaded: bool


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _download_url_to_path(
    url: str,
    dest: Path,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> Path:
    dest = dest.expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    req = Request(str(url), headers={"User-Agent": "PechaBridge/1.0"})
    with urlopen(req) as resp, tmp.open("wb") as fh:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)

    if expected_size is not None and tmp.stat().st_size != int(expected_size):
        raise ValueError(
            f"download size mismatch for {dest.name}: expected {expected_size}, got {tmp.stat().st_size}"
        )
    if expected_sha256:
        actual_sha256 = _sha256_file(tmp)
        if actual_sha256.lower() != str(expected_sha256).strip().lower():
            raise ValueError(
                f"download checksum mismatch for {dest.name}: expected {expected_sha256}, got {actual_sha256}"
            )

    tmp.replace(dest)
    return dest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ensure_default_bdrc_line_assets(
    base_dir: str | Path = DEFAULT_BDRC_MODELS_ROOT,
    *,
    force: bool = False,
    include_layout: bool = True,
) -> EnsureBDRCLineAssetsResult:
    base = Path(base_dir).expanduser().resolve()
    line_dir = base / "Lines"
    layout_dir = base / "Layout"
    downloaded: List[str] = []

    line_model_path = line_dir / "PhotiLines.onnx"
    line_cfg_path = line_dir / "config.json"
    line_ok = line_model_path.exists() and line_cfg_path.exists()
    if force or not line_ok:
        _download_url_to_path(
            DEFAULT_BDRC_LINE_URL,
            line_model_path,
            expected_sha256=DEFAULT_BDRC_LINE_SHA256,
            expected_size=DEFAULT_BDRC_LINE_SIZE,
        )
        _write_json(line_cfg_path, DEFAULT_BDRC_LINE_CONFIG)
        downloaded.append("line")

    layout_model_path = layout_dir / "photi.onnx"
    layout_cfg_path = layout_dir / "config.json"
    layout_ok = layout_model_path.exists() and layout_cfg_path.exists()
    if include_layout and (force or not layout_ok):
        _download_url_to_path(
            DEFAULT_BDRC_LAYOUT_URL,
            layout_model_path,
            expected_sha256=DEFAULT_BDRC_LAYOUT_SHA256,
            expected_size=DEFAULT_BDRC_LAYOUT_SIZE,
        )
        _write_json(layout_cfg_path, DEFAULT_BDRC_LAYOUT_CONFIG)
        downloaded.append("layout")

    return EnsureBDRCLineAssetsResult(
        root=base,
        line_dir=line_dir,
        layout_dir=layout_dir,
        downloaded_items=tuple(downloaded),
    )


def _pick_existing_ocr_model_root(root: Path) -> tuple[Path, ...]:
    found = find_bdrc_ocr_model_dirs(root)
    return tuple(sorted(found, key=lambda p: p.name.lower()))


def _detect_extracted_ocr_root(extract_root: Path) -> Path:
    preferred = extract_root / "OCRModels"
    if preferred.exists() and preferred.is_dir() and _pick_existing_ocr_model_root(preferred):
        return preferred

    children = [p for p in sorted(extract_root.iterdir()) if p.is_dir()]
    if len(children) == 1 and _pick_existing_ocr_model_root(children[0]):
        return children[0]

    if _pick_existing_ocr_model_root(extract_root):
        return extract_root

    raise FileNotFoundError(
        f"Could not locate extracted BDRC OCR models under archive root: {extract_root}"
    )


def ensure_default_bdrc_ocr_models(
    base_dir: str | Path = DEFAULT_BDRC_MODELS_ROOT,
    *,
    force: bool = False,
) -> EnsureBDRCOCRModelsResult:
    base = Path(base_dir).expanduser().resolve()
    ocr_root = base / "OCRModels"
    existing = _pick_existing_ocr_model_root(ocr_root)
    if existing and not force:
        return EnsureBDRCOCRModelsResult(root=ocr_root, model_dirs=existing, downloaded=False)

    with tempfile.TemporaryDirectory(prefix="pechabridge-bdrc-ocr-") as tmp_dir_s:
        tmp_dir = Path(tmp_dir_s)
        archive_path = tmp_dir / "bdrc_ocr_models_1.0.zip"
        extract_root = tmp_dir / "extract"
        extract_root.mkdir(parents=True, exist_ok=True)

        _download_url_to_path(DEFAULT_BDRC_OCR_ZIP_URL, archive_path)
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_root)

        source_root = _detect_extracted_ocr_root(extract_root)
        if ocr_root.exists():
            shutil.rmtree(ocr_root)
        shutil.copytree(source_root, ocr_root)

    found = _pick_existing_ocr_model_root(ocr_root)
    if not found:
        raise FileNotFoundError(f"BDRC OCR model bundle extracted but no models were found under: {ocr_root}")
    return EnsureBDRCOCRModelsResult(root=ocr_root, model_dirs=found, downloaded=True)


def choose_default_bdrc_ocr_model_dir(base_dir: str | Path = DEFAULT_BDRC_MODELS_ROOT) -> Path:
    root = Path(base_dir).expanduser().resolve()
    model_dirs = list(_pick_existing_ocr_model_root(root))
    if not model_dirs:
        raise FileNotFoundError(f"No BDRC OCR models found under: {root}")
    preferred = sorted(
        model_dirs,
        key=lambda p: (0 if p.name.strip().lower() == "woodblock" else 1, p.name.lower()),
    )
    return preferred[0]


def create_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download the default BDRC OCR app assets into models/bdrc. "
            "This mirrors the original app setup: line/layout ONNX files from the app repo "
            "and the OCR model bundle ZIP from the app release."
        ),
        add_help=add_help,
    )
    parser.add_argument(
        "--dest",
        default=str(DEFAULT_BDRC_MODELS_ROOT),
        help="Destination root directory (default: ./models/bdrc)",
    )
    parser.add_argument(
        "--assets",
        default="all",
        help="Comma-separated asset groups: all, line, layout, ocr",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite existing BDRC assets.",
    )
    return parser


def _parse_asset_selection(raw: str) -> set[str]:
    selected = {x.strip().lower() for x in str(raw or "all").split(",") if x.strip()}
    if not selected or "all" in selected:
        return {"line", "layout", "ocr"}
    invalid = selected - {"line", "layout", "ocr"}
    if invalid:
        raise ValueError(f"Unknown asset selection: {sorted(invalid)}")
    return selected


def run(args: argparse.Namespace) -> int:
    dest = Path(str(getattr(args, "dest", DEFAULT_BDRC_MODELS_ROOT))).expanduser().resolve()
    selected = _parse_asset_selection(str(getattr(args, "assets", "all") or "all"))
    force = bool(getattr(args, "force", False))

    if "line" in selected or "layout" in selected:
        res = ensure_default_bdrc_line_assets(
            dest,
            force=force,
            include_layout=("layout" in selected),
        )
        if res.downloaded_items:
            print(f"Downloaded BDRC line assets: {', '.join(res.downloaded_items)} -> {res.root}")
        else:
            print(f"BDRC line assets already present under {res.root}")

    if "ocr" in selected:
        res_ocr = ensure_default_bdrc_ocr_models(dest, force=force)
        if res_ocr.downloaded:
            print(f"Downloaded BDRC OCR models bundle -> {res_ocr.root}")
        else:
            print(f"BDRC OCR models already present under {res_ocr.root}")
        print(f"Detected OCR models: {', '.join(p.name for p in res_ocr.model_dirs)}")

    return 0


__all__ = [
    "BDRC_APP_RELEASE_TAG",
    "BDRC_OCR_BUNDLE_TAG",
    "DEFAULT_BDRC_MODELS_ROOT",
    "DEFAULT_BDRC_LINE_URL",
    "DEFAULT_BDRC_LAYOUT_URL",
    "DEFAULT_BDRC_OCR_ZIP_URL",
    "EnsureBDRCLineAssetsResult",
    "EnsureBDRCOCRModelsResult",
    "choose_default_bdrc_ocr_model_dir",
    "create_parser",
    "ensure_default_bdrc_line_assets",
    "ensure_default_bdrc_ocr_models",
    "run",
]
