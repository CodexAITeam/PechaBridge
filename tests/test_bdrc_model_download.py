from __future__ import annotations

import json
import zipfile
from pathlib import Path

from pechabridge.ocr.bdrc_model_download import (
    choose_default_bdrc_ocr_model_dir,
    ensure_default_bdrc_line_assets,
    ensure_default_bdrc_ocr_models,
)


def test_ensure_default_bdrc_line_assets_from_file_urls(tmp_path, monkeypatch):
    line_src = tmp_path / "PhotiLines.onnx"
    line_src.write_bytes(b"fake-line-model")
    layout_src = tmp_path / "photi.onnx"
    layout_src.write_bytes(b"fake-layout-model")

    import pechabridge.ocr.bdrc_model_download as mod

    monkeypatch.setattr(mod, "DEFAULT_BDRC_LINE_URL", line_src.resolve().as_uri())
    monkeypatch.setattr(mod, "DEFAULT_BDRC_LAYOUT_URL", layout_src.resolve().as_uri())
    monkeypatch.setattr(mod, "DEFAULT_BDRC_LINE_SHA256", mod._sha256_file(line_src))
    monkeypatch.setattr(mod, "DEFAULT_BDRC_LAYOUT_SHA256", mod._sha256_file(layout_src))
    monkeypatch.setattr(mod, "DEFAULT_BDRC_LINE_SIZE", line_src.stat().st_size)
    monkeypatch.setattr(mod, "DEFAULT_BDRC_LAYOUT_SIZE", layout_src.stat().st_size)

    dest = tmp_path / "models" / "bdrc"
    result = ensure_default_bdrc_line_assets(dest, include_layout=True)
    assert result.line_dir.exists()
    assert result.layout_dir.exists()
    assert (result.line_dir / "PhotiLines.onnx").read_bytes() == b"fake-line-model"
    assert (result.layout_dir / "photi.onnx").read_bytes() == b"fake-layout-model"
    assert json.loads((result.line_dir / "config.json").read_text(encoding="utf-8"))["onnx-model"] == "PhotiLines.onnx"
    assert json.loads((result.layout_dir / "config.json").read_text(encoding="utf-8"))["onnx-model"] == "photi.onnx"
    assert set(result.downloaded_items) == {"line", "layout"}

    second = ensure_default_bdrc_line_assets(dest, include_layout=True)
    assert second.downloaded_items == ()


def test_ensure_default_bdrc_ocr_models_from_local_zip(tmp_path, monkeypatch):
    archive_path = tmp_path / "bdrc_ocr_models_1.0.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(
            "OCRModels/Woodblock/model_config.json",
            json.dumps({"onnx-model": "woodblock.onnx", "input_width": 2000, "input_height": 80}),
        )
        zf.writestr("OCRModels/Woodblock/woodblock.onnx", b"fake-woodblock")
        zf.writestr(
            "OCRModels/Print/model_config.json",
            json.dumps({"onnx-model": "print.onnx", "input_width": 2000, "input_height": 80}),
        )
        zf.writestr("OCRModels/Print/print.onnx", b"fake-print")

    import pechabridge.ocr.bdrc_model_download as mod

    monkeypatch.setattr(mod, "DEFAULT_BDRC_OCR_ZIP_URL", archive_path.resolve().as_uri())

    dest = tmp_path / "models" / "bdrc"
    result = ensure_default_bdrc_ocr_models(dest)
    names = [p.name for p in result.model_dirs]
    assert result.downloaded is True
    assert result.root.exists()
    assert "Woodblock" in names
    assert "Print" in names

    chosen = choose_default_bdrc_ocr_model_dir(result.root)
    assert chosen.name == "Woodblock"

    second = ensure_default_bdrc_ocr_models(dest)
    assert second.downloaded is False
    assert {p.name for p in second.model_dirs} == {"Woodblock", "Print"}
