from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import pechabridge.ocr.bdrc_inference as bdrc_inference
from pechabridge.ocr.bdrc_inference import (
    find_bdrc_line_model_dirs,
    find_bdrc_ocr_model_dirs,
    is_bdrc_line_model_dir,
    is_bdrc_ocr_model_dir,
    resolve_bdrc_line_model_config,
    resolve_bdrc_ocr_model_config,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_resolve_bdrc_line_model_config_for_line_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "Models" / "Lines"
    model_dir.mkdir(parents=True)
    (model_dir / "PhotiLines.onnx").write_bytes(b"dummy")
    _write_json(
        model_dir / "config.json",
        {
            "onnx-model": "PhotiLines.onnx",
            "patch_size": 512,
        },
    )

    assert is_bdrc_line_model_dir(model_dir)
    cfg = resolve_bdrc_line_model_config(model_dir)
    assert cfg.model_kind == "line"
    assert cfg.patch_size == 512
    assert Path(cfg.model_file).name == "PhotiLines.onnx"


def test_resolve_bdrc_line_model_config_for_layout_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "Models" / "Layout"
    model_dir.mkdir(parents=True)
    (model_dir / "photi.onnx").write_bytes(b"dummy")
    _write_json(
        model_dir / "config.json",
        {
            "onnx-model": "photi.onnx",
            "patch_size": 512,
            "classes": ["background", "image", "line", "margin", "caption"],
        },
    )

    cfg = resolve_bdrc_line_model_config(model_dir / "config.json")
    assert cfg.model_kind == "layout"
    assert "line" in cfg.classes


def test_find_bdrc_line_model_dirs_discovers_both_types(tmp_path: Path) -> None:
    lines_dir = tmp_path / "a" / "Lines"
    lines_dir.mkdir(parents=True)
    (lines_dir / "line.onnx").write_bytes(b"dummy")
    _write_json(lines_dir / "config.json", {"onnx-model": "line.onnx", "patch_size": 512})

    layout_dir = tmp_path / "b" / "Layout"
    layout_dir.mkdir(parents=True)
    (layout_dir / "layout.onnx").write_bytes(b"dummy")
    _write_json(
        layout_dir / "config.json",
        {"onnx-model": "layout.onnx", "patch_size": 512, "classes": ["background", "line"]},
    )

    found = find_bdrc_line_model_dirs(tmp_path)
    found_set = {p.name for p in found}
    assert found_set == {"Lines", "Layout"}


def test_resolve_bdrc_ocr_model_config(tmp_path: Path) -> None:
    model_dir = tmp_path / "OCRModels" / "Woodblock"
    model_dir.mkdir(parents=True)
    (model_dir / "ocr.onnx").write_bytes(b"dummy")
    _write_json(
        model_dir / "model_config.json",
        {
            "onnx-model": "ocr.onnx",
            "architecture": "CRNN",
            "version": "1.0",
            "input_width": 2048,
            "input_height": 80,
            "input_layer": "input",
            "output_layer": "output",
            "encoder": "wylie",
            "squeeze_channel_dim": "no",
            "swap_hw": "no",
            "charset": ["a", "b", "c"],
            "add_blank": "yes",
        },
    )

    assert is_bdrc_ocr_model_dir(model_dir)
    cfg = resolve_bdrc_ocr_model_config(model_dir)
    assert cfg.input_width == 2048
    assert cfg.input_height == 80
    assert cfg.encoder == "wylie"
    assert cfg.add_blank is True


def test_find_bdrc_ocr_model_dirs(tmp_path: Path) -> None:
    model_dir = tmp_path / "OCRModels" / "Pecha"
    model_dir.mkdir(parents=True)
    (model_dir / "ocr.onnx").write_bytes(b"dummy")
    _write_json(
        model_dir / "model_config.json",
        {
            "onnx-model": "ocr.onnx",
            "architecture": "CRNN",
            "input_width": 2000,
            "input_height": 80,
            "input_layer": "input",
            "output_layer": "output",
            "encoder": "wylie",
            "squeeze_channel_dim": "no",
            "swap_hw": "no",
            "charset": ["a"],
            "add_blank": "yes",
        },
    )

    found = find_bdrc_ocr_model_dirs(tmp_path)
    assert [p.name for p in found] == ["Pecha"]


def test_run_tps_identity_mapping_preserves_image_geometry() -> None:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[2:10, 4:12, 0] = 180
    image[5:8, 6:10, 1] = 255
    ctrl = np.asarray(
        [
            [1.0, 1.0],
            [1.0, 14.0],
            [10.0, 1.0],
            [10.0, 14.0],
            [6.0, 8.0],
        ],
        dtype=np.float64,
    )

    warped, mapping = bdrc_inference._run_tps(image, ctrl, ctrl)

    assert warped.shape == image.shape
    assert warped.dtype == np.uint8
    assert np.max(np.abs(warped.astype(np.int16) - image.astype(np.int16))) <= 1

    pts = np.asarray([[6.0, 8.0], [3.0, 5.0]], dtype=np.float64)
    mapped = mapping.transform(pts)
    assert np.allclose(mapped, pts, atol=1e-3)
