from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image


gr = pytest.importorskip("gradio")
ui_workbench = pytest.importorskip("ui_workbench")


def _make_segment_dataset(root: Path) -> Path:
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    Image.fromarray(np.full((32, 48, 3), 255, dtype=np.uint8)).save(root / "images" / "train" / "sample.png")
    (root / "labels" / "train" / "sample.txt").write_text(
        "0 0.1 0.1 0.9 0.1 0.9 0.3 0.1 0.3\n",
        encoding="utf-8",
    )
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "names": {0: "line"},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return root


def test_resolve_yolo_split_io_dirs_supports_images_train_layout(tmp_path: Path) -> None:
    dataset_root = _make_segment_dataset(tmp_path / "openpecha_line_segmentation")

    image_dir, label_dir, resolved_root = ui_workbench._resolve_yolo_split_io_dirs(dataset_root, "train")

    assert image_dir == (dataset_root / "images" / "train").resolve()
    assert label_dir == (dataset_root / "labels" / "train").resolve()
    assert resolved_root == dataset_root.resolve()
    assert ui_workbench._is_supported_yolo_dataset_root(dataset_root)


def test_draw_yolo_boxes_uses_yaml_class_names_for_segment_dataset(tmp_path: Path) -> None:
    dataset_root = _make_segment_dataset(tmp_path / "line_dataset")

    _, summary = ui_workbench._draw_yolo_boxes(
        dataset_root / "images" / "train" / "sample.png",
        dataset_root / "labels" / "train" / "sample.txt",
    )

    assert "label=line" in summary


def test_infer_ultralytics_task_detects_segment_dataset_from_polygon_labels(tmp_path: Path) -> None:
    dataset_root = _make_segment_dataset(tmp_path / "line_dataset_task")

    task = ui_workbench._infer_ultralytics_task(str(dataset_root), "best.pt")

    assert task == "segment"
