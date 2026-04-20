from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from scripts.train_line_segmentation import (
    _build_preprocessed_training_dataset,
    create_parser,
)


def _make_raw_segment_dataset(root: Path) -> Path:
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    arr = np.zeros((24, 32, 3), dtype=np.uint8)
    arr[:, :, 0] = 220
    arr[:, :, 1] = 40
    arr[:, :, 2] = 10
    Image.fromarray(arr, mode="RGB").save(root / "images" / "train" / "sample.png")
    (root / "labels" / "train" / "sample.txt").write_text(
        "0 0.1 0.1 0.9 0.1 0.9 0.4 0.1 0.4\n",
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
    return root / "data.yaml"


def test_train_parser_defaults_to_gray_preprocess() -> None:
    parser = create_parser()
    args = parser.parse_args(["--dataset", "/tmp/raw-line-seg"])
    assert args.image_preprocess_pipeline == "gray"


def test_build_preprocessed_training_dataset_writes_temp_segment_dataset(tmp_path: Path) -> None:
    source_yaml = _make_raw_segment_dataset(tmp_path / "src")

    out_yaml, summary = _build_preprocessed_training_dataset(
        source_yaml,
        output_dir=tmp_path / "prepared",
        pipeline="gray",
        preprocess_workers=2,
    )

    prepared_image = tmp_path / "prepared" / "images" / "train" / "sample.png"
    prepared_label = tmp_path / "prepared" / "labels" / "train" / "sample.txt"
    assert out_yaml == (tmp_path / "prepared" / "data.yaml")
    assert prepared_image.exists()
    assert prepared_label.exists()
    assert summary["image_preprocess_pipeline"] == "gray"
    assert summary["splits"]["train"]["image_count"] == 1
    assert summary["splits"]["train"]["preprocess_workers"] == 2
    assert prepared_label.read_text(encoding="utf-8").startswith("0 ")
