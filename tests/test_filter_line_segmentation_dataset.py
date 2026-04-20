from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from scripts.filter_line_segmentation_dataset import create_parser, run


def _make_source_dataset(root: Path) -> Path:
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    (root / "meta").mkdir(parents=True)
    Image.fromarray(np.full((32, 48, 3), 255, dtype=np.uint8)).save(root / "images" / "train" / "sample.png")
    (root / "labels" / "train" / "sample.txt").write_text(
        "\n".join(
            [
                "0 0.100000 0.400000 0.900000 0.400000 0.900000 0.500000 0.100000 0.500000",
                "0 0.100000 0.100000 0.200000 0.100000 0.200000 0.700000 0.100000 0.700000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "meta" / "source_note.txt").write_text("hello", encoding="utf-8")
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


def test_filter_line_segmentation_dataset_removes_tall_polygons(tmp_path: Path) -> None:
    source_root = _make_source_dataset(tmp_path / "source_dataset")
    output_root = tmp_path / "filtered_dataset"
    args = create_parser().parse_args(
        [
            "--dataset",
            str(source_root),
            "--output-dir",
            str(output_root),
            "--copy-mode",
            "copy",
            "--min-width-height-ratio",
            "1.0",
        ]
    )

    summary = run(args)

    out_label = output_root / "labels" / "train" / "sample.txt"
    assert (output_root / "images" / "train" / "sample.png").exists()
    assert out_label.exists()
    assert (output_root / "meta" / "source_note.txt").exists()
    kept_lines = [line for line in out_label.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(kept_lines) == 1
    assert kept_lines[0].startswith("0 0.100000 0.400000")

    saved_summary = json.loads((output_root / "meta" / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary["transform"]["min_width_height_ratio"] == 1.0
    assert saved_summary["totals"]["kept_instances"] == 1
    assert saved_summary["totals"]["filtered_instances"] == 1
    assert summary["output_dir"] == str(output_root.resolve())


def test_filter_line_segmentation_dataset_parser_defaults() -> None:
    args = create_parser().parse_args(["--dataset", "/tmp/in", "--output-dir", "/tmp/out"])
    assert args.min_width_height_ratio == 1.0
    assert args.copy_mode == "hardlink"
