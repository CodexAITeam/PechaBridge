import pytest
import numpy as np
from PIL import Image

from pechabridge.ocr.line_segmentation import (
    LinePrediction,
    apply_line_segmentation_preprocess,
    normalize_line_segmentation_preprocess_pipeline,
    polygon_to_box,
    polygon_to_yolo_segment_line,
    sort_line_predictions,
)


def test_polygon_to_yolo_segment_line_clips_and_normalizes():
    line = polygon_to_yolo_segment_line(
        [[-5, 5], [55, 5], [55, 25], [-5, 25], [-5, 5]],
        width=100,
        height=50,
        class_id=0,
    )
    assert line is not None
    parts = line.split()
    assert parts[0] == "0"
    coords = [float(v) for v in parts[1:]]
    assert len(coords) == 8
    assert all(0.0 <= v <= 1.0 for v in coords)
    assert coords[0] == pytest.approx(0.0)
    assert coords[1] == pytest.approx(0.1)


def test_polygon_to_box_uses_exclusive_max_corner():
    box = polygon_to_box(
        [[10, 20], [30, 20], [30, 40], [10, 40]],
        width=100,
        height=100,
    )
    assert box == (10, 20, 31, 41)


def test_sort_line_predictions_orders_top_to_bottom_then_left_to_right():
    preds = [
        LinePrediction(box=(100, 80, 150, 100), polygon=((100, 80), (149, 80), (149, 99), (100, 99)), confidence=0.9, class_id=0, label="line"),
        LinePrediction(box=(10, 20, 60, 40), polygon=((10, 20), (59, 20), (59, 39), (10, 39)), confidence=0.9, class_id=0, label="line"),
        LinePrediction(box=(70, 20, 120, 40), polygon=((70, 20), (119, 20), (119, 39), (70, 39)), confidence=0.9, class_id=0, label="line"),
    ]
    ordered = sort_line_predictions(preds)
    assert [pred.box for pred in ordered] == [
        (10, 20, 60, 40),
        (70, 20, 120, 40),
        (100, 80, 150, 100),
    ]


def test_normalize_line_segmentation_preprocess_pipeline_maps_alias():
    assert normalize_line_segmentation_preprocess_pipeline("bdrc_no_bin") == "gray"
    assert normalize_line_segmentation_preprocess_pipeline("weird") == "none"


def test_apply_line_segmentation_preprocess_gray_preserves_rgb_shape():
    image = Image.fromarray(np.full((12, 18, 3), 200, dtype=np.uint8), mode="RGB")
    out = apply_line_segmentation_preprocess(image, pipeline="gray")
    assert out.shape == (12, 18, 3)
    assert out.dtype == np.uint8
