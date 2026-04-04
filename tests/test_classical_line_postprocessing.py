import pytest


ui_workbench = pytest.importorskip("ui_workbench")


def test_filter_line_boxes_by_mean_height_removes_large_outlier() -> None:
    line_boxes = [
        (0, 0, 40, 10),
        (0, 15, 40, 25),
        (0, 30, 40, 60),
    ]

    filtered = ui_workbench._filter_line_boxes_by_mean_height(line_boxes, tolerance_ratio=0.50)

    assert filtered == [
        (0, 0, 40, 10),
        (0, 15, 40, 25),
    ]


def test_expand_line_boxes_vertically_adds_quarter_height_above_and_below() -> None:
    expanded = ui_workbench._expand_line_boxes_vertically(
        [(5, 10, 25, 20)],
        image_height=100,
        pad_ratio=0.25,
    )

    assert expanded == [(5, 7, 25, 23)]
