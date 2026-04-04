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


def test_expand_line_boxes_vertically_adds_thirty_percent_only_above() -> None:
    expanded = ui_workbench._expand_line_boxes_vertically(
        [(5, 10, 25, 20)],
        image_height=100,
        top_pad_ratio=0.30,
        bottom_pad_ratio=0.0,
    )

    assert expanded == [(5, 7, 25, 20)]


def test_filter_tall_narrow_boxes_removes_vertical_short_boxes_anywhere() -> None:
    filtered = ui_workbench._filter_tall_narrow_boxes(
        [
            (0, 0, 14, 42),
            (90, 5, 104, 41),
            (30, 10, 150, 24),
            (160, 6, 192, 26),
        ],
        image_width=200,
    )

    assert filtered == [
        (30, 10, 150, 24),
        (160, 6, 192, 26),
    ]
