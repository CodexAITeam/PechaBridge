"""OCR weak labeling package."""

from .preprocess import PreprocessConfig, preprocess_patch_image
from .preprocess_bdrc import BDRCPreprocessConfig, preprocess_image_bdrc, pad_ocr_line, bdrc_image_to_normalized_tensor
from .preprocess_rgb import RGBLinePreprocessConfig, preprocess_image_rgb_lines
from .line_segmentation import (
    DEFAULT_LINE_SEGMENTATION_CONF,
    DEFAULT_LINE_SEGMENTATION_IMGSZ,
    LinePrediction,
    coerce_polygon_points,
    polygon_to_box,
    polygon_to_yolo_segment_line,
    predict_line_regions,
    sort_line_predictions,
)
from .weak_labeler import WeakOCRConfig, run_weak_ocr_labeler

__all__ = [
    "WeakOCRConfig",
    "run_weak_ocr_labeler",
    "PreprocessConfig",
    "preprocess_patch_image",
    "BDRCPreprocessConfig",
    "preprocess_image_bdrc",
    "RGBLinePreprocessConfig",
    "preprocess_image_rgb_lines",
    "pad_ocr_line",
    "bdrc_image_to_normalized_tensor",
    "DEFAULT_LINE_SEGMENTATION_CONF",
    "DEFAULT_LINE_SEGMENTATION_IMGSZ",
    "LinePrediction",
    "coerce_polygon_points",
    "polygon_to_box",
    "polygon_to_yolo_segment_line",
    "predict_line_regions",
    "sort_line_predictions",
]
