"""OCR weak labeling package."""

from .preprocess import PreprocessConfig, preprocess_patch_image
from .preprocess_bdrc import BDRCPreprocessConfig, preprocess_image_bdrc, pad_ocr_line, bdrc_image_to_normalized_tensor
from .weak_labeler import WeakOCRConfig, run_weak_ocr_labeler

__all__ = [
    "WeakOCRConfig",
    "run_weak_ocr_labeler",
    "PreprocessConfig",
    "preprocess_patch_image",
    "BDRCPreprocessConfig",
    "preprocess_image_bdrc",
    "pad_ocr_line",
    "bdrc_image_to_normalized_tensor",
]
