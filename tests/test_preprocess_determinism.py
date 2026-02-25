from io import BytesIO

import numpy as np
from PIL import Image

from pechabridge.ocr.preprocess import PreprocessConfig, preprocess_patch_image


def _to_png_bytes(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_preprocess_is_deterministic_bytes():
    # Deterministic synthetic image with gradients and dark strokes.
    h, w = 48, 160
    arr = np.full((h, w, 3), 240, dtype=np.uint8)
    arr[:, ::7, :] = 30
    arr[10:14, 20:140, :] = 20
    arr[30:34, 40:120, :] = 10
    img = Image.fromarray(arr, mode="RGB")

    cfg = PreprocessConfig(
        grayscale=True,
        invert="auto",
        pad_px=8,
        resize_height=192,
        denoise=True,
        binarize="otsu",
        clahe=True,
    )

    out1 = preprocess_patch_image(img, cfg)
    out2 = preprocess_patch_image(img, cfg)

    assert _to_png_bytes(out1) == _to_png_bytes(out2)
