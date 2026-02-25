from pechabridge.ocr.weak_labeler import stable_config_hash


def test_stable_config_hash_is_order_independent():
    a = {"backend": {"name": "tesseract", "lang": "bod", "psm": 6}, "preprocess": {"pad_px": 8, "binarize": "otsu"}}
    b = {"preprocess": {"binarize": "otsu", "pad_px": 8}, "backend": {"psm": 6, "lang": "bod", "name": "tesseract"}}
    assert stable_config_hash(a) == stable_config_hash(b)


def test_stable_config_hash_changes_on_value_change():
    a = {"backend": {"name": "tesseract", "lang": "bod"}}
    b = {"backend": {"name": "tesseract", "lang": "eng"}}
    assert stable_config_hash(a) != stable_config_hash(b)
