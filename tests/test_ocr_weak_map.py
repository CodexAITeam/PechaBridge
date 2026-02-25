from pathlib import Path

import pandas as pd

from pechabridge.training.weak_ocr_pairs import load_ocr_weak_map


def test_ocr_weak_map_groups_equal_text_with_conf_filter(tmp_path: Path):
    p = tmp_path / "weak_ocr.parquet"
    df = pd.DataFrame(
        [
            {"patch_id": 1, "text": "བཀྲ ཤིས", "confidence": 0.9, "char_count": 6, "error_code": None},
            {"patch_id": 2, "text": "བཀྲ   ཤིས", "confidence": 0.8, "char_count": 6, "error_code": ""},
            {"patch_id": 3, "text": "གཞན", "confidence": 0.95, "char_count": 3, "error_code": None},
            {"patch_id": 4, "text": "བཀྲ ཤིས", "confidence": 0.05, "char_count": 6, "error_code": None},
        ]
    )
    df.to_parquet(p, index=False)

    m = load_ocr_weak_map(
        weak_ocr_parquet=p,
        pair_min_confidence=0.2,
        min_chars=2,
        max_group_size=128,
        max_neighbors_per_anchor=0,
        weight_scale=1.0,
        require_no_error=True,
        patch_id_allowlist={1, 2, 3, 4},
    )
    assert 1 in m and 2 in m
    assert any(dst == 2 for dst, _ in m[1])
    assert any(dst == 1 for dst, _ in m[2])
    assert 4 not in m  # filtered by confidence
    assert 3 not in m  # singleton group

