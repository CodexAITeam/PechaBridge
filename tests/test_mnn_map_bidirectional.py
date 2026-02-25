from pathlib import Path

import pandas as pd

from pechabridge.training.mnn_pairs import load_mnn_map


def test_mnn_map_bidirectional(tmp_path: Path):
    pairs_path = tmp_path / "pairs.parquet"
    df = pd.DataFrame(
        [
            {
                "src_patch_id": 10,
                "dst_patch_id": 20,
                "sim": 0.8,
                "stability_ratio": 0.5,
                "multi_scale_ok": True,
            }
        ]
    )
    df.to_parquet(pairs_path, index=False)

    m = load_mnn_map(
        pairs_parquet=pairs_path,
        pair_min_sim=0.1,
        pair_min_stability_ratio=0.1,
        require_multi_scale_ok=False,
        weight_scale=1.0,
        max_neighbors_per_anchor=0,
    )

    assert 10 in m
    assert 20 in m
    assert any(dst == 20 for dst, _ in m[10])
    assert any(dst == 10 for dst, _ in m[20])
    w_10_20 = next(w for dst, w in m[10] if dst == 20)
    w_20_10 = next(w for dst, w in m[20] if dst == 10)
    assert abs(float(w_10_20) - 0.4) < 1e-6
    assert abs(float(w_20_10) - 0.4) < 1e-6

