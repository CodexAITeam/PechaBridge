from pathlib import Path

from pechabridge.mining.mnn_miner import PatchRecord, select_center_match


def _rec(pid: int, center: float) -> PatchRecord:
    w = 0.1
    x0 = max(0.0, center - w * 0.5)
    x1 = min(1.0, center + w * 0.5)
    return PatchRecord(
        patch_id=int(pid),
        doc_id="d1",
        page_id="p1",
        line_id=1,
        scale_w=256,
        k=0,
        x0_norm=float(x0),
        x1_norm=float(x1),
        line_w_px=1000,
        line_h_px=112,
        boundary_score=0.2,
        ink_ratio=0.3,
        image_path=Path("/tmp/x.png"),
    )


def test_select_center_match_with_eps():
    pool = [_rec(1, 0.12), _rec(2, 0.50), _rec(3, 0.88)]
    m1 = select_center_match(pool, center=0.49, center_eps=0.03)
    assert m1 is not None
    assert int(m1.patch_id) == 2

    m2 = select_center_match(pool, center=0.49, center_eps=0.001)
    assert m2 is None

