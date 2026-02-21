from pathlib import Path

from pechabridge.mining.mnn_miner import MiningConfig, PatchRecord, is_excluded_pair


def _rec(patch_id: int, doc: str, page: str, line: int, k: int = 0) -> PatchRecord:
    return PatchRecord(
        patch_id=int(patch_id),
        doc_id=str(doc),
        page_id=str(page),
        line_id=int(line),
        scale_w=256,
        k=int(k),
        x0_norm=0.1,
        x1_norm=0.2,
        line_w_px=1000,
        line_h_px=112,
        boundary_score=0.0,
        ink_ratio=0.4,
        image_path=Path("/tmp/x.png"),
    )


def test_exclusion_same_page_and_line_rules():
    cfg = MiningConfig(
        exclude_same_page=True,
        exclude_same_doc=False,
        exclude_same_line=True,
        exclude_nearby_lines=1,
        exclude_nearby_k=0,
    )
    a = _rec(1, "docA", "1", 10)
    b_same_page = _rec(2, "docA", "1", 99)
    c_near_line = _rec(3, "docA", "1", 11)
    d_other_page = _rec(4, "docA", "2", 10)
    e_other_doc = _rec(5, "docB", "1", 10)

    assert is_excluded_pair(a, b_same_page, cfg) is True
    assert is_excluded_pair(a, c_near_line, cfg) is True
    assert is_excluded_pair(a, d_other_page, cfg) is False
    assert is_excluded_pair(a, e_other_doc, cfg) is False

