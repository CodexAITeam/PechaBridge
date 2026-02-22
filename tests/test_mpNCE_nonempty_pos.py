from pathlib import Path

import torch

from pechabridge.training.losses import MpNCEConfig, multi_positive_infonce
from pechabridge.training.mnn_pairs import PatchMeta


def _meta(pid: int, doc: str, page: str, line: int, scale: int, x0: float, x1: float) -> PatchMeta:
    return PatchMeta(
        patch_id=int(pid),
        doc_id=str(doc),
        page_id=str(page),
        line_id=int(line),
        scale_w=int(scale),
        k=0,
        x0_norm=float(x0),
        x1_norm=float(x1),
        ink_ratio=0.2,
        boundary_score=0.2,
        image_path=Path("/tmp/x.png"),
    )


def test_mpNCE_nonempty_pos():
    torch.manual_seed(0)
    z = torch.randn(4, 16, dtype=torch.float32)
    metas = [
        _meta(1, "d1", "p1", 1, 256, 0.1, 0.2),
        _meta(2, "d2", "p2", 1, 256, 0.1, 0.2),
        _meta(1, "d1", "p1", 1, 256, 0.1, 0.2),
        _meta(2, "d2", "p2", 1, 256, 0.1, 0.2),
    ]
    patch_ids = [1, 2, 1, 2]
    mnn_map = {
        1: [(2, 0.7)],
        2: [(1, 0.7)],
    }
    cfg = MpNCEConfig(
        tau=0.07,
        w_overlap=0.0,
        w_multiscale=0.0,
        t_iou=0.6,
        eps_center=0.06,
        min_positives_per_anchor=1,
        allow_self_fallback=False,
        exclude_same_page_in_denominator=False,
        lambda_smooth=0.0,
    )
    loss, stats = multi_positive_infonce(
        z=z,
        metas=metas,
        patch_ids=patch_ids,
        mnn_map=mnn_map,
        cfg=cfg,
    )
    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0
    assert float(stats.get("valid_anchors", 0.0)) >= 2.0
    assert float(stats.get("mnn_pos", 0.0)) > 0.0


def test_mpNCE_ocr_only_nonempty_pos():
    torch.manual_seed(1)
    z = torch.randn(4, 8, dtype=torch.float32)
    metas = [
        _meta(10, "d1", "p1", 1, 256, 0.1, 0.2),
        _meta(20, "d2", "p9", 1, 256, 0.1, 0.2),
        _meta(10, "d1", "p1", 1, 256, 0.1, 0.2),
        _meta(20, "d2", "p9", 1, 256, 0.1, 0.2),
    ]
    patch_ids = [10, 20, 10, 20]
    cfg = MpNCEConfig(
        tau=0.1,
        w_ocr=0.7,
        w_overlap=0.0,
        w_multiscale=0.0,
        use_mnn=False,
        use_ocr=True,
        allow_self_fallback=False,
        lambda_smooth=0.0,
    )
    loss, stats = multi_positive_infonce(
        z=z,
        metas=metas,
        patch_ids=patch_ids,
        mnn_map={},
        cfg=cfg,
        ocr_map={10: [(20, 0.8)], 20: [(10, 0.8)]},
    )
    assert torch.isfinite(loss)
    assert float(stats.get("valid_anchors", 0.0)) >= 2.0
    assert float(stats.get("ocr_pos", 0.0)) > 0.0
    assert float(stats.get("mnn_pos", 0.0)) == 0.0
