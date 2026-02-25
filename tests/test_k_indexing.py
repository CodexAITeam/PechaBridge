from collections import defaultdict

from pechabridge.data_gen.patch_sampler import assign_option_a_k


def test_assign_option_a_k_sorted_and_contiguous():
    rows = [
        {"patch_id": 3, "doc_id": "d1", "page_id": "p1", "line_id": 1, "scale_w": 256, "x0_px": 40, "x1_px": 80},
        {"patch_id": 1, "doc_id": "d1", "page_id": "p1", "line_id": 1, "scale_w": 256, "x0_px": 10, "x1_px": 50},
        {"patch_id": 2, "doc_id": "d1", "page_id": "p1", "line_id": 1, "scale_w": 256, "x0_px": 25, "x1_px": 65},
        {"patch_id": 4, "doc_id": "d1", "page_id": "p1", "line_id": 1, "scale_w": 384, "x0_px": 5, "x1_px": 60},
        {"patch_id": 5, "doc_id": "d2", "page_id": "p3", "line_id": 7, "scale_w": 256, "x0_px": 12, "x1_px": 24},
    ]

    out = assign_option_a_k(rows)
    groups = defaultdict(list)
    for r in out:
        key = (r["doc_id"], r["page_id"], int(r["line_id"]), int(r["scale_w"]))
        groups[key].append(r)

    for key, recs in groups.items():
        recs_sorted_by_k = sorted(recs, key=lambda r: int(r["k"]))
        ks = [int(r["k"]) for r in recs_sorted_by_k]
        assert ks == list(range(len(recs_sorted_by_k))), f"k not contiguous for group={key}"

        x0s = [int(r["x0_px"]) for r in recs_sorted_by_k]
        assert x0s == sorted(x0s), f"x0 is not sorted by k for group={key}"

