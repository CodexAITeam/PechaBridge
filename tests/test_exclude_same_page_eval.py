from pechabridge.eval.eval_faiss_crosspage import EvalPatch, filter_ranked_indices


def test_exclude_same_page_eval():
    rows = [
        EvalPatch(patch_id=1, doc_id="d1", page_id="1", line_id=1, scale_w=256, x0_norm=0.0, x1_norm=0.1),
        EvalPatch(patch_id=2, doc_id="d1", page_id="1", line_id=2, scale_w=256, x0_norm=0.2, x1_norm=0.3),
        EvalPatch(patch_id=3, doc_id="d1", page_id="2", line_id=1, scale_w=256, x0_norm=0.1, x1_norm=0.2),
        EvalPatch(patch_id=4, doc_id="d2", page_id="9", line_id=1, scale_w=256, x0_norm=0.1, x1_norm=0.2),
    ]
    ranked = [0, 1, 2, 3]

    filtered = filter_ranked_indices(
        query_index=0,
        ranked_indices=ranked,
        rows=rows,
        exclude_same_page=True,
    )
    assert filtered == [2, 3]

    unfiltered = filter_ranked_indices(
        query_index=0,
        ranked_indices=ranked,
        rows=rows,
        exclude_same_page=False,
    )
    assert unfiltered == [1, 2, 3]

