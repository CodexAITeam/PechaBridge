from pechabridge.mining.mnn_miner import find_mutual_ranks


def test_find_mutual_ranks_accepts_valid_mnn():
    src = 10
    dst = 20
    src_list = [(20, 0.82, 1), (30, 0.77, 2), (40, 0.72, 3)]
    dst_list = [(10, 0.84, 1), (41, 0.79, 2), (42, 0.70, 3)]
    ok, r1, r2 = find_mutual_ranks(src, dst, src_list, dst_list, mutual_topk=3)
    assert ok is True
    assert r1 == 1
    assert r2 == 1


def test_find_mutual_ranks_rejects_non_mutual():
    src = 10
    dst = 20
    src_list = [(20, 0.82, 1), (30, 0.77, 2), (40, 0.72, 3)]
    dst_list = [(41, 0.84, 1), (10, 0.79, 4)]  # src only appears beyond mutual_topk
    ok, r1, r2 = find_mutual_ranks(src, dst, src_list, dst_list, mutual_topk=3)
    assert ok is False
    assert r1 == -1 or r2 == -1

