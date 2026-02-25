from pechabridge.ocr.weak_labeler import belongs_to_shard


def test_sharding_uses_patch_id_modulo():
    patch_ids = list(range(20))
    shard0 = [pid for pid in patch_ids if belongs_to_shard(pid, shard_id=0, num_shards=2)]
    shard1 = [pid for pid in patch_ids if belongs_to_shard(pid, shard_id=1, num_shards=2)]

    assert set(shard0).isdisjoint(set(shard1))
    assert sorted(shard0 + shard1) == patch_ids
    assert shard0 == [pid for pid in patch_ids if pid % 2 == 0]
    assert shard1 == [pid for pid in patch_ids if pid % 2 == 1]
