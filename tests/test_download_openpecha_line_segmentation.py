from scripts.download_openpecha_line_segmentation import (
    _assign_canonical_split,
    _make_output_stem,
    _parse_image_size,
    create_parser,
)


def test_parse_image_size_accepts_common_hf_format():
    assert _parse_image_size("2000x379") == (2000, 379)
    assert _parse_image_size("3000×1026") == (3000, 1026)


def test_assign_canonical_split_is_deterministic():
    split_a = _assign_canonical_split("W24769", seed=42, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    split_b = _assign_canonical_split("W24769", seed=42, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    assert split_a == split_b
    assert split_a in {"train", "val", "test"}


def test_make_output_stem_uses_work_id_source_name_and_hash():
    stem = _make_output_stem(
        {
            "bdrc_work_id": "W24769",
            "source_image": "38460539.jpg",
            "image_url": "https://example.com/38460539.jpg",
        }
    )
    assert stem.startswith("W24769__38460539__")
    assert len(stem.split("__")[-1]) == 10


def test_downloader_no_longer_exposes_image_preprocess_pipeline():
    parser = create_parser()
    args = parser.parse_args(["--output-dir", "/tmp/test-line-seg"])
    assert not hasattr(args, "image_preprocess_pipeline")
