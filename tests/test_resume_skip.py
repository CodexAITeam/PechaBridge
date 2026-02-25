import pandas as pd

from pechabridge.ocr.weak_labeler import filter_patch_ids_for_resume, load_existing_patch_ids


def test_resume_skips_existing_patch_ids(tmp_path):
    out_path = tmp_path / "weak_ocr.parquet"
    pd.DataFrame({"patch_id": [2, 4], "text": ["a", "b"]}).to_parquet(out_path, index=False)

    existing = load_existing_patch_ids(out_path)
    filtered = filter_patch_ids_for_resume(
        [1, 2, 3, 4, 5],
        existing_patch_ids=existing,
        resume=True,
        overwrite=False,
    )
    assert filtered == [1, 3, 5]
