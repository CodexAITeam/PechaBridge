import json

from PIL import Image

from scripts import download_merge_openpecha_ocr_lines as mod


class _FakeSplit:
    def __init__(self, rows):
        self._rows = list(rows)
        self.column_names = list(self._rows[0].keys()) if self._rows else []
        self.features = None
        self.num_rows = len(self._rows)

    def __iter__(self):
        return iter(self._rows)


def test_run_writes_flat_images_without_text_hierarchy(tmp_path, monkeypatch):
    row = {
        "image": Image.new("L", (8, 4), color=255),
        "text": "བཀྲ་ཤིས།",
        "doc_id": "DOC-1",
        "page_id": "PAGE-2",
        "line_id": 7,
    }

    def _fake_load_dataset_bundle(*args, **kwargs):
        return [(None, {"train": _FakeSplit([row])})]

    monkeypatch.setattr(mod, "_load_dataset_bundle", _fake_load_dataset_bundle)

    args = mod.create_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--dataset",
            "openpecha/OCR-Dergetenjur",
            "--num-workers",
            "1",
            "--skip-parquet",
        ]
    )

    summary = mod.run(args)

    manifest_path = tmp_path / "train" / "meta" / "lines.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1

    rec = rows[0]
    expected_rel = "train/images/dataset=OCR-Dergetenjur/doc=DOC-1/page=PAGE-2/line_000007.png"
    assert rec["line_path"] == expected_rel
    assert rec["src__image"] == expected_rel
    assert rec["line_path_dup_idx"] == 0
    assert "TextHierarchy" not in rec["line_path"]
    assert (tmp_path / expected_rel).exists()
    assert not any(path.name == "TextHierarchy" for path in tmp_path.rglob("*"))
    assert summary["split_outputs"]["train"]["image_root"] == str((tmp_path / "train" / "images").resolve())

