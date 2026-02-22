# Weak OCR Labeler

This tool generates weak OCR labels per patch image and stores them in `weak_ocr.parquet` keyed by `patch_id`.

## Install Tesseract

macOS (Homebrew):

```bash
brew install tesseract
brew install tesseract-lang
```

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-bod
```

Notes:

- Tibetan language data (`bod`) is not always installed by default.
- If `bod` is unavailable, the backend automatically falls back to `eng`.

## Run

```bash
python -m pechabridge.cli.weak_ocr_label \
  --dataset /path/to/out_dataset \
  --meta /path/to/out_dataset/meta/patches.parquet \
  --out /path/to/out_dataset/meta/weak_ocr.parquet \
  --backend tesseract \
  --config configs/weak_ocr.yaml \
  --num_workers 8 \
  --shard_id 0 --num_shards 1 \
  --resume
```

Debug dumps:

```bash
python -m pechabridge.cli.weak_ocr_label ... --debug_dump 20
```

This writes:

- `<dataset>/debug/weak_ocr/*_orig.png`
- `<dataset>/debug/weak_ocr/*_prep.png`
- `<dataset>/debug/weak_ocr/*_text.txt`

## Confidence

`confidence` is computed from `image_to_data` word confidences:

- mean(valid_word_confidence) / 100
- `-1` values are ignored
- if no valid word confidence exists, `confidence` is `NaN`
- a fallback heuristic is still computed and stored in backend raw payload when raw storage is enabled

## Resume and Sharding

- Resume: use `--resume` to skip `patch_id`s already present in output parquet.
- Overwrite: use `--overwrite` to recompute this shard’s patch ids.
- Shard rule: `patch_id % num_shards == shard_id`.

## Output Schema

`weak_ocr.parquet` includes:

- `patch_id`, `doc_id`, `page_id`, `line_id`, `scale_w`
- `text`, `confidence`, `char_count`, `word_count`
- `lang_used`, `backend`
- `preprocess_hash`, `ocr_config_hash`
- `error_code`, `error_msg`
- optional `raw_json` when `output.store_raw=true`

## Swapping to OCR-VLM Later

The pipeline calls only the backend interface in:

- `pechabridge/ocr/backends/base.py`

Current VLM placeholder:

- `pechabridge/ocr/backends/vlm_backend_stub.py`

To switch from Tesseract to VLM later:

1. Implement API call logic inside `VLMBackendStub.ocr_image`.
2. Keep return type as `OCRResult`.
3. Run CLI with `--backend vlm` and corresponding backend config.
