# PechaBridge Workbench

PechaBridge is intended to be used via the **Workbench UI**.

## Install

```bash
pip install -r requirements-ui.txt
```

If you want transformer/VLM features in the Workbench (`Batch VLM Layout (SBB)` and `VLM Layout` tabs), also install:

```bash
pip install -r requirements-vlm.txt
```

## Start the Workbench

```bash
python ui_workbench.py
```

## Recommended Workflow (UI only)

1. `Synthetic Data`: generate synthetic YOLO datasets.
2. `Batch VLM Layout (SBB)`: run VLM-based layout on SBB PPN pages (test-only), combine with synthetic data, export.
3. `Dataset Preview`: inspect images and label boxes.
4. `Ultralytics Training`: train detection models.
5. `Model Inference`: run trained model inference.
6. `VLM Layout`: single-image VLM layout parsing.
7. `Label Studio Export`: convert YOLO splits to Label Studio tasks and optionally launch Label Studio.
8. `PPN Downloader`: download and inspect SBB pages.
9. `CLI Audit`: view script options.

## Label Studio Notes

For local file serving in Label Studio, set:

```bash
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/absolute/path/to/your/dataset/root
```

Then use the Workbench export actions.

## CLI Documentation

CLI usage is documented separately in:

- [README_CLI.md](README_CLI.md)
- [README_PSEUDO_LABELING_LABEL_STUDIO.md](README_PSEUDO_LABELING_LABEL_STUDIO.md)

## License

MIT, see [LICENSE](LICENSE).
