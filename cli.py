#!/usr/bin/env python3
"""Unified PechaBridge CLI entrypoint for diffusion and retrieval-encoder workflows."""

from __future__ import annotations

import argparse
import logging

from tibetan_utils.arg_utils import (
    create_eval_text_hierarchy_vit_parser,
    create_faiss_text_hierarchy_search_parser,
    create_prepare_texture_lora_dataset_parser,
    create_prepare_donut_ocr_dataset_parser,
    create_run_donut_ocr_workflow_parser,
    create_train_donut_ocr_parser,
    create_train_image_encoder_parser,
    create_train_text_hierarchy_vit_parser,
    create_train_text_encoder_parser,
    create_texture_augment_parser,
    create_train_texture_lora_parser,
)
from pechabridge.cli.gen_patches import create_parser as create_gen_patches_parser, run as run_gen_patches
from pechabridge.cli.mine_mnn_pairs import create_parser as create_mnn_pairs_parser, run as run_mnn_pairs
from pechabridge.cli.weak_ocr_label import create_parser as create_weak_ocr_label_parser, run as run_weak_ocr_label
from pechabridge.eval.eval_faiss_crosspage import create_parser as create_eval_faiss_crosspage_parser
from pechabridge.eval.eval_faiss_crosspage import run as run_eval_faiss_crosspage
from scripts.download_merge_openpecha_ocr_lines import (
    create_parser as create_download_openpecha_ocr_lines_parser,
)
from scripts.download_bosentencepiece_tokenizer import (
    create_parser as create_download_bosentencepiece_tokenizer_parser,
)
from scripts.eval_ocr_tokenizer import create_parser as create_eval_ocr_tokenizer_parser
from scripts.warm_line_clip_workbench_cache import (
    create_parser as create_warm_line_clip_workbench_cache_parser,
)

LOGGER = logging.getLogger("pechabridge_cli")


def _build_root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PechaBridge command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parent = create_prepare_texture_lora_dataset_parser(add_help=False)
    prepare_parser = subparsers.add_parser(
        "prepare-texture-lora-dataset",
        parents=[prepare_parent],
        help="Prepare real-page texture crops + JSONL metadata for LoRA training",
        description=prepare_parent.description,
    )
    prepare_parser.set_defaults(handler=_run_prepare_texture_lora_dataset)

    train_parent = create_train_texture_lora_parser(add_help=False)
    train_parser = subparsers.add_parser(
        "train-texture-lora",
        parents=[train_parent],
        help="Train SDXL texture LoRA adapters using accelerate",
        description=train_parent.description,
    )
    train_parser.set_defaults(handler=_run_train_texture_lora)

    augment_parent = create_texture_augment_parser(add_help=False)
    augment_parser = subparsers.add_parser(
        "texture-augment",
        parents=[augment_parent],
        help="Apply SDXL + ControlNet Canny texture augmentation",
        description=augment_parent.description,
    )
    augment_parser.set_defaults(handler=_run_texture_augment)

    train_image_parent = create_train_image_encoder_parser(add_help=False)
    train_image_parser = subparsers.add_parser(
        "train-image-encoder",
        parents=[train_image_parent],
        help="Train self-supervised image encoder for Tibetan page retrieval",
        description=train_image_parent.description,
    )
    train_image_parser.set_defaults(handler=_run_train_image_encoder)

    train_text_parent = create_train_text_encoder_parser(add_help=False)
    train_text_parser = subparsers.add_parser(
        "train-text-encoder",
        parents=[train_text_parent],
        help="Train unsupervised Tibetan text encoder",
        description=train_text_parent.description,
    )
    train_text_parser.set_defaults(handler=_run_train_text_encoder)

    train_hierarchy_parent = create_train_text_hierarchy_vit_parser(add_help=False)
    train_hierarchy_parser = subparsers.add_parser(
        "train-text-hierarchy-vit",
        parents=[train_hierarchy_parent],
        help="Train ViT retrieval encoder on TextHierarchy or patch-parquet dataset",
        description=train_hierarchy_parent.description,
    )
    train_hierarchy_parser.set_defaults(handler=_run_train_text_hierarchy_vit)

    eval_hierarchy_parent = create_eval_text_hierarchy_vit_parser(add_help=False)
    eval_hierarchy_parser = subparsers.add_parser(
        "eval-text-hierarchy-vit",
        parents=[eval_hierarchy_parent],
        help="Evaluate ViT retrieval encoder on TextHierarchy or patch-parquet dataset",
        description=eval_hierarchy_parent.description,
    )
    eval_hierarchy_parser.set_defaults(handler=_run_eval_text_hierarchy_vit)

    faiss_hierarchy_parent = create_faiss_text_hierarchy_search_parser(add_help=False)
    faiss_hierarchy_parser = subparsers.add_parser(
        "faiss-text-hierarchy-search",
        parents=[faiss_hierarchy_parent],
        help="FAISS similarity search on TextHierarchy/patch-parquet embeddings",
        description=faiss_hierarchy_parent.description,
    )
    faiss_hierarchy_parser.set_defaults(handler=_run_faiss_text_hierarchy_search)

    prepare_donut_parent = create_prepare_donut_ocr_dataset_parser(add_help=False)
    prepare_donut_parser = subparsers.add_parser(
        "prepare-donut-ocr-dataset",
        parents=[prepare_donut_parent],
        help="Prepare label-filtered OCR manifests (JSONL) for Donut-style training",
        description=prepare_donut_parent.description,
    )
    prepare_donut_parser.set_defaults(handler=_run_prepare_donut_ocr_dataset)

    eval_ocr_tokenizer_parent = create_eval_ocr_tokenizer_parser(add_help=False)
    eval_ocr_tokenizer_parser = subparsers.add_parser(
        "eval-ocr-tokenizer",
        parents=[eval_ocr_tokenizer_parent],
        help="Evaluate tokenizer coverage/length behavior on OCR manifests (e.g. BoSentencePiece)",
        description=eval_ocr_tokenizer_parent.description,
    )
    eval_ocr_tokenizer_parser.set_defaults(handler=_run_eval_ocr_tokenizer)

    train_donut_parent = create_train_donut_ocr_parser(add_help=False)
    train_donut_parser = subparsers.add_parser(
        "train-donut-ocr",
        parents=[train_donut_parent],
        help="Train Donut-style OCR model (VisionEncoderDecoder) on OCR crops",
        description=train_donut_parent.description,
    )
    train_donut_parser.set_defaults(handler=_run_train_donut_ocr)

    workflow_parent = create_run_donut_ocr_workflow_parser(add_help=False)
    workflow_parser = subparsers.add_parser(
        "run-donut-ocr-workflow",
        parents=[workflow_parent],
        help="Run full label-1 OCR workflow: generate -> prepare -> train",
        description=workflow_parent.description,
    )
    workflow_parser.set_defaults(handler=_run_donut_ocr_workflow)

    hierarchy_parser = subparsers.add_parser(
        "export-text-hierarchy",
        help="Run YOLO on an input folder and export line + word-block hierarchy crops",
        description="Detect text regions and export Tibetan line hierarchy plus number crops.",
    )
    hierarchy_parser.add_argument("--model", type=str, required=True, help="Path to YOLO model (.pt)")
    hierarchy_parser.add_argument("--input-dir", type=str, required=True, help="Input image directory (recursive scan)")
    hierarchy_parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    hierarchy_parser.add_argument(
        "--no-samples",
        "--no_samples",
        dest="no_samples",
        type=int,
        default=0,
        help="Randomly sample at most N images from input_dir (0 = use all images)",
    )
    hierarchy_parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    hierarchy_parser.add_argument("--imgsz", type=int, default=1024, help="YOLO inference image size")
    hierarchy_parser.add_argument("--device", type=str, default="", help="Inference device (e.g. cpu, cuda:0)")
    hierarchy_parser.add_argument("--min-line-height", type=int, default=10, help="Minimum detected line height in pixels")
    hierarchy_parser.add_argument("--line-projection-smooth", type=int, default=9, help="Smoothing window for vertical line profile")
    hierarchy_parser.add_argument("--line-projection-threshold-rel", type=float, default=0.20, help="Relative threshold for vertical line profile")
    hierarchy_parser.add_argument("--line-merge-gap-px", type=int, default=5, help="Merge gap for neighboring line segments")
    hierarchy_parser.add_argument("--horizontal-profile-smooth-cols", type=int, default=21, help="Smoothing window for horizontal profile")
    hierarchy_parser.add_argument("--horizontal-profile-threshold-rel", type=float, default=0.20, help="Relative threshold for horizontal profile")
    hierarchy_parser.add_argument("--horizontal-seg-min-width-px", type=int, default=14, help="Minimum horizontal segment width")
    hierarchy_parser.add_argument("--horizontal-seg-merge-gap-px", type=int, default=6, help="Merge gap for horizontal segments")
    hierarchy_parser.add_argument(
        "--hierarchy-levels",
        type=str,
        default="2,4,8",
        help="Comma-separated hierarchy levels (e.g. 2,4,8)",
    )
    hierarchy_parser.set_defaults(handler=_run_export_text_hierarchy)

    openpecha_ocr_parent = create_download_openpecha_ocr_lines_parser(add_help=False)
    openpecha_ocr_parser = subparsers.add_parser(
        "download-openpecha-ocr-lines",
        aliases=["download-merge-openpecha-ocr-lines"],
        parents=[openpecha_ocr_parent],
        help="Download and merge OpenPecha OCR Hugging Face datasets into line dataset format",
        description=openpecha_ocr_parent.description,
    )
    openpecha_ocr_parser.set_defaults(handler=_run_download_openpecha_ocr_lines)

    bosentencepiece_parent = create_download_bosentencepiece_tokenizer_parser(add_help=False)
    bosentencepiece_parser = subparsers.add_parser(
        "download-bosentencepiece-tokenizer",
        aliases=["download-bosentencepiece"],
        parents=[bosentencepiece_parent],
        help="Download and verify OpenPecha BoSentencePiece tokenizer into ext/BoSentencePiece",
        description=bosentencepiece_parent.description,
    )
    bosentencepiece_parser.set_defaults(handler=_run_download_bosentencepiece_tokenizer)

    gen_patches_parent = create_gen_patches_parser(add_help=False)
    gen_patches_parser = subparsers.add_parser(
        "gen-patches",
        parents=[gen_patches_parent],
        help="Generate line sub-patch dataset with Option-A neighborhood metadata",
        description=gen_patches_parent.description,
    )
    gen_patches_parser.set_defaults(handler=_run_gen_patches)

    weak_ocr_parent = create_weak_ocr_label_parser(add_help=False)
    weak_ocr_parser = subparsers.add_parser(
        "weak-ocr-label",
        parents=[weak_ocr_parent],
        help="Generate weak OCR labels for patch datasets",
        description=weak_ocr_parent.description,
    )
    weak_ocr_parser.set_defaults(handler=_run_weak_ocr_label)

    mnn_parent = create_mnn_pairs_parser(add_help=False)
    mnn_parser = subparsers.add_parser(
        "mine-mnn-pairs",
        parents=[mnn_parent],
        help="Mine robust cross-page MNN positives from patch dataset",
        description=mnn_parent.description,
    )
    mnn_parser.set_defaults(handler=_run_mine_mnn_pairs)

    eval_cross_parent = create_eval_faiss_crosspage_parser(add_help=False)
    eval_cross_parser = subparsers.add_parser(
        "eval-faiss-crosspage",
        parents=[eval_cross_parent],
        help="Evaluate cross-page retrieval with FAISS from exported embeddings",
        description=eval_cross_parent.description,
    )
    eval_cross_parser.set_defaults(handler=_run_eval_faiss_crosspage)

    warm_line_clip_cache_parent = create_warm_line_clip_workbench_cache_parser(add_help=False)
    warm_line_clip_cache_parser = subparsers.add_parser(
        "warm-line-clip-workbench-cache",
        parents=[warm_line_clip_cache_parent],
        help="Build/persist line_clip Workbench corpus embeddings for all available OCR splits using the best line_clip model",
        description=warm_line_clip_cache_parent.description,
    )
    warm_line_clip_cache_parser.set_defaults(handler=_run_warm_line_clip_workbench_cache)

    return parser


def _run_prepare_texture_lora_dataset(args: argparse.Namespace) -> int:
    from scripts.prepare_texture_lora_dataset import run

    run(args)
    return 0


def _run_train_texture_lora(args: argparse.Namespace) -> int:
    from scripts.train_texture_lora_sdxl import run

    run(args)
    return 0


def _run_texture_augment(args: argparse.Namespace) -> int:
    from scripts.texture_augment import run

    run(args)
    return 0


def _run_train_image_encoder(args: argparse.Namespace) -> int:
    from scripts.train_image_encoder import run

    run(args)
    return 0


def _run_train_text_encoder(args: argparse.Namespace) -> int:
    from scripts.train_text_encoder import run

    run(args)
    return 0


def _run_train_text_hierarchy_vit(args: argparse.Namespace) -> int:
    from scripts.train_text_hierarchy_vit import run

    run(args)
    return 0


def _run_eval_text_hierarchy_vit(args: argparse.Namespace) -> int:
    from scripts.eval_text_hierarchy_vit import run

    run(args)
    return 0


def _run_faiss_text_hierarchy_search(args: argparse.Namespace) -> int:
    from scripts.faiss_text_hierarchy_search import run

    run(args)
    return 0


def _run_prepare_donut_ocr_dataset(args: argparse.Namespace) -> int:
    from scripts.prepare_donut_ocr_dataset import run

    run(args)
    return 0


def _run_eval_ocr_tokenizer(args: argparse.Namespace) -> int:
    from scripts.eval_ocr_tokenizer import run

    run(args)
    return 0


def _run_train_donut_ocr(args: argparse.Namespace) -> int:
    from scripts.train_donut_ocr import run

    run(args)
    return 0


def _run_donut_ocr_workflow(args: argparse.Namespace) -> int:
    from scripts.run_donut_ocr_workflow import run

    run(args)
    return 0


def _run_export_text_hierarchy(args: argparse.Namespace) -> int:
    from scripts.export_text_hierarchy import run

    run(args)
    return 0


def _run_download_openpecha_ocr_lines(args: argparse.Namespace) -> int:
    from scripts.download_merge_openpecha_ocr_lines import run

    run(args)
    return 0


def _run_download_bosentencepiece_tokenizer(args: argparse.Namespace) -> int:
    from scripts.download_bosentencepiece_tokenizer import run

    return int(run(args))


def _run_gen_patches(args: argparse.Namespace) -> int:
    run_gen_patches(args)
    return 0


def _run_weak_ocr_label(args: argparse.Namespace) -> int:
    run_weak_ocr_label(args)
    return 0


def _run_mine_mnn_pairs(args: argparse.Namespace) -> int:
    run_mnn_pairs(args)
    return 0


def _run_eval_faiss_crosspage(args: argparse.Namespace) -> int:
    run_eval_faiss_crosspage(args)
    return 0


def _run_warm_line_clip_workbench_cache(args: argparse.Namespace) -> int:
    from scripts.warm_line_clip_workbench_cache import run

    return int(run(args))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = _build_root_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error("No subcommand selected")
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
