#!/usr/bin/env python3
"""
Apply OCR/layout parsing to images.

Supports multiple parser backends:
- legacy: YOLO detection + Tesseract OCR
- mineru25: MinerU2.5 CLI integration
- paddleocr_vl: Transformer-based PaddleOCR-VL backend
- qwen25vl: Transformer-based Qwen-VL backend
- granite_docling: Transformer-based Granite-Docling backend
- deepseek_ocr: Transformer-based DeepSeek-VL backend
- qwen3_vl: Qwen3-VL layout-only backend (no OCR text)
- groundingdino: GroundingDINO layout-only backend (no OCR text)
- florence2: Florence-2 layout-only backend (no OCR text)
"""

from pathlib import Path

from tibetan_utils.arg_utils import create_ocr_parser
from tibetan_utils.io_utils import find_images, save_json
from tibetan_utils.parsers import create_parser, list_parser_specs, parser_availability


def print_parser_table() -> None:
    """Print registered parser backends and current availability."""
    print("Verfügbare Parser:")
    for spec in list_parser_specs():
        available, reason = parser_availability(spec.key)
        status = "OK" if available else "N/A"
        print(
            f"  - {spec.key:9s} | {status:3s} | {spec.display_name} | "
            f"Layout={spec.supports_layout} OCR={spec.supports_ocr} | {reason}"
        )


def build_backend(args):
    """Create selected parser backend from CLI args."""
    if args.parser == "legacy":
        return create_parser(
            "legacy",
            model_path=args.model,
            lang=args.lang,
            conf=args.conf,
            tesseract_config=args.tesseract_config,
            save_crops=args.save_crops,
        )
    if args.parser == "mineru25":
        return create_parser(
            "mineru25",
            mineru_command=args.mineru_command,
            timeout_sec=args.mineru_timeout,
        )
    if args.parser in (
        "paddleocr_vl",
        "qwen25vl",
        "granite_docling",
        "deepseek_ocr",
        "qwen3_vl",
        "groundingdino",
        "florence2",
    ):
        kwargs = {
            "prompt": args.vlm_prompt if args.vlm_prompt else None,
            "max_new_tokens": args.vlm_max_new_tokens,
            "hf_device": args.hf_device,
            "hf_dtype": args.hf_dtype,
        }
        if args.hf_model_id:
            kwargs["model_id"] = args.hf_model_id
        return create_parser(args.parser, **kwargs)
    raise ValueError(f"Unbekannter Parser: {args.parser}")


def process_local_image(image_path, backend, output_dir):
    """Process one local image with selected parser backend."""
    print(f"Verarbeite Bild: {image_path}")
    doc = backend.parse(image_path, output_dir=output_dir)
    out_data = doc.to_dict()

    output_path = Path(output_dir) / f"{Path(doc.image_name).stem}_ocr.json"
    save_json(out_data, str(output_path))

    print(f"  Erkannt: {len(out_data['detections'])} Blöcke")
    for j, det in enumerate(out_data["detections"][:5]):
        text_preview = det.get("text", "").replace("\n", " ")[:50]
        if len(det.get("text", "")) > 50:
            text_preview += "..."
        print(f"    Block {j + 1}: {text_preview}")
    if len(out_data["detections"]) > 5:
        print(f"    ... +{len(out_data['detections']) - 5} weitere")
    return out_data


def process_sbb_image(image, backend, output_dir):
    """Process one SBB image with selected parser backend."""
    doc = backend.parse(image, output_dir=output_dir)
    out_data = doc.to_dict()
    output_path = Path(output_dir) / f"{Path(doc.image_name).stem}_ocr.json"
    save_json(out_data, str(output_path))

    print(f"  Erkannt: {len(out_data['detections'])} Blöcke")
    for j, det in enumerate(out_data["detections"][:5]):
        text_preview = det.get("text", "").replace("\n", " ")[:50]
        if len(det.get("text", "")) > 50:
            text_preview += "..."
        print(f"    Block {j + 1}: {text_preview}")
    if len(out_data["detections"]) > 5:
        print(f"    ... +{len(out_data['detections']) - 5} weitere")
    return out_data


def main():
    parser = create_ocr_parser()
    args = parser.parse_args()

    if args.list_parsers:
        print_parser_table()
        return

    if not args.source and not args.ppn:
        parser.error("Entweder --source oder --ppn muss angegeben werden")

    available, reason = parser_availability(args.parser)
    if not available:
        raise RuntimeError(
            f"Parser '{args.parser}' ist nicht verfügbar: {reason}\n"
            "Nutze --list-parsers für Status aller Backends."
        )

    backend = build_backend(args)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    if args.source:
        source_path = Path(args.source)
        if not source_path.exists():
            print(f"Fehler: Quelle nicht gefunden: {source_path}")
            return

        if source_path.is_file():
            image_paths = [str(source_path)]
        else:
            image_paths = find_images(str(source_path), recursive=True)

        if not image_paths:
            print(f"Keine Bilder gefunden in: {source_path}")
            return

        print(f"Gefunden: {len(image_paths)} Bilder")
        for i, img_path in enumerate(image_paths):
            print(f"Verarbeite Bild {i + 1}/{len(image_paths)}: {img_path}")
            process_local_image(img_path, backend, args.output)

    elif args.ppn:
        from tibetan_utils.sbb_utils import process_sbb_images

        process_args = {
            "backend": backend,
            "output_dir": args.output,
        }
        process_sbb_images(
            args.ppn,
            lambda img, **kwargs: process_sbb_image(img, **kwargs),
            max_images=args.max_images,
            download=args.download,
            output_dir=args.output,
            verify_ssl=not args.no_ssl_verify,
            **process_args,
        )

    print(f"\nVerarbeitung abgeschlossen. Ergebnisse gespeichert unter: {args.output}")


if __name__ == "__main__":
    main()
