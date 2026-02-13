#!/usr/bin/env python3
"""
Skript zur Generierung von Multi-Klassen-Trainingsdaten für Tibetische OCR.
Erstellt synthetische Bilder mit Tibetischem Text, chinesischen Zahlen und allgemeinem Text für YOLO-Training.

Unterstützt 3 Klassen:
- Klasse 0: tibetan_number_word (Tibetische Zahlen)
- Klasse 1: tibetan_text (Allgemeiner tibetischer Text)
- Klasse 2: chinese_number_word (Chinesische Zahlen)
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

from tibetanDataGenerator.dataset_generator import generate_dataset
from tibetan_utils.arg_utils import create_generate_dataset_parser


def _parse_csv_items(spec: str) -> list[str]:
    return [item.strip() for item in str(spec).split(",") if item.strip()]


def _has_any_images(folder: Path) -> bool:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return any(p.is_file() and p.suffix.lower() in exts for p in folder.rglob("*"))


def _run_lora_augmentation_on_dir(input_dir: Path, args, seed_offset: int = 0) -> dict:
    from scripts.texture_augment import run as run_texture_augment

    if not input_dir.exists() or not input_dir.is_dir():
        return {"images_processed": 0, "output_dir": str(input_dir), "skipped": "missing_dir"}
    if not _has_any_images(input_dir):
        return {"images_processed": 0, "output_dir": str(input_dir), "skipped": "no_images"}

    effective_seed = None
    if args.lora_augment_seed is not None:
        effective_seed = int(args.lora_augment_seed) + int(seed_offset)

    aug_args = argparse.Namespace(
        model_family=args.lora_augment_model_family,
        input_dir=str(input_dir),
        output_dir=str(input_dir),  # in-place keeps references stable
        strength=float(args.lora_augment_strength),
        steps=int(args.lora_augment_steps),
        guidance_scale=float(args.lora_augment_guidance_scale),
        seed=effective_seed,
        controlnet_scale=float(args.lora_augment_controlnet_scale),
        lora_path=str(args.lora_augment_path),
        lora_scale=float(args.lora_augment_scale),
        prompt=str(args.lora_augment_prompt),
        base_model_id=str(args.lora_augment_base_model_id),
        controlnet_model_id=str(args.lora_augment_controlnet_model_id),
        canny_low=int(args.lora_augment_canny_low),
        canny_high=int(args.lora_augment_canny_high),
    )
    return run_texture_augment(aug_args)


def _augment_generated_dataset(full_dataset_path: Path, args) -> list[dict]:
    if not str(args.lora_augment_path).strip():
        return []

    splits = _parse_csv_items(args.lora_augment_splits)
    if not splits:
        splits = ["train"]

    targets = ["images"]
    if args.lora_augment_targets == "images_and_ocr_crops":
        targets.append("ocr_crops")

    reports: list[dict] = []
    seed_offset = 0
    for split in splits:
        for target in targets:
            folder = full_dataset_path / split / target
            print(f"LoRA-Augmentierung: split={split} target={target} path={folder}")
            rep = _run_lora_augmentation_on_dir(folder, args, seed_offset=seed_offset)
            rep["split"] = split
            rep["target"] = target
            reports.append(rep)
            seed_offset += 1000
    return reports


def main():
    parser = create_generate_dataset_parser()
    args = parser.parse_args()

    full_dataset_path = Path(args.output_dir) / args.dataset_name
    original_dataset_name = args.dataset_name
    args.dataset_name = str(full_dataset_path)

    print(f"Generiere Multi-Klassen YOLO-Datensatz in {args.dataset_name}...")
    print("Speicherort kann geändert werden per `yolo settings`.")
    print("Unterstützte Klassen:")
    print("  - Klasse 0: tibetan_number_word (Tibetische Zahlen)")
    print("  - Klasse 1: tibetan_text (Allgemeiner tibetischer Text)")
    print("  - Klasse 2: chinese_number_word (Chinesische Zahlen)")

    train_dataset_info = generate_dataset(args, validation=False)
    _ = generate_dataset(args, validation=True)

    lora_reports = _augment_generated_dataset(full_dataset_path, args)

    yaml_content = OrderedDict()
    yaml_content["path"] = original_dataset_name
    yaml_content["train"] = "train/images"
    yaml_content["val"] = "val/images"
    yaml_content["test"] = ""

    if "nc" not in train_dataset_info or "names" not in train_dataset_info:
        raise ValueError("generate_dataset did not return 'nc' or 'names' in its info dictionary.")
    yaml_content["nc"] = train_dataset_info["nc"]
    yaml_content["names"] = train_dataset_info["names"]

    yaml_file_path = Path(args.output_dir) / f"{original_dataset_name}.yaml"
    import yaml

    def represent_ordereddict(dumper, data):
        return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())

    yaml.add_representer(OrderedDict, represent_ordereddict)
    with open(yaml_file_path, "w", encoding="utf-8") as f:
        yaml.dump(dict(yaml_content), f, sort_keys=False, allow_unicode=True)

    print(f"\nMulti-Klassen-Datensatzgenerierung abgeschlossen. YAML-Konfiguration gespeichert: {yaml_file_path}")
    if lora_reports:
        augmented_total = sum(int(r.get("images_processed", 0) or 0) for r in lora_reports)
        print(f"LoRA-Augmentierung abgeschlossen. Verarbeitete Bilder: {augmented_total}")
        for rep in lora_reports:
            print(f"  - {rep.get('split')}/{rep.get('target')}: {rep.get('images_processed', 0)}")
    print("Training kann mit folgendem Befehl gestartet werden:\n")
    print(f"yolo detect train data={yaml_file_path} epochs=100 imgsz=[{args.image_height},{args.image_width}] model=yolov8n.pt")


if __name__ == "__main__":
    main()
