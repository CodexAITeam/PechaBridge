#!/usr/bin/env python3
"""
Skript zur Generierung von Multi-Klassen-Trainingsdaten für Tibetische OCR.
Erstellt synthetische Bilder mit Tibetischem Text, chinesischen Zahlen und allgemeinem Text für YOLO-Training.

Unterstützt 3 Klassen:
- Klasse 0: tibetan_number_word (Tibetische Zahlen)
- Klasse 1: tibetan_text (Allgemeiner tibetischer Text)  
- Klasse 2: chinese_number_word (Chinesische Zahlen)
"""

from pathlib import Path
from collections import OrderedDict
try:
    from ultralytics.data.utils import DATASETS_DIR
except ImportError:
    DATASETS_DIR = "./datasets"  # Fallback if ultralytics not installed
from tibetanDataGenerator.dataset_generator import generate_dataset

# Importiere Funktionen aus der tibetan_utils-Bibliothek
from tibetan_utils.arg_utils import create_generate_dataset_parser
from tibetan_utils.io_utils import save_yaml


def main():
    # Parse arguments (Multi-Klassen-Support)
    parser = create_generate_dataset_parser()
    args = parser.parse_args()

    # Set dataset path
    full_dataset_path = Path(args.output_dir) / args.dataset_name
    original_dataset_name = args.dataset_name
    args.dataset_name = str(full_dataset_path)

    print(f"Generiere Multi-Klassen YOLO-Datensatz in {args.dataset_name}...")
    print("Speicherort kann geändert werden per `yolo settings`.")
    print("Unterstützte Klassen:")
    print("  - Klasse 0: tibetan_number_word (Tibetische Zahlen)")
    print("  - Klasse 1: tibetan_text (Allgemeiner tibetischer Text)")
    print("  - Klasse 2: chinese_number_word (Chinesische Zahlen)")

    # Generate training dataset (Multi-Klassen)
    train_dataset_info = generate_dataset(args, validation=False)

    # Generate validation dataset (Multi-Klassen)
    val_dataset_info = generate_dataset(args, validation=True)

    # Multi-Klassen YAML-Konfiguration
    yaml_content = OrderedDict()
    yaml_content['path'] = original_dataset_name
    yaml_content['train'] = 'train/images'
    yaml_content['val'] = 'val/images'
    yaml_content['test'] = ''

    if 'nc' not in train_dataset_info or 'names' not in train_dataset_info:
        raise ValueError("generate_dataset did not return 'nc' or 'names' in its info dictionary.")
    yaml_content['nc'] = train_dataset_info['nc']
    yaml_content['names'] = train_dataset_info['names']

    # YAML speichern mit korrekter Funktion
    yaml_file_path = Path(args.output_dir) / f"{original_dataset_name}.yaml"
    
    # Verwende die modulare save_yaml Funktion
    import yaml
    def represent_ordereddict(dumper, data):
        return dumper.represent_mapping('tag:yaml.org,2002:map', data.items())
    
    yaml.add_representer(OrderedDict, represent_ordereddict)
    
    with open(yaml_file_path, 'w', encoding='utf-8') as f:
        yaml.dump(dict(yaml_content), f, sort_keys=False, allow_unicode=True)

    print(f"\nMulti-Klassen-Datensatzgenerierung abgeschlossen. YAML-Konfiguration gespeichert: {yaml_file_path}")
    print("Training kann mit folgendem Befehl gestartet werden:\n")
    print(f"yolo detect train data={yaml_file_path} epochs=100 imgsz=[{args.image_height},{args.image_width}] model=yolov8n.pt")


if __name__ == "__main__":
    main()
