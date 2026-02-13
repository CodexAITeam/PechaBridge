#!/usr/bin/env python3
"""
Skript zur Inferenz mit einem trainierten YOLO-Modell für Tibetische OCR
mit Daten von der Staatsbibliothek zu Berlin.
"""

import os
from pathlib import Path
import tempfile

# Importiere Funktionen aus der tibetan_utils-Bibliothek
from tibetan_utils.arg_utils import create_sbb_inference_parser
from tibetan_utils.model_utils import ModelManager
from tibetan_utils.sbb_utils import process_sbb_images, get_sbb_metadata


def process_image(image, model, **kwargs):
    """
    Process an image with the YOLO model.
    
    Args:
        image: Image to process
        model: YOLO model
        **kwargs: Additional arguments for prediction
        
    Returns:
        dict: Processing results
    """
    # Run inference
    results = model.predict(source=image, **kwargs)
    
    # Count detections
    detection_count = sum(len(r.boxes) for r in results if hasattr(r, 'boxes'))
    
    return {
        'results': results,
        'detection_count': detection_count
    }


def main():
    # Parse arguments
    parser = create_sbb_inference_parser()
    
    # Add YOLO-specific options
    parser.add_argument('--save', action='store_true', default=True,
                        help='Speichere Ergebnisse')
    parser.add_argument('--show', action='store_true',
                        help='Zeige Ergebnisse während der Inferenz an')
    parser.add_argument('--save-txt', action='store_true',
                        help='Speichere Ergebnisse als .txt Dateien')
    parser.add_argument('--save-conf', action='store_true',
                        help='Speichere Konfidenzwerte in .txt Dateien')
    parser.add_argument('--download-workers', type=int, default=8,
                        help='Anzahl paralleler Downloader-Threads (nur mit --download)')
    
    args = parser.parse_args()

    # Get document metadata
    metadata = get_sbb_metadata(args.ppn, verify_ssl=not args.no_ssl_verify)
    if metadata['title']:
        print(f"Dokument: {metadata['title']}")
        if metadata['author']:
            print(f"Autor: {metadata['author']}")
        if metadata['date']:
            print(f"Datum: {metadata['date']}")
        print(f"Seiten: {metadata['pages']}")

    model = None
    model_path = Path(args.model)
    download_only = bool(args.download and not model_path.exists())
    if download_only:
        print(f"Hinweis: Modell nicht gefunden ({model_path}). Starte reinen Download-Modus.")
    else:
        # Check if model exists for inference mode
        if not model_path.exists():
            print(f"Fehler: Modell nicht gefunden: {model_path}")
            return
        # Load model
        print(f"Lade Modell: {model_path}")
        model = ModelManager.load_model(str(model_path))
    
    # Prepare prediction arguments
    predict_args = {
        'imgsz': args.imgsz,
        'conf': args.conf,
        'device': getattr(args, 'device', ''),
        'save': getattr(args, 'save', True),
        'project': getattr(args, 'project', 'runs/detect'),
        'name': getattr(args, 'name', 'predict'),
        'show': getattr(args, 'show', False),
        'save_txt': getattr(args, 'save_txt', False),
        'save_conf': getattr(args, 'save_conf', False)
    }
    
    # Process images
    if download_only:
        results = process_sbb_images(
            args.ppn,
            lambda img, **kwargs: {'downloaded_path': str(img)},
            max_images=args.max_images,
            download=True,
            output_dir=args.output,
            verify_ssl=not args.no_ssl_verify,
            download_workers=max(1, int(args.download_workers)),
        )
        print(f"\nDownload abgeschlossen. Bilder gespeichert unter: {Path(args.output).resolve()}")
        print(f"Heruntergeladene Bilder: {len(results)}")
        return
    else:
        results = process_sbb_images(
            args.ppn,
            lambda img, **kwargs: process_image(img, model, **kwargs),
            max_images=args.max_images,
            download=args.download,
            output_dir=args.output,
            verify_ssl=not args.no_ssl_verify,
            download_workers=max(1, int(args.download_workers)),
            **predict_args
        )
    
    # Output directory
    output_dir = Path(args.project) / args.name
    print(f"\nInferenz abgeschlossen. Ergebnisse gespeichert unter: {output_dir}")
    
    # Results summary
    if results:
        total_detections = sum(result['detection_count'] for result in results)
        print(f"Insgesamt {total_detections} Tibetische Textblöcke erkannt.")


if __name__ == "__main__":
    main()
