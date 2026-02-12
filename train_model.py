#!/usr/bin/env python3
"""
Skript zum Trainieren eines YOLO-Modells mit den generierten Tibetischen OCR-Daten.
Unterstützt Weights & Biases (wandb) Logging für Experiment-Tracking.
"""

import os
import tempfile
from pathlib import Path
import yaml
from ultralytics import __version__ as ultralytics_version
from ultralytics.data.utils import DATASETS_DIR
from packaging.version import Version, InvalidVersion
try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

# Importiere Funktionen aus der tibetan_utils-Bibliothek
from tibetan_utils.arg_utils import create_train_parser
from tibetan_utils.model_utils import ModelManager


def initialize_wandb(args):
    """
    Initialize Weights & Biases logging.
    
    Args:
        args: Command-line arguments
        
    Returns:
        bool: Whether wandb was initialized
    """
    if not args.wandb:
        return False
    if wandb is None:
        print("Warnung: --wandb gesetzt, aber Paket 'wandb' ist nicht installiert. Logging wird deaktiviert.")
        return False
        
    wandb_name = args.wandb_name if args.wandb_name else args.name
    wandb_tags = args.wandb_tags.split(',') if args.wandb_tags else None
    
    print(f"Initialisiere Weights & Biases Logging")
    print(f"  Projekt: {args.wandb_project}")
    print(f"  Entity: {args.wandb_entity or 'Standard'}")
    print(f"  Run-Name: {wandb_name}")
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=wandb_name,
        tags=wandb_tags,
        config={
            "model": args.model,
            "dataset": args.dataset,
            "epochs": args.epochs,
            "batch_size": args.batch,
            "image_size": args.imgsz,
            "patience": args.patience
        }
    )
    
    return True


def save_model_to_wandb(model_path, export_path=None):
    """
    Save model to Weights & Biases as an artifact.
    
    Args:
        model_path: Path to the model file
        export_path: Path to the exported model file
    """
    if wandb is None or wandb.run is None:
        return
        
    artifact = wandb.Artifact(name=f"model-{wandb.run.id}", type="model")
    artifact.add_file(str(model_path))
    
    if export_path and os.path.exists(export_path):
        artifact.add_file(str(export_path))
        
    wandb.log_artifact(artifact)


def normalize_dataset_yaml_for_ultralytics(yaml_path: Path) -> Path:
    """
    Create a temporary dataset YAML with absolute paths so Ultralytics does not
    resolve relative paths against its global DATASETS_DIR.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {yaml_path}")

    raw_root = cfg.get("path", "")
    if raw_root:
        root = Path(raw_root).expanduser()
        if not root.is_absolute():
            root = (yaml_path.parent / root).resolve()
    else:
        root = yaml_path.parent.resolve()

    cfg["path"] = str(root)

    for key in ("train", "val", "test"):
        if key not in cfg or cfg[key] in ("", None):
            continue
        split = Path(str(cfg[key])).expanduser()
        if not split.is_absolute():
            split = (root / split).resolve()
        cfg[key] = str(split)

    fd, tmp_name = tempfile.mkstemp(prefix="pechabridge_dataset_", suffix=".yaml")
    os.close(fd)
    tmp_path = Path(tmp_name)
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return tmp_path


def _check_model_compatibility(model_name: str) -> None:
    """
    Fail early with a clear message if the selected model family is not
    supported by the installed Ultralytics version.
    """
    if not model_name.lower().startswith("yolo26"):
        return
    try:
        current = Version(ultralytics_version)
    except InvalidVersion:
        return
    minimum = Version("8.4.0")
    if current < minimum:
        raise RuntimeError(
            f"Model '{model_name}' requires a newer Ultralytics release "
            f"(installed: {ultralytics_version}, required: >= {minimum}). "
            "Please upgrade with: pip install -U ultralytics"
        )


def main():
    # Parse arguments
    parser = create_train_parser()
    args = parser.parse_args()

    # Path to dataset configuration
    script_root = Path(__file__).resolve().parent
    dataset_arg = Path(str(args.dataset)).expanduser()
    candidates = []

    # 1) Direct YAML path passed by user/UI.
    if dataset_arg.suffix.lower() in {".yml", ".yaml"}:
        candidates.append(dataset_arg)
    # 2) Direct dataset folder path containing data.yml.
    candidates.append(dataset_arg / "data.yml")
    # 3) Project-local relative path.
    candidates.append(script_root / dataset_arg)
    # 4) Project-local datasets folder (name or yaml filename).
    candidates.append(script_root / "datasets" / dataset_arg)
    if dataset_arg.suffix.lower() not in {".yml", ".yaml"}:
        candidates.append(script_root / "datasets" / f"{str(args.dataset)}.yaml")
        candidates.append(script_root / "datasets" / f"{str(args.dataset)}.yml")
    # 5) Project-local datasets/<name>/data.yml (folder layout).
    candidates.append(script_root / "datasets" / str(args.dataset) / "data.yml")
    # 6) Ultralytics global datasets dir (name, yaml, folder/data.yml).
    if dataset_arg.suffix.lower() in {".yml", ".yaml"}:
        candidates.append(Path(DATASETS_DIR) / dataset_arg.name)
    else:
        candidates.append(Path(DATASETS_DIR) / f"{str(args.dataset)}.yaml")
        candidates.append(Path(DATASETS_DIR) / f"{str(args.dataset)}.yml")
    candidates.append(Path(DATASETS_DIR) / str(args.dataset) / "data.yml")

    resolved_candidates = []
    for p in candidates:
        p = p.resolve()
        if p.is_file() and p.suffix.lower() in {".yml", ".yaml"}:
            resolved_candidates.append(p)
        elif p.is_dir():
            yml = p / "data.yml"
            yaml_alt = p / "data.yaml"
            if yml.exists():
                resolved_candidates.append(yml.resolve())
            elif yaml_alt.exists():
                resolved_candidates.append(yaml_alt.resolve())

    data_path = resolved_candidates[0] if resolved_candidates else None
    
    if data_path is None:
        print(f"Fehler: Datensatz-Konfiguration nicht gefunden fuer --dataset={args.dataset}")
        print("Erwartet eine der folgenden Eingaben fuer --dataset:")
        print("  - Datensatzname (z.B. tibetan-yolo)")
        print("  - Absoluter/relativer Pfad zu einem Datensatzordner mit data.yml")
        print("  - Absoluter/relativer Pfad direkt auf data.yml")
        print("Beispiel:")
        print("python train_model.py --dataset ./datasets/tibetan-yolo --epochs 100")
        return

    normalized_data_path = normalize_dataset_yaml_for_ultralytics(data_path)

    print(f"Starte Training mit Datensatz: {data_path}")
    if normalized_data_path != data_path:
        print(f"Nutze normalisierte Dataset-YAML (absolute Pfade): {normalized_data_path}")
    print(f"Basis-Modell: {args.model}")
    print(f"Epochen: {args.epochs}")
    print(f"Bildgröße: {args.imgsz}x{args.imgsz}")

    _check_model_compatibility(args.model)
    
    # Initialize Weights & Biases if enabled
    wandb_enabled = initialize_wandb(args)

    # Load model
    try:
        model = ModelManager.load_model(args.model)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load model '{args.model}'. "
            "If this is a named pretrained model (e.g. yolo26n.pt), Ultralytics "
            "will auto-download it when internet access is available."
        ) from exc

    # Start training
    # Train model
    results = ModelManager.train_model(
        model,
        data_path=str(normalized_data_path),
        epochs=args.epochs,
        image_size=args.imgsz,
        batch_size=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        use_wandb=wandb_enabled,
    )

    # Best model path
    best_model_path = Path(args.project) / args.name / 'weights' / 'best.pt'
    
    print(f"\nTraining abgeschlossen. Bestes Modell gespeichert unter: {best_model_path}")

    # Export model if requested
    export_path = None
    if args.export and best_model_path.exists():
        print("\nExportiere Modell als TorchScript...")
        export_model = ModelManager.load_model(str(best_model_path))
        export_path = ModelManager.export_model(export_model, format='torchscript')
        print(f"Modell exportiert nach: {export_path}")
        
        # Example command for inference
        print("\nBeispiel für Inferenz mit dem exportierten Modell:")
        print(f"yolo predict task=detect model={export_path} imgsz={args.imgsz} source=data/my_inference_data/*.jpg")
    
    # Save model to wandb if enabled
    if wandb_enabled:
        save_model_to_wandb(best_model_path, export_path)
        wandb.finish()


if __name__ == "__main__":
    main()
