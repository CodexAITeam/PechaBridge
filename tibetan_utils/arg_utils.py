"""
Argument parsing utilities for the TibetanOCR project.
Multi-class support with Tibetan numbers, Tibetan text, and Chinese numbers.
"""

import argparse
from pathlib import Path
try:
    from ultralytics.data.utils import DATASETS_DIR
except ImportError:
    DATASETS_DIR = "./datasets"  # Fallback if ultralytics not installed

from .config import (
    DEFAULT_BACKGROUND_TRAIN_PATH,
    DEFAULT_BACKGROUND_VAL_PATH,
    DEFAULT_CORPORA_PATH,
    DEFAULT_FONT_PATH,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_WORKERS,
    DEFAULT_TRAIN_SAMPLES,
    DEFAULT_VAL_SAMPLES,
    DEFAULT_AUGMENTATION,
    DEFAULT_ANNOTATION_FILE_PATH
)


def add_model_arguments(parser):
    """Add model-related arguments."""
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Path to the model file')
    parser.add_argument('--imgsz', type=int, default=DEFAULT_IMAGE_SIZE,
                       help='Image size for inference')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for detections')


def add_output_arguments(parser):
    """Add output-related arguments."""
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--save-crops', action='store_true',
                       help='Save cropped text regions')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with verbose output')


def add_dataset_generation_arguments(parser):
    """Add dataset generation arguments for multi-class support."""
    parser.add_argument('--background_train', type=str, default=DEFAULT_BACKGROUND_TRAIN_PATH,
                       help='Folder with background images for training')
    parser.add_argument('--background_val', type=str, default=DEFAULT_BACKGROUND_VAL_PATH,
                       help='Folder with background images for validation')
    parser.add_argument('--output_dir', type=str, default=str(Path(DATASETS_DIR)),
                       help='Base directory to save the generated dataset. (Default: Ultralytics DATASETS_DIR)')
    parser.add_argument('--dataset_name', type=str, default='yolo_tibetan_dataset',
                       help='Name for the generated dataset folder.')
    
    # Multi-class corpora paths
    parser.add_argument('--corpora_tibetan_numbers_path', type=str, 
                       default='./data/corpora/Tibetan Number Words/',
                       help='Folder with Tibetan number words (maps to class_id 0: "tibetan_number_word").')
    parser.add_argument('--corpora_tibetan_text_path', type=str, 
                       default='./data/corpora/UVA Tibetan Spoken Corpus/',
                       help='Folder with general Tibetan text (maps to class_id 1: "tibetan_text").')
    parser.add_argument('--corpora_chinese_numbers_path', type=str, 
                       default='./data/corpora/Chinese Number Words/',
                       help='Folder with Chinese number words (maps to class_id 2: "chinese_number_word").')
    
    # Sample counts
    parser.add_argument('--train_samples', type=int, default=DEFAULT_TRAIN_SAMPLES,
                       help='Number of training samples to generate')
    parser.add_argument('--val_samples', type=int, default=DEFAULT_VAL_SAMPLES,
                       help='Number of validation samples to generate')
    
    # Multi-font support
    parser.add_argument('--font_path_tibetan', type=str, required=True, 
                       default='ext/Microsoft Himalaya.ttf',
                       help='Path to a font file that supports Tibetan characters')
    parser.add_argument('--font_path_chinese', type=str, required=True, 
                       default='ext/simkai.ttf',
                       help='Path to a font file that supports Chinese characters')
    
    # Image dimensions
    parser.add_argument('--image_width', type=int, default=1024,
                       help='Width (pixels) of each generated image.')
    parser.add_argument('--image_height', type=int, default=361,
                       help='Height (pixels) of each generated image.')
    
    # Labels and augmentation
    parser.add_argument('--single_label', action='store_true',
                       help='Use a single label "tibetan" for all files instead of using filenames as labels')
    parser.add_argument("--augmentation", choices=['rotate', 'noise', 'none'], default=DEFAULT_AUGMENTATION,
                       help="Type of augmentation to apply")
    
    # YOLO annotations support
    parser.add_argument('--annotations_file_path', type=str,
                       default=DEFAULT_ANNOTATION_FILE_PATH,
                       help='Path to a YOLO annotation file to load and draw bounding boxes from.')


def add_training_arguments(parser):
    """Add training-related arguments."""
    parser.add_argument('--dataset', type=str, default='yolo_tibetan/',
                       help='Path to dataset YAML file')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=DEFAULT_BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                       help='Number of worker threads')
    parser.add_argument('--device', type=str, default='',
                       help='Device to use for training')
    parser.add_argument('--project', type=str, default='runs/detect',
                       help='Project directory')
    parser.add_argument('--name', type=str, default='train',
                       help='Experiment name')
    parser.add_argument('--export', action='store_true',
                       help='Export model after training')
    parser.add_argument('--patience', type=int, default=50,
                       help='EarlyStopping patience')


def add_wandb_arguments(parser):
    """Add Weights & Biases arguments."""
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='PechaBridge',
                       help='W&B project name')
    parser.add_argument('--wandb-entity', type=str,
                       help='W&B entity (team or username)')
    parser.add_argument('--wandb-tags', type=str,
                       help='Comma-separated tags for the experiment')
    parser.add_argument('--wandb-name', type=str,
                       help='Name of the experiment in wandb')


def add_sbb_arguments(parser):
    """Add Staatsbibliothek zu Berlin arguments."""
    parser.add_argument('--ppn', type=str, required=True,
                       help='PPN (Pica Production Number) of the document')
    parser.add_argument('--download', action='store_true',
                       help='Download images instead of processing them directly')
    parser.add_argument('--max-images', type=int, default=0,
                       help='Maximum number of images to process (0 = all)')
    parser.add_argument('--no-ssl-verify', action='store_true',
                       help='Disable SSL certificate verification')


def add_ocr_arguments(parser):
    """Add OCR-related arguments."""
    parser.add_argument('--lang', type=str, default='eng+deu',
                       help='Language for Tesseract OCR')
    parser.add_argument('--tesseract-config', type=str, default='',
                       help='Additional Tesseract configuration')


def add_source_argument(parser):
    """Add source argument for input files."""
    parser.add_argument('--source', type=str,
                       help='Path to image file or directory')


def create_generate_dataset_parser():
    """Create parser for multi-class dataset generation."""
    parser = argparse.ArgumentParser(description="Generate YOLO dataset for Tibetan text detection")
    add_dataset_generation_arguments(parser)
    add_output_arguments(parser)
    return parser


def create_train_parser():
    """Create parser for model training."""
    parser = argparse.ArgumentParser(description="Train YOLO model for Tibetan text detection")
    add_training_arguments(parser)
    add_wandb_arguments(parser)
    return parser


def create_inference_parser():
    """Create parser for inference."""
    parser = argparse.ArgumentParser(description="Run inference on images")
    add_model_arguments(parser)
    add_source_argument(parser)
    add_output_arguments(parser)
    return parser


def create_sbb_inference_parser():
    """Create parser for SBB inference."""
    parser = argparse.ArgumentParser(description="Run inference on SBB data")
    add_model_arguments(parser)
    add_sbb_arguments(parser)
    add_output_arguments(parser)
    return parser


def create_ocr_parser():
    """Create parser for OCR on detections."""
    parser = argparse.ArgumentParser(description="Apply OCR to detected text blocks")
    add_model_arguments(parser)
    add_source_argument(parser)
    add_sbb_arguments(parser)
    add_ocr_arguments(parser)
    add_output_arguments(parser)
    return parser
