"""
Configuration utilities for the TibetanOCR project.
"""

import os
import yaml
from pathlib import Path

# Default paths
DEFAULT_FONT_PATH = 'ext/Microsoft Himalaya.ttf'
DEFAULT_OUTPUT_DIR = 'runs/detect'
DEFAULT_DATASET_DIR = 'yolo_tibetan/'

# Additional default constants for multi-class support
DEFAULT_ANNOTATION_FILE_PATH = './data/tibetan numbers/annotations/tibetan_chinese_no/bg_PPN337138764X_00000005.txt'
DEFAULT_BACKGROUND_TRAIN_PATH = './data/tibetan numbers/backgrounds/'
DEFAULT_BACKGROUND_VAL_PATH = './data/tibetan numbers/backgrounds/'
DEFAULT_CORPORA_PATH = './data/corpora/'
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 100
DEFAULT_WORKERS = 8

# Default model settings
DEFAULT_MODEL_PATH = 'yolov8n.pt'
DEFAULT_IMAGE_SIZE = 1024
DEFAULT_CONFIDENCE = 0.25

# Default OCR settings
DEFAULT_OCR_LANG = 'eng+deu'
TIBETAN_OCR_LANG = 'bod'  # Tibetan language code for Tesseract

# Default dataset generation settings
DEFAULT_TRAIN_SAMPLES = 100
DEFAULT_VAL_SAMPLES = 100
DEFAULT_AUGMENTATION = 'noise'

# Default SBB settings
DEFAULT_SBB_OUTPUT = 'sbb_images'
DEFAULT_OCR_OUTPUT = 'ocr_results'


class Config:
    """Configuration manager for TibetanOCR project."""
    
    @staticmethod
    def load_config(config_file=None):
        """
        Load configuration from a YAML file or use defaults.
        
        Args:
            config_file: Path to a YAML configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        config = {
            'paths': {
                'font': DEFAULT_FONT_PATH,
                'output': DEFAULT_OUTPUT_DIR,
                'dataset': DEFAULT_DATASET_DIR,
            },
            'model': {
                'path': DEFAULT_MODEL_PATH,
                'image_size': DEFAULT_IMAGE_SIZE,
                'confidence': DEFAULT_CONFIDENCE,
            },
            'ocr': {
                'language': DEFAULT_OCR_LANG,
                'tibetan_language': TIBETAN_OCR_LANG,
                'output': DEFAULT_OCR_OUTPUT,
            },
            'dataset': {
                'train_samples': DEFAULT_TRAIN_SAMPLES,
                'val_samples': DEFAULT_VAL_SAMPLES,
                'augmentation': DEFAULT_AUGMENTATION,
            },
            'sbb': {
                'output': DEFAULT_SBB_OUTPUT,
            }
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = yaml.safe_load(f)
                # Update config with user values
                for section, values in user_config.items():
                    if section in config:
                        config[section].update(values)
                    else:
                        config[section] = values
        
        return config
    
    @staticmethod
    def save_config(config, config_file):
        """
        Save configuration to a YAML file.
        
        Args:
            config: Configuration dictionary
            config_file: Path to save the configuration
        """
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
