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

DEFAULT_TEXTURE_PROMPT = (
    "scanned printed Tibetan pecha page, paper texture, ink bleed, aged grayscale scan, "
    "realistic Tibetan glyph stroke thickness, subtle hand-written-like ink edge variation"
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
    parser.add_argument('--dataset_name', type=str, default='tibetan-yolo',
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
    parser.add_argument('--save_rendered_text_targets', action='store_true',
                       help='Save per-sample OCR targets with deterministic line linearization.')
    parser.add_argument('--save_ocr_crops', action='store_true',
                       help='Save per-region OCR crop images under <split>/ocr_crops.')
    parser.add_argument('--ocr_crop_labels', type=str, default='2',
                       help='Comma-separated class IDs for OCR crop export (e.g. "2" or "0,2"). Default: 2.')
    parser.add_argument('--target_newline_token', type=str, choices=['\\n', '<NL>'], default='\\n',
                       help="Line-break token used in OCR target_text: real newline ('\\n') or '<NL>'.")
    parser.add_argument('--lora_augment_path', type=str, default='',
                       help='Optional LoRA path to texture-augment generated data in-place.')
    parser.add_argument('--lora_augment_model_family', type=str, choices=['sdxl', 'sd21'], default='sdxl',
                       help='Diffusion model family used for LoRA texture augmentation.')
    parser.add_argument('--lora_augment_base_model_id', type=str,
                       default='stabilityai/stable-diffusion-xl-base-1.0',
                       help='Base diffusion model ID for LoRA augmentation.')
    parser.add_argument('--lora_augment_controlnet_model_id', type=str,
                       default='diffusers/controlnet-canny-sdxl-1.0',
                       help='ControlNet model ID for LoRA augmentation.')
    parser.add_argument('--lora_augment_prompt', type=str, default=DEFAULT_TEXTURE_PROMPT,
                       help='Prompt used for LoRA augmentation.')
    parser.add_argument('--lora_augment_scale', type=float, default=0.8,
                       help='LoRA cross-attention scale for augmentation.')
    parser.add_argument('--lora_augment_strength', type=float, default=0.2,
                       help='Img2img strength for LoRA augmentation (clamped to <=0.25 by backend).')
    parser.add_argument('--lora_augment_steps', type=int, default=28,
                       help='Diffusion inference steps for LoRA augmentation.')
    parser.add_argument('--lora_augment_guidance_scale', type=float, default=1.0,
                       help='Classifier-free guidance scale for LoRA augmentation.')
    parser.add_argument('--lora_augment_controlnet_scale', type=float, default=2.0,
                       help='ControlNet conditioning scale for LoRA augmentation.')
    parser.add_argument('--lora_augment_seed', type=int, default=None,
                       help='Optional random seed for deterministic LoRA augmentation.')
    parser.add_argument('--lora_augment_splits', type=str, default='train',
                       help='Comma-separated splits for augmentation (e.g. "train" or "train,val").')
    parser.add_argument('--lora_augment_targets', type=str, choices=['images', 'images_and_ocr_crops'],
                       default='images', help='Which generated assets to augment with LoRA.')
    parser.add_argument('--lora_augment_canny_low', type=int, default=100,
                       help='Canny low threshold used for LoRA augmentation conditioning.')
    parser.add_argument('--lora_augment_canny_high', type=int, default=200,
                       help='Canny high threshold used for LoRA augmentation conditioning.')


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


def add_sbb_arguments(parser, ppn_required: bool = True):
    """Add Staatsbibliothek zu Berlin arguments."""
    parser.add_argument('--ppn', type=str, required=ppn_required,
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
    parser.add_argument('--parser', type=str, default='legacy',
                       choices=[
                           'legacy',
                           'mineru25',
                           'paddleocr_vl',
                           'qwen25vl',
                           'granite_docling',
                           'deepseek_ocr',
                           'qwen3_vl',
                           'groundingdino',
                           'florence2',
                       ],
                       help='OCR/layout backend to use')
    parser.add_argument('--list-parsers', action='store_true',
                       help='List available parser backends and exit')
    parser.add_argument('--mineru-command', type=str, default='mineru',
                       help='MinerU CLI command name/path (used when --parser mineru25)')
    parser.add_argument('--mineru-timeout', type=int, default=300,
                       help='Timeout in seconds for MinerU CLI call')
    parser.add_argument('--hf-model-id', type=str, default='',
                       help='Override Hugging Face model ID for transformer backends')
    parser.add_argument('--vlm-prompt', type=str, default='',
                       help='Custom extraction prompt for transformer VLM backends')
    parser.add_argument('--vlm-max-new-tokens', type=int, default=1024,
                       help='Maximum generated tokens for transformer VLM backends')
    parser.add_argument('--hf-device', type=str, default='auto',
                       help='Device mode for HF backend (e.g. auto, cpu, cuda)')
    parser.add_argument('--hf-dtype', type=str, default='auto',
                       help='Torch dtype for HF model loading (e.g. auto, float16, bfloat16)')


def add_source_argument(parser):
    """Add source argument for input files."""
    parser.add_argument('--source', type=str,
                       help='Path to image file or directory')
    
# ---- Verifier CLI ----

def add_verifier_arguments(parser):
    """Arguments for training/evaluating the region-level image+text verifier."""
    parser.add_argument('--train_manifest', type=str, required=True,
                        help='Path to training manifest (tsv/jsonl) with region crops and texts')
    parser.add_argument('--val_manifest', type=str, required=True,
                        help='Path to validation manifest (tsv/jsonl) with region crops and texts')
    parser.add_argument('--image_root', type=str, required=True,
                        help='Root directory for region/crop images')

    parser.add_argument('--region_class', type=str, choices=[
        'TIBETAN_TEXT', 'TIBETAN_NUMBER', 'CHINESE_NUMBER'
    ], default='TIBETAN_TEXT', help='Which region head to train/use')

    # model dims (must match your frozen encoders)
    parser.add_argument('--image_embed_dim', type=int, default=768,
                        help='Dimensionality of frozen image embeddings')
    parser.add_argument('--text_embed_dim', type=int, default=512,
                        help='Dimensionality of frozen text embeddings')
    parser.add_argument('--proj_dim', type=int, default=512,
                        help='Hidden size of projector head')

    # training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='')

    # contrastive setup
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--num_hard_negatives', type=int, default=3,
                        help='Negatives per positive sample in a batch')

    # sliding-window over width (in feature steps, not pixels)
    parser.add_argument('--window_width_steps', type=int, default=32,
                        help='Window width in feature steps for pooled similarity')
    parser.add_argument('--window_stride_steps', type=int, default=16,
                        help='Stride in feature steps')

    # output
    parser.add_argument('--output_dir', type=str, default='outputs/verifier',
                        help='Where to save trained projector weights')

def add_verifier_preproc_arguments(parser):
    parser.add_argument('--vh', '--verifier-height', dest='verifier_height', type=int, default=64,
                        help='Fixed input height for verifier crops')
    parser.add_argument('--pad-multiple', type=int, default=8,
                        help='Pad width to nearest multiple')
    parser.add_argument('--bin', '--binarize', dest='bin', action='store_true',
                        help='Apply Sauvola-like binarization before feeding encoder')
    parser.add_argument('--edge-ch', action='store_true',
                        help='Append a Canny edge channel (second channel)')
    parser.add_argument('--norm-mean', type=float, default=0.5)
    parser.add_argument('--norm-std', type=float, default=0.5)

def add_verifier_textenc_arguments(parser):
    parser.add_argument('--st-model', type=str, default='billingsmoore/minilm-bo',
                        help='SentenceTransformer model name')
    parser.add_argument('--st-device', type=str, default='',
                        help='Device for ST model (default: auto)')
    parser.add_argument('--st-batch', type=int, default=64,
                        help='Batch size for ST encoding')
    parser.add_argument('--st-norm', action='store_true',
                        help='L2-normalize text embeddings (default on)')

def create_verifier_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Train region-level image+text verifier")
    add_verifier_arguments(parser)
    add_verifier_preproc_arguments(parser)
    add_verifier_textenc_arguments(parser)  
    add_output_arguments(parser)
    return parser



def create_generate_dataset_parser():
    """Create parser for multi-class dataset generation."""
    parser = argparse.ArgumentParser(description="Generate YOLO dataset for Tibetan text detection")
    add_dataset_generation_arguments(parser)
    add_output_arguments(parser)
    return parser


def create_train_parser():
    """Create parser for model training."""
    parser = argparse.ArgumentParser(description="Train YOLO model for Tibetan text detection")
    add_model_arguments(parser)
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
    add_sbb_arguments(parser, ppn_required=True)
    add_output_arguments(parser)
    return parser


def create_ocr_parser():
    """Create parser for OCR on detections."""
    parser = argparse.ArgumentParser(description="Apply OCR to detected text blocks")
    add_model_arguments(parser)
    add_source_argument(parser)
    add_sbb_arguments(parser, ppn_required=False)
    add_ocr_arguments(parser)
    add_output_arguments(parser)
    return parser


def add_prepare_texture_lora_dataset_arguments(parser):
    """Arguments for preparing texture LoRA crops from real pecha pages."""
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Folder with real pecha page images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output folder for generated crops + metadata.jsonl')
    parser.add_argument('--crop_size', type=int, default=1024,
                       help='Square crop size in pixels (e.g. 768 or 1024)')
    parser.add_argument('--num_crops_per_page', type=int, default=12,
                       help='Number of crops sampled from each page')
    parser.add_argument('--min_edge_density', type=float, default=0.025,
                       help='Minimum Canny edge density to treat a crop as text-rich')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for deterministic crop sampling')
    parser.add_argument('--canny_low', type=int, default=100,
                       help='Lower threshold for Canny edges')
    parser.add_argument('--canny_high', type=int, default=200,
                       help='Upper threshold for Canny edges')
    parser.add_argument('--lora_augment_path', type=str, default='',
                       help='Optional LoRA path to augment generated crops in-place.')
    parser.add_argument('--lora_augment_model_family', type=str, choices=['sdxl', 'sd21'], default='sdxl',
                       help='Diffusion model family used for optional crop augmentation.')
    parser.add_argument('--lora_augment_base_model_id', type=str,
                       default='stabilityai/stable-diffusion-xl-base-1.0',
                       help='Base diffusion model ID for optional crop augmentation.')
    parser.add_argument('--lora_augment_controlnet_model_id', type=str,
                       default='diffusers/controlnet-canny-sdxl-1.0',
                       help='ControlNet model ID for optional crop augmentation.')
    parser.add_argument('--lora_augment_prompt', type=str, default=DEFAULT_TEXTURE_PROMPT,
                       help='Prompt used for optional crop augmentation.')
    parser.add_argument('--lora_augment_scale', type=float, default=0.8,
                       help='LoRA scale used for optional crop augmentation.')
    parser.add_argument('--lora_augment_strength', type=float, default=0.2,
                       help='Img2img strength for optional crop augmentation.')
    parser.add_argument('--lora_augment_steps', type=int, default=28,
                       help='Diffusion steps for optional crop augmentation.')
    parser.add_argument('--lora_augment_guidance_scale', type=float, default=1.0,
                       help='Guidance scale for optional crop augmentation.')
    parser.add_argument('--lora_augment_controlnet_scale', type=float, default=2.0,
                       help='ControlNet scale for optional crop augmentation.')
    parser.add_argument('--lora_augment_seed', type=int, default=None,
                       help='Optional random seed for deterministic crop augmentation.')
    parser.add_argument('--lora_augment_canny_low', type=int, default=100,
                       help='Canny low threshold for optional crop augmentation.')
    parser.add_argument('--lora_augment_canny_high', type=int, default=200,
                       help='Canny high threshold for optional crop augmentation.')


def add_train_texture_lora_arguments(parser):
    """Arguments for SDXL texture LoRA training."""
    parser.add_argument('--model_family', type=str, default='sdxl',
                       choices=['sdxl', 'sd21'],
                       help='Model family to train LoRA for')
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='Folder containing texture crops (or an "images/" subfolder)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory where LoRA weights and training config are saved')
    parser.add_argument('--resolution', type=int, default=1024,
                       help='Training resolution for SDXL crops')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Per-device train batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--max_train_steps', type=int, default=1500,
                       help='Total optimization steps')
    parser.add_argument('--rank', type=int, default=16,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=16.0,
                       help='LoRA alpha')
    parser.add_argument('--mixed_precision', type=str, default='no',
                       choices=['no', 'fp16', 'bf16'],
                       help='Accelerate mixed precision mode')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                       help='Enable gradient checkpointing')
    parser.add_argument('--prompt', type=str, default=DEFAULT_TEXTURE_PROMPT,
                       help='Generic prompt used for texture LoRA training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for deterministic training')
    parser.add_argument('--base_model_id', type=str,
                       default='stabilityai/stable-diffusion-xl-base-1.0',
                       help='Base model ID (SDXL or SD2.1 depending on --model_family)')
    parser.add_argument('--train_text_encoder', action='store_true',
                       help='Also train text encoder LoRA adapters (off by default)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Dataloader workers')
    parser.add_argument('--lora_weights_name', type=str, default='texture_lora.safetensors',
                       help='Output LoRA filename')
    parser.add_argument('--checkpoint_every_steps', type=int, default=1000,
                       help='Save checkpoint every N optimizer steps (0 disables checkpointing). Step checkpoints are never overwritten.')
    parser.add_argument('--checkpoint_every_epochs', type=int, default=0,
                       help='Optional: save checkpoint every N epochs (0 disables epoch checkpointing)')
    parser.add_argument('--checkpoint_weights_name', type=str, default='texture_lora_checkpoint.safetensors',
                       help='Checkpoint LoRA filename when overwriting is enabled')
    parser.add_argument('--checkpoint_keep_all', dest='checkpoint_overwrite', action='store_false',
                       help='Keep all checkpoints instead of overwriting a single checkpoint file')
    parser.set_defaults(checkpoint_overwrite=True)


def add_texture_augment_arguments(parser):
    """Arguments for SDXL + ControlNet texture augmentation."""
    parser.add_argument('--model_family', type=str, default='sdxl',
                       choices=['sdxl', 'sd21'],
                       help='Model family for inference')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Folder with synthetic input images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Folder where texture-augmented images are written')
    parser.add_argument('--strength', type=float, default=0.2,
                       help='Img2img strength (kept conservative; values >0.25 are clamped)')
    parser.add_argument('--steps', type=int, default=28,
                       help='Diffusion inference steps')
    parser.add_argument('--guidance_scale', type=float, default=1.0,
                       help='Classifier-free guidance scale')
    parser.add_argument('--seed', type=int, default=None,
                       help='Base random seed; if set, outputs are deterministic')
    parser.add_argument('--controlnet_scale', type=float, default=2.0,
                       help='ControlNet conditioning scale (high to preserve structure)')
    parser.add_argument('--disable_controlnet', action='store_true',
                       help='Disable ControlNet completely and run plain img2img inference.')
    parser.add_argument('--lora_path', type=str, default='',
                       help='Optional path to LoRA directory or .safetensors file')
    parser.add_argument('--lora_scale', type=float, default=0.8,
                       help='LoRA scale for cross attention')
    parser.add_argument('--prompt', type=str, default=DEFAULT_TEXTURE_PROMPT,
                       help='Prompt for texture transfer (can be empty)')
    parser.add_argument('--base_model_id', type=str,
                       default='stabilityai/stable-diffusion-xl-base-1.0',
                       help='Base model ID (SDXL or SD2.1 depending on --model_family)')
    parser.add_argument('--controlnet_model_id', type=str,
                       default='diffusers/controlnet-canny-sdxl-1.0',
                       help='SDXL ControlNet Canny model ID')
    parser.add_argument('--canny_low', type=int, default=100,
                       help='Lower threshold for Canny conditioning map')
    parser.add_argument('--canny_high', type=int, default=200,
                       help='Upper threshold for Canny conditioning map')


def create_prepare_texture_lora_dataset_parser(add_help: bool = True):
    """Create parser for preparing a texture LoRA crop dataset."""
    parser = argparse.ArgumentParser(
        description="Prepare texture-focused LoRA crops from real pecha pages",
        add_help=add_help,
    )
    add_prepare_texture_lora_dataset_arguments(parser)
    return parser


def create_train_texture_lora_parser(add_help: bool = True):
    """Create parser for training SDXL texture LoRA."""
    parser = argparse.ArgumentParser(
        description="Train SDXL texture LoRA adapters for pecha renders",
        add_help=add_help,
    )
    add_train_texture_lora_arguments(parser)
    return parser


def create_texture_augment_parser(add_help: bool = True):
    """Create parser for SDXL texture augmentation with Canny ControlNet."""
    parser = argparse.ArgumentParser(
        description="Run structure-preserving texture augmentation on synthetic renders",
        add_help=add_help,
    )
    add_texture_augment_arguments(parser)
    return parser


def add_train_image_encoder_arguments(parser):
    """Arguments for self-supervised image encoder training."""
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Folder with training images (recursive scan)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory where trained image encoder artifacts are saved')
    parser.add_argument('--model_name_or_path', type=str, default='facebook/dinov2-base',
                       help='Backbone model ID or local path (transformers AutoModel)')
    parser.add_argument('--resolution', type=int, default=448,
                       help='Input image resolution for training')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Per-device batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--num_train_epochs', type=int, default=5,
                       help='Number of training epochs (used when max_train_steps=0)')
    parser.add_argument('--max_train_steps', type=int, default=0,
                       help='Override total train steps (0 = derive from num_train_epochs)')
    parser.add_argument('--warmup_steps', type=int, default=200,
                       help='Warmup steps for lr scheduler')
    parser.add_argument('--projection_dim', type=int, default=256,
                       help='Output dimension of projection head')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='NT-Xent temperature')
    parser.add_argument('--mixed_precision', type=str, default='fp16',
                       choices=['no', 'fp16', 'bf16'],
                       help='Accelerate mixed precision mode')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                       help='Enable backbone gradient checkpointing when supported')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone and train projection head only')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--checkpoint_every_steps', type=int, default=0,
                       help='Save a checkpoint every N optimizer steps (0 disables)')
    parser.add_argument('--checkpoint_name', type=str, default='checkpoint',
                       help='Checkpoint prefix in output_dir')
    parser.add_argument('--checkpoint_keep_all', dest='checkpoint_overwrite', action='store_false',
                       help='Keep all checkpoints instead of overwriting')
    parser.set_defaults(checkpoint_overwrite=True)


def add_train_text_encoder_arguments(parser):
    """Arguments for unsupervised Tibetan text encoder training."""
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Folder with text files (.txt/.jsonl/.csv/.tsv)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory where trained text encoder artifacts are saved')
    parser.add_argument('--model_name_or_path', type=str, default='google/byt5-small',
                       help='Backbone model ID or local path (transformers AutoModel)')
    parser.add_argument('--normalization', type=str, default='NFC',
                       choices=['NFC', 'NFKC', 'NFD', 'NFKD', 'none'],
                       help='Unicode normalization strategy')
    parser.add_argument('--min_chars', type=int, default=2,
                       help='Minimum number of characters per sample')
    parser.add_argument('--max_chars', type=int, default=512,
                       help='Maximum number of characters per sample')
    parser.add_argument('--max_length', type=int, default=256,
                       help='Tokenizer max sequence length')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Per-device batch size')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--num_train_epochs', type=int, default=5,
                       help='Number of training epochs (used when max_train_steps=0)')
    parser.add_argument('--max_train_steps', type=int, default=0,
                       help='Override total train steps (0 = derive from num_train_epochs)')
    parser.add_argument('--warmup_steps', type=int, default=200,
                       help='Warmup steps for lr scheduler')
    parser.add_argument('--projection_dim', type=int, default=256,
                       help='Output dimension of projection head')
    parser.add_argument('--temperature', type=float, default=0.05,
                       help='NT-Xent temperature')
    parser.add_argument('--mixed_precision', type=str, default='fp16',
                       choices=['no', 'fp16', 'bf16'],
                       help='Accelerate mixed precision mode')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                       help='Enable backbone gradient checkpointing when supported')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone and train projection head only')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--checkpoint_every_steps', type=int, default=0,
                       help='Save a checkpoint every N optimizer steps (0 disables)')
    parser.add_argument('--checkpoint_name', type=str, default='checkpoint',
                       help='Checkpoint prefix in output_dir')
    parser.add_argument('--checkpoint_keep_all', dest='checkpoint_overwrite', action='store_false',
                       help='Keep all checkpoints instead of overwriting')
    parser.set_defaults(checkpoint_overwrite=True)


def add_train_text_hierarchy_vit_arguments(parser):
    """Arguments for ViT retrieval training on exported TextHierarchy crops."""
    parser.add_argument('--dataset_dir', '--dataset-dir', dest='dataset_dir', type=str, required=True,
                       help='Dataset root (legacy TextHierarchy/ or new patches/ + meta/patches.parquet)')
    parser.add_argument('--output_dir', '--output-dir', dest='output_dir', type=str, required=True,
                       help='Directory where trained ViT artifacts are saved')
    parser.add_argument('--model_name_or_path', '--model-name-or-path', dest='model_name_or_path', type=str,
                       default='google/vit-base-patch16-224-in21k',
                       help='HF vision backbone ID/path (e.g. DINOv2, ViT, BEiT, Swin)')

    parser.add_argument('--include_line_images', '--include-line-images', dest='include_line_images',
                       action='store_true', help='Use line.png assets from TextHierarchy')
    parser.add_argument('--no_include_line_images', '--no-include-line-images', dest='include_line_images',
                       action='store_false', help='Disable line.png assets')
    parser.set_defaults(include_line_images=True)
    parser.add_argument('--include_word_crops', '--include-word-crops', dest='include_word_crops',
                       action='store_true', help='Use hierarchy word crops (word_*.png)')
    parser.add_argument('--no_include_word_crops', '--no-include-word-crops', dest='include_word_crops',
                       action='store_false', help='Disable hierarchy word crops')
    parser.set_defaults(include_word_crops=True)
    parser.add_argument('--include_number_crops', '--include-number-crops', dest='include_number_crops',
                       action='store_true', help='Include NumberCrops as singleton groups')
    parser.add_argument('--min_assets_per_group', '--min-assets-per-group', dest='min_assets_per_group', type=int, default=1,
                       help='Minimum assets required per positive group')

    parser.add_argument('--target_height', '--target-height', dest='target_height', type=int, default=64,
                       help='Fixed normalized input height in pixels')
    parser.add_argument('--width_buckets', '--width-buckets', dest='width_buckets', type=str, default='256,384,512,768',
                       help='Comma-separated target width buckets for right-padding')
    parser.add_argument('--max_width', '--max-width', dest='max_width', type=int, default=1024,
                       help='Maximum normalized width before clipping')
    parser.add_argument('--patch_multiple', '--patch-multiple', dest='patch_multiple', type=int, default=16,
                       help='Snap widths to a multiple of this value')

    parser.add_argument('--batch_size', '--batch-size', dest='batch_size', type=int, default=16,
                       help='Per-device batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', '--weight-decay', dest='weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--num_train_epochs', '--num-train-epochs', dest='num_train_epochs', type=int, default=8,
                       help='Training epochs (used when max_train_steps=0)')
    parser.add_argument('--max_train_steps', '--max-train-steps', dest='max_train_steps', type=int, default=0,
                       help='Override total train steps (0 = derive from epochs)')
    parser.add_argument('--warmup_steps', '--warmup-steps', dest='warmup_steps', type=int, default=200,
                       help='Warmup steps for lr scheduler')
    parser.add_argument('--projection_dim', '--projection-dim', dest='projection_dim', type=int, default=256,
                       help='Projection head output dimension')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='NT-Xent temperature')
    parser.add_argument('--mixed_precision', '--mixed-precision', dest='mixed_precision', type=str, default='fp16',
                       choices=['no', 'fp16', 'bf16'],
                       help='Accelerate mixed precision mode')
    parser.add_argument('--gradient_checkpointing', '--gradient-checkpointing', dest='gradient_checkpointing',
                       action='store_true', help='Enable backbone gradient checkpointing when supported')
    parser.add_argument('--freeze_backbone', '--freeze-backbone', dest='freeze_backbone',
                       action='store_true', help='Freeze backbone and train projection head only')
    parser.add_argument('--num_workers', '--num-workers', dest='num_workers', type=int, default=4,
                       help='Dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--checkpoint_every_steps', '--checkpoint-every-steps', dest='checkpoint_every_steps',
                       type=int, default=0, help='Save checkpoint every N optimizer steps (0 disables)')
    parser.add_argument('--train_mode', '--train-mode', dest='train_mode', type=str, default='auto',
                       choices=['auto', 'legacy', 'patch_mpnce', 'patch_clip', 'line_clip'],
                       help='Training mode selection (auto prefers patch_mpnce when patch parquet exists)')
    parser.add_argument('--train_manifest', '--train-manifest', dest='train_manifest', type=str, default='',
                       help='JSONL manifest for line_clip training (expects {"image":..., "text":...})')
    parser.add_argument('--val_manifest', '--val-manifest', dest='val_manifest', type=str, default='',
                       help='Optional JSONL manifest for line_clip validation/eval/export metadata context')
    parser.add_argument('--text_encoder_name_or_path', '--text-encoder-name-or-path', dest='text_encoder_name_or_path', type=str,
                       default='google/byt5-small',
                       help='HF text encoder ID/path for CLIP-style patch_clip training')
    parser.add_argument('--text_max_length', '--text-max-length', dest='text_max_length', type=int, default=128,
                       help='Tokenizer max length for CLIP-style patch_clip training')
    parser.add_argument('--freeze_text_encoder', '--freeze-text-encoder', dest='freeze_text_encoder',
                       action='store_true', help='Freeze text encoder in patch_clip mode and train projection heads only')
    parser.add_argument('--image_preprocess_pipeline', '--image-preprocess-pipeline', dest='image_preprocess_pipeline', type=str,
                       default='none', choices=['none', 'pb', 'bdrc'],
                       help='Optional deterministic pre-processing pipeline applied to images before ViT normalization')
    parser.add_argument('--patch_meta_parquet', '--patch-meta-parquet', dest='patch_meta_parquet', type=str, default='',
                       help='Optional explicit path to patch metadata parquet (default: <dataset>/meta/patches.parquet)')
    parser.add_argument('--pairs_parquet', '--pairs-parquet', dest='pairs_parquet', type=str, default='',
                       help='Optional explicit path to MNN pairs parquet (default: <dataset>/meta/mnn_pairs.parquet)')
    parser.add_argument('--ink_ratio_min', '--ink-ratio-min', dest='ink_ratio_min', type=float, default=0.0,
                       help='Patch filter: minimum ink_ratio')
    parser.add_argument('--boundary_score_min', '--boundary-score-min', dest='boundary_score_min', type=float, default=0.0,
                       help='Patch filter: minimum boundary_score')
    parser.add_argument('--positive_sources', '--positive-sources', dest='positive_sources', type=str, default='mnn',
                       choices=['mnn', 'ocr', 'both'],
                       help='Which weak positive sources to use in mp-InfoNCE')
    parser.add_argument('--require_pairs', '--require-pairs', dest='require_pairs', action='store_true',
                       help='Fail if no selected weak positives (MNN/OCR) remain after filtering')
    parser.add_argument('--pair_min_sim', '--pair-min-sim', dest='pair_min_sim', type=float, default=0.25,
                       help='Minimum pair similarity from mnn_pairs.parquet')
    parser.add_argument('--pair_min_stability_ratio', '--pair-min-stability-ratio', dest='pair_min_stability_ratio',
                       type=float, default=0.5, help='Minimum stability_ratio from mnn_pairs.parquet')
    parser.add_argument('--pair_require_multi_scale_ok', '--pair-require-multi-scale-ok', dest='pair_require_multi_scale_ok',
                       action='store_true', help='Require multi_scale_ok=true in mnn_pairs parquet')
    parser.add_argument('--max_neighbors_per_anchor', '--max-neighbors-per-anchor', dest='max_neighbors_per_anchor',
                       type=int, default=0, help='Cap loaded MNN neighbors per anchor (0 = unlimited)')
    parser.add_argument('--p_pair', '--p-pair', dest='p_pair', type=float, default=0.6,
                       help='Pair-aware sampler probability of adding an in-batch MNN partner')
    parser.add_argument('--hard_negative_ratio', '--hard-negative-ratio', dest='hard_negative_ratio', type=float, default=0.2,
                       help='Sampler fraction of same-page different-line hard negatives in each batch')
    parser.add_argument('--pair_sampling_seed', '--pair-sampling-seed', dest='pair_sampling_seed', type=int, default=42,
                       help='Random seed for pair-aware batch sampler')
    parser.add_argument('--w_mnn_scale', '--w-mnn-scale', dest='w_mnn_scale', type=float, default=1.0,
                       help='Global scale factor applied to MNN edge weights')
    parser.add_argument('--weak_ocr_parquet', '--weak-ocr-parquet', dest='weak_ocr_parquet', type=str, default='',
                       help='Optional weak OCR labels parquet (default: <dataset>/meta/weak_ocr.parquet)')
    parser.add_argument('--ocr_min_confidence', '--ocr-min-confidence', dest='ocr_min_confidence', type=float, default=0.2,
                       help='Weak OCR filter: minimum OCR confidence')
    parser.add_argument('--ocr_min_chars', '--ocr-min-chars', dest='ocr_min_chars', type=int, default=2,
                       help='Weak OCR filter: minimum character count')
    parser.add_argument('--ocr_max_group_size', '--ocr-max-group-size', dest='ocr_max_group_size', type=int, default=128,
                       help='Skip weak OCR text clusters larger than this size (0 disables)')
    parser.add_argument('--ocr_max_neighbors_per_anchor', '--ocr-max-neighbors-per-anchor', dest='ocr_max_neighbors_per_anchor',
                       type=int, default=0, help='Cap OCR-derived neighbors per anchor (0 = unlimited)')
    parser.add_argument('--ocr_require_no_error', '--ocr-require-no-error', dest='ocr_require_no_error', action='store_true',
                       help='Only use weak OCR rows without error_code')
    parser.add_argument('--no_ocr_require_no_error', '--no-ocr-require-no-error', dest='ocr_require_no_error', action='store_false',
                       help='Allow weak OCR rows with error_code')
    parser.set_defaults(ocr_require_no_error=True)
    parser.add_argument('--w_ocr_scale', '--w-ocr-scale', dest='w_ocr_scale', type=float, default=1.0,
                       help='Global scale factor applied while building OCR weak edges')
    parser.add_argument('--w_ocr', '--w-ocr', dest='w_ocr', type=float, default=0.5,
                       help='Loss weight multiplier for OCR weak positives in mp-InfoNCE numerator')
    parser.add_argument('--w_overlap', '--w-overlap', dest='w_overlap', type=float, default=0.3,
                       help='Weight for overlap positives in mp-InfoNCE')
    parser.add_argument('--w_multiscale', '--w-multiscale', dest='w_multiscale', type=float, default=0.2,
                       help='Weight for multiscale positives in mp-InfoNCE')
    parser.add_argument('--t_iou', '--t-iou', dest='t_iou', type=float, default=0.6,
                       help='1D IoU threshold for overlap positives')
    parser.add_argument('--eps_center', '--eps-center', dest='eps_center', type=float, default=0.06,
                       help='Normalized center distance for multiscale positives')
    parser.add_argument('--min_positives_per_anchor', '--min-positives-per-anchor', dest='min_positives_per_anchor',
                       type=int, default=1, help='Minimum positives required per anchor for mp-InfoNCE')
    parser.add_argument('--min_valid_anchors_per_batch', '--min-valid-anchors-per-batch', dest='min_valid_anchors_per_batch',
                       type=int, default=1, help='Skip optimizer step if batch has fewer valid anchors')
    parser.add_argument('--allow_self_fallback', '--allow-self-fallback', dest='allow_self_fallback', action='store_true',
                       help='Allow same patch_id second-view fallback positives when no other positives exist')
    parser.add_argument('--no_allow_self_fallback', '--no-allow-self-fallback', dest='allow_self_fallback', action='store_false',
                       help='Disable same patch_id fallback positives')
    parser.set_defaults(allow_self_fallback=True)
    parser.add_argument('--exclude_same_page_in_denominator', '--exclude-same-page-in-denominator',
                       dest='exclude_same_page_in_denominator', action='store_true',
                       help='Exclude same-page non-positive samples from InfoNCE denominator')
    parser.add_argument('--lambda_smooth', '--lambda-smooth', dest='lambda_smooth', type=float, default=0.05,
                       help='Overlap-only smoothness regularizer weight')
    parser.add_argument('--phase1_epochs', '--phase1-epochs', dest='phase1_epochs', type=int, default=0,
                       help='Phase-1 epochs (backbone frozen, head training). 0 => auto split')
    parser.add_argument('--phase2_epochs', '--phase2-epochs', dest='phase2_epochs', type=int, default=0,
                       help='Phase-2 epochs (unfreeze last transformer blocks). 0 => auto split')
    parser.add_argument('--unfreeze_last_n_blocks', '--unfreeze-last-n-blocks', dest='unfreeze_last_n_blocks',
                       type=int, default=2, help='Number of final transformer blocks to unfreeze in phase 2')
    parser.add_argument('--phase2_lr_scale', '--phase2-lr-scale', dest='phase2_lr_scale', type=float, default=0.1,
                       help='Learning-rate scale factor applied at phase-2 transition')


def create_train_image_encoder_parser(add_help: bool = True):
    """Create parser for image encoder training."""
    parser = argparse.ArgumentParser(
        description="Train a self-supervised Tibetan page image encoder",
        add_help=add_help,
    )
    add_train_image_encoder_arguments(parser)
    return parser


def create_train_text_encoder_parser(add_help: bool = True):
    """Create parser for text encoder training."""
    parser = argparse.ArgumentParser(
        description="Train an unsupervised Tibetan text encoder",
        add_help=add_help,
    )
    add_train_text_encoder_arguments(parser)
    return parser


def create_train_text_hierarchy_vit_parser(add_help: bool = True):
    """Create parser for ViT training on TextHierarchy crops."""
    parser = argparse.ArgumentParser(
        description="Train ViT retrieval encoder on legacy TextHierarchy or new patch-parquet dataset",
        add_help=add_help,
    )
    add_train_text_hierarchy_vit_arguments(parser)
    return parser


def add_eval_text_hierarchy_vit_arguments(parser):
    """Arguments for retrieval evaluation of a trained TextHierarchy ViT encoder."""
    parser.add_argument('--dataset_dir', '--dataset-dir', dest='dataset_dir', type=str, required=True,
                       help='Dataset root (legacy TextHierarchy/ or new patches/ + meta/patches.parquet)')
    parser.add_argument('--backbone_dir', '--backbone-dir', dest='backbone_dir', type=str, required=True,
                       help='Path to trained backbone directory (e.g. text_hierarchy_vit_backbone)')
    parser.add_argument('--projection_head_path', '--projection-head-path', dest='projection_head_path', type=str, default='',
                       help='Optional projection head checkpoint (.pt)')
    parser.add_argument('--output_dir', '--output-dir', dest='output_dir', type=str, required=True,
                       help='Directory where eval report/CSV are written')
    parser.add_argument('--config_path', '--config-path', dest='config_path', type=str, default='',
                       help='Optional explicit training_config.json path')

    parser.add_argument('--include_line_images', '--include-line-images', dest='include_line_images',
                       action='store_true', help='Use line.png assets from TextHierarchy')
    parser.add_argument('--no_include_line_images', '--no-include-line-images', dest='include_line_images',
                       action='store_false', help='Disable line.png assets')
    parser.set_defaults(include_line_images=True)
    parser.add_argument('--include_word_crops', '--include-word-crops', dest='include_word_crops',
                       action='store_true', help='Use hierarchy word crops (word_*.png)')
    parser.add_argument('--no_include_word_crops', '--no-include-word-crops', dest='include_word_crops',
                       action='store_false', help='Disable hierarchy word crops')
    parser.set_defaults(include_word_crops=True)
    parser.add_argument('--include_number_crops', '--include-number-crops', dest='include_number_crops',
                       action='store_true', help='Include NumberCrops in retrieval gallery')
    parser.add_argument('--min_assets_per_group', '--min-assets-per-group', dest='min_assets_per_group', type=int, default=1,
                       help='Minimum assets required to include a group in gallery')
    parser.add_argument('--min_positives_per_query', '--min-positives-per-query', dest='min_positives_per_query', type=int, default=1,
                       help='Required positives per query (1 => group size at least 2)')

    parser.add_argument('--target_height', '--target-height', dest='target_height', type=int, default=0,
                       help='Override normalized input height (0 = from training config/default)')
    parser.add_argument('--max_width', '--max-width', dest='max_width', type=int, default=0,
                       help='Override maximum normalized width (0 = from training config/default)')
    parser.add_argument('--patch_multiple', '--patch-multiple', dest='patch_multiple', type=int, default=0,
                       help='Override width snap multiple (0 = from training config/default)')
    parser.add_argument('--width_buckets', '--width-buckets', dest='width_buckets', type=str, default='',
                       help='Override comma-separated width buckets (empty = from training config/default)')

    parser.add_argument('--batch_size', '--batch-size', dest='batch_size', type=int, default=32,
                       help='Per-device eval batch size')
    parser.add_argument('--num_workers', '--num-workers', dest='num_workers', type=int, default=4,
                       help='DataLoader workers')
    parser.add_argument('--device', type=str, default='auto',
                       help='Embedding device (auto/cpu/cuda:0/mps)')
    parser.add_argument('--l2_normalize_embeddings', '--l2-normalize-embeddings', dest='l2_normalize_embeddings',
                       action='store_true', help='L2-normalize embeddings before retrieval (default on)')
    parser.add_argument('--no_l2_normalize_embeddings', '--no-l2-normalize-embeddings', dest='l2_normalize_embeddings',
                       action='store_false', help='Disable embedding L2 normalization')
    parser.set_defaults(l2_normalize_embeddings=True)
    parser.add_argument('--recall_ks', '--recall-ks', dest='recall_ks', type=str, default='1,5,10',
                       help='Comma-separated Recall@K values (e.g. 1,5,10)')
    parser.add_argument('--max_queries', '--max-queries', dest='max_queries', type=int, default=0,
                       help='Randomly sample at most N evaluable queries (0 = all)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (sampling and reproducibility)')

    parser.add_argument('--write_per_query_csv', '--write-per-query-csv', dest='write_per_query_csv',
                       action='store_true', help='Write per-query CSV report (default on)')
    parser.add_argument('--no_write_per_query_csv', '--no-write-per-query-csv', dest='write_per_query_csv',
                       action='store_false', help='Disable per-query CSV output')
    parser.set_defaults(write_per_query_csv=True)


def create_eval_text_hierarchy_vit_parser(add_help: bool = True):
    """Create parser for evaluating a TextHierarchy ViT retrieval encoder."""
    parser = argparse.ArgumentParser(
        description="Evaluate ViT retrieval encoder on legacy TextHierarchy or new patch-parquet dataset",
        add_help=add_help,
    )
    add_eval_text_hierarchy_vit_arguments(parser)
    return parser


def add_faiss_text_hierarchy_search_arguments(parser):
    """Arguments for FAISS-based TextHierarchy similarity search."""
    parser.add_argument('--query_image', '--query-image', dest='query_image', type=str, required=True,
                       help='Query image/crop path for similarity search')
    parser.add_argument('--backbone_dir', '--backbone-dir', dest='backbone_dir', type=str, required=True,
                       help='Path to trained backbone directory used for embedding')
    parser.add_argument('--projection_head_path', '--projection-head-path', dest='projection_head_path', type=str, default='',
                       help='Optional projection head checkpoint (.pt)')
    parser.add_argument('--output_dir', '--output-dir', dest='output_dir', type=str, required=True,
                       help='Directory where search report is written')

    parser.add_argument('--index_path', '--index-path', dest='index_path', type=str, default='',
                       help='Existing FAISS index path (.faiss). If set, loads DB instead of rebuilding.')
    parser.add_argument('--meta_path', '--meta-path', dest='meta_path', type=str, default='',
                       help='Optional explicit metadata sidecar path for existing FAISS DB')
    parser.add_argument('--dataset_dir', '--dataset-dir', dest='dataset_dir', type=str, default='',
                       help='Dataset root to build FAISS DB (legacy TextHierarchy or new patch-parquet)')
    parser.add_argument('--save_index_path', '--save-index-path', dest='save_index_path', type=str, default='',
                       help='Path where newly built FAISS DB is saved (default: <output_dir>/text_hierarchy.faiss)')

    parser.add_argument('--metric', type=str, default='cosine', choices=['cosine', 'l2'],
                       help='FAISS similarity metric')
    parser.add_argument('--top_k', '--top-k', dest='top_k', type=int, default=10,
                       help='Number of nearest neighbors to return')

    parser.add_argument('--include_line_images', '--include-line-images', dest='include_line_images',
                       action='store_true', help='Use line.png assets from TextHierarchy')
    parser.add_argument('--no_include_line_images', '--no-include-line-images', dest='include_line_images',
                       action='store_false', help='Disable line.png assets')
    parser.set_defaults(include_line_images=True)
    parser.add_argument('--include_word_crops', '--include-word-crops', dest='include_word_crops',
                       action='store_true', help='Use hierarchy word crops (word_*.png)')
    parser.add_argument('--no_include_word_crops', '--no-include-word-crops', dest='include_word_crops',
                       action='store_false', help='Disable hierarchy word crops')
    parser.set_defaults(include_word_crops=True)
    parser.add_argument('--include_number_crops', '--include-number-crops', dest='include_number_crops',
                       action='store_true', help='Include NumberCrops assets in FAISS DB')
    parser.add_argument('--min_assets_per_group', '--min-assets-per-group', dest='min_assets_per_group', type=int, default=1,
                       help='Minimum assets required to include a group while building DB')

    parser.add_argument('--config_path', '--config-path', dest='config_path', type=str, default='',
                       help='Optional training_config.json path for normalization defaults')
    parser.add_argument('--target_height', '--target-height', dest='target_height', type=int, default=0,
                       help='Override normalized input height (0 = from config/DB)')
    parser.add_argument('--max_width', '--max-width', dest='max_width', type=int, default=0,
                       help='Override maximum normalized width (0 = from config/DB)')
    parser.add_argument('--patch_multiple', '--patch-multiple', dest='patch_multiple', type=int, default=0,
                       help='Override width snap multiple (0 = from config/DB)')
    parser.add_argument('--width_buckets', '--width-buckets', dest='width_buckets', type=str, default='',
                       help='Override comma-separated width buckets (empty = from config/DB)')

    parser.add_argument('--batch_size', '--batch-size', dest='batch_size', type=int, default=32,
                       help='Per-device embedding batch size while building DB')
    parser.add_argument('--num_workers', '--num-workers', dest='num_workers', type=int, default=4,
                       help='DataLoader workers')
    parser.add_argument('--device', type=str, default='auto',
                       help='Embedding device (auto/cpu/cuda:0/mps)')
    parser.add_argument('--l2_normalize_embeddings', '--l2-normalize-embeddings', dest='l2_normalize_embeddings',
                       action='store_true', help='L2-normalize embeddings (default on)')
    parser.add_argument('--no_l2_normalize_embeddings', '--no-l2-normalize-embeddings', dest='l2_normalize_embeddings',
                       action='store_false', help='Disable embedding L2 normalization')
    parser.set_defaults(l2_normalize_embeddings=True)


def create_faiss_text_hierarchy_search_parser(add_help: bool = True):
    """Create parser for FAISS-based TextHierarchy similarity search."""
    parser = argparse.ArgumentParser(
        description="FAISS similarity search on TextHierarchy/patch-parquet embeddings",
        add_help=add_help,
    )
    add_faiss_text_hierarchy_search_arguments(parser)
    return parser


def add_prepare_donut_ocr_dataset_arguments(parser):
    """Arguments for preparing label-filtered OCR manifests for Donut-style training."""
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='Dataset root containing legacy train/val ocr_targets or canonical train/test/eval meta/lines.*')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for train/val manifests')
    parser.add_argument('--label_id', type=int, default=1,
                       help='Class label ID to keep for legacy ocr_targets format (ignored for line-meta datasets)')
    parser.add_argument('--splits', type=str, default='train,val',
                       help='Comma-separated output splits to process (default: train,val; val auto-falls back to eval)')
    parser.add_argument('--text_field', type=str, default='target_text',
                       help='Preferred text field; auto-falls back to text/text_raw for line-meta datasets')
    parser.add_argument('--normalization', type=str, default='NFC',
                       choices=['NFC', 'NFKC', 'NFD', 'NFKD', 'none'],
                       help='Unicode normalization strategy for output text')
    parser.add_argument('--output_newline_token', type=str, choices=['keep', '<NL>', '\\n'], default='keep',
                       help="How to represent newlines in output text.")
    parser.add_argument('--min_chars', type=int, default=1,
                       help='Drop samples with fewer characters after normalization')
    parser.add_argument('--max_chars', type=int, default=0,
                       help='Truncate samples to max chars (0 = no truncation)')
    parser.add_argument('--max_samples_per_split', type=int, default=0,
                       help='Optional cap per split after filtering (0 = no cap)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed used when max_samples_per_split > 0')

    parser.add_argument('--wrap_task_tokens', dest='wrap_task_tokens', action='store_true',
                       help='Wrap targets with task start/end tokens (enabled by default)')
    parser.add_argument('--no_wrap_task_tokens', dest='wrap_task_tokens', action='store_false',
                       help='Disable task start/end wrapping')
    parser.set_defaults(wrap_task_tokens=True)
    parser.add_argument('--task_start_token', type=str, default='<s_ocr>',
                       help='Task start token for decoder targets')
    parser.add_argument('--task_end_token', type=str, default='</s_ocr>',
                       help='Task end token for decoder targets')

    parser.add_argument('--include_class_token', dest='include_class_token', action='store_true',
                       help='Prefix targets with class token (enabled by default)')
    parser.add_argument('--no_include_class_token', dest='include_class_token', action='store_false',
                       help='Disable class token prefix')
    parser.set_defaults(include_class_token=True)
    parser.add_argument('--class_token', type=str, default='<s_cls1>',
                       help='Class token to prepend when include_class_token is enabled')

    parser.add_argument('--dedupe', dest='dedupe', action='store_true',
                       help='Remove duplicate (image,text) pairs (enabled by default)')
    parser.add_argument('--no_dedupe', dest='dedupe', action='store_false',
                       help='Keep duplicates')
    parser.set_defaults(dedupe=True)


def add_train_donut_ocr_arguments(parser):
    """Arguments for Donut-style OCR model training."""
    parser.add_argument('--train_manifest', type=str, required=True,
                       help='Path to training JSONL manifest')
    parser.add_argument('--val_manifest', type=str, default='',
                       help='Optional validation JSONL manifest')
    parser.add_argument('--val_eval_max_samples', '--val-eval-max-samples', dest='val_eval_max_samples', type=int, default=0,
                       help='Randomly sample at most N validation rows for eval/CER during training (0 = use all)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for checkpoints and final model')
    parser.add_argument('--model_name_or_path', type=str, default='microsoft/trocr-base-stage1',
                       help='VisionEncoderDecoder checkpoint to fine-tune')
    parser.add_argument('--image_processor_path', type=str, default='',
                       help='Optional image processor checkpoint/path override')
    parser.add_argument('--tokenizer_path', type=str, default='openpecha/BoSentencePiece',
                       help='Tokenizer checkpoint/path (default: openpecha/BoSentencePiece; empty string falls back to model tokenizer)')
    parser.add_argument('--train_tokenizer', action='store_true',
                       help='Train a new tokenizer from training targets')
    parser.add_argument('--tokenizer_vocab_size', type=int, default=16000,
                       help='Vocabulary size when train_tokenizer is enabled')
    parser.add_argument('--tokenizer_output_dir', type=str, default='',
                       help='Optional explicit path where tokenizer is saved')
    parser.add_argument('--extra_special_tokens', type=str,
                       default='<NL>,<s_ocr>,</s_ocr>,<s_cls1>',
                       help='Comma-separated special tokens to ensure in tokenizer vocab')
    parser.add_argument('--decoder_start_token', type=str, default='<s_ocr>',
                       help='Token used as decoder start token')
    parser.add_argument('--image_size', type=int, default=384,
                       help='Square resize used by image processor')
    parser.add_argument('--image_preprocess_pipeline', '--image-preprocess-pipeline', dest='image_preprocess_pipeline', type=str,
                       default='none', choices=['none', 'pb', 'bdrc'],
                       help='Optional deterministic image preprocessing before Donut image processor')
    parser.add_argument('--max_target_length', type=int, default=512,
                       help='Maximum target token length for training labels')
    parser.add_argument('--generation_max_length', type=int, default=512,
                       help='Maximum generated token length during eval')
    parser.add_argument('--per_device_train_batch_size', type=int, default=4,
                       help='Train batch size per device')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4,
                       help='Eval batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Number of grad accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--num_train_epochs', type=float, default=8.0,
                       help='Number of training epochs')
    parser.add_argument('--warmup_steps', type=int, default=200,
                       help='Warmup steps')
    parser.add_argument('--logging_steps', type=int, default=20,
                       help='Logging interval (steps)')
    parser.add_argument('--eval_steps', type=int, default=200,
                       help='Evaluation interval (steps)')
    parser.add_argument('--save_steps', type=int, default=200,
                       help='Checkpoint interval (steps)')
    parser.add_argument('--save_total_limit', type=int, default=3,
                       help='Maximum number of checkpoints to keep')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--fp16', action='store_true',
                       help='Enable fp16 training')
    parser.add_argument('--bf16', action='store_true',
                       help='Enable bf16 training')
    parser.add_argument('--metric_newline_token', type=str, choices=['<NL>', '\\n'], default='<NL>',
                       help='Newline token normalization used for CER computation')
    parser.add_argument('--debug_train_trace', '--debug-train-trace', dest='debug_train_trace', action='store_true',
                       help='Enable verbose per-step DONUT training trace (inputs/encoder/logits/top-k decode debug)')
    parser.add_argument('--debug_train_trace_every_steps', '--debug-train-trace-every-steps', dest='debug_train_trace_every_steps', type=int, default=1,
                       help='Log verbose DONUT training trace every N steps when --debug_train_trace is enabled')
    parser.add_argument('--debug_train_trace_topk', '--debug-train-trace-topk', dest='debug_train_trace_topk', type=int, default=5,
                       help='Top-k token candidates to log per timestep in verbose DONUT training trace')
    parser.add_argument('--debug_train_trace_max_positions', '--debug-train-trace-max-positions', dest='debug_train_trace_max_positions', type=int, default=8,
                       help='Number of decoding positions to log in verbose DONUT training trace')
    parser.add_argument('--resume_from_checkpoint', type=str, default='',
                       help='Optional checkpoint path to resume training from')


def add_run_donut_ocr_workflow_arguments(parser):
    """Arguments for end-to-end synthetic generation + OCR prep + Donut training."""
    parser.add_argument('--dataset_name', type=str, default='tibetan-donut-ocr-label1',
                       help='Dataset name used in generate_training_data.py')
    parser.add_argument('--dataset_output_dir', type=str, default='./datasets',
                       help='Base output directory for generated dataset')
    parser.add_argument('--train_samples', type=int, default=2000,
                       help='Number of synthetic train samples')
    parser.add_argument('--val_samples', type=int, default=200,
                       help='Number of synthetic val samples')
    parser.add_argument('--font_path_tibetan', type=str, required=True,
                       help='Path to Tibetan font file')
    parser.add_argument('--font_path_chinese', type=str, required=True,
                       help='Path to Chinese font file')
    parser.add_argument('--augmentation', type=str, choices=['rotate', 'noise', 'none'], default='noise',
                       help='Synthetic augmentation mode')
    parser.add_argument('--target_newline_token', type=str, choices=['\\n', '<NL>'], default='<NL>',
                       help='Target newline token for generated OCR texts')
    parser.add_argument('--prepared_output_dir', type=str, default='',
                       help='Optional explicit output directory for prepared manifests')
    parser.add_argument('--model_output_dir', type=str, default='./models/donut-ocr-label1',
                       help='Output directory for trained OCR model')
    parser.add_argument('--model_name_or_path', type=str, default='microsoft/trocr-base-stage1',
                       help='VisionEncoderDecoder checkpoint to fine-tune')
    parser.add_argument('--tokenizer_path', type=str, default='openpecha/BoSentencePiece',
                       help='Tokenizer checkpoint/path for Donut training step (default: openpecha/BoSentencePiece)')
    parser.add_argument('--train_tokenizer', action='store_true',
                       help='Train a tokenizer from label-1 OCR targets')
    parser.add_argument('--tokenizer_vocab_size', type=int, default=16000,
                       help='Tokenizer vocab size when train_tokenizer is enabled')
    parser.add_argument('--per_device_train_batch_size', type=int, default=4,
                       help='Train batch size per device')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4,
                       help='Eval batch size per device')
    parser.add_argument('--num_train_epochs', type=float, default=8.0,
                       help='Training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--max_target_length', type=int, default=512,
                       help='Maximum target length')
    parser.add_argument('--image_size', type=int, default=384,
                       help='Image size for OCR model input')
    parser.add_argument('--image_preprocess_pipeline', '--image-preprocess-pipeline', dest='image_preprocess_pipeline', type=str,
                       default='none', choices=['none', 'pb', 'bdrc'],
                       help='Optional deterministic image preprocessing before Donut image processor in workflow train step')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--skip_generation', action='store_true',
                       help='Skip synthetic generation and use existing dataset dir')
    parser.add_argument('--skip_prepare', action='store_true',
                       help='Skip OCR manifest preparation')
    parser.add_argument('--skip_train', action='store_true',
                       help='Skip OCR training step')
    parser.add_argument('--lora_augment_path', type=str, default='',
                       help='Optional LoRA path for augmentation during synthetic generation.')
    parser.add_argument('--lora_augment_model_family', type=str, choices=['sdxl', 'sd21'], default='sdxl',
                       help='Diffusion model family for optional LoRA augmentation.')
    parser.add_argument('--lora_augment_base_model_id', type=str,
                       default='stabilityai/stable-diffusion-xl-base-1.0',
                       help='Base model ID for optional LoRA augmentation.')
    parser.add_argument('--lora_augment_controlnet_model_id', type=str,
                       default='diffusers/controlnet-canny-sdxl-1.0',
                       help='ControlNet model ID for optional LoRA augmentation.')
    parser.add_argument('--lora_augment_prompt', type=str, default=DEFAULT_TEXTURE_PROMPT,
                       help='Prompt for optional LoRA augmentation.')
    parser.add_argument('--lora_augment_scale', type=float, default=0.8,
                       help='LoRA scale for optional augmentation.')
    parser.add_argument('--lora_augment_strength', type=float, default=0.2,
                       help='Img2img strength for optional augmentation.')
    parser.add_argument('--lora_augment_steps', type=int, default=28,
                       help='Diffusion steps for optional augmentation.')
    parser.add_argument('--lora_augment_guidance_scale', type=float, default=1.0,
                       help='Guidance scale for optional augmentation.')
    parser.add_argument('--lora_augment_controlnet_scale', type=float, default=2.0,
                       help='ControlNet scale for optional augmentation.')
    parser.add_argument('--lora_augment_seed', type=int, default=None,
                       help='Seed for optional LoRA augmentation.')
    parser.add_argument('--lora_augment_splits', type=str, default='train',
                       help='Comma-separated splits for optional LoRA augmentation.')
    parser.add_argument('--lora_augment_targets', type=str, choices=['images', 'images_and_ocr_crops'],
                       default='images_and_ocr_crops',
                       help='Assets to augment during synthetic generation in workflow mode.')
    parser.add_argument('--lora_augment_canny_low', type=int, default=100,
                       help='Canny low threshold for optional LoRA augmentation.')
    parser.add_argument('--lora_augment_canny_high', type=int, default=200,
                       help='Canny high threshold for optional LoRA augmentation.')


def create_prepare_donut_ocr_dataset_parser(add_help: bool = True):
    parser = argparse.ArgumentParser(
        description="Prepare label-filtered OCR manifests for Donut-style OCR training",
        add_help=add_help,
    )
    add_prepare_donut_ocr_dataset_arguments(parser)
    return parser


def create_train_donut_ocr_parser(add_help: bool = True):
    parser = argparse.ArgumentParser(
        description="Train a Donut-style OCR model (VisionEncoderDecoder) on OCR crops",
        add_help=add_help,
    )
    add_train_donut_ocr_arguments(parser)
    return parser


def create_run_donut_ocr_workflow_parser(add_help: bool = True):
    parser = argparse.ArgumentParser(
        description="Run full label-1 OCR workflow: synthetic data -> manifests -> Donut-style training",
        add_help=add_help,
    )
    add_run_donut_ocr_workflow_arguments(parser)
    return parser
