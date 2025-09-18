import argparse
import multiprocessing
import random
import re
import os
import csv
import time
import traceback
from typing import Tuple, Dict, List, Optional  # Added Optional

import yaml
from pathlib import Path
from collections import OrderedDict
try:
    from ultralytics.data.utils import DATASETS_DIR
except ImportError:
    DATASETS_DIR = "./datasets"  # Fallback if ultralytics not installed
from tibetanDataGenerator.utils.data_loader import TextFactory
from tibetanDataGenerator.data.text_renderer import ImageBuilder
from tibetanDataGenerator.data.augmentation import RotateAugmentation, NoiseAugmentation, \
     AugmentationStrategy
from tibetan_utils.image_utils import BoundingBoxCalculator
from tibetan_utils.io_utils import hash_current_time

# Define a dictionary of augmentation strategies
augmentation_strategies: Dict[str, AugmentationStrategy] = {
    'rotate': RotateAugmentation(),
    'noise': NoiseAugmentation()
}

def _parse_yolo_annotations(file_path: str) -> List[Tuple[int, float, float, float, float]]:
    """
    Parses a YOLO annotation file.
    Each line is expected to be: class_id center_x center_y width height
    Returns a list of tuples (class_id, x_center, y_center, width, height).
    """
    annotations = []
    if not file_path:  # If file_path is None or empty string
        return annotations

    if not os.path.exists(file_path):
        print(f"Warning: Annotation file '{file_path}' not found. No annotations will be loaded from this file.")
        return annotations

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line_strip = line.strip()
                if not line_strip:  # Skip empty lines
                    continue
                parts = line_strip.split()
                if len(parts) == 5:
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        # Basic validation for YOLO coordinates (normalized)
                        if not (0.0 <= x_center <= 1.0 and \
                                0.0 <= y_center <= 1.0 and \
                                0.0 <= width <= 1.0 and \
                                0.0 <= height <= 1.0):
                            # This warning can be made conditional on debug flag if too verbose
                            # print(f"Debug: Annotation values out of [0,1] range in {file_path}, line {i+1}: {line_strip}")
                            pass

                            # Ensure width and height are positive for valid bounding box
                        if width <= 0 or height <= 0:
                            print(
                                f"Warning: Non-positive width/height in annotation file {file_path}, line {i + 1}: {line_strip}. Skipping this annotation.")
                            continue

                        annotations.append((class_id, x_center, y_center, width, height))
                    except ValueError:
                        print(
                            f"Warning: Malformed line (numeric conversion) in annotation file {file_path}, line {i + 1}: {line_strip}")
                else:  # Incorrect number of parts
                    print(
                        f"Warning: Incorrect number of parts in line in annotation file {file_path}, line {i + 1}: {line_strip}")
    except Exception as e:
        print(f"Error reading or parsing annotation file {file_path}: {e}")
    return annotations


def generate_dataset(args: argparse.Namespace, validation: bool = False) -> Dict:
    """
    Generate a dataset for training or validation.

    Args:
        args (argparse.Namespace): Command-line arguments.
        validation (bool): Whether to generate validation dataset. Defaults to False.

    Returns:
        Dict: A dictionary containing dataset information.
    """
    print(f"Starting dataset generation (validation={validation})...")
    start_time = time.time()
    
    dataset_info = _setup_dataset_info(args, validation)
    print(f"Dataset info setup completed. Target samples: {dataset_info['no_samples']}")
    
    label_dict = _create_label_dict(args)
    print(f"Label dictionary created with {len(label_dict)} labels: {list(label_dict.keys())}")
    
    background_images = _load_background_images(dataset_info['background_folder'])
    print(f"Loaded {len(background_images)} background images from {dataset_info['background_folder']}")

    # _prepare_generation_args now gets annotations_file_path from args
    generation_args_tuple = _prepare_generation_args(args, dataset_info, label_dict, background_images)
    print("Generation arguments prepared")

    results = _generate_images_in_parallel(generation_args_tuple, dataset_info['no_samples'])
    
    elapsed = time.time() - start_time
    successful_results = [r for r in results if r[0] and r[1]]  # Filter out failed generations
    print(f"Dataset generation completed in {elapsed:.1f}s. Success rate: {len(successful_results)}/{len(results)}")

    return _create_dataset_dict(str(dataset_info['folder']), label_dict)


def generate_synthetic_image(
        images: List[str],
        label_dict: Dict[str, int],
        folder_with_background: str,
        corpora_tibetan_numbers_path: str,
        corpora_tibetan_text_path: str,
        corpora_chinese_numbers_path: str,
        folder_for_train_data: str,
        debug: bool = True,
        font_path_tibetan: str = 'res/Microsoft Himalaya.ttf',
        font_path_chinese: str = 'res/simkai.ttf',
        single_label: bool = False,
        image_width: int = 1024,
        image_height: int = 361,
        augmentation: str = "noise",
        annotations_file_path: Optional[str] = None
) -> Tuple[str, str]:
    """
    Generate a synthetic image with improved error handling and resource management.
    """
    try:
        return _generate_synthetic_image_impl(
            images, label_dict, folder_with_background,
            corpora_tibetan_numbers_path, corpora_tibetan_text_path, corpora_chinese_numbers_path,
            folder_for_train_data, debug, font_path_tibetan, font_path_chinese,
            single_label, image_width, image_height, augmentation, annotations_file_path
        )
    except Exception as e:
        # Log the error and return empty paths to indicate failure
        print(f"Error in generate_synthetic_image: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return "", ""


def _generate_synthetic_image_impl(
        images: List[str],
        label_dict: Dict[str, int],
        folder_with_background: str,
        corpora_tibetan_numbers_path: str,
        corpora_tibetan_text_path: str,
        corpora_chinese_numbers_path: str,
        folder_for_train_data: str,
        debug: bool = True,
        font_path_tibetan: str = 'res/Microsoft Himalaya.ttf',
        font_path_chinese: str = 'res/simkai.ttf',
        single_label: bool = False,
        image_width: int = 1024,
        image_height: int = 361,
        augmentation: str = "noise",
        annotations_file_path: Optional[str] = None
) -> Tuple[str, str]:
    # Font configuration
    BORDER_OFFSET_RATIO = 0.05
    font_size_class1 = None
    font_size_0_2 = None

    ctr = hash_current_time()

    border_offset_x = int(BORDER_OFFSET_RATIO * image_width)
    border_offset_y = int(BORDER_OFFSET_RATIO * image_height)

    image_path_bg = _select_random_background(folder_with_background, images)
    # Determine which font to use based on annotation class
    current_font_path = font_path_tibetan  # Default to Tibetan font
    if annotations_file_path:
        parsed_annotations = _parse_yolo_annotations(annotations_file_path)
        if parsed_annotations and parsed_annotations[0][0] == 2:  # Check first annotation's class_id
            current_font_path = font_path_chinese

    builder = _setup_image_builder(image_path_bg, image_width, image_height, current_font_path, 24)  # Default font size only used if no annotations

    bbox_str_list = []  # Collect bounding box strings for all text instances
    tibetan_number_match = None  # Store the matching number if we find a Tibetan number file

    # ---- Start: Draw bounding boxes from YOLO annotation file ----
    if annotations_file_path:
        parsed_annotations = _parse_yolo_annotations(annotations_file_path)
        for ann_class_id, norm_cx, norm_cy, norm_w, norm_h in parsed_annotations:
            # Convert YOLO normalized coordinates to pixel coordinates for drawing
            x_center_pixel = norm_cx * image_width
            y_center_pixel = norm_cy * image_height
            pixel_w = norm_w * image_width
            pixel_h = norm_h * image_height

            # Calculate top-left corner for add_bounding_box
            tl_x = x_center_pixel - (pixel_w / 2)
            tl_y = y_center_pixel - (pixel_h / 2)

            draw_tl_pos = (int(round(tl_x)), int(round(tl_y)))
            draw_box_size = (int(round(pixel_w)), int(round(pixel_h)))

            # Draw only if width and height are positive
            if draw_box_size[0] > 0 and draw_box_size[1] > 0:
                # Select the text corpus based on ann_class_id
                if ann_class_id == 0:  # Tibetan numbers
                    text, file_name_from_corpus = _generate_text(corpora_tibetan_numbers_path)
                    # Calculate font size for class 1 with bounding box constraints
                    text_for_sizing = text if text else "default"
                    max_font = BoundingBoxCalculator.find_max_font(
                        text_for_sizing,
                        (draw_box_size[0], draw_box_size[1]),
                        font_path_tibetan,
                        max_size=100,
                        debug=debug
                    )
                    font_size_class1 = random.randint(24, max(24, min(100, max_font)))
                    
                    # Set sibling classes to be ±1-2 sizes different
                    delta = random.choice([-2, -1, 1, 2])
                    font_size_0_2 = max(1, min(100, font_size_class1 + delta))
                    
                    builder.set_font(font_path_tibetan, font_size_class1)
                    # Extract the number part from the Tibetan filename
                    try:
                        tibetan_number_match = re.search(r'tib_no_(\d+)', file_name_from_corpus)
                        if tibetan_number_match:
                            tibetan_number_match = tibetan_number_match.group(1)
                    except:
                        tibetan_number_match = None
                elif ann_class_id == 1:  # Tibetan text
                    text, file_name_from_corpus = _generate_text(corpora_tibetan_text_path)
                    # Calculate font size for class 1 with bounding box constraints
                    text_for_sizing = text if text else "default"
                    max_font = BoundingBoxCalculator.find_max_font(
                        text_for_sizing,
                        (draw_box_size[0], draw_box_size[1]),
                        font_path_tibetan,
                        max_size=100,
                        debug=debug
                    )
                    font_size_class1 = random.randint(24, max(24, min(100, max_font)))
                    builder.set_font(font_path_tibetan, font_size_class1)
                elif ann_class_id == 2:  # Chinese numbers
                    # Use the same number as Tibetan if available
                    chinese_number = f"chi_no_{tibetan_number_match}" if tibetan_number_match else None
                    text, file_name_from_corpus = _generate_text(corpora_chinese_numbers_path, chinese_number)
                    builder.set_font(font_path_chinese, font_size_0_2)
                else:
                    if debug:
                        print(f"Debug: Unknown ann_class_id {ann_class_id}. Skipping this annotation box.")
                    continue

                # Ensure the text fits within the bounding box
                # Calculate actual text dimensions and centered position
                actual_text_box_size = BoundingBoxCalculator.fit(
                    text,
                    draw_box_size,
                    font_size=font_size_class1 if ann_class_id == 1 else font_size_0_2,
                    font_path=current_font_path,
                    debug=debug
                )
                # Calculate random offset based on class ID
                def get_offset(box_dim, percentage):
                    max_offset = box_dim * percentage / 100
                    return random.uniform(-max_offset, max_offset)
                
                # Apply different variation based on class ID
                if ann_class_id in [0, 2]:  # Tibetan and Chinese numbers
                    x_offset = get_offset(draw_box_size[0], 10)
                    y_offset = get_offset(draw_box_size[1], 10)
                else:  # Tibetan text (class 1)
                    x_offset = get_offset(draw_box_size[0], 10)
                    y_offset = get_offset(draw_box_size[1], 10)
                
                # Calculate centered position with random offset
                base_x = draw_tl_pos[0] + (draw_box_size[0] - actual_text_box_size[0]) // 2
                base_y = draw_tl_pos[1] + (draw_box_size[1] - actual_text_box_size[1]) // 2
                
                # Apply offsets and clamp to stay within bounding box
                text_tl_x = int(base_x + x_offset)
                text_tl_y = int(base_y + y_offset)
                
                # Ensure text stays within bounding box
                text_tl_x = max(draw_tl_pos[0], min(text_tl_x, draw_tl_pos[0] + draw_box_size[0] - actual_text_box_size[0]))
                text_tl_y = max(draw_tl_pos[1], min(text_tl_y, draw_tl_pos[1] + draw_box_size[1] - actual_text_box_size[1]))
                text_render_top_left_pos = (text_tl_x, text_tl_y)
                yolo_box_center_pos = (int(round(x_center_pixel)), int(round(y_center_pixel)))

                # Apply rotation for Tibetan numbers (class 0)
                rotation_angle = 90 if ann_class_id == 0 else 0
                builder.add_text(text, text_render_top_left_pos, actual_text_box_size, rotation=rotation_angle)
                # Get the base filename without extension
                label_key = os.path.splitext(file_name_from_corpus)[0]

                # For Tibetan numbers (class 0), ensure we use the tib_no_ prefix
                if ann_class_id == 0:
                    if not label_key.startswith('tib_no_'):
                        # Extract the number from the filename if it exists
                        try:
                            num_part = re.search(r'\d+', label_key).group()
                            label_key = f'tib_no_{num_part.zfill(4)}'  # Format as tib_no_0001
                        except AttributeError:
                            label_key = 'tib_no_0001'  # Default fallback

                # For ann_class_id 0, always use 0 as the label_id
                # For other classes, get label ID from dictionary or use class ID as fallback
                if ann_class_id == 0:
                    label_id = 0
                else:
                    label_id = label_dict.get(label_key, ann_class_id)
                    if label_key not in label_dict and debug:
                        print(f"Debug: Label '{label_key}' not found in label_dict. Using class_id {ann_class_id}")

                bbox_str = _create_bbox_string(
                    label_id,
                    yolo_box_center_pos,
                    actual_text_box_size,
                    image_width,
                    image_height
                )
                bbox_str_list.append(bbox_str)

                if debug:
                    builder.add_bounding_box(text_render_top_left_pos, actual_text_box_size, color=(0, 255, 0))  # Green
                    builder.add_bounding_box(draw_tl_pos, draw_box_size, color=(255, 0, 0))  # Red

            else:
                if debug:
                    print(
                        f"Debug: Skipping drawing annotation box from file (class {ann_class_id}) due to non-positive dimensions: size {draw_box_size}")

    if augmentation.lower() != 'none' and augmentation.lower() in augmentation_strategies:
        _apply_augmentation(builder, augmentation)
    elif augmentation.lower() != 'none':
        print(f"Warning: Augmentation strategy '{augmentation}' not found. Skipping augmentation.")

    image_filename_saved = f"{ctr}.png"
    image_full_path = os.path.join(folder_for_train_data, 'images', image_filename_saved)
    os.makedirs(os.path.dirname(image_full_path), exist_ok=True)
    builder.save(image_full_path)

    labels_dir = os.path.join(folder_for_train_data, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    label_filename_saved = f"{ctr}.txt"
    label_full_path = os.path.join(labels_dir, label_filename_saved)
    with open(label_full_path, 'w', encoding='utf-8') as f:
        f.writelines(bbox_str_list)  # Write all bounding box strings into the file

    if debug:
        print(f"Generated sample: {image_full_path}")
        print(f"Label file: {label_full_path}")
        print(f"Bounding boxes (YOLO format for synthetic text):\n{''.join(bbox_str_list).strip()}")

    return image_full_path, label_full_path


def _select_random_background(folder: str, images: List[str]) -> str:
    if not images:
        raise ValueError(f"No images found in background folder: {folder}. Cannot select a random background.")
    return os.path.join(folder, random.choice(images))


def _setup_image_builder(image_path_bg: str, image_width: int, image_height: int,
                        font_path: str, font_size: int) -> ImageBuilder:
    builder = ImageBuilder(image_size=(image_width, image_height))
    try:
        if image_path_bg and os.path.exists(image_path_bg):
            builder.set_background(image_path_bg)
        else:
            if image_path_bg:
                print(f"Warning: Background image {image_path_bg} not found. Using default white background.")
    except FileNotFoundError:
        print(f"Warning: Background image {image_path_bg} not found during set_background. Using default white background.")
    except Exception as e:
        print(f"Error setting background {image_path_bg}: {e}. Using default white background.")

    # Font will be set separately during text rendering
    return builder


def _generate_text(folder_with_corpora: str, matching_number: str = None) -> Tuple[str, str]:
    text_generator = TextFactory.create_text_source("corpus", folder_with_corpora)
    if matching_number:
        # If a matching number is specified, try to find the exact file
        matching_file = f"{matching_number}.txt"
        file_path = os.path.join(folder_with_corpora, matching_file)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            return text, matching_file
    # Fall back to random selection if no matching number or file not found
    return text_generator.generate_text()


def _calculate_text_layout(
        text: str,
        image_width: int,
        image_height: int,
        border_offset_x: int,
        border_offset_y: int,
        font_path: str,
        font_size: int
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    min_text_box_width = font_size * 2
    min_text_box_height = int(font_size * 1.2)

    max_width_for_text_area = image_width - 2 * border_offset_x
    max_height_for_text_area = image_height - 2 * border_offset_y

    if max_width_for_text_area < min_text_box_width or max_height_for_text_area < min_text_box_height:
        # Fallback: try to use at least minimal dimensions if text is very short.
        # This might not be ideal if text is truly too large for the area.
        # A more robust solution might involve text wrapping or scaling, but that's complex.
        print(
            f"Warning: Text area ({max_width_for_text_area}x{max_height_for_text_area}) might be too small for text. Attempting to fit.")
        max_width_for_text_area = max(max_width_for_text_area, min_text_box_width)
        max_height_for_text_area = max(max_height_for_text_area, min_text_box_height)

    conceptual_box_w = random.randint(min_text_box_width, max_width_for_text_area)
    conceptual_box_h = random.randint(min_text_box_height, max_height_for_text_area)
    max_placement_box = (conceptual_box_w, conceptual_box_h)

    actual_text_box_size = BoundingBoxCalculator.fit(text, max_placement_box, font_size=font_size, font_path=font_path, debug=False)
    actual_w, actual_h = actual_text_box_size

    if actual_w <= 0 or actual_h <= 0:
        print(
            f"Warning: BoundingBoxCalculator.fit returned non-positive dimensions ({actual_w}x{actual_h}) for text: '{text[:50]}...'. Defaulting to minimal.")
        actual_w = max(actual_w, font_size // 2 if text else 1)
        actual_h = max(actual_h, font_size // 2 if text else 1)
        actual_text_box_size = (actual_w, actual_h)

    pos_x_upper_bound = image_width - border_offset_x - actual_w
    pos_y_upper_bound = image_height - border_offset_y - actual_h

    # Ensure random range is valid: lower_bound <= upper_bound
    # If upper bound is less than lower, it means the box is too large.
    # We should place it at the border_offset in such cases.
    tl_pos_x = random.randint(border_offset_x, max(border_offset_x,
                                                   pos_x_upper_bound)) if pos_x_upper_bound >= border_offset_x else border_offset_x
    tl_pos_y = random.randint(border_offset_y, max(border_offset_y,
                                                   pos_y_upper_bound)) if pos_y_upper_bound >= border_offset_y else border_offset_y

    text_render_top_left_pos = (tl_pos_x, tl_pos_y)

    center_x = tl_pos_x + actual_w // 2
    center_y = tl_pos_y + actual_h // 2
    yolo_box_center_pos = (center_x, center_y)

    return text_render_top_left_pos, yolo_box_center_pos, actual_text_box_size


def _apply_augmentation(builder: ImageBuilder, augmentation_name: str):
    augmentation_strategy = augmentation_strategies[augmentation_name.lower()]
    builder.apply_augmentation(augmentation_strategy)


def _save_image_and_label(
        builder: ImageBuilder,
        text_content: str,
        ctr: str,
        folder_for_train_data: str,
        label_dict: Dict[str, int],
        single_label: bool,
        file_name_from_corpus: str,
        yolo_box_center_pos: Tuple[int, int],
        actual_text_box_size: Tuple[int, int],
        image_width: int,
        image_height: int,
        debug: bool
) -> Tuple[str, str]:
    label_str = next(iter(label_dict.keys())) if single_label else os.path.splitext(file_name_from_corpus)[0]
    if label_str not in label_dict:
        print(
            f"Warning: Label '{label_str}' from corpus file '{file_name_from_corpus}' not found in label_dict. Defaulting to first available label.")
        if not label_dict:
            raise ValueError("Label dictionary is empty. Cannot determine a label.")
        label_str = next(iter(label_dict.keys()))
    label_id = label_dict[label_str]

    image_base_filename = f"{label_str}_{ctr}.png"
    image_full_path = os.path.join(folder_for_train_data, 'images', image_base_filename)
    builder.save(image_full_path)

    bbox_str = _create_bbox_string(
        label_id, yolo_box_center_pos, actual_text_box_size, image_width, image_height
    )

    labels_dir = os.path.join(folder_for_train_data, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    label_base_filename = f"{label_str}_{ctr}.txt"
    label_full_path = os.path.join(labels_dir, label_base_filename)
    with open(label_full_path, 'w', encoding='utf-8') as f:
        f.write(bbox_str)

    if debug:
        print(f"Generated sample: {image_full_path}")
        print(f"Label file: {label_full_path}")
        print(f"Bounding box (YOLO format for synthetic text):\n{bbox_str.strip()}")

    return image_full_path, label_full_path


def _create_bbox_string(
        label_id: int,
        box_center_xy: Tuple[int, int],
        box_wh: Tuple[int, int],
        image_width: int = 1024,
        image_height: int = 361
) -> str:
    center_x, center_y = box_center_xy
    box_w, box_h = box_wh

    if image_width == 0: raise ValueError("image_width cannot be zero.")
    if image_height == 0: raise ValueError("image_height cannot be zero.")

    norm_center_x = max(0.0, min(1.0, center_x / image_width))
    norm_center_y = max(0.0, min(1.0, center_y / image_height))
    norm_w = max(0.0, min(1.0, box_w / image_width))
    norm_h = max(0.0, min(1.0, box_h / image_height))

    return f"{label_id} {norm_center_x:.6f} {norm_center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n"


def _fill_label_dict(folder_path: str) -> Dict[str, int]:
    label_dict = OrderedDict()
    label_id_counter = 0

    if not os.path.isdir(folder_path):
        print(f"Warning: Corpora folder '{folder_path}' not found. Returning empty label dict.")
        return label_dict

    # Get all .txt files and sort them numerically by their suffix
    files = [f for f in os.listdir(folder_path) if f.endswith(".txt") and f.startswith("tib_no_")]

    try:
        # Sort files by their numeric suffix (tib_no_0001.txt -> 1)
        sorted_files = sorted(
            files,
            key=lambda x: int(x.split("_")[-1].split(".")[0])
        )
    except (ValueError, IndexError):
        print("Warning: Could not sort corpus files numerically. Using simple alphabetical sort.")
        sorted_files = sorted(files)

    for filename in sorted_files:
        label_name = os.path.splitext(filename)[0]  # Gets 'tib_no_0001' from 'tib_no_0001.txt'
        if label_name not in label_dict:
            label_dict[label_name] = label_id_counter
            label_id_counter += 1

    if not label_dict:
        print(f"Warning: No valid .txt files found in corpora folder '{folder_path}'. Label dictionary is empty.")
    return label_dict


def _setup_dataset_info(args: argparse.Namespace, validation: bool) -> Dict:
    base_output_folder = Path(args.dataset_name)

    if validation:
        folder_path = base_output_folder / 'val'
        num_samples = args.val_samples
        bg_folder = args.background_val
    else:
        folder_path = base_output_folder / 'train'
        num_samples = args.train_samples
        bg_folder = args.background_train

    os.makedirs(folder_path / 'images', exist_ok=True)
    os.makedirs(folder_path / 'labels', exist_ok=True)

    return {
        'background_folder': bg_folder,
        'folder': folder_path,
        'no_samples': num_samples
    }


def _read_labels_from_csv(csv_path: str) -> Dict[str, int]:
    """
    Read label names from a CSV file.
    The CSV file should have columns 'yolo_label' and 'label_name'.
    Returns a dictionary mapping label names to their corresponding class IDs.
    """
    label_dict = OrderedDict()
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'yolo_label' in row and 'label_name' in row:
                    class_id = int(row['yolo_label'])
                    label_name = row['label_name']
                    label_dict[label_name] = class_id
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
    
    if not label_dict:
        print(f"Warning: No valid labels found in CSV file '{csv_path}'. Label dictionary is empty.")
    
    return label_dict

def _create_label_dict(args: argparse.Namespace) -> Dict[str, int]:
    if args.single_label:
        return {'tibetan': 0}
    else:
        # Check if annotations_file_path is provided and has a corresponding CSV file
        if args.annotations_file_path and os.path.exists(args.annotations_file_path):
            # Try to find the corresponding CSV file
            csv_path = args.annotations_file_path.replace('.txt', '.csv')
            if os.path.exists(csv_path):
                return _read_labels_from_csv(csv_path)
        
        # Fallback to the original method if CSV doesn't exist
        return _fill_label_dict(args.corpora_tibetan_numbers_path)


def _load_background_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        print(f"Warning: Background folder '{folder}' not found. No background images will be loaded.")
        return []
    return [file for file in os.listdir(folder) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]


def _prepare_generation_args(args: argparse.Namespace, dataset_info: Dict, label_dict: Dict,
                             images_bg_list: List[str]) -> Tuple:
    """Prepare arguments for each call to generate_synthetic_image."""
    return (
        images_bg_list,
        label_dict,
        dataset_info['background_folder'],
        args.corpora_tibetan_numbers_path,
        args.corpora_tibetan_text_path,
        args.corpora_chinese_numbers_path,
        dataset_info['folder'],
        args.debug,
        args.font_path_tibetan,
        args.font_path_chinese,
        args.single_label,
        args.image_width,
        args.image_height,
        args.augmentation,
        args.annotations_file_path
    )


def _generate_images_in_parallel(generation_args_tuple: Tuple, no_samples: int) -> List:
    if no_samples <= 0:
        return []
    
    list_of_generation_args = [generation_args_tuple] * no_samples
    # Ensure os.cpu_count() returns a valid number or default to 1
    num_cpus = os.cpu_count()
    # Reduce parallel processes to avoid resource conflicts
    max_parallel_calls = min((num_cpus // 2) if num_cpus and num_cpus > 2 else 1, no_samples, 4)
    
    if max_parallel_calls == 0:
        max_parallel_calls = 1  # Ensure at least one process
    
    print(f"Generating {no_samples} images using {max_parallel_calls} parallel processes...")
    
    results = []
    pool = None
    
    try:
        # Use spawn method to avoid potential issues with fork on some systems
        ctx = multiprocessing.get_context('spawn')
        pool = ctx.Pool(processes=max_parallel_calls)
        
        # Add timeout and progress tracking
        import time
        start_time = time.time()
        timeout_seconds = 300  # 5 minutes timeout
        
        # Use starmap_async for better control
        async_result = pool.starmap_async(generate_synthetic_image, list_of_generation_args)
        
        # Wait with timeout and progress updates
        while not async_result.ready():
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                print(f"Timeout after {timeout_seconds} seconds. Terminating processes...")
                pool.terminate()
                pool.join()
                raise TimeoutError(f"Image generation timed out after {timeout_seconds} seconds")
            
            # Show progress every 10 seconds
            if int(elapsed) % 10 == 0 and elapsed > 0:
                print(f"Still generating... ({elapsed:.0f}s elapsed)")
            
            time.sleep(1)
        
        results = async_result.get()
        elapsed = time.time() - start_time
        print(f"Successfully generated {len(results)} images in {elapsed:.1f} seconds")
        
    except Exception as e:
        print(f"Error during parallel image generation: {e}")
        if pool:
            try:
                pool.terminate()  # Forcefully terminate worker processes
                pool.join(timeout=10)  # Wait max 10 seconds for cleanup
            except Exception as cleanup_error:
                print(f"Error during pool cleanup: {cleanup_error}")
        
        # Fallback to sequential processing
        print("Falling back to sequential processing...")
        results = _generate_images_sequentially(generation_args_tuple, no_samples)
        
    finally:
        if pool:
            try:
                pool.close()
                pool.join()
            except Exception:
                pass  # Ignore cleanup errors
    
    return results


def _generate_images_sequentially(generation_args_tuple: Tuple, no_samples: int) -> List:
    """Fallback sequential image generation when parallel processing fails."""
    print(f"Generating {no_samples} images sequentially...")
    results = []
    start_time = time.time()
    
    for i in range(no_samples):
        try:
            if i % 10 == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (no_samples - i) / rate if rate > 0 else 0
                print(f"Generated {i}/{no_samples} images... ({rate:.1f} img/s, ETA: {eta:.0f}s)")
            
            img_start = time.time()
            result = generate_synthetic_image(*generation_args_tuple)
            img_time = time.time() - img_start
            
            if result[0] and result[1]:  # Check if generation was successful
                results.append(result)
            else:
                print(f"Warning: Image {i+1} generation failed (took {img_time:.2f}s)")
                
        except Exception as e:
            print(f"Error generating image {i+1}: {e}")
            if generation_args_tuple[7]:  # debug flag
                traceback.print_exc()
            continue
    
    elapsed = time.time() - start_time
    success_rate = len(results) / no_samples * 100 if no_samples > 0 else 0
    print(f"Sequential generation completed: {len(results)}/{no_samples} images ({success_rate:.1f}% success) in {elapsed:.1f}s")
    return results


def _create_dataset_dict(output_folder_str: str, label_dict: Dict[str, int]) -> OrderedDict:
    # Create a mapping from class IDs to label names
    # If the label_dict contains entries like {'tibetan_no': 0, 'text_body': 1, 'chinese_no': 2},
    # then class_names will be {0: 'tibetan_no', 1: 'text_body', 2: 'chinese_no'}
    class_names = {}
    
    # First, create a reverse mapping from class IDs to label names
    for label_name, class_id in label_dict.items():
        class_names[class_id] = label_name
    
    # Ensure we have entries for class IDs 0, 1, and 2 if they're not in the dictionary
    if 0 not in class_names:
        class_names[0] = 'tibetan_no'
    if 1 not in class_names:
        class_names[1] = 'text_body'
    if 2 not in class_names:
        class_names[2] = 'chinese_no'
    
    dataset_name_part = Path(output_folder_str).parent.name
    split_name = Path(output_folder_str).name

    return OrderedDict([
        ('path', f"../{dataset_name_part}"),
        (split_name, f'{split_name}/images'),
        ('nc', len(class_names)),
        ('names', class_names)
    ])
