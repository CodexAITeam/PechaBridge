import re
from collections import OrderedDict
from pathlib import Path

from ultralytics.cfg import get_cfg
from ultralytics.data.utils import DATASETS_DIR

from PIL import Image, ImageDraw, ImageFont
from yolo_utils import prepare_bbox_string


import io
import os
import random
import yaml
import numpy as np
from tqdm import tqdm
from utils import hash_current_time
import argparse
import multiprocessing


def _parse_csv_items(spec):
    return [item.strip() for item in str(spec).split(",") if item.strip()]


def _has_any_images(folder: Path) -> bool:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return any(p.is_file() and p.suffix.lower() in exts for p in folder.rglob("*"))


def _run_lora_augmentation_on_dir(input_dir: Path, args, seed_offset: int = 0):
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
        output_dir=str(input_dir),
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


def _augment_generated_dataset(dataset_root: Path, args):
    if not str(args.lora_augment_path).strip():
        return []

    splits = _parse_csv_items(args.lora_augment_splits)
    if not splits:
        splits = ["train"]

    reports = []
    for idx, split in enumerate(splits):
        folder = dataset_root / split / "images"
        print(f"LoRA augmentation: split={split} target=images path={folder}")
        rep = _run_lora_augmentation_on_dir(folder, args, seed_offset=idx * 1000)
        rep["split"] = split
        rep["target"] = "images"
        reports.append(rep)
    return reports


def ensure_directory_exists(file_path):
    # Extract the directory part of the file_path
    directory = os.path.dirname(file_path)

    # Check if the directory exists
    if not os.path.exists(directory):
        # If the directory does not exist, create it
        os.makedirs(directory, exist_ok=True)


def calculate_wrapped_text_bounding_box(text, box_size, font_path='res/Microsoft Himalaya.ttf', font_size=24):
    """
    Calculate the true bounding box size for the specified text when it is wrapped and terminated to fit a given box size.

    :param text: Text to be measured.
    :param box_size: Tuple (width, height) specifying the size of the box to fit the text.
    :param font_path: Path to the font file.
    :param font_size: Size of the font.
    :return: Tuple (width, height) representing the actual bounding box size of the wrapped and terminated text.
    """
    # Create a dummy image to get a drawing context
    dummy_image = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy_image)

    # Define the font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
        print("Warning: Default font used, may not accurately measure text.")

    box_w, box_h = box_size
    actual_text_width, actual_text_height = 0, 0
    y_offset = 0

    # Process each line
    for line in text.split('\n'):
        while line:
            # Find the breakpoint for wrapping
            for i in range(len(line)):
                if draw.textlength(line[:i + 1], font=font) > box_w:
                    break
            else:
                i = len(line)

            # Add the line to wrapped text
            wrapped_line = line[:i]

            left, top, right, bottom = font.getbbox(wrapped_line)
            line_width, line_height = right - left, bottom - top

            actual_text_width = max(actual_text_width, line_width)
            y_offset += line_height

            # Check if the next line exceeds the box height
            if y_offset > box_h:
                y_offset -= line_height  # Remove the last line's height if it exceeds
                break

            line = line[i:]

        if y_offset > box_h:
            break

    return actual_text_width, y_offset


def embed_text_in_box_with_limit(image, text, box_position, box_size, font_path, font_size=24):
    """
    Embed text within a specified rectangular box on an image, terminating the text if it surpasses the bounding box.

    :param image: PIL Image object to embed text on.
    :param text: Text to be embedded.
    :param box_position: Tuple (x, y) specifying the top left corner of the box.
    :param box_size: Tuple (width, height) specifying the size of the box.
    :param font_path: Path to the font file.
    :param font_size: Size of the font.
    :return: Image object with text embedded within the box.
    """
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
        print("Warning: Default font used, may not display text as expected.")

    box_x, box_y = box_position
    box_w, box_h = box_size
    max_y = box_y + box_h
    wrapped_text = []
    for line in text.split('\n'):
        while line:
            for i in range(len(line)):
                if draw.textlength(line[:i + 1], font=font) > box_w:
                    break
            else:
                i = len(line)

            wrapped_text.append(line[:i])
            line = line[i:]

    y_offset = 0
    for line in wrapped_text:
        left, top, right, bottom = font.getbbox(line)
        line_height = bottom - top
        if box_y + y_offset + line_height > max_y:
            break  # Stop if the next line exceeds the box height
        draw.text((box_x, box_y + y_offset), line, font=font, fill=(0, 0, 0))
        y_offset += line_height

    return image


def generate_sample(images, label_dict, folder_with_background, folder_with_corpoare, folder_for_train_data, debug = False, font_path ='res/Microsoft Himalaya.ttf', no_cols_max=3, single_label = False, image_size=1024):
    ctr = hash_current_time()

    image_id = random.randint(0, len(images)-1)
    image_path = folder_with_background + "/" + images[image_id]
    with open(image_path, "rb") as image_file:
        magazine_image = Image.open(io.BytesIO(image_file.read()))

    magazine_image = magazine_image.resize((image_size,image_size), Image.Resampling.LANCZOS)

    dx, dy = magazine_image.size
    no_cols = random.randint(1, no_cols_max)
    dx_multicol = int(dx / no_cols)

    max_box_size_w = random.randint(100, dx_multicol-5)
    max_box_size = (max_box_size_w, 400)    # maximum size of text box (text will be wrapped if longer)
    box_pos_x = random.randint(0, dx_multicol - max_box_size[0])
    box_pos_y = random.randint(0, dy - max_box_size[1])

    bbox_str = ""

    if(debug):
        print(f"\n\n[{ctr}] image size: ({dx},{dy})")

    for i in range(no_cols):
        #text_to_embed = generate_lorem_like_tibetan_text(500)
        text_to_embed, filename = read_random_tibetan_file(folder_with_corpoare)


        label = next(iter(label_dict.keys()))
        label_id = label_dict[label]
        if(not single_label):
            label = os.path.splitext(filename)[0]
            label_id = label_dict[label]

        # position of bounding box
        if(i>0): # shift to the right for columns 2+
            box_pos_x += max_box_size_w + i*random.randint(5, 30)
        box_position = (box_pos_x, box_pos_y)  # position of text box
        
        if(debug):
            print(f"position of box in col {i}: ({box_position[0]},{box_position[1]})")
            print(f" >> max size ({max_box_size[0]},{max_box_size[1]})")

        bbox = calculate_wrapped_text_bounding_box(text_to_embed, max_box_size, font_path)
        magazine_image = embed_text_in_box_with_limit(magazine_image, text_to_embed, box_position, max_box_size, font_path)
        bbox = np.array(bbox)
        x = box_position[0]
        y = box_position[1]
        w = bbox[0]
        h = bbox[1]

        bbox_str += prepare_bbox_string(label_id,x,y,h,w,dx,dy) + "\n"

    pImg = folder_for_train_data + "/images/" + label + "_" + str(ctr) + ".png"
    pBB = folder_for_train_data + "/labels/" + label + "_" + str(ctr) + ".txt" 

    ensure_directory_exists(pImg)
    ensure_directory_exists(pBB)

    # Save the image
    magazine_image.save(pImg)

    # Open the file in write mode
    with open(pBB, 'w') as file:
        # Write the string to the file
        file.write(bbox_str)


def fill_label_dict(folder_path):
    label_dict = {}
    label_id = 0

    # Get all txt files
    files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

    # Sort files based on the numeric part
    sorted_files = sorted(files, key=lambda x: int(re.findall(r'\d+', x)[-1]))

    for filename in sorted_files:
        label = os.path.splitext(filename)[0]
        if label not in label_dict:
            label_dict[label] = label_id
            label_id += 1

    return label_dict


def generate_data(args, validation = False):
    folder_with_background = args.background_train
    folder_for_train_data = f'{args.dataset_name}/train/'
    no_samples = args.train_samples
    if(validation):
        folder_with_background = args.background_val
        folder_for_train_data = f'{args.dataset_name}/val/'
        no_samples = args.val_samples

    label_dict = {'tibetan': 0}
    if not args.single_label:
        label_dict = fill_label_dict(args.corpora_folder)

    # Load background images
    images = [file for file in os.listdir(folder_with_background) if file.lower().endswith(('.jpg', '.png'))]

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    args = (images, label_dict, folder_with_background, args.corpora_folder, folder_for_train_data, False, args.font_path, args.no_cols, args.single_label, args.image_size)

    number_of_calls = no_samples
    max_parallel_calls = os.cpu_count()

    # Create a pool of workers, limited to no. cpu parallel processes for generation of training data
    with multiprocessing.Pool(max_parallel_calls) as pool:
        results = pool.starmap(generate_sample, [args] * number_of_calls)

    label_dict_swap = {v: k for k, v in label_dict.items()} # swap key & value of dictionary for ultralytics yolo file format

    dataset_dict = OrderedDict([
        ('path', f"../{folder_for_train_data}"),
        ('train', 'train/images'),
        ('val', 'val/images'),
        ('nc', len(label_dict_swap)),  # Add the number of classes
        ('names', label_dict_swap)
    ])
    return dataset_dict


def main():
    parser = argparse.ArgumentParser(description="Generate YOLO dataset for Tibetan text detection")

    parser.add_argument('--background_train', type=str, default='./data/background_images_train/',
                        help='Folder with background images for training')
    parser.add_argument('--background_val', type=str, default='./data/background_images_val/',
                        help='Folder with background images for validation')
    parser.add_argument('--dataset_name', type=str, default='yolo_tibetan/',
                        help='Folder for the generated YOLO dataset')
    parser.add_argument('--corpora_folder', type=str, default='./data/corpora/UVA Tibetan Spoken Corpus/',
                        help='Folder with Tibetan tibetan numbers corpora')
    parser.add_argument('--train_samples', type=int, default=1000,
                        help='Number of training samples to generate')
    parser.add_argument('--val_samples', type=int, default=100,
                        help='Number of validation samples to generate')
    parser.add_argument('--no_cols', type=int, default=1,
                        help='Number of text columns to generate [1....5]')
    parser.add_argument('--font_path', type=str, default='ext/Microsoft Himalaya.ttf',
                        help='Path to a font file that supports Tibetan characters')
    parser.add_argument('--single_label', action='store_true',
                        help='Use a single label "tibetan" for all files instead of using filenames as labels')
    parser.add_argument('--image_size', type=int, default=1024,
                        help='size (pixels) of each image')
    parser.add_argument('--lora_augment_path', type=str, default='',
                        help='Optional LoRA path to texture-augment generated data in-place.')
    parser.add_argument('--lora_augment_model_family', type=str, choices=['sdxl', 'sd21'], default='sdxl',
                        help='Diffusion model family used for optional LoRA augmentation.')
    parser.add_argument('--lora_augment_base_model_id', type=str,
                        default='stabilityai/stable-diffusion-xl-base-1.0',
                        help='Base model ID for optional LoRA augmentation.')
    parser.add_argument('--lora_augment_controlnet_model_id', type=str,
                        default='diffusers/controlnet-canny-sdxl-1.0',
                        help='ControlNet model ID for optional LoRA augmentation.')
    parser.add_argument('--lora_augment_prompt', type=str, default='scanned printed page',
                        help='Prompt for optional LoRA augmentation.')
    parser.add_argument('--lora_augment_scale', type=float, default=0.8,
                        help='LoRA scale for optional LoRA augmentation.')
    parser.add_argument('--lora_augment_strength', type=float, default=0.2,
                        help='Img2img strength for optional LoRA augmentation.')
    parser.add_argument('--lora_augment_steps', type=int, default=28,
                        help='Diffusion steps for optional LoRA augmentation.')
    parser.add_argument('--lora_augment_guidance_scale', type=float, default=1.0,
                        help='Guidance scale for optional LoRA augmentation.')
    parser.add_argument('--lora_augment_controlnet_scale', type=float, default=2.0,
                        help='ControlNet scale for optional LoRA augmentation.')
    parser.add_argument('--lora_augment_seed', type=int, default=None,
                        help='Optional random seed for deterministic LoRA augmentation.')
    parser.add_argument('--lora_augment_splits', type=str, default='train',
                        help='Comma-separated splits for augmentation (e.g. "train" or "train,val").')
    parser.add_argument('--lora_augment_canny_low', type=int, default=100,
                        help='Canny low threshold for optional LoRA augmentation.')
    parser.add_argument('--lora_augment_canny_high', type=int, default=200,
                        help='Canny high threshold for optional LoRA augmentation.')

    args = parser.parse_args()
    # Read default settings

    datasets_dir = Path(DATASETS_DIR)
    path = str(datasets_dir / args.dataset_name)
    args.dataset_name = path
    print(f"Generating YOLO dataset {args.dataset_name}...")

    dataset_dict = generate_data(args, validation = False)
    generate_data(args, validation = True)
    lora_reports = _augment_generated_dataset(Path(args.dataset_name), args)

    dataset_dict['path'] = args.dataset_name


    def represent_ordereddict(dumper, data):
        return dumper.represent_mapping('tag:yaml.org,2002:map', data.items())

    yaml.add_representer(OrderedDict, represent_ordereddict)

    with open(f"{args.dataset_name}/ultralytics.yml", 'w') as yaml_file:
        yaml.dump(dataset_dict, yaml_file, default_flow_style=False)

    if lora_reports:
        augmented_total = sum(int(r.get("images_processed", 0) or 0) for r in lora_reports)
        print(f"LoRA augmentation complete. Processed images: {augmented_total}")
        for rep in lora_reports:
            print(f"  - {rep.get('split')}/{rep.get('target')}: {rep.get('images_processed', 0)}")

    print("Training can be started with the following command:\n\n"
          f"yolo detect train data=yolo detect train data={args.dataset_name}/ultralytics.yml epochs=100 imgsz=1024 model=yolo11n.pt\n"
          "")

if __name__ == "__main__":
    main()
