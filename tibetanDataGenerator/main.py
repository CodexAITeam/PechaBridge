import argparse
from pathlib import Path
import yaml
from collections import OrderedDict
from ultralytics.data.utils import DATASETS_DIR
from tibetanDataGenerator.dataset_generator_tib_no import generate_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate YOLO dataset for Tibetan text detection")

    parser.add_argument('--background_train', type=str, default='./data/background_images_train/',
                        help='Folder with background images for training')
    parser.add_argument('--background_val', type=str, default='./data/background_images_val/',
                        help='Folder with background images for validation')
    parser.add_argument('--output_dir', type=str, default=str(Path(DATASETS_DIR)),
                        help='Base directory to save the generated dataset. (Default: Ultralytics DATASETS_DIR)')
    parser.add_argument('--dataset_name', type=str, default='yolo_tibetan_dataset',
                        help='Name for the generated dataset folder.')
    parser.add_argument('--corpora_tibetan_numbers_path', type=str, default='./data/corpora/Tibetan Number Words/',
                        help='Folder with Tibetan number words (maps to class_id 0: "tibetan_number_word").')
    parser.add_argument('--corpora_tibetan_text_path', type=str, default='./data/corpora/UVA Tibetan Spoken Corpus/',
                        help='Folder with general Tibetan text (maps to class_id 1: "tibetan_text").')
    parser.add_argument('--corpora_chinese_numbers_path', type=str, default='./data/corpora/Chinese Number Words/',
                        help='Folder with Chinese number words (maps to class_id 2: "chinese_number_word").')
    parser.add_argument('--train_samples', type=int, default=100,
                        help='Number of training samples to generate')
    parser.add_argument('--val_samples', type=int, default=20,
                        help='Number of validation samples to generate')
    parser.add_argument('--font_path_tibetan', type=str, required=True, default='ext/Microsoft Himalaya.ttf',
                        help='Path to a font file that supports Tibetan characters')
    parser.add_argument('--font_path_chinese', type=str, required=True, default='ext/simkai.ttf',
                        help='Path to a font file that supports Chinese characters')
    parser.add_argument('--single_label', action='store_true',
                        help='Use a single label "tibetan" for all files instead of using filenames as labels')
    parser.add_argument('--debug', action='store_true',
                        help='More verbose output with debug information about the image generation process.')
    parser.add_argument('--image_width', type=int, default=1024,
                        help='Width (pixels) of each generated image.')
    parser.add_argument('--image_height', type=int, default=361,
                        help='Height (pixels) of each generated image.')
    parser.add_argument("--augmentation", choices=['rotate', 'noise', 'none'], default='noise',
                        help="Type of augmentation to apply")
    parser.add_argument('--annotations_file_path', type=str,
                        default='./data/tibetan numbers/annotations/tibetan_chinese_no',
                        help='Path to a YOLO annotation file to load and draw bounding boxes from.')


    args = parser.parse_args()

    full_dataset_path = Path(args.output_dir) / args.dataset_name
    original_dataset_name = args.dataset_name
    args.dataset_name = str(full_dataset_path)

    print(f"Generating YOLO dataset in {args.dataset_name}...")

    # Generate training dataset
    # args object (containing args.annotations_file_path) is passed to generate_dataset
    train_dataset_info = generate_dataset(args, validation=False)

    # Generate validation dataset
    val_dataset_info = generate_dataset(args, validation=True)

    yaml_content = OrderedDict()
    yaml_content['path'] = original_dataset_name
    yaml_content['train'] = 'train/images'
    yaml_content['val'] = 'val/images'
    yaml_content['test'] = ''

    if 'nc' not in train_dataset_info or 'names' not in train_dataset_info:
        raise ValueError("generate_dataset did not return 'nc' or 'names' in its info dictionary.")
    yaml_content['nc'] = train_dataset_info['nc']
    yaml_content['names'] = train_dataset_info['names']

    def represent_ordereddict(dumper, data):
        return dumper.represent_mapping('tag:yaml.org,2002:map', data.items())

    yaml.add_representer(OrderedDict, represent_ordereddict)

    yaml_file_path = Path(args.output_dir) / f"{original_dataset_name}.yaml"

    with open(yaml_file_path, 'w', encoding='utf-8') as f:  # Added encoding='utf-8'
        yaml.dump(dict(yaml_content), f, sort_keys=False, allow_unicode=True)

    print(f"\nDataset generation completed. YAML configuration saved to: {yaml_file_path}")
    print("Training can be started with a command like:\n")
    print(
        f"yolo detect train data={yaml_file_path} epochs=100 imgsz=[{args.image_height},{args.image_width}] model=yolov8n.pt")


if __name__ == "__main__":
    main()
