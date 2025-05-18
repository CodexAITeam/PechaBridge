# Tibetan Text Detection Dataset Generator

A tool for generating synthetic YOLO-formatted datasets for detecting Tibetan text, numbers, and their Chinese number counterparts in document images.

## Features
- Generates synthetic document images with Tibetan text, numbers, and Chinese numbers
- Creates corresponding YOLO-format annotations
- Maintains consistent numbering between Tibetan and Chinese number representations
- Supports multiple text corpora with intelligent text placement
- Includes data augmentation options (rotation, noise)

## New Options
python main.py \
  --corpora_tibetan_numbers_path ./data/corpora/Tibetan\ Number\ Words/ \
  --corpora_tibetan_text_path ./data/corpora/UVA\ Tibetan\ Spoken\ Corpus/ \
  --corpora_chinese_numbers_path ./data/corpora/Chinese\ Number\ Words/ 
  --font_path_tibetan ./fonts/Microsoft\ Himalaya.ttf \
  --font_path_chinese ./fonts/simkai.ttf \
  --image_width 1024 \
  --image_height 361 \
  --annotations_file_path ./data/annotations/tibetan_chinese_no.txt \

## Script Details
The script loads the Corpus path inputs from main.py to their corresponding bounding boxes of their ann_class_id (YOLO CLASS ID) in order to produce different texts in generate_dataset_tib_chi_no.py. 
Here is the table of the label mapping: 

| Corpus Path                     | ID Range | YOLO Class ID |
|---------------------------------|----------|---------------|
| `corpora_tibetan_numbers_path`  | 000-009  | 0             |
| `corpora_tibetan_text_path`     | 101-110  | 1             |
| `corpora_chinese_numbers_path`  | 201-210  | 2             |

-Tibetan Numbers: tib_no_0001.txt to tib_no_0010.txt
-Chinese Numbers: chi_no_0001.txt to chi_no_0010.txt
-Tibetan Text: Randomly selected from corpus files


## Limitations and Outline for future development
- Label_dict is still not producing correct results because it only uses tibetan number file labels so far. 
- Augmentations are still very limited.
