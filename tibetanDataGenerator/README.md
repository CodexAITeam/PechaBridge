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

## Sample Command
python path/to/main.py \
  --corpora_tibetan_numbers_path "path/to/data/corpora/Tibetan Number Words" \
  --corpora_tibetan_text_path "path/to/data/corpora/UVA Tibetan Spoken Corpus" \
  --corpora_chinese_numbers_path "path/to/data/corpora/Chinese Number Words" \
  --background_train "path/to/data/background_images_train" \
  --background_val "path/to/data/background_images_val" \
  --annotations_file_path "path/to/data/annotations/tibetan_chinese_no/bg_example_0001.txt" \
  --font_path_tibetan "path/to/fonts/Microsoft Himalaya.ttf" \
  --font_path_chinese "path/to/fonts/simkai.ttf" \
  --train_samples 2 \
  --val_samples 2
  
## List of altered scripts
- main.py (for correct use shift the script to the [initial project directory](https://github.com/CodexAITeam/TibetanOCR/tree/synthetic_generation_tib_chi_no)
- dataset_generator.py => altered to dataset_generator_tib_chi_no.py
- text_renderer.py =>altered to text_renderer_img_size.py

## Script Details
The script loads the Corpus path inputs from main.py to their corresponding bounding boxes of their ann_class_id (YOLO CLASS ID) in order to produce different texts with generate_dataset_tib_chi_no.py. 
The ann_class_id are parsed from a preconfigured annotation template named bg_PPN337138764X_00000005.txt which is located in the Tibetan Layout Analyser project. See our [Tibetan Numbers Dataset Folder](https://github.com/CodexAITeam/TibetanLayoutAnalyzer/tree/main/data/tibetan%20numbers) for sample files. Furthermore, the script uses different background image from that project in the format 1024x361 
because it reflects the original historical data format. The argparse input font_path_tibetan is used to display generated tibetan text, while is font_path_chinese used for chinese text.

Here is the table of the label mapping: 

| Class Name            | Corpus Path                     | Planned Label ID Range* | ann_class_id / YOLO Class ID |
|-----------------------|---------------------------------|-------------------------|------------------------------|
| Tibetan Number Words  | `corpora_tibetan_numbers_path`  | 000-009                 | 0                            |
| Tibetan Text Body     | `corpora_tibetan_text_path`     | 101-110                 | 1                            |
| Chinese Number Words  | `corpora_chinese_numbers_path`  | 201-210                 | 2                            |
* see Limitations

The different text inputs are given by:
-Tibetan Numbers: tib_no_0001.txt to tib_no_0010.txt: Randomly selected
-Tibetan Text: uvrip*.txt: Randomly selected
-Chinese Numbers: chi_no_0001.txt to chi_no_0010.txt: Simultaneosly selected (for instance chi_no_001.txt is selected when tib_no_0001.txt is selected)  
See our [Corpora Folder](https://github.com/CodexAITeam/TibetanOCR/tree/synthetic_generation_tib_chi_no/data/corpora) for sample files.

## Limitations and Outline for future development
- Label_dict is still not producing correct results of Planned Label ID Ranges because it only uses tibetan number file labels so far. 
- Augmentations are still very limited and will be expanded.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/CodexAITeam/TibetanOCR/blob/synthetic_generation_tib_chi_no/LICENSE) file for details.
