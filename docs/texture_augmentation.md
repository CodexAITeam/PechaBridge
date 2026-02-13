# Texture Augmentation (SDXL + ControlNet Canny + LoRA)

This workflow changes paper/ink/scan texture while preserving glyph geometry.

## Requirements

Install core dependencies:

```bash
pip install torch torchvision diffusers accelerate transformers opencv-python safetensors peft datasets tqdm
```

Or install the project-wide unified dependency file:

```bash
pip install -r requirements.txt
```

Base models used by default:

- `stabilityai/stable-diffusion-xl-base-1.0`
- `diffusers/controlnet-canny-sdxl-1.0`

## 1) Prepare real texture crops for LoRA

Generate square crops from real pecha pages using Canny edge density sampling.

```bash
python cli.py prepare-texture-lora-dataset \
  --input_dir /path/to/real_pecha_pages \
  --output_dir /path/to/texture_lora_dataset \
  --crop_size 1024 \
  --num_crops_per_page 12 \
  --min_edge_density 0.025 \
  --seed 42
```

Outputs:

- Crops: `/path/to/texture_lora_dataset/images/*.png`
- Metadata: `/path/to/texture_lora_dataset/metadata.jsonl`

## 2) Train SDXL texture LoRA

Single-process run:

```bash
python cli.py train-texture-lora \
  --dataset_dir /path/to/texture_lora_dataset \
  --output_dir /path/to/texture_lora_output \
  --resolution 1024 \
  --batch_size 1 \
  --lr 1e-4 \
  --max_train_steps 1500 \
  --rank 16 \
  --lora_alpha 16 \
  --mixed_precision fp16 \
  --gradient_checkpointing \
  --seed 42
```

Distributed/multi-GPU run (recommended when available):

```bash
accelerate launch scripts/train_texture_lora_sdxl.py \
  --dataset_dir /path/to/texture_lora_dataset \
  --output_dir /path/to/texture_lora_output \
  --resolution 1024 \
  --batch_size 1 \
  --lr 1e-4 \
  --max_train_steps 1500 \
  --rank 16 \
  --lora_alpha 16 \
  --mixed_precision fp16 \
  --gradient_checkpointing \
  --seed 42
```

Outputs:

- LoRA weights: `/path/to/texture_lora_output/texture_lora.safetensors`
- Config: `/path/to/texture_lora_output/training_config.json`

## 3) Run texture augmentation inference

Conservative defaults for structure preservation:

- `strength=0.2` (hard-clamped to `<=0.25`)
- `controlnet_scale=2.0`
- `guidance_scale=1.0`

```bash
python cli.py texture-augment \
  --input_dir /path/to/synthetic_renders \
  --output_dir /path/to/synthetic_textured \
  --strength 0.2 \
  --steps 28 \
  --guidance_scale 1.0 \
  --controlnet_scale 2.0 \
  --seed 123
```

With LoRA:

```bash
python cli.py texture-augment \
  --input_dir /path/to/synthetic_renders \
  --output_dir /path/to/synthetic_textured \
  --lora_path /path/to/texture_lora_output/texture_lora.safetensors \
  --lora_scale 0.8 \
  --strength 0.2 \
  --controlnet_scale 2.0 \
  --seed 123
```

## Direct script entrypoints

You can also run scripts directly:

- `python scripts/prepare_texture_lora_dataset.py ...`
- `python scripts/train_texture_lora_sdxl.py ...`
- `python scripts/texture_augment.py ...`
