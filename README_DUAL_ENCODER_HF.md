# PechaBridgeDualEncoder

Tibetan dual image-text encoder trained with a CLIP-style symmetric InfoNCE loss
on Tibetan pecha line images paired with their OCR transcripts.
The model enables cross-modal retrieval between line images and Tibetan text.

## Architecture

| Component | Base model | Role |
|-----------|-----------|------|
| **Image encoder** (ViT backbone) | `facebook/dinov2-base` | Encodes line-image crops → 256-d embedding |
| **Text encoder** (CLIP text encoder) | `google/byt5-small` | Encodes Tibetan transcript lines → 256-d embedding |

- **Projection dimension**: 256
- **Image preprocessing pipeline**: `bdrc` (target height 64 px, max width 1024 px, aspect-preserving)
- **Training step**: 45 000
- **Training pairs**: 190 787

## Retrieval performance (validation set, 14 877 pairs)

| Metric | Score |
|--------|-------|
| Image→Text R@1 | 96.26 % |
| Text→Image R@1 | 97.02 % |
| Image→Text R@5 | 99.02 % |
| Text→Image R@5 | 99.32 % |

## Recommended usage — PechaBridge CLI

```bash
# 1. Clone PechaBridge and install dependencies
git clone https://github.com/CodexAITeam/PechaBridge.git && cd PechaBridge
pip install -r requirements.txt

# 2. Download this encoder (alongside OCR + line segmentation models)
python cli.py download-models --models encoder

# 3. Launch the Semantic Search Workbench
#    Set huggingface_model_id in your workbench config to:
#      TibetanCodexAITeam/PechaBridgeDualEncoder/text_encoder
python cli.py semantic-search-workbench --config configs/semantic_search.yaml
```

## Repository layout

```
PechaBridgeDualEncoder/
  text_encoder/          ← fine-tuned ByT5 text encoder + tokenizer
    config.json
    model.safetensors
    tokenizer_config.json
    added_tokens.json
  vit_backbone/          ← fine-tuned DINOv2 image encoder + preprocessor
    config.json
    model.safetensors
    preprocessor_config.json
  training_config.json   ← full training hyper-parameters & validation metrics
  README.md
```

## Python usage

### Text-only embedding (Semantic Search Workbench)

The Semantic Search Workbench uses only the **text encoder** sub-directory
via `AutoModel` + `AutoTokenizer`. Point `huggingface_model_id` in your
workbench YAML config to the `text_encoder` subfolder:

```yaml
# configs/semantic_search.yaml  (excerpt)
embedding:
  huggingface_model_id: TibetanCodexAITeam/PechaBridgeDualEncoder/text_encoder
  device: cpu          # or cuda
  batch_size: 64
  max_length: 256
  normalize_embeddings: true
  trust_remote_code: false
  document_prefix: ""
  query_prefix: ""
```

Or load it directly in Python:

```python
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F

repo = "TibetanCodexAITeam/PechaBridgeDualEncoder"

tokenizer = AutoTokenizer.from_pretrained(repo, subfolder="text_encoder")
text_model = AutoModel.from_pretrained(repo, subfolder="text_encoder").eval()

texts = ["བོད་ཀྱི་གནའ་རབས་ལོ་རྒྱུས།"]
enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
with torch.no_grad():
    out = text_model(**enc)
    # mean-pool over sequence length
    mask = enc["attention_mask"].unsqueeze(-1).float()
    emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    emb = F.normalize(emb, p=2, dim=1)
print(emb.shape)  # (1, 1472)  — raw hidden size before projection
```

### Image embedding (cross-modal retrieval)

```python
from transformers import AutoModel, AutoImageProcessor
from pechabridge.ocr.preprocess_bdrc import BDRCPreprocessConfig, preprocess_image_bdrc
from PIL import Image
import torch
import torch.nn.functional as F

repo = "TibetanCodexAITeam/PechaBridgeDualEncoder"

processor = AutoImageProcessor.from_pretrained(repo, subfolder="vit_backbone")
vit_model = AutoModel.from_pretrained(repo, subfolder="vit_backbone").eval()

# Load and BDRC-preprocess a line-image crop
image = Image.open("line_crop.png").convert("RGB")
cfg = BDRCPreprocessConfig(pipeline="bdrc", target_height=64, max_width=1024)
preprocessed = preprocess_image_bdrc(image, cfg)

inputs = processor(images=preprocessed, return_tensors="pt")
with torch.no_grad():
    out = vit_model(**inputs)
    # CLS token
    emb = F.normalize(out.last_hidden_state[:, 0, :], p=2, dim=1)
print(emb.shape)  # (1, 768)
```

## Training details

| Parameter | Value |
|-----------|-------|
| Batch size | 32 |
| Learning rate | 4e-5 |
| Loss | CLIP symmetric InfoNCE |
| Mixed precision | bf16 |
| Warmup steps | 1 500 |
| Temperature | 0.1 |
| Epochs | 20 |
| Steps per epoch | 5 962 |

## Training framework

- **Framework**: [PechaBridge](https://github.com/CodexAITeam/PechaBridge)
- **Training data**: Tibetan pecha line images from OpenPecha and BDRC collections
- **Image preprocessing**: BDRC-style adaptive binarisation, background normalisation,
  aspect-preserving resize and padding
