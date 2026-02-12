"""
Region-level image+text verifier (dual-encoder) for TibetanOCR.

- Supports region classes: Tibetan text, Tibetan numbers, Chinese numbers
- Frozen text & image embeddings, tiny trainable projector head
- Sliding-window pooling for long regions
- Gating utilities (match score + simple geometry checks)

Fill in the TODOs to wire up your actual embeddings, datasets, and metrics.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from tibetan_utils.arg_utils import create_verifier_parser
from preprocessing import Preprocessor, PreprocCfg
from text_embedding_st import STTextEmbeddingProvider, STConfig

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional: add these when you implement real image/text backbones
# import timm
# from your_text_pkg import TibetanTokenizer, TibetanTextEncoder

# ---------- Region classes ----------

class RegionClass(Enum):
    TIBETAN_NUMBER = 0
    TIBETAN_TEXT = 1
    CHINESE_NUMBER = 2

# ---------- Config ----------

@dataclass
class VerifierConfig:
    # data
    train_manifest: Path
    val_manifest: Path
    image_root: Path
    # model
    image_embed_dim: int = 256     # set to your frozen image encoder output
    text_embed_dim: int = 512      # set to your frozen text embedding size
    proj_dim: int = 512
    freeze_image_encoder: bool = True
    freeze_text_encoder: bool = True
    # training
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 10
    device: str = "cuda"
    # contrastive
    temperature: float = 0.07
    num_hard_negatives: int = 3
    # sliding-window
    window_width_steps: int = 16
    window_stride_steps: int = 8
    # gating thresholds (tune later)
    thresh_match: float = 0.9
    thresh_iou_bin: float = 0.985
    thresh_edge_f1: float = 0.98
    thresh_swt_delta: float = 0.02

# ---------- Embedding providers (frozen) ----------

# -- NEW: drop-in ResNet31 image encoder (MMOCR) --
class ImageEmbeddingProvider(nn.Module):
    def __init__(self, out_dim: int = 256, in_channels: int = 1):
        super().__init__()
        self.backbone = ResNet31Like(in_ch=in_channels, base=32, out_ch=out_dim)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.out_dim = out_dim

    def forward(self, x):  # x: (B,C,H,W) where C=1 or 2 (edge channel)
        return self.backbone(x)


class TextEmbeddingProvider(nn.Module):
    """
    Wrapper around SentenceTransformer-based text embedding.
    """
    def __init__(self, out_dim: int, region_class):
        super().__init__()
        # Note: out_dim is ignored here; we read the real dim from the ST model.
        self.st = STTextEmbeddingProvider()  # uses default cfg; override below if needed
        self.out_dim = self.st.out_dim
        self.region_class = region_class  # keep for future class-specific logic

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        # Return L2-normalized embeddings as torch.FloatTensor on CPU;
        # caller can .to(device)
        return self.st.encode(texts)

# ---------- Projector head (trainable) ----------

class Projector(nn.Module):
    """
    Tiny projector that maps image features to the text-embedding space.
    """
    def __init__(self, image_dim: int, text_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, text_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ---------- Verifier model ----------

class RegionVerifier(nn.Module):
    """
    Dual-encoder verifier with sliding-window pooling.
    """
    def __init__(self, cfg: VerifierConfig, region_class: RegionClass):
        super().__init__()
        self.cfg = cfg
        self.region_class = region_class

        self.image_enc = ImageEmbeddingProvider(out_dim=cfg.image_embed_dim)
        self.text_enc = TextEmbeddingProvider(out_dim=cfg.text_embed_dim, region_class=region_class)

        # Use the actual text dim discovered from ST model:
        text_dim = getattr(self.text_enc, "out_dim", cfg.text_embed_dim)
        self.projector = Projector(cfg.image_embed_dim, text_dim, cfg.proj_dim)

    def similarity(self, img_feats_seq: torch.Tensor, txt_vec: torch.Tensor) -> torch.Tensor:
        """
        img_feats_seq: (B, T, Dv)   sequence over width (after encoder)
        txt_vec:      (B, Dt)       normalized text embeddings
        Returns:      (B,)          pooled cosine similarity
        """
        # Project each timestep to text space
        B, T, Dv = img_feats_seq.shape
        x = self.projector(img_feats_seq.reshape(B * T, Dv)).reshape(B, T, -1)  # (B, T, Dt)
        x = F.normalize(x, dim=-1)
        txt = F.normalize(txt_vec, dim=-1).unsqueeze(1)  # (B,1,Dt)
        # Cosine per step
        cos = (x * txt).sum(dim=-1)  # (B, T)
        # Sliding-window pooling across width
        pooled = self._sliding_pool(cos, self.cfg.window_width_px, self.cfg.window_stride_px)
        return pooled.max(dim=1).values  # best window score per sample

    @staticmethod
    def _sliding_pool(seq: torch.Tensor, win: int, stride: int) -> torch.Tensor:
        """
        seq: (B, T) similarity over width-steps; pool over windows.
        NOTE: here 'win'/'stride' are in steps; map from pixels in your dataloader.
        """
        B, T = seq.shape
        if win >= T:
            return seq.mean(dim=1, keepdim=True)
        windows = []
        for start in range(0, T - win + 1, stride):
            w = seq[:, start:start + win].mean(dim=1)
            windows.append(w)
        return torch.stack(windows, dim=1)  # (B, num_windows)

# ---------- Datasets (skeleton) ----------

class RegionPairSample:
    """
    One training sample: (region_image_tensor, positive_text, [hard_negatives], class_id)
    """
    def __init__(self, image: torch.Tensor, pos_text: str, neg_texts: List[str], cls: RegionClass):
        self.image = image
        self.pos_text = pos_text
        self.neg_texts = neg_texts
        self.cls = cls

class RegionPairDataset(torch.utils.data.Dataset):
    def __init__(self, manifest: Path, image_root: Path, region_class: RegionClass,
                 num_hard_neg: int, preproc: Preprocessor):
        super().__init__()
        self.manifest = manifest
        self.image_root = image_root
        self.region_class = region_class
        self.num_hard_neg = num_hard_neg
        self.preproc = preproc
        # TODO: load your rows (path, class_id, text, etc.) into self.rows
        self.rows = []  # fill this

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> RegionPairSample:
        row = self.rows[idx]
        img_path = (self.image_root / row["rel_path"]).as_posix()
        # load BGR or gray (uint8)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # or IMREAD_COLOR
        if img is None:
            raise FileNotFoundError(img_path)

        # >>> PREPROCESS HERE <<<
        tensor = self.preproc(img)          # (C,H,W) float32

        pos_text = row["text"]
        neg_texts = self._sample_hard_negatives(pos_text)
        return RegionPairSample(tensor, pos_text, neg_texts, self.region_class)

    def _sample_hard_negatives(self, s: str) -> List[str]:
        # TODO: implement class-specific hard negatives
        return []

def collate_pairs(batch: List[RegionPairSample]):
    images = torch.stack([b.image for b in batch], dim=0)     # (B,C,H,W)
    pos_texts = [b.pos_text for b in batch]
    neg_texts = [b.neg_texts for b in batch]
    return {"images": images, "pos_texts": pos_texts, "neg_texts": neg_texts}


# ---------- Contrastive loss ----------

def info_nce_loss(img_txt_sims: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    img_txt_sims: (B, 1+K) similarities for [positive, negatives...]
    """
    logits = img_txt_sims / temperature
    targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)  # positive at index 0
    return F.cross_entropy(logits, targets)

# ---------- Trainer (minimal loop) ----------

def train_one_epoch(model: RegionVerifier,
                    loader: torch.utils.data.DataLoader,
                    cfg: VerifierConfig) -> float:
    model.train()
    opt = torch.optim.AdamW(model.projector.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = 0.0
    for batch in loader:
        # EXPECTED batch collation (implement collate_fn in your dataset):
        # images: (B,1,H,W), pos_texts: List[str], neg_texts: List[List[str]]
        images, pos_texts, neg_texts = batch["images"], batch["pos_texts"], batch["neg_texts"]
        images = images.to(cfg.device)

        with torch.no_grad():
            img_feats_seq = model.image_enc(images)                    # (B, T, Dv)
            txt_pos = model.text_enc.encode(pos_texts).to(cfg.device)  # (B, Dt)
            # encode negatives flat
            neg_flat = sum(neg_texts, [])
            txt_neg = model.text_enc.encode(neg_flat).to(cfg.device)   # (B*K, Dt)

        # similarities
        pos_sim = model.similarity(img_feats_seq, txt_pos)             # (B,)
        # reshape negatives back to (B,K,Dt)
        K = len(neg_texts[0]) if neg_texts else 0
        if K > 0:
            txt_neg = txt_neg.view(len(pos_texts), K, -1)              # (B,K,Dt)
            # compute sim per negative
            neg_sims = []
            for k in range(K):
                neg_sims.append(model.similarity(img_feats_seq, txt_neg[:, k, :]))
            neg_sims = torch.stack(neg_sims, dim=1)                    # (B,K)
            sims = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)  # (B,1+K)
        else:
            sims = pos_sim.unsqueeze(1)                                # degenerate

        loss = info_nce_loss(sims, cfg.temperature)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += loss.item()
    return total / max(1, len(loader))

@torch.no_grad()
def evaluate(model: RegionVerifier,
             loader: torch.utils.data.DataLoader,
             cfg: VerifierConfig) -> Dict[str, float]:
    model.eval()
    # TODO: compute ROC-AUC / EER from sims; placeholder returns mean similarity
    sims_all, labels_all = [], []
    for batch in loader:
        images, pos_texts, neg_texts = batch["images"], batch["pos_texts"], batch["neg_texts"]
        images = images.to(cfg.device)
        img_feats_seq = model.image_enc(images)
        txt_pos = model.text_enc.encode(pos_texts).to(cfg.device)
        pos_sim = model.similarity(img_feats_seq, txt_pos)  # (B,)
        sims_all.append(pos_sim.cpu())
        labels_all.append(torch.ones_like(pos_sim, dtype=torch.int))
        # TODO: also evaluate negatives for proper curves
    sims = torch.cat(sims_all) if sims_all else torch.tensor([])
    return {"mean_pos_sim": sims.mean().item() if sims.numel() else float("nan")}

# ---------- Geometry checks (stubs) ----------

def binarized_iou(input_img: torch.Tensor, output_img: torch.Tensor) -> float:
    """
    TODO: implement Sauvola/Otsu binarization + IoU; here a stub.
    """
    return 1.0

def edge_f1(input_img: torch.Tensor, output_img: torch.Tensor) -> float:
    """
    TODO: implement Canny/Sobel + F1; here a stub.
    """
    return 1.0

def stroke_width_delta(input_img: torch.Tensor, output_img: torch.Tensor) -> float:
    """
    TODO: implement SWT or proxy; here a stub returning 0 (no change).
    """
    return 0.0

# ---------- Inference gate (per region) ----------

@torch.no_grad()
def verify_region(model: RegionVerifier,
                  region_image: torch.Tensor,
                  region_text: str,
                  cfg: VerifierConfig) -> Tuple[bool, Dict[str, float]]:
    """
    Returns: (accept, metrics)
    """
    model.eval()
    img = region_image.to(cfg.device).unsqueeze(0)  # (1,1,H,W)
    feats = model.image_enc(img)                    # (1,T,Dv)
    txt = model.text_enc.encode([region_text]).to(cfg.device)  # (1,Dt)
    sim = model.similarity(feats, txt)[0].item()

    # Optional: run geometry checks comparing pre/post stylization
    # For gating stylized outputs you’ll pass both images to these functions.
    iou = 1.0
    f1 = 1.0
    dsw = 0.0

    accept = (sim >= cfg.thresh_match) and \
             (iou >= cfg.thresh_iou_bin) and \
             (f1 >= cfg.thresh_edge_f1) and \
             (dsw <= cfg.thresh_swt_delta)

    return accept, {"sim": sim, "iou": iou, "edge_f1": f1, "delta_swt": dsw}

# ---------- CLI entry (hooks into your arg_utils parser) ----------

def main_verifier_train(args) -> None:
    cfg = VerifierConfig(
        train_manifest=Path(args.train_manifest),
        val_manifest=Path(args.val_manifest),
        image_root=Path(args.image_root),
        image_embed_dim=args.image_embed_dim,
        text_embed_dim=args.text_embed_dim,
        proj_dim=args.proj_dim,
        batch_size=args.batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        temperature=args.temperature,
        num_hard_negatives=args.num_hard_negatives,
        window_width_px=args.window_width_steps,
        window_stride_px=args.window_stride_steps,
    )

    region = RegionClass[args.region_class]  # e.g., "TIBETAN_TEXT"
    
    
    preproc_cfg = PreprocCfg(
        target_height=args.verifier_height,
        pad_to=args.pad_multiple,
        binarize=args.bin,
        add_edge_channel=args.edge_ch,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std
    )
    preproc = Preprocessor(preproc_cfg)

    in_ch = 2 if preproc_cfg.add_edge_channel else 1
    from text_embedding_st import STConfig
    # Build text encoder with CLI cfg
    st_cfg = STConfig(
        model_name=getattr(args, 'st_model', 'billingsmoore/minilm-bo'),
        device=(args.st_device if getattr(args, 'st_device', '') else ("cuda" if torch.cuda.is_available() else "cpu")),
        batch_size=getattr(args, 'st_batch', 64),
        normalize=getattr(args, 'st_norm', True),
    )

    model = RegionVerifier(cfg, region).to(cfg.device)
    model.image_enc = ImageEmbeddingProvider(out_dim=cfg.image_embed_dim, in_channels=in_ch).to(cfg.device)
    model.text_enc = STTextEmbeddingProvider(st_cfg)  # sets out_dim internally

    text_dim = model.text_enc.out_dim
    model.projector = Projector(cfg.image_embed_dim, text_dim, cfg.proj_dim).to(cfg.device)

    train_ds = RegionPairDataset(cfg.train_manifest, cfg.image_root, region, cfg.num_hard_negatives, preproc)
    val_ds   = RegionPairDataset(cfg.val_manifest,   cfg.image_root, region, cfg.num_hard_negatives, preproc)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size,
                                            shuffle=True, collate_fn=collate_pairs)
    val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size,
                                            collate_fn=collate_pairs)

    for epoch in range(cfg.epochs):
        loss = train_one_epoch(model, train_loader, cfg)
        metrics = evaluate(model, val_loader, cfg)
        print(f"[{epoch+1}/{cfg.epochs}] loss={loss:.4f} | val={metrics}")

    # Save only projector (frozen encoders assumed external)
    out = Path(args.output_dir) / f"verifier_{args.region_class.lower()}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"projector": model.projector.state_dict(),
                "cfg": cfg}, out)
    print(f"Saved verifier head to {out}")


def main() -> None:
    parser = create_verifier_parser()
    args = parser.parse_args()
    main_verifier_train(args)


if __name__ == "__main__":
    main()
