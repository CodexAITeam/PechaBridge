# preprocessing.py
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
import torch

@dataclass
class PreprocCfg:
    target_height: int = 64          # fixed H
    pad_to: int = 8                   # pad W to nearest multiple (e.g., 4/8/16)
    binarize: bool = False            # use Sauvola binarization
    add_edge_channel: bool = False    # append Canny/Lineart channel
    norm_mean: float = 0.5            # simple [0,1] -> normalize
    norm_std: float = 0.5

class Preprocessor:
    def __init__(self, cfg: PreprocCfg):
        self.cfg = cfg

    def __call__(self, img_bgr_or_gray: np.ndarray) -> torch.Tensor:
        """
        Input: HxWx3 (BGR) or HxW (uint8)
        Output: Tensor (C,H,W) float32 in [-1,1] (if mean/std = .5/.5)
        """
        g = self._to_gray(img_bgr_or_gray)                     # HxW uint8
        g = self._resize_keep_aspect_to_height(g, self.cfg.target_height)
        if self.cfg.binarize:
            g = self._sauvola(g)
        g = self._pad_width_multiple(g, self.cfg.pad_to)
        # base channel
        chs = [g.astype(np.float32) / 255.0]                   # [0,1]

        if self.cfg.add_edge_channel:
            ed = self._canny_edges(g)
            chs.append(ed.astype(np.float32) / 255.0)

        x = np.stack(chs, axis=0)                              # (C,H,W)
        # normalize to mean/std
        x = (x - self.cfg.norm_mean) / self.cfg.norm_std
        return torch.from_numpy(x)                             # float32

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img
        if img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        raise ValueError("Unsupported image shape")

    @staticmethod
    def _resize_keep_aspect_to_height(gray: np.ndarray, H: int) -> np.ndarray:
        h, w = gray.shape[:2]
        if h == 0 or w == 0:
            raise ValueError("Empty image")
        scale = H / float(h)
        new_w = max(1, int(round(w * scale)))
        return cv2.resize(gray, (new_w, H), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _pad_width_multiple(gray: np.ndarray, m: int) -> np.ndarray:
        if m <= 1:  # no pad
            return gray
        H, W = gray.shape
        rem = W % m
        if rem == 0:
            return gray
        pad = m - rem
        return cv2.copyMakeBorder(gray, 0, 0, 0, pad, borderType=cv2.BORDER_CONSTANT, value=255)

    @staticmethod
    def _sauvola(gray: np.ndarray, win: int = 25, k: float = 0.2) -> np.ndarray:
        # OpenCV has adaptive threshold; simple Sauvola approx:
        g = gray.astype(np.float32)
        mean = cv2.blur(g, (win, win))
        sqmean = cv2.blur(g * g, (win, win))
        var = np.maximum(sqmean - mean * mean, 1e-6)
        std = np.sqrt(var)
        R = 128.0
        thr = mean * (1 + k * ((std / R) - 1))
        out = (g > thr).astype(np.uint8) * 255
        return out

    @staticmethod
    def _canny_edges(gray: np.ndarray, t1: int = 50, t2: int = 150) -> np.ndarray:
        return cv2.Canny(gray, t1, t2)
