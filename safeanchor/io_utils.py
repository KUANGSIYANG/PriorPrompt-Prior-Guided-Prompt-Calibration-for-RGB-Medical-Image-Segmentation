from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image


def load_rgb_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_mask_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def load_prob_map(path: Path) -> np.ndarray:
    obj = np.load(path)
    p = np.asarray(obj["probabilities"], dtype=np.float32)
    if p.ndim != 4 or p.shape[1] != 1 or p.shape[0] < 2:
        raise ValueError(f"unexpected probability shape: {p.shape}")
    return p[1, 0]


def image_to_tensor(image_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(image_rgb.transpose(2, 0, 1))).float().unsqueeze(0)
    return x.to(device)


def map_to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)


def save_mask_png(mask01: np.ndarray, path: Path) -> None:
    arr = (np.asarray(mask01, dtype=np.uint8) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def bbox_from_mask(
    mask: np.ndarray,
    *,
    margin_ratio: float = 0.05,
    margin_px: int = 0,
    scale: float = 1.0,
) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    h, w = int(m.shape[0]), int(m.shape[1])
    if not bool(np.any(m)):
        return np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    rows = np.flatnonzero(m.any(axis=1))
    cols = np.flatnonzero(m.any(axis=0))
    x1, x2 = int(cols[0]), int(cols[-1])
    y1, y2 = int(rows[0]), int(rows[-1])
    dw = int(round((x2 - x1 + 1) * float(margin_ratio))) + int(margin_px)
    dh = int(round((y2 - y1 + 1) * float(margin_ratio))) + int(margin_px)
    x1 = max(0, x1 - dw)
    y1 = max(0, y1 - dh)
    x2 = min(w - 1, x2 + dw)
    y2 = min(h - 1, y2 + dh)
    if abs(float(scale) - 1.0) > 1e-6:
        cx = 0.5 * float(x1 + x2)
        cy = 0.5 * float(y1 + y2)
        bw = max(1.0, float(x2 - x1 + 1) * float(scale))
        bh = max(1.0, float(y2 - y1 + 1) * float(scale))
        x1 = int(round(cx - 0.5 * bw))
        x2 = int(round(cx + 0.5 * bw))
        y1 = int(round(cy - 0.5 * bh))
        y2 = int(round(cy + 0.5 * bh))
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
    return np.asarray(
        [
            float(x1) / float(max(1, w - 1)),
            float(y1) / float(max(1, h - 1)),
            float(x2) / float(max(1, w - 1)),
            float(y2) / float(max(1, h - 1)),
        ],
        dtype=np.float32,
    )


def bbox_xyxy_from_mask(
    mask: np.ndarray,
    *,
    margin_ratio: float = 0.05,
    margin_px: int = 0,
) -> tuple[int, int, int, int]:
    m = np.asarray(mask, dtype=bool)
    h, w = int(m.shape[0]), int(m.shape[1])
    if not bool(np.any(m)):
        return 0, 0, w - 1, h - 1
    rows = np.flatnonzero(m.any(axis=1))
    cols = np.flatnonzero(m.any(axis=0))
    x1, x2 = int(cols[0]), int(cols[-1])
    y1, y2 = int(rows[0]), int(rows[-1])
    dw = int(round((x2 - x1 + 1) * float(margin_ratio))) + int(margin_px)
    dh = int(round((y2 - y1 + 1) * float(margin_ratio))) + int(margin_px)
    x1 = max(0, x1 - dw)
    y1 = max(0, y1 - dh)
    x2 = min(w - 1, x2 + dw)
    y2 = min(h - 1, y2 + dh)
    return x1, y1, x2, y2
