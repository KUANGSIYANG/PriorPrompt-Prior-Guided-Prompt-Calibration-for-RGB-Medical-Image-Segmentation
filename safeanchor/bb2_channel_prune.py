from __future__ import annotations

import numpy as np
import torch


def prune_bb2_channels_np(bb2_maps: np.ndarray, mode: str = "all") -> np.ndarray:
    mode = str(mode).lower().strip()
    x = np.asarray(bb2_maps, dtype=np.float32)
    if mode in {"", "all", "none"}:
        return x
    if mode in {"logit_grad2", "logit+grad2", "lg2", "two"}:
        if x.ndim != 3 or x.shape[0] < 3:
            raise ValueError(f"BB2 maps must be CxHxW with at least 3 channels, got {x.shape}")
        return np.stack([x[0], x[2]], axis=0).astype(np.float32, copy=False)
    if mode not in {"logit_grad", "logit+grad", "lg"}:
        raise ValueError(f"unknown BB2 channel mode: {mode}")
    if x.ndim != 3 or x.shape[0] < 3:
        raise ValueError(f"BB2 maps must be CxHxW with at least 3 channels, got {x.shape}")
    y = np.zeros_like(x, dtype=np.float32)
    y[0] = x[0]
    y[2] = x[2]
    return y


def prune_bb2_channels_torch(bb2_maps: torch.Tensor, mode: str = "all") -> torch.Tensor:
    mode = str(mode).lower().strip()
    if mode in {"", "all", "none"}:
        return bb2_maps
    if mode in {"logit_grad2", "logit+grad2", "lg2", "two"}:
        if bb2_maps.ndim != 4 or bb2_maps.shape[1] < 3:
            raise ValueError(f"BB2 maps must be BCHW with at least 3 channels, got {tuple(bb2_maps.shape)}")
        return torch.cat([bb2_maps[:, 0:1], bb2_maps[:, 2:3]], dim=1)
    if mode not in {"logit_grad", "logit+grad", "lg"}:
        raise ValueError(f"unknown BB2 channel mode: {mode}")
    if bb2_maps.ndim != 4 or bb2_maps.shape[1] < 3:
        raise ValueError(f"BB2 maps must be BCHW with at least 3 channels, got {tuple(bb2_maps.shape)}")
    y = torch.zeros_like(bb2_maps)
    y[:, 0:1] = bb2_maps[:, 0:1]
    y[:, 2:3] = bb2_maps[:, 2:3]
    return y
