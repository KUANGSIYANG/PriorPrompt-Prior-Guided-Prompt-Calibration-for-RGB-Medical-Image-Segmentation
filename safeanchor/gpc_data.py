from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from safeanchor.coarse_prior_calibrator import (
    CoarsePriorCalibratorCfg,
    FailureAwareCoarsePriorCalibrator,
)
from safeanchor.io_utils import load_prob_map


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def safe_logit(prob: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=np.float32), 1.0e-4, 1.0 - 1.0e-4)
    return np.log(p / (1.0 - p)).astype(np.float32)


class PriorDataset(Dataset):
    """Dataset wrapper for image, GT mask, and upstream nnUNet probability."""

    def __init__(self, *, root: Path, split_cfg: dict, max_train_side: int = 768) -> None:
        self.root = root
        self.images_dir = resolve_path(root, split_cfg["images_dir"])
        self.labels_dir = resolve_path(root, split_cfg["labels_dir"])
        self.prob_dir = resolve_path(root, split_cfg["prob_dir"])
        self.ids_file = resolve_path(root, split_cfg["ids_file"])
        self.ids = [
            line.strip().lstrip("\ufeff").removesuffix(".png")
            for line in self.ids_file.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        self.max_train_side = int(max_train_side)

    def __len__(self) -> int:
        return len(self.ids)

    @staticmethod
    def _resize_rgb(x: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        img = Image.fromarray(np.asarray(np.clip(x, 0.0, 1.0) * 255.0, dtype=np.uint8), mode="RGB")
        img = img.resize((out_w, out_h), resample=Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0

    @staticmethod
    def _resize_bilinear(x: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        img = Image.fromarray(np.asarray(x, dtype=np.float32), mode="F")
        img = img.resize((out_w, out_h), resample=Image.BILINEAR)
        return np.asarray(img, dtype=np.float32)

    @staticmethod
    def _resize_mask(x: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        img = Image.fromarray((np.asarray(x, dtype=np.float32) > 0.5).astype(np.uint8) * 255, mode="L")
        img = img.resize((out_w, out_h), resample=Image.NEAREST)
        return (np.asarray(img, dtype=np.uint8) > 127).astype(np.float32)

    def _image_path(self, case_id: str) -> Path:
        path = self.images_dir / f"{case_id}.png"
        if path.exists():
            return path
        path = self.images_dir / f"{case_id}_0000.png"
        if path.exists():
            return path
        raise FileNotFoundError(f"missing image for {case_id}")

    def __getitem__(self, index: int) -> dict:
        case_id = self.ids[index]
        image = np.asarray(Image.open(self._image_path(case_id)).convert("RGB"), dtype=np.float32) / 255.0
        gt = (np.asarray(Image.open(self.labels_dir / f"{case_id}.png").convert("L"), dtype=np.uint8) > 0).astype(np.float32)
        prob = load_prob_map(self.prob_dir / f"{case_id}.npz")
        h, w = prob.shape
        if max(h, w) > self.max_train_side:
            scale = float(self.max_train_side) / float(max(h, w))
            out_h = max(32, int(round(h * scale)))
            out_w = max(32, int(round(w * scale)))
            image = self._resize_rgb(image, out_h, out_w)
            gt = self._resize_mask(gt, out_h, out_w)
            prob = self._resize_bilinear(prob, out_h, out_w)
        return {
            "case_id": case_id,
            "image": image.transpose(2, 0, 1).astype(np.float32),
            "gt": gt[None].astype(np.float32),
            "logit_raw": safe_logit(prob)[None].astype(np.float32),
        }


def collate_pad(batch: list[dict]) -> dict:
    out = {"case_id": [str(item["case_id"]) for item in batch]}
    for key in ("image", "gt", "logit_raw"):
        arrays = [np.asarray(item[key], dtype=np.float32) for item in batch]
        h_max = max(array.shape[-2] for array in arrays)
        w_max = max(array.shape[-1] for array in arrays)
        padded = []
        for array in arrays:
            pad = np.zeros((array.shape[0], h_max, w_max), dtype=np.float32)
            pad[:, : array.shape[-2], : array.shape[-1]] = array
            padded.append(torch.from_numpy(pad))
        out[key] = torch.stack(padded, dim=0)
    return out


def prob_to_bbox_torch(prob: torch.Tensor, *, threshold: float = 0.4, margin_ratio: float = 0.05) -> torch.Tensor:
    boxes = []
    for i in range(prob.shape[0]):
        p = prob[i, 0].detach()
        mask = p > float(threshold)
        h, w = int(mask.shape[0]), int(mask.shape[1])
        ys, xs = torch.where(mask)
        if ys.numel() == 0:
            boxes.append(torch.tensor([0.0, 0.0, 1.0, 1.0], device=prob.device, dtype=prob.dtype))
            continue
        x1 = int(xs.min().item())
        x2 = int(xs.max().item())
        y1 = int(ys.min().item())
        y2 = int(ys.max().item())
        dw = int(round((x2 - x1 + 1) * float(margin_ratio)))
        dh = int(round((y2 - y1 + 1) * float(margin_ratio)))
        x1 = max(0, x1 - dw)
        y1 = max(0, y1 - dh)
        x2 = min(w - 1, x2 + dw)
        y2 = min(h - 1, y2 + dh)
        boxes.append(
            torch.tensor(
                [
                    float(x1) / float(max(1, w - 1)),
                    float(y1) / float(max(1, h - 1)),
                    float(x2) / float(max(1, w - 1)),
                    float(y2) / float(max(1, h - 1)),
                ],
                device=prob.device,
                dtype=prob.dtype,
            )
        )
    return torch.stack(boxes, dim=0)


def run_bb2_logits(adapter: torch.nn.Module, maps: torch.Tensor, logit: torch.Tensor) -> torch.Tensor:
    try:
        out = adapter.run_bb2_adapter(maps, logit)
    except TypeError:
        out = adapter.run_bb2_adapter(maps)
    if isinstance(out, (tuple, list)):
        return out[-1]
    return out


def load_coarse_calibrator(ckpt_path: Path, device: torch.device) -> FailureAwareCoarsePriorCalibrator:
    state = torch.load(str(ckpt_path), map_location="cpu")
    manifest = state.get("manifest", {}) if isinstance(state, dict) else {}
    cfg_dict = dict(manifest.get("coarse_calibrator_cfg", {}))
    cfg = CoarsePriorCalibratorCfg(
        hidden_channels=int(cfg_dict.get("hidden_channels", 24)),
        residual_scale=float(cfg_dict.get("residual_scale", 1.10)),
        support_floor=float(cfg_dict.get("support_floor", 0.04)),
        basis_scale=float(cfg_dict.get("basis_scale", 0.20)),
        temperature_scale=float(cfg_dict.get("temperature_scale", 0.30)),
        bias_scale=float(cfg_dict.get("bias_scale", 0.70)),
        support_learn_scale=float(cfg_dict.get("support_learn_scale", 0.35)),
        point_gate_scale=float(cfg_dict.get("point_gate_scale", 1.0)),
    )
    model = FailureAwareCoarsePriorCalibrator(cfg).to(device)
    model.load_state_dict(state["coarse_calibrator_state_dict"], strict=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model
