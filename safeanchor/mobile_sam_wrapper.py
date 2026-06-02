from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from vendor.mobile_sam.build_sam import sam_model_registry


@dataclass
class SamImageInfo:
    original_size: Tuple[int, int]
    input_size: Tuple[int, int]
    scale: float


def _resize_longest_side_hw(h: int, w: int, target_longest: int) -> Tuple[int, int, float]:
    longest = max(h, w)
    scale = 1.0 if longest == 0 else float(target_longest) / float(longest)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    return new_h, new_w, scale


def _pad_to_square(x: torch.Tensor, target: int) -> torch.Tensor:
    h, w = x.shape[-2:]
    return F.pad(x, (0, max(0, target - w), 0, max(0, target - h)))


def preprocess_sam_image(
    image_rgb: torch.Tensor,
    *,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
    target_longest: int = 1024,
) -> Tuple[torch.Tensor, SamImageInfo]:
    b, c, h, w = image_rgb.shape
    new_h, new_w, scale = _resize_longest_side_hw(h, w, target_longest)
    x = image_rgb
    if float(x.max().detach().cpu()) <= 1.5:
        x = x * 255.0
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    x = (x - pixel_mean) / pixel_std
    x = _pad_to_square(x, target_longest)
    info = SamImageInfo(original_size=(int(h), int(w)), input_size=(int(new_h), int(new_w)), scale=float(scale))
    return x, info


def boxes_norm_to_resized_xyxy(
    boxes_norm: torch.Tensor,
    *,
    original_size: Tuple[int, int],
    scale: float,
) -> torch.Tensor:
    h, w = original_size
    b = boxes_norm.clamp(0.0, 1.0)
    x1 = b[:, 0] * float(w - 1) * float(scale)
    y1 = b[:, 1] * float(h - 1) * float(scale)
    x2 = b[:, 2] * float(w - 1) * float(scale)
    y2 = b[:, 3] * float(h - 1) * float(scale)
    return torch.stack([x1, y1, x2, y2], dim=1)


def points_norm_to_resized_xy(
    points_norm: torch.Tensor,
    *,
    original_size: Tuple[int, int],
    scale: float,
) -> torch.Tensor:
    h, w = original_size
    p = points_norm.clamp(0.0, 1.0)
    x = p[..., 0] * float(w - 1) * float(scale)
    y = p[..., 1] * float(h - 1) * float(scale)
    return torch.stack([x, y], dim=-1)


def mask_prompt_to_mask_input(
    mask_prompt: torch.Tensor,
    *,
    input_size: Tuple[int, int],
    target_longest: int = 1024,
    low_res: int = 256,
    already_low_res: bool = False,
) -> torch.Tensor:
    x = mask_prompt
    if float(x.min().detach().cpu()) >= 0.0 and float(x.max().detach().cpu()) <= 1.0:
        eps = 1e-4
        x = x.clamp(eps, 1.0 - eps)
        x = torch.log(x / (1.0 - x))
    if bool(already_low_res):
        if x.shape[-2:] != (low_res, low_res):
            x = F.interpolate(x, size=(low_res, low_res), mode="bilinear", align_corners=False)
        return x
    x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
    x = _pad_to_square(x, target_longest)
    x = F.interpolate(x, size=(low_res, low_res), mode="bilinear", align_corners=False)
    return x


def target_mask_to_low_res_input(
    mask: torch.Tensor,
    *,
    info: SamImageInfo,
    target_longest: int = 1024,
    low_res: int = 256,
) -> torch.Tensor:
    """Map an original-image binary target to SAM's padded low-res mask grid."""

    x = F.interpolate(mask.float(), size=info.input_size, mode="nearest")
    x = _pad_to_square(x, target_longest)
    x = F.interpolate(x, size=(low_res, low_res), mode="nearest")
    return x


class FrozenMobileSAM(nn.Module):
    def __init__(
        self,
        checkpoint: Path | str,
        *,
        model_type: str = "vit_t",
        device: Optional[str] = None,
        target_longest: int = 1024,
        sam_state_dict: Optional[dict] = None,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.target_longest = int(target_longest)
        sam = sam_model_registry[str(model_type)](checkpoint=str(checkpoint))
        if sam_state_dict is not None:
            sam.load_state_dict(sam_state_dict, strict=False)
        sam.to(self.device)
        sam.eval()
        for p in sam.parameters():
            p.requires_grad = bool(trainable)
        self.sam = sam

    @property
    def pixel_mean(self) -> torch.Tensor:
        return self.sam.pixel_mean.to(self.device)

    @property
    def pixel_std(self) -> torch.Tensor:
        return self.sam.pixel_std.to(self.device)

    def encode_image(self, image_rgb: torch.Tensor) -> Tuple[torch.Tensor, SamImageInfo]:
        image_in, info = preprocess_sam_image(
            image_rgb.to(self.device),
            pixel_mean=self.pixel_mean,
            pixel_std=self.pixel_std,
            target_longest=self.target_longest,
        )
        with torch.no_grad():
            image_embeddings = self.sam.image_encoder(image_in)
        return image_embeddings, info

    def decode_from_embeddings(
        self,
        image_embeddings: torch.Tensor,
        info: SamImageInfo,
        *,
        boxes_norm: Optional[torch.Tensor],
        point_coords_norm: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        mask_prompt: Optional[torch.Tensor],
        bb2_mask_prompt_logits: Optional[torch.Tensor] = None,
        bb2_alpha: Optional[object] = None,
        mask_prompt_low_res: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        boxes = None
        if boxes_norm is not None:
            boxes = boxes_norm_to_resized_xyxy(
                boxes_norm.to(self.device),
                original_size=info.original_size,
                scale=info.scale,
            )
        points = None
        if point_coords_norm is not None:
            if point_labels is None:
                raise ValueError("point_labels must be provided when point_coords_norm is not None")
            coords = points_norm_to_resized_xy(
                point_coords_norm.to(self.device),
                original_size=info.original_size,
                scale=info.scale,
            )
            labels = point_labels.to(device=self.device, dtype=torch.int64)
            points = (coords, labels)
        if bb2_mask_prompt_logits is not None:
            bb2 = bb2_mask_prompt_logits.to(self.device)
            if mask_prompt is None:
                mask_prompt = bb2
            else:
                if torch.is_tensor(bb2_alpha):
                    a = bb2_alpha.to(device=self.device, dtype=mask_prompt.dtype)
                    if a.ndim == 0:
                        pass
                    elif a.ndim == 3:
                        a = a.unsqueeze(1)
                    elif a.ndim != 4:
                        raise ValueError(f"bb2_alpha tensor must be scalar/B1HW/BHW, got {tuple(a.shape)}")
                    if a.ndim == 4 and a.shape[-2:] != mask_prompt.shape[-2:]:
                        a = F.interpolate(a, size=mask_prompt.shape[-2:], mode="bilinear", align_corners=False)
                else:
                    a = 0.5 if bb2_alpha is None else float(bb2_alpha)
                mask_prompt = a * mask_prompt.to(self.device) + (1.0 - a) * bb2
        masks = None
        if mask_prompt is not None:
            masks = mask_prompt_to_mask_input(
                mask_prompt.to(self.device),
                input_size=info.input_size,
                target_longest=self.target_longest,
                low_res=256,
                already_low_res=bool(mask_prompt_low_res),
            )
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
        )
        low_res_masks, iou_pred = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        masks_full = self.sam.postprocess_masks(low_res_masks, info.input_size, info.original_size)
        return masks_full, iou_pred

    def encode_prompts(
        self,
        info: SamImageInfo,
        *,
        boxes_norm: Optional[torch.Tensor],
        point_coords_norm: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        mask_prompt: Optional[torch.Tensor],
        mask_prompt_low_res: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        boxes = None
        if boxes_norm is not None:
            boxes = boxes_norm_to_resized_xyxy(
                boxes_norm.to(self.device),
                original_size=info.original_size,
                scale=info.scale,
            )
        points = None
        if point_coords_norm is not None:
            if point_labels is None:
                raise ValueError("point_labels must be provided when point_coords_norm is not None")
            coords = points_norm_to_resized_xy(
                point_coords_norm.to(self.device),
                original_size=info.original_size,
                scale=info.scale,
            )
            labels = point_labels.to(device=self.device, dtype=torch.int64)
            points = (coords, labels)
        masks = None
        if mask_prompt is not None:
            masks = mask_prompt_to_mask_input(
                mask_prompt.to(self.device),
                input_size=info.input_size,
                target_longest=self.target_longest,
                low_res=256,
                already_low_res=bool(mask_prompt_low_res),
            )
        return self.sam.prompt_encoder(points=points, boxes=boxes, masks=masks)

    def decode_low_res_from_embeddings(
        self,
        image_embeddings: torch.Tensor,
        *,
        boxes_norm: Optional[torch.Tensor],
        point_coords_norm: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        mask_prompt: Optional[torch.Tensor],
        info: SamImageInfo,
        mask_prompt_low_res: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sparse_embeddings, dense_embeddings = self.encode_prompts(
            info,
            boxes_norm=boxes_norm,
            point_coords_norm=point_coords_norm,
            point_labels=point_labels,
            mask_prompt=mask_prompt,
            mask_prompt_low_res=bool(mask_prompt_low_res),
        )
        return self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

    def decode_from_prompt_embeddings(
        self,
        image_embeddings: torch.Tensor,
        *,
        info: SamImageInfo,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        low_res_masks, iou_pred = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=False,
        )
        masks_full = self.sam.postprocess_masks(low_res_masks, info.input_size, info.original_size)
        return masks_full, iou_pred

    def forward(
        self,
        image_rgb: torch.Tensor,
        *,
        boxes_norm: Optional[torch.Tensor],
        point_coords_norm: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        mask_prompt: Optional[torch.Tensor],
        bb2_mask_prompt_logits: Optional[torch.Tensor] = None,
        bb2_alpha: Optional[object] = None,
        mask_prompt_low_res: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_embeddings, info = self.encode_image(image_rgb)
        return self.decode_from_embeddings(
            image_embeddings=image_embeddings,
            info=info,
            boxes_norm=boxes_norm,
            point_coords_norm=point_coords_norm,
            point_labels=point_labels,
            mask_prompt=mask_prompt,
            bb2_mask_prompt_logits=bb2_mask_prompt_logits,
            bb2_alpha=bb2_alpha,
            mask_prompt_low_res=bool(mask_prompt_low_res),
        )
