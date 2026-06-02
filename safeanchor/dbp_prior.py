from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DBPPriorCfg:
    in_channels: int = 8
    hidden_channels: int = 32
    depth: int = 4
    residual_scale_init: float = 0.0
    dbp_scale_init: float = 0.0
    binarization_sharpness_init: float = 8.0
    box_scale_init: float = 2.5


def gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    dx = F.pad(x[..., :, 1:] - x[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx * dx + dy * dy + 1.0e-6)


def image_luma_and_grad(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = image
    if float(x.max().detach().cpu()) > 2.0:
        x = x / 255.0
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    luma = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 1.0)
    return luma, gradient_magnitude(luma)


def soft_moment_box(mask_prob: torch.Tensor, box_scale: torch.Tensor) -> torch.Tensor:
    """Build a continuous box from soft foreground mass.

    This avoids dataset-specific binary thresholds. The scale is predicted by
    the module as a trainable/domain-specific parameter, not hand-selected at
    inference time.
    """

    if mask_prob.ndim != 4 or mask_prob.shape[1] != 1:
        raise ValueError(f"mask_prob must be B1HW, got {tuple(mask_prob.shape)}")
    b, _, h, w = mask_prob.shape
    device = mask_prob.device
    dtype = mask_prob.dtype
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, h, device=device, dtype=dtype),
        torch.linspace(0.0, 1.0, w, device=device, dtype=dtype),
        indexing="ij",
    )
    xx = xx.view(1, 1, h, w)
    yy = yy.view(1, 1, h, w)
    mass = mask_prob.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)
    cx = (mask_prob * xx).sum(dim=(-2, -1), keepdim=True) / mass
    cy = (mask_prob * yy).sum(dim=(-2, -1), keepdim=True) / mass
    vx = (mask_prob * (xx - cx).square()).sum(dim=(-2, -1), keepdim=True) / mass
    vy = (mask_prob * (yy - cy).square()).sum(dim=(-2, -1), keepdim=True) / mass
    sx = torch.sqrt(vx.clamp_min(1.0e-8)) * box_scale.view(b, 1, 1, 1)
    sy = torch.sqrt(vy.clamp_min(1.0e-8)) * box_scale.view(b, 1, 1, 1)
    x1 = (cx - sx).view(b).clamp(0.0, 1.0)
    y1 = (cy - sy).view(b).clamp(0.0, 1.0)
    x2 = (cx + sx).view(b).clamp(0.0, 1.0)
    y2 = (cy + sy).view(b).clamp(0.0, 1.0)
    x_low = torch.minimum(x1, x2)
    y_low = torch.minimum(y1, y2)
    x_high = torch.maximum(x1, x2)
    y_high = torch.maximum(y1, y2)
    return torch.stack([x_low, y_low, x_high, y_high], dim=1)


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        groups = 8 if out_ch % 8 == 0 else 4 if out_ch % 4 == 0 else 1
        pad = int(kernel_size) // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DifferentiableBinarizedPrior(nn.Module):
    """Single-path learnable prior adapter.

    The module replaces manual threshold/box/alpha decisions with a continuous
    coarse residual, a differentiable threshold map, and a soft-moment box.
    """

    def __init__(self, cfg: DBPPriorCfg | None = None) -> None:
        super().__init__()
        self.cfg = cfg or DBPPriorCfg()
        hidden = int(self.cfg.hidden_channels)
        blocks: list[nn.Module] = [ConvGNAct(int(self.cfg.in_channels), hidden, 5)]
        for _ in range(max(0, int(self.cfg.depth) - 1)):
            blocks.append(ConvGNAct(hidden, hidden, 3))
        self.encoder = nn.Sequential(*blocks)
        self.residual_head = nn.Conv2d(hidden, 1, 1)
        self.threshold_head = nn.Conv2d(hidden, 1, 1)
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2, 1),
        )
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.threshold_head.weight)
        nn.init.zeros_(self.threshold_head.bias)
        nn.init.zeros_(self.global_head[-1].weight)
        nn.init.zeros_(self.global_head[-1].bias)
        self.residual_scale_logit = nn.Parameter(_inverse_softplus(float(self.cfg.residual_scale_init)))
        self.dbp_scale_logit = nn.Parameter(_inverse_softplus(float(self.cfg.dbp_scale_init)))
        self.log_sharpness = nn.Parameter(_inverse_softplus(float(self.cfg.binarization_sharpness_init)))
        self.log_box_scale = nn.Parameter(_inverse_softplus(float(self.cfg.box_scale_init)))

        # Learned loss balancers used by the training script.
        self.loss_log_vars = nn.Parameter(torch.zeros(5, dtype=torch.float32))

    def cfg_dict(self) -> dict:
        return asdict(self.cfg)

    def forward(self, image: torch.Tensor, logit_c: torch.Tensor, bb2_lg: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        prob = torch.sigmoid(logit_c)
        uncertainty = (4.0 * prob * (1.0 - prob)).clamp(0.0, 1.0)
        grad_l = gradient_magnitude(logit_c)
        luma, grad_i = image_luma_and_grad(image)
        if luma.shape[-2:] != logit_c.shape[-2:]:
            luma = F.interpolate(luma, size=logit_c.shape[-2:], mode="bilinear", align_corners=False)
            grad_i = F.interpolate(grad_i, size=logit_c.shape[-2:], mode="bilinear", align_corners=False)
        if bb2_lg is None:
            bb2_lg = torch.cat([logit_c, grad_l], dim=1)
        else:
            bb2_lg = bb2_lg.to(device=logit_c.device, dtype=logit_c.dtype)
            if bb2_lg.shape[-2:] != logit_c.shape[-2:]:
                bb2_lg = F.interpolate(bb2_lg, size=logit_c.shape[-2:], mode="bilinear", align_corners=False)
        z = torch.cat([logit_c, prob, uncertainty, grad_l, luma, grad_i, bb2_lg[:, 0:1], bb2_lg[:, 1:2]], dim=1)
        feat = self.encoder(z)
        raw_residual = torch.tanh(self.residual_head(feat))
        residual_scale = F.softplus(self.residual_scale_logit).to(dtype=logit_c.dtype, device=logit_c.device)
        residual = residual_scale * raw_residual
        refined_logit = logit_c + residual
        refined_prob = torch.sigmoid(refined_logit)
        threshold = torch.sigmoid(self.threshold_head(feat))
        sharpness = F.softplus(self.log_sharpness).to(dtype=logit_c.dtype, device=logit_c.device).clamp_min(1.0)
        dbp_logit = sharpness * (refined_prob - threshold)
        dbp_mask = torch.sigmoid(dbp_logit)
        dbp_scale = F.softplus(self.dbp_scale_logit).to(dtype=logit_c.dtype, device=logit_c.device)
        prompt = refined_logit + dbp_scale * dbp_logit
        global_delta = self.global_head(feat).flatten(1)
        box_scale = F.softplus(self.log_box_scale + 0.25 * torch.tanh(global_delta[:, 0])).clamp_min(0.5)
        temperature = F.softplus(global_delta[:, 1]).view(-1, 1, 1, 1).clamp_min(0.5) + 0.5
        boxes = soft_moment_box(dbp_mask, box_scale)
        return {
            "prompt": prompt,
            "boxes": boxes,
            "refined_logit": refined_logit,
            "refined_prob": refined_prob,
            "threshold": threshold,
            "dbp_logit": dbp_logit,
            "dbp_mask": dbp_mask,
            "residual": residual,
            "temperature": temperature,
            "box_scale": box_scale,
            "sharpness": sharpness.view(1),
            "dbp_scale": dbp_scale.view(1),
        }


def uncertainty_weighted_sum(losses: list[torch.Tensor], log_vars: torch.Tensor) -> torch.Tensor:
    total = torch.zeros((), device=losses[0].device, dtype=losses[0].dtype)
    for idx, loss in enumerate(losses):
        s = log_vars[idx].to(device=loss.device, dtype=loss.dtype)
        total = total + torch.exp(-s) * loss + s
    return total


def _inverse_softplus(value: float) -> torch.Tensor:
    v = torch.tensor(max(float(value), 1.0e-6), dtype=torch.float32)
    return torch.log(torch.expm1(v))
