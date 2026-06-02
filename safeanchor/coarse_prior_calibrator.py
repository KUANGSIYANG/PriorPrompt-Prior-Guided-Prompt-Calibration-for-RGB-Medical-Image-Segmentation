from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class CoarsePriorCalibratorCfg:
    hidden_channels: int = 24
    residual_scale: float = 1.10
    support_floor: float = 0.04
    basis_scale: float = 0.20
    temperature_scale: float = 0.30
    bias_scale: float = 0.70
    clip_logit: float = 8.0
    support_learn_scale: float = 0.35
    point_gate_scale: float = 1.0


def _sobel_mag(x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    device = x.device
    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=dtype, device=device).view(1, 1, 3, 3)
    ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=dtype, device=device).view(1, 1, 3, 3)
    gx = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), kx)
    gy = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), ky)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
    scale = torch.amax(mag.flatten(2), dim=-1, keepdim=True).view(x.shape[0], 1, 1, 1).clamp_min(1e-4)
    return (mag / scale).clamp(0.0, 1.0)


def _coord_grid_like(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b, _, h, w = x.shape
    yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
    xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
    return xx, yy


def coarse_moment_features(prob: torch.Tensor) -> torch.Tensor:
    b, _, h, w = prob.shape
    xx, yy = _coord_grid_like(prob)
    mass = prob.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
    area = mass / float(max(1, h * w))
    cx = (prob * xx).sum(dim=(-2, -1), keepdim=True) / mass
    cy = (prob * yy).sum(dim=(-2, -1), keepdim=True) / mass
    dx = xx - cx
    dy = yy - cy
    var_x = (prob * dx * dx).sum(dim=(-2, -1), keepdim=True) / mass
    var_y = (prob * dy * dy).sum(dim=(-2, -1), keepdim=True) / mass
    cov_xy = (prob * dx * dy).sum(dim=(-2, -1), keepdim=True) / mass
    entropy = -(prob.clamp(1e-4, 1.0 - 1e-4) * torch.log(prob.clamp(1e-4, 1.0 - 1e-4))
                + (1.0 - prob).clamp(1e-4, 1.0) * torch.log((1.0 - prob).clamp(1e-4, 1.0)))
    ent = entropy.mean(dim=(-2, -1), keepdim=True)
    return torch.cat([area, cx, cy, var_x, var_y, cov_xy, ent], dim=1).flatten(1)


def build_bb2_maps_from_logit(
    logit_p: torch.Tensor,
    *,
    t_box: float = 0.4,
    clip_logit: float = 6.0,
    band_sigma: float = 0.08,
    morph_kernel: int = 3,
) -> torch.Tensor:
    """Differentiable torch counterpart of safeanchor.bb2_maps.build_bb2_maps."""

    if logit_p.ndim != 4 or logit_p.shape[1] != 1:
        raise ValueError(f"logit_p must be B1HW, got {tuple(logit_p.shape)}")
    prob = torch.sigmoid(logit_p)
    logit_clip = logit_p.clamp(-float(clip_logit), float(clip_logit))
    band = torch.exp(-torch.abs(prob - 0.5) / max(1e-6, float(band_sigma))).clamp(0.0, 1.0)
    grad = _sobel_mag(logit_clip)

    # A smooth surrogate for the binary morphological edge channel.  It keeps
    # training differentiable while matching the meaning of the numpy channel:
    # "where would thresholded coarse morphology be unstable?"
    soft_mask = torch.sigmoid((prob - float(t_box)) / 0.06)
    k = max(1, int(morph_kernel))
    if k % 2 == 0:
        k += 1
    dil = F.max_pool2d(soft_mask, kernel_size=k, stride=1, padding=k // 2)
    ero = -F.max_pool2d(-soft_mask, kernel_size=k, stride=1, padding=k // 2)
    edge = (dil - ero).clamp(0.0, 1.0)
    return torch.cat([logit_clip, band, grad, edge], dim=1)


class FailureAwareCoarsePriorCalibrator(nn.Module):
    """Identity-initialized upstream calibrator for nnU-Net coarse logits.

    The module edits the coarse prior before it creates the SafeAnchor box,
    BB2 maps, and dense mask prompt.  It is deliberately not a post-refiner:
    the final mask is still produced by SafeAnchor/MobileSAM.
    """

    def __init__(self, cfg: CoarsePriorCalibratorCfg | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CoarsePriorCalibratorCfg()
        hidden = int(self.cfg.hidden_channels)
        in_ch = 12
        groups = 4 if hidden % 4 == 0 else 1
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
        )
        self.context = nn.Sequential(
            nn.Linear(7, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden * 2),
        )
        self.support_head = nn.Conv2d(hidden, 1, kernel_size=1, bias=True)
        self.delta_head = nn.Conv2d(hidden, 3, kernel_size=1, bias=True)
        self.point_head = nn.Conv2d(hidden, 2, kernel_size=1, bias=True)
        self.global_head = nn.Sequential(
            nn.Linear(7, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 8),
        )
        nn.init.zeros_(self.support_head.weight)
        nn.init.zeros_(self.support_head.bias)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.point_head.weight)
        nn.init.zeros_(self.point_head.bias)
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)
        nn.init.zeros_(self.global_head[-1].weight)
        nn.init.zeros_(self.global_head[-1].bias)

    @staticmethod
    def _image_gray_and_grad(image: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        x = image.to(dtype=dtype)
        if float(x.detach().amax().cpu()) > 1.5:
            x = x / 255.0
        if x.shape[1] == 1:
            gray = x
        else:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        return gray.clamp(0.0, 1.0), _sobel_mag(gray.clamp(0.0, 1.0))

    def _features(self, image: torch.Tensor, logit_p: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logit_clip = logit_p.clamp(-float(self.cfg.clip_logit), float(self.cfg.clip_logit))
        prob = torch.sigmoid(logit_p)
        uncertainty = (4.0 * prob * (1.0 - prob)).clamp(0.0, 1.0)
        grad_p = _sobel_mag(logit_clip)
        gray, grad_i = self._image_gray_and_grad(image, dtype=logit_p.dtype)
        if gray.shape[-2:] != logit_p.shape[-2:]:
            gray = F.interpolate(gray, size=logit_p.shape[-2:], mode="bilinear", align_corners=False)
            grad_i = F.interpolate(grad_i, size=logit_p.shape[-2:], mode="bilinear", align_corners=False)
        xx, yy = _coord_grid_like(logit_p)
        soft_mask = torch.sigmoid((prob - 0.4) / 0.06)
        dil = F.max_pool2d(soft_mask, kernel_size=7, stride=1, padding=3)
        ero = -F.max_pool2d(-soft_mask, kernel_size=7, stride=1, padding=3)
        outside_band = ((1.0 - prob) * dil).clamp(0.0, 1.0)
        inside_band = (prob * (1.0 - ero)).clamp(0.0, 1.0)
        morph = (dil - ero).clamp(0.0, 1.0)
        support = (
            float(self.cfg.support_floor)
            + 0.40 * uncertainty
            + 0.25 * grad_p
            + 0.20 * morph
            + 0.15 * grad_i
        ).clamp(0.0, 1.0)
        feats = torch.cat(
            [
                logit_clip / float(self.cfg.clip_logit),
                prob,
                uncertainty,
                grad_p,
                gray,
                grad_i,
                outside_band,
                inside_band,
                morph,
                support,
                xx,
                yy,
            ],
            dim=1,
        )
        aux = {
            "prob": prob,
            "support": support,
            "outside_band": outside_band,
            "inside_band": inside_band,
            "morph": morph,
        }
        return feats, aux

    def forward(
        self,
        image: torch.Tensor,
        logit_p: torch.Tensor,
        *,
        return_aux: bool = False,
    ):
        feats, aux = self._features(image, logit_p)
        prob = aux["prob"]
        context = coarse_moment_features(prob)
        h = self.stem(feats)
        gamma_beta = self.context(context).view(h.shape[0], 2, h.shape[1], 1, 1)
        gamma = 0.25 * torch.tanh(gamma_beta[:, 0])
        beta = 0.25 * torch.tanh(gamma_beta[:, 1])
        h = h * (1.0 + gamma) + beta
        base_support = aux["support"]
        support_delta = float(self.cfg.support_learn_scale) * torch.tanh(self.support_head(h))
        learned_support = (base_support + support_delta).clamp(0.0, 1.0)
        aux["base_support"] = base_support
        aux["support_delta"] = support_delta
        aux["support"] = learned_support
        raw = self.delta_head(h)
        zero = math.log(2.0)
        expand = aux["outside_band"] * (F.softplus(raw[:, 0:1]) - zero)
        shrink = aux["inside_band"] * (F.softplus(raw[:, 1:2]) - zero)
        local = aux["support"] * torch.tanh(raw[:, 2:3])

        global_raw = self.global_head(context)
        temp = torch.exp(float(self.cfg.temperature_scale) * torch.tanh(global_raw[:, 0:1])).view(-1, 1, 1, 1)
        bias = (float(self.cfg.bias_scale) * torch.tanh(global_raw[:, 1:2])).view(-1, 1, 1, 1)
        coeff = torch.tanh(global_raw[:, 2:8]).view(-1, 6, 1, 1)
        xx, yy = _coord_grid_like(logit_p)
        basis = torch.cat([torch.ones_like(xx), xx, yy, xx * xx, yy * yy, xx * yy], dim=1)
        basis_delta = (basis * coeff).sum(dim=1, keepdim=True)
        basis_delta = float(self.cfg.basis_scale) * aux["support"] * basis_delta

        point_raw = self.point_head(h)
        point_gate = float(self.cfg.point_gate_scale) * torch.clamp(torch.sigmoid(point_raw) - 0.5, min=0.0) * 2.0
        pos_point_score = point_gate[:, 0:1] * (
            0.55 * aux["outside_band"] + 0.30 * aux["support"] + 0.15 * aux["morph"]
        )
        neg_point_score = point_gate[:, 1:2] * (
            0.55 * aux["inside_band"] + 0.30 * aux["support"] + 0.15 * aux["morph"]
        )

        residual = float(self.cfg.residual_scale) * (expand - shrink + local) + basis_delta
        calibrated = logit_p / temp + bias + residual
        calibrated = torch.nan_to_num(calibrated, nan=0.0, posinf=20.0, neginf=-20.0)
        if not return_aux:
            return calibrated
        aux.update(
            {
                "residual": residual,
                "temperature": temp,
                "bias": bias,
                "expand": expand,
                "shrink": shrink,
                "local": local,
                "point_raw": point_raw,
                "pos_point_score": pos_point_score,
                "neg_point_score": neg_point_score,
            }
        )
        return calibrated, aux
