from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class RepVGGBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.rbr_dense = nn.Sequential(
            nn.Conv2d(self.in_ch, self.out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_ch),
        )
        self.rbr_1x1 = nn.Sequential(
            nn.Conv2d(self.in_ch, self.out_ch, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.out_ch),
        )
        if self.in_ch == self.out_ch:
            self.rbr_identity = nn.BatchNorm2d(self.out_ch)
        else:
            self.rbr_identity = None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rbr_dense(x) + self.rbr_1x1(x)
        if self.rbr_identity is not None:
            out = out + self.rbr_identity(x)
        return self.act(out)


def _support_from_bb2_maps(bb2_maps: torch.Tensor) -> torch.Tensor:
    if bb2_maps.ndim != 4:
        raise ValueError(f"support extraction expects BCHW, got {tuple(bb2_maps.shape)}")
    if bb2_maps.shape[1] < 2:
        return torch.zeros((bb2_maps.shape[0], 1, bb2_maps.shape[2], bb2_maps.shape[3]), device=bb2_maps.device, dtype=bb2_maps.dtype)
    band = bb2_maps[:, 1:2].clamp(0.0, 1.0)
    grad = bb2_maps[:, 2:3].clamp(0.0, 1.0) if bb2_maps.shape[1] >= 3 else band
    morph = bb2_maps[:, 3:4].clamp(0.0, 1.0) if bb2_maps.shape[1] >= 4 else band
    edge = 1.0 - (1.0 - 0.35 * grad) * (1.0 - 0.65 * morph)
    return (band + (1.0 - band) * edge).clamp(0.0, 1.0)


@dataclass
class BB2AdapterCfg:
    arch: str = "repvgg"
    in_channels: int = 4
    base_channels: int = 32
    depth: int = 4
    alpha_init: float = 0.5
    alpha_learnable: bool = True
    support_residual_channels: int = 16
    support_residual_scale: float = 0.25


class BB2PromptAdapter(nn.Module):
    def __init__(self, cfg: Optional[BB2AdapterCfg] = None) -> None:
        super().__init__()
        self.cfg = cfg or BB2AdapterCfg()
        c = int(self.cfg.base_channels)
        self.backbone = nn.Sequential(
            RepVGGBlock(int(self.cfg.in_channels), c),
            RepVGGBlock(c, c),
            RepVGGBlock(c, c),
            RepVGGBlock(c, c),
        )
        self.head = nn.Conv2d(c, 1, kernel_size=1, bias=True)
        alpha = float(self.cfg.alpha_init)
        alpha = min(max(alpha, 0.0), 1.0)
        init = torch.log(torch.tensor(alpha / max(1e-6, 1.0 - alpha), dtype=torch.float32))
        if bool(self.cfg.alpha_learnable):
            self.alpha_logit = nn.Parameter(init)
        else:
            self.register_buffer("alpha_logit", init)
        support_hidden = int(self.cfg.support_residual_channels)
        support_groups = 4 if support_hidden % 4 == 0 else 1
        self.support_residual = nn.Sequential(
            nn.Conv2d(int(self.cfg.in_channels) + 5, support_hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(support_groups, support_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(support_hidden, support_hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(support_groups, support_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(support_hidden, 1, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.support_residual[-1].weight)
        nn.init.zeros_(self.support_residual[-1].bias)
        self.register_buffer("support_residual_scale", torch.tensor(float(self.cfg.support_residual_scale), dtype=torch.float32))
        support_groups = 4 if support_hidden % 4 == 0 else 1
        self.support_context = nn.Sequential(
            nn.Conv2d(int(self.cfg.in_channels) + 5, support_hidden, kernel_size=5, padding=2, groups=1, bias=False),
            nn.GroupNorm(support_groups, support_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(support_hidden, support_hidden, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(support_groups, support_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(support_hidden, 1, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.support_context[-1].weight)
        nn.init.zeros_(self.support_context[-1].bias)

    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)

    @staticmethod
    def _grad_mag(x: torch.Tensor) -> torch.Tensor:
        dx = F.pad(x[..., :, 1:] - x[..., :, :-1], (0, 1, 0, 0))
        dy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
        return torch.sqrt(dx * dx + dy * dy + 1e-6)

    def run_bb2_adapter(
        self,
        bb2_maps: torch.Tensor,
        logit_p: Optional[torch.Tensor] = None,
        *,
        condition: Optional[dict] = None,
    ) -> torch.Tensor:
        feat = self.backbone(bb2_maps)
        bb2_logits = torch.nan_to_num(self.head(feat), nan=0.0, posinf=20.0, neginf=-20.0)
        support = None
        if isinstance(condition, dict):
            support = condition.get("support_region")
        if support is None:
            support = _support_from_bb2_maps(bb2_maps)
        else:
            support = support.to(device=bb2_maps.device, dtype=bb2_maps.dtype)
            if support.ndim == 3:
                support = support.unsqueeze(1)
            if support.shape[-2:] != bb2_maps.shape[-2:]:
                support = F.interpolate(support, size=bb2_maps.shape[-2:], mode="bilinear", align_corners=False)
            support = support.clamp(0.0, 1.0)
        if logit_p is None:
            logit_p = bb2_maps[:, :1]
        prob = torch.sigmoid(logit_p)
        uncertainty = torch.clamp(4.0 * prob * (1.0 - prob), 0.0, 1.0)
        response_gap = torch.tanh((bb2_logits - logit_p) / 6.0)
        grad_p = torch.clamp(self._grad_mag(prob) * 8.0, 0.0, 1.0)
        support_feat = torch.cat([bb2_maps, prob, uncertainty, support, response_gap, grad_p], dim=1)
        residual = self.support_residual_scale.to(dtype=bb2_logits.dtype, device=bb2_logits.device) * support * torch.tanh(self.support_residual(support_feat))
        context_residual = self.support_residual_scale.to(dtype=bb2_logits.dtype, device=bb2_logits.device) * support * torch.tanh(self.support_context(support_feat))
        residual = residual + context_residual
        return torch.nan_to_num(bb2_logits + residual, nan=0.0, posinf=20.0, neginf=-20.0)

    def forward(
        self,
        bb2_maps: torch.Tensor,
        logit_p: torch.Tensor,
        *,
        alpha_override: Optional[float] = None,
        condition: Optional[dict] = None,
        return_bb2_logits: bool = False,
    ):
        bb2_logits = self.run_bb2_adapter(bb2_maps, logit_p, condition=condition)
        alpha = self.alpha().to(device=logit_p.device, dtype=logit_p.dtype)
        if alpha_override is not None:
            alpha = torch.tensor(float(alpha_override), device=logit_p.device, dtype=logit_p.dtype)
        fused = alpha * logit_p + (1.0 - alpha) * bb2_logits
        if return_bb2_logits:
            return fused, bb2_logits
        return fused


class ConvNeXtLiteBlock(nn.Module):
    """Large-kernel depthwise block used by the clean BB2v2 branch."""

    def __init__(self, channels: int, dilation: int = 1, expansion: int = 2) -> None:
        super().__init__()
        padding = int(dilation) * 3
        hidden = int(channels) * int(expansion)
        self.dw = nn.Conv2d(
            int(channels),
            int(channels),
            kernel_size=7,
            padding=padding,
            dilation=int(dilation),
            groups=int(channels),
            bias=False,
        )
        self.norm = nn.GroupNorm(1, int(channels))
        self.pw1 = nn.Conv2d(int(channels), hidden, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, int(channels), kernel_size=1)
        nn.init.zeros_(self.pw2.weight)
        nn.init.zeros_(self.pw2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.norm(y)
        y = self.pw2(self.act(self.pw1(y)))
        return x + y


class BB2PromptAdapterV2(nn.Module):
    """Two-channel logit/gradient prompt adapter with learnable multiscale context.

    The branch is intentionally small: it does not contain hand-coded support gates,
    area heuristics, point prompts, or threshold policies.  It learns a boundary
    prompt from only the calibrated prior logit and its gradient.
    """

    def __init__(self, cfg: Optional[BB2AdapterCfg] = None) -> None:
        super().__init__()
        self.cfg = cfg or BB2AdapterCfg(arch="v2_multiscale", in_channels=2, base_channels=32, depth=4)
        c = int(self.cfg.base_channels)
        in_ch = int(self.cfg.in_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, c),
            nn.GELU(),
        )
        depth = max(1, int(self.cfg.depth))
        self.local_blocks = nn.Sequential(*[ConvNeXtLiteBlock(c, dilation=1) for _ in range(depth)])
        self.mid_block = ConvNeXtLiteBlock(c, dilation=2)
        self.wide_block = ConvNeXtLiteBlock(c, dilation=4)
        self.fuse = nn.Sequential(
            nn.Conv2d(c * 3, c, kernel_size=1, bias=False),
            nn.GroupNorm(1, c),
            nn.GELU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, c),
            nn.GELU(),
        )
        self.head = nn.Conv2d(c, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        alpha = float(self.cfg.alpha_init)
        alpha = min(max(alpha, 0.0), 1.0)
        init = torch.log(torch.tensor(alpha / max(1e-6, 1.0 - alpha), dtype=torch.float32))
        if bool(self.cfg.alpha_learnable):
            self.alpha_logit = nn.Parameter(init)
        else:
            self.register_buffer("alpha_logit", init)

    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)

    def run_bb2_adapter(
        self,
        bb2_maps: torch.Tensor,
        logit_p: Optional[torch.Tensor] = None,
        *,
        condition: Optional[dict] = None,
    ) -> torch.Tensor:
        del condition
        if bb2_maps.ndim != 4:
            raise ValueError(f"BB2v2 expects BCHW maps, got {tuple(bb2_maps.shape)}")
        if bb2_maps.shape[1] == 2:
            x = bb2_maps
            anchor = bb2_maps[:, 0:1]
        elif bb2_maps.shape[1] >= 3:
            x = torch.cat([bb2_maps[:, 0:1], bb2_maps[:, 2:3]], dim=1)
            anchor = bb2_maps[:, 0:1]
        else:
            raise ValueError(f"BB2v2 expects logit+grad maps, got {tuple(bb2_maps.shape)}")
        if logit_p is not None:
            anchor = logit_p
        feat = self.stem(torch.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0))
        local = self.local_blocks(feat)
        mid = self.mid_block(local)
        wide = self.wide_block(local)
        delta = self.head(self.fuse(torch.cat([local, mid, wide], dim=1)))
        return torch.nan_to_num(anchor + delta, nan=0.0, posinf=20.0, neginf=-20.0)

    def forward(
        self,
        bb2_maps: torch.Tensor,
        logit_p: torch.Tensor,
        *,
        alpha_override: Optional[float] = None,
        condition: Optional[dict] = None,
        return_bb2_logits: bool = False,
    ):
        bb2_logits = self.run_bb2_adapter(bb2_maps, logit_p, condition=condition)
        alpha = self.alpha().to(device=logit_p.device, dtype=logit_p.dtype)
        if alpha_override is not None:
            alpha = torch.tensor(float(alpha_override), device=logit_p.device, dtype=logit_p.dtype)
        fused = alpha * logit_p + (1.0 - alpha) * bb2_logits
        if return_bb2_logits:
            return fused, bb2_logits
        return fused


def load_bb2_adapter(checkpoint_path: Path | str, device: torch.device, alpha_override: Optional[float] = None) -> nn.Module:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    cfg_dict = dict(ckpt["cfg"])
    arch = str(cfg_dict.get("arch", "repvgg"))
    cfg = BB2AdapterCfg(
        arch=arch,
        in_channels=int(cfg_dict["in_channels"]),
        base_channels=int(cfg_dict["base_channels"]),
        depth=int(cfg_dict["depth"]),
        alpha_init=float(cfg_dict["alpha_init"]),
        alpha_learnable=bool(cfg_dict["alpha_learnable"]),
        support_residual_channels=int(cfg_dict.get("support_residual_channels", 16)),
        support_residual_scale=float(cfg_dict.get("support_residual_scale", 0.25)),
    )
    if arch in {"v2_multiscale", "bb2v2", "multiscale"}:
        model = BB2PromptAdapterV2(cfg)
    else:
        model = BB2PromptAdapter(cfg)
    missing, unexpected = model.load_state_dict(ckpt["adapter"], strict=False)
    unexpected = [k for k in unexpected if k]
    if unexpected:
        raise RuntimeError(f"Unexpected BB2 checkpoint keys: {unexpected[:8]}")
    allowed_missing = [
        k for k in missing
        if (
            not str(k).startswith("support_residual.")
            and not str(k).startswith("support_context.")
            and str(k) != "support_residual_scale"
        )
    ]
    if allowed_missing:
        raise RuntimeError(f"Missing required BB2 checkpoint keys: {allowed_missing[:8]}")
    model.to(device)
    model.eval()
    if alpha_override is not None:
        with torch.no_grad():
            a = float(alpha_override)
            model.alpha_logit.copy_(torch.log(torch.tensor(a / max(1e-6, 1.0 - a), device=device)))
    return model
