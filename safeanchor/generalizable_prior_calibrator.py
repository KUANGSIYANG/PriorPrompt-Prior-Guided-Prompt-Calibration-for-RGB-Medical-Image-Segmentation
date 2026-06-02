from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy.ndimage import distance_transform_edt

from safeanchor.dbp_prior import gradient_magnitude, soft_moment_box, uncertainty_weighted_sum


@dataclass
class GPCCfg:
    hidden_channels: int = 24
    depth: int = 4
    encoder_arch: str = "convnext_lite"
    context_kernel_size: int = 13
    context_expansion: int = 3
    layer_scale_init: float = 1.0e-3
    residual_scale_init: float = 0.05
    alpha_init: float = 0.78
    box_scale_init: float = 2.5
    temperature_init: float = 1.0
    logit_clip: float = 8.0
    use_residual: bool = True
    use_case_alpha: bool = True
    use_spatial_alpha: bool = False
    spatial_alpha_scale_init: float = 0.25
    use_temperature: bool = True
    use_learned_threshold: bool = False
    use_binary_shape_losses: bool = False
    use_binary_distance_loss: bool = False
    use_distill_loss: bool = True
    global_delta_scale: float = 0.25
    use_learned_global_scale: bool = False
    use_global_tanh: bool = True
    threshold_init: float = 0.6
    threshold_sharpness_init: float = 12.0


def _inverse_softplus(value: float) -> torch.Tensor:
    v = torch.tensor(max(float(value), 1.0e-6), dtype=torch.float32)
    return torch.log(torch.expm1(v))


def _logit01(value: float) -> torch.Tensor:
    v = min(max(float(value), 1.0e-6), 1.0 - 1.0e-6)
    return torch.log(torch.tensor(v / (1.0 - v), dtype=torch.float32))


def image_luma_and_grad(image: torch.Tensor, *, size: tuple[int, int], dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    x = image.to(dtype=dtype)
    if float(x.detach().amax().cpu()) > 1.5:
        x = x / 255.0
    if x.shape[1] == 1:
        luma = x
    else:
        luma = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).clamp(0.0, 1.0)
    if luma.shape[-2:] != size:
        luma = F.interpolate(luma, size=size, mode="bilinear", align_corners=False)
    return luma, gradient_magnitude(luma)


class LargeKernelResidualBlock(nn.Module):
    """Small ConvNeXt-style block for coarse-prior calibration."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 4 if channels % 4 == 0 else 1
        self.dw = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False)
        self.norm = nn.GroupNorm(groups, channels)
        self.pw1 = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.pw2 = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.norm(y)
        y = self.pw2(F.gelu(self.pw1(y)))
        return x + y


class GlobalResponseNorm2d(nn.Module):
    """ConvNeXt-V2 style global response normalization for BCHW tensors."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, int(channels), 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, int(channels), 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(-2, -1), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True).clamp_min(1.0e-6))
        return x + self.gamma * (x * nx) + self.beta


class ContextMixerBlock(nn.Module):
    """Higher-capacity prior context mixer for the v9 research branch.

    The block keeps the V8C philosophy: all complexity stays inside the encoder,
    while the output degrees of freedom remain alpha, box scale, and threshold.
    """

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 13,
        expansion: int = 3,
        layer_scale_init: float = 1.0e-3,
    ) -> None:
        super().__init__()
        channels = int(channels)
        k = max(3, int(kernel_size))
        if k % 2 == 0:
            k += 1
        hidden = channels * max(2, int(expansion))
        groups = 8 if channels % 8 == 0 else 4 if channels % 4 == 0 else 1
        self.local_dw = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False)
        self.context_dw = nn.Conv2d(channels, channels, kernel_size=k, padding=k // 2, groups=channels, bias=False)
        self.norm = nn.GroupNorm(groups, channels)
        self.pw_gate = nn.Conv2d(channels, hidden * 2, kernel_size=1)
        self.grn = GlobalResponseNorm2d(hidden)
        self.pw_out = nn.Conv2d(hidden, channels, kernel_size=1)
        self.layer_scale = nn.Parameter(torch.full((1, channels, 1, 1), float(layer_scale_init), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.local_dw(x) + self.context_dw(x)
        y = self.norm(y)
        u, v = self.pw_gate(y).chunk(2, dim=1)
        y = F.gelu(u) * torch.sigmoid(v)
        y = self.pw_out(self.grn(y))
        return x + self.layer_scale.to(dtype=x.dtype, device=x.device) * y


class GeneralizablePriorCalibrator(nn.Module):
    """Learnable coarse-prior calibration without hand-crafted gates.

    GPC only edits the upstream coarse logit and predicts continuous SafeAnchor
    prompt parameters.  It avoids area gates, point prompts, candidate routing,
    and thresholded boxes inside the model; boxes are soft moments of the
    calibrated prior.
    """

    def __init__(self, cfg: GPCCfg | None = None) -> None:
        super().__init__()
        self.cfg = cfg or GPCCfg()
        h = int(self.cfg.hidden_channels)
        in_ch = 6
        groups = 8 if h % 8 == 0 else 4 if h % 4 == 0 else 1
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, h, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, h),
            nn.SiLU(inplace=True),
        )
        arch = str(self.cfg.encoder_arch).lower().strip()
        if arch in {"convnext_lite", "v8c", "large_kernel"}:
            self.blocks = nn.Sequential(*[LargeKernelResidualBlock(h) for _ in range(max(1, int(self.cfg.depth)))])
        elif arch in {"context_mixer", "v9_context", "gpc_v9"}:
            self.blocks = nn.Sequential(
                *[
                    ContextMixerBlock(
                        h,
                        kernel_size=int(self.cfg.context_kernel_size),
                        expansion=int(self.cfg.context_expansion),
                        layer_scale_init=float(self.cfg.layer_scale_init),
                    )
                    for _ in range(max(1, int(self.cfg.depth)))
                ]
            )
        else:
            raise ValueError(f"unsupported GPC encoder_arch: {self.cfg.encoder_arch}")
        self.residual_head = nn.Conv2d(h, 1, kernel_size=1)
        if bool(self.cfg.use_spatial_alpha):
            self.spatial_alpha_head = nn.Conv2d(h, 1, kernel_size=1)
        global_outputs = 4 if bool(self.cfg.use_learned_threshold) else 3
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(h, h, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(h, global_outputs, kernel_size=1),
        )
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        if bool(self.cfg.use_spatial_alpha):
            nn.init.zeros_(self.spatial_alpha_head.weight)
            nn.init.zeros_(self.spatial_alpha_head.bias)
        nn.init.zeros_(self.global_head[-1].weight)
        nn.init.zeros_(self.global_head[-1].bias)
        self.residual_scale = nn.Parameter(_inverse_softplus(float(self.cfg.residual_scale_init)))
        self.alpha_logit = nn.Parameter(_logit01(float(self.cfg.alpha_init)))
        if bool(self.cfg.use_spatial_alpha):
            self.spatial_alpha_scale_log = nn.Parameter(_inverse_softplus(float(self.cfg.spatial_alpha_scale_init)))
        self.box_scale_log = nn.Parameter(_inverse_softplus(float(self.cfg.box_scale_init)))
        self.temperature_log = nn.Parameter(_inverse_softplus(float(self.cfg.temperature_init)))
        if bool(self.cfg.use_learned_global_scale):
            self.global_delta_scale_log = nn.Parameter(_inverse_softplus(float(self.cfg.global_delta_scale)))
        if bool(self.cfg.use_learned_threshold):
            self.threshold_logit = nn.Parameter(_logit01(float(self.cfg.threshold_init)))
            self.threshold_sharpness_log = nn.Parameter(_inverse_softplus(float(self.cfg.threshold_sharpness_init)))
        loss_count = 5 + int(bool(self.cfg.use_distill_loss))
        if bool(self.cfg.use_learned_threshold):
            loss_count += 1
        if bool(self.cfg.use_learned_threshold) and bool(self.cfg.use_binary_shape_losses):
            loss_count += 2
        if bool(self.cfg.use_learned_threshold) and bool(self.cfg.use_binary_distance_loss):
            loss_count += 1
        self.loss_log_vars = nn.Parameter(torch.zeros(loss_count, dtype=torch.float32))

    def cfg_dict(self) -> dict:
        return asdict(self.cfg)

    def _features(self, image: torch.Tensor, logit_base: torch.Tensor) -> torch.Tensor:
        clip = float(self.cfg.logit_clip)
        logit_clip = logit_base.clamp(-clip, clip)
        prob = torch.sigmoid(logit_clip)
        uncertainty = (4.0 * prob * (1.0 - prob)).clamp(0.0, 1.0)
        grad_l = gradient_magnitude(logit_clip)
        luma, grad_i = image_luma_and_grad(image, size=logit_base.shape[-2:], dtype=logit_base.dtype)
        return torch.cat([logit_clip / clip, prob, uncertainty, grad_l, luma, grad_i], dim=1)

    def forward(self, image: torch.Tensor, logit_base: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.blocks(self.stem(self._features(image, logit_base)))
        residual_scale = F.softplus(self.residual_scale).to(device=logit_base.device, dtype=logit_base.dtype)
        if bool(self.cfg.use_residual):
            residual = residual_scale * torch.tanh(self.residual_head(feat))
        else:
            residual = torch.zeros_like(logit_base)
        logit_cal = torch.nan_to_num(logit_base + residual, nan=0.0, posinf=20.0, neginf=-20.0)
        global_delta = self.global_head(feat).flatten(1)
        if bool(self.cfg.use_global_tanh):
            global_delta = torch.tanh(global_delta)
        if bool(self.cfg.use_learned_global_scale):
            global_scale = F.softplus(self.global_delta_scale_log).to(device=logit_base.device, dtype=logit_base.dtype)
        else:
            global_scale = torch.tensor(float(self.cfg.global_delta_scale), device=logit_base.device, dtype=logit_base.dtype)
        global_delta = global_scale * global_delta
        if bool(self.cfg.use_case_alpha):
            alpha_logit = (self.alpha_logit + global_delta[:, 0:1]).view(-1, 1, 1, 1)
        else:
            alpha_logit = self.alpha_logit.view(1, 1, 1, 1).expand(logit_base.shape[0], 1, 1, 1)
        if bool(self.cfg.use_spatial_alpha):
            spatial_scale = F.softplus(self.spatial_alpha_scale_log).to(device=logit_base.device, dtype=logit_base.dtype)
            alpha_logit = alpha_logit + spatial_scale * torch.tanh(self.spatial_alpha_head(feat))
        alpha = torch.sigmoid(alpha_logit)
        box_scale = F.softplus(self.box_scale_log + global_delta[:, 1]).clamp_min(0.5)
        if bool(self.cfg.use_temperature):
            temperature = F.softplus(self.temperature_log + global_delta[:, 2]).view(-1, 1, 1, 1).clamp_min(0.5)
        else:
            temperature = torch.ones((logit_base.shape[0], 1, 1, 1), device=logit_base.device, dtype=logit_base.dtype)
        if bool(self.cfg.use_learned_threshold):
            threshold = torch.sigmoid(self.threshold_logit + global_delta[:, 3:4]).view(-1, 1, 1, 1)
            threshold_sharpness = F.softplus(self.threshold_sharpness_log).to(device=logit_base.device, dtype=logit_base.dtype).clamp_min(1.0)
        else:
            threshold = torch.full(
                (logit_base.shape[0], 1, 1, 1),
                float(self.cfg.threshold_init),
                device=logit_base.device,
                dtype=logit_base.dtype,
            )
            threshold_sharpness = torch.tensor(float(self.cfg.threshold_sharpness_init), device=logit_base.device, dtype=logit_base.dtype)
        boxes = soft_moment_box(torch.sigmoid(logit_cal), box_scale)
        return {
            "logit_cal": logit_cal,
            "residual": residual,
            "alpha": alpha.to(device=logit_base.device, dtype=logit_base.dtype),
            "boxes": boxes,
            "box_scale": box_scale,
            "temperature": temperature.to(device=logit_base.device, dtype=logit_base.dtype),
            "threshold": threshold.to(device=logit_base.device, dtype=logit_base.dtype),
            "threshold_sharpness": threshold_sharpness.view(1),
        }


def dice_bce_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = torch.sum(prob * target, dim=(-2, -1))
    denom = torch.sum(prob, dim=(-2, -1)) + torch.sum(target, dim=(-2, -1))
    dice = torch.mean(1.0 - (2.0 * inter + 1.0) / (denom + 1.0))
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return dice + bce


def normalized_target_distance(target: torch.Tensor) -> torch.Tensor:
    """Per-case distance-to-boundary weights for soft binary supervision.

    The map is computed from GT only, normalized by its own maximum, and used
    to make far false positives/negatives more expensive without adding an
    inference-time gate or a hand-tuned spatial threshold.
    """

    with torch.no_grad():
        target_np = target.detach().float().cpu().numpy()
        weights: list[np.ndarray] = []
        for case in target_np:
            mask = case[0] > 0.5
            if mask.any() and (~mask).any():
                inside = distance_transform_edt(mask)
                outside = distance_transform_edt(~mask)
                dist = np.where(mask, inside, outside).astype(np.float32)
                max_dist = float(dist.max())
                if max_dist > 1.0e-6:
                    dist = dist / max_dist
            else:
                dist = np.zeros_like(mask, dtype=np.float32)
            weights.append(dist[None])
        weight = torch.from_numpy(np.stack(weights, axis=0))
    return weight.to(device=target.device, dtype=target.dtype)


def soft_iou_box_loss(pred_box: torch.Tensor, target_box: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(pred_box[:, 0], target_box[:, 0])
    y1 = torch.maximum(pred_box[:, 1], target_box[:, 1])
    x2 = torch.minimum(pred_box[:, 2], target_box[:, 2])
    y2 = torch.minimum(pred_box[:, 3], target_box[:, 3])
    inter = (x2 - x1).clamp_min(0.0) * (y2 - y1).clamp_min(0.0)
    area_p = (pred_box[:, 2] - pred_box[:, 0]).clamp_min(0.0) * (pred_box[:, 3] - pred_box[:, 1]).clamp_min(0.0)
    area_t = (target_box[:, 2] - target_box[:, 0]).clamp_min(0.0) * (target_box[:, 3] - target_box[:, 1]).clamp_min(0.0)
    union = (area_p + area_t - inter).clamp_min(1.0e-6)
    return torch.mean(1.0 - inter / union)


def gpc_loss(
    *,
    model: GeneralizablePriorCalibrator,
    final_logits: torch.Tensor,
    gt: torch.Tensor,
    logit_cal: torch.Tensor,
    logit_base: torch.Tensor,
    boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    teacher_logits: torch.Tensor | None,
    threshold: torch.Tensor | None = None,
    threshold_sharpness: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    final = dice_bce_logits(final_logits, gt)
    prior = dice_bce_logits(logit_cal, gt)
    boundary = F.smooth_l1_loss(gradient_magnitude(torch.sigmoid(logit_cal)), gradient_magnitude(gt))
    box = soft_iou_box_loss(boxes, gt_boxes)
    anchor = F.smooth_l1_loss(logit_cal, logit_base.detach())
    distill = torch.zeros((), device=final_logits.device, dtype=final_logits.dtype)
    if teacher_logits is not None:
        distill = F.smooth_l1_loss(final_logits, teacher_logits.detach())
    losses = [final, prior, boundary, box, anchor]
    log_var_indices = [0, 1, 2, 3, 4]
    next_log_var = 5
    if bool(model.cfg.use_distill_loss):
        if teacher_logits is not None:
            losses.append(distill)
            log_var_indices.append(next_log_var)
        next_log_var += 1
    binarize = torch.zeros((), device=final_logits.device, dtype=final_logits.dtype)
    binary_distance = torch.zeros((), device=final_logits.device, dtype=final_logits.dtype)
    if bool(model.cfg.use_learned_threshold) and threshold is not None:
        sharpness = threshold_sharpness
        if sharpness is None:
            sharpness = torch.tensor(float(model.cfg.threshold_sharpness_init), device=final_logits.device, dtype=final_logits.dtype)
        prob = torch.sigmoid(final_logits)
        soft_binary_logits = sharpness.to(device=final_logits.device, dtype=final_logits.dtype) * (prob - threshold.to(device=final_logits.device, dtype=final_logits.dtype))
        binarize = dice_bce_logits(soft_binary_logits, gt)
        losses.append(binarize)
        log_var_indices.append(next_log_var)
        next_log_var += 1
        soft_binary = torch.sigmoid(soft_binary_logits)
        if bool(model.cfg.use_binary_shape_losses):
            binary_boundary = F.smooth_l1_loss(gradient_magnitude(soft_binary), gradient_magnitude(gt))
            binary_area = F.smooth_l1_loss(soft_binary.mean(dim=(-2, -1)), gt.mean(dim=(-2, -1)))
            losses.extend([binary_boundary, binary_area])
            log_var_indices.extend([next_log_var, next_log_var + 1])
            next_log_var += 2
        else:
            binary_boundary = torch.zeros((), device=final_logits.device, dtype=final_logits.dtype)
            binary_area = torch.zeros((), device=final_logits.device, dtype=final_logits.dtype)
        if bool(model.cfg.use_binary_distance_loss):
            distance_weight = normalized_target_distance(gt)
            binary_distance = torch.mean(F.smooth_l1_loss(soft_binary, gt, reduction="none") * distance_weight)
            losses.append(binary_distance)
            log_var_indices.append(next_log_var)
    else:
        binary_boundary = torch.zeros((), device=final_logits.device, dtype=final_logits.dtype)
        binary_area = torch.zeros((), device=final_logits.device, dtype=final_logits.dtype)
    selected_log_vars = model.loss_log_vars[torch.tensor(log_var_indices, device=model.loss_log_vars.device)]
    total = uncertainty_weighted_sum(losses, selected_log_vars)
    return total, {
        "final": float(final.detach().cpu()),
        "prior": float(prior.detach().cpu()),
        "boundary": float(boundary.detach().cpu()),
        "box": float(box.detach().cpu()),
        "anchor": float(anchor.detach().cpu()),
        "distill": float(distill.detach().cpu()),
        "binarize": float(binarize.detach().cpu()),
        "binary_boundary": float(binary_boundary.detach().cpu()),
        "binary_area": float(binary_area.detach().cpu()),
        "binary_distance": float(binary_distance.detach().cpu()),
    }
