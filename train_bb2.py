from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from safeanchor.bb2_adapter import BB2AdapterCfg, BB2PromptAdapter, BB2PromptAdapterV2
from safeanchor.bb2_channel_prune import prune_bb2_channels_torch
from safeanchor.coarse_prior_calibrator import build_bb2_maps_from_logit
from safeanchor.generalizable_prior_calibrator import dice_bce_logits, gradient_magnitude, uncertainty_weighted_sum
from safeanchor.gpc_data import PriorDataset, collate_pad, load_json, prob_to_bbox_torch, resolve_path
from safeanchor.mobile_sam_wrapper import FrozenMobileSAM


class BB2TrainingLoss(nn.Module):
    """Learned-weight segmentation and boundary supervision for BB2 training."""

    def __init__(self, use_boundary: bool = True) -> None:
        super().__init__()
        self.use_boundary = bool(use_boundary)
        self.loss_log_vars = nn.Parameter(torch.zeros(2 if self.use_boundary else 1, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        if logits.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
        seg = dice_bce_logits(logits, target)
        losses = [seg]
        parts = {"seg": float(seg.detach().cpu())}
        if self.use_boundary:
            boundary = F.smooth_l1_loss(gradient_magnitude(torch.sigmoid(logits)), gradient_magnitude(target))
            losses.append(boundary)
            parts["boundary"] = float(boundary.detach().cpu())
        total = uncertainty_weighted_sum(losses, self.loss_log_vars[: len(losses)])
        parts["loss"] = float(total.detach().cpu())
        return total, parts


def _write_history(out_dir: Path, history: list[dict]) -> None:
    (out_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if not history:
        return
    keys = list(history[0].keys())
    with (out_dir / "train_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def _build_adapter(args: argparse.Namespace) -> nn.Module:
    arch = str(args.arch).lower().strip()
    if arch in {"v2_multiscale", "bb2v2", "multiscale"}:
        cfg = BB2AdapterCfg(
            arch="v2_multiscale",
            in_channels=2,
            base_channels=int(args.base_channels),
            depth=int(args.depth),
            alpha_init=float(args.alpha_init),
            alpha_learnable=bool(args.alpha_learnable),
        )
        return BB2PromptAdapterV2(cfg)
    cfg = BB2AdapterCfg(
        arch="repvgg",
        in_channels=4,
        base_channels=int(args.base_channels),
        depth=int(args.depth),
        alpha_init=float(args.alpha_init),
        alpha_learnable=bool(args.alpha_learnable),
    )
    return BB2PromptAdapter(cfg)


def _select_prompt(adapter: nn.Module, maps: torch.Tensor, logit: torch.Tensor, prompt_mode: str) -> torch.Tensor:
    mode = str(prompt_mode).lower().strip()
    if mode == "bb2_logits":
        return adapter.run_bb2_adapter(maps, logit)
    if mode == "adapter_mix":
        return adapter(maps, logit)
    raise ValueError(f"unknown BB2 training prompt mode: {prompt_mode}")


def _sam_trainable_parameters(sam: FrozenMobileSAM, train_prompt_decoder: bool) -> list[nn.Parameter]:
    for p in sam.sam.parameters():
        p.requires_grad = False
    if not bool(train_prompt_decoder):
        return []
    for module_name in ("prompt_encoder", "mask_decoder"):
        module = getattr(sam.sam, module_name)
        for p in module.parameters():
            p.requires_grad = True
    return [p for p in sam.sam.parameters() if p.requires_grad]


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def main() -> None:
    root_default = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Train the BB2 boundary prompt adapter for the clean GPC mainline.")
    ap.add_argument("--root", type=str, default=str(root_default))
    ap.add_argument("--config", type=str, default="configs/branch_safeanchor_gpc_v8c_paper_release.json")
    ap.add_argument("--dataset", choices=["kvasir", "ph2", "idrid", "dfuc", "busi", "isic2017", "tn3k"], required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5.0e-5)
    ap.add_argument("--sam_lr", type=float, default=1.0e-5)
    ap.add_argument("--weight_decay", type=float, default=1.0e-4)
    ap.add_argument("--max_train_side", type=int, default=768)
    ap.add_argument("--arch", type=str, default="v2_multiscale", choices=["v2_multiscale", "repvgg"])
    ap.add_argument("--base_channels", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--alpha_init", type=float, default=0.5)
    ap.add_argument("--alpha_learnable", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--bb2_channel_mode", type=str, default="logit_grad2")
    ap.add_argument("--prompt_mode", type=str, default="bb2_logits", choices=["bb2_logits", "adapter_mix"])
    ap.add_argument("--train_sam_prompt_decoder", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--use_boundary_loss", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--print_every", type=int, default=110)
    ap.add_argument("--max_steps", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    cfg = load_json(resolve_path(root, args.config))
    split_cfg = cfg[f"{args.dataset}_train"]
    device = torch.device(str(args.device))
    dataset = PriorDataset(root=root, split_cfg=split_cfg, max_train_side=int(args.max_train_side))
    default_name = f"bb2v2_{args.dataset}_train{len(dataset)}_3ep_clean"
    out_dir = resolve_path(root, args.out_dir) if str(args.out_dir).strip() else root / "outputs" / default_name
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=collate_pad)
    adapter = _build_adapter(args).to(device)
    loss_model = BB2TrainingLoss(use_boundary=bool(args.use_boundary_loss)).to(device)
    sam = FrozenMobileSAM(
        checkpoint=resolve_path(root, cfg["mobile_sam_ckpt"]),
        model_type="vit_t",
        device=str(device),
        trainable=True,
    )
    sam_train_params = _sam_trainable_parameters(sam, bool(args.train_sam_prompt_decoder))
    param_groups: list[dict] = [
        {"params": list(adapter.parameters()) + list(loss_model.parameters()), "lr": float(args.lr)},
    ]
    if sam_train_params:
        param_groups.append({"params": sam_train_params, "lr": float(args.sam_lr)})
    opt = torch.optim.AdamW(param_groups, weight_decay=float(args.weight_decay))

    manifest = {
        "method": "SafeAnchor-BB2v2" if str(args.arch) == "v2_multiscale" else "SafeAnchor-BB2-RepVGG",
        "dataset": str(args.dataset),
        "train_cases": len(dataset),
        "same_architecture_for_all_release_datasets": True,
        "coarse_source": "dataset-specific converged nnUNet probability maps from the release split",
        "frozen": {"nnunet": True, "sam_image_encoder": True},
        "trainable": {
            "bb2_adapter": True,
            "sam_prompt_encoder": bool(args.train_sam_prompt_decoder),
            "sam_mask_decoder": bool(args.train_sam_prompt_decoder),
            "loss_log_vars": True,
        },
        "bb2_channels": "logit_and_gradient_only" if str(args.bb2_channel_mode).lower().strip() in {"logit_grad2", "lg2", "two"} else str(args.bb2_channel_mode),
        "prompt_mode": str(args.prompt_mode),
        "selection_protocol": "checkpoint_last_only",
        "args": vars(args),
        "max_train_side": int(args.max_train_side),
    }
    (out_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    history: list[dict] = []
    global_step = 0
    for epoch in range(int(args.epochs)):
        adapter.train()
        loss_model.train()
        sam.sam.prompt_encoder.train(bool(args.train_sam_prompt_decoder))
        sam.sam.mask_decoder.train(bool(args.train_sam_prompt_decoder))
        epoch_parts = {"loss": 0.0, "seg": 0.0, "boundary": 0.0}
        epoch_count = 0
        for batch in loader:
            global_step += 1
            image = batch["image"].to(device)
            gt = batch["gt"].to(device)
            logit = batch["logit_raw"].to(device)
            maps = prune_bb2_channels_torch(build_bb2_maps_from_logit(logit), str(args.bb2_channel_mode))
            prompt = _select_prompt(adapter, maps, logit, str(args.prompt_mode))
            boxes = prob_to_bbox_torch(torch.sigmoid(logit))
            image_embeddings, info = sam.encode_image(image)
            pred, _ = sam.decode_from_embeddings(
                image_embeddings,
                info,
                boxes_norm=boxes,
                point_coords_norm=None,
                point_labels=None,
                mask_prompt=None,
                bb2_mask_prompt_logits=prompt,
            )
            loss, parts = loss_model(pred, gt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(adapter.parameters()) + sam_train_params + list(loss_model.parameters()), 1.0)
            opt.step()
            for key in epoch_parts:
                epoch_parts[key] += float(parts.get(key, 0.0))
            epoch_count += 1
            if global_step % int(args.print_every) == 0:
                print(
                    "[bb2] "
                    f"epoch={epoch + 1} step={global_step} "
                    f"loss={parts['loss']:.6f} seg={parts['seg']:.6f} "
                    f"boundary={parts.get('boundary', 0.0):.6f} alpha={float(adapter.alpha().detach().cpu()):.4f}",
                    flush=True,
                )
            if int(args.max_steps) > 0 and global_step >= int(args.max_steps):
                break
        row = {
            "epoch": epoch + 1,
            "step": global_step,
            "mean_loss": epoch_parts["loss"] / max(1, epoch_count),
            "mean_seg": epoch_parts["seg"] / max(1, epoch_count),
            "mean_boundary": epoch_parts["boundary"] / max(1, epoch_count),
            "alpha": float(adapter.alpha().detach().cpu()),
        }
        for i, value in enumerate(loss_model.loss_log_vars.detach().cpu().tolist()):
            row[f"loss_log_var_{i}"] = float(value)
        history.append(row)
        _write_history(out_dir, history)
        torch.save(
            {
                "epoch": epoch + 1,
                "mean_loss": row["mean_loss"],
                "adapter": _cpu_state_dict(adapter),
                "cfg": adapter.cfg.__dict__,
                "sam_model_type": "vit_t",
                "sam_state_dict": _cpu_state_dict(sam.sam) if bool(args.train_sam_prompt_decoder) else None,
                "loss_state_dict": _cpu_state_dict(loss_model),
                "history": history,
                "manifest": manifest,
                "args": vars(args),
            },
            out_dir / "checkpoint_last.pth",
        )
        if int(args.max_steps) > 0 and global_step >= int(args.max_steps):
            break
    print(f"[bb2] checkpoint written to: {out_dir / 'checkpoint_last.pth'}", flush=True)


if __name__ == "__main__":
    main()
