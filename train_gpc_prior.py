from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from safeanchor.bb2_adapter import load_bb2_adapter
from safeanchor.bb2_channel_prune import prune_bb2_channels_torch
from safeanchor.coarse_prior_calibrator import build_bb2_maps_from_logit
from safeanchor.gpc_data import (
    PriorDataset,
    collate_pad,
    load_coarse_calibrator,
    load_json,
    prob_to_bbox_torch,
    resolve_path,
    run_bb2_logits,
)
from safeanchor.generalizable_prior_calibrator import GPCCfg, GeneralizablePriorCalibrator, gpc_loss
from safeanchor.mobile_sam_wrapper import FrozenMobileSAM


def _average_state(avg: dict[str, torch.Tensor], state: dict[str, torch.Tensor], n: int) -> dict[str, torch.Tensor]:
    if not avg:
        return {k: v.detach().cpu().clone() for k, v in state.items()}
    out = {}
    for key, value in state.items():
        out[key] = avg[key] + (value.detach().cpu() - avg[key]) / float(n)
    return out


def _write_history(out_dir: Path, history: list[dict]) -> None:
    (out_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if not history:
        return
    keys = list(history[0].keys())
    with (out_dir / "train_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def build_argparser() -> argparse.ArgumentParser:
    root_default = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(root_default))
    ap.add_argument("--config", type=str, default="configs/branch_safeanchor_gpc_v1.json")
    ap.add_argument("--dataset", type=str, default="kvasir", choices=["kvasir", "ph2", "idrid", "dfuc", "busi", "isic2017", "tn3k"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5.0e-5)
    ap.add_argument("--weight_decay", type=float, default=1.0e-4)
    ap.add_argument("--max_train_side", type=int, default=768)
    ap.add_argument("--seed", type=int, default=20260522)
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--print_every", type=int, default=220)
    ap.add_argument("--save_step_ckpts", action="store_true")
    ap.add_argument("--teacher_mode", type=str, default="config", choices=["config", "fixed078", "none"])
    ap.add_argument("--teacher_alpha", type=float, default=-1.0)
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    root = Path(args.root).resolve()
    cfg = load_json(resolve_path(root, args.config))
    device = torch.device(str(args.device))
    teacher_mode = str(args.teacher_mode)
    if teacher_mode == "config":
        teacher_mode = str(cfg.get("teacher_mode", "fixed078"))
    teacher_alpha = float(cfg.get("teacher_alpha", 0.78))
    if float(args.teacher_alpha) >= 0.0:
        teacher_alpha = float(args.teacher_alpha)
    if teacher_mode not in {"fixed078", "none"}:
        raise ValueError(f"unsupported teacher_mode: {teacher_mode}")
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset_name = str(args.dataset)
    split_key = f"{dataset_name}_train"
    if split_key not in cfg:
        raise KeyError(f"missing train split config: {split_key}")
    split_cfg = cfg[split_key]
    ds = PriorDataset(root=root, split_cfg=split_cfg, max_train_side=int(args.max_train_side))
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_pad,
        generator=torch.Generator().manual_seed(seed),
    )

    sam_ckpt_key = f"{dataset_name}_sam_ckpt"
    if sam_ckpt_key not in cfg:
        raise KeyError(f"missing SAM/BB2 checkpoint config: {sam_ckpt_key}")
    sam_ckpt_path = resolve_path(root, cfg[sam_ckpt_key])
    sam_ckpt = torch.load(str(sam_ckpt_path), map_location="cpu")
    adapter = load_bb2_adapter(sam_ckpt_path, device=device)
    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad = False
    base_coarse = None
    if bool(split_cfg.get("use_base_coarse_calibrator", False)):
        base_coarse = load_coarse_calibrator(resolve_path(root, cfg["coarse_calib_ckpt"]), device)
    sam = FrozenMobileSAM(
        checkpoint=resolve_path(root, cfg["mobile_sam_ckpt"]),
        model_type=str(sam_ckpt.get("sam_model_type", "vit_t")),
        device=str(device),
        sam_state_dict=sam_ckpt.get("sam_state_dict"),
        trainable=False,
    )
    sam.eval()

    gpc_cfg = dict(cfg.get("gpc_cfg", {}))
    if teacher_mode == "none":
        gpc_cfg["use_distill_loss"] = False
    model = GeneralizablePriorCalibrator(GPCCfg(**gpc_cfg)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay), foreach=False)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        default_suffix = "teacherfree" if teacher_mode == "none" else "v1"
        out_dir = root / (str(args.out_dir).strip() or f"outputs/gpc_{dataset_name}_train{len(ds)}_3ep_{default_suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": "SafeAnchor-GPC",
        "dataset": dataset_name,
        "train_ids": len(ds),
        "epochs": int(args.epochs),
        "max_train_side": int(args.max_train_side),
        "max_steps": int(args.max_steps),
        "single_path": True,
        "uses_area_gate": False,
        "uses_candidate_gate": False,
        "uses_point_prompt": False,
        "uses_fixed_alpha_at_inference": False,
        "uses_thresholded_box_in_model": False,
        "bb2_channels": str(cfg.get("bb2_channel_mode", "logit_grad")),
        "box": "learned soft-moment box from calibrated coarse probability",
        "loss_weighting": "homoscedastic_uncertainty_learned",
        "teacher_mode": teacher_mode,
        "teacher_alpha": teacher_alpha if teacher_mode == "fixed078" else None,
        "uses_distill_loss": bool(model.cfg.use_distill_loss),
        "base_coarse_calibrator": str(cfg["coarse_calib_ckpt"]) if base_coarse is not None else "",
        "sam_ckpt": str(sam_ckpt_path),
        "cfg": model.cfg_dict(),
        "seed": seed,
    }
    (out_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    history: list[dict] = []
    sums = {
        k: 0.0
        for k in [
            "loss",
            "final",
            "prior",
            "boundary",
            "box",
            "anchor",
            "distill",
            "binarize",
            "binary_boundary",
            "binary_area",
            "binary_distance",
            "residual",
            "alpha",
            "alpha_std",
            "alpha_min",
            "alpha_max",
            "box_scale",
            "temperature",
            "threshold",
            "threshold_sharpness",
            "w_final",
            "w_prior",
            "w_boundary",
            "w_box",
            "w_anchor",
            "w_distill",
            "w_binarize",
            "w_binary_boundary",
            "w_binary_area",
            "w_binary_distance",
        ]
    }
    step = 0
    window_count = 0
    avg_state: dict[str, torch.Tensor] = {}
    avg_n = 0

    def _save(path: Path, state_dict: dict[str, torch.Tensor]) -> None:
        torch.save(
            {
                "gpc_state_dict": state_dict,
                "gpc_cfg": model.cfg_dict(),
                "manifest": manifest,
                "history": history,
                "step": int(step),
                "sam_model_type": str(sam_ckpt.get("sam_model_type", "vit_t")),
            },
            path,
        )

    def _flush_history(tag: str) -> None:
        nonlocal sums, window_count
        if window_count <= 0:
            return
        row = {k: sums[k] / float(window_count) for k in sums}
        row.update({"step": step, "epoch": epoch, "window_n": window_count, "tag": tag})
        history.append(row)
        print(json.dumps(row), flush=True)
        _write_history(out_dir, history)
        sums = {k: 0.0 for k in sums}
        window_count = 0

    model.train()
    for epoch in range(1, int(args.epochs) + 1):
        for batch in loader:
            image = batch["image"].to(device)
            gt = batch["gt"].to(device)
            logit_raw = batch["logit_raw"].to(device)
            with torch.no_grad():
                image_embeddings, info = sam.encode_image(image)
                logit_base = base_coarse(image, logit_raw) if base_coarse is not None else logit_raw
                teacher_logits = None
                if teacher_mode == "fixed078":
                    base_maps = prune_bb2_channels_torch(build_bb2_maps_from_logit(logit_base), str(cfg.get("bb2_channel_mode", "logit_grad")))
                    base_bb2 = run_bb2_logits(adapter, base_maps, logit_base)
                    teacher_prompt = teacher_alpha * logit_base + (1.0 - teacher_alpha) * base_bb2
                    teacher_box = prob_to_bbox_torch(torch.sigmoid(logit_base), threshold=0.4, margin_ratio=0.05)
                    teacher_logits, _ = sam.decode_from_embeddings(
                        image_embeddings=image_embeddings,
                        info=info,
                        boxes_norm=teacher_box,
                        point_coords_norm=None,
                        point_labels=None,
                        mask_prompt=None,
                        bb2_mask_prompt_logits=teacher_prompt,
                    )
                gt_box = prob_to_bbox_torch(gt, threshold=0.5, margin_ratio=0.05)

            out = model(image, logit_base.detach())
            maps = prune_bb2_channels_torch(build_bb2_maps_from_logit(out["logit_cal"]), str(cfg.get("bb2_channel_mode", "logit_grad")))
            bb2_logits = run_bb2_logits(adapter, maps, out["logit_cal"])
            prompt = out["alpha"] * out["logit_cal"] + (1.0 - out["alpha"]) * bb2_logits
            logits, _ = sam.decode_from_embeddings(
                image_embeddings=image_embeddings,
                info=info,
                boxes_norm=out["boxes"],
                point_coords_norm=None,
                point_labels=None,
                mask_prompt=None,
                bb2_mask_prompt_logits=prompt,
            )
            logits = logits / out["temperature"].clamp_min(1.0e-4)
            loss, parts = gpc_loss(
                model=model,
                final_logits=logits,
                gt=gt,
                logit_cal=out["logit_cal"],
                logit_base=logit_base,
                boxes=out["boxes"],
                gt_boxes=gt_box,
                teacher_logits=teacher_logits,
                threshold=out.get("threshold"),
                threshold_sharpness=out.get("threshold_sharpness"),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            step += 1
            avg_n += 1
            avg_state = _average_state(avg_state, model.state_dict(), avg_n)
            raw_weights = torch.exp(-model.loss_log_vars.detach()).cpu().tolist()
            weight_names = ["final", "prior", "boundary", "box", "anchor"]
            if bool(model.cfg.use_distill_loss):
                weight_names.append("distill")
            if bool(model.cfg.use_learned_threshold):
                weight_names.append("binarize")
            if bool(model.cfg.use_learned_threshold) and bool(model.cfg.use_binary_shape_losses):
                weight_names.extend(["binary_boundary", "binary_area"])
            if bool(model.cfg.use_learned_threshold) and bool(model.cfg.use_binary_distance_loss):
                weight_names.append("binary_distance")
            weight_map = {name: float(raw_weights[i]) for i, name in enumerate(weight_names)}
            vals = {
                "loss": float(loss.detach().cpu()),
                **parts,
                "residual": float(out["residual"].abs().mean().detach().cpu()),
                "alpha": float(out["alpha"].mean().detach().cpu()),
                "alpha_std": float(out["alpha"].std(unbiased=False).detach().cpu()),
                "alpha_min": float(out["alpha"].amin().detach().cpu()),
                "alpha_max": float(out["alpha"].amax().detach().cpu()),
                "box_scale": float(out["box_scale"].mean().detach().cpu()),
                "temperature": float(out["temperature"].mean().detach().cpu()),
                "threshold": float(out["threshold"].mean().detach().cpu()),
                "threshold_sharpness": float(out["threshold_sharpness"].mean().detach().cpu()),
                "w_final": weight_map.get("final", 0.0),
                "w_prior": weight_map.get("prior", 0.0),
                "w_boundary": weight_map.get("boundary", 0.0),
                "w_box": weight_map.get("box", 0.0),
                "w_anchor": weight_map.get("anchor", 0.0),
                "w_distill": weight_map.get("distill", 0.0),
                "w_binarize": weight_map.get("binarize", 0.0),
                "w_binary_boundary": weight_map.get("binary_boundary", 0.0),
                "w_binary_area": weight_map.get("binary_area", 0.0),
                "w_binary_distance": weight_map.get("binary_distance", 0.0),
            }
            for k, v in vals.items():
                sums[k] += v
            window_count += 1
            if step % int(args.print_every) == 0:
                _flush_history("step")
                _save(out_dir / "checkpoint_last.pth", {k: v.detach().cpu() for k, v in model.state_dict().items()})
                _save(out_dir / "checkpoint_swa.pth", avg_state)
                if bool(args.save_step_ckpts):
                    _save(out_dir / f"checkpoint_step{step:04d}.pth", {k: v.detach().cpu() for k, v in model.state_dict().items()})
            if int(args.max_steps) > 0 and step >= int(args.max_steps):
                _flush_history("max_steps")
                _save(out_dir / "checkpoint_last.pth", {k: v.detach().cpu() for k, v in model.state_dict().items()})
                _save(out_dir / "checkpoint_swa.pth", avg_state)
                _write_history(out_dir, history)
                return
        _flush_history("epoch_end")
        _save(out_dir / "checkpoint_last.pth", {k: v.detach().cpu() for k, v in model.state_dict().items()})
        _save(out_dir / "checkpoint_swa.pth", avg_state)
        if int(args.max_steps) > 0 and step >= int(args.max_steps):
            break
    _save(out_dir / "checkpoint_last.pth", {k: v.detach().cpu() for k, v in model.state_dict().items()})
    _save(out_dir / "checkpoint_swa.pth", avg_state)
    _write_history(out_dir, history)


if __name__ == "__main__":
    main()
