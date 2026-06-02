from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval_metrics import evaluate
from safeanchor.bb2_adapter import load_bb2_adapter
from safeanchor.bb2_channel_prune import prune_bb2_channels_torch
from safeanchor.coarse_prior_calibrator import build_bb2_maps_from_logit
from safeanchor.gpc_data import (
    load_coarse_calibrator,
    load_json,
    prob_to_bbox_torch,
    resolve_path,
    run_bb2_logits,
    safe_logit,
)
from safeanchor.generalizable_prior_calibrator import GPCCfg, GeneralizablePriorCalibrator
from safeanchor.io_utils import image_to_tensor, load_prob_map, load_rgb_image, map_to_tensor, save_mask_png
from safeanchor.mobile_sam_wrapper import FrozenMobileSAM


def _load_ids(path: Path) -> list[str]:
    return [line.strip().lstrip("\ufeff").removesuffix(".png") for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _image_path(images_dir: Path, cid: str) -> Path:
    p = images_dir / f"{cid}.png"
    if p.exists():
        return p
    p = images_dir / f"{cid}_0000.png"
    if p.exists():
        return p
    raise FileNotFoundError(f"missing image for {cid}")


def _resize_rgb_np(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    x = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1)).unsqueeze(0)
    x = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return x.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)


def _resize_prob_np(prob: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    x = torch.from_numpy(np.asarray(prob, dtype=np.float32)).view(1, 1, prob.shape[0], prob.shape[1])
    x = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return x[0, 0].cpu().numpy().astype(np.float32)


def _resize_mask_to_original(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    x = torch.from_numpy(np.asarray(mask, dtype=np.float32)).view(1, 1, mask.shape[0], mask.shape[1])
    x = F.interpolate(x, size=(out_h, out_w), mode="nearest")
    return (x[0, 0].cpu().numpy() > 0.5).astype(np.uint8)


def _split_defaults(root: Path, split: str) -> dict:
    split = str(split).lower().strip()
    if split == "val50":
        return {
            "dataset": "kvasir",
            "images_dir": root / "data/kvasir/val50/images",
            "labels_dir": root / "data/kvasir/val50/labels",
            "prob_dir": root / "data/kvasir/val50/probs",
            "ids_file": root / "data/kvasir/val50/ids.txt",
            "use_base_coarse_calibrator": False,
        }
    if split == "test70":
        return {
            "dataset": "kvasir",
            "images_dir": root / "data/kvasir/test70/images",
            "labels_dir": root / "data/kvasir/test70/labels",
            "prob_dir": root / "data/kvasir/test70/probs",
            "ids_file": root / "data/kvasir/test70/ids.txt",
            "use_base_coarse_calibrator": False,
        }
    if split == "ph2_val30":
        return {
            "dataset": "ph2",
            "images_dir": root / "data/ph2/val30/images",
            "labels_dir": root / "data/ph2/val30/labels",
            "prob_dir": root / "data/ph2/val30/probs",
            "ids_file": root / "data/ph2/val30/ids.txt",
            "use_base_coarse_calibrator": False,
        }
    if split == "ph2_test30":
        return {
            "dataset": "ph2",
            "images_dir": root / "data/ph2/test30/images",
            "labels_dir": root / "data/ph2/test30/labels",
            "prob_dir": root / "data/ph2/test30/probs",
            "ids_file": root / "data/ph2/test30/ids.txt",
            "use_base_coarse_calibrator": False,
        }
    for dataset in ("idrid", "dfuc", "busi", "isic2017", "tn3k"):
        if split == f"{dataset}_val":
            return {
                "dataset": dataset,
                "images_dir": root / f"data/{dataset}/val/images",
                "labels_dir": root / f"data/{dataset}/val/labels",
                "prob_dir": root / f"data/{dataset}/val/probs",
                "ids_file": root / f"data/{dataset}/val/ids.txt",
                "use_base_coarse_calibrator": False,
            }
        if split == f"{dataset}_test":
            return {
                "dataset": dataset,
                "images_dir": root / f"data/{dataset}/test/images",
                "labels_dir": root / f"data/{dataset}/test/labels",
                "prob_dir": root / f"data/{dataset}/test/probs",
                "ids_file": root / f"data/{dataset}/test/ids.txt",
                "use_base_coarse_calibrator": False,
            }
    raise ValueError(f"unknown split: {split}")


def _apply_eval_overrides(root: Path, split_cfg: dict, override: dict) -> None:
    for key in ["images_dir", "labels_dir", "prob_dir", "ids_file"]:
        if key in override:
            split_cfg[key] = resolve_path(root, override[key])
    if "use_base_coarse_calibrator" in override:
        split_cfg["use_base_coarse_calibrator"] = bool(override["use_base_coarse_calibrator"])
    if "max_infer_side" in override:
        split_cfg["max_infer_side"] = int(override["max_infer_side"])


def build_argparser() -> argparse.ArgumentParser:
    root_default = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(root_default))
    ap.add_argument("--config", type=str, default="configs/branch_safeanchor_gpc_v1.json")
    ap.add_argument(
        "--split",
        type=str,
        default="val50",
        choices=["val50", "test70", "ph2_val30", "ph2_test30", "idrid_test", "dfuc_test", "busi_test", "isic2017_val", "isic2017_test", "tn3k_test"],
    )
    ap.add_argument("--gpc_ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_name", type=str, default="")
    ap.add_argument("--max_cases", type=int, default=0)
    ap.add_argument("--box_mode", type=str, default="gpc_soft", choices=["gpc_soft", "threshold"])
    ap.add_argument("--alpha_mode", type=str, default="gpc", choices=["gpc", "fixed078"])
    ap.add_argument("--residual_mode", type=str, default="gpc", choices=["gpc", "identity"])
    ap.add_argument("--temperature_mode", type=str, default="gpc", choices=["gpc", "none"])
    ap.add_argument("--prompt_mode", type=str, default="mix", choices=["mix", "coarse_only", "bb2_only"])
    ap.add_argument("--bb2_channel_mode", type=str, default="", choices=["", "logit_grad", "logit_grad2", "all"])
    ap.add_argument("--pred_mode", type=str, default="auto", choices=["auto", "fixed06", "gpc_threshold"])
    ap.add_argument("--max_infer_side", type=int, default=0)
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    root = Path(args.root).resolve()
    cfg = load_json(resolve_path(root, args.config))
    split_cfg = _split_defaults(root, args.split)
    _apply_eval_overrides(root, split_cfg, cfg.get(f"{split_cfg['dataset']}_eval", {}))
    _apply_eval_overrides(root, split_cfg, cfg.get(f"{str(args.split).lower()}_eval", {}))
    device = torch.device(str(args.device))
    max_infer_side = int(split_cfg.get("max_infer_side", int(args.max_infer_side)))
    ids = _load_ids(split_cfg["ids_file"])
    if int(args.max_cases) > 0:
        ids = ids[: int(args.max_cases)]

    ckpt = torch.load(str(resolve_path(root, args.gpc_ckpt)), map_location="cpu")
    model = GeneralizablePriorCalibrator(GPCCfg(**dict(ckpt.get("gpc_cfg", cfg.get("gpc_cfg", {}))))).to(device)
    model.load_state_dict(ckpt["gpc_state_dict"], strict=True)
    model.eval()

    sam_ckpt_key = f"{split_cfg['dataset']}_sam_ckpt"
    if sam_ckpt_key not in cfg:
        raise KeyError(f"missing SAM/BB2 checkpoint config: {sam_ckpt_key}")
    sam_ckpt_path = resolve_path(root, cfg[sam_ckpt_key])
    sam_ckpt = torch.load(str(sam_ckpt_path), map_location="cpu")
    adapter = load_bb2_adapter(sam_ckpt_path, device=device)
    adapter.eval()
    base_coarse = load_coarse_calibrator(resolve_path(root, cfg["coarse_calib_ckpt"]), device) if bool(split_cfg["use_base_coarse_calibrator"]) else None
    sam = FrozenMobileSAM(
        checkpoint=resolve_path(root, cfg["mobile_sam_ckpt"]),
        model_type=str(sam_ckpt.get("sam_model_type", "vit_t")),
        device=str(device),
        sam_state_dict=sam_ckpt.get("sam_state_dict"),
        trainable=False,
    )
    sam.eval()

    out_name = str(args.out_name).strip() or f"gpc_{Path(args.gpc_ckpt).parent.name}_{str(args.split).lower()}"
    pred_dir = root / "outputs" / f"pred_{out_name}"
    metrics_dir = root / "outputs/metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    eval_ids = metrics_dir / f"ids_{out_name}.txt"
    eval_ids.write_text("\n".join(ids) + "\n", encoding="utf-8")
    aux = []
    with torch.no_grad():
        for cid in ids:
            image = load_rgb_image(_image_path(split_cfg["images_dir"], cid))
            prob = load_prob_map(split_cfg["prob_dir"] / f"{cid}.npz")
            orig_h, orig_w = int(prob.shape[0]), int(prob.shape[1])
            if max_infer_side > 0 and max(orig_h, orig_w) > max_infer_side:
                scale = float(max_infer_side) / float(max(orig_h, orig_w))
                infer_h = max(32, int(round(orig_h * scale)))
                infer_w = max(32, int(round(orig_w * scale)))
                image = _resize_rgb_np(image, infer_h, infer_w)
                prob = _resize_prob_np(prob, infer_h, infer_w)
            image_t = image_to_tensor(image, device)
            logit_raw = map_to_tensor(safe_logit(prob), device)
            logit_base = base_coarse(image_t, logit_raw) if base_coarse is not None else logit_raw
            out = model(image_t, logit_base)
            logit_cal = out["logit_cal"] if str(args.residual_mode) == "gpc" else logit_base
            boxes = out["boxes"] if str(args.box_mode) == "gpc_soft" else prob_to_bbox_torch(torch.sigmoid(logit_cal), threshold=0.4, margin_ratio=0.05)
            alpha = out["alpha"] if str(args.alpha_mode) == "gpc" else torch.full_like(out["alpha"], 0.78)
            bb2_channel_mode = str(args.bb2_channel_mode).strip() or str(cfg.get("bb2_channel_mode", "logit_grad"))
            maps = prune_bb2_channels_torch(build_bb2_maps_from_logit(logit_cal), bb2_channel_mode)
            bb2_logits = run_bb2_logits(adapter, maps, logit_cal)
            if str(args.prompt_mode) == "coarse_only":
                prompt = logit_cal
            elif str(args.prompt_mode) == "bb2_only":
                prompt = bb2_logits
            else:
                prompt = alpha * logit_cal + (1.0 - alpha) * bb2_logits
            image_embeddings, info = sam.encode_image(image_t)
            logits, _ = sam.decode_from_embeddings(
                image_embeddings=image_embeddings,
                info=info,
                boxes_norm=boxes,
                point_coords_norm=None,
                point_labels=None,
                mask_prompt=None,
                bb2_mask_prompt_logits=prompt,
            )
            if str(args.temperature_mode) == "gpc":
                logits = logits / out["temperature"].clamp_min(1.0e-4)
            prob_mask = torch.sigmoid(logits)
            pred_mode = str(args.pred_mode)
            if pred_mode == "auto":
                pred_mode = "gpc_threshold" if bool(getattr(model.cfg, "use_learned_threshold", False)) else "fixed06"
            threshold = out["threshold"] if pred_mode == "gpc_threshold" else torch.full_like(out["threshold"], 0.6)
            pred = (prob_mask[0, 0] > threshold[0, 0]).detach().cpu().numpy().astype(np.uint8)
            if pred.shape != (orig_h, orig_w):
                pred = _resize_mask_to_original(pred, orig_h, orig_w)
            save_mask_png(pred, pred_dir / f"{cid}.png")
            aux.append(
                {
                    "case_id": cid,
                    "box": [float(v) for v in boxes[0].detach().cpu().tolist()],
                    "alpha": float(alpha.mean().detach().cpu()),
                    "alpha_std": float(alpha.std(unbiased=False).detach().cpu()),
                    "alpha_min": float(alpha.amin().detach().cpu()),
                    "alpha_max": float(alpha.amax().detach().cpu()),
                    "residual_abs": float((logit_cal - logit_base).abs().mean().detach().cpu()),
                    "box_scale": float(out["box_scale"].mean().detach().cpu()),
                    "temperature": float(out["temperature"].mean().detach().cpu()),
                    "threshold": float(threshold.mean().detach().cpu()),
                    "threshold_raw": float(out["threshold"].mean().detach().cpu()),
                    "threshold_sharpness": float(out["threshold_sharpness"].mean().detach().cpu()),
                    "box_mode": str(args.box_mode),
                    "alpha_mode": str(args.alpha_mode),
                    "residual_mode": str(args.residual_mode),
                    "temperature_mode": str(args.temperature_mode),
                    "prompt_mode": str(args.prompt_mode),
                    "bb2_channel_mode": bb2_channel_mode,
                    "pred_mode": pred_mode,
                    "max_infer_side": max_infer_side,
                }
            )
    (metrics_dir / f"aux_{out_name}.json").write_text(json.dumps(aux, indent=2), encoding="utf-8")
    evaluate(pred_dir=pred_dir, gt_dir=split_cfg["labels_dir"], out_json=metrics_dir / f"metrics_{out_name}.json", out_csv=metrics_dir / f"metrics_{out_name}.csv", ids_file=eval_ids)
    summary = load_json(metrics_dir / f"metrics_{out_name}.json")["summary"]
    print(f"{out_name}: {summary['dice_mean_pct']:.4f} / {summary['asd_mean_mm']:.4f} / {summary['hd95_mean_mm']:.4f}")


if __name__ == "__main__":
    main()
