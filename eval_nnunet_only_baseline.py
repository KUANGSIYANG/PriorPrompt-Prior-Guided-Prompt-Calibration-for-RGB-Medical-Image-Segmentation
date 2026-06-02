from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval_metrics import evaluate
from safeanchor.gpc_data import load_json, resolve_path
from safeanchor.io_utils import load_prob_map, save_mask_png


def _load_ids(path: Path) -> list[str]:
    return [
        line.strip().lstrip("\ufeff").removesuffix(".png")
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _split_defaults(root: Path, split: str) -> dict[str, Path]:
    split = str(split).lower().strip()
    if split == "val50":
        return {
            "prob_dir": root / "data/kvasir/val50/probs",
            "gt_dir": root / "data/kvasir/val50/labels",
            "ids_file": root / "data/kvasir/val50/ids.txt",
        }
    if split == "test70":
        return {
            "prob_dir": root / "data/kvasir/test70/probs",
            "gt_dir": root / "data/kvasir/test70/labels",
            "ids_file": root / "data/kvasir/test70/ids.txt",
        }
    if split == "ph2_val30":
        return {
            "prob_dir": root / "data/ph2/val30/probs",
            "gt_dir": root / "data/ph2/val30/labels",
            "ids_file": root / "data/ph2/val30/ids.txt",
        }
    if split == "ph2_test30":
        return {
            "prob_dir": root / "data/ph2/test30/probs",
            "gt_dir": root / "data/ph2/test30/labels",
            "ids_file": root / "data/ph2/test30/ids.txt",
        }
    if split == "idrid_test":
        return {
            "prob_dir": root / "data/idrid/test/probs",
            "gt_dir": root / "data/idrid/test/labels",
            "ids_file": root / "data/idrid/test/ids.txt",
        }
    if split == "dfuc_test":
        return {
            "prob_dir": root / "data/dfuc/test/probs",
            "gt_dir": root / "data/dfuc/test/labels",
            "ids_file": root / "data/dfuc/test/ids.txt",
        }
    if split == "busi_test":
        return {
            "prob_dir": root / "data/busi/test/probs",
            "gt_dir": root / "data/busi/test/labels",
            "ids_file": root / "data/busi/test/ids.txt",
        }
    if split == "isic2017_val":
        return {
            "prob_dir": root / "data/isic2017/val/probs",
            "gt_dir": root / "data/isic2017/val/labels",
            "ids_file": root / "data/isic2017/val/ids.txt",
        }
    if split == "isic2017_test":
        return {
            "prob_dir": root / "data/isic2017/test/probs",
            "gt_dir": root / "data/isic2017/test/labels",
            "ids_file": root / "data/isic2017/test/ids.txt",
        }
    if split == "tn3k_test":
        return {
            "prob_dir": root / "data/tn3k/test/probs",
            "gt_dir": root / "data/tn3k/test/labels",
            "ids_file": root / "data/tn3k/test/ids.txt",
        }
    raise ValueError(f"unknown split: {split}")


def _apply_split_overrides(root: Path, cfg: dict[str, Path], override: dict) -> None:
    if "prob_dir" in override:
        cfg["prob_dir"] = resolve_path(root, override["prob_dir"])
    if "labels_dir" in override:
        cfg["gt_dir"] = resolve_path(root, override["labels_dir"])
    if "gt_dir" in override:
        cfg["gt_dir"] = resolve_path(root, override["gt_dir"])
    if "ids_file" in override:
        cfg["ids_file"] = resolve_path(root, override["ids_file"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pure nnUNet probability baseline without SafeAnchor.")
    parser.add_argument(
        "--split",
        choices=["val50", "test70", "ph2_val30", "ph2_test30", "idrid_test", "dfuc_test", "busi_test", "isic2017_val", "isic2017_test", "tn3k_test"],
        required=True,
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out_name", default="")
    args = parser.parse_args()

    root = args.root.resolve()
    cfg = _split_defaults(root, args.split)
    if args.config is not None:
        config = load_json(resolve_path(root, args.config))
        split_name = str(args.split).lower()
        if split_name.startswith("ph2_"):
            dataset = "ph2"
        elif split_name.startswith("idrid_"):
            dataset = "idrid"
        elif split_name.startswith("dfuc_"):
            dataset = "dfuc"
        elif split_name.startswith("busi_"):
            dataset = "busi"
        elif split_name.startswith("isic2017_"):
            dataset = "isic2017"
        elif split_name.startswith("tn3k_"):
            dataset = "tn3k"
        else:
            dataset = "kvasir"
        _apply_split_overrides(root, cfg, dict(config.get(f"{dataset}_eval", {})))
        _apply_split_overrides(root, cfg, dict(config.get(split_name, {})))
        _apply_split_overrides(root, cfg, dict(config.get(f"{split_name}_eval", {})))
    ids = _load_ids(cfg["ids_file"])
    out_name = args.out_name.strip() or f"nnunet_only_{args.split}"
    pred_dir = root / "outputs" / f"pred_{out_name}"
    metrics_dir = root / "outputs/metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    eval_ids = metrics_dir / f"ids_{out_name}.txt"
    eval_ids.write_text("\n".join(ids) + "\n", encoding="utf-8")

    missing: list[str] = []
    for cid in ids:
        prob_path = cfg["prob_dir"] / f"{cid}.npz"
        if not prob_path.exists():
            missing.append(cid)
            continue
        prob = load_prob_map(prob_path)
        pred = (np.asarray(prob, dtype=np.float32) > float(args.threshold)).astype(np.uint8)
        save_mask_png(pred, pred_dir / f"{cid}.png")
    if missing:
        raise FileNotFoundError(f"missing probability files for {len(missing)} cases, first={missing[:5]}")

    meta = {
        "method": "nnUNet-only",
        "split": args.split,
        "threshold": float(args.threshold),
        "prob_dir": str(cfg["prob_dir"]),
        "gt_dir": str(cfg["gt_dir"]),
        "ids_file": str(cfg["ids_file"]),
        "count": len(ids),
        "safeanchor_used": False,
        "bb2_used": False,
        "gpc_used": False,
    }
    (metrics_dir / f"meta_{out_name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    evaluate(
        pred_dir=pred_dir,
        gt_dir=cfg["gt_dir"],
        out_json=metrics_dir / f"metrics_{out_name}.json",
        out_csv=metrics_dir / f"metrics_{out_name}.csv",
        ids_file=eval_ids,
    )
    summary = json.loads((metrics_dir / f"metrics_{out_name}.json").read_text(encoding="utf-8"))["summary"]
    print(f"{out_name}: {summary['dice_mean_pct']:.4f} / {summary['asd_mean_mm']:.4f} / {summary['hd95_mean_mm']:.4f}")


if __name__ == "__main__":
    main()
