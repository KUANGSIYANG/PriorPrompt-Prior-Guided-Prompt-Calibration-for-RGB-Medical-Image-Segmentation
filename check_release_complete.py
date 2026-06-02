from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "DATA_AND_WEIGHT_MANIFEST.json",
    "configs/branch_safeanchor_gpc_v8c_paper_release.json",
    "configs/branch_safeanchor_gpc_v8c_teacher_free_nores_notemp_learnthr_unified.json",
    "train.py",
    "train_nnunet.py",
    "train_bb2.py",
    "export_nnunet_probs.py",
    "train_gpc_prior.py",
    "run_safeanchor_gpc.py",
    "eval_nnunet_only_baseline.py",
    "audit_gpc_clean_mainline.py",
    "eval_metrics.py",
    "scripts/run_train_kvasir.ps1",
    "scripts/run_train_ph2.ps1",
    "scripts/run_train_nnunet_kvasir.ps1",
    "scripts/run_train_nnunet_ph2.ps1",
    "scripts/run_train_bb2_kvasir.ps1",
    "scripts/run_train_bb2_ph2.ps1",
    "scripts/run_full_from_scratch_protocol.ps1",
    "scripts/run_eval_all.ps1",
    "scripts/run_full_protocol.ps1",
    "scripts/run_audit.ps1",
    "safeanchor/bb2_adapter.py",
    "safeanchor/bb2_channel_prune.py",
    "safeanchor/coarse_prior_calibrator.py",
    "safeanchor/dbp_prior.py",
    "safeanchor/generalizable_prior_calibrator.py",
    "safeanchor/gpc_data.py",
    "safeanchor/io_utils.py",
    "safeanchor/mobile_sam_wrapper.py",
    "assets/checkpoints/mobile_sam.pt",
    "assets/checkpoints/kvasir_bb2_checkpoint_best.pth",
    "assets/checkpoints/ph2_bb2_checkpoint_best.pth",
    "assets/nnunet_weights/kvasir_100ep_old512hexa/fold_0/checkpoint_final.pth",
    "assets/nnunet_weights/ph2_30ep/fold_0/checkpoint_best.pth",
    "outputs/gpc_v8c_learnthr_unified_kvasir_train880_3ep_v1/checkpoint_last.pth",
    "outputs/gpc_v8c_learnthr_unified_ph2_train140_3ep_v1/checkpoint_last.pth",
    "outputs/gpc_v8c_learnthr_unified_kvasir_train880_3ep_v1/train_history.json",
    "outputs/gpc_v8c_learnthr_unified_ph2_train140_3ep_v1/train_history.json",
    "outputs/metrics/gpc_v8c_clean_mainline_audit.json",
]


DATASETS = {
    "kvasir_train880": ("data/kvasir/train/images", "data/kvasir/train/labels", "data/kvasir/train/probs", "data/kvasir/train/ids.txt", 880),
    "kvasir_val50": ("data/kvasir/val50/images", "data/kvasir/val50/labels", "data/kvasir/val50/probs", "data/kvasir/val50/ids.txt", 50),
    "kvasir_test70": ("data/kvasir/test70/images", "data/kvasir/test70/labels", "data/kvasir/test70/probs", "data/kvasir/test70/ids.txt", 70),
    "ph2_train140": ("data/ph2/train/images", "data/ph2/train/labels", "data/ph2/train/probs", "data/ph2/train/ids.txt", 140),
    "ph2_val30": ("data/ph2/val30/images", "data/ph2/val30/labels", "data/ph2/val30/probs", "data/ph2/val30/ids.txt", 30),
    "ph2_test30": ("data/ph2/test30/images", "data/ph2/test30/labels", "data/ph2/test30/probs", "data/ph2/test30/ids.txt", 30),
}


METRICS = [
    "metrics_gpc_v8c_learnthr_unified_kvasir_val50_last.json",
    "metrics_gpc_v8c_learnthr_unified_kvasir_test70_last.json",
    "metrics_gpc_v8c_learnthr_unified_ph2_val30_last.json",
    "metrics_gpc_v8c_learnthr_unified_ph2_test30_last.json",
    "metrics_nnunet_only_val50.json",
    "metrics_nnunet_only_test70.json",
    "metrics_nnunet_only_ph2_val30.json",
    "metrics_nnunet_only_ph2_test30.json",
]


PRED_DIRS = [
    ("outputs/pred_gpc_v8c_learnthr_unified_kvasir_val50_last", 50),
    ("outputs/pred_gpc_v8c_learnthr_unified_kvasir_test70_last", 70),
    ("outputs/pred_gpc_v8c_learnthr_unified_ph2_val30_last", 30),
    ("outputs/pred_gpc_v8c_learnthr_unified_ph2_test30_last", 30),
]


def _count_files(path: Path, suffix: str | None = None) -> int:
    if not path.exists():
        return 0
    files = [p for p in path.rglob("*") if p.is_file()]
    if suffix is not None:
        files = [p for p in files if p.suffix.lower() == suffix]
    return len(files)


def _read_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip().lstrip("\ufeff").removesuffix(".png") for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def main() -> None:
    missing = [item for item in REQUIRED_FILES if not (ROOT / item).exists()]
    data_report = {}
    for name, (image_dir, label_dir, prob_dir, ids_file, expected) in DATASETS.items():
        ids = _read_ids(ROOT / ids_file)
        data_report[name] = {
            "expected_cases": expected,
            "ids": len(ids),
            "images": _count_files(ROOT / image_dir, ".png"),
            "labels": _count_files(ROOT / label_dir, ".png"),
            "prob_files": _count_files(ROOT / prob_dir, ".npz"),
            "ok": len(ids) == expected,
        }
    metric_report = {name: (ROOT / "outputs/metrics" / name).exists() for name in METRICS}
    pred_report = {
        path: {"expected": expected, "png_files": _count_files(ROOT / path, ".png"), "ok": _count_files(ROOT / path, ".png") == expected}
        for path, expected in PRED_DIRS
    }
    pycache = [str(p.relative_to(ROOT)) for p in ROOT.rglob("__pycache__")]
    ok = (
        not missing
        and all(item["ok"] for item in data_report.values())
        and all(metric_report.values())
        and all(item["ok"] for item in pred_report.values())
        and not pycache
    )
    out = {
        "release_root": str(ROOT),
        "complete": ok,
        "scope": "v8c GPC training/inference plus frozen upstream nnUNet/BB2/MobileSAM weights and probabilities",
        "missing_required_files": missing,
        "datasets": data_report,
        "metrics": metric_report,
        "predictions": pred_report,
        "pycache_dirs": pycache,
        "note": "nnUNet and BB2 are frozen upstream stages in v8c; their trained weights and probabilities are included. The executable training scripts in this release retrain the v8c GPC heads for Kvasir and PH2.",
    }
    out_path = ROOT / "RELEASE_COMPLETENESS_AUDIT.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"complete": ok, "out": str(out_path)}, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
