from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(path: Path) -> dict[str, float]:
    summary = _load_json(path)["summary"]
    return {
        "dice": float(summary["dice_mean_pct"]),
        "asd": float(summary["asd_mean_mm"]),
        "hd95": float(summary["hd95_mean_mm"]),
    }


def _weighted(a: dict[str, float], na: int, b: dict[str, float], nb: int) -> dict[str, float]:
    n = float(na + nb)
    return {k: (float(a[k]) * na + float(b[k]) * nb) / n for k in ["dice", "asd", "hd95"]}


def _same_except(left: Any, right: Any, *, ignore_keys: set[str]) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        keys = set(left) | set(right)
        for key in keys:
            if key in ignore_keys:
                continue
            if not _same_except(left.get(key), right.get(key), ignore_keys=ignore_keys):
                return False
        return True
    return left == right


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "metrics" / "gpc_v8c_clean_mainline_audit.json",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    metrics_dir = root / "outputs" / "metrics"
    k_manifest = _load_json(root / "outputs" / "gpc_v8c_learnthr_unified_kvasir_train880_3ep_v1" / "train_manifest.json")
    p_manifest = _load_json(root / "outputs" / "gpc_v8c_learnthr_unified_ph2_train140_3ep_v1" / "train_manifest.json")
    k_cfg = _load_json(root / "configs" / "branch_safeanchor_gpc_v8c_teacher_free_nores_notemp_learnthr_unified.json")

    ignore_manifest = {
        "dataset",
        "train_ids",
        "sam_ckpt",
        "seed",
    }
    manifest_same_arch = _same_except(k_manifest, p_manifest, ignore_keys=ignore_manifest)
    forbidden_flags_ok = all(
        not bool(k_manifest.get(key, False)) and not bool(p_manifest.get(key, False))
        for key in [
            "uses_area_gate",
            "uses_candidate_gate",
            "uses_point_prompt",
            "uses_fixed_alpha_at_inference",
            "uses_thresholded_box_in_model",
        ]
    )
    gpc_cfg_ok = (
        k_manifest["cfg"] == p_manifest["cfg"]
        and not bool(k_manifest["cfg"]["use_residual"])
        and not bool(k_manifest["cfg"]["use_temperature"])
        and bool(k_manifest["cfg"]["use_learned_threshold"])
        and not bool(k_manifest["uses_distill_loss"])
        and not bool(p_manifest["uses_distill_loss"])
        and k_manifest["base_coarse_calibrator"] == ""
        and p_manifest["base_coarse_calibrator"] == ""
    )

    table = {
        "nnunet_kvasir_val50": _metric(metrics_dir / "metrics_nnunet_only_val50.json"),
        "nnunet_kvasir_test70": _metric(metrics_dir / "metrics_nnunet_only_test70.json"),
        "nnunet_ph2_val30": _metric(metrics_dir / "metrics_nnunet_only_ph2_val30.json"),
        "nnunet_ph2_test30": _metric(metrics_dir / "metrics_nnunet_only_ph2_test30.json"),
        "gpc_v8c_kvasir_val50": _metric(metrics_dir / "metrics_gpc_v8c_learnthr_unified_kvasir_val50_last.json"),
        "gpc_v8c_kvasir_test70": _metric(metrics_dir / "metrics_gpc_v8c_learnthr_unified_kvasir_test70_last.json"),
        "gpc_v8c_ph2_val30": _metric(metrics_dir / "metrics_gpc_v8c_learnthr_unified_ph2_val30_last.json"),
        "gpc_v8c_ph2_test30": _metric(metrics_dir / "metrics_gpc_v8c_learnthr_unified_ph2_test30_last.json"),
    }
    table["nnunet_kvasir_test120"] = _weighted(table["nnunet_kvasir_val50"], 50, table["nnunet_kvasir_test70"], 70)
    table["gpc_v8c_kvasir_test120"] = _weighted(table["gpc_v8c_kvasir_val50"], 50, table["gpc_v8c_kvasir_test70"], 70)
    table["nnunet_ph2_test60"] = _weighted(table["nnunet_ph2_val30"], 30, table["nnunet_ph2_test30"], 30)
    table["gpc_v8c_ph2_test60"] = _weighted(table["gpc_v8c_ph2_val30"], 30, table["gpc_v8c_ph2_test30"], 30)

    gains = {
        "kvasir_test120": {
            key: table["gpc_v8c_kvasir_test120"][key] - table["nnunet_kvasir_test120"][key]
            for key in ["dice", "asd", "hd95"]
        },
        "ph2_test60": {
            key: table["gpc_v8c_ph2_test60"][key] - table["nnunet_ph2_test60"][key]
            for key in ["dice", "asd", "hd95"]
        },
    }
    passed = bool(manifest_same_arch and forbidden_flags_ok and gpc_cfg_ok)
    out = {
        "mainline": "SafeAnchor-GPC-v8c Teacher-Free NoResidual NoTemperature LearnedThreshold",
        "config": str((root / "configs" / "branch_safeanchor_gpc_v8c_teacher_free_nores_notemp_learnthr_unified.json").resolve()),
        "protocol": {
            "checkpoint_policy": "checkpoint_last",
            "validation_selection": False,
            "architecture_same_for_kvasir_and_ph2": passed,
            "allowed_dataset_specific_items": [
                "dataset probabilities",
                "dataset images and labels",
                "dataset-specific nnUNet weights",
                "dataset-specific BB2/SafeAnchor weights",
                "dataset-specific GPC weights",
            ],
        },
        "audit": {
            "manifest_same_architecture_except_allowed_fields": manifest_same_arch,
            "forbidden_flags_ok": forbidden_flags_ok,
            "gpc_cfg_ok": gpc_cfg_ok,
            "config_method_family": k_cfg.get("method_family"),
        },
        "metrics": table,
        "gains_vs_nnunet_only": gains,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"passed": passed, "out": str(args.out), "mainline": out["mainline"]}, indent=2))


if __name__ == "__main__":
    main()
