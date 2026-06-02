from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt, label as nd_label


def _case_id_from_path(path: Path) -> str:
    return path.stem


def _load_ids(ids_file: Optional[Path]) -> Optional[set[str]]:
    if ids_file is None:
        return None
    ids: set[str] = set()
    for line in ids_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.endswith(".png"):
            line = line[:-4]
        ids.add(line)
    return ids


def _load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def _binary_mask(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.floating) and arr.min() >= 0.0 and arr.max() <= 1.0:
        return arr > 0.5
    return arr > 0


def _dice(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    inter = (gt & pred).sum()
    denom = gt.sum() + pred.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def _surface_distances(mask_ref: np.ndarray, mask_pred: np.ndarray, spacing: Tuple[float, float]) -> np.ndarray:
    footprint = np.ones([3] * mask_ref.ndim, dtype=bool)
    ref_border = mask_ref ^ binary_erosion(mask_ref, footprint, border_value=0)
    pred_border = mask_pred ^ binary_erosion(mask_pred, footprint, border_value=0)
    if not ref_border.any() and not pred_border.any():
        return np.array([0.0], dtype=np.float32)
    if not ref_border.any() or not pred_border.any():
        return np.array([math.inf], dtype=np.float32)
    dt_ref = distance_transform_edt(~ref_border, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_border, sampling=spacing)
    return np.concatenate([dt_pred[ref_border], dt_ref[pred_border]]).astype(np.float32)


def _asd(gt: np.ndarray, pred: np.ndarray, spacing: Tuple[float, float]) -> float:
    return float(np.nanmean(_surface_distances(gt.astype(bool), pred.astype(bool), spacing)))


def _hd95(gt: np.ndarray, pred: np.ndarray, spacing: Tuple[float, float]) -> float:
    return float(np.nanpercentile(_surface_distances(gt.astype(bool), pred.astype(bool), spacing), 95.0))


def _label_failures_from_masks(gt: np.ndarray, pred: np.ndarray) -> Dict:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    gt_lbl, gt_n = nd_label(gt)
    pred_lbl, pred_n = nd_label(pred)
    omission_ids: List[int] = []
    omission_components: List[Dict] = []
    for gid in range(1, int(gt_n) + 1):
        gm = gt_lbl == gid
        area = int(gm.sum())
        if area <= 0:
            continue
        pred_ids, counts = np.unique(pred_lbl[gm & pred], return_counts=True)
        valid = pred_ids > 0
        best_recall = 0.0
        best_pred = 0
        if np.any(valid):
            pred_ids = pred_ids[valid]
            counts = counts[valid]
            idx = int(np.argmax(counts))
            best_pred = int(pred_ids[idx])
            best_recall = float(counts[idx]) / float(area)
        if best_recall < 0.5:
            omission_ids.append(gid)
            omission_components.append(
                {
                    "gt_component_id": gid,
                    "gt_area": area,
                    "best_pred_component_id": best_pred,
                    "best_recall": best_recall,
                    "best_precision": 0.0,
                    "severity": float(1.0 - best_recall),
                }
            )
    merge_pred_ids: List[int] = []
    merge_pairs: List[List[int]] = []
    merge_components: List[Dict] = []
    for pid in range(1, int(pred_n) + 1):
        pm = pred_lbl == pid
        pred_area = int(pm.sum())
        if pred_area <= 0:
            continue
        gt_ids, counts = np.unique(gt_lbl[pm & gt], return_counts=True)
        valid = gt_ids > 0
        if not np.any(valid):
            continue
        gt_ids = gt_ids[valid].astype(int)
        counts = counts[valid]
        matched = []
        recalls = []
        for gid, ov in zip(gt_ids, counts):
            gt_area = int((gt_lbl == int(gid)).sum())
            recall = float(ov) / float(max(1, gt_area))
            if recall >= 0.5:
                matched.append(int(gid))
                recalls.append(recall)
        if len(matched) >= 2:
            merge_pred_ids.append(pid)
            for i in range(len(matched)):
                for j in range(i + 1, len(matched)):
                    merge_pairs.append([matched[i], matched[j]])
            merge_components.append(
                {
                    "pred_component_id": pid,
                    "pred_area": pred_area,
                    "matched_gt_component_ids": matched,
                    "matched_gt_count": len(matched),
                    "matched_gt_recalls": recalls,
                    "matched_gt_precisions": [],
                    "matched_gt_overlap_areas": [],
                    "dominant_gt_component_id": matched[0],
                    "severity": float(len(matched) - 1),
                }
            )
    gt_area_total = float(max(1, gt.sum()))
    pred_area_total = float(max(1, pred.sum()))
    omission_severity = float(sum(c["severity"] * c["gt_area"] for c in omission_components) / gt_area_total)
    merge_severity = float(sum(c["severity"] for c in merge_components) / pred_area_total)
    return {
        "gt_component_count": int(gt_n),
        "pred_component_count": int(pred_n),
        "omission_count": len(omission_ids),
        "merge_count": len(merge_pred_ids),
        "omission_component_ids": omission_ids,
        "merge_pred_component_ids": merge_pred_ids,
        "merge_gt_component_pairs": merge_pairs,
        "omission_components": omission_components,
        "merge_components": merge_components,
        "severity": {"omission": omission_severity, "merge": merge_severity, "total": omission_severity + merge_severity},
    }


def evaluate(pred_dir: Path, gt_dir: Path, out_json: Path, out_csv: Path, ids_file: Optional[Path] = None) -> None:
    ids = _load_ids(ids_file)
    pred_files = [p for p in sorted(pred_dir.glob("*.png")) if ids is None or _case_id_from_path(p) in ids]
    rows: List[Dict] = []
    dice_vals: List[float] = []
    asd_vals: List[float] = []
    hd95_vals: List[float] = []
    finite_asd: List[float] = []
    finite_hd95: List[float] = []
    missing_gt = 0
    for pred_path in pred_files:
        case_id = _case_id_from_path(pred_path)
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            missing_gt += 1
            continue
        pred_bin = _binary_mask(_load_gray(pred_path))
        gt_bin = _binary_mask(_load_gray(gt_path))
        is_gt_empty = not gt_bin.any()
        is_pred_empty = not pred_bin.any()
        if is_gt_empty and is_pred_empty:
            dice, asd, hd95 = 1.0, 0.0, 0.0
        elif is_gt_empty != is_pred_empty:
            dice, asd, hd95 = 0.0, math.inf, math.inf
        else:
            dice = _dice(gt_bin, pred_bin)
            asd = _asd(gt_bin, pred_bin, (1.0, 1.0))
            hd95 = _hd95(gt_bin, pred_bin, (1.0, 1.0))
        failure = _label_failures_from_masks(gt_bin, pred_bin)
        row = {
            "case_id": case_id,
            "dice": dice * 100.0,
            "asd_mm": asd,
            "hd95_mm": hd95,
            "gt_area": int(gt_bin.sum()),
            "pred_area": int(pred_bin.sum()),
            "is_gt_empty": is_gt_empty,
            "is_pred_empty": is_pred_empty,
            "gt_component_count": int(failure["gt_component_count"]),
            "pred_component_count": int(failure["pred_component_count"]),
            "omission_count": int(failure["omission_count"]),
            "merge_count": int(failure["merge_count"]),
            "omission_component_ids": list(failure["omission_component_ids"]),
            "merge_pred_component_ids": list(failure["merge_pred_component_ids"]),
            "merge_gt_component_pairs": list(failure["merge_gt_component_pairs"]),
            "omission_components": list(failure["omission_components"]),
            "merge_components": list(failure["merge_components"]),
            "omission_severity": float(failure["severity"]["omission"]),
            "merge_severity": float(failure["severity"]["merge"]),
            "failure_total_severity": float(failure["severity"]["total"]),
        }
        rows.append(row)
        dice_vals.append(row["dice"])
        asd_vals.append(asd)
        hd95_vals.append(hd95)
        if math.isfinite(asd):
            finite_asd.append(asd)
        if math.isfinite(hd95):
            finite_hd95.append(hd95)
    summary = {
        "count": len(rows),
        "missing_gt": missing_gt,
        "dice_mean_pct": float(np.mean(dice_vals)) if dice_vals else float("nan"),
        "dice_std_pct": float(np.std(dice_vals)) if dice_vals else float("nan"),
        "asd_mean_mm": float(np.mean(asd_vals)) if asd_vals else float("nan"),
        "asd_std_mm": float(np.std(asd_vals)) if asd_vals else float("nan"),
        "finite_asd_mean_mm": float(np.mean(finite_asd)) if finite_asd else float("nan"),
        "finite_asd_std_mm": float(np.std(finite_asd)) if finite_asd else float("nan"),
        "finite_asd_median_mm": float(np.median(finite_asd)) if finite_asd else float("nan"),
        "trimmed_asd_mean_mm": float(np.mean(finite_asd)) if finite_asd else float("nan"),
        "finite_asd_count": int(len(finite_asd)),
        "finite_asd_total": int(len(rows)),
        "hd95_mean_mm": float(np.mean(hd95_vals)) if hd95_vals else float("nan"),
        "hd95_std_mm": float(np.std(hd95_vals)) if hd95_vals else float("nan"),
        "finite_hd95_mean_mm": float(np.mean(finite_hd95)) if finite_hd95 else float("nan"),
        "finite_hd95_std_mm": float(np.std(finite_hd95)) if finite_hd95 else float("nan"),
        "finite_hd95_median_mm": float(np.median(finite_hd95)) if finite_hd95 else float("nan"),
        "finite_hd95_count": int(len(finite_hd95)),
        "finite_hd95_total": int(len(rows)),
        "empty_pred_case_ids": sorted([r["case_id"] for r in rows if r["is_pred_empty"]]),
        "empty_gt_case_ids": sorted([r["case_id"] for r in rows if r["is_gt_empty"]]),
        "omission_count_total": int(sum(int(r["omission_count"]) for r in rows)),
        "merge_count_total": int(sum(int(r["merge_count"]) for r in rows)),
        "omission_case_count": int(sum(int(r["omission_count"]) > 0 for r in rows)),
        "merge_case_count": int(sum(int(r["merge_count"]) > 0 for r in rows)),
        "omission_severity_mean": float(np.mean([float(r["omission_severity"]) for r in rows])) if rows else 0.0,
        "merge_severity_mean": float(np.mean([float(r["merge_severity"]) for r in rows])) if rows else 0.0,
        "failure_total_severity_mean": float(np.mean([float(r["failure_total_severity"]) for r in rows])) if rows else 0.0,
        "failure_total_severity_sum": float(np.sum([float(r["failure_total_severity"]) for r in rows])) if rows else 0.0,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"cases": rows, "summary": summary}, indent=2, allow_nan=True), encoding="utf-8")
    keys = sorted({k for row in rows for k in row.keys()})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

