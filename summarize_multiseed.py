from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _read_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("summary", data)


def _metric(summary: dict, key: str) -> float:
    if key == "dice":
        for name in ("dice_mean_pct", "dice_mean", "mean_dice"):
            if name in summary:
                return float(summary[name])
    if key == "asd":
        return float(summary["asd_mean_mm"])
    if key == "hd95":
        return float(summary["hd95_mean_mm"])
    raise KeyError(key)


def _weighted_pair(a: dict, b: dict, key: str) -> float:
    na = int(a.get("count", a.get("num_cases", 0)))
    nb = int(b.get("count", b.get("num_cases", 0)))
    if na <= 0 or nb <= 0:
        raise ValueError(f"missing case counts for weighted {key}: {na}, {nb}")
    return (_metric(a, key) * na + _metric(b, key) * nb) / float(na + nb)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--seeds", type=str, default="20260522,20260527,20260530")
    parser.add_argument("--out_name", type=str, default="safeprompt_bed_multiseed_last_checkpoint")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    metrics_dir = root / "outputs" / "metrics"
    out_dir = root / "outputs" / "multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [s.strip() for s in str(args.seeds).split(",") if s.strip()]

    rows = []
    for seed in seeds:
        kv = _read_summary(metrics_dir / f"metrics_safeprompt_bed_kvasir_val50_seed{seed}.json")
        kt = _read_summary(metrics_dir / f"metrics_safeprompt_bed_kvasir_test70_seed{seed}.json")
        pv = _read_summary(metrics_dir / f"metrics_safeprompt_bed_ph2_val30_seed{seed}.json")
        pt = _read_summary(metrics_dir / f"metrics_safeprompt_bed_ph2_test30_seed{seed}.json")
        for dataset, first, second in (("Kvasir-120", kv, kt), ("PH2-60", pv, pt)):
            row = {"seed": seed, "dataset": dataset}
            for key in ("dice", "asd", "hd95"):
                row[key] = _weighted_pair(first, second, key)
            rows.append(row)

    summary = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        ds_rows = [r for r in rows if r["dataset"] == dataset]
        summary[dataset] = {}
        for key in ("dice", "asd", "hd95"):
            values = np.array([float(r[key]) for r in ds_rows], dtype=np.float64)
            summary[dataset][f"{key}_mean"] = float(values.mean())
            summary[dataset][f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            summary[dataset][f"{key}_values"] = values.tolist()

    csv_path = out_dir / f"{args.out_name}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "dataset", "dice", "asd", "hd95"])
        writer.writeheader()
        writer.writerows(rows)

    json_path = out_dir / f"{args.out_name}.json"
    json_path.write_text(json.dumps({"seeds": seeds, "rows": rows, "summary": summary}, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
