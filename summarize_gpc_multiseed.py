from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _load_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["summary"]


def _row(label: str, values: list[float]) -> dict:
    return {
        "metric": label,
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
        "values": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize V8C checkpoint_last multi-seed metrics.")
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--out", type=str, default="outputs/metrics/gpc_v8c_multiseed_summary.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    metrics_dir = root / "outputs" / "metrics"
    groups = {
        "kvasir_val50": [
            metrics_dir / "metrics_gpc_v8c_learnthr_unified_kvasir_val50_last.json",
            metrics_dir / "metrics_gpc_v8c_seed20260527_kvasir_val50_last.json",
        ],
        "kvasir_test70": [
            metrics_dir / "metrics_gpc_v8c_learnthr_unified_kvasir_test70_last.json",
            metrics_dir / "metrics_gpc_v8c_seed20260527_kvasir_test70_last.json",
        ],
        "ph2_val30": [
            metrics_dir / "metrics_gpc_v8c_learnthr_unified_ph2_val30_last.json",
            metrics_dir / "metrics_gpc_v8c_seed20260527_ph2_val30_last.json",
        ],
        "ph2_test30": [
            metrics_dir / "metrics_gpc_v8c_learnthr_unified_ph2_test30_last.json",
            metrics_dir / "metrics_gpc_v8c_seed20260527_ph2_test30_last.json",
        ],
    }
    output: dict[str, object] = {
        "protocol": "SafeAnchor-GPC-v8c full 3ep checkpoint_last multi-seed summary",
        "seed_labels": ["20260522_release", "20260527"],
        "splits": {},
    }
    for split, paths in groups.items():
        summaries = []
        missing = []
        for path in paths:
            if path.exists():
                summaries.append(_load_summary(path))
            else:
                missing.append(str(path))
        if not summaries:
            output["splits"][split] = {"missing": missing}
            continue
        dice = [float(s["dice_mean_pct"]) for s in summaries]
        asd = [float(s["asd_mean_mm"]) for s in summaries]
        hd95 = [float(s["hd95_mean_mm"]) for s in summaries]
        output["splits"][split] = {
            "missing": missing,
            "dice": _row("dice_mean_pct", dice),
            "asd": _row("asd_mean_mm", asd),
            "hd95": _row("hd95_mean_mm", hd95),
        }
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
