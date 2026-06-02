from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_aux(path: Path, dataset: str, split: str, seed: str) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"expected a list in {path}")
    out = []
    for row in rows:
        if "alpha" not in row:
            continue
        out.append(
            {
                "dataset": dataset,
                "split": split,
                "seed": seed,
                "case_id": str(row.get("case_id", "")),
                "alpha": float(row["alpha"]),
                "threshold": float(row.get("threshold", np.nan)),
                "box_scale": float(row.get("box_scale", np.nan)),
            }
        )
    return out


def collect_alpha_rows(metrics_dir: Path, seeds: list[str]) -> list[dict]:
    specs = [
        ("Kvasir-120", "kvasir_val50", "aux_safeprompt_bed_kvasir_val50_seed{seed}.json"),
        ("Kvasir-120", "kvasir_test70", "aux_safeprompt_bed_kvasir_test70_seed{seed}.json"),
        ("PH2-60", "ph2_val30", "aux_safeprompt_bed_ph2_val30_seed{seed}.json"),
        ("PH2-60", "ph2_test30", "aux_safeprompt_bed_ph2_test30_seed{seed}.json"),
    ]
    rows: list[dict] = []
    for seed in seeds:
        for dataset, split, pattern in specs:
            path = metrics_dir / pattern.format(seed=seed)
            if not path.exists():
                raise FileNotFoundError(path)
            rows.extend(_load_aux(path, dataset, split, seed))
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "split", "seed", "case_id", "alpha", "threshold", "box_scale"])
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        vals = np.array([r["alpha"] for r in rows if r["dataset"] == dataset], dtype=np.float64)
        summary[dataset] = {
            "n": int(vals.size),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "min": float(vals.min()),
            "p25": float(np.percentile(vals, 25)),
            "median": float(np.median(vals)),
            "p75": float(np.percentile(vals, 75)),
            "max": float(vals.max()),
        }
    return summary


def plot_alpha_distribution(rows: list[dict], summary: dict, out_path: Path) -> None:
    rng = np.random.default_rng(20260522)
    datasets = ["Kvasir-120", "PH2-60"]
    colors = {
        "Kvasir-120": "#C93A4A",
        "PH2-60": "#2A86B8",
    }
    edge_colors = {
        "Kvasir-120": "#6E1530",
        "PH2-60": "#0D3F63",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)
    data = [np.array([r["alpha"] for r in rows if r["dataset"] == d], dtype=np.float64) for d in datasets]

    parts = ax.violinplot(data, positions=[1, 2], widths=0.62, showmeans=False, showmedians=False, showextrema=False)
    for body, dataset in zip(parts["bodies"], datasets):
        body.set_facecolor(colors[dataset])
        body.set_edgecolor(edge_colors[dataset])
        body.set_alpha(0.24)
        body.set_linewidth(1.1)

    bp = ax.boxplot(
        data,
        positions=[1, 2],
        widths=0.24,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.4},
        whiskerprops={"color": "#333333", "linewidth": 1.0},
        capprops={"color": "#333333", "linewidth": 1.0},
    )
    for patch, dataset in zip(bp["boxes"], datasets):
        patch.set_facecolor(colors[dataset])
        patch.set_edgecolor(edge_colors[dataset])
        patch.set_alpha(0.36)
        patch.set_linewidth(1.0)

    for idx, dataset in enumerate(datasets, start=1):
        vals = np.array([r["alpha"] for r in rows if r["dataset"] == dataset], dtype=np.float64)
        jitter = rng.normal(0.0, 0.045, size=vals.size)
        ax.scatter(
            np.full(vals.size, idx) + jitter,
            vals,
            s=13,
            alpha=0.48,
            color=colors[dataset],
            edgecolor="white",
            linewidth=0.25,
            zorder=3,
        )
        st = summary[dataset]
        ax.scatter([idx], [st["mean"]], marker="D", s=42, color=edge_colors[dataset], edgecolor="white", linewidth=0.7, zorder=4)
        ax.text(
            idx,
            min(0.98, st["max"] + 0.012),
            f"{st['mean']:.3f} +/- {st['std']:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=edge_colors[dataset],
        )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel("GPC fusion coefficient alpha", fontsize=11)
    ax.set_title("Case-wise alpha distribution of SafePrompt-BED V8C", fontsize=12, pad=10)
    ax.set_ylim(0.50, 0.82)
    ax.grid(axis="y", color="#D8D8D8", linestyle="--", linewidth=0.6, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.01,
        -0.18,
        "Each dot is one case from one seed; Kvasir combines val50+test70, PH2 combines val30+test30 across seeds 20260522/20260527/20260530.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_alpha_histogram(rows: list[dict], summary: dict, out_path: Path) -> None:
    datasets = ["Kvasir-120", "PH2-60"]
    colors = {
        "Kvasir-120": "#C93A4A",
        "PH2-60": "#2A86B8",
    }
    edge_colors = {
        "Kvasir-120": "#6E1530",
        "PH2-60": "#0D3F63",
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.25), dpi=300, sharey=True)
    bins = {
        "Kvasir-120": np.linspace(0.732, 0.755, 18),
        "PH2-60": np.linspace(0.618, 0.634, 18),
    }
    for ax, dataset in zip(axes, datasets):
        vals = np.array([r["alpha"] for r in rows if r["dataset"] == dataset], dtype=np.float64)
        ax.hist(
            vals,
            bins=bins[dataset],
            color=colors[dataset],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.82,
        )
        st = summary[dataset]
        ax.axvline(st["mean"], color=edge_colors[dataset], linewidth=1.5)
        ax.axvspan(st["mean"] - st["std"], st["mean"] + st["std"], color=colors[dataset], alpha=0.15, linewidth=0)
        ax.set_title(f"{dataset}\nmean {st['mean']:.3f} +/- {st['std']:.3f}", fontsize=10)
        ax.set_xlabel("alpha", fontsize=10)
        ax.grid(axis="y", color="#D8D8D8", linestyle="--", linewidth=0.55, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Case count across 3 seeds", fontsize=10)
    fig.suptitle("Histogram of GPC fusion coefficient alpha", fontsize=12, y=1.02)
    fig.text(
        0.01,
        -0.04,
        "Kvasir combines val50+test70; PH2 combines val30+test30. Counts include seeds 20260522/20260527/20260530.",
        ha="left",
        va="top",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot V8C GPC alpha distributions for Kvasir and PH2.")
    parser.add_argument("--metrics_dir", type=Path, default=Path("outputs/metrics"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/figures/v8c_alpha_distribution"))
    parser.add_argument("--seeds", type=str, default="20260522,20260527,20260530")
    args = parser.parse_args()

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    rows = collect_alpha_rows(args.metrics_dir, seeds)
    summary = summarize(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "v8c_alpha_distribution_cases.csv")
    (args.out_dir / "v8c_alpha_distribution_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_alpha_distribution(rows, summary, args.out_dir / "v8c_alpha_distribution_kvasir_ph2.png")
    plot_alpha_histogram(rows, summary, args.out_dir / "v8c_alpha_histogram_kvasir_ph2.png")

    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out_dir / 'v8c_alpha_distribution_kvasir_ph2.png'}")
    print(f"wrote {args.out_dir / 'v8c_alpha_histogram_kvasir_ph2.png'}")


if __name__ == "__main__":
    main()
