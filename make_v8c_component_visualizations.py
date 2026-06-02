from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion

from safeanchor.bb2_adapter import load_bb2_adapter
from safeanchor.bb2_channel_prune import prune_bb2_channels_torch
from safeanchor.coarse_prior_calibrator import build_bb2_maps_from_logit
from safeanchor.generalizable_prior_calibrator import GPCCfg, GeneralizablePriorCalibrator
from safeanchor.gpc_data import load_json, resolve_path, run_bb2_logits, safe_logit
from safeanchor.io_utils import image_to_tensor, load_prob_map, load_rgb_image, map_to_tensor


@dataclass(frozen=True)
class VizSpec:
    name: str
    split: str
    title: str
    images_dir: Path
    labels_dir: Path
    prob_dir: Path
    nnunet_pred_dir: Path
    v8c_pred_dir: Path
    nnunet_metrics: Path
    v8c_metrics: Path
    aux_json: Path
    gpc_ckpt: Path
    sam_ckpt_key: str
    gt_color: tuple[int, int, int]
    pred_color: tuple[int, int, int]
    overlap_color: tuple[int, int, int]
    box_color: tuple[int, int, int]
    prompt_cmap: str


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _image_path(images_dir: Path, case_id: str) -> Path:
    path = images_dir / f"{case_id}.png"
    if path.exists():
        return path
    path = images_dir / f"{case_id}_0000.png"
    if path.exists():
        return path
    raise FileNotFoundError(f"missing image for {case_id} in {images_dir}")


def _load_mask(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0).astype(np.uint8)


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if tuple(mask.shape) == tuple(shape):
        return mask.astype(np.uint8)
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img = img.resize((int(shape[1]), int(shape[0])), resample=Image.NEAREST)
    return (np.asarray(img, dtype=np.uint8) > 127).astype(np.uint8)


def _resize_float(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if tuple(arr.shape) == tuple(shape):
        return arr.astype(np.float32)
    x = torch.from_numpy(np.asarray(arr, dtype=np.float32)).view(1, 1, arr.shape[0], arr.shape[1])
    y = F.interpolate(x, size=shape, mode="bilinear", align_corners=False)
    return y[0, 0].cpu().numpy().astype(np.float32)


def _case_table(metrics_path: Path) -> dict[str, dict]:
    data = _read_json(metrics_path)
    return {str(row["case_id"]): row for row in data["cases"]}


def _all_cases(spec: VizSpec) -> list[str]:
    nn = _case_table(spec.nnunet_metrics)
    v8c = _case_table(spec.v8c_metrics)
    return [case_id for case_id in v8c.keys() if case_id in nn]


def _select_cases(spec: VizSpec, count: int) -> list[str]:
    nn = _case_table(spec.nnunet_metrics)
    v8c = _case_table(spec.v8c_metrics)
    rows = []
    for case_id, vrow in v8c.items():
        if case_id not in nn:
            continue
        nd = float(nn[case_id].get("dice", 0.0))
        vd = float(vrow.get("dice", 0.0))
        rows.append((vd - nd, vd, nd, case_id))
    positive = [row for row in rows if row[0] > 0.0]
    positive.sort(reverse=True)
    if len(positive) >= count:
        return [row[-1] for row in positive[:count]]
    rows.sort(reverse=True)
    return [row[-1] for row in rows[:count]]


def _to_float_image(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    if x.max(initial=0.0) > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def _contour(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return np.zeros_like(m, dtype=bool)
    eroded = binary_erosion(m, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return np.logical_xor(m, eroded)


def _thicken(mask: np.ndarray, radius: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    if not out.any() or int(radius) <= 0:
        return out
    structure = np.ones((3, 3), dtype=bool)
    for _ in range(int(radius)):
        out = binary_dilation(out, structure=structure)
    return out


def _overlay_mask(
    image: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    pred_color: tuple[int, int, int],
    gt_color: tuple[int, int, int],
    overlap_color: tuple[int, int, int],
) -> np.ndarray:
    out = _to_float_image(image).copy()
    pred_m = np.asarray(pred, dtype=bool)
    gt_m = np.asarray(gt, dtype=bool)
    pred_rgb = np.asarray(pred_color, dtype=np.float32) / 255.0
    gt_rgb = np.asarray(gt_color, dtype=np.float32) / 255.0
    overlap_rgb = np.asarray(overlap_color, dtype=np.float32) / 255.0
    # Three-region error map: GT-only, mask-only, and overlap are distinct.
    # Kvasir GT is read with >0, so 0/1 labels are rendered correctly.
    gt_only = gt_m & ~pred_m
    pred_only = pred_m & ~gt_m
    overlap = gt_m & pred_m
    out[gt_only] = 0.34 * out[gt_only] + 0.66 * gt_rgb
    out[pred_only] = 0.38 * out[pred_only] + 0.62 * pred_rgb
    out[overlap] = 0.32 * out[overlap] + 0.68 * overlap_rgb
    pred_edge = _contour(pred_m)
    gt_edge = _contour(gt_m)
    out[pred_edge] = pred_rgb
    out[gt_edge] = gt_rgb
    return np.clip(out, 0.0, 1.0)


def _draw_gt_contour(ax, gt: np.ndarray, *, color: tuple[int, int, int], halo: bool = False) -> None:
    gt_edge = _contour(gt)
    if not gt_edge.any():
        return
    rgba = np.zeros((*gt_edge.shape, 4), dtype=np.float32)
    if halo:
        halo_edge = _thicken(gt_edge, 1)
        rgba[halo_edge, :3] = 0.0
        rgba[halo_edge, 3] = 0.85
        ax.imshow(rgba)
        rgba = np.zeros((*gt_edge.shape, 4), dtype=np.float32)
    rgba[gt_edge, :3] = np.asarray(color, dtype=np.float32) / 255.0
    rgba[gt_edge, 3] = 1.0
    ax.imshow(rgba)


def _add_overlay_legend(
    ax,
    *,
    pred_color: tuple[int, int, int],
    gt_color: tuple[int, int, int],
    overlap_color: tuple[int, int, int],
) -> None:
    pred_rgb = np.asarray(pred_color, dtype=np.float32) / 255.0
    gt_rgb = np.asarray(gt_color, dtype=np.float32) / 255.0
    overlap_rgb = np.asarray(overlap_color, dtype=np.float32) / 255.0
    handles = [
        plt.Line2D([0], [0], color=gt_rgb, lw=4.0, label="GT only"),
        plt.Line2D([0], [0], color=pred_rgb, lw=4.0, label="Mask only"),
        plt.Line2D([0], [0], color=overlap_rgb, lw=4.0, label="Overlap"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.72, facecolor="white")


def _norm_box_to_xyxy(box: list[float] | tuple[float, ...], shape: tuple[int, int]) -> tuple[float, float, float, float]:
    h, w = int(shape[0]), int(shape[1])
    x0, y0, x1, y1 = [float(v) for v in box]
    return x0 * max(1, w - 1), y0 * max(1, h - 1), x1 * max(1, w - 1), y1 * max(1, h - 1)


def _draw_box(ax, box: list[float], shape: tuple[int, int], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = _norm_box_to_xyxy(box, shape)
    rgb = np.asarray(color, dtype=np.float32) / 255.0
    rect = plt.Rectangle((x0, y0), max(1.0, x1 - x0), max(1.0, y1 - y0), fill=False, color=rgb, linewidth=1.0)
    ax.add_patch(rect)


def _load_gpc_and_bed(root: Path, cfg: dict, spec: VizSpec, device: torch.device):
    ckpt = torch.load(str(spec.gpc_ckpt), map_location="cpu")
    gpc = GeneralizablePriorCalibrator(GPCCfg(**dict(ckpt.get("gpc_cfg", cfg.get("gpc_cfg", {}))))).to(device)
    gpc.load_state_dict(ckpt["gpc_state_dict"], strict=True)
    gpc.eval()
    bed = load_bb2_adapter(resolve_path(root, cfg[spec.sam_ckpt_key]), device=device)
    bed.eval()
    return gpc, bed


def _reconstruct_prompt(
    *,
    image: np.ndarray,
    prob: np.ndarray,
    gpc: GeneralizablePriorCalibrator,
    bed: torch.nn.Module,
    bb2_channel_mode: str,
    device: torch.device,
) -> tuple[np.ndarray, list[float], dict[str, float]]:
    with torch.no_grad():
        image_t = image_to_tensor(image, device)
        logit_base = map_to_tensor(safe_logit(prob), device)
        out = gpc(image_t, logit_base)
        logit_cal = out["logit_cal"]
        maps = prune_bb2_channels_torch(build_bb2_maps_from_logit(logit_cal), bb2_channel_mode)
        bb2_logits = run_bb2_logits(bed, maps, logit_cal)
        prompt = out["alpha"] * logit_cal + (1.0 - out["alpha"]) * bb2_logits
        prompt_prob = torch.sigmoid(prompt)[0, 0].detach().cpu().numpy().astype(np.float32)
        meta = {
            "alpha": float(out["alpha"].mean().detach().cpu()),
            "box_scale": float(out["box_scale"].mean().detach().cpu()),
            "threshold": float(out["threshold"].mean().detach().cpu()),
        }
        return prompt_prob, [float(v) for v in out["boxes"][0].detach().cpu().tolist()], meta


def _format_metric(row: dict | None) -> str:
    if not row:
        return "Dice n/a"
    return f"Dice {float(row.get('dice', 0.0)):.2f} | ASD {float(row.get('asd_mm', 0.0)):.2f}"


def _input_panel_title(spec: VizSpec) -> str:
    if spec.name.startswith("ph2_"):
        return "PH$^2$ RGB input"
    if spec.name.startswith("kvasir_"):
        return "Kvasir-SEG RGB input"
    return "RGB input"


def _draw_panels(
    axes,
    *,
    case_id: str,
    image: np.ndarray,
    gt: np.ndarray,
    nn_overlay: np.ndarray,
    v8c_overlay: np.ndarray,
    prompt_prob: np.ndarray,
    box: list[float],
    shape: tuple[int, int],
    spec: VizSpec,
    nn_metric: dict | None,
    v8c_metric: dict | None,
    meta: dict[str, float],
) -> None:
    axes[0].imshow(image)
    axes[0].set_title(_input_panel_title(spec))
    axes[1].imshow(nn_overlay)
    axes[1].set_title(f"nnU-Net vs GT\n{_format_metric(nn_metric)}")
    _add_overlay_legend(axes[1], pred_color=spec.pred_color, gt_color=spec.gt_color, overlap_color=spec.overlap_color)
    axes[2].imshow(v8c_overlay)
    _draw_box(axes[2], box, shape, spec.box_color)
    axes[2].set_title(f"Ours vs GT\n{_format_metric(v8c_metric)}")
    _add_overlay_legend(axes[2], pred_color=spec.pred_color, gt_color=spec.gt_color, overlap_color=spec.overlap_color)
    axes[3].imshow(image, alpha=0.35)
    axes[3].imshow(prompt_prob, cmap=spec.prompt_cmap, alpha=0.78, vmin=0.0, vmax=1.0)
    _draw_gt_contour(axes[3], gt, color=spec.gt_color)
    _draw_box(axes[3], box, shape, spec.box_color)
    axes[3].set_title(
        "Dense prompt + learned box\n"
        f"alpha {meta['alpha']:.3f} | scale {meta['box_scale']:.2f} | tau {meta['threshold']:.3f}"
    )
    for ax in axes:
        ax.set_axis_off()


def _render_spec(
    root: Path,
    cfg: dict,
    spec: VizSpec,
    cases: list[str],
    out_dir: Path,
    device: torch.device,
    *,
    write_grid: bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    gpc, bed = _load_gpc_and_bed(root, cfg, spec, device)
    bb2_channel_mode = str(cfg.get("bb2_channel_mode", "logit_grad"))
    nn_metrics = _case_table(spec.nnunet_metrics)
    v8c_metrics = _case_table(spec.v8c_metrics)
    aux = {str(row["case_id"]): row for row in _read_json(spec.aux_json)}
    rows_for_manifest = []

    fig = None
    axes = None
    if write_grid:
        fig, axes = plt.subplots(len(cases), 4, figsize=(18, 4.2 * len(cases)), squeeze=False)
    for row_idx, case_id in enumerate(cases):
        image = load_rgb_image(_image_path(spec.images_dir, case_id))
        shape = (int(image.shape[0]), int(image.shape[1]))
        gt = _resize_mask(_load_mask(spec.labels_dir / f"{case_id}.png"), shape)
        nn_pred = _resize_mask(_load_mask(spec.nnunet_pred_dir / f"{case_id}.png"), shape)
        v8c_pred = _resize_mask(_load_mask(spec.v8c_pred_dir / f"{case_id}.png"), shape)
        prob = load_prob_map(spec.prob_dir / f"{case_id}.npz")
        prompt_prob, box, meta = _reconstruct_prompt(
            image=image,
            prob=prob,
            gpc=gpc,
            bed=bed,
            bb2_channel_mode=bb2_channel_mode,
            device=device,
        )
        prompt_prob = _resize_float(prompt_prob, shape)
        aux_box = aux.get(case_id, {}).get("box")
        if aux_box is not None:
            box = [float(v) for v in aux_box]

        nn_overlay = _overlay_mask(
            image,
            nn_pred,
            gt,
            pred_color=spec.pred_color,
            gt_color=spec.gt_color,
            overlap_color=spec.overlap_color,
        )
        v8c_overlay = _overlay_mask(
            image,
            v8c_pred,
            gt,
            pred_color=spec.pred_color,
            gt_color=spec.gt_color,
            overlap_color=spec.overlap_color,
        )
        if write_grid and axes is not None:
            _draw_panels(
                axes[row_idx],
                case_id=case_id,
                image=image,
                gt=gt,
                nn_overlay=nn_overlay,
                v8c_overlay=v8c_overlay,
                prompt_prob=prompt_prob,
                box=box,
                shape=shape,
                spec=spec,
                nn_metric=nn_metrics.get(case_id),
                v8c_metric=v8c_metrics.get(case_id),
                meta=meta,
            )

        single = out_dir / f"{spec.name}_{case_id}_component.png"
        single_fig, single_axes = plt.subplots(1, 4, figsize=(18, 4.2), squeeze=False)
        _draw_panels(
            single_axes[0],
            case_id=case_id,
            image=image,
            gt=gt,
            nn_overlay=nn_overlay,
            v8c_overlay=v8c_overlay,
            prompt_prob=prompt_prob,
            box=box,
            shape=shape,
            spec=spec,
            nn_metric=nn_metrics.get(case_id),
            v8c_metric=v8c_metrics.get(case_id),
            meta=meta,
        )
        single_fig.tight_layout()
        single_fig.savefig(single, dpi=220, bbox_inches="tight")
        plt.close(single_fig)
        rows_for_manifest.append(
            {
                "case_id": case_id,
                "nnunet": nn_metrics.get(case_id, {}),
                "v8c": v8c_metrics.get(case_id, {}),
                "box": box,
                "alpha": meta["alpha"],
                "box_scale": meta["box_scale"],
                "threshold": meta["threshold"],
                "figure": str(single.relative_to(root)),
            }
        )

    grid_path = None
    if write_grid and fig is not None:
        fig.suptitle(
            f"{spec.title}: GT color {spec.gt_color}, prediction color {spec.pred_color}; "
            "box and dense prompt are reconstructed from the V8C mainline.",
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        grid_path = out_dir / f"{spec.name}_component_grid.png"
        fig.savefig(grid_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
    manifest = {
        "dataset": spec.name,
        "split": spec.split,
        "grid": str(grid_path.relative_to(root)) if grid_path is not None else None,
        "cases": rows_for_manifest,
        "color_policy": {
            "gt_rgb": spec.gt_color,
            "prediction_rgb": spec.pred_color,
            "overlap_rgb": spec.overlap_color,
            "box_rgb": spec.box_color,
            "prompt_cmap": spec.prompt_cmap,
        },
    }
    (out_dir / f"{spec.name}_component_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_argparser() -> argparse.ArgumentParser:
    root_default = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(root_default))
    ap.add_argument("--config", type=str, default="configs/branch_safeanchor_gpc_v8c_paper_release.json")
    ap.add_argument("--seed", type=str, default="20260522")
    ap.add_argument("--cases-per-dataset", type=int, default=3)
    ap.add_argument(
        "--case-ids",
        type=str,
        default="",
        help="Optional comma-separated case ids. When set, each split renders only matching cases.",
    )
    ap.add_argument("--all-cases", action="store_true", help="Render every case in the selected splits.")
    ap.add_argument("--no-grid", action="store_true", help="Only write per-case figures; do not write stitched grids.")
    ap.add_argument(
        "--splits",
        type=str,
        default="kvasir_val50,kvasir_test70,ph2_val30,ph2_test30",
        help="Comma-separated subset: kvasir_val50,kvasir_test70,ph2_val30,ph2_test30.",
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out_dir", type=str, default="outputs/figures/v8c_component_visualization_seed20260522")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    root = Path(args.root).resolve()
    cfg = load_json(resolve_path(root, args.config))
    seed = str(args.seed)
    out_dir = resolve_path(root, args.out_dir)
    device = torch.device(str(args.device))

    specs_by_key = {
        "kvasir_val50": VizSpec(
            name="kvasir_polyp_val50",
            split="val50",
            title="Kvasir polyp val50",
            images_dir=root / "data/kvasir/val50/images",
            labels_dir=root / "data/kvasir/val50/labels",
            prob_dir=root / "data/kvasir/val50/probs",
            nnunet_pred_dir=root / "outputs/pred_nnunet_only_val50",
            v8c_pred_dir=root / f"outputs/pred_safeprompt_bed_kvasir_val50_seed{seed}",
            nnunet_metrics=root / "outputs/metrics/metrics_nnunet_only_val50.json",
            v8c_metrics=root / f"outputs/metrics/metrics_safeprompt_bed_kvasir_val50_seed{seed}.json",
            aux_json=root / f"outputs/metrics/aux_safeprompt_bed_kvasir_val50_seed{seed}.json",
            gpc_ckpt=root / f"outputs/gpc_safeprompt_bed_kvasir_train880_3ep_seed{seed}/checkpoint_last.pth",
            sam_ckpt_key="kvasir_sam_ckpt",
            gt_color=(255, 35, 35),
            pred_color=(160, 65, 230),
            overlap_color=(255, 115, 210),
            box_color=(255, 35, 35),
            prompt_cmap="magma",
        ),
        "kvasir_test70": VizSpec(
            name="kvasir_polyp_test70",
            split="test70",
            title="Kvasir polyp test70",
            images_dir=root / "data/kvasir/test70/images",
            labels_dir=root / "data/kvasir/test70/labels",
            prob_dir=root / "data/kvasir/test70/probs",
            nnunet_pred_dir=root / "outputs/pred_nnunet_only_test70",
            v8c_pred_dir=root / f"outputs/pred_safeprompt_bed_kvasir_test70_seed{seed}",
            nnunet_metrics=root / "outputs/metrics/metrics_nnunet_only_test70.json",
            v8c_metrics=root / f"outputs/metrics/metrics_safeprompt_bed_kvasir_test70_seed{seed}.json",
            aux_json=root / f"outputs/metrics/aux_safeprompt_bed_kvasir_test70_seed{seed}.json",
            gpc_ckpt=root / f"outputs/gpc_safeprompt_bed_kvasir_train880_3ep_seed{seed}/checkpoint_last.pth",
            sam_ckpt_key="kvasir_sam_ckpt",
            gt_color=(255, 35, 35),
            pred_color=(160, 65, 230),
            overlap_color=(255, 115, 210),
            box_color=(255, 35, 35),
            prompt_cmap="magma",
        ),
        "ph2_val30": VizSpec(
            name="ph2_lesion_val30",
            split="ph2_val30",
            title="PH2 dermoscopic lesion val30",
            images_dir=root / "data/ph2/val30/images",
            labels_dir=root / "data/ph2/val30/labels",
            prob_dir=root / "data/ph2/val30/probs",
            nnunet_pred_dir=root / "outputs/pred_nnunet_only_ph2_val30",
            v8c_pred_dir=root / f"outputs/pred_safeprompt_bed_ph2_val30_seed{seed}",
            nnunet_metrics=root / "outputs/metrics/metrics_nnunet_only_ph2_val30.json",
            v8c_metrics=root / f"outputs/metrics/metrics_safeprompt_bed_ph2_val30_seed{seed}.json",
            aux_json=root / f"outputs/metrics/aux_safeprompt_bed_ph2_val30_seed{seed}.json",
            gpc_ckpt=root / f"outputs/gpc_safeprompt_bed_ph2_train140_3ep_seed{seed}/checkpoint_last.pth",
            sam_ckpt_key="ph2_sam_ckpt",
            gt_color=(255, 35, 35),
            pred_color=(160, 65, 230),
            overlap_color=(255, 115, 210),
            box_color=(255, 35, 35),
            prompt_cmap="magma",
        ),
        "ph2_test30": VizSpec(
            name="ph2_lesion_test30",
            split="ph2_test30",
            title="PH2 dermoscopic lesion test30",
            images_dir=root / "data/ph2/test30/images",
            labels_dir=root / "data/ph2/test30/labels",
            prob_dir=root / "data/ph2/test30/probs",
            nnunet_pred_dir=root / "outputs/pred_nnunet_only_ph2_test30",
            v8c_pred_dir=root / f"outputs/pred_safeprompt_bed_ph2_test30_seed{seed}",
            nnunet_metrics=root / "outputs/metrics/metrics_nnunet_only_ph2_test30.json",
            v8c_metrics=root / f"outputs/metrics/metrics_safeprompt_bed_ph2_test30_seed{seed}.json",
            aux_json=root / f"outputs/metrics/aux_safeprompt_bed_ph2_test30_seed{seed}.json",
            gpc_ckpt=root / f"outputs/gpc_safeprompt_bed_ph2_train140_3ep_seed{seed}/checkpoint_last.pth",
            sam_ckpt_key="ph2_sam_ckpt",
            gt_color=(255, 35, 35),
            pred_color=(160, 65, 230),
            overlap_color=(255, 115, 210),
            box_color=(255, 35, 35),
            prompt_cmap="magma",
        ),
    }
    requested = [item.strip() for item in str(args.splits).split(",") if item.strip()]
    unknown = [item for item in requested if item not in specs_by_key]
    if unknown:
        raise ValueError(f"unknown split keys: {unknown}")
    specs = [specs_by_key[item] for item in requested]
    selected_case_ids = {item.strip() for item in str(args.case_ids).split(",") if item.strip()}
    summary = {}
    for spec in specs:
        if selected_case_ids:
            cases = [case_id for case_id in _all_cases(spec) if case_id in selected_case_ids]
        else:
            cases = _all_cases(spec) if bool(args.all_cases) else _select_cases(spec, int(args.cases_per_dataset))
        if not cases:
            continue
        manifest = _render_spec(root, cfg, spec, cases, out_dir / spec.name, device, write_grid=not bool(args.no_grid))
        summary[spec.name] = {
            "split": spec.split,
            "cases": cases,
            "count": len(cases),
            "grid": manifest.get("grid"),
        }
    (out_dir / "v8c_component_visualization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
