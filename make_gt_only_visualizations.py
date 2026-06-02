from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion


def _image_path(images_dir: Path, case_id: str) -> Path:
    for suffix in (".png", "_0000.png"):
        path = images_dir / f"{case_id}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"missing image for {case_id} in {images_dir}")


def _load_ids(path: Path) -> list[str]:
    return [line.strip().lstrip("\ufeff").removesuffix(".png") for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_mask(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0).astype(np.uint8)


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if tuple(mask.shape) == tuple(shape):
        return mask.astype(np.uint8)
    img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    img = img.resize((int(shape[1]), int(shape[0])), resample=Image.NEAREST)
    return (np.asarray(img, dtype=np.uint8) > 127).astype(np.uint8)


def _contour(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return np.zeros_like(m, dtype=bool)
    eroded = binary_erosion(m, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return np.logical_xor(m, eroded)


def _thicken(mask: np.ndarray, radius: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(radius))):
        out = binary_dilation(out, structure=np.ones((3, 3), dtype=bool))
    return out


def _gt_overlay(image: np.ndarray, gt: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = np.asarray(image, dtype=np.float32) / 255.0
    gt_m = np.asarray(gt, dtype=bool)
    rgb = np.asarray(color, dtype=np.float32) / 255.0
    out[gt_m] = 0.45 * out[gt_m] + 0.55 * rgb
    edge = _thicken(_contour(gt_m), 1)
    halo = _thicken(edge, 1)
    out[halo] = 0.0
    out[edge] = rgb
    return np.clip(out, 0.0, 1.0)


def _render_split(
    *,
    root: Path,
    name: str,
    images_dir: Path,
    labels_dir: Path,
    ids_file: Path,
    out_dir: Path,
    color: tuple[int, int, int],
    max_cases: int,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = _load_ids(ids_file)
    if max_cases > 0:
        ids = ids[:max_cases]
    written = []
    for case_id in ids:
        image = _load_rgb(_image_path(images_dir, case_id))
        gt = _resize_mask(_load_mask(labels_dir / f"{case_id}.png"), (image.shape[0], image.shape[1]))
        overlay = _gt_overlay(image, gt, color)
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), squeeze=False)
        axes[0, 0].imshow(image)
        axes[0, 0].set_title(f"{case_id}\nRGB input")
        axes[0, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[0, 1].set_title(f"GT binary mask\narea={int(gt.sum())}")
        axes[0, 2].imshow(overlay)
        axes[0, 2].set_title("GT overlay only")
        for ax in axes[0]:
            ax.set_axis_off()
        fig.tight_layout()
        path = out_dir / f"{name}_{case_id}_gt_only.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path.relative_to(root)))
    return written


def build_argparser() -> argparse.ArgumentParser:
    root_default = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(root_default))
    ap.add_argument("--splits", type=str, default="kvasir_test70,ph2_test30")
    ap.add_argument("--max-cases", type=int, default=3)
    ap.add_argument("--out_dir", type=str, default="outputs/figures/gt_only_visualization")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    specs = {
        "kvasir_val50": ("kvasir_val50", root / "data/kvasir/val50/images", root / "data/kvasir/val50/labels", root / "data/kvasir/val50/ids.txt", (255, 35, 35)),
        "kvasir_test70": ("kvasir_test70", root / "data/kvasir/test70/images", root / "data/kvasir/test70/labels", root / "data/kvasir/test70/ids.txt", (255, 35, 35)),
        "ph2_val30": ("ph2_val30", root / "data/ph2/val30/images", root / "data/ph2/val30/labels", root / "data/ph2/val30/ids.txt", (255, 215, 0)),
        "ph2_test30": ("ph2_test30", root / "data/ph2/test30/images", root / "data/ph2/test30/labels", root / "data/ph2/test30/ids.txt", (255, 215, 0)),
    }
    summary = {}
    for key in [item.strip() for item in str(args.splits).split(",") if item.strip()]:
        if key not in specs:
            raise ValueError(f"unknown split: {key}")
        name, images_dir, labels_dir, ids_file, color = specs[key]
        summary[key] = _render_split(
            root=root,
            name=name,
            images_dir=images_dir,
            labels_dir=labels_dir,
            ids_file=ids_file,
            out_dir=out_dir / name,
            color=color,
            max_cases=int(args.max_cases),
        )
    print(summary)


if __name__ == "__main__":
    main()
