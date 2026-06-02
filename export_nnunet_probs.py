from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


DATASETS = {
    "kvasir": {"dataset_id": "902", "dataset_name": "Dataset902_KvasirSEG2D", "trainer": "nnUNetTrainer", "plans": "nnUNetPlans"},
    "ph2": {"dataset_id": "903", "dataset_name": "Dataset903_PH2", "trainer": "nnUNetTrainer", "plans": "nnUNetPlans"},
    "idrid": {"dataset_id": "904", "dataset_name": "Dataset904_IDRiDLesion", "trainer": "nnUNetTrainer_100epochs", "plans": "nnUNetPlans_old512"},
    "dfuc": {"dataset_id": "905", "dataset_name": "Dataset905_DFUCUlcer", "trainer": "nnUNetTrainer_100epochs", "plans": "nnUNetPlans_old512"},
    "busi": {"dataset_id": "906", "dataset_name": "Dataset906_BUSILesion", "trainer": "nnUNetTrainer_100epochs", "plans": "nnUNetPlans_old512"},
    "isic2017": {"dataset_id": "907", "dataset_name": "Dataset907_ISIC2017Lesion", "trainer": "nnUNetTrainer_100epochs", "plans": "nnUNetPlans_old512"},
    "tn3k": {"dataset_id": "908", "dataset_name": "Dataset908_TN3KThyroidNodule", "trainer": "nnUNetTrainer_100epochs", "plans": "nnUNetPlans_old512"},
}


def _read_ids(path: Path) -> list[str]:
    return [line.strip().lstrip("\ufeff").removesuffix(".png") for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _image_path(images_dir: Path, case_id: str) -> Path:
    for name in (f"{case_id}_0000.png", f"{case_id}.png"):
        path = images_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"missing image for {case_id} in {images_dir}")


def _prepare_images(root: Path, dataset: str, split: str, work_dir: Path) -> Path:
    split_root = root / "data" / dataset / split
    ids = _read_ids(split_root / "ids.txt")
    out = work_dir / "imagesTs"
    out.mkdir(parents=True, exist_ok=True)
    for case_id in ids:
        shutil.copy2(_image_path(split_root / "images", case_id), out / f"{case_id}_0000.png")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Export nnU-Net probability maps for a release split.")
    ap.add_argument("--root", type=str, default=str(Path(__file__).resolve().parent))
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--split", required=True, help="kvasir: train/val50/test70; ph2: train/val30/test30; idrid/dfuc/busi/isic2017/tn3k: train/test, isic2017 also supports val")
    ap.add_argument("--work_dir", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--config_name", type=str, default="2d")
    ap.add_argument("--fold", type=str, default="0")
    ap.add_argument("--trainer", type=str, default="")
    ap.add_argument("--plans", type=str, default="")
    ap.add_argument("--checkpoint", type=str, default="checkpoint_final.pth")
    ap.add_argument("--npp", type=int, default=1)
    ap.add_argument("--nps", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda"], help="Release protocol uses GPU export only; no CPU fallback.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    meta = DATASETS[str(args.dataset)]
    work_dir = Path(args.work_dir).resolve() if args.work_dir else root / "outputs" / "nnunet_retrain" / str(args.dataset)
    images_ts = _prepare_images(root, str(args.dataset), str(args.split), work_dir / f"predict_{args.split}")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else root / "data" / str(args.dataset) / str(args.split) / "probs_retrained"
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["nnUNet_raw"] = str(work_dir / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(work_dir / "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(work_dir / "nnUNet_results")

    from nnunetv2.utilities.file_path_utilities import get_output_folder
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    import torch
    if str(args.device) == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export was requested by the release protocol, but torch.cuda.is_available() is False.")

    model_training_output_dir = get_output_folder(
        str(meta["dataset_id"]),
        str(args.trainer or meta["trainer"]),
        str(args.plans or meta["plans"]),
        str(args.config_name),
    )
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device(str(args.device)),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=str(model_training_output_dir),
        use_folds=(int(args.fold),),
        checkpoint_name=str(args.checkpoint),
    )
    predictor.predict_from_files(
        list_of_lists_or_source_folder=str(images_ts),
        output_folder_or_list_of_truncated_output_files=str(out_dir),
        save_probabilities=True,
        overwrite=True,
        num_processes_preprocessing=int(args.npp),
        num_processes_segmentation_export=int(args.nps),
    )
    print(f"[nnunet] probabilities written to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
