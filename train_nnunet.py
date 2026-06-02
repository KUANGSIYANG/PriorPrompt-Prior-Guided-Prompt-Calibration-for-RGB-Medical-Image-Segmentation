from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
from time import sleep
from pathlib import Path

import numpy as np
import torch
from PIL import Image


DATASETS = {
    "kvasir": {
        "dataset_id": 902,
        "dataset_name": "Dataset902_KvasirSEG2D",
        "label_name": "polyp",
        "trainer": "nnUNetTrainer",
        "plans": "nnUNetPlans",
    },
    "ph2": {
        "dataset_id": 903,
        "dataset_name": "Dataset903_PH2",
        "label_name": "lesion",
        "trainer": "nnUNetTrainer",
        "plans": "nnUNetPlans",
    },
    "idrid": {
        "dataset_id": 904,
        "dataset_name": "Dataset904_IDRiDLesion",
        "label_name": "lesion",
        "trainer": "nnUNetTrainer_100epochs",
        "plans": "nnUNetPlans_old512",
    },
    "dfuc": {
        "dataset_id": 905,
        "dataset_name": "Dataset905_DFUCUlcer",
        "label_name": "ulcer",
        "trainer": "nnUNetTrainer_100epochs",
        "plans": "nnUNetPlans_old512",
    },
    "busi": {
        "dataset_id": 906,
        "dataset_name": "Dataset906_BUSILesion",
        "label_name": "lesion",
        "trainer": "nnUNetTrainer_100epochs",
        "plans": "nnUNetPlans_old512",
    },
    "isic2017": {
        "dataset_id": 907,
        "dataset_name": "Dataset907_ISIC2017Lesion",
        "label_name": "lesion",
        "trainer": "nnUNetTrainer_100epochs",
        "plans": "nnUNetPlans_old512",
    },
    "tn3k": {
        "dataset_id": 908,
        "dataset_name": "Dataset908_TN3KThyroidNodule",
        "label_name": "nodule",
        "trainer": "nnUNetTrainer_100epochs",
        "plans": "nnUNetPlans_old512",
    },
}

OLD512_PLAN_DATASETS = {"idrid", "dfuc", "busi", "isic2017", "tn3k"}


def _read_ids(path: Path) -> list[str]:
    return [line.strip().lstrip("\ufeff").removesuffix(".png") for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_label(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(Image.open(src).convert("L"), dtype=np.uint8)
    Image.fromarray((arr > 0).astype(np.uint8), mode="L").save(dst)


def _image_path(images_dir: Path, case_id: str) -> Path:
    for name in (f"{case_id}_0000.png", f"{case_id}.png"):
        path = images_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"missing image for {case_id} in {images_dir}")


def prepare_raw_dataset(root: Path, dataset: str, raw_root: Path, train_split: str = "train") -> Path:
    meta = DATASETS[dataset]
    split_root = root / "data" / dataset / str(train_split)
    ids = _read_ids(split_root / "ids.txt")
    dataset_dir = raw_root / str(meta["dataset_name"])
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    for case_id in ids:
        _copy_image(_image_path(split_root / "images", case_id), images_tr / f"{case_id}_0000.png")
        _write_label(split_root / "labels" / f"{case_id}.png", labels_tr / f"{case_id}.png")
    dataset_json = {
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": {"background": 0, str(meta["label_name"]): 1},
        "numTraining": len(ids),
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")
    return dataset_dir


def _run(cmd: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, env=env)


def _configure_python_spawn(env: dict[str, str]) -> None:
    """Keep multiprocessing workers on the current interpreter.

    On this Windows setup, nnU-Net preprocessing can otherwise spawn workers
    via the base Anaconda interpreter instead of the active env interpreter.
    """
    exe = str(Path(sys.executable).resolve())
    env["PYTHONEXECUTABLE"] = exe
    if os.name == "nt":
        mp.set_executable(exe)


def _patch_windows_writable_crops() -> None:
    """Make nnUNet dataloader crops writable before torch.from_numpy on Windows.

    This preserves official nnUNet hyperparameters and data flow. We only avoid
    undefined behavior from non-writable numpy views that can crash the process
    on this setup.
    """
    if os.name != "nt":
        return
    import nnunetv2.training.dataloading.data_loader as dl_mod

    original_crop_and_pad_nd = dl_mod.crop_and_pad_nd

    def _crop_and_pad_nd_writable(*args, **kwargs):
        out = original_crop_and_pad_nd(*args, **kwargs)
        arr = np.asarray(out)
        if not arr.flags.writeable:
            arr = np.array(arr, copy=True)
        return arr

    dl_mod.crop_and_pad_nd = _crop_and_pad_nd_writable


def _patch_windows_multithreaded_augmenter() -> None:
    """Stabilize nnUNet DA workers on Windows without changing training hyperparameters.

    The user-requested setup keeps `batch_size=12`, `patch_size=512`, `epochs=100`,
    and `n_proc_da=4`. On this machine, the Windows batchgenerators pin-memory path
    can crash with `resource already mapped` / shared-object initialization errors.
    We also keep the worker cache shallow to reduce queue memory pressure.
    """
    if os.name != "nt":
        return

    from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
        NonDetMultiThreadedAugmenter,
    )

    original_init = NonDetMultiThreadedAugmenter.__init__
    if getattr(original_init, "_safeanchor_windows_patched", False):
        return

    def _init_windows_safe(
        self,
        data_loader,
        transform,
        num_processes,
        num_cached=2,
        seeds=None,
        pin_memory=False,
        wait_time=0.02,
    ):
        pin_memory = False
        num_cached = min(int(num_cached), 2)
        return original_init(
            self,
            data_loader,
            transform,
            num_processes,
            num_cached,
            seeds,
            pin_memory,
            wait_time,
        )

    _init_windows_safe._safeanchor_windows_patched = True
    NonDetMultiThreadedAugmenter.__init__ = _init_windows_safe


def _patch_windows_training_runtime() -> None:
    _patch_windows_writable_crops()
    _patch_windows_multithreaded_augmenter()


def _build_old512_plans_from_default(default_plans: dict, plans_name: str) -> dict:
    target_plans = json.loads(json.dumps(default_plans))
    default_cfg = default_plans["configurations"]["2d"]
    cfg = target_plans["configurations"]["2d"]
    n_stages = 8

    cfg["data_identifier"] = f"{plans_name}_2d"
    cfg["batch_size"] = 12
    cfg["patch_size"] = [512, 512]
    cfg["median_image_size_in_voxels"] = list(default_cfg["median_image_size_in_voxels"])
    cfg["spacing"] = list(default_cfg["spacing"])
    cfg["normalization_schemes"] = list(default_cfg["normalization_schemes"])
    cfg["use_mask_for_norm"] = list(default_cfg["use_mask_for_norm"])
    cfg["UNet_class_name"] = "PlainConvUNet"
    cfg["UNet_base_num_features"] = 32
    cfg["n_conv_per_stage_encoder"] = [2] * n_stages
    cfg["n_conv_per_stage_decoder"] = [2] * (n_stages - 1)
    cfg["num_pool_per_axis"] = [7, 7]
    cfg["pool_op_kernel_sizes"] = [[1, 1]] + [[2, 2]] * (n_stages - 1)
    cfg["conv_kernel_sizes"] = [[3, 3]] * n_stages
    cfg["unet_max_num_features"] = 512
    cfg["batch_dice"] = True
    cfg.pop("architecture", None)
    target_plans["plans_name"] = plans_name
    return target_plans


def _materialize_old512_variant(preprocessed_root: Path, dataset_name: str, plans_name: str) -> None:
    dataset_dir = preprocessed_root / dataset_name
    default_plans_file = dataset_dir / "nnUNetPlans.json"
    target_plans_file = dataset_dir / f"{plans_name}.json"
    default_data_dir = dataset_dir / "nnUNetPlans_2d"
    target_data_dir = dataset_dir / f"{plans_name}_2d"
    if not default_plans_file.exists():
        raise FileNotFoundError(f"default plans not found: {default_plans_file}")
    # When called before preprocessing, only the default plans JSON exists.
    # nnU-Net will create <plans_name>_2d during the subsequent preprocess call.
    # Older resumed runs may already have nnUNetPlans_2d materialized; in that
    # case we keep the compatibility symlink below.
    default_data_exists = default_data_dir.exists()

    default_plans = json.loads(default_plans_file.read_text(encoding="utf-8"))
    target_plans = _build_old512_plans_from_default(default_plans, plans_name)
    target_plans_file.write_text(json.dumps(target_plans, indent=4), encoding="utf-8")

    if not default_data_exists:
        return

    if target_data_dir.exists():
        if target_data_dir.is_symlink():
            target_data_dir.unlink()
        else:
            raise FileExistsError(f"target preprocessed dir already exists and is not a symlink: {target_data_dir}")
    os.symlink(default_data_dir, target_data_dir, target_is_directory=True)


def _run_preprocess_via_api(
    *,
    dataset_id: int,
    dataset: str,
    config_name: str,
    plans_name: str,
    preprocess_np: int,
    verify_dataset: bool,
    dry_run: bool,
) -> None:
    cmd_preview = [
        sys.executable,
        "-m",
        "nnunetv2.experiment_planning.plan_and_preprocess_entrypoints",
        "-d",
        str(dataset_id),
        "-c",
        str(config_name),
        "-np",
        str(preprocess_np),
    ]
    if bool(verify_dataset):
        cmd_preview.append("--verify_dataset_integrity")
    print("[RUN]", " ".join(cmd_preview), flush=True)
    if dry_run:
        return

    from nnunetv2.experiment_planning.plan_and_preprocess_api import (
        extract_fingerprints,
        plan_experiments,
        preprocess,
    )
    from batchgenerators.utilities.file_and_folder_operations import (
        isdir,
        isfile,
        join,
        load_json,
        maybe_mkdir_p,
    )
    from tqdm import tqdm
    from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
    from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets
    from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetNumpy
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    if int(preprocess_np) <= 1:
        def _run_fingerprint_serial(self, overwrite_existing: bool = False):
            preprocessed_output_folder = join(nnUNet_preprocessed, self.dataset_name)
            maybe_mkdir_p(preprocessed_output_folder)
            properties_file = join(preprocessed_output_folder, "dataset_fingerprint.json")

            if not isfile(properties_file) or overwrite_existing:
                from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
                from batchgenerators.utilities.file_and_folder_operations import save_json

                reader_writer_class = determine_reader_writer_from_dataset_json(
                    self.dataset_json,
                    self.dataset[self.dataset.keys().__iter__().__next__()]["images"][0],
                )
                num_foreground_samples_per_case = int(
                    self.num_foreground_voxels_for_intensitystats // len(self.dataset)
                )
                results = []
                iterable = self.dataset.keys()
                progress = tqdm(
                    iterable,
                    desc="Extracting dataset fingerprint",
                    total=len(self.dataset),
                    disable=not getattr(self, "show_progress_bar", True),
                )
                for k in progress:
                    results.append(
                        DatasetFingerprintExtractor.analyze_case(
                            self.dataset[k]["images"],
                            self.dataset[k]["label"],
                            reader_writer_class,
                            num_foreground_samples_per_case,
                        )
                    )

                shapes_after_crop = [r[0] for r in results]
                spacings = [r[1] for r in results]
                foreground_intensities_per_channel = [
                    np.concatenate([r[2][i] for r in results]) for i in range(len(results[0][2]))
                ]
                foreground_intensities_per_channel = np.array(foreground_intensities_per_channel)
                median_relative_size_after_cropping = np.median([r[4] for r in results], 0)
                num_channels = len(
                    self.dataset_json["channel_names"].keys()
                    if "channel_names" in self.dataset_json.keys()
                    else self.dataset_json["modality"].keys()
                )
                intensity_statistics_per_channel = {}
                percentiles = np.array((0.5, 50.0, 99.5))
                for i in range(num_channels):
                    percentile_00_5, median, percentile_99_5 = np.percentile(
                        foreground_intensities_per_channel[i], percentiles
                    )
                    intensity_statistics_per_channel[i] = {
                        "mean": float(np.mean(foreground_intensities_per_channel[i])),
                        "median": float(median),
                        "std": float(np.std(foreground_intensities_per_channel[i])),
                        "min": float(np.min(foreground_intensities_per_channel[i])),
                        "max": float(np.max(foreground_intensities_per_channel[i])),
                        "percentile_99_5": float(percentile_99_5),
                        "percentile_00_5": float(percentile_00_5),
                    }

                fingerprint = {
                    "spacings": spacings,
                    "shapes_after_crop": shapes_after_crop,
                    "foreground_intensity_properties_per_channel": intensity_statistics_per_channel,
                    "median_relative_size_after_cropping": median_relative_size_after_cropping,
                }
                save_json(fingerprint, properties_file)
            else:
                fingerprint = load_json(properties_file)
            return fingerprint

        def _run_preprocessor_serial(self, dataset_name_or_id, configuration_name, plans_identifier, num_processes):
            del num_processes
            dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
            assert isdir(join(nnUNet_raw, dataset_name)), "The requested dataset could not be found in nnUNet_raw"

            plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + ".json")
            assert isfile(plans_file), (
                "Expected plans file (%s) not found. Run corresponding nnUNet_plan_experiment first." % plans_file
            )
            plans = load_json(plans_file)
            plans_manager = PlansManager(plans)
            configuration_manager = plans_manager.get_configuration(configuration_name)

            dataset_json_file = join(nnUNet_preprocessed, dataset_name, "dataset.json")
            dataset_json = load_json(dataset_json_file)
            output_directory = join(nnUNet_preprocessed, dataset_name, configuration_manager.data_identifier)
            if isdir(output_directory):
                shutil.rmtree(output_directory)
            maybe_mkdir_p(output_directory)

            dataset = get_filenames_of_train_images_and_targets(join(nnUNet_raw, dataset_name), dataset_json)
            progress = tqdm(
                dataset.keys(),
                desc="Preprocessing cases",
                total=len(dataset),
                disable=not getattr(self, "show_progress_bar", True),
            )
            for k in progress:
                data, seg, properties = self.run_case(
                    dataset[k]["images"],
                    dataset[k]["label"],
                    plans_manager,
                    configuration_manager,
                    dataset_json,
                )
                nnUNetDatasetNumpy.save_case(
                    data.astype(np.float32, copy=False),
                    seg.astype(np.int16, copy=False),
                    properties,
                    join(output_directory, k),
                )
                sleep(0.0)

        DatasetFingerprintExtractor.run = _run_fingerprint_serial
        DefaultPreprocessor.run = _run_preprocessor_serial

    extract_fingerprints(
        [int(dataset_id)],
        num_processes=int(preprocess_np),
        check_dataset_integrity=bool(verify_dataset),
        clean=False,
        verbose=True,
        show_progress_bar=True,
    )
    plan_experiments([int(dataset_id)], overwrite_plans_name="nnUNetPlans")
    if str(dataset) in OLD512_PLAN_DATASETS and str(plans_name) == "nnUNetPlans_old512":
        _materialize_old512_variant(Path(nnUNet_preprocessed), maybe_convert_to_dataset_name(dataset_id), plans_name)
    preprocess(
        [int(dataset_id)],
        plans_identifier=str(plans_name),
        configurations=[str(config_name)],
        num_processes=[int(preprocess_np)],
        verbose=True,
        show_progress_bar=True,
    )


def _run_training_via_api(
    *,
    dataset_id: str,
    config_name: str,
    fold: str,
    trainer: str,
    plans: str,
    num_gpus: int,
    device: str,
    continue_train: bool,
    dry_run: bool,
) -> None:
    cmd_preview = [
        sys.executable,
        "-m",
        "nnunetv2.run.run_training",
        str(dataset_id),
        str(config_name),
        str(fold),
        "-tr",
        str(trainer),
        "-p",
        str(plans),
        "-num_gpus",
        str(num_gpus),
        "-device",
        str(device),
    ]
    if bool(continue_train):
        cmd_preview.append("--c")
    print("[RUN]", " ".join(cmd_preview), flush=True)
    if dry_run:
        return

    from nnunetv2.run.run_training import run_training

    _patch_windows_training_runtime()
    run_training(
        dataset_name_or_id=str(dataset_id),
        configuration=str(config_name),
        fold=str(fold),
        trainer_class_name=str(trainer),
        plans_identifier=str(plans),
        pretrained_weights=None,
        num_gpus=int(num_gpus),
        export_validation_probabilities=False,
        continue_training=bool(continue_train),
        only_run_validation=False,
        disable_checkpointing=False,
        val_with_best=False,
        device=torch.device(str(device)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare and train the upstream nnU-Net coarse model for the clean release splits.")
    ap.add_argument("--root", type=str, default=str(Path(__file__).resolve().parent))
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--train_split", type=str, default="train", help="Release data split used as nnU-Net training set.")
    ap.add_argument("--work_dir", type=str, default="")
    ap.add_argument("--config_name", type=str, default="2d")
    ap.add_argument("--fold", type=str, default="0")
    ap.add_argument("--trainer", type=str, default="")
    ap.add_argument("--plans", type=str, default="")
    ap.add_argument("--num_gpus", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--preprocess_np", type=int, default=8)
    ap.add_argument("--n_proc_da", type=int, default=-1)
    ap.add_argument("--skip_preprocess", action="store_true")
    ap.add_argument("--verify_dataset", action="store_true")
    ap.add_argument("--continue_train", action="store_true")
    ap.add_argument("--prepare_only", action="store_true")
    ap.add_argument("--preprocess_only", action="store_true")
    ap.add_argument("--materialize_old512_only", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    meta = DATASETS[str(args.dataset)]
    work_dir = Path(args.work_dir).resolve() if args.work_dir else root / "outputs" / "nnunet_retrain" / str(args.dataset)
    raw_root = work_dir / "nnUNet_raw"
    preprocessed_root = work_dir / "nnUNet_preprocessed"
    results_root = work_dir / "nnUNet_results"
    dataset_dir = prepare_raw_dataset(root, str(args.dataset), raw_root, str(args.train_split))
    print(f"[nnunet] prepared raw dataset: {dataset_dir}", flush=True)
    if args.prepare_only:
        return

    env = os.environ.copy()
    env["nnUNet_raw"] = str(raw_root)
    env["nnUNet_preprocessed"] = str(preprocessed_root)
    env["nnUNet_results"] = str(results_root)
    os.environ["nnUNet_raw"] = str(raw_root)
    os.environ["nnUNet_preprocessed"] = str(preprocessed_root)
    os.environ["nnUNet_results"] = str(results_root)
    requested_n_proc_da = int(args.n_proc_da)
    if requested_n_proc_da < 0:
        requested_n_proc_da = 4 if str(args.dataset) in OLD512_PLAN_DATASETS else requested_n_proc_da
    if requested_n_proc_da >= 0:
        env["nnUNet_n_proc_DA"] = str(requested_n_proc_da)
        os.environ["nnUNet_n_proc_DA"] = str(requested_n_proc_da)
    _configure_python_spawn(env)
    trainer = str(args.trainer or meta["trainer"])
    plans = str(args.plans or meta["plans"])
    dataset_id = str(meta["dataset_id"])

    if bool(args.materialize_old512_only):
        _materialize_old512_variant(preprocessed_root, str(meta["dataset_name"]), str(plans))
        print(f"[nnunet] materialized {plans} for {meta['dataset_name']}", flush=True)
        return

    if not bool(args.skip_preprocess):
        _run_preprocess_via_api(
            dataset_id=int(dataset_id),
            dataset=str(args.dataset),
            config_name=str(args.config_name),
            plans_name=str(plans),
            preprocess_np=int(args.preprocess_np),
            verify_dataset=bool(args.verify_dataset),
            dry_run=bool(args.dry_run),
        )
    if bool(args.preprocess_only):
        return
    _run_training_via_api(
        dataset_id=dataset_id,
        config_name=str(args.config_name),
        fold=str(args.fold),
        trainer=trainer,
        plans=plans,
        num_gpus=int(args.num_gpus),
        device=str(args.device),
        continue_train=bool(args.continue_train),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
