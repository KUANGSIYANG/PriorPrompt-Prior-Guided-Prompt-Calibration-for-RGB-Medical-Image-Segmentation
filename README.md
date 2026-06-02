# SafePrompt-BED V8C Paper Release

This directory is a clean paper repository for the V8C mainline only. It contains the code, configs, data splits, frozen upstream weights, archived predictions, metrics, and scripts needed to train and evaluate the complete pipeline:

```text
dataset-specific nnU-Net prior
-> BED / BB2 boundary-evidence prompt adapter
-> GPC global prompt calibration
-> frozen MobileSAM one-pass decoder
-> learned final threshold
-> final binary mask
```

The release intentionally excludes non-mainline branches such as area gates, candidate selectors, point prompts, recurrent decoding, second-pass decoding, post-refiners, native-prompt branches, BB2v2 branches, external-dataset experiments, validation-best checkpoint selection, and checkpoint soups.

## Mainline Scope

The paper method is `SafePrompt-BED V8C`, also recorded in configs as:

```text
SafeAnchor-GPC-v8c Teacher-Free NoResidual NoTemperature LearnedThreshold
```

The executable graph is the same for Kvasir-SEG and PH2. Dataset-specific items are limited to images, labels, nnU-Net probability priors, the frozen BED/BB2 checkpoint, and the GPC checkpoint.

## Repository Layout

```text
paper_release_v8c/
  configs/
    branch_safeanchor_gpc_v8c_paper_release.json
    branch_safeanchor_gpc_v8c_teacher_free_nores_notemp_learnthr_unified.json

  safeanchor/
    bb2_adapter.py
    bb2_channel_prune.py
    generalizable_prior_calibrator.py
    gpc_data.py
    mobile_sam_wrapper.py
    ...

  vendor/mobile_sam/
    MobileSAM dependency used by the frozen decoder wrapper.

  data/
    kvasir/{train,val50,test70}/{images,labels,probs,ids.txt}
    ph2/{train,val30,test30}/{images,labels,probs,ids.txt}

  assets/
    checkpoints/
      mobile_sam.pt
      kvasir_bb2_checkpoint_best.pth
      ph2_bb2_checkpoint_best.pth
    nnunet_weights/
      kvasir_100ep_old512hexa/fold_0/checkpoint_final.pth
      ph2_30ep/fold_0/checkpoint_best.pth

  outputs/
    gpc_v8c_learnthr_unified_kvasir_train880_3ep_v1/
    gpc_v8c_learnthr_unified_ph2_train140_3ep_v1/
    metrics/
    figures/

  scripts/
    run_train_nnunet_kvasir.ps1
    run_train_nnunet_ph2.ps1
    run_train_bb2_kvasir.ps1
    run_train_bb2_ph2.ps1
    run_train_kvasir.ps1
    run_train_ph2.ps1
    run_eval_all.ps1
    run_full_protocol.ps1
    run_full_from_scratch_protocol.ps1
    run_gpc_multiseed_last_checkpoint.ps1
    run_release_check.ps1
```

## Training Stages

The standard dispatcher is:

```powershell
python .\train.py --stage nnunet --dataset kvasir --device cuda
python .\train.py --stage bb2 --dataset kvasir --epochs 3 --device cuda
python .\train.py --stage gpc --dataset kvasir --epochs 3 --device cuda
```

The three stages are:

1. `nnunet`: train a dataset-specific nnU-Net prior and export frozen foreground probability maps.
2. `bb2`: train the BED/BB2 boundary-evidence prompt adapter used as the frozen dense-prompt bridge.
3. `gpc`: train the V8C global prompt calibration policy with the upstream nnU-Net prior, BED/BB2 adapter, and MobileSAM decoder frozen.

## Full Protocol

Retrain only the final V8C GPC stage:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_full_protocol.ps1
```

Run the full from-scratch protocol including nnU-Net, probability export, BB2, GPC, evaluation, and audit:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_full_from_scratch_protocol.ps1
```

Evaluate archived `checkpoint_last.pth` weights:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_eval_all.ps1
```

Check repository completeness:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_release_check.ps1
```

## Archived Main Results

All archived V8C results use full 3-epoch GPC training and `checkpoint_last.pth`, not validation-best selection or checkpoint averaging.

| Dataset split | Dice | ASD | HD95 |
| --- | ---: | ---: | ---: |
| Kvasir val50 | 90.5115 | 8.4902 | 32.0095 |
| Kvasir test70 | 85.2218 | 19.8053 | 63.6669 |
| Kvasir test120 weighted | 87.4258 | 15.0907 | 50.4763 |
| PH2 val30 | 95.4095 | 9.8026 | 32.7226 |
| PH2 test30 | 92.3030 | 15.4886 | 47.4019 |
| PH2 test60 weighted | 93.8562 | 12.6456 | 40.0622 |

## Baseline

The paper baseline is `nnUNet-only`: thresholding the upstream nnU-Net probability map directly, without BED/BB2, MobileSAM, or GPC.

```powershell
python .\eval_nnunet_only_baseline.py --config .\configs\branch_safeanchor_gpc_v8c_paper_release.json --split val50
python .\eval_nnunet_only_baseline.py --config .\configs\branch_safeanchor_gpc_v8c_paper_release.json --split test70
python .\eval_nnunet_only_baseline.py --config .\configs\branch_safeanchor_gpc_v8c_paper_release.json --split ph2_val30
python .\eval_nnunet_only_baseline.py --config .\configs\branch_safeanchor_gpc_v8c_paper_release.json --split ph2_test30
```

## Notes

The PH2 nnU-Net weight path contains the historical directory name `ph2_30ep`, but the manuscript wording should describe the upstream prior as a dataset-specific converged/frozen nnU-Net probability prior rather than presenting the directory name as a method claim.
