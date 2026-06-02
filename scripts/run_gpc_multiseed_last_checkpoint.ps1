$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Py = $env:SAFEANCHOR_PYTHON
if (-not $Py) { $Py = "python" }
$Config = ".\configs\branch_safeanchor_gpc_v8c_paper_release.json"
$Seeds = @(20260522, 20260527, 20260530)

Set-Location $Root

foreach ($Seed in $Seeds) {
  Write-Host "==== SafePrompt-BED GPC Kvasir seed $Seed ===="
  & $Py .\train.py --stage gpc `
    --config $Config `
    --dataset kvasir `
    --epochs 3 `
    --device cuda `
    --seed $Seed `
    --out_dir ".\outputs\gpc_safeprompt_bed_kvasir_train880_3ep_seed$Seed" `
    --print_every 220

  & $Py .\run_safeanchor_gpc.py `
    --config $Config `
    --split val50 `
    --gpc_ckpt ".\outputs\gpc_safeprompt_bed_kvasir_train880_3ep_seed$Seed\checkpoint_last.pth" `
    --device cuda `
    --out_name "safeprompt_bed_kvasir_val50_seed$Seed"

  & $Py .\run_safeanchor_gpc.py `
    --config $Config `
    --split test70 `
    --gpc_ckpt ".\outputs\gpc_safeprompt_bed_kvasir_train880_3ep_seed$Seed\checkpoint_last.pth" `
    --device cuda `
    --out_name "safeprompt_bed_kvasir_test70_seed$Seed"

  Write-Host "==== SafePrompt-BED GPC PH2 seed $Seed ===="
  & $Py .\train.py --stage gpc `
    --config $Config `
    --dataset ph2 `
    --epochs 3 `
    --device cuda `
    --seed $Seed `
    --out_dir ".\outputs\gpc_safeprompt_bed_ph2_train140_3ep_seed$Seed" `
    --print_every 140

  & $Py .\run_safeanchor_gpc.py `
    --config $Config `
    --split ph2_val30 `
    --gpc_ckpt ".\outputs\gpc_safeprompt_bed_ph2_train140_3ep_seed$Seed\checkpoint_last.pth" `
    --device cuda `
    --out_name "safeprompt_bed_ph2_val30_seed$Seed"

  & $Py .\run_safeanchor_gpc.py `
    --config $Config `
    --split ph2_test30 `
    --gpc_ckpt ".\outputs\gpc_safeprompt_bed_ph2_train140_3ep_seed$Seed\checkpoint_last.pth" `
    --device cuda `
    --out_name "safeprompt_bed_ph2_test30_seed$Seed"
}

& $Py .\summarize_multiseed.py --seeds $($Seeds -join ",")
