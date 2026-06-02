$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Py = $env:SAFEANCHOR_PYTHON
if (-not $Py) { $Py = "python" }
$Config = ".\configs\branch_safeanchor_gpc_v8c_paper_release.json"
Set-Location $Root
& $Py .\train.py --stage gpc `
  --config $Config `
  --dataset ph2 `
  --epochs 3 `
  --device cuda `
  --out_dir .\outputs\gpc_v8c_learnthr_unified_ph2_train140_3ep_v1
