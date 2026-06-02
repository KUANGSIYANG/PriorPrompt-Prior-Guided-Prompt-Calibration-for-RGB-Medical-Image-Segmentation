$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Py = $env:SAFEANCHOR_PYTHON
if (-not $Py) { $Py = "python" }
$Config = ".\configs\branch_safeanchor_gpc_v8c_paper_release.json"
Set-Location $Root
& $Py .\train.py --stage bb2 --config $Config --dataset kvasir --epochs 3 --device cuda --out_dir .\outputs\bb2_kvasir_train880_3ep_retrain
