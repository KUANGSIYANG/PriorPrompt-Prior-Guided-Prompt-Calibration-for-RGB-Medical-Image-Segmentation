$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Py = $env:SAFEANCHOR_PYTHON
if (-not $Py) { $Py = "python" }
$Config = ".\configs\branch_safeanchor_gpc_v8c_paper_release.json"
Set-Location $Root
& $Py .\run_safeanchor_gpc.py --config $Config --split val50 --gpc_ckpt .\outputs\gpc_v8c_learnthr_unified_kvasir_train880_3ep_seed20260527\checkpoint_last.pth --device cuda --out_name gpc_v8c_seed20260527_kvasir_val50_last
& $Py .\run_safeanchor_gpc.py --config $Config --split test70 --gpc_ckpt .\outputs\gpc_v8c_learnthr_unified_kvasir_train880_3ep_seed20260527\checkpoint_last.pth --device cuda --out_name gpc_v8c_seed20260527_kvasir_test70_last
& $Py .\run_safeanchor_gpc.py --config $Config --split ph2_val30 --gpc_ckpt .\outputs\gpc_v8c_learnthr_unified_ph2_train140_3ep_seed20260527\checkpoint_last.pth --device cuda --out_name gpc_v8c_seed20260527_ph2_val30_last
& $Py .\run_safeanchor_gpc.py --config $Config --split ph2_test30 --gpc_ckpt .\outputs\gpc_v8c_learnthr_unified_ph2_train140_3ep_seed20260527\checkpoint_last.pth --device cuda --out_name gpc_v8c_seed20260527_ph2_test30_last
