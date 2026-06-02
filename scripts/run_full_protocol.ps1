$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Py = $env:SAFEANCHOR_PYTHON
if (-not $Py) { $Py = "python" }
$Config = ".\configs\branch_safeanchor_gpc_v8c_paper_release.json"
Set-Location $Root
& powershell -ExecutionPolicy Bypass -File .\scripts\run_train_kvasir.ps1
& powershell -ExecutionPolicy Bypass -File .\scripts\run_train_ph2.ps1
& powershell -ExecutionPolicy Bypass -File .\scripts\run_eval_all.ps1
& powershell -ExecutionPolicy Bypass -File .\scripts\run_audit.ps1
