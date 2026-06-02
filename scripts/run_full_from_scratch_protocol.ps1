$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

powershell -ExecutionPolicy Bypass -File .\scripts\run_train_nnunet_kvasir.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_train_nnunet_ph2.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_train_bb2_kvasir.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_train_bb2_ph2.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_full_protocol.ps1
