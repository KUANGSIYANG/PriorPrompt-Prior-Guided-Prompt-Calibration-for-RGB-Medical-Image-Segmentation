$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Py = $env:SAFEANCHOR_PYTHON
if (-not $Py) { $Py = "python" }
$Config = ".\configs\branch_safeanchor_gpc_v8c_paper_release.json"
Set-Location $Root
& $Py .\audit_gpc_clean_mainline.py --root $Root
