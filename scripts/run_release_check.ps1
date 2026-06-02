$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Py = $env:SAFEANCHOR_PYTHON
if (-not $Py) { $Py = "python" }
Set-Location $Root

& $Py .\check_release_complete.py
& $Py .\audit_gpc_clean_mainline.py --root $Root
