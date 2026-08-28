# run.ps1 - start the screener service
$root = $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "run setup.ps1 first" -ForegroundColor Red; exit 1 }
Write-Host "http://127.0.0.1:8790" -ForegroundColor Green
& $py (Join-Path $root "server.py")
