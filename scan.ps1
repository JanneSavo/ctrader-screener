# scan.ps1 - one headless scan, prints the table, exits. Good for Task Scheduler.
$root = $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "run setup.ps1 first" -ForegroundColor Red; exit 1 }
& $py (Join-Path $root "server.py") --scan
