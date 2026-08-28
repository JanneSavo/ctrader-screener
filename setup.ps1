# setup.ps1 - create the venv, install dependencies, verify the install.
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Write-Host "cTrader Screener setup -> $root" -ForegroundColor Cyan

if (-not (Test-Path (Join-Path $root "server.py"))) {
    Write-Host "run this from the repository root" -ForegroundColor Red; exit 1
}

if (-not (Test-Path (Join-Path $root ".venv"))) {
    Write-Host "creating .venv..." -ForegroundColor Cyan
    python -m venv (Join-Path $root ".venv")
}
$py = Join-Path $root ".venv\Scripts\python.exe"

Write-Host "installing dependencies (a minute or two)..." -ForegroundColor Cyan
& $py -m pip install --upgrade pip --quiet
& $py -m pip install -r (Join-Path $root "requirements.txt") --quiet

$cfg = Join-Path $root "config.yaml"
if (-not (Test-Path $cfg)) {
    Copy-Item (Join-Path $root "config.example.yaml") $cfg
    Write-Host "created config.yaml from the example" -ForegroundColor Yellow
}

Write-Host "verifying..." -ForegroundColor Cyan
& $py -c "import fastapi, uvicorn, yaml, pandas, numpy, httpx, mcp; print('  dependencies ok')"
& $py -c "import sys; sys.path.insert(0,'.'); from strategies import registry; print('  strategies:', ', '.join(sorted(registry())))"

Write-Host ""
Write-Host "done." -ForegroundColor Green
Write-Host "next:" -ForegroundColor Yellow
Write-Host "  1. start cTrader Desktop with AI Agent Connect enabled"
Write-Host "  2. .\.venv\Scripts\python.exe ctrader_mcp.py --dump-tools"
Write-Host "     check the resolved tool names, and pin any that are wrong"
Write-Host "     in config.yaml under ctrader.tools"
Write-Host "  3. .\run.ps1   ->  http://127.0.0.1:8790"
