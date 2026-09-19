$ErrorActionPreference = "Stop"

Write-Host "== PayPal Agentic System setup ==" -ForegroundColor Cyan

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python was not found. Install Python 3.11+ and make sure 'python' is on PATH."
}

$py = python
$version = & $py --version
Write-Host "Using $version"

& $py -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 'Python 3.11+ is required')"

if (-not (Test-Path .venv)) {
    & $py -m venv .venv
}

$venvPy = Join-Path $PWD ".venv\Scripts\python.exe"
& $venvPy -m pip install --upgrade pip
& $venvPy -m pip install -e ".[test]"

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "Created .env from .env.example. Add GROQ_API_KEY only if you need LLM mode." -ForegroundColor Yellow
}

Write-Host "Running tests..." -ForegroundColor Cyan
& $venvPy -m pytest -q

Write-Host "Setup and tests completed successfully." -ForegroundColor Green
