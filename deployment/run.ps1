# Starts Ollama (if needed), the FastAPI backend and the React dev server.   Usage: .\run.ps1
# The backend listens on all interfaces so the Raspberry Pi client (pi_client.py) can reach it over the LAN.
param([int]$Port = 8000)
$root = Split-Path -Parent $PSScriptRoot        # deployment/ sits one level below the repo root
$backend = Join-Path $root "src\backend"
$py = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) { Write-Error "Create the venv first (see README Quick start)."; exit 1 }
if (-not (Test-Path (Join-Path $root "data\patients.sqlite"))) { & $py (Join-Path $backend "scripts\generate_patients.py") }
if (-not (Test-Path (Join-Path $root "data\chroma"))) { & $py (Join-Path $backend "scripts\build_index.py") }

# Check Ollama by its API, not process name: the Windows tray app runs as "ollama app.exe", and starting a
# second `ollama serve` next to it makes the two fight over port 11434 and drop requests.
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollama) {
    try { Invoke-RestMethod http://127.0.0.1:11434/api/tags -TimeoutSec 2 | Out-Null }
    catch { Start-Process $ollama.Source -ArgumentList "serve" -WindowStyle Hidden }
}

if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    $healthy = $false
    try { Invoke-RestMethod "http://127.0.0.1:$Port/api/health" -TimeoutSec 5 | Out-Null; $healthy = $true } catch {}
    if (-not $healthy) {
        Write-Error "Port $Port is in use by something that is not this backend. Stop it first or pass -Port."
        exit 1
    }
    Write-Host "Backend already running on port $Port - reusing it." -ForegroundColor Yellow
} else {
    Start-Process $py -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "$Port" -WorkingDirectory $backend
}

$ip = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
       Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" -and $_.PrefixOrigin -ne "WellKnown" } |
       Select-Object -First 1).IPAddress
Write-Host "Backend: http://localhost:$Port  (Pi / LAN: http://${ip}:$Port)" -ForegroundColor Green
Write-Host "UI:      http://localhost:5173" -ForegroundColor Green

Push-Location (Join-Path $root "src\frontend")
if (-not (Test-Path node_modules)) { npm install }
npm run dev
Pop-Location
