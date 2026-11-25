# Vitalis AI - one-shot setup + run + test script (Windows / PowerShell)
#
# Usage:
#   .\run_and_test.ps1                 # setup (if needed) + start backend + run all checks, then stop backend
#   .\run_and_test.ps1 -SkipTests      # setup + start backend only, leave it running (Ctrl+C to stop)
#   .\run_and_test.ps1 -Heldout        # also run the held-out generalisation set (evaluate.py --heldout)
#   .\run_and_test.ps1 -KeepRunning    # leave the backend running after tests finish
#   .\run_and_test.ps1 -Consult        # also run the Patient Consult probe against the real Pi-hosted medical model (~15 min)
#
# What it does, in order:
#   1. Create the venv and install backend/requirements.txt if missing.
#   2. Generate the synthetic patient DB and build the vector index if missing (first run only).
#   3. Make sure Ollama is running and the configured model is pulled.
#   4. Start the FastAPI backend (uvicorn) in the background and wait for /api/health.
#   5. Run: pytest unit tests -> smoke_test.py -> voice_roundtrip.py -> evaluate.py.
#   6. Print a pass/fail summary and stop the backend it started (unless -KeepRunning).

param(
    [switch]$SkipTests,
    [switch]$Heldout,
    [switch]$KeepRunning,
    [switch]$Consult,
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot        # deployment/ sits one level below the repo root
$backend = Join-Path $root "src\backend"
$py = Join-Path $root ".venv\Scripts\python.exe"
$results = @()  # (name, pass:bool)

function Add-Result($name, $ok) {
    $script:results += [pscustomobject]@{ Name = $name; Pass = $ok }
    if ($ok) { Write-Host "[PASS] $name" -ForegroundColor Green }
    else     { Write-Host "[FAIL] $name" -ForegroundColor Red }
}

# ---------- 1. venv + deps ----------
if (-not (Test-Path $py)) {
    Write-Host "Creating venv..." -ForegroundColor Cyan
    python -m venv (Join-Path $root ".venv")
}
Write-Host "Checking backend deps (fast no-op if already installed)..." -ForegroundColor Cyan
& $py -m pip install -q -r (Join-Path $backend "requirements.txt")

# ---------- 2. data: synthetic patients + vector index ----------
if (-not (Test-Path (Join-Path $root "data\patients.sqlite"))) {
    Write-Host "Generating synthetic patient DB..." -ForegroundColor Cyan
    Push-Location $backend; & $py "scripts\generate_patients.py"; Pop-Location
}
if (-not (Test-Path (Join-Path $root "data\chroma"))) {
    Write-Host "Building vector index (downloads the embedding model once)..." -ForegroundColor Cyan
    Push-Location $backend; & $py "scripts\build_index.py"; Pop-Location
}

# ---------- 3. Ollama ----------
# Checked by hitting its API, not by process name: the Windows app runs as "ollama app.exe"
# (tray app with its own embedded server), which `Get-Process ollama` does not match. Starting
# a second `ollama serve` while that's already up makes both instances fight over port 11434 -
# causing exactly the kind of mid-request connection drop this check exists to avoid.
$ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollamaCmd) {
    $ollamaUp = $false
    try { Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 2 | Out-Null; $ollamaUp = $true } catch {}
    if (-not $ollamaUp) {
        Write-Host "Starting Ollama..." -ForegroundColor Cyan
        Start-Process $ollamaCmd.Source -ArgumentList "serve" -WindowStyle Hidden
        for ($i = 0; $i -lt 15; $i++) {
            Start-Sleep -Seconds 1
            try { Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 2 | Out-Null; $ollamaUp = $true; break } catch {}
        }
    }
    $envExample = Get-Content (Join-Path $root "config\.env.example") -Raw
    $model = if ($envExample -match 'HVA_LLM_MODEL=(\S+)') { $Matches[1] } else { "qwen2.5:3b-instruct" }
    $have = (& $ollamaCmd.Source list) -join "`n"
    if ($have -notmatch [regex]::Escape($model)) {
        Write-Host "Pulling LLM model $model (first run only, this can take a while)..." -ForegroundColor Cyan
        & $ollamaCmd.Source pull $model
    }
} else {
    Write-Warning "Ollama not found on PATH - LLM-backed answers will fail health/evaluate checks. Install from https://ollama.com"
}

# ---------- 4. start backend ----------
$healthUrl = "http://127.0.0.1:$Port/api/health"
$proc = $null
$reused = $false
try {
    Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2 | Out-Null
    $reused = $true
    Write-Host "Backend already running and healthy on port $Port - reusing it (will not be stopped at the end)." -ForegroundColor Yellow
} catch {
    $portBusy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($portBusy) {
        $pids = ($portBusy.OwningProcess | Select-Object -Unique) -join ", "
        Write-Error "Port $Port is already in use by PID(s) $pids but isn't answering $healthUrl. Stop it first (Stop-Process -Id <pid>) or pass -Port <other>."
        exit 1
    }
    Write-Host "Starting backend on port $Port..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Force (Join-Path $root "logs") | Out-Null
    $outLog = Join-Path $root "logs\server.out.log"
    $errLog = Join-Path $root "logs\server.err.log"
    # The test instance runs with Google sign-in switched off: the smoke/eval scripts cannot sign in.
    $env:HVA_AUTH_DISABLED = "true"
    $proc = Start-Process $py -ArgumentList "-m", "uvicorn", "app.main:app", "--port", "$Port" `
        -WorkingDirectory $backend -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog
    Remove-Item Env:HVA_AUTH_DISABLED -ErrorAction SilentlyContinue

    $up = $false
    Write-Host "Waiting for it to become healthy (first run loads the embedding/whisper models, can take ~1-2 min)..." -ForegroundColor Cyan
    for ($i = 0; $i -lt 90; $i++) {
        Start-Sleep -Seconds 2
        try { Invoke-RestMethod -Uri $healthUrl -TimeoutSec 3 | Out-Null; $up = $true; break } catch {}
        if (-not (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue)) {
            Write-Error "Backend process exited before becoming healthy - see server.err.log"
            break
        }
    }
    Add-Result "Backend health ($healthUrl)" $up
    if (-not $up) {
        Write-Host "--- server.err.log ---"; Get-Content $errLog -Tail 40 -ErrorAction SilentlyContinue
    }
}
if ($reused) { $up = $true; Add-Result "Backend health ($healthUrl)" $true }

# ---------- 5. tests ----------
if (-not $SkipTests -and $up) {
    Push-Location $backend
    try {
        Write-Host "`nRunning unit tests..." -ForegroundColor Cyan
        & $py -m pytest (Join-Path $root "tests") -q -c (Join-Path $root "pytest.ini") --rootdir $root
        Add-Result "Unit tests (pytest)" ($LASTEXITCODE -eq 0)

        Write-Host "`nRunning smoke test (end-to-end API)..." -ForegroundColor Cyan
        & $py "scripts\smoke_test.py" "http://localhost:$Port"
        Add-Result "Smoke test" ($LASTEXITCODE -eq 0)

        Write-Host "`nRunning voice round-trip (Piper -> Whisper -> pipeline -> Piper)..." -ForegroundColor Cyan
        & $py "scripts\voice_roundtrip.py" "http://localhost:$Port"
        Add-Result "Voice round-trip" ($LASTEXITCODE -eq 0)

        Write-Host "`nRunning accuracy/latency evaluation..." -ForegroundColor Cyan
        & $py "scripts\evaluate.py" --show-failures
        Add-Result "Evaluate (accuracy/latency)" ($LASTEXITCODE -eq 0)

        if ($Consult) {
            Write-Host "`nRunning Patient Consult probe (real medical model, dialogue index, web, map - slow on a Pi)..." -ForegroundColor Cyan
            $env:PYTHONIOENCODING = "utf-8"
            & $py "scripts\consult_probe.py" "http://localhost:$Port"
            Add-Result "Patient Consult probe" ($LASTEXITCODE -eq 0)
        }

        if ($Heldout) {
            Write-Host "`nRunning held-out generalisation set..." -ForegroundColor Cyan
            & $py "scripts\evaluate.py" --heldout
            Add-Result "Evaluate (held-out)" ($LASTEXITCODE -eq 0)
        }
    } finally {
        Pop-Location
    }
} elseif ($SkipTests) {
    Write-Host "`n-SkipTests set: backend left running, no checks executed." -ForegroundColor Yellow
}

# ---------- 6. summary + cleanup ----------
if (-not $SkipTests) {
    Write-Host "`n==== Summary ====" -ForegroundColor Cyan
    $results | ForEach-Object {
        $color = if ($_.Pass) { "Green" } else { "Red" }
        Write-Host ("{0,-40} {1}" -f $_.Name, $(if ($_.Pass) { "PASS" } else { "FAIL" })) -ForegroundColor $color
    }
    $failed = ($results | Where-Object { -not $_.Pass }).Count
}

if ($reused) {
    Write-Host "`nBackend was already running before this script started - left it running on http://localhost:$Port" -ForegroundColor Yellow
} elseif ($KeepRunning -or $SkipTests) {
    Write-Host "`nBackend still running (PID $($proc.Id)) on http://localhost:$Port - stop it with: Stop-Process -Id $($proc.Id)" -ForegroundColor Yellow
} elseif ($proc) {
    Write-Host "`nStopping backend (PID $($proc.Id))..." -ForegroundColor Cyan
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
}

if (-not $SkipTests) {
    if ($failed -gt 0) { exit 1 } else { exit 0 }
}
