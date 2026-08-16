param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

function Test-Url {
    param([string]$Url)

    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
        return $response.StatusCode -eq 200
    }
    catch {
        return $false
    }
}

Write-Host "[PunditRAG] Checking Docker..." -ForegroundColor Cyan
try {
    docker info *> $null
}
catch {
    Write-Host "Docker Desktop is not running. Start Docker Desktop first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[PunditRAG] Building changed layers and starting all services..." -ForegroundColor Cyan
docker compose up -d --build

if ($LASTEXITCODE -ne 0) {
    Write-Host "Startup failed. Run: docker compose logs app" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit $LASTEXITCODE
}

Write-Host "[PunditRAG] Waiting for services..." -ForegroundColor Cyan
$deadline = (Get-Date).AddMinutes(5)
$queryReady = $false
$importReady = $false

while ((Get-Date) -lt $deadline) {
    $queryReady = Test-Url "http://127.0.0.1:8001/health"
    $importReady = Test-Url "http://127.0.0.1:8000/health"

    if ($queryReady -and $importReady) {
        break
    }

    $importStatus = if ($importReady) { "ready" } else { "waiting" }
    $queryStatus = if ($queryReady) { "ready" } else { "waiting" }
    Write-Host "`r[PunditRAG] Import API: $importStatus | Query API: $queryStatus   " -NoNewline
    Start-Sleep -Seconds 3
}

Write-Host ""
if (-not ($queryReady -and $importReady)) {
    Write-Host "Services did not become ready. Recent app logs:" -ForegroundColor Yellow
    docker compose logs --tail 80 app
    Read-Host "Press Enter to exit"
    exit 1
}

$workspaceUrl = "http://127.0.0.1:8001/query/html"
Write-Host "[PunditRAG] Workspace: $workspaceUrl" -ForegroundColor Green
Write-Host "[PunditRAG] Import API: http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host "[PunditRAG] Query API: http://127.0.0.1:8001/docs" -ForegroundColor Green
Start-Process $workspaceUrl
