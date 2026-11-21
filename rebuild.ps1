$ErrorActionPreference = "Stop"

$ImageName = "rags_img"
$ContainerName = "global_rag_app"
$ProjectRoot = $PSScriptRoot

# Ensure host mount directories and files exist before starting the container
$ChromaDir = Join-Path $ProjectRoot "chroma_db_global"
$LogsDir = Join-Path $ProjectRoot "logs"
$EnvFile = Join-Path $ProjectRoot ".env"
$ChatHistoryFile = Join-Path $ProjectRoot "chat_history.json"
New-Item -ItemType Directory -Force -Path $ChromaDir, $LogsDir | Out-Null
if (-not (Test-Path $ChatHistoryFile)) {
    [System.IO.File]::WriteAllText($ChatHistoryFile, "[]")
}

Write-Host "Building Docker image '$ImageName'..."
docker build -t $ImageName $ProjectRoot
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker build failed. Keeping existing container (if any) untouched."
    exit $LASTEXITCODE
}

# Remove any running container currently occupying port 8501
$portContainers = docker ps -q --filter "publish=8501"
if ($portContainers) {
    Write-Host "Stopping container(s) using port 8501..."
    docker rm -f $portContainers | Out-Null
}

# Remove existing container named $ContainerName (if stopped or still present)
$existing = docker ps -aq -f "name=^${ContainerName}$"
if ($existing) {
    Write-Host "Removing previous container '$ContainerName'..."
    docker rm -f $ContainerName | Out-Null
}

Write-Host "Starting container '$ContainerName' on port 8501..."
docker run -d `
    --name $ContainerName `
    -p 8501:8501 `
    -v "${EnvFile}:/app/.env:ro" `
    -v "${ChromaDir}:/app/chroma_db_global" `
    -v "${LogsDir}:/app/logs" `
    -v "${ChatHistoryFile}:/app/chat_history.json" `
    $ImageName

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Done. App is running at http://localhost:8501"
