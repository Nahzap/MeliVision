# Inicia MetricLearning desde cualquier directorio de PowerShell.
# Uso: .\start-metriclearning.ps1
#      .\start-metriclearning.ps1 --cli train --config config.yaml

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$App = Join-Path $ProjectRoot "app.py"

if (-not (Test-Path $Python)) {
    Write-Error "No se encontró el entorno virtual en: $Python"
    exit 1
}

Set-Location $ProjectRoot
& $Python $App @args
