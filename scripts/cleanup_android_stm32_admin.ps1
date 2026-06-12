# Ejecutar como Administrador (clic derecho -> Ejecutar con PowerShell como administrador)
# Elimina Android Studio + STM32 en Program Files (requiere permisos elevados)

$ErrorActionPreference = 'Continue'
$targets = @(
    'C:\Program Files\Android',
    'C:\Program Files\STMicroelectronics',
    'C:\Program Files (x86)\STMicroelectronics'
)

foreach ($p in $targets) {
    if (-not (Test-Path $p)) {
        Write-Host "SKIP (no existe): $p"
        continue
    }
    $size = (Get-ChildItem $p -Recurse -File -Force -ErrorAction SilentlyContinue |
        Measure-Object Length -Sum).Sum
    Write-Host ("Borrando {0:N2} GB -> {1}" -f ($size / 1GB), $p)
    Remove-Item -LiteralPath $p -Recurse -Force -ErrorAction SilentlyContinue
    if (Test-Path $p) {
        Write-Warning "No se pudo borrar por completo: $p (desinstalar desde Configuracion > Aplicaciones)"
    } else {
        Write-Host "OK: $p"
    }
}

Write-Host ""
Write-Host "Libre en C: $([math]::Round((Get-PSDrive C).Free / 1GB, 1)) GB"
