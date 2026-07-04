# run_benchmarks_rtx.ps1 — matriz de benchmarks de la RTX 4070 Ti (CUDA puro).
# Corre en la RTX (C:\km\parallel_project). La energía la mide el propio binario
# por NVML (columnas avg_power_w/energy_wh); speedup/efficiency los completa
# aggregate_power.py. Ver PLAN §6, docs/runbook_rtx.md.
#
# REANUDABLE: salta las filas ya presentes en el CSV (platform,rows,rep).
#
# Uso:  powershell -ExecutionPolicy Bypass -File scripts\run_benchmarks_rtx.ps1 `
#            [-Reps 5] [-Sizes 100K,1M,10M] [-Out results\benchmark_rtx.csv] [-DryRun]

param(
  [int]    $Reps  = 5,
  [string[]]$Sizes = @("100K","1M","10M"),
  [string] $Out   = "results\benchmark_rtx.csv",
  [string] $DataDir = "C:\km\data",
  [switch] $DryRun
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$bin = "bin\kmeans_rtx.exe"
if (-not (Test-Path $bin)) { throw "falta $bin — compila con scripts\build_rtx.ps1" }
New-Item -ItemType Directory -Force (Split-Path $Out) | Out-Null

# fila ya presente? platform=rtx4070ti, col3=rows, col6=repetition
function Ya-Hecha($rows, $rep) {
  if (-not (Test-Path $Out)) { return $false }
  Select-String -Path $Out -Pattern "^rtx4070ti,cuda,$rows,1,1,$rep," -Quiet
}
function N-DeTam($t) { switch ($t) { "100K"{100000} "1M"{1000000} "10M"{10000000} default{0} } }

Write-Host "=== run_benchmarks_rtx  reps=$Reps sizes=$($Sizes -join ',') out=$Out ==="
foreach ($tam in $Sizes) {
  $rows = N-DeTam $tam
  $data = Join-Path $DataDir "nyc_$tam.bin"
  if (-not (Test-Path $data)) { Write-Warning "falta $data (bajalo del NFS); salto $tam"; continue }
  foreach ($rep in 0..$Reps) {          # rep 0 = corrida en frío (se filtra al graficar)
    if (Ya-Hecha $rows $rep) { Write-Host "skip rtx $tam rep=$rep (ya)"; continue }
    Write-Host ">>> rtx $tam rep=$rep"
    if ($DryRun) { Write-Host "    [dry-run] $bin --data $data --rep $rep"; continue }
    & ".\$bin" --data $data --out $Out --rep $rep --platform rtx4070ti --quiet
    if ($LASTEXITCODE -ne 0) { throw "kmeans_rtx fallo en $tam rep=$rep (exit $LASTEXITCODE)" }
  }
}
$filas = (Get-Content $Out | Measure-Object -Line).Lines - 1
Write-Host "=== fin. Filas en ${Out}: $filas ==="
Write-Host "Copia este CSV al maestro y unelo con el de los clusteres antes de aggregate_power.py."
