# build_rtx.ps1 — compila la variante CUDA en Windows (RTX 4070 Ti, sm_89).
# El Makefile es para Linux; esto es el equivalente para la estación.
#
# Requisitos (ya instalados en .161): CUDA 11.8 (nvcc), VS Build Tools 2022
# (nvcc encuentra cl automáticamente si corres desde un shell con vcvars, o
# nvcc lo localiza vía el registro). Salida: bin\kmeans_rtx.exe
#
# Uso:  powershell -ExecutionPolicy Bypass -File scripts\build_rtx.ps1

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

if (-not (Test-Path bin)) { New-Item -ItemType Directory bin | Out-Null }

$common = @(
    "src\common\rng.c",
    "src\common\io_dataset.c",
    "src\common\kmeans_core.c",
    "src\common\metrics.c"
)
$cuda = @(
    "src\rtx\main_cuda.cu",
    "src\rtx\kmeans_kernel.cu",
    "src\rtx\energia_nvml.cu"
)

# -arch=sm_89 (Ada). -O3. Host MSVC en modo C++ para los .cu; los .c los
# trata nvcc como C. Enlaza nvml.lib (viene con el toolkit).
$nvmlLib = "`"$env:CUDA_PATH\lib\x64\nvml.lib`""

$args = @(
    "-O3", "-arch=sm_89", "-std=c++14",
    "-Xcompiler", "/O2",
    "-o", "bin\kmeans_rtx.exe"
) + $common + $cuda + @($nvmlLib)

Write-Host "nvcc $($args -join ' ')"
& nvcc @args
if ($LASTEXITCODE -ne 0) { throw "nvcc fallo (exit $LASTEXITCODE)" }
Write-Host "OK -> bin\kmeans_rtx.exe"
