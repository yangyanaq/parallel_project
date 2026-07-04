# build_rtx.ps1 — compila la variante CUDA en Windows (RTX 4070 Ti, sm_89).
# El Makefile es para Linux; esto es el equivalente para la estación.
#
# Requisitos (ya instalados en .161): CUDA 11.8 (nvcc), VS Build Tools 2022,
# nvml.lib (viene con el toolkit). Salida: bin\kmeans_rtx.exe
#
# GOTCHAS resueltos aquí (ver docs/runbook_rtx.md):
#  1) nvcc falla ("Could not set up the environment" / exit 2 sin log) si %TEMP%
#     tiene un espacio (el perfil es "C:\Users\Windows 11\..."). -> forzamos
#     TMP/TEMP a C:\kmtmp (sin espacios) antes de nvcc.
#  2) CUDA 11.8 no soporta MSVC 14.40 (VS 17.10+) -> -allow-unsupported-compiler.
#  3) -use-local-env para que nvcc NO re-invoque vcvars (usa el que cargamos).
#
# Uso:  powershell -ExecutionPolicy Bypass -File scripts\build_rtx.ps1

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
if (-not (Test-Path bin))     { New-Item -ItemType Directory bin     | Out-Null }
if (-not (Test-Path C:\kmtmp)){ New-Item -ItemType Directory C:\kmtmp | Out-Null }

# --- localizar vcvars64.bat (Build Tools o Community) ---
$vcCandidatos = @(
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)
$vcvars = $vcCandidatos | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $vcvars) { throw "no encontre vcvars64.bat (VS Build Tools 2022)" }

$nvmlLib = "$env:CUDA_PATH\lib\x64\nvml.lib"
$fuentes = @(
    "src\common\rng.c", "src\common\io_dataset.c",
    "src\common\kmeans_core.c", "src\common\metrics.c",
    "src\rtx\main_cuda.cu", "src\rtx\kmeans_kernel.cu", "src\rtx\energia_nvml.cu"
) -join " "

$nvccCmd = "nvcc -O3 -arch=sm_89 -std=c++14 -use-local-env " +
           "-allow-unsupported-compiler -Xcompiler /O2 " +
           "-o bin\kmeans_rtx.exe $fuentes `"$nvmlLib`""

$bat = Join-Path "C:\kmtmp" "km_build.bat"
@"
@echo off
call "$vcvars" >nul 2>&1
set TMP=C:\kmtmp
set TEMP=C:\kmtmp
cd /d "$repo"
$nvccCmd
"@ | Set-Content -Encoding ascii $bat

Write-Host "=> $nvccCmd"
& cmd /c $bat
$rc = $LASTEXITCODE
if ($rc -ne 0) { throw "nvcc fallo (exit $rc)" }
Write-Host "OK -> bin\kmeans_rtx.exe"
