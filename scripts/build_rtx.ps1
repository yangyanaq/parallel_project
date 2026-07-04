# build_rtx.ps1 — compila la variante CUDA en Windows (RTX 4070 Ti, sm_89).
# El Makefile es para Linux; esto es el equivalente para la estación.
#
# Requisitos (ya instalados en .161): CUDA 11.8 (nvcc), VS Build Tools 2022.
# nvcc necesita cl.exe en el PATH -> cargamos vcvars64.bat primero (por eso
# construimos un .bat temporal que hace 'call vcvars64 && nvcc ...', porque
# el entorno de vcvars no persiste entre procesos de PowerShell).
#
# Uso:  powershell -ExecutionPolicy Bypass -File scripts\build_rtx.ps1

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

if (-not (Test-Path bin)) { New-Item -ItemType Directory bin | Out-Null }

# --- localizar vcvars64.bat (Build Tools o Community) ---
$vcCandidatos = @(
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)
$vcvars = $vcCandidatos | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $vcvars) { throw "no encontre vcvars64.bat (VS Build Tools 2022)" }

$nvmlLib = "$env:CUDA_PATH\lib\x64\nvml.lib"

# fuentes: los .c del common (nvcc los trata como C) + los .cu de la RTX
$fuentes = @(
    "src\common\rng.c", "src\common\io_dataset.c",
    "src\common\kmeans_core.c", "src\common\metrics.c",
    "src\rtx\main_cuda.cu", "src\rtx\kmeans_kernel.cu", "src\rtx\energia_nvml.cu"
) -join " "

# CUDA 11.8 NO soporta MSVC 14.40 (VS 17.10+). En esta maquina hay varios
# toolsets; pedimos el 14.29 (VS 2019, soportado por 11.8) a vcvars con
# -vcvars_ver=14.29 y usamos -use-local-env para que nvcc NO re-invoque
# vcvars (evita el "Could not set up the environment"). El toolset se puede
# sobreescribir con la env var KM_VCVARS_VER.
$vcverToolset = if ($env:KM_VCVARS_VER) { $env:KM_VCVARS_VER } else { "14.29" }

# -arch=sm_89 (Ada), -O3, host MSVC /O2, enlaza nvml.lib
$nvccCmd = "nvcc -O3 -arch=sm_89 -std=c++14 -use-local-env -Xcompiler /O2 " +
           "-o bin\kmeans_rtx.exe $fuentes `"$nvmlLib`""

$bat = Join-Path $env:TEMP "km_build_rtx.bat"
@"
@echo off
call "$vcvars" -vcvars_ver=$vcverToolset >nul
if errorlevel 1 exit /b 1
$nvccCmd
"@ | Set-Content -Encoding ascii $bat

Write-Host "=> $nvccCmd"
& cmd /c $bat
$rc = $LASTEXITCODE
Remove-Item $bat -ErrorAction SilentlyContinue
if ($rc -ne 0) { throw "nvcc fallo (exit $rc)" }
Write-Host "OK -> bin\kmeans_rtx.exe"
