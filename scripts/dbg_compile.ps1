# temporal: compila kmeans_kernel.cu y vuelca la salida a C:\kmtmp\err.txt
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
if (-not (Test-Path C:\kmtmp)) { New-Item -ItemType Directory C:\kmtmp | Out-Null }
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$bat = "$env:TEMP\kmc.bat"
@"
@echo off
call "$vcvars" -vcvars_ver=14.29 >nul 2>&1
nvcc -arch=sm_89 -std=c++14 -use-local-env -c src\rtx\kmeans_kernel.cu -o C:\kmtmp\kk.obj
"@ | Set-Content -Encoding ascii $bat
& cmd /c $bat *>&1 | Tee-Object -FilePath C:\kmtmp\err.txt | Out-Null
Write-Output "EXITCODE=$LASTEXITCODE"
Write-Output "=== err.txt ==="
Get-Content C:\kmtmp\err.txt
