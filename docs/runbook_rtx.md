# Runbook — variante CUDA en la RTX 4070 Ti (Windows 11)

> Fase 3. Compilar y correr `kmeans_rtx` en la estación (`192.168.77.161`).
> Acceso: `ssh -J cris@10.144.101.22 "windows 11"@192.168.77.161` (túnel vía
> el maestro; ver `runbook_habilitar_ssh_rtx.md`).

## Estado de la máquina (Fase 0 de la RTX, hecho 2026-07-03)

- GPU **RTX 4070 Ti**, 12 GB, `sm_89`. CUDA **11.8** (`nvcc`), `nvml.lib` OK.
- MSVC **VS Build Tools 2022** (varios toolsets: 14.16 / 14.29 / 14.40).
- git 2.42.

## GOTCHA importante: el repo NO puede vivir en una ruta con espacios

El perfil del usuario es `C:\Users\Windows 11` (con espacio). **nvcc falla de
formas opacas** (exit 2 sin log, o "Could not set up the environment") cuando su
`%TEMP%` o el repo están en rutas con espacios. Solución adoptada:

- **El repo de trabajo de la RTX vive en `C:\km\parallel_project`** (sin espacios),
  clonado aparte del `C:\Users\Windows 11\parallel_project`.
- `build_rtx.ps1` fuerza `TMP`/`TEMP` a `C:\kmtmp` antes de llamar a nvcc.

```powershell
# primera vez
if (-not (Test-Path C:\km)) { mkdir C:\km }
cd C:\km
git clone https://github.com/yangyanaq/parallel_project.git
cd parallel_project
```

## Compilar

```powershell
cd C:\km\parallel_project
git pull
powershell -ExecutionPolicy Bypass -File scripts\build_rtx.ps1
# -> bin\kmeans_rtx.exe
```

El script resuelve tres cosas (detalle en el propio `.ps1`): `TEMP` sin espacios,
`-allow-unsupported-compiler` (CUDA 11.8 no “soporta” MSVC 14.40 pero funciona),
y `-use-local-env` para que nvcc no re-invoque vcvars.

## Datos (una vez, por scp desde el NFS del maestro, misma LAN)

```powershell
if (-not (Test-Path C:\km\data)) { mkdir C:\km\data }
scp cris@192.168.77.10:/home/cris/kmeans_share/data/nyc_100K.bin C:\km\data\
scp cris@192.168.77.10:/home/cris/kmeans_share/data/nyc_1M.bin   C:\km\data\
scp cris@192.168.77.10:/home/cris/kmeans_share/data/nyc_10M.bin  C:\km\data\
```

## Correr

```powershell
cd C:\km\parallel_project
bin\kmeans_rtx.exe --data C:\km\data\nyc_1M.bin --out C:\kmtmp\bench_rtx.csv --quiet
```

CLI uniforme (igual que seq/mpi): `--data --k --d --iters --seed --out --conv
--rep --platform --quiet`. La energía se mide sola (hilo NVML, columnas
`avg_power_w`/`energy_wh`); no requiere medidor externo (a diferencia de los
clústeres). Sin `--quiet` imprime los centroides iniciales y por rank.

## Validación (puerta F3) — HECHA 2026-07-03

WCSS idéntico a seq/numpy dentro de 1e-4 (de hecho ~1e-12) en los 3 tamaños:

| n | WCSS CUDA | wall_s | throughput | avg_W |
|---|---|---|---|---|
| 100K | 156.4779546281535 | 0.077 | 130 M pts/s | 6.8 |
| 1M | 1571.3412310829622 | 0.123 | 813 M pts/s | 7.0 |
| 10M | 15741.323367872177 | 0.571 | 1 753 M pts/s | 40.2 |

Referencia (seq RPi 10M = 49.2 s): la RTX es **~86× más rápida** en 10M.
`nvidia-smi` durante la corrida confirma la potencia reportada por NVML.
