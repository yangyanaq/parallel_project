# Runbook — Fase 5: correr la matriz de benchmarks

> Genera `results/benchmark.csv` (165 corridas) y `benchmark_final.csv` con
> speedup/efficiency/energía. Ver PLAN §6, CLAUDE.md §9/§11.

## Matriz (165 corridas)

| Plataforma | Configs | Tamaños | Reps | Corridas |
|---|---|---|---|---|
| seq (RPi, T_1) | 1 | 3 | 5 | 15 |
| RPi MPI | np ∈ {1,2,4,8,16,20} | 3 | 5 | 90 |
| Jetson híbrido | np ∈ {1,2,3} | 3 | 5 | 45 |
| RTX CUDA | 1 | 3 | 5 | 15 |

Cada script corre además la **rep 0 en frío** (se filtra al graficar). Todo es
**reanudable**: si cortas y relanzas, salta las combinaciones ya en el CSV.

## 1. Preparar binarios (maestro .10)

```bash
cd ~/parallel_project && git pull
make seq && make install-rpi          # kmeans_seq (local) + kmeans_rpi (NFS)
# kmeans_jetson: compilar en la .23 (CUDA 10.0) y copiar al NFS (ver runbook_jetson.md)
```

## 2. Clústeres — seq + RPi MPI + Jetson (maestro .10)

```bash
cd ~/parallel_project
bash scripts/run_benchmarks.sh                 # matriz completa, reps=5
# o por partes / reanudando:
bash scripts/run_benchmarks.sh --only seq
bash scripts/run_benchmarks.sh --only rpi  --sizes 100K,1M
bash scripts/run_benchmarks.sh --only jetson --reps 5
bash scripts/run_benchmarks.sh --dry-run       # ver qué correría, sin ejecutar
```

- CSV en el NFS: `/home/cris/kmeans_share/results/benchmark.csv`; al terminar se
  copia a `results/benchmark.csv` del repo (**versionar** — dato central).
- **Jetson escribe distinto:** el rank 0 (en una Jetson) no puede `fopen(append)`
  el CSV del NFS bajo mpirun (caché NFSv3); cada corrida escribe a `/tmp` local y
  el maestro recupera la fila y la anexa. Transparente para el usuario.
- **Energía Jetson:** el script arranca/para `power_log_tegra.sh` (INA3221) en las
  3 Jetson por corrida; los logs van a `/home/cris/kmeans_share/power/`.
- Duración: dominada por 10M en seq/np=1 (decenas de min × reps). Correr de noche;
  reanudable si se corta.

## 3. RTX (por SSH a la estación, o AnyDesk)

```powershell
# en C:\km\parallel_project (repo sin espacios en la ruta)
git pull
powershell -ExecutionPolicy Bypass -File scripts\build_rtx.ps1        # si cambió el codigo
powershell -ExecutionPolicy Bypass -File scripts\run_benchmarks_rtx.ps1 -Reps 5
```

**OJO al invocar por SSH/cmd:** `-Sizes 100K,1M,10M` NO se separa como array al
pasar por cmd (llega como un solo string "100K,1M,10M" y no encuentra el .bin).
Usar la forma de array explícita:

```powershell
powershell -Command "& { .\scripts\run_benchmarks_rtx.ps1 -Reps 5 -Sizes @('100K','1M','10M') }"
```
(Ejecutado localmente con `-File`, el default de `-Sizes` ya es los 3 tamaños.)

- La energía la mide el binario por NVML (columnas ya llenas). Salida:
  `results\benchmark_rtx.csv`. **Cópialo al maestro** y concaténalo (sin cabecera)
  al `benchmark.csv` de los clústeres antes del post-proceso.

## 4. Post-proceso (maestro .10)

```bash
python3 scripts/aggregate_power.py \
    --in results/benchmark.csv \
    --out results/benchmark_final.csv \
    --power-dir /home/cris/kmeans_share/power \
    [--wall-log medidor_pared.csv]
```

Calcula:
- `speedup = T_1/T_p` (T_1 = media de seq rep>0 por tamaño), `efficiency = speedup/num_procs`.
- Energía Jetson: integra INA3221 por rango de timestamps de cada corrida.
- Energía de pared RPi: desde `--wall-log` (CSV manual del medidor: `timestamp,power_w`),
  alineada por timestamp. Sin él, queda NaN (se anota a mano después).
- RTX: respeta la energía NVML del binario.

## Puerta F5

`results/benchmark.csv` con las 165 filas (o su versión reanudada) y
`benchmark_final.csv` con speedup/efficiency/energía completos. Listo para los
gráficos (Fase 6, `plot_results.py`).

## Estado de la validación (2026-07-04)

Orquestación probada end-to-end con reps reducidas: seq, RPi MPI (speedup np=2 ≈
1.99), Jetson np∈{1,2,3} (WCSS ok, energía INA3221 capturada ~1.3–2.3 W/riel), y
`aggregate_power.py` calculando speedup/efficiency. Reanudación verificada. Falta
solo **lanzar la matriz completa** (el usuario, por su duración).
