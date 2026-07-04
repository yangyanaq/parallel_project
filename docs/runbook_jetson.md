# Runbook — variante híbrida MPI+CUDA (clúster Jetson Nano)

> Fase 4. Compilar y lanzar `kmeans_jetson` en las 3 Jetson (`.21/.22/.23`).
> El maestro RPi `.10` orquesta `mpirun` y sirve NFS pero NO computa (G8).

## Inventario relevante (confirmado 2026-07-04)

- Jetson `.21`, `.22`: **CUDA 10.2** (JetPack más nuevo). `.23`: **CUDA 10.0**
  (JetPack más viejo, driver más antiguo). arch `sm_53` en las 3.
- `nvcc` en `/usr/local/cuda/bin` (NO en el PATH → `export PATH=/usr/local/cuda/bin:$PATH`).
- Open MPI **4.1.6** en `/usr/local/bin` (coincide con el maestro; NO usar el de apt 2.1.1).
- gcc 7.5. NFS montado en `/home/cris/kmeans_share`. Hostname: `.21/.22` = `nano-desktop`, `.23` = `nano1`.
- Riel de energía INA3221: `.../iio:device0/in_power0_input` (mW). Ver `scripts/power_log_tegra.sh`.

## GOTCHAS (los cuatro se resolvieron en la sesión F4)

1. **Interfaz de red equivocada** → MPI multinodo se cuelga. Las Jetson tienen
   `docker0`/`sppp` además de `eth0`; MPI elegía la incorrecta. **Siempre** lanzar con:
   `--mca btl_tcp_if_include eth0 --mca oob_tcp_if_include eth0`.
2. **Clúster heterogéneo (CUDA 10.0 vs 10.2)** → binario 10.2 falla en la `.23`
   con "CUDA driver version is insufficient for CUDA runtime version". Solución:
   **compilar en la `.23` (CUDA 10.0)**; ese binario corre en las 3 (10.0 es
   retrocompatible). El binario oficial del NFS se genera en la `.23`.
3. **libcudart dinámica** → "libcudart.so.10.2: cannot open shared object" en la
   `.23`. Solución: link **estático** (`-lcudart_static -ldl -lrt -lpthread`),
   ya fijado en el Makefile. (Con el fix #2 esto es doble seguro.)
4. **Binario en NFS** (como Fase 2): las Jetson no clonan el repo salvo para
   compilar; el binario va al NFS y se lanza desde ahí, misma ruta en las 3.

## Compilar (en la Jetson .23, que tiene el CUDA mínimo 10.0)

```bash
ssh nano@192.168.77.23           # (desde el maestro)
cd ~/parallel_project && git pull
export PATH=/usr/local/cuda/bin:$PATH
make clean && make jetson CUDA_HOME=/usr/local/cuda MPICC=/usr/local/bin/mpicc
cp bin/kmeans_jetson /home/cris/kmeans_share/bin/kmeans_jetson   # o: make install-jetson
```

## Lanzar (desde el maestro .10)

```bash
BIN=/home/cris/kmeans_share/bin/kmeans_jetson
DATA=/home/cris/kmeans_share/data
MCA="--mca btl_tcp_if_include eth0 --mca oob_tcp_if_include eth0"
# barrido de la tesis: -np ∈ {1,2,3} (1 rank por Jetson/GPU; la Nano no tiene MPS)
/usr/local/bin/mpirun --hostfile ~/parallel_project/hosts_jetson -np 3 $MCA \
    $BIN --data $DATA/nyc_1M.bin --out results/benchmark.csv --quiet
```

`hosts_jetson`: 3 líneas `.21/.22/.23 slots=1`. El `--out` conviene a una ruta
escribible por el usuario (`/home/cris/...` o el NFS), no `/tmp` (dueño distinto).

## Energía (INA3221, en paralelo durante la corrida)

En cada Jetson, antes de lanzar el benchmark:
```bash
./scripts/power_log_tegra.sh /home/cris/power_jetsonXX.csv 100 &
```
Se alinea por `timestamp` con el benchmark en el post-proceso (Fase 5). Energía
"de dispositivo" (como la RTX por NVML), distinta de la de pared del clúster.

## Validación (puerta F4) — HECHA 2026-07-04

WCSS idéntico a seq (1e-4) en np ∈ {1,2,3}. Ej. 100K:
np=1 → 156.47795462815358, np=2 → ...367, np=3 → ...367 (dif ~1e-12 por orden
de reducción). La emulación `atomicCAS` de atomicAdd(double) (G3, sm_53 sin
nativo) da el mismo resultado que el `atomicAdd` nativo de la RTX.

Nota de rendimiento (esperada, PLAN §5.3 / G5): la Nano en FP64 es lenta
(1/32 de FP32); np=1 en 100K ≈ 5–6M pts/s, MÁS LENTO que el RPi seq (~19M).
No es un bug: es material de discusión de la tesis (costo/energía vs velocidad).
