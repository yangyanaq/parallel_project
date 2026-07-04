# PLAN MAESTRO — Implementación K-Means paralelo (tesis)

> Complementa a `CLAUDE.md` (la especificación). Este documento es el **cómo y en qué orden**.
> Cada fase tiene una **puerta de salida** (criterio verificable). No se avanza a la siguiente fase sin pasar la puerta.

---

## 0. Decisiones ya tomadas (2026-07-02)

| Decisión | Valor |
|---|---|
| Clúster Jetson real | **3 nodos** (`.21/.22/.23`), 12 núcleos, 3 GPUs. El maestro RPi `.10` lanza `mpirun` y sirve NFS, no computa en jobs Jetson |
| Ranks en Jetson | **1 rank MPI por nodo/GPU** → barrido `-np` = 1, 2, 3 (la Nano no tiene MPS; el paralelismo intra-nodo lo da el kernel) |
| Distribución de código | Repo **GitHub**: se edita en esta PC Windows, los maestros hacen `git pull`, la RTX hace `git pull` vía AnyDesk |
| Estación RTX | **Windows nativo** (nvcc + MSVC). Todo por instalar (Fase 0) |
| Datos | `preprocess.py` corre en **esta PC** (aquí están los CSV crudos); los `.bin` se copian por `scp` al NFS del `.10` y por AnyDesk a la RTX |
| Fuente de datos | Solo `yellow_tripdata_2015-01.csv` (~12.7M filas). Los CSV 2016 quedan como respaldo, no se usan |

---

## 1. Fase 0 — Infraestructura

### 1.1 Repo Git (en esta PC)

- `git init` en `c:\Users\Ren\Desktop\tesis`, repo GitHub privado, un solo repo para código + LaTeX.
- `.gitignore`: `data/` (CSV crudos y .bin), `pass.txt`, `bin/`, `build/`, `*.o`, artefactos LaTeX (`*.aux`, `*.log`, `*.synctex.gz`, …), `results/figs/` se versiona pero los PNG pesados pueden regenerarse.
- **`results/benchmark.csv` SÍ se versiona** — es el dato central de la tesis.
- Mover `data/preprocess.py` → `scripts/preprocess.py` (estructura de §13 del CLAUDE.md).

### 1.2 Inventario de los clústeres (por SSH desde esta PC)

Comandos a correr en `.10` (y vía éste, en workers) para fijar hechos antes de escribir una línea de C:

```bash
ssh <user>@192.168.77.10  'uname -a; lscpu | head; mpicc --version; mpirun --version; df -h; cat /etc/exports'
# en cada Jetson (.21-.23):
ssh <user>@192.168.77.2X 'nvcc --version; cat /etc/nv_tegra_release; mount | grep nfs; ls /sys/bus/i2c/drivers/ina3221x/ 2>/dev/null || ls /sys/class/hwmon/'
```

Qué queda fijado: usuario SSH, ruta del NFS montado en todos los nodos (ej. `/mnt/nfs` o `/home/pi/shared`), versiones reales de Open MPI y CUDA, ruta de los rieles INA3221 para energía Jetson, y que los relojes estén sincronizados (NTP/chrony) — crítico para alinear timestamps con el medidor de pared.

### 1.3 Setup estación RTX 4070 Ti — HECHO 2026-07-03 (acceso por SSH, no AnyDesk)

GPU real: **RTX 4070 Ti** (12 GB, `sm_89`), no 4090. Windows 11, `192.168.77.161`.
Acceso: **OpenSSH Server habilitado** → `ssh -J cris@10.144.101.22 <usuario>@192.168.77.161`
(túnel vía el maestro; ver `docs/runbook_habilitar_ssh_rtx.md`).

1. ✅ Visual Studio **Build Tools 2022** (workload C++, `cl` vía vcvars).
2. ✅ CUDA Toolkit **11.8** (`nvcc` V11.8.89, `nvml.lib` presente). 11.8 = mínimo para sm_89.
3. ✅ Git para Windows 2.42.
4. Pendiente: **clonar el repo** en la RTX (`git clone` en el HOME).

### 1.4 Datos

1. En esta PC: `python scripts/preprocess.py --inputs data/yellow_tripdata_2015-01.csv --outdir data` → `nyc_100K.bin`, `nyc_1M.bin`, `nyc_10M.bin` (+ meta + csv chicos).
2. Sanity-check automático contra los estadísticos del CLAUDE.md §5 (medias ≈ (40.75, −73.97), std ≤ 0.036°).
3. `scp data/*.bin data/*.meta.json <user>@192.168.77.10:<ruta_NFS>/data/`.
4. A la RTX: está en la LAN del clúster → desde la propia RTX (vía AnyDesk) bajar del maestro: `scp <user>@192.168.77.10:<ruta_NFS>/data/*.bin C:\datos\` (Windows 10/11 trae cliente OpenSSH). Fallback: transferencia de archivos de AnyDesk (~355 MB).

**Puerta de salida F0:** repo en GitHub clonado en `.10`; `.bin` visibles desde un worker RPi y una Jetson vía NFS; `nvcc` funcionando en la RTX; inventario documentado en `docs/inventario.md`.

---

## 2. Fase 1 — Código común + secuencial (la base de todo)

### 2.1 Archivos

```
src/common/config.h        # ConfigKMeans, Dataset, Centroides, AcumuladorLocal, Metricas (§6)
src/common/rng.{c,h}       # PRNG PROPIO (ver gotcha G1) — NO rand()
src/common/io_dataset.{c,h}
src/common/kmeans_core.{c,h}
src/common/metrics.{c,h}
src/seq/main_seq.c
Makefile                   # targets: seq, rpi, jetson, all, clean (Linux)
scripts/validate_wcss.py   # referencia numpy pura (solo verificación, no es "el código")
```

### 2.2 Firmas del núcleo (contrato entre las 4 variantes)

```c
// rng.h — determinista e idéntico en glibc/MSVC/nvcc
void     rng_init(uint64_t semilla);
uint64_t rng_next(void);                       // xorshift64* o LCG de 64 bits
long     rng_indice(long n);                   // índice uniforme en [0, n)

// io_dataset.h
double  *cargar_binario(const char *ruta, long *n_out);      // contrato §5.1
void     inicializar_centroides(const double *datos, long n, int k, int d,
                                unsigned int semilla, double *centroides);
                                               // k índices aleatorios DISTINTOS; AoS k*d

// kmeans_core.h — TODA la matemática; compila igual en gcc/mpicc/MSVC
void  asignar_y_acumular(const double *datos, long n_local, long n_stride,
                         const double *centroides, int k, int d,
                         double *sumas, long *conteos);       // un paso (a)+(b)
void  actualizar_centroides(const double *sumas, const long *conteos,
                            int k, int d, double *centroides); // paso (c); vacío ⇒ conserva
double calcular_wcss(const double *datos, long n_local, long n_stride,
                     const double *centroides, int k, int d);

// metrics.h
double tiempo_ahora(void);                     // CLOCK_MONOTONIC / QueryPerformanceCounter (G2)
void   metrics_escribir_fila(const char *csv, const Metricas *m, /* + identificación */);
void   convergencia_registrar(const char *csv, int iter, double wcss);
```

Nota: `n_stride` es el `n` del layout SoA (`datos[dim*n_stride + i]`) — permite que la misma función sirva para el dataset completo (seq/RTX) y para el bloque local (MPI) sin copiar.

### 2.3 CLI uniforme de los 4 binarios

```
kmeans_X --data <ruta.bin> [--k 5] [--d 4] [--iters 100] [--seed 42]
         [--out results/benchmark.csv] [--conv results/wcss_convergence.csv]
         [--rep <num_repeticion>] [--quiet]
```

Salida por stdout: una línea legible con `wall_time_s`, `wcss` y los timestamps de inicio/fin (ISO-8601 con milisegundos, para cruzar con el medidor de pared).

### 2.4 Validación

- `kmeans_seq` sobre `nyc_100K.bin` en esta PC (compilado con MSVC o en el RPi) vs `scripts/validate_wcss.py` (Lloyd en numpy, misma semilla y misma init): WCSS coincide < `1e-4`.
- Correr también sobre 1M en el RPi maestro para tener el primer `T_1` real.

**Puerta de salida F1:** WCSS seq == numpy (1e-4) en 100K y 1M; `benchmark.csv` recibe filas bien formadas; `wcss_convergence.csv` muestra curva monótona no creciente.

---

## 3. Fase 2 — MPI puro (RPi)

### 3.1 Archivos

```
src/rpi/main_mpi.c
hosts_rpi                  # 5 líneas: 192.168.77.10..14 slots=4
```

### 3.2 Diseño (§8.1 del CLAUDE.md, protocolo exacto)

- Rank 0 carga el `.bin` desde NFS e inicializa centroides (misma `rng`).
- `MPI_Scatterv` (no Scatter: `n` no siempre divide entre `p`) de **cada una de las 4 dimensiones** por separado, o re-empaquetar a bloques por rank; decisión: **scatterv por dimensión** (4 llamadas, 1 vez) mantiene SoA local sin copias extra.
- Bucle ×100: `asignar_y_acumular` local → `MPI_Allreduce` de `sumas[20]` y `conteos[5]` → `actualizar_centroides` replicado.
- Cronómetro: `MPI_Wtime()`; `comm_time_s` = suma de tiempo dentro de colectivas (Allreduce ×200 + Scatterv + Bcast + Reduce final); `compute_time_s` = resto del bucle.
- WCSS final: local + `MPI_Reduce`.

### 3.3 Validación

1. En esta PC no hay MPI: compilar y probar directo en `.10` (`git pull && make rpi`).
2. `mpirun -np 4` solo en el maestro → WCSS == seq (1e-4).
3. `mpirun --hostfile hosts_rpi -np 20` → mismo WCSS, en 100K/1M/10M.
4. Confirmar el comportamiento esperado: en 100K con np=20 el speedup se degrada (overhead de latencia) — si no aparece, revisar cronometraje, no "arreglarlo".

**Puerta de salida F2:** WCSS idéntico (1e-4) para np ∈ {1,2,4,8,16,20} en los 3 tamaños; `comm_time_s` + `compute_time_s` ≈ `wall_time_s` (±5%).

---

## 4. Fase 3 — CUDA puro (RTX) *(antes que Jetson: el kernel de Jetson deriva de éste)*

### 4.1 Archivos

```
src/rtx/main_cuda.cu
src/rtx/kmeans_kernel.cu   # + kmeans_kernel.cuh
scripts/build_rtx.ps1      # nvcc en Windows (el Makefile es para Linux)
```

### 4.2 Diseño (§8.3)

- Una `cudaMemcpy` H2D del dataset antes del bucle (10M×4 doubles = 320 MB, sobra en 12 GB). D2H solo de `sumas/conteos` (o mantener actualización en device y bajar solo al final — decisión: **actualización de centroides en device** con un kernel trivial de k×d hilos; así el bucle no toca PCIe).
- Kernel de asignación: 1 hilo/punto, centroides en `__constant__` (20 doubles), acumulación con **reducción en memoria compartida por bloque + `atomicAdd` global por bloque** (sm_89 tiene atomicAdd(double) nativo; la reducción por bloque reduce contención de 10M atómicas a ~40K).
- La MISMA matemática de `kmeans_core` se replica en el kernel; para no divergir, el `.cuh` documenta la correspondencia línea a línea y la validación WCSS es el juez.
- Tiempos: eventos CUDA para kernel y transferencias (`transfer_time_s` = H2D + D2H); `wall_time_s` con `tiempo_ahora()`.
- Energía: hilo de muestreo NVML (`nvmlDeviceGetPowerUsage` cada 100 ms, hilo Win32) → integra a Wh (G4).

### 4.3 Validación

- WCSS == seq (1e-4) en los 3 tamaños (la RTX puede correr también `kmeans_seq` compilado con MSVC para comparar in situ, pero el T_1 oficial es el del RPi).
- `nvidia-smi` durante la corrida confirma potencia ≈ la reportada por NVML.

**Puerta de salida F3:** WCSS ok en 3 tamaños; `energy_wh > 0` coherente; fila completa en `benchmark.csv`.

---

## 5. Fase 4 — MPI + CUDA (Jetson)

### 5.1 Archivos

```
src/jetson/main_hybrid.c       # host C11, compilado con mpicc
src/jetson/kmeans_kernel.cu    # port del kernel RTX a sm_53
hosts_jetson                   # 3 líneas: 192.168.77.21..23 slots=1
```

### 5.2 Diseño (§8.2)

- Estructura MPI idéntica a `main_mpi.c` (mismo Scatterv/Bcast/Allreduce/Reduce); la única diferencia: `asignar_y_acumular` se reemplaza por `lanzar_asignacion_gpu(...)` (declarada `extern "C"` en el `.cu`).
- Port sm_53: **no hay `atomicAdd(double)` nativo** → emulación con `atomicCAS` de 64 bits (G3). CUDA 10.2 ⇒ `-std=c++11`, sin features modernas.
- `cudaMemcpy` H2D del bloque local **una vez** antes del bucle (1M/3 ≈ 11 MB, cabe de sobra); por iteración solo suben centroides (160 B, o `__constant__` con `cudaMemcpyToSymbol`) y bajan `sumas+conteos` (200 B). `transfer_time_s` los acumula.
- Cuidado con los 4 GB compartidos CPU/GPU de la Nano (memoria física unificada): en 10M el bloque local es ~107 MB ×2 (host+device) — ok.
- Energía: script `scripts/power_log_tegra.sh` que muestrea los rieles INA3221 (`in_power0_input`) cada 100 ms a un log con timestamps; corre en cada Jetson durante el benchmark; la integración/alineación la hace `plot_results.py`/`aggregate_power.py` por timestamps (el binario no mide — igual que el medidor de pared).

### 5.3 Validación

- Compilar en una Jetson (`git pull` en NFS o local): `make jetson`.
- np=1 en una sola Jetson → WCSS == seq (1e-4). Luego np=3.
- Comparar `t_computo` np=1 GPU vs CPU (curiosidad: la Nano en double es lenta, FP64 = 1/32 de FP32 — resultado esperado y digno de discusión en la tesis, no un bug).

**Puerta de salida F4:** WCSS ok para np ∈ {1,2,3} en los 3 tamaños; logs de potencia generados y alineables por timestamp.

---

## 6. Fase 5 — Orquestación de benchmarks

### 6.1 Archivos

```
scripts/run_benchmarks.sh      # corre en .10 (clústeres) — recorre la matriz
scripts/run_benchmarks_rtx.ps1 # corre en la RTX
scripts/aggregate_power.py     # integra logs de potencia (tegra/pared) → completa columnas del CSV
```

### 6.2 Matriz de experimentos (total de corridas)

| Plataforma | Configs | Tamaños | Reps | Corridas |
|---|---|---|---|---|
| seq (RPi, T_1) | 1 | 3 | 5 | 15 |
| RPi MPI | np ∈ {1,2,4,8,16,20} = 6 | 3 | 5 | 90 |
| Jetson híbrido | np ∈ {1,2,3} = 3 | 3 | 5 | 45 |
| RTX CUDA | 1 | 3 | 5 | 15 |
| **Total** | | | | **165** |

- Cada corrida = 1 fila en `benchmark.csv` (append; el script pasa `--rep i`).
- Descartar/marcar la corrida 0 en frío (columna `repetition = 0`, se filtra al graficar) — política documentada.
- `speedup`/`efficiency` NO los calcula el binario en caliente (necesitan el T_1 promedio): los completa `aggregate_power.py`/`plot_results.py` al post-procesar. El binario deja esas columnas en `NaN` y el post-proceso genera `benchmark_final.csv`. *(Alternativa: pasarle `--t1 <segundos>` al binario; decidir en implementación — el CSV de salida final debe cumplir §9 igual.)*
- Estimación de duración: dominada por 10M en seq/np=1 (decenas de minutos por corrida × reps) — el script debe poder reanudar (saltar combinaciones ya presentes en el CSV).

**Puerta de salida F5:** `benchmark.csv` con las 165 filas (o su versión reanudada), energía completada para RTX (NVML), Jetson (INA3221) y clústeres (pared, manual desde el log del medidor).

---

## 7. Fase 6 — Gráficos y cierre

- `scripts/plot_results.py` (matplotlib, PNG 200 dpi → `results/figs/`): los 9 gráficos de §12, con media ± desv. est. filtrando `repetition == 0`.
- Verificación final del checklist §15 completo.
- `docs/` con: inventario de hardware, política de clúster vacío, política de corrida en frío, qué energía es "de dispositivo" vs "de pared".

---

## 8. Gotchas técnicos críticos (leer antes de codificar)

- **G1 — `rand()` NO es portable.** glibc, MSVC y la libc de la Jetson dan secuencias distintas para la misma semilla ⇒ centroides iniciales distintos ⇒ WCSS distinto entre plataformas y la validación 1e-4 muere. Solución: PRNG propio en `rng.c` (xorshift64* — 5 líneas, determinista, sin UB). El CLAUDE.md dice "srand(semilla)"; se implementa el espíritu (semilla fija reproducible), no la letra.
- **G2 — Reloj portable.** MSVC no tiene `clock_gettime` ⇒ `metrics.c` con `#ifdef _WIN32` → `QueryPerformanceCounter`, else `clock_gettime(CLOCK_MONOTONIC)`. En MPI, usar `MPI_Wtime`.
- **G3 — `atomicAdd(double)` no existe en sm_53** (requiere sm_60+). El kernel Jetson usa la emulación clásica con `atomicCAS(unsigned long long)`. El kernel RTX (sm_89) usa el nativo. Mismo resultado numérico salvo orden de suma (cubierto por la tolerancia 1e-4).
- **G4 — NVML en Windows:** headers y `nvml.lib` vienen con el CUDA Toolkit; el hilo de muestreo usa Win32 `CreateThread` (MSVC no trae `<threads.h>` de C11).
- **G5 — FP64 en las GPUs de consumo es lento** (RTX 4070 Ti ≈ 1/64 de FP32, igual que toda la gama Ada de consumo; Nano ≈ 1/32). El spec exige `double` — se respeta. El resultado seguirá siendo rápido vs los clústeres por el ancho de banda de memoria; documentar en la tesis, no "optimizar" cambiando a float.
- **G6 — `Scatterv`, no `Scatter`:** `n` (p.ej. 100 000) no es divisible entre p=16; los conteos/desplazamientos se calculan por rank. El layout SoA obliga a 4 Scatterv (uno por dimensión) o a empaquetar: preferir 4 Scatterv.
- **G7 — Relojes sincronizados** en todos los nodos y la PC (NTP) — sin esto la energía de pared no se puede alinear por timestamp.
- **G8 — Línea `mpirun` desde `.10` hacia Jetsons:** el binario `kmeans_jetson` NO corre en el RPi; `mpirun --hostfile hosts_jetson` con `.10` fuera del hostfile lanza los ranks solo en las Jetsons (Open MPI lanza remoto vía ssh sin necesidad de que el nodo local participe).
- **G9 — WCSS del bucle vs final:** el WCSS de validación se calcula con los centroides FINALES (tras la iteración 100), recomputando asignaciones; el de convergencia (opcional §9) se registra por iteración. Ambos deben usar la misma fórmula de `kmeans_core`.

---

## 9. Orden de sesiones de trabajo sugerido (para las sesiones de código)

1. **S1:** Fase 0 completa (repo, .gitignore, discovery SSH, preprocess + scp de datos). Sin C todavía.
2. **S2:** Fase 1 (common + seq + validate_wcss.py + Makefile). Validar en RPi por SSH.
3. **S3:** Fase 2 (MPI RPi) + validación en clúster real.
4. **S4:** Fase 3 (CUDA RTX) — el build/tests en la RTX los ejecuta el usuario por AnyDesk siguiendo un runbook que la sesión deja escrito (`docs/runbook_rtx.md`).
5. **S5:** Fase 4 (Jetson híbrido) + validación por SSH.
6. **S6:** Fase 5 (orquestación + energía) y lanzamiento de la matriz.
7. **S7:** Fase 6 (plots, checklist §15, limpieza).

---

## 10. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| 10M no cabe en RAM del RPi maestro al cargar (320 MB datos + copias) | Rank 0 carga y hace Scatterv por dimensión sin duplicar; 4 GB alcanzan, pero vigilar; peor caso: `mmap` del `.bin` |
| WCSS no coincide entre plataformas | Primero verificar centroides INICIALES idénticos (imprimirlos con `%.17g` y comparar); el 90% de las veces el bug es la init, no la reducción |
| Toolchain Jetson vieja (CUDA 10.2 / gcc 7) rechaza algo | Núcleo en C11 conservador, kernels `-std=c++11`; nada de C17/C++14+ |
| Corridas de 10M muy largas para 5 reps | `run_benchmarks.sh` reanudable + correr matrices por la noche |
| Medidor de pared sin log digital | Anotar manualmente W promedio por corrida usando los timestamps de stdout; formato de entrada de `aggregate_power.py` lo contempla |
| AnyDesk sin automatización | Runbook paso a paso (`docs/runbook_rtx.md`); los scripts `.ps1` minimizan los pasos manuales |
