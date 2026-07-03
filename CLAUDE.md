# HANDOFF — Implementación de K-Means paralelo (tesis)

> **Uso:** guarda este archivo como `CLAUDE.md` en la raíz del repositorio. Claude Code lo lee como contexto. Es la **única fuente de verdad** del proyecto: no hace falta pasarle los capítulos de la tesis. Si algo aquí contradice una idea previa, manda esto.

---

## 0. Qué se construye (en una frase)

Tres implementaciones **desde cero, en C11**, del algoritmo **K-Means (Lloyd)** que resuelven el *mismo* problema con la *misma* lógica, para comparar rendimiento, energía y costo entre tres plataformas: un clúster Raspberry Pi 5 (MPI puro), un clúster Jetson Nano (MPI + CUDA) y una estación RTX 4090 (CUDA puro). El resultado central no es el clustering, sino un **`benchmark.csv`** con métricas comparables y los **gráficos** que lo acompañan.

---

## 1. Reglas duras (no negociables)

1. **Lenguaje:** host y núcleo algorítmico en **C11** (`-std=c11`). Los kernels de GPU en CUDA C (`.cu`). **Prohibido** usar bibliotecas de alto nivel de ML o de clustering (nada de scikit-learn, cuML, Thrust para el núcleo, etc.). Todo el K-Means se escribe a mano.
2. **Hardware de referencia:** Raspberry Pi **5** (Cortex-A76 @ 2.4 GHz), **no** Pi 4. Nunca menciones ni asumas Pi 4 ni MPI4py/Python.
3. **Conteo de nodos fijo:** clúster RPi = **5 nodos / 20 núcleos**; clúster Jetson = **3 nodos / 12 núcleos + 3 GPU** (los ranks MPI corren solo en las Jetsons; el maestro RPi `.10` orquesta el lanzamiento y sirve NFS).
4. **Parámetros del algoritmo fijos:** `k = 5`, `d = 4`, **100 iteraciones fijas** (sin parada anticipada), inicialización aleatoria con **semilla fija**.
5. **Misma lógica en las 3 variantes.** Toda diferencia de tiempo debe venir del hardware/modelo de paralelismo, no de la implementación. El núcleo (`kmeans_core`) es compartido.
6. **Verificación:** el WCSS de las tres variantes debe coincidir dentro de **tolerancia `1e-4`** contra la línea base secuencial. Si no coincide, hay un bug.
7. **Nada de datos inventados.** El código mide; no rellena.

---

## 2. Hardware objetivo

| Plataforma | Nodos | CPU | GPU | Memoria | Red | Variante |
|---|---|---|---|---|---|---|
| **Clúster RPi 5** | 5 (1 maestro + 4 trabajadores) | 20× ARM Cortex-A76 @2.4 GHz (4/nodo) | — | 4 GB/nodo (20 GB agregados) | Gigabit Ethernet + switch, NFS en maestro | **MPI puro** |
| **Clúster Jetson Nano** | 3 | 12× ARM Cortex-A57 (4/nodo) | 3× Maxwell, 128 CUDA cores c/u (`sm_53`) | 4 GB/nodo | Gigabit Ethernet + switch, NFS servido por el maestro RPi | **MPI + CUDA** |
| **Estación RTX 4090** | 1 (x86-64) | host CPU | Ada Lovelace, 16 384 CUDA cores (`sm_89`), 24 GB GDDR6X | — | sin red (sin MPI) | **CUDA puro** |

La línea base secuencial (`T_1`) se corre en **un solo núcleo del nodo maestro de la RPi 5**.

### 2.1 Infraestructura real (red y acceso)

- Subred única `192.168.77.0/24`; ambos clústeres comparten switch y el NFS del maestro RPi.
- **Maestro RPi (punto de entrada y lanzador de AMBOS clústeres):** `192.168.77.10`.
- **Workers RPi:** `192.168.77.11`–`192.168.77.14` (4 workers; el maestro también computa → 5 nodos / 20 núcleos).
- **Workers Jetson:** `192.168.77.21`–`192.168.77.23` (3 Jetson Nano; el maestro RPi NO computa en los jobs Jetson, solo lanza `mpirun` y sirve NFS).
- **Estación RTX 4090:** máquina Windows **dentro de la red del clúster** (`192.168.77.x`), pero se opera solo por **AnyDesk** (sin SSH desde la PC de trabajo); CUDA/MSVC/git por instalar. Compila nativo Windows (`nvcc` + MSVC). Al estar en la LAN, puede bajar los `.bin` directo del NFS/maestro `.10` (scp desde la propia RTX), sin transferir por AnyDesk. Credenciales en `pass.txt` (fuera del repo — el repo es público).
- **PC de trabajo (esta):** Windows, edita el código, corre `preprocess.py`, accede por SSH a `192.168.77.10`.
- Flujo de código: repo **GitHub**; los maestros hacen `git pull` y compilan; la RTX hace `git pull` vía AnyDesk.
- `pass.txt` (credenciales) vive en la raíz: **jamás** debe entrar al repo (va en `.gitignore`).

---

## 3. Toolchain y compilación

- **MPI:** Open MPI 4.1.6 (`mpicc`) en RPi y Jetson.
- **CUDA:** Toolkit **10.2** en Jetson (JetPack 4.6, arch `sm_53`); Toolkit **12.x** en RTX (arch `sm_89`).
- Host/núcleo compilan como **C11**; los `.cu` los compila `nvcc` (CUDA C++).

Banderas de referencia (ajústalas si hace falta):

```bash
# RPi 5 (MPI puro)
mpicc -O3 -std=c11 -march=native -funroll-loops \
      src/common/*.c src/rpi/main_mpi.c -o bin/kmeans_rpi -lm

# Jetson Nano (MPI + CUDA): compilar kernels con nvcc, host con mpicc, enlazar con mpicc
nvcc -O3 -arch=sm_53 -std=c++11 -c src/jetson/kmeans_kernel.cu -o build/kmeans_kernel.o
mpicc -O3 -std=c11 -c src/common/*.c src/jetson/main_hybrid.c
mpicc *.o build/kmeans_kernel.o -o bin/kmeans_jetson \
      -L/usr/local/cuda/lib64 -lcudart -lm

# RTX 4090 (CUDA puro): main puede ser .cu
nvcc -O3 -arch=sm_89 -std=c++11 \
     src/common/io_dataset.c src/common/metrics.c \
     src/rtx/main_cuda.cu src/rtx/kmeans_kernel.cu -o bin/kmeans_rtx -lm
```

Provee un **Makefile** con targets `rpi`, `jetson`, `rtx`, `seq` (secuencial), `all`, `clean`. Detecta plataforma o usa targets explícitos.

Lanzamiento en clúster:

```bash
mpirun --hostfile hosts_rpi -np 20 ./bin/kmeans_rpi --data data/nyc_1M.bin --n 1000000
```

`hosts_rpi`: un nodo por línea con `slots=4` (5 líneas, maestro incluido). `hosts_jetson`: un nodo por línea con `slots=1` (3 líneas, solo Jetsons — 1 rank por GPU; ver §11).

---

## 4. El algoritmo (especificación exacta)

K-Means de Lloyd sobre `n` puntos en `R^d`, `d = 4` (las 4 coordenadas), `k = 5`.

- **Distancia:** euclidiana **al cuadrado** (no calcular `sqrt`; solo se usa para comparar).
- **Objetivo (WCSS):** `sum_i min_j || x_i - mu_j ||^2`.
- **Inicialización:** elegir `k` puntos al azar del dataset con `srand(semilla)` fija (determinista y replicable en todas las plataformas). No usar K-Means++.
- **Bucle:** exactamente 100 iteraciones. En cada una: (a) asignar cada punto a su centroide más cercano; (b) acumular sumas y conteos por clúster; (c) recalcular `mu_j = suma_j / conteo_j`. Si un clúster queda vacío, mantener su centroide anterior (documentar esta política).
- **Sin parada anticipada.** Aunque converja antes, corre las 100.
- **Nota de punto flotante:** el orden de sumatoria difiere entre secuencial, MPI y GPU; por eso la equivalencia se valida con tolerancia `1e-4`, no con igualdad exacta. Usar `double` en todo.

---

## 5. Los datos

- **Origen:** NYC Yellow Taxi Trip Data, **enero 2015** (tiene lat/long reales; posterior a jun-2016 solo trae `LocationID`, inservible).
- **Columnas usadas (d = 4), en este orden lógico:** `pickup_latitude`, `pickup_longitude`, `dropoff_latitude`, `dropoff_longitude`. Ojo: en el CSV crudo el orden es `pickup_longitude, pickup_latitude, ... dropoff_longitude, dropoff_latitude` (longitud antes que latitud).
- **Tres tamaños:** 100K, 1M, 10M filas (muestreo aleatorio con semilla fija).
- **Limpieza:** descartar filas con NaN y coordenadas fuera de la caja de NYC (`lon ∈ [-74.27, -73.68]`, `lat ∈ [40.49, 40.92]`). **No normalizar ni estandarizar.**
- **Formatos:** CSV (legible) y **binario `double`** (carga sin parseo). Preprocesar una vez con un script Python (`scripts/preprocess.py`) que hace muestreo, limpieza y exporta ambos formatos. El formato binario está **fijado** en la sección 5.1 (SoA `float64` little-endian, sin cabecera). No lo cambies.
- **Ubicación:** NFS del maestro (clústeres) y disco local (RTX). Carpeta `data/` en `.gitignore`.

> Estadísticos reales ya calculados sobre 100K (para sanity-check del cargador): medias ≈ pickup (40.751, −73.975), dropoff (40.752, −73.974); std ≤ 0.036°. Si tu cargador da algo muy distinto, está mal leyendo columnas.

---

### 5.1 Preprocesamiento y formato binario (`scripts/preprocess.py`)

Los datasets **no** se generan en los clústeres. `scripts/preprocess.py` corre **una vez en la workstation**: lee los CSV mensuales crudos (~1.8 GB c/u), extrae solo las 4 coordenadas, limpia (NaN, ceros, fuera de bbox), baraja con semilla fija y produce tres muestras **anidadas** (100K ⊂ 1M ⊂ 10M). Enero 2015 por sí solo tiene ~12.7M filas, suficiente para el 10M.

```bash
python3 scripts/preprocess.py --inputs yellow_tripdata_2015-01.csv --outdir data
```

Salida en `data/`: `nyc_{100K,1M,10M}.bin` (+ `.meta.json`; + `.csv` para 100K y 1M). **A los clústeres se copian solo los `.bin`** (~355 MB los tres). El `.csv` de 100K alimenta el análisis exploratorio del Cap. 5.

**Contrato del formato binario (entre `preprocess.py` y el cargador en C — NO cambiar):**

- `float64` little-endian, **sin cabecera**.
- Layout **SoA**, `d = 4`, orden: `[pickup_lat × n][pickup_lon × n][dropoff_lat × n][dropoff_lon × n]`.
- `n = filesize_bytes / (4 * 8)`. Acceso en C: `datos[dim*n + i]` (dim 0..3, i 0..n-1).

Cargador recíproco que `io_dataset` debe implementar:

```c
double *cargar_binario(const char *ruta, long *n_out) {
    FILE *f = fopen(ruta, "rb");
    fseek(f, 0, SEEK_END); long bytes = ftell(f); fseek(f, 0, SEEK_SET);
    long n = bytes / (4 * sizeof(double));            // 4 = d
    double *datos = malloc((size_t)n * 4 * sizeof(double));
    fread(datos, sizeof(double), (size_t)n * 4, f);   // SoA
    fclose(f);
    *n_out = n;
    return datos;                                     // datos[dim*n + i]
}
```

## 6. Estructuras de datos (compartidas, en `config.h`/`common`)

```c
typedef struct {            // parámetros del experimento
    int   k;                // = 5
    int   d;                // = 4
    int   max_iter;         // = 100
    unsigned int semilla;   // fija
    const char *ruta_datos;
    long  n;                // filas totales
} ConfigKMeans;

typedef struct {            // datos (Structure of Arrays)
    long    n;              // filas totales
    int     d;              // = 4
    double *datos;          // tamaño n*d, layout SoA
    long    n_local;        // filas de este proceso tras Scatter
    double *datos_local;    // n_local*d
} Dataset;

typedef struct {
    int    k, d;
    double coords[/*k*d*/ 20];   // k*d = 20
} Centroides;

typedef struct {                 // parciales por proceso/hilo
    double sumas[/*k*d*/ 20];
    long   conteos[/*k*/ 5];
} AcumuladorLocal;

typedef struct {                 // lo que se exporta a benchmark.csv
    double t_total, t_computo, t_comunicacion, t_transferencia;
    double wcss;
    double speedup, eficiencia;
    double potencia_w, energia_wh;
    double throughput;           // puntos/segundo
} Metricas;
```

---

## 7. Arquitectura del software

**Módulos comunes** (misma lógica en las 3 variantes):

- `io_dataset` — leer CSV/binario, particionar, (des)serializar.
- `kmeans_core` — asignación, actualización de centroides, cálculo de WCSS. **Toda la matemática vive aquí.**
- `metrics` — cronometraje (usar `MPI_Wtime` en clústeres y `clock_gettime(CLOCK_MONOTONIC)` / eventos CUDA en GPU), speedup, eficiencia, energía, throughput, escritura del CSV.

**Tres ejecutables:**

- `kmeans_rpi` — MPI puro. Todo en CPU.
- `kmeans_jetson` — MPI entre nodos + kernel CUDA para la asignación dentro de cada nodo.
- `kmeans_rtx` — CUDA puro, un solo proceso, sin MPI.

Más un **cuarto binario `kmeans_seq`** (secuencial, 1 núcleo) para la línea base `T_1`.

---

## 8. Paralelización por variante

### 8.1 MPI puro (RPi) — grano grueso, por particionado de datos
Protocolo por corrida:
1. `MPI_Scatter` (1 vez): reparte `n/p` filas contiguas desde el maestro.
2. `MPI_Bcast` (1 vez): centroides iniciales a todos.
3. Bucle ×100: asignación local → `MPI_Allreduce(sumas, MPI_SUM)` + `MPI_Allreduce(conteos, MPI_SUM)` → actualización replicada en todos.
4. `MPI_Reduce` (1 vez): WCSS local → WCSS global en el maestro.

Carga del `Allreduce` por iteración: `k*d` doubles (160 B) + `k` conteos (~20–40 B) ≈ **constante**, independiente de `n`. El costo relevante es la **latencia** (`O(log p)` en árbol), no el volumen. Cronometra por separado cómputo vs comunicación (envuelve los `Allreduce`).

### 8.2 MPI + CUDA (Jetson) — híbrido
Igual que 8.1 entre nodos, pero la **asignación** de cada nodo corre en su GPU: `cudaMemcpy` del bloque local al device, kernel (un hilo por punto) que calcula el `argmin` sobre los `k` centroides y acumula sumas por clúster con `atomicAdd`; el parcial vuelve a la CPU para entrar al `MPI_Allreduce`. Centroides en `__constant__` (solo 20 doubles).

### 8.3 CUDA puro (RTX) — grano fino, sin MPI
Una `cudaMemcpy` H2D antes del bucle. Las 100 iteraciones corren en GPU: kernel de asignación (un hilo por punto), reducción de sumas por clúster (usar `atomicAdd` o reducción en memoria compartida por bloque + combinación), actualización de centroides. Centroides en memoria constante. El dataset de 10M×4 doubles = 320 MB entra de sobra en 24 GB, así que **no fragmentar**.

---

## 9. Métricas y salida — CRÍTICO para Cap. 6 y 7

Cada corrida agrega **una fila** a `results/benchmark.csv` con este esquema exacto:

```
platform,variant,dataset_rows,num_procs,num_gpus,repetition,
wall_time_s,compute_time_s,comm_time_s,transfer_time_s,
wcss,speedup,efficiency,avg_power_w,energy_wh,throughput_pts_s,
k,d,iterations,seed,timestamp
```

Definiciones:
- `speedup = T_1 / T_p` (T_1 = secuencial en 1 núcleo RPi, mismo dataset).
- `efficiency = speedup / num_procs` (en GPU, reportar vs T_1; documentar el "p" usado).
- `throughput_pts_s = (dataset_rows * iterations) / wall_time_s`.
- `comm_time_s`: solo para MPI (suma de tiempos en colectivas). `transfer_time_s`: solo GPU (H2D/D2H).
- `energy_wh`, `avg_power_w`: ver §10.

Además, un **`results/wcss_convergence.csv`** opcional pero recomendado (`platform,variant,dataset_rows,iteration,wcss`) para graficar la curva de convergencia y demostrar equivalencia entre plataformas.

**Repeticiones:** correr cada configuración **≥ 5 veces** (idealmente 10) y guardar cada repetición; los gráficos usan media y desviación (Cap. 7 pide estadística y validación). Descartar la primera corrida en frío si hace falta (documentarlo).

---

## 10. Medición de energía

- **RTX 4090:** muestrear potencia con **NVML** (`nvmlDeviceGetPowerUsage`) en un hilo cada ~100 ms durante la corrida; integrar potencia×tiempo → Wh. Registrar `avg_power_w` y `energy_wh`.
- **Jetson Nano:** potencia por `tegrastats` o rieles INA3221 (`/sys/bus/i2c/.../in_power*`); mismo esquema de muestreo/integración.
- **Clústeres (energía total de pared):** medidor físico en la entrada de CA (o enchufe inteligente). El código **no** mide esto; deja las columnas `avg_power_w`/`energy_wh` que se completan desde el log del medidor, alineando por `timestamp`. Emite timestamps de inicio/fin precisos para poder cruzarlos.

Documenta claramente qué energía es "de dispositivo" (GPU) vs "de pared" (clúster) para que la comparación del Cap. 7 sea honesta.

---

## 11. Matriz de experimentos

- **3 plataformas × 3 tamaños (100K/1M/10M) × ≥5 repeticiones.**
- **Escalabilidad (para curvas de speedup/eficiencia):** en los clústeres, barrer número de procesos: RPi con `-np` = 1, 2, 4, 8, 16, 20; Jetson con `-np` = 1, 2, 3 (**1 rank por nodo/GPU** — la Nano no soporta MPS y compartir la GPU entre ranks serializa contextos; el paralelismo intra-nodo lo da el kernel CUDA, no más ranks). La RTX no escala por procesos (reporta su tiempo absoluto).
- **Línea base:** `kmeans_seq` en 1 núcleo RPi, por cada tamaño.
- Un `scripts/run_benchmarks.sh` que recorra la matriz y vaya agregando filas al CSV.

---

## 12. Gráficos a producir (`scripts/plot_results.py`, matplotlib)

Leen `benchmark.csv` y generan PNG (200 dpi) para los Capítulos 6 y 7:

1. **Speedup vs nº de procesos** (por tamaño), con la línea ideal `y=x`.
2. **Eficiencia paralela vs nº de procesos**.
3. **Tiempo de ejecución vs tamaño del dataset** (100K/1M/10M, escala log-log), las tres plataformas.
4. **Descomposición cómputo vs comunicación vs transferencia** (barras apiladas) — muestra el overhead de MPI.
5. **Consumo de energía (Wh)** por plataforma y tamaño.
6. **Rendimiento por vatio** (throughput / potencia).
7. **Rendimiento por dólar** (throughput / costo del hardware) — la métrica costo-eficiencia central de la tesis.
8. **Convergencia del WCSS** por iteración (verifica que las 3 plataformas convergen al mismo valor).
9. **Throughput (puntos/s) vs tamaño**.

Cada gráfico con media y barras de error (±desv. est.) cuando haya repeticiones. Exportar también PDF si se pide.

---

## 13. Estructura de repositorio sugerida

```
kmeans-tesis/
├── CLAUDE.md                 # este documento
├── Makefile
├── hosts_rpi  hosts_jetson
├── src/
│   ├── common/  config.h  io_dataset.{c,h}  kmeans_core.{c,h}  metrics.{c,h}
│   ├── seq/     main_seq.c
│   ├── rpi/     main_mpi.c
│   ├── jetson/  main_hybrid.c  kmeans_kernel.cu
│   └── rtx/     main_cuda.cu   kmeans_kernel.cu
├── scripts/  preprocess.py  run_benchmarks.sh  plot_results.py
├── data/                     # .gitignore
└── results/  benchmark.csv  wcss_convergence.csv  figs/
```

---

## 14. Pseudocódigo de referencia (variante MPI; las otras derivan de esta)

```
En cada proceso r de p:
  MPI_Init()
  si r == 0: leer X desde NFS; inicializar mu con semilla fija
  MPI_Scatter(X -> X_local)          # n/p filas
  MPI_Bcast(mu)
  para t = 1..100:
     suma_local=0; conteo_local=0
     para cada x en X_local:          # asignación (paralela)
        j* = argmin_j ||x - mu_j||^2
        suma_local[j*] += x; conteo_local[j*] += 1
     MPI_Allreduce(suma_local  -> suma_global,  MPI_SUM)   # sincronización
     MPI_Allreduce(conteo_local-> conteo_global,MPI_SUM)
     para j: mu_j = (conteo_global[j]>0) ? suma_global[j]/conteo_global[j] : mu_j
  wcss_local = sum ||x - mu_asig(x)||^2 sobre X_local
  MPI_Reduce(wcss_local -> WCSS, MPI_SUM, raiz=0)
  MPI_Finalize()
```

---

## 15. Criterios de aceptación (checklist)

- [ ] Las 4 binarios (`seq`, `rpi`, `jetson`, `rtx`) compilan con sus toolchains.
- [ ] Sobre el mismo dataset y semilla, los 4 producen el mismo WCSS dentro de `1e-4`.
- [ ] `benchmark.csv` se llena con el esquema de §9; una fila por corrida/repetición.
- [ ] Cómputo, comunicación y transferencia se cronometran por separado.
- [ ] La curva de speedup muestra la degradación esperada del clúster RPi en 100K (overhead de comunicación domina en particiones chicas).
- [ ] `plot_results.py` genera los 9 gráficos de §12 desde el CSV.
- [ ] Reproducible: misma semilla ⇒ mismo resultado; corridas repetidas ⇒ media + desv.
- [ ] Cero bibliotecas de ML/clustering de alto nivel en el núcleo.

---

## 16. Por qué (decisiones, por si preguntas)

- **100 iteraciones fijas** y no convergencia: para que el tiempo mida velocidad de cómputo y no trayectorias de convergencia distintas entre plataformas.
- **Distancia al cuadrado:** evita `sqrt`, irrelevante para el `argmin`.
- **Semilla fija / init aleatoria (no K-Means++):** el objeto de estudio es el rendimiento, no la calidad del agrupamiento; se necesita determinismo.
- **SoA:** accesos coalescentes en GPU y vectorización en CPU.
- **Tolerancia `1e-4`:** el orden de reducción en punto flotante difiere entre CPU secuencial, MPI y GPU; la igualdad exacta no es alcanzable ni esperable.
