# Análisis extra para la redacción (figuras 10 y 11 + cuantificaciones Cap 7/9)

> **Para el chat de redacción LaTeX.** Este documento es la FUENTE de dos figuras
> nuevas y varios números verificados; no es texto final. Toda cifra sale de
> `results/benchmark_final.csv` (repeticiones 1-5, la 0 se descarta) y de
> `results/wcss_convergence.csv`, igual que el resto de los caps 6-7.
> Recordatorios de estilo vigentes: sin raya larga en la prosa, sin fechas.
> Los PNG ya están copiados a `latex/Nueva carpeta/Figures/`; se regeneran con
> `python scripts/plot_results.py` (funciones `g10_*` y `g11_*`).

## 1. Figura 10 — Speedup del cómputo puro vs total (RPi, 10M)

**Archivo:** `Figures/10_speedup_computo_vs_total.png` (también .pdf en results/figs)

**Qué muestra.** Dos curvas de speedup sobre el mismo eje para el clúster RPi
sobre 10M: la calculada con el tiempo total (`wall_time_s`) y la calculada solo
con el tiempo de cómputo (`compute_time_s`), más la recta ideal y una flecha
que anota la brecha como "costo de sincronización".

**Números exactos (medias de reps 1-5):**

| p | S total | S cómputo puro | eficiencia del cómputo |
|---|---------|----------------|------------------------|
| 1 | 0.99x | 0.99x | 99.2% |
| 2 | 1.97x | 1.99x | 99.4% |
| 4 | 3.86x | 3.95x | 98.9% |
| 8 | 2.41x | 7.91x | 98.9% |
| 16 | 1.98x | 15.91x | 99.5% |
| 20 | 1.90x | **19.88x** | **99.4%** |

**La lectura (idea central, redactar sin raya larga):** el algoritmo paraleliza
de forma casi perfecta; con 20 procesos el cómputo puro alcanza 19.88x de un
ideal de 20 (eficiencia 99.4%). Lo que colapsa el resultado total a 1.90x es
íntegramente la sincronización. La brecha vertical entre ambas curvas es,
literalmente, el costo de las colectivas. Esto refuerza la conclusión de la
sección "frontera del nodo" del Cap 7: el reparto de datos y el balance de
carga funcionan; el problema es la barrera de red.

**Destino sugerido:** Cap 7, sección de la frontera del nodo (complementa la
tabla `tab:frontera_nodo`; la figura dice en una imagen lo que la tabla dice
en columnas). Alternativa: junto a las figuras de speedup del Cap 6, pero su
tono es analítico, encaja mejor en el 7.

## 2. Figura 11 — Firma de la desincronización (RPi multinodo)

**Archivo:** `Figures/11_firma_desincronizacion.png`

**Qué muestra.** Comunicación por iteración (ms) contra el tamaño del dataset,
una línea por cada p multinodo (8, 16, 20), en un solo tono de azul con
marcadores distintos. Anotación interna: el mensaje es constante (180 B por
iteración), así que si el costo fuera solo latencia de red las líneas serían
planas.

**Números exactos (ms de comunicación por iteración, medias):**

| p | 100K | 1M | 10M |
|---|------|-----|-----|
| 8 | 6.4 | 18.9 | 141.7 |
| 16 | 10.0 | 29.0 | 217.4 |
| 20 | 11.4 | 30.4 | 233.3 |

**La lectura:** las líneas crecen con el tamaño del dataset aunque el volumen
transmitido no cambia. Eso descarta el ancho de banda y la latencia pura como
explicación dominante y señala la desincronización: a más cómputo por
iteración, más se desalinean los procesos, y la colectiva espera al último.
Este es el respaldo visual del párrafo del Cap 7 que empieza "Vale la pena
precisar de qué está hecho ese costo".

**Números de contexto para ese argumento (ya calculados y verificados):**
- Volumen total por corrida: 200 colectivas x 180 B = 17.6 KB.
- Transmitir 17.6 KB a velocidad Gigabit real (~115 MB/s) toma 0.16 ms.
- La comunicación medida con p=20 sobre 10M es 23.33 s, es decir 116.7 ms por
  colectiva: cinco órdenes de magnitud sobre el costo de transmisión.
- Referencia de latencias: TCP sobre Ethernet de consumo, cientos de
  microsegundos por salto; interconexiones HPC (InfiniBand), 1-2 us.

**Destino sugerido:** Cap 7, misma sección, inmediatamente después del párrafo
citado. Si se considera que dos figuras nuevas son demasiadas, esta es la
prescindible (la 10 es la principal); su contenido puede quedar como tabla.

## 3. Tabla de costo unitario (ns por punto por iteración, 10M, cómputo puro)

Normaliza todas las plataformas a una sola escala intuitiva. Valores medidos:

| Plataforma (config) | Cómputo (s) | ns/punto/iteración |
|---|---|---|
| Secuencial, 1 núcleo A76 | 49.03 | 49.0 |
| RPi 5 MPI, p=4 | 12.40 | 12.4 |
| RPi 5 MPI, p=20 | 2.47 | 2.5 |
| Jetson, 1 nodo (GPU) | 93.49 | 93.5 |
| Jetson, 3 nodos | 31.08 | 31.1 |
| RTX 4070 Ti | 0.46 | 0.46 |

**Lecturas:** (a) la GPU embebida cuesta 93.5 ns por punto donde un núcleo de
CPU cuesta 49: la penalización de la doble precisión en Maxwell hecha número
unitario; (b) la RTX procesa cada punto 107 veces más barato que el núcleo
A76 y 203 veces más barato que la GPU del Nano; (c) el cómputo del RPi con 20
procesos baja a 2.5 ns/punto, mejor incluso que su marca con p=4, lo que
confirma otra vez que el cómputo escala y lo que se paga es la sincronización.

**Destino sugerido:** Cap 7, en la comparación global o en la sección Jetson.

## 4. Cuantificaciones para el Cap 9 (trabajo futuro)

Solo los tres puntos siguientes. **Exclusión deliberada:** la fusión de los
dos Allreduce en uno y la reducción jerárquica en dos niveles quedan FUERA de
este documento por decisión del autor; no añadir cuantificaciones de esas dos
ideas desde aquí (si el Cap 9 ya las menciona cualitativamente, se dejan como
están).

### 4a. Parada anticipada (dato medido, no estimado)

La curva de convergencia registrada (`wcss_convergence.csv`, dataset 100K)
muestra que el WCSS entra en el 1% de su valor final en la **iteración 15**,
en las tres plataformas por igual (la trayectoria es numéricamente idéntica).
Las 100 iteraciones fijas fueron una decisión deliberada del diseño para que
todas las plataformas midieran el mismo trabajo; en un despliegue real, un
criterio de parada por convergencia recortaría el trabajo unas **6.7 veces**
en todas las plataformas por igual. No cambia el ranking de la comparación
(el recorte es proporcional), pero sí los tiempos y la energía absolutos, y
para el clúster RPi reduce en la misma proporción el número de barreras de
sincronización, que es su costo dominante.

### 4b. Precisión simple en la Jetson (estimación acotada, no medida)

El desglose medido muestra que la variante Jetson es compute-bound: el kernel
concentra el 99.1% del tiempo en 1M y el 99.4% en 10M con un nodo (columnas
compute/comm/transfer del CSV). Sobre esa base, pasar a FP32:
- La mejora NO sería 32x (el ratio nominal FP64/FP32 de Maxwell) porque el
  kernel está limitado por memoria, no por unidades de cómputo: el trabajo
  por punto es leer 4 coordenadas y comparar 5 distancias.
- La cota realista la da el ancho de banda: los datos pesan la mitad en FP32,
  de modo que la ganancia esperable es **cercana a 2x**, aplicada a casi todo
  el tiempo de ejecución (por ser compute-bound al 99%).
- Beneficio adicional no cuantificado: sm_53 tiene atomicAdd nativo en FP32,
  lo que elimina la emulación por compare-and-swap.
- Advertencia metodológica que el Cap 9 ya recoge: cambia la naturaleza de la
  comparación (deja de resolverse el mismo problema numérico).

### 4c. Red de baja latencia, no de más ancho de banda (cálculo con datos medidos)

Con los números de la sección 2 de este documento: el volumen transmitido por
corrida es 17.6 KB (0.16 ms a velocidad Gigabit) y la comunicación medida con
p=20 sobre 10M es 23.33 s. La conclusión cuantitativa: multiplicar el ancho
de banda (2.5G o 10G Ethernet) no ataca el problema, porque el costo no está
en mover bytes sino en la latencia por colectiva (116.7 ms efectivos) y en la
espera al proceso más lento. Lo que movería la frontera del nodo es una
interconexión de baja latencia (clase InfiniBand, 1-2 us por salto) o reducir
el número de sincronizaciones. Esto refina la mención de "red 2.5/10G" del
Cap 9 si sigue presente: el upgrade de ancho de banda, por sí solo, no cambia
el resultado.

## 5. Procedencia y reproducibilidad

- Fig 10 y 11: `scripts/plot_results.py` (g10, g11), corren con
  `python scripts/plot_results.py --pdf` sobre `results/benchmark_final.csv`.
- Tablas 1-3: medias de reps 1-5 del mismo CSV; el cálculo de ns/punto es
  compute_time_s / (n x 100 iteraciones).
- Parada anticipada: primera iteración con |WCSS - final|/final < 1% en
  `results/wcss_convergence.csv` (da 15 en las tres plataformas).
- Cálculo de red: 200 colectivas x 180 B; Gigabit real 115 MB/s.
