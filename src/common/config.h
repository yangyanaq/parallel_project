/*
 * config.h — Estructuras y parámetros compartidos por las 4 variantes
 * (seq, rpi/MPI, jetson/híbrido, rtx/CUDA). Ver CLAUDE.md §6.
 *
 * Compila como C11 estricto en gcc (RPi), mpicc, gcc-7 (Jetson) y MSVC (RTX).
 */
#ifndef KM_CONFIG_H
#define KM_CONFIG_H

/* Parámetros fijos del experimento (CLAUDE.md §1.4). La CLI puede
 * sobreescribirlos para pruebas, pero los oficiales son estos. */
#define KM_K_DEF      5
#define KM_D_DEF      4
#define KM_ITERS_DEF  100
#define KM_SEED_DEF   42u

/* Cotas para los arreglos estáticos de abajo (k*d = 20 en el experimento;
 * se deja margen por si la CLI pide algo mayor en pruebas locales). */
#define KM_MAX_K 16
#define KM_MAX_D 8

typedef struct {            /* parámetros del experimento */
    int   k;                /* = 5 */
    int   d;                /* = 4 */
    int   max_iter;         /* = 100 (sin parada anticipada) */
    unsigned int semilla;   /* fija */
    const char *ruta_datos;
    long  n;                /* filas totales */
} ConfigKMeans;

typedef struct {            /* datos (Structure of Arrays) */
    long    n;              /* filas totales */
    int     d;              /* = 4 */
    double *datos;          /* tamaño n*d, layout SoA: datos[dim*n + i] */
    long    n_local;        /* filas de este proceso tras Scatter */
    double *datos_local;    /* n_local*d, mismo layout con stride n_local */
} Dataset;

typedef struct {                 /* centroides en AoS: coords[j*d + dim] */
    int    k, d;
    double coords[KM_MAX_K * KM_MAX_D];
} Centroides;

typedef struct {                 /* parciales por proceso/hilo */
    double sumas[KM_MAX_K * KM_MAX_D];
    long   conteos[KM_MAX_K];
} AcumuladorLocal;

typedef struct {                 /* lo que se exporta a benchmark.csv (§9) */
    double t_total, t_computo, t_comunicacion, t_transferencia;
    double wcss;
    double speedup, eficiencia;  /* NaN en caliente; los completa el post-proceso */
    double potencia_w, energia_wh;
    double throughput;           /* puntos*iteraciones / segundo */
} Metricas;

#endif /* KM_CONFIG_H */
