/*
 * metrics.h — cronometraje portable (gotcha G2) y escritura de
 * results/benchmark.csv con el esquema exacto de CLAUDE.md §9.
 *
 * Nota MPI: en los binarios de clúster el cronómetro fino es MPI_Wtime;
 * tiempo_ahora() queda para seq/RTX y para timestamps de pared.
 */
#ifndef KM_METRICS_H
#define KM_METRICS_H

#include <stddef.h>
#include "config.h"

/* Identificación de la corrida: las columnas de benchmark.csv que no son
 * métricas. Cada binario la llena con sus valores. */
typedef struct {
    const char  *plataforma;   /* "rpi5" | "jetson" | "rtx4090" | ... */
    const char  *variante;     /* "seq" | "mpi" | "hybrid" | "cuda"   */
    long         dataset_rows;
    int          num_procs;
    int          num_gpus;
    int          repeticion;   /* 0 = corrida en frío (se filtra al graficar) */
    int          k, d, iteraciones;
    unsigned int semilla;
} InfoCorrida;

/* Segundos monotónicos de alta resolución:
 * CLOCK_MONOTONIC (POSIX) / QueryPerformanceCounter (Windows). */
double tiempo_ahora(void);

/* Timestamp UTC ISO-8601 con milisegundos ("2026-07-02T15:04:05.123Z")
 * para cruzar con el medidor de pared (G7). buf >= 32 bytes. */
void timestamp_iso8601(char *buf, size_t tam);

/* Agrega UNA fila a benchmark.csv (crea el archivo con cabecera si no
 * existe). Los double en NaN se escriben como "nan" (speedup/eficiencia
 * y la energía de pared los completa el post-proceso).
 * Devuelve 0 si ok, -1 si no pudo escribir. */
int metrics_escribir_fila(const char *csv, const InfoCorrida *info,
                          const Metricas *m, const char *timestamp);

/* Agrega una fila a wcss_convergence.csv:
 * platform,variant,dataset_rows,iteration,wcss */
int convergencia_registrar(const char *csv, const InfoCorrida *info,
                           int iter, double wcss);

#endif /* KM_METRICS_H */
