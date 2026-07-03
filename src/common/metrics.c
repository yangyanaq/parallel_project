#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L   /* clock_gettime, gmtime_r con -std=c11 */
#endif

#include "metrics.h"

#include <math.h>
#include <stdio.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

double tiempo_ahora(void)
{
#ifdef _WIN32
    static LARGE_INTEGER frec = { 0 };
    LARGE_INTEGER cuenta;
    if (frec.QuadPart == 0)
        QueryPerformanceFrequency(&frec);
    QueryPerformanceCounter(&cuenta);
    return (double)cuenta.QuadPart / (double)frec.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

void timestamp_iso8601(char *buf, size_t tam)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);              /* C11: existe en glibc y MSVC */
    struct tm tmv;
#ifdef _WIN32
    gmtime_s(&tmv, &ts.tv_sec);
#else
    gmtime_r(&ts.tv_sec, &tmv);
#endif
    snprintf(buf, tam, "%04d-%02d-%02dT%02d:%02d:%02d.%03ldZ",
             tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
             tmv.tm_hour, tmv.tm_min, tmv.tm_sec, ts.tv_nsec / 1000000L);
}

/* NaN explícito como "nan": MSVC imprime "nan(ind)", que rompería el
 * parseo del CSV en pandas. */
static void poner_double(FILE *f, double v, const char *fmt)
{
    if (isnan(v)) fputs("nan", f);
    else          fprintf(f, fmt, v);
}

static FILE *abrir_con_cabecera(const char *csv, const char *cabecera)
{
    FILE *f = fopen(csv, "a");
    if (!f) {
        fprintf(stderr, "ERROR: no se pudo abrir %s para append\n", csv);
        return NULL;
    }
    /* con "a" la posición está al final: 0 => archivo recién creado/vacío */
    if (ftell(f) == 0)
        fputs(cabecera, f);
    return f;
}

int metrics_escribir_fila(const char *csv, const InfoCorrida *info,
                          const Metricas *m, const char *timestamp)
{
    FILE *f = abrir_con_cabecera(csv,
        "platform,variant,dataset_rows,num_procs,num_gpus,repetition,"
        "wall_time_s,compute_time_s,comm_time_s,transfer_time_s,"
        "wcss,speedup,efficiency,avg_power_w,energy_wh,throughput_pts_s,"
        "k,d,iterations,seed,timestamp\n");
    if (!f) return -1;

    fprintf(f, "%s,%s,%ld,%d,%d,%d,",
            info->plataforma, info->variante, info->dataset_rows,
            info->num_procs, info->num_gpus, info->repeticion);
    poner_double(f, m->t_total,         "%.6f"); fputc(',', f);
    poner_double(f, m->t_computo,       "%.6f"); fputc(',', f);
    poner_double(f, m->t_comunicacion,  "%.6f"); fputc(',', f);
    poner_double(f, m->t_transferencia, "%.6f"); fputc(',', f);
    poner_double(f, m->wcss,       "%.17g"); fputc(',', f);
    poner_double(f, m->speedup,    "%.6f"); fputc(',', f);
    poner_double(f, m->eficiencia, "%.6f"); fputc(',', f);
    poner_double(f, m->potencia_w, "%.3f"); fputc(',', f);
    poner_double(f, m->energia_wh, "%.6f"); fputc(',', f);
    poner_double(f, m->throughput, "%.1f"); fputc(',', f);
    fprintf(f, "%d,%d,%d,%u,%s\n",
            info->k, info->d, info->iteraciones, info->semilla, timestamp);
    fclose(f);
    return 0;
}

int convergencia_registrar(const char *csv, const InfoCorrida *info,
                           int iter, double wcss)
{
    FILE *f = abrir_con_cabecera(csv,
        "platform,variant,dataset_rows,iteration,wcss\n");
    if (!f) return -1;
    fprintf(f, "%s,%s,%ld,%d,%.17g\n",
            info->plataforma, info->variante, info->dataset_rows, iter, wcss);
    fclose(f);
    return 0;
}
