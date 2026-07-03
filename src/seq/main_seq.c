/*
 * main_seq.c — línea base secuencial (T_1), 1 núcleo, sin MPI ni CUDA.
 * Oficialmente corre en un núcleo del maestro RPi 5; compila también con
 * MSVC para pruebas locales (metrics.c resuelve el reloj, gotcha G2).
 *
 * CLI uniforme de los 4 binarios (PLAN §2.3):
 *   kmeans_seq --data <ruta.bin> [--k 5] [--d 4] [--iters 100] [--seed 42]
 *              [--out results/benchmark.csv] [--conv <csv>] [--rep <i>]
 *              [--platform rpi5] [--quiet]
 *
 * Cronometraje: wall_time_s cubre init de centroides + bucle + WCSS final
 * (la carga del .bin queda FUERA: depende del disco/NFS, no del cómputo).
 * Si se pasa --conv, el WCSS por iteración se cronometra aparte y se
 * excluye de compute_time_s — las corridas de benchmark van sin --conv.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/config.h"
#include "../common/io_dataset.h"
#include "../common/kmeans_core.h"
#include "../common/metrics.h"

static void uso(const char *prog)
{
    fprintf(stderr,
        "Uso: %s --data <ruta.bin> [--k %d] [--d %d] [--iters %d] [--seed %u]\n"
        "        [--out results/benchmark.csv] [--conv <csv>] [--rep <i>]\n"
        "        [--platform rpi5] [--quiet]\n",
        prog, KM_K_DEF, KM_D_DEF, KM_ITERS_DEF, KM_SEED_DEF);
}

int main(int argc, char **argv)
{
    const char *ruta_datos = NULL;
    const char *ruta_out   = "results/benchmark.csv";
    const char *ruta_conv  = NULL;
    const char *plataforma = "rpi5";
    int k = KM_K_DEF, d = KM_D_DEF, iters = KM_ITERS_DEF;
    unsigned int semilla = KM_SEED_DEF;
    int rep = 1, quiet = 0;

    for (int a = 1; a < argc; a++) {
        if      (!strcmp(argv[a], "--data")     && a + 1 < argc) ruta_datos = argv[++a];
        else if (!strcmp(argv[a], "--out")      && a + 1 < argc) ruta_out   = argv[++a];
        else if (!strcmp(argv[a], "--conv")     && a + 1 < argc) ruta_conv  = argv[++a];
        else if (!strcmp(argv[a], "--platform") && a + 1 < argc) plataforma = argv[++a];
        else if (!strcmp(argv[a], "--k")     && a + 1 < argc) k       = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--d")     && a + 1 < argc) d       = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--iters") && a + 1 < argc) iters   = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--seed")  && a + 1 < argc) semilla = (unsigned)strtoul(argv[++a], NULL, 10);
        else if (!strcmp(argv[a], "--rep")   && a + 1 < argc) rep     = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--quiet")) quiet = 1;
        else { uso(argv[0]); return 1; }
    }
    if (!ruta_datos || k <= 0 || d <= 0 || iters <= 0 ||
        k > KM_MAX_K || d > KM_MAX_D) {
        uso(argv[0]);
        return 1;
    }

    /* d != 4 no tiene sentido con el formato binario fijado (§5.1) */
    if (d != 4)
        fprintf(stderr, "AVISO: el .bin esta fijado a d=4; --d %d solo para pruebas\n", d);

    long n = 0;
    double *datos = cargar_binario(ruta_datos, &n);
    if (!datos) return 1;

    double *centroides = (double *)malloc((size_t)k * d * sizeof(double));
    double *sumas      = (double *)malloc((size_t)k * d * sizeof(double));
    long   *conteos    = (long *)malloc((size_t)k * sizeof(long));
    if (!centroides || !sumas || !conteos) {
        fprintf(stderr, "ERROR: sin memoria\n");
        return 1;
    }

    char ts_inicio[32], ts_fin[32];
    timestamp_iso8601(ts_inicio, sizeof ts_inicio);

    double t0 = tiempo_ahora();
    inicializar_centroides(datos, n, k, d, semilla, centroides);

    if (!quiet) {
        fprintf(stderr, "[seq] centroides iniciales (semilla %u):\n", semilla);
        for (int j = 0; j < k; j++) {
            fprintf(stderr, "  mu[%d] =", j);
            for (int dim = 0; dim < d; dim++)
                fprintf(stderr, " %.17g", centroides[j * d + dim]);
            fputc('\n', stderr);
        }
    }

    InfoCorrida info = {
        .plataforma = plataforma, .variante = "seq",
        .dataset_rows = n, .num_procs = 1, .num_gpus = 0,
        .repeticion = rep, .k = k, .d = d, .iteraciones = iters,
        .semilla = semilla
    };

    double t_computo = 0.0, t_conv = 0.0;
    for (int it = 1; it <= iters; it++) {           /* exactamente `iters`, sin parada anticipada */
        double ta = tiempo_ahora();
        asignar_y_acumular(datos, n, n, centroides, k, d, sumas, conteos);
        actualizar_centroides(sumas, conteos, k, d, centroides);
        t_computo += tiempo_ahora() - ta;

        if (ruta_conv) {                            /* curva de convergencia (G9) */
            double tc = tiempo_ahora();
            double w = calcular_wcss(datos, n, n, centroides, k, d);
            convergencia_registrar(ruta_conv, &info, it, w);
            t_conv += tiempo_ahora() - tc;
        }
    }

    double tw = tiempo_ahora();
    double wcss = calcular_wcss(datos, n, n, centroides, k, d);
    double t_fin = tiempo_ahora();
    t_computo += t_fin - tw;

    timestamp_iso8601(ts_fin, sizeof ts_fin);
    double wall = (t_fin - t0) - t_conv;            /* sin el costo de la curva */

    Metricas m = {
        .t_total = wall, .t_computo = t_computo,
        .t_comunicacion = 0.0, .t_transferencia = 0.0,
        .wcss = wcss,
        .speedup = NAN, .eficiencia = NAN,           /* los completa el post-proceso */
        .potencia_w = NAN, .energia_wh = NAN,        /* energía de pared: manual */
        .throughput = ((double)n * iters) / wall
    };
    if (metrics_escribir_fila(ruta_out, &info, &m, ts_inicio) != 0)
        return 1;

    printf("[seq] n=%ld k=%d d=%d iters=%d seed=%u rep=%d\n"
           "      wcss=%.17g\n"
           "      wall_time_s=%.6f compute_time_s=%.6f throughput=%.0f pts/s\n"
           "      inicio=%s fin=%s\n",
           n, k, d, iters, semilla, rep, wcss, wall, t_computo,
           m.throughput, ts_inicio, ts_fin);

    free(centroides); free(sumas); free(conteos); free(datos);
    return 0;
}
