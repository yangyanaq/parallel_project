/*
 * main_cuda.cu — variante CUDA pura (RTX 4070 Ti, sm_89). CLAUDE.md §8.3, PLAN §4.
 *
 * Un solo proceso, sin MPI. El dataset sube UNA vez (H2D); las 100 iteraciones
 * corren en device (centroides/sumas/conteos nunca bajan por PCIe). La init de
 * centroides se hace en HOST con el mismo rng que seq (inicializar_centroides),
 * así el WCSS coincide con la línea base dentro de 1e-4 (G1, G9).
 *
 * CLI uniforme (PLAN §2.3), idéntica a seq salvo variante/plataforma/gpus.
 * Tiempos: eventos CUDA. wall = tiempo_ahora() (metrics.c). transfer_time_s =
 * H2D del dataset + D2H del WCSS. Energía: hilo NVML (energia_nvml, G4).
 *
 * Se compila con nvcc (host = MSVC). io_dataset/metrics son C11 del common.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
#include "../common/config.h"
#include "../common/io_dataset.h"
#include "../common/metrics.h"
}
#include "kmeans_kernel.cuh"
#include "energia_nvml.h"

static void uso(const char *prog)
{
    fprintf(stderr,
        "Uso: %s --data <ruta.bin> [--k %d] [--d %d] [--iters %d] [--seed %u]\n"
        "        [--out results/benchmark.csv] [--conv <csv>] [--rep <i>]\n"
        "        [--platform rtx4070ti] [--quiet]\n",
        prog, KM_K_DEF, KM_D_DEF, KM_ITERS_DEF, KM_SEED_DEF);
}

int main(int argc, char **argv)
{
    const char *ruta_datos = NULL;
    const char *ruta_out   = "results/benchmark.csv";
    const char *ruta_conv  = NULL;
    const char *plataforma = "rtx4070ti";
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

    long n = 0;
    double *datos = cargar_binario(ruta_datos, &n);   /* SoA host, stride n */
    if (!datos) return 1;

    double *centroides = (double *)malloc((size_t)k * d * sizeof(double));
    if (!centroides) { fprintf(stderr, "ERROR: sin memoria\n"); return 1; }

    char ts_inicio[32], ts_fin[32];
    timestamp_iso8601(ts_inicio, sizeof ts_inicio);

    /* init de centroides en HOST (mismo rng que seq) */
    inicializar_centroides(datos, n, k, d, semilla, centroides);
    if (!quiet) {
        fprintf(stderr, "[cuda] centroides iniciales (semilla %u):\n", semilla);
        for (int j = 0; j < k; j++) {
            fprintf(stderr, "  mu[%d] =", j);
            for (int dim = 0; dim < d; dim++)
                fprintf(stderr, " %.17g", centroides[j * d + dim]);
            fputc('\n', stderr);
        }
    }

    InfoCorrida info;
    info.plataforma = plataforma; info.variante = "cuda";
    info.dataset_rows = n; info.num_procs = 1; info.num_gpus = 1;
    info.repeticion = rep; info.k = k; info.d = d;
    info.iteraciones = iters; info.semilla = semilla;

    /* energía: arranca el muestreo NVML antes del cómputo */
    EnergiaNVML *ener = energia_iniciar(100);

    float ms_h2d = 0.0f, ms_kernel = 0.0f, ms_d2h = 0.0f;
    double wall0 = tiempo_ahora();

    KmCuda *cu = km_cuda_crear(datos, n, k, d, centroides, &ms_h2d);
    if (!cu) { fprintf(stderr, "ERROR: fallo init CUDA\n"); return 1; }

    for (int it = 1; it <= iters; it++) {
        km_cuda_iterar(cu, &ms_kernel);
        if (ruta_conv) {                              /* curva de convergencia (G9) */
            double w = km_cuda_wcss(cu, &ms_d2h);
            convergencia_registrar(ruta_conv, &info, it, w);
        }
    }

    double wcss = km_cuda_wcss(cu, &ms_d2h);          /* WCSS final (centroides finales) */
    double wall = tiempo_ahora() - wall0;
    timestamp_iso8601(ts_fin, sizeof ts_fin);

    double avg_w = NAN, wh = NAN;
    energia_detener(ener, &avg_w, &wh);

    km_cuda_leer_centroides(cu, centroides, &ms_d2h);
    km_cuda_destruir(cu);

    double transfer_s = (ms_h2d + ms_d2h) / 1000.0;   /* H2D dataset + D2H wcss/centroides */
    double compute_s  = ms_kernel / 1000.0;

    Metricas m;
    m.t_total = wall; m.t_computo = compute_s;
    m.t_comunicacion = 0.0; m.t_transferencia = transfer_s;
    m.wcss = wcss; m.speedup = NAN; m.eficiencia = NAN;
    m.potencia_w = avg_w; m.energia_wh = wh;
    m.throughput = ((double)n * iters) / wall;
    if (metrics_escribir_fila(ruta_out, &info, &m, ts_inicio) != 0)
        return 1;

    printf("[cuda] n=%ld k=%d d=%d iters=%d seed=%u rep=%d gpu=%s\n"
           "       wcss=%.17g\n"
           "       wall_time_s=%.6f compute_time_s=%.6f transfer_time_s=%.6f\n"
           "       avg_power_w=%.2f energy_wh=%.6f throughput=%.0f pts/s\n"
           "       inicio=%s fin=%s\n",
           n, k, d, iters, semilla, rep, plataforma, wcss,
           wall, compute_s, transfer_s, avg_w, wh, m.throughput,
           ts_inicio, ts_fin);

    free(centroides); free(datos);
    return 0;
}
