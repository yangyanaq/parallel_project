/*
 * main_hybrid.c — variante MPI + CUDA (clúster Jetson Nano). CLAUDE.md §8.2, PLAN §5.
 *
 * Estructura MPI IDÉNTICA a src/rpi/main_mpi.c (mismo Scatterv por dimensión,
 * Bcast, Allreduce, Reduce), con UNA diferencia: la asignación+acumulación
 * local de cada rank corre en su GPU (km_gpu_asignar, declarada extern "C" en
 * el .cu), no en CPU. La actualización de centroides sigue replicada en CPU
 * tras el Allreduce (regla §1.5: misma lógica, distinto ejecutor).
 *
 * 1 rank MPI por nodo/GPU (la Nano no tiene MPS): barrido -np = 1,2,3.
 * Cronometraje MPI_Wtime: comm_time = colectivas; transfer_time = H2D/D2H de
 * la GPU (bloque inicial + centroides/parciales por iter); compute = resto.
 */
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/config.h"
#include "../common/io_dataset.h"
#include "../common/kmeans_core.h"      /* actualizar_centroides (CPU, replicado) */
#include "../common/metrics.h"
#include "kmeans_kernel.cuh"            /* km_gpu_* (asignación en GPU) */

static void uso(const char *prog)
{
    fprintf(stderr,
        "Uso: %s --data <ruta.bin> [--k %d] [--d %d] [--iters %d] [--seed %u]\n"
        "        [--out results/benchmark.csv] [--conv <csv>] [--rep <i>]\n"
        "        [--platform jetson] [--quiet]\n",
        prog, KM_K_DEF, KM_D_DEF, KM_ITERS_DEF, KM_SEED_DEF);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const char *ruta_datos = NULL;
    const char *ruta_out   = "results/benchmark.csv";
    const char *ruta_conv  = NULL;
    const char *plataforma = "jetson";
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
        else { if (rank == 0) uso(argv[0]); MPI_Finalize(); return 1; }
    }
    if (!ruta_datos || k <= 0 || d <= 0 || iters <= 0 ||
        k > KM_MAX_K || d > KM_MAX_D) {
        if (rank == 0) uso(argv[0]);
        MPI_Finalize();
        return 1;
    }

    char ts_inicio[32], ts_fin[32];
    if (rank == 0) timestamp_iso8601(ts_inicio, sizeof ts_inicio);

    /* --- Rank 0 carga el dataset completo desde el NFS --- */
    double *datos = NULL;
    long n = 0;
    if (rank == 0) {
        datos = cargar_binario(ruta_datos, &n);
        if (!datos) { n = -1; }
    }
    MPI_Bcast(&n, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    if (n < 0) { MPI_Finalize(); return 1; }
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iters, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* --- Reparto de filas (G6, idéntico a main_mpi.c) --- */
    int *conteos_filas = (int *)malloc((size_t)nprocs * sizeof(int));
    int *desplaz       = (int *)malloc((size_t)nprocs * sizeof(int));
    long base = n / nprocs, resto = n % nprocs;
    int off = 0;
    for (int r = 0; r < nprocs; r++) {
        conteos_filas[r] = (int)(base + (r < resto ? 1 : 0));
        desplaz[r] = off;
        off += conteos_filas[r];
    }
    long n_local = conteos_filas[rank];

    double *datos_local = (double *)malloc((size_t)n_local * d * sizeof(double));
    double *centroides  = (double *)malloc((size_t)k * d * sizeof(double));
    double *sumas_loc   = (double *)malloc((size_t)k * d * sizeof(double));
    double *sumas_glob  = (double *)malloc((size_t)k * d * sizeof(double));
    long   *cont_loc    = (long *)malloc((size_t)k * sizeof(long));
    long   *cont_glob   = (long *)malloc((size_t)k * sizeof(long));
    if (!conteos_filas || !desplaz || !datos_local || !centroides ||
        !sumas_loc || !sumas_glob || !cont_loc || !cont_glob) {
        fprintf(stderr, "[rank %d] ERROR: sin memoria\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double t0 = MPI_Wtime();
    double t_comm = 0.0;
    float  ms_h2d = 0.0f, ms_kernel = 0.0f, ms_transfer = 0.0f;

    if (rank == 0)
        inicializar_centroides(datos, n, k, d, semilla, centroides);

    /* --- 4 Scatterv (uno por dimensión) + Bcast de centroides --- */
    double tc = MPI_Wtime();
    for (int dim = 0; dim < d; dim++) {
        const double *envio = (rank == 0) ? datos + (long)dim * n : NULL;
        MPI_Scatterv(envio, conteos_filas, desplaz, MPI_DOUBLE,
                     datos_local + (long)dim * n_local, (int)n_local, MPI_DOUBLE,
                     0, MPI_COMM_WORLD);
    }
    MPI_Bcast(centroides, k * d, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    t_comm += MPI_Wtime() - tc;

    if (rank == 0) { free(datos); datos = NULL; }

    /* --- Subir el bloque local a la GPU (1 vez) --- */
    KmGpu *gpu = km_gpu_crear(datos_local, n_local, k, d, &ms_h2d);
    if (!gpu) {
        fprintf(stderr, "[rank %d] ERROR: fallo init GPU\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    InfoCorrida info = {
        .plataforma = plataforma, .variante = "hybrid",
        .dataset_rows = n, .num_procs = nprocs, .num_gpus = nprocs,
        .repeticion = rep, .k = k, .d = d, .iteraciones = iters,
        .semilla = semilla
    };

    if (!quiet) {
        char host[MPI_MAX_PROCESSOR_NAME];
        int hlen;
        MPI_Get_processor_name(host, &hlen);
        printf("[rank %d/%d] host=%s n_local=%ld (n=%ld) GPU\n",
               rank, nprocs, host, n_local, n);
        fflush(stdout);
    }
    if (rank == 0 && !quiet) {
        fprintf(stderr, "[hybrid] centroides iniciales (semilla %u):\n", semilla);
        for (int j = 0; j < k; j++) {
            fprintf(stderr, "  mu[%d] =", j);
            for (int dim = 0; dim < d; dim++)
                fprintf(stderr, " %.17g", centroides[j * d + dim]);
            fputc('\n', stderr);
        }
    }

    /* --- Bucle de Lloyd: asignación en GPU, actualización replicada en CPU --- */
    for (int it = 1; it <= iters; it++) {
        km_gpu_asignar(gpu, centroides, sumas_loc, cont_loc,
                       &ms_kernel, &ms_transfer);           /* (a)+(b) en GPU */

        double ta = MPI_Wtime();
        MPI_Allreduce(sumas_loc, sumas_glob, k * d, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(cont_loc,  cont_glob,  k,     MPI_LONG,   MPI_SUM, MPI_COMM_WORLD);
        t_comm += MPI_Wtime() - ta;

        actualizar_centroides(sumas_glob, cont_glob, k, d, centroides);  /* (c) en CPU */

        if (ruta_conv) {
            double w_loc = km_gpu_wcss(gpu, centroides, &ms_kernel);
            double w_glob = 0.0;
            MPI_Reduce(&w_loc, &w_glob, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0)
                convergencia_registrar(ruta_conv, &info, it, w_glob);
        }
    }

    /* --- WCSS final: local en GPU + Reduce al rank 0 --- */
    double wcss_loc = km_gpu_wcss(gpu, centroides, &ms_kernel);
    double wcss = 0.0;
    double tr = MPI_Wtime();
    MPI_Reduce(&wcss_loc, &wcss, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    t_comm += MPI_Wtime() - tr;

    double t_fin = MPI_Wtime();
    double wall = t_fin - t0;
    double transfer = (ms_h2d + ms_transfer) / 1000.0;

    /* Reducción por MAX del rank crítico (igual que MPI puro) */
    double wall_max, comm_max, transfer_max;
    MPI_Reduce(&wall,     &wall_max,     1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm,   &comm_max,     1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&transfer, &transfer_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        timestamp_iso8601(ts_fin, sizeof ts_fin);
        double t_computo = wall_max - comm_max - transfer_max;
        Metricas m = {
            .t_total = wall_max, .t_computo = t_computo,
            .t_comunicacion = comm_max, .t_transferencia = transfer_max,
            .wcss = wcss,
            .speedup = NAN, .eficiencia = NAN,
            .potencia_w = NAN, .energia_wh = NAN,   /* Jetson: INA3221 por timestamp */
            .throughput = ((double)n * iters) / wall_max
        };
        if (metrics_escribir_fila(ruta_out, &info, &m, ts_inicio) != 0)
            MPI_Abort(MPI_COMM_WORLD, 1);

        printf("[hybrid] n=%ld k=%d d=%d iters=%d seed=%u rep=%d np=%d\n"
               "         wcss=%.17g\n"
               "         wall_time_s=%.6f compute_time_s=%.6f comm_time_s=%.6f transfer_time_s=%.6f\n"
               "         throughput=%.0f pts/s  inicio=%s fin=%s\n",
               n, k, d, iters, semilla, rep, nprocs, wcss,
               wall_max, t_computo, comm_max, transfer_max, m.throughput,
               ts_inicio, ts_fin);
    }

    km_gpu_destruir(gpu);
    free(conteos_filas); free(desplaz); free(datos_local);
    free(centroides); free(sumas_loc); free(sumas_glob);
    free(cont_loc); free(cont_glob);
    MPI_Finalize();
    return 0;
}
