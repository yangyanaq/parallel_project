#include "io_dataset.h"
#include "rng.h"

#include <stdio.h>
#include <stdlib.h>

double *cargar_binario(const char *ruta, long *n_out)
{
    *n_out = 0;
    FILE *f = fopen(ruta, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: no se pudo abrir %s\n", ruta);
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);

    const long fila = 4 * (long)sizeof(double);      /* d=4 fijado por el formato */
    if (bytes <= 0 || bytes % fila != 0) {
        fprintf(stderr, "ERROR: %s corrupto (%ld bytes no es multiplo de %ld)\n",
                ruta, bytes, fila);
        fclose(f);
        return NULL;
    }
    long n = bytes / fila;

    double *datos = (double *)malloc((size_t)n * 4 * sizeof(double));
    if (!datos) {
        fprintf(stderr, "ERROR: sin memoria para %ld filas\n", n);
        fclose(f);
        return NULL;
    }
    size_t leidos = fread(datos, sizeof(double), (size_t)n * 4, f);
    fclose(f);
    if (leidos != (size_t)n * 4) {
        fprintf(stderr, "ERROR: lectura incompleta de %s\n", ruta);
        free(datos);
        return NULL;
    }
    *n_out = n;
    return datos;                                    /* datos[dim*n + i] */
}

void inicializar_centroides(const double *datos, long n, int k, int d,
                            unsigned int semilla, double *centroides)
{
    /* k índices distintos por rechazo: mismo consumo del stream del PRNG
     * en C y en Python (validate_wcss.py replica este bucle tal cual). */
    long idx[64];                                    /* k <= KM_MAX_K << 64 */
    int elegidos = 0;

    rng_init((uint64_t)semilla);
    while (elegidos < k) {
        long cand = rng_indice(n);
        int repetido = 0;
        for (int j = 0; j < elegidos; j++)
            if (idx[j] == cand) { repetido = 1; break; }
        if (!repetido)
            idx[elegidos++] = cand;
    }
    for (int j = 0; j < k; j++)
        for (int dim = 0; dim < d; dim++)
            centroides[j * d + dim] = datos[(long)dim * n + idx[j]];
}
