#include "kmeans_core.h"

#include <float.h>

/* Distancia euclidiana al cuadrado punto i vs centroide j (sin sqrt:
 * solo se compara). Se mantiene como helper para que los kernels CUDA
 * tengan una correspondencia 1:1 con esta matemática. */
static double dist2(const double *datos, long i, long n_stride,
                    const double *centroides, int j, int d)
{
    double acc = 0.0;
    for (int dim = 0; dim < d; dim++) {
        double diff = datos[(long)dim * n_stride + i] - centroides[j * d + dim];
        acc += diff * diff;
    }
    return acc;
}

/* argmin_j dist2(i, j); con < estricto el empate queda en el índice
 * menor, igual que np.argmin (primera ocurrencia del mínimo). */
static int mas_cercano(const double *datos, long i, long n_stride,
                       const double *centroides, int k, int d,
                       double *dist_min_out)
{
    int    mejor = 0;
    double dist_min = DBL_MAX;
    for (int j = 0; j < k; j++) {
        double dist = dist2(datos, i, n_stride, centroides, j, d);
        if (dist < dist_min) {
            dist_min = dist;
            mejor = j;
        }
    }
    if (dist_min_out) *dist_min_out = dist_min;
    return mejor;
}

void asignar_y_acumular(const double *datos, long n_local, long n_stride,
                        const double *centroides, int k, int d,
                        double *sumas, long *conteos)
{
    for (int j = 0; j < k * d; j++) sumas[j] = 0.0;
    for (int j = 0; j < k; j++)     conteos[j] = 0;

    for (long i = 0; i < n_local; i++) {
        int j = mas_cercano(datos, i, n_stride, centroides, k, d, 0);
        for (int dim = 0; dim < d; dim++)
            sumas[j * d + dim] += datos[(long)dim * n_stride + i];
        conteos[j]++;
    }
}

void actualizar_centroides(const double *sumas, const long *conteos,
                           int k, int d, double *centroides)
{
    for (int j = 0; j < k; j++) {
        if (conteos[j] > 0)
            for (int dim = 0; dim < d; dim++)
                centroides[j * d + dim] = sumas[j * d + dim] / (double)conteos[j];
        /* conteo 0: se conserva el centroide anterior */
    }
}

double calcular_wcss(const double *datos, long n_local, long n_stride,
                     const double *centroides, int k, int d)
{
    double wcss = 0.0;
    for (long i = 0; i < n_local; i++) {
        double dist_min;
        mas_cercano(datos, i, n_stride, centroides, k, d, &dist_min);
        wcss += dist_min;
    }
    return wcss;
}
