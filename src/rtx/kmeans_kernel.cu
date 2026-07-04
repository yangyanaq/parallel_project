/*
 * kmeans_kernel.cu — kernels CUDA de la variante RTX (sm_89, CUDA 11.8).
 *
 * Correspondencia línea a línea con src/common/kmeans_core.c:
 *   - dist2()        <-> mismo bucle sobre d, diferencia al cuadrado.
 *   - argmin con <   <-> empate al índice MENOR (igual que mas_cercano()).
 *   - clúster vacío  <-> conserva el centroide anterior (actualizar_centroides).
 * El orden de sumatoria difiere (atómicas en paralelo) -> equivalencia por
 * tolerancia 1e-4, no igualdad exacta (CLAUDE.md §4, G9).
 *
 * sm_89 tiene atomicAdd(double) NATIVO. La reducción shared-mem por bloque
 * baja la contención de ~10M atómicas globales a ~(bloques * k) (§8.3).
 */
#include "kmeans_kernel.cuh"

#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include "../common/config.h"     /* KM_MAX_K, KM_MAX_D */

#define BLK 256                   /* hilos por bloque */

/* Conteos en device: SIEMPRE 64 bits (ver nota en k_asignar). */
typedef unsigned long long u64;

/* Los centroides viven en memoria GLOBAL (d_centroides) y se pasan como
 * argumento a los kernels. Nota de diseño: probamos centroides en
 * __constant__ + cudaMemcpyToSymbol D2D por iteración, pero esa copia
 * device->device al símbolo dejaba el stream en error (wcss=0); pasar el
 * puntero global es correcto y, con k*d=20 doubles cacheados, igual de
 * rápido en la práctica. */

struct KmCuda {
    long   n;
    int    k, d;
    double *d_datos;        /* SoA n*d en device */
    double *d_centroides;   /* AoS k*d */
    double *d_sumas;        /* AoS k*d */
    u64    *d_conteos;      /* k (64 bits: atomicAdd + long de 4 bytes en Win) */
    double *d_wcss_parcial; /* 1 (reducción global) */
    int     nbloques;
};

#define CHK(call) do {                                              \
    cudaError_t _e = (call);                                        \
    if (_e != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(_e));        \
        return NULL;                                                \
    }                                                               \
} while (0)

/* ---- kernel (a)+(b): asignar cada punto y acumular sumas/conteos ----
 * Réplica de asignar_y_acumular(). Acumulación en 2 niveles:
 *  1) cada bloque acumula en shared (s_sumas[k*d], s_conteos[k]) con átomicas
 *     de bloque (baratas);
 *  2) un representante del bloque vuelca a global con atomicAdd. */
/* En Windows/MSVC `long` es 4 bytes, así que tratar un long* como
 * (unsigned long long*) para atomicAdd daba "misaligned address". Por eso los
 * conteos son u64 en todo el device; km_cuda_iterar no baja conteos a host. */
__global__ void k_asignar(const double *datos, long n, int k, int d,
                          const double *centroides, double *g_sumas, u64 *g_conteos)
{
    extern __shared__ double s_mem[];
    double *s_sumas   = s_mem;                      /* k*d doubles */
    u64    *s_conteos = (u64 *)(s_sumas + k * d);   /* k u64 (alineado a 8) */

    for (int t = threadIdx.x; t < k * d; t += blockDim.x) s_sumas[t] = 0.0;
    for (int t = threadIdx.x; t < k;     t += blockDim.x) s_conteos[t] = 0ULL;
    __syncthreads();

    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        /* argmin_j ||x_i - mu_j||^2 con < estricto (empate -> j menor) */
        int    mejor = 0;
        double dist_min = DBL_MAX;
        for (int j = 0; j < k; j++) {
            double acc = 0.0;
            for (int dim = 0; dim < d; dim++) {
                double diff = datos[(long)dim * n + i] - centroides[j * d + dim];
                acc += diff * diff;
            }
            if (acc < dist_min) { dist_min = acc; mejor = j; }
        }
        for (int dim = 0; dim < d; dim++)
            atomicAdd(&s_sumas[mejor * d + dim], datos[(long)dim * n + i]);
        atomicAdd(&s_conteos[mejor], 1ULL);
    }
    __syncthreads();

    for (int t = threadIdx.x; t < k * d; t += blockDim.x)
        atomicAdd(&g_sumas[t], s_sumas[t]);
    for (int t = threadIdx.x; t < k; t += blockDim.x)
        atomicAdd(&g_conteos[t], s_conteos[t]);
}

/* ---- kernel (c): actualizar centroides (k*d hilos) ----
 * Réplica de actualizar_centroides(): mu_j = suma_j/conteo_j; vacío conserva. */
__global__ void k_actualizar(double *centroides, const double *sumas,
                             const u64 *conteos, int k, int d)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < k * d) {
        int j = t / d;
        if (conteos[j] > 0ULL)
            centroides[t] = sumas[t] / (double)conteos[j];
        /* conteo 0: se conserva el valor previo (no se toca) */
    }
}

/* ---- kernel WCSS: sum_i min_j ||x_i - mu_j||^2 ----
 * Reducción shared por bloque + atomicAdd global. Réplica de calcular_wcss(). */
__global__ void k_wcss(const double *datos, long n, int k, int d,
                       const double *centroides, double *g_wcss)
{
    __shared__ double s_red[BLK];
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;

    double dist_min = 0.0;
    if (i < n) {
        double m = DBL_MAX;
        for (int j = 0; j < k; j++) {
            double acc = 0.0;
            for (int dim = 0; dim < d; dim++) {
                double diff = datos[(long)dim * n + i] - centroides[j * d + dim];
                acc += diff * diff;
            }
            if (acc < m) m = acc;
        }
        dist_min = m;
    }
    s_red[threadIdx.x] = dist_min;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_red[threadIdx.x] += s_red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(g_wcss, s_red[0]);
}

/* -------------------- API (extern "C") -------------------- */

KmCuda *km_cuda_crear(const double *h_datos, long n, int k, int d,
                      const double *h_centroides_ini, float *ms_h2d)
{
    KmCuda *c = (KmCuda *)calloc(1, sizeof(KmCuda));
    if (!c) return NULL;
    c->n = n; c->k = k; c->d = d;
    c->nbloques = (int)((n + BLK - 1) / BLK);

    CHK(cudaMalloc(&c->d_datos,        (size_t)n * d * sizeof(double)));
    CHK(cudaMalloc(&c->d_centroides,   (size_t)k * d * sizeof(double)));
    CHK(cudaMalloc(&c->d_sumas,        (size_t)k * d * sizeof(double)));
    CHK(cudaMalloc(&c->d_conteos,      (size_t)k * sizeof(u64)));
    CHK(cudaMalloc(&c->d_wcss_parcial, sizeof(double)));

    /* H2D del dataset (1 vez), cronometrado con eventos CUDA. */
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    CHK(cudaMemcpy(c->d_datos, h_datos, (size_t)n * d * sizeof(double),
                   cudaMemcpyHostToDevice));
    cudaEventRecord(b); cudaEventSynchronize(b);
    if (ms_h2d) cudaEventElapsedTime(ms_h2d, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);

    CHK(cudaMemcpy(c->d_centroides, h_centroides_ini,
                   (size_t)k * d * sizeof(double), cudaMemcpyHostToDevice));
    return c;
}

void km_cuda_iterar(KmCuda *c, float *ms_kernel)
{
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);

    cudaMemset(c->d_sumas,   0, (size_t)c->k * c->d * sizeof(double));
    cudaMemset(c->d_conteos, 0, (size_t)c->k * sizeof(u64));

    size_t shmem = (size_t)c->k * c->d * sizeof(double) + (size_t)c->k * sizeof(u64);
    k_asignar<<<c->nbloques, BLK, shmem>>>(c->d_datos, c->n, c->k, c->d,
                                           c->d_centroides, c->d_sumas, c->d_conteos);
    int nb_upd = (c->k * c->d + BLK - 1) / BLK;
    k_actualizar<<<nb_upd, BLK>>>(c->d_centroides, c->d_sumas, c->d_conteos,
                                  c->k, c->d);

    cudaEventRecord(b); cudaEventSynchronize(b);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        fprintf(stderr, "CUDA error en iterar: %s\n", cudaGetErrorString(e));
    float ms = 0.0f; cudaEventElapsedTime(&ms, a, b);
    if (ms_kernel) *ms_kernel += ms;
    cudaEventDestroy(a); cudaEventDestroy(b);
}

double km_cuda_wcss(KmCuda *c, float *ms_d2h)
{
    cudaMemset(c->d_wcss_parcial, 0, sizeof(double));
    k_wcss<<<c->nbloques, BLK>>>(c->d_datos, c->n, c->k, c->d,
                                 c->d_centroides, c->d_wcss_parcial);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        fprintf(stderr, "CUDA error en wcss: %s\n", cudaGetErrorString(e));

    double wcss = 0.0;
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    cudaMemcpy(&wcss, c->d_wcss_parcial, sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0.0f; cudaEventElapsedTime(&ms, a, b);
    if (ms_d2h) *ms_d2h += ms;
    cudaEventDestroy(a); cudaEventDestroy(b);
    return wcss;
}

void km_cuda_leer_centroides(KmCuda *c, double *h_centroides, float *ms_d2h)
{
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    cudaMemcpy(h_centroides, c->d_centroides,
               (size_t)c->k * c->d * sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0.0f; cudaEventElapsedTime(&ms, a, b);
    if (ms_d2h) *ms_d2h += ms;
    cudaEventDestroy(a); cudaEventDestroy(b);
}

void km_cuda_destruir(KmCuda *c)
{
    if (!c) return;
    cudaFree(c->d_datos); cudaFree(c->d_centroides);
    cudaFree(c->d_sumas); cudaFree(c->d_conteos);
    cudaFree(c->d_wcss_parcial);
    free(c);
}
