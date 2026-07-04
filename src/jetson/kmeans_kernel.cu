/*
 * kmeans_kernel.cu — kernel CUDA de la variante Jetson (sm_53, CUDA 10.2).
 *
 * Port del kernel RTX (src/rtx/kmeans_kernel.cu). MISMA matemática, misma
 * correspondencia con kmeans_core.c (dist2, argmin con < -> índice menor,
 * clúster vacío conserva). Diferencias respecto a la RTX:
 *
 *  G3 — sm_53 NO tiene atomicAdd(double) nativo (requiere sm_60+). Se emula
 *       con atomicCAS de 64 bits (patrón clásico del CUDA C Programming Guide).
 *  §8.2 — la GPU hace SOLO asignación+acumulación local; la actualización de
 *       centroides es replicada en CPU tras el MPI_Allreduce. Por eso aquí no
 *       hay k_actualizar: km_gpu_asignar baja sumas/conteos locales a host.
 *  CUDA 10.2 -> -std=c++11, nada de features modernas.
 */
#include "kmeans_kernel.cuh"

#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include "../common/config.h"

#define BLK 256

typedef unsigned long long u64;

/* --- G3: atomicAdd(double) emulado para sm_53 (sin nativo) --- */
__device__ double atomicAddDouble(double *dir, double val)
{
    u64 *dir_u64 = (u64 *)dir;
    u64  old = *dir_u64, assumed;
    do {
        assumed = old;
        old = atomicCAS(dir_u64, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);   /* reintenta si otro hilo cambió el valor */
    return __longlong_as_double(old);
}

struct KmGpu {
    long   n_local;
    int    k, d;
    double *d_datos;        /* bloque local SoA n_local*d */
    double *d_centroides;   /* k*d (se sube cada iter) */
    double *d_sumas;        /* k*d */
    u64    *d_conteos;      /* k */
    double *d_wcss_parcial; /* 1 */
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

/* ---- asignación + acumulación local (a)+(b) ----
 * Idéntico al kernel RTX salvo atomicAddDouble en vez de atomicAdd. */
__global__ void k_asignar(const double *datos, long n, int k, int d,
                          const double *centroides, double *g_sumas, u64 *g_conteos)
{
    extern __shared__ double s_mem[];
    double *s_sumas   = s_mem;                      /* k*d doubles */
    u64    *s_conteos = (u64 *)(s_sumas + k * d);   /* k u64 */

    for (int t = threadIdx.x; t < k * d; t += blockDim.x) s_sumas[t] = 0.0;
    for (int t = threadIdx.x; t < k;     t += blockDim.x) s_conteos[t] = 0ULL;
    __syncthreads();

    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
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
            atomicAddDouble(&s_sumas[mejor * d + dim], datos[(long)dim * n + i]);
        atomicAdd(&s_conteos[mejor], 1ULL);         /* atomicAdd(u64) sí existe en sm_53 */
    }
    __syncthreads();

    for (int t = threadIdx.x; t < k * d; t += blockDim.x)
        atomicAddDouble(&g_sumas[t], s_sumas[t]);
    for (int t = threadIdx.x; t < k; t += blockDim.x)
        atomicAdd(&g_conteos[t], s_conteos[t]);
}

/* ---- WCSS local: sum_i min_j ||x_i - mu_j||^2 ---- */
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
    if (threadIdx.x == 0) atomicAddDouble(g_wcss, s_red[0]);
}

/* -------------------- API (extern "C") -------------------- */

KmGpu *km_gpu_crear(const double *h_datos_local, long n_local, int k, int d,
                    float *ms_h2d)
{
    KmGpu *g = (KmGpu *)calloc(1, sizeof(KmGpu));
    if (!g) return NULL;
    g->n_local = n_local; g->k = k; g->d = d;
    g->nbloques = (int)((n_local + BLK - 1) / BLK);

    CHK(cudaMalloc(&g->d_datos,        (size_t)n_local * d * sizeof(double)));
    CHK(cudaMalloc(&g->d_centroides,   (size_t)k * d * sizeof(double)));
    CHK(cudaMalloc(&g->d_sumas,        (size_t)k * d * sizeof(double)));
    CHK(cudaMalloc(&g->d_conteos,      (size_t)k * sizeof(u64)));
    CHK(cudaMalloc(&g->d_wcss_parcial, sizeof(double)));

    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    CHK(cudaMemcpy(g->d_datos, h_datos_local, (size_t)n_local * d * sizeof(double),
                   cudaMemcpyHostToDevice));
    cudaEventRecord(b); cudaEventSynchronize(b);
    if (ms_h2d) cudaEventElapsedTime(ms_h2d, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return g;
}

void km_gpu_asignar(KmGpu *g, const double *h_centroides,
                    double *h_sumas, long *h_conteos,
                    float *ms_kernel, float *ms_transfer)
{
    cudaEvent_t a, b;
    float ms;

    /* transfer: subir centroides */
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    cudaMemcpy(g->d_centroides, h_centroides, (size_t)g->k * g->d * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemset(g->d_sumas,   0, (size_t)g->k * g->d * sizeof(double));
    cudaMemset(g->d_conteos, 0, (size_t)g->k * sizeof(u64));
    cudaEventRecord(b); cudaEventSynchronize(b);
    ms = 0.0f; cudaEventElapsedTime(&ms, a, b); if (ms_transfer) *ms_transfer += ms;
    cudaEventDestroy(a); cudaEventDestroy(b);

    /* kernel */
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    size_t shmem = (size_t)g->k * g->d * sizeof(double) + (size_t)g->k * sizeof(u64);
    k_asignar<<<g->nbloques, BLK, shmem>>>(g->d_datos, g->n_local, g->k, g->d,
                                           g->d_centroides, g->d_sumas, g->d_conteos);
    cudaEventRecord(b); cudaEventSynchronize(b);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        fprintf(stderr, "CUDA error en asignar: %s\n", cudaGetErrorString(e));
    ms = 0.0f; cudaEventElapsedTime(&ms, a, b); if (ms_kernel) *ms_kernel += ms;
    cudaEventDestroy(a); cudaEventDestroy(b);

    /* transfer: bajar sumas (double) y conteos (u64 -> long host) */
    u64 *tmp_conteos = (u64 *)malloc((size_t)g->k * sizeof(u64));
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    cudaMemcpy(h_sumas, g->d_sumas, (size_t)g->k * g->d * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(tmp_conteos, g->d_conteos, (size_t)g->k * sizeof(u64),
               cudaMemcpyDeviceToHost);
    cudaEventRecord(b); cudaEventSynchronize(b);
    ms = 0.0f; cudaEventElapsedTime(&ms, a, b); if (ms_transfer) *ms_transfer += ms;
    cudaEventDestroy(a); cudaEventDestroy(b);

    for (int j = 0; j < g->k; j++) h_conteos[j] = (long)tmp_conteos[j];
    free(tmp_conteos);
}

double km_gpu_wcss(KmGpu *g, const double *h_centroides, float *ms_kernel)
{
    cudaMemcpy(g->d_centroides, h_centroides, (size_t)g->k * g->d * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemset(g->d_wcss_parcial, 0, sizeof(double));

    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    k_wcss<<<g->nbloques, BLK>>>(g->d_datos, g->n_local, g->k, g->d,
                                 g->d_centroides, g->d_wcss_parcial);
    cudaEventRecord(b); cudaEventSynchronize(b);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        fprintf(stderr, "CUDA error en wcss: %s\n", cudaGetErrorString(e));
    float ms = 0.0f; cudaEventElapsedTime(&ms, a, b); if (ms_kernel) *ms_kernel += ms;
    cudaEventDestroy(a); cudaEventDestroy(b);

    double wcss_loc = 0.0;
    cudaMemcpy(&wcss_loc, g->d_wcss_parcial, sizeof(double), cudaMemcpyDeviceToHost);
    return wcss_loc;
}

void km_gpu_destruir(KmGpu *g)
{
    if (!g) return;
    cudaFree(g->d_datos); cudaFree(g->d_centroides);
    cudaFree(g->d_sumas); cudaFree(g->d_conteos);
    cudaFree(g->d_wcss_parcial);
    free(g);
}
