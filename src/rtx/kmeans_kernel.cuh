/*
 * kmeans_kernel.cuh — interfaz host<->device de la variante CUDA (RTX).
 *
 * El host (main_cuda.cu) sube el dataset UNA vez (H2D) y luego llama a
 * km_cuda_iterar() 100 veces sin tocar PCIe: centroides, sumas y conteos
 * viven en device toda la corrida. La MISMA matemática de kmeans_core.c
 * se replica en los kernels; la validación WCSS 1e-4 es el juez (G9).
 *
 * Layout idéntico al núcleo CPU:
 *   d_datos     : SoA en device, stride n  -> d_datos[dim*n + i]
 *   d_centroides: AoS (k*d)                -> d_centroides[j*d + dim]
 * Los centroides ADEMÁS viven en memoria __constant__ para el kernel de
 * asignación (solo k*d = 20 doubles); km_cuda_iterar los refresca cada iter.
 */
#ifndef KM_KERNEL_CUH
#define KM_KERNEL_CUH

#ifdef __cplusplus
extern "C" {
#endif

/* Handle opaco con todos los buffers de device y el estado de la corrida. */
typedef struct KmCuda KmCuda;

/* Reserva buffers en device y sube el dataset (H2D, 1 vez). Copia los
 * centroides iniciales (calculados en host con el mismo rng que seq).
 * Devuelve NULL si falla alguna llamada CUDA. *ms_h2d recibe el tiempo
 * de la transferencia H2D del dataset (para transfer_time_s). */
KmCuda *km_cuda_crear(const double *h_datos, long n, int k, int d,
                      const double *h_centroides_ini, float *ms_h2d);

/* Una iteración de Lloyd EN DEVICE: (a) asignar+acumular, (b) actualizar
 * centroides. No transfiere nada por PCIe. Acumula en *ms_kernel el tiempo
 * de los kernels de esta iteración (medido con eventos CUDA). */
void km_cuda_iterar(KmCuda *c, float *ms_kernel);

/* WCSS global con los centroides actuales (recomputa asignaciones, G9).
 * Reduce en device y baja UN double (D2H). Suma *ms_d2h. */
double km_cuda_wcss(KmCuda *c, float *ms_d2h);

/* Baja los centroides actuales a host (k*d doubles) para imprimir/depurar. */
void km_cuda_leer_centroides(KmCuda *c, double *h_centroides, float *ms_d2h);

/* Libera todos los buffers de device. */
void km_cuda_destruir(KmCuda *c);

#ifdef __cplusplus
}
#endif

#endif /* KM_KERNEL_CUH */
