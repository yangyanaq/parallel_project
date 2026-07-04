/*
 * kmeans_kernel.cuh — interfaz host<->device del kernel Jetson (sm_53).
 *
 * A diferencia de la RTX (que hace TODO el bucle en device), aquí la GPU solo
 * ejecuta el paso de ASIGNACIÓN+ACUMULACIÓN local (a+b); la actualización de
 * centroides es replicada en CPU tras el MPI_Allreduce (main_hybrid.c). Por
 * eso la interfaz devuelve sumas/conteos LOCALES en host, listos para reducir.
 *
 * El bloque local sube UNA vez (H2D) antes del bucle; por iteración solo suben
 * los centroides (k*d doubles) y bajan sumas+conteos (k*d + k). transfer_time_s
 * los acumula (CLAUDE.md §8.2, PLAN §5.2).
 *
 * La MATEMÁTICA es idéntica a la del kernel RTX y a kmeans_core.c; la única
 * diferencia es G3: sm_53 no tiene atomicAdd(double) nativo -> emulación con
 * atomicCAS(unsigned long long) en el .cu. La validación WCSS 1e-4 es el juez.
 */
#ifndef KM_JETSON_KERNEL_CUH
#define KM_JETSON_KERNEL_CUH

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KmGpu KmGpu;

/* Reserva buffers en device y sube el bloque LOCAL (n_local*d, SoA con stride
 * n_local). Se llama una vez por rank. Devuelve NULL si falla CUDA.
 * *ms_h2d recibe el tiempo del H2D del bloque (transfer_time_s). */
KmGpu *km_gpu_crear(const double *h_datos_local, long n_local, int k, int d,
                    float *ms_h2d);

/* Paso (a)+(b) en GPU: dado los centroides actuales (host, k*d), calcula
 * sumas y conteos LOCALES y los deja en host (sumas: k*d doubles; conteos:
 * k long). Acumula en *ms_kernel el tiempo de kernel y en *ms_transfer el
 * de las transferencias (centroides H2D + parciales D2H) de esta iteración. */
void km_gpu_asignar(KmGpu *g, const double *h_centroides,
                    double *h_sumas, long *h_conteos,
                    float *ms_kernel, float *ms_transfer);

/* WCSS local con los centroides dados (recomputa asignaciones en GPU, G9).
 * Devuelve el parcial local (el host lo reduce con MPI_Reduce). */
double km_gpu_wcss(KmGpu *g, const double *h_centroides, float *ms_kernel);

/* Libera los buffers de device. */
void km_gpu_destruir(KmGpu *g);

#ifdef __cplusplus
}
#endif

#endif /* KM_JETSON_KERNEL_CUH */
