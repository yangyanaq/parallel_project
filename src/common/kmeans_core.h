/*
 * kmeans_core.h — TODA la matemática del K-Means (Lloyd) vive aquí.
 * Las 4 variantes llaman a estas funciones (los kernels CUDA replican
 * esta matemática línea a línea; la validación WCSS 1e-4 es el juez).
 *
 * Convenciones de layout:
 *   datos      : SoA con stride n_stride -> datos[dim*n_stride + i]
 *   centroides : AoS               -> centroides[j*d + dim]
 *   sumas      : AoS como centroides (k*d), conteos: k
 *
 * n_stride es el n del layout SoA: permite usar la misma función sobre
 * el dataset completo (seq/RTX, n_stride = n) o sobre el bloque local
 * de un rank MPI (n_stride = n_local) sin copiar.
 */
#ifndef KM_KMEANS_CORE_H
#define KM_KMEANS_CORE_H

/* Un paso (a)+(b): asigna cada punto a su centroide más cercano
 * (distancia euclidiana AL CUADRADO; empates -> índice menor) y acumula
 * sumas y conteos por clúster. sumas/conteos se ponen a cero adentro. */
void asignar_y_acumular(const double *datos, long n_local, long n_stride,
                        const double *centroides, int k, int d,
                        double *sumas, long *conteos);

/* Paso (c): mu_j = sumas_j / conteos_j. Clúster vacío (conteo 0) ->
 * conserva el centroide anterior (política documentada, CLAUDE.md §4). */
void actualizar_centroides(const double *sumas, const long *conteos,
                           int k, int d, double *centroides);

/* WCSS = sum_i min_j ||x_i - mu_j||^2, recomputando asignaciones con los
 * centroides dados (gotcha G9: misma fórmula para validación y curva). */
double calcular_wcss(const double *datos, long n_local, long n_stride,
                     const double *centroides, int k, int d);

#endif /* KM_KMEANS_CORE_H */
