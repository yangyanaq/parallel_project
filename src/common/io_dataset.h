/*
 * io_dataset.h — carga del binario SoA e inicialización de centroides.
 * Contrato del formato: CLAUDE.md §5.1 (float64 LE, sin cabecera, SoA,
 * orden pickup_lat / pickup_lon / dropoff_lat / dropoff_lon).
 */
#ifndef KM_IO_DATASET_H
#define KM_IO_DATASET_H

/* Carga el .bin completo a memoria. Devuelve el puntero (malloc, el
 * llamador libera) y deja en *n_out el número de filas deducido del
 * tamaño del archivo: n = bytes / (4*8). Acceso: datos[dim*n + i].
 * Devuelve NULL (y *n_out = 0) si el archivo no existe o está corrupto
 * (tamaño no múltiplo de 32 bytes). */
double *cargar_binario(const char *ruta, long *n_out);

/* Elige k índices aleatorios DISTINTOS con el PRNG propio (rng_init(semilla)
 * se llama adentro) y copia esos puntos como centroides iniciales.
 * Salida AoS: centroides[j*d + dim], tamaño k*d (provisto por el llamador).
 * Idéntico en las 4 variantes y replicado en validate_wcss.py. */
void inicializar_centroides(const double *datos, long n, int k, int d,
                            unsigned int semilla, double *centroides);

#endif /* KM_IO_DATASET_H */
