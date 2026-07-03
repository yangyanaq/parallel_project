/*
 * rng.h — PRNG propio (xorshift64*), determinista e idéntico en
 * glibc / MSVC / nvcc / numpy (gotcha G1: rand() NO es portable).
 *
 * La réplica exacta en Python vive en scripts/validate_wcss.py; si se
 * toca este algoritmo hay que tocar los dos.
 */
#ifndef KM_RNG_H
#define KM_RNG_H

#include <stdint.h>

/* Siembra el generador. semilla != 0 (si llega 0 se sustituye por una
 * constante fija, documentado en rng.c). */
void     rng_init(uint64_t semilla);

/* Siguiente valor crudo de 64 bits (xorshift64*). */
uint64_t rng_next(void);

/* Índice uniforme en [0, n) vía módulo (sesgo despreciable para n<<2^64
 * y, sobre todo, replicable exactamente en Python). n > 0. */
long     rng_indice(long n);

#endif /* KM_RNG_H */
