#include "rng.h"

/* xorshift64* (Vigna, 2016). Estado de 64 bits, un solo hilo lo usa
 * (la init de centroides ocurre solo en rank 0 / host). */

static uint64_t estado = 0x9E3779B97F4A7C15ULL;

void rng_init(uint64_t semilla)
{
    /* xorshift muere con estado 0; se reemplaza por una constante fija
     * para que siga siendo determinista. */
    estado = (semilla != 0) ? semilla : 0x9E3779B97F4A7C15ULL;
}

uint64_t rng_next(void)
{
    uint64_t x = estado;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    estado = x;
    return x * 0x2545F4914F6CDD1DULL;
}

long rng_indice(long n)
{
    return (long)(rng_next() % (uint64_t)n);
}
