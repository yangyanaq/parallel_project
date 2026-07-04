/*
 * energia_nvml.cu — implementación del muestreo NVML (Win32 thread).
 * Se compila con nvcc (arrastra <nvml.h> del toolkit; enlaza nvml.lib).
 */
#include "energia_nvml.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <nvml.h>
#include <windows.h>

struct EnergiaNVML {
    nvmlDevice_t dev;
    HANDLE       hilo;
    volatile LONG parar;
    int          intervalo_ms;

    /* integración por trapecios: acumulamos energía (Wh) y tiempo total */
    double energia_wh;
    double suma_w;         /* para el promedio simple */
    long   muestras;
    int    nvml_ok;
};

/* Lee potencia en W (nvmlDeviceGetPowerUsage da miliwatts). -1 si falla. */
static double leer_w(nvmlDevice_t dev)
{
    unsigned int mw = 0;
    if (nvmlDeviceGetPowerUsage(dev, &mw) != NVML_SUCCESS) return -1.0;
    return (double)mw / 1000.0;
}

static DWORD WINAPI bucle_muestreo(LPVOID arg)
{
    EnergiaNVML *e = (EnergiaNVML *)arg;
    LARGE_INTEGER frec, t_prev, t_now;
    QueryPerformanceFrequency(&frec);
    QueryPerformanceCounter(&t_prev);
    double w_prev = leer_w(e->dev);

    while (!InterlockedCompareExchange(&e->parar, 0, 0)) {
        Sleep(e->intervalo_ms);
        double w_now = leer_w(e->dev);
        QueryPerformanceCounter(&t_now);
        double dt = (double)(t_now.QuadPart - t_prev.QuadPart) / (double)frec.QuadPart;

        if (w_now >= 0.0 && w_prev >= 0.0) {
            /* trapecio: (w_prev+w_now)/2 * dt [W·s] -> /3600 [Wh] */
            e->energia_wh += (w_prev + w_now) * 0.5 * dt / 3600.0;
            e->suma_w += w_now;
            e->muestras++;
        }
        w_prev = w_now;
        t_prev = t_now;
    }
    return 0;
}

EnergiaNVML *energia_iniciar(int intervalo_ms)
{
    if (nvmlInit() != NVML_SUCCESS) {
        fprintf(stderr, "AVISO: NVML no disponible; energia quedara en NaN\n");
        return NULL;
    }
    EnergiaNVML *e = (EnergiaNVML *)calloc(1, sizeof(EnergiaNVML));
    if (!e) { nvmlShutdown(); return NULL; }
    e->intervalo_ms = intervalo_ms > 0 ? intervalo_ms : 100;

    if (nvmlDeviceGetHandleByIndex(0, &e->dev) != NVML_SUCCESS) {
        fprintf(stderr, "AVISO: NVML sin device 0; energia en NaN\n");
        free(e); nvmlShutdown(); return NULL;
    }
    e->nvml_ok = 1;
    e->parar = 0;
    e->hilo = CreateThread(NULL, 0, bucle_muestreo, e, 0, NULL);
    if (!e->hilo) { free(e); nvmlShutdown(); return NULL; }
    return e;
}

void energia_detener(EnergiaNVML *e, double *avg_w, double *wh)
{
    if (!e) { if (avg_w) *avg_w = NAN; if (wh) *wh = NAN; return; }
    InterlockedExchange(&e->parar, 1);
    WaitForSingleObject(e->hilo, INFINITE);
    CloseHandle(e->hilo);

    if (e->muestras > 0) {
        if (avg_w) *avg_w = e->suma_w / (double)e->muestras;
        if (wh)    *wh    = e->energia_wh;
    } else {
        if (avg_w) *avg_w = NAN;
        if (wh)    *wh    = NAN;
    }
    nvmlShutdown();
    free(e);
}
