/*
 * energia_nvml.h — muestreo de potencia GPU con NVML en un hilo aparte (G4).
 *
 * Arranca un hilo Win32 que lee nvmlDeviceGetPowerUsage() cada ~100 ms;
 * al parar, integra potencia×tiempo -> Wh y da el promedio en W.
 * Si NVML no está disponible, energia_iniciar devuelve NULL y el resto
 * del programa sigue (energía queda en NaN, no aborta la corrida).
 */
#ifndef KM_ENERGIA_NVML_H
#define KM_ENERGIA_NVML_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct EnergiaNVML EnergiaNVML;

/* Inicializa NVML sobre el device 0 y lanza el hilo de muestreo.
 * Devuelve NULL si NVML falla (se sigue sin energía). */
EnergiaNVML *energia_iniciar(int intervalo_ms);

/* Detiene el hilo e integra. avg_w = potencia media (W); wh = energía (Wh).
 * Ambos quedan en NAN si nunca hubo una muestra válida. */
void energia_detener(EnergiaNVML *e, double *avg_w, double *wh);

#ifdef __cplusplus
}
#endif

#endif /* KM_ENERGIA_NVML_H */
