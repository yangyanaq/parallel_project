#!/bin/bash
# power_log_tegra.sh — muestrea la potencia de una Jetson Nano por los rieles
# INA3221 y la escribe con timestamps para alinearla luego con el benchmark
# (PLAN §5.2, energía "de dispositivo"). El binario NO mide: esto corre en
# paralelo en cada Jetson durante la corrida, igual que el medidor de pared.
#
# Uso:  ./power_log_tegra.sh <salida.csv> [intervalo_ms]
#   Ctrl-C para parar. Salida: timestamp_iso8601,power_mw,power_w
#
# El riel se detecta automáticamente; si tu Jetson expone otra ruta, exporta
# INA_RAIL con el fichero in_powerX_input correcto.
#
# Riel confirmado en las Nano del clúster (2026-07-04):
#   /sys/devices/50000000.host1x/546c0000.i2c/i2c-6/6-0040/iio:device0/in_power0_input
# (in_power0 = módulo POM_5V_IN, potencia total de entrada en mW.)

SALIDA="${1:-power_tegra.csv}"
INTERVALO_MS="${2:-100}"

# localizar el riel de potencia total (in_power0_input) si no viene dado
if [ -z "$INA_RAIL" ]; then
    INA_RAIL=$(find /sys -name 'in_power0_input' 2>/dev/null | head -1)
fi
if [ -z "$INA_RAIL" ] || [ ! -r "$INA_RAIL" ]; then
    echo "ERROR: no encuentro in_power0_input (rieles INA3221). Exporta INA_RAIL." >&2
    exit 1
fi
echo "[power] riel: $INA_RAIL  intervalo: ${INTERVALO_MS}ms  -> $SALIDA" >&2

echo "timestamp,power_mw,power_w" > "$SALIDA"
sleep_s=$(awk "BEGIN{print $INTERVALO_MS/1000.0}")

trap 'echo "[power] fin" >&2; exit 0' INT TERM
while true; do
    mw=$(cat "$INA_RAIL" 2>/dev/null)
    # ISO-8601 UTC con milisegundos, igual formato que timestamp_iso8601() en C
    ts=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
    w=$(awk "BEGIN{printf \"%.3f\", $mw/1000.0}")
    echo "$ts,$mw,$w" >> "$SALIDA"
    sleep "$sleep_s"
done
