#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 aggregate_power.py — post-proceso del benchmark.csv crudo -> benchmark_final.csv
=============================================================================
El binario deja speedup/efficiency en NaN (no conoce el T_1 promedio) y, en los
clústeres, también la energía. Este script cierra esas columnas:

  1) speedup   = T_1 / T_p     (T_1 = media de seq por tamaño, rep>0)
     efficiency= speedup / num_procs
     (para GPU se reporta speedup vs T_1 con num_procs=1; se documenta el "p").
  2) energía Jetson: integra los logs INA3221 (scripts/power_log_tegra.sh) por
     rango de timestamps [inicio,fin] de cada corrida -> avg_power_w, energy_wh.
     Suma los rieles de las 3 Jetson activas según num_procs.
  3) energía RTX: ya viene del binario (NVML); se respeta si no es NaN.
  4) energía de pared (RPi): se completa desde un CSV manual del medidor
     (--wall-log) alineando por timestamp; si no se pasa, queda NaN.

La corrida en frío (repetition==0) NO se descarta aquí; se marca y el graficado
la filtra. Ver PLAN §6, CLAUDE.md §9/§10.

USO:
  python aggregate_power.py --in results/benchmark.csv --out results/benchmark_final.csv \
     [--power-dir <dir con tegra_*.csv>] [--wall-log <medidor.csv>]
=============================================================================
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ITERS_DEFAULT = 100


def cargar_logs_potencia(power_dir):
    """Lee tegra_<ip>_<tag>.csv (timestamp,power_mw,power_w) -> lista de series
    ordenadas por timestamp, una por archivo. Devuelve un DataFrame concatenado
    con columna 'ts' (datetime UTC) y 'w'."""
    if not power_dir or not os.path.isdir(power_dir):
        return None
    marcos = []
    for f in glob.glob(os.path.join(power_dir, "tegra_*.csv")):
        try:
            df = pd.read_csv(f)
            if "timestamp" not in df or "power_w" not in df:
                continue
            df["ts"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["ts"]).sort_values("ts")
            marcos.append(df[["ts", "power_w"]].rename(columns={"power_w": "w"}))
        except Exception as e:
            print(f"[aviso] no pude leer {f}: {e}", file=sys.stderr)
    if not marcos:
        return None
    return pd.concat(marcos, ignore_index=True).sort_values("ts")


def integrar_potencia(pot, ini, fin):
    """Media (W) y energía (Wh) de las muestras de potencia en [ini,fin].
    Suma la potencia instantánea de todas las Jetson activas (los logs vienen
    concatenados); integra por trapecios sobre el tiempo. NaN si no hay muestras."""
    if pot is None or pd.isna(ini) or pd.isna(fin):
        return np.nan, np.nan
    m = pot[(pot["ts"] >= ini) & (pot["ts"] <= fin)]
    if len(m) < 2:
        return np.nan, np.nan
    # potencia total del sistema en cada instante ~ media de rieles * n muestras
    # simultáneas; como los logs están concatenados por nodo, agrupamos por ts.
    porslot = m.groupby("ts")["w"].sum().sort_index()
    t = porslot.index.view("int64") / 1e9          # segundos
    w = porslot.values
    dt = np.diff(t)
    energia_wh = float(np.sum((w[:-1] + w[1:]) * 0.5 * dt) / 3600.0)
    avg_w = float(np.mean(w))
    return avg_w, energia_wh


def main():
    ap = argparse.ArgumentParser(description="Completa speedup/efficiency/energía del benchmark.")
    ap.add_argument("--in", dest="entrada", default="results/benchmark.csv")
    ap.add_argument("--out", dest="salida", default="results/benchmark_final.csv")
    ap.add_argument("--power-dir", default=None, help="dir con tegra_*.csv (INA3221 Jetson)")
    ap.add_argument("--wall-log", default=None, help="CSV del medidor de pared (RPi)")
    args = ap.parse_args()

    if not os.path.exists(args.entrada):
        sys.exit(f"ERROR: no existe {args.entrada}")
    df = pd.read_csv(args.entrada)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # ---- T_1 por tamaño: media del seq (rep>0), 1 núcleo RPi ----
    seq = df[(df["variant"] == "seq") & (df["repetition"] > 0)]
    if seq.empty:
        print("[aviso] no hay filas seq rep>0; speedup quedará NaN", file=sys.stderr)
    t1 = seq.groupby("dataset_rows")["wall_time_s"].mean().to_dict()
    print("[T_1] (s) por tamaño:", {int(k): round(v, 4) for k, v in t1.items()})

    # ---- speedup / efficiency ----
    def calc_speedup(fila):
        base = t1.get(fila["dataset_rows"])
        if base is None or fila["wall_time_s"] <= 0:
            return np.nan
        return base / fila["wall_time_s"]

    df["speedup"] = df.apply(calc_speedup, axis=1)
    # efficiency = speedup / num_procs (documentar el 'p' usado en GPU: num_procs)
    df["efficiency"] = df["speedup"] / df["num_procs"].replace(0, np.nan)

    # ---- energía Jetson (INA3221) por rango de timestamps de cada corrida ----
    pot = cargar_logs_potencia(args.power_dir)
    if pot is not None:
        print(f"[power] {len(pot)} muestras INA3221 cargadas")
        # duración de cada corrida ~ wall_time_s; ventana [ts, ts+wall]
        for i, fila in df[df["variant"] == "hybrid"].iterrows():
            ini = fila["timestamp"]
            fin = ini + pd.to_timedelta(fila["wall_time_s"], unit="s") if pd.notna(ini) else pd.NaT
            avg_w, wh = integrar_potencia(pot, ini, fin)
            if not np.isnan(avg_w):
                df.at[i, "avg_power_w"] = round(avg_w, 3)
                df.at[i, "energy_wh"] = round(wh, 6)

    # ---- energía de pared (RPi) desde el medidor manual, si se da ----
    if args.wall_log and os.path.exists(args.wall_log):
        wall = pd.read_csv(args.wall_log)
        wall["ts"] = pd.to_datetime(wall["timestamp"], utc=True, errors="coerce")
        wall = wall.dropna(subset=["ts"]).sort_values("ts")
        wpot = wall.rename(columns={"power_w": "w"})[["ts", "w"]]
        for i, fila in df[df["platform"] == "rpi5"].iterrows():
            ini = fila["timestamp"]
            fin = ini + pd.to_timedelta(fila["wall_time_s"], unit="s") if pd.notna(ini) else pd.NaT
            avg_w, wh = integrar_potencia(wpot, ini, fin)
            if not np.isnan(avg_w):
                df.at[i, "avg_power_w"] = round(avg_w, 3)
                df.at[i, "energy_wh"] = round(wh, 6)

    df.to_csv(args.salida, index=False)
    n_ok = df["speedup"].notna().sum()
    print(f"[ok] {args.salida}: {len(df)} filas, speedup calculado en {n_ok}")
    print("     (repetition==0 se conserva; plot_results.py la filtra)")


if __name__ == "__main__":
    main()
