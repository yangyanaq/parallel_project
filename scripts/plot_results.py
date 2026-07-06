#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 plot_results.py — genera los 9 gráficos de la tesis (CLAUDE.md §12)
=============================================================================
Lee results/benchmark_final.csv (o benchmark.csv) y produce PNG 200 dpi (y PDF
opcional) en results/figs/, para los Capítulos 6 y 7.

Principios de visualización (skill dataviz, adaptados a matplotlib para LaTeX):
  - la FORMA la elige el trabajo del dato (speedup->línea con ideal; energía->
    barras; convergencia->línea; costo/$->barras);
  - paleta CATEGÓRICA en orden FIJO por entidad (Okabe-Ito, colorblind-safe),
    nunca cicla ni repinta según el filtro;
  - un solo eje; leyenda siempre presente; grid recesivo; marcas finas;
  - media ± desv. est. (barras de error) sobre las repeticiones, filtrando la
    corrida en frío (repetition == 0).

USO:
  python plot_results.py [--in results/benchmark_final.csv] [--figs results/figs]
                         [--conv results/wcss_convergence.csv] [--pdf]
=============================================================================
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")            # sin display (corre en el maestro/PC)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- paleta categórica Okabe-Ito (colorblind-safe), orden FIJO por plataforma ---
# La identidad sigue a la entidad, no al orden de aparición ni al conteo de series.
COLOR = {
    "rpi5":      "#0072B2",   # azul
    "jetson":    "#E69F00",   # naranja
    "rtx4070ti": "#009E73",   # verde
    "ideal":     "#999999",   # gris (línea ideal / referencias)
}
ETIQUETA = {"rpi5": "RPi 5 (MPI)", "jetson": "Jetson (MPI+CUDA)",
            "rtx4070ti": "RTX 4070 Ti (CUDA)", "seq": "Secuencial"}
# Orden fijo de tamaños y su etiqueta legible
TAMS = [100000, 1000000, 10000000]
TAM_LBL = {100000: "100K", 1000000: "1M", 10000000: "10M"}

# Costo del hardware (USD) para rendimiento/dólar (Cap 7). Ajustable si cambia.
COSTO_USD = {
    "rpi5":      5 * 80,      # 5 nodos RPi 5 (~$80 c/u)
    "jetson":    3 * 99,      # 3 Jetson Nano (~$99 c/u)
    "rtx4070ti": 800,         # 1 estación GPU (aprox. la GPU)
}

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 7,
})


def guardar(fig, figs, nombre, pdf):
    fig.tight_layout()
    fig.savefig(os.path.join(figs, nombre + ".png"), bbox_inches="tight")
    if pdf:
        fig.savefig(os.path.join(figs, nombre + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {nombre}.png")


def agg(df, por, val):
    """media y desv. est. de `val` agrupando por `por` (lista de columnas)."""
    g = df.groupby(por)[val].agg(["mean", "std"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    return g


# ------------------------- gráficos -------------------------

def g1_speedup(df, figs, pdf):
    """1. Speedup vs nº de procesos (por tamaño), con línea ideal y=x."""
    mpi = df[(df["platform"] == "rpi5") & (df["variant"] == "mpi")]
    if mpi.empty:
        print("  [skip] g1 speedup: sin datos rpi/mpi"); return
    fig, axes = plt.subplots(1, len(TAMS), figsize=(4 * len(TAMS), 3.6), sharey=True)
    if len(TAMS) == 1:
        axes = [axes]
    for ax, tam in zip(axes, TAMS):
        sub = mpi[mpi["dataset_rows"] == tam]
        if sub.empty:
            ax.set_visible(False); continue
        s = agg(sub, ["num_procs"], "speedup")
        ax.errorbar(s["num_procs"], s["mean"], yerr=s["std"], marker="o",
                    color=COLOR["rpi5"], capsize=3, label="RPi 5 medido")
        lim = s["num_procs"].max()
        ax.plot([1, lim], [1, lim], "--", color=COLOR["ideal"], label="ideal (y=x)")
        ax.set_title(TAM_LBL[tam]); ax.set_xlabel("nº de procesos")
    axes[0].set_ylabel("speedup (T₁ / Tₚ)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Speedup vs nº de procesos — clúster RPi 5", y=1.02)
    guardar(fig, figs, "01_speedup_vs_procs", pdf)


def g2_eficiencia(df, figs, pdf):
    """2. Eficiencia paralela vs nº de procesos."""
    mpi = df[(df["platform"] == "rpi5") & (df["variant"] == "mpi")]
    if mpi.empty:
        print("  [skip] g2 eficiencia: sin datos"); return
    fig, ax = plt.subplots(figsize=(6, 4))
    for tam in TAMS:
        sub = mpi[mpi["dataset_rows"] == tam]
        if sub.empty:
            continue
        e = agg(sub, ["num_procs"], "efficiency")
        ax.errorbar(e["num_procs"], e["mean"], yerr=e["std"], marker="o",
                    capsize=3, label=TAM_LBL[tam])
    ax.axhline(1.0, ls="--", color=COLOR["ideal"], label="ideal (1.0)")
    ax.set_xlabel("nº de procesos"); ax.set_ylabel("eficiencia (speedup / p)")
    ax.set_title("Eficiencia paralela — clúster RPi 5"); ax.legend(title="tamaño")
    guardar(fig, figs, "02_eficiencia_vs_procs", pdf)


def g3_tiempo_vs_tam(df, figs, pdf):
    """3. Tiempo vs tamaño (log-log), mejor config de cada plataforma."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for plat in ["rpi5", "jetson", "rtx4070ti"]:
        sub = df[df["platform"] == plat]
        if plat == "rpi5":
            sub = sub[sub["variant"] == "mpi"]
        if sub.empty:
            continue
        # mejor (mínimo) tiempo por tamaño (sobre configs y reps)
        best = sub.groupby("dataset_rows")["wall_time_s"].min().reindex(TAMS).dropna()
        ax.plot(best.index, best.values, marker="o", color=COLOR[plat],
                label=ETIQUETA[plat])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("nº de puntos"); ax.set_ylabel("tiempo (s)")
    ax.set_title("Tiempo de ejecución vs tamaño (log-log)")
    ax.set_xticks(TAMS); ax.set_xticklabels([TAM_LBL[t] for t in TAMS])
    ax.legend()
    guardar(fig, figs, "03_tiempo_vs_tamano", pdf)


def g4_descomposicion(df, figs, pdf):
    """4. Cómputo vs comunicación vs transferencia (barras apiladas)."""
    # una barra por (plataforma, tamaño) en su mejor config; muestra overhead MPI
    filas = []
    for plat in ["rpi5", "jetson", "rtx4070ti"]:
        sub = df[df["platform"] == plat]
        if plat == "rpi5":
            sub = sub[sub["variant"] == "mpi"]
        for tam in TAMS:
            s = sub[sub["dataset_rows"] == tam]
            if s.empty:
                continue
            # config de menor wall
            idx = s.groupby("num_procs")["wall_time_s"].mean().idxmin()
            s = s[s["num_procs"] == idx]
            filas.append((f"{ETIQUETA[plat]}\n{TAM_LBL[tam]}",
                          s["compute_time_s"].mean(), s["comm_time_s"].mean(),
                          s["transfer_time_s"].mean()))
    if not filas:
        print("  [skip] g4 descomposición: sin datos"); return
    lbl = [f[0] for f in filas]
    comp = np.array([f[1] for f in filas]); comm = np.array([f[2] for f in filas])
    trans = np.array([f[3] for f in filas])
    fig, ax = plt.subplots(figsize=(max(6, len(lbl) * 1.1), 4.2))
    x = np.arange(len(lbl))
    ax.bar(x, comp, color=COLOR["rtx4070ti"], label="cómputo")
    ax.bar(x, comm, bottom=comp, color=COLOR["rpi5"], label="comunicación")
    ax.bar(x, trans, bottom=comp + comm, color=COLOR["jetson"], label="transferencia")
    ax.set_xticks(x); ax.set_xticklabels(lbl, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("tiempo (s)"); ax.set_title("Descomposición del tiempo (mejor config)")
    ax.legend()
    guardar(fig, figs, "04_descomposicion_tiempo", pdf)


def _barras_por_plataforma_tam(df, val, titulo, ylabel, nombre, figs, pdf, reducir="mean"):
    """helper: barras agrupadas plataforma × tamaño para una métrica."""
    plats = ["rpi5", "jetson", "rtx4070ti"]
    x = np.arange(len(TAMS)); w = 0.25
    fig, ax = plt.subplots(figsize=(6.5, 4))
    algo = False
    for i, plat in enumerate(plats):
        sub = df[df["platform"] == plat]
        if plat == "rpi5":
            sub = sub[sub["variant"] == "mpi"]
        vals, errs = [], []
        for tam in TAMS:
            s = sub[sub["dataset_rows"] == tam][val].dropna()
            if s.empty:
                vals.append(np.nan); errs.append(0)
            else:
                vals.append(getattr(s, reducir)()); errs.append(s.std() if len(s) > 1 else 0)
        if np.all(np.isnan(vals)):
            continue
        algo = True
        ax.bar(x + (i - 1) * w, vals, w, yerr=errs, capsize=2,
               color=COLOR[plat], label=ETIQUETA[plat])
    if not algo:
        print(f"  [skip] {nombre}: sin datos"); plt.close(fig); return
    ax.set_xticks(x); ax.set_xticklabels([TAM_LBL[t] for t in TAMS])
    ax.set_xlabel("tamaño"); ax.set_ylabel(ylabel); ax.set_title(titulo); ax.legend()
    guardar(fig, figs, nombre, pdf)


def g5_energia(df, figs, pdf):
    """5. Consumo de energía (Wh) por plataforma y tamaño."""
    _barras_por_plataforma_tam(df, "energy_wh", "Consumo de energía",
                               "energía (Wh)", "05_energia_wh", figs, pdf)


def g6_throughput_por_watt(df, figs, pdf):
    """6. Rendimiento por vatio (throughput / potencia)."""
    d = df.copy()
    d["tput_por_w"] = d["throughput_pts_s"] / d["avg_power_w"]
    _barras_por_plataforma_tam(d, "tput_por_w", "Rendimiento por vatio",
                               "puntos/s por W", "06_throughput_por_watt", figs, pdf)


def g7_throughput_por_dolar(df, figs, pdf):
    """7. Rendimiento por dólar (throughput / costo hardware) — métrica central."""
    d = df.copy()
    d["tput_por_usd"] = d.apply(
        lambda r: r["throughput_pts_s"] / COSTO_USD.get(r["platform"], np.nan), axis=1)
    _barras_por_plataforma_tam(d, "tput_por_usd", "Rendimiento por dólar (costo-eficiencia)",
                               "puntos/s por USD", "07_throughput_por_dolar", figs, pdf)


def g8_convergencia(conv_path, figs, pdf):
    """8. Convergencia del WCSS por iteración (las plataformas convergen igual)."""
    if not conv_path or not os.path.exists(conv_path):
        print("  [skip] g8 convergencia: sin wcss_convergence.csv"); return
    c = pd.read_csv(conv_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    for plat in c["platform"].unique():
        sub = c[c["platform"] == plat].sort_values("iteration")
        col = COLOR.get(plat, None)
        ax.plot(sub["iteration"], sub["wcss"], color=col,
                label=ETIQUETA.get(plat, plat))
    ax.set_xlabel("iteración"); ax.set_ylabel("WCSS")
    ax.set_title("Convergencia del WCSS por iteración"); ax.legend()
    guardar(fig, figs, "08_convergencia_wcss", pdf)


def g9_throughput_vs_tam(df, figs, pdf):
    """9. Throughput (puntos/s) vs tamaño."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for plat in ["rpi5", "jetson", "rtx4070ti"]:
        sub = df[df["platform"] == plat]
        if plat == "rpi5":
            sub = sub[sub["variant"] == "mpi"]
        if sub.empty:
            continue
        best = sub.groupby("dataset_rows")["throughput_pts_s"].max().reindex(TAMS).dropna()
        ax.plot(best.index, best.values, marker="o", color=COLOR[plat],
                label=ETIQUETA[plat])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("nº de puntos"); ax.set_ylabel("throughput (puntos/s)")
    ax.set_title("Throughput vs tamaño")
    ax.set_xticks(TAMS); ax.set_xticklabels([TAM_LBL[t] for t in TAMS]); ax.legend()
    guardar(fig, figs, "09_throughput_vs_tamano", pdf)


def main():
    ap = argparse.ArgumentParser(description="Genera los 9 gráficos de la tesis (§12).")
    ap.add_argument("--in", dest="entrada", default="results/benchmark_final.csv")
    ap.add_argument("--figs", default="results/figs")
    ap.add_argument("--conv", default="results/wcss_convergence.csv")
    ap.add_argument("--pdf", action="store_true", help="exportar también PDF")
    args = ap.parse_args()

    if not os.path.exists(args.entrada):
        # fallback al crudo si no se ha post-procesado
        alt = "results/benchmark.csv"
        if os.path.exists(alt):
            print(f"[aviso] {args.entrada} no existe; uso {alt} (sin speedup/energía completos)")
            args.entrada = alt
        else:
            sys.exit(f"ERROR: no existe {args.entrada} ni results/benchmark.csv")
    os.makedirs(args.figs, exist_ok=True)

    df = pd.read_csv(args.entrada)
    # filtrar corrida en frío (repetition == 0) — política documentada
    n0 = len(df)
    df = df[df["repetition"] != 0]
    print(f"[datos] {n0} filas, {len(df)} tras filtrar rep 0. "
          f"Plataformas: {sorted(df['platform'].unique())}")

    print("[figuras]")
    g1_speedup(df, args.figs, args.pdf)
    g2_eficiencia(df, args.figs, args.pdf)
    g3_tiempo_vs_tam(df, args.figs, args.pdf)
    g4_descomposicion(df, args.figs, args.pdf)
    g5_energia(df, args.figs, args.pdf)
    g6_throughput_por_watt(df, args.figs, args.pdf)
    g7_throughput_por_dolar(df, args.figs, args.pdf)
    g8_convergencia(args.conv, args.figs, args.pdf)
    g9_throughput_vs_tam(df, args.figs, args.pdf)
    print(f"\nListo. PNG (200 dpi) en {os.path.abspath(args.figs)}")


if __name__ == "__main__":
    main()
