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
    # Las curvas son numéricamente idénticas y se tapan entre sí: se dibujan
    # con anchos y trazos distintos (sólido/discontinuo/punteado) para que
    # las tres queden visibles una sobre otra.
    ESTILO = {"rpi5": ("-", 3.5), "jetson": ("--", 2.0), "rtx4070ti": (":", 1.4)}
    orden = [p for p in ["rpi5", "jetson", "rtx4070ti"] if p in set(c["platform"])]
    orden += [p for p in c["platform"].unique() if p not in orden]
    for plat in orden:
        sub = c[c["platform"] == plat].sort_values("iteration")
        col = COLOR.get(plat, None)
        ls, lw = ESTILO.get(plat, ("-", 2.0))
        ax.plot(sub["iteration"], sub["wcss"], color=col, ls=ls, lw=lw,
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


def g10_speedup_computo_vs_total(df, figs, pdf):
    """10. Speedup del cómputo puro vs speedup total (RPi, 10M).

    Aísla al culpable de la degradación: el speedup calculado SOLO con
    compute_time_s escala casi ideal (19.9x con p=20, eficiencia 99%),
    mientras el total colapsa a 1.9x. La brecha vertical entre ambas
    curvas ES el costo de sincronización. Análisis del Cap. 7 (frontera
    del nodo); ver docs/analisis_extras_para_redaccion.md."""
    tam = 10_000_000
    seq = df[(df["variant"] == "seq") & (df["dataset_rows"] == tam)]
    mpi = df[(df["platform"] == "rpi5") & (df["variant"] == "mpi") &
             (df["dataset_rows"] == tam)]
    if seq.empty or mpi.empty:
        print("  [skip] g10: sin datos seq/mpi de 10M"); return
    t1 = seq["wall_time_s"].mean()

    # speedup por corrida (rep a rep) para tener barras de error honestas
    g_tot = mpi.assign(s=t1 / mpi["wall_time_s"]) \
               .groupby("num_procs")["s"].agg(["mean", "std"])
    g_cmp = mpi.assign(s=t1 / mpi["compute_time_s"]) \
               .groupby("num_procs")["s"].agg(["mean", "std"])

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    lim = int(g_tot.index.max())
    ax.plot([1, lim], [1, lim], "--", color=COLOR["ideal"], label="ideal (y=x)")
    ax.errorbar(g_cmp.index, g_cmp["mean"], yerr=g_cmp["std"].fillna(0),
                marker="s", capsize=3, color=COLOR["rtx4070ti"],
                label="cómputo puro (sin sincronización)")
    ax.errorbar(g_tot.index, g_tot["mean"], yerr=g_tot["std"].fillna(0),
                marker="o", capsize=3, color=COLOR["rpi5"],
                label="total (con sincronización)")
    # anotar la brecha en p máximo
    ax.annotate("", xy=(lim, g_tot["mean"].iloc[-1]),
                xytext=(lim, g_cmp["mean"].iloc[-1]),
                arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.2))
    ax.text(lim - 0.4, (g_tot["mean"].iloc[-1] + g_cmp["mean"].iloc[-1]) / 2,
            "costo de\nsincronización", ha="right", va="center",
            fontsize=9, color="#555555")
    ax.set_xlabel("nº de procesos"); ax.set_ylabel("speedup (T₁ / Tₚ)")
    ax.set_title("Speedup del cómputo puro vs total — clúster RPi 5, 10M")
    ax.set_xticks(sorted(g_tot.index)); ax.legend(loc="upper left")
    guardar(fig, figs, "10_speedup_computo_vs_total", pdf)


def g11_firma_desincronizacion(df, figs, pdf):
    """11. Firma de la desincronización: comm por iteración vs tamaño, a p fijo.

    El Allreduce mueve 180 bytes por iteración, independientes de n: si el
    costo fuera solo latencia de red, las líneas serían PLANAS. Crecen con
    el tamaño del dataset porque más cómputo desalinea más a los procesos
    y la colectiva espera al último (desincronización). Evidencia del
    argumento del Cap. 7 §3; ver docs/analisis_extras_para_redaccion.md."""
    mpi = df[(df["platform"] == "rpi5") & (df["variant"] == "mpi") &
             (df["num_procs"] >= 8)]                 # solo configs multinodo
    if mpi.empty:
        print("  [skip] g11: sin datos multinodo"); return
    iters = int(mpi["iterations"].iloc[0])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    marcas = {8: "o", 16: "s", 20: "^"}
    azules = {8: "#7BAFD4", 16: "#3D85C6", 20: "#0B5394"}   # secuencia de un solo tono
    for p in sorted(mpi["num_procs"].unique()):
        sub = mpi[mpi["num_procs"] == p]
        # ms de comunicación por iteración (cada iteración = 2 Allreduce)
        g = (sub.assign(ms=sub["comm_time_s"] * 1000.0 / iters)
                .groupby("dataset_rows")["ms"].agg(["mean", "std"])
                .reindex(TAMS).dropna())
        ax.errorbar(g.index, g["mean"], yerr=g["std"].fillna(0),
                    marker=marcas.get(p, "o"), capsize=3,
                    color=azules.get(p), label=f"p = {p}")
    ax.set_xscale("log")
    ax.set_xticks(TAMS); ax.set_xticklabels([TAM_LBL[t] for t in TAMS])
    ax.set_xlabel("tamaño del dataset")
    ax.set_ylabel("comunicación por iteración (ms)")
    ax.set_title("Firma de la desincronización — clúster RPi 5 multinodo")
    ax.text(0.02, 0.96, "mensaje constante: 180 B/iteración\n"
            "si fuera solo latencia de red, las líneas serían planas",
            transform=ax.transAxes, fontsize=8.5, va="top", color="#555555")
    ax.legend(title="procesos", loc="center left")
    guardar(fig, figs, "11_firma_desincronizacion", pdf)


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
    # figuras de análisis adicionales (Cap. 7: aislar el costo de sincronización)
    g10_speedup_computo_vs_total(df, args.figs, args.pdf)
    g11_firma_desincronizacion(df, args.figs, args.pdf)
    print(f"\nListo. PNG (200 dpi) en {os.path.abspath(args.figs)}")


if __name__ == "__main__":
    main()
