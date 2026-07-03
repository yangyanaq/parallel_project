#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 validate_wcss.py — referencia numpy pura del K-Means (Lloyd) de la tesis
=============================================================================
NO es "el código" de la tesis: es el juez. Replica EXACTAMENTE la lógica de
src/common/ (mismo PRNG xorshift64*, misma init, mismas 100 iteraciones,
política de clúster vacío = conservar) y calcula el WCSS de referencia.
Puerta de salida F1: |wcss_C - wcss_numpy| dentro de 1e-4 en 100K y 1M.

USO:
    python scripts/validate_wcss.py --data data/nyc_100K.bin
    python scripts/validate_wcss.py --data data/nyc_1M.bin --expected <wcss_del_binario_C>

Con --expected compara y termina con exit code 0 (PASA) o 1 (FALLA).
La tolerancia acepta si la diferencia es < --tol ABSOLUTA o RELATIVA
(el orden de sumatoria difiere entre C secuencial y numpy; CLAUDE.md §4).
=============================================================================
"""
import argparse
import sys

import numpy as np

MASK64 = (1 << 64) - 1


class Xorshift64s:
    """Réplica bit a bit de src/common/rng.c — si se toca uno, se tocan ambos."""

    def __init__(self, semilla):
        self.estado = (semilla & MASK64) or 0x9E3779B97F4A7C15

    def next(self):
        x = self.estado
        x ^= x >> 12
        x ^= (x << 25) & MASK64
        x ^= x >> 27
        self.estado = x
        return (x * 0x2545F4914F6CDD1D) & MASK64

    def indice(self, n):
        return self.next() % n


def cargar_binario(ruta):
    """Contrato §5.1: float64 LE sin cabecera, SoA (4, n). Devuelve (n, 4)."""
    plano = np.fromfile(ruta, dtype="<f8")
    if plano.size == 0 or plano.size % 4 != 0:
        sys.exit(f"ERROR: {ruta} corrupto ({plano.size} doubles no es multiplo de 4)")
    n = plano.size // 4
    return plano.reshape(4, n).T.copy()          # (n, 4) contiguo


def inicializar_centroides(X, k, semilla):
    """Misma selección por rechazo de io_dataset.c (k índices distintos)."""
    rng = Xorshift64s(semilla)
    idx = []
    while len(idx) < k:
        cand = rng.indice(len(X))
        if cand not in idx:
            idx.append(cand)
    return X[idx].astype(np.float64), idx


def distancias2(X, mu):
    """(n, k) de distancias euclidianas al cuadrado (sin sqrt)."""
    d2 = np.empty((X.shape[0], mu.shape[0]), dtype=np.float64)
    for j in range(mu.shape[0]):                 # bucle en k: evita el (n,k,d)
        diff = X - mu[j]
        d2[:, j] = np.einsum("ij,ij->i", diff, diff)
    return d2


def lloyd(X, mu, iters, conv_out=None):
    k = mu.shape[0]
    mu = mu.copy()
    for it in range(1, iters + 1):
        asig = np.argmin(distancias2(X, mu), axis=1)   # empate -> índice menor, como C
        for j in range(k):
            miembros = X[asig == j]
            if len(miembros) > 0:
                mu[j] = miembros.sum(axis=0) / len(miembros)
            # clúster vacío: conserva el centroide anterior
        if conv_out is not None:
            conv_out.append((it, float(distancias2(X, mu).min(axis=1).sum())))
    wcss = float(distancias2(X, mu).min(axis=1).sum())  # G9: recomputa asignaciones
    return wcss, mu


def main():
    ap = argparse.ArgumentParser(description="Referencia numpy del K-Means (juez del WCSS).")
    ap.add_argument("--data", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--expected", type=float, default=None,
                    help="WCSS reportado por el binario C; compara con --tol.")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--dump-conv", default=None,
                    help="CSV opcional con la curva WCSS por iteración (referencia).")
    args = ap.parse_args()

    X = cargar_binario(args.data)
    n = len(X)

    # Sanity-check del cargador (CLAUDE.md §5): medias ~ (40.75, -73.97)
    medias = X.mean(axis=0)
    print(f"[datos] n={n:,}  medias= pickup({medias[0]:.3f}, {medias[1]:.3f}) "
          f"dropoff({medias[2]:.3f}, {medias[3]:.3f})  std_max={X.std(axis=0).max():.4f}")
    if not (40.4 < medias[0] < 41.0 and -74.3 < medias[1] < -73.6):
        sys.exit("ERROR: medias fuera de la caja de NYC — el cargador lee mal las columnas")

    mu0, idx = inicializar_centroides(X, args.k, args.seed)
    print(f"[init] semilla={args.seed}  indices={idx}")
    for j, fila in enumerate(mu0):
        print(f"  mu[{j}] = " + " ".join(f"{v:.17g}" for v in fila))

    conv = [] if args.dump_conv else None
    wcss, _ = lloyd(X, mu0, args.iters, conv)
    print(f"[numpy] wcss={wcss:.17g}  (k={args.k}, iters={args.iters})")

    if args.dump_conv:
        with open(args.dump_conv, "w") as f:
            f.write("iteration,wcss\n")
            for it, w in conv:
                f.write(f"{it},{w:.17g}\n")
        # la curva debe ser monótona no creciente (puerta F1)
        ws = [w for _, w in conv]
        mono = all(b <= a + 1e-12 for a, b in zip(ws, ws[1:]))
        print(f"[conv] {args.dump_conv} escrita; monotona_no_creciente={mono}")

    if args.expected is not None:
        dif = abs(wcss - args.expected)
        rel = dif / abs(wcss) if wcss != 0 else dif
        ok = dif < args.tol or rel < args.tol
        print(f"[compara] esperado={args.expected:.17g}  dif_abs={dif:.3g}  "
              f"dif_rel={rel:.3g}  tol={args.tol:g}  -> {'PASA' if ok else 'FALLA'}")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
