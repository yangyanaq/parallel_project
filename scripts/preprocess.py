#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 preprocess.py  --  Construye los sub-datasets del experimento (tesis K-Means)
=============================================================================
Toma los CSV mensuales crudos del NYC Yellow Taxi (los de Kaggle, ~1.8 GB c/u),
extrae SOLO las 4 coordenadas del clustering, limpia, y genera tres muestras
ANIDADAS y deterministas:  100K  subconjunto de  1M  subconjunto de  10M.

Cada muestra se guarda en dos formatos:
  * .bin  -> binario compacto que se copia a los clústeres (carga sin parseo)
  * .csv  -> legible, para inspección y para el análisis exploratorio (Cap. 5)
  * .meta.json -> metadatos (n, d, orden de columnas, semilla, limpieza)

Este script corre UNA VEZ en la workstation (la del RTX), NO en los Raspberry Pi.

---------------------------------------------------------------------------
 FORMATO BINARIO  (debe coincidir EXACTO con el cargador en C, io_dataset)
---------------------------------------------------------------------------
  * float64 little-endian ('<f8'), sin cabecera.
  * Layout SoA (Structure of Arrays), d = 4, en este orden de columnas:
        [ pickup_latitude  * n ]
        [ pickup_longitude * n ]
        [ dropoff_latitude * n ]
        [ dropoff_longitude* n ]
  * El número de filas se deduce del tamaño del archivo:
        n = filesize_bytes / (4 * 8)
  * Indexado en C:  datos[dim * n + i]   (dim en 0..3, i en 0..n-1)

USO:
    python3 preprocess.py --inputs yellow_tripdata_2015-01.csv
    python3 preprocess.py --inputs 2015-01.csv 2016-01.csv --outdir data --seed 42

RECOMENDACIÓN: usa solo 2015-01 (tiene ~12.7M filas, suficiente para 10M) para
mantener coherencia temporal con el análisis exploratorio del Capítulo 5.
=============================================================================
"""
import argparse, os, sys, json, time
import numpy as np
import pandas as pd

# Orden LÓGICO de las 4 dimensiones (d=4). OJO: en el CSV crudo el orden es
# lon,lat; aquí las reordenamos a lat,lon para origen y destino.
COLS = ["pickup_latitude", "pickup_longitude",
        "dropoff_latitude", "dropoff_longitude"]

NYC_BBOX = {"lon_min": -74.27, "lon_max": -73.68,
            "lat_min":  40.49, "lat_max":  40.92}


def detectar_columnas(cols):
    """Mapea nombres reales del CSV -> nombres canónicos COLS (case-insensitive)."""
    lower = {c.lower().strip(): c for c in cols}
    alias = {
        "pickup_latitude":   ["pickup_latitude", "pickup_lat", "start_lat"],
        "pickup_longitude":  ["pickup_longitude", "pickup_lon", "pickup_lng"],
        "dropoff_latitude":  ["dropoff_latitude", "dropoff_lat", "end_lat"],
        "dropoff_longitude": ["dropoff_longitude", "dropoff_lon", "dropoff_lng"],
    }
    mapa = {}
    for canon, ops in alias.items():
        for o in ops:
            if o in lower:
                mapa[lower[o]] = canon
                break
    return mapa


def cargar_y_limpiar(inputs, aplicar_bbox):
    partes = []
    total_bruto = 0
    for path in inputs:
        if not os.path.exists(path):
            print(f"ERROR: no existe {path}"); sys.exit(1)
        print(f"[carga] {path} ...")
        cab = pd.read_csv(path, nrows=5)
        mapa = detectar_columnas(cab.columns)
        if len(mapa) < 4:
            print("ERROR: no se detectaron las 4 coordenadas. Columnas:", list(cab.columns))
            sys.exit(1)
        df = pd.read_csv(path, usecols=list(mapa.keys()))
        df = df.rename(columns=mapa)[COLS].astype(np.float64)
        total_bruto += len(df)
        partes.append(df)
    df = pd.concat(partes, ignore_index=True) if len(partes) > 1 else partes[0]

    n0 = len(df)
    df = df.dropna()
    if aplicar_bbox:
        b = NYC_BBOX
        m = (df["pickup_longitude"].between(b["lon_min"], b["lon_max"]) &
             df["pickup_latitude"].between(b["lat_min"], b["lat_max"]) &
             df["dropoff_longitude"].between(b["lon_min"], b["lon_max"]) &
             df["dropoff_latitude"].between(b["lat_min"], b["lat_max"]))
        df = df[m]
    removidas = n0 - len(df)
    print(f"[limpieza] brutas={n0:,}  removidas={removidas:,} "
          f"({100*removidas/max(n0,1):.2f}%)  validas={len(df):,}")
    return df, total_bruto, removidas


def tag_de(n):
    if n % 1_000_000 == 0 and n >= 1_000_000: return f"{n//1_000_000}M"
    if n % 1_000 == 0 and n >= 1_000:         return f"{n//1_000}K"
    return str(n)


def escribir(a, size, outdir, escribir_csv, meta_base):
    """a: array (size x 4) en orden COLS. Escribe .bin (SoA), .csv y .meta.json"""
    tag = tag_de(size)
    base = os.path.join(outdir, f"nyc_{tag}")

    # --- binario SoA float64 little-endian ---
    soa = np.ascontiguousarray(a.T).astype("<f8")   # (4, size) -> plano = SoA
    soa.tofile(base + ".bin")

    # --- csv (opcional para los grandes) ---
    if escribir_csv:
        pd.DataFrame(a, columns=COLS).to_csv(base + ".csv", index=False,
                                             float_format="%.6f")

    # --- metadatos ---
    meta = dict(meta_base)
    meta.update(n=int(size), d=4, columnas=COLS, tag=tag,
                bin_bytes=os.path.getsize(base + ".bin"),
                layout="SoA float64 little-endian, sin cabecera, n=filesize/(4*8)")
    with open(base + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    mb = meta["bin_bytes"] / 1e6
    print(f"  nyc_{tag}: {size:,} filas  ->  .bin {mb:,.1f} MB"
          + ("  + .csv" if escribir_csv else "  (sin csv)"))


def main():
    ap = argparse.ArgumentParser(description="Genera los sub-datasets del experimento.")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Uno o más CSV mensuales crudos del NYC Yellow Taxi.")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--sizes", default="100000,1000000,10000000",
                    help="Tamaños a generar, separados por coma.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-bbox", action="store_true", help="No filtrar por caja de NYC.")
    ap.add_argument("--csv-max", type=int, default=1_000_000,
                    help="Escribe .csv solo para tamaños <= este valor (los grandes solo .bin).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sizes = sorted(int(s) for s in args.sizes.split(","))

    t0 = time.time()
    df, bruto, removidas = cargar_y_limpiar(args.inputs, not args.no_bbox)

    # Baraja UNA vez con semilla fija -> las muestras quedan anidadas al tomar prefijos.
    print(f"[muestreo] barajando con semilla {args.seed} (muestras anidadas)")
    idx = np.random.RandomState(args.seed).permutation(len(df))
    a_full = df.to_numpy()[idx]        # (N, 4) barajado, orden COLS

    if a_full.shape[0] < max(sizes):
        print(f"AVISO: solo hay {a_full.shape[0]:,} filas válidas; "
              f"el tamaño {max(sizes):,} se recortará. Agrega más meses si necesitas más.")

    meta_base = dict(fuente=[os.path.basename(p) for p in args.inputs],
                     semilla=args.seed, filas_brutas=int(bruto),
                     filas_removidas=int(removidas), bbox=(not args.no_bbox),
                     anidadas=True)

    print("[escritura]")
    for size in sizes:
        s = min(size, a_full.shape[0])
        escribir(a_full[:s], s, args.outdir, s <= args.csv_max, meta_base)

    print(f"\nListo en {time.time()-t0:.1f}s. Archivos en: {os.path.abspath(args.outdir)}")
    print("A los clústeres solo copia los .bin (y el .meta.json). "
          "El análisis exploratorio (Cap. 5) corre sobre nyc_100K.csv.")


if __name__ == "__main__":
    main()
