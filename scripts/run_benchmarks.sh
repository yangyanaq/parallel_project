#!/bin/bash
# run_benchmarks.sh — orquesta la matriz de benchmarks de los CLÚSTERES.
# Corre en el maestro RPi .10. Recorre: seq (T_1), RPi MPI y Jetson híbrido.
# La RTX va aparte (scripts/run_benchmarks_rtx.ps1). Ver PLAN §6, CLAUDE.md §11.
#
# REANUDABLE: antes de cada corrida comprueba si la fila ya está en el CSV
# (misma platform,variant,rows,np,rep) y la salta. Así se puede cortar y seguir.
#
# El binario deja speedup/efficiency/energía en NaN; los completa
# scripts/aggregate_power.py al post-procesar (necesita el T_1 promedio y los
# logs de potencia). Este script SÍ arranca el muestreo INA3221 en las Jetson.
#
# Uso:
#   ./run_benchmarks.sh [--reps N] [--sizes 100K,1M,10M] [--only seq|rpi|jetson]
#                       [--out results/benchmark.csv] [--dry-run]

set -u

# ---------------- config ----------------
REPO=~/parallel_project
NFS=/home/cris/kmeans_share
DATA=$NFS/data
BIN_RPI=$NFS/bin/kmeans_rpi
BIN_JET=$NFS/bin/kmeans_jetson
HF_RPI=$REPO/hosts_rpi
HF_JET=$REPO/hosts_jetson
MPIRUN=/usr/local/bin/mpirun
MCA_JET="--mca btl_tcp_if_include eth0 --mca oob_tcp_if_include eth0"
JETSONS="192.168.77.21 192.168.77.22 192.168.77.23"
POWER_DIR=$NFS/power          # logs INA3221 por corrida (alineables por timestamp)

OUT=results/benchmark.csv
REPS=5                        # reps "en caliente" (1..REPS); además la rep 0 en frío
SIZES="100K 1M 10M"
NP_RPI="1 2 4 8 16 20"
NP_JET="1 2 3"
ONLY=""
DRY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --reps)  REPS="$2"; shift 2;;
    --sizes) SIZES="$(echo "$2" | tr ',' ' ')"; shift 2;;
    --only)  ONLY="$2"; shift 2;;
    --out)   OUT="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    *) echo "arg desconocido: $1"; exit 1;;
  esac
done

cd "$REPO" || exit 1
mkdir -p "$(dirname "$OUT")" "$POWER_DIR"

# fila ya presente? (platform,variant,dataset_rows,num_procs,repetition)
# columnas: 1=platform 2=variant 3=rows 4=num_procs 6=repetition
ya_hecha() {
  local plat="$1" var="$2" rows="$3" np="$4" rep="$5"
  [ -f "$OUT" ] || return 1
  awk -F, -v p="$plat" -v v="$var" -v r="$rows" -v n="$np" -v e="$rep" \
    'NR>1 && $1==p && $2==v && $3==r && $4==n && $6==e {found=1} END{exit !found}' "$OUT"
}

n_de_tam() { case "$1" in 100K) echo 100000;; 1M) echo 1000000;; 10M) echo 10000000;; *) echo 0;; esac; }

# --------- energía Jetson: arranca/para el muestreo INA3221 en las 3 ---------
power_start() {
  local tag="$1"
  for ip in $JETSONS; do
    ssh -o BatchMode=yes nano@$ip \
      "nohup bash $REPO/scripts/power_log_tegra.sh $POWER_DIR/tegra_${ip##*.}_${tag}.csv 100 >/dev/null 2>&1 & echo \$! > /tmp/power_pid" &
  done
  wait
}
power_stop() {
  for ip in $JETSONS; do
    ssh -o BatchMode=yes nano@$ip "kill \$(cat /tmp/power_pid 2>/dev/null) 2>/dev/null" &
  done
  wait
}

correr() {   # correr <desc> <cmd...>
  local desc="$1"; shift
  echo ">>> $desc"
  if [ "$DRY" = 1 ]; then echo "    [dry-run] $*"; return 0; fi
  "$@"
}

# ============================ SEQ (T_1) ============================
bench_seq() {
  for tam in $SIZES; do
    local rows; rows=$(n_de_tam "$tam")
    for rep in $(seq 0 "$REPS"); do
      if ya_hecha "rpi5" "seq" "$rows" 1 "$rep"; then
        echo "skip seq $tam rep=$rep (ya)"; continue; fi
      correr "seq $tam rep=$rep" \
        ./bin/kmeans_seq --data "$DATA/nyc_${tam}.bin" --out "$OUT" --rep "$rep" --quiet
    done
  done
}

# ============================ RPi MPI ============================
bench_rpi() {
  for tam in $SIZES; do
    local rows; rows=$(n_de_tam "$tam")
    for np in $NP_RPI; do
      for rep in $(seq 0 "$REPS"); do
        if ya_hecha "rpi5" "mpi" "$rows" "$np" "$rep"; then
          echo "skip rpi $tam np=$np rep=$rep (ya)"; continue; fi
        correr "rpi $tam np=$np rep=$rep" \
          "$MPIRUN" --hostfile "$HF_RPI" -np "$np" "$BIN_RPI" \
            --data "$DATA/nyc_${tam}.bin" --out "$OUT" --rep "$rep" --quiet
      done
    done
  done
}

# ============================ Jetson híbrido ============================
bench_jetson() {
  for tam in $SIZES; do
    local rows; rows=$(n_de_tam "$tam")
    for np in $NP_JET; do
      for rep in $(seq 0 "$REPS"); do
        if ya_hecha "jetson" "hybrid" "$rows" "$np" "$rep"; then
          echo "skip jetson $tam np=$np rep=$rep (ya)"; continue; fi
        [ "$DRY" = 0 ] && power_start "${tam}_np${np}_r${rep}"
        correr "jetson $tam np=$np rep=$rep" \
          "$MPIRUN" --hostfile "$HF_JET" -np "$np" $MCA_JET "$BIN_JET" \
            --data "$DATA/nyc_${tam}.bin" --out "$OUT" --rep "$rep" --quiet
        [ "$DRY" = 0 ] && power_stop
      done
    done
  done
}

echo "=== run_benchmarks.sh  reps=$REPS sizes='$SIZES' only='${ONLY:-todo}' out=$OUT ==="
case "$ONLY" in
  seq)    bench_seq;;
  rpi)    bench_rpi;;
  jetson) bench_jetson;;
  "")     bench_seq; bench_rpi; bench_jetson;;
  *) echo "only invalido: $ONLY"; exit 1;;
esac
echo "=== fin. Filas en $OUT: $(( $(wc -l < "$OUT" 2>/dev/null || echo 1) - 1 )) ==="
echo "Ahora: completar RTX (run_benchmarks_rtx.ps1) y post-procesar (aggregate_power.py)."
