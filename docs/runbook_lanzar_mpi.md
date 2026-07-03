# Runbook — lanzar la variante MPI (clúster RPi)

> Fase 2. Cómo compilar y lanzar `kmeans_rpi` en los 5 nodos sin el crash SIGILL.

## El gotcha (por qué esto importa)

Los **workers RPi `.11`–`.14` NO clonan el repo** — solo montan el NFS.
Solo el maestro `.10` tiene `~/parallel_project`. Cuando `mpirun` lanza un rank
en un worker, ejecuta la ruta que le pasas; si es relativa (`./bin/kmeans_rpi`)
Open MPI la resuelve desde el HOME del worker, **donde no hay binario** → el rank
crashea con **SIGILL / exit 132** (o corre uno viejo). Diagnóstico de la sesión F2:
np=1/2/4 funcionaban (todo intra-nodo en `.10`), np≥8 crasheaban al tocar workers.

**Regla:** el binario MPI debe estar en la **misma ruta absoluta en todos los
nodos**. La forma limpia con este clúster: ponerlo en el NFS que todos montan.

## Flujo correcto

```bash
# en el maestro .10
cd ~/parallel_project
git pull                      # traer el código
make install-rpi              # compila (make rpi) + copia a /home/cris/kmeans_share/bin/

# lanzar SIEMPRE con la ruta NFS absoluta, no ./bin/...
NFS=/home/cris/kmeans_share
mpirun --hostfile hosts_rpi -np 20 \
       $NFS/bin/kmeans_rpi \
       --data $NFS/data/nyc_1M.bin --out /tmp/bench_mpi.csv --quiet
```

`hosts_rpi` (5 líneas, `slots=4`) incluye al maestro `.10` → 20 núcleos.
Barrido de la tesis: `-np` ∈ {1, 2, 4, 8, 16, 20}.

## Verificación rápida (paralelismo real)

Sin `--quiet`, cada rank imprime `host=<nodo> n_local=<filas>`. Con np=20 deben
verse los **5 nodos** (worker1=`.10`, worker2/3/4 y `cris`), 4 ranks cada uno,
`n_local ≈ n/20`. Si todos los ranks caen en un solo host, el hostfile no se está
aplicando.

## Resultado esperado (no es bug)

En 1M el `wall_time` **baja de np=1 a np=4** (~4.9s → ~1.24s, escala casi lineal
intra-nodo) y luego **sube de np=8 a np=20** porque el `comm_time` (200 Allreduce
sobre Gigabit) domina cuando el cómputo por rank es chico. Es la degradación por
overhead de comunicación que la tesis busca demostrar (PLAN §3.3, checklist §15).
El punto óptimo se corre hacia más procesos conforme crece el dataset (en 10M np=4
sigue ganando claramente).
