# Inventario real del clúster (Fase 0)

> Capturado 2026-07-02 vía SSH al maestro por ZeroTier. Fuente de verdad de rutas/versiones para compilar y lanzar.

## Acceso

| Qué | Valor |
|---|---|
| Maestro RPi — IP física | `192.168.77.10` (eth0); también `192.168.77.145` (wlan0) |
| Maestro RPi — IP ZeroTier | `10.144.101.22` (red `c7c8172af1a15747`, iface `zt5u43m7nw`) |
| Usuario / pass RPi (maestro y workers) | `cris` / ver `pass.txt` |
| Usuario / pass Jetson | `nano` / ver `pass.txt` |
| RTX 4090 | AnyDesk (acceso libre), en la LAN `192.168.77.x` |

**Nota:** el `hostname` del maestro es `worker1` (nombre heredado, no implica rol). El rol de maestro/lanzador lo cumple `.10`.

## Software del maestro (RPi 5)

| Componente | Versión / ruta |
|---|---|
| SO | Debian GNU/Linux 13 (trixie), kernel 6.12.75 `aarch64` |
| CPU | Cortex-A76, 4 cores, hasta 2.4 GHz |
| RAM | 4.0 GiB (+ 2 GiB swap) |
| Disco | 115 GB en `/` (101 GB libres) — sobra para los datos |
| **Open MPI** | **4.1.6** — `mpicc`/`mpirun` en `/usr/local/bin` |
| **gcc** | **14.2.0** (soporta C11 de sobra) |
| make, git, python3, scp, rsync | presentes |
| zerotier-cli | instalado y funcionando (aunque `command -v` no lo ve en el PATH no-interactivo) |

## NFS

- **Export:** `/home/cris/kmeans_share` → `192.168.77.0/24(rw,sync,no_subtree_check,no_root_squash)`.
- **Carpeta de datos:** `/home/cris/kmeans_share/data/` (creada; aquí van los `.bin`).
- El export es a la subred física `192.168.77.0/24`: los workers montan por esa red, no por ZeroTier.

## Rutas de trabajo

| Qué | Ruta |
|---|---|
| Repo clonado | `/home/cris/parallel_project` (en el HOME, NO en el NFS) |
| Datos (NFS) | `/home/cris/kmeans_share/data/nyc_{100K,1M,10M}.bin` |

**Decisión:** el repo se clona en `$HOME` de cada nodo que compile; los datos viven en el NFS compartido. Al lanzar, `--data /home/cris/kmeans_share/data/nyc_XXX.bin`.

## Estado de la red (todos los nodos UP, 2026-07-02)

Workers RPi `192.168.77.11-.14` ✅ · Jetson `192.168.77.21-.23` ✅ — los 7 responden ping desde el maestro por la LAN física.

## Reloj / energía

- NTP **activo y sincronizado**; zona `America/Bogota` (−05:00). Timestamps alineables con el medidor de pared ✅ (gotcha G7 resuelto).
- Rieles INA3221 de las Jetson: **por inventariar** cuando se entre a una Jetson (Fase 4).

## Pendientes de Fase 0

- [ ] Confirmar subida de los 3 `.bin` al NFS (en curso).
- [ ] Verificar que un worker RPi y una Jetson montan el NFS y ven los `.bin`.
- [ ] Inventario de una Jetson: versión CUDA/JetPack, `sm_53`, ruta INA3221, que monte el NFS.
- [ ] Setup RTX (CUDA 12.x + MSVC + git) — runbook aparte.
