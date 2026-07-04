# Inventario real del clúster (Fase 0)

> Capturado 2026-07-02 vía SSH al maestro por ZeroTier. Fuente de verdad de rutas/versiones para compilar y lanzar.

## Acceso

| Qué | Valor |
|---|---|
| Maestro RPi — IP física | `192.168.77.10` (eth0); también `192.168.77.145` (wlan0) |
| Maestro RPi — IP ZeroTier | `10.144.101.22` (red `c7c8172af1a15747`, iface `zt5u43m7nw`) |
| Usuario / pass RPi (maestro y workers) | `cris` / ver `pass.txt` |
| Usuario / pass Jetson | `nano` / ver `pass.txt` |
| RTX 4070 Ti (Win 11) | `192.168.77.161`, usuario `windows 11` (con espacio), equipo `DESKTOP-4APJUUH`. SSH por túnel: `ssh -J cris@10.144.101.22 "windows 11"@192.168.77.161`. AnyDesk de respaldo |

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

## Jetson (workers .21–.23) — confirmado 2026-07-02

- **CUDA 10.2** en `/usr/local/cuda-10.2` (symlink `/usr/local/cuda`). `nvcc` no está en el PATH por defecto → exportar `PATH=/usr/local/cuda/bin:$PATH` al compilar.
- **JetPack R32.4.4** (L4T), board `t210ref`, `aarch64` → arch **`sm_53`** ✅ (coincide con el plan).
- `nfs-common` presente; NFS montado ✅.
- Rieles INA3221 para energía: **por inventariar** al entrar a compilar la variante Jetson (Fase 4).

## Estación RTX 4070 Ti (`.161`, Win 11) — confirmado 2026-07-03

- **GPU real: NVIDIA GeForce RTX 4070 Ti, 12 GB** (driver 591.86), arch **`sm_89`**. NO es 4090 (toda la doc previa lo asumía; corregido).
- **CUDA Toolkit 11.8** (`nvcc` V11.8.89) — 11.8 es la versión mínima que soporta Ada/`sm_89`. `nvml.lib` presente (energía G4).
- **MSVC:** Visual Studio Build Tools 2022 (workload C++). `cl` vía vcvars.
- **git** 2.42 para Windows.
- Acceso: **OpenSSH Server** activo, `ssh -J cris@10.144.101.22 "windows 11"@192.168.77.161` (llave ed25519 de la PC de trabajo en `%ProgramData%\ssh\administrators_authorized_keys`).
- Pendiente: clonar el repo en el HOME de la RTX (Fase 3).

## Estado NFS en workers (2026-07-02)

Los **7 workers montan** `/home/cris/kmeans_share` y ven los 3 `.bin` ✅:
- RPi `.11–.14`: se instaló `nfs-common` (faltaba) + `mount -a`. El fstab ya tenía la entrada → remonta al reboot.
- Jetson `.21–.23`: solo `mount -a` (nfs-common ya estaba).

## Pendientes de Fase 0

- [x] Subir los 3 `.bin` al NFS.
- [x] Verificar que los 7 workers montan el NFS y ven los `.bin`.
- [x] Inventario Jetson: CUDA 10.2 / JetPack R32.4.4 / `sm_53`.
- [ ] Rutas INA3221 de las Jetson (se hace en Fase 4).
- [x] Setup RTX (CUDA 11.8 + MSVC 2022 + git) — HECHO 2026-07-03; SSH habilitado. Falta solo clonar el repo.
