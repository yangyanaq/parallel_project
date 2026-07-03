# Runbook — Montar el NFS en los workers (Fase 0, cierre)

**Diagnóstico (2026-07-02):** el maestro `.10` exporta `/home/cris/kmeans_share` y ya tiene
los `.bin`, pero los workers **no lo tienen montado**:

- **RPi `.11`–`.14`:** fstab correcto, pero **falta el paquete `nfs-common`**.
- **Jetson `.21`–`.23`:** `nfs-common` presente, fstab correcto, **solo falta ejecutar `mount`**.
  (CUDA 10.2 + JetPack R32.4.4 confirmado, arch `sm_53` ✅.)

Todo requiere `sudo` (escritura en infraestructura compartida).

## Opción A — Ejecutar en cada nodo (manual, seguro)

### En cada RPi worker (.11, .12, .13, .14), usuario `cris` / pass `nano`:

```bash
sudo apt-get update && sudo apt-get install -y nfs-common
sudo mount -a                      # monta lo del fstab
ls /home/cris/kmeans_share/data/   # debe listar los 3 .bin
```

### En cada Jetson (.21, .22, .23), usuario `nano` / pass `nano`:

```bash
sudo mount -a
ls /home/cris/kmeans_share/data/   # debe listar los 3 .bin
```

## Opción B — Que Claude lo haga por SSH anidado (previa autorización)

Claude puede recorrer los 7 workers desde el maestro (salto `direct-tcpip` con paramiko)
y ejecutar los mismos comandos con `sudo -S` (pass `nano` por stdin). Es más rápido pero
son escrituras `sudo` en 7 máquinas compartidas — requiere que autorices explícitamente.

## Verificación final (Claude la corre solo, es de lectura)

```
ls -l /home/cris/kmeans_share/data/*.bin   # en .11 y en .21 → deben aparecer 3 archivos de 3.2/32/320 MB
```

## Nota de arranque

`mount -a` monta ahora; el fstab ya garantiza el remontaje al reiniciar. Si algún worker
no reconecta tras reboot, revisar que el maestro (`nfs-kernel-server`) esté activo:
`sudo systemctl status nfs-kernel-server`.
