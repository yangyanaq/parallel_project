# Runbook — Unir el maestro RPi (.10) a ZeroTier

**Objetivo:** que la PC de trabajo (que NO está en la red física `192.168.77.x`) pueda
llegar por SSH al maestro del clúster desde cualquier lugar, vía ZeroTier. Con eso se
automatiza el inventario, la copia de datos al NFS y el lanzamiento de benchmarks.

**Se hace UNA sola vez.** Camino: AnyDesk → estación RTX → SSH al `.10`.

## Pasos (pegar en la terminal del maestro `.10`)

```bash
# 1. Instalar ZeroTier (necesita internet; es el instalador oficial)
curl -s https://install.zerotier.com | sudo bash

# 2. Unirse a la red (ID de la red ZeroTier de la PC de trabajo)
sudo zerotier-cli join <NETWORK_ID>

# 3. Ver el ID del nodo (10 caracteres hex) — anotarlo
sudo zerotier-cli info
```

## Autorización

En **my.zerotier.com** → tu red → sección *Members*: aparecerá el nuevo nodo del RPi.
Marcar **Auth** (checkbox) y ponerle nombre (`rpi-master`). Anotar la **IP administrada**
que le asigne la red (esa es la IP con la que la PC de trabajo hará `ssh`).

## Verificación (desde la PC de trabajo)

```bash
ping <ip_zerotier_del_rpi>
ssh <usuario>@<ip_zerotier_del_rpi>
```

## Notas

- Solo el maestro `.10` necesita ZeroTier: los workers (`.11-.14`, `.21-.23`) se alcanzan
  desde el maestro por la LAN física, y el maestro ya es el lanzador de ambos clústeres.
- La estación RTX podría unirse igual (opcional) si algún día se quiere evitar AnyDesk.
- ZeroTier no interfiere con la red `192.168.77.x`: agrega una interfaz virtual aparte.
