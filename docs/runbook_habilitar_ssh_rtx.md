# Runbook — habilitar SSH en la RTX 4070 Ti (Windows 11), una sola vez

> Objetivo: que Claude (desde la PC de trabajo, vía el maestro RPi) entre por SSH a
> la RTX y ejecute/valide la Fase 3 sin que tú teclees todo por AnyDesk.
>
> La RTX es `192.168.77.161` (Windows 11) en la LAN. Pasos **una vez por AnyDesk**.

## 1. Activar OpenSSH Server

Abre **PowerShell como Administrador** (clic derecho → "Ejecutar como administrador") y pega:

```powershell
# instalar el servidor SSH (viene con Windows 10/11, solo hay que añadirlo)
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0

# arrancarlo y dejarlo automático al reiniciar
Start-Service sshd
Set-Service  sshd -StartupType Automatic

# abrir el puerto 22 en el firewall
New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH Server' `
  -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
```

Verifica que quedó corriendo:

```powershell
  Get-Service sshd     

     # Status debe decir "Running"
whoami                      # anota el usuario (ej. rtx\cris) — lo necesito
ipconfig | findstr IPv4     # confirma que la IPv4 en la LAN es 192.168.77.161
```

## 2. Instalar la llave pública de la PC de trabajo (sin contraseña)

Mi llave pública (de la PC de trabajo, ya generada):

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEnjHK01EQrtj6O4Ngny6S78aufnkukyZSJNxcaGJXlk ren-pc-tesis
```

En la **misma PowerShell (Admin)** de la RTX, pega (crea el archivo de llaves
autorizadas para tu usuario):

```powershell
$k = 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEnjHK01EQrtj6O4Ngny6S78aufnkukyZSJNxcaGJXlk ren-pc-tesis'
$d = "$env:USERPROFILE\.ssh"
New-Item -ItemType Directory -Force $d | Out-Null
Add-Content "$d\authorized_keys" $k
```

**OJO administradores en Windows:** OpenSSH ignora `~/.ssh/authorized_keys` para
cuentas del grupo Administradores y usa un archivo global. Si tu usuario es admin
(lo normal en esta PC), corre además:

```powershell
$k = 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEnjHK01EQrtj6O4Ngny6S78aufnkukyZSJNxcaGJXlk ren-pc-tesis'
Add-Content "$env:ProgramData\ssh\administrators_authorized_keys" $k
# usar los SID (independientes del idioma): *S-1-5-32-544=Administradores, *S-1-5-18=SYSTEM.
# Con nombres ("Administrators") falla en Windows en español ("no se efectuó asignación").
icacls "$env:ProgramData\ssh\administrators_authorized_keys" /inheritance:r /grant "*S-1-5-32-544:F" /grant "*S-1-5-18:F"
Restart-Service sshd
```

## 3. Qué me pasas a mí

- El **usuario de Windows** (salida de `whoami`, ej. `cris`).
- Confirmación de que `ipconfig` muestra `192.168.77.161` (si es otra, esa).

Con eso yo entro por **túnel a través del maestro**:
`ssh -J cris@10.144.101.22 <usuario>@192.168.77.161`
(la RTX no necesita ZeroTier; el maestro ya está en la LAN y me hace de salto.)

## 4. (Opcional) Confirmar que quedó bien

Desde tu propia PC de trabajo, en PowerShell:

```powershell
ssh -J cris@10.144.101.22 <usuario>@192.168.77.161 "hostname; nvidia-smi --query-gpu=name --format=csv,noheader"
```

Si responde el nombre de la PC y "NVIDIA GeForce RTX 4070 Ti", está listo.

---

**Nota de seguridad:** esto abre SSH por llave (no por contraseña) solo dentro de tu
LAN. No expone la RTX a internet. La llave privada nunca sale de la PC de trabajo.
Las contraseñas siguen fuera del repo (`pass.txt`).
