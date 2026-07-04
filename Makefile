# Makefile — clústeres Linux (RPi 5 y Jetson Nano).
# La RTX (Windows) compila con scripts/build_rtx.ps1 (Fase 3), no con esto.
#
# Targets: seq | rpi | jetson | all | clean

CC      ?= gcc
MPICC   ?= mpicc
NVCC    ?= nvcc
CFLAGS  ?= -O3 -std=c11 -march=native -funroll-loops -Wall -Wextra
LDLIBS   = -lm

COMMON_SRC = src/common/rng.c src/common/io_dataset.c \
             src/common/kmeans_core.c src/common/metrics.c
COMMON_HDR = src/common/config.h src/common/rng.h src/common/io_dataset.h \
             src/common/kmeans_core.h src/common/metrics.h

# Fase 1: 'all' construye lo que ya existe; rpi (F2) y jetson (F4) se
# suman a 'all' cuando sus fuentes entren al repo.
all: seq

seq: bin/kmeans_seq
bin/kmeans_seq: $(COMMON_SRC) src/seq/main_seq.c $(COMMON_HDR)
	@mkdir -p bin
	$(CC) $(CFLAGS) $(COMMON_SRC) src/seq/main_seq.c -o $@ $(LDLIBS)

rpi: bin/kmeans_rpi
bin/kmeans_rpi: $(COMMON_SRC) src/rpi/main_mpi.c $(COMMON_HDR)
	@mkdir -p bin
	$(MPICC) $(CFLAGS) $(COMMON_SRC) src/rpi/main_mpi.c -o $@ $(LDLIBS)

# MPI multi-nodo exige el binario en la MISMA ruta en todos los nodos.
# Los workers NO clonan el repo: solo montan el NFS. 'install-rpi' copia el
# binario al NFS compartido (ver docs/runbook_lanzar_mpi.md). Lanzar SIEMPRE
# con la ruta NFS: mpirun --hostfile hosts_rpi -np N $(NFS_BIN) ...
NFS_DIR ?= /home/cris/kmeans_share
install-rpi: rpi
	@mkdir -p $(NFS_DIR)/bin
	cp bin/kmeans_rpi $(NFS_DIR)/bin/kmeans_rpi
	@echo "binario en $(NFS_DIR)/bin/kmeans_rpi (visible por todos los nodos)"

# Jetson: kernels con nvcc (CUDA 10.2, sm_53), host con mpicc, enlace con mpicc.
# nvcc no está en el PATH de la Jetson -> CUDA_HOME apunta al toolkit. gcc 7.5
# y el host híbrido en C11 conservador (sin -march=native para no arriesgar en
# la toolchain vieja; ver PLAN §5).
CUDA_HOME ?= /usr/local/cuda
JCFLAGS    = -O3 -std=c11 -funroll-loops -Wall -Wextra
jetson: bin/kmeans_jetson
bin/kmeans_jetson: $(COMMON_SRC) src/jetson/main_hybrid.c src/jetson/kmeans_kernel.cu $(COMMON_HDR)
	@mkdir -p bin build
	$(CUDA_HOME)/bin/nvcc -O3 -arch=sm_53 -std=c++11 \
	    -c src/jetson/kmeans_kernel.cu -o build/kmeans_kernel.o
	$(MPICC) $(JCFLAGS) -c $(COMMON_SRC) src/jetson/main_hybrid.c
	@mv *.o build/ 2>/dev/null || true
	$(MPICC) build/rng.o build/io_dataset.o build/kmeans_core.o build/metrics.o \
	         build/main_hybrid.o build/kmeans_kernel.o -o $@ \
	         -L$(CUDA_HOME)/lib64 -lcudart_static -ldl -lrt -lpthread $(LDLIBS)
# NOTA: -lcudart_static (no -lcudart) porque el clúster Jetson es HETEROGÉNEO:
# .21/.22 traen CUDA 10.2 y .23 trae CUDA 10.0. Enlazar el runtime dinámico
# ataba el binario a libcudart.so.10.2, que falta en la .23 ("cannot open
# shared object"). El runtime estático hace el binario portable entre las 3.

# Igual que install-rpi: el binario al NFS para que las 3 Jetson lo vean en la
# misma ruta (ver docs/runbook_lanzar_mpi.md, gotcha binario-en-NFS).
install-jetson: jetson
	@mkdir -p $(NFS_DIR)/bin
	cp bin/kmeans_jetson $(NFS_DIR)/bin/kmeans_jetson
	@echo "binario en $(NFS_DIR)/bin/kmeans_jetson (visible por las 3 Jetson)"

clean:
	rm -rf bin build *.o

.PHONY: all seq rpi install-rpi jetson install-jetson clean
