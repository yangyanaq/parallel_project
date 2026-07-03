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

# Jetson: kernels con nvcc (CUDA 10.2, sm_53), host con mpicc, enlace con mpicc
jetson: bin/kmeans_jetson
bin/kmeans_jetson: $(COMMON_SRC) src/jetson/main_hybrid.c src/jetson/kmeans_kernel.cu $(COMMON_HDR)
	@mkdir -p bin build
	$(NVCC) -O3 -arch=sm_53 -std=c++11 -c src/jetson/kmeans_kernel.cu -o build/kmeans_kernel.o
	$(MPICC) $(CFLAGS) -c $(COMMON_SRC) src/jetson/main_hybrid.c
	@mv *.o build/ 2>/dev/null || true
	$(MPICC) build/rng.o build/io_dataset.o build/kmeans_core.o build/metrics.o \
	         build/main_hybrid.o build/kmeans_kernel.o -o $@ \
	         -L/usr/local/cuda/lib64 -lcudart $(LDLIBS)

clean:
	rm -rf bin build *.o

.PHONY: all seq rpi jetson clean
