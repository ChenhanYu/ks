CC = gcc
CXX = g++

ARCH = ar
ARCHFLAGS = cr
RANLIB = ranlib

LIBGSKS = $(GSKS_DIR)/lib/libgsks.a
SHAREDLIBGSKS = $(GSKS_DIR)/lib/libgsks.so
CFLAGS = -O3 -fopenmp -mavx -fPIC
LDLIBS = $(LIBGSKS) -lpthread -lm -fopenmp
LDFLAGS = -I$(GSKS_DIR)/include -I$(GSKS_DIR)/micro_kernel/$(GSKS_ARCH)

ifeq ($(GSKS_USE_BLAS),true)
CFLAGS += -DUSE_BLAS
LDLIBS += -lblas
endif
