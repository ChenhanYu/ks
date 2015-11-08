CC = icc
CXX = icpc

ARCH = ar
ARCHFLAGS = cr
RANLIB = ranlib

LIBGSKS = $(GSKS_DIR)/lib/libgsks.a
SHAREDLIBGSKS = $(GSKS_DIR)/lib/libgsks.so
CFLAGS = -O3 -openmp -mavx -fPIC
LDLIBS = $(LIBGSKS) -lpthread -lm -openmp -Werror -Wall -pedantic
LDFLAGS = -I$(GSKS_DIR)/include -I$(GSKS_DIR)/micro_kernel/$(GSKS_ARCH) -I$(GSKS_MKL_DIR)/include

ifeq ($(GSKS_USE_BLAS),true)
CFLAGS += -DUSE_BLAS
LDLIBS += -mkl=parallel
endif

ifeq ($(GSKS_USE_VML),true)
CFLAGS += -DUSE_VML
ifeq ($(GSKS_USE_BLAS),false)
LDLIBS += -mkl=parallel
endif
endif
