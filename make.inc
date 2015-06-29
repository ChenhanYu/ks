CC = icc
CXX = icpc
ARCH = ar
ARCHFLAGS = cr
RANLIB = ranlib
CFLAGS = -O3 -openmp -mavx
LIBKS = $(KS_DIR)/lib/libks.a
#LDFLAGS = -I$(KS_DIR)/include -I$(KS_MKL_DIR)/include
LDFLAGS = -I$(KS_DIR)/include -I$(KS_DIR)/micro_kernel/$(KS_ARCH) -I$(KS_MKL_DIR)/include
LDLIBS = $(LIBKS) -lpthread -lm -openmp -mkl=parallel -Werror -Wall -pedantic
