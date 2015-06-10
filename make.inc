CC = icc
#CC = gcc
CXX = icpc
#CXX = g++
NVCC = nvcc

ARCH = ar
ARCHFLAGS = cr
RANLIB = ranlib

CFLAGS = -O3 -openmp -mavx

# For Intel Xeon Phi on Stampede
CFLAGSMIC = -g -O3 -openmp -mmic -fasm-blocks

NVCCFLAGS = -O4 -arch sm_35 
#NVCCFLAGS = -O4 -arch sm_35 --maxrregcount=128


MIC_LD_LIBRARY_PATH=$(TACC_MKL_DIR)/lib/mic
MIC_VML_DIR=/opt/apps/intel/13/composer_xe_2013.2.146/compiler/lib/mic


LIBKS = $(KS_DIR)/lib/libks.a
LIBKSMIC = $(KS_DIR)/lib/libks-mic.a

CUDA_DIR = $(TACC_CUDA_DIR)

#LDFLAGS = -I$(KS_DIR)/include -I$(TACC_MKL_DIR)/include
LDFLAGS = -I$(CUDA_DIR)/include -I$(KS_DIR)/include -I$(TACC_MKL_DIR)/include
#LDFLAGS = -I$(CUDA_DIR)/include -I$(KS_DIR)/include -I/opt/intel/mkl/include

#LDLIBS = $(LIBKS) -lpthread -lm -openmp -mkl=sequential -Werror -Wall -pedantic
LDLIBS = $(LIBKS) -lpthread -lm -openmp -mkl=parallel -Werror -Wall -pedantic

LDLIBSMIC = $(LIBKSMIC) -lpthread -lm -openmp -L$(MIC_LD_LIBRARY_PATH) \
-L$(MIC_VML_DIR) -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lsvml


#LDLIBS = $(LIBKS) -L$(CUDA_DIR)/lib64 -lcudart -lcublas -lpthread -lm  -openmp \
#-mkl=parallel -Werror -Wall -pedantic
