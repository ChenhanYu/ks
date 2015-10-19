#!/bin/bash
export KS_DIR=$PWD
echo "KS_DIR = $KS_DIR"


# Manually set the target architecture.
export KS_ARCH_MAJOR=x86_64
export KS_ARCH_MINOR=haswell
#export KS_ARCH_MINOR=sandybridge
export KS_ARCH=$KS_ARCH_MAJOR/$KS_ARCH_MINOR
echo "KS_ARCH = $KS_ARCH"


# Manually set the mkl path
export KS_MKL_DIR=/opt/intel/mkl
echo "KS_MKL_DIR = $KS_MKL_DIR"


# Parallel options
export KMP_AFFINITY=compact
export OMP_NUM_THREADS=1
export KS_IC_NT=1


# Add search paths.
export PATH=${PATH}:/opt/intel/bin
export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:/opt/intel/lib:${KS_MKL_DIR}/lib
echo "DYLD_LIBRARY_PATH = ${DYLD_LIBRARY_PATH}"
