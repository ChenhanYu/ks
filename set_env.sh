#!/bin/bash
export KS_DIR=$PWD
echo "KS_DIR = $KS_DIR"

# Manually set the target architecture.
export KS_ARCH_MAJOR=x86_64
export KS_ARCH_MINOR=sandybridge
echo "KS_ARCH = $KS_ARCH_MAJOR/$KS_ARCH_MINOR"

# Manually set the mkl path
export KS_MKL_DIR=/opt/intel/mkl
echo "KS_MKL_DIR = $KS_MKL_DIR"

# Parallel options
export KMP_AFFINITY=compact
export OMP_NUM_THREADS=1
export KS_IC_NT=1
