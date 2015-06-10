#!/bin/bash
export KS_DIR=$PWD
echo "KS_DIR = $KS_DIR"

export KMP_AFFINITY=compact
export OMP_NUM_THREADS=1
export KS_IC_NT=1
export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:/opt/intel/lib:/opt/intel/mkl/lib


