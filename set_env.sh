#!/bin/bash
export GSKS_DIR=$PWD
echo "GSKS_DIR = $GSKS_DIR"

## Manually set the target architecture.
# export GSKS_ARCH_MAJOR=x86_64
# export GSKS_ARCH_MINOR=sandybridge

export GSKS_ARCH_MAJOR=x86_64
export GSKS_ARCH_MINOR=haswell


export GSKS_ARCH=$GSKS_ARCH_MAJOR/$GSKS_ARCH_MINOR
echo "GSKS_ARCH = $GSKS_ARCH"

## Compiler options (if false, then use GNU compilers)
export GSKS_USE_INTEL=true
echo "GSKS_USE_INTEL = $GSKS_USE_INTEL"

## Whether use BLAS or not?
export GSKS_USE_BLAS=true
echo "GSKS_USE_BLAS = $GSKS_USE_BLAS"

## Whether use VML or not? (only if you have MKL)
export GSKS_USE_VML=true
echo "GSKS_USE_VML = $GSKS_USE_VML"


## Manually set the mkl path
export GSKS_MKL_DIR=$TACC_MKL_DIR
# export GSKS_MKL_DIR=/opt/intel/mkl
echo "GSKS_MKL_DIR = $GSKS_MKL_DIR"

## Parallel options
export KMP_AFFINITY=compact
export OMP_NUM_THREADS=24
export KS_IC_NT=24
