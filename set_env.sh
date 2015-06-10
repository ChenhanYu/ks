#!/bin/bash
export KS_DIR=$PWD
echo "KS_DIR = $KS_DIR"

export KMP_AFFINITY=compact
export OMP_NUM_THREADS=1
export KS_IC_NT=1

export MIC_LD_LIBRARY_PATH=/opt/apps/intel/13/composer_xe_2013.2.146/mkl/lib/mic:/opt/apps/intel/13/composer_xe_2013.2.146/compiler/lib/mic



# Intel Xeon Phi on Stampede
export MIC_USE_2MB_BUFFERS=16K
#export MIC_OMP_NESTED=
export MIC_ENV_PREFIX=MIC
#export MIC_OMP_NUM_THREADS=4
export MIC_OMP_NUM_THREADS=240
export MIC_KMP_AFFINITY=compact
#export MIC_KMP_PLACE_THREADS=
#export MIC_KMP_PLACE_THREADS=60c,4t
#export MIC_KMP_AFFINITY=granularity=fine,scatter
#export MIC_KMP_AFFINITY=granularity=fine,compact

#export MIC_BLIS_IC_NT=15
#export MIC_BLIS_JR_NT=16




# For macbook pro
export DYLD_LIBRARY_PATH=/opt/intel/lib:/opt/intel/mkl/lib
echo "DYLD_LIBRARY_PATH = $DYLD_LIBRARY_PATH"
#ulimit -s 65532
