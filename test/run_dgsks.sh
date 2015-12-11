#!/bin/bash
export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:/opt/intel/lib:${GSKS_MKL_DIR}/lib

m=3600
n=4097
kmin=4
kmax=2048
kinc=31

echo 'Gaussian = ['
for (( k=kmin; k<kmax; k+=kinc ))
do
  ./test_dgsks.x Gaussian $m $n $k
done
echo '];'

echo 'Polynomial = ['
for (( k=kmin; k<kmax; k+=kinc ))
do
  ./test_dgsks.x Polynomial $m $n $k
done
echo '];'

echo 'Laplace = ['
for (( k=kmin; k<kmax; k+=kinc ))
do
  ./test_dgsks.x Laplace $m $n $k
done
echo '];'

echo 'Var_bandwidth = ['
for (( k=kmin; k<kmax; k+=kinc ))
do
  ./test_dgsks.x Var_bandwidth $m $n $k
done
echo '];'

echo 'Tanh = ['
for (( k=kmin; k<kmax; k+=kinc ))
do
  ./test_dgsks.x Tanh $m $n $k
done
echo '];'

echo 'Quartic = ['
for (( k=kmin; k<kmax; k+=kinc ))
do
  ./test_dgsks.x Quartic $m $n $k
done
echo '];'

echo 'Multiquadratic = ['
for (( k=kmin; k<kmax; k+=kinc ))
do
  ./test_dgsks.x Multiquadratic $m $n $k
done
echo '];'

echo 'Epanechnikov = ['
for (( k=kmin; k<kmax; k+=kinc ))
do
  ./test_dgsks.x Epanechnikov $m $n $k
done
echo '];'
