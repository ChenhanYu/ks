#!/bin/bash

#n=512
#k=512
#
#for (( m=4; m<1600; m+=4 ))
#do
#  ./test_dgsks.x $m $n $k
#done

#m=4096
#k=512
#
#for (( n=4; n<8192; n+=128 ))
#do
#  ./test_dgsks.x $m $n $k
#done


m=4097
n=4097

echo 'd_mn2048_gaussian = ['
for (( k=240; k<1028; k+=16 ))
do
  ./test_dgsks.x $m $n $k
done


#m=4096
#n=4096
#
#echo 'd_mn4096_tanh = ['
#for (( k=4; k<260; k+=4 ))
#do
#  ./test_dgsks.x $m $n $k
#done
#echo '];'


#m=8192
#n=8192
#
#echo 'd_mn8192_gaussian = ['
#for (( k=4; k<260; k+=4 ))
#do
#  ./test_dgsks.x $m $n $k
#done
#echo '];'





#k=8
#
#echo 'p1_d8_mn_gaussian = ['
#for (( m=32; m<4100; m+=32 ))
#do
#  ./test_dgsks.x $m $m $k
#done
#echo '];'
#
#
#k=64
#
#echo 'p1_d64_mn_gaussian = ['
#for (( m=32; m<4100; m+=32 ))
#do
#  ./test_dgsks.x $m $m $k
#done
#echo '];'
#
#
#k=512
#
#echo 'p1_d64_mn_gaussian = ['
#for (( m=32; m<4100; m+=32 ))
#do
#  ./test_dgsks.x $m $m $k
#done
#echo '];'











#./test_dgsks.x 16 4 1

#./test_dgsks.x 8192 8192 4
#./test_dgsks.x 8192 8192 32
#./test_dgsks.x 8192 8192 256
#./test_dgsks.x 8192 8192 1024


#./test_dgsks.x 16384 512 256
#./test_dgsks.x 16384 512 256
#./test_dgsks.x 16384 512 257
#./test_dgsks.x 16384 512 257


#m=20000
#n=1024
#for (( k=4; k<1604; k+=32 ))
#do
#  ./test_dgsks_mic.x $m $n $k
#done


#./test_dgsks_mic.x 28800 28800 8
#./test_dgsks_mic.x 4096 4096 240
#./test_dgsks_mic.x 96 60 1
