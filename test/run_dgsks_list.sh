#!/bin/bash

#n=512
#k=288
#
#for (( m=4; m<8192; m+=64 ))
#do
#  ./test_dgsks.x $m $n $k
#done

#m=4096
#k=512
#
#for (( n=4; n<8192; n+=63 ))
#do
#  ./test_dgsks.x $m $n $k
#done

#rangem=4096
#rangen=512
#ntask=500
#
#for (( k=4; k<1604; k+=64 ))
#do
#  ./test_dgsks_list.x $rangem $rangen $k $ntask
#done


#./test_dgsks_list.x 4096 512 240 500
./test_dgsks_list.x 4097 513 16 100

#./test_dgsks_list_mic.x 8192 600 240 500
