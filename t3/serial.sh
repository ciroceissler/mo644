#!/bin/bash

for i in `seq 1 1000`;
do
  V1=$(($V1+$(/home/ciroceissler/workspace/github/parallel_programming/t3/./hist_s < /home/ciroceissler/workspace/github/parallel_programming/t3/arq1.in | tail -1)))
  V2=$(($V2+$(/home/ciroceissler/workspace/github/parallel_programming/t3/./hist_s < /home/ciroceissler/workspace/github/parallel_programming/t3/arq2.in | tail -1)))
  V3=$(($V3+$(/home/ciroceissler/workspace/github/parallel_programming/t3/./hist_s < /home/ciroceissler/workspace/github/parallel_programming/t3/arq3.in | tail -1)))
  V4=$(($V4+$(/home/ciroceissler/workspace/github/parallel_programming/t3/./hist_p < /home/ciroceissler/workspace/github/parallel_programming/t3/arq1.in | tail -1)))
  V5=$(($V5+$(/home/ciroceissler/workspace/github/parallel_programming/t3/./hist_p < /home/ciroceissler/workspace/github/parallel_programming/t3/arq2.in | tail -1)))
  V6=$(($V6+$(/home/ciroceissler/workspace/github/parallel_programming/t3/./hist_p < /home/ciroceissler/workspace/github/parallel_programming/t3/arq3.in | tail -1)))
  V7=$(($V7+$(/home/ciroceissler/workspace/github/parallel_programming/t3/./hist_p < /home/ciroceissler/workspace/github/parallel_programming/t3/arq1_4.in | tail -1)))
  V8=$(($V8+$(/home/ciroceissler/workspace/github/parallel_programming/t3/./hist_p < /home/ciroceissler/workspace/github/parallel_programming/t3/arq2_4.in | tail -1)))
  V9=$(($V9+$(/home/ciroceissler/workspace/github/parallel_programming/t3/./hist_p < /home/ciroceissler/workspace/github/parallel_programming/t3/arq3_4.in | tail -1)))
  V10=$(($V10+$(/home/ciroceissler/workspace/github/parallel_programming/t3/./hist_p < /home/ciroceissler/workspace/github/parallel_programming/t3/arq1_8.in | tail -1)))
  V11=$(($V11+$(/home/ciroceissler/workspace/github/parallel_programming/t3/./hist_p < /home/ciroceissler/workspace/github/parallel_programming/t3/arq2_8.in | tail -1)))
  V12=$(($V12+$(/home/ciroceissler/workspace/github/parallel_programming/t3/./hist_p < /home/ciroceissler/workspace/github/parallel_programming/t3/arq3_8.in | tail -1)))
  V13=$(($V13+$(/home/ciroceissler/workspace/github/parallel_programming/t3/./hist_p < /home/ciroceissler/workspace/github/parallel_programming/t3/arq1_16.in | tail -1)))
  V14=$(($V14+$(/home/ciroceissler/workspace/github/parallel_programming/t3/./hist_p < /home/ciroceissler/workspace/github/parallel_programming/t3/arq2_16.in | tail -1)))
  V15=$(($V15+$(/home/ciroceissler/workspace/github/parallel_programming/t3/./hist_p < /home/ciroceissler/workspace/github/parallel_programming/t3/arq3_16.in | tail -1)))
  echo $i
done

echo '--------------------------'
echo 'serial:'
echo $V1
echo $(($V1/1000))
echo $(($V2/1000))
echo $(($V3/1000))

echo '2 threads:'
echo $(($V4/1000))
echo $(($V5/1000))
echo $(($V6/1000))

echo '4 threads:'
echo $(($V7/1000))
echo $(($V8/1000))
echo $(($V9/1000))

echo '8 threads:'
echo $(($V10/1000))
echo $(($V11/1000))
echo $(($V12/1000))

echo '16 threads:'
echo $(($V13/1000))
echo $(($V14/1000))
echo $(($V15/1000))

