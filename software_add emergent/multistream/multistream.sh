#!/bin/bash
clear
TIME=$1
echo ${TIME}
for (( c=1; c<=${TIME}; c++ ))
do  
   echo "Welcome $c times"
   sudo ./multistream -p 1 -c 12 -t 10 -b 32 -e 44
   echo "Test done $c ############################################################## $c"
   #sleep 5
done

