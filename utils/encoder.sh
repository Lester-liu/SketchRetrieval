#!/usr/bin/env bash

#Ex: bash encoder.sh ../Sketch/pipeline/bin/ /Sketch/pipeline/dict.txt ../Sketch/dic/ 36

Path="$1" #path of input(ex: binary folder)
Dict="$2" #path of dictionary
Output="$3" #path of output folder
Cases="$4" #case number

for Folder in $Path*/
do
    Dest_Name="${Folder%/}"
    Dest_Name="${Dest_Name##*/}"
    #echo Dest_Name
    #echo "./KMean/Debug/./k_mean -f $Folder -d $Dict -s 32 -c $Cases -a 1089 -o $Output$Dest_Name.trans"
    ./KMean/Release/./k_mean -f $Folder -d $Dict -s 32 -c $Cases -a 784 -o $Output$Dest_Name.trans
done