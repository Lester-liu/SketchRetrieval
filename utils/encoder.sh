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
    echo "./KMean/Debug/./k_mean -f $Folder -d $Dict -s 32 -c $Cases -a 784 -o $Output$Dest_Name.trans"
    ./KMean/Release/./k_mean -4 -f $Folder -d $Dict -s 32 -c $Cases -a 784 -o $Output$Dest_Name.trans
    #-f [Path_to_file] -r [Folder_to_contour] -d [Path_to_dictionary] -s [8|32] -o [output_file] -c [Cases] -a [data_size]
    k_mean -4 -f ../../../../data/TinySketch/views/m87/view.txt -r ../../../../data/TinySketch/contours/m87/ -d ../../../../data/TinySketch/result.dict -s 32 -o ../../../../data/TinySketch/encodes/m87.trans -c 36 -a 784

done