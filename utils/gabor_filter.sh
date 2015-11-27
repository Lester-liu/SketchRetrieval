#!/usr/bin/env bash

# One folders with PNG format files and one folder to the contour
# Ex: bash gabor_filter.sh ../../../data/Sketch/pipeline/view/ ../../../Sketch/pipeline/contour/ ../../../data/Sketch/pipeline/bin/ view.txt

View_Foler="$1"
Contour_Folder="$2"
Dest_Folder="$3"
Name="$4"

for Folder in $View_Foler*/
do
    Dest_Name="${Folder%/}"
    Dest_Name="${Dest_Name##*/}"
    #echo "./Gabor/Debug/./gabor -i $Folder$Name -o $Dest_Folder$Dest_Name.bin -a $Contour_Folder$Dest_Name/"
    ./Gabor/Debug/./gabor -i $Folder$Name -o $Dest_Folder$Dest_Name.bin -a $Contour_Folder$Dest_Name/

done