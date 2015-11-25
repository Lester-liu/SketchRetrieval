#!/usr/bin/env bash

# One folders with PNG format files and one folder to the contour
#gabor_filter.sh ../../../data/Sketch/pipeline/contour/ ../../../data/Sketch/pipeline/bin/

Files="$1"
Dest_Folder="$2"

for Folder in $Files*/
do
    Dest_Folder_Name="${Folder%/}"
    Dest_Folder_Name="${Dest_Folder_Name##*/}"
    # Create the folder
    rm -rf "$Dest_Folder$Dest_Folder_Name/";
    mkdir "$Dest_Folder$Dest_Folder_Name/";

    for Src in $Folder*.png
    do
            #echo $Src
            # Get the file name (after the last '/')
            File="${Src##*/}"
            File="${File%.*}"
            #echo $Dest_Folder$Dest_Folder_Name/$File
            # Run the contour_extraction
            ./Gabor/Debug/gabor -i $Src -o $Dest_Folder$Dest_Folder_Name/$File.bin;
    done
done