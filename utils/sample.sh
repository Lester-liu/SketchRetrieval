#!/usr/bin/env bash
#exemple sample.sh ../Sketch/views/ ../Sketch/bin/ view.txt ../Sketch/sample/

Files="$1"
Bin_Path="$2"
List="$3"
Result_Path="$4"

for Folder in $Files*/
do
    Dest_Folder_Name="${Folder%/}"
    Dest_Folder_Name="${Dest_Folder_Name##*/}"

    ./Sample/Debug/./sample $Folder/$List $Bin_Path/$Dest_Folder_Name/ $Result_Path/$Dest_Folder_Name.bin
done