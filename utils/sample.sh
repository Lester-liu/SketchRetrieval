#!/usr/bin/env bash

Files="$1"
Dest_Folder="$2"
Name="$3"

for Folder in $Files*/
do
    Dest_Folder_Name="${Folder%/}"
    Dest_Folder_Name="${Dest_Folder_Name##*/}"

    ./Sample/Debug/./sample $Folder/view.txt $Dest_Folder/$Dest_Folder_Name/ $Dest_Folder/$Dest_Folder_Name/$Name
done