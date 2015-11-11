#!/usr/bin/env bash

# One folders with OFF format files

EXT='ply'
Folders="$1*.off"

for Src in $Folders
do
        # Replace all 'off' in the path by 'ply'
        Dest="${Src//off/$EXT}";
        # Run the converter
        ./PLYConverter/Debug/ply_converter $Src $Dest;
done
