#!/usr/bin/env bash

# One folders with OFF format files

EXT='ply'
Folders="$1*.off"

for Src in $Folders
do
        Dest="${Src//off/$EXT}";
        ./PLYConverter/Debug/ply_converter $Src $Dest;
done
