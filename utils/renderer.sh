#!/usr/bin/env bash

# One folders with PLY format files and one folder to the image
# Ex: bash renderer.sh ../../../data/TinySketch/models_ply/ ../../../data/TinySketch/view/ 6

Files="$1*.ply"
Dest_Folder="$2"

Number="$3"

for Src in $Files
do
        # Get the file name (after the last '/')
        File="${Src##*/}"
        # Remove the extension
        Dest="${File:0:-4}";
        # Create the folder
        rm -rf "$Dest_Folder$Dest";
        mkdir "$Dest_Folder$Dest";
        # Run the renderer
        ./Renderer/Debug/renderor -g -n $Number -f $Src -t $Dest_Folder$Dest/;
done