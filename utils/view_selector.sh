#!/usr/bin/env bash

# One folders with PNG format files and the name of the output file

Files="$1"
Name="$2"
for Folder in $Files*/
do
    ./ViewSelector/Debug/selector $Folder $Folder/$Name
done
