#!/usr/bin/env bash

Path="$1"
Dict="$2"
Name="$3"
Output="$4"
Cases="$5"

for Folder in $Path*/
do
    ./KMean/Debug/./k_mean -f $Folder -d $Dict -s 32 -c $Cases -a 1089 -o $Output
done