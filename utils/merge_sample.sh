#!/usr/bin/env bash

#Ex: bash merge_sample.sh ../../../data/TinySketch/gabors/ ../../../data/TinySketch/vectors.bin 282240 512 2

Path="$1"
Output="$2"
Line="$3"
Dim="$4"
Skip="$5"

./Merge/Debug/merge -a $Path -o $Output -n $Line -d $Dim -s $Skip