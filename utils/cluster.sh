#!/usr/bin/env bash

# Use K-Means to cluster all vectors
# Ex: bash cluster.sh ../../../../data/TinySketch/vectors.bin 512 ../../../../data/TinySketch/result.dict 20 0.01

Vectors="$1"
Dict="$2"
K="$3"
Iteration="$4"
V="$5"

./KMean/Release/k_mean -1 -f $Vectors -k $K -d $Dict -s 32 -i $Iteration -v $V
