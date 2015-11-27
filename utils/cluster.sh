#!/usr/bin/env bash

# Use K-Means to cluster all vectors
# Ex: bash cluster.sh ../../../data/Sketch/vectors.bin 1024 ../../../data/Sketch/result.dict 20 0.01

Vectors="$1"
Dict="$3"
K="$2"
Iteration="$4"
V="$5"

./KMean/Release/k_mean -1 -f $Vectors -k $K -d $Dict -s 32 -i $Iteration -v $V
