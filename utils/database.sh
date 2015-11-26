#!/usr/bin/env bash

# Ex: bash database.sh ../../../data/TinySketch/encode/ ../../../data/TinySketch/tf-idf ../../../data/TinySketch/label 100

Input="$1" # input folder with encoded files
Output_data="$2" # output tf-idf file
Output_index="$3" # output label file
K="$4" # number of words

./Database/Debug/./build_database -i $Input -d $Output_data -m $Output_index -k $K
