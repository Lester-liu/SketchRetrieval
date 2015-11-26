#!/usr/bin/env bash

# Ex:

Input="$1" # input folder with encoded files
Output_data="$2" # output tf-idf file
Output_index="$3" # output label file
K="$4" # number of words

./Database/Debug/./build_database -i $Input -d $Output_data -m $Output_index -k $K
