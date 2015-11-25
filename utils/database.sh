#!/usr/bin/env bash
Input="$1" # input folder with encoded files
Output_data="$2" # output file
Output_index="$3"
K="$4" # number of words

./Database/Debug/./build_database -i $Input -d $Output_data -m $Output_index -k $K
