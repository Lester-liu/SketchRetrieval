#!/usr/bin/env bash
Input="$1" # input folder with encoded files
Output="$2" # output file
K="$3" # number of words

./Database/Debug/./database $Input $Output $K
