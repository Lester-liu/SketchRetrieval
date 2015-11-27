#!/usr/bin/env bash

Path="$1"
Output="$2"
Line="$3"
Dim="$4"

./Merge/Debug/merge -a $Path -o $Output -n $Line -d $Dim