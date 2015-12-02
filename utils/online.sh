#!/usr/bin/env bash

# Online query takes one image as input

Sketch="$1"

Center="2048"

Data="../../../data/Sketch/tf-idf_$Center"
Dict="../../../data/Sketch/result_$Center.dict"
Label="../../../data/Sketch/label_$Center"
Model="../../../data/Sketch/models_ply/"
View="../../../data/NormalSketch/views/"
Sorted="../../../data/Sketch/views/"

././../Release/sketch -d $Data -w $Dict -l $Label -m $Model -f $Sketch -v $View -s $Sorted -t