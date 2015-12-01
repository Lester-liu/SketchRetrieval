#!/usr/bin/env bash

# Online query takes one image as input

Sketch="$1"

Data="../../../data/Sketch/tf-idf_2048"
Dict="../../../data/Sketch/result_2048.dict"
Label="../../../data/Sketch/label_2048"
Model="../../../data/Sketch/models_ply/"
View="../../../data/NormalSketch/views/"

././../Release/sketch -d $Data -w $Dict -l $Label -m $Model -f $Sketch -v $View