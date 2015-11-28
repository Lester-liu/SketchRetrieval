#!/usr/bin/env bash

# Online query takes one image as input

Sketch="$1"

Data="../../../data/Sketch/tf-idf"
Dict="../../../data/Sketch/result.dict"
Label="../../../data/Sketch/label"
Model="../../../data/Sketch/models_ply/"

././../Debug/sketch -d $Data -w $Dict -l $Label -m $Model -f $Sketch