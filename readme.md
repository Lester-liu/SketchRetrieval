# Sketch-based 3D model retrieval

## About

This project is based on Mathias Eitz's paper "[Sketch-Based Shape Retrieval](http://cybertron.cg.tu-berlin.de/eitz/pdf/2012_siggraph_sbsr.pdf)".

## Idea

The idea is quite simple, 3D models are complex and can not be given directly to any machine learning mecanisms. So, instead of using the whole
model, we use different views of the model. Then we summarize the view with words using K-Mean and run a nearest neigbor classifier on these bag-of-features.

There are two parts in the project, the online query is coded in the src directory, whereas the offline preparation is in the utils folder. Some of the codes
are similar. The offline part consists of multiple modules governed by shell script.

## Interesting components

This repository contains some modules that can be used in other jobs.

1. K-Means with CUDA (CuBLAS, CuSPARSE, etc.)
2. TF-IDF database for document searching
3. Gabor filter
4. Contour extractor
5. PLY 3D model to 2D image with different angles
6. OFF to PLY convertor

## How to use

To run this code, you will need:

1. [Data](http://www.itl.nist.gov/iad/vug/sharp/contest/2013/SBR/): images and 3D models
2. VTK library
3. OpenCV
4. CUDA and Nvidia GPU
5. C++ 11 compiler

Then create your own pipeline as follow:

1. Create a folder named 'pipeline' as the same level of 'src'
2. Put the model folder inside it
3. Generate PLY models and views pictures with script inside the 'utils' (You will need to compile first the program inside 'utils')