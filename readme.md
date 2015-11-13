# Sketch-based 3D model retrieval

## About

This project is based on Mathias Eitz's paper "[Sketch-Based Shape Retrieval](http://cybertron.cg.tu-berlin.de/eitz/pdf/2012_siggraph_sbsr.pdf)".

## Idea

The idea is quite simple, 3D models are complex and can not be given directly to any machine learning mecanisms. So, instead of using the whole
model, we use different views of the model. Then we summarize the view with words using K-Mean and run a SVM on these bags of features.

## How to use

To run this code, you will need:

1. [Data](http://www.itl.nist.gov/iad/vug/sharp/contest/2013/SBR/): images and 3D models
2. VTK library
3. OpenCV
4. C++ 11 compiler

Then create your own pipeline as follow:

1. Create a folder named 'pipeline' as the same level of 'src'
2. Put the model folder inside it
3. Generate PLY models and views pictures with script inside the 'utils' (You will need to compile first the program inside 'utils')