/*
 * Sketch-based 3D model retrieval
 *
 * This program is based on Eitz et al. paper, it has two component: a offline one and an online one. In the main
 * structure, we will code the online part which take a sketch as input and output a 3D model. The offline part is
 * coded in other project under the folder "utils/".
 *
 * Whenever you want to use this program, run the offline part first, which builds a database for online query.
 *
 * Usage 1 (file based query): sketch -d [database_file] -w [label_file] -f [input_file]
 * Usage 2 (real-time query with camera): sketch -d [database_file] -w [label_file] -c
 */

#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

enum Mode {Camera, File};
Mode mode = File;

string database_file, label_file, input_file;

// return the index of model
int retrieve(Mat& image) {
    // use Gabor filter

    // translate into words

    // compute TF-IDF

    // get nearest neighbor
}

int main(int argc, char** argv) {

    string input;

    // choose a mode
    switch (argc) {
        case 2:
            if (argv[1] == "-c")
                mode = Camera;
            else
                return EXIT_FAILURE;
            break;
        case 3:
            if (argv[1] == "-f")
                input = argv[2];
            else
                return EXIT_FAILURE;
            break;
        default:
            return EXIT_FAILURE;
    }

    if (mode == File) {

    }
    else if (mode == Camera) {

    }

    return EXIT_SUCCESS;
}