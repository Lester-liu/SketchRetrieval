/*
 * Contour extraction with Canny algorithm
 *
 * This program uses Canny module from OpenCV to extract the contour of an image. For the algorithm's detail,
 * please read the OpenCV documentation.
 *
 * Parameters:
 *      f (path to the input image)
 *      t (path to store the result)
 *      s (parameters to tune the result)
 * Ex: contour -f [Path_to_input_image] -t [Path_to_contour_image] [-s max_threshold min_threshold]
 */

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
char *input,*output;
int max_threshold, min_threshold;

void show_help(){
    printf("/*\n"
                   " * Contour extraction with Canny algorithm\n"
                   " *\n"
                   " * This program uses Canny module from OpenCV to extract the contour of an image. For the algorithm's detail,\n"
                   " * please read the OpenCV documentation.\n"
                   " *\n"
                   " * Parameters:\n"
                   " *      f (path to the input image)\n"
                   " *      t (path to store the result)\n"
                   " *      s (parameters to tune the result)\n"
                   " * Ex: contour -f [Path_to_input_image] -t [Path_to_contour_image] [-s max_threshold min_threshold]\n"
                   " */");
}

bool parse_command_line(int argc, char **argv) {

    int i = 1;
    while(i < argc) {
        if (argv[i][0] != '-')
            break;
        switch(argv[i][1]) {
            case 'h': // help
                show_help();
                return false;
            case 's': // single mode flag
                max_threshold = atoi(argv[++i]);
                min_threshold = atoi(argv[++i]);
                break;
            case 'f': // input model file
                input = argv[++i];
                break;
            case 't': // output image file
                output = argv[++i];
                break;
        }
        i++;
    }
    if (input == NULL || output == NULL) { // invalid file name
        show_help();
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    input = NULL;
    output = NULL;
    min_threshold = 300;
    max_threshold = 500;
    if (!parse_command_line(argc,argv))
        return 0;
    Mat image = imread(input, IMREAD_GRAYSCALE);
    Mat outline;
    Canny(image, outline, min_threshold, max_threshold);
    imwrite(output, outline);
    return 0;
}