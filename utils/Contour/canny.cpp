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

int main(int argc, char** argv) {
    if (argc != 3){
        cerr << "Please enter the path of input and the path of output!" << endl;
        return 0;
    }
    Mat image = imread(argv[1], IMREAD_GRAYSCALE);
    Mat outline;
    Canny(image, outline, 300, 500);
    imwrite(argv[2], outline);
    return 0;
}