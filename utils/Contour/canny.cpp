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

int find_second_last_of(string s, char c){
    int i = s.length();
    for(int j = 0; j < 2; j++) {
        while (i >= 0 && s[i] != c)
            i--;
    }
    if (s[i]!=c)
        return -1;
    else
        return i;
}

int main(int argc, char** argv) {
    if (argc != 3){
        cerr << "Please enter the path of input and the path of output!" << endl;
        return 0;
    }
    Mat image = imread(argv[1]);
    Mat gray_image;
    cvtColor( image, gray_image, CV_BGR2GRAY );
    Mat outline;
    Canny(gray_image,outline,300,500);
    imwrite(argv[2],outline);
    return 0;
}