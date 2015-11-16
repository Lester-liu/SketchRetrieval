/*
 * Gabor filter
 *
 * Gabor filter is one way to extract edge or other features from an image. By using multiple filters with
 * different orientation, one can build a bag-of-features for a given image. This program takes as parameters
 * the kernel size, orientation, etc..
 *
 * Parameters:
 *      k: number of orientations
 *      p: number of points
 *      n: size of the kernel
 *      s[sigma]: standard deviation of the gaussian envelope
 *      t[theta]: orientation of the normal to the parallel stripes of a Gabor function
 *      l[lambda]: wavelength of the sinusoidal factor
 *      g[gamma]: spatial aspect ratio
 *      i: input image
 *      o: output image folder (with '/' in the end)
 *
 * Usage:
 *      gabor -k [k] -n [n] -s [sigma] -t [theta] -l [lambda] -b [beta] -f [Path_to_input] -t [Path_to_output]
 */

#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

string input, output;

int kernel_size = 15;
int k = 8;
int window_size = 4;
int point_count = 1000;
double sigma = 4;
double theta = 0;
double lambda = 10.0;
double beta = 0.5;

vector<Mat> filters(k);
vector<vector<uchar> > result;

vector<uchar> getVector(int x, int y) {
    vector<uchar> result;

    for(int i = 0; i < k; i++)
        for(int u = 0; u < window_size; u++)
            for(int v = 0; v < window_size; v++)
                result.push_back(filters[i].at<uchar>(u + x, v + y));

    return result;
}

void output_result(string output) {
    ofstream out(output);

    for(vector<uchar> v: result) {
        for(uchar u: v)
            out << u;
        out << endl;
    }

    out.close();
}

void show_help() {
    printf("Gabor filter\n"
                   "s\n"
                   "Gabor filter is one way to extract edge or other features from an image. By using multiple filters with\n"
                   "different orientation, one can build a bag-of-features for a given image. This program takes as parameters\n"
                   "the kernel size, orientation, etc..\n"
                   "\n"
                   "Parameters:\n"
                   "     k: number of orientations\n"
                   "     p: number of points\n"
                   "     n: size of the kernel\n"
                   "     s[sigma]: standard deviation of the gaussian envelope\n"
                   "     t[theta]: orientation of the normal to the parallel stripes of a Gabor function\n"
                   "     l[lambda]: wavelength of the sinusoidal factor\n"
                   "     g[gamma]: spatial aspect ratio\n"
                   "     i: input image\n"
                   "     o: output image folder (with '/' in the end)\n"
                   "\n"
                   "Usage:\n"
                   "     gabor -k [k] -n [n] -s [sigma] -t [theta] -l [lambda] -b [beta] -f [Path_to_input] -t [Path_to_output]");

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
            case 'k':
                k = atoi(argv[++i]);
                break;
            case 'p':
                point_count = atoi(argv[++i]);
                break;
            case 'n':
                kernel_size = atoi(argv[++i]);
                break;
            case 's':
                sigma = atof(argv[++i]);
                break;
            case 't':
                theta = atof(argv[++i]);
                break;
            case 'l':
                lambda = atof(argv[++i]);
                break;
            case 'g':
                beta = atof(argv[++i]);
                break;
            case 'i': // input file
                input = argv[++i];
                break;
            case 'o': // output file
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

    if (!parse_command_line(argc, argv))
        return 0;

    Mat img = imread(input);
    Mat src;
    img.convertTo(src, CV_32F);

    double step = CV_PI / k;

    for(int i = 0; i < k; i++) {
        Mat kernel = getGaborKernel(Size(kernel_size, kernel_size), sigma,
                                    theta + step * (double)i, lambda, beta, CV_PI * 0.5, CV_32F);
        filter2D(src, filters[i], -1, kernel, Point(-1, -1), 0, BORDER_DEFAULT);
        filters[i].convertTo(filters[i], CV_8U, 255.0);
    }

    int row_gap = img.rows / ((int)sqrt(point_count));
    int col_gap = img.cols / ((int)sqrt(point_count));

    for(int i = 0; i < img.rows; i += row_gap){
        for(int j = 0; j < img.cols; j += col_gap){
            if (i + window_size < img.rows && j + window_size < img.cols)
                result.push_back(getVector(i, j));
        }
    }

    output_result(output);

    return 0;
}