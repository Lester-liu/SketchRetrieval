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
 *      o: output file
 *      a: picture_path
 *      m: number of pictures to read
 *      d: draw gabor or not
 *
 * Usage:
 *      gabor [-k k] [-p number of points] [-n n] [-s sigma] [-t theta] [-l lambda] [-b beta] -i [Path_to_input] -o [Path_to_output] -a [picture_path] [-d on/off]
 */

#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

string input, output, picture_path;

int kernel_size = 15;
int k = 8; // number of directions for Gabor filter
int window_size = 8; // local feature area (not size)
int point_per_row = 28;
int picture_number = 1;
double sigma = 4;
double theta = 0;
double lambda = 10.0;
double beta = 0.5;
bool draw_gabor = false;

vector<Mat> filters(k); // k filters
vector<float*> result; // matrix representing the image

// get local feature for a specific point
float* get_vector(int x, int y) {

    float* res = new float[k * window_size * window_size]; // size of one local feature

    int d = 0;
    for(int i = 0; i < k; i++) // iterate over all filters
        for(int u = 0; u < window_size; u++)
            for(int v = 0; v < window_size; v++)
                res[d++] = filters[i].at<float>(u + x, v + y);

    // return a pointer, need to call delete
    return res;

}


void save(string output) {

    ofstream out(output);
    int lines = result.size(); // number of local features per image
    int dimension = k * window_size * window_size;

    out.write((char*)&lines, sizeof(int));
    out.write((char*)&dimension, sizeof(int));

    for(float* v: result)
        out.write((char*)v, sizeof(float) * dimension);

    out.close();

    for(float* v: result)
        delete[] v;
}

void show_help() {
    printf("Gabor filter\n"
                   "\n"
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
                   "     gabor -k [k] -n [n] -s [sigma] -t [theta] -l [lambda] -b [beta] -i [Path_to_input] -o [Path_to_output]\n");

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
                point_per_row = atoi(argv[++i]);
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
            case 'a':
                picture_path = argv[++i];
                break;
            case 'm':
                picture_number = atoi(argv[++i]);
                break;
            case 'd':
                if (atoi(argv[++i]) == 1)
                    draw_gabor = true;
                break;

        }
        i++;
    }
    if (input == "" || output == "" || picture_path == "") { // invalid file name
        show_help();
        return false;
    }
    return true;
}

void show_mat(Mat &m){
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++)
            cout << m.at<float>(i, j) << ' ';
        cout << endl;
    }
}

void show_middle_row(Mat &m){
    for(int i = 0; i < m.cols; i++)
        cout << m.at<float>(m.rows / 2, i) << ' ';
    cout << endl;
}

void convertTo(Mat & a, Mat &b){
    double maxi, mini;
    minMaxLoc(a,&mini, &maxi);
    for(int i =  0; i < a.rows; i++){
        for(int j = 0; j < a.cols; j++){
            b.at<uchar>(i,j) = (uchar)((a.at<float>(i,j)-mini)/(maxi - mini) * 255.0);
        }
    }
}

int main(int argc, char** argv) {

    if (!parse_command_line(argc, argv))
        return 0;

    ifstream in(input);
    string picture_name;

    int tmp;

    //calculate of kernels
    vector<Mat> kernels;
    double step = CV_PI / k;
    for(int i = 0; i < k; i++) {
        Mat kernel = getGaborKernel(Size(kernel_size, kernel_size), sigma,
                                    theta + step * (double) i, lambda, beta, CV_PI * 0.5, CV_32F);
        kernels.push_back(kernel);
        if (draw_gabor){
            Mat kernel_grey(Size(kernel_size,kernel_size),CV_8U), heatkernel;

            convertTo(kernel,kernel_grey);
            applyColorMap(kernel_grey, heatkernel, COLORMAP_HSV);
            string output_path = output+"kernel" + to_string(i) +".png";
            //cout << output_path << endl;
            imwrite(output_path,heatkernel);
        }
    }

    for(int pictures  = 0; pictures < picture_number; pictures++) {
        in >> picture_name >> tmp;

        //cout << input <<' ' << output <<' ' << picture_path << ' ' << picture_name << endl;

        Mat img = imread(picture_path + picture_name, CV_LOAD_IMAGE_GRAYSCALE);
        Mat src;
        img.convertTo(src, CV_32F);


        // build all Gabor filters
        for (int i = 0; i < k; i++) {

            filter2D(src, filters[i], -1, kernels[i], Point(-1, -1), 0, BORDER_DEFAULT);

            if (draw_gabor){
                Mat filter_grey(Size(filters[i].cols,filters[i].rows), CV_8U), filter_heat;

                convertTo(filters[i],filter_grey);

                applyColorMap(filter_grey, filter_heat, COLORMAP_HSV);
                string output_path = output + "filter" + to_string(i) + ".png";
               // cout << output_path << endl;
                imwrite(output_path, filter_heat);
            }
        }



        // uniformly distributed points
        int row_gap = (img.rows - window_size) / point_per_row;
        int col_gap = (img.cols - window_size) / point_per_row;

        /*int line_count = (img.rows - window_size)/row_gap * (img.cols - window_size) / col_gap;
        Mat gabor(Size(window_size * window_size * k, line_count), CV_32F);
        int line_index = 0;
*/
        for (int i = 0; i < img.rows; i += row_gap)
            for (int j = 0; j < img.cols; j += col_gap)
                if (i + window_size < img.rows && j + window_size < img.cols)
                    result.push_back(get_vector(i, j)); // append new local feature
    }

    if (!draw_gabor)
        save(output);
    cout << input << " Done!" << endl;
    return 0;
}