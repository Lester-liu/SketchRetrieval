/*
 * K-mean clustering
 *
 * This program implements the K-Mean clustering method. It takes a set of points (lines in a binary file) and
 * returns the group number of each point and an extra file to describe the group, namely the group center.
 *
 * Parameters:
 *      f: local features to be clustered
 *      k: number of cluster
 *      t: path to the output dictionary
 *
 * Usage 1:
 *      k_mean -f [Path_to_file] -k Number_of_center -t [Path_to_output_file]
 *
 * N.B. The file format is very specific, it is a binary file with integers and floats, so please pay attention to the
 * big / little endian problem. You may want to generate the file by program in case of theses sorts of problems.
 * The first number is the number of points: N, the second is the dimension of the point: d. Then there should
 * be N * d float numbers after. So the binary file looks like:
 *
 *      N (32 bits integer) d (32 bits integer)
 *      P_1 (d * 32 bits float)
 *      ...
 *      P_N (d * 32 bits float)
 *
 * Parameters:
 *      f: Gabor features of an image file
 *      t: output binary file
 *      d: dictionary to be used
 *
 * Usage 2:
 *      k_mean -f [Path_to_input] -t [Path_to_output] -d [Path_to_dictionary]
 *
 * Parameters:
 *      f: file containing all Gabor features files name
 *      d: dictionary to be used
 *
 * Usage 3:
 *      k_mean -f [Path_to_file] -d [Path_to_dictionary]
 */

#include "k_mean.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace k_mean;
using namespace cv;

enum Mode {Testing, Training}; // Testing for usage 2, Training for usage 1

Mode mode = Testing;

string input, output;
int center_count, data_count, dim;
float *data;

void test_2d() {

    data_count = 12;
    center_count = 3;
    dim = 2;

    assert(data_count >= center_count);

    data = new float[dim * data_count];
    data[0] = 0;
    data[1] = 0;
    data[2] = 1;
    data[3] = 0;
    data[4] = 1;
    data[5] = 1;
    data[6] = 0;
    data[7] = 1;
    data[8] = 10;
    data[9] = 10;
    data[10] = 11;
    data[11] = 10;
    data[12] = 11;
    data[13] = 11;
    data[14] = 10;
    data[15] = 11;
    data[16] = 10;
    data[17] = 0;
    data[18] = 11;
    data[19] = 0;
    data[20] = 11;
    data[21] = 1;
    data[22] = 10;
    data[23] = 1;

    K_Mean model(data, data_count, dim, center_count);
    model.execute(10, 0.1);

    delete[] data;

}

void test_mnist() {

    input = "t10k-images.idx3-ubyte";
    center_count = 10;

    ifstream file(input);
    int tmp, row, col;
    read_int(file, &tmp);
    read_int(file, &data_count);
    read_int(file, &row);
    read_int(file, &col);
    dim = row * col;
    uint8_t *_data = new uint8_t[dim * data_count];
    read_bytes(file, _data, dim * data_count);
    file.close();

    data = new float[dim * data_count];
    for (int i = 0; i < dim * data_count; i++)
        data[i] = float(_data[i]);

    K_Mean model(data, data_count, dim, center_count);
    model.execute(50, 0.05);

    float *center = new float[dim * center_count];
    model.get_clusters(center);

    for (int i = 0; i < center_count; i++) {
        Mat m(row, col, CV_32F, center + i * dim);
        Mat img;
        m.convertTo(img, CV_8U);
        imshow("Image" + to_string(i), img);
    }
    waitKey(0);

    delete[] center;
    delete[] data;

}

void read_file() {
    ifstream in(input);
    in >> data_count;
    in >> dim;
    data = new float[data_count*dim];
    string s;
    getline(in, s);
    for(int i = 0; i < data_count; i++){
        getline(in, s);
        for(int j = 0; j < s.length(); j++){
            data[dim*i+j] = (float)((int)((unsigned char)s[j]));
        }
    }
}

int main() {
    srand(time(NULL));
    //test_2d();
    test_mnist();
    /*
    input = "test_k_mean.bin";
    read_file();
    for (int i = 0; i < 128; i++) {
        cout << data[i] << endl;
    }
    delete[] data;*/
}