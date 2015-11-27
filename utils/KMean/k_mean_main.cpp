/*
 * K-means clustering
 *
 * This program implements the K-Means clustering method. It takes a set of points (lines in a binary file) and
 * returns the group number of each point and an extra file to describe the group, namely the group center.
 *
 * Parameters:
 *      f: local features to be clustered
 *      k: number of cluster
 *      d: path to the output dictionary
 *      s: size of the value: 8 for uint8_t, 32 for float
 *      i: number of iteration
 *      v: value of variation
 *
 * Usage 1 (Training):
 *      k_mean -1 -f [Path_to_file] -k Number_of_center -d [Path_to_output_file] -s [8|32] -i [Iteration] -v [Variation]
 *
 * N.B. The file format is very specific, it is a binary file with integers and floats, so please pay attention to the
 * big / little endian problem. You may want to generate the file by program in case of theses sorts of problems.
 * The first number is the number of points: N, the second is the dimension of the point: d. Then there should
 * be N * d float numbers after. So the binary file looks like:
 *
 *      N (32 bits integer) d (32 bits integer)
 *      P_1 (d * 32 bits float or d * 8 bits integer)
 *      ...
 *      P_N (d * 32 bits float or d * 8 bits intege)
 *
 * Parameters:
 *      f: Gabor features of an image file (line + dim + data in float)
 *      t: output binary file (vector of index (int))
 *      d: dictionary to be used
 *      s: size of the value: 8 for uint8_t, 32 for float
 *
 * Usage 2 (Testing):
 *      k_mean -2 -f [Path_to_input] -t [Path_to_output] -d [Path_to_dictionary] -s [8|32]
 *
 * Parameters:
 *      f: file containing all Gabor features files name (cases + line per case + names, text file)
 *      d: dictionary to be used
 *      s: size of the value: 8 for uint8_t, 32 for float
 *      c: case number
 *      a: data size (number of features in an image)
 *      o: output file (a binary file for all files in the group)
 *
 * Output: translated images:
 *
 *      Line_Count (32 bits integer) Dimension (32 bits integer)
 *      Image_1 (Dimension * 32 bits int)
 *      ...
 *      Image_Line_Count (Dimension * 32 bits)
 *
 * Usage 3 (Group_Testing):
 *      k_mean -3 -f [Path_to_folder] -d [Path_to_dictionary] -s [8|32] -c [case_number] -a [data_size] -o [output_file]
 *
 * Parameters:
 *      f: input file containing the name of all views
 *      r: folder containing all contour image
 *      d: path to the dictionary
 *      s: size of the value
 *      o: output encoded file
 *      c: number of image to be considered
 *      a: data size (number of features in an image)
 *
 * Input file:
 *      Name_i Importance_i
 *
 * Usage 4 (Group_Testing_From_Contour):
 *      k_mean -4 -f [Path_to_file] -r [Folder_to_contour] -d [Path_to_dictionary] -s [8|32] -o [output_file] -c [Cases] -a [data_size]
 */

#include "k_mean.h"
#include <dirent.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace k_mean;
using namespace std;
using namespace cv;

enum Mode {Group_Testing, Testing, Training, Contour_Testing}; // Group_Testing for usage 3, Testing for usage 2, Training for usage 1
enum Format {Integer, Float}; // format of initial data

Mode mode = Contour_Testing;
Format data_format = Integer;

string input, output, dictionary, output_file, root_folder;
float *data;
int center_count = 0;
int data_count = 0;
int dim = 0;
int cases = 0;

int iteration = 0;
float delta = 0;


int kernel_size = 15;
int k = 8; // number of directions for Gabor filter
int window_size = 8; // local feature area (not size)
int point_per_row = 28;
double sigma = 4;
double theta = 0;
double lambda = 10.0;
double beta = 0.5;



void show_help() {

}

/*
 * Process all arguments
 */
bool parse_command_line(int argc, char **argv) {

    int i = 1;
    while(i < argc) {
        if (argv[i][0] != '-')
            break;
        switch(argv[i][1]) {
            case '1':
                mode = Training;
                break;
            case '2':
                mode = Testing;
                break;
            case '3':
                mode = Group_Testing;
                break;
            case '4':
                mode = Contour_Testing;
                break;
            case 'h': // help
                show_help();
                return false;
            case 's': // data_format
                data_format = (argv[++i] == "8") ? Integer : Float;
                break;
            case 'k': // training mode
                center_count = atoi(argv[++i]);
                break;
            case 'd': // dictionary file
                dictionary = argv[++i];
                break;
            case 'f': // input file
                input = argv[++i];
                break;
            case 't': // output file
                output = argv[++i];
                break;
            case 'v': // variation rate
                delta = float(atof(argv[++i]));
                break;
            case 'i': // number of iteration
                iteration = atoi(argv[++i]);
                break;
            case 'c':
                cases = atoi(argv[++i]);
                break;
            case 'a':
                data_count = atoi(argv[++i]);
                break;
            case 'o':
                output_file = argv[++i];
                break;
            case 'r':
                root_folder = argv[++i];
                break;
        }
        i++;
    }
    if (input.length() == 0) { // invalid file name
        show_help();
        return false;
    }
    return true;
}

void training() {

    cout << "Training" << endl;

    ifstream in(input);
    read_int(in, &data_count); // read meta-info
    read_int(in, &dim);

    // read data
    if (data_format == Integer) {
        uint8_t *_data = new uint8_t[dim * data_count];
        read_bytes(in, _data, dim * data_count);

        data = new float[dim * data_count];
        for (int i = 0; i < dim * data_count; i++)
            data[i] = float(_data[i]);

        delete[] _data;
    }
    else {
        data = new float[dim * data_count];
        read_floats(in, data, dim * data_count);
    }

    in.close();
    cout << data_count << ' ' << dim << ' ' << center_count << endl;

    // train the model
    K_Mean model(data, data_count, dim, center_count);
    model.execute(iteration, delta);

    // save the dictionary
    model.save(dictionary, false);

    delete[] data;

}

void group_testing() {

    cout << "Group Testing" << endl;

    DIR *dir;
    struct dirent *ent;

    // read the dictionary
    ifstream dict(dictionary);
    read_int(dict, &center_count); // read meta-info
    read_int(dict, &dim); // should be the same

    float *center = new float[dim * center_count];
    read_floats(dict, center, dim * center_count);

    dict.close();

    data = new float[dim * data_count];
    int *allocation = new int[data_count];

    // prepare the output file
    ofstream out(output_file);
    out.write((char*)&cases, sizeof(int));
    out.write((char*)&data_count, sizeof(int));

    K_Mean model(data, center, data_count, dim, center_count);

    string file;
    if ((dir = opendir(input.c_str()))!=NULL) {
        while ((ent = readdir(dir)) != NULL) {
            file = ent->d_name;
            if (file[0] != 'm')
                continue;
            ifstream f(input + file);
            cout << input + file << endl;

            read_int(f, &data_count); // read meta-info
            read_int(f, &dim);

            // read data
            if (data_format == Integer) {
                uint8_t *_data = new uint8_t[dim * data_count];
                read_bytes(f, _data, dim * data_count);

                for (int i = 0; i < dim * data_count; i++)
                    data[i] = float(_data[i]);

                delete[] _data;
            }
            else {
                read_floats(f, data, dim * data_count);
            }

            f.close();

            // replace the data with new image
            model.update_data();
            // translate the local features into words
            model.translate(allocation);

            out.write((char *) allocation, sizeof(int) * data_count);
        }
    }

    out.close();

    delete[] center;
    delete[] data;
    delete[] allocation;

}

void contour_testing() {

    cout << "Contour Testing" << endl;

    ifstream in(input);

    // read the dictionary
    ifstream dict(dictionary);
    read_int(dict, &center_count); // read meta-info
    read_int(dict, &dim); // should be the same

    float *center = new float[dim * center_count];
    read_floats(dict, center, dim * center_count);

    dict.close();

    data = new float[dim * data_count];
    int *allocation = new int[data_count];

    // prepare the output file
    ofstream out(output_file);
    out.write((char*)&cases, sizeof(int));
    out.write((char*)&data_count, sizeof(int));

    K_Mean model(data, center, data_count, dim, center_count);

    string file;
    int tmp;
    int d = 0;

    //calculate of kernels
    double step = CV_PI / k;
    vector<Mat> kernels;
    for(int i = 0; i < k; i++) {
        Mat kernel = getGaborKernel(Size(kernel_size, kernel_size), sigma,
                                    theta + step * (double) i, lambda, beta, CV_PI * 0.5, CV_32F);
        kernels.push_back(kernel);
    }

    for (int z = 0; z < cases; z++) {
        in >> file >> tmp;

        vector<Mat> filter(k);
        // read image
        Mat img = imread(root_folder + file, CV_LOAD_IMAGE_GRAYSCALE);
        img.convertTo(img,CV_32F);


        //use gabor filter
        for(int i = 0; i < k; i++){
            filter2D(src, filters[i], -1, kernels[i], Point(-1, -1), 0, BORDER_DEFAULT);
        }

        // compute the new value

        // replace the data with new image
        int row_gap = (img.rows - window_size) / point_per_row;
        int col_gap = (img.cols - window_size) / point_per_row;

        for(int i = 0; i < img.rows - window_size; i += row_gap){
            for(int j = 0; j < img.cols - window_size; j += col_gap){
                for(int kk = 0; kk < k; kk++){
                    for(int u = 0; u < window_size; u++){
                        for(int v = 0; v < window_size; v++){
                            data[d++] = filters[kk].at<float>(u + i, v + j);
                        }
                    }
                }
            }
        }

        model.update_data();
        // translate the local features into words
        model.translate(allocation);

        out.write((char *) allocation, sizeof(int) * data_count);

    }

    in.close();
    out.close();

    delete[] center;
    delete[] data;
    delete[] allocation;

}

void testing() {

    cout << "Testing" << endl;

    ifstream in(input);
    read_int(in, &data_count); // read meta-info
    read_int(in, &dim);

    cout << data_count << ' ' << dim << endl;

    // read data
    if (data_format == Integer) {
        uint8_t *_data = new uint8_t[dim * data_count];
        read_bytes(in, _data, dim * data_count);

        data = new float[dim * data_count];
        for (int i = 0; i < dim * data_count; i++)
            data[i] = float(_data[i]);

        delete[] _data;
    }
    else {
        data = new float[dim * data_count];
        read_floats(in, data, dim * data_count);
    }

    in.close();

    // read the dictionary
    ifstream dir(dictionary);
    read_int(dir, &center_count); // read meta-info
    read_int(dir, &dim); // should be the same

    float *center = new float[dim * center_count];
    read_floats(dir, center, dim * center_count);

    dir.close();

    // build the model
    K_Mean model(data, center, data_count, dim, center_count);

    // translate the local features into words
    int *allocation = new int[data_count];
    model.translate(allocation);

    /*
    for (int i = 0; i < data_count; i++)
        cout << allocation[i] << ' ';
    cout << endl;
    */

    ofstream out(output);
    out.write((char*)&dim, sizeof(int));
    out.write((char*)allocation, sizeof(int) * data_count);
    out.close();

    delete[] allocation;
    delete[] data;
    delete[] center;

}

int main(int argc, char **argv) {

    if (!parse_command_line(argc, argv))
        return EXIT_FAILURE;

    srand(time(NULL));

    switch (mode) {
        case Group_Testing:
            group_testing();
            break;
        case Testing:
            testing();
            break;
        case Training:
            training();
            break;
        case Contour_Testing:
            contour_testing();
            break;
        default:
            return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}