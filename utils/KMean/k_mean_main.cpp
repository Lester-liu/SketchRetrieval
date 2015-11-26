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
 *      k_mean -f [Path_to_file] -k Number_of_center -d [Path_to_output_file] -s [8|32] -i [Iteration] -v [Variation]
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
 *      k_mean -f [Path_to_input] -t [Path_to_output] -d [Path_to_dictionary] -s [8|32]
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
 *      k_mean -f [Path_to_folder] -d [Path_to_dictionary] -s [8|32] -c [case_number] -a [data_size] -o [output_file]
 */

#include "k_mean.h"
#include <dirent.h>

using namespace k_mean;

enum Mode {Group_Testing, Testing, Training}; // Group_Testing for usage 3, Testing for usage 2, Training for usage 1
enum Format {Integer, Float}; // format of initial data

Mode mode = Group_Testing;
Format format = Integer;

string input, output, dictionary, output_file;
float *data;
int center_count = 0;
int data_count = 0;
int dim = 0;
int cases = 0;

int iteration = 0;
float delta = 0;

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
            case 'h': // help
                show_help();
                return false;
            case 's': // format
                format = (argv[++i] == "8") ? Integer : Float;
                break;
            case 'k': // training mode
                center_count = atoi(argv[++i]);
                mode = Training;
                break;
            case 'd': // dictionary file
                dictionary = argv[++i];
                break;
            case 'f': // input file
                input = argv[++i];
                break;
            case 't': // output file
                output = argv[++i];
                if (mode != Training)
                    mode = Testing;
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
    if (format == Integer) {
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
            if (format == Integer) {
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

void testing() {

    cout << "Testing" << endl;

    ifstream in(input);
    read_int(in, &data_count); // read meta-info
    read_int(in, &dim);

    // read data
    if (format == Integer) {
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
        default:
            return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}