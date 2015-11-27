/*
 * Merge Binary Files
 *
 * Merge will merge all the binary files in a folder into one binary file
 *
 * Parameters:
 *      a: [Folder_path]
 *      o: [Output_file_name]
 *      n: [Number_of_total_lines]
 *      d: [Dimension of each line]
 *
 * Usage:
 *      merge -a [Folder_path] -o [Output_file_name] -n [Number_of_total_lines] -d [Dimension]
 */

#include <iostream>
#include <dirent.h>
#include <fstream>

using namespace std;

string path, name, output;
int k, dim, N;
float *result;

bool parse_command_line(int argc, char **argv) {
    int i = 1;
    while(i < argc) {
        if (argv[i][0] != '-')
            break;
        switch(argv[i][1]) {
            case 'n':
                N = atoi(argv[++i]);
                break;
            case 'a': // input file
                path = argv[++i];
                break;
            case 'o': // output file
                output = argv[++i];
                break;
            case 'd':
                dim = atoi(argv[++i]);
                break;

        }
        i++;
    }
    if (path == "" || output == "" ) { // invalid file name
        return false;
    }
    return true;
}

void merge(string filename){

    if (!ifstream(filename))
        return;
    ifstream input(filename);
    int n, size;
    if (!input.read((char*)&n, sizeof(int)))
        return;
    if (!input.read((char*)&size, sizeof(int)))
        return;
    float tmp;
    //cout << n <<' ' << size;
    for(int i = 0; i < n * size; i++){
        input.read((char*)&tmp, sizeof(float));
        result[k++] = tmp;
    }

}
int main(int argc, char** argv) {

    if (!parse_command_line(argc, argv))
        return EXIT_FAILURE;

    DIR *dir;
    struct dirent *ent;
    result = new float[N * dim];
    k = 0;
    if ((dir = opendir(path.c_str())) != NULL){
        while((ent = readdir(dir)) != NULL){
            string filename = ent->d_name;
            //cout << path + "/" + filename << endl;
            merge(path + "/" + filename);
        }
    }

    ofstream out(output);
    int lines = k / dim;
    out.write((char*)&lines, sizeof(int));
    out.write((char*)&dim, sizeof(int));
    out.write((char*)result, sizeof(float) * k);
    out.close();

    return EXIT_SUCCESS;
}