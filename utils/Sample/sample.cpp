/* Sample
 *
 * Sample selects qualified vectors among vectors of each view.
 * (qualified vector is the one who has different colors)
 *
 * Parameters:
 *      txt file with sorted view files' names
 *      folder of views' binary files
 *      output file with selected vectors
 *
 * Output format:
 *      N (32 bits integer):number of vectors   m (32 bits integer):size of vectors
 *      vectors(32 bits float)
 *
 * Usage:
 *      sample [txt file path] [folder path] [output path]
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
const int lines = 25000;
const int size = 512;

string input, bin_path, output;
int line_number;
float *result;

bool same_color(float *val){
    int i = 1;
    float k = val[0];
    while(i < size) {
        //cout << val[i] << ' ';
        if (abs(val[i] - k) > 5.0)
            return false;
        i++;
    }
    return true;
}


void get_vectors(string file){
    ifstream in(file);

    int a,b;
    if(!in.read((char*)&a,sizeof(int)))
        return;
    if(!in.read((char*)&b, sizeof(int)))
        return;

    int k = 0;
    float* val = new float[size];
    while(line_number < lines && (!in.eof()) && (k < lines)){
        in.read((char*)val,size * sizeof(float));
        k++;
        if (!same_color(val)){
            for(int i = 0; i < size; i++)
                result[line_number * size + i] = val[i];
            line_number++;
        }
    }
    delete[] val;
    in.close();
}


int main(int argc, char** argv) {

    input = argv[1];
    bin_path = argv[2];
    output = argv[3];

    ifstream in(input);
    int tmp;
    line_number = 0;
    string file_name;

    result = new float[lines * size];

    while(line_number < lines && (!in.eof())){
        in >> file_name >> tmp;
        file_name = file_name.substr(0, file_name.length() - 4);
        get_vectors(bin_path + file_name +".bin");
    }

    ofstream out(output);

    out.write((char*)&line_number, sizeof(int));
    out.write((char*)&size, sizeof(int));
    out.write((char*)result, sizeof(float) * size * line_number);

    out.close();
    cout << output << endl;

    delete[] result;
    return EXIT_SUCCESS;
}