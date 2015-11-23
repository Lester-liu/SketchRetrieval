#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

const int lines = 1000;
const int size = 128;
string input, bin_path, output;

int line_number;
uint8_t *result;

bool same_color(unsigned char *val){
    int i = 1;
    int k = (int)(val[0]);
    while(i < size) {
        if (((int)val[i]) != k)
            return false;
        i++;
    }
    return true;
}


void get_vectors(string file){
    ifstream in(file);
    uint8_t* val = new uint8_t[size];

    while(line_number < lines && (!in.eof())){
        in.read((char*)val,size);
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

    stringstream ss;
    ss << argv[1];
    ss >> input;
    ss.clear();
    ss << argv[2];
    ss >> bin_path;
    ss.clear();
    ss << argv[3];
    ss >> output;

    ifstream in(input);
    int tmp;
    line_number = 0;
    string file_name;

    result = new uint8_t[lines * size];

    while(line_number < lines){
        in >> file_name >> tmp;
        file_name = file_name.substr(0,file_name.length()-4);
        get_vectors(bin_path + file_name);
    }

    ofstream out(output);

    out.write((char*)&line_number, sizeof(int));
    out.write((char*)&size, sizeof(int));
    out.write((char*)result, sizeof(uint8_t) * size * line_number);

    out.close();

    delete[] result;
    return 0;
}