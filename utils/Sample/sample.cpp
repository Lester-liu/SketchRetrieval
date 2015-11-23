#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <opencv2/highgui/highgui.hpp>
#include <set>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

const int lines = 1000;
const int file_number = 10;
const int n = 100;
const int size = 128;
string input, binpath,output;


set<string> result;

bool totalBlack(string val, int size){
    int i = 0;
    while(i < size) {
        //cout << (int) ((unsigned char) val[i]) << ' ';
        if ((int) ((unsigned char) val[i]) != 0)
            return false;
        i++;
    }
    return true;
}


void getVectors(string file){
    ifstream in(file.c_str());
    int k = 0;
    string s;
    while(k < n && getline(in,s)){
        if (!totalBlack(s,size)){
            result.insert(s);
            k++;
        }
    }
    in.close();
}

int main(int argc, char** argv) {

    stringstream ss;
    ss << argv[1];
    ss >> input;
    ss.clear();
    ss << argv[2];
    ss >> binpath;
    ss.clear();
    ss << argv[3];
    ss >> output;

    ifstream in(input);
    string s;
    int size;
    for( int i = 0; i < file_number; i++){
        in >> s >> size;
        s = s.substr(0,s.length()-4);
        //cout << binpath+s << endl;
        getVectors(binpath+s);
    }

    ofstream out(output.c_str());

    cout << result.size() <<' ';
    for(string s:result){
        out << s <<endl;
    }
    out.close();
    return 0;
}