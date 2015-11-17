#include <fstream>
#include <dirent.h>
#include <vector>

using namespace std;

int k = 1000;

string input,output;
string filenames[36];
vector<vector<unsigned char> > result;

bool totalBlack(string s){
    int i = 0;

}

void getVectors(int n, string file){
    ifstream input(file);
    int k = 0, line = 0;
    string lines;
    while(k < n && line < 1000){
        getline(input,lines);

    }

}

int main(int argc, char** argv) {
    DIR *dir;
    struct  dirent *ent;
    int size = 0;
    if ((dir = opendir(input.c_str())) != NULL){
        while((ent  = readdir(dir)) != NULL){
            filenames[size] = ent->d_name;
            size++;
        }
    }

    return 0;
}