#include <iostream>
#include<sstream>
#include<dirent.h>
#include<fstream>

using namespace std;
string path,name,output;
int k, dim, N;
float *result;

void merge(string filename){
    if (!ifstream(filename))
        return;
    ifstream input(filename);
    int n,size;
    input.read((char*)&n,sizeof(int));
    input.read((char*)&size,sizeof(int));
    float tmp;
    for(int i = 0; i < n*size; i++){
        input.read((char*)&tmp,sizeof(float));
        result[k++] = tmp;
    }
}
int main(int argc, char** argv) {
    stringstream ss;
    ss << argv[1];
    ss >> path;
    ss.clear();

    ss << argv[2];
    ss >> name;
    ss.clear();

    ss << argv[3];
    ss >> output;

    N = atoi(argv[4]);
    dim = atoi(argv[5]);

    cout << N << ' ' << dim << endl;

    DIR *dir;
    struct dirent *ent;
    result = new float[N * dim];
    k = 0;
    if ((dir = opendir(path.c_str())) != NULL){
        while((ent = readdir(dir)) != NULL){
            string filename = ent->d_name;
            cout << path + filename + "/" + name << endl;
            merge(path + filename + "/" + name);
        }
    }

    k;
    ofstream out(output);
    int lines = k/dim;
    out.write((char*)&lines,sizeof(int));
    out.write((char*)&dim,sizeof(int));
    out.write((char*)result,sizeof(float) * k * dim);
    out.close();

    return 0;
}