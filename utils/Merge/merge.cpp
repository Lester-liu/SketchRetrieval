#include <iostream>
#include <dirent.h>
#include <fstream>

using namespace std;

string path, name, output;
int k, dim, N;
float *result;

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

    path = argv[1];
    output = argv[2];
    N = atoi(argv[3]);
    dim = atoi(argv[4]);

    cout << N << ' ' << dim << endl;

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

    cout << ' ' << k << endl;
    ofstream out(output);
    int lines = k / dim;
    out.write((char*)&lines, sizeof(int));
    out.write((char*)&dim, sizeof(int));
    out.write((char*)result, sizeof(float) * k);
    out.close();

    return EXIT_SUCCESS;
}