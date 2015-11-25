/*
 * Database generator
 *
 * Database generator creates the tf-idf index table with a set of given documents.
 * By calculating the idf value of each word in the document set and the tf value of each word in each file,
 * we can get the keyword of each file.
 *
 * Parameters:
 *      i: input folder containing translated images (one file per model)
 *      o: database with TF-IDF value for all pair of word and image
 *      k: number of word
 *
 * N.B. The translated image file per model has the following format
 *
 *      Image_Count (32 bits integer) Dim (32 bits integer)
 *      Image_1 (Dim * 32 bits integer)
 *      ...
 *      Image_2 (Dim * 32 bits integer)
 *
 *      The Dim represent the number of selected point or the number of local features of an image
 *
 * Output format: Database of TF-IDF value:
 *
 *      Image_Count (32 bits integer) Word_Count (32 bits integer)
 *      IDF_1 ... (32 bits float)
 *      Value_Image_1_Word_1 ...
 *      ...
 *      Value_Image_i_Word_1 ... Value_Image_i_Word_j ... (32 bits float)
 *      ...
 *
 * Usage:
 *      build_database -i [input_path] -o [output_database_path] -k [number of word]
 */

#include <iostream>
#include <fstream>
#include <dirent.h>
#include <set>
#include <vector>
#include <cmath>

using namespace std;

int word_count, view_count; // total number of image
string input, output;
vector<float> idf;
vector<vector<float> > dict; // tf-idf values

//read one file
void read_file(string file_name){
    set<int> word_set;
    vector<float> tf(word_count);

    ifstream input(file_name);

    int image_count, dim, tmp; // number of image per model
    input.read((char*)&image_count, sizeof(int));
    input.read((char*)&dim, sizeof(int));
    view_count += image_count;

    for(int i = 0; i < image_count; i++){
        word_set.clear();
        for(int j = 0; j < word_count; j++)
            tf[j] = 0;

        //calculate the local tf value
        for(int j = 0; j < dim; j++){
            input.read((char*)&tmp, sizeof(int));
            word_set.insert(tmp);
            tf[tmp]++;
        }
        dict.push_back(tf);

        //update the occurence of each word
        for(auto j: word_set)
            idf[j]++;
    }
}

int main(int argc, char** argv) {

    DIR *dir;
    struct dirent *ent;

    if((dir = opendir(input.c_str()))==NULL)
        return EXIT_FAILURE;

    //treat each file
    while((ent = readdir(dir))!=NULL){
        read_file(ent->d_name);
    }

    //calculate the idf value
    for(int i = 0; i < word_count; i++)
        idf[i] = idf[i] == 0 ? 0 : log((float)view_count / idf[i]);

    //calculate the tf-idf value
    for(int i = 0; i < view_count; i++){
        for(int j = 0; j < word_count; j++){
            dict[i][j] *= idf[j];
        }
    }
    return EXIT_SUCCESS;
}