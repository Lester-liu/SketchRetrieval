/*
 * Database generator
 *
 * Database generator creates the tf-idf index table with a set of given documents.
 * By calculating the idf value of each word in the document set and the tf value of each word in each file,
 * we can get the keyword of each file.
 *
 * Parameters:
 *      i: input folder containing translated images (one file per model)
 *      d: database with TF-IDF value for all pair of word and image
 *      m: txt file indicate the correspond model of each image
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
string input, output_data, output_index;
float* idf;
vector<float*> dict; // tf-idf values
vector<string> models;

//read one file
void read_file(string file_name){

    if (!ifstream(file_name))
        return;

    set<int> word_set;

    ifstream input(file_name);

    int image_count, dim, tmp; // number of image per model

    if (!input.read((char*)&image_count, sizeof(int)))
        return;
    if (!input.read((char*)&dim, sizeof(int)))
        return;
    view_count += image_count;

    for(int i = 0; i < image_count; i++){
        models.push_back(file_name);
        word_set.clear();
        float* tf = new float[word_count];
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

bool parse_command_line(int argc, char **argv) {
    int i = 1;
    while(i < argc) {
        if (argv[i][0] != '-')
            break;
        switch(argv[i][1]) {
            case 'h': // help
                return false;
            case 'k': // threshold flag
                word_count = atoi(argv[++i]);
                break;
            case 'i': // input file
                input = argv[++i];
                break;
            case 'd': // output file
                output_data = argv[++i];
                break;
            case 'm':
                output_index = argv[++i];
                break;
        }
        i++;
    }
    if (input == "" || output_data == "" || output_index == "") { // invalid file name
        return false;
    }
    return true;
}

int main(int argc, char** argv) {

    if (!parse_command_line(argc, argv))
        return EXIT_FAILURE;

    idf = new float[word_count];

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

    ofstream outd(output_data);

    outd.write((char*)&view_count, sizeof(int));
    outd.write((char*)&word_count, sizeof(int));

    for(int i = 0; i < view_count; i++){
        outd.write((char*)dict[i], sizeof(float) * word_count);
        delete[] dict[i];
    }
    outd.close();

    ofstream outi(output_index);
    outi << models.size() <<  endl;
    for(int i = 0; i < models.size(); i++)
         outi << models[i] << endl;
    outi.close();

    return EXIT_SUCCESS;
}