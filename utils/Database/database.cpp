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
 *Output index format:
 *      Model_count Views_per_model
 *      Model_index
 *      ...
 *
 * Usage:
 *      build_database -i [input_path] -d [output_database_path] -m [output_index_path] -k [number of word]
 */

#include <iostream>
#include <fstream>
#include <dirent.h>
#include <set>
#include <vector>
#include <cmath>
#include <cassert>

using namespace std;

int center_count, document_count, image_per_model; // total number of image
string input, output_data, output_index;
float* idf;
vector<vector<int>> tf; // temporary tf values
float* dict; // tf-idf values
vector<string> models;

//read one file
void read_file(string path, string file_name){

    if (file_name[0] == '.')
        return;

    set<int> word_set;

    ifstream input(path + file_name);

    int image_count, feature_count; // number of image per model

    if (!input.read((char*)&image_count, sizeof(int)))
        return;
    if (!input.read((char*)&feature_count, sizeof(int)))
        return;

    image_per_model = image_count;

    int *feature = new int[feature_count];

    for(int i = 0; i < image_count; i++){
        word_set.clear();

        //calculate the local tf value
        vector<int> tf_value(center_count); // use vector is easier than pointer
        fill(tf_value.begin(), tf_value.end(), 0); // init at 0

        input.read((char*) feature, sizeof(int) * feature_count);
        for(int j = 0; j < feature_count; j++){
            word_set.insert(feature[j]);
            tf_value[feature[j]]++; // build tf vector
        }

        tf.push_back(tf_value);

        //update the occurrence of each word
        for(auto j: word_set)
            idf[j]++;

        document_count++;
        assert(document_count == tf.size());
    }

    delete[] feature;

    string name = file_name.substr(1,file_name.find('.')-1);
    models.push_back(name);

    input.close();
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
                center_count = atoi(argv[++i]);
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
    if (input.length() <= 0 || output_data.length() <= 0 || output_index.length() <= 0) { // invalid file name
        return false;
    }
    return true;
}

int main(int argc, char** argv) {

    if (!parse_command_line(argc, argv))
        return EXIT_FAILURE;

    idf = new float[center_count];
    document_count = 0;

    DIR *dir;
    struct dirent *ent;

    if((dir = opendir(input.c_str()))==NULL)
        return EXIT_FAILURE;

    //treat each file
    while((ent = readdir(dir))!=NULL)
        read_file(input, ent->d_name);

    //calculate the idf value
    for(int i = 0; i < center_count; i++) {
        idf[i] = idf[i] == 0 ? 0 : log((float) document_count / idf[i]);
    }

    dict = new float[document_count * center_count];

    //calculate the tf-idf value
    for(int i = 0; i < document_count; i++)
        for(int j = 0; j < center_count; j++)
            dict[i * center_count + j] = tf[i][j] * idf[j];

    // write the dictionary
    ofstream out_dict(output_data);

    out_dict.write((char*)&document_count, sizeof(int));
    out_dict.write((char*)&center_count, sizeof(int));

    out_dict.write((char*)idf, sizeof(float) * center_count);
    out_dict.write((char*)dict, sizeof(float) * document_count * center_count);

    delete[] idf;
    delete[] dict;

    out_dict.close();

    // write down the model number
    ofstream out_index(output_index);
    out_index << models.size() << ' ' << image_per_model <<  endl;
    for(int i = 0; i < models.size(); i++)
         out_index << models[i] << endl;
    out_index.close();

    return EXIT_SUCCESS;
}