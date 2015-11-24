/*
 * Database generator
 *
 * Database generator creates the tf-idf index table with a set of given documents.
 * By calculating the idf value of each word in the document set and the tf value of each word in each file,
 * we can get the keyword of each file.
 *
 * Parameters:
 *      i: input folder containing translated images
 *      o: database with TF-IDF value for all pair of word and image
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
 *      build_database -i [input_path] -o [output_database_path]
 */

#include <iostream>
#include <dirent.h>
#include <map>
#include <vector>

using namespace std;

string input, output;
map<int, int> words;
vector<map<int> > dict;

void read_file(string file_name){

}

int main() {
    DIR *dir;
    struct dirent *ent;

    if((dir = opendir(input.c_str()))==NULL)
        return 0;

    while((ent = readdir(dir))!=NULL){

    }

    return 0;
}