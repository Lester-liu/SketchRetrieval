/*
 * Database generator
 *
 * Database generator creates the tf-idf index table with a set of given documents.
 * By calculating the idf value of each word in the document set and the tf value of each word in each file,
 * we can get the key word of each file.
 *
 * Parameters:
 *      i: input folder
 *      o: output file
 *
 * Usage:
 *      database -i [input_path] -o [output_path]
 */

#include <iostream>
#include "dirent.h"
#include<bits/stdc++.h>
using namespace std;

string input;
map<int,int> words;
vector<map<int> > dic;

void readfile(string file_name){

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