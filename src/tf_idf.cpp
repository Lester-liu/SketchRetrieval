//
// Created by lyx on 25/11/15.
//

#include "tf_idf.h"
#include <fstream>

TF_IDF::TF_IDF() {
    word_count = 0;
}

TF_IDF::TF_IDF(string database_file) {
    ifstream input(database_file);
    int picture_count;
    input.read((char*)&picture_count, sizeof(int));
    input.read((char*)&word_count, sizeof(int));

    idf = new float[word_count];
    input.read((char*)idf, sizeof(float) * word_count);

    float *data = new float[word_count * picture_count];
    input.read((char*)data, sizeof(float) * word_count * picture_count);
    tf_idf = Blob(data,Dim(picture_count,word_count,1));
}

int TF_IDF::find_nearest(Blob &tf_value) {
    assert(tf_value.row() == 1);
    assert(tf_value.col() == word_count);
    float norm = 0;
    for(int i = 0; i < word_count; i++){
        float tmp = tf_value.at(0,i,0) * idf[i];
        norm += tmp * tmp;
    }
    norm = sqrt(norm);

    int mini = -1;
    float mini_value = 0;
    for(int i = 1; i < tf_idf.row(); i++){
        float tmp = 0;
        float normi = 0;
        for(int j = 0; j < word_count; j++){
            tmp += tf_idf.at(i,j,0) * tf_value.at(0,j,0) * idf[j];
            normi += tf_idf.at(i,j,0) * tf_idf.at(i,j,0);
        }
        tmp = tmp / norm / sqrt(normi);
        if (mini == -1  || tmp < mini_value){
            mini = i;
            mini_value = tmp;
        }
    }
    return mini;
}

int TF_IDF::get_word_count() {
    return word_count;
}