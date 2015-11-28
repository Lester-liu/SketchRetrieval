//
// Created by lyx on 25/11/15.
//

#include "tf_idf.h"

TF_IDF::TF_IDF() {
    word_count = 0;
    document_count = 0;
    tf_idf = NULL;
    idf = NULL;
}

TF_IDF::TF_IDF(string database_file) {
    ifstream input(database_file);

    input.read((char*)&document_count, sizeof(int));
    input.read((char*)&word_count, sizeof(int)); // dimension of each tf-idf vector

    idf = new float[word_count];
    input.read((char*)idf, sizeof(float) * word_count);

    tf_idf = new float[document_count * word_count];
    input.read((char*)tf_idf, sizeof(float) * document_count * word_count);

    input.close();
}

int TF_IDF::find_nearest(int *tf_value) {

    float norm = 0;
    float tmp = 0;

    for(int j = 0; j < word_count; j++){
        tmp = tf_value[j] * idf[j];
        norm += tmp * tmp;
    }
    norm = sqrt(norm);

    int max_index = -1;
    float max_value = -1;
    for(int i = 0; i < document_count; i++){
        tmp = 0; // dot product
        float norm_i = 0;
        for(int j = 0; j < word_count; j++){
            tmp += tf_idf[i * word_count + j] * tf_value[j] * idf[j];
            norm_i += tf_idf[i * word_count + j] * tf_idf[i * word_count + j];
        }
        tmp = tmp / norm / sqrt(norm_i);
        if (max_index == -1  || tmp > max_value){
            max_index = i;
            max_value = tmp;
        }
    }
    return max_index;
}

TF_IDF::~TF_IDF() {
    if (idf)
        delete[] idf;
    if (tf_idf)
        delete[] tf_idf;
};