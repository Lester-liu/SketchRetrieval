//
// Created by lyx on 25/11/15.
//

#include "tf_idf.h"
#include <fstream>

TF_IDF::TF_IDF() {
    word_count = 0;
    idf = new float[0];
}

TF_IDF::TF_IDF(string database_file) {
    ifstream input(database_file);

    int picture_count;
    input.read((char*)&picture_count, sizeof(int));
    input.read((char*)&word_count, sizeof(int));

    idf = new float[word_count];
    input.read((char*)idf, sizeof(float) * word_count);

    float* data = new float[word_count * picture_count];
    input.read((char*)data, sizeof(float) * word_count * picture_count);

    int d = 0;

    Mat b(Size(word_count,picture_count),CV_32F);
    for(int i = 0; i < picture_count; i++){
        for(int j = 0; j < word_count; j++){
            b.at<float>(i,j) = data[d++];
        }
    }
    tf_idf = b;

    input.close();
}

int TF_IDF::find_nearest(Mat &tf_value) {
    assert(tf_value.rows == 1);
    assert(tf_value.cols == word_count);
    float norm = 0;
    for(int i = 0; i < word_count; i++){
        float tmp = (float)tf_value.at<int>(0,i) * idf[i];
        norm += tmp * tmp;

    }
    norm = sqrt(norm);

    int maxi = -1;
    float max_value = 0;
    for(int i = 0; i < tf_idf.rows; i++){
        float tmp = 0;
        float normi = 0;
        for(int j = 0; j < word_count; j++){
            tmp += tf_idf.at<float>(i,j) * (float)tf_value.at<int>(0,j) * idf[j];
            normi += tf_idf.at<float>(i,j) * tf_idf.at<float>(i,j);
        }
        tmp = tmp / norm / sqrt(normi);
        if (maxi == -1  || tmp > max_value){
            maxi = i;
            max_value = tmp;
        }
    }
    cout << max_value <<endl;
    return maxi;
}

TF_IDF::~TF_IDF() { };

int TF_IDF::get_word_count() {
    return word_count;
}