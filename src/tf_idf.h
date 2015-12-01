//
// Created by lyx on 25/11/15.
//

#ifndef SKETCH_TF_IDF_H
#define SKETCH_TF_IDF_H

#include <fstream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

/*
 * The class TF_IDF simulates a real database, given a new document with word frequency (TF), it looks into its
 * database and find the most similar document. We use the cosines method to define the similarity.
 */
class TF_IDF {

private:
    int word_count; // number of words
    int document_count; // number of documents in the databases
    float *tf_idf;
    float *idf; // IDF value of each word
    float scalor_product(float *a, float *b,int n);

public:
    TF_IDF();
    TF_IDF(string database_file);

    virtual ~TF_IDF();

    vector<pair<float,int> > find_nearest(int *tf_value); // given Tf vector, find the nearest document
};


#endif //SKETCH_TF_IDF_H
