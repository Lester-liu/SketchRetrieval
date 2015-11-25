//
// Created by lyx on 25/11/15.
//

#ifndef SKETCH_TF_IDF_H
#define SKETCH_TF_IDF_H

#include "clusters.h"

using namespace std;

/*
 * The class TF_IDF simulates a real database, given a new document with word frequency (TF), it looks into its
 * database and find the most similar document. We use the cosines method to define the similarity.
 */
class TF_IDF {

private:
    int word_count; // number of words
    Clusters tf_idf;
    float *idf; // IDF value of each word

public:
    TF_IDF(string database_file);
    virtual ~TF_IDF();

    int find_center(Blob& tf_value); // given Tf vector, find the nearest document

};


#endif //SKETCH_TF_IDF_H
