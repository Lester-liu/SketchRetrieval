//
// Created by lyx on 10/11/15.
//

#ifndef SKETCH_ML_H
#define SKETCH_ML_H

#include "../structure/blob.h"

/*
 * Abstract class for Machine Learning methods
 */
class ML {

private:

    Blob *training; // training set
    Blob *validation; // validation set
    Blob *testing; // testing set
    Blob *label; // correct label information
    Blob *prediction; // prediction of the system

public:

    ML();
    ML(Blob *training, Blob *validation, Blob *testing, Blob *label, Blob *prediction);
    ~ML();

    virtual void train() = 0; // main training method
    virtual void predict() = 0; // prediction method

};

#endif //SKETCH_ML_H
