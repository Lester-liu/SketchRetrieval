//
// Created by lyx on 10/11/15.
//

#ifndef SKETCH_ML_H
#define SKETCH_ML_H

#include "../structure/blob.h"

class ML {

private:

    Blob* training;
    Blob* validation;
    Blob* testing;
    Blob* label;
    Blob* prediction;

public:

    ML();
    ML(Blob* training, Blob* validation, Blob* testing, Blob* label, Blob* prediction);
    ~ML();

    virtual void train() = 0;
    virtual void predict() = 0;
};


#endif //SKETCH_ML_H
