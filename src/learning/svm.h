//
// Created by lyx on 10/11/15.
//

#ifndef SKETCH_SVM_H
#define SKETCH_SVM_H

#include "ml.h"

class SVM: ML {

public:

    SVM();
    SVM(Blob *training, Blob *validation, Blob *testing, Blob *label, Blob *prediction);
    ~SVM();

    void train();
    void predict();

};


#endif //SKETCH_SVM_H
