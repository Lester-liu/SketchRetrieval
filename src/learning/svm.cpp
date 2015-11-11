//
// Created by lyx on 10/11/15.
//

#include "svm.h"

SVM::SVM(): ML() { }

SVM::SVM(Blob *training, Blob *validation, Blob *testing, Blob *label, Blob *prediction):
        ML(training, validation, testing, label, prediction) { }

SVM::~SVM() { }

void SVM::train() {

}

void SVM::predict() {

}