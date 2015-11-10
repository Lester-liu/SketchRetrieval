//
// Created by lyx on 10/11/15.
//

#include "ml.h"

ML::ML() {

    this->training = NULL;
    this->validation = NULL;
    this->testing = NULL;
    this->label = NULL;
    this->prediction = NULL;

}

ML::ML(Blob *training, Blob *validation, Blob *testing, Blob *label, Blob *prediction) {

    this->training = training;
    this->validation = validation;
    this->testing = testing;
    this->label = label;
    this->prediction = prediction;

}

ML::~ML() {

    this->training = NULL;
    this->validation = NULL;
    this->testing = NULL;
    this->label = NULL;
    this->prediction = NULL;

}