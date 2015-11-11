//
// Created by lyx on 10/11/15.
//

#include "blob.h"

using namespace std;

Blob::Blob() {

    this->data = NULL;
    this->size = Dim();

}

Blob::Blob(Dim size) {

    this->size = Dim(size);
    this->data = new float[size.get_size()];

    fill(this->data, this->data + size.get_size(), 0);

}

Blob::Blob(float *data, Dim size) {

    this->size = Dim(size);
    this->data = new float[size.get_size()];

    for (int i = 0; i < size.get_size(); i++)
        this->data[i] = data[i];

}

Blob::~Blob() {

    delete[] this->data;

}