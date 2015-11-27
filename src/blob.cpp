//
// Created by lyx on 10/11/15.
//

#include "blob.h"

using namespace std;

Blob::Blob() {

    this->data = NULL;
    this->dim = Dim();

}

Blob::Blob(Dim dim) {

    this->dim(dim);
    this->data = new float[dim.size()];

    fill(this->data, this->data + dim.size(), 0);

}

Blob::Blob(float *data, Dim dim) {

    this->dim(dim);
    this->data = new float[dim.size()];

    for (int i = 0; i < dim.size(); i++)
        this->data[i] = data[i];

}

Blob::Blob(const Blob &blob) : Blob(blob.data, blob.dim) { }

Blob::~Blob() {

    delete[] this->data;

}

int Blob::row() const {
    return dim.x;
}

int Blob::col() const{
    return dim.y;
}

float Blob::at(int x, int y, int z) const {
    return data[dim.get_index(x, y, z)];
}

/*void Blob::operator()(const Blob& blob) {

}*/