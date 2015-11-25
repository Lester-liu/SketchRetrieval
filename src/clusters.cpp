//
// Created by lyx on 25/11/15.
//

#include "clusters.h"

Clusters::Clusters(Blob& centers) : centers(centers) { }

Clusters::~Clusters() { }

int Clusters::size() const {
    return centers.dim.x;
}

int Clusters::dimension() const {
    return centers.dim.y;
}

void Clusters::find_center(Blob& vector, int *allocation, int n) const {
    assert(vector.dim.x == n); // x is the number of lines, y is the dimension of vector
    // find neighbor for all instances
    for (int i = 0; i < n; i++)
        allocation[i] = find_center(vector.data + i * vector.dim.y);
}

int Clusters::find_center(float *vector) const {
    int index = -1;
    float dist = std::numeric_limits<float>::max();

    // find the nearest neighbor
    for (int i = 0; i < size(); i++) {
        float tmp = 0;
        for (int j = 0; j < dimension(); j++)
            tmp += (vector[j] - centers.at(i, j)) * (vector[j] - centers.at(i, j));
        if (tmp < dist) {
            dist = tmp;
            index = i;
        }
    }

    return index;
}