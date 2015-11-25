//
// Created by lyx on 25/11/15.
//

#ifndef SKETCH_CLUSTERS_H
#define SKETCH_CLUSTERS_H

#include <cassert>
#include <climits>
#include "blob.h"

/*
 * This class is similar to K-means, but it is a CPU version to find a nearest neighbor
 */
class Clusters {

private:
    int find_center(float *vector) const; // get cluster number of one vector

public:
    Blob centers;

    Clusters(Blob& centers);
    virtual ~Clusters();
    int size() const; // the number of clusters
    int dimension() const; // the center dimension
    void find_center(Blob& vectors, int *allocation, int n) const; // get nearest neighbor for n vectors

};

#endif //SKETCH_CLUSTERS_H
