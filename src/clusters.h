//
// Created by lyx on 25/11/15.
//

#ifndef SKETCH_CLUSTERS_H
#define SKETCH_CLUSTERS_H

#include <cassert>
#include <climits>
#include "opencv2/opencv.hpp"

using namespace cv;

/*
 * This class is similar to K-means, but it is a CPU version to find a nearest neighbor
 */
class Clusters {

private:
    int center_count;
    int dim; // the dimension by default is the same as all query vectors

public:
    float *centers;
    Clusters();
    Clusters(float *centers, int center_count, int dim);
    virtual ~Clusters();
    void find_center(float *vectors, int *allocation, int n) const; // get nearest neighbor for n vectors

};

#endif //SKETCH_CLUSTERS_H
