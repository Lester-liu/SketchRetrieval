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
    int find_center(Mat& vector , int x) const; // get cluster number of one vector

public:
    Mat centers;
    Clusters();
    Clusters(Mat& centers);
    virtual ~Clusters();
    int row() const; // the number of centers
    int col() const; // the center dimension
    void find_center(Mat& vectors, int *allocation, int n) const; // get nearest neighbor for n vectors

};

#endif //SKETCH_CLUSTERS_H
