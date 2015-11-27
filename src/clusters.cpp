//
// Created by lyx on 25/11/15.
//

#include "clusters.h"

Clusters::Clusters() { }
Clusters::Clusters(Mat& centers) : centers(centers) { }

Clusters::~Clusters() { }

int Clusters::row() const {
    return centers.rows;
}

int Clusters::col() const {
    return centers.cols;
}

void Clusters::find_center(Mat& vector, int *allocation, int n) const {
    assert(vector.rows == n); // x is the number of lines, y is the dimension of vector
    // find neighbor for all instances
    for (int i = 0; i < n; i++)
        allocation[i] = find_center(vector,i);
}

int Clusters::find_center(Mat& vector, int x) const {
    int index = -1;
    float dist = std::numeric_limits<float>::max();

    // find the nearest neighbor
    for (int i = 0; i < row(); i++) {
        float tmp = 0;
        for (int j = 0; j < col(); j++)
            tmp += (vector.at<float>(x,j) - centers.at<float>(i, j)) * (vector.at<float>(x,j) - centers.at<float>(i, j));
        if (tmp < dist) {
            dist = tmp;
            index = i;
        }
    }

    return index;
}