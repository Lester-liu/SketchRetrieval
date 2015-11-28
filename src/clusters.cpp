//
// Created by lyx on 25/11/15.
//

#include "clusters.h"

Clusters::Clusters() { }
Clusters::Clusters(float *centers, int center_count, int dim) :
        centers(centers), center_count(center_count), dim(dim) { }

Clusters::~Clusters() { }

void Clusters::find_center(float *vectors, int *allocation, int n) const {
    // find neighbor for all instances
    for (int k = 0; k < n; k++) {
        int index = -1;
        float dist = std::numeric_limits<float>::max();

        // find the nearest neighbor
        for (int i = 0; i < center_count; i++) {
            float tmp = 0;
            for (int j = 0; j < dim; j++)
                tmp += (vectors[k * dim + j] - centers[i * dim + j]) * (vectors[k * dim + j] - centers[i * dim + j]);
            if (tmp < dist) {
                dist = tmp;
                index = i;
            }
        }

        allocation[k] = index;
    }
}
