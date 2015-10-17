//
// Created by lyx on 17/10/15.
//

#ifndef SKETCH_TRANSFORM_H
#define SKETCH_TRANSFORM_H

#include "opencv2/opencv.hpp"

using namespace cv;

namespace sketch {

    /*
     * rotation around the center
     *
     * x, y, z -> r, phi, theta
     * phi: angle with x axis
     * theta: angle with z axis
     *
     */
    void rotation(Point3f pt, float phi, float theta);

}

#endif //SKETCH_TRANSFORM_H
