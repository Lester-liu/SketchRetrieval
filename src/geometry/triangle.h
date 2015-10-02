//
// Created by lyx on 02/10/15.
//

#ifndef SKETCH_TRIANGLE_H
#define SKETCH_TRIANGLE_H

#include <iostream>
#include "opencv2/opencv.hpp"

namespace sketch {

    using namespace cv;
    using namespace std;

    class Triangle {
    public:
        Triangle();
        Triangle(const Point3f &a, const Point3f &b, const Point3f &c);

        friend ostream& operator<< (ostream &out, const Triangle& triangle);

        Point3f a, b, c;
    };


}

#endif //SKETCH_TRIANGLE_H
