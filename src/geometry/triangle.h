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
        Triangle(const Point2f &a, const Point2f &b, const Point2f &c);

        friend ostream& operator<< (ostream &out, const Triangle& triangle);

        Point2f a, b, c;
    };


}

#endif //SKETCH_TRIANGLE_H
