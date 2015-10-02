//
// Created by lyx on 02/10/15.
//

#ifndef SKETCH_LINE_H
#define SKETCH_LINE_H

#include <iostream>
#include <climits>
#include "opencv2/opencv.hpp"

namespace sketch {

    using namespace std;
    using namespace cv;

    class Line {
    public:
        Line();
        Line(const Point2f& a, const Point2f& b);
        Line(const Point2f& a, const float theta);

        friend ostream& operator<<(ostream &out, const Line& line);

        Point2f a, b;
        float theta; // tangent of the direction (tan(theta))
    };

}

#endif //SKETCH_LINE_H
