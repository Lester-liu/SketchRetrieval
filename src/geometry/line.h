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
        Line(const Point3f& a, const Point3f& b);
        Line(const Point3f& a, const float delta_y, const float delta_z);

        friend ostream& operator<<(ostream &out, const Line& line);

        Point3f a;
        float delta_y, delta_z; // direction

    };

}

#endif //SKETCH_LINE_H
