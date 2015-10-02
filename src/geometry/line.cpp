//
// Created by lyx on 02/10/15.
//

#include "line.h"

namespace sketch {

    Line::Line() {
        a = Point3f(0, 0, 0);
        delta_y = 0;
        delta_z = 0;
    }

    Line::Line(const Point3f &a, const Point3f &b) {
        this->a = Point3f(a);
        if (a.x == b.x) {
            delta_y = numeric_limits<float>::max();
            delta_z = numeric_limits<float>::max();
        }
        else {
            delta_y = (b.y - a.y) / (b.x - a.x);
            delta_z = (b.z - a.z) / (b.x - a.x);
        }
    }

    Line::Line(const Point3f &a, const float delta_y, const float delta_z) :
            delta_y(delta_y), delta_z(delta_z) {
        this->a = Point3f(a);
    }

    ostream& operator<<(ostream &out, const Line &line) {
        out << "Line: ";
        out << line.a << '|' << line.delta_y << '|' << line.delta_z << endl;
    }
}