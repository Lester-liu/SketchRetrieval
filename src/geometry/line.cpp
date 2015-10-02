//
// Created by lyx on 02/10/15.
//

#include "line.h"

namespace sketch {

    Line::Line() {
        a = Point2f(0, 0);
        b = Point2f(0, 0);
        theta = 0;
    }

    Line::Line(const Point2f &a, const Point2f &b) {
        this->a = Point2f(a);
        this->b = Point2f(b);
        if (a.x == b.x)
            theta = numeric_limits<float>::max();
        else
            theta = (b.y - a.y) / (b.x - a.x);
    }

    Line::Line(const Point2f &a, const float theta) : theta(theta) {
        this->a = Point2f(a);
        if (theta == numeric_limits<float>::max())
            this->b = Point2f(a.x, a.y + 1);
        else
            this->b = Point2f(a.x + 1, a.y + theta);
    }

    ostream& operator<<(ostream &out, const Line &line) {
        out << "Line: ";
        out << line.a << '|' << line.b << '|' << line.theta << endl;
    }
}