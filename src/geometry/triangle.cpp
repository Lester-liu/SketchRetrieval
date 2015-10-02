//
// Created by lyx on 02/10/15.
//

#include "triangle.h"

namespace sketch {

    Triangle::Triangle() {
        a = Point2f(0, 0);
        b = Point2f(0, 0);
        c = Point2f(0, 0);
    }

    Triangle::Triangle(const Point2f &a, const Point2f &b, const Point2f &c) {
        this->a = Point2f(a);
        this->b = Point2f(b);
        this->c = Point2f(c);
    }

    ostream& operator<<(ostream &out, const Triangle &triangle) {
        out << "Triangle: ";
        cout << triangle.a << '|' << triangle.b << '|' << triangle.c << endl;
    }

}