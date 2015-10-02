//
// Created by lyx on 02/10/15.
//

#include "triangle.h"

namespace sketch {

    Triangle::Triangle() {
        a = Point3f(0, 0, 0);
        b = Point3f(0, 0, 0);
        c = Point3f(0, 0, 0);
    }

    Triangle::Triangle(const Point3f &a, const Point3f &b, const Point3f &c) {
        this->a = Point3f(a);
        this->b = Point3f(b);
        this->c = Point3f(c);
    }

    ostream& operator<<(ostream &out, const Triangle &triangle) {
        out << "Triangle: ";
        cout << triangle.a << '|' << triangle.b << '|' << triangle.c << endl;
    }

}