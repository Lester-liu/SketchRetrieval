#include <iostream>
#include "opencv2/opencv.hpp"

#include "geometry/triangle.h"
#include "geometry/line.h"

using namespace cv;
using namespace std;

using namespace sketch;

int main() {
    cout << "Hello, World!" << endl;
    Triangle tri(Point2f(1.23f, 1), Point2f(2, 2), Point2f(3, 4));
    Line line(Point2f(1, 1), Point2f(2.26, 5));
    cout << tri << line;
    return 0;
}