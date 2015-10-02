#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "vtkRenderWindow.h"
#include "vtkSmartPointer.h"

#include "geometry/triangle.h"
#include "geometry/line.h"
#include "geometry/model.h"

using namespace cv;
using namespace std;

using namespace sketch;

int main() {
    cout << "Hello, World!" << endl;

    Triangle tri(Point3f(1.23f, 1, 0.5), Point3f(2, 2, 1), Point3f(3, 4, -3));
    Line line(Point3f(1, 1, 1), Point3f(2.26, 5, 0.4));
    ifstream input("../models/models/m0.off");
    Model model(input);
    cout << tri << line << model;

    return 0;
}