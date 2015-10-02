//
// Created by lyx on 02/10/15.
//

#ifndef SKETCH_MODEL_H
#define SKETCH_MODEL_H

#include <iostream>
#include <cassert>
#include <vector>
#include "opencv2/opencv.hpp"

#include "triangle.h"

namespace sketch {

    using namespace std;
    using namespace cv;

    class Model {
    public:
        Model();
        Model(istream& in);
        void Add(const Triangle& triangle);
        void Pop();

        friend ostream& operator<<(ostream& out, const Model& model);

        vector<Triangle> triangles;
        int count;
    };

}

#endif //SKETCH_MODEL_H
