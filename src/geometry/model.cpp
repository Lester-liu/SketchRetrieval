//
// Created by lyx on 02/10/15.
//

#include "model.h"

namespace sketch {

    Model::Model() {

    }

    Model::Model(istream &in) {
        string head;
        in >> head;
        assert(head == "OFF");
        int pts;
        int tmp;
        in >> pts >> count >> tmp;
        Point3f* points = new Point3f[pts + 1];
        float tmp_x, tmp_y, tmp_z;
        for (int i = 1; i <= pts; i++) {
            in >> tmp_x >> tmp_y >> tmp_z;
            points[i] = Point3f(tmp_x, tmp_y, tmp_z);
        }
        int index_a, index_b, index_c;
        for (int i = 0; i < count; i++) {
            in >> tmp >> index_a >> index_b >> index_c;
            triangles.push_back(Triangle(points[index_a], points[index_b], points[index_c]));
        }
        delete[] points;
    }

    void Model::Add(const Triangle &triangle) {

    }

    void Model::Pop() {

    }

    ostream& operator<<(ostream &out, const Model &model) {
        out << "Model: " << model.count << " triangles" << endl;
        for (Triangle t : model.triangles)
            cout << '\t' << t;
    }

}
