//
// Created by lyx on 10/11/15.
//

#ifndef SKETCH_BLOB_H
#define SKETCH_BLOB_H

#include <cstddef>
#include <algorithm>

/*
 * 3D dimension
 */
struct Dim {

    int x, y, z;

    Dim() {
        x = 0;
        y = 0;
        z = 0;
    }

    Dim(int x, int y, int z): x(x), y(y), z(z) {}

    Dim(Dim& d) {
        x = d.x;
        y = d.y;
        z = d.z;
    }

    /*
     * Return the position of a 3D coordinates in a 1D array
     *
     * i: x-axis -> channel
     * j: y-axis -> width
     * k: z-axis -> height
     */
    int get_index(int i, int j, int k) const {
        return i * y * z + j * z + k;
    }

    /*
     * Return the size of equivalent 1D array
     */
    int get_size() const {
        return x * y * z;
    }

    bool operator== (const Dim& dim) const {
        return x == dim.x && y == dim.y && z == dim.z;
    }
};

/*
 * Data structure or wrapper
 */
class Blob {

public:

    float *data; // 1D array content
    Dim size; // Real dimension of the content

    Blob();
    Blob(Dim size); // 0 array
    Blob(float *data, Dim size); // Copy the value into new array
    virtual ~Blob();

};


#endif //SKETCH_BLOB_H
