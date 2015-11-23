//
// Created by lyx on 23/11/15.
//

#include "k_mean.h"

using namespace k_mean;

void test_2d() {

    int data_count = 12;
    int center_count = 3;
    int dim = 2;

    assert(data_count >= center_count);

    float *data = new float[dim * data_count];
    data[0] = 0;
    data[1] = 0;
    data[2] = 1;
    data[3] = 0;
    data[4] = 1;
    data[5] = 1;
    data[6] = 0;
    data[7] = 1;
    data[8] = 10;
    data[9] = 10;
    data[10] = 11;
    data[11] = 10;
    data[12] = 11;
    data[13] = 11;
    data[14] = 10;
    data[15] = 11;
    data[16] = 10;
    data[17] = 0;
    data[18] = 11;
    data[19] = 0;
    data[20] = 11;
    data[21] = 1;
    data[22] = 10;
    data[23] = 1;

    K_Mean model(data, data_count, dim, center_count);
    model.execute(10, 0.1);

    delete[] data;

}

void test_mnist() {
    /*
    input = "t10k-images.idx3-ubyte";
    center_count = 10;

    ifstream file(input);
    int tmp, row, col;
    read_int(file, &tmp);
    read_int(file, &data_count);
    read_int(file, &row);
    read_int(file, &col);
    dim = row * col;
    uint8_t *_data = new uint8_t[dim * data_count];
    read_bytes(file, _data, dim * data_count);
    file.close();
    */

}

int main() {
    srand(time(NULL));
    test_2d();
}