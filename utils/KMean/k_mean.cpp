/*
 * K-mean clustering
 *
 * This program implements the K-Mean clustering method. It takes a set of points (lines in a binary file) and
 * returns the group number of each point and an extra file to describe the group, namely the group center.
 *
 * Usage: k_mean [Path_to_file]
 *
 * N.B. The file format is very specific, it is a binary file with integers and floats, so please pay attention to the
 * big / little endian problem. You may want to generate the file by program in case of theses sorts of problems.
 * The first number is the number of points: N, the second is the dimension of the point: d. Then there should
 * be N * d float numbers after. So the binary file looks like:
 *
 *      N (32 bits integer) d (32 bits integer)
 *      P_1 (d * 32 bits float)
 *      ...
 *      P_N (d * 32 bits float)
 */

#include <fstream>
#include <iostream>
#include <set>

#ifdef CUDA_ENABLED

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>

inline void callCuda(cudaError_t e) {
    if (e != cudaSuccess)
        printf("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
}

__global__ void kernel_minus(float *d_center, float *point, float *d_dist, int center_count, int dim) {
    int id_center = blockIdx.x;
    int id_pos = threadIdx.x;

}

#endif

using namespace std;

float *data; // data to be clustered (vectors are stored line after line)
float *d_data; // device copy of data
int data_count; // number of vectors

float *center; // centers of k-mean
float *d_center; // device copy of centers
float *d_tmp_diff; // temporary array to calculate the nearest neighbor
float *d_tmp_dist; // temporary array to store distance
int center_count; // number of centers (k)

int *allocation; // index of all vectors (one dimension array)
float *d_allocation; // allocation result of all vectors (matrix)

int *cluster_size; // size of all clusters
float *d_cluster_size; // device copy

int dim; // vector dimension

string input, output; // file name

void initialize() {

    // select k different centers

    set<int> selected;
    while (selected.size() < center_count) // different
        selected.insert(rand() % data_count);

    // copy theirs coordinates

    int column = 0;
    for (int i: selected) {
        for (int j = 0; j < dim; j++)
            center[column * dim + j] = data[i * dim + j];
        column++;
    }

    // copy to the device

    callCuda(cudaMemcpy(d_data, data, dim * data_count, cudaMemcpyHostToDevice));
    callCuda(cudaMemcpy(d_center, center, dim * center_count, cudaMemcpyHostToDevice));

}

void find_nearest_center() {

    // find nearest neighbor for all data vectors

    for (int i = 0; i < data_count; i++) {

    }
}

void update_center() {

}

void k_mean(int iteration) {
    initialize();
    for (int i = 0; i < iteration; i++) {
        find_nearest_center();
        update_center();
    }
}

int main(int argc, char** argv) {

    data = new float[dim * data_count];
    center = new float[dim * center_count];
    allocation = new int[data_count];
    cluster_size = new int[center_count];

    callCuda(cudaMalloc(&d_data, sizeof(float) * dim * data_count));
    callCuda(cudaMalloc(&d_center, sizeof(float) * dim * center_count));
    callCuda(cudaMalloc(&d_tmp_diff, sizeof(float) * dim * center_count));
    callCuda(cudaMalloc(&d_tmp_dist, sizeof(float) * center_count));
    callCuda(cudaMalloc(&d_allocation, sizeof(float) * data_count * center_count));
    callCuda(cudaMalloc(&d_cluster_size, sizeof(float) * center_count));



    callCuda(cudaFree(d_data));
    callCuda(cudaFree(d_center));
    callCuda(cudaFree(d_tmp_diff));
    callCuda(cudaFree(d_tmp_dist));
    callCuda(cudaFree(d_allocation));
    callCuda(cudaFree(d_cluster_size));

    delete[] data;
    delete[] center;
    delete[] allocation;
    delete[] cluster_size;

    return EXIT_SUCCESS;
}