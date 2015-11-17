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
#include <sstream>
#include <set>

#ifdef CUDA_ENABLED

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 128

#define fatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define callCuda(status) do {                                  		   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
    	_error << "Cuda failure: " << status;                          \
    	fatalError(_error.str());                                      \
    }                                                                  \
} while(0)

template <typename T>
__device__ T square(const T x) {
    return x * x;
}

template <typename T>
__global__ void kernel_sequence(T *d_m, int n, T start, T step) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        d_m[i] = start + step * i;
}

__global__ void kernal_set_value(float *d_m, int n, float value) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        d_m[i] = value;
}

__global__ void kernel_square_minus(float *d_center, float *point, float *d_diff) {
    int id_center = blockIdx.x; // blockDim.x == dim && max(blockIdx.x) == center_count
    int id_pos = threadIdx.x; // max(threadIdx.x) == dim
    d_diff[id_center * blockDim.x + id_pos] = square(d_center[id_center * blockDim.x + id_pos] - point[id_pos]);
}

void set_value(float *d_m, int n, float value) {
    kernal_set_value<<<(n - 1 + BLOCK_SIZE) / BLOCK_SIZE, BLOCK_SIZE>>>(d_m, n, value);
}

void set_sequence(int)

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
float *d_allocation_val; // allocation result of all vectors (sparse matrix)
float *d_allocation_row; // row numbers (sparse matrix)
float *d_allocation_col; // column numbers (sparse matrix)


int *cluster_size; // size of all clusters
int *d_cluster_size; // device copy

float *d_one; // all one vector (the length is large enough)

int dim; // vector dimension

string input, output; // file name

cublasHandle_t handle;

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

    callCuda(cudaMemcpy(d_data, data, size_t(dim * data_count), cudaMemcpyHostToDevice));
    callCuda(cudaMemcpy(d_center, center, size_t(dim * center_count), cudaMemcpyHostToDevice));

}

void find_nearest_center() {

    // find nearest neighbor for all data vectors
    float one = 1;
    float zero = 0;
    int index = -1;

    for (int i = 0; i < data_count; i++) {
        // compute the distance
        kernel_square_minus<<<center_count, dim>>>(d_center, d_data[i * dim], d_tmp_diff);
        callCuda(cublasSgemv(handle, CUBLAS_OP_N, dim, center_count, &one, d_tmp_dist,
                             dim, d_one, 1, &zero, d_tmp_dist, 1));
        // get the minimal one
        callCuda(cublasIsamin(handle, center_count, d_tmp_dist, 1, &index));
        allocation[i] = index;
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
    callCuda(cudaMalloc(&d_allocation_val, sizeof(float) * data_count));
    callCuda(cudaMalloc(&d_allocation_val, sizeof(int) * data_count));
    callCuda(cudaMalloc(&d_allocation_val, sizeof(int) * data_count));
    callCuda(cudaMalloc(&d_cluster_size, sizeof(int) * center_count));
    callCuda(cudaMalloc(&d_one, sizeof(float) * max(data_count, max(dim, center_count)))); // long enough

    set_value(d_one, max(data_count, max(dim, center_count)), 1);
    set_value(d_allocation_val, data_count, 1);

    callCuda(cublasCreate(&handle));



    callCuda(cublasDestroy(handle));

    callCuda(cudaFree(d_data));
    callCuda(cudaFree(d_center));
    callCuda(cudaFree(d_tmp_diff));
    callCuda(cudaFree(d_tmp_dist));
    callCuda(cudaFree(d_allocation_val));
    callCuda(cudaFree(d_allocation_row));
    callCuda(cudaFree(d_allocation_col));
    callCuda(cudaFree(d_cluster_size));
    callCuda(cudaFree(d_one));

    delete[] data;
    delete[] center;
    delete[] allocation;
    delete[] cluster_size;

    return EXIT_SUCCESS;
}