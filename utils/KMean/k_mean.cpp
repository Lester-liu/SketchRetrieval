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

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include "kernel.h"

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



using namespace std;

float *data; // data to be clustered (vectors are stored line after line)
float *d_data; // device copy of data
int data_count; // number of vectors

float *center; // centers of k-mean
float *d_center; // device copy of centers
float *d_center_transpose; // temporary matrix
float *d_tmp_diff; // temporary matrix to calculate the nearest neighbor
float *d_tmp_dist; // temporary array to store distance
int center_count; // number of centers (k)

int *allocation; // index of all vectors (one dimension array), same as d_allocation_col

cusparseMatDescr_t d_allocation_descr;
float *d_allocation_val; // allocation result of all vectors (sparse matrix of 0 or 1)
int *d_allocation_row; // row numbers (from 0 to data_count)
int *d_allocation_col; // column numbers (from 0 to center_count)


float *cluster_size; // size of all clusters
float *d_cluster_size; // device copy

float *d_one; // all one vector (the length is large enough)

int dim; // vector dimension

string input, output; // file name

cublasHandle_t cublas_handle;
cusparseHandle_t cusparse_handle;

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

    // clear previous allocation status
    fill(cluster_size, cluster_size + center_count, 0);

    // find nearest neighbor for all data vectors
    float one = 1;
    float zero = 0;
    int index = -1;

    for (int i = 0; i < data_count; i++) {
        // compute the distance
        square_minus(d_center, dim, center_count, d_data + i * dim, d_tmp_diff);
        callCuda(cublasSgemv(cublas_handle, CUBLAS_OP_N, dim, center_count, &one, d_tmp_dist,
                             dim, d_one, 1, &zero, d_tmp_dist, 1));
        // get the minimal one
        callCuda(cublasIsamin(cublas_handle, center_count, d_tmp_dist, 1, &index));
        allocation[i] = index;
        cluster_size[allocation[i]]++;
    }

}

void update_center() {

    float one = 1;
    float zero = 0;

    // update the allocation information
    callCuda(cudaMemcpy(d_allocation_col, allocation, size_t(data_count), cudaMemcpyHostToDevice));
    callCuda(cudaMemcpy(d_cluster_size, cluster_size, size_t(center_count), cudaMemcpyHostToDevice));

    // compute the new center
    callCuda(cusparseScsrmm2(cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                             data_count, dim, center_count, data_count, &one, d_allocation_descr, d_allocation_val,
                             d_allocation_row, d_allocation_col, d_data, dim, &zero, d_center_transpose,
                             center_count));
    transpose_scale(d_center, dim, center_count, d_center_transpose, d_cluster_size);

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
    cluster_size = new float[center_count];

    callCuda(cudaMalloc(&d_data, sizeof(float) * dim * data_count));
    callCuda(cudaMalloc(&d_center, sizeof(float) * dim * center_count));
    callCuda(cudaMalloc(&d_center_transpose, sizeof(float) * center_count * dim));
    callCuda(cudaMalloc(&d_tmp_diff, sizeof(float) * dim * center_count));
    callCuda(cudaMalloc(&d_tmp_dist, sizeof(float) * center_count));
    callCuda(cudaMalloc(&d_allocation_val, sizeof(float) * data_count));
    callCuda(cudaMalloc(&d_allocation_val, sizeof(int) * data_count));
    callCuda(cudaMalloc(&d_allocation_val, sizeof(int) * data_count));
    callCuda(cudaMalloc(&d_cluster_size, sizeof(float) * center_count));
    callCuda(cudaMalloc(&d_one, sizeof(float) * max(data_count, max(dim, center_count)))); // long enough

    set_value(d_one, max(data_count, max(dim, center_count)), 1.0f);

    set_value(d_allocation_val, data_count, 1.0f);
    set_sequence(d_allocation_row, data_count, 0, 1); // one 1 per row
    set_value(d_allocation_col, data_count, 0);

    callCuda(cublasCreate(&cublas_handle));
    callCuda(cusparseCreate(&cusparse_handle));

    callCuda(cusparseCreateMatDescr(&d_allocation_descr));


    callCuda(cusparseDestroyMatDescr(d_allocation_descr));

    callCuda(cusparseDestroy(cusparse_handle));
    callCuda(cublasDestroy(cublas_handle));

    callCuda(cudaFree(d_data));
    callCuda(cudaFree(d_center));
    callCuda(cudaFree(d_center_transpose));
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