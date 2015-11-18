/*
 * K-mean clustering
 *
 * This program implements the K-Mean clustering method. It takes a set of points (lines in a binary file) and
 * returns the group number of each point and an extra file to describe the group, namely the group center.
 *
 * Usage: k_mean [Path_to_file] Number_of_center
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
#include <unordered_set>
#include <iomanip>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include "kernel.h"
#include "read_data.h"

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

int *allocation; // index of all vectors (one dimension array), same as d_allocation_col_csr

cusparseMatDescr_t d_allocation_descr;
float *d_allocation_val_csr; // allocation result of all vectors (sparse matrix of 0 or 1)
int *d_allocation_row_csr; // row numbers (from 0 to data_count)
int *d_allocation_col_csr; // column numbers (from 0 to center_count)
float *d_allocation_val_csc; // csc is just the transpose in some way
int *d_allocation_row_csc; // this one still store the pointer
int *d_allocation_col_csc;

float *cluster_size; // size of all clusters
float *d_cluster_size; // device copy

float *d_one; // all one vector (the length is large enough)

int dim; // vector dimension

string input, output; // file name

cublasHandle_t cublas_handle;
cusparseHandle_t cusparse_handle;

template <typename T>
void printCpuMatrix(T* m, int n, int r, int c, int precision) {
    cout << "Size: " << r << " * " << c << endl;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++)
            cout << fixed << setprecision(precision) << m[i + j * r] << '\t';
        cout << endl;
    }
    cout << endl;
}

template <typename T>
void printGpuMatrix(T* d_m, int n, int r, int c, int precision) {
    T* m = new T[n];
    cublasGetVector(n, sizeof(*m), d_m, 1, m, 1);
    printCpuMatrix(m, n, r, c, precision);
    delete[] m;
}

void initialize_monoid() {

    // select k different centers
    unordered_set<int> selected;
    while (selected.size() < center_count) // different
        selected.insert(rand() % data_count);

    // copy theirs coordinates
    int column = 0;
    for (int i: selected) {
        cout << i << endl;
        for (int j = 0; j < dim; j++)
            center[column * dim + j] = data[i * dim + j];
        column++;
    }

    // copy to the device
    callCuda(cudaMemcpy(d_data, data, sizeof(float) * dim * data_count, cudaMemcpyHostToDevice));
    callCuda(cudaMemcpy(d_center, center, sizeof(float) * dim * center_count, cudaMemcpyHostToDevice));

}

void initialize_centroid() {

    set_uniform_value(d_center, dim * center_count, 256);
    callCuda(cudaMemcpy(d_data, data, sizeof(float) * dim * data_count, cudaMemcpyHostToDevice));
    callCuda(cudaMemcpy(center, d_center, sizeof(float) * dim * center_count, cudaMemcpyDeviceToHost));

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
        //printGpuMatrix(d_tmp_diff, 4, 2, 2, 0);
        callCuda(cublasSgemv(cublas_handle, CUBLAS_OP_T, dim, center_count, &one, d_tmp_diff,
                             dim, d_one, 1, &zero, d_tmp_dist, 1));

        // get the minimal one
        callCuda(cublasIsamin(cublas_handle, center_count, d_tmp_dist, 1, &index));
        //printGpuMatrix(d_tmp_dist, 2, 1, 2, 0);
        allocation[i] = index - 1;
        cluster_size[allocation[i]]++;
    }

    printCpuMatrix(cluster_size, center_count, 1, center_count, 0);
    printCpuMatrix(allocation, data_count, 1, data_count, 0);

}

void update_center() {

    float one = 1;
    float zero = 0;

    // update the allocation information
    callCuda(cudaMemcpy(d_allocation_col_csr, allocation, sizeof(int) * size_t(data_count), cudaMemcpyHostToDevice));
    callCuda(cudaMemcpy(d_cluster_size, cluster_size, sizeof(float) * size_t(center_count), cudaMemcpyHostToDevice));

    // conversion method, use dense matrix
    /*
    float *tmp;
    callCuda(cudaMalloc(&tmp, sizeof(float) * center_count * data_count));
    callCuda(cusparseScsr2dense(cusparse_handle, data_count, center_count, d_allocation_descr, d_allocation_val_csr,
                                d_allocation_row_csr, d_allocation_col_csr, tmp, data_count));
    callCuda(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, center_count, dim, data_count, &one, tmp,
                         data_count, d_data, dim, &zero, d_center_transpose, center_count));
    callCuda(cudaFree(tmp));
    */

    // transpose the allocation matrix
    callCuda(cusparseScsr2csc(cusparse_handle, data_count, center_count, data_count, d_allocation_val_csr,
                              d_allocation_row_csr, d_allocation_col_csr, d_allocation_val_csc, d_allocation_col_csc,
                              d_allocation_row_csc, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO));

    // compute the new center
    // attention: while the second matrix is transposed, the first one should be normal
    callCuda(cusparseScsrmm2(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                             center_count, dim, data_count, data_count, &one, d_allocation_descr, d_allocation_val_csc,
                             d_allocation_row_csc, d_allocation_col_csc, d_data, dim, &zero, d_center_transpose,
                             center_count));

    transpose_scale(d_center, dim, center_count, d_center_transpose, d_cluster_size);
    callCuda(cudaMemcpy(center, d_center, sizeof(float) * dim * center_count, cudaMemcpyDeviceToHost));

}

void show_center() {

    for (int i = 0; i < center_count; i++) {
        cv::Mat m(28, 28, CV_32F, center + i * dim);
        cv::Mat img;
        m.convertTo(img, CV_8U);
        cv::imshow("Image" + to_string(i), img);
    }
    cv::waitKey(0);

}

void print_center() {
    printCpuMatrix(center, dim * center_count, dim, center_count, 1);
}

void k_mean(int iteration) {

    initialize_monoid();
    //initialize_centroid();

    //print_center();
    //show_center();
    for (int i = 0; i < iteration; i++) {
        cout << "Iteration #" << i << endl;
        find_nearest_center();
        update_center();
        print_center();
        //show_center();
    }

}

int main(int argc, char** argv) {
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
    data_count = 12;
    center_count = 3;
    dim = 2;

    assert(data_count >= center_count);

    data = new float[dim * data_count];
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
    /*
    for (int i = 0; i < dim * data_count; i++)
        data[i] = float(_data[i]);
    delete[] _data;
    */


    center = new float[dim * center_count];
    allocation = new int[data_count];
    cluster_size = new float[center_count];

    callCuda(cudaMalloc(&d_data, sizeof(float) * dim * data_count));
    callCuda(cudaMalloc(&d_center, sizeof(float) * dim * center_count));
    callCuda(cudaMalloc(&d_center_transpose, sizeof(float) * center_count * dim));
    callCuda(cudaMalloc(&d_tmp_diff, sizeof(float) * dim * center_count));
    callCuda(cudaMalloc(&d_tmp_dist, sizeof(float) * center_count));
    callCuda(cudaMalloc(&d_allocation_val_csr, sizeof(float) * (data_count)));
    callCuda(cudaMalloc(&d_allocation_row_csr, sizeof(int) * (data_count + 1))); // CRS format
    callCuda(cudaMalloc(&d_allocation_col_csr, sizeof(int) * (data_count)));
    callCuda(cudaMalloc(&d_allocation_val_csc, sizeof(float) * (data_count)));
    callCuda(cudaMalloc(&d_allocation_col_csc, sizeof(int) * (data_count))); // CRS format
    callCuda(cudaMalloc(&d_allocation_row_csc, sizeof(int) * (center_count + 1)));
    callCuda(cudaMalloc(&d_cluster_size, sizeof(float) * center_count));
    callCuda(cudaMalloc(&d_one, sizeof(float) * max(data_count, max(dim, center_count)))); // long enough

    set_value(d_one, max(data_count, max(dim, center_count)), 1.0f);

    set_value(d_allocation_val_csr, data_count, 1.0f);
    set_sequence(d_allocation_row_csr, data_count + 1, 0, 1); // one 1 per row
    set_value(d_allocation_col_csr, data_count, 0);

    callCuda(cublasCreate(&cublas_handle));
    callCuda(cusparseCreate(&cusparse_handle));

    callCuda(cusparseCreateMatDescr(&d_allocation_descr));
    callCuda(cusparseSetMatType(d_allocation_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    callCuda(cusparseSetMatIndexBase(d_allocation_descr, CUSPARSE_INDEX_BASE_ZERO));

    k_mean(10);

    callCuda(cusparseDestroyMatDescr(d_allocation_descr));

    callCuda(cusparseDestroy(cusparse_handle));
    callCuda(cublasDestroy(cublas_handle));

    callCuda(cudaFree(d_data));
    callCuda(cudaFree(d_center));
    callCuda(cudaFree(d_center_transpose));
    callCuda(cudaFree(d_tmp_diff));
    callCuda(cudaFree(d_tmp_dist));
    callCuda(cudaFree(d_allocation_val_csr));
    callCuda(cudaFree(d_allocation_row_csr));
    callCuda(cudaFree(d_allocation_col_csr));
    callCuda(cudaFree(d_allocation_val_csc));
    callCuda(cudaFree(d_allocation_row_csc));
    callCuda(cudaFree(d_allocation_col_csc));
    callCuda(cudaFree(d_cluster_size));
    callCuda(cudaFree(d_one));

    delete[] data;
    delete[] center;
    delete[] allocation;
    delete[] cluster_size;

    return EXIT_SUCCESS;
}