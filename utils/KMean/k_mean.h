//
// Created by lyx on 23/11/15.
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

#ifndef K_MEAN_H
#define K_MEAN_H

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

class K_Mean {

private:

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

    cublasHandle_t cublas_handle;
    cusparseHandle_t cusparse_handle;

    void initialize_monoid();
    void initialize_centroid();
    void find_nearest_center();
    void update_center();
    void show_center();
    void print_center();

public:

    K_Mean(float* data, int data_count, int dim, int center_count);
    virtual ~K_Mean();
    void execute(int iteration);


};

#endif //K_MEAN_H
