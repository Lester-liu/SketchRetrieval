/*
 * K-Mean algorithm
 *
 * Created by lyx on 23/11/15.
 */

#ifndef K_MEAN_H
#define K_MEAN_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <iomanip>
#include <cassert>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include "kernel.h"
#include "read_data.h"

using namespace std;

namespace k_mean {

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

        void initialize_monoid(); // use existing points as centers

        void initialize_centroid(); // use random centers

        void find_nearest_center();

        void update_center(); // compute the new barycenter of the group

        void shake_center(float delta); // move a little the existing center to leave local minimum

        void print_center();

    public:

        K_Mean(float *data, int data_count, int dim, int center_count); // train a clustering function

        K_Mean(float *data, float* center, int dim, int center_count); // classify an image with a dictionary

        virtual ~K_Mean();

        void execute(int iteration, float delta); // train

        void save(string file); // save the center information into a file

        int get_cluster(int i); // get cluster number of the i-th vector

        void get_clusters(float *dest); // get center coordinates

    };
}

#endif //K_MEAN_H
