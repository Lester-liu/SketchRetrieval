//
// Created by lyx on 17/11/15.
//

#include "k_mean.h"

namespace k_mean {

    template<typename T>
    void printCpuMatrix(T *m, int n, int r, int c, int precision) {
        cout << "Size: " << r << " * " << c << endl;
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++)
                cout << fixed << setprecision(precision) << m[i + j * r] << '\t';
            cout << endl;
        }
        cout << endl;
    }

    template<typename T>
    void printGpuMatrix(T *d_m, int n, int r, int c, int precision) {
        T *m = new T[n];
        cublasGetVector(n, sizeof(*m), d_m, 1, m, 1);
        printCpuMatrix(m, n, r, c, precision);
        delete[] m;
    }

    void K_Mean::initialize_monoid() {

        // select k different centers
        unordered_set<int> selected;
        while (selected.size() < center_count) // different
            selected.insert(rand() % data_count);

        // copy theirs coordinates
        int column = 0;
        for (int i: selected) {
            //cout << i << endl;
            for (int j = 0; j < dim; j++)
                center[column * dim + j] = data[i * dim + j];
            column++;
        }

        // copy to the device
        callCuda(cudaMemcpy(d_data, data, sizeof(float) * dim * data_count, cudaMemcpyHostToDevice));
        callCuda(cudaMemcpy(d_center, center, sizeof(float) * dim * center_count, cudaMemcpyHostToDevice));

    }

    void K_Mean::initialize_centroid() {

        set_uniform_value(d_center, dim * center_count, 256);
        callCuda(cudaMemcpy(d_data, data, sizeof(float) * dim * data_count, cudaMemcpyHostToDevice));
        callCuda(cudaMemcpy(center, d_center, sizeof(float) * dim * center_count, cudaMemcpyDeviceToHost));

    }

    void K_Mean::find_nearest_center() {

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

        //printCpuMatrix(cluster_size, min(100, center_count), 1, min(100, center_count), 0);
        //printCpuMatrix(allocation, data_count, 1, data_count, 0);

    }

    void K_Mean::update_center() {

        float one = 1;
        float zero = 0;

        // update the allocation information
        callCuda(
                cudaMemcpy(d_allocation_col_csr, allocation, sizeof(int) * size_t(data_count), cudaMemcpyHostToDevice));
        callCuda(
                cudaMemcpy(d_cluster_size, cluster_size, sizeof(float) * size_t(center_count), cudaMemcpyHostToDevice));

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
                                  d_allocation_row_csr, d_allocation_col_csr, d_allocation_val_csc,
                                  d_allocation_col_csc,
                                  d_allocation_row_csc, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO));

        // compute the new center
        // attention: while the second matrix is transposed, the first one should be normal
        callCuda(cusparseScsrmm2(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                                 center_count, dim, data_count, data_count, &one, d_allocation_descr,
                                 d_allocation_val_csc,
                                 d_allocation_row_csc, d_allocation_col_csc, d_data, dim, &zero, d_center_transpose,
                                 center_count));

        transpose_scale(d_center, dim, center_count, d_center_transpose, d_cluster_size);
        callCuda(cudaMemcpy(center, d_center, sizeof(float) * dim * center_count, cudaMemcpyDeviceToHost));

    }

    void K_Mean::shake_center(float delta) {
        float *d_scale;
        callCuda(cudaMalloc(&d_scale, sizeof(float) * center_count * dim));
        // use random variation
        set_uniform_value(d_scale, center_count * dim, -delta, delta);
        shake(d_center, d_scale, center_count * dim);
        callCuda(cudaFree(d_scale));
    }

    void K_Mean::print_center() {
        printCpuMatrix(center, dim * center_count, dim, center_count, 3);
    }

    K_Mean::K_Mean(float *_data, int _data_count, int _dim, int _center_count) {

        data = _data;
        data_count = _data_count;
        center_count = _center_count;
        dim = _dim;

        assert(data_count >= center_count);

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

    }

    K_Mean::K_Mean(float *_data, float* _center, int _data_count, int _dim, int _center_count) {

        data = _data;
        data_count = _data_count;
        center_count = _center_count;
        dim = _dim;

        center = _center;
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

        callCuda(cudaMemcpy(d_center, center, sizeof(float) * dim * center_count, cudaMemcpyHostToDevice));
        callCuda(cudaMemcpy(d_data, data, sizeof(float) * dim * data_count, cudaMemcpyHostToDevice));

        set_value(d_one, max(data_count, max(dim, center_count)), 1.0f);

        set_value(d_allocation_val_csr, data_count, 1.0f);
        set_sequence(d_allocation_row_csr, data_count + 1, 0, 1); // one 1 per row
        set_value(d_allocation_col_csr, data_count, 0);

        callCuda(cublasCreate(&cublas_handle));
        callCuda(cusparseCreate(&cusparse_handle));

        callCuda(cusparseCreateMatDescr(&d_allocation_descr));
        callCuda(cusparseSetMatType(d_allocation_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
        callCuda(cusparseSetMatIndexBase(d_allocation_descr, CUSPARSE_INDEX_BASE_ZERO));

    }

    K_Mean::~K_Mean() {
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
    }

    void K_Mean::execute(int iteration, float delta) {

        initialize_monoid();
        //initialize_centroid();

        //print_center();
        for (int i = 0; i < iteration; i++) {
            cout << "Iteration #" << i << endl;
            shake_center(delta);
            //callCuda(cudaMemcpy(center, d_center, sizeof(float) * dim * center_count, cudaMemcpyDeviceToHost));
            //print_center();
            find_nearest_center();
            update_center();
            //print_center();
        }

        //print_center();

    }

    void K_Mean::save(string file, bool add_null) {
        ofstream out(file);

        // add a all zero center
        if (add_null) {
            center_count++;
            out.write((char *)&center_count, sizeof(int));
            center_count--;
        }
        else
            out.write((char*)&center_count, sizeof(int));

        out.write((char*)&dim, sizeof(int));
        out.write((char*)center, sizeof(float) * dim * center_count);
        if (add_null) {
            float *tmp = new float[dim];
            fill(tmp, tmp + dim, 0);
            out.write((char*)tmp, sizeof(float) * dim);
            delete[] tmp;
        }
        out.close();
    }

    void K_Mean::translate(int *result) {
        find_nearest_center();
        callCuda(cudaMemcpy(result, allocation, sizeof(int) * data_count, cudaMemcpyHostToHost));
    }

    void K_Mean::get_clusters(float *dest) {
        callCuda(cudaMemcpy(dest, d_center, sizeof(float) * dim * center_count, cudaMemcpyHostToHost));
    }

    void K_Mean::update_data() {
        callCuda(cudaMemcpy(d_data, data, sizeof(float) * dim * data_count, cudaMemcpyHostToDevice));
    }

}