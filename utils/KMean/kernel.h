/*
 * Kernel file, must be separated from cpp files
 *
 * Created by lyx on 17/11/15.
 */

#ifndef KERNEL_H
#define KERNEL_H

#include <ctime>
#include <iostream>
#include <sstream>
#include <curand.h>
#include <cuda_runtime.h>

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

void set_value(float *d_m, int n, float value);
void set_value(int *d_m, int n, int value);
void set_sequence(int *d_m, int n, int start, int step);
void transpose_scale(float *d_center, int n, int m, float *d_center_transpose, float *d_cluster_size);
void square_minus(float *d_center, int n, int m, float *d_point, float *d_diff);
void set_uniform_value(float* d_m, int n, float min, float max);
void set_uniform_value(float* d_m, int n, float epsilon);
void shake(float *d_m, float *d_scale ,int n);

#endif //KERNEL_H
