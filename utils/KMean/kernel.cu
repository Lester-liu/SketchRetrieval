//
// Created by lyx on 17/11/15.
//

#include "kernel.h"
#include <device_launch_parameters.h>

#define BLOCK_SIZE 256

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

template <typename T>
__global__ void kernal_set_value(T *d_m, int n, T value) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        d_m[i] = value;
}

template <typename T>
__global__ void kernel_scale(T *d_m, float *d_scale, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        d_m[i] *= 1 + d_scale[i];
}

template <typename T>
__global__ void kernel_square_minus(T *d_center, int n, int m, T *point, T *d_diff) {
    int id_center = blockIdx.x; // blockDim.x == dim && max(blockIdx.x) == center_count
    int id_pos = threadIdx.x; // max(threadIdx.x) == dim
    if (id_pos < n && id_center < m)
        d_diff[id_center * blockDim.x + id_pos] = square(d_center[id_center * blockDim.x + id_pos] - point[id_pos]);
}

template <typename T>
__global__ void kernel_transpose_scale(T *d_center, int n, int m, T *d_center_tmp, T *size) {
    // each column of d_center (n * m) correspond to each row of d_center_tmp * size[row]
    int row = threadIdx.x;
    int col = blockIdx.x;
    // m == center_count && n == dim
    if (row + m * col < n * m && row * n + col < n * m) {
        if (size[col] != 0)
            d_center[row + m * col] = d_center_tmp[row * n + col] / size[col];
        else
            d_center[row + m * col] = d_center_tmp[row * n + col];
    }
}

template <typename T>
__global__ void kernel_uniform_scale(T *x, int n, T min, T max) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        x[i] = x[i] * (max - min) + min;
}

void set_value(float *d_m, int n, float value) {
    if (value == 0)
        callCuda(cudaMemset(d_m, 0, sizeof(float) * n));
    else
        kernal_set_value<<<(n - 1 + BLOCK_SIZE) / BLOCK_SIZE, BLOCK_SIZE>>>(d_m, n, value);
}

void set_value(int *d_m, int n, int value) {
    if (value == 0)
        callCuda(cudaMemset(d_m, 0, sizeof(int) * n));
    else
        kernal_set_value<<<(n - 1 + BLOCK_SIZE) / BLOCK_SIZE, BLOCK_SIZE>>>(d_m, n, value);
}

void set_sequence(int *d_m, int n, int start, int step) {
    kernel_sequence<<<(n - 1 + BLOCK_SIZE) / BLOCK_SIZE, BLOCK_SIZE>>>(d_m, n, start, step);
}

void transpose_scale(float *d_center, int n, int m, float *d_center_transpose, float *d_cluster_size) {
    kernel_transpose_scale<<<m, n>>>(d_center, m, n, d_center_transpose, d_cluster_size);
}

void square_minus(float *d_center, int n, int m, float *d_point, float *d_diff) {
    kernel_square_minus<<<m, n>>>(d_center, n, m, d_point, d_diff);
}

void set_uniform_value(float *d_m, int n, float min, float max) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(generator, rand());
    curandGenerateUniform(generator, d_m, n);
    kernel_uniform_scale <<<blocksPerGrid, threadsPerBlock>>>(d_m, n, min, max);
    curandDestroyGenerator(generator);
}

void set_uniform_value(float *d_m, int n, float epsilon) {
    set_uniform_value(d_m, n, 0, epsilon);
}

void shake(float *d_m, float *d_scale ,int n) {
    kernel_scale<<<(n - 1 + BLOCK_SIZE) / BLOCK_SIZE, BLOCK_SIZE>>>(d_m, d_scale, n);
}