/*
 * Kernel file, must be separated from cpp files
 *
 * Created by lyx on 17/11/15.
 */

#ifndef KERNEL_H
#define KERNEL_H

#include <ctime>
#include <curand.h>

void set_value(float *d_m, int n, float value);
void set_value(int *d_m, int n, int value);
void set_sequence(int *d_m, int n, int start, int step);
void transpose_scale(float *d_center, int n, int m, float *d_center_transpose, float *d_cluster_size);
void square_minus(float *d_center, int n, int m, float *d_point, float *d_diff);
void set_uniform_value(float* d_m, int n, float epsilon);

#endif //KERNEL_H
