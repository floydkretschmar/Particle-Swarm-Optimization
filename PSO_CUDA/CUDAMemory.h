#ifndef __CUDA_MEMORY_H
#define __CUDA_MEMORY_H

size_t *create_array_double(int length);

size_t *create_array_int(int length);

size_t *create_matrix_double(int length, int height);

size_t *create_matrix_int(int length, int height);

void *free_array_int(int n);

void *free_array_double(int n);

void *free_matrix_int(int n);

void *free_matrix_double(int n);

void copy_array(double * destination, double * source, int n);

void RandomArray(int * arr, int n);

void SwapArrayDbl(double * arr1, double * arr2, int n);

#endif