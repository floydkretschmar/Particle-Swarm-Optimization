#ifndef __PSO_UTILITY_H
#define __PSO_UTILITY_H

#ifdef __CUDACC__
#define CUDA_KERNEL_CALL_ARGS2(grid, block) <<< grid, block >>>
#define CUDA_KERNEL_CALL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define CUDA_KERNEL_CALL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define CUDA_KERNEL_CALL_ARGUMENTS2(grid, block)
#define CUDA_KERNEL_CALL_ARGUMENTS3(grid, block, sh_mem)
#define CUDA_KERNEL_CALL_ARGUMENTS4(grid, block, sh_mem, stream)
#endif

#define CUDA_METHOD_CALL(x) if((x)!=cudaSuccess) { fprintf(stdout, "Error at %s:%d\n",__FILE__,__LINE__); }
#define CURAND_METHOD_CALL(x) if((x)!=CURAND_STATUS_SUCCESS) { fprintf(stdout, "Error at %s:%d\n",__FILE__,__LINE__); } 

#define MIN(a,b) ( (a) < (b) ? (a) : (b) )
#define MAX(a,b) ( (a) > (b) ? (a) : (b) )

/**
* Creates a matrix with the given dimensions on the device and fills it with random values between 0 and 1.
*
* @param matrix The matrix filled with random numbers.
* @param rows The number rows the matrix should have.
* @param columns The number columns the matrix should have.
* @return The pitch of the created matrix.
*/
size_t CreateRandomDoubleMatrix(double **matrix, size_t rows, size_t columns);
#endif