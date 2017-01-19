#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Utility.h"

/**
* Creates a matrix with the given dimensions on the device and fills it with random values between 0 and 1.
*
* @param matrix The matrix filled with random numbers.
* @param rows The number rows the matrix should have.
* @param columns The number columns the matrix should have.
* @return The pitch of the created matrix.
*/
size_t CreateRandomDoubleMatrix(double **matrix, size_t rows, size_t columns)
{
	size_t pitch;
	
	CUDA_METHOD_CALL(cudaMallocPitch((void**)matrix, &pitch, columns*sizeof(double), rows));

	//! use Mersenne Twister algorithm for pseudo random number generation.
	curandGenerator_t gen;
	CURAND_METHOD_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32));

	//! set seed 
	CURAND_METHOD_CALL(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));

	//! create random numbers 
	CURAND_METHOD_CALL(curandGenerateUniformDouble(gen, *matrix, pitch*rows / sizeof(double)));

	//! cleanup
	CURAND_METHOD_CALL(curandDestroyGenerator(gen));

	return pitch;
}

