#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <omp.h>

#include "Parameter.h"
#include "Utility.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/**
* Initializes the parameters by reading from a given text file. /SERIAL/
* @param paramsFile The path of the file specifying the parameters to start the algorithm with.
* @return The initialized parameters.
*/
Parameters *InitializeParameters(const char * paramsFile)
{
	int i;
	//! allocate memory for parameters
	Parameters *parameters = (Parameters*)calloc(1, sizeof(Parameters));

	FILE * pf = fopen(paramsFile, "r");
	if (!pf){
		printf("Provide parameters: \
			   		\n\t 1.dimension \
					\n\t 2.number of solutions \
					\n\t 3.number of iterations \
					\n\t 4.inertial coefficient (0.8 - 1.2) \
					\n\t 5.personal best coefficient (0.0 - 2.0) \
					\n\t 6.global best coefficient (0.0 - 2.0) \
					\n\t 7.velocity coefficient (0.0 - 1.0) \
					\n\t 8.value indicating whether the velocityC is exponentiated (0 or 1)\
					\n\t 9.problem id \
					\n\t 10.bounds matrix \n");
		pf = stdin;
	}

	int exponentiate = 0;
	//! get GA parameters
	fscanf(pf, "%d", &(parameters->dimensions));
	fscanf(pf, "%d", &(parameters->numberSolutions));
	fscanf(pf, "%d", &(parameters->iterations));
	fscanf(pf, "%lf", &(parameters->inertialC));
	fscanf(pf, "%lf", &(parameters->personalBestC));
	fscanf(pf, "%lf", &(parameters->globalBestC));
	fscanf(pf, "%lf", &(parameters->velocityC));
	fscanf(pf, "%d", &(exponentiate));
	parameters->exponentiateVelocityC = (bool)exponentiate;
	//! get problem id
	fscanf(pf, "%d", &(parameters->problemId));

	//! allocate memory for Bounds
	double *bounds = (double*)malloc(2 * sizeof(double)*parameters->dimensions);
	for (i = 0; i<parameters->dimensions; i++){
		fscanf(pf, "%lf %lf", &(bounds[i*2]), &(bounds[i*2+1]));
	}

	cudaError_t error;

	error = cudaMallocPitch((void**)&(parameters->dev_bounds), &(parameters->boundsPitch), 2 * sizeof(double), parameters->dimensions);
	CUDA_METHOD_CALL(cudaMemcpy2D(parameters->dev_bounds, parameters->boundsPitch, bounds, 2 * sizeof(double), 2 * sizeof(double), parameters->dimensions, cudaMemcpyHostToDevice));
	
	free(bounds);

	if (pf != stdin){
		fclose(pf);
	}

	return parameters;
}

/**
* Cleans up the memory used to store the parameters. /SERIAL/
* @param parameter The parameters that will be cleaned up.
*/
void FreeParameter(Parameters * parameters)
{
	cudaFree(parameters->dev_bounds);
	free(parameters);
}