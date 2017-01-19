#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ParticleSwarmOptimization.h"
#include "ParticleSwarmOptimizationKernels.cuh"
#include "Population.h"
#include "Parameter.h"
#include "Utility.h"

size_t CreateRandomArray(double **dev_randomC1C2, Parameters *parameters);

/**
* Calculates the new velocity using the initial velocity, the personal best of every particle in the
* population and the global best. /PARALLEL/
* @param population The populations whose velocity will be calculated.
* @param parameters The parameters that describe the input given by the user to unfluence the
*					simulation.
*/
void CalculateVelocity(
	Population *population, 
	Parameters *parameters,
	int iteration)
{
	double *dev_randomC1C2;

	size_t randomC1C2Pitch = CreateRandomArray(&dev_randomC1C2, parameters);

	CalculateVelocityKernel << <parameters->numberSolutions, parameters->dimensions >> > (
		population->dev_solutions, 
		population->solutionsPitch, 
		population->dev_pBestSolutions, 
		population->pBestSolutionsPitch, 
		dev_randomC1C2,
		randomC1C2Pitch,
		population->dev_gBestSolution, 
		parameters->personalBestC, 
		parameters->globalBestC, 
		parameters->inertialC, 
		parameters->velocityC,
		parameters->exponentiateVelocityC,
		population->dev_velocity, 
		population->velocityPitch,
		parameters->iterations,
		iteration);

	CUDA_METHOD_CALL(cudaGetLastError());
	CUDA_METHOD_CALL(cudaDeviceSynchronize());
	CUDA_METHOD_CALL(cudaFree(dev_randomC1C2));
}


/**
* Creates the three arrays based on chance needed for calculating the velocity.
*
* @param dev_randomC1C2 The matrix that contains the random values between [0, 1] that will be used in the
*					    calculation of the new velocity.
* @return The pitch of the dev_randomC1C2 matrix.
*/
size_t CreateRandomArray(
		double **dev_randomC1C2,
		Parameters *parameters)
{
	//! Matrix that contains the random values that will be used in conjunction with the personal best and the 
	//! global best when calculating the new velocity
	//!
	//!					dimensions * 2
	//!	----------------------------------------------->
	//!	________________________________________________
	//!	|						|						|	^
	//!	|						|						|	|
	//!	|						|						|	|
	//!	|		rands used		|		rands used		|	|
	//!	|		with c1			|		with c2			|	|	Number of
	//!	|						|						|	|	solutions
	//!	|						|						|	|
	//!	|						|						|	|
	//!	|_______________________|_______________________|	|

	size_t randPitch = CreateRandomDoubleMatrix(dev_randomC1C2, parameters->numberSolutions, parameters->dimensions * 2);
	
	return randPitch;
}


/**
* Moves the particles to their new positions. /PARALLEL/
* @param population The populations whose particles will be moved.
* @param parameters The parameters that describe the input given by the user to unfluence the
*					simulation.
*/
void MoveParticles(
	Population *population, 
	Parameters *parameters)
{
	MoveParticlesKernel << <parameters->numberSolutions, parameters->dimensions >> >(
		population->dev_solutions, 
		population->solutionsPitch, 
		parameters->dev_bounds,
		parameters->boundsPitch,
		population->dev_velocity, 
		population->velocityPitch);

	CUDA_METHOD_CALL(cudaGetLastError());
	CUDA_METHOD_CALL(cudaDeviceSynchronize());
}


/**
* Updates the best solution the populations has ever collectively found. /SERIAL BUT ON DEVICE/
* @param population The populations that will be updated.
* @param parameters The parameters that describe the input given by the user to unfluence the
*					simulation.
*/
void UpdateGlobalBest(Population *population, Parameters *parameters, int iteration)
{
	int bestIndex;
	int* dev_bestIndex;
	int improved;
	int* dev_improved;

	CUDA_METHOD_CALL(cudaMalloc(&dev_bestIndex, sizeof(int)));
	CUDA_METHOD_CALL(cudaMalloc(&dev_improved, sizeof(int)));

	FindGlobalBestKernel << <1, 1 >> >(
		population->dev_pBestCosts, 
		population->dev_gBestCost, 
		parameters->numberSolutions, 
		dev_bestIndex, 
		dev_improved);

	CUDA_METHOD_CALL(cudaGetLastError());
	CUDA_METHOD_CALL(cudaDeviceSynchronize());

	CUDA_METHOD_CALL(cudaMemcpy(
		&bestIndex,
		dev_bestIndex,
		sizeof(int),
		cudaMemcpyDeviceToHost));
	CUDA_METHOD_CALL(cudaMemcpy(
		&improved,
		dev_improved,
		sizeof(int),
		cudaMemcpyDeviceToHost));

	//! in the first iteration gBest is not set at all so the new best solution improves it in any case
	if (improved || iteration == 0){
		CUDA_METHOD_CALL(cudaMemcpy(
			population->dev_gBestCost,
			&(population->dev_pBestCosts[bestIndex]),
			sizeof(double),
			cudaMemcpyDeviceToDevice));
		CUDA_METHOD_CALL(cudaMemcpy(
			population->dev_gBestSolution,
			&(population->dev_pBestSolutions[bestIndex * population->pBestSolutionsPitch / sizeof(double)]),
			sizeof(double)*parameters->dimensions,
			cudaMemcpyDeviceToDevice));
	}

	CUDA_METHOD_CALL(cudaMemcpy(
		&(population->dev_bestCosts[iteration]),
		population->dev_gBestCost,
		sizeof(double),
		cudaMemcpyDeviceToDevice));
	CUDA_METHOD_CALL(cudaMemcpy(
		&(population->dev_bestSolutions[iteration * population->bestSolutionsPitch / sizeof(double)]),
		population->dev_gBestSolution,
		sizeof(double)*(parameters->dimensions),
		cudaMemcpyDeviceToDevice));
}


/**
* Updates the best solution every single particle in the population has ever found.
* /PARALLEL/
* @param population The populations that will be updated.
* @param parameters The parameters that describe the input given by the user to unfluence the
*					simulation.
*/
void UpdatePersonalBest(Population *population, Parameters *parameters)
{
	bool* dev_improved;

	CUDA_METHOD_CALL(cudaMalloc(&dev_improved, sizeof(bool)*parameters->numberSolutions));

	FindPersonalBestImprovedKernel << <1, parameters->numberSolutions >> >(population->dev_costs, population->dev_pBestCosts, dev_improved);
	CUDA_METHOD_CALL(cudaGetLastError());
	CUDA_METHOD_CALL(cudaDeviceSynchronize());

	UpdatePersonalBestKernel << <parameters->numberSolutions, parameters->dimensions >> >(
		population->dev_solutions, 
		population->solutionsPitch, 
		population->dev_pBestSolutions, 
		population->pBestSolutionsPitch, 
		dev_improved);

	CUDA_METHOD_CALL(cudaGetLastError());
	CUDA_METHOD_CALL(cudaDeviceSynchronize());
}