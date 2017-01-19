#ifndef __PSO_POPULATION_KERNELS_H
#define __PSO_POPULATION_KERNELS_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include "math.h"

#include "ProblemKernels.cuh"
#include "Utility.h"


/**
* Skales a matrix in the size of the population filled with random values towards the bounds
* of the solution.
*
* @param matrix The matrix that will be scaled towards the bounds.
* @param matrixPitch The pitch indicating how the matrix is aligned.
* @param bounds The bounds that limit the solutions space.
* @param boundsPitch The pitch indicating how the matrix of bounds is aligned.
*/
__global__ void AdjustMatrixToBoundsKernel(double *matrix, size_t matrixPitch, double *bounds, size_t boundsPitch)
{
	int threadId = threadIdx.x;
	int blockId = blockIdx.x;

 	double rand = matrix[blockId * matrixPitch / sizeof(double) + threadId];
	double lowerBound = bounds[threadId * boundsPitch / sizeof(double)];
	double upperBound = bounds[threadId * boundsPitch / sizeof(double) + 1];

	double value = lowerBound + (upperBound - lowerBound) * rand;

	matrix[blockId * matrixPitch / sizeof(double) + threadId] = value;
}


/**
* Evaluates the problem cost of a solution.
*
* @param solutions The solutions/particles the populations is made of.
* @param solutionsPitch The pitch indicating how the matrix of solutions is aligned.
* @param costs The costs of all solutions.
* @param numberSolutions The number of particles in the population.
* @param dimensions The number of dimensions of the problem.
* @param problemId The id that indicate which problem the solution should 
*					 be tested against.
*/
__global__ void EvaluatePopulationCostKernel(double *solutions, size_t solutionsPitch, double *costs, int numberSolutions, int dimensions, int problemId)
{	
	int threadId = threadIdx.x;

	double cost = Problem(solutions, solutionsPitch, dimensions, problemId);
	costs[threadId] = cost;
}


/**
* Repairs the solutions in the case that it has left the bounds of the solution space.
*
* @param matrix The matrix that will be scaled towards the bounds.
* @param matrixPitch The pitch indicating how the matrix is aligned.
* @param bounds The bounds that limit the solutions space.
* @param boundsPitch The pitch indicating how the matrix of bounds is aligned.
*/
__global__ void RepairPopulationKernel(double *solutions, size_t solutionsPitch, double *bounds, size_t boundsPitch)
{
	int threadId = threadIdx.x;
	int blockId = blockIdx.x;

	//! I have no fucking idea why I have to use a register for the index but otherwise I get access violations
	int solIndex = blockId * solutionsPitch / sizeof(double) + threadId;
	int boundIndex = threadId * boundsPitch / sizeof(double);
	double sol = solutions[solIndex];

	sol = MAX(sol, bounds[boundIndex]);
	solutions[solIndex] = MIN(sol, bounds[boundIndex + 1]);
}

#endif