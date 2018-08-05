#ifndef __PSO_KERNELS_H
#define __PSO_KERNELS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>
#include "math.h"

#include "Utility.h"

/**
* Calculates the new velocity of a particle based on its current velocity, its personal
* best positon and the global best position. This will be influenced by multiple parameters.

* @param solutions The solutions/particles the populations is made of.
* @param solutionsPitch The pitch indicating how the matrix of solutions is aligned.
* @param pBestSolutions The personal best positions of every single particle in the solution.
* @param pBestSolutionsPitch The pitch indicating how the matrix of personal best solutions is aligned.
* @param randomC1C2 The matrix that contains the random values between [0, 1] that will be used in the
*					calculation of the new velocity.
* @param randomC1C2Pitch The pitch indicating how the matrix of random values is aligned.
* @param gBestSolution The global best solution of the population.
* @param c1 The parameter that indicates how much influence the personal best should have on the new velocity.
* @param c2 The parameter that indicates how much influence the global best should have on the new velocity.
* @param k The parameter that indicates how much influence the initial velocity should have on the new velocity.
* @param velocityC The value that constrains the new velocity.
* @param exponentiateVelocityC The value that indicates whether the velocity escalates depending on the iteration.
* @param velocity The matrix that contains all velocities for the entire population.
* @param velocityPitch The pitch indicating how the matrix of velocities is aligned.
* @param iterations The total number of iterations of the PSO.
* @param iteration The current iteration of the PSO.
*/
__global__ void CalculateVelocityKernel(
	double *solutions,
	size_t solutionsPitch,
	double *pBestSolutions,
	size_t pBestSolutionsPitch,
	double *randomC1C2,
	size_t randomC1C2Pitch,
	double *gBestSolution,
	double c1,
	double c2,
	double k,
	double velocityC,
	bool exponentiateVelocityC,
	double *velocity,
	size_t velocityPitch,
	int iterations,
	int currentIteration)
{
	int threadId = threadIdx.x;
	int blockId = blockIdx.x;

	int indexSolutions = blockId * solutionsPitch / sizeof(double)+threadId;
	int indexPBestSolutions = blockId * pBestSolutionsPitch / sizeof(double)+threadId;
	int indexVelocity = blockId * velocityPitch / sizeof(double)+threadId;
	int indexRand = blockId * randomC1C2Pitch / sizeof(double)+threadId;


	//! The later the iteration the more the solution converges towards the global best. To try to improve the quality of the solution in later stages
	//! of the algorithm: Make the random value multiplied with the global best smaller, and increase the random value multiplied with the personal best
	double random = randomC1C2[indexRand];
	double quotient = (double)currentIteration / (double)iterations;
	double globalBestInfluence = 1 - pow(random, 1 - quotient);
	double personalBestInfluence = pow(random, 1 - quotient);

	double velC = velocityC;
	if (exponentiateVelocityC)
		velC = pow(velocityC, 1 - quotient);
	
	//! Original PSO formula: Problem = r1 and r2 independently generated, inevitably there are cases in which they are both too large or too small
	/*double newVelocity = k * velocity[indexVelocity]
		+ c1 * randomC1C2[indexRand] * (pBestSolutions[indexPBestSolutions] - solutions[indexSolutions])
		+ c2 * randomC1C2[indexRand + 1] * (gBestSolution[threadId] - solutions[indexSolutions]);*/

	//! Improvement:
	double newVelocity = velC * (k * velocity[indexVelocity]
	+ c1 * personalBestInfluence * (pBestSolutions[indexPBestSolutions] - solutions[indexSolutions])
	+ c2 * globalBestInfluence * (gBestSolution[threadId] - solutions[indexSolutions]));

	velocity[indexVelocity] = newVelocity;
}

/**
* Moves a particle into its new direction based on the velocity calculated beforehand.

* @param solutions The solutions/particles the populations is made of.
* @param solutionsPitch The pitch indicating how the matrix of solutions is aligned.
* @param velocity The matrix that contains all velocities for the entire population.
* @param velocityPitch The pitch indicating how the matrix of velocities is aligned.
*/
__global__ void MoveParticlesKernel(
	double *solutions, 
	size_t solutionsPitch,
	double *bounds,
	size_t boundsPitch, 
	double *velocity, 
	size_t velocityPitch)
{
	int threadId = threadIdx.x;
	int blockId = blockIdx.x;

	int solIndex = blockId * solutionsPitch / sizeof(double)+threadId;
	int velIndex = blockId * solutionsPitch / sizeof(double)+threadId;
	int indexBounds = threadId * boundsPitch / sizeof(double);
	
	double lowerBounds = bounds[indexBounds];
	double upperBounds = bounds[indexBounds + 1];
	double range = upperBounds - lowerBounds;
	double distance = 0;

	solutions[solIndex] = solutions[solIndex] + velocity[velIndex];
}

/**
* Finds the global best solution and the information whether the global best solution is actually 
* better than the previous one.
*
* @param costs The costs of all particles.
* @param previousCost The cost of the previous best solution.
* @param numberSolutions The number of particles in the population.
* @param bestIndex The index of the currently best solution.
* @param improved The value indicating whether the current best solution is better than the previous 
*				  one.
*/
__global__ void FindGlobalBestKernel(double *costs, double* previousCost, int numberSolutions, int* bestIndex, int* improved)
{
	*improved = 0;

	// serialize search for best
	int i;
	*bestIndex = 0;
	//! find the best solution in the population
	for (i = 1; i < numberSolutions; i++){
		if (costs[i] < costs[*bestIndex]) {
			*bestIndex = i;
		}
	}

	//! if better than previous global best, update
	if (costs[*bestIndex] < *previousCost){
		*improved = 1;
	}
}


/**
* Finds out which particles have improved upon their personal best position.
*
* @param costs The costs of all particles.
* @param pBestCosts The costs of the best positions of all particles.
* @param improved The array that contains the values indication whether a particle has improved
*				  upon its personal best.
*/
__global__ void FindPersonalBestImprovedKernel(double *costs, double *pBestCosts, bool *improved)
{
	int threadId = threadIdx.x;
	//int index = threadId * sizeof(double);
	improved[threadId] = false;

	if (costs[threadId] < pBestCosts[threadId]) {
		improved[threadId] = true;
		pBestCosts[threadId] = costs[threadId];
	}
}

/**
* Updates all personal best solutions that actually have improved.
*
* @param solutions The solutions/particles the populations is made of.
* @param solutionsPitch The pitch indicating how the matrix of solutions is aligned.
* @param pBestSolutions The personal best positions of every single particle in the solution.
* @param pBestSolutionsPitch The pitch indicating how the matrix of personal best solutions is aligned.
* @param improved The array that contains the values indication whether a particle has improved
*				  upon its personal best.
*/
__global__ void UpdatePersonalBestKernel(double *solutions, size_t solutionsPitch, double *pBestSolutions, size_t pBestsSolutionsPitch, bool *improved)
{
	int threadId = threadIdx.x;
	int blockId = blockIdx.x;

	if (improved[blockId])
		pBestSolutions[blockId * pBestsSolutionsPitch / sizeof(double) + threadId] = solutions[blockId * solutionsPitch / sizeof(double) + threadId];
}

#endif
