#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <float.h>

#include "Population.h"
#include "PopulationKernels.cuh"
#include "Utility.h"

/**
* Skales a matrix in the size of the population filled with random values towards the bounds
* of the solution.
*
* @param solutions The solutions/particles the populations is made of.
* @param solutionsPitch The pitch indicating how the matrix of solutions is aligned.
* @param bounds The bounds that limit the solutions space.
* @param boundsPitch The pitch indicating how the matrix of bounds is aligned.
* @param dimensions The number of dimensions of the problem.
* @param numberSolutions The number of particles in the population.
*/
void AdjustRandomMatrixToBounds(double *solutions, size_t solutionsPitch, double *bounds, size_t boundsPitch, int dimensions, int numberSolutions)
{
	AdjustMatrixToBoundsKernel <<<numberSolutions, dimensions>>>(solutions, solutionsPitch, bounds, boundsPitch);
	CUDA_METHOD_CALL(cudaGetLastError());
	CUDA_METHOD_CALL(cudaDeviceSynchronize());
}

/**
* Calculates the new function costs for all particles/solutions of the population. /PARALLEL/
* @param population The populations whose fuction costs will be evaluated.
* @param parameters The parameters that describe the input given by the user to unfluence the
*					simulation.
*/
void EvaluatePopulationCost(Population *population, Parameters *parameters, bool initialEvaluation){
	EvaluatePopulationCostKernel <<<1, parameters->numberSolutions>>>(
		population->dev_solutions,
		population->solutionsPitch,
		population->dev_costs,
		parameters->numberSolutions,
		parameters->dimensions,
		parameters->problemId); 
	CUDA_METHOD_CALL(cudaGetLastError());
	CUDA_METHOD_CALL(cudaDeviceSynchronize());

	if (initialEvaluation){
		CUDA_METHOD_CALL(cudaMemcpy(population->dev_pBestCosts, population->dev_costs, parameters->numberSolutions*sizeof(double), cudaMemcpyDeviceToDevice));

		CUDA_METHOD_CALL(cudaGetLastError());
		CUDA_METHOD_CALL(cudaDeviceSynchronize());
	}
}

/**
* Cleans up the memory used to store the population. /SERIAL/
* @param population The populations that will be cleaned up.
*/
void FreePopulation(Population *population)
{
	CUDA_METHOD_CALL(cudaFree(population->dev_bestCosts));
	CUDA_METHOD_CALL(cudaFree(population->dev_bestSolutions));
	CUDA_METHOD_CALL(cudaFree(population->dev_costs));
	CUDA_METHOD_CALL(cudaFree(population->dev_gBestCost));
	CUDA_METHOD_CALL(cudaFree(population->dev_gBestSolution));
	CUDA_METHOD_CALL(cudaFree(population->dev_pBestCosts));
	CUDA_METHOD_CALL(cudaFree(population->dev_pBestSolutions));
	CUDA_METHOD_CALL(cudaFree(population->dev_solutions));
	CUDA_METHOD_CALL(cudaFree(population->dev_velocity));
}

/**
* Creates and Initializes a new random population. /SERIAL/
* @param parameters The parameters that describe the input given by the user to unfluence the
*					simulation.
* @return The initialized population.
*/
Population *GeneratePopulation(Parameters* parameter)
{
	Population *population = (Population*)calloc(1, sizeof(Population));

	//! create solutions an fill them with random start values and repair population to make sure it is within its bounds
	population->solutionsPitch = CreateRandomDoubleMatrix(&(population->dev_solutions), parameter->numberSolutions, parameter->dimensions);
	AdjustMatrixToBoundsKernel << <parameter->numberSolutions, parameter->dimensions >> >(population->dev_solutions, population->solutionsPitch, parameter->dev_bounds, parameter->boundsPitch);
	CUDA_METHOD_CALL(cudaGetLastError());
	CUDA_METHOD_CALL(cudaDeviceSynchronize());

	CUDA_METHOD_CALL(cudaMallocPitch<double>(&(population->dev_bestSolutions), &(population->bestSolutionsPitch), parameter->dimensions*sizeof(double), parameter->iterations));
	
	//! create the personal best for all solutions and initialize it with the start position of all solutions
	CUDA_METHOD_CALL(cudaMallocPitch<double>(&(population->dev_pBestSolutions), &(population->pBestSolutionsPitch), parameter->dimensions*sizeof(double), parameter->numberSolutions));
	CUDA_METHOD_CALL(cudaMemcpy2D(
		population->dev_pBestSolutions,
		population->pBestSolutionsPitch,
		population->dev_solutions,
		population->solutionsPitch,
		parameter->dimensions*sizeof(double),
		parameter->numberSolutions,
		cudaMemcpyDeviceToDevice));

	CUDA_METHOD_CALL(cudaMallocPitch<double>(&(population->dev_velocity), &(population->velocityPitch), parameter->dimensions*sizeof(double), parameter->numberSolutions));
	CUDA_METHOD_CALL(cudaMemset2D(population->dev_velocity, population->velocityPitch, 0, parameter->dimensions*sizeof(double), parameter->numberSolutions));


	CUDA_METHOD_CALL(cudaMalloc<double>(&(population->dev_gBestSolution), parameter->dimensions*sizeof(double)));
	CUDA_METHOD_CALL(cudaMalloc<double>(&(population->dev_costs), parameter->numberSolutions*sizeof(double)));
	CUDA_METHOD_CALL(cudaMalloc<double>(&(population->dev_bestCosts), parameter->iterations*sizeof(double)));

	CUDA_METHOD_CALL(cudaMalloc<double>(&(population->dev_pBestCosts), parameter->numberSolutions*sizeof(double)));
	CUDA_METHOD_CALL(cudaMemcpy(population->dev_pBestCosts, population->dev_costs, parameter->numberSolutions*sizeof(double), cudaMemcpyDeviceToDevice));

	CUDA_METHOD_CALL(cudaMalloc<double>(&(population->dev_gBestCost), sizeof(double)));
	CUDA_METHOD_CALL(cudaMemset(population->dev_gBestCost, DBL_MAX, sizeof(double)));

	return population;
}


/**
* Copys the array of best solutions over time to the host and prints them.
* @param population The populations whose best solutions will be printed.
* @param parameters The parameters that describe the input given by the user to unfluence the
*					simulation.
*/
void PrintBestSolutions(Population* population, Parameters *parameters, FILE *out)
{
	FILE * file = fopen("PSO.csv", "w");

	size_t pitch = population->bestSolutionsPitch;

	double *bestSolutions = (double*)malloc(sizeof(double)*(parameters->iterations)*(parameters->dimensions));
	double *bestCosts = (double*)malloc(sizeof(double)*parameters->iterations);
	int i, j;

	CUDA_METHOD_CALL(cudaMemcpy2D(
		bestSolutions,
		parameters->dimensions*sizeof(double),
		population->dev_bestSolutions,
		pitch,
		parameters->dimensions*sizeof(double),
		parameters->iterations,
		cudaMemcpyDeviceToHost));

	CUDA_METHOD_CALL(cudaMemcpy(bestCosts, population->dev_bestCosts, parameters->iterations*sizeof(double), cudaMemcpyDeviceToHost));

	for (i = 0; i < parameters->iterations; i++){
		fprintf(file, "%lf;", bestCosts[i]);
		for (j = 0; j < parameters->dimensions; j++){
			double value = bestSolutions[i * parameters->dimensions + j];
			fprintf(file, "%lf;", value);
		}
		fprintf(file, "\n");
	}

	fclose(file);
}

/**
* Repairs the population and makes sure that the solution does not leav the bounds specified
* by the user.
* @param population The populations that will be repaired.
* @param parameters The parameters that describe the input given by the user to unfluence the
*					simulation.
*/
void RepairPopulation(Population* population, Parameters *parameters)
{
	RepairPopulationKernel <<<parameters->numberSolutions, parameters->dimensions>>>(
		population->dev_solutions,
		population->solutionsPitch,
		parameters->dev_bounds,
		parameters->boundsPitch); 

	CUDA_METHOD_CALL(cudaGetLastError());
	CUDA_METHOD_CALL(cudaDeviceSynchronize());
}


