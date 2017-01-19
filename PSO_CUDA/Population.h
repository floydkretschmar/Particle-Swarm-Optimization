#ifndef __PSO_POPULATION_H
#define __PSO_POPULATION_H

#include <stdlib.h>
#include <stdio.h>
#include "Parameter.h"

typedef struct Population {
	double *dev_solutions;				//!< particles
	size_t solutionsPitch;				//!< pitch for the matrix of solutions
	double* dev_costs;					//!< current cost functions values

	double* dev_bestSolutions;			//!< the history of best solutions over all iterations
	size_t bestSolutionsPitch;			//!< pitch for the matrix of best solutions over all iterations
	double* dev_bestCosts;				//!< the history of best solution cost values

	double *dev_gBestCost;				//!< all time global best solution cost value
	double *dev_gBestSolution;			//!< all time global best solution

	double *dev_pBestCosts;				//!< all time personal best solution cost functions values
	double *dev_pBestSolutions;			//!< all time personal best solutions
	size_t pBestSolutionsPitch;			//!< pitch for the matrix of the personal best solutions

	double *dev_velocity;				//!< velocity 
	size_t velocityPitch;				//!< pitch for the matrix of velocities

} Population;


/**
* Calculates the new function costs for all particles/solutions of the population. /PARALLEL/
* @param population The populations whose fuction costs will be evaluated.
* @param parameters The parameters that describe the input given by the user to unfluence the 
*					simulation.
*/
void EvaluatePopulationCost(Population *population, Parameters *parameters, bool initialEvaluation);

/**
* Cleans up the memory used to store the population. /SERIAL/
* @param population The populations that will be cleaned up.
*/
void FreePopulation(Population *population);

/**
* Creates and Initializes a new random population. /SERIAL/
* @param parameters The parameters that describe the input given by the user to unfluence the 
*					simulation.
* @return The initialized population.
*/
Population *GeneratePopulation(Parameters *parameter);

/**
* Copys the array of best solutions over time to the host and prints them.
* @param population The populations whose best solutions will be printed.
* @param parameters The parameters that describe the input given by the user to unfluence the
*					simulation.
* @param out The output stream into which the results will be written.
*/
void PrintBestSolutions(Population* population, Parameters *parameters, FILE *out);

/**
* Repairs the population and makes sure that the solution does not leav the bounds specified 
* by the user.
* @param population The populations that will be repaired.
* @param parameters The parameters that describe the input given by the user to unfluence the
*					simulation.
*/
void RepairPopulation(Population* population, Parameters *parameters);


#endif