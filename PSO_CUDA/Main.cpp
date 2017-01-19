#include <stdlib.h>
#include <stdio.h>

#include "Parameter.h"
#include "Population.h"
#include "ParticleSwarmOptimization.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Utility.h"

void CUDA_PSO();

void Generation(Population * population, Parameters *parameters, int iteration);

int main(int argc, char ** argv){

	int testId = 0;
	int probId = 1;

	printf("|---------------------------------------------------------------| \n");
	printf("|\t\t\t\t\t\t\t\t| \n");
	printf("|\t\t\t\t\t\t\t\t| \n");
	printf("|\t\tParticle Swarm Optimization\t\t\t| \n");
	printf("|\t\t\t\t\t\t\t\t| \n");
	printf("|\t\t\t\t\t\t\t\t| \n");
	printf("|\tChose one of the following tests:\t\t\t| \n");
	printf("|\t 1) Particle Swarm Optimization CUDA\t\t\t\t\| \n");
	//printf("|\t 2) Particle Swarm Optimization\t\t\t\| \n");
	printf("|\t\t\t\t\t\t\t\t| \n");
	printf("|\t\t\t\t\t\t\t\t| \n");
	printf("|---------------------------------------------------------------| \n");
	printf("\n");
	printf("My choice: \n");
	testId = getchar() - '0';

	switch (testId){
	case 1:
		CUDA_PSO();
		break;
	case 2:
		//CUDA_OMP();
		break;
	default:
		printf("nothing selected.\n");
	}

	return 0;
}

void CUDA_PSO()
{
	int i;
	Parameters *params = InitializeParameters("PSO.txt");
	FILE *output = fopen("PSO.csv", "w");
	Population *population = GeneratePopulation(params);

	EvaluatePopulationCost(population, params, true);

	for (i = 0; i < params->iterations; i++){
		fprintf(stdout, "Iteration %d of %d.\n", i, params->iterations);

		Generation(population, params, i);

		//fprintf(out, "%d , ", i);
		//PrintSolutionCSV(Pop->bestSolution, Pop->bestF, DIM, out);
		//fputc('\n', out);
	}

	PrintBestSolutions(population, params, output);

	fclose(output);
	
	FreePopulation(population);
	FreeParameter(params);
}

void Generation(Population * population, Parameters *parameters, int iteration){
	//! Find out whether any solutions have reached a new personal best
	UpdatePersonalBest(population, parameters);

	//! Find out whether a new global Best has been reached
	UpdateGlobalBest(population, parameters, iteration);
	
	//! calculate the new velocity and move the particles accordingly
	CalculateVelocity(population, parameters, iteration);
	MoveParticles(population, parameters);

	//! repair the population
	RepairPopulation(population, parameters);

	//! evaluate the new costs
	EvaluatePopulationCost(population, parameters, false);
}