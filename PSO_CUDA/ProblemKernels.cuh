#ifndef __PSO_PROBLEM_KERNELS_H
#define __PSO_PROBLEM_KERNELS_H

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"

#include <stdio.h>
#include <stdlib.h>


__device__ double SCHWEFEL(double *solution, size_t solutionsPitch, int dimensions){
	int threadId = threadIdx.x;	
	double sum = 0.0;

	for (int i = 0; i < dimensions; i++){
		sum = sum +(-1.0*solution[threadId*solutionsPitch / sizeof(double) + i])*sinf(sqrt(fabs(solution[threadId*solutionsPitch / sizeof(double) + i])));
	}

	return sum;
}

__device__ double DE_JONG_ONE(double *solution, size_t solutionsPitch, int dimensions){
	int threadId = threadIdx.x;	
	double sum = 0.0;

	for (int i = 0; i < dimensions; i++){
		sum = sum + (solution[threadId*solutionsPitch / sizeof(double) + i] * solution[threadId*solutionsPitch / sizeof(double) + i]);
	}

	return sum;
}

__device__ double DE_JONG_THREE(double *solution, size_t solutionsPitch, int dimensions){
	int threadId = threadIdx.x;
	double sum = 0.0;

	for (int i = 0; i < dimensions; i++){
		sum = sum + fabs(solution[threadId*solutionsPitch / sizeof(double) + i]);
	}
	return sum;
}

__device__ double DE_JONG_FOUR(double *solution, size_t solutionsPitch, int dimensions){
	int threadId = threadIdx.x;
	double sum = 0.0;

	for (int i = 0; i < dimensions; i++){
		sum = sum + ((threadId*solutionsPitch / sizeof(double) + i)*(pow(solution[threadId*solutionsPitch / sizeof(double) + i], 4)));
	}
	return sum;
}

__device__ double ROSENBROCK(double *solution, size_t solutionsPitch, int dimensions){
	int threadId = threadIdx.x;
	double sum = 0.0;

	for (int i = 0; i < dimensions; i++){
		sum = sum + (100 * (pow((solution[1 * solutionsPitch / sizeof(double)] * solution[threadId*solutionsPitch / sizeof(double) + i]) - solution[threadId*solutionsPitch / sizeof(double) + i + 1], 2) + pow(1 - solution[threadId*solutionsPitch / sizeof(double) + i], 2)));
	}
	return sum;
}

__device__ double RASTRIGIN(double *solution, size_t solutionsPitch, int dimensions){
	int threadId = threadIdx.x;
	double sum = 0.0;

	for (int i = 0; i < dimensions; i++){
		sum = sum + (pow(solution[threadId*solutionsPitch / sizeof(double) + i], 2) - (10 * cos(2 * 3.141592*solution[threadId*solutionsPitch / sizeof(double) + i])));
	}
	sum = sum * 2 * dimensions;	

	return sum;
}

__device__ double GRIEWANGK(double *solution, size_t solutionsPitch, int dimensions){
	int threadId = threadIdx.x;
	double prod = 1.0;
	double sum = 0.0;
	double cosinus = 0.0;
	double square = 0.0;
	
	for (int i = 0; i < dimensions; i++){
		sum = sum + (pow(solution[threadId*solutionsPitch / sizeof(double) + i], 2) / 4000);

		square = sqrt((double)i);
		cosinus = cos(solution[threadId*solutionsPitch / sizeof(double)+i] / square);

		prod = prod * cosinus;
	}

	sum = 1 + sum - prod;	
	return sum;
}

__device__ double SINE_ENVELOPE_SINE_WAVE(double *solution, size_t solutionsPitch, int dimensions){
	int threadId = threadIdx.x;
	double sum = 0.0;

	for (int i = 0; i < dimensions; i++){
		sum = sum + (0.5 + ((pow(sin(pow(solution[threadId*solutionsPitch / sizeof(double) + i], 2) + pow(solution[threadId*solutionsPitch / sizeof(double) + i + 1], 2) - 0.5), 2)) /
			pow((1 + 0.001*(pow(solution[threadId*solutionsPitch / sizeof(double) + i], 2) + pow(solution[threadId*solutionsPitch / sizeof(double) + i + 1], 2))), 2)));
	}
	sum = -1.0*sum;

	return sum;
}

__device__ double STRETCH_V_SINE_WAVE(double *solution, size_t solutionsPitch, int dimensions){
	int threadId = threadIdx.x;
	double sum = 0.0;

	for (int i = 0; i < dimensions; i++){
		sum = sum + (pow(pow(solution[threadId*solutionsPitch / sizeof(double) + i], 2) + pow(solution[threadId*solutionsPitch / sizeof(double) + i + 1], 2), 1 / 4)
			*pow(sin(50 * (pow(pow(solution[threadId*solutionsPitch / sizeof(double) + i], 2) + pow(solution[threadId*solutionsPitch / sizeof(double) + i + 1], 2), 1 / 10))), 2) + 1);
	}
	return sum;
}

__device__ double ACKLEY_ONE(double *solution, size_t solutionsPitch, int dimensions){
	int threadId = threadIdx.x;
	double sum = 0.0;
	
	for (int i = 0; i < dimensions; i++){
		sum = sum + ((1 / exp(5.0))*(sqrt(pow(solution[threadId*solutionsPitch / sizeof(double) + i], 2) + pow(solution[threadId*solutionsPitch / sizeof(double) + i + 1], 2)))
			+ 3 * (cos(2 * solution[threadId*solutionsPitch / sizeof(double) + i]) + sin(2 * solution[threadId*solutionsPitch / sizeof(double) + i + 1])));
	}

	return sum;
}

__device__ double ACKLEY_TWO(double *solution, size_t solutionsPitch, int dimensions){
	int threadId = threadIdx.x;
	double sum = 0.0;

	for (int i = 0; i < dimensions; i++){
		sum = sum + (20 + exp(1.0) - (20 / exp(0.2*sqrt((pow(solution[threadId*solutionsPitch / sizeof(double) + i], 2) + pow(solution[threadId*solutionsPitch / sizeof(double) + i + 1], 2)) / 2)))
			- exp(0.5*(cos(2 * 3.141592*solution[threadId*solutionsPitch / sizeof(double) + i]) + cos(2 * 3.141592*solution[threadId*solutionsPitch / sizeof(double) + i + 1]))));
	}

	return sum;
}

__device__ double EGG_HOLDER(double *solution, size_t solutionsPitch, int dimensions){
	int threadId = threadIdx.x;
	double sum = 0.0;

	for (int i = 0; i < dimensions; i++){
		sum = sum + (-1.0*sin(sqrt(fabs(solution[threadId*solutionsPitch / sizeof(double) + i] - solution[threadId*solutionsPitch / sizeof(double) + i + 1] - 47)))
			- ((solution[threadId*solutionsPitch / sizeof(double) + i + 1] + 47)*sin(sqrt(fabs(solution[threadId*solutionsPitch / sizeof(double) + i + 1] + 47 + (solution[threadId*solutionsPitch / sizeof(double) + i] / 2))))));
	}

	return sum;
}

__device__ double RANA(double *solution, size_t solutionsPitch, int dimensions){
	int threadId = threadIdx.x;
	double sum = 0.0;

	for (int i = 0; i < dimensions; i++){
		sum = sum + ((solution[threadId*solutionsPitch / sizeof(double) + i] * sin(sqrt(fabs(solution[threadId*solutionsPitch / sizeof(double) + i + 1] + 1 - solution[threadId*solutionsPitch / sizeof(double) + i])))*cos(sqrt(fabs(solution[threadId*solutionsPitch / sizeof(double) + i + 1] + 1 + solution[threadId*solutionsPitch / sizeof(double) + i]))))
			+ (solution[threadId*solutionsPitch / sizeof(double) + i + 1] * cos(sqrt(fabs(solution[threadId*solutionsPitch / sizeof(double) + i + 1] + 1 - solution[threadId*solutionsPitch / sizeof(double) + i])))*sin(sqrt(fabs(solution[threadId*solutionsPitch / sizeof(double) + i + 1] + 1 + solution[threadId*solutionsPitch / sizeof(double) + i])))));
	}

	return sum;
}

__device__ double PATHOLOGICAL(double *solution, size_t solutionsPitch, int dimensions){
	int threadId = threadIdx.x;
	double sum = 0.0;

	for (int i = 0; i < dimensions; i++){
		sum = sum + (0.5 + (pow(sin(sqrt(100 * (pow(solution[threadId*solutionsPitch / sizeof(double) + i], 2) - pow(solution[threadId*solutionsPitch / sizeof(double) + i + 1], 2)))), 2) - 0.5
			/ (1 + 0.001*pow((pow(solution[threadId*solutionsPitch / sizeof(double) + i], 2) - 2 * solution[threadId*solutionsPitch / sizeof(double) + i] * solution[threadId*solutionsPitch / sizeof(double) + i + 1] + pow(solution[threadId*solutionsPitch / sizeof(double) + i + 1], 2)), 2))));
	}

	return sum;
}

__device__ double MICHALEWICZ(double *solution, size_t solutionsPitch, int dimensions){
	int threadId = threadIdx.x;

	double sum = 0.0;


	for (int i = 0; i < dimensions; i++){
		sum = sum + (-1.0*(sin(solution[threadId*solutionsPitch / sizeof(double) + i] * pow(sin(pow(solution[threadId*solutionsPitch / sizeof(double) + i], 2) / 3.141592), 20))
			+ sin(solution[threadId*solutionsPitch / sizeof(double) + i + 1] * pow(sin(2 * pow(solution[threadId*solutionsPitch / sizeof(double) + i], 2) / 3.141592), 20))));
	}

	return sum;
}

__device__ double MASTERS_COSINE_WAVE(double *solution, size_t solutionsPitch, int dimensions){
	int threadId = threadIdx.x;

	double sum = 0.0;

	
	for (int i = 0; i < dimensions; i++){
		sum = sum + (exp(-1.0*(pow(solution[threadId*solutionsPitch / sizeof(double) + i], 2) + pow(solution[threadId*solutionsPitch / sizeof(double) + i + 1], 2) + 0.5*solution[threadId*solutionsPitch / sizeof(double) + i] * solution[threadId*solutionsPitch / sizeof(double) + i + 1]) / 8)
			*cos(4 * sqrt(pow(solution[threadId*solutionsPitch / sizeof(double) + i], 2) + pow(solution[threadId*solutionsPitch / sizeof(double) + i + 1], 2) + 0.5*solution[threadId*solutionsPitch / sizeof(double) + i] * solution[threadId*solutionsPitch / sizeof(double) + i + 1])));
	}


	return sum;
}

//double SHEKELS_FOXHOLE(double *SOL, int NVAR){
//	int i, j;
//	double sum = 0, tmp;
//	double C[] = { 0.806, 0.517, 0.1, 0.908, 0.965, 0.669, 0.524, 0.902, 0.351, 0.876, 0.462,
//		0.491, 0.463, 0.741, 0.352, 0.869, 0.813, 0.811, 0.0828, 0.964, 0.789, 0.360, 0.369,
//		0.992, 0.332, 0.817, 0.632, 0.883, 0.608, 0.326 };
//	double A[][10] = { { 9.681, 0.667, 4.783, 9.095, 3.517, 9.325, 6.544, 0.211, 5.122, 2.02 },
//	{ 9.4, 2.041, 3.788, 7.931, 2.882, 2.672, 3.568, 1.284, 7.033, 7.374 },
//	{ 8.025, 9.152, 5.114, 7.621, 4.564, 4.711, 2.996, 6.126, 0.734, 4.982 },
//	{ 2.196, 0.415, 5.649, 6.979, 9.510, 9.166, 6.304, 6.054, 9.377, 1.426 },
//	{ 8.074, 8.777, 3.467, 1.863, 6.708, 6.349, 4.534, 0.276, 7.633, 1.567 },
//	{ 7.650, 5.658, 0.720, 2.764, 3.278, 5.283, 7.474, 6.274, 1.409, 8.208 },
//	{ 1.256, 3.605, 8.623, 6.905, 4.584, 8.133, 6.071, 6.888, 4.187, 5.448 },
//	{ 8.314, 2.261, 4.24, 1.781, 4.124, 0.932, 8.129, 8.658, 1.208, 5.762 },
//	{ 0.226, 8.858, 1.42, 0.954, 1.622, 4.698, 6.228, 9.096, 0.972, 7.637 },
//	{ 7.305, 2.228, 1.242, 5.928, 9.133, 1.826, 4.06, 5.204, 8.713, 8.247 },
//	{ 0.652, 7.027, 0.508, 4.876, 8.807, 4.632, 5.808, 6.937, 3.291, 7.016 },
//	{ 2.699, 3.516, 5.847, 4.119, 4.461, 7.496, 8.817, 0.69, 6.593, 9.789 },
//	{ 8.327, 3.897, 2.017, 9.57, 9.825, 1.15, 1.395, 3.885, 6.354, 0.109 },
//	{ 2.132, 7.006, 7.136, 2.641, 1.882, 5.943, 7.273, 7.691, 2.88, 0.564 },
//	{ 4.707, 5.579, 4.08, 0.581, 9.698, 8.542, 8.077, 8.515, 9.231, 4.67 },
//	{ 8.304, 7.559, 8.567, 0.322, 7.128, 8.392, 1.472, 8.524, 2.277, 7.826 },
//	{ 8.632, 4.409, 4.832, 5.768, 7.05, 6.715, 1.711, 4.323, 4.405, 4.591 },
//	{ 4.887, 9.112, 0.17, 8.967, 9.693, 9.867, 7.508, 7.77, 8.382, 6.74 },
//	{ 2.44, 6.686, 4.299, 1.007, 7.008, 1.427, 9.398, 8.48, 9.95, 1.675 },
//	{ 6.306, 8.583, 6.084, 1.138, 4.350, 3.134, 7.853, 6.061, 7.457, 2.258 },
//	{ 0.652, 2.343, 1.37, 0.821, 1.31, 1.063, 0.689, 8.819, 8.833, 9.07 },
//	{ 5.558, 1.272, 5.756, 9.857, 2.279, 2.764, 1.284, 1.677, 1.244, 1.234 },
//	{ 3.352, 7.549, 9.817, 9.437, 8.687, 4.167, 2.57, 6.54, 0.228, 0.027 },
//	{ 8.798, 0.88, 2.37, 0.168, 1.701, 3.68, 1.231, 2.39, 2.499, 0.064 },
//	{ 1.46, 8.057, 1.337, 7.217, 7.914, 3.615, 9.981, 9.198, 5.292, 1.224 },
//	{ 0.432, 8.645, 8.774, 0.249, 8.081, 7.461, 4.416, 0.652, 4.002, 4.644 },
//	{ 0.679, 2.8, 5.523, 3.049, 2.968, 7.225, 6.73, 4.199, 9.614, 9.229 },
//	{ 4.263, 1.074, 7.286, 5.599, 8.291, 5.2, 9.214, 8.272, 4.398, 4.506 },
//	{ 9.496, 4.83, 3.15, 8.27, 5.079, 1.231, 5.731, 9.494, 1.883, 9.732 },
//	{ 4.138, 2.562, 2.532, 9.661, 5.611, 5.5, 6.886, 2.341, 9.699, 6.5 } };
//
//	for (i = 1; i <= 30; i++){
//		tmp = 0;
//		for (j = 1; j <= NVAR; j++){
//			tmp = tmp + pow((solution[threadId*solutionsPitch + i] - A[threadId][j]), 2);
//		}
//		sum = sum + (1 / (C[threadId] + tmp));
//	}
//	sum = -1.0*sum;
//
//	return sum;
//}

__device__ double Problem(double * solution, size_t solutionPitch, int dimensions, int probId){
	double val = 0;

	//solution--;
	if (probId == 1){
		val = SCHWEFEL(solution, solutionPitch, dimensions);
	}
	else if (probId == 2){
		val = DE_JONG_ONE(solution, solutionPitch, dimensions);
	}
	else if (probId == 3){
		val = DE_JONG_THREE(solution, solutionPitch, dimensions);
	}
	else if (probId == 4){
		val = DE_JONG_FOUR(solution, solutionPitch, dimensions);
	}
	else if (probId == 5){
		val = ROSENBROCK(solution, solutionPitch, dimensions);
	}
	else if (probId == 6){
		val = RASTRIGIN(solution, solutionPitch, dimensions);
	}
	else if (probId == 7){
		val = GRIEWANGK(solution, dimensions, dimensions);
	}
	else if (probId == 8){
		val = SINE_ENVELOPE_SINE_WAVE(solution, solutionPitch, dimensions);
	}
	else if (probId == 9){
		val = STRETCH_V_SINE_WAVE(solution, solutionPitch, dimensions);
	}
	else if (probId == 10){
		val = ACKLEY_ONE(solution, solutionPitch, dimensions);
	}
	else if (probId == 11){
		val = ACKLEY_TWO(solution, solutionPitch, dimensions);
	}
	else if (probId == 12){
		val = EGG_HOLDER(solution, solutionPitch, dimensions);
	}
	else if (probId == 13){
		val = PATHOLOGICAL(solution, solutionPitch, dimensions);
	}
	else if (probId == 14){
		val = MICHALEWICZ(solution, solutionPitch, dimensions);
	}
	else if (probId == 15){
		val = MASTERS_COSINE_WAVE(solution, solutionPitch, dimensions);
	}
	/*else if (probId == 16){
	val = SHEKELS_FOXHOLE(solution, dimensions);
	}*/

	return val;
}

#endif