#ifndef __PARAMETER_H
#define __PARAMETER_H

typedef struct Parameters {
	int dimensions;					//!< dimension
	int numberSolutions;			//!< number of solutions
	int problemId;					//!< problem id

	int iterations;					//!< number of iterations (cca 500)
	double inertialC;				//!< inertial coefficient (0.8 - 1.2)
	double personalBestC;			//!< personal best coefficient (0.0 - 2.0)
	double globalBestC;				//!< global best coefficient (0.0 - 2.0)
	double velocityC;				//!< velocity coefficient (0.0 - 1.0)
	double exponentiateVelocityC;	//!< value indicating whether the velocityC is exponentiated (0 or 1)
	double *dev_bounds;				//!< search space bounds
	size_t boundsPitch;
} Parameters;

//! allocate and initialise parameters

/**
* Initializes the parameters by reading from a given text file. /SERIAL/
* @param paramsFile The path of the file specifying the parameters to start the algorithm with.
* @return The initialized parameters.
*/
Parameters *InitializeParameters(const char *paramsFile);

/**
* Cleans up the memory used to store the parameters. /SERIAL/
* @param parameter The parameters that will be cleaned up.
*/
void FreeParameter(Parameters *parameter);

#endif