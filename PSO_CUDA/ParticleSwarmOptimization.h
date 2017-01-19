#ifndef __PSO_H
#define __PSO_H

#include <stdlib.h>
#include <stdio.h>
#include "Parameter.h"
#include "Population.h"

/**
* Calculates the new velocity using the initial velocity, the personal best of every particle in the
* population and the global best. /PARALLEL/
* @param population The populations whose velocity will be calculated.
* @param parameters The parameters that describe the input given by the user to unfluence the 
*					simulation.
* @param iteration The current iteration of the PSO.
* @param
*/
void CalculateVelocity(Population *population, Parameters *parameters, int iteration);


/**
* Moves the particles to their new positions. /PARALLEL/
* @param population The populations whose particles will be moved.
* @param parameters The parameters that describe the input given by the user to unfluence the 
*					simulation.
*/
void MoveParticles(Population *population, Parameters *parameters);


/**
* Updates the best solution the populations has ever collectively found. /SERIAL BUT ON DEVICE/
* @param population The populations that will be updated.
* @param parameters The parameters that describe the input given by the user to unfluence the 
*					simulation.
* @param iteration The current iteration of the PSO.
*/
void UpdateGlobalBest(Population *population, Parameters *parameters, int iteration);

/**
* Updates the best solution every single particle in the population has ever found. 
* /PARALLEL/
* @param population The populations that will be updated.
* @param parameters The parameters that describe the input given by the user to unfluence the 
*					simulation.
*/
void UpdatePersonalBest(Population *population, Parameters *parameters);

#endif