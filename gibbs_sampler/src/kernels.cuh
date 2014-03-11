/*
 * parameters.cuh
 *
 *  Created on: Jul 2, 2013
 *      Author: brandonkelly
 */

#ifndef KERNELS_H__
#define KERNELS_H__

// Cuda Includes
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

// local includes
#include "parameters.hpp"

// FUNCTION DEFINITIONS

__device__ __host__
void chol_update_r1(double* cholfactor, double* v, int dim_v, bool downdate);

__device__ __host__
void Propose(double* chi, double* cholfact, double* proposed_chi, double* snorm_deviate,
		double* scaled_proposal, int pchi, curandState* p_state);

__device__ __host__
void AdaptProp(double* cholfact, double* snorm_deviate, double* scaled_proposal,
		double metro_ratio, int pchi, int current_iter);

__device__ __host__
bool AcceptProp(double logdens_prop, double logdens_current, double forward_dens,
		double backward_dens, double& ratio, curandState* p_state);


// Initialize the parallel random number generator state on the device
__global__ void initialize_rng(curandState *state);

// calculate initial value of characteristics
template<int mfeat, int pchi> __global__
void initial_chi_value(double* chi, double* meas, double* meas_unc, double* cholfact, double* logdens,
		int ndata, pLogDensMeas LogDensityMeas);

// kernel to update the values of the characteristics in parallel on the GPU
template<int mfeat, int pchi, int dtheta> __global__
void update_characteristic(double* meas, double* meas_unc, double* chi, double* theta, double* cholfact,
		double* logdens_meas, double* logdens_pop, curandState* devStates, pLogDensMeas LogDensityMeas,
		pLogDensPop LogDensityPop, int current_iter, int* naccept, int ndata);

// compute the conditional log-posterior density of the characteristics given the population parameter
template<int mfeat, int pchi> __global__
void logdensity_meas(double* meas, double* meas_unc, double* chi, double* logdens, pLogDensMeas LogDensityMeas,
		int ndata);

// compute the conditional log-posterior density of the characteristics given the population parameter
template<int pchi, int dtheta> __global__
void logdensity_pop(double* theta, double* chi, double* logdens, pLogDensPop LogDensityPop, int ndata);

#endif /* KERNELS_H__ */
