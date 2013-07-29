/*
 * DataAugmentation.hpp
 *
 *  Created on: Jul 28, 2013
 *      Author: brandonkelly
 */

#ifndef _DATA_AUGMENTATION_HPP__
#define _DATA_AUGMENTATION_HPP__

// macro for checking for CUDA errors
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

// Cuda Includes
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

// Standard includes
#include <cmath>
#include <vector>
#include <stdio.h>

// Boost includes
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

// Thrust includes
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// convenience typedefs
typedef std::vector<std::vector<double> > vecvec;
typedef thrust::host_vector<double> hvector;
typedef thrust::device_vector<double> dvector;

// pointers to functions that must be supplied by the user for computed the conditional log-densities
typedef double (*pLogDensMeas)(double*, double*, double*, int, int);
typedef double (*pLogDensPop)(double*, double*, int, int);

class PopulationPar; // forward declaration

// class for a data augmentation.
class DataAugmentation
{
public:
	// Constructor
	DataAugmentation(double** meas, double** meas_unc, int n, int m, int p, dim3& nB, dim3& nT);

	virtual ~DataAugmentation() { cudaFree(p_devStates); }

	// launch the update kernel on the GPU
	void Update();

	// setters and getters
	void SetChi(dvector& chi, bool update_logdens = true);

	// make sure that the data augmentation knows about the population parameters
	void SetPopulationPtr(PopulationPar* t) { p_Theta = t; }

	void SetLogDens(dvector& logdens) {
		d_logdens = logdens;
		h_logdens = d_logdens;
	}
	void SetCholFact(dvector& cholfact) {
		d_cholfact = cholfact;
		h_cholfact = d_cholfact;
	}

	vecvec GetChi(); // return the value of the characteristic in a std::vector of std::vectors for convenience
	hvector GetHostLogDens() { return h_logdens; }
	dvector GetDevLogDens() { return d_logdens; }
	double* GetDevLogDensPtr() { return thrust::raw_pointer_cast(&d_logdens[0]); }
	hvector GetHostChi() { return h_chi; }
	dvector GetDevChi() { return d_chi; }
	double* GetDevChiPtr() { return thrust::raw_pointer_cast(&d_chi[0]); }
	int GetDataDim() { return ndata; }
	int GetChiDim() { return pchi; }
	PopulationPar* GetPopulationPtr() { return p_Theta; }
	pLogDensMeas GetLogDensPtr() { return p_logdens_function; }
	thrust::host_vector<int> GetNaccept() {
		h_naccept = d_naccept;
		return h_naccept;
	}

protected:
	// set the sizes of the data members
	void _SetArraySizes();
	// measurements and their uncertainties
	hvector h_meas;
	hvector h_meas_unc;
	dvector d_meas;
	dvector d_meas_unc;
	int ndata;
	int mfeat;
	int pchi;
	// characteristics
	hvector h_chi;
	dvector d_chi;
	// population-level parameters
	PopulationPar* p_Theta;
	// logarithm of conditional posterior densities
	hvector h_logdens; // probability of meas|chi
	dvector d_logdens;
	// cholesky factors of Metropolis proposal covariance matrix
	hvector h_cholfact;
	dvector d_cholfact;
	// state of parallel random number generator on the device
	curandState* p_devStates;
	// CUDA kernel launch specifications
	dim3& nBlocks;
	dim3& nThreads;
	// pointer to device-side function that compute the conditional log-posterior of measurements|characteristics
	pLogDensMeas p_logdens_function;
	// MCMC sampler parameters
	int current_iter;
	thrust::host_vector<int> h_naccept;
	thrust::device_vector<int> d_naccept;
};

// class for a population level parameter
class PopulationPar
{
public:
	// constructors
	PopulationPar(int dtheta, dim3& nB, dim3& nT);
	PopulationPar(int dtheta, DataAugmentation* D, dim3& nB, dim3& nT);

	// calculate the initial value of the population parameters
	virtual void InitialValue();

	// return the log-prior of the population parameters
	virtual double LogPrior(hvector theta) { return 0.0; }

	// propose a new value of the population parameters
	virtual hvector Propose();

	// adapt the covariance matrix (i.e., the cholesky factors) of the theta proposals
	virtual void AdaptProp(double metro_ratio);

	// calculate whether to accept or reject the metropolist-hastings proposal
	bool AcceptProp(double logdens_prop, double logdens_current, double& ratio, double forward_dens = 0.0, double backward_dens = 0.0);

	// update the value of the population parameter value using a robust adaptive metropolis algorithm
	virtual void Update();

	// setters and getters
	void SetDataAugPtr(DataAugmentation* DataAug) { Daug = DataAug; }
	void SetTheta(dvector& theta, bool update_logdens = true);
	void SetLogDens(dvector& logdens) {
		h_logdens = logdens;
		d_logdens = logdens;
	}
	void SetCholFact(hvector cholfact_new) { cholfact = cholfact_new; }
	void SetCurrentIter(int iter) { current_iter = iter; }

	hvector GetHostTheta() { return h_theta; }
	dvector GetDevTheta() { return d_theta; }
	std::vector<double> GetTheta() { // return the current value of theta as a std::vector
		std::vector<double> std_theta(h_theta.size());
		thrust::copy(h_theta.begin(), h_theta.end(), std_theta.begin());
		return std_theta;
	}
	double* GetDevThetaPtr() { return thrust::raw_pointer_cast(&d_theta[0]); }
	hvector GetLogDens() {
		h_logdens = d_logdens;
		return h_logdens;
	}
	dvector GetDevLogDens() { return d_logdens; }
	double* GetDevLogDensPtr() { return thrust::raw_pointer_cast(&d_logdens[0]); }
	int GetDim() { return dim_theta; }
	hvector GetCholFactor() { return cholfact; }
	int GetNaccept() { return naccept; }
	pLogDensPop GetLogDensPtr() { return p_logdens_function; }

protected:
	int dim_theta;
	int pchi;
	// the value of the population parameter
	hvector h_theta;
	dvector d_theta;
	// log of the value the probability of the characteristics given the population parameter
	hvector h_logdens;
	dvector d_logdens;
	// make sure that the population parameter knows about the characteristics
	DataAugmentation* Daug;
	// cholesky factors of Metropolis proposal covariance matrix
	hvector cholfact;
	// interval variables used in robust adaptive metropolis algorithm
	hvector snorm_deviate;
	hvector scaled_proposal;
	// CUDA kernel launch specifications
	dim3& nBlocks;
	dim3& nThreads;
	// pointer to device-side function that compute the conditional log-posterior of characteristics|population
	pLogDensPop p_logdens_function;
	// MCMC parameters
	int naccept;
	int current_iter;
};

#endif // _DATA_AUGMENTATION_HPP__ //
