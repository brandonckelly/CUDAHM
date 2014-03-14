/*
 * UnitTests.cuh
 *
 *  Created on: Jul 17, 2013
 *      Author: brandonkelly
 * list of unit tests:
 *
 *	- test rank-1 cholesky update
 *	- make sure Chi::Propose follows a multivariate normal distribution
 *	- make sure Chi::Accept always accepts when the proposal and the current values are the same
 *	- make sure we accept and save a Chi value with a much higher posterior
 *	- Test Chi::Adapt acceptance rate and covariance by running a simple MCMC sampler
 *	- make sure PopulationPar::Propose follow a multivariate normal distribution
 *	- make sure that PopulationPar::Accept always accepts when the logdensities are the same
 *	- make sure PopulationPar::Update always accepts when the proposed and current theta values are the same
 *	- make sure we accept and save a PopulationPar value with the posterior is much higher
 *	- Test PopulationPar::Adapt acceptance rate and covariance by running a simple MCMC sampler
 *	- Test DataAugmentation::GetChi
 *	- make sure DataAugmentation::Update always accepts when the proposed and current chi values are the same
 *	- make sure we accept and save a Chi value when the posterior is much higher
 *
 */

#ifndef UNITTESTS_HPP_
#define UNITTESTS_HPP_

// standard includes
#include <vector>
// local includes
#include "GibbsSampler.hpp"

// function definitions
__device__ __host__
double LogDensityMeas(double* chi, double* meas, double* meas_unc);

__device__ __host__
double LogDensityPop(double* chi, double* theta);

// class definition for running the unit tests

class UnitTests {
public:
	// constructor
	UnitTests(int n, dim3& nB, dim3& nT);

	// destructor
	virtual ~UnitTests();

	// save the measurement values to a text file
	void SaveMeasurements();

	// test rank-1 cholesky update
	void R1CholUpdate();
	// check that Chi::Propose follows a multivariate normal distribution
	void ChiPropose();
	// check that Chi::Accept always accepts when the proposal and the current values are the same
	void ChiAcceptSame();
	// test Chi::Adapt acceptance rate and covariance by running a simple MCMC sampler
	void ChiAdapt();
	// check that PopulationPar::Propose follow a multivariate normal distribution
	void ThetaPropose();
	// check that PopulationPar::Accept always accepts when the logdensities are the same
	void ThetaAcceptSame();
	// test PopulationPar::Adapt acceptance rate and covariance by running a simple MCMC sampler
	void ThetaAdapt();
	// test DataAugmentation::GetChi
	void DaugGetChi();
	// check that pointers to device-side LogDensityMeas and LogDensityPop are properly set
	void DaugLogDensPtr();
	// check that proposals for chi generated on the GPU have correct distribution
	void DevicePropose();
	// check that the device-side Accept method works and updates the chi-values
	void DeviceAccept();
	// check that the device-side Adapt method works and updates the cholesky factor of the chi proposal covariances
	void DeviceAdapt();
	// check that DataAugmentation::Update always accepts when the proposed and current chi values are the same
	void DaugAcceptSame();
	// make sure that DataAugmentation::Update() accepts and saves Chi values when the posterior is much higher
	void DaugAcceptBetter();
	// check that constructor for GibbsSampler correctly sets the DataAugmentation and PopulationPar pointers.
	void GibbsSamplerPtr();
	// test the Gibbs Sampler for a Normal-Normal model keeping the characteristics fixed
	void FixedChar();
	// test the Gibbs Sampler for a Normal-Normal model keeping the population parameter fixed
	void FixedPopPar();
	// test the Gibbs Sampler for a Normal-Normal model
	void NormNorm();
	// print out summary of test results
	void Finish();

private:
	// array sizes
	int ndata;
	// data
	double** meas;
	hvector h_meas;
	double** meas_unc;
	hvector h_meas_unc;
	// true values of parameters
	hvector h_true_chi;
	dvector d_true_chi;
	hvector h_true_theta;
	// parameters related to population covariance
	double covar[3][3];
	double covar_inv[9];
	double cholfact[6];
	// CUDA grid launch parameters
	dim3& nBlocks;
	dim3& nThreads;
	// number of tests performed and passed
	int npassed;
	int nperformed;
};

#endif /* UNITTESTS_HPP_ */
