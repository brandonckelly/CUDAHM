/*
 * UnitTests.hpp
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

#include "device_launch_parameters.h"

// standard includes
#include <vector>

class UnitTests {
public:
	// constructor
	UnitTests(int n, int p, int m, int d, dim3& nB, dim3& nT) : ndata(n), mfeat(m), pchi(p), dim_theta(d),
		nBlocks(nB), nThreads(nT)
	{
		// fill data arrays
	    meas = new double* [ndata];
	    meas_unc = new double* [ndata];
	    for (int i = 0; i < ndata; ++i) {
			meas[i] = new double [mfeat];
			meas_unc[i] = new double [mfeat];
			for (int j = 0; j < mfeat; ++j) {
				meas[i][j] = 0.0;
				meas_unc[i][j] = 0.0;
			}
		}
	    // resize host vectors of parameters
	    true_chi.resize(pchi);
	    true_theta.resize(dim_theta);

	    epsilon = 1e-6; // set fractional precision for equality tests
	    nperformed = 0;
	    npassed = 0;
	}

	// destructor
	virtual ~UnitTests();

	// test rank-1 cholesky update
	void R1CholUpdate();
	// check that Chi::Propose follows a multivariate normal distribution
	void ChiPropose();
	// check that Chi::Accept always accepts when the proposal and the current values are the same
	void ChiAcceptSame();
	// make sure we accept and save a Chi value with a much higher posterior
	void ChiAcceptBetter();
	// test Chi::Adapt acceptance rate and covariance by running a simple MCMC sampler
	void ChiAdapt();
	// check that PopulationPar::Propose follow a multivariate normal distribution
	void ThetaPropose();
	// check that PopulationPar::Accept always accepts when the logdensities are the same
	void ThetaAcceptSame();
	// make sure we accept and save a population parameter value with a much higher posterior
	void ThetaAcceptBetter();
	// test PopulationPar::Adapt acceptance rate and covariance by running a simple MCMC sampler
	void ThetaAdapt();
	// check that constructor for population parameter correctly set the pointer data member of DataAugmentation
	void DaugPopPtr();
	// test DataAugmentation::GetChi
	void DaugGetChi();
	// check that DataAugmentation::Update always accepts when the proposed and current chi values are the same
	void DaugAcceptSame();
	// make sure that DataAugmentation::Update() accepts and saves Chi values when the posterior is much higher
	void DaugAcceptBetter();

	// print out summary of test results
	void Finish();

private:
	// array sizes
	int ndata, pchi, mfeat, dim_theta;
	// data
	double** meas;
	double** meas_unc;
	// true values of parameters
	std::vector<double> true_chi;
	std::vector<double> true_theta;
	// CUDA grid launch parameters
	dim3& nBlocks;
	dim3& nThreads;
	// fractional precision of equality tests
	double epsilon;
	// number of tests performed and passed
	int npassed;
	int nperformed;
};

#endif /* UNITTESTS_HPP_ */
