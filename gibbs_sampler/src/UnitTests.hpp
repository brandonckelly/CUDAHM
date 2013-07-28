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

// standard includes
#include <vector>
// local includes
#include "parameters.hpp"
#include "GibbsSampler.hpp"

class UnitTests {
public:
	// constructor
	UnitTests(int n, dim3& nB, dim3& nT) : ndata(n), nBlocks(nB), nThreads(nT)
	{
		mfeat = 3;
		pchi = 3;
		dim_theta = 3;
		// known population parameters
	    h_true_theta.resize(dim_theta);
		double mu[3] = {1.2, -0.4, 3.4};
	    h_true_theta[0] = mu[0];
	    h_true_theta[1] = mu[1];
	    h_true_theta[2] = mu[2];

	    covar[0][0] = 5.29;
	    covar[0][1] = 0.3105;
	    covar[0][2] = -15.41;
	    covar[1][0] = 0.3105;
	    covar[1][1] = 0.2025;
	    covar[1][2] = 3.2562;
	    covar[2][0] = -15.41;
	    covar[2][1] = 3.2562;
	    covar[2][2] = 179.56;

		cholfact[0] = 2.3;
		cholfact[1] = 0.135;
		cholfact[2] = 0.42927264;
		cholfact[3] = -6.7;
		cholfact[4] = 9.6924416;
		cholfact[5] = 6.38173768;

		covar_inv[0] = 0.64880351;
		covar_inv[1] = -2.66823952;
		covar_inv[2] = 0.10406763;
		covar_inv[3] = -2.66823952;
		covar_inv[4] = 17.94430089;
		covar_inv[5] = -0.55439855;
		covar_inv[6] = 0.10406763;
		covar_inv[7] = -0.55439855;
		covar_inv[8] = 0.02455399;

		// generate some chi values as chi_i|theta ~ N(mu,covar)
	    h_true_chi.resize(ndata * pchi);
		thrust::fill(h_true_chi.begin(), h_true_chi.end(), 0.0);
		for (int i = 0; i < ndata; ++i) {
			double snorm_deviate[3];
			for (int j = 0; j < dim_theta; ++j) {
				snorm_deviate[j] = snorm(rng);
			}
			int cholfact_index = 0;
			for (int j = 0; j < dim_theta; ++j) {
				h_true_chi[j * ndata + i] = mu[j];
				for (int k = 0; k < (j+1); ++k) {
					h_true_chi[j * ndata + i] += cholfact[cholfact_index] * snorm_deviate[k];
					cholfact_index++;
				}
			}
		}
		d_true_chi = h_true_chi;
		d_true_theta = h_true_theta;

		// fill data arrays
	    meas = new double* [ndata];
	    meas_unc = new double* [ndata];
	    double meas_err[3] = {1.2, 0.4, 0.24};
	    for (int i = 0; i < ndata; ++i) {
			meas[i] = new double [mfeat];
			meas_unc[i] = new double [mfeat];
			for (int j = 0; j < mfeat; ++j) {
				// y_ij|chi_ij ~ N(chi_ij, meas_err_j^2)
				meas[i][j] = h_true_chi[j * ndata + i] + meas_err[j] * snorm(rng);
				meas_unc[i][j] = meas_err[j];
			}
		}

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
	// test Chi::Adapt acceptance rate and covariance by running a simple MCMC sampler
	void ChiAdapt();
	// check that PopulationPar::Propose follow a multivariate normal distribution
	void ThetaPropose();
	// check that PopulationPar::Accept always accepts when the logdensities are the same
	void ThetaAcceptSame();
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
	int ndata, pchi, mfeat, dim_theta;
	// data
	double** meas;
	double** meas_unc;
	// true values of parameters
	hvector h_true_chi;
	dvector d_true_chi;
	hvector h_true_theta;
	dvector d_true_theta;
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
