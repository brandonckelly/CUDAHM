/*
 * UnitTests.cpp
 *
 *  Created on: Jul 17, 2013
 *      Author: brandonkelly
 */

// standard includes
#include <iostream>
#include <fstream>
#include <cassert>

// local includes
#include "UnitTests.hpp"
#include "kernels.cuh"

// Global random number generator and distributions for generating random numbers on the host. The random number generator used
// is the Mersenne Twister mt19937 from the BOOST library. These are instantiated in data_augmentation.cu.
extern boost::random::mt19937 rng;
extern boost::random::normal_distribution<> snorm; // Standard normal distribution
extern boost::random::uniform_real_distribution<> uniform; // Uniform distribution from 0.0 to 1.0

/*
 * Pointers to the device-side functions used to compute the conditional log-posteriors
 */
__constant__ pLogDensMeas c_LogDensMeas = LogDensityMeas;
__constant__ pLogDensPop c_LogDensPop = LogDensityPop;

// calculate transpose(x) * covar_inv * x for an nx-element vector x and an (nx,nx)-element matrix covar_inv
__host__ __device__
double mahalanobis_distance(double** covar_inv, double* x, int nx) {
	double distance = 0.0;
	double* x_temp;
	x_temp = new double [nx];
	for (int i = 0; i < nx; ++i) {
		x_temp[i] = 0.0;
		for (int j = 0; j < nx; ++j) {
			x_temp[i] += covar_inv[i][j] * x[j];
		}
	}
	for (int i = 0; i < nx; ++i) {
		distance += x_temp[i] * x[i];
	}
	delete x_temp;
	return distance;
}

// calculate transpose(x) * covar_inv * x
__device__ __host__
double ChiSqr(double* x, double* covar_inv, int nx)
{
	double chisqr = 0.0;
	for (int i = 0; i < nx; ++i) {
		for (int j = 0; j < nx; ++j) {
			chisqr += x[i] * covar_inv[i * nx + j] * x[j];
		}
	}
	return chisqr;
}

// compute the conditional log-posterior density of the measurements given the characteristic
__device__ __host__
double LogDensityMeas(double* chi, double* meas, double* meas_unc, int mfeat, int pchi)
{
	double logdens = 0.0;
	for (int i = 0; i < 3; ++i) {
		double chi_std = (meas[i] - chi[i]) / meas_unc[i];
		logdens += -0.5 * chi_std * chi_std;
	}

	return logdens;
}

// compute the conditional log-posterior density of the characteristic given the population mean
__device__ __host__
double LogDensityPop(double* chi, double* theta, int pchi, int dim_theta)
{
	// known inverse covariance matrix of the characteristics
	double covar_inv[9] =
	{
			0.64880351, -2.66823952, 0.10406763,
			-2.66823952, 17.94430089, -0.55439855,
			0.10406763, -0.55439855, 0.02455399
	};
	// subtract off the population mean
	double chi_cent[3];
	for (int i = 0; i < 3; ++i) {
		chi_cent[i] = chi[i] - theta[i];
	}

	double logdens = -0.5 * ChiSqr(chi_cent, covar_inv, 3);
	return logdens;
}


double mean(double* x, int nx) {
	double mu = 0.0;
	for (int i = 0; i < nx; ++i) {
		mu += x[i] / nx;
	}
	return mu;
}

double variance(double* x, int nx) {
	double sigsqr = 0.0;
	double mu = mean(x, nx);
	for (int i = 0; i < nx; ++i) {
		sigsqr += (x[i] - mu) * (x[i] - mu);
	}
	sigsqr /= nx;
	return sigsqr;
}

// constructor for class that performs unit tests
UnitTests::UnitTests(int n, dim3& nB, dim3& nT) : ndata(n), nBlocks(nB), nThreads(nT)
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
UnitTests::~UnitTests() {
	// free memory used by data arrays
	for (int i = 0; i < ndata; ++i) {
		delete [] meas[i];
		delete [] meas_unc[i];
	}
	delete meas;
	delete meas_unc;
}

// test rank-1 cholesky update
void UnitTests::R1CholUpdate() {

	double v[3] = {-0.39108095, -0.0668706, -0.30427621};
	// cholesky factor for update (covar + v * transpose(v)), computed from python
	double Lup0[6] = {2.33301185, 0.14429923, 0.43145036, -6.55419017, 9.78632116,  6.39711602};
	// cholesky factor for downdate (covar - v * transpose(v)), computed from python
	double Ldown0[6] = {2.26650738, 0.12545654, 0.42695313, -6.85150942, 9.59219932,  6.36505672};

	// now get rank-1 updated and downdated factors using the fast method
	double Lup[6] = {2.3, 0.135, 0.42927264, -6.7, 9.6924416, 6.38173768};
	double vup[3] = {-0.39108095, -0.0668706, -0.30427621};
	bool downdate = false;
	chol_update_r1(Lup, vup, 3, downdate);

	double Ldown[6] = {2.3, 0.135, 0.42927264, -6.7, 9.6924416, 6.38173768};
	double vdown[3] = {-0.39108095, -0.0668706, -0.30427621};
	downdate = true;
	chol_update_r1(Ldown, vdown, 3, downdate);

	// test if the cholesky factors agree by finding the maximum fraction difference between the two
	double max_frac_diff_up = 0.0;
	double max_frac_diff_down = 0.0;
	for (int i = 0; i < 6; ++i) {
		double frac_diff = abs(Lup[i] - Lup0[i]) / abs(Lup0[i]);
		max_frac_diff_up = max(frac_diff, max_frac_diff_up);
		frac_diff = abs(Ldown[i] - Ldown0[i]) / abs(Ldown0[i]);
		max_frac_diff_down = max(frac_diff, max_frac_diff_down);
	}

	if ((max_frac_diff_down < 1e-6) && (max_frac_diff_up < 1e-6)) {
		// cholesky factors agree, so test passed
		npassed++;
	} else {
		// test failed
		std::cerr << "Rank-1 Cholesky update test failed." << std::endl;
	}

	nperformed++;
}

// check that Propose follows a multivariate normal distribution
void UnitTests::ChiPropose() {

	double** covar_inv_local;
	covar_inv_local = new double* [3];
	for (int i = 0; i < 3; ++i) {
		covar_inv_local[i] = new double [3];
	}
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			covar_inv_local[i][j] = covar_inv[j * 3 + i];
		}
	}

	int p = 3, ntrials = 100000;

	double snorm_deviate[p];
	double scaled_proposal[p];
	double proposed_chi[p];
	double chi[3] = {1.2, 0.4, -0.7};
	double chisqr[ntrials];

	curandState* p_state;

	for (int i = 0; i < ntrials; ++i) {
		// get the ntrials proposals
		Propose(chi, cholfact, proposed_chi, snorm_deviate, scaled_proposal, pchi, p_state);
		for (int j = 0; j < p; ++j) {
			proposed_chi[j] -= chi[j]; // center the proposals
		}
		chisqr[i] = mahalanobis_distance(covar_inv_local, proposed_chi, p);
	}

	/*
	 * check that the values of chisqr are consistent with being drawn from a chi-square distribution
	 * with p = 3 degrees of freedom.
	*/

	// first compare the average with the known value
	double true_mean = p;
	double true_var = 2.0 * p;
	double mu_sigma = sqrt(true_var / ntrials); // standard deviation in the average
	double mean_chisqr = mean(chisqr, ntrials);

	double zdiff_mean = abs(mean_chisqr - true_mean) / mu_sigma;
	if (zdiff_mean < 3.0) {
		npassed++;
	} else {
		std::cerr << "Test for Chi::Propose failed: average chi-square value more than 3-sigma away from true value" << std::endl;
	}
	nperformed++;

	// compare empirical quantiles with known ones
	double chi2_low = 0.3, chi2_high = 8.0;
	int nlow_low = 3800, nlow_high = 4200; // # of chisqr < chi2_low should fall within this interval
	int nhigh_low = 95200, nhigh_high = 95600; // # of chisqr < chi2_high should fall within this interval
	int count_low = 0, count_high = 0;
	for (int i = 0; i < ntrials; ++i) {
		// count the number of elements of chisqr that are below the 4.0 and 95.4 percentiles
		if (chisqr[i] < chi2_low) {
			count_low++;
		}
		if (chisqr[i] < chi2_high) {
			count_high++;
		}
	}
	if ((count_low > nlow_low) && (count_low < nlow_high)) {
		npassed++;
	} else {
		std::cerr << "Test for Chi::Propose failed: empirical 4.0 percentile inconsistent with true value" << std::endl;
	}
	nperformed++;
	if ((count_high > nhigh_low) && (count_high < nhigh_high)) {
		npassed++;
	} else {
		std::cerr << "Test for Chi::Propose failed: empirical 95.4 percentile inconsistent with true value" << std::endl;
	}
	nperformed++;

	// free memory
	for (int i = 0; i < 3; ++i) {
		delete [] covar_inv_local[i];
	}
	delete covar_inv_local;
}

// check that Chi::Accept always accepts when the proposal and the current values are the same
void UnitTests::ChiAcceptSame() {
	int p = 3, ntrials = 100000;

	bool accept;
	int naccept = 0;
	double logdens = -1.32456;
	double ratio = 0.0;
	curandState* p_state;
	for (int i = 0; i < ntrials; ++i) {
		AcceptProp(logdens, logdens, 0.0, 0.0, ratio, p_state);
		if (abs(ratio - 1.0) < 1e-6) {
			naccept++;
		}
	}

	if (naccept == ntrials) {
		npassed++;
	} else {
		std::cerr << "Test for Chi::Accept failed: Failed to always accept when the log-posteriors are the same." << std::endl;
	}
	nperformed++;
}

// test Chi::Adapt acceptance rate and covariance by running a simple MCMC sampler for a 3-dimensional normal distribution
void UnitTests::ChiAdapt() {

	int p = 3, niter = 300000;
	int current_iter = 1;
	double target_rate = 0.4; // MCMC sampler target acceptance rate

	double chi[3];
	for (int j = 0; j < p; ++j) {
		chi[j] = 0.0; // initialize chi values to zero
	}

	// set initial covariance matrix of the chi proposals as the identity matrix
	int diag_index = 0;
	double cholfact[6];
	for (int j=0; j<p; j++) {
		cholfact[diag_index] = 1.0;
		diag_index += j + 2;
	}

	// run the MCMC sampler
	double theta[3] = {1.2, 0.4, -0.7};
	double snorm_deviate[p], scaled_proposal[p], proposed_chi[p];
	double logdens;
	logdens = LogDensityPop(chi, theta, pchi, dim_theta);
	int naccept = 0, start_counting = 10000;

	curandState* p_state;

	for (int i = 0; i < niter; ++i) {
		// propose a new value of chi
		Propose(chi, cholfact, proposed_chi, snorm_deviate, scaled_proposal, pchi, p_state);

		// get value of log-posterior for proposed chi value
		double logdens_prop;
		logdens_prop = LogDensityPop(proposed_chi, theta, pchi, dim_theta);

		// accept the proposed value of the characteristic?
		double metro_ratio = 0.0;
		bool accept = AcceptProp(logdens_prop, logdens, 0.0, 0.0, metro_ratio, p_state);

		// adapt the covariance matrix of the characteristic proposal distribution
		AdaptProp(cholfact, snorm_deviate, scaled_proposal, metro_ratio, pchi, current_iter);

		if (accept) {
			// accepted this proposal, so save new value of chi and log-densities
			for (int j=0; j<p; j++) {
				chi[j] = proposed_chi[j];
			}
			logdens = logdens_prop;
			if (current_iter >= start_counting) {
				// don't start counting # of accepted proposals until we've down start_counting iterations
				naccept++;
			}
		}
		current_iter++;
	}
	double accept_rate = double(naccept) / double(niter - start_counting);
	double frac_diff = abs(accept_rate - target_rate) / target_rate;
	// make sure acceptance rate is within 5% of the target rate
	if (frac_diff < 0.05) {
		npassed++;
	} else {
		std::cerr << "Test for Chi::Adapt failed: Acceptance rate is not within 5% of the target rate." << std::endl;
		std::cout << accept_rate << ", " << target_rate << std::endl;
	}
	nperformed++;
}

// check that PopulationPar::Propose follow a multivariate normal distribution
void UnitTests::ThetaPropose() {
	double** covar_inv_local;
	covar_inv_local = new double* [3];
	for (int i = 0; i < 3; ++i) {
		covar_inv_local[i] = new double [3];
	}
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			covar_inv_local[i][j] = covar_inv[i + 3*j];
		}
	}

	hvector cholfact(6);
	cholfact[0] = 2.3;
	cholfact[1] = 0.135;
	cholfact[2] = 0.42927264;
	cholfact[3] = -6.7;
	cholfact[4] = 9.6924416;
	cholfact[5] = 6.38173768;

	int ntrials = 100000;
	int current_iter = 1;
    PopulationPar Theta(dim_theta, nBlocks, nThreads);

    hvector h_theta(3);
    h_theta[0] = 1.2;
    h_theta[1] = 0.4;
    h_theta[2] = -0.7;
    dvector d_theta = h_theta;
    Theta.SetTheta(d_theta, false);
    Theta.SetCholFact(cholfact);

	double chisqr[ntrials];
	for (int i = 0; i < ntrials; ++i) {
		// get the ntrials proposals
		hvector theta_prop = Theta.Propose();
		for (int j = 0; j < dim_theta; ++j) {
			theta_prop[j] -= h_theta[j]; // center the proposals
		}
		double* p_theta = thrust::raw_pointer_cast(&theta_prop[0]);
		chisqr[i] = mahalanobis_distance(covar_inv_local, p_theta, dim_theta);
	}

	/*
	 * check that the values of chisqr are consistent with being drawn from a chi-square distribution
	 * with dt = 3 degrees of freedom.
	*/

	// first compare the average with the known value
	double true_mean = dim_theta;
	double true_var = 2.0 * dim_theta;
	double mu_sigma = sqrt(true_var / ntrials); // standard deviation in the average
	double mean_chisqr = mean(chisqr, ntrials);

	double zdiff_mean = abs(mean_chisqr - true_mean) / mu_sigma;
	if (zdiff_mean < 3.0) {
		npassed++;
	} else {
		std::cerr << "Test for Theta::Propose failed: average chi-square value more than 3-sigma away from true value" << std::endl;
	}
	nperformed++;

	// compare empirical quantiles with known ones
	double chi2_low = 0.3, chi2_high = 8.0;
	int nlow_low = 3800, nlow_high = 4200; // # of chisqr < chi2_low should fall within this interval
	int nhigh_low = 95200, nhigh_high = 95600; // # of chisqr < chi2_high should fall within this interval
	int count_low = 0, count_high = 0;
	for (int i = 0; i < ntrials; ++i) {
		// count the number of elements of chisqr that are below the 4.0 and 95.4 percentiles
		if (chisqr[i] < chi2_low) {
			count_low++;
		}
		if (chisqr[i] < chi2_high) {
			count_high++;
		}
	}
	if ((count_low > nlow_low) && (count_low < nlow_high)) {
		npassed++;
	} else {
		std::cerr << "Test for Theta::Propose failed: empirical 4.0 percentile inconsistent with true value" << std::endl;
	}
	nperformed++;
	if ((count_high > nhigh_low) && (count_high < nhigh_high)) {
		npassed++;
	} else {
		std::cerr << "Test for Theta::Propose failed: empirical 95.4 percentile inconsistent with true value" << std::endl;
	}
	nperformed++;

	// free memory
	for (int i = 0; i < 3; ++i) {
		delete [] covar_inv_local[i];
	}
	delete covar_inv_local;
}

// check that PopulationPar::Accept always accepts when the logdensities are the same
void UnitTests::ThetaAcceptSame() {
	int ntrials = 100000;
    PopulationPar Theta(dim_theta, nBlocks, nThreads);

	bool accept;
	int naccept = 0;
	double logdens = -1.32456;
	double ratio = 0.0;
	for (int i = 0; i < ntrials; ++i) {
		accept = Theta.AcceptProp(logdens, logdens, ratio);
		if (abs(ratio - 1.0) < 1e-6) {
			naccept++;
		}
	}

	if (naccept == ntrials) {
		npassed++;
	} else {
		std::cerr << "Test for Theta::Accept failed: Failed to always accept when the log-posteriors are the same." << std::endl;
	}
	nperformed++;
}

// test PopulationPar::Adapt acceptance rate and covariance by running a simple MCMC sampler
void UnitTests::ThetaAdapt() {

	double mu[3] = {1.2, 0.4, -0.7};

	PopulationPar Theta(dim_theta, nBlocks, nThreads);

	hvector h_theta = Theta.GetHostTheta();
	dvector d_theta = h_theta;
	double* p_theta = thrust::raw_pointer_cast(&h_theta[0]);

	// run the MCMC sampler
	double logdens_current = LogDensityPop(p_theta, mu, 3, 3);
	int naccept = 0, start_counting = 10000;
	int niter = 300000, current_iter = 1;
	double target_rate = 0.4; // MCMC sampler target acceptance rate

	for (int i = 0; i < niter; ++i) {
		// propose a new value of theta
		hvector theta_prop(dim_theta);
		theta_prop = Theta.Propose();
		double* p_theta_prop = thrust::raw_pointer_cast(&theta_prop[0]);
		// get value of log-posterior for proposed theta value
		double logdens_prop;
		logdens_prop = LogDensityPop(p_theta_prop, mu, 3, 3);

		// accept the proposed value of the characteristic?
		double metro_ratio = 0.0;
		bool accept = Theta.AcceptProp(logdens_prop, logdens_current, metro_ratio);

		// adapt the covariance matrix of the characteristic proposal distribution
		Theta.AdaptProp(metro_ratio);

		if (accept) {
			// accepted this proposal, so save new value of chi and log-densities
			d_theta = theta_prop;
			h_theta = theta_prop;
			Theta.SetTheta(d_theta, false);
			logdens_current = logdens_prop;
			if (current_iter >= start_counting) {
				// don't start counting # of accepted proposals until we've down start_counting iterations
				naccept++;
			}
		}
		current_iter++;
		Theta.SetCurrentIter(current_iter);
	}
	double accept_rate = double(naccept) / double(niter - start_counting);
	double frac_diff = abs(accept_rate - target_rate) / target_rate;
	// make sure acceptance rate is within 5% of the target rate
	if (frac_diff < 0.05) {
		npassed++;
	} else {
		std::cerr << "Test for Theta::Adapt failed: Acceptance rate is not within 5% of the target rate." << std::endl;
		std::cout << accept_rate << ", " << target_rate << std::endl;
	}
	nperformed++;
}

// check that constructor for population parameter correctly set the pointer data member of DataAugmentation
void UnitTests::DaugPopPtr() {
	DataAugmentation Daug(meas, meas_unc, ndata, mfeat, pchi, nBlocks, nThreads);
	PopulationPar Theta(dim_theta, &Daug, nBlocks, nThreads);

	PopulationPar* p_Theta = Daug.GetPopulationPtr();

	if (p_Theta == &Theta) {
		npassed++;
	} else {
		std::cerr << "Test for PopulationPar constructor failed: Pointer to DataAugmentation member not correctly initialized."
				<< std::endl;
	}
	nperformed++;
}

// test DataAugmentation::GetChi
void UnitTests::DaugGetChi() {

	DataAugmentation Daug(meas, meas_unc, ndata, mfeat, pchi, nBlocks, nThreads);
	hvector h_chi = Daug.GetHostChi();

	assert(h_chi.size() == ndata * pchi);

	// generate some chi-values
	double** chi2d;
    chi2d = new double* [ndata];
    for (int i = 0; i < ndata; ++i) {
		chi2d[i] = new double [pchi];
		for (int j = 0; j < pchi; ++j) {
			chi2d[i][j] = snorm(rng);
			h_chi[ndata * j + i] = chi2d[i][j];
		}
	}

    dvector d_chi = h_chi;
    Daug.SetChi(d_chi, false);

    // return chi-values as a std::vector of std::vectors
    vecvec vchi = Daug.GetChi();
    int nequal = 0;
    for (int i = 0; i < ndata; ++i) {
		for (int j = 0; j < pchi; ++j) {
			if (vchi[i][j] == chi2d[i][j]) {
				nequal++;
			}
		}
	}
    if (nequal == ndata * pchi) {
		npassed++;
	} else {
		std::cerr << "Test for Daug::GetChi failed: Values returned do not match the input values." << std::endl;
	}

    for (int i = 0; i < ndata; ++i) {
		delete [] chi2d[i];
	}
    delete chi2d;
    nperformed++;
}

// check that pointers to device-side LogDensityMeas and LogDensityPop are properly set by comparing output
// with that obtained from the host-side functions
void UnitTests::DaugLogDensPtr()
{

}

// check that DataAugmentation::Update always accepts when the proposed and current chi values are the same
void UnitTests::DaugAcceptSame()
{
	std::cout << "Testing DaugAcceptSame...." << std::endl;
	DataAugmentation Daug(meas, meas_unc, ndata, mfeat, pchi, nBlocks, nThreads);
	PopulationPar Theta(dim_theta, &Daug, nBlocks, nThreads);

	Daug.SetChi(d_true_chi);
	Theta.SetTheta(d_true_theta);

	// PopulationPar<Characteristic> Theta(dim_theta, &Daug, nBlocks, nThreads);
	//Theta.SetTheta(d_true_theta);
	//Daug.SetChi(d_true_chi);

	/*
	hvector h_chi = Daug.GetHostChi();
	std::cout << "Daug chi: ";
	for (int i = 0; i < h_chi.size(); ++i) {
		std::cout << h_chi[i] << " ";
	}
	std::cout << std::endl;

	std::cout << "true chi: ";
	for (int i = 0; i < d_true_chi.size(); ++i) {
		std::cout << d_true_chi[i] << " ";
	}
	std::cout << std::endl;

	dvector d_logdens_meas = Daug.GetDevLogDens();
	hvector h_logdens_meas = Daug.GetHostLogDens();
	dvector d_logdens_pop = Theta.GetDevLogDens();
	hvector h_logdens_pop= Theta.GetLogDens();

	std::cout << "logdens_meas: ";
	for (int i = 0; i < 10; ++i) {
		std::cout << d_logdens_meas[i] << " ";
	}
	std::cout << std::endl;

	std::cout << "logdens_pop: ";
	double logdens = 0.0;
	for (int i = 0; i < 10; ++i) {
		logdens += d_logdens_pop[i];
		std::cout << d_logdens_pop[i] << " ";
	}
	std::cout << std::endl;
	std::cout << "total logdens_pop: " << logdens << std::endl;
	double logdens_meas = 0.0;
	logdens_meas = thrust::reduce(d_logdens_meas.begin(), d_logdens_meas.end(), 0.0, thrust::plus<double>());
	double logdens_pop = 0.0;
	logdens_pop = thrust::reduce(d_logdens_pop.begin(), d_logdens_pop.begin(), 0.0, thrust::plus<double>());
	std::cout << "on device logdens_meas, logdens_pop: " << logdens_meas << ", " << logdens_pop << std::endl;

	// make sure posteriors saved in Daug and PopulationPar match those computed manually
	NormalVariate3d Chi(pchi, mfeat, dim_theta, 1);
	Chi.SetRNG(&rng);
	logdens_meas = 0.0;
	logdens_pop = 0.0;

	h_chi = Daug.GetDevChi();
	hvector h_theta = Theta.GetDevTheta();

	for (int i = 0; i < ndata; ++i) {
		double local_meas[10], local_chi[10], local_meas_unc[10];
		for (int j = 0; j < mfeat; ++j) {
			local_meas[j] = meas[i][j];
			local_meas_unc[j] = meas_unc[i][j];
		}
		for (int j = 0; j < pchi; ++j) {
			local_chi[j] = h_chi[j * ndata + i];
		}
		logdens_meas += Chi.LogDensityMeas(local_chi, local_meas, local_meas_unc);
		double* p_theta = thrust::raw_pointer_cast(&h_theta[0]);
		logdens_pop += Chi.LogDensityPop(local_chi, p_theta);
	}
	std::cout << "manual logdens_meas, logdens_pop: " << logdens_meas << ", " << logdens_pop << std::endl;
	*/

	// set the cholesky factors to zero so that NormalVariate.Propose() just returns the same chi value
	int dim_cholfact = pchi * pchi - ((pchi - 1) * pchi) / 2;
	hvector h_cholfact(ndata * dim_cholfact);
	thrust::fill(h_cholfact.begin(), h_cholfact.end(), 0.0);
	dvector d_cholfact = h_cholfact;
	Daug.SetCholFact(d_cholfact);

	Daug.Update();

	thrust::host_vector<int> h_naccept = Daug.GetNaccept();
	// make sure all of the proposals are accepted, since the proposed chi values are the same as the current ones
	int naccept = 0;
	for (int i = 0; i < h_naccept.size(); ++i) {
		naccept += h_naccept[i];
	}
	if (naccept == ndata) {
		npassed++;
	} else {
		std::cerr << "Test for Daug::Accept() failed: Did not accept all of the proposed characteristics "
				<< "when they are the same." << std::endl;
		std::cerr <<"naccept = " << naccept << std::endl;
	}
	nperformed++;

	// Now do the same thing, but for updating the population parameter values
	dim_cholfact = dim_theta * dim_theta - ((dim_theta - 1) * dim_theta) / 2;
	h_cholfact.resize(dim_cholfact);
	thrust::fill(h_cholfact.begin(), h_cholfact.end(), 0.0);
	d_cholfact = h_cholfact;

	int ntrials = 1000;
	for (int i = 0; i < ntrials; ++i) {
		Theta.SetCholFact(d_cholfact);
		Theta.Update();
	}
	naccept = Theta.GetNaccept();
	if (naccept == ntrials) {
		npassed++;
	} else {
		std::cerr << "Test for PopulationPar::Accept() failed: Did not accept all of the proposed population "
				<< "values when they are the same." << std::endl;
	}
	nperformed++;
}

// make sure that DataAugmentation::Update() accepts and saves Chi values when the posterior is much higher
void UnitTests::DaugAcceptBetter() {
	std::cout << "Testing DaugAcceptBetter...." << std::endl;
	DataAugmentation Daug(meas, meas_unc, ndata, mfeat, pchi, nBlocks, nThreads);
	PopulationPar Theta(dim_theta, &Daug, nBlocks, nThreads);

	Daug.SetChi(d_true_chi);
	Theta.SetTheta(d_true_theta);

	// artificially set the conditional log-posteriors to a really low value to make sure we accept the proposal
	hvector h_logdens_meas(ndata);
	thrust::fill(h_logdens_meas.begin(), h_logdens_meas.end(), -1e10);
	dvector d_logdens_meas = h_logdens_meas;
	Daug.SetLogDens(d_logdens_meas);

	Daug.Update();
	thrust::host_vector<int> h_naccept = Daug.GetNaccept();

	// make sure all of the proposals are accepted
	int naccept = 0;
	for (int i = 0; i < h_naccept.size(); ++i) {
		naccept += h_naccept[i];
	}
	if (naccept == ndata) {
		npassed++;
	} else {
		std::cerr << "Test for Daug::Accept() failed: Did not accept all of the proposed characteristics when "
				<< "the posterior is improved." << std::endl;
	}
	nperformed++;

	// make sure that the proposed values and new posteriors are saved
	hvector h_new_chi = Daug.GetDevChi();
	int ndiff_chi = 0;
	for (int i = 0; i < h_true_chi.size(); ++i) {
		if (h_new_chi[i] != h_true_chi[i]) {
			ndiff_chi++;
		}
	}
	if (ndiff_chi == h_true_chi.size()) {
		npassed++;
	} else {
		std::cerr << "Test for Daug::Accept() failed: Did not update the characteristics when the "
				<< "proposals are accepted." << std::endl;
	}
	nperformed++;
	hvector h_new_logdens = Daug.GetDevLogDens();
	int ndiff_logdens = 0;
	for (int i = 0; i < h_new_logdens.size(); ++i) {
		if (h_new_logdens[i] != h_logdens_meas[i]) {
			ndiff_logdens++;
		}
	}
	if (ndiff_logdens == h_logdens_meas.size()) {
		npassed++;
	} else {
		std::cerr << "Test for Daug::Accept() failed: Did not update the posteriors when the "
				<< "proposals are accepted." << std::endl;
	}
	nperformed++;

	/*
	 * Now do the same thing, but for the population parameters.
	 */

	hvector h_logdens_pop(ndata);
	thrust::fill(h_logdens_pop.begin(), h_logdens_pop.end(), -1e10);
	dvector d_logdens_pop = h_logdens_pop;
	Theta.SetLogDens(d_logdens_pop);

	Theta.Update();

	// make sure we accepted the proposal
	naccept = Theta.GetNaccept();
	if (naccept == 1) {
		npassed++;
	} else {
		std::cerr << "Test for PopulationPar::Accept() failed: Did not accept the proposed population "
				<< "values when the posterior is improved" << std::endl;
	}
	nperformed++;

	// make sure that the proposed values and new posteriors are saved
	hvector h_new_theta = Theta.GetHostTheta();
	int ndiff_theta = 0;
	for (int i = 0; i < h_new_theta.size(); ++i) {
		if (h_new_theta[i] != h_true_theta[i]) {
			ndiff_theta++;
		}
	}
	if (ndiff_theta == h_new_theta.size()) {
		// did we save the accepted theta?
		npassed++;
	} else {
		std::cerr << "Test for PopulationPar::Accept() failed: Did not update the population parameter value when the "
				<< "proposal is accepted." << std::endl;
	}
	nperformed++;
	h_new_logdens = Theta.GetLogDens();
	ndiff_logdens = 0;
	for (int i = 0; i < h_new_logdens.size(); ++i) {
		if (h_new_logdens[i] != h_logdens_meas[i]) {
			ndiff_logdens++;
		}
	}
	if (ndiff_logdens == h_logdens_meas.size()) {
		// did we update the posterior?
		npassed++;
	} else {
		std::cerr << "Test for PopulationPar::Accept() failed: Did not update the posteriors when the "
				<< "proposals are accepted." << std::endl;
	}
	nperformed++;
}

// test the Gibbs Sampler for a Normal-Normal model keeping the characteristics fixed
void UnitTests::FixedChar() {

	// create the parameter objects
	DataAugmentation Daug(meas, meas_unc, ndata, mfeat, pchi, nBlocks, nThreads);
	Daug.SetChi(d_true_chi);
	PopulationPar Theta(dim_theta, &Daug, nBlocks, nThreads);

	// setup the Gibbs sampler object
	int niter(100000), nburn(10000);
	GibbsSampler Sampler(Daug, Theta, niter, nburn);
	Sampler.FixChar(); // keep the characteristics fixed

	// run the MCMC sampler
	Sampler.Run();

	// check the acceptance rate
	double target_rate = 0.4;
	double naccept = Theta.GetNaccept();
	double arate = naccept / double(niter + nburn);
	double frac_diff = abs(arate - target_rate) / target_rate;
	// make sure acceptance rate is within 5% of the target rate
	if (frac_diff < 0.05) {
		npassed++;
	} else {
		std::cerr << "Test for GibbsSampler with fixed Characteristics failed: Acceptance rate "
				<< "is not within 5% of the target rate." << std::endl;
		std::cout << arate << ", " << target_rate << std::endl;
	}
	nperformed++;

	// grab the MCMC samples of the population parameter
	vecvec tsamples = Sampler.GetPopSamples();

	// get the estimated posterior mean of normal mean parameter
	double theta_mean[3] = {0.0, 0.0, 0.0};
	for (int i = 0; i < tsamples.size(); ++i) {
		theta_mean[0] += tsamples[i][0];
		theta_mean[1] += tsamples[i][1];
		theta_mean[2] += tsamples[i][2];
	}
	for (int j = 0; j < dim_theta; ++j) {
		theta_mean[j] /= tsamples.size();
	}
	// get true value of posterior mean
	double theta_mean_true[3] = {0.0, 0.0, 0.0};
	for (int i = 0; i < ndata; ++i) {
		theta_mean_true[0] += h_true_chi[i];
		theta_mean_true[1] += h_true_chi[ndata + i];
		theta_mean_true[2] += h_true_chi[2 * ndata + i];
	}
	for (int j = 0; j < dim_theta; ++j) {
		theta_mean_true[j] /= ndata;
	}

	// make sure estimated value and true value are within 2% of eachother
	frac_diff = 0.0;
	for (int j = 0; j < dim_theta; ++j) {
		frac_diff += abs(theta_mean[j] - theta_mean_true[j]) / abs(theta_mean_true[j]);
	}
	frac_diff /= dim_theta;
	if (frac_diff < 0.02) {
		npassed++;
	} else {
		std::cerr << "Test for GibbsSampler with fixed Characteristics failed: Average fractional difference "
				<< "between estimated posterior mean and true value is greater than 2%." << std::endl;
	}
	nperformed++;

	// get the estimated posterior covariance of the normal mean parameter
	double mean_covar[3][3];
	double mean_covar_true[3][3];
	for (int j = 0; j < dim_theta; ++j) {
		for (int k = 0; k < dim_theta; ++k) {
			// initialize to zero
			mean_covar[j][k] = 0.0;
			// calculate true value of posterior covariance of normal mean parameter
			mean_covar_true[j][k] = covar[j][k] / ndata;
		}
	}
	for (int i = 0; i < tsamples.size(); ++i) {
		for (int j = 0; j < dim_theta; ++j) {
			for (int k = 0; k < dim_theta; ++k) {
				mean_covar[j][k] += (tsamples[i][j] - theta_mean[j]) * (tsamples[i][k] - theta_mean[k]);
			}
		}
	}
	for (int j = 0; j < dim_theta; ++j) {
		for (int k = 0; k < dim_theta; ++k) {
			mean_covar[j][k] /= tsamples.size();
		}
	}

	// make sure estimated value and true value are within 2% of eachother
	frac_diff = 0.0;
	for (int j = 0; j < dim_theta; ++j) {
		for (int k = 0; k < dim_theta; ++k) {
			frac_diff += abs(mean_covar[j][k] - mean_covar_true[j][k]) / abs(mean_covar[j][k]);
		}
	}
	frac_diff /= (dim_theta * dim_theta) ;
	if (frac_diff < 0.02) {
		npassed++;
	} else {
		std::cerr << "Test for GibbsSampler with fixed Characteristics failed: Average fractional difference "
				<< "between estimated posterior covariance in mean parameter and the true value is greater than 2%." << std::endl;
	}
	nperformed++;
}

// test the Gibbs Sampler for a Normal-Normal model keeping the population parameter fixed
void UnitTests::FixedPopPar() {

}

// test the Gibbs Sampler for a Normal-Normal model
void UnitTests::NormNorm() {

}

// print out summary of test results
void UnitTests::Finish() {
	std::cout << npassed << " tests passed out of " << nperformed << " tests performed." << std::endl;
	npassed = 0;
	nperformed = 0;
}
