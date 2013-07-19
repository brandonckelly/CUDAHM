/*
 * UnitTests.cpp
 *
 *  Created on: Jul 17, 2013
 *      Author: brandonkelly
 */

// standard includes
#include <iostream>

// local includes
#include "UnitTests.cuh"
#include "data_augmentation.cuh"

// Global random number generator and distributions for generating random numbers on the host. The random number generator used
// is the Mersenne Twister mt19937 from the BOOST library. These are instantiated in data_augmentation.cu.
extern boost::random::mt19937 rng;
extern boost::random::normal_distribution<> snorm; // Standard normal distribution
extern boost::random::uniform_real_distribution<> uniform; // Uniform distribution from 0.0 to 1.0

// calculate transpose(x) * covar_inv * x for an nx-element vector x and an (nx,nx)-element matrix covar_inv
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
	std::cout << "Testing rank-1 Cholesky Update..." << std::endl;
	double covar[3][3] =
	{
			{5.29, 0.3105, -15.41},
			{0.3105, 0.2025, 3.2562},
			{-15.41, 3.2562, 179.56}
	};
	double cholfact0[6] = {2.3, 0.135, 0.42927264, -6.7, 9.6924416, 6.38173768};
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

	if ((max_frac_diff_down < epsilon) && (max_frac_diff_up < epsilon)) {
		// cholesky factors agree, so test passed
		npassed++;
	} else {
		// test failed
		std::cerr << "Rank-1 Cholesky update test failed." << std::endl;
	}

	nperformed++;
}

// check that Chi::Propose follows a multivariate normal distribution
void UnitTests::ChiPropose() {
	std::cout << "Testing Chi::Propose..." << std::endl;
	double covar[3][3] =
	{
			{5.29, 0.3105, -15.41},
			{0.3105, 0.2025, 3.2562},
			{-15.41, 3.2562, 179.56}
	};
	double covar_inv0[3][3] =
	{
			{0.64880351, -2.66823952, 0.10406763},
			{-2.66823952, 17.94430089, -0.55439855},
			{0.10406763, -0.55439855, 0.02455399}
	};
	double** covar_inv;
	covar_inv = new double* [3];
	for (int i = 0; i < 3; ++i) {
		covar_inv[i] = new double [3];
	}
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			covar_inv[i][j] = covar_inv0[i][j];
		}
	}

	double cholfact[6] = {2.3, 0.135, 0.42927264, -6.7, 9.6924416, 6.38173768};

	int p = 3, ntrials = 100000;
	int m = 1, dt = 1, current_iter = 1;

	Characteristic Chi(p, m, dt, current_iter);
	Chi.SetRNG(&rng);

	double snorm_deviate[p];
	double scaled_proposal[p];
	double proposed_chi[p];
	double chi[3] = {1.2, 0.4, -0.7};
	double chisqr[ntrials];

	for (int i = 0; i < ntrials; ++i) {
		// get the ntrials proposals
		Chi.Propose(chi, cholfact, proposed_chi, snorm_deviate, scaled_proposal);
		for (int j = 0; j < p; ++j) {
			proposed_chi[j] -= chi[j]; // center the proposals
		}
		chisqr[i] = mahalanobis_distance(covar_inv, proposed_chi, p);
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
		delete [] covar_inv[i];
	}
	delete covar_inv;
}

// check that Chi::Accept always accepts when the proposal and the current values are the same
void UnitTests::ChiAcceptSame() {
	double chi[3] = {1.2, 0.4, -0.7};
	int p = 3, ntrials = 100000;
	int m = 1, dt = 1, current_iter = 1;

	Characteristic Chi(p, m, dt, current_iter);
	Chi.SetRNG(&rng);

	bool accept;
	int naccept = 0;
	double logdens = -1.32456;
	double ratio = 0.0;
	for (int i = 0; i < ntrials; ++i) {
		accept = Chi.AcceptProp(logdens, logdens, 0.0, 0.0, ratio);
		if (abs(ratio - 1.0) < epsilon) {
			naccept++;
		}
	}

	if (naccept == ntrials) {
		npassed++;
	} else {
		std::cerr << "Test for Chi::Accept failed: Failes to always accept when the log-posteriors are the same." << std::endl;
	}
	nperformed++;
}

// test Chi::Adapt acceptance rate and covariance by running a simple MCMC sampler
void UnitTests::ChiAdapt() {

}

// check that PopulationPar::Propose follow a multivariate normal distribution
void UnitTests::ThetaPropose() {

}

// check that PopulationPar::Accept always accepts when the logdensities are the same
void UnitTests::ThetaAcceptSame() {

}

// make sure we accept and save a population parameter value with a much higher posterior
void UnitTests::ThetaAcceptBetter() {

}

// test PopulationPar::Adapt acceptance rate and covariance by running a simple MCMC sampler
void UnitTests::ThetaAdapt() {

}

// check that constructor for population parameter correctly set the pointer data member of DataAugmentation
void UnitTests::DaugPopPtr() {

}

// test DataAugmentation::GetChi
void UnitTests::DaugGetChi() {

}

// check that DataAugmentation::Update always accepts when the proposed and current chi values are the same
void UnitTests::DaugAcceptSame() {

}

// make sure that DataAugmentation::Update() accepts and saves Chi values when the posterior is much higher
void UnitTests::DaugAcceptBetter() {

}

// print out summary of test results
void UnitTests::Finish() {
	std::cout << npassed << " tests passed out of " << nperformed << " tests performed." << std::endl;
	npassed = 0;
	nperformed = 0;
}
