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

}

// check that Chi::Accept always accepts when the proposal and the current values are the same
void UnitTests::ChiAcceptSame() {

}

// make sure we accept and save a Chi value with a much higher posterior
void UnitTests::ChiAcceptBetter() {

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
