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
#include "UnitTests.cuh"
#include "data_augmentation.cuh"

// Global random number generator and distributions for generating random numbers on the host. The random number generator used
// is the Mersenne Twister mt19937 from the BOOST library. These are instantiated in data_augmentation.cu.
extern boost::random::mt19937 rng;
extern boost::random::normal_distribution<> snorm; // Standard normal distribution
extern boost::random::uniform_real_distribution<> uniform; // Uniform distribution from 0.0 to 1.0

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

// characteristic class for a simple p-dimensional gaussian distribution with known mean and inverse-covariance matrix
class SimpleNormalVariate : public Characteristic {
public:
	__device__ __host__
	SimpleNormalVariate(int p, int m, int dimt, int iter) : Characteristic(p, m, dimt, iter) {}

	// compute the conditional log-posterior density of the measurements given the characteristic
	__device__ __host__ double LogDensityMeas(double* chi, double* meas, double* meas_unc)
	{
		double** covar_inv;
		covar_inv = new double* [pchi];
		for (int i = 0; i < pchi; ++i) {
			covar_inv[i] = new double [pchi];
		}
		int index = 0;
		for (int i = 0; i < pchi; ++i) {
			for (int j = 0; j < pchi; ++j) {
				// input meas_unc contains the values of the inverse-covariance matrix
				covar_inv[i][j] = meas_unc[index];
				index++;
			}
		}
		double* chi_cent = new double [pchi];
		for (int i = 0; i < pchi; ++i) {
			// meas is the p-dimensional mean vector
			chi_cent[i] = chi[i] - meas[i];
		}
		double logdens = -0.5 * mahalanobis_distance(covar_inv, chi_cent, pchi);

		// delete memory
		for (int i = 0; i < pchi; ++i) {
			delete [] covar_inv[i];
		}
		delete covar_inv;
		delete chi_cent;

		return logdens;
	}

	// compute the conditional log-posterior dentity of the characteristic given the population parameter
	__device__ __host__ double LogDensityPop(double* chi, double* theta)
	{
		return 0.0;
	}

	__device__ __host__ void SetCurrentIter(int iter) {
		current_iter = iter;
	}

private:
};

// characteristic class for a 3-dimensional Normal-Normal Model
class NormalVariate3d : public Characteristic {
public:
	__device__ __host__
	NormalVariate3d(int p, int m, int dimt, int iter) : Characteristic(3, 3, 3, iter) {}

	// invert a 3-dimensional covariance matrix
	__device__ __host__ void InvertCovar(double* covar, double* covar_inv)
	{
		double a, b, c, d, e, f, g, h, k;
		a = covar[0];
		b = covar[1];
		c = covar[2];
		d = covar[3];
		e = covar[4];
		f = covar[5];
		g = covar[6];
		h = covar[7];
		k = covar[8];

		double determ_inv = 0.0;
		determ_inv = 1.0 / (a * (e * k - f * h) - b * (k * d - f * g) + c * (d * h - e * g));

		covar_inv[0] = determ_inv * (e * k - f * h);
		covar_inv[1] = -determ_inv * (b * k - c * h);
		covar_inv[2] = determ_inv * (b * f - c * e);
		covar_inv[3] = -determ_inv * (d * k - f * g);
		covar_inv[4] = determ_inv * (a * k - c * g);
		covar_inv[5] = -determ_inv * (a * f - c * d);
		covar_inv[6] = determ_inv * (d * h - e * g);
		covar_inv[7] = -determ_inv * (a * h - b * g);
		covar_inv[8] = determ_inv * (a * e - b * d);
	}

	// calculate transpose(x) * covar_inv * x
	__device__ __host__ double ChiSqr(double* x, double* covar_inv, int nx)
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
	__device__ __host__ double LogDensityMeas(double* chi, double* meas, double* meas_unc)
	{
		double chi_cent[3];
		for (int i = 0; i < 3; ++i) {
			// meas is the p-dimensional mean vector
			chi_cent[i] = chi[i] - meas[i];
		}

		double meas_unc_inv[9];
		InvertCovar(meas_unc, meas_unc_inv);

		double logdens = -0.5 * ChiSqr(chi_cent, meas_unc_inv, 3);
		return logdens;
	}

	// compute the conditional log-posterior density of the characteristic given the population mean
	__device__ __host__ double LogDensityPop(double* chi, double* theta)
	{
		// known inverse covariance matrix at the characteristic population level
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

private:
};


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
		std::cerr << "Test for Chi::Accept failed: Failed to always accept when the log-posteriors are the same." << std::endl;
	}
	nperformed++;
}

// test Chi::Adapt acceptance rate and covariance by running a simple MCMC sampler for a 3-dimensional normal distribution
void UnitTests::ChiAdapt() {
	double covar[3][3] =
	{
			{5.29, 0.3105, -15.41},
			{0.3105, 0.2025, 3.2562},
			{-15.41, 3.2562, 179.56}
	};
	// store in a 1-d array since that is what Chi::Propose expects for the measurement uncertainties
	double covar_inv[9] =
	{
			0.64880351, -2.66823952, 0.10406763,
			-2.66823952, 17.94430089, -0.55439855,
			0.10406763, -0.55439855, 0.02455399
	};
	double meas[3] = {1.2, 0.4, -0.7};

	int p = 3, niter = 200000;
	int m = 3, dt = 2, current_iter = 1;
	double target_rate = 0.4; // MCMC sampler target acceptance rate

	SimpleNormalVariate Chi(p, m, dt, current_iter);
	Chi.SetRNG(&rng);

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
	double theta[2] = {0.0, 0.0};
	double snorm_deviate[p], scaled_proposal[p], proposed_chi[p];
	double logdens_meas, logdens_pop;
	logdens_meas = Chi.LogDensityMeas(chi, meas, covar_inv);
	logdens_pop = Chi.LogDensityPop(chi, theta);
	int naccept = 0, start_counting = 10000;

//	std::ofstream output("/users/brandonkelly/chi.dat");
//	output << chi[0] << " " << chi[1] << " " << chi[2] << " " << logdens_meas << std::endl;

	for (int i = 0; i < niter; ++i) {
		// propose a new value of chi
		Chi.Propose(chi, cholfact, proposed_chi, snorm_deviate, scaled_proposal);

		// get value of log-posterior for proposed chi value
		double logdens_pop_prop, logdens_meas_prop;
		logdens_meas_prop = Chi.LogDensityMeas(proposed_chi, meas, covar_inv);
		logdens_pop_prop = Chi.LogDensityPop(proposed_chi, theta);
		double logpost_prop = logdens_meas_prop + logdens_pop_prop;

		// accept the proposed value of the characteristic?
		double logpost_current = logdens_meas + logdens_pop;
		double metro_ratio = 0.0;
		bool accept = Chi.AcceptProp(logpost_prop, logpost_current, 0.0, 0.0, metro_ratio);

		// adapt the covariance matrix of the characteristic proposal distribution
		Chi.AdaptProp(cholfact, snorm_deviate, scaled_proposal, metro_ratio);
		int dim_cholfact = p * p - ((p - 1) * p) / 2;

		if (accept) {
			// accepted this proposal, so save new value of chi and log-densities
			for (int j=0; j<p; j++) {
				chi[j] = proposed_chi[j];
			}
			logdens_meas = logdens_meas_prop;
			logdens_pop = logdens_pop_prop;
			if (current_iter >= start_counting) {
				// don't start counting # of accepted proposals until we've down start_counting iterations
				naccept++;
			}
		}
		current_iter++;
		Chi.SetCurrentIter(current_iter);
//		output << chi[0] << " " << chi[1] << " " << chi[2] << " " << logdens_meas << std::endl;
	}
//	output.close();
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

	hvector cholfact(6);
	cholfact[0] = 2.3;
	cholfact[1] = 0.135;
	cholfact[2] = 0.42927264;
	cholfact[3] = -6.7;
	cholfact[4] = 9.6924416;
	cholfact[5] = 6.38173768;

	int ntrials = 100000;
	int current_iter = 1;
    PopulationPar<Characteristic> Theta(dim_theta, nBlocks, nThreads);

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
		chisqr[i] = mahalanobis_distance(covar_inv, p_theta, dim_theta);
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
		delete [] covar_inv[i];
	}
	delete covar_inv;
}

// check that PopulationPar::Accept always accepts when the logdensities are the same
void UnitTests::ThetaAcceptSame() {
	int ntrials = 100000;
    PopulationPar<Characteristic> Theta(dim_theta, nBlocks, nThreads);

	bool accept;
	int naccept = 0;
	double logdens = -1.32456;
	double ratio = 0.0;
	for (int i = 0; i < ntrials; ++i) {
		accept = Theta.AcceptProp(logdens, logdens, ratio);
		if (abs(ratio - 1.0) < epsilon) {
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
	double covar[3][3] =
	{
			{5.29, 0.3105, -15.41},
			{0.3105, 0.2025, 3.2562},
			{-15.41, 3.2562, 179.56}
	};
	// store in a 1-d array since that is what Chi::Propose expects for the measurement uncertainties
	double covar_inv[9] =
	{
			0.64880351, -2.66823952, 0.10406763,
			-2.66823952, 17.94430089, -0.55439855,
			0.10406763, -0.55439855, 0.02455399
	};
	double mu[3] = {1.2, 0.4, -0.7};

	SimpleNormalVariate NormDist(pchi, mfeat, dim_theta, 1);
	NormDist.SetRNG(&rng);
	PopulationPar<SimpleNormalVariate> Theta(dim_theta, nBlocks, nThreads);

	hvector h_theta = Theta.GetHostTheta();
	dvector d_theta = h_theta;
	double* p_theta = thrust::raw_pointer_cast(&h_theta[0]);

	// run the MCMC sampler
	double logdens_current = NormDist.LogDensityMeas(p_theta, mu, covar_inv);
	int naccept = 0, start_counting = 10000;
	int niter = 200000, current_iter = 1;
	double target_rate = 0.4; // MCMC sampler target acceptance rate

	//std::ofstream output("/users/brandonkelly/theta.dat");
	//output << h_theta[0] << " " << h_theta[1] << " " << h_theta[2] << " " << logdens_current << std::endl;

	for (int i = 0; i < niter; ++i) {
		// propose a new value of theta
		hvector theta_prop(dim_theta);
		theta_prop = Theta.Propose();
		double* p_theta_prop = thrust::raw_pointer_cast(&theta_prop[0]);
		// get value of log-posterior for proposed theta value
		double logdens_prop;
		logdens_prop = NormDist.LogDensityMeas(p_theta_prop, mu, covar_inv);

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
		//output << h_theta[0] << " " << h_theta[1] << " " << h_theta[2] << " " << logdens_current << std::endl;
	}
	//output.close();
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

// check that constructor for population parameter correctly set the pointer data member of DataAugmentation
void UnitTests::DaugPopPtr() {
	DataAugmentation<Characteristic> Daug(meas, meas_unc, ndata, mfeat, pchi, nBlocks, nThreads);
	PopulationPar<Characteristic> Theta(dim_theta, &Daug, nBlocks, nThreads);

	PopulationPar<Characteristic>* p_Theta = Daug.GetPopulationPtr();

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

	DataAugmentation<Characteristic> Daug(meas, meas_unc, ndata, mfeat, pchi, nBlocks, nThreads);
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

// check that DataAugmentation::Update always accepts when the proposed and current chi values are the same
void UnitTests::DaugAcceptSame()
{
	DataAugmentation<NormalVariate3d> Daug(meas, meas_unc, ndata, mfeat, pchi, nBlocks, nThreads);
	PopulationPar<NormalVariate3d> Theta(dim_theta, &Daug, nBlocks, nThreads);

	// generate some chi values from a standard normal
	hvector h_chi(ndata * pchi);
	for (int i = 0; i < h_chi.size(); ++i) {
		h_chi[i] = snorm(rng);
	}
	dvector d_chi = h_chi;
	Daug.SetChi(d_chi);

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
	}
	nperformed++;

	// Now do the same thing, but for updating the population parameter values
	dim_cholfact = dim_theta * dim_theta - ((dim_theta - 1) * dim_theta) / 2;
	h_cholfact.resize(dim_cholfact);
	thrust::fill(h_cholfact.begin(), h_cholfact.end(), 0.0);
	d_cholfact = h_cholfact;

	hvector h_theta(dim_theta);
	for (int i = 0; i < h_theta.size(); ++i) {
		h_theta[i] = snorm(rng);
	}
	dvector d_theta = h_theta;
	Theta.SetTheta(d_theta);

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
	DataAugmentation<NormalVariate3d> Daug(meas, meas_unc, ndata, mfeat, pchi, nBlocks, nThreads);
	PopulationPar<NormalVariate3d> Theta(dim_theta, &Daug, nBlocks, nThreads);

	// generate some chi values from a standard normal
	hvector h_chi(ndata * pchi);
	for (int i = 0; i < h_chi.size(); ++i) {
		h_chi[i] = snorm(rng);
	}
	dvector d_chi = h_chi;
	Daug.SetChi(d_chi);

	// artificially set the conditional log-posteriors to a really low value to make sure we accept the proposal
	hvector h_logdens_meas(ndata);
	thrust::fill(h_logdens_meas.begin(), h_logdens_meas.end(), -1e10);
	dvector d_logdens_meas = h_logdens_meas;
	Daug.SetLogDens(d_logdens_meas);

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
		std::cerr << "Test for Daug::Accept() failed: Did not accept all of the proposed characteristics when "
				<< "the posterior is improved." << std::endl;
	}
	nperformed++;

	// make sure that the proposed values and new posteriors are saved
	hvector h_new_chi = Daug.GetDevChi();
	int ndiff_chi = 0;
	for (int i = 0; i < h_chi.size(); ++i) {
		if (h_new_chi[i] != h_chi[i]) {
			ndiff_chi++;
		}
	}
	if (ndiff_chi == h_chi.size()) {
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

	hvector h_theta(dim_theta);
	for (int i = 0; i < h_theta.size(); ++i) {
		h_theta[i] = snorm(rng);
	}
	dvector d_theta = h_theta;
	Theta.SetTheta(d_theta);

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
		if (h_new_theta[i] != h_theta[i]) {
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
	h_new_logdens = Theta.GetHostLogDens();
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

// print out summary of test results
void UnitTests::Finish() {
	std::cout << npassed << " tests passed out of " << nperformed << " tests performed." << std::endl;
	npassed = 0;
	nperformed = 0;
}
