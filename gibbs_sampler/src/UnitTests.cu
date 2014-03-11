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
	for (int i = 0; i < pchi; ++i) {
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

/*
 * kernels that will be used to test each component of update_characteristic
 */

// test the function that proposes a new characteristic on the device
__global__
void test_propose(double* chi, double* cholfact, curandState* devStates, int ndata, int pchi)
{
	int idata = blockDim.x * blockIdx.x + threadIdx.x;
	if (idata < ndata)
	{
		curandState localState = devStates[idata]; // grab state of this random number generator

		// copy values for this data point to registers for speed
		double snorm_deviate[3], scaled_proposal[3], proposed_chi[3], local_chi[3], local_cholfact[6];
		int cholfact_index = 0;
		for (int j = 0; j < pchi; ++j) {
			local_chi[j] = chi[j * ndata + idata];
			for (int k = 0; k < (j+1); ++k) {
				local_cholfact[cholfact_index] = cholfact[cholfact_index * ndata + idata];
				cholfact_index++;
			}
		}
		// propose a new value of chi
		Propose(local_chi, local_cholfact, proposed_chi, snorm_deviate, scaled_proposal, pchi, &localState);

		// copy local RNG state back to global memory
		devStates[idata] = localState;

		// copy proposed value of chi back to global memory
		for (int j=0; j<pchi; j++) {
			chi[ndata * j + idata] = proposed_chi[j];
		}
	}
}

// test the function perform the Metropolis-Hasting acceptance step
__global__
void test_accept(double* chi, double* cholfact, double* logdens_meas, double* logdens_pop, curandState* devStates,
		int* naccept, int ndata, int pchi)
{
	int idata = blockDim.x * blockIdx.x + threadIdx.x;
	if (idata < ndata)
	{
		curandState localState = devStates[idata]; // grab state of this random number generator

		// copy values for this data point to registers for speed
		double snorm_deviate[3] = {1.2, -0.6, 0.8};
		double scaled_proposal[3], local_cholfact[6], proposed_chi[3];
		int cholfact_index = 0;
		for (int j = 0; j < pchi; ++j) {
			double scaled_proposal_j = 0.0;
			for (int k = 0; k < (j+1); ++k) {
				local_cholfact[cholfact_index] = cholfact[cholfact_index * ndata + idata];
				scaled_proposal_j += local_cholfact[cholfact_index] * snorm_deviate[k];
				cholfact_index++;
			}
			scaled_proposal[j] = scaled_proposal_j;
			proposed_chi[j] = chi[ndata * j + idata] + scaled_proposal[j];
		}

		// should always accept when proposed posterior is higher
		double logpost_current = 0.0;
		double logpost_prop = 1.0;
		double metro_ratio = 0.0;
		bool accept = AcceptProp(logpost_prop, logpost_current, 0.0, 0.0, metro_ratio, &localState);

		// copy local RNG state back to global memory
		devStates[idata] = localState;

		if (accept) {
			// accepted this proposal, so save new value of chi and log-densities
			for (int j=0; j<pchi; j++) {
				chi[ndata * j + idata] = proposed_chi[j];
			}
			logdens_meas[idata] = 0.5;
			logdens_pop[idata] = 0.5;
			naccept[idata] += 1;
		}
	}
}

// test that we adapt the proposal covariance matrix
__global__
void test_adapt(double* cholfact, int ndata, int pchi, int current_iter)
{
	int idata = blockDim.x * blockIdx.x + threadIdx.x;
	if (idata < ndata)
	{
		// copy values for this data point to registers for speed
		double snorm_deviate[3] = {1.2, -0.6, 0.8};
		double scaled_proposal[3], local_cholfact[6];
		int cholfact_index = 0;
		for (int j = 0; j < pchi; ++j) {
			double scaled_proposal_j = 0.0;
			for (int k = 0; k < (j+1); ++k) {
				local_cholfact[cholfact_index] = cholfact[cholfact_index * ndata + idata];
				scaled_proposal_j += local_cholfact[cholfact_index] * snorm_deviate[k];
				cholfact_index++;
			}
			scaled_proposal[j] = scaled_proposal_j;
		}

		// adapt the covariance matrix of the characteristic proposal distribution using some predetermined values
		double metro_ratio = 0.34;
		AdaptProp(local_cholfact, snorm_deviate, scaled_proposal, metro_ratio, pchi, current_iter);

		int dim_cholfact = pchi * pchi - ((pchi - 1) * pchi) / 2;
		for (int j = 0; j < dim_cholfact; ++j) {
			// copy value of this adapted cholesky factor back to global memory
			cholfact[j * ndata + idata] = local_cholfact[j];
		}
	}
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

std::vector<double> covariance(vecvec X) {
	std::vector<double> xcovar(X[0].size() * X[0].size(), 0.0);
	std::vector<double> xmean(X[0].size(), 0.0);
	for (int j = 0; j < X[0].size(); ++j) {
		for (int i = 0; i < X.size(); ++i) {
			xmean[j] += X[i][j];
		}
		xmean[j] /= X.size();
	}
	for (int i = 0; i < X.size(); ++i) {
		for (int j = 0; j < X[0].size(); ++j) {
			for (int k = 0; k < X[0].size(); ++k) {
				xcovar[j * X[0].size() + k] += (X[i][j] - xmean[j]) * (X[i][k] - xmean[k]) / X.size();
			}
		}
	}
	return xcovar;
}

void matrix_invert3d(double* A, double* A_inv) {
	double a, b, c, d, e, f, g, h, k;
	a = A[0];
	b = A[1];
	c = A[2];
	d = A[3];
	e = A[4];
	f = A[5];
	g = A[6];
	h = A[7];
	k = A[8];

	double determ_inv = 0.0;
	determ_inv = 1.0 / (a * (e * k - f * h) - b * (k * d - f * g) + c * (d * h - e * g));

	A_inv[0] = determ_inv * (e * k - f * h);
	A_inv[1] = -determ_inv * (b * k - c * h);
	A_inv[2] = determ_inv * (b * f - c * e);
	A_inv[3] = -determ_inv * (d * k - f * g);
	A_inv[4] = determ_inv * (a * k - c * g);
	A_inv[5] = -determ_inv * (a * f - c * d);
	A_inv[6] = determ_inv * (d * h - e * g);
	A_inv[7] = -determ_inv * (a * h - b * g);
	A_inv[8] = determ_inv * (a * e - b * d);
}

bool approx_equal(double a, double b, double eps) {
	if (abs(a - b) < eps) {
		return true;
	} else {
		return false;
	}
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
    h_meas.resize(ndata * mfeat);
    h_meas_unc.resize(ndata * mfeat);
    double meas_err[3] = {1.2, 0.4, 0.24};
    for (int i = 0; i < ndata; ++i) {
		meas[i] = new double [mfeat];
		meas_unc[i] = new double [mfeat];
		for (int j = 0; j < mfeat; ++j) {
			// y_ij|chi_ij ~ N(chi_ij, meas_err_j^2)
			meas[i][j] = h_true_chi[j * ndata + i] + meas_err[j] * snorm(rng);
			h_meas[ndata * j + i] = meas[i][j];
			meas_unc[i][j] = meas_err[j];
			h_meas_unc[ndata * j + i] = meas_err[j];
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

// save measurements to "measurement.txt"
void UnitTests::SaveMeasurements()
{
	std::ofstream meas_file;
	meas_file.open("measurements.txt");
	meas_file << "# (measurement, sigma), ndata=" << ndata << ", mfeat=" << mfeat << std::endl;
	for (int i = 0; i < ndata; ++i) {
		for (int j = 0; j < mfeat; ++j) {
			meas_file << h_meas[ndata * j + i] << " " << h_meas_unc[ndata * j + i] << " ";
		}
		meas_file << std::endl;
	}
}

// test rank-1 cholesky update
void UnitTests::R1CholUpdate() {

	std::cout << "Testing rank-1 Cholesky update..." << std::endl;

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
		std::cout << "... passed." << std::endl;
	} else {
		// test failed
		std::cerr << "Rank-1 Cholesky update test failed." << std::endl;
	}

	nperformed++;
}

// check that Propose follows a multivariate normal distribution
void UnitTests::ChiPropose() {

	std::cout << "Testing chi proposal generation..." << std::endl;

	int local_passed = 0;

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
		local_passed++;
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
		local_passed++;
	} else {
		std::cerr << "Test for Chi::Propose failed: empirical 4.0 percentile inconsistent with true value" << std::endl;
	}
	nperformed++;
	if ((count_high > nhigh_low) && (count_high < nhigh_high)) {
		npassed++;
		local_passed++;
	} else {
		std::cerr << "Test for Chi::Propose failed: empirical 95.4 percentile inconsistent with true value" << std::endl;
	}
	nperformed++;

	// free memory
	for (int i = 0; i < 3; ++i) {
		delete [] covar_inv_local[i];
	}
	delete covar_inv_local;
	if (local_passed == 3) {
		std::cout << "... passed." << std::endl;
	}
}

// check that Chi::Accept always accepts when the proposal and the current values are the same
void UnitTests::ChiAcceptSame() {

	std::cout << "Testing chi acceptance step for same posteriors..." << std::endl;

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
		std::cout << "... passed." << std::endl;
	} else {
		std::cerr << "Test for Chi::Accept failed: Failed to always accept when the log-posteriors are the same." << std::endl;
	}
	nperformed++;
}

// test Chi::Adapt acceptance rate and covariance by running a simple MCMC sampler for a 3-dimensional normal distribution
void UnitTests::ChiAdapt() {

	std::cout << "Testing that RAM MCMC sampler achieves desired acceptance rate for a single chi using host functions."
			<< std::endl;

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
		std::cout << "... passed." << std::endl;
	} else {
		std::cerr << "Test for Chi::Adapt failed: Acceptance rate is not within 5% of the target rate." << std::endl;
		std::cout << accept_rate << ", " << target_rate << std::endl;
	}
	nperformed++;
}

// check that PopulationPar::Propose follow a multivariate normal distribution
void UnitTests::ThetaPropose() {

	std::cout << "Testing that population parameter proposal follows correct distribution..." << std::endl;
	int local_passed = 0;

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
    PopulationPar<3,3,3> Theta(nBlocks, nThreads);

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
		local_passed++;
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
		local_passed++;
	} else {
		std::cerr << "Test for Theta::Propose failed: empirical 4.0 percentile inconsistent with true value" << std::endl;
	}
	nperformed++;
	if ((count_high > nhigh_low) && (count_high < nhigh_high)) {
		npassed++;
		local_passed++;
	} else {
		std::cerr << "Test for Theta::Propose failed: empirical 95.4 percentile inconsistent with true value" << std::endl;
	}
	nperformed++;

	// free memory
	for (int i = 0; i < 3; ++i) {
		delete [] covar_inv_local[i];
	}
	delete covar_inv_local;

	if (local_passed == 3) {
		std::cout << "... passed." << std::endl;
	}
}

// check that PopulationPar::Accept always accepts when the logdensities are the same
void UnitTests::ThetaAcceptSame() {
	std::cout << "Testing that population parameter updates always accept when posterior are the same." << std::endl;

	int ntrials = 100000;
    PopulationPar<3,3,3> Theta(nBlocks, nThreads);

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
		std::cout << "... passed." << std::endl;
	} else {
		std::cerr << "Test for Theta::Accept failed: Failed to always accept when the log-posteriors are the same." << std::endl;
	}
	nperformed++;
}

// test PopulationPar::Adapt acceptance rate and covariance by running a simple MCMC sampler
void UnitTests::ThetaAdapt() {

	std::cout << "Testing that population parameter RAM sampler achieves desired acceptance rate." << std::endl;

	double mu[3] = {1.2, 0.4, -0.7};

	PopulationPar<3,3,3> Theta(nBlocks, nThreads);
	boost::shared_ptr<DataAugmentation<3,3,3> > DaugPtr(new DataAugmentation<3,3,3>(meas, meas_unc, ndata, nBlocks, nThreads));
	Theta.SetDataAugPtr(DaugPtr);

	Theta.Initialize();

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
		std::cout << "... passed." << std::endl;
	} else {
		std::cerr << "Test for Theta::Adapt failed: Acceptance rate is not within 5% of the target rate." << std::endl;
		std::cout << accept_rate << ", " << target_rate << std::endl;
	}
	nperformed++;
}

// test DataAugmentation::GetChi
void UnitTests::DaugGetChi() {

	std::cout << "Testing DataAugmentation::GetChi..." << std::endl;

	DataAugmentation<3,3,3> Daug(meas, meas_unc, ndata, nBlocks, nThreads);
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
		std::cout << "... passed." << std::endl;
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
	std::cout << "Testing that pointers to log density functions are properly set..." << std::endl;
	int local_passed = 0;

	boost::shared_ptr<DataAugmentation<3,3,3> > Daug(new DataAugmentation<3,3,3> (meas, meas_unc, ndata, nBlocks, nThreads));
	boost::shared_ptr<PopulationPar<3,3,3> > Theta(new PopulationPar<3,3,3> (nBlocks, nThreads));

	Daug->SetPopulationPtr(Theta);
	Theta->SetDataAugPtr(Daug);

	// first test that pointer is set to point to LogDensityPop()
	Theta->SetTheta(d_true_theta);
	hvector h_logdens_from_theta = Theta->GetHostLogDens();
	double logdens_from_theta = 0.0;
	for (int i = 0; i < h_logdens_from_theta.size(); ++i) {
		logdens_from_theta += h_logdens_from_theta[i];
	}
	hvector h_chi = Daug->GetHostChi();
	double* p_theta = thrust::raw_pointer_cast(&h_true_theta[0]);
	double local_chi[3];
	double logdens_from_host = 0.0;
	for (int i = 0; i < ndata; ++i) {
		for (int j = 0; j < pchi; ++j) {
			local_chi[j] = h_chi[j * ndata + i];
		}
		logdens_from_host += LogDensityPop(local_chi, p_theta, pchi, dim_theta);
	}
	double frac_diff = abs(logdens_from_theta - logdens_from_host) / abs(logdens_from_host);
	if (frac_diff < 1e-8) {
		npassed++;
		local_passed++;
	} else {
		std::cerr << "Test for PopulationPar constructor failed: Pointer to LogDensityPop() not correctly set." << std::endl;
		std::cerr << "Log-Density of characteristics|theta from host-side function: " << logdens_from_host << std::endl;
		std::cerr << "Log-Density of characteristics|theta from PopulationPar: " << logdens_from_theta << std::endl;
	}
	nperformed++;

	// TODO: make this calculate the sum, as above

	// now test that pointer is set to point to LogDensityMeas()
	Daug->SetChi(d_true_chi);
	dvector d_logdens_from_daug = Daug->GetDevLogDens();
	double logdens_from_daug = thrust::reduce(d_logdens_from_daug.begin(), d_logdens_from_daug.end());
	h_chi = Daug->GetHostChi();
	logdens_from_host = 0.0;
	double local_meas[3];
	double local_meas_unc[3];
	for (int i = 0; i < ndata; ++i) {
		for (int j = 0; j < pchi; ++j) {
			local_chi[j] = h_chi[j * ndata + i];
			local_meas[j] = h_meas[j * ndata + i];
			local_meas_unc[j] = h_meas_unc[j * ndata + i];
		}
		double logdens_from_host_i = LogDensityMeas(local_chi, local_meas, local_meas_unc, mfeat, pchi);
		logdens_from_host += logdens_from_host_i;
	}

	// TODO: CHECK DAUG.LOGDENSPOP

	frac_diff = abs(logdens_from_daug - logdens_from_host) / abs(logdens_from_host);
	if (frac_diff < 1e-8) {
		npassed++;
		local_passed++;
	} else {
		std::cerr << "Test for DataAugmentation constructor failed: Pointer to LogDensityMeas() not correctly set." << std::endl;
		std::cerr << "Log-Density of measurements|characteristics from host-side function: " << logdens_from_host << std::endl;
		std::cerr << "Log-Density of measurements|characteristics from DataAugmentation: " << logdens_from_daug << std::endl;
	}
	nperformed++;

	if (local_passed == 2) {
		std::cout << "... passed." << std::endl;
	}
}

// test the device side function to generate the chi proposals. this is needed for DataAugmenation::Update().
void UnitTests::DevicePropose()
{
	std::cout << "Testing device-side function that generates proposals for characteristics..." << std::endl;
	int local_passed = 0;

	int local_ndata = 10000;
	// Cuda grid launch
    dim3 nT(256);
    dim3 nB((local_ndata + nT.x-1) / nT.x);
    if (nB.x > 65535)
    {
        std::cerr << "ERROR: Block is too large" << std::endl;
        return;
    }

	// initialize the RNG seed on the GPU first
	curandState* p_devStates;

	// Allocate memory on GPU for RNG states
	CUDA_CHECK_RETURN(cudaMalloc((void **)&p_devStates, nT.x * nB.x * sizeof(curandState)));
	initialize_rng<<<nB,nT>>>(p_devStates);
	CUDA_CHECK_RETURN(cudaPeekAtLastError());
	// Wait until RNG stuff is done running on the GPU, make sure everything went OK
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	// build the chi values on the host and device. just use a values of all ones.
	hvector h_chi(local_ndata * pchi);
	thrust::fill(h_chi.begin(), h_chi.end(), 1.0);
	dvector d_chi = h_chi;
	double* p_chi = thrust::raw_pointer_cast(&d_chi[0]);

	// set covariance matrix of the chi proposals as the identity matrix
	int dim_cholfact = pchi * pchi - ((pchi - 1) * pchi) / 2;
	hvector h_cholfact(local_ndata * dim_cholfact);
	for (int i=0; i<local_ndata; i++) {
		int diag_index = 0;
		for (int j=0; j<pchi; j++) {
			h_cholfact[i + local_ndata * diag_index] = 1.0;
			diag_index += j + 2;
		}
	}

	dvector d_cholfactor = h_cholfact;
	double* p_cholfactor = thrust::raw_pointer_cast(&d_cholfactor[0]);

	// get proposals from the device. since the cholesky factor corresponds to an identity covariance matrix,
	// the proposed chi values should follow a standard normal distribution
	test_propose<<<nB,nT>>>(p_chi, p_cholfactor, p_devStates, local_ndata, pchi);
	CUDA_CHECK_RETURN(cudaPeekAtLastError());

	hvector h_chi_proposed = d_chi;

	bool doprint = false;
	if (doprint) {
		// print out the results to a file
		std::ofstream data_file;
		data_file.open("device_proposal_samples.txt");
		data_file << "# nsamples = " << local_ndata << ", pchi = " << pchi << std::endl;
		for (int i = 0; i < local_ndata; ++i) {
			for (int j = 0; j < pchi; ++j) {
				// chi_samples is [nmcmc, ndata, mfeat]
				data_file << h_chi_proposed[i + local_ndata * j] << " ";
			}
			data_file << std::endl;
		}
		data_file.close();
	}

	std::vector<double> avg_chi(pchi);
	std::vector<double> avg_chi_sqr(pchi);
	std::fill(avg_chi.begin(), avg_chi.end(), 0.0);
	std::fill(avg_chi_sqr.begin(), avg_chi_sqr.end(), 0.0);
	for (int j = 0; j < pchi; ++j) {
		for (int i = 0; i < local_ndata; ++i) {
			double chi_diff = h_chi_proposed[i + local_ndata * j] - h_chi[i + local_ndata * j];
			avg_chi[j] += chi_diff / local_ndata;
			avg_chi_sqr[j] += chi_diff * chi_diff / local_ndata;
		}
	}

	// make sure average value of proposed chi within 3-sigma of expected value
	int npassed_zscore = 0;
	for (int j=0; j<pchi; j++) {
		double zscore = avg_chi[j] * sqrt(double(local_ndata));
		if (abs(zscore) < 3) {
			npassed_zscore++;
		}
	}

	if (npassed_zscore == pchi) {
		npassed++;
		local_passed++;
	} else {
		std::cerr << "Test for device-side Proposal failed: mean of random variables outside of 3-sigma for " <<
				pchi - npassed_zscore << " out of " << pchi << " characteristics." << std::endl;
	}
	nperformed++;

	// make sure variance of proposed chi within 3-sigma of expected value
	npassed_zscore = 0;
	double expected_var = 1.0;
	double var_sigma = 2.0 / sqrt(double(local_ndata));
	for (int j = 0; j < pchi; ++j) {
		double var_chi = avg_chi_sqr[j] - avg_chi[j] * avg_chi[j];
		double zscore = (var_chi - expected_var) / var_sigma;
		if (abs(zscore) < 3) {
			npassed_zscore++;
		}
	}
	if (npassed_zscore == pchi) {
		npassed++;
		local_passed++;
	} else {
		std::cerr << "Test for device-side Proposal failed: variance of random variables outside of 3-sigma for " <<
				pchi - npassed_zscore << " out of " << pchi << " characteristics." << std::endl;
	}
	nperformed++;

	if (local_passed == 2) {
		std::cout << "... passed." << std::endl;
	}

	cudaFree(p_devStates);
}

// check that Accept accepts better proposals on the GPU, and updates the chi values
void UnitTests::DeviceAccept()
{
	std::cout << "Testing device-side Metropolis acceptance for characteristics..." << std::endl;

	// initialize the RNG seed on the GPU first
	curandState* p_devStates;

	// Allocate memory on GPU for RNG states
	CUDA_CHECK_RETURN(cudaMalloc((void **)&p_devStates, nThreads.x * nBlocks.x * sizeof(curandState)));
	initialize_rng<<<nBlocks, nThreads>>>(p_devStates);
	CUDA_CHECK_RETURN(cudaPeekAtLastError());
	// Wait until RNG stuff is done running on the GPU, make sure everything went OK
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	// first update the chi-values on the device. start with the true values.
	int dim_cholfact = pchi * pchi - ((pchi - 1) * pchi) / 2;
	hvector h_cholfact(ndata * dim_cholfact);
	for (int i=0; i<ndata; i++) {
		// set covariance matrix of the chi proposals as the identity matrix
		int diag_index = 0;
		for (int j=0; j<pchi; j++) {
			h_cholfact[i + ndata * diag_index] = 1.0;
			diag_index += j + 2;
		}
	}
	dvector d_cholfact = h_cholfact;

	hvector h_logdens_meas(ndata);  // just set initial log-densities to zero
	thrust::fill(h_logdens_meas.begin(), h_logdens_meas.end(), 0.0);
	dvector d_logdens_meas = h_logdens_meas;
	hvector h_logdens_pop(ndata);
	thrust::fill(h_logdens_pop.begin(), h_logdens_pop.end(), 0.0);
	dvector d_logdens_pop = h_logdens_pop;

	thrust::host_vector<int> h_naccept(ndata);
	thrust::fill(h_naccept.begin(), h_naccept.end(), 0);
	thrust::device_vector<int> d_naccept = h_naccept;

	dvector d_chi = d_true_chi;
	hvector h_chi = h_true_chi;

	double* p_chi = thrust::raw_pointer_cast(&d_chi[0]);
	double* p_cholfact = thrust::raw_pointer_cast(&d_cholfact[0]);
	double* p_logdens_meas = thrust::raw_pointer_cast(&d_logdens_meas[0]);
	double* p_logdens_pop = thrust::raw_pointer_cast(&d_logdens_pop[0]);
	int* p_naccept = thrust::raw_pointer_cast(&d_naccept[0]);

	test_accept<<<nBlocks, nThreads>>>(p_chi, p_cholfact, p_logdens_meas, p_logdens_pop, p_devStates, p_naccept, ndata, pchi);

	hvector h_proposed_chi = d_chi;
	hvector h_proposed_logdens_meas = d_logdens_meas;
	hvector h_proposed_logdens_pop = d_logdens_pop;

	double snorm_deviate[3] = {1.2, -0.6, 0.8};
	int nmatch_chi(0), nmatch_logdens(0);
	// make sure Metropolis step saved the new values of chi and the logdensities
	for (int i=0; i<ndata; ++i) {
		int cholfact_index = 0;
		for (int j = 0; j < pchi; ++j) {
			double scaled_proposal_j = 0.0;
			for (int k = 0; k < (j+1); ++k) {
				scaled_proposal_j += h_cholfact[cholfact_index * ndata + i] * snorm_deviate[k];
				cholfact_index++;
			}
			double proposed_chi_ij = h_chi[ndata * j + i] + scaled_proposal_j;
			if (approx_equal(proposed_chi_ij, h_proposed_chi[ndata * j + i], 1e-8)) {
				nmatch_chi++;
			}
		}
		if ((h_proposed_logdens_meas[i] == 0.5) && (h_proposed_logdens_pop[i] == 0.5)) {
			nmatch_logdens++;
		}
	}

	if ((nmatch_chi != h_proposed_chi.size()) || (nmatch_logdens != ndata)) {
		std::cerr << "Test for device-side Accept failed: values updated on device do not match those updated on the host."
				<< std::endl;
	} else {
		std::cout << "... passed." << std::endl;
		npassed++;
	}

	nperformed++;
	cudaFree(p_devStates);
}

// check that Adapt updates the cholesky factor of the chi proposals on the GPU
void UnitTests::DeviceAdapt()
{
	std::cout << "Testing device-side adapt step via rank-1 cholesky update..." << std::endl;

	int dim_cholfact = pchi * pchi - ((pchi - 1) * pchi) / 2;
	hvector h_cholfact(ndata * dim_cholfact);
	for (int i=0; i<ndata; i++) {
		// set covariance matrix of the chi proposals as the identity matrix
		int diag_index = 0;
		for (int j=0; j<pchi; j++) {
			h_cholfact[i + ndata * diag_index] = 1.0;
			diag_index += j + 2;
		}
	}
	dvector d_cholfact = h_cholfact;

	double* p_cholfact = thrust::raw_pointer_cast(&d_cholfact[0]);
	int current_iter = 5;

	test_adapt<<<nBlocks, nThreads>>>(p_cholfact, ndata, pchi, current_iter);

	hvector h_updated_cholfact = d_cholfact;

	// now adapt the cholesky factor manually on the CPU
	double snorm_deviate[3] = {1.2, -0.6, 0.8};
	double scaled_proposal[3];
	double this_cholfact[6];
	int nmatch(0);
	double metro_ratio = 0.34;
	for (int i=0; i<ndata; ++i) {
		int cholfact_index = 0;
		// first get scaled proposed chi value
		for (int j = 0; j < pchi; ++j) {
			double scaled_proposal_j = 0.0;
			for (int k = 0; k < (j+1); ++k) {
				scaled_proposal_j += h_cholfact[cholfact_index * ndata + i] * snorm_deviate[k];
				cholfact_index++;
			}
			scaled_proposal[j] = scaled_proposal_j;
		}
		// store this cholesky factor in an array with shape needed by AdaptProp
		for (int j=0; j<dim_cholfact; j++) {
			this_cholfact[j] = h_cholfact[i + ndata * j];
		}
		AdaptProp(this_cholfact, snorm_deviate, scaled_proposal, metro_ratio, pchi, current_iter);

		// now make sure the two cholesky factors agree
		for (int j = 0; j < dim_cholfact; ++j) {
			if (approx_equal(this_cholfact[j], h_updated_cholfact[ndata * j + i], 1e-8)) {
				nmatch++;
			}
		}
	}

	if (nmatch == h_updated_cholfact.size()) {
		std::cout << "... passed." << std::endl;
		npassed++;
	} else {
		std::cerr << "Test for RAM Cholesky adaption step on the GPU failed: device and host results do not agree." << std::endl;
	}
	nperformed++;
}

// check that DataAugmentation::Update always accepts when the proposed and current chi values are the same
void UnitTests::DaugAcceptSame()
{
	std::cout << "Testing that update for DataAugmentation always accepts when chi values are unchanged...." << std::endl;
	int local_passed = 0;

	boost::shared_ptr<DataAugmentation<3,3,3> > Daug(new DataAugmentation<3,3,3> (meas, meas_unc, ndata, nBlocks, nThreads));
	boost::shared_ptr<PopulationPar<3,3,3> > Theta(new PopulationPar<3,3,3> (nBlocks, nThreads));

	Daug->SetPopulationPtr(Theta);
	Theta->SetDataAugPtr(Daug);

	Daug->SetChi(d_true_chi);
	Theta->SetTheta(d_true_theta);

	// set the cholesky factors to zero so that NormalPropose() just returns the same chi value
	int dim_cholfact = pchi * pchi - ((pchi - 1) * pchi) / 2;
	hvector h_cholfact(ndata * dim_cholfact);
	thrust::fill(h_cholfact.begin(), h_cholfact.end(), 0.0);
	dvector d_cholfact = h_cholfact;
	Daug->SetCholFact(d_cholfact);

	Daug->Update();

	thrust::host_vector<int> h_naccept = Daug->GetNaccept();
	// make sure all of the proposals are accepted, since the proposed chi values are the same as the current ones
	int naccept = 0;
	for (int i = 0; i < h_naccept.size(); ++i) {
		naccept += h_naccept[i];
	}
	if (naccept == ndata) {
		npassed++;
		local_passed++;
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

	int ntrials = 1000;
	for (int i = 0; i < ntrials; ++i) {
		Theta->SetCholFact(h_cholfact);
		Theta->Update();
	}
	naccept = Theta->GetNaccept();
	if (naccept == ntrials) {
		npassed++;
		local_passed++;
	} else {
		std::cerr << "Test for PopulationPar::Accept() failed: Did not accept all of the proposed population "
				<< "values when they are the same." << std::endl;
	}
	nperformed++;
	if (local_passed == 2) {
		std::cout << "... passed." << std::endl;
	}
}

// make sure that DataAugmentation::Update() accepts and saves Chi values when the posterior is much higher
void UnitTests::DaugAcceptBetter() {
	std::cout << "Testing DaugAccept.Update() always accepts a better proposal...." << std::endl;
	int local_passed = 0;

	boost::shared_ptr<DataAugmentation<3,3,3> > Daug(new DataAugmentation<3,3,3> (meas, meas_unc, ndata, nBlocks, nThreads));
	boost::shared_ptr<PopulationPar<3,3,3> > Theta(new PopulationPar<3,3,3> (nBlocks, nThreads));

	Daug->SetPopulationPtr(Theta);
	Theta->SetDataAugPtr(Daug);

	Daug->Initialize();
	Theta->Initialize();

	Daug->SetChi(d_true_chi);
	Theta->SetTheta(d_true_theta);

	// artificially set the conditional log-posteriors to a really low value to make sure we accept the proposal
	hvector h_logdens_meas(ndata);
	thrust::fill(h_logdens_meas.begin(), h_logdens_meas.end(), -1e10);
	dvector d_logdens_meas = h_logdens_meas;
	Daug->SetLogDens(d_logdens_meas);

	Daug->Update();
	thrust::host_vector<int> h_naccept = Daug->GetNaccept();

	// make sure all of the proposals are accepted
	int naccept = 0;
	for (int i = 0; i < h_naccept.size(); ++i) {
		naccept += h_naccept[i];
	}
	if (naccept == ndata) {
		npassed++;
		local_passed++;
	} else {
		std::cerr << "Test for Daug::Accept() failed: Did not accept all of the proposed characteristics when "
				<< "the posterior is improved." << std::endl;
	}
	nperformed++;

	// make sure that the proposed values and new posteriors are saved
	hvector h_new_chi = Daug->GetDevChi();
	int ndiff_chi = 0;
	for (int i = 0; i < h_true_chi.size(); ++i) {
		// if the proposal is saved, then new chi should not equal initial chi
		if (!approx_equal(h_new_chi[i], h_true_chi[i], 1e-6)) {
			ndiff_chi++;
		}
	}
	if (ndiff_chi == h_true_chi.size()) {
		npassed++;
		local_passed++;
	} else {
		std::cerr << "Test for Daug::Accept() failed: Did not update the characteristics when the "
				<< "proposals are accepted." << std::endl;
	}
	nperformed++;
	hvector h_new_logdens = Daug->GetDevLogDens();
	int ndiff_logdens = 0;
	for (int i = 0; i < h_new_logdens.size(); ++i) {
		if (!approx_equal(h_new_logdens[i], h_logdens_meas[i], 1e-6)) {
			ndiff_logdens++;
		}
	}
	if (ndiff_logdens == h_logdens_meas.size()) {
		npassed++;
		local_passed++;
	} else {
		std::cerr << "Test for Daug::Accept() failed: Did not update the posteriors when the "
				<< "proposals are accepted." << std::endl;
	}
	nperformed++;

	if (local_passed == 3) {
		std::cout << "... passed." << std::endl;
	}

	/*
	 * Now do the same thing, but for the population parameters.
	 */
	std::cout << "Testing Theta.Update() always accepts a better proposal...." << std::endl;
	local_passed = 0;

	hvector h_logdens_pop(ndata);
	thrust::fill(h_logdens_pop.begin(), h_logdens_pop.end(), -1e10);
	dvector d_logdens_pop = h_logdens_pop;
	Theta->SetLogDens(d_logdens_pop);

	Theta->Update();

	// make sure we accepted the proposal
	naccept = Theta->GetNaccept();
	if (naccept == 1) {
		npassed++;
		local_passed++;
	} else {
		std::cerr << "Test for PopulationPar::Accept() failed: Did not accept the proposed population "
				<< "values when the posterior is improved" << std::endl;
	}
	nperformed++;

	// make sure that the proposed values and new posteriors are saved
	hvector h_new_theta = Theta->GetHostTheta();
	int ndiff_theta = 0;
	for (int i = 0; i < h_new_theta.size(); ++i) {
		if (!approx_equal(h_new_theta[i], h_true_theta[i], 1e-6)) {
			ndiff_theta++;
		}
	}
	if (ndiff_theta == h_new_theta.size()) {
		// did we save the accepted theta?
		npassed++;
		local_passed++;
	} else {
		std::cerr << "Test for PopulationPar::Accept() failed: Did not update the population parameter value when the "
				<< "proposal is accepted." << std::endl;
	}
	nperformed++;
	h_new_logdens = Theta->GetHostLogDens();
	ndiff_logdens = 0;
	for (int i = 0; i < h_new_logdens.size(); ++i) {
		if (!approx_equal(h_new_logdens[i], h_logdens_pop[i], 1e-6)) {
			ndiff_logdens++;
		}
	}
	if (ndiff_logdens == h_logdens_pop.size()) {
		// did we update the posterior?
		npassed++;
		local_passed++;
	} else {
		std::cerr << "Test for PopulationPar::Accept() failed: Did not update the posteriors when the "
				<< "proposals are accepted." << std::endl;
	}
	nperformed++;
	if (local_passed == 3) {
		std::cout << "... passed." << std::endl;
	}
}

// check that constructor for population parameter correctly set the pointer data member of DataAugmentation
void UnitTests::GibbsSamplerPtr() {
	std::cout << "Making sure DataAugmentation and PopulationPar know about eachother." << std::endl;
	int local_passed = 0;

	int niter = 10;
	int nburn = 10;
	GibbsSampler<3, 3, 3> Sampler(meas, meas_unc, ndata, nBlocks, nThreads, niter, nburn);

	boost::shared_ptr<DataAugmentation<3,3,3> > DaugPtr = Sampler.GetDaugPtr();
	boost::shared_ptr<PopulationPar<3,3,3> > ThetaPtr = Sampler.GetThetaPtr();

	if (DaugPtr->GetPopulationPtr() == ThetaPtr) {
		npassed++;
		local_passed++;
	} else {
		std::cerr << "Test for GibbsSampler constructor failed: Data Augmentation does not know about the Population Parameter."
				<< std::endl;
	}

	nperformed++;

	if (ThetaPtr->GetDataAugPtr() == DaugPtr) {
		npassed++;
		local_passed++;
	} else {
		std::cerr << "Test for GibbsSampler constructor failed: Population Parameter does not know about the Data Augmentation."
				<< std::endl;
	}

	nperformed++;

	if (local_passed == 2) {
		std::cout << "... passed." << std::endl;
	}

}

// test the Gibbs Sampler for a Normal-Normal model keeping the characteristics fixed
void UnitTests::FixedChar() {

	std::cout << "Testing Gibbs Sampler for fixed characteristics..." << std::endl;;
	int local_passed = 0;

	// setup the Gibbs sampler object
	int niter(50000), nburn(25000);
	GibbsSampler<3,3,3> Sampler(meas, meas_unc, ndata, nBlocks, nThreads, niter, nburn);

	// keep the characteristics fixed
	Sampler.GetDaugPtr()->SetChi(d_true_chi);
	Sampler.FixChar();

	// start at the true values
	Sampler.GetThetaPtr()->SetTheta(d_true_theta);
	// run the MCMC sampler
	Sampler.Run();

	// check the acceptance rate
	double target_rate = 0.4;
	double naccept = Sampler.GetThetaPtr()->GetNaccept();
	double arate = naccept / double(niter);
	double frac_diff = abs(arate - target_rate) / target_rate;
	// make sure acceptance rate is within 5% of the target rate
	if (frac_diff < 0.05) {
		npassed++;
		local_passed++;
	} else {
		std::cerr << "Test for GibbsSampler with fixed Characteristics failed: Acceptance rate "
				<< "is not within 5% of the target rate." << std::endl;
		std::cerr << arate << ", " << target_rate << std::endl;
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

	// make sure estimated value and true value are within 5% of each other
	frac_diff = 0.0;
	for (int j = 0; j < dim_theta; ++j) {
		frac_diff += abs(theta_mean[j] - theta_mean_true[j]) / abs(theta_mean_true[j]);
	}
	frac_diff /= dim_theta;
	if (frac_diff < 0.05) {
		npassed++;
		local_passed++;
	} else {
		std::cerr << "Test for GibbsSampler with fixed Characteristics failed: Average fractional difference "
				<< "between estimated posterior mean and true value is greater than 5%." << std::endl;
		std::cerr << "Average fractional difference: " << frac_diff << std::endl;
		for (int j = 0; j < dim_theta; ++j) {
			std::cerr << "Estimated, True:" << std::endl;
			std::cerr << theta_mean[j] << ", " << theta_mean_true[j] << std::endl;
		}
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

	// make sure estimated value and true value are within 5% of eachother
	frac_diff = 0.0;
	for (int j = 0; j < dim_theta; ++j) {
		for (int k = 0; k < dim_theta; ++k) {
			frac_diff += abs(mean_covar[j][k] - mean_covar_true[j][k]) / abs(mean_covar[j][k]);
		}
	}
	frac_diff /= (dim_theta * dim_theta) ;
	if (frac_diff < 0.05) {
		npassed++;
		local_passed++;
	} else {
		std::cerr << "Test for GibbsSampler with fixed Characteristics failed: Average fractional difference "
				<< "between estimated posterior covariance in mean parameter and the true value is greater than 5%." << std::endl;
		std::cerr << "Average fractional difference:" << frac_diff << std::endl;
		std::cerr << "Estimated posterior covariance:" << std::endl;
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				std::cerr << mean_covar[i][j] << "  ";
			}
			std::cerr << std::endl;
		}
		std::cerr << "True posterior covariance:" << std::endl;
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				std::cerr << mean_covar_true[i][j] << "  ";
			}
			std::cerr << std::endl;
		}
	}
	nperformed++;

	bool print_thetas = false;
	if (print_thetas) {
		// print out results to a file
		vecvec theta_samples = Sampler.GetPopSamples();
		std::ofstream mcmc_file;
		mcmc_file.open("const_chi_samples.txt");
		mcmc_file << "# nsamples = " << theta_samples.size() << ", ndata = " << ndata << ", mfeat = " << mfeat << std::endl;
		for (int l = 0; l < theta_samples.size(); ++l) {
			for (int i = 0; i < dim_theta; ++i) {
				// theta_samples is [nmcmc, dim_theta]
				mcmc_file << theta_samples[l][i] << " ";
			}
			mcmc_file << std::endl;
		}
		mcmc_file.close();
	}

	if (local_passed == 3) {
		std::cout << "... passed." << std::endl;
	}

}

// test the Gibbs Sampler for a Normal-Normal model keeping the population parameter fixed
void UnitTests::FixedPopPar() {

	std::cout << "Testing Gibbs Sampler for fixed population parameter...";
	int local_passed = 0;

	// setup the Gibbs sampler object
	int niter(50000), nburn(25000);
	GibbsSampler<3,3,3> Sampler(meas, meas_unc, ndata, nBlocks, nThreads, niter, nburn);

	// keep the population parameter fixed
	Sampler.GetThetaPtr()->SetTheta(d_true_theta);
	Sampler.FixPopPar();

	// start at the true values
	Sampler.GetDaugPtr()->SetChi(d_true_chi);
	// run the MCMC sampler
	Sampler.Run();

	// check the acceptance rate
	double target_rate = 0.4;
	thrust::host_vector<int> naccept = Sampler.GetDaugPtr()->GetNaccept();
	std::vector<double> frac_diff(naccept.size());
	int npass = 0;
	for (int i = 0; i < naccept.size(); ++i) {
		double arate = naccept[i] / double(niter);
		frac_diff[i] = abs(arate - target_rate) / target_rate;
		// make sure acceptance rate is within 5% of the target rate
		if (frac_diff[i] < 0.05) {
			npass++;
		}
	}

	// make sure acceptance rate is within 5% of the target rate
	if (npass == ndata) {
		npassed++;
		local_passed++;
	} else {
		int nbad = ndata - npass;
		std::cerr << "Test for GibbsSampler with fixed PopulationPar failed: Acceptance rate "
				<< "is not within 5% of the target rate for " << nbad << " characteristics." << std::endl;
		for (int i = 0; i < frac_diff.size(); ++i) {
			if (frac_diff[i] > 0.05) {
				std::cout << naccept[i] / double(niter) << ", " << target_rate << std::endl;
			}
		}
	}
	nperformed++;

	// grab the MCMC samples of the population parameter
	std::vector<vecvec> csamples = Sampler.GetCharSamples();

	// compare the estimated posterior mean and covariance of characteristics with true values
	// for the NormalNormal model
	int npass_mean = 0;
	int npass_covar = 0;
	for (int i = 0; i < ndata; ++i) {
		double chi_mean[3] = {0.0, 0.0, 0.0};
		vecvec this_csamples(csamples.size());
		for (int j = 0; j < csamples.size(); ++j) {
			this_csamples[j] = csamples[j][i];
			chi_mean[0] += csamples[j][i][0] / csamples.size();
			chi_mean[1] += csamples[j][i][1] / csamples.size();
			chi_mean[2] += csamples[j][i][2] / csamples.size();
		}
		// get true value of posterior covariance matrix
		double meas_var_inv[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		meas_var_inv[0] = 1.0 / meas_unc[i][0] / meas_unc[i][0];
		meas_var_inv[4] = 1.0 / meas_unc[i][1] / meas_unc[i][1];
		meas_var_inv[8] = 1.0 / meas_unc[i][2] / meas_unc[i][2];
		double post_var_inv[9];
		for (int k = 0; k < mfeat * mfeat; ++k) {
			post_var_inv[k] = meas_var_inv[k] + covar_inv[k];
		}
		double post_var[9];
		matrix_invert3d(post_var_inv, post_var);
		// get true value of posterior mean
		double wchi_mean_true[3] = {0.0, 0.0, 0.0};
		for (int k = 0; k < mfeat; ++k) {
			// contribution to posterior mean from measured value of characteristic
			wchi_mean_true[k] = meas[i][k] / meas_unc[i][k] / meas_unc[i][k];
			for (int l = 0; l < mfeat; ++l) {
				// add in prior contribution
				wchi_mean_true[k] += covar_inv[k * mfeat + l] * h_true_theta[l];
			}
		}
		double chi_mean_true[3] = {0.0, 0.0, 0.0};
		for (int k = 0; k < mfeat; ++k) {
			for (int l = 0; l < mfeat; ++l) {
				// true value of posterior mean
				chi_mean_true[k] += post_var[k * mfeat + l] * wchi_mean_true[l];
			}
		}
		// make sure estimated and true posterior means are within 5% of each other
		double frac_diff = 0.0;
		for (int k = 0; k < mfeat; ++k) {
			frac_diff += abs(chi_mean_true[k] - chi_mean[k]) / abs(chi_mean_true[k]) / dim_theta;
		}
		if (frac_diff < 0.05) {
			npass_mean++;
		}
		// make sure estimated and true posterior standard deviations are within 5% of eachother
		std::vector<double> post_covar_est = covariance(this_csamples);
		frac_diff = 0.0;
		for (int j = 0; j < pchi; ++j) {
			for (int k = 0; k < pchi; ++k) {
				if (j == k) {
					double post_sigma_est = sqrt(post_covar_est[j * pchi + k]);
					double post_sigma_true = sqrt(post_var[j * pchi + k]);
					frac_diff += abs(post_sigma_est - post_sigma_true) / post_sigma_true;
				}
			}
		}
		frac_diff /= (pchi);
		if (frac_diff < 0.05) {
			npass_covar++;
		}
	}

	if (npass_mean > 0.9 * ndata) {
		npassed++;
		local_passed++;
	}
	else {
		int nfailed = ndata - npass_mean;
		std::cerr << "Test for GibbsSampler with fixed PopulationPar failed: Average fractional difference "
				<< "between estimated posterior mean and true value is greater than 5% for " << nfailed
				<< " out of " << ndata << " characteristics." << std::endl;
	}
	nperformed++;
	if (npass_covar > 0.9 * ndata) {
		npassed++;
		local_passed++;
	}
	else {
		int nfailed = ndata - npass_covar;
		std::cerr << "Test for GibbsSampler with fixed PopulationPar failed: Average fractional difference "
				<< "between estimated posterior covariance and true value is greater than 5% for " << nfailed
				<< "out of " << ndata << " characteristics." << std::endl;
	}
	nperformed++;

	if (local_passed == 3) {
		std::cout << "... passed." << std::endl;
	}

	bool print_chis = false;
	if (print_chis) {
		// print out the results to a file
		std::ofstream mcmc_file;
		mcmc_file.open("const_theta_samples.txt");
		mcmc_file << "# nsamples = " << csamples.size() << ", ndata = " << ndata << ", pchi = " << pchi << std::endl;
		for (int l = 0; l < csamples.size(); ++l) {
			for (int i = 0; i < ndata; ++i) {
				for (int j = 0; j < pchi; ++j) {
					// chi_samples is [nmcmc, ndata, mfeat]
					mcmc_file << csamples[l][i][j] << " ";
				}
			}
			mcmc_file << std::endl;
		}
		mcmc_file.close();

		std::vector<double> logdens_samples = Sampler.GetLogDensMeas();
		std::ofstream logdens_file;
		logdens_file.open("const_theta_logdens.txt");
		for (int i = 0; i < csamples.size(); ++i) {
			logdens_file << logdens_samples[i] << std::endl;
		}
		logdens_file.close();
	}
}

// test the Gibbs Sampler for a Normal-Normal model
void UnitTests::NormNorm()
{
	std::cout << "Testing Gibbs Sampler for normal-normal model...";
	int local_passed = 0;

	// setup the Gibbs sampler object
	int niter(50000), nburn(25000);
	GibbsSampler<3,3,3> Sampler(meas, meas_unc, ndata, nBlocks, nThreads, niter, nburn);

	// run the MCMC sampler
	Sampler.Run();

	/*
	 * FIRST CHECK THAT ACCEPTANCE RATES HAVE CONVERGED TO WITHIN 5% OF THE TARGET RATE
	 */

	double target_rate = 0.4;
	// first check the acceptance rate for the population level parameter
	double naccept_theta = Sampler.GetThetaPtr()->GetNaccept();
	double arate = naccept_theta / double(niter);
	double frac_diff_theta = abs(arate - target_rate) / target_rate;
	// make sure acceptance rate is within 5% of the target rate
	if (frac_diff_theta < 0.05) {
		npassed++;
		local_passed++;
	} else {
		std::cerr << "Test for GibbsSampler under normal-normal model failed: Acceptance rate "
				<< "is not within 5% of the target rate for the population parameter." << std::endl;
		std::cerr << arate << ", " << target_rate << std::endl;
	}
	nperformed++;

	// check the acceptance rate for the characteristics
	thrust::host_vector<int> naccept_chi = Sampler.GetDaugPtr()->GetNaccept();
	std::vector<double> frac_diff_chi(naccept_chi.size());
	int npass = 0;
	for (int i = 0; i < naccept_chi.size(); ++i) {
		arate = naccept_chi[i] / double(niter);
		frac_diff_chi[i] = abs(arate - target_rate) / target_rate;
		// make sure acceptance rate is within 5% of the target rate
		if (frac_diff_chi[i] < 0.05) {
			npass++;
		}
	}

	// make sure acceptance rate is within 5% of the target rate
	if (npass > 0.9 * ndata) {
		npassed++;
		local_passed++;
	} else {
		int nbad = ndata - npass;
		std::cerr << "Test for GibbsSampler under normal-normal model failed: Acceptance rate "
				<< "is not within 5% of the target rate for " << nbad << " out of " << ndata <<
				" characteristics." << std::endl;
		for (int i = 0; i < frac_diff_chi.size(); ++i) {
			if (frac_diff_chi[i] > 0.05) {
				std::cout << i << ", " << naccept_chi[i] / double(niter) << ", " << target_rate << std::endl;
			}
		}
	}
	nperformed++;

	/*
	 * NOW CHECK THAT THE ESTIMATED VALUE OF THE POPULATION PARAMETER IS ABOUT 3-SIGMA OF THE TRUE VALUE
	 */
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
	// get the estimated posterior covariance of the normal mean parameter
	double tmean_covar[9];
	int idx(0);
	for (int j = 0; j < dim_theta; ++j) {
		for (int k = 0; k < dim_theta; ++k) {
			// initialize to zero
			tmean_covar[idx] = 0.0;
			idx++;
		}
	}
	for (int i = 0; i < tsamples.size(); ++i) {
		idx = 0;
		for (int j = 0; j < dim_theta; ++j) {
			for (int k = 0; k < dim_theta; ++k) {
				tmean_covar[idx] += (tsamples[i][j] - theta_mean[j]) * (tsamples[i][k] - theta_mean[k]);
				idx++;
			}
		}
	}
	idx = 0;
	for (int j = 0; j < dim_theta; ++j) {
		for (int k = 0; k < dim_theta; ++k) {
			tmean_covar[idx] /= tsamples.size();
			idx++;
		}
	}

	// get number of mahalonabis distance to true value
	double tmean_covar_inv[9];
	matrix_invert3d(tmean_covar, tmean_covar_inv);

	double** tmean_covar_inv2d;
	tmean_covar_inv2d = new double* [3];
	for (int j = 0; j < 3; ++j) {
		tmean_covar_inv2d[j] = new double [3];
	}
	double tmean_diff[3];
	idx = 0;
	for (int i = 0; i < 3; ++i) {
		tmean_diff[i] = theta_mean[i] - h_true_theta[i];
		for (int j = 0; j < 3; ++j) {
			tmean_covar_inv2d[i][j] = tmean_covar_inv[idx];
			idx++;
		}
	}

	double chisqr = mahalanobis_distance(tmean_covar_inv2d, tmean_diff, dim_theta);
	double chi3_99 = 11.34;  // 99th percentile of chi-square distribution with 3 degrees of freedom

	if (chisqr < chi3_99) {
		npassed++;
		local_passed++;
	} else {
		std::cerr << "Test for GibbsSampler under normal-normal model failed: Estimated value of theta is too far from true value"
				<< std::endl;
		std::cerr << "Mahalanobis distance is " << sqrt(chisqr) << std::endl;
	}
	nperformed++;

	// free memory
	for (int i = 0; i < 3; ++i) {
		delete [] tmean_covar_inv2d[i];
	}
	delete tmean_covar_inv2d;
	/*
	 * NOW CHECK THAT THE ESTIMATED VALUE OF THE CHARACTERISTICS ARE WITHIN ABOUT 3-SIGMA OF THE TRUE VALUES AT
	 * LEAST 95% OF THE TIME
	 */

	// grab the MCMC samples of the population parameter
	std::vector<vecvec> csamples = Sampler.GetCharSamples();
	int nchi_passed = 0;
	for (int i = 0; i < ndata; ++i) {
		// get the estimated posterior mean for this characteristic
		double chi_mean[3] = {0.0, 0.0, 0.0};
		vecvec this_csamples(csamples.size());
		for (int j = 0; j < csamples.size(); ++j) {
			this_csamples[j] = csamples[j][i];
			chi_mean[0] += csamples[j][i][0] / csamples.size();
			chi_mean[1] += csamples[j][i][1] / csamples.size();
			chi_mean[2] += csamples[j][i][2] / csamples.size();
		}
		// get the estimated posterior covariance for this characteristic
		std::vector<double> chi_covar = covariance(this_csamples);

		// convert format
		double chi_covar_inv[9];
		matrix_invert3d(&chi_covar.front(), chi_covar_inv);

		double** chi_covar_inv2d;
		chi_covar_inv2d = new double* [3];
		for (int j = 0; j < 3; ++j) {
			chi_covar_inv2d[j] = new double [3];
		}

		double chi_diff[3];
		idx = 0;
		for (int j = 0; j < 3; ++j) {
			chi_diff[j] = chi_mean[j] - h_true_chi[j * ndata + i];
			for (int k = 0; k < 3; ++k) {
				chi_covar_inv2d[j][k] = chi_covar_inv[idx];
				idx++;
			}
		}
		// get mahalanobis distance to true value for this characteristic
		chisqr = mahalanobis_distance(chi_covar_inv2d, chi_diff, pchi);

		if (chisqr < chi3_99) {
			nchi_passed++;
		}
		// free memory
		for (int j = 0; j < 3; ++j) {
			delete [] chi_covar_inv2d[j];
		}
		delete chi_covar_inv2d;
	}

	if (nchi_passed > 0.95 * ndata) {
		npassed++;
		local_passed++;
	} else {
		int nfailed = ndata - nchi_passed;
		std::cerr << "Test for GibbsSampler under normal-normal model failed: Estimated value of the characteristic is too far "
				<< "from true value for " << nfailed << " out of " << ndata << " characteristics." << std::endl;
	}
	nperformed++;

	if (local_passed == 4) {
		std::cout << "... passed." << std::endl;
	}

	bool print_values = false;
	if (print_values) {
		// print out the results to a file
		vecvec theta_samples = Sampler.GetPopSamples();
		std::ofstream mcmc_file;
		mcmc_file.open("normnorm_theta_samples.txt");
		mcmc_file << "# nsamples = " << theta_samples.size() << ", ndata = " << ndata << ", mfeat = " << mfeat << std::endl;
		for (int l = 0; l < theta_samples.size(); ++l) {
			for (int i = 0; i < dim_theta; ++i) {
				// theta_samples is [nmcmc, dim_theta]
				mcmc_file << theta_samples[l][i] << " ";
			}
			mcmc_file << std::endl;
		}
		mcmc_file.close();

		std::vector<double> logdens_samples = Sampler.GetLogDensPop();
		std::ofstream logdens_file;
		logdens_file.open("normnorm_pop_logdens.txt");
		for (int i = 0; i < csamples.size(); ++i) {
			logdens_file << logdens_samples[i] << std::endl;
		}
		logdens_file.close();

		mcmc_file.open("normnorm_chi_samples.txt");
		mcmc_file << "# nsamples = " << csamples.size() << ", ndata = " << ndata << ", pchi = " << pchi << std::endl;
		for (int l = 0; l < csamples.size(); ++l) {
			for (int i = 0; i < ndata; ++i) {
				for (int j = 0; j < pchi; ++j) {
					// chi_samples is [nmcmc, ndata, mfeat]
					mcmc_file << csamples[l][i][j] << " ";
				}
			}
			mcmc_file << std::endl;
		}
		mcmc_file.close();

		logdens_samples = Sampler.GetLogDensMeas();
		logdens_file.open("normnorm_meas_logdens.txt");
		for (int i = 0; i < csamples.size(); ++i) {
			logdens_file << logdens_samples[i] << std::endl;
		}
		logdens_file.close();
	}
}

// print out summary of test results
void UnitTests::Finish() {
	std::cout << npassed << " tests passed out of " << nperformed << " tests performed." << std::endl;
	npassed = 0;
	nperformed = 0;
}
