/*
 * dusthm.cu
 *
 *  Created on: Mar 26, 2014
 *      Author: Brandon C. Kelly
 *
 * This file illustrates how to setup a hierarchical model for parameters of spectral energy distributions (SEDs) of astronomical dust
 * in the far-IR and sample from this using CUDAHM. The model is:
 *
 * 		y_ij | chi_i ~ N(SED_j(chi_i), sigma^2_ij), i = 1, ..., N; j = 1, ..., 5   (measurement model)
 * 		SED_j(chi_i) = C_i * (nu_j / nu_0)^beta_i * BlackBody_{nu_j}(T_i),   chi = (log C, beta, log T)
 * 		chi_i | theta ~ t_8(mu, Covar),   theta = (mu, Covar)  					   (population model for unknown characteristics)
 *
 * 	Prior:
 *
 * 		mu ~ N(mu_0, V_0)
 * 		p(Covar) = 'Separation Strategy Prior' of Barnard, McCulloch, & Meng 2000, Statistical Sinica, 10, 1281-1311
 *
 * Under this model the N values of the unknown SED parameters, chi, are drawn from a 3-dimensional multivariate student's t density
 * with 8 degrees of freedom and unknown mean, mu, and covariance matrix, Covar. The prior on mu is Normal, broad, and centered at
 * values that are reasonable based on previous scientific investigations. The prior on the covariance matrix is the 'separation strategy'
 * prior of Barnard et al. (2000). This prior places independent prior distributions on the standard deviations and correlations. The
 * prior on the correlations is marginally uniform and given by Equation (8) of Barnard et al. (2000), while the prior on the standard
 * deviations are independent broad log-normal distributions with geometric means of 1.0. This is a simplified verson of the model described
 * in Kelly et al. (2012, The Astrophysical Journal, 752, 55).
 *
 * We desire to infer the values of the characteristics (chi) and their mean and covariance (theta) from a set of SEDs contaminated with
 * Gaussian measurement noise with known variances. We do this by using CUDAHM to construct a Metropolis-within-Gibbs MCMC sampler to sample
 * from the posterior probability distribution of chi_1, ..., chi_N, theta | {y_ij | i = 1, ..., N; j = 1, ..., 5}. In order to accomplish this
 * we subclass the PopulationPar class to create a new class DustPopPar which overrides the uninformative prior distribution of PopulationPar.
 * In addition, we also override the methods that set the initial values of theta and chi, which in the base classes are just set to zero. In
 * order to set the initial values of the chis to anything other than zero we also need to subclass the DataAugmentation class, which we do here
 * with the ConstBetaTemp class.
 */

// standard library includes
#include <iostream>
#include <time.h>

// local CUDAHM includes
#include "../../cudahm/src/GibbsSampler.hpp"
#include "input_output.hpp"
#include "ConstBetaTemp.cuh"
#include "DustPopPar.hpp"

/*
 * First you need to set the values of the parameter dimensions as const int types. These must be supplied
 * at compile-time in order to efficiently make use of GPU memory. These also need to be placed before the
 * functions LogDensityMeas and LogDensityPop since they need to know the dimensions of the features and parameters.
 *
 */

const int mfeat = 5;
const int pchi = 3;  // chi = {log C, beta, log T}, where C \propto N_H (column density)
const int dtheta = 9;
// frequencies correspond to {500, 350, 250, 170, 70} microns, the Herschel bands
__constant__ const double c_nu[mfeat] = {5.99584916e11, 8.56549880e11, 1.19916983e12, 1.87370286e12, 2.99792458e12};
const double nu_ref = 2.3e11;  // 230 GHz
__constant__ double c_nu_ref = nu_ref;

const int dof = 8;  // population-level model is a multivariate student's t-distribution with dof degrees of freedom
__constant__ int c_dof = dof;

// physical constants, cgs
const double clight = 2.99792458e10;
__constant__ double c_clight = clight;
const double hplanck = 6.6260755e-27;
__constant__ double c_hplanck = hplanck;
const double kboltz = 1.380658e-16;
__constant__ double c_kboltz = kboltz;

// Compute the model dust SED, a modified blackbody
__device__
double modified_blackbody(double nu, double C, double beta, double T) {
	double sed = 2.0 * c_hplanck * nu * nu * nu / (c_clight * c_clight) / (exp(c_hplanck * nu / (c_kboltz * T)) - 1.0);
	sed *= C * pow(nu / c_nu_ref, beta);

	return sed;
}

/*
 * This function returns the logarithm of the conditional density of the measurements given the
 * characteristics for a single data point, log p(y_i | chi_i). For this model p(y_i | chi_i) is a product of
 * 5 independent normal distributions with means given by the SED values at the j^th observational bandpass and
 * known variances.
 *
 * The input parameters are:
 *
 * 	chi      - Pointer to the values of the characteristics (log C, beta, log T) for the i^th data point.
 * 	meas     - Pointer to the measurements for the i^th data point.
 * 	meas_unc - Pointer to the standard deviations in the measurement errors for y_ij.
 *
 */

__device__
double LogDensityMeas(double* chi, double* meas, double* meas_unc)
{
	double C = exp(chi[0]);
	double T = exp(chi[2]);
	double logdens_meas = 0.0;
	for (int j = 0; j < mfeat; ++j) {
		// p(y_ij | chi_ij) is a normal density centered at the model SED
		double model_sed = modified_blackbody(c_nu[j], C, chi[1], T);
		logdens_meas += -0.5 * (meas[j] - model_sed) * (meas[j] - model_sed) / (meas_unc[j] * meas_unc[j]);
	}
	return logdens_meas;
}

/*
 * Helper functions used by LogDensityPop to compute the log-density of log C, beta, log T | theta
 */

// calculate the inverse of a 3 x 3 matrix
__device__ __host__
double matrix_invert3d(double* A, double* A_inv) {
	double determ_inv = 0.0;
	determ_inv = 1.0 / (A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[8] * A[3] - A[5] * A[6]) +
			A[2] * (A[3] * A[7] - A[4] * A[6]));

	A_inv[0] = determ_inv * (A[4] * A[8] - A[5] * A[7]);
	A_inv[1] = -determ_inv * (A[1] * A[8] - A[2] * A[7]);
	A_inv[2] = determ_inv * (A[1] * A[5]- A[2] * A[4]);
	A_inv[3] = -determ_inv * (A[3] * A[8] - A[5] * A[6]);
	A_inv[4] = determ_inv * (A[0] * A[8] - A[2] * A[6]);
	A_inv[5] = -determ_inv * (A[0] * A[5] - A[2] * A[3]);
	A_inv[6] = determ_inv * (A[3] * A[7] - A[4] * A[6]);
	A_inv[7] = -determ_inv * (A[0] * A[7] - A[1] * A[6]);
	A_inv[8] = determ_inv * (A[0] * A[4] - A[1] * A[3]);

	return determ_inv;
}

// calculate chisqr = transpose(x) * covar_inv * x
__device__ __host__
double chisqr(double* x, double* covar_inv, int nx)
{
	double chisqr = 0.0;
	for (int i = 0; i < nx; ++i) {
		for (int j = 0; j < nx; ++j) {
			chisqr += x[i] * covar_inv[i * nx + j] * x[j];
		}
	}
	return chisqr;
}

/*
 * This function returns the logarithm of the conditional density of the characteristics given the
 * population parameter theta for a single data point, log p(chi_i | theta). For this model p(chi_i | theta) is
 * a 3-dimensional student's t-distribution of c_dof (= 8) degrees of freedom.
 *
 * The input parameters are:
 *
 * 	chi       - Pointer to the values of the characteristics for the i^th data point.
 * 	theta     - Pointer to the population parameter.
 *
 */
__device__
double LogDensityPop(double* chi, double* theta)
{
	double covar[pchi * pchi];
	double covar_inv[pchi * pchi];
	double cov_determ_inv;

	// theta = (mu, log(sigma), arctanh(corr)) to allow more efficient sampling, so we need to transform theta values to
	// the values of the covariance matrix of (log C, beta, log T)
	covar[0] = exp(2.0 * theta[pchi]);  // Covar[0,0], variance in log C
	covar[1] = tanh(theta[2 * pchi]) * exp(theta[pchi] + theta[pchi+1]);  // Covar[0,1] = cov(log C, beta)
	covar[2] = tanh(theta[2 * pchi + 1]) * exp(theta[pchi] + theta[pchi+2]);  // Covar[0,2] = cov(log C, log T)
	covar[3] = covar[1];  // Covar[1,0]
	covar[4] = exp(2.0 * theta[pchi + 1]);  // Covar[1,1], variance in beta
	covar[5] = tanh(theta[2 * pchi + 2]) * exp(theta[pchi+1] + theta[pchi+2]);  // Covar[1,2] = cov(beta, log T)
	covar[6] = covar[2];  // Covar[2,0]
	covar[7] = covar[5];  // Covar[2,1]
	covar[8] = exp(2.0 * theta[pchi + 2]);  // Covar[2,2], variance in log T

	cov_determ_inv = matrix_invert3d(covar, covar_inv);
	double chi_cent[pchi];
	for (int j = 0; j < pchi; ++j) {
		chi_cent[j] = chi[j] - theta[j];
	}
	double zsqr = chisqr(chi_cent, covar_inv, pchi);

	// multivariate student's t-distribution with DOF degrees of freedom
	double logdens_pop = 0.5 * log(cov_determ_inv) - (pchi + c_dof) / 2.0 * log(1.0 + zsqr / c_dof);

	return logdens_pop;
}

/*
 * Pointers to the GPU functions used to compute the conditional log-densities for a single data point.
 * These functions live on the GPU in constant memory.
 *
 * IF YOU ARE USING LogDensityMeas and LogDensityPop TO COMPUTE YOUR CONDITIONAL DENSITIES, DO NOT MODIFY THESE.
 * Otherwise you will need to set these points to whichever functions you are using to compute these quantities.
 *
 */
__constant__ pLogDensMeas c_LogDensMeas = LogDensityMeas;  // log p(y_i|chi_i)
__constant__ pLogDensPop c_LogDensPop = LogDensityPop;  // log p(chi_i|theta)

/*
 * Pointer to the population parameter (theta), stored in constant memory on the GPU. Originally defined in
 * kernels.cu and kernels.cuh. Needed by LogDensityPop, which computes the conditional posterior of the
 * characteristics given the population parameters: log p(chi_i|theta). This assumes a maximum of 100 elements
 * in the theta parameter vector.
 *
 * If you need more than this, you will have to change this manually here and in
 * the kernels.cuh and kernels.cu files.
 *
 * YOU SHOULD NOT MODIFY THIS UNLESS YOU KNOW WHAT YOU ARE DOING.
 */
extern __constant__ double c_theta[100];


int main(int argc, char** argv)
{
	time_t timer1, timer2;  // keep track of how long the program takes to run
	time(&timer1);
	/*
	 * Read in the data for the measurements, meas, and their standard deviations, meas_unc.
	 */
	std::string datafile = "../data/cbt_sed_100000.dat";
	int ndata = get_file_lines(datafile) - 1;  // subtract off one line for the header
	std::cout << "Loaded " << ndata << " data points." << std::endl;

	vecvec fnu(ndata);  // the measured SEDs
	vecvec fnu_sig(ndata);  // the standard deviation in the measurement errors
	read_data(datafile, fnu, fnu_sig, ndata, mfeat);

	/*
	 * Set the number of MCMC iterations and the amount of thinning for the chi and theta samples.
	 *
	 * NOTE THAT IF YOU HAVE A LARGE DATA SET, YOU WILL PROBABLY WANT TO THIN THE CHI VALUES SIGNIFICANTLY SO
	 * YOU DO NOT RUN OUR OF MEMORY.
	 */

	int nmcmc_iter = 50000;
	int nburnin = nmcmc_iter / 2;
	int nchi_samples = 100;
	int nthin_chi = nmcmc_iter / nchi_samples;

	// first create pointers to instantiated subclassed DataAugmentation and PopulationPar objects, since we need to give them to the
	// constructor for the GibbsSampler class.
	boost::shared_ptr<DataAugmentation<mfeat, pchi, dtheta> > CBT(new ConstBetaTemp<mfeat, pchi, dtheta>(fnu, fnu_sig));
	boost::shared_ptr<PopulationPar<mfeat, pchi, dtheta> > Theta(new DustPopPar<mfeat, pchi, dtheta>);

	// instantiate the GibbsSampler object and run the sampler
	GibbsSampler<mfeat, pchi, dtheta> Sampler(CBT, Theta, nmcmc_iter, nburnin, nthin_chi);

	/*
	 * DEBUGGING
	 */
//	vecvec cbt_true(ndata);
//	std::string cbtfile = "../data/true_cbt_1000.dat";
//	load_cbt(cbtfile, cbt_true, ndata);
//
//	// copy input data to data members
//	hvector h_cbt(ndata * pchi);
//	dvector d_cbt;
//	for (int j = 0; j < pchi; ++j) {
//		for (int i = 0; i < ndata; ++i) {
//			h_cbt[ndata * j + i] = cbt_true[i][j];
//		}
//	}
//	// copy data from host to device
//	d_cbt = h_cbt;
//
//	hvector h_theta(dtheta);
//	h_theta[0] = 15.0;
//	h_theta[1] = 2.0;
//	h_theta[2] = log(15.0);
//	h_theta[3] = log(1.0);
//	h_theta[4] = log(0.1);
//	h_theta[5] = log(0.3);
//	h_theta[6] = atanh(-0.5);
//	h_theta[7] = atanh(0.0);
//	h_theta[8] = atanh(0.25);
//
//	Sampler.GetDaugPtr()->SetChi(d_cbt, true);
//	Sampler.GetThetaPtr()->SetTheta(h_theta, true);
//	Sampler.FixPopPar();

	// run the MCMC sampler
	Sampler.Run();

    // grab the samples
	vecvec theta_samples = Sampler.GetPopSamples();
	std::vector<vecvec> chi_samples = Sampler.GetCharSamples();

	std::cout << "Writing results to text files..." << std::endl;

	// write the sampled theta values to a file. Output will have nsamples rows and dtheta columns.
	std::string thetafile("dusthm_thetas.dat");
	write_thetas(thetafile, theta_samples);

	// write the posterior means and standard deviations of the characteristics to a file. output will have ndata rows and
	// 2 * pchi columns, where the column format is posterior mean 1, posterior sigma 1, posterior mean 2, posterior sigma 2, etc.
	std::string chifile("dusthm_chi_summary.dat");
	write_chis(chifile, chi_samples);

	time(&timer2);
	double seconds = difftime(timer2, timer1);

	std::cout << "Program took " << seconds << " seconds." << std::endl;

}

