/*
 * cudahm_blueprint.cu
 *
 *  Created on: Mar 13, 2014
 *      Author: Brandon C. Kelly
 *
 * This file provides a blueprint for using the CUDAHM API. In order to use CUDAHM the user must supply a function
 * to compute the logarithm of the conditional probability densities of the measurements (y) given the
 * characteristics (chi), and the characteristics given the population parameter theta. The user must also set
 * the pointers c_pLogDensMeas and c_pLogDensPop to these functions, in order to correctly place these functions
 * in GPU constant memory. The purpose of this file is to provide an easy way of setting the pointers and stream-line
 * the use of CUDAHM to build MCMC sampler. Using this blueprint, the user need only modify the LogDensityMeas and
 * LogDensityPop functions to ensure that the pointers are set correctly.
 *
 * The MCMC sampler should be constructed within the main function using the GibbSampler class. Further details
 * are provided below. A complete working example is provided in normnorm.cu.
 *
 */

// standard library includes
#include <iostream>

// local CUDAHM includes
#include "GibbsSampler.hpp"
#include "input_output.hpp"
#include "ConstBetaTemp.cuh"
#include "DustPopPar.hpp"

/*
 * First you need to set the values of the parameter dimensions as const int types. These must be supplied
 * at compile-time in order to efficiently make use of GPU memory. These also need to be placed before the
 * functions LogDensityMeas and LogDensityPop if they need to know the dimensions of the features and parameters.
 *
 */

const int mfeat = 5;
const int pchi = 3;  // chi = {log C, beta, log T}, where C \propto N_H
const int dtheta = 9;
__constant__ const double c_nu[mfeat] = {6.0e11, 8.571e11, 1.2e11, 1.765e12, 4.286e12};  // {500, 350, 250, 170, 70} microns, Herschel bands
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
 * characteristics for a single data point, log p(y_i | chi_i). This function must be supplied by the user
 * and written in CUDA. The input parameters are:
 *
 * 	chi      - Pointer to the values of the characteristics for the i^th data point.
 * 	meas     - Pointer to the measurements for the i^th data point.
 * 	meas_unc - Pointer to the standard deviations in the measurement errors for y_ij.
 * 	mfeat    - The number of features measured for each data point, i.e., the length of the meas array.
 *  pchi     - The number of characteristics for each data point, i.e., the length of the chi array.
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
 * Helper functions used by the function that computes the log-density of log C, beta, log T | theta
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

// calculate transpose(x) * covar_inv * x
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

//__device__ __host__
//double tanh(double x) {
//	return (exp(2.0 * x) - 1.0) / (exp(2.0 * x) + 1.0);
//}

/*
 * This function returns the logarithm of the conditional density of the characteristics given the
 * population parameter theta for a single data point, log p(chi_i | theta). This function must be supplied by
 * the user and written in CUDA. The input parameters are:
 *
 * 	chi       - Pointer to the values of the characteristics for the i^th data point.
 * 	theta     - Pointer to the population parameter.
 *  pchi      - The number of characteristics for each data point, i.e., the length of the chi array.
 *  dim_theta - The dimension of the population parameter vector theta, i.e., the length of the theta array.
 *
 */
__device__
double LogDensityPop(double* chi, double* theta)
{
	double covar[pchi * pchi];
	double covar_inv[pchi * pchi];
	double cov_determ_inv;

	// transform theta values to covariance matrix of (log C, beta, log T)
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
	double logdens_pop = 0.5 * log(cov_determ_inv) - (pchi + c_dof) / 2.0 * log(1.0 + zsqr / pchi);

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
	/*
	 * Read in the data for the measurements, meas, and their standard deviations, meas_unc.
	 */

	std::string datafile = "../data/cbt_sed_1000.dat";
	int ndata = get_file_lines(datafile);
	std::cout << "Loaded " << ndata << " data points." << std::endl;

	vecvec fnu(ndata);
	vecvec fnu_sig(ndata);
	read_data(datafile, fnu, fnu_sig, ndata, mfeat);

	/*
	 * Set the number of MCMC iterations and the amount of thinning for the chi and theta samples.
	 *
	 * NOTE THAT IF YOU HAVE A LARGE DATA SET, YOU WILL PROBABLY WANT TO THIN THE CHI VALUES SIGNIFICANTLY SO
	 * YOU DO NOT RUN OUR OF MEMORY.
	 */

	int nmcmc_iter = 10;
	// int nburnin = nmcmc_iter / 2;
	int nburnin = 1;
	int nchi_samples = 10;
	int nthin_chi = nmcmc_iter / nchi_samples;

	/*
	 * Instantiate the GibbsSampler<mfeat, pchi, dtheta> object here. Once you've instantiated it, use the
	 * GibbSampler::Run() method to run the MCMC sampler. Finally, use GibbSampler::GetCharSampler() to get
	 * the sampled characteristics as a std::vector<std::vector<std::vector<double> > > (three nested vectors,
	 * dimensions nchi_samples x ndata x pchi) object. Similarly, use the GibbsSampler::GetPopSamples() to
	 * retrieve the sampled theta values as a std::vector<std::vector<double> > (two nested vectors,
	 * dimensions nsamples x dtheta) object.
	 */

	// first instantiate the subclassed DataAugmentation and PopulationPar objects
	boost::shared_ptr<DataAugmentation<mfeat, pchi, dtheta> > CBT(new ConstBetaTemp<mfeat, pchi, dtheta>(fnu, fnu_sig));
	boost::shared_ptr<PopulationPar<mfeat, pchi, dtheta> > Theta(new DustPopPar<mfeat, pchi, dtheta>);

	// instantiate the GibbsSampler object and run the sampler
	GibbsSampler<mfeat, pchi, dtheta> Sampler(CBT, Theta, nmcmc_iter, nburnin, nthin_chi);
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

}

